import { fmt, now, rng } from 'lib/util.js';
import * as ops from 'lib/ops.js';
import { Tensor } from 'lib/tensor.js';
import { attention } from 'lib/attention.js';

export default async function demo(m, lab) {
  const hw = m.H100, L = m.LLAMA3_8B;
  const b = m.DTYPE_BYTES.bf16;
  const ridge = m.ridgePoint(hw.flops, hw.bandwidth);
  lab.log(`${hw.name}: ${fmt(hw.flops)}FLOP/s bf16, ${fmt(hw.bandwidth)}B/s HBM -> ridge point ${ridge.toFixed(1)} FLOP/byte`);

  // ---------- 1. every matmul of one Llama-3-8B layer, at decode and at prefill ----------
  const layerOps = (M) => [
    ['qkv projection', M, L.dModel, L.dModel + 2 * L.nKvHeads * L.headDim],
    ['attention out', M, L.dModel, L.dModel],
    ['mlp gate+up', M, L.dModel, 2 * L.dFF],
    ['mlp down', M, L.dFF, L.dModel],
  ];
  const rows = [], analysed = { 1: [], 2048: [] };
  for (const M of [1, 2048]) {
    for (const [name, a, k, n] of layerOps(M)) {
      const r = m.analyseOp({ name, ...m.matmulCost(a, k, n, b) }, hw);
      analysed[M].push(r);
      rows.push([
        M === 1 ? 'decode' : 'prefill 2048', name, `[${a},${k}]x[${k},${n}]`,
        fmt(r.flops / 1e9) + ' GFLOP', fmt(r.bytes / 1e6) + ' MB',
        +r.intensity.toFixed(2), r.bound,
        +(r.seconds * 1e6).toFixed(1), +(100 * r.fractionOfPeak).toFixed(2),
      ]);
    }
    await lab.tick();
  }
  const attn = m.analyseOp({ name: 'attention (naive)', ...m.attentionCost(4096, L.headDim, { bytesPerElement: b }) }, hw);
  const flash = m.analyseOp({ name: 'attention (flash)', ...m.attentionCost(4096, L.headDim, { bytesPerElement: b, flash: true }) }, hw);
  for (const r of [attn, flash]) {
    rows.push(['one head, 4096 ctx', r.name, `[4096,${L.headDim}]`, fmt(r.flops / 1e9) + ' GFLOP', fmt(r.bytes / 1e6) + ' MB',
      +r.intensity.toFixed(2), r.bound, +(r.seconds * 1e6).toFixed(1), +(100 * r.fractionOfPeak).toFixed(2)]);
  }
  lab.table({
    title: `Llama-3-8B ops on an ${hw.name} (one layer; ridge = ${ridge.toFixed(0)} FLOP/byte)`,
    columns: ['phase', 'op', 'shape', 'FLOPs', 'HBM bytes', 'FLOP/byte', 'bound', 'time (us)', '% of peak'],
    rows,
  });

  // ---------- 2. the roofline itself ----------
  const marks = [
    { label: 'decode matmul', intensity: analysed[1][1].intensity },
    { label: 'attention, naive', intensity: attn.intensity },
    { label: 'prefill matmul', intensity: analysed[2048][1].intensity },
    { label: 'attention, flash', intensity: flash.intensity },
  ];
  const grid = [];
  for (let e = -2; e <= 12; e++) grid.push(e);
  const xs = [...new Set([...grid, ...marks.map((p) => Math.log2(p.intensity))])].sort((p, q) => p - q);
  const curve = (device) => xs.map((e) => m.attainable(2 ** e, device.flops, device.bandwidth) / 1e12);
  const markValues = xs.map((e) => (marks.some((p) => Math.abs(Math.log2(p.intensity) - e) < 1e-9) ? m.attainable(2 ** e, hw.flops, hw.bandwidth) / 1e12 : NaN));
  lab.plot({
    title: 'Roofline: attainable TFLOP/s vs arithmetic intensity',
    x: xs,
    series: [
      { name: m.H100.name, values: curve(m.H100) },
      { name: m.A100.name, values: curve(m.A100) },
      { name: m.RTX_4090.name, values: curve(m.RTX_4090) },
      { name: m.B200.name, values: curve(m.B200) },
      { name: 'Llama-3-8B ops (H100)', values: markValues },
    ],
    xlabel: 'log2(arithmetic intensity, FLOP/byte)',
    ylabel: 'TFLOP/s',
    yscale: 'log',
  });
  for (const p of marks) {
    const pct = 100 * m.attainable(p.intensity, hw.flops, hw.bandwidth) / hw.flops;
    lab.log(`${p.label}: ${p.intensity.toFixed(2)} FLOP/byte -> ${pct.toFixed(2)}% of ${hw.name} peak (${p.intensity < ridge ? 'memory' : 'compute'}-bound)`);
  }

  const b200Ridge = m.ridgePoint(m.B200.flops, m.B200.bandwidth);
  lab.log(`${m.B200.name}: ${fmt(m.B200.flops)}FLOP/s bf16, ${fmt(m.B200.bandwidth)}B/s HBM3e -> ridge point ${b200Ridge.toFixed(1)} FLOP/byte ` +
    `(${(m.B200.flops / hw.flops).toFixed(2)}x the ${hw.name}'s FLOP/s, ${(m.B200.bandwidth / hw.bandwidth).toFixed(2)}x its bandwidth)`);

  // ---------- 2b. your online-softmax attention against module 05's ----------
  const seq = 256, dh = 64;
  const q = ops.randn([seq, dh], rng(71)), k = ops.randn([seq, dh], rng(72)), v = ops.randn([seq, dh], rng(73));
  const refOut = attention(new Tensor(q), new Tensor(k), new Tensor(v), { causal: true }).out;
  const flashRows = [];
  let flashWorst = 0;
  for (const bs of [16, 64, seq]) {
    const got = m.tiledAttention(q, k, v, bs);
    let diff = 0;
    for (let i = 0; i < got.data.length; i++) diff = Math.max(diff, Math.abs(got.data[i] - refOut.data[i]));
    lab.check(diff < 1e-4, `tiledAttention with blockSize ${bs} disagrees with lib/attention.js by ${diff}`);
    flashWorst = Math.max(flashWorst, diff);
    flashRows.push([bs, `${bs} scores`, `${seq} scores`, diff.toExponential(1)]);
  }
  await lab.tick();
  lab.table({
    title: `Your tiledAttention vs lib/attention.js, one causal head, T = ${seq}, dh = ${dh}`,
    columns: ['blockSize', 'score state per query row', 'naive kernel holds', 'max |difference|'],
    rows: flashRows,
  });

  // ---------- 3. your own matmul: naive vs tiled ----------
  const sizes = [256, 512, 1024];
  const blockSize = 64;
  const naiveGf = [], tiledGf = [];
  // The tests have already called your kernels with plain Arrays and a Proxy, which leaves the JIT's
  // type feedback for these functions polymorphic and can make them several times slower. Time fresh
  // copies built from your source instead, checked against the originals on a small Float32Array
  // product (which also warms them up). If a copy cannot stand alone, for example because it calls a
  // helper defined elsewhere in your file, fall back to the original function.
  const fresh = (fn, args) => {
    try {
      const copy = new Function(`return (${fn.toString()});`)();
      const want = fn(...args), got = copy(...args);
      if (got.length !== want.length) return fn;
      for (let i = 0; i < want.length; i++) if (Math.abs(got[i] - want[i]) > 1e-3) return fn;
      return copy;
    } catch { return fn; }
  };
  const warmA = m.randomMatrix(96, 5), warmB = m.randomMatrix(96, 6);
  const naiveKernel = fresh(m.naiveMatmul, [warmA, warmB, 96]);
  const tiledKernel = fresh(m.tiledMatmul, [warmA, warmB, 96, 16]);
  for (const n of sizes) {
    const A = m.randomMatrix(n, n + 1), B = m.randomMatrix(n, n + 2);
    let t0 = now();
    const cN = naiveKernel(A, B, n);
    naiveGf.push(m.gflops(2 * n ** 3, (now() - t0) / 1000));
    await lab.tick();
    t0 = now();
    const cT = tiledKernel(A, B, n, blockSize);
    tiledGf.push(m.gflops(2 * n ** 3, (now() - t0) / 1000));
    let maxDiff = 0;
    for (let i = 0; i < cN.length; i++) maxDiff = Math.max(maxDiff, Math.abs(cN[i] - cT[i]));
    lab.check(maxDiff < 1e-2, `tiledMatmul disagrees with naiveMatmul at n=${n} by ${maxDiff}`);
    lab.log(`n=${n}: naive ${naiveGf.at(-1).toFixed(2)} GFLOP/s, tiled(${blockSize}) ${tiledGf.at(-1).toFixed(2)} GFLOP/s, speedup ${(tiledGf.at(-1) / naiveGf.at(-1)).toFixed(2)}x`);
    lab.progress((sizes.indexOf(n) + 1) / sizes.length, `matmul ${n}x${n}`);
    await lab.tick();
  }
  lab.plot({
    title: `Your matmul: naive vs ${blockSize}x${blockSize} tiles`,
    x: sizes,
    series: [{ name: 'naive (i,j,k)', values: naiveGf }, { name: `tiled (${blockSize})`, values: tiledGf }],
    xlabel: 'n (n x n matrices)', ylabel: 'GFLOP/s',
  });
  const big = sizes.at(-1);
  lab.bar({
    title: 'Measured GFLOP/s, naive vs tiled, at each size',
    labels: sizes.flatMap((n) => [`naive ${n}`, `tiled ${n}`]),
    values: sizes.flatMap((n, i) => [+naiveGf[i].toFixed(3), +tiledGf[i].toFixed(3)]),
  });
  lab.log(`traffic model at n=${big}: naive intensity ${m.arithmeticIntensity(2 * big ** 3, m.blockTraffic(big, 1, 4)).toFixed(2)} FLOP/byte, ` +
    `tiled(${blockSize}) ${m.arithmeticIntensity(2 * big ** 3, m.blockTraffic(big, blockSize, 4)).toFixed(1)} FLOP/byte`);

  // ---------- 4. the hierarchy the tiling is exploiting ----------
  const cap = (x) => (x >= 1e12 ? (x / 1e12).toFixed(1) + ' TB' : x >= 1e9 ? (x / 1e9).toFixed(0) + ' GB' : (x / 1e6).toFixed(1) + ' MB');
  const bw = (x) => (x >= 1e12 ? (x / 1e12).toFixed(1) + ' TB/s' : (x / 1e9).toFixed(0) + ' GB/s');
  lab.table({
    title: 'Approximate H100 memory hierarchy (capacities from NVIDIA\'s H100 whitepaper; upper-level bandwidths and latencies are rounded orders of magnitude)',
    columns: ['level', 'capacity', 'bandwidth', 'latency (ns)', 'bandwidth vs HBM', 'note'],
    rows: m.MEMORY_HIERARCHY.map((h) => [h.level, cap(h.capacity), bw(h.bandwidth), h.latencyNs, (h.bandwidth / 3.35e12).toFixed(2) + 'x', h.note]),
  });

  // ---------- 5. the summary ----------
  const decode = analysed[1];
  const perLayer = decode.reduce((s, r) => s + r.seconds, 0);
  const tokPerSec = 1 / (perLayer * L.layers);
  const single = m.decodeThroughput(hw, { params: L.params, bytesPerParam: b, batch: 1 });
  const int4 = m.decodeThroughput(hw, { params: L.params, bytesPerParam: m.DTYPE_BYTES.int4, batch: 1 });
  const minBatch = m.minBatchForCompute(hw, { K: L.dModel, N: L.dModel, bytesPerElement: b });
  const sram = m.flashBlockSize(228 * 1024, L.headDim, b);
  const speedup = tiledGf.at(-1) / naiveGf.at(-1);
  const smallRatio = tiledGf[0] / naiveGf[0];
  lab.done(`
Your roofline calculator puts one **${L.name}** decode step at **${decode[1].intensity.toFixed(2)} FLOP/byte** on an ${hw.name},
against a ridge point of **${ridge.toFixed(0)} FLOP/byte** — **${(100 * decode[1].fractionOfPeak).toFixed(2)}% of peak bf16**.
Summing the four matmuls over ${L.layers} layers gives **${(perLayer * L.layers * 1e3).toFixed(2)} ms/token**, about **${tokPerSec.toFixed(0)} tokens/s**,
and the whole-model estimate is **${single.tokensPerSecond.toFixed(0)} tokens/s** for a single stream (${single.bound}-bound): the layer sum comes out
about ${(100 * (tokPerSec / single.tokensPerSecond - 1)).toFixed(0)}% high because it leaves out the embedding table and the output projection (together **${(2 * L.vocab * L.dModel / 1e9).toFixed(2)}B** parameters) that the whole-model estimate counts.
A real step reads all of the output projection, about **${(L.vocab * L.dModel * b / 1e9).toFixed(2)} GB** in bf16, but only one row of the embedding table, so its weight traffic lies between the two estimates (before counting the KV cache).
The same matmuls at prefill 2048 sit at **${analysed[2048][1].intensity.toFixed(0)} FLOP/byte** and reach **100% of peak**;
the crossover is a batch of **${minBatch.toFixed(0)}** once the activations are counted (**${(ridge * b / 2).toFixed(0)}** if you count only the weight stream). Quantising the weights to int4 raises the single-stream ceiling to
**${int4.tokensPerSecond.toFixed(0)} tokens/s** without changing a single FLOP.

Attention at 4096 tokens moves **${fmt(attn.bytes / 1e6)} MB** per head naively and **${fmt(flash.bytes / 1e6)} MB** when the
score tiles stay in SRAM — **${(attn.bytes / flash.bytes).toFixed(0)}x less traffic** for identical arithmetic, which is FlashAttention's
whole argument; at headDim ${L.headDim} in bf16, 228 KiB (233,472 bytes) of shared memory holds a block of **${sram}** rows.
Your online-softmax \`tiledAttention\` matched module 05's attention to within **${flashWorst.toExponential(1)}** at every block size
while holding only one block of scores per query row. On the plot, the ${m.B200.name} line sits higher on both roofs, but its
ridge of **${b200Ridge.toFixed(0)} FLOP/byte** is close to the H100's, so a batch-1 decode stays just as far on the memory side.

Your own kernels, measured just now: naive **${naiveGf.at(-1).toFixed(2)} GFLOP/s** and tiled(${blockSize}) **${tiledGf.at(-1).toFixed(2)} GFLOP/s**
at n=${big} (**${speedup.toFixed(2)}x**) — ${speedup < 1
    ? `so on this run tiling did not pay off even at the largest size. A single JavaScript timing is noisy (other tabs, garbage collection and JIT tiering all move it), so run the demo again; if tiling keeps losing, compare your tile loop with the worked naive loop for extra work in the innermost loop`
    : smallRatio < speedup
    ? `against **${smallRatio.toFixed(2)}x** at n=${sizes[0]}, where B (${(4 * sizes[0] ** 2 / 1024).toFixed(0)} KiB) already fits in cache, so there is little traffic to save and the extra loop bookkeeping ${smallRatio < 1 ? 'costs more than it saves' : 'eats most of the gain'}`
    : `and **${smallRatio.toFixed(2)}x** at n=${sizes[0]}; on this machine the cache effect did not separate the sizes, so run the demo again to see how much of the difference is timing noise`}.
The trend across sizes in the chart is the lesson, not the headline speedup. Meanwhile the traffic model says tiling raises intensity from
**${m.arithmeticIntensity(2 * big ** 3, m.blockTraffic(big, 1, 4)).toFixed(2)}** to **${m.arithmeticIntensity(2 * big ** 3, m.blockTraffic(big, blockSize, 4)).toFixed(1)} FLOP/byte**.
That peak is roughly **${(hw.flops / 1e9 / Math.max(naiveGf.at(-1), tiledGf.at(-1))).toExponential(1)}x** below the ${hw.name}'s bf16 peak — JavaScript never gets near either roof.
`);
}
