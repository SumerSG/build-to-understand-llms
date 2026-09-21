import { fmt, now } from 'lib/util.js';

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

  // ---------- 3. your own matmul: naive vs tiled ----------
  const sizes = [256, 512, 768, 1024];
  const blockSize = 64;
  const naiveGf = [], tiledGf = [];
  for (const n of sizes) {
    const A = m.randomMatrix(n, n + 1), B = m.randomMatrix(n, n + 2);
    let t0 = now();
    const cN = m.naiveMatmul(A, B, n);
    naiveGf.push(m.gflops(2 * n ** 3, (now() - t0) / 1000));
    await lab.tick();
    t0 = now();
    const cT = m.tiledMatmul(A, B, n, blockSize);
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
    title: 'Approximate H100 memory hierarchy (NVIDIA H100 whitepaper; latencies are rounded orders of magnitude)',
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
  lab.done(`
Your roofline calculator puts one **${L.name}** decode step at **${decode[1].intensity.toFixed(2)} FLOP/byte** on an ${hw.name},
against a ridge point of **${ridge.toFixed(0)} FLOP/byte** — **${(100 * decode[1].fractionOfPeak).toFixed(2)}% of peak bf16**.
Summing the four matmuls over ${L.layers} layers gives **${(perLayer * L.layers * 1e3).toFixed(2)} ms/token**, about **${tokPerSec.toFixed(0)} tokens/s**,
and the whole-model estimate agrees: **${single.tokensPerSecond.toFixed(0)} tokens/s** for a single stream (${single.bound}-bound).
The same matmuls at prefill 2048 sit at **${analysed[2048][1].intensity.toFixed(0)} FLOP/byte** and reach **100% of peak**;
the crossover is a batch of **${minBatch.toFixed(0)}**. Quantising the weights to int4 raises the single-stream ceiling to
**${int4.tokensPerSecond.toFixed(0)} tokens/s** without changing a single FLOP.

Attention at 4096 tokens moves **${fmt(attn.bytes / 1e6)} MB** per head naively and **${fmt(flash.bytes / 1e6)} MB** when the
score tiles stay in SRAM — **${(attn.bytes / flash.bytes).toFixed(0)}x less traffic** for identical arithmetic, which is FlashAttention's
whole argument; at headDim ${L.headDim} in bf16, 228 KB of shared memory holds a block of **${sram}** rows.

Your own kernels, measured just now: naive **${naiveGf.at(-1).toFixed(2)} GFLOP/s** and tiled(${blockSize}) **${tiledGf.at(-1).toFixed(2)} GFLOP/s**
at n=${big} (**${speedup.toFixed(2)}x**), while the traffic model says tiling raises intensity from
**${m.arithmeticIntensity(2 * big ** 3, m.blockTraffic(big, 1, 4)).toFixed(2)}** to **${m.arithmeticIntensity(2 * big ** 3, m.blockTraffic(big, blockSize, 4)).toFixed(1)} FLOP/byte**.
That peak is roughly **${(hw.flops / 1e9 / Math.max(naiveGf.at(-1), tiledGf.at(-1))).toExponential(1)}x** below the ${hw.name}'s bf16 peak — JavaScript never gets near either roof.
`);
}
