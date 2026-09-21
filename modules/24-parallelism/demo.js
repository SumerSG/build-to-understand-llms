import { fmt } from 'lib/util.js';

const GB = (bytes) => +(bytes / 1e9).toFixed(1);
const S = (sec) => +sec.toFixed(2);

export default async function demo(m, lab) {
  const model = m.LLAMA3_70B;
  const gpus = 64;
  const tokens = model.batchSeqs * model.seqLen;

  // ---------- 1. the ring: why message size decides everything ----------
  lab.log(`Pricing a ring all-reduce on ${m.INFINIBAND.name}: ${m.INFINIBAND.latency * 1e6} us latency, ${m.INFINIBAND.bandwidth / 1e9} GB/s.`);
  const ranks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
  const smallMsg = [], largeMsg = [];
  for (const n of ranks) {
    smallMsg.push(m.ringAllReduceTime(1e6, n, m.INFINIBAND) * 1e3);
    largeMsg.push(m.ringAllReduceTime(16e9, n, m.INFINIBAND) * 1e3);
    lab.progress(ranks.indexOf(n) / ranks.length, `ring over ${n} GPUs`);
    await lab.tick();
  }
  lab.plot({
    title: 'Ring all-reduce time vs ring size (InfiniBand)',
    x: ranks,
    series: [{ name: '1 MB buffer', values: smallMsg }, { name: '16 GB buffer', values: largeMsg }],
    xlabel: 'GPUs in the ring', ylabel: 'milliseconds', yscale: 'log',
  });
  const smallGrowth = smallMsg[smallMsg.length - 1] / smallMsg[0];
  const largeGrowth = largeMsg[largeMsg.length - 1] / largeMsg[0];
  lab.log(`From 2 to 1024 GPUs the 1 MB all-reduce gets ${smallGrowth.toFixed(0)}x slower (latency), the 16 GB one ${largeGrowth.toFixed(2)}x (bandwidth: 2(n-1)/n saturates at 2).`);

  // ---------- 2. data parallelism: how far does it scale? ----------
  const base = { params: m.LLAMA3_8B.params, gpu: m.H100, link: m.INFINIBAND };
  const curves = [
    { name: '1 seq/GPU, 60% overlap', cfg: { ...base, tokensPerGpu: m.LLAMA3_8B.seqLen, overlap: 0.6 } },
    { name: '8 seq/GPU, no overlap', cfg: { ...base, tokensPerGpu: 8 * m.LLAMA3_8B.seqLen, overlap: 0 } },
    { name: '1 seq/GPU, no overlap', cfg: { ...base, tokensPerGpu: m.LLAMA3_8B.seqLen, overlap: 0 } },
  ];
  const dps = [8, 16, 32, 64, 128, 256, 512, 1024];
  const series = [];
  for (const c of curves) {
    const values = [];
    for (const dp of dps) { values.push(+(m.dpEfficiency(c.cfg, dp) * 100).toFixed(2)); await lab.tick(); }
    series.push({ name: c.name, values });
  }
  lab.plot({
    title: 'Data-parallel scaling efficiency, Llama-3-8B gradients (16 GB bf16)',
    x: dps, series, xlabel: 'data-parallel replicas', ylabel: 'efficiency (%)',
  });
  lab.log(`At 1024 replicas: ${series.map((s) => `${s.name} -> ${s.values[s.values.length - 1]}%`).join(', ')}.`);

  // ---------- 3. what one GPU holds under tensor parallelism ----------
  const tpDemo = 4;
  const whole = m.tpShards(model, 1);
  const shards = m.tpShards(model, tpDemo);
  lab.table({
    title: `Megatron sharding of one ${model.name} layer across tp=${tpDemo} GPUs`,
    columns: ['matrix', 'split', 'full shape', 'shape on one GPU', 'weights on one GPU'],
    rows: shards.map((s, i) => [s.name, s.split, `[${whole[i].shape}]`, `[${s.shape}]`, fmt(s.shape[0] * s.shape[1])]),
  });
  const layerComm = m.tpLayerComm({ dModel: model.dModel, tokens: model.seqLen }, tpDemo, m.NVLINK);
  lab.log(`Each layer costs ${layerComm.allReduces} all-reduces of ${fmt(layerComm.bytesPerAllReduce)}B = ${(layerComm.time * 1e3).toFixed(2)} ms on NVLink.`);

  // ---------- 4. the layout table ----------
  const candidates = [
    { dp: 64, tp: 1, pp: 1, zero: 3 },
    { dp: 32, tp: 2, pp: 1, zero: 3 },
    { dp: 16, tp: 4, pp: 1, zero: 3 },
    { dp: 8, tp: 8, pp: 1, zero: 1 },
    { dp: 8, tp: 8, pp: 1, zero: 3 },
    { dp: 4, tp: 4, pp: 4, zero: 2 },
    { dp: 2, tp: 4, pp: 8, zero: 1 },
    { dp: 1, tp: 8, pp: 8, zero: 0 },
  ];
  const rows = [];
  for (const s of candidates) {
    const p = m.plan({ model, gpus, strategy: s });
    rows.push([
      `dp=${p.dp} tp=${p.tp} pp=${p.pp}`, `ZeRO-${p.zero}`, p.m,
      S(p.stepTime), S(p.breakdown.tpComm), S(p.breakdown.bubble), S(p.breakdown.dpExposed),
      GB(p.memory.activations), GB(p.memoryPerGpu), p.fits ? 'yes' : 'NO',
    ]);
    await lab.tick();
  }
  lab.table({
    title: `${model.name} on ${gpus} H100s, global batch ${fmt(tokens)} tokens — seconds per step and GB per GPU`,
    columns: ['layout', 'zero', 'micro-batches', 'step (s)', 'tp comm (s)', 'bubble (s)', 'dp exposed (s)', 'activations (GB)', 'total (GB)', 'fits in 80 GB'],
    rows,
  });

  // ---------- 5. choose ----------
  lab.progress(0.9, 'searching layouts');
  const all = m.enumerateStrategies(model, gpus);
  const best = m.bestPlan({ model, gpus });
  await lab.tick();
  lab.check(best !== null, 'bestPlan found no layout that fits in 80 GB');
  const pureDp = m.plan({ model, gpus, strategy: { dp: gpus, tp: 1, pp: 1, zero: 3 } });
  lab.bar({
    title: `Memory per GPU of the winning layout (dp=${best.dp}, tp=${best.tp}, pp=${best.pp}, ZeRO-${best.zero})`,
    labels: ['params', 'grads', 'optimizer', 'activations'],
    values: [GB(best.memory.params), GB(best.memory.grads), GB(best.memory.optimizer), GB(best.memory.activations)],
  });

  const tax = ((best.stepTime / pureDp.stepTime - 1) * 100).toFixed(1);
  lab.done(`
Searched **${all.length}** legal layouts of **${model.name}** (${fmt(model.params)} parameters, ${model.layers} layers) on **${gpus} H100s** with a global batch of **${fmt(tokens)} tokens**.

**Fastest layout that fits:** \`dp=${best.dp}, tp=${best.tp}, pp=${best.pp}, ZeRO-${best.zero}\` — **${S(best.stepTime)} s** per step, **${GB(best.memoryPerGpu)} GB** per GPU, **${fmt(Math.round(best.tokensPerSec))}** tokens/s across the cluster (${fmt(Math.round(best.tokensPerSec / gpus))} per GPU).

**The layout that would be fastest if memory were free:** pure data parallelism, \`dp=${gpus}\` — ${S(pureDp.stepTime)} s per step, but **${GB(pureDp.memoryPerGpu)} GB** per GPU, of which ${GB(pureDp.memory.activations)} GB is activations. It does not fit, so tensor parallelism costs you **${tax}%** of step time to make the run possible at all.

Where the winning step goes: ${S(best.breakdown.compute)} s of arithmetic, ${S(best.breakdown.tpComm)} s of tensor-parallel all-reduces, ${S(best.breakdown.bubble)} s of pipeline bubble, ${S(best.breakdown.dpExposed)} s of exposed gradient all-reduce (of ${S(best.breakdown.dpComm)} s issued — the rest hides behind compute).

On the wire: a 1 MB all-reduce over 1024 GPUs takes ${smallMsg[smallMsg.length - 1].toFixed(1)} ms and is **${smallGrowth.toFixed(0)}x** slower than over 2 GPUs, while a 16 GB one takes ${(largeMsg[largeMsg.length - 1] / 1e3).toFixed(2)} s and is only ${largeGrowth.toFixed(2)}x slower. Small collectives are latency-bound; that is why frameworks bucket gradients.
`);
}
