import { fmt } from 'lib/util.js';

const S = (sec) => +sec.toFixed(2);
const GB = (bytes) => +(bytes / 1e9).toFixed(1);
const gb = (bytes) => (bytes / 1e9).toFixed(2) + ' GB';
const mb = (bytes) => (bytes / 1e6).toFixed(0) + ' MB';
const range = (n) => [...Array(n).keys()];

export default async function demo(m, lab) {
  const topo = m.CLUSTER;
  const model = m.LLAMA3_70B;
  const gpus = 128;
  const nodes = gpus / topo.gpusPerNode;
  const tokens = model.batchSeqs * model.seqLen;

  // ---------- 1. the hierarchy your topology encodes ----------
  lab.log(`Cluster: ${gpus} ${topo.gpu.name} in ${nodes} nodes of ${topo.gpusPerNode}, ${topo.nodesPerPod} nodes per pod.`);
  const rungs = [topo.links.self, topo.links.node, topo.links.pod, topo.links.cluster];
  lab.bar({
    title: 'The bandwidth hierarchy: approximate one-way GB/s per GPU (HBM ≈ 7x NVLink ≈ 9x one IB port)',
    labels: rungs.map((l) => l.name),
    values: rungs.map((l) => +(l.bandwidth / 1e9).toFixed(1)),
  });
  lab.log(`GPU 7 and GPU 8 are adjacent ids but use ${m.linkBetween(7, 8, topo).name}: ${(m.linkBetween(7, 7, topo).bandwidth / m.linkBetween(7, 8, topo).bandwidth).toFixed(0)}x below GPU 7's own memory.`);

  // ---------- 2. flat vs hierarchical all-reduce ----------
  const sizes = [8, 16, 32, 64, 128, 256, 512, 1024];
  const flat = [], hier = [];
  const gradBytes = 16e9; // an 8B-parameter bf16 gradient buffer
  for (const n of sizes) {
    const group = range(n);
    flat.push(+(m.flatAllReduceTime(gradBytes, group, topo) * 1e3).toFixed(2));
    hier.push(+(m.hierarchicalAllReduce(gradBytes, group, topo).time * 1e3).toFixed(2));
    lab.progress(sizes.indexOf(n) / sizes.length, `all-reduce over ${n} GPUs`);
    await lab.tick();
  }
  lab.plot({
    title: 'All-reduce of a 16 GB gradient buffer: one flat ring vs the two-level schedule',
    x: sizes,
    series: [{ name: 'flat ring on the slowest link', values: flat }, { name: 'hierarchical (in node, across nodes, in node)', values: hier }],
    xlabel: 'GPUs in the group', ylabel: 'milliseconds', yscale: 'log',
  });
  const at128 = m.hierarchicalAllReduce(gradBytes, range(128), topo);
  const speedup = flat[sizes.indexOf(128)] / hier[sizes.indexOf(128)];
  lab.table({
    title: 'Where the 16 GB goes on 128 GPUs (16 nodes) under the hierarchical schedule',
    columns: ['phase', 'bytes on the wire', 'link', 'ms'],
    rows: at128.phases.map((p) => [p.name, gb(p.bytes), p.link.name, +(p.time * 1e3).toFixed(2)]),
  });
  lab.log(`Hierarchical is ${speedup.toFixed(2)}x faster at 128 GPUs: the spine carries ${gb(at128.phases[1].bytes)} instead of ${gb(gradBytes)}.`);

  // ---------- 3. plan the 70B ----------
  lab.progress(0.5, 'ranking layouts');
  const ranked = m.rankLayouts({ model, gpus, topo });
  await lab.tick();
  const fitting = ranked.filter((r) => r.fits);
  lab.check(fitting.length > 0, 'no layout of the 70B fits in 80 GB per GPU');
  const best = fitting[0];
  lab.table({
    title: `Top 5 of ${ranked.length} legal layouts: ${model.name} on ${gpus} H100s, global batch ${fmt(tokens)} tokens`,
    columns: ['tp/pp/dp', 'tp link', 'dp link', 'step (s)', 'compute (s)', 'tp comm (s)', 'bubble (s)', 'pp comm (s)', 'dp exposed (s)', 'GB/GPU'],
    rows: fitting.slice(0, 5).map((r) => [
      `${r.layout.tp}/${r.layout.pp}/${r.layout.dp}`,
      r.links.tp.tier, r.links.dp.tier,
      S(r.stepTime), S(r.compute), S(r.tpComm), S(r.bubble), S(r.ppComm), S(r.dpExposed), GB(r.memory.total),
    ]),
  });

  const tp8 = ranked.find((r) => r.layout.tp === 8 && r.layout.pp === 1);
  const tp16 = ranked.find((r) => r.layout.tp === 16 && r.layout.pp === 1);
  const pureDp = ranked.find((r) => r.layout.tp === 1 && r.layout.pp === 1);
  lab.bar({
    title: 'The same 320 all-reduces per micro-batch, on two different links (seconds per step)',
    labels: [`tp=8 on ${tp8.links.tp.name}`, `tp=16 on ${tp16.links.tp.name}`],
    values: [S(tp8.tpComm), S(tp16.tpComm)],
  });
  // Where the chosen layout's communication actually runs, grouped by the link each axis lands on.
  const byLink = new Map();
  for (const [axis, t] of [['tp', best.tpComm], ['pp', best.ppComm], ['dp', best.dpComm]]) {
    if (t <= 0) continue;
    const name = best.links[axis].name;
    byLink.set(name, (byLink.get(name) || 0) + t);
  }
  lab.bar({
    title: `Communication per step of the chosen layout ${best.layout.tp}/${best.layout.pp}/${best.layout.dp}, by link (seconds, before overlap)`,
    labels: [...byLink.keys()],
    values: [...byLink.values()].map(S),
  });
  lab.log(`tp=16 needs ${GB(tp16.memory.total)} GB per GPU against tp=8's ${GB(tp8.memory.total)} GB — less memory — and is still ${((tp16.stepTime / tp8.stepTime - 1) * 100).toFixed(0)}% slower.`);

  // ---------- 4. the MoE all-to-all ----------
  const moe = m.DEEPSEEK_V3;
  const epNodes = [1, 2, 4, 8, 16];
  const free = [], limited = [];
  for (const n of epNodes) {
    const group = range(n * topo.gpusPerNode);
    const a = m.moeAllToAll({ tokensPerGpu: moe.seqLen, topK: moe.topK, dModel: moe.dModel, gpus: group, topo });
    const b = m.moeAllToAll({ tokensPerGpu: moe.seqLen, topK: moe.topK, dModel: moe.dModel, gpus: group, topo, maxNodes: moe.maxNodes });
    free.push(+(a.time * 1e3).toFixed(3));
    limited.push(+(b.time * 1e3).toFixed(3));
    await lab.tick();
  }
  lab.plot({
    title: `${moe.name}-shaped MoE layer: all-to-all per GPU, ${moe.seqLen} tokens, top-${moe.topK}, hidden ${moe.dModel}`,
    x: epNodes,
    series: [{ name: 'unrestricted routing', values: free }, { name: `node-limited to ${moe.maxNodes} nodes`, values: limited }],
    xlabel: 'nodes the experts are spread over', ylabel: 'milliseconds per layer',
  });
  const wide = m.moeAllToAll({ tokensPerGpu: moe.seqLen, topK: moe.topK, dModel: moe.dModel, gpus: range(8 * topo.gpusPerNode), topo });
  const wideLimited = m.moeAllToAll({ tokensPerGpu: moe.seqLen, topK: moe.topK, dModel: moe.dModel, gpus: range(8 * topo.gpusPerNode), topo, maxNodes: moe.maxNodes });
  lab.log(`Over 8 nodes a token reaches ${wide.nodesPerToken.toFixed(2)} of them; the cap takes that to ${wideLimited.nodesPerToken.toFixed(2)} and cuts inter-node traffic from ${mb(wide.interNodeBytes)} to ${mb(wideLimited.interNodeBytes)} per GPU per layer.`);

  // ---------- 5. failures and checkpoints ----------
  const MTBF_HOURS = 50677; // per GPU; reproduces the Llama 3 paper's 419 unexpected interruptions in 54 days on 16,384 GPUs
  const ckptSeconds = (model.params * m.BYTES.total) / 10e9; // 1.1 TB of state to storage at an assumed ~10 GB/s aggregate
  const clusterSizes = [128, 512, 1024, 4096, 8192, 16384, 32768];
  const perDay = [], wasted = [], intervals = [];
  for (const n of clusterSizes) {
    const f = m.failuresPerDay(n, MTBF_HOURS);
    const jobMtbf = 86400 / f;
    const interval = m.youngInterval(jobMtbf, ckptSeconds);
    perDay.push(+f.toFixed(2));
    intervals.push(+(interval / 60).toFixed(1));
    wasted.push(+(m.wastedFraction({ mtbfSeconds: jobMtbf, checkpointSeconds: ckptSeconds, intervalSeconds: interval }) * 100).toFixed(2));
    await lab.tick();
  }
  lab.progress(0.95, 'reliability');
  lab.plot({
    title: `Failures and the checkpoint tax (per-GPU MTBF ${MTBF_HOURS} h, ${ckptSeconds.toFixed(0)} s to write ${gb(model.params * m.BYTES.total)})`,
    x: clusterSizes,
    series: [
      { name: 'interruptions per day', values: perDay },
      { name: "Young's checkpoint interval (minutes)", values: intervals },
      { name: 'wall clock wasted (%)', values: wasted },
    ],
    xlabel: 'GPUs in the job', ylabel: 'value', yscale: 'log',
  });
  const i16k = clusterSizes.indexOf(16384);

  const power = m.clusterPowerMW(gpus, topo.gpu);
  lab.done(`
Planned **${model.name}** (${fmt(model.params)} parameters, ${model.layers} layers) on **${gpus} ${topo.gpu.name}** — ${nodes} nodes, ${(gpus / (topo.gpusPerNode * topo.nodesPerPod))} pods, approximately **${power.toFixed(2)} MW** of facility power (the same accounting puts 10,000 H100s at approximately ${m.clusterPowerMW(10000, topo.gpu).toFixed(0)} MW).

**The winner:** \`tp=${best.layout.tp}, pp=${best.layout.pp}, dp=${best.layout.dp}\` at **${S(best.stepTime)} s** per step and **${GB(best.memory.total)} GB** per GPU — ${fmt(Math.round(best.tokensPerSec))} tokens/s across the cluster. Its tensor-parallel group is ${best.links.tp.tier === 'node' ? 'exactly one node, so its all-reduces stay on NVLink' : 'spread across nodes'}; its gradient all-reduce (${S(best.dpComm)} s) runs on the ${best.links.dp.name} and ${best.dpExposed <= 0 ? `hides entirely behind ${S(best.compute)} s of arithmetic` : `leaves ${S(best.dpExposed)} s exposed after overlapping with compute`}.

**Why not wider tensor parallelism:** \`tp=16\` needs only ${GB(tp16.memory.total)} GB per GPU against ${GB(tp8.memory.total)} GB, yet its ${model.layers * 4} all-reduces per micro-batch add up to **${S(tp16.tpComm)} s** per step instead of **${S(tp8.tpComm)} s** — the group straddles two nodes, so the same bytes move at approximately ${(topo.links.pod.bandwidth / 1e9).toFixed(0)} GB/s instead of ${(topo.links.node.bandwidth / 1e9).toFixed(0)} GB/s. Net result: ${((tp16.stepTime / tp8.stepTime - 1) * 100).toFixed(0)}% slower. Pure data parallelism (\`tp=1\`) would take ${S(pureDp.stepTime)} s${pureDp.stepTime < best.stepTime ? ', faster than the winner,' : ''} but wants ${GB(pureDp.memory.total)} GB per GPU and ${pureDp.fits ? 'still fits' : 'does not fit'}.

**On the wire:** a flat 16 GB all-reduce over 128 GPUs takes ${flat[sizes.indexOf(128)].toFixed(0)} ms; your two-level schedule does it in ${hier[sizes.indexOf(128)].toFixed(0)} ms, **${speedup.toFixed(2)}x** faster, because the spine carries ${gb(at128.phases[1].bytes)} rather than ${gb(gradBytes)}.

**Mixture of experts:** spreading ${moe.name}'s experts over 8 nodes costs ${free[epNodes.indexOf(8)].toFixed(2)} ms of all-to-all per layer; node-limited routing to ${moe.maxNodes} nodes brings that to ${limited[epNodes.indexOf(8)].toFixed(2)} ms, a ${((1 - limited[epNodes.indexOf(8)] / free[epNodes.indexOf(8)]) * 100).toFixed(0)}% cut, without changing which experts the token uses.

**Reliability:** at 16,384 GPUs this model predicts **${perDay[i16k].toFixed(1)} interruptions a day** (the Llama 3 paper reports 419 unexpected interruptions in 54 days, about 7.8 a day, roughly 78% of them hardware). Young's rule then says checkpoint every **${intervals[i16k].toFixed(0)} minutes**, which costs **${wasted[i16k].toFixed(1)}%** of wall clock. At ${gpus} GPUs the same job would lose only ${wasted[0].toFixed(2)}%.
`);
}
