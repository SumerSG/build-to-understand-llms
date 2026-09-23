// Module 25 — Nodes, interconnects & clusters (reference solution).
// Every quantity is a plain number in SI units: seconds, bytes, bytes/s, FLOP/s, watts.
// A GPU is an integer id in [0, gpus). The topology turns that id into a place in the machine.

// ---------- hardware constants (done for you) ----------

/**
 * One H100 SXM. `flops` is the dense bf16 tensor-core peak and `memory` the HBM3 capacity, both
 * approximately as listed in NVIDIA's H100 datasheet; `watts` is the 700 W SXM board power from the
 * same datasheet. `mfu` (model FLOPs utilisation) is a typical large-scale training value from
 * module 08, not a datasheet number.
 */
export const H100 = { name: 'H100 SXM', flops: 989e12, memory: 80e9, mfu: 0.4, watts: 700 };

/**
 * The four rungs of the bandwidth hierarchy, each an alpha–beta model: `latency` is the fixed cost of
 * one message in seconds, `bandwidth` the sustained bytes per second one GPU gets in one direction.
 * All figures are approximate.
 *   HBM       — on-package memory, ~3.35 TB/s (NVIDIA H100 SXM datasheet). No message ever crosses it.
 *   NVLINK    — NVLink 4 through the node's NVSwitches, ~900 GB/s bidirectional per GPU, so ~450 GB/s
 *               in one direction (NVIDIA H100 datasheet).
 *   IB_LEAF   — one InfiniBand NDR 400 port per GPU, 400 Gb/s = 50 GB/s, one leaf-switch hop
 *               (NVIDIA Quantum-2 datasheet).
 *   IB_SPINE  — the same port, but three hops through a spine that we make 2:1 oversubscribed, so one
 *               GPU sustains ~25 GB/s. Large rail-optimised fabrics are often built non-blocking;
 *               oversubscribing here makes the planner's choice visible.
 * Latencies are round figures for a switched hop, not measurements.
 */
export const HBM = { name: 'HBM3 (on-package)', tier: 'self', latency: 0, bandwidth: 3.35e12 };
export const NVLINK = { name: 'NVLink 4 / NVSwitch (in node)', tier: 'node', latency: 5e-6, bandwidth: 450e9 };
export const IB_LEAF = { name: 'InfiniBand NDR 400 (in pod)', tier: 'pod', latency: 15e-6, bandwidth: 50e9 };
export const IB_SPINE = { name: 'InfiniBand NDR 400 (across pods)', tier: 'cluster', latency: 25e-6, bandwidth: 25e9 };

/** The tiers from fastest to slowest. The index of a tier is how far apart two GPUs are. */
export const TIERS = ['self', 'node', 'pod', 'cluster'];

/**
 * A cluster: 8 H100s per node (an HGX/DGX H100 baseboard), 8 nodes per pod behind one leaf switch,
 * pods joined by the spine. `overlap` is the fraction of a step's compute during which gradient
 * traffic can run in the background (PyTorch DDP and Megatron-LM bucket gradients to get this;
 * 0.6 is a plausible value, not a measurement).
 */
export const CLUSTER = {
  name: 'H100 pod cluster',
  gpu: H100,
  gpusPerNode: 8,
  nodesPerPod: 8,
  links: { self: HBM, node: NVLINK, pod: IB_LEAF, cluster: IB_SPINE },
  overlap: 0.6,
};

/**
 * Llama-3-style dense config, identical to module 24 so the two planners can be compared.
 * `batchSeqs` is the global batch in sequences, so the global batch is 512 × 8192 = 4.19M tokens.
 */
export const LLAMA3_70B = { name: 'Llama-3-70B', params: 70e9, layers: 80, dModel: 8192, dFF: 28672, seqLen: 8192, batchSeqs: 512 };

/**
 * A DeepSeek-V3-shaped mixture of experts, for the all-to-all model in step 5. The DeepSeek-V3
 * technical report gives 671B total parameters, 37B activated per token, 61 layers, hidden size 7168,
 * 256 routed experts with top-8 routing, and node-limited routing that caps each token at 4 nodes.
 */
export const DEEPSEEK_V3 = { name: 'DeepSeek-V3', params: 671e9, activated: 37e9, layers: 61, dModel: 7168, experts: 256, topK: 8, maxNodes: 4, seqLen: 4096 };

/** Bytes of persistent training state per parameter: bf16 weights 2, bf16 gradients 2, fp32 AdamW 12. */
export const BYTES = { params: 2, grads: 2, optimizer: 12, total: 16 };

/**
 * Activation bytes kept for the backward pass per token, per layer, per unit of dModel.
 * Korthikanti et al. 2022 give `s·b·h·(34 + 5·a·s/h)` bytes per transformer layer; FlashAttention
 * removes the second term, leaving the 34.
 */
export const ACT_BYTES_PER_TOKEN_LAYER_DIM = 34;

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/**
 * Alpha–beta cost of moving `bytes` over one link: a fixed latency plus bytes divided by bandwidth.
 * This is the whole cost model — every collective below is built out of calls to it.
 * commTime(1e9, NVLINK) ≈ 0.00222 s (bandwidth-bound); commTime(1, NVLINK) = 5e-6 s (all latency).
 */
export function commTime(bytes, link) {
  return link.latency + bytes / link.bandwidth;
}

/**
 * Ring all-reduce over n GPUs that all share one link — the collective you built in module 24.
 * 2(n−1) sequential steps, each moving one chunk of bytes/n. n = 1 is free.
 */
export function ringAllReduceTime(bytes, n, link) {
  if (!(n >= 1)) throw new Error(`ringAllReduceTime: n must be >= 1, got ${n}`);
  if (n === 1) return 0;
  return 2 * (n - 1) * commTime(bytes / n, link);
}

/**
 * Bytes one GPU holds under a Megatron-style 3D layout with ZeRO-1 (optimizer state sharded over the
 * data-parallel ranks), from module 24. Tensor and pipeline parallelism cut the parameters a GPU owns;
 * 1F1B keeps min(pp, m) micro-batches of activations alive at once.
 * Returns { params, grads, optimizer, activations, total } in bytes.
 */
export function memoryPerGpu({ model, layout, microBatchSeqs = 1 }) {
  const { tp, pp, dp } = layout;
  const shardParams = model.params / (tp * pp);
  const layersPerStage = model.layers / pp;
  const microTokens = microBatchSeqs * model.seqLen;
  const m = model.batchSeqs / (dp * microBatchSeqs);
  const params = BYTES.params * shardParams;
  const grads = BYTES.grads * shardParams;
  const optimizer = (BYTES.optimizer * shardParams) / dp;
  const activations = (Math.min(pp, m) * ACT_BYTES_PER_TOKEN_LAYER_DIM * microTokens * model.dModel * layersPerStage) / tp;
  return { params, grads, optimizer, activations, total: params + grads + optimizer + activations };
}

/**
 * Megawatts drawn by `gpus` accelerators, including everything else in the hall.
 * `pue` (power usage effectiveness) is the ratio of facility power to IT power; the Uptime Institute's
 * annual survey puts a typical large data centre near 1.5, and hyperscalers report closer to 1.1.
 * We also charge 1.8× the GPU board power for the CPUs, NICs, NVSwitches, storage and fans in the node:
 * NVIDIA's DGX H100 datasheet gives ~10.2 kW maximum for 8 GPUs, about 1.8 × 8 × 700 W.
 * clusterPowerMW(10000, H100) ≈ 16.4 MW.
 */
export function clusterPowerMW(gpus, gpu, pue = 1.3) {
  return (gpus * gpu.watts * 1.8 * pue) / 1e6;
}

// ---------- step 1: the bandwidth hierarchy ----------

/**
 * Where GPU `gpu` sits in the machine: which node, which pod, and which slot inside the node.
 * GPUs are numbered so that a node is a contiguous block of `gpusPerNode` ids and a pod a contiguous
 * block of `gpusPerNode · nodesPerPod` ids. Throws on a negative or non-integer id.
 */
export function location(gpu, topo) {
  if (!Number.isInteger(gpu) || gpu < 0) throw new Error(`location: gpu must be a non-negative integer, got ${gpu}`);
  const perPod = topo.gpusPerNode * topo.nodesPerPod;
  return {
    gpu,
    node: Math.floor(gpu / topo.gpusPerNode),
    pod: Math.floor(gpu / perPod),
    slot: gpu % topo.gpusPerNode,
  };
}

/**
 * The link two GPUs must use to talk: their own memory if they are the same GPU, NVLink if they share
 * a node, the leaf switch if they share a pod, the spine otherwise. Returns one of `topo.links`.
 */
export function linkBetween(a, b, topo) {
  const la = location(a, topo), lb = location(b, topo);
  if (la.gpu === lb.gpu) return topo.links.self;
  if (la.node === lb.node) return topo.links.node;
  if (la.pod === lb.pod) return topo.links.pod;
  return topo.links.cluster;
}

/**
 * The slowest link any pair in `gpus` must use — the link that sets the cost of a collective over
 * that set. The tiers form a hierarchy (two GPUs in the same node are in the same pod), so the
 * slowest pair always includes the first GPU and one comparison per GPU is enough.
 * A single GPU communicates with nobody, so its slowest link is `topo.links.self`.
 */
export function slowestLink(gpus, topo) {
  if (!gpus.length) throw new Error('slowestLink: needs at least one GPU, got an empty group');
  let worst = topo.links.self;
  for (const g of gpus) {
    const link = linkBetween(gpus[0], g, topo);
    if (TIERS.indexOf(link.tier) > TIERS.indexOf(worst.tier)) worst = link;
  }
  return worst;
}

// ---------- step 2: collectives on the hierarchy ----------

/**
 * A flat ring all-reduce: one ring through every GPU in the group, every hop priced at the group's
 * slowest link. This is what NCCL falls back to when it cannot see the hierarchy.
 */
export function flatAllReduceTime(bytes, gpus, topo) {
  return ringAllReduceTime(bytes, gpus.length, slowestLink(gpus, topo));
}

/**
 * The hierarchical (two-level) all-reduce every real library uses across nodes:
 *   1. reduce-scatter inside each node over NVLink — every GPU ends up with 1/g of the buffer, reduced
 *      across its node. g−1 sequential steps of bytes/g.
 *   2. all-reduce those bytes/g chunks across the nodes over the slow link. The g GPUs of a node run
 *      g such rings at once (one per slot, each on its own NIC), so the phase costs one of them.
 *   3. all-gather inside each node over NVLink to put the full result back on every GPU.
 * The slow link therefore carries bytes/g instead of bytes — that factor g is the whole point.
 * Requires the same number of GPUs on every node it touches.
 * Returns { nodes, gpusPerNode, crossLink, phases: [{ name, bytes, link, time }], time }.
 */
export function hierarchicalAllReduce(bytes, gpus, topo) {
  if (!gpus.length) throw new Error('hierarchicalAllReduce: needs at least one GPU, got an empty group');
  const byNode = new Map();
  for (const g of gpus) {
    const n = location(g, topo).node;
    if (!byNode.has(n)) byNode.set(n, []);
    byNode.get(n).push(g);
  }
  const nodes = [...byNode.keys()].sort((x, y) => x - y);
  const g = byNode.get(nodes[0]).length;
  for (const n of nodes) {
    if (byNode.get(n).length !== g) {
      throw new Error(`hierarchicalAllReduce: node ${nodes[0]} holds ${g} GPUs of the group but node ${n} holds ${byNode.get(n).length}; the group must be spread evenly over nodes`);
    }
  }
  const leaders = nodes.map((n) => byNode.get(n)[0]);
  const crossLink = slowestLink(leaders, topo);
  const nodeLink = topo.links.node;
  const chunk = bytes / g;
  const phases = [];
  if (g > 1) phases.push({ name: 'reduce-scatter in node', bytes: chunk, link: nodeLink, time: (g - 1) * commTime(chunk, nodeLink) });
  if (nodes.length > 1) phases.push({ name: 'all-reduce across nodes', bytes: chunk, link: crossLink, time: ringAllReduceTime(chunk, nodes.length, crossLink) });
  if (g > 1) phases.push({ name: 'all-gather in node', bytes: chunk, link: nodeLink, time: (g - 1) * commTime(chunk, nodeLink) });
  let time = 0;
  for (const p of phases) time += p.time;
  return { nodes: nodes.length, gpusPerNode: g, crossLink, phases, time };
}

/**
 * What an all-reduce of `bytes` over `gpus` actually costs: the better of the flat ring and the
 * hierarchical schedule. One GPU costs nothing.
 */
export function allReduceTime(bytes, gpus, topo) {
  if (gpus.length <= 1) return 0;
  return Math.min(flatAllReduceTime(bytes, gpus, topo), hierarchicalAllReduce(bytes, gpus, topo).time);
}

// ---------- step 3: placing ranks on the hierarchy ----------

/**
 * Which GPUs share a parallel group with global rank `rank`, under layout { tp, pp, dp }.
 * Ranks are laid out with tensor parallelism innermost, then data parallelism, then pipeline
 * parallelism — Megatron-LM's default order:
 *     rank = (ppIndex · dp + dpIndex) · tp + tpIndex
 * so a tensor-parallel group is `tp` consecutive ids (which land inside one node when tp ≤ 8),
 * a data-parallel group is spaced `tp` apart, and a pipeline group `tp · dp` apart.
 * `kind` is 'tp', 'dp' or 'pp'. Returns the group's ids in ascending order, including `rank` itself.
 */
export function groupFor(kind, rank, layout) {
  const { tp, pp, dp } = layout;
  const total = tp * pp * dp;
  if (!Number.isInteger(rank) || rank < 0 || rank >= total) throw new Error(`groupFor: rank must be an integer in [0, ${total}), got ${rank}`);
  const tpIndex = rank % tp;
  const dpIndex = Math.floor(rank / tp) % dp;
  const ppIndex = Math.floor(rank / (tp * dp));
  const at = (t, d, p) => (p * dp + d) * tp + t;
  const out = [];
  if (kind === 'tp') for (let i = 0; i < tp; i++) out.push(at(i, dpIndex, ppIndex));
  else if (kind === 'dp') for (let i = 0; i < dp; i++) out.push(at(tpIndex, i, ppIndex));
  else if (kind === 'pp') for (let i = 0; i < pp; i++) out.push(at(tpIndex, dpIndex, i));
  else throw new Error(`groupFor: kind must be 'tp', 'dp' or 'pp', got ${JSON.stringify(kind)}`);
  return out.sort((a, b) => a - b);
}

/**
 * The slowest link used by ANY group of this kind under this layout — the link that decides what
 * that axis of parallelism costs. Groups of one kind are congruent, but a group can straddle a node
 * or pod boundary depending on where it starts, so check every distinct group.
 */
export function groupLink(kind, layout, topo) {
  const total = layout.tp * layout.pp * layout.dp;
  const seen = new Set();
  let worst = topo.links.self;
  for (let rank = 0; rank < total; rank++) {
    const group = groupFor(kind, rank, layout);
    if (seen.has(group[0])) continue;
    seen.add(group[0]);
    const link = slowestLink(group, topo);
    if (TIERS.indexOf(link.tier) > TIERS.indexOf(worst.tier)) worst = link;
  }
  return worst;
}

// ---------- step 4: the placement planner ----------

/**
 * One training step of `model` on `gpus` GPUs under layout { tp, pp, dp }, with every collective
 * priced on the link its group actually lands on.
 *   compute    — 6 FLOPs per parameter per token (module 08), divided over the whole cluster.
 *   tpComm     — 4 all-reduces of the [microTokens, dModel] activation per layer per micro-batch.
 *   bubble     — the pipeline fill and drain, (pp−1)/m of a stage (module 24).
 *   ppComm     — one activation across each of the pp−1 stage boundaries, forward and backward,
 *                for every micro-batch.
 *   dpExposed  — the gradient all-reduce that does not hide behind compute.
 * Returns { layout, m, compute, tpComm, bubble, ppComm, dpComm, dpExposed, stepTime, tokensPerSec,
 *           links: { tp, pp, dp }, memory, fits }.
 */
export function layoutStepTime({ model, gpus, layout, topo = CLUSTER, microBatchSeqs = 1, memoryCap = topo.gpu.memory }) {
  const { tp, pp, dp } = layout;
  if (tp * pp * dp !== gpus) throw new Error(`layoutStepTime: tp·pp·dp = ${tp * pp * dp} must equal gpus = ${gpus}`);
  if (model.layers % pp !== 0) throw new Error(`layoutStepTime: ${model.layers} layers do not divide into pp = ${pp} stages`);
  if (model.dModel % tp !== 0 || model.dFF % tp !== 0) throw new Error(`layoutStepTime: dModel = ${model.dModel} and dFF = ${model.dFF} must both be divisible by tp = ${tp}`);
  if (model.batchSeqs % (dp * microBatchSeqs) !== 0) throw new Error(`layoutStepTime: a batch of ${model.batchSeqs} sequences does not split over dp = ${dp} replicas of ${microBatchSeqs}-sequence micro-batches`);

  const m = model.batchSeqs / (dp * microBatchSeqs);
  const microTokens = microBatchSeqs * model.seqLen;
  const tokens = model.batchSeqs * model.seqLen;
  const layersPerStage = model.layers / pp;
  const shardParams = model.params / (tp * pp);
  const actBytes = BYTES.params * microTokens * model.dModel;

  const compute = (6 * model.params * tokens) / (topo.gpu.flops * topo.gpu.mfu * gpus);
  const tpComm = tp > 1 ? m * layersPerStage * 4 * allReduceTime(actBytes, groupFor('tp', 0, layout), topo) : 0;
  const bubble = ((compute + tpComm) * (pp - 1)) / m;
  const ppComm = pp > 1 ? m * 2 * (pp - 1) * commTime(actBytes, groupLink('pp', layout, topo)) : 0;
  const dpComm = dp > 1 ? allReduceTime(BYTES.grads * shardParams, groupFor('dp', 0, layout), topo) : 0;
  const dpExposed = Math.max(0, dpComm - topo.overlap * compute);
  const stepTime = compute + tpComm + bubble + ppComm + dpExposed;
  const memory = memoryPerGpu({ model, layout, microBatchSeqs });
  return {
    layout: { tp, pp, dp },
    m,
    compute,
    tpComm,
    bubble,
    ppComm,
    dpComm,
    dpExposed,
    stepTime,
    tokensPerSec: tokens / stepTime,
    links: { tp: groupLink('tp', layout, topo), pp: groupLink('pp', layout, topo), dp: groupLink('dp', layout, topo) },
    memory,
    fits: memory.total <= memoryCap,
  };
}

/**
 * Every legal layout of `model` on `gpus` GPUs, priced and sorted: the ones that fit in memory first,
 * fastest first, then the ones that do not fit. A layout is legal when tp·pp·dp = gpus, tp divides
 * dModel and dFF, pp divides the layer count, and the batch divides over dp.
 * Micro-batches are one sequence each.
 */
export function rankLayouts({ model, gpus, topo = CLUSTER, microBatchSeqs = 1, memoryCap = topo.gpu.memory }) {
  const out = [];
  for (let tp = 1; tp <= gpus; tp++) {
    if (gpus % tp !== 0 || model.dModel % tp !== 0 || model.dFF % tp !== 0) continue;
    for (let pp = 1; pp <= gpus / tp; pp++) {
      if ((gpus / tp) % pp !== 0 || model.layers % pp !== 0) continue;
      const dp = gpus / (tp * pp);
      if (model.batchSeqs % (dp * microBatchSeqs) !== 0) continue;
      out.push(layoutStepTime({ model, gpus, layout: { tp, pp, dp }, topo, microBatchSeqs, memoryCap }));
    }
  }
  out.sort((a, b) => (a.fits === b.fits ? a.stepTime - b.stepTime : a.fits ? -1 : 1));
  return out;
}

// ---------- step 5: mixture-of-experts all-to-all ----------

/**
 * How many distinct nodes one token's `topK` expert copies have to reach, when the experts are spread
 * uniformly over `nodes` nodes. Each of the topK choices lands on a given node with probability
 * 1/nodes, so the expected number of nodes touched is `nodes·(1 − (1 − 1/nodes)^topK)`.
 * `maxNodes` is DeepSeek-V3's node-limited routing: the router is forbidden from picking experts on
 * more than that many nodes, so the answer is capped.
 */
export function expectedNodesPerToken(topK, nodes, maxNodes = Infinity) {
  if (!(topK >= 1)) throw new Error(`expectedNodesPerToken: topK must be >= 1, got ${topK}`);
  if (!(nodes >= 1)) throw new Error(`expectedNodesPerToken: nodes must be >= 1, got ${nodes}`);
  const expected = nodes * (1 - Math.pow(1 - 1 / nodes, topK));
  return Math.min(expected, nodes, topK, maxNodes);
}

/**
 * The all-to-all one MoE layer costs one GPU, in bytes and seconds. `gpus` is the expert-parallel
 * group (the GPUs the experts are spread over).
 *   Dispatch sends each token's hidden vector to the experts that want it; combine sends the results
 *   back, so every byte is paid twice — that is the leading 2.
 *   Across nodes, DeepSeek-V3 sends ONE copy per destination node and fans out over NVLink inside it,
 *   so the inter-node volume is the expected node count, not topK. Of the nodes a token reaches,
 *   a fraction (nodes−1)/nodes is remote.
 *   Inside a node every one of the topK copies is delivered over NVLink.
 *   `capacityFactor` ≥ 1 is GShard's expert capacity (Lepikhin et al. 2020): experts accept
 *   capacityFactor · tokens/experts tokens and drop the rest, so the traffic scales with it.
 * A switch delivers all pairs at once, so the cost is one GPU's own egress, charged one latency per
 * phase — not the sum over peers.
 * Returns { nodes, nodesPerToken, interNodeBytes, intraNodeBytes, interNodeTime, intraNodeTime, time }.
 */
export function moeAllToAll({ tokensPerGpu, topK, dModel, gpus, topo = CLUSTER, bytesPerElement = 2, capacityFactor = 1, maxNodes = Infinity }) {
  if (!gpus.length) throw new Error('moeAllToAll: the expert-parallel group must contain at least one GPU');
  const nodeIds = new Set(gpus.map((g) => location(g, topo).node));
  const nodes = nodeIds.size;
  const nodesPerToken = expectedNodesPerToken(topK, nodes, maxNodes);
  const perToken = 2 * dModel * bytesPerElement * capacityFactor;
  const remoteShare = (nodes - 1) / nodes;
  const interNodeBytes = tokensPerGpu * perToken * nodesPerToken * remoteShare;
  const intraNodeBytes = tokensPerGpu * perToken * topK;
  const leaders = [...nodeIds].sort((a, b) => a - b).map((n) => gpus.find((g) => location(g, topo).node === n));
  const crossLink = nodes > 1 ? slowestLink(leaders, topo) : topo.links.node;
  const interNodeTime = nodes > 1 ? commTime(interNodeBytes, crossLink) : 0;
  const intraNodeTime = commTime(intraNodeBytes, topo.links.node);
  return { nodes, nodesPerToken, interNodeBytes, intraNodeBytes, interNodeTime, intraNodeTime, time: interNodeTime + intraNodeTime };
}

// ---------- step 6: failures and checkpoints ----------

/**
 * Expected hardware interruptions per day for `gpus` accelerators whose individual mean time between
 * failures is `mtbfHours`. Failures are independent, so the rates add: gpus · 24 / mtbfHours.
 */
export function failuresPerDay(gpus, mtbfHours) {
  if (!(gpus >= 0)) throw new Error(`failuresPerDay: gpus must be >= 0, got ${gpus}`);
  if (!(mtbfHours > 0)) throw new Error(`failuresPerDay: mtbfHours must be > 0, got ${mtbfHours}`);
  return (gpus * 24) / mtbfHours;
}

/**
 * Young's optimal checkpoint interval (Young 1974; Daly 2006 gives the higher-order version):
 * `sqrt(2 · mtbf · checkpointTime)` seconds, where `mtbf` is the mean time between failures of the
 * WHOLE job. Checkpoint more often and you pay for writes; less often and you redo more work.
 */
export function youngInterval(mtbfSeconds, checkpointSeconds) {
  if (!(mtbfSeconds > 0)) throw new Error(`youngInterval: mtbfSeconds must be > 0, got ${mtbfSeconds}`);
  if (!(checkpointSeconds >= 0)) throw new Error(`youngInterval: checkpointSeconds must be >= 0, got ${checkpointSeconds}`);
  return Math.sqrt(2 * mtbfSeconds * checkpointSeconds);
}

/**
 * Fraction of wall-clock time the job does NOT spend making progress, for a given checkpoint interval:
 *   checkpointSeconds / intervalSeconds        — the writes themselves
 * + intervalSeconds / (2 · mtbfSeconds)        — on average half an interval of work is lost per failure
 * This is minimised exactly at `youngInterval`, where it equals `sqrt(2·checkpointSeconds/mtbfSeconds)`.
 */
export function wastedFraction({ mtbfSeconds, checkpointSeconds, intervalSeconds }) {
  if (!(intervalSeconds > 0)) throw new Error(`wastedFraction: intervalSeconds must be > 0, got ${intervalSeconds}`);
  if (!(mtbfSeconds > 0)) throw new Error(`wastedFraction: mtbfSeconds must be > 0, got ${mtbfSeconds}`);
  return checkpointSeconds / intervalSeconds + intervalSeconds / (2 * mtbfSeconds);
}
