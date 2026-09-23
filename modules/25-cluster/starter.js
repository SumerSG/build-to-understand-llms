// Nodes, interconnects & clusters.
// Every quantity is a plain number in SI units: seconds, bytes, bytes/s, FLOP/s, watts.
// A GPU is an integer id in [0, gpus). The topology turns that id into a place in the machine.
// Everything below the "worked examples" line is yours to implement.

// ---------- hardware constants (done for you) ----------

/**
 * One H100 SXM. `flops` is the dense bf16 tensor-core peak and `memory` the HBM3 capacity, both
 * approximately as listed in NVIDIA's H100 datasheet; `watts` is the 700 W SXM board power from the
 * same datasheet. `mfu` (model FLOPs utilisation) is a typical large-scale training value from
 * the scaling-laws module, not a datasheet number.
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
 * Llama-3-style dense config, identical to the parallelism module so the two planners can be compared.
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
 * Ring all-reduce over n GPUs that all share one link — the collective you built in the parallelism module.
 * 2(n−1) sequential steps, each moving one chunk of bytes/n. n = 1 is free.
 */
export function ringAllReduceTime(bytes, n, link) {
  if (!(n >= 1)) throw new Error(`ringAllReduceTime: n must be >= 1, got ${n}`);
  if (n === 1) return 0;
  return 2 * (n - 1) * commTime(bytes / n, link);
}

/**
 * Bytes one GPU holds under a Megatron-style 3D layout with ZeRO-1 (optimizer state sharded over the
 * data-parallel ranks), from the parallelism module. Tensor and pipeline parallelism cut the parameters a GPU owns;
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
 * Where GPU `gpu` sits in the machine. Return { gpu, node, pod, slot }, where `slot` is the position
 * inside its node. GPUs are numbered so that a node is a contiguous block of `topo.gpusPerNode` ids
 * and a pod a contiguous block of `topo.gpusPerNode * topo.nodesPerPod` ids.
 * Throw if `gpu` is negative or not an integer.
 */
export function location(gpu, topo) {
  if (!Number.isInteger(gpu) || gpu < 0) throw new Error(`location: gpu must be a non-negative integer, got ${gpu}`);
  const node = Math.floor(gpu / topo.gpusPerNode);
  // TODO: step 1 — a pod is a contiguous block of topo.gpusPerNode * topo.nodesPerPod ids,
  // and slot is the position inside the node.
  const pod = 0;
  const slot = 0;
  return { gpu, node, pod, slot };
}

/**
 * The link two GPUs must use to talk. Return one of `topo.links`: `self` if they are the same GPU,
 * `node` if they share a node, `pod` if they share a pod, `cluster` otherwise.
 */
export function linkBetween(a, b, topo) {
  // TODO: step 1
  return topo.links.self;
}

/**
 * The slowest link any pair in `gpus` (an array of ids) must use — the link that sets the cost of a
 * collective over that set. A group of one uses `topo.links.self`. Throw on an empty group.
 * `TIERS` tells you which of two links is the slower: compare `TIERS.indexOf(link.tier)`.
 */
export function slowestLink(gpus, topo) {
  // TODO: step 1
  return topo.links.self;
}

// ---------- step 2: collectives on the hierarchy ----------

/**
 * A flat ring all-reduce: one ring through every GPU in the group, every hop priced at the group's
 * slowest link. Build it out of `ringAllReduceTime` and `slowestLink`.
 */
export function flatAllReduceTime(bytes, gpus, topo) {
  // TODO: step 2
  return 0;
}

/**
 * The hierarchical (two-level) all-reduce, in three sequential phases:
 *   1. reduce-scatter inside each node over `topo.links.node`: g−1 steps of bytes/g, where g is the
 *      number of GPUs of the group on one node.
 *   2. all-reduce the bytes/g chunks across the nodes over the slowest link between the nodes (find
 *      it from one representative GPU per node). The g GPUs of a node run g such rings at once, one
 *      per slot, each on its own NIC, so the phase costs one ring of bytes/g over the node count.
 *   3. all-gather inside each node, the mirror of phase 1.
 * Skip a phase that has nothing to do (g = 1, or a single node). Throw if the group does not have
 * the same number of GPUs on every node it touches.
 * Return { nodes, gpusPerNode, crossLink, phases: [{ name, bytes, link, time }], time }.
 */
export function hierarchicalAllReduce(bytes, gpus, topo) {
  // TODO: step 2
  return { nodes: 1, gpusPerNode: 1, crossLink: topo.links.self, phases: [], time: 0 };
}

/**
 * What an all-reduce really costs: the better of the two schedules above. One GPU costs nothing.
 * If the group holds a different number of GPUs on different nodes, the hierarchical schedule does
 * not exist (it throws), so return the flat ring's time instead of letting the error escape.
 */
export function allReduceTime(bytes, gpus, topo) {
  // TODO: step 2
  return 0;
}

// ---------- step 3: placing ranks on the hierarchy ----------

/**
 * Which GPUs share a parallel group with global rank `rank`, under layout { tp, pp, dp }.
 * Megatron-LM's default rank order puts tensor parallelism innermost, then data, then pipeline:
 *     rank = (ppIndex * dp + dpIndex) * tp + tpIndex
 * `kind` is 'tp', 'dp' or 'pp'. Return the group's ids in ascending order, including `rank` itself.
 * Throw on an unknown kind or a rank outside [0, tp*pp*dp).
 */
export function groupFor(kind, rank, layout) {
  // TODO: step 3
  return [rank];
}

/**
 * The slowest link used by ANY group of this kind under this layout. Groups of one kind all have the
 * same shape, but where a group starts decides whether it straddles a node or pod boundary, so look
 * at every distinct group, not only the one containing rank 0.
 */
export function groupLink(kind, layout, topo) {
  // TODO: step 3
  return topo.links.self;
}

// ---------- step 4: the placement planner ----------

/**
 * One training step of `model` on `gpus` GPUs under layout { tp, pp, dp }, with every collective
 * priced on the link its group actually lands on.
 *   compute    — 6 FLOPs per parameter per token (the scaling-laws module), divided over the whole cluster:
 *                `6 * model.params * tokens / (topo.gpu.flops * topo.gpu.mfu * gpus)`.
 *   tpComm     — m micro-batches x layersPerStage x 4 all-reduces of the [microTokens, dModel]
 *                activation (BYTES.params bytes each), over the tensor-parallel group. Zero if tp = 1.
 *   bubble     — `(compute + tpComm) * (pp - 1) / m` (the parallelism module).
 *   ppComm     — one activation across each of the pp−1 boundaries, forward and backward, per
 *                micro-batch: `m * 2 * (pp - 1) * commTime(actBytes, ppLink)`. Zero if pp = 1.
 *   dpComm     — an all-reduce of BYTES.grads per owned parameter over the data-parallel group.
 *   dpExposed  — `max(0, dpComm - topo.overlap * compute)`.
 * with `tokens = model.batchSeqs * model.seqLen` (the global batch in tokens),
 * `m = model.batchSeqs / (dp * microBatchSeqs)` micro-batches, `microTokens = microBatchSeqs *
 * model.seqLen`, `layersPerStage = model.layers / pp` and `shardParams = model.params / (tp * pp)`.
 * tokensPerSec is `tokens / stepTime`.
 * stepTime is compute + tpComm + bubble + ppComm + dpExposed.
 * Throw unless tp*pp*dp === gpus, pp divides model.layers, tp divides model.dModel and model.dFF,
 * and dp*microBatchSeqs divides model.batchSeqs.
 * Return { layout, m, compute, tpComm, bubble, ppComm, dpComm, dpExposed, stepTime, tokensPerSec,
 *          links: { tp, pp, dp }, memory, fits }.
 */
export function layoutStepTime({ model, gpus, layout, topo = CLUSTER, microBatchSeqs = 1, memoryCap = topo.gpu.memory }) {
  // TODO: step 4
  return { layout, m: 1, compute: 0, tpComm: 0, bubble: 0, ppComm: 0, dpComm: 0, dpExposed: 0, stepTime: 0, tokensPerSec: 0, links: {}, memory: null, fits: false };
}

/**
 * Every legal layout of `model` on `gpus` GPUs, priced and sorted: the ones that fit in memory first,
 * fastest first, then the ones that do not fit. A layout is legal when tp*pp*dp = gpus, tp divides
 * dModel and dFF, pp divides the layer count, and the batch divides over dp.
 */
export function rankLayouts({ model, gpus, topo = CLUSTER, microBatchSeqs = 1, memoryCap = topo.gpu.memory }) {
  // TODO: step 4
  return [];
}

// ---------- step 5: mixture-of-experts all-to-all ----------

/**
 * How many distinct nodes one token's `topK` expert copies have to reach, when the experts are spread
 * uniformly over `nodes` nodes: `nodes * (1 - (1 - 1/nodes)^topK)`, never more than `topK`, never more
 * than `nodes`, and never more than `maxNodes` (DeepSeek-V3's node-limited routing).
 * Throw if topK or nodes is below 1.
 */
export function expectedNodesPerToken(topK, nodes, maxNodes = Infinity) {
  // TODO: step 5
  return 1;
}

/**
 * The all-to-all one MoE layer costs one GPU. `gpus` is the expert-parallel group.
 * Dispatch sends each token's hidden vector out and combine sends the result back, so every byte is
 * paid twice: `perToken = 2 * dModel * bytesPerElement * capacityFactor` per copy.
 *   intraNodeBytes = tokensPerGpu * perToken * topK          (every copy is delivered over NVLink)
 *   interNodeBytes = tokensPerGpu * perToken * nodesPerToken * (nodes - 1) / nodes
 * — one copy per DESTINATION NODE, which then fans out over NVLink inside it, and of the nodes a
 * token reaches, a fraction (nodes−1)/nodes is remote.
 * A switch delivers all pairs at once, so charge each phase one `commTime` of this GPU's own egress:
 * the inter-node phase on the slowest link between the nodes' representatives (zero if there is only
 * one node), the intra-node phase on `topo.links.node` (always paid, even on one node). The phases
 * run one after the other: `time = interNodeTime + intraNodeTime`. Throw on an empty group.
 * Return { nodes, nodesPerToken, interNodeBytes, intraNodeBytes, interNodeTime, intraNodeTime, time }.
 */
export function moeAllToAll({ tokensPerGpu, topK, dModel, gpus, topo = CLUSTER, bytesPerElement = 2, capacityFactor = 1, maxNodes = Infinity }) {
  // TODO: step 5
  return { nodes: 1, nodesPerToken: 1, interNodeBytes: 0, intraNodeBytes: 0, interNodeTime: 0, intraNodeTime: 0, time: 0 };
}

// ---------- step 6: failures and checkpoints ----------

/**
 * Expected hardware interruptions per day for `gpus` accelerators whose individual mean time between
 * failures is `mtbfHours`. Failures are independent, so the rates add.
 * Throw if gpus is negative or mtbfHours is not positive.
 */
export function failuresPerDay(gpus, mtbfHours) {
  // TODO: step 6
  return 0;
}

/**
 * Young's optimal checkpoint interval in seconds (Young 1974), where `mtbfSeconds` is the mean time
 * between failures of the WHOLE job. Throw if mtbfSeconds is not positive or checkpointSeconds is negative.
 */
export function youngInterval(mtbfSeconds, checkpointSeconds) {
  // TODO: step 6
  return 0;
}

/**
 * Fraction of wall-clock time the job does NOT spend making progress at a given checkpoint interval:
 * the writes themselves, plus the work redone after a failure (on average half an interval).
 * Throw if intervalSeconds or mtbfSeconds is not positive.
 */
export function wastedFraction({ mtbfSeconds, checkpointSeconds, intervalSeconds }) {
  // TODO: step 6
  return 0;
}
