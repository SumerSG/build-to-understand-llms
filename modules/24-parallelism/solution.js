// Module 24 — Data, tensor & pipeline parallelism (reference solution).
// A cost simulator: every quantity is a plain number in SI units (seconds, bytes, bytes/s, FLOP/s).
// Nothing here runs on a GPU; the formulas are the ones Megatron-LM, DeepSpeed and FSDP are built on.

// ---------- hardware and model constants (done for you) ----------

/**
 * One H100 SXM. flops is the dense bf16 tensor-core peak and memory the HBM3 capacity, both
 * approximately as listed in NVIDIA's H100 datasheet; mfu (model FLOPs utilisation) is a typical
 * large-scale training value from module 08, not a datasheet number.
 */
export const H100 = { name: 'H100 SXM', flops: 989e12, memory: 80e9, mfu: 0.4 };

/**
 * Links as alpha–beta models: `latency` is the fixed cost of one message in seconds, `bandwidth`
 * the sustained bytes per second one GPU gets in one direction. Both approximate: NVLink 4 is
 * ~450 GB/s per GPU per direction (NVIDIA H100 datasheet), InfiniBand NDR 400 is 400 Gb/s = 50 GB/s
 * per port (NVIDIA Quantum-2 datasheet). Latencies are round figures for a switched hop.
 */
export const NVLINK = { name: 'NVLink 4 (intra-node)', latency: 5e-6, bandwidth: 450e9 };
export const INFINIBAND = { name: 'InfiniBand NDR 400 (inter-node)', latency: 15e-6, bandwidth: 50e9 };

/**
 * A cluster: 8 GPUs per node joined by NVLink, nodes joined by InfiniBand.
 * `overlap` is the fraction of a step's compute during which gradient traffic can run in the
 * background (PyTorch DDP's gradient bucketing does this; 0.6 is a plausible value, not a measurement).
 */
export const HARDWARE = { gpu: H100, gpusPerNode: 8, intraNode: NVLINK, interNode: INFINIBAND, overlap: 0.6 };

/**
 * Llama-3-style configs. `batchSeqs` is the global batch in sequences, so the global batch in tokens
 * is batchSeqs × seqLen: 4.2M for the 70B here. (Llama 3 ramped its real batch up to ~16M tokens;
 * 4.2M keeps the numbers in this module easy to check by hand.)
 */
export const LLAMA3_70B = { name: 'Llama-3-70B', params: 70e9, layers: 80, dModel: 8192, dFF: 28672, seqLen: 8192, batchSeqs: 512 };
export const LLAMA3_8B = { name: 'Llama-3-8B', params: 8e9, layers: 32, dModel: 4096, dFF: 14336, seqLen: 8192, batchSeqs: 128 };

/**
 * Bytes per parameter of persistent training state, for bf16 weights and gradients with a fp32 AdamW
 * (fp32 master copy 4 + momentum 4 + variance 4 = 12). This is the 16 bytes/parameter of the ZeRO
 * paper, Rajbhandari et al. 2020, section 3.
 */
export const BYTES = { params: 2, grads: 2, optimizer: 12, total: 16 };

/**
 * Activation bytes kept for the backward pass, per token, per layer, per unit of dModel.
 * Korthikanti et al. 2022 give `s·b·h·(34 + 5·a·s/h)` bytes per transformer layer; the second term is
 * the materialised attention matrix, which FlashAttention removes, leaving the 34.
 */
export const ACT_BYTES_PER_TOKEN_LAYER_DIM = 34;

/**
 * ZeRO-3 (and FSDP) moves 1.5× the bytes of a plain data-parallel all-reduce: a reduce-scatter of the
 * gradients plus an all-gather of the parameters in each of forward and backward (Rajbhandari et al. 2020,
 * table 1). Stages 0–2 move exactly the all-reduce volume.
 */
export const ZERO3_COMM_FACTOR = 1.5;

// ---------- worked examples (done for you) ----------

/**
 * Time for one GPU to do the forward+backward FLOPs of `tokens` tokens through `params` parameters.
 * Module 08: training costs approximately 6 FLOPs per parameter per token. `gpu.mfu` scales the
 * datasheet peak down to what a real step achieves.
 * computeTime(8e9, 16384, H100) ≈ 1.99 s
 */
export function computeTime(params, tokens, gpu) {
  return (6 * params * tokens) / (gpu.flops * gpu.mfu);
}

/**
 * Alpha–beta cost of moving `bytes` over one link: a fixed latency plus bytes divided by bandwidth.
 * commTime(1e9, NVLINK) ≈ 0.00222 s (bandwidth-bound); commTime(1, NVLINK) = 5e-6 s (all latency).
 */
export function commTime(bytes, link) {
  return link.latency + bytes / link.bandwidth;
}

// ---------- step 1: the ring all-reduce ----------

/**
 * Bytes each GPU sends during a ring all-reduce of a `bytes`-sized buffer across n GPUs.
 * The ring does n−1 reduce-scatter steps and n−1 all-gather steps, each moving one chunk of
 * bytes/n, so each GPU sends 2(n−1)/n · bytes. n = 1 sends nothing.
 */
export function ringAllReduceBytes(bytes, n) {
  if (!(n >= 1)) throw new Error(`ringAllReduceBytes: n must be >= 1, got ${n}`);
  return ((2 * (n - 1)) / n) * bytes;
}

/**
 * Time of a ring all-reduce: the 2(n−1) steps are sequential, and each one pays the link's latency
 * plus one chunk of bytes/n over its bandwidth. n = 1 is free.
 */
export function ringAllReduceTime(bytes, n, link) {
  if (!(n >= 1)) throw new Error(`ringAllReduceTime: n must be >= 1, got ${n}`);
  if (n === 1) return 0;
  return 2 * (n - 1) * commTime(bytes / n, link);
}

// ---------- step 2: data parallelism ----------

/**
 * One data-parallel training step. Every GPU holds the full model, processes `tokensPerGpu` tokens,
 * then all-reduces its bf16 gradient (BYTES.grads per parameter) across the `dp` replicas.
 * `overlap` is the fraction of the compute time during which that all-reduce can run in the background.
 * Returns { compute, comm, exposed, step } in seconds.
 */
export function dataParallelStep({ params, tokensPerGpu, dp, gpu, link, overlap = 0 }) {
  const compute = computeTime(params, tokensPerGpu, gpu);
  const comm = ringAllReduceTime(BYTES.grads * params, dp, link);
  const hidden = Math.min(comm, overlap * compute);
  const exposed = comm - hidden;
  return { compute, comm, exposed, step: compute + exposed };
}

/**
 * Weak-scaling efficiency at `dp` replicas: the ideal step time (compute alone) over the real one.
 * 1.0 means the all-reduce is completely hidden; 0.5 means half the step is waiting for gradients.
 */
export function dpEfficiency(cfg, dp) {
  const r = dataParallelStep({ ...cfg, dp });
  return r.compute / r.step;
}

// ---------- step 3: tensor parallelism ----------

/**
 * Megatron-style sharding of one transformer layer's four big matrices across `tp` GPUs.
 * The first matrix of each pair (attn.qkv, mlp.fc) is split by COLUMNS, the second (attn.proj,
 * mlp.proj) by ROWS, so the pair needs no communication between its two matmuls.
 * Returns [{ name, split: 'column' | 'row', shape: [rows, cols] }] describing what ONE GPU holds.
 */
export function tpShards({ dModel, dFF }, tp) {
  if (!(tp >= 1)) throw new Error(`tpShards: tp must be >= 1, got ${tp}`);
  if (dModel % tp !== 0 || dFF % tp !== 0) throw new Error(`tpShards: dModel=${dModel} and dFF=${dFF} must both be divisible by tp=${tp}`);
  return [
    { name: 'attn.qkv', split: 'column', shape: [dModel, (3 * dModel) / tp] },
    { name: 'attn.proj', split: 'row', shape: [dModel / tp, dModel] },
    { name: 'mlp.fc', split: 'column', shape: [dModel, dFF / tp] },
    { name: 'mlp.proj', split: 'row', shape: [dFF / tp, dModel] },
  ];
}

/**
 * What one transformer layer costs in communication under tensor parallelism, for a micro-batch of
 * `tokens` tokens. Each row-parallel matmul leaves every GPU with a partial sum of the full
 * [tokens, dModel] activation, so it takes an all-reduce: one after attention and one after the MLP
 * in the forward pass, and one each in the backward pass — four per layer. tp = 1 needs none.
 * Returns { allReduces, bytesPerAllReduce, time }.
 */
export function tpLayerComm({ dModel, tokens }, tp, link) {
  const allReduces = tp > 1 ? 4 : 0;
  const bytesPerAllReduce = BYTES.params * tokens * dModel;
  return { allReduces, bytesPerAllReduce, time: allReduces * ringAllReduceTime(bytesPerAllReduce, tp, link) };
}

// ---------- step 4: pipeline parallelism ----------

/** Fraction of a pipeline step that the stages spend idle filling and draining: (p−1)/m. */
export function pipelineBubble(p, m) {
  if (!(p >= 1) || !(m >= 1)) throw new Error(`pipelineBubble: p and m must be >= 1, got p=${p}, m=${m}`);
  return (p - 1) / m;
}

/**
 * One pipeline-parallel step under the 1F1B schedule.
 *   modelTime — seconds for ONE worker to push the whole batch through ALL p stages (so p × stageTime).
 *   p — pipeline stages, m — micro-batches per step.
 *   activationBytesPerMicroBatch — activation memory one stage keeps for one micro-batch.
 * 1F1B starts the backward pass of micro-batch 1 as soon as it can, so a stage holds at most
 * min(p, m) micro-batches of activations; GPipe holds all m.
 * Returns { stageTime, bubble, step, activationBytes1F1B, activationBytesGPipe }.
 */
export function pipelineStep({ modelTime, p, m, activationBytesPerMicroBatch = 0 }) {
  const bubble = pipelineBubble(p, m);
  const stageTime = modelTime / p;
  return {
    stageTime,
    bubble,
    step: stageTime * (1 + bubble),
    activationBytes1F1B: Math.min(p, m) * activationBytesPerMicroBatch,
    activationBytesGPipe: m * activationBytesPerMicroBatch,
  };
}

// ---------- step 5: ZeRO / FSDP memory ----------

/**
 * Bytes per GPU of persistent training state for `params` parameters across `dp` data-parallel ranks.
 *   stage 0 — everything replicated: 16 bytes per parameter.
 *   stage 1 — optimizer state sharded (the 12 bytes).
 *   stage 2 — optimizer state and gradients sharded.
 *   stage 3 — optimizer state, gradients and parameters sharded (this is PyTorch FSDP).
 * Activations are not training state and are counted separately.
 * Returns { params, grads, optimizer, total }.
 */
export function zeroMemoryPerGpu(params, dp, stage) {
  if (![0, 1, 2, 3].includes(stage)) throw new Error(`zeroMemoryPerGpu: stage must be 0, 1, 2 or 3, got ${stage}`);
  if (!(dp >= 1)) throw new Error(`zeroMemoryPerGpu: dp must be >= 1, got ${dp}`);
  const optimizer = (BYTES.optimizer * params) / (stage >= 1 ? dp : 1);
  const grads = (BYTES.grads * params) / (stage >= 2 ? dp : 1);
  const p = (BYTES.params * params) / (stage >= 3 ? dp : 1);
  return { params: p, grads, optimizer, total: p + grads + optimizer };
}

// ---------- step 6: the planner ----------

/**
 * Simulate one training step of `model` on `gpus` GPUs under strategy { dp, tp, pp, zero, microBatchSeqs }.
 * TP always uses the intra-node link (it is only legal inside a node). DP uses the intra-node link
 * only when the whole tp × dp group fits in one node; PP likewise.
 * Returns { dp, tp, pp, zero, m, stepTime, memoryPerGpu, fits, tokensPerSec, breakdown, memory }.
 */
export function plan({ model, gpus, strategy, hardware = HARDWARE }) {
  const { dp, tp, pp, zero = 1, microBatchSeqs = 1 } = strategy;
  const { gpu, gpusPerNode, intraNode, interNode, overlap } = hardware;
  if (dp * tp * pp !== gpus) throw new Error(`plan: dp·tp·pp = ${dp * tp * pp} must equal gpus = ${gpus}`);
  if (tp > gpusPerNode) throw new Error(`plan: tp=${tp} exceeds the ${gpusPerNode} GPUs in a node; tensor parallelism must stay on the intra-node link`);
  if (model.layers % pp !== 0) throw new Error(`plan: ${model.layers} layers do not divide into pp=${pp} stages`);
  if (model.batchSeqs % (dp * microBatchSeqs) !== 0) throw new Error(`plan: a batch of ${model.batchSeqs} sequences does not split over dp=${dp} replicas of ${microBatchSeqs}-sequence micro-batches`);
  tpShards(model, tp); // throws if the matrices cannot be sharded

  const m = model.batchSeqs / (dp * microBatchSeqs);      // micro-batches one pipeline runs per step
  const microTokens = microBatchSeqs * model.seqLen;
  const tokens = model.batchSeqs * model.seqLen;          // global batch in tokens
  const layersPerStage = model.layers / pp;
  const shardParams = model.params / (tp * pp);           // parameters one GPU owns before ZeRO
  const dpLink = tp * dp <= gpusPerNode ? intraNode : interNode;
  const ppLink = gpus <= gpusPerNode ? intraNode : interNode;

  // --- time ---
  const compute = computeTime(model.params, tokens, gpu) / gpus;
  const tpComm = m * layersPerStage * tpLayerComm({ dModel: model.dModel, tokens: microTokens }, tp, intraNode).time;
  const actPerMicroBatch = (ACT_BYTES_PER_TOKEN_LAYER_DIM * microTokens * model.dModel * layersPerStage) / tp;
  const pipe = pipelineStep({ modelTime: (compute + tpComm) * pp, p: pp, m, activationBytesPerMicroBatch: actPerMicroBatch });
  const ppComm = pp > 1 ? 2 * (pp - 1) * commTime(BYTES.params * microTokens * model.dModel, ppLink) : 0;
  const dpComm = ringAllReduceTime(BYTES.grads * shardParams * (zero === 3 ? ZERO3_COMM_FACTOR : 1), dp, dpLink);
  const dpExposed = dpComm - Math.min(dpComm, overlap * compute);
  const stepTime = pipe.step + ppComm + dpExposed;

  // --- memory ---
  const state = zeroMemoryPerGpu(shardParams, dp, zero);
  const activations = pipe.activationBytes1F1B;
  const memoryPerGpu = state.total + activations;
  return {
    dp, tp, pp, zero, m, stepTime, memoryPerGpu,
    fits: memoryPerGpu <= gpu.memory,
    tokensPerSec: tokens / stepTime,
    breakdown: { compute, tpComm, bubble: pipe.step - (compute + tpComm), ppComm, dpComm, dpExposed },
    memory: { params: state.params, grads: state.grads, optimizer: state.optimizer, activations },
  };
}

/**
 * Every (dp, tp, pp, zero) layout that is legal for this model on this many GPUs:
 * the three degrees multiply to `gpus`, tp fits in a node and divides dModel and dFF,
 * pp divides the layer count, and the batch divides over dp. Micro-batches are one sequence each.
 */
export function enumerateStrategies(model, gpus, hardware = HARDWARE) {
  const out = [];
  for (let tp = 1; tp <= Math.min(gpus, hardware.gpusPerNode); tp++) {
    if (gpus % tp !== 0 || model.dModel % tp !== 0 || model.dFF % tp !== 0) continue;
    for (let pp = 1; pp <= gpus / tp; pp++) {
      if ((gpus / tp) % pp !== 0 || model.layers % pp !== 0) continue;
      const dp = gpus / (tp * pp);
      if (model.batchSeqs % dp !== 0) continue;
      for (const zero of [0, 1, 2, 3]) out.push({ dp, tp, pp, zero });
    }
  }
  return out;
}

/**
 * The fastest legal layout whose memory per GPU stays under `memoryCap` (default: the GPU's memory).
 * Ties on step time are broken towards the layout that uses less memory. null if nothing fits.
 */
export function bestPlan({ model, gpus, hardware = HARDWARE, memoryCap = hardware.gpu.memory }) {
  let best = null;
  for (const strategy of enumerateStrategies(model, gpus, hardware)) {
    const p = plan({ model, gpus, strategy, hardware });
    if (p.memoryPerGpu > memoryCap) continue;
    if (!best || p.stepTime < best.stepTime || (p.stepTime === best.stepTime && p.memoryPerGpu < best.memoryPerGpu)) best = p;
  }
  return best;
}
