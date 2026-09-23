// Module 24 — Data, tensor & pipeline parallelism.
// A cost simulator. Every quantity is a plain number in SI units: seconds, bytes, bytes/s, FLOP/s.
// Nothing here runs on a GPU; you are writing the formulas that Megatron-LM, DeepSpeed and FSDP are built on.
//
// Everything above the "step 1" line is done for you. Read it: the constants and the two worked
// functions set every convention the rest of the file follows.

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
 * Korthikanti et al. 2022 give `s·b·h·(34 + 5·a·s/h)` bytes per GPT-style transformer layer; the second
 * term is the materialised attention matrix, which FlashAttention removes, leaving the 34. With tensor
 * parallelism PLUS sequence parallelism all 34 bytes shrink by tp; with tensor parallelism alone only 24
 * of them do. This module assumes sequence parallelism is on, so the planner divides all 34 by tp.
 */
export const ACT_BYTES_PER_TOKEN_LAYER_DIM = 34;

/**
 * ZeRO-3 (and FSDP) moves 1.5× the bytes of a plain data-parallel all-reduce: a reduce-scatter of the
 * gradients plus an all-gather of the parameters in each of forward and backward (Rajbhandari et al. 2020,
 * section 7). Stages 0–2 move exactly the all-reduce volume.
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
 * This is the whole cost model of the module — every collective below is built out of calls to it.
 * commTime(1e9, NVLINK) ≈ 0.00222 s (bandwidth-bound); commTime(1, NVLINK) = 5e-6 s (all latency).
 */
export function commTime(bytes, link) {
  return link.latency + bytes / link.bandwidth;
}

// ---------- step 1: the ring all-reduce ----------

/**
 * Bytes each GPU sends during a ring all-reduce of a `bytes`-sized buffer across n GPUs.
 * Throw if n is less than 1. n = 1 sends nothing.
 */
export function ringAllReduceBytes(bytes, n) {
  // TODO: step 1
  return 0;
}

/**
 * Wall-clock time of a ring all-reduce of `bytes` across n GPUs over `link`.
 * The steps are sequential, so price each one with commTime and add them up. n = 1 is free.
 * Throw if n is less than 1.
 */
export function ringAllReduceTime(bytes, n, link) {
  // TODO: step 1
  return 0;
}

// ---------- step 2: data parallelism ----------

/**
 * One data-parallel training step. Every GPU holds the full model, processes `tokensPerGpu` tokens,
 * then all-reduces its bf16 gradient (BYTES.grads per parameter) across the `dp` replicas.
 * `overlap` is the fraction of the compute time during which that all-reduce can run in the background.
 * Returns { compute, comm, exposed, step } in seconds. `exposed` is never negative.
 */
export function dataParallelStep({ params, tokensPerGpu, dp, gpu, link, overlap = 0 }) {
  // TODO: step 2
  return { compute: 0, comm: 0, exposed: 0, step: 0 };
}

/**
 * Weak-scaling efficiency at `dp` replicas: the ideal step time (compute alone) over the real one.
 * `cfg` is a dataParallelStep argument object without `dp`.
 * 1.0 means the all-reduce is completely hidden; 0.5 means half the step is waiting for gradients.
 */
export function dpEfficiency(cfg, dp) {
  // TODO: step 2
  return 1;
}

// ---------- step 3: tensor parallelism ----------

/**
 * Megatron-style sharding of one transformer layer's four big matrices across `tp` GPUs.
 * Return, in this order, `attn.qkv` [dModel, 3·dModel], `attn.proj` [dModel, dModel],
 * `mlp.fc` [dModel, dFF] and `mlp.proj` [dFF, dModel], each shrunk by tp along one axis:
 * [{ name, split: 'column' | 'row', shape: [rows, cols] }] describing what ONE GPU holds.
 * Throw if dModel or dFF is not divisible by tp, or if tp is less than 1.
 */
export function tpShards({ dModel, dFF }, tp) {
  // TODO: step 3
  return [];
}

/**
 * What one transformer layer costs in communication under tensor parallelism, for a micro-batch of
 * `tokens` tokens. Returns { allReduces, bytesPerAllReduce, time }. tp = 1 needs no communication.
 */
export function tpLayerComm({ dModel, tokens }, tp, link) {
  // TODO: step 3
  return { allReduces: 0, bytesPerAllReduce: 0, time: 0 };
}

// ---------- step 4: pipeline parallelism ----------

/** Bubble ratio of a pipeline step: idle time filling and draining divided by a stage's useful work. Throw if p or m is below 1. */
export function pipelineBubble(p, m) {
  // TODO: step 4
  return 0;
}

/**
 * One pipeline-parallel step under the 1F1B schedule.
 *   modelTime — seconds for ONE worker to push the whole batch through ALL p stages (so p × stageTime).
 *   p — pipeline stages, m — micro-batches per step.
 *   activationBytesPerMicroBatch — activation memory one stage keeps for one micro-batch.
 * Returns { stageTime, bubble, step, activationBytes1F1B, activationBytesGPipe }.
 */
export function pipelineStep({ modelTime, p, m, activationBytesPerMicroBatch = 0 }) {
  // TODO: step 4
  return { stageTime: 0, bubble: 0, step: 0, activationBytes1F1B: 0, activationBytesGPipe: 0 };
}

// ---------- step 5: ZeRO / FSDP memory ----------

/**
 * Bytes per GPU of persistent training state for `params` parameters across `dp` data-parallel ranks
 * at ZeRO `stage` 0, 1, 2 or 3. Activations are not training state and are counted elsewhere.
 * Returns { params, grads, optimizer, total }. Throw on an unknown stage or a dp below 1.
 */
export function zeroMemoryPerGpu(params, dp, stage) {
  // TODO: step 5
  return { params: 0, grads: 0, optimizer: 0, total: 0 };
}

// ---------- step 6: the planner ----------

/**
 * Simulate one training step of `model` on `gpus` GPUs under strategy { dp, tp, pp, zero, microBatchSeqs }.
 * Returns { dp, tp, pp, zero, m, stepTime, memoryPerGpu, fits, tokensPerSec, breakdown, memory }
 * where breakdown is { compute, tpComm, bubble, ppComm, dpComm, dpExposed } and
 * memory is { params, grads, optimizer, activations }.
 */
export function plan({ model, gpus, strategy, hardware = HARDWARE }) {
  const { dp, tp, pp, zero = 1, microBatchSeqs = 1 } = strategy;
  // TODO: step 6 — validate the layout, then compute time and memory from the pieces you built above.
  void microBatchSeqs; void hardware; void model; void gpus;
  return {
    dp, tp, pp, zero, m: 0, stepTime: 0, memoryPerGpu: 0, fits: false, tokensPerSec: 0,
    breakdown: { compute: 0, tpComm: 0, bubble: 0, ppComm: 0, dpComm: 0, dpExposed: 0 },
    memory: { params: 0, grads: 0, optimizer: 0, activations: 0 },
  };
}

/**
 * Every (dp, tp, pp, zero) layout that is legal for this model on this many GPUs.
 * Micro-batches are one sequence each.
 */
export function enumerateStrategies(model, gpus, hardware = HARDWARE) {
  // TODO: step 6
  return [];
}

/**
 * The fastest legal layout whose memory per GPU stays under `memoryCap` (default: the GPU's memory).
 * Ties on step time are broken towards the layout that uses less memory. null if nothing fits.
 */
export function bestPlan({ model, gpus, hardware = HARDWARE, memoryCap = hardware.gpu.memory }) {
  // TODO: step 6
  return null;
}
