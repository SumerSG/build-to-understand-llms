// Scaling laws & the arithmetic of compute.
// Nothing here is a simulation. Every function you write is a handful of multiplications
// applied to four numbers: parameters (N), tokens (D), chips, and seconds.
// Symbols used throughout: N = parameter count, D = training tokens, C = training FLOPs,
// MFU = model FLOPs utilisation (the fraction of a chip's peak throughput your run actually uses).

import { fmt } from 'lib/util.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/** Bytes per stored value for each numeric precision. */
export const PRECISION_BYTES = { fp32: 4, bf16: 2, fp16: 2, fp8: 1 };

/**
 * Optimizer state in bytes per parameter. Adam keeps two fp32 buffers (the first moment m and
 * the second moment v), SGD with momentum keeps one, plain SGD keeps none.
 */
export const OPTIMIZER_BYTES = { adamw: 8, 'sgd-momentum': 4, sgd: 0 };

/**
 * Approximate accelerator figures, dense bf16 tensor-core peak, from the vendors' datasheets.
 * NVIDIA's H100 datasheet lists approximately 989 TFLOP/s dense bf16 and 80 GB of HBM3;
 * the A100 datasheet lists approximately 312 TFLOP/s dense bf16 and 80 GB of HBM2e.
 * Real runs reach 30–50% of these peaks (see MFU in step 2).
 */
export const GPUS = {
  H100: { name: 'H100 SXM (80 GB HBM3)', peakFlops: 989e12, memoryBytes: 80e9 },
  A100: { name: 'A100 SXM (80 GB HBM2e)', peakFlops: 312e12, memoryBytes: 80e9 },
};

/** The loss fit printed in Hoffmann et al. 2022 ("Chinchilla"), Approach 3. */
export const CHINCHILLA = { E: 1.69, A: 406.4, B: 410.7, alpha: 0.34, beta: 0.28 };

/** The re-fit of the same data by Besiroglu et al. 2024 ("Chinchilla Scaling: A replication attempt"). */
export const CHINCHILLA_REFIT = { E: 1.8172, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658 };

/** Human-readable duration: 90 -> "1.5 min", 1e6 -> "11.6 days". */
export function formatDuration(seconds) {
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)} h`;
  if (seconds < 86400 * 365) return `${(seconds / 86400).toFixed(1)} days`;
  return `${(seconds / (86400 * 365)).toFixed(1)} years`;
}

/** Human-readable byte count: 1.28e11 -> "128.00 GB", 2.8e12 -> "2.80 TB". */
export function formatBytes(bytes) {
  const abs = Math.abs(bytes);
  if (abs >= 1e12) return `${(bytes / 1e12).toFixed(2)} TB`;
  if (abs >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (abs >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  return `${fmt(bytes)} bytes`;
}

// ---------- step 1: FLOPs ----------

/**
 * Total floating-point operations to train a dense transformer with N parameters on D tokens.
 * The standard estimate is 6·N·D: per parameter per token, 2 FLOPs forward (one multiply-add)
 * and 4 FLOPs backward (gradient with respect to the input, and with respect to the weight).
 * Throw an Error if N or D is not positive — a zero-parameter model is a bug, not a free run.
 */
export function trainingFlops(N, D) {
  // TODO: step 1
  return 0;
}

/**
 * FLOPs to push `tokens` tokens through the same model at inference: 2·N each, because
 * inference is the forward pass only. (The attention terms that depend on context length
 * are small for short contexts; the KV cache module puts them back: `2N + 4·L·T·C` per token with a KV cache.)
 */
export function inferenceFlops(N, tokens = 1) {
  // TODO: step 1
  return 0;
}

// ---------- step 2: wall-clock ----------

/**
 * Seconds to finish `flops` on `gpus` chips of `peakFlops` each, achieving `mfu`.
 * mfu is a fraction in (0, 1] — 0.4, never 40. Throw if it is outside that range,
 * or if gpus < 1, or if peakFlops <= 0.
 */
export function wallClockSeconds(flops, { gpus, peakFlops, mfu }) {
  // TODO: step 2
  return 0;
}

/** The same work priced in GPU-hours: the single-GPU time, in hours. */
export function gpuHours(flops, { peakFlops, mfu }) {
  // TODO: step 2
  return 0;
}

// ---------- step 3: memory and the run planner ----------

/**
 * Bytes of persistent training state for N parameters, split into the four things that
 * actually sit in HBM for the whole run:
 *   { weights, grads, optimizerStates, masterWeights, total, bytesPerParam }
 * Weights and grads cost PRECISION_BYTES[precision] each; the optimizer costs
 * OPTIMIZER_BYTES[optimizer]; mixed precision (anything but fp32) also keeps an fp32 master
 * copy of the weights, 4 bytes per parameter, which is what the update is applied to.
 * Throw on an unknown precision or optimizer name rather than returning NaN.
 */
export function trainingMemory(N, { precision = 'bf16', optimizer = 'adamw' } = {}) {
  // TODO: step 3
  return {};
}

/**
 * Bytes of activations stored for the backward pass, using the Megatron-LM estimate
 * (Korthikanti et al. 2022): per layer, `s·b·h·(34 + 5·a·s/h)` bytes in 16-bit, where
 * s = seq, b = batch, h = hidden, a = heads. The second term is the a·s·s attention score
 * matrices; FlashAttention never materialises them, so set it to zero when flashAttention is true.
 * bytesPerValue lets you price the same formula at another precision.
 */
export function activationBytes({ batch, seq, hidden, layers, heads, flashAttention = false, bytesPerValue = 2 }) {
  // TODO: step 3
  return 0;
}

/**
 * The artifact: one call that turns a run description into its bill. Returns
 *   { flops, seconds, days, gpuHours, dollars, memory, minGpusForState, tokensPerParam }
 * where memory is the trainingMemory object, minGpusForState is how many chips it takes just
 * to hold that state, and dollars is GPU-hours times the assumed price.
 * Call the functions you already wrote; do not repeat their arithmetic.
 */
export function planRun({ params, tokens, gpus, gpuFlops = GPUS.H100.peakFlops, gpuMemoryBytes = GPUS.H100.memoryBytes,
  mfu = 0.4, precision = 'bf16', optimizer = 'adamw', pricePerGpuHour = 2 }) {
  // TODO: step 3
  return {};
}

// ---------- step 4: the Chinchilla fit ----------

/**
 * Predicted loss of a model with N parameters trained on D tokens:
 * `L(N, D) = E + A / N^alpha + B / D^beta`. E is the irreducible loss of the data itself.
 * `fit` supplies the five constants and defaults to the paper's.
 */
export function scalingLoss(N, D, fit = CHINCHILLA) {
  // TODO: step 4
  return 0;
}

/**
 * The compute-optimal split of a training budget C FLOPs: the (N, D) that minimise
 * scalingLoss subject to `6·N·D = C`. Returns { N, D, loss, tokensPerParam }.
 * Throw if C is not positive.
 */
export function chinchillaOptimal(C, fit = CHINCHILLA) {
  // TODO: step 4
  return {};
}

// ---------- step 5: over-training and the inference bill ----------

/**
 * Invert scalingLoss in D: how many tokens a model of N parameters needs to reach targetLoss.
 * Return Infinity when the target is at or below what N alone can reach
 * (`E + A / N^alpha`), because no amount of data gets there.
 */
export function tokensForLoss(N, targetLoss, fit = CHINCHILLA) {
  // TODO: step 5
  return 0;
}

/**
 * The cheapest run that reaches targetLoss: search the compute budget C for the smallest one
 * whose chinchillaOptimal model hits the target. Returns { N, D, flops, loss }.
 * Throw if the target is at or below the irreducible loss E.
 */
export function optimalForLoss(targetLoss, fit = CHINCHILLA) {
  // TODO: step 5
  return {};
}

/**
 * Compare an actual run (N parameters, D tokens) against the compute-optimal run that reaches
 * the same loss, and price the difference over a serving lifetime of `inferenceTokens` tokens.
 * Returns { loss, optimal, trainingFlops, extraTrainingFlops, savingPerInferenceToken,
 *           breakEvenInferenceTokens, lifetimeFlops, lifetimeFlopsOptimal }.
 * breakEvenInferenceTokens is Infinity when the actual model is not smaller than the optimal one.
 */
export function overtrainingAnalysis({ N, D, inferenceTokens = 0, fit = CHINCHILLA }) {
  // TODO: step 5
  return {};
}
