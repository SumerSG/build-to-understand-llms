// Module 08 — reference solution: the arithmetic of a training run.
// Every function here is a few multiplications. The point of the module is that this is enough.

import { fmt } from 'lib/util.js';

// ---------- constants shared by starter and solution ----------

/** Bytes per value for each storage precision. */
export const PRECISION_BYTES = { fp32: 4, bf16: 2, fp16: 2, fp8: 1 };

/** Optimizer state in bytes per parameter (states are kept in fp32, as in PyTorch/DeepSpeed). */
export const OPTIMIZER_BYTES = { adamw: 8, 'sgd-momentum': 4, sgd: 0 };

/** Hardware figures, approximate, from the vendors' datasheets (dense bf16 tensor-core peak). */
export const GPUS = {
  H100: { name: 'H100 SXM (80 GB HBM3)', peakFlops: 989e12, memoryBytes: 80e9 },
  A100: { name: 'A100 SXM (80 GB HBM2e)', peakFlops: 312e12, memoryBytes: 80e9 },
};

/** The fit printed in Hoffmann et al. 2022 ("Chinchilla"), Approach 3, equation (10). */
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

export function trainingFlops(N, D) {
  if (!(N > 0) || !(D > 0)) throw new Error(`trainingFlops: N and D must be positive, got N=${N}, D=${D}`);
  return 6 * N * D;
}

export function inferenceFlops(N, tokens = 1) {
  if (!(N > 0) || !(tokens >= 0)) throw new Error(`inferenceFlops: bad arguments N=${N}, tokens=${tokens}`);
  return 2 * N * tokens;
}

// ---------- step 2: wall-clock ----------

function checkMfu(mfu) {
  if (!(mfu > 0 && mfu <= 1)) throw new Error(`mfu must be a fraction in (0, 1], got ${mfu} (write 0.4, not 40)`);
}

export function wallClockSeconds(flops, { gpus, peakFlops, mfu }) {
  checkMfu(mfu);
  if (!(gpus >= 1) || !(peakFlops > 0)) throw new Error(`wallClockSeconds: need gpus >= 1 and peakFlops > 0, got gpus=${gpus}, peakFlops=${peakFlops}`);
  return flops / (gpus * peakFlops * mfu);
}

export function gpuHours(flops, { peakFlops, mfu }) {
  return wallClockSeconds(flops, { gpus: 1, peakFlops, mfu }) / 3600;
}

// ---------- step 3: memory and the run planner ----------

export function trainingMemory(N, { precision = 'bf16', optimizer = 'adamw' } = {}) {
  const pb = PRECISION_BYTES[precision];
  const ob = OPTIMIZER_BYTES[optimizer];
  if (pb === undefined) throw new Error(`unknown precision "${precision}"; use one of ${Object.keys(PRECISION_BYTES).join(', ')}`);
  if (ob === undefined) throw new Error(`unknown optimizer "${optimizer}"; use one of ${Object.keys(OPTIMIZER_BYTES).join(', ')}`);
  const weights = pb * N;
  const grads = pb * N;
  const optimizerStates = ob * N;
  const masterWeights = precision === 'fp32' ? 0 : 4 * N;
  const total = weights + grads + optimizerStates + masterWeights;
  return { weights, grads, optimizerStates, masterWeights, total, bytesPerParam: total / N };
}

export function activationBytes({ batch, seq, hidden, layers, heads, flashAttention = false, bytesPerValue = 2 }) {
  const perLayerPerValue = seq * batch * hidden * 17 + (flashAttention ? 0 : 2.5 * heads * seq * seq * batch);
  return layers * perLayerPerValue * bytesPerValue;
}

export function planRun({ params, tokens, gpus, gpuFlops = GPUS.H100.peakFlops, gpuMemoryBytes = GPUS.H100.memoryBytes,
  mfu = 0.4, precision = 'bf16', optimizer = 'adamw', pricePerGpuHour = 2 }) {
  const flops = trainingFlops(params, tokens);
  const seconds = wallClockSeconds(flops, { gpus, peakFlops: gpuFlops, mfu });
  const hours = gpuHours(flops, { peakFlops: gpuFlops, mfu });
  const memory = trainingMemory(params, { precision, optimizer });
  return {
    flops,
    seconds,
    days: seconds / 86400,
    gpuHours: hours,
    dollars: hours * pricePerGpuHour,
    memory,
    minGpusForState: Math.ceil(memory.total / gpuMemoryBytes),
    tokensPerParam: tokens / params,
  };
}

// ---------- step 4: the Chinchilla fit ----------

export function scalingLoss(N, D, fit = CHINCHILLA) {
  const { E, A, B, alpha, beta } = fit;
  return E + A / Math.pow(N, alpha) + B / Math.pow(D, beta);
}

export function chinchillaOptimal(C, fit = CHINCHILLA) {
  if (!(C > 0)) throw new Error(`chinchillaOptimal: compute budget must be positive, got ${C}`);
  const { A, B, alpha, beta } = fit;
  const G = Math.pow((alpha * A) / (beta * B), 1 / (alpha + beta));
  const a = beta / (alpha + beta);
  const N = G * Math.pow(C / 6, a);
  const D = C / (6 * N);
  return { N, D, loss: scalingLoss(N, D, fit), tokensPerParam: D / N };
}

// ---------- step 5: over-training and inference ----------

export function tokensForLoss(N, targetLoss, fit = CHINCHILLA) {
  const { E, A, B, alpha, beta } = fit;
  const remaining = targetLoss - E - A / Math.pow(N, alpha);
  if (!(remaining > 0)) return Infinity;
  return Math.pow(B / remaining, 1 / beta);
}

export function optimalForLoss(targetLoss, fit = CHINCHILLA) {
  if (!(targetLoss > fit.E)) throw new Error(`optimalForLoss: target ${targetLoss} is at or below the irreducible loss E=${fit.E}; no finite model reaches it`);
  let lo = 10, hi = 40; // log10 of the compute budget; loss on the optimal frontier falls monotonically with C
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (chinchillaOptimal(Math.pow(10, mid), fit).loss > targetLoss) lo = mid; else hi = mid;
  }
  const C = Math.pow(10, (lo + hi) / 2);
  const { N, D, loss } = chinchillaOptimal(C, fit);
  return { N, D, flops: C, loss };
}

export function overtrainingAnalysis({ N, D, inferenceTokens = 0, fit = CHINCHILLA }) {
  const loss = scalingLoss(N, D, fit);
  const optimal = optimalForLoss(loss, fit);
  const train = trainingFlops(N, D);
  const extraTrainingFlops = Math.max(0, train - optimal.flops);
  const savingPerInferenceToken = inferenceFlops(optimal.N) - inferenceFlops(N);
  const breakEvenInferenceTokens = savingPerInferenceToken > 0 ? extraTrainingFlops / savingPerInferenceToken : Infinity;
  return {
    loss,
    optimal,
    trainingFlops: train,
    extraTrainingFlops,
    savingPerInferenceToken,
    breakEvenInferenceTokens,
    lifetimeFlops: train + inferenceFlops(N, inferenceTokens),
    lifetimeFlopsOptimal: optimal.flops + inferenceFlops(optimal.N, inferenceTokens),
  };
}
