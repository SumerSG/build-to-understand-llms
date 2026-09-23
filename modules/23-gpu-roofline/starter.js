// GPUs, memory bandwidth & the roofline.
// Every quantity is a plain number in SI units: FLOP, bytes, seconds, FLOP/s, bytes/s.
// Nothing here runs on a GPU: you are building the model you would use *before* writing a kernel.
// Everything below the "worked examples" line is yours to implement.

import { rng } from 'lib/util.js';

// ---------- hardware constants (done for you) ----------
// All figures are approximate. GPU FLOP/s are dense (non-sparse) bf16 with fp32 accumulate
// from NVIDIA's H100 and A100 datasheets and the Ada/RTX 4090 datasheet; the B200 row is per GPU,
// derived from NVIDIA's 8-GPU DGX B200 figures. The CPU row is a rough figure for one 32-core
// AVX-512 server socket, not a measurement.

export const DTYPE_BYTES = { fp32: 4, tf32: 4, fp16: 2, bf16: 2, fp8: 1, int8: 1, int4: 0.5 };

export const H100 = { name: 'H100 SXM', flops: 989e12, bandwidth: 3.35e12, memory: 80e9, dtype: 'bf16' };
export const A100 = { name: 'A100 SXM 80GB', flops: 312e12, bandwidth: 2.04e12, memory: 80e9, dtype: 'bf16' };
export const RTX_4090 = { name: 'RTX 4090', flops: 165e12, bandwidth: 1.01e12, memory: 24e9, dtype: 'bf16' };
export const CPU = { name: 'server CPU (32 cores)', flops: 2e12, bandwidth: 0.2e12, memory: 512e9, dtype: 'fp32' };
// Blackwell: about 2.3x the H100's bf16 FLOP/s and about 2.4x its bandwidth, so the ridge barely
// moves (about 281 FLOP/byte). At FP4, approximately 9 PFLOP/s dense on the same 8 TB/s, the ridge is
// 4x higher (about 1125 FLOP/byte), so even more of inference sits on the memory side.
export const B200 = { name: 'B200', flops: 2.25e15, bandwidth: 8e12, memory: 180e9, dtype: 'bf16' };
export const HARDWARE = [H100, A100, RTX_4090, B200, CPU];

/** Llama-3-8B shapes (Meta's model card): grouped-query attention with 8 key/value heads. */
export const LLAMA3_8B = {
  name: 'Llama-3-8B', params: 8.03e9, layers: 32, dModel: 4096,
  nHeads: 32, nKvHeads: 8, headDim: 128, dFF: 14336, vocab: 128256,
};

/**
 * One GPU's memory hierarchy, from fastest to slowest. Capacities and the HBM3 bandwidth are
 * approximate H100 figures from NVIDIA's H100 whitepaper. The register, SRAM and L2 bandwidths and
 * all latencies are rounded orders of magnitude from published microbenchmarks, not datasheet
 * numbers — treat them as ratios, not measurements. Used by the demo table.
 */
export const MEMORY_HIERARCHY = [
  { level: 'registers', capacity: 256 * 1024 * 132, bandwidth: 100e12, latencyNs: 1, note: '256 KiB per SM x 132 SMs' },
  { level: 'SRAM (L1 / shared)', capacity: 228 * 1024 * 132, bandwidth: 30e12, latencyNs: 25, note: 'up to 228 KiB (233,472 bytes) per SM; where FlashAttention keeps its tiles' },
  { level: 'L2 cache', capacity: 50e6, bandwidth: 7e12, latencyNs: 200, note: 'shared by all SMs' },
  { level: 'HBM3', capacity: 80e9, bandwidth: 3.35e12, latencyNs: 500, note: 'the "memory" in memory-bound' },
  { level: 'host DRAM over PCIe 5 x16', capacity: 2e12, bandwidth: 64e9, latencyNs: 2000, note: 'offloading lands here' },
];

// ---------- worked examples (done for you; they set the conventions) ----------

/** Throughput in GFLOP/s: gflops(2e9, 0.5) === 4. */
export function gflops(flops, seconds) {
  return flops / seconds / 1e9;
}

/** A deterministic n x n matrix as a flat Float32Array, row-major (the tensors module's layout). */
export function randomMatrix(n, seed = 1) {
  const next = rng(seed);
  const a = new Float32Array(n * n);
  for (let i = 0; i < a.length; i++) a[i] = next() * 2 - 1;
  return a;
}

/**
 * The textbook matmul: one dot product per output element (the i, j, k loop order).
 * A and B are flat Float32Arrays of n*n, row-major. Returns a new Float32Array.
 * The inner loop walks a COLUMN of B, jumping n*4 bytes per step, so it touches a fresh
 * cache line on every multiply. Step 4 fixes exactly that.
 */
export function naiveMatmul(A, B, n) {
  const C = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j];
      C[i * n + j] = s;
    }
  }
  return C;
}

// ---------- step 1: the roofline ----------

/** FLOPs per byte of memory traffic. Throw if `bytes` is not greater than 0. */
export function arithmeticIntensity(flops, bytes) {
  // TODO: step 1
  return 0;
}

/** The intensity where the two roofs meet, in FLOP/byte. */
export function ridgePoint(peakFlops, bandwidth) {
  // TODO: step 1
  return 0;
}

/** Best achievable FLOP/s at a given intensity: the lower of the compute roof and the memory roof. */
export function attainable(intensity, peakFlops, bandwidth) {
  // TODO: step 1
  return 0;
}

/** Time for a kernel assuming compute and memory traffic overlap perfectly. */
export function kernelTime(flops, bytes, peakFlops, bandwidth) {
  // TODO: step 1
  return 0;
}

// ---------- step 2: what one matmul costs ----------

/** FLOPs and HBM bytes for [M,K] x [K,N] -> [M,N]. Returns { flops, bytes, intensity }. */
export function matmulCost(M, K, N, bytesPerElement = 2) {
  // TODO: step 2
  return { flops: 0, bytes: 0, intensity: 0 };
}

/**
 * Put one op on the roofline of one device. `op` is { name?, flops, bytes }, `hw` is a row of HARDWARE.
 * Returns { name, flops, bytes, intensity, bound, seconds, attainedFlops, fractionOfPeak }.
 */
export function analyseOp(op, hw) {
  // TODO: step 2
  return { name: op.name ?? 'op', flops: 0, bytes: 0, intensity: 0, bound: 'memory', seconds: 0, attainedFlops: 0, fractionOfPeak: 0 };
}

// ---------- step 3: how much batch buys you ----------

/** The batch size M at which [M,K] x [K,N] first reaches the ridge point. Infinity if it never does. */
export function minBatchForCompute(hw, { K, N, bytesPerElement = 2 }) {
  // TODO: step 3
  return 0;
}

/** Decode ceilings in tokens/s: { memoryBound, computeBound, tokensPerSecond, bound }. */
export function decodeThroughput(hw, { params, bytesPerParam = 2, batch = 1 }) {
  // TODO: step 3
  return { memoryBound: 0, computeBound: 0, tokensPerSecond: 0, bound: 'memory' };
}

// ---------- step 4: raising intensity by tiling ----------

/**
 * Blocked matmul: the same arithmetic as naiveMatmul, reordered so a blockSize x blockSize tile
 * of A, of B and of C are in use at once. Must handle a blockSize that does not divide n.
 */
export function tiledMatmul(A, B, n, blockSize = 32) {
  // TODO: step 4
  return new Float32Array(n * n);
}

/** Bytes a blocked matmul reads from the slow level: (2*n^3/blockSize + n^2) * bytesPerElement. */
export function blockTraffic(n, blockSize, bytesPerElement = 4) {
  // TODO: step 4
  return 0;
}

// ---------- step 5: the memory hierarchy and FlashAttention ----------

/** FLOPs and HBM bytes for one attention head. Returns { flops, bytes, intensity }. */
export function attentionCost(seqLen, headDim, { bytesPerElement = 2, flash = false } = {}) {
  // TODO: step 5
  return { flops: 0, bytes: 0, intensity: 0 };
}

/** Largest key/value block whose four tiles (Q, K, V, O) fit in `sramBytes`. At least 1. */
export function flashBlockSize(sramBytes, headDim, bytesPerElement = 2) {
  // TODO: step 5
  return 0;
}

/**
 * Attention for one head via the online softmax: stream k and v in blocks of blockSize rows, keeping
 * a running max m, a running sum l and an output accumulator per query row. q is { shape: [Tq, dh], data },
 * k and v are { shape: [Tk, dh], data }. causal defaults to true. Returns { shape: [Tq, dh], data }.
 */
export function tiledAttention(q, k, v, blockSize, { causal = true } = {}) {
  // TODO: step 5
  return { shape: [q.shape[0], q.shape[1]], data: new Float32Array(q.shape[0] * q.shape[1]) };
}
