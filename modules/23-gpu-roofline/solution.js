// Module 23 — GPUs, memory bandwidth & the roofline (reference solution).
// Every quantity is a plain number in SI units: FLOP, bytes, seconds, FLOP/s, bytes/s.
// Nothing here runs on a GPU. The formulas are the ones you would use with Nsight Compute
// or with a spreadsheet before you write a kernel at all.

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

/** A deterministic n x n matrix as a flat Float32Array, row-major (module 01's layout). */
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

/** FLOPs per byte of memory traffic. This single number decides which roof you hit. */
export function arithmeticIntensity(flops, bytes) {
  if (!(bytes > 0)) throw new Error(`arithmeticIntensity: bytes must be > 0, got ${bytes}`);
  return flops / bytes;
}

/** The intensity where the two roofs meet: peakFlops / bandwidth, in FLOP/byte. */
export function ridgePoint(peakFlops, bandwidth) {
  if (!(bandwidth > 0)) throw new Error(`ridgePoint: bandwidth must be > 0, got ${bandwidth}`);
  return peakFlops / bandwidth;
}

/** Best achievable FLOP/s at a given intensity: the lower of the compute roof and the memory roof. */
export function attainable(intensity, peakFlops, bandwidth) {
  return Math.min(peakFlops, intensity * bandwidth);
}

/** Time for a kernel, assuming compute and memory traffic overlap perfectly: the max, never the sum. */
export function kernelTime(flops, bytes, peakFlops, bandwidth) {
  return Math.max(flops / peakFlops, bytes / bandwidth);
}

// ---------- step 2: what one matmul costs ----------

/**
 * FLOPs and HBM bytes for [M,K] x [K,N] -> [M,N], counting each matrix exactly once.
 * flops = 2*M*K*N (one multiply and one add per term); bytes = (M*K + K*N + M*N) * bytesPerElement.
 */
export function matmulCost(M, K, N, bytesPerElement = 2) {
  if (!(M > 0 && K > 0 && N > 0)) throw new Error(`matmulCost: dimensions must be > 0, got [${M},${K},${N}]`);
  const flops = 2 * M * K * N;
  const bytes = (M * K + K * N + M * N) * bytesPerElement;
  return { flops, bytes, intensity: arithmeticIntensity(flops, bytes) };
}

/**
 * Put one op on the roofline of one device.
 * op is { name?, flops, bytes }; hw is a row of HARDWARE ({ flops, bandwidth }).
 * Returns the intensity, which roof it hits, how long it takes and what fraction of peak it reaches.
 */
export function analyseOp(op, hw) {
  const intensity = arithmeticIntensity(op.flops, op.bytes);
  const ridge = ridgePoint(hw.flops, hw.bandwidth);
  const attained = attainable(intensity, hw.flops, hw.bandwidth);
  return {
    name: op.name ?? 'op',
    flops: op.flops,
    bytes: op.bytes,
    intensity,
    bound: intensity < ridge ? 'memory' : 'compute',
    seconds: kernelTime(op.flops, op.bytes, hw.flops, hw.bandwidth),
    attainedFlops: attained,
    fractionOfPeak: attained / hw.flops,
  };
}

// ---------- step 3: how much batch buys you ----------

/**
 * The batch size M at which [M,K] x [K,N] first reaches the ridge point, i.e. stops being memory-bound.
 * Solving 2*M*K*N = ridge * (M*K + K*N + M*N) * b for M gives
 *   M = ridge*b*K*N / (2*K*N - ridge*b*(K + N)).
 * Returns Infinity when the denominator is <= 0: no batch size is large enough on that device.
 */
export function minBatchForCompute(hw, { K, N, bytesPerElement = 2 }) {
  const rb = ridgePoint(hw.flops, hw.bandwidth) * bytesPerElement;
  const denom = 2 * K * N - rb * (K + N);
  if (denom <= 0) return Infinity;
  return (rb * K * N) / denom;
}

/**
 * Decode throughput ceilings for one model on one device, in tokens per second.
 * Memory roof: every token re-reads all the weights, so batch tokens cost params*bytesPerParam bytes.
 * Compute roof: a forward pass is about 2 FLOPs per parameter per token (module 08), and both the
 * FLOPs and the batch grow together, so the compute ceiling does not depend on the batch size.
 */
export function decodeThroughput(hw, { params, bytesPerParam = 2, batch = 1 }) {
  if (!(batch >= 1)) throw new Error(`decodeThroughput: batch must be >= 1, got ${batch}`);
  const memoryBound = (batch * hw.bandwidth) / (params * bytesPerParam);
  const computeBound = hw.flops / (2 * params);
  return {
    memoryBound,
    computeBound,
    tokensPerSecond: Math.min(memoryBound, computeBound),
    bound: memoryBound <= computeBound ? 'memory' : 'compute',
  };
}

// ---------- step 4: raising intensity by tiling ----------

/**
 * Blocked matmul: the same arithmetic as naiveMatmul, reordered so that a blockSize x blockSize
 * tile of A, of B and of C are in use at once and are reused blockSize times before being evicted.
 * Tiles at the edges are partial when blockSize does not divide n.
 */
export function tiledMatmul(A, B, n, blockSize = 32) {
  if (!(blockSize >= 1)) throw new Error(`tiledMatmul: blockSize must be >= 1, got ${blockSize}`);
  const C = new Float32Array(n * n);
  for (let i0 = 0; i0 < n; i0 += blockSize) {
    const iMax = Math.min(i0 + blockSize, n);
    for (let j0 = 0; j0 < n; j0 += blockSize) {
      const jMax = Math.min(j0 + blockSize, n);
      for (let k0 = 0; k0 < n; k0 += blockSize) {
        const kMax = Math.min(k0 + blockSize, n);
        for (let i = i0; i < iMax; i++) {
          for (let j = j0; j < jMax; j++) {
            let s = C[i * n + j];
            for (let k = k0; k < kMax; k++) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
          }
        }
      }
    }
  }
  return C;
}

/**
 * Bytes a blocked matmul reads from the slow level, assuming a tile stays resident while it is used.
 * There are (n/blockSize)^3 tile products; each reads one tile of A and one of B (blockSize^2 each),
 * and C is written once: (2*n^3/blockSize + n^2) * bytesPerElement.
 * blockSize = 1 reproduces the naive count, so the traffic falls like 1/blockSize.
 */
export function blockTraffic(n, blockSize, bytesPerElement = 4) {
  if (!(blockSize >= 1)) throw new Error(`blockTraffic: blockSize must be >= 1, got ${blockSize}`);
  return ((2 * n ** 3) / blockSize + n * n) * bytesPerElement;
}

// ---------- step 5: the memory hierarchy and FlashAttention ----------

/**
 * FLOPs and HBM bytes for one attention head over seqLen tokens with head dimension headDim.
 * flops = 4*seqLen^2*headDim for both matmuls (Q·Kᵀ and P·V); the softmax itself is not counted.
 * bytes, naive: Q, K, V and O move once (4*seqLen*headDim) and the [seqLen, seqLen] score matrix
 *   moves four times (write S, read S, write P, read P) -> 4*seqLen^2.
 * bytes, flash: the scores never leave SRAM, so only Q, K, V and O move -> 4*seqLen*headDim.
 */
export function attentionCost(seqLen, headDim, { bytesPerElement = 2, flash = false } = {}) {
  if (!(seqLen > 0 && headDim > 0)) throw new Error(`attentionCost: seqLen and headDim must be > 0, got ${seqLen}, ${headDim}`);
  const flops = 4 * seqLen * seqLen * headDim;
  const streamed = 4 * seqLen * headDim;
  const bytes = (flash ? streamed : streamed + 4 * seqLen * seqLen) * bytesPerElement;
  return { flops, bytes, intensity: arithmeticIntensity(flops, bytes) };
}

/**
 * Largest key/value block a FlashAttention-style kernel can hold in SRAM: four tiles
 * (Q, K, V and the running output O) of blockSize x headDim elements must fit. At least 1.
 */
export function flashBlockSize(sramBytes, headDim, bytesPerElement = 2) {
  if (!(headDim > 0)) throw new Error(`flashBlockSize: headDim must be > 0, got ${headDim}`);
  return Math.max(1, Math.floor(sramBytes / (4 * headDim * bytesPerElement)));
}

/**
 * Attention for one head without ever holding a full row of scores: FlashAttention's online softmax.
 * q is a raw tensor { shape: [Tq, dh], data }, k and v are { shape: [Tk, dh], data }, row-major.
 * Keys and values are streamed in blocks of blockSize rows. For each query row the kernel keeps only
 * a running max m, a running sum l of exp(score - m) and an unnormalised output accumulator acc;
 * when a block raises the max, the old l and acc are rescaled by exp(mOld - mNew) so that every term
 * is measured against the same max. Dividing acc by l at the end gives exactly softmax(q·kᵀ/sqrt(dh))·v.
 * causal (default true, as in lib/attention.js) lets query i see keys j <= i only.
 * Returns { shape: [Tq, dh], data: Float32Array }.
 */
export function tiledAttention(q, k, v, blockSize, { causal = true } = {}) {
  if (!(blockSize >= 1)) throw new Error(`tiledAttention: blockSize must be >= 1, got ${blockSize}`);
  const [Tq, dh] = q.shape, Tk = k.shape[0];
  if (causal && Tq !== Tk) throw new Error(`tiledAttention: causal masking needs as many queries as keys (${Tq} vs ${Tk})`);
  const scale = 1 / Math.sqrt(dh);
  const out = new Float32Array(Tq * dh);
  const acc = new Float64Array(dh);
  const s = new Float64Array(blockSize);
  for (let i = 0; i < Tq; i++) {
    const kEnd = causal ? i + 1 : Tk;
    let m = -Infinity, l = 0;
    acc.fill(0);
    for (let j0 = 0; j0 < kEnd; j0 += blockSize) {
      const jMax = Math.min(j0 + blockSize, kEnd);
      // 1. Scores for this block only, and the block's own max.
      let blockMax = -Infinity;
      for (let j = j0; j < jMax; j++) {
        let dot = 0;
        for (let c = 0; c < dh; c++) dot += q.data[i * dh + c] * k.data[j * dh + c];
        s[j - j0] = dot * scale;
        if (s[j - j0] > blockMax) blockMax = s[j - j0];
      }
      // 2. Move the running statistics onto the new max. On the first block m is -Infinity,
      //    so the correction is exp(-Infinity) = 0 and multiplies an l and acc that are already 0.
      const mNew = Math.max(m, blockMax);
      const correction = Math.exp(m - mNew);
      l *= correction;
      for (let c = 0; c < dh; c++) acc[c] *= correction;
      // 3. Add this block's contribution, measured against the new max.
      for (let j = j0; j < jMax; j++) {
        const p = Math.exp(s[j - j0] - mNew);
        l += p;
        for (let c = 0; c < dh; c++) acc[c] += p * v.data[j * dh + c];
      }
      m = mNew;
    }
    for (let c = 0; c < dh; c++) out[i * dh + c] = acc[c] / l;
  }
  return { shape: [Tq, dh], data: out };
}
