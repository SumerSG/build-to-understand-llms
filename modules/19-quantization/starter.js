// Module 19 — Quantisation.
//
// A quantised tensor stores each float as a small integer `q` times a shared float `scale`
// (plus an integer `zero` point when the scheme is asymmetric): x ≈ (q − zero) · scale.
// The scale is set by the largest value in whatever group shares it, so the whole module is
// about choosing how many values share one scale: the tensor, a row, or a group of 32–128.
//
// Conventions: raw tensors are { shape, data: Float32Array } as in lib/ops.js. Integer codes are kept
// in an Int8Array whatever the bit width (int4 codes occupy a byte each here; real kernels pack two
// per byte). A quantised 2-D matrix is
//   { shape: [rows, cols], q: Int8Array, scales: Float32Array, zeros: Float32Array | null, bits, groupSize }
// with one scale (and zero) per contiguous group of `groupSize` values along each row.
// Everything below the "worked examples" line is yours to implement.

import * as ops from 'lib/ops.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/** The integer range for `bits`: symmetric [-2^(b-1), 2^(b-1)-1], asymmetric [0, 2^b - 1]. */
export function qrange(bits, symmetric = true) {
  if (symmetric) {
    const qmax = 2 ** (bits - 1) - 1;
    return { qmin: -qmax - 1, qmax };
  }
  return { qmin: 0, qmax: 2 ** bits - 1 };
}

/** Accept a raw tensor, a typed array or a plain array and return the flat values. */
function flat(x) {
  if (x && x.data) return x.data;
  return x;
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * How far a reconstruction `y` is from the original `x`: mean squared error, largest absolute
 * error, cosine similarity, and relative L2 error ‖x − y‖ / ‖x‖.
 */
export function errorStats(x, y) {
  const a = flat(x), b = flat(y);
  if (a.length !== b.length) throw new Error(`errorStats: lengths differ (${a.length} vs ${b.length})`);
  let se = 0, maxAbs = 0, dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    se += d * d;
    if (Math.abs(d) > maxAbs) maxAbs = Math.abs(d);
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const n = a.length || 1;
  return {
    mse: se / n,
    maxAbs,
    cosine: dot / (Math.sqrt(na * nb) || 1),
    rel: Math.sqrt(se) / (Math.sqrt(na) || 1),
  };
}

// ---------- step 1: absmax symmetric, one scale for the whole tensor ----------

/**
 * Symmetric absmax quantisation of a whole tensor: scale = max|x| / qmax, q = round(x / scale)
 * clamped to [qmin, qmax]. Returns { q: Int8Array, scale, bits }. An all-zero input must not
 * divide by zero (use scale = 1).
 */
export function quantizeAbsmax(x, bits = 8) {
  const data = flat(x);
  const { qmin, qmax } = qrange(bits, true);
  let amax = 0;
  for (let i = 0; i < data.length; i++) if (Math.abs(data[i]) > amax) amax = Math.abs(data[i]);
  // TODO: step 1 — compute the scale from amax and qmax, then fill the codes.
  const scale = 1;
  const q = new Int8Array(data.length);
  return { q, scale, bits };
}

/** q * scale for every code: Float32Array of the same length as qt.q. */
export function dequantizeAbsmax(qt) {
  // TODO: step 1
  return new Float32Array(qt.q.length);
}

// ---------- step 2: one scale per row (per channel) ----------

/**
 * One symmetric absmax scale per row of a [rows, cols] raw tensor. Returns the quantised-matrix
 * struct described at the top of the file with groupSize = cols and zeros = null.
 */
export function quantizePerChannel(w, bits = 8) {
  // TODO: step 2
  const [rows, cols] = w.shape;
  return { shape: [rows, cols], q: new Int8Array(rows * cols), scales: new Float32Array(rows), zeros: null, bits, groupSize: cols };
}

/**
 * Rebuild a raw [rows, cols] tensor from a quantised matrix. Value [r, c] belongs to group
 * g = floor(c / groupSize) of row r, whose scale is scales[r * nGroups + g] (nGroups = cols / groupSize).
 * If qw.zeros is present, the value is (q − zeros[same index]) * scale; otherwise q * scale.
 */
export function dequantize(qw) {
  // TODO: step 2
  return { shape: qw.shape.slice(), data: new Float32Array(qw.q.length) };
}

// ---------- step 3: group-wise scales, symmetric or with a zero point ----------

/**
 * Split every row into contiguous groups of `groupSize` values (throw if cols is not divisible)
 * and give each group its own scale. Symmetric: scale = max|group| / qmax, zero = 0, zeros = null.
 * Asymmetric: scale = (max − min) / (qmax − qmin), zero = round(−min / scale), and every code is
 * round(x / scale) + zero clamped to [qmin, qmax]; zeros holds one zero point per group.
 */
export function quantizeGroups(w, { bits = 4, groupSize = 64, symmetric = true } = {}) {
  // TODO: step 3
  const [rows, cols] = w.shape;
  return { shape: [rows, cols], q: new Int8Array(rows * cols), scales: new Float32Array((rows * cols) / groupSize), zeros: null, bits, groupSize };
}

// ---------- step 4: weight-only quantised matmul ----------

/**
 * Quantise a linear layer's weight W [K, N] (y = x · W) for the decode kernel: rows of the stored
 * matrix are OUTPUT channels ([N, K]) and groups run along the input dimension K, which is the
 * layout GPTQ, AWQ and Marlin use.
 */
export function quantizeWeight(w, opts = {}) {
  // TODO: step 4
  return quantizeGroups(w, opts);
}

/**
 * x [T, K] times a quantised weight stored as [N, K] → raw [T, N]. Dequantise inside the loop,
 * group by group: accumulate Σ x[k] · (q[n, k] − zero) over the group, then multiply by the group's
 * scale once. Never materialise the full float weight. Throw if K does not match.
 */
export function quantizedMatmul(x, qw) {
  // TODO: step 4
  const [T] = x.shape;
  const [N] = qw.shape;
  return { shape: [T, N], data: new Float32Array(T * N) };
}

// ---------- step 5: the memory calculator ----------

// Llama 3 (Meta, 2024): 8B has 32 layers, 8 KV heads of dimension 128 (GQA); 70B has 80 layers, 8 KV heads.
export const LLAMA3_8B = { name: 'Llama-3-8B', params: 8.03e9, nLayer: 32, nKvHeads: 8, headDim: 128 };
export const LLAMA3_70B = { name: 'Llama-3-70B', params: 70.6e9, nLayer: 80, nKvHeads: 8, headDim: 128 };

/**
 * Bytes to store `params` weights at `bits` per value plus one scale (and optional zero point)
 * per group of `groupSize` values. groupSize = Infinity means a single scale per tensor (no overhead).
 */
export function weightBytes(params, { bits, groupSize = Infinity, scaleBits = 16, zeroBits = 0 } = {}) {
  // TODO: step 5
  return 0;
}

/** Effective bits per parameter including the scale overhead: e.g. int4 g128 with fp16 scales = 4.125. */
export function bitsPerParam(opts) {
  // TODO: step 5
  return 0;
}

/** KV cache bytes: keys and values, every layer, every KV head, every position, every sequence. */
export function kvCacheBytes({ nLayer, nKvHeads, headDim }, { contextLen, batch = 1, bits = 16 } = {}) {
  // TODO: step 5
  return 0;
}
