// Module 19 — Quantisation. Reference solution.
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

import * as ops from 'lib/ops.js';

// ---------- helpers ----------

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

export function quantizeAbsmax(x, bits = 8) {
  const data = flat(x);
  const { qmin, qmax } = qrange(bits, true);
  let amax = 0;
  for (let i = 0; i < data.length; i++) if (Math.abs(data[i]) > amax) amax = Math.abs(data[i]);
  const scale = amax > 0 ? amax / qmax : 1;
  const q = new Int8Array(data.length);
  for (let i = 0; i < data.length; i++) q[i] = clamp(Math.round(data[i] / scale), qmin, qmax);
  return { q, scale, bits };
}

export function dequantizeAbsmax(qt) {
  const out = new Float32Array(qt.q.length);
  for (let i = 0; i < out.length; i++) out[i] = qt.q[i] * qt.scale;
  return out;
}

// ---------- step 2: one scale per row (per channel) ----------

export function quantizePerChannel(w, bits = 8) {
  const [rows, cols] = w.shape;
  const { qmin, qmax } = qrange(bits, true);
  const q = new Int8Array(rows * cols);
  const scales = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let amax = 0;
    for (let c = 0; c < cols; c++) {
      const v = Math.abs(w.data[r * cols + c]);
      if (v > amax) amax = v;
    }
    const scale = amax > 0 ? amax / qmax : 1;
    scales[r] = scale;
    for (let c = 0; c < cols; c++) q[r * cols + c] = clamp(Math.round(w.data[r * cols + c] / scale), qmin, qmax);
  }
  return { shape: [rows, cols], q, scales, zeros: null, bits, groupSize: cols };
}

export function dequantize(qw) {
  const [rows, cols] = qw.shape;
  const G = qw.groupSize;
  const nGroups = cols / G;
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let g = 0; g < nGroups; g++) {
      const idx = r * nGroups + g;
      const s = qw.scales[idx];
      const z = qw.zeros ? qw.zeros[idx] : 0;
      const base = r * cols + g * G;
      for (let k = 0; k < G; k++) out[base + k] = (qw.q[base + k] - z) * s;
    }
  }
  return { shape: [rows, cols], data: out };
}

// ---------- step 3: group-wise scales, symmetric or with a zero point ----------

export function quantizeGroups(w, { bits = 4, groupSize = 64, symmetric = true } = {}) {
  const [rows, cols] = w.shape;
  if (cols % groupSize !== 0) throw new Error(`quantizeGroups: cols ${cols} not divisible by groupSize ${groupSize}`);
  const { qmin, qmax } = qrange(bits, symmetric);
  const nGroups = cols / groupSize;
  const q = new Int8Array(rows * cols);
  const scales = new Float32Array(rows * nGroups);
  const zeros = symmetric ? null : new Float32Array(rows * nGroups);
  for (let r = 0; r < rows; r++) {
    for (let g = 0; g < nGroups; g++) {
      const base = r * cols + g * groupSize;
      let lo = Infinity, hi = -Infinity;
      for (let k = 0; k < groupSize; k++) {
        const v = w.data[base + k];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      let scale, zero;
      if (symmetric) {
        const amax = Math.max(Math.abs(lo), Math.abs(hi));
        scale = amax > 0 ? amax / qmax : 1;
        zero = 0;
      } else {
        const range = hi - lo;
        scale = range > 0 ? range / (qmax - qmin) : 1;
        zero = Math.round(-lo / scale);
        zeros[r * nGroups + g] = zero;
      }
      scales[r * nGroups + g] = scale;
      for (let k = 0; k < groupSize; k++) q[base + k] = clamp(Math.round(w.data[base + k] / scale) + zero, qmin, qmax);
    }
  }
  return { shape: [rows, cols], q, scales, zeros, bits, groupSize };
}

// ---------- step 4: weight-only quantised matmul ----------

/**
 * Quantise a linear layer's weight W [K, N] (y = x · W) for the decode kernel: rows of the stored
 * matrix are OUTPUT channels ([N, K]) and groups run along the input dimension K, which is the
 * layout GPTQ, AWQ and Marlin use.
 */
export function quantizeWeight(w, opts = {}) {
  return quantizeGroups(ops.transpose(w), opts);
}

/** x [T, K] times a quantised weight stored as [N, K]; dequantises group by group inside the loop. */
export function quantizedMatmul(x, qw) {
  const [T, K] = x.shape;
  const [N, K2] = qw.shape;
  if (K !== K2) throw new Error(`quantizedMatmul: x has ${K} channels but the weight expects ${K2}`);
  const G = qw.groupSize;
  const nGroups = K / G;
  const out = new Float32Array(T * N);
  for (let t = 0; t < T; t++) {
    const xRow = t * K;
    for (let n = 0; n < N; n++) {
      const wRow = n * K;
      let acc = 0;
      for (let g = 0; g < nGroups; g++) {
        const idx = n * nGroups + g;
        const s = qw.scales[idx];
        const z = qw.zeros ? qw.zeros[idx] : 0;
        let part = 0;
        for (let k = g * G; k < (g + 1) * G; k++) part += x.data[xRow + k] * (qw.q[wRow + k] - z);
        acc += s * part;
      }
      out[t * N + n] = acc;
    }
  }
  return { shape: [T, N], data: out };
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
  const codes = (params * bits) / 8;
  const overhead = groupSize === Infinity ? 0 : (params / groupSize) * ((scaleBits + zeroBits) / 8);
  return codes + overhead;
}

/** Effective bits per parameter including the scale overhead: e.g. int4 g128 with fp16 scales = 4.125. */
export function bitsPerParam(opts) {
  return weightBytes(1, opts) * 8;
}

/** KV cache bytes: keys and values, every layer, every KV head, every position, every sequence. */
export function kvCacheBytes({ nLayer, nKvHeads, headDim }, { contextLen, batch = 1, bits = 16 } = {}) {
  return 2 * nLayer * nKvHeads * headDim * contextLen * batch * (bits / 8);
}
