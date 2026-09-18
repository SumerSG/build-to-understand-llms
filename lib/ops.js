// lib/ops.js — raw tensor kernels. This file is the REFERENCE SOLUTION for module 01 (Tensors).
//
// A "raw tensor" is a plain object: { shape: number[], data: Float32Array } stored row-major
// (C order): the last dimension varies fastest. Every function here is pure and allocates a new
// tensor. There is no autograd here — that is built on top in lib/tensor.js.

// ---------- creation ----------

export function size(shape) {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

export function raw(shape, data) {
  const n = size(shape);
  if (data === undefined) data = new Float32Array(n);
  else if (!(data instanceof Float32Array)) data = Float32Array.from(data);
  if (data.length !== n) throw new Error(`raw: data length ${data.length} != size(shape) ${n} for shape [${shape}]`);
  return { shape: shape.slice(), data };
}

export function zeros(shape) {
  return raw(shape);
}

export function ones(shape) {
  return full(shape, 1);
}

export function full(shape, value) {
  const t = raw(shape);
  t.data.fill(value);
  return t;
}

/** Gaussian init. `next` is a uniform rng function (see lib/util.js rng). */
export function randn(shape, next, std = 1) {
  const t = raw(shape);
  for (let i = 0; i < t.data.length; i++) {
    let u = 0, v = 0;
    while (u === 0) u = next();
    while (v === 0) v = next();
    t.data[i] = std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return t;
}

/** Nested JS arrays -> raw tensor. fromArray([[1,2],[3,4]]) has shape [2,2]. */
export function fromArray(nested) {
  const shape = [];
  let cur = nested;
  while (Array.isArray(cur) || ArrayBuffer.isView(cur)) {
    shape.push(cur.length);
    cur = cur[0];
  }
  const data = new Float32Array(size(shape));
  let i = 0;
  (function walk(x, depth) {
    if (depth === shape.length) { data[i++] = x; return; }
    if (x.length !== shape[depth]) throw new Error('fromArray: ragged input');
    for (let j = 0; j < x.length; j++) walk(x[j], depth + 1);
  })(nested, 0);
  return { shape, data };
}

/** raw tensor -> nested JS arrays of plain numbers. */
export function toArray(t) {
  const { shape, data } = t;
  if (shape.length === 0) return data[0];
  let i = 0;
  function build(depth) {
    const out = new Array(shape[depth]);
    for (let j = 0; j < shape[depth]; j++) {
      out[j] = depth === shape.length - 1 ? data[i++] : build(depth + 1);
    }
    return out;
  }
  return build(0);
}

export function clone(t) {
  return { shape: t.shape.slice(), data: new Float32Array(t.data) };
}

export function reshape(t, shape) {
  const inferIdx = shape.indexOf(-1);
  if (inferIdx >= 0) {
    const known = shape.reduce((p, d, i) => (i === inferIdx ? p : p * d), 1);
    shape = shape.slice();
    shape[inferIdx] = t.data.length / known;
  }
  if (size(shape) !== t.data.length) throw new Error(`reshape: cannot reshape [${t.shape}] to [${shape}]`);
  return { shape: shape.slice(), data: t.data };
}

// ---------- broadcasting binary ops ----------

export function broadcastShapes(a, b) {
  const n = Math.max(a.length, b.length);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const da = a[a.length - 1 - i] ?? 1;
    const db = b[b.length - 1 - i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) throw new Error(`cannot broadcast [${a}] with [${b}]`);
    out[n - 1 - i] = Math.max(da, db);
  }
  return out;
}

function stridesFor(shape) {
  const s = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) { s[i] = acc; acc *= shape[i]; }
  return s;
}

/** Strides of `shape` when viewed as the broadcast target `outShape` (0 on broadcast dims). */
function broadcastStrides(shape, outShape) {
  const s = stridesFor(shape);
  const out = new Array(outShape.length).fill(0);
  const off = outShape.length - shape.length;
  for (let i = 0; i < shape.length; i++) out[off + i] = shape[i] === 1 ? 0 : s[i];
  return out;
}

/** Elementwise op with numpy-style broadcasting. fn(x, y) -> number. */
export function binary(a, b, fn) {
  if (typeof b === 'number') return unary(a, (x) => fn(x, b));
  if (typeof a === 'number') return unary(b, (y) => fn(a, y));
  const A = a.data, B = b.data;
  // Fast path: identical shapes.
  if (a.shape.length === b.shape.length && a.shape.every((d, i) => d === b.shape[i])) {
    const out = new Float32Array(A.length);
    for (let i = 0; i < A.length; i++) out[i] = fn(A[i], B[i]);
    return { shape: a.shape.slice(), data: out };
  }
  const outShape = broadcastShapes(a.shape, b.shape);
  const n = size(outShape);
  const out = new Float32Array(n);
  const sa = broadcastStrides(a.shape, outShape);
  const sb = broadcastStrides(b.shape, outShape);
  const idx = new Array(outShape.length).fill(0);
  let ia = 0, ib = 0;
  for (let i = 0; i < n; i++) {
    out[i] = fn(A[ia], B[ib]);
    // increment multi-index
    for (let d = outShape.length - 1; d >= 0; d--) {
      idx[d]++;
      ia += sa[d];
      ib += sb[d];
      if (idx[d] < outShape[d]) break;
      ia -= sa[d] * outShape[d];
      ib -= sb[d] * outShape[d];
      idx[d] = 0;
    }
  }
  return { shape: outShape, data: out };
}

export function unary(a, fn) {
  const out = new Float32Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = fn(a.data[i]);
  return { shape: a.shape.slice(), data: out };
}

export const add = (a, b) => binary(a, b, (x, y) => x + y);
export const sub = (a, b) => binary(a, b, (x, y) => x - y);
export const mul = (a, b) => binary(a, b, (x, y) => x * y);
export const div = (a, b) => binary(a, b, (x, y) => x / y);
export const scale = (a, s) => unary(a, (x) => x * s);
export const neg = (a) => unary(a, (x) => -x);
export const exp = (a) => unary(a, Math.exp);
export const log = (a) => unary(a, Math.log);
export const sqrt = (a) => unary(a, Math.sqrt);
export const tanh = (a) => unary(a, Math.tanh);
export const pow = (a, p) => unary(a, (x) => Math.pow(x, p));
export const relu = (a) => unary(a, (x) => (x > 0 ? x : 0));
export const sigmoid = (a) => unary(a, (x) => 1 / (1 + Math.exp(-x)));
/** GELU, tanh approximation (the one GPT-2 uses). */
export const gelu = (a) => unary(a, geluScalar);
export function geluScalar(x) {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

// ---------- matmul & transpose ----------

/**
 * Matrix multiply. a: [..., n, k], b: [..., k, m] (same leading dims) or b: [k, m] (shared).
 * Returns [..., n, m]. This is the classic triple loop, ordered i-k-j so the inner loop streams
 * through contiguous memory.
 */
export function matmul(a, b) {
  if (a.shape.length < 2 || b.shape.length < 2) throw new Error('matmul: need at least 2D tensors');
  const n = a.shape[a.shape.length - 2], k = a.shape[a.shape.length - 1];
  const k2 = b.shape[b.shape.length - 2], m = b.shape[b.shape.length - 1];
  if (k !== k2) throw new Error(`matmul: inner dims differ (${k} vs ${k2}) for [${a.shape}] x [${b.shape}]`);
  const batchA = a.shape.slice(0, -2), batchB = b.shape.slice(0, -2);
  const batch = size(batchA);
  const bShared = batchB.length === 0;
  if (!bShared && (batchA.length !== batchB.length || batchA.some((d, i) => d !== batchB[i]))) {
    throw new Error(`matmul: batch dims differ [${batchA}] vs [${batchB}]`);
  }
  const out = new Float32Array(batch * n * m);
  const A = a.data, B = b.data;
  for (let bi = 0; bi < batch; bi++) {
    const aOff = bi * n * k, bOff = bShared ? 0 : bi * k * m, oOff = bi * n * m;
    for (let i = 0; i < n; i++) {
      const aRow = aOff + i * k, oRow = oOff + i * m;
      for (let p = 0; p < k; p++) {
        const av = A[aRow + p];
        if (av === 0) continue;
        const bRow = bOff + p * m;
        for (let j = 0; j < m; j++) out[oRow + j] += av * B[bRow + j];
      }
    }
  }
  return { shape: [...batchA, n, m], data: out };
}

/** Swap the last two dimensions. */
export function transpose(a) {
  if (a.shape.length < 2) throw new Error('transpose: need at least 2D');
  const n = a.shape[a.shape.length - 2], m = a.shape[a.shape.length - 1];
  const batch = size(a.shape.slice(0, -2));
  const out = new Float32Array(a.data.length);
  for (let b = 0; b < batch; b++) {
    const off = b * n * m;
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[off + j * n + i] = a.data[off + i * m + j];
  }
  return { shape: [...a.shape.slice(0, -2), m, n], data: out };
}

// ---------- reductions ----------

function normAxis(axis, ndim) {
  if (axis === null || axis === undefined) return null;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) throw new Error(`axis ${axis} out of range for ndim ${ndim}`);
  return axis;
}

/** Reduce along `axis` (or all elements when axis is null). keepDims keeps a size-1 axis. */
export function reduce(a, axis, init, fn, finish = (x) => x, keepDims = false) {
  axis = normAxis(axis, a.shape.length);
  if (axis === null) {
    let acc = init;
    for (let i = 0; i < a.data.length; i++) acc = fn(acc, a.data[i]);
    return { shape: keepDims ? a.shape.map(() => 1) : [], data: Float32Array.of(finish(acc, a.data.length)) };
  }
  const outer = size(a.shape.slice(0, axis));
  const len = a.shape[axis];
  const inner = size(a.shape.slice(axis + 1));
  const out = new Float32Array(outer * inner);
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      let acc = init;
      const base = o * len * inner + i;
      for (let l = 0; l < len; l++) acc = fn(acc, a.data[base + l * inner]);
      out[o * inner + i] = finish(acc, len);
    }
  }
  const shape = a.shape.slice();
  if (keepDims) shape[axis] = 1; else shape.splice(axis, 1);
  return { shape, data: out };
}

export const sum = (a, axis = null, keepDims = false) => reduce(a, axis, 0, (acc, x) => acc + x, (x) => x, keepDims);
export const mean = (a, axis = null, keepDims = false) => reduce(a, axis, 0, (acc, x) => acc + x, (x, n) => x / n, keepDims);
export const max = (a, axis = null, keepDims = false) => reduce(a, axis, -Infinity, (acc, x) => (x > acc ? x : acc), (x) => x, keepDims);

/** Index of the max along the last axis. Returns a plain Array of ints with shape a.shape[:-1]. */
export function argmax(a) {
  const d = a.shape[a.shape.length - 1];
  const rows = a.data.length / d;
  const out = new Array(rows);
  for (let r = 0; r < rows; r++) {
    let best = 0, bv = a.data[r * d];
    for (let j = 1; j < d; j++) {
      const v = a.data[r * d + j];
      if (v > bv) { bv = v; best = j; }
    }
    out[r] = best;
  }
  return out;
}

// ---------- row-wise ops along the last dimension ----------

/** Numerically stable softmax along the last dimension. */
export function softmax(a) {
  const d = a.shape[a.shape.length - 1];
  const out = new Float32Array(a.data.length);
  for (let r = 0; r < a.data.length; r += d) {
    let m = -Infinity;
    for (let j = 0; j < d; j++) if (a.data[r + j] > m) m = a.data[r + j];
    let z = 0;
    for (let j = 0; j < d; j++) { const e = Math.exp(a.data[r + j] - m); out[r + j] = e; z += e; }
    for (let j = 0; j < d; j++) out[r + j] /= z;
  }
  return { shape: a.shape.slice(), data: out };
}

export function logSoftmax(a) {
  const d = a.shape[a.shape.length - 1];
  const out = new Float32Array(a.data.length);
  for (let r = 0; r < a.data.length; r += d) {
    let m = -Infinity;
    for (let j = 0; j < d; j++) if (a.data[r + j] > m) m = a.data[r + j];
    let z = 0;
    for (let j = 0; j < d; j++) z += Math.exp(a.data[r + j] - m);
    const lz = Math.log(z) + m;
    for (let j = 0; j < d; j++) out[r + j] = a.data[r + j] - lz;
  }
  return { shape: a.shape.slice(), data: out };
}

/** LayerNorm along the last dimension. gamma/beta: raw [d] (or null for 1/0). */
export function layerNorm(a, gamma = null, beta = null, eps = 1e-5) {
  const d = a.shape[a.shape.length - 1];
  const out = new Float32Array(a.data.length);
  for (let r = 0; r < a.data.length; r += d) {
    let mu = 0;
    for (let j = 0; j < d; j++) mu += a.data[r + j];
    mu /= d;
    let v = 0;
    for (let j = 0; j < d; j++) { const c = a.data[r + j] - mu; v += c * c; }
    v /= d;
    const inv = 1 / Math.sqrt(v + eps);
    for (let j = 0; j < d; j++) {
      const g = gamma ? gamma.data[j] : 1, b = beta ? beta.data[j] : 0;
      out[r + j] = (a.data[r + j] - mu) * inv * g + b;
    }
  }
  return { shape: a.shape.slice(), data: out };
}

// ---------- indexing helpers ----------

/** Embedding lookup: table [V, d], ids: array of ints (any nesting) -> [...idsShape, d]. */
export function embed(table, ids) {
  const d = table.shape[1];
  const flat = Array.isArray(ids) ? ids.flat(Infinity) : Array.from(ids);
  const shape = Array.isArray(ids) ? shapeOf(ids) : [flat.length];
  const out = new Float32Array(flat.length * d);
  for (let i = 0; i < flat.length; i++) {
    const id = flat[i];
    if (id < 0 || id >= table.shape[0]) throw new Error(`embed: id ${id} out of range`);
    out.set(table.data.subarray(id * d, (id + 1) * d), i * d);
  }
  return { shape: [...shape, d], data: out };
}

function shapeOf(nested) {
  const s = [];
  let c = nested;
  while (Array.isArray(c)) { s.push(c.length); c = c[0]; }
  return s;
}

/** [n, n] lower-triangular mask: 1 where key j <= query i (allowed), else 0. */
export function causalMask(n) {
  const t = raw([n, n]);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) t.data[i * n + j] = 1;
  return t;
}

/** Wherever mask (broadcastable) is 0, replace with `value`. */
export function maskedFill(a, mask, value) {
  return binary(a, mask, (x, m) => (m === 0 ? value : x));
}

/** Slice along `axis` from start (inclusive) to end (exclusive). */
export function slice(a, axis, start, end) {
  axis = normAxis(axis, a.shape.length);
  const len = a.shape[axis];
  if (end === undefined) end = len;
  if (start < 0) start += len;
  if (end < 0) end += len;
  const outer = size(a.shape.slice(0, axis));
  const inner = size(a.shape.slice(axis + 1));
  const w = end - start;
  const out = new Float32Array(outer * w * inner);
  for (let o = 0; o < outer; o++) {
    const src = o * len * inner + start * inner;
    out.set(a.data.subarray(src, src + w * inner), o * w * inner);
  }
  const shape = a.shape.slice();
  shape[axis] = w;
  return { shape, data: out };
}

/** Concatenate two tensors along `axis`. */
export function concat(a, b, axis = 0) {
  axis = normAxis(axis, a.shape.length);
  for (let i = 0; i < a.shape.length; i++) {
    if (i !== axis && a.shape[i] !== b.shape[i]) throw new Error(`concat: shapes [${a.shape}] and [${b.shape}] differ off-axis`);
  }
  const outer = size(a.shape.slice(0, axis));
  const inner = size(a.shape.slice(axis + 1));
  const la = a.shape[axis], lb = b.shape[axis];
  const out = new Float32Array(outer * (la + lb) * inner);
  for (let o = 0; o < outer; o++) {
    out.set(a.data.subarray(o * la * inner, (o + 1) * la * inner), o * (la + lb) * inner);
    out.set(b.data.subarray(o * lb * inner, (o + 1) * lb * inner), o * (la + lb) * inner + la * inner);
  }
  const shape = a.shape.slice();
  shape[axis] = la + lb;
  return { shape, data: out };
}

// ---------- comparison ----------

export function allClose(a, b, tol = 1e-4) {
  const A = a.data ?? a, B = b.data ?? b;
  if (A.length !== B.length) return false;
  for (let i = 0; i < A.length; i++) {
    const diff = Math.abs(A[i] - B[i]);
    if (Number.isNaN(diff) || diff > tol * Math.max(1, Math.abs(A[i]), Math.abs(B[i]))) return false;
  }
  return true;
}

// ---------- permute (general axis reorder) ----------

/** Reorder axes: permute(a, [0,2,1,3]) turns [B,T,H,d] into [B,H,T,d]. */
export function permute(a, order) {
  const nd = a.shape.length;
  if (order.length !== nd) throw new Error('permute: order length must equal ndim');
  const outShape = order.map((o) => a.shape[o]);
  const inStrides = stridesFor(a.shape);
  const srcStrides = order.map((o) => inStrides[o]);
  const out = new Float32Array(a.data.length);
  const idx = new Array(nd).fill(0);
  let src = 0;
  for (let i = 0; i < out.length; i++) {
    out[i] = a.data[src];
    for (let d = nd - 1; d >= 0; d--) {
      idx[d]++;
      src += srcStrides[d];
      if (idx[d] < outShape[d]) break;
      src -= srcStrides[d] * outShape[d];
      idx[d] = 0;
    }
  }
  return { shape: outShape, data: out };
}
