// Tensors from scratch — reference solution (the same kernels live in lib/ops.js in a more general form).

export function size(shape) {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

export function raw(shape, data) {
  const n = size(shape);
  if (data === undefined) data = new Float32Array(n);
  else if (!(data instanceof Float32Array)) data = Float32Array.from(data);
  if (data.length !== n) throw new Error(`raw: data length ${data.length} != size(shape) ${n}`);
  return { shape: shape.slice(), data };
}

export function fromArray(nested) {
  const shape = [];
  let cur = nested;
  while (Array.isArray(cur)) { shape.push(cur.length); cur = cur[0]; }
  return raw(shape, nested.flat(Infinity));
}

export function toArray(t) {
  const { shape, data } = t;
  let i = 0;
  const build = (d) => {
    const out = [];
    for (let j = 0; j < shape[d]; j++) out.push(d === shape.length - 1 ? data[i++] : build(d + 1));
    return out;
  };
  return shape.length ? build(0) : data[0];
}

export function offset(shape, indices) {
  let off = 0, stride = 1;
  for (let d = shape.length - 1; d >= 0; d--) {
    off += indices[d] * stride;
    stride *= shape[d];
  }
  return off;
}

export function transpose(a) {
  const [n, m] = a.shape;
  const out = new Float32Array(n * m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[j * n + i] = a.data[i * m + j];
  return { shape: [m, n], data: out };
}

export function matmul(a, b) {
  const [n, k] = a.shape, [k2, m] = b.shape;
  if (k !== k2) throw new Error(`matmul: inner dims differ (${k} vs ${k2})`);
  const out = new Float32Array(n * m);
  const A = a.data, B = b.data;
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const av = A[i * k + p];
      for (let j = 0; j < m; j++) out[i * m + j] += av * B[p * m + j];
    }
  }
  return { shape: [n, m], data: out };
}

function binary(a, b, fn) {
  const out = new Float32Array(a.data.length);
  const d = a.shape[a.shape.length - 1];
  if (typeof b === 'number') {
    for (let i = 0; i < out.length; i++) out[i] = fn(a.data[i], b);
  } else if (b.shape.length === a.shape.length && b.shape.every((s, i) => s === a.shape[i])) {
    for (let i = 0; i < out.length; i++) out[i] = fn(a.data[i], b.data[i]);
  } else if (b.shape.length === 1 && b.shape[0] === d) {
    for (let i = 0; i < out.length; i++) out[i] = fn(a.data[i], b.data[i % d]);
  } else {
    throw new Error(`cannot broadcast [${b.shape}] onto [${a.shape}]`);
  }
  return { shape: a.shape.slice(), data: out };
}

export const add = (a, b) => binary(a, b, (x, y) => x + y);
export const mul = (a, b) => binary(a, b, (x, y) => x * y);

export function sum(a) {
  const d = a.shape[a.shape.length - 1];
  const rows = a.data.length / d;
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += a.data[r * d + j];
    out[r] = s;
  }
  return { shape: a.shape.slice(0, -1), data: out };
}

export function argmax(a) {
  const d = a.shape[a.shape.length - 1];
  const rows = a.data.length / d;
  const out = [];
  for (let r = 0; r < rows; r++) {
    let best = 0;
    for (let j = 1; j < d; j++) if (a.data[r * d + j] > a.data[r * d + best]) best = j;
    out.push(best);
  }
  return out;
}

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
    for (let j = 0; j < d; j++) out[r + j] = (a.data[r + j] - mu) * inv * (gamma ? gamma.data[j] : 1) + (beta ? beta.data[j] : 0);
  }
  return { shape: a.shape.slice(), data: out };
}
