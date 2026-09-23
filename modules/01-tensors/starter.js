// Tensors from scratch.
// A "raw tensor" is { shape: number[], data: Float32Array }, stored row-major: the last index varies fastest.
// Everything below the "worked examples" line is yours to implement.

// ---------- worked examples (done for you; read them, they set the conventions) ----------
// The comments marked "JS:" explain JavaScript that the "JavaScript for this lab" module may not have shown you.

/** Number of elements in a shape: size([2, 3]) === 6. */
export function size(shape) {
  let n = 1;
  for (const d of shape) n *= d;          // JS: n *= d is short for n = n * d
  return n;
}

/** Create a tensor. `data` defaults to zeros; arrays are converted to Float32Array. */
export function raw(shape, data) {
  const n = size(shape);
  // JS: a parameter the caller left out is undefined. `x instanceof Float32Array` asks "is x a Float32Array?",
  // and `!` turns the answer around. Float32Array.from([1, 2]) copies a plain array into a Float32Array.
  if (data === undefined) data = new Float32Array(n);
  else if (!(data instanceof Float32Array)) data = Float32Array.from(data);
  // JS: `throw new Error(message)` stops the function and reports the message. The backtick string is a
  // template literal: ${...} inside it is replaced by the value of the expression.
  if (data.length !== n) throw new Error(`raw: data length ${data.length} != size(shape) ${n}`);
  // JS: { shape: ..., data } is short for { shape: ..., data: data }. shape.slice() copies the array, so the
  // caller changing their array later cannot change this tensor's shape.
  return { shape: shape.slice(), data };
}

/** Nested JS arrays -> tensor: fromArray([[1, 2], [3, 4]]) has shape [2, 2]. */
export function fromArray(nested) {
  const shape = [];
  let cur = nested;
  // Walk into the first element again and again, recording each level's length: [[1, 2], [3, 4]] gives [2, 2].
  while (Array.isArray(cur)) { shape.push(cur.length); cur = cur[0]; }
  // JS: .flat(Infinity) flattens every level of nesting: [[1, 2], [3, 4]] becomes [1, 2, 3, 4].
  return raw(shape, nested.flat(Infinity));
}

/** Tensor -> nested JS arrays (handy for printing and for tests). */
export function toArray(t) {
  // JS: const { shape, data } = t; is short for const shape = t.shape; const data = t.data;
  const { shape, data } = t;
  let i = 0;
  // JS: build is a function stored in a constant (an arrow function), and it calls itself for the next axis.
  // cond ? x : y means "x if cond is true, otherwise y". data[i++] reads data[i], then adds 1 to i.
  const build = (d) => {
    const out = [];
    for (let j = 0; j < shape[d]; j++) out.push(d === shape.length - 1 ? data[i++] : build(d + 1));
    return out;
  };
  return shape.length ? build(0) : data[0];
}

// ---------- step 1: indexing and transpose ----------

/**
 * Flat offset of a multi-index in a row-major tensor.
 * offset([2, 3], [1, 2]) === 5   because row 1 starts at 3 and column 2 adds 2.
 */
export function offset(shape, indices) {
  // Completion problem: the loop is written for you. It walks the axes from last to first,
  // carrying `stride`, the number of flat elements you skip when index d goes up by one.
  let off = 0;
  let stride = 1;                          // the last axis is contiguous
  for (let d = shape.length - 1; d >= 0; d--) {
    // TODO: step 1. Two lines: add axis d's contribution to `off`,
    // then update `stride` so it is correct for axis d - 1.
  }
  return off;
}

/** Swap the two axes of a 2-D tensor: [n, m] -> [m, n]. */
export function transpose(a) {
  // TODO: step 1
  return a;
}

// ---------- step 2: matrix multiply ----------

/** 2-D matrix multiply: [n, k] x [k, m] -> [n, m]. Throw if the inner dimensions differ. */
export function matmul(a, b) {
  // TODO: step 2
  return a;
}

// ---------- step 3: elementwise ops with broadcasting ----------

/**
 * The shared helper for add and mul: combine a and b element by element with fn, for example
 * fn = (x, y) => x + y. It stays here, on its own at the top level, not inside add or any other function,
 * so that both add and mul can call it.
 */
// JS: no `export` in front: the helper is private to this file (the tests do not call it directly).
function binary(a, b, fn) {
  // TODO: step 3
}

/**
 * Elementwise a + b. `b` may be: a tensor of the same shape, a plain number,
 * or a 1-D tensor of length a.shape[last] that is broadcast across every row.
 */
export function add(a, b) {
  // TODO: step 3
  return a;
}

/** Elementwise a * b, same broadcasting rules as add. */
export function mul(a, b) {
  // TODO: step 3
  return a;
}

// ---------- step 4: row-wise reductions and softmax ----------

/** Sum along the last axis: [.., d] -> [..]. */
export function sum(a) {
  // TODO: step 4
  return a;
}

/** Index of the largest value along the last axis, as a plain array of ints. */
export function argmax(a) {
  // TODO: step 4
  return [];
}

/** Softmax along the last axis. Must be numerically stable for large logits. */
export function softmax(a) {
  // TODO: step 4
  return a;
}

// ---------- step 5: layer normalisation ----------

/**
 * LayerNorm along the last axis: normalise each row to mean 0, variance 1 (using the
 * biased variance, divide by d), then scale by gamma and shift by beta (1-D tensors of length d,
 * or null for 1 and 0). eps is added to the variance before the square root.
 */
export function layerNorm(a, gamma = null, beta = null, eps = 1e-5) {
  // TODO: step 5
  return a;
}
