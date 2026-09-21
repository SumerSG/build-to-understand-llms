// lib/tensor.js — reverse-mode autograd on top of lib/ops.js. This file is the REFERENCE SOLUTION for
// module 02 (Autograd).
//
// A Tensor wraps a raw tensor ({ shape, data: Float32Array }) and, while gradients are enabled, remembers
// how it was made: the Tensors it came from (_children), the op name (_op) and a closure (_backward) that
// takes the gradient sitting in this.grad and adds each input's share into input.grad. backward() on a
// scalar walks that record from the output back to the leaves: the chain rule, run in reverse.
// Gradients ACCUMULATE with +=, so a tensor used twice receives both contributions; call zeroGrad() (or
// the optimizer's zeroGrad) before the next backward pass.

import * as ops from './ops.js';

let gradEnabled = true;

/** True while ops record the graph (the default); false inside noGrad. */
export function isGradEnabled() {
  return gradEnabled;
}

/** Run fn with graph recording disabled (inference, evaluation, parameter updates); returns fn(). */
export function noGrad(fn) {
  const previous = gradEnabled;
  gradEnabled = false;
  try {
    return fn();
  } finally {
    gradEnabled = previous;
  }
}

/** Sum `grad` (laid out as gradShape) down to targetShape, the input shape it was broadcast from. */
export function unbroadcast(grad, gradShape, targetShape) {
  let t = { shape: gradShape, data: grad };
  // Extra leading dims were created by broadcasting: sum them away.
  while (t.shape.length > targetShape.length) t = ops.sum(t, 0);
  // Dims the input had as size 1 were stretched: sum them back to size 1.
  for (let i = 0; i < targetShape.length; i++) {
    if (targetShape[i] === 1 && t.shape[i] !== 1) t = ops.sum(t, i, true);
  }
  return t.data;
}

/** t.grad, allocated as zeros on first use. */
function gradOf(t) {
  if (t.grad === null) t.grad = new Float32Array(t.data.length);
  return t.grad;
}

/** Add the Float32Array g into t.grad (+=, so repeated uses of t all contribute). */
function accumulate(t, g) {
  if (!t.requiresGrad) return;
  const G = gradOf(t);
  for (let i = 0; i < G.length; i++) G[i] += g[i];
}

/** Wrap an op's raw result; when recording, remember the inputs and how to push gradient back to them. */
function fromOp(raw, op, inputs, backwardFn /* (g: raw gradient of the output) => void */) {
  const out = new Tensor(raw);
  if (gradEnabled && inputs.some((t) => t.requiresGrad)) {
    out.requiresGrad = true;
    out._op = op;
    out._children = inputs;
    out._backward = () => backwardFn({ shape: out.shape, data: out.grad });
  }
  return out;
}

/** The Tensor inputs of a binary op (a plain-number or raw operand has no gradient). */
function tensorInputs(a, b) {
  return b instanceof Tensor ? [a, b] : [a];
}

/** Negative axes count from the end, as in numpy; null means "all elements". */
function resolveAxis(axis, ndim) {
  if (axis === null || axis === undefined) return null;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) throw new Error(`axis ${axis} out of range for ndim ${ndim}`);
  return axis;
}

/** Spread a reduced gradient back over the axis it was summed along (the reverse of ops.reduce), times factor. */
function expandAlongAxis(g, shape, axis, factor) {
  const out = new Float32Array(ops.size(shape));
  if (axis === null) {
    out.fill(g[0] * factor);
    return out;
  }
  const outer = ops.size(shape.slice(0, axis)), len = shape[axis], inner = ops.size(shape.slice(axis + 1));
  for (let o = 0; o < outer; o++) {
    for (let l = 0; l < len; l++) {
      for (let i = 0; i < inner; i++) out[(o * len + l) * inner + i] = g[o * inner + i] * factor;
    }
  }
  return out;
}

/** Every node reachable from root, children before parents (post-order depth-first search). */
function topoSort(root) {
  const order = [];
  const seen = new Set();
  (function visit(t) {
    if (seen.has(t)) return;
    seen.add(t);
    for (const child of t._children) visit(child);
    order.push(t);
  })(root);
  return order;
}

/** A raw tensor plus gradient bookkeeping. Every op returns a new Tensor; nothing is modified in place. */
export class Tensor {
  /** Wraps raw ({ shape, data }) without copying. requiresGrad marks a leaf whose gradient we want. */
  constructor(raw, { requiresGrad = false } = {}) {
    if (!raw || !raw.shape || !raw.data) throw new Error('Tensor: expected a raw tensor { shape, data }');
    this.shape = raw.shape;
    this.data = raw.data;
    this.grad = null;
    this.requiresGrad = requiresGrad;
    this._children = [];
    this._backward = null;
    this._op = '';
  }

  // ---------- creation ----------

  /** Tensor from nested JS arrays: Tensor.from([[1, 2], [3, 4]]) has shape [2, 2]. */
  static from(nested, opts) {
    return new Tensor(ops.fromArray(nested), opts);
  }

  /** All zeros. */
  static zeros(shape, opts) {
    return new Tensor(ops.zeros(shape), opts);
  }

  /** All ones. */
  static ones(shape, opts) {
    return new Tensor(ops.ones(shape), opts);
  }

  /** Every element equal to v. */
  static full(shape, v, opts) {
    return new Tensor(ops.full(shape, v), opts);
  }

  /** Gaussian init with the given std; `next` is an rng function from lib/util.js. */
  static randn(shape, next, std = 1, opts) {
    return new Tensor(ops.randn(shape, next, std), opts);
  }

  /** A trainable leaf: wraps raw with requiresGrad = true. */
  static param(raw) {
    return new Tensor(raw, { requiresGrad: true });
  }

  // ---------- inspection ----------

  /** Number of elements. */
  get size() {
    return this.data.length;
  }

  /** The single value of a size-1 tensor as a plain number. */
  item() {
    if (this.data.length !== 1) throw new Error(`item: tensor has ${this.data.length} elements, not 1`);
    return this.data[0];
  }

  /** Nested JS arrays of plain numbers. */
  toArray() {
    return ops.toArray(this);
  }

  /** The same data as a fresh leaf that is cut off from the graph (no history, no gradient). */
  detach() {
    return new Tensor({ shape: this.shape.slice(), data: this.data });
  }

  /** Forget the accumulated gradient. */
  zeroGrad() {
    this.grad = null;
  }

  // ---------- elementwise, with broadcasting; o is a Tensor or a number ----------

  /** this + o. Each input receives the output gradient summed back to its own shape. */
  add(o) {
    return fromOp(ops.add(this, o), 'add', tensorInputs(this, o), (g) => {
      accumulate(this, unbroadcast(g.data, g.shape, this.shape));
      if (o instanceof Tensor) accumulate(o, unbroadcast(g.data, g.shape, o.shape));
    });
  }

  /** this - o. */
  sub(o) {
    return fromOp(ops.sub(this, o), 'sub', tensorInputs(this, o), (g) => {
      accumulate(this, unbroadcast(g.data, g.shape, this.shape));
      if (o instanceof Tensor) accumulate(o, unbroadcast(ops.neg(g).data, g.shape, o.shape));
    });
  }

  /** this * o. d(a*b)/da = b, d(a*b)/db = a. */
  mul(o) {
    return fromOp(ops.mul(this, o), 'mul', tensorInputs(this, o), (g) => {
      accumulate(this, unbroadcast(ops.mul(g, o).data, g.shape, this.shape));
      if (o instanceof Tensor) accumulate(o, unbroadcast(ops.mul(g, this).data, g.shape, o.shape));
    });
  }

  /** this / o. d(a/b)/da = 1/b, d(a/b)/db = -a/b². */
  div(o) {
    return fromOp(ops.div(this, o), 'div', tensorInputs(this, o), (g) => {
      accumulate(this, unbroadcast(ops.div(g, o).data, g.shape, this.shape));
      if (o instanceof Tensor) {
        const dB = ops.div(ops.mul(g, ops.neg(this)), ops.mul(o, o));
        accumulate(o, unbroadcast(dB.data, g.shape, o.shape));
      }
    });
  }

  /** this * s for a plain number s. */
  scale(s) {
    return fromOp(ops.scale(this, s), 'scale', [this], (g) => accumulate(this, ops.scale(g, s).data));
  }

  /** -this. */
  neg() {
    return fromOp(ops.neg(this), 'neg', [this], (g) => accumulate(this, ops.neg(g).data));
  }

  // ---------- linear algebra and shape ops ----------

  /** Matrix product (2D or batched; o may be a shared 2D matrix). dA = dC·Bᵀ, dB = Aᵀ·dC. */
  matmul(o) {
    return fromOp(ops.matmul(this, o), 'matmul', tensorInputs(this, o), (g) => {
      // Skip the expensive product for an input that is a constant.
      if (this.requiresGrad) accumulate(this, ops.matmul(g, ops.transpose(o)).data);
      if (o.requiresGrad) {
        // A shared 2D B was used by every batch element, so its gradient is the sum over the batch.
        const dB = ops.matmul(ops.transpose(this), g);
        accumulate(o, unbroadcast(dB.data, dB.shape, o.shape));
      }
    });
  }

  /** Swap the last two dims; the gradient is transposed back. */
  transpose() {
    return fromOp(ops.transpose(this), 'transpose', [this], (g) => accumulate(this, ops.transpose(g).data));
  }

  /** Reorder axes; the gradient is permuted with the inverse order. */
  permute(order) {
    const inverse = new Array(order.length);
    for (let i = 0; i < order.length; i++) inverse[order[i]] = i;
    return fromOp(ops.permute(this, order), 'permute', [this], (g) => accumulate(this, ops.permute(g, inverse).data));
  }

  /** Same elements, new shape (one -1 allowed); the gradient just takes the old shape back. */
  reshape(shape) {
    return fromOp(ops.reshape(this, shape), 'reshape', [this], (g) => accumulate(this, g.data));
  }

  /** Slice along axis from start (inclusive) to end (exclusive); the gradient is scattered back into place. */
  slice(axis, start, end) {
    axis = resolveAxis(axis, this.shape.length);
    const len = this.shape[axis];
    if (end === undefined) end = len;
    if (start < 0) start += len;
    if (end < 0) end += len;
    const outer = ops.size(this.shape.slice(0, axis)), inner = ops.size(this.shape.slice(axis + 1));
    const width = end - start;
    return fromOp(ops.slice(this, axis, start, end), 'slice', [this], (g) => {
      const full = new Float32Array(this.data.length);
      for (let o = 0; o < outer; o++) {
        full.set(g.data.subarray(o * width * inner, (o + 1) * width * inner), (o * len + start) * inner);
      }
      accumulate(this, full);
    });
  }

  /** Row lookup: this is a [V, d] table, ids any nesting of ints. Gradient rows are scatter-added into table.grad. */
  embed(ids) {
    const flat = Array.isArray(ids) ? ids.flat(Infinity) : Array.from(ids);
    const d = this.shape[1];
    return fromOp(ops.embed(this, ids), 'embed', [this], (g) => {
      if (!this.requiresGrad) return;
      const G = gradOf(this);
      for (let i = 0; i < flat.length; i++) {
        const row = flat[i] * d;
        for (let j = 0; j < d; j++) G[row + j] += g.data[i * d + j];
      }
    });
  }

  // ---------- reductions ----------

  /** Sum over axis (or everything). The gradient is copied to every element that was summed. */
  sum(axis = null, keepDims = false) {
    axis = resolveAxis(axis, this.shape.length);
    return fromOp(ops.sum(this, axis, keepDims), 'sum', [this], (g) => {
      accumulate(this, expandAlongAxis(g.data, this.shape, axis, 1));
    });
  }

  /** Mean over axis (or everything). Like sum, with the gradient divided by the number of elements averaged. */
  mean(axis = null, keepDims = false) {
    axis = resolveAxis(axis, this.shape.length);
    const count = axis === null ? this.data.length : this.shape[axis];
    return fromOp(ops.mean(this, axis, keepDims), 'mean', [this], (g) => {
      accumulate(this, expandAlongAxis(g.data, this.shape, axis, 1 / count));
    });
  }

  // ---------- elementwise nonlinearities ----------

  /** e^x; derivative is the output itself. */
  exp() {
    const y = ops.exp(this);
    return fromOp(y, 'exp', [this], (g) => {
      const dx = new Float32Array(y.data.length);
      for (let i = 0; i < dx.length; i++) dx[i] = g.data[i] * y.data[i];
      accumulate(this, dx);
    });
  }

  /** ln x; derivative 1/x. */
  log() {
    return fromOp(ops.log(this), 'log', [this], (g) => {
      const dx = new Float32Array(this.data.length);
      for (let i = 0; i < dx.length; i++) dx[i] = g.data[i] / this.data[i];
      accumulate(this, dx);
    });
  }

  /** tanh x; derivative 1 - tanh²x. */
  tanh() {
    const y = ops.tanh(this);
    return fromOp(y, 'tanh', [this], (g) => {
      const dx = new Float32Array(y.data.length);
      for (let i = 0; i < dx.length; i++) dx[i] = g.data[i] * (1 - y.data[i] * y.data[i]);
      accumulate(this, dx);
    });
  }

  /** max(x, 0); the gradient passes only where x > 0. */
  relu() {
    return fromOp(ops.relu(this), 'relu', [this], (g) => {
      const dx = new Float32Array(this.data.length);
      for (let i = 0; i < dx.length; i++) dx[i] = this.data[i] > 0 ? g.data[i] : 0;
      accumulate(this, dx);
    });
  }

  /** GELU (tanh approximation). With u = c(x + 0.044715x³): f' = ½(1 + tanh u) + ½x(1 - tanh²u)·u'. */
  gelu() {
    return fromOp(ops.gelu(this), 'gelu', [this], (g) => {
      const c = Math.sqrt(2 / Math.PI);
      const dx = new Float32Array(this.data.length);
      for (let i = 0; i < dx.length; i++) {
        const x = this.data[i];
        const t = Math.tanh(c * (x + 0.044715 * x * x * x));
        const du = c * (1 + 3 * 0.044715 * x * x);
        dx[i] = g.data[i] * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du);
      }
      accumulate(this, dx);
    });
  }

  /** √x; derivative 1/(2√x). */
  sqrt() {
    const y = ops.sqrt(this);
    return fromOp(y, 'sqrt', [this], (g) => {
      const dx = new Float32Array(y.data.length);
      for (let i = 0; i < dx.length; i++) dx[i] = g.data[i] * 0.5 / y.data[i];
      accumulate(this, dx);
    });
  }

  /** x^p for a plain number p; derivative p·x^(p-1). */
  pow(p) {
    return fromOp(ops.pow(this, p), 'pow', [this], (g) => {
      const dx = new Float32Array(this.data.length);
      for (let i = 0; i < dx.length; i++) dx[i] = g.data[i] * p * Math.pow(this.data[i], p - 1);
      accumulate(this, dx);
    });
  }

  // ---------- row ops along the last dimension ----------

  /** Softmax over the last dim. With p the output: g_x = p·(g - Σ g·p) per row. */
  softmax() {
    const p = ops.softmax(this);
    const d = this.shape[this.shape.length - 1];
    return fromOp(p, 'softmax', [this], (g) => {
      const dx = new Float32Array(p.data.length);
      for (let r = 0; r < dx.length; r += d) {
        let dot = 0;
        for (let j = 0; j < d; j++) dot += g.data[r + j] * p.data[r + j];
        for (let j = 0; j < d; j++) dx[r + j] = p.data[r + j] * (g.data[r + j] - dot);
      }
      accumulate(this, dx);
    });
  }

  /** log(softmax) over the last dim, computed stably. g_x = g - softmax(x)·Σ g per row. */
  logSoftmax() {
    const y = ops.logSoftmax(this);
    const d = this.shape[this.shape.length - 1];
    return fromOp(y, 'logSoftmax', [this], (g) => {
      const dx = new Float32Array(y.data.length);
      for (let r = 0; r < dx.length; r += d) {
        let total = 0;
        for (let j = 0; j < d; j++) total += g.data[r + j];
        for (let j = 0; j < d; j++) dx[r + j] = g.data[r + j] - Math.exp(y.data[r + j]) * total;
      }
      accumulate(this, dx);
    });
  }

  /**
   * LayerNorm over the last dim: y = x̂·γ + β with x̂ = (x - μ)/√(σ² + eps). gamma/beta are Tensors [d] or null.
   * Backward per row, with gγ = g·γ: dx = inv·(gγ - mean(gγ) - x̂·mean(gγ·x̂)); dγ = Σrows g·x̂; dβ = Σrows g.
   */
  layerNorm(gamma = null, beta = null, eps = 1e-5) {
    const d = this.shape[this.shape.length - 1];
    const inputs = [this];
    if (gamma) inputs.push(gamma);
    if (beta) inputs.push(beta);
    return fromOp(ops.layerNorm(this, gamma, beta, eps), 'layerNorm', inputs, (g) => {
      const x = this.data, G = g.data;
      const dx = new Float32Array(x.length);
      const dGamma = gamma ? new Float32Array(d) : null;
      const dBeta = beta ? new Float32Array(d) : null;
      const xhat = new Float32Array(d), gg = new Float32Array(d);
      for (let r = 0; r < x.length; r += d) {
        // Recompute this row's statistics, exactly as the forward did.
        let mu = 0;
        for (let j = 0; j < d; j++) mu += x[r + j];
        mu /= d;
        let variance = 0;
        for (let j = 0; j < d; j++) { const c = x[r + j] - mu; variance += c * c; }
        variance /= d;
        const inv = 1 / Math.sqrt(variance + eps);
        let mean1 = 0, mean2 = 0;
        for (let j = 0; j < d; j++) {
          xhat[j] = (x[r + j] - mu) * inv;
          gg[j] = G[r + j] * (gamma ? gamma.data[j] : 1);
          mean1 += gg[j];
          mean2 += gg[j] * xhat[j];
        }
        mean1 /= d;
        mean2 /= d;
        for (let j = 0; j < d; j++) {
          dx[r + j] = inv * (gg[j] - mean1 - xhat[j] * mean2);
          if (dGamma) dGamma[j] += G[r + j] * xhat[j];
          if (dBeta) dBeta[j] += G[r + j];
        }
      }
      accumulate(this, dx);
      if (gamma) accumulate(gamma, dGamma);
      if (beta) accumulate(beta, dBeta);
    });
  }

  /** Replace elements where mask (raw or Tensor, broadcastable) is 0 with value; gradient passes where mask != 0. */
  maskedFill(mask, value) {
    return fromOp(ops.maskedFill(this, mask, value), 'maskedFill', [this], (g) => {
      const passed = ops.binary(g, mask, (x, m) => (m === 0 ? 0 : x));
      accumulate(this, unbroadcast(passed.data, g.shape, this.shape));
    });
  }

  // ---------- backward ----------

  /** Reverse-mode pass from this scalar: seed grad = 1, then run every _backward in reverse topological order. */
  backward() {
    if (this.data.length !== 1) throw new Error(`backward: root must have size 1, got shape [${this.shape}]`);
    if (!this.requiresGrad) throw new Error('backward: this tensor has no recorded graph (nothing requires grad)');
    const order = topoSort(this);
    // Intermediate grads are scratch space for one pass; only leaves keep accumulating across calls.
    for (const t of order) if (t._children.length > 0) t.grad = null;
    accumulate(this, Float32Array.of(1));
    for (let i = order.length - 1; i >= 0; i--) {
      if (order[i]._backward !== null) order[i]._backward();
    }
  }
}

/** Mean negative log-likelihood of targets under logits [N,V] or [B,T,V] (targets nested to match). */
export function crossEntropy(logits, targets) {
  const V = logits.shape[logits.shape.length - 1];
  const ids = Array.isArray(targets) ? targets.flat(Infinity) : Array.from(targets);
  const N = ids.length;
  if (N * V !== logits.data.length) {
    throw new Error(`crossEntropy: ${N} targets do not match logits of shape [${logits.shape}]`);
  }
  // Fused log-softmax + pick keeps this stable for large logits (no exp overflow, no log(0)).
  const logProbs = ops.logSoftmax(logits);
  let loss = 0;
  for (let i = 0; i < N; i++) {
    if (!(ids[i] >= 0 && ids[i] < V)) throw new Error(`crossEntropy: target ${ids[i]} out of range for V=${V}`);
    loss -= logProbs.data[i * V + ids[i]];
  }
  loss /= N;
  return fromOp({ shape: [], data: Float32Array.of(loss) }, 'crossEntropy', [logits], (g) => {
    // d loss / d logits = (softmax - onehot) / N, times the incoming scalar gradient.
    const s = g.data[0] / N;
    const dLogits = new Float32Array(logProbs.data.length);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < V; j++) dLogits[i * V + j] = Math.exp(logProbs.data[i * V + j]) * s;
      dLogits[i * V + ids[i]] -= s;
    }
    accumulate(logits, dLogits);
  });
}

/** Up to `count` element indices of an n-element tensor, spread evenly (deterministic). */
function checkIndices(n, count) {
  const k = Math.min(n, count);
  const indices = [];
  for (let j = 0; j < k; j++) indices.push(Math.floor((j * n) / k));
  return indices;
}

/**
 * Compare analytic gradients of the scalar fn(...inputs) with central differences, element by element
 * (at most 40 per input). Data is float32, so eps ≈ 1e-3 and tol ≈ 1e-2 are the useful settings; the error is
 * relative for values above 1 in magnitude and absolute below (where float32 rounding noise dominates).
 */
export function gradCheck(fn, inputs, { eps = 1e-3, tol = 1e-2 } = {}) {
  inputs.forEach((t, k) => {
    if (!t.requiresGrad) throw new Error(`gradCheck: input ${k} must have requiresGrad (use Tensor.param)`);
    t.zeroGrad();
  });
  const out = fn(...inputs);
  if (out.size !== 1) throw new Error('gradCheck: fn must return a scalar Tensor');
  out.backward();
  const analytic = inputs.map((t) => (t.grad ? Float32Array.from(t.grad) : new Float32Array(t.size)));

  const details = [];
  let maxRelErr = 0;
  inputs.forEach((t, k) => {
    for (const index of checkIndices(t.size, 40)) {
      const saved = t.data[index];
      // Perturb the float32 storage in place and measure the perturbation that actually landed.
      t.data[index] = saved + eps;
      const hi = t.data[index];
      const fPlus = noGrad(() => fn(...inputs)).item();
      t.data[index] = saved - eps;
      const lo = t.data[index];
      const fMinus = noGrad(() => fn(...inputs)).item();
      t.data[index] = saved;
      const numeric = (fPlus - fMinus) / (hi - lo);
      const a = analytic[k][index];
      const relErr = Math.abs(a - numeric) / Math.max(1, Math.abs(a), Math.abs(numeric));
      details.push({ input: k, index, analytic: a, numeric, relErr });
      if (!(relErr <= maxRelErr)) maxRelErr = relErr; // also catches NaN
    }
  });
  return { ok: maxRelErr <= tol, maxRelErr, details };
}
