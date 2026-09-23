// Module 02 — reference solution. The same engine, with more ops, lives in lib/tensor.js.
//
// A Tensor wraps a raw tensor ({ shape, data: Float32Array }) and, when any input requires a gradient,
// remembers how it was made: the input Tensors (_children), the op name (_op) and a closure
// (_backward) that reads the gradient sitting in this.grad and adds each input's share into
// input.grad. backward() on a scalar walks that record from the output back to the leaves.

import * as ops from 'lib/ops.js';

// ---------- gradient bookkeeping ----------

/** t.grad, allocated as zeros on first use. */
function gradOf(t) {
  if (t.grad === null) t.grad = new Float32Array(t.data.length);
  return t.grad;
}

/** Add the Float32Array g into t.grad with +=, so every use of t contributes its share. */
function accumulate(t, g) {
  if (!t.requiresGrad) return;
  const G = gradOf(t);
  if (g.length !== G.length) throw new Error(`accumulate: gradient has ${g.length} values but the tensor has ${G.length}`);
  for (let i = 0; i < G.length; i++) G[i] += g[i];
}

/** Wrap an op's raw result and, if any input requires a gradient, record how to push gradient back. */
function fromOp(raw, op, inputs, backwardFn /* (g: raw gradient of the output) => void */) {
  const out = new Tensor(raw);
  if (inputs.some((t) => t.requiresGrad)) {
    out.requiresGrad = true;
    out._op = op;
    out._children = inputs;
    out._backward = () => backwardFn({ shape: out.shape, data: out.grad });
  }
  return out;
}

/** The Tensor inputs of a binary op (a plain-number operand has no gradient). */
function tensorInputs(a, b) {
  return b instanceof Tensor ? [a, b] : [a];
}

/** Negative axes count from the end; null means "all elements". */
function resolveAxis(axis, ndim) {
  if (axis === null || axis === undefined) return null;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) throw new Error(`axis ${axis} out of range for ndim ${ndim}`);
  return axis;
}

/** Spread a reduced gradient g back over every element of `shape` that was reduced along axis, times factor. */
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

// ---------- step 1: the graph ----------

/** Every node reachable from root, each exactly once, children before parents (post-order DFS). */
export function topoSort(root) {
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

// ---------- step 3: broadcasting ----------

/** Sum `grad` (laid out as gradShape) down to targetShape, the input shape it was broadcast from. */
export function unbroadcast(grad, gradShape, targetShape) {
  let t = { shape: gradShape, data: grad };
  // Leading dims that broadcasting created: sum them away.
  while (t.shape.length > targetShape.length) t = ops.sum(t, 0);
  // Dims the input had as size 1 were stretched: sum them back to size 1.
  for (let i = 0; i < targetShape.length; i++) {
    if (targetShape[i] === 1 && t.shape[i] !== 1) t = ops.sum(t, i, true);
  }
  return t.data;
}

/** A raw tensor plus gradient bookkeeping. Every op returns a new Tensor; nothing is modified in place. */
export class Tensor {
  /** Wraps raw ({ shape, data }) without copying. requiresGrad marks a leaf whose gradient you want. */
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

  /** Tensor from nested JS arrays (or a single number): Tensor.from([[1, 2], [3, 4]]) has shape [2, 2]. */
  static from(nested, opts) {
    return new Tensor(ops.fromArray(nested), opts);
  }

  static zeros(shape, opts) {
    return new Tensor(ops.zeros(shape), opts);
  }

  static ones(shape, opts) {
    return new Tensor(ops.ones(shape), opts);
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

  get size() {
    return this.data.length;
  }

  /** The single value of a size-1 tensor as a plain number. */
  item() {
    if (this.data.length !== 1) throw new Error(`item: tensor has ${this.data.length} elements, not 1`);
    return this.data[0];
  }

  toArray() {
    return ops.toArray(this);
  }

  /** Forget the accumulated gradient (the optimizer calls this before every backward pass). */
  zeroGrad() {
    this.grad = null;
  }

  // ---------- elementwise, with broadcasting; o is a Tensor or a number ----------

  /** this + o. Each input receives the output gradient, summed back to its own shape. */
  add(o) {
    return fromOp(ops.add(this, o), 'add', tensorInputs(this, o), (g) => {
      accumulate(this, unbroadcast(g.data, g.shape, this.shape));
      if (o instanceof Tensor) accumulate(o, unbroadcast(g.data, g.shape, o.shape));
    });
  }

  /** this - o. The second input receives the negated gradient. */
  sub(o) {
    return fromOp(ops.sub(this, o), 'sub', tensorInputs(this, o), (g) => {
      accumulate(this, unbroadcast(g.data, g.shape, this.shape));
      if (o instanceof Tensor) accumulate(o, unbroadcast(ops.neg(g).data, g.shape, o.shape));
    });
  }

  /** this * o. d(a*b)/da = b and d(a*b)/db = a, each multiplied by the incoming gradient. */
  mul(o) {
    return fromOp(ops.mul(this, o), 'mul', tensorInputs(this, o), (g) => {
      accumulate(this, unbroadcast(ops.mul(g, o).data, g.shape, this.shape));
      if (o instanceof Tensor) accumulate(o, unbroadcast(ops.mul(g, this).data, g.shape, o.shape));
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

  // ---------- step 2: matmul and reductions ----------

  /** Matrix product A·B (2-D, or batched with a shared 2-D B). dA = dC·Bᵀ and dB = Aᵀ·dC. */
  matmul(o) {
    return fromOp(ops.matmul(this, o), 'matmul', tensorInputs(this, o), (g) => {
      if (this.requiresGrad) accumulate(this, ops.matmul(g, ops.transpose(o)).data);
      if (o.requiresGrad) {
        // A shared 2-D B was used by every batch element, so its gradient is the sum over the batch.
        const dB = ops.matmul(ops.transpose(this), g);
        accumulate(o, unbroadcast(dB.data, dB.shape, o.shape));
      }
    });
  }

  /** Sum over axis (or everything). The gradient is copied to every element that was summed. */
  sum(axis = null, keepDims = false) {
    axis = resolveAxis(axis, this.shape.length);
    return fromOp(ops.sum(this, axis, keepDims), 'sum', [this], (g) => {
      accumulate(this, expandAlongAxis(g.data, this.shape, axis, 1));
    });
  }

  /** Mean over axis (or everything): like sum, with the gradient divided by the number of elements averaged. */
  mean(axis = null, keepDims = false) {
    axis = resolveAxis(axis, this.shape.length);
    const count = axis === null ? this.data.length : this.shape[axis];
    return fromOp(ops.mean(this, axis, keepDims), 'mean', [this], (g) => {
      accumulate(this, expandAlongAxis(g.data, this.shape, axis, 1 / count));
    });
  }

  // ---------- step 4: nonlinearities ----------

  /** e^x; the derivative is the output itself, so keep it instead of recomputing. */
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

  /** max(x, 0); the gradient passes only where x > 0. */
  relu() {
    return fromOp(ops.relu(this), 'relu', [this], (g) => {
      const dx = new Float32Array(this.data.length);
      for (let i = 0; i < dx.length; i++) dx[i] = this.data[i] > 0 ? g.data[i] : 0;
      accumulate(this, dx);
    });
  }

  // ---------- step 1: backward ----------

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

// ---------- step 4: fused cross-entropy ----------

/** Mean negative log-likelihood of integer targets under logits [N, V]. Gradient: (softmax − onehot) / N. */
export function crossEntropy(logits, targets) {
  const V = logits.shape[logits.shape.length - 1];
  const ids = Array.isArray(targets) ? targets.flat(Infinity) : Array.from(targets);
  const N = ids.length;
  if (N * V !== logits.data.length) {
    throw new Error(`crossEntropy: ${N} targets do not match logits of shape [${logits.shape}]`);
  }
  // Fused log-softmax + pick: no exp overflow for large logits and no log(0) for confident wrong answers.
  const logProbs = ops.logSoftmax(logits);
  let loss = 0;
  for (let i = 0; i < N; i++) {
    if (!(ids[i] >= 0 && ids[i] < V)) throw new Error(`crossEntropy: target ${ids[i]} out of range for V=${V}`);
    loss -= logProbs.data[i * V + ids[i]];
  }
  loss /= N;
  return fromOp({ shape: [], data: Float32Array.of(loss) }, 'crossEntropy', [logits], (g) => {
    const s = g.data[0] / N;
    const dLogits = new Float32Array(logProbs.data.length);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < V; j++) dLogits[i * V + j] = Math.exp(logProbs.data[i * V + j]) * s;
      dLogits[i * V + ids[i]] -= s;
    }
    accumulate(logits, dLogits);
  });
}

// ---------- step 5: numerical gradient check ----------

/**
 * Compare the analytic gradient of the scalar fn(...inputs) with central differences, element by element.
 * Data is float32, so eps ≈ 1e-3 and tol ≈ 1e-2 are the useful settings. The error is relative for values
 * above 1 in magnitude and absolute below, where float32 rounding noise dominates.
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
    for (let index = 0; index < t.size; index++) {
      const saved = t.data[index];
      // Perturb the float32 storage in place and measure the perturbation that actually landed.
      t.data[index] = saved + eps;
      const hi = t.data[index];
      const fPlus = fn(...inputs).item();
      t.data[index] = saved - eps;
      const lo = t.data[index];
      const fMinus = fn(...inputs).item();
      t.data[index] = saved;
      const numeric = (fPlus - fMinus) / (hi - lo);
      const a = analytic[k][index];
      const relErr = Math.abs(a - numeric) / Math.max(1, Math.abs(a), Math.abs(numeric));
      details.push({ input: k, index, analytic: a, numeric, relErr });
      // NaN compares false with everything: let it in, and once in, nothing larger can replace it.
      if (Number.isNaN(relErr) || relErr > maxRelErr) maxRelErr = relErr;
    }
  });
  return { ok: maxRelErr <= tol, maxRelErr, details };
}

// ---------- step 6: SGD and linear regression ----------

/** One gradient-descent update, p.data -= lr * p.grad, for every parameter that has a gradient. Leaves .grad alone. */
export function sgdStep(params, lr) {
  for (const p of params) {
    if (!p.grad) continue;
    for (let i = 0; i < p.data.length; i++) p.data[i] -= lr * p.grad[i];
  }
}

/**
 * Fit y ≈ w·x + b by full-batch gradient descent on the mean squared error.
 * xs, ys: plain arrays of N numbers. Returns the fitted w and b plus the per-step loss, w and b.
 */
export function trainLinear(xs, ys, { steps = 200, lr = 0.1 } = {}) {
  const X = Tensor.from(xs.map((x) => [x]));   // [N, 1]
  const Y = Tensor.from(ys.map((y) => [y]));   // [N, 1]
  const w = Tensor.zeros([1, 1], { requiresGrad: true });
  const b = Tensor.zeros([1], { requiresGrad: true });
  const losses = [], ws = [], bs = [];
  for (let step = 0; step < steps; step++) {
    const pred = X.matmul(w).add(b);           // [N, 1]; b is broadcast over the rows
    const diff = pred.sub(Y);
    const loss = diff.mul(diff).mean();
    losses.push(loss.item());
    w.zeroGrad();
    b.zeroGrad();
    loss.backward();
    sgdStep([w, b], lr);
    ws.push(w.data[0]);
    bs.push(b.data[0]);
  }
  return { w: w.data[0], b: b.data[0], losses, ws, bs };
}
