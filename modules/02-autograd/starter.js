// Autograd from scratch.
// A Tensor wraps a raw tensor ({ shape, data: Float32Array }, as built in the Tensors module) and, when any input requires a
// gradient, remembers how it was made: the input Tensors (_children), the op name (_op) and a closure
// (_backward) that reads the gradient sitting in this.grad and ADDS each input's share into input.grad.
// backward() on a scalar loss walks that record from the output back to the leaves: the chain rule, run
// in reverse. Every forward op below is already written with lib/ops.js; you write the backward side.
// The comments marked "JS:" explain JavaScript that earlier modules did not need.

// JS: `import * as ops from 'lib/ops.js'` loads every exported function of the lab's reference tensor library
// into one object named ops, so ops.matmul is its matmul and ops.transpose its transpose (the same kernels you
// wrote in the Tensors module, in a more general form).
import * as ops from 'lib/ops.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

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

/**
 * Wrap an op's raw result and, if any input requires a gradient, record how to push gradient back.
 * backwardFn receives g = { shape, data }: the gradient of the loss with respect to this op's OUTPUT.
 * Its job is to call accumulate(input, dInput) for each input. This is the micrograd pattern.
 */
function fromOp(raw, op, inputs, backwardFn) {
  const out = new Tensor(raw);
  if (inputs.some((t) => t.requiresGrad)) {
    out.requiresGrad = true;
    out._op = op;
    out._children = inputs;
    // JS: `() => backwardFn(...)` is a closure: a small function made here and stored for later. It still
    // "remembers" out and backwardFn after fromOp has returned, so backward() can call it much later.
    // Every op below passes its own closure, (g) => { ... }, as backwardFn in the same way.
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

/**
 * The reverse of a reduction: spread the reduced gradient g back over every element of `shape` that was
 * reduced along `axis` (null = all elements), multiplied by `factor`. Index arithmetic as in the Tensors module.
 */
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

/** Every node reachable from root, each exactly once, with every node's _children listed before it; root last. */
export function topoSort(root) {
  // TODO: step 1
  return [];
}

// ---------- step 3: broadcasting ----------

/**
 * Sum `grad` (laid out as gradShape) down to targetShape, the input shape it was broadcast from.
 * Example: a [3] bias added to a [2, 3] tensor got a [2, 3] output gradient; the bias needs [3].
 */
export function unbroadcast(grad, gradShape, targetShape) {
  // TODO: step 3 — this placeholder is only right when gradShape equals targetShape
  return grad;
}

/** A raw tensor plus gradient bookkeeping. Every op returns a new Tensor; nothing is modified in place. */
// JS: a class is a recipe for objects that share the same methods. `new Tensor(raw)` makes one object (an
// "instance") and runs its constructor; every method below, such as add or backward, is then called on an
// instance with a dot, x.add(y), and inside it `this` is that instance (x).
export class Tensor {
  /** Wraps raw ({ shape, data }) without copying. requiresGrad marks a leaf whose gradient you want. */
  // JS: the constructor runs once, inside `new Tensor(...)`, and fills in this object's fields (this.shape, ...).
  // `{ requiresGrad = false } = {}` unpacks an options object with a default: new Tensor(raw) and
  // new Tensor(raw, { requiresGrad: true }) both work.
  constructor(raw, { requiresGrad = false } = {}) {
    if (!raw || !raw.shape || !raw.data) throw new Error('Tensor: expected a raw tensor { shape, data }');
    this.shape = raw.shape;
    this.data = raw.data;
    this.grad = null;             // Float32Array of the same length as data, or null until a backward pass
    this.requiresGrad = requiresGrad;
    this._children = [];          // the Tensors this one was computed from
    this._backward = null;        // closure that pushes this.grad into the children
    this._op = '';                // the op name, for debugging and for the demo
  }

  // ---------- creation (done) ----------

  /** Tensor from nested JS arrays (or a single number): Tensor.from([[1, 2], [3, 4]]) has shape [2, 2]. */
  // JS: `static` means the method belongs to the class itself, not to one tensor: you call Tensor.from(...),
  // not x.from(...). It is a way of making new tensors.
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

  // ---------- inspection (done) ----------

  // JS: `get` makes a getter: t.size reads like a field (no brackets) but runs this function each time.
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

  /** Forget the accumulated gradient (call this before every backward pass of a training loop). */
  zeroGrad() {
    this.grad = null;
  }

  // ---------- elementwise, with broadcasting; o is a Tensor or a number ----------

  /** this + o. Fully worked: each input receives the output gradient, summed back to its own shape. */
  add(o) {
    return fromOp(ops.add(this, o), 'add', tensorInputs(this, o), (g) => {
      accumulate(this, unbroadcast(g.data, g.shape, this.shape));
      if (o instanceof Tensor) accumulate(o, unbroadcast(g.data, g.shape, o.shape));
    });
  }

  /** this - o. Worked: the second input receives the negated gradient. */
  sub(o) {
    return fromOp(ops.sub(this, o), 'sub', tensorInputs(this, o), (g) => {
      accumulate(this, unbroadcast(g.data, g.shape, this.shape));
      if (o instanceof Tensor) accumulate(o, unbroadcast(ops.neg(g).data, g.shape, o.shape));
    });
  }

  /** this * o. d(a*b)/da = b and d(a*b)/db = a, each multiplied by the incoming gradient g. */
  mul(o) {
    return fromOp(ops.mul(this, o), 'mul', tensorInputs(this, o), (g) => {
      accumulate(this, unbroadcast(ops.mul(g, o).data, g.shape, this.shape));
      // TODO: step 3 — the gradient for o (only when o is a Tensor)
    });
  }

  /** this * s for a plain number s. Worked. */
  scale(s) {
    return fromOp(ops.scale(this, s), 'scale', [this], (g) => accumulate(this, ops.scale(g, s).data));
  }

  /** -this. Worked. */
  neg() {
    return fromOp(ops.neg(this), 'neg', [this], (g) => accumulate(this, ops.neg(g).data));
  }

  // ---------- step 2: matmul and reductions ----------

  /** Matrix product C = A·B (2-D). With dC the output gradient: dA = dC·Bᵀ and dB = Aᵀ·dC. */
  matmul(o) {
    return fromOp(ops.matmul(this, o), 'matmul', tensorInputs(this, o), (g) => {
      // TODO: step 2
    });
  }

  /** Sum over axis (or everything when axis is null). The gradient is copied to every element that was summed. */
  sum(axis = null, keepDims = false) {
    axis = resolveAxis(axis, this.shape.length);
    return fromOp(ops.sum(this, axis, keepDims), 'sum', [this], (g) => {
      // TODO: step 2
    });
  }

  /** Mean over axis (or everything): like sum, with the gradient divided by the number of elements averaged. */
  mean(axis = null, keepDims = false) {
    axis = resolveAxis(axis, this.shape.length);
    return fromOp(ops.mean(this, axis, keepDims), 'mean', [this], (g) => {
      // TODO: step 2
    });
  }

  // ---------- step 4: nonlinearities ----------

  /** e^x. The derivative is the output itself. */
  exp() {
    const y = ops.exp(this);
    return fromOp(y, 'exp', [this], (g) => {
      // TODO: step 4
    });
  }

  /** ln x. The derivative is 1/x. */
  log() {
    return fromOp(ops.log(this), 'log', [this], (g) => {
      // TODO: step 4
    });
  }

  /** max(x, 0). The gradient passes only where x > 0. */
  relu() {
    return fromOp(ops.relu(this), 'relu', [this], (g) => {
      // TODO: step 4
    });
  }

  // ---------- step 1: backward ----------

  /** Reverse-mode pass from this scalar: seed grad = 1, then run every _backward in reverse topological order. */
  backward() {
    if (this.data.length !== 1) throw new Error(`backward: root must have size 1, got shape [${this.shape}]`);
    if (!this.requiresGrad) throw new Error('backward: this tensor has no recorded graph (nothing requires grad)');
    // TODO: step 1
  }
}

// ---------- step 4: fused cross-entropy ----------

/**
 * Mean negative log-likelihood of integer targets under logits [N, V]. The forward pass is done:
 * log-softmax is computed in one stable step (row max subtracted inside ops.logSoftmax).
 * The gradient with respect to the logits is (softmax − onehot) / N, times the incoming gradient.
 */
export function crossEntropy(logits, targets) {
  const V = logits.shape[logits.shape.length - 1];
  const ids = Array.isArray(targets) ? targets.flat(Infinity) : Array.from(targets);
  const N = ids.length;
  if (N * V !== logits.data.length) {
    throw new Error(`crossEntropy: ${N} targets do not match logits of shape [${logits.shape}]`);
  }
  const logProbs = ops.logSoftmax(logits);   // raw [N, V]; softmax(logits) is exp(logProbs)
  let loss = 0;
  for (let i = 0; i < N; i++) {
    if (!(ids[i] >= 0 && ids[i] < V)) throw new Error(`crossEntropy: target ${ids[i]} out of range for V=${V}`);
    loss -= logProbs.data[i * V + ids[i]];
  }
  loss /= N;
  return fromOp({ shape: [], data: Float32Array.of(loss) }, 'crossEntropy', [logits], (g) => {
    // TODO: step 4
  });
}

// ---------- step 5: a numeric derivative of one element ----------

/**
 * The central-difference derivative of the scalar fn(...inputs) with respect to ONE number,
 * inputs[k].data[index]:
 *   (fn(x + eps) − fn(x − eps)) / (hi − lo)
 * where hi and lo are the perturbed values read back from the Float32Array (float32 rounds, so they can
 * differ from x ± eps). The value is restored before returning.
 */
export function numericDerivative(fn, inputs, k, index, eps = 1e-3) {
  // TODO: step 5
  return NaN;
}

// ---------- step 6: the gradient check ----------

/**
 * Compare the analytic gradient of the scalar fn(...inputs) with numericDerivative, element by element:
 *   relErr = |analytic − numeric| / max(1, |analytic|, |numeric|)
 * Returns { ok: maxRelErr <= tol, maxRelErr, details: [{ input, index, analytic, numeric, relErr }] }.
 * input is the 0-based position of the tensor in `inputs`; index is the flat element index in its data.
 */
export function gradCheck(fn, inputs, { eps = 1e-3, tol = 1e-2 } = {}) {
  const analytic = [];   // analytic[k] will be a copy of inputs[k].grad
  // TODO: step 6, part 1. Throw an Error if any input has requiresGrad false, and call zeroGrad() on every
  // input. Run fn(...inputs) once, throw unless the result has size 1, and call backward() on it.
  // Then push a copy of each input's gradient into `analytic` (zeros when its grad is still null).

  const details = [];
  let maxRelErr = 0;
  for (let k = 0; k < inputs.length; k++) {                   // every input tensor...
    for (let index = 0; index < inputs[k].size; index++) {    // ...and every element of it
      // TODO: step 6, part 2. Compare analytic[k][index] with numericDerivative(fn, inputs, k, index, eps),
      // push one { input: k, index, analytic, numeric, relErr } into details, and update maxRelErr so that a
      // NaN relErr gets in and nothing afterwards can replace it.
    }
  }
  return { ok: maxRelErr <= tol, maxRelErr, details };
}

// ---------- step 7: SGD and linear regression ----------

/** One gradient-descent update, p.data -= lr * p.grad, for every parameter that has a gradient. Leaves .grad alone. */
export function sgdStep(params, lr) {
  // TODO: step 7
}

/**
 * Fit y ≈ w·x + b by full-batch gradient descent on the mean squared error, starting from w = 0, b = 0.
 * xs, ys: plain arrays of N numbers. Returns { w, b, losses, ws, bs }: the fitted numbers and, per step,
 * the loss before the update and w and b after it.
 */
export function trainLinear(xs, ys, { steps = 200, lr = 0.1 } = {}) {
  // TODO: step 7
  return { w: 0, b: 0, losses: [], ws: [], bs: [] };
}
