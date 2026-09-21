import * as ops from 'lib/ops.js';
import { rng } from 'lib/util.js';

/** A tensor filled deterministically from the seeded rng, uniform in [-1, 1). */
function randomTensor(m, shape, seed, opts) {
  const next = rng(seed);
  const t = new m.Tensor(ops.zeros(shape), opts);
  for (let i = 0; i < t.data.length; i++) t.data[i] = next() * 2 - 1;
  return t;
}

/** Central-difference gradient of the scalar fn() with respect to every element of t (t.data is restored). */
function numericGrad(fn, t, eps = 1e-3) {
  const out = new Float32Array(t.data.length);
  for (let i = 0; i < out.length; i++) {
    const saved = t.data[i];
    t.data[i] = saved + eps; const hi = t.data[i]; const fp = fn().item();
    t.data[i] = saved - eps; const lo = t.data[i]; const fm = fn().item();
    t.data[i] = saved;
    out[i] = (fp - fm) / (hi - lo);
  }
  return out;
}

export const tests = [
  // ---------- step 1: graph ----------
  { step: 'graph', name: 'topoSort lists every node once, children before parents, root last', run(m, T) {
    const a = m.Tensor.from(1, { requiresGrad: true }), b = m.Tensor.from(2, { requiresGrad: true });
    const x = a.add(b);
    const p = x.add(a), q = x.add(b);   // a diamond: x is reachable through both p and q
    const y = p.add(q);
    const order = m.topoSort(y);
    T.eq(order.length, 6, 'the graph has 6 distinct nodes (a, b, x, p, q, y); each must appear exactly once even though x is reached twice');
    T.ok(order[order.length - 1] === y, 'the root must come last');
    for (const t of order) for (const c of t._children) T.ok(order.indexOf(c) < order.indexOf(t), `child (${c._op || 'leaf'}) must be listed before its parent (${t._op})`);
    T.eq(m.topoSort(a).length, 1, 'a leaf on its own is a one-node graph');
  } },
  { step: 'graph', name: 'backward seeds the root with 1 and accumulates every path into the leaves', run(m, T) {
    const a = m.Tensor.from(2, { requiresGrad: true }), b = m.Tensor.from(3, { requiresGrad: true });
    const x = a.add(b);
    const y = x.add(a);        // y = 2a + b, and a is used twice
    y.backward();
    T.ok(y.grad !== null && a.grad !== null, 'after backward() the root and every leaf that requires a gradient must hold a Float32Array in .grad, not null: seed the root with 1 and run every _backward');
    T.eq(Array.from(y.grad), [1], 'the root gradient dy/dy is 1');
    T.eq(Array.from(x.grad), [1], 'x feeds y once, so x.grad = 1');
    T.eq(Array.from(a.grad), [2], 'a is used twice (in x and directly in y): the two contributions must ADD to 2');
    T.eq(Array.from(b.grad), [1]);
    const c = m.Tensor.from(5);   // no requiresGrad: a constant
    const z = a.add(c);
    z.backward();
    T.ok(c.grad === null, 'a tensor without requiresGrad never receives a gradient');
  } },
  { step: 'graph', name: 'backward rejects non-scalars; leaf grads accumulate across calls until zeroGrad', run(m, T) {
    const v = m.Tensor.from([1, 2], { requiresGrad: true });
    T.throws(() => v.add(v).backward(), 'backward on a size-2 tensor must throw: reverse mode starts from one scalar loss');
    const a = m.Tensor.from(2, { requiresGrad: true }), b = m.Tensor.from(3, { requiresGrad: true });
    const x = a.add(b);
    const y = x.add(a);
    y.backward();
    y.backward();
    T.ok(a.grad !== null, 'backward() must fill a.grad');
    T.eq(Array.from(a.grad), [4], 'two backward passes on the same graph must give exactly twice the leaf gradient: intermediates (x, y) are scratch space and must be reset each pass, leaves accumulate');
    T.eq(Array.from(b.grad), [2]);
    a.zeroGrad();
    T.ok(a.grad === null, 'zeroGrad forgets the accumulated gradient');
    y.backward();
    T.eq(Array.from(a.grad), [2], 'after zeroGrad a fresh pass starts from nothing');
  } },

  // ---------- step 2: matmul and reductions ----------
  { step: 'matmul', name: 'sum() spreads the gradient to every element; mean() divides it by the count', run(m, T) {
    const x = randomTensor(m, [2, 3], 1, { requiresGrad: true });
    x.sum().backward();
    T.close(x.grad, [1, 1, 1, 1, 1, 1], 1e-6, 'd(sum)/dx is 1 for every element');
    x.zeroGrad();
    x.mean().backward();
    T.close(x.grad, new Array(6).fill(1 / 6), 1e-6, 'd(mean)/dx is 1/N for every element (N = 6 here); forgetting the 1/N makes the gradient 6× too large');
    x.zeroGrad();
    x.sum().scale(3).backward();
    T.close(x.grad, new Array(6).fill(3), 1e-6, 'the incoming gradient (3 from scale) must be spread, not a constant 1');
  } },
  { step: 'matmul', name: 'sum(axis) and mean(axis) route the gradient back along the reduced axis', run(m, T) {
    const x = m.Tensor.from([[1, 2, 3], [4, 5, 6]], { requiresGrad: true });
    const s = x.sum(0);
    T.shape(s, [3]);
    s.mul(m.Tensor.from([1, 10, 100])).sum().backward();
    T.close(x.grad, [1, 10, 100, 1, 10, 100], 1e-6, 'summing over axis 0: column j of every row receives the gradient of output j');
    x.zeroGrad();
    const mu = x.mean(1);
    T.shape(mu, [2]);
    T.close(mu, [2, 5], 1e-6);
    mu.mul(m.Tensor.from([3, 30])).sum().backward();
    T.close(x.grad, [1, 1, 1, 10, 10, 10], 1e-6, 'mean over axis 1 (3 elements): row 0 gets 3/3, row 1 gets 30/3');
    x.zeroGrad();
    T.eq(x.sum(-1).shape, [2], 'negative axes count from the end');
  } },
  { step: 'matmul', name: 'matmul: dA = dC·Bᵀ and dB = Aᵀ·dC, and a constant operand gets no gradient', run(m, T) {
    const A = randomTensor(m, [2, 3], 2, { requiresGrad: true });
    const B = randomTensor(m, [3, 4], 3, { requiresGrad: true });
    const W = randomTensor(m, [2, 4], 4);       // fixed weights make dC non-uniform
    A.matmul(B).mul(W).sum().backward();
    const dA = ops.matmul(W, ops.transpose(B)), dB = ops.matmul(ops.transpose(A), W);
    T.close(A.grad, dA.data, 1e-4, 'dA must be dC·Bᵀ, shape [2,3]; the two products are easy to swap');
    T.close(B.grad, dB.data, 1e-4, 'dB must be Aᵀ·dC, shape [3,4]');
    const num = numericGrad(() => A.matmul(B).mul(W).sum(), A);
    T.close(A.grad, num, 2e-2, 'analytic dA disagrees with central differences');
    const C = randomTensor(m, [3, 4], 5);       // no requiresGrad
    A.zeroGrad();
    A.matmul(C).sum().backward();
    T.ok(C.grad === null, 'a constant right operand must not receive a gradient');
    T.close(A.grad, ops.matmul(ops.ones([2, 4]), ops.transpose(C)).data, 1e-4);
  } },

  // ---------- step 3: broadcasting ----------
  { step: 'broadcast', name: 'unbroadcast sums a gradient back to the shape it was broadcast from', run(m, T) {
    const g = new Float32Array([1, 2, 3, 4, 5, 6]);
    T.close(m.unbroadcast(g, [2, 3], [3]), [5, 7, 9], 1e-6, 'a [3] bias added to every row of a [2,3] tensor collects one gradient per row: sum over the leading axis');
    T.close(m.unbroadcast(g, [2, 3], [2, 1]), [6, 15], 1e-6, 'a [2,1] column stretched to 3 columns collects the row sums');
    T.close(m.unbroadcast(g, [2, 3], [2, 3]), [1, 2, 3, 4, 5, 6], 1e-6, 'same shape: nothing to sum');
    T.close(m.unbroadcast(g, [2, 3], []), [21], 1e-6, 'a scalar broadcast everywhere collects the total');
    T.close(m.unbroadcast(g, [2, 3], [1, 3]), [5, 7, 9], 1e-6, 'a [1,3] row keeps its leading size-1 dim');
  } },
  { step: 'broadcast', name: 'a broadcast bias receives the column sums of the output gradient', run(m, T) {
    const X = randomTensor(m, [4, 3], 6);
    const bias = m.Tensor.from([0.1, 0.2, 0.3], { requiresGrad: true });
    const W = randomTensor(m, [4, 3], 7);
    X.add(bias).mul(W).sum().backward();
    T.shape(bias.grad, [3], 'bias.grad must have the bias shape [3], not the output shape [4,3]');
    T.close(bias.grad, ops.sum(W, 0).data, 1e-5, 'each bias entry was added to 4 rows, so its gradient is the sum of that column of dY');
    const col = m.Tensor.from([[1], [2]], { requiresGrad: true });
    const R = m.Tensor.from([[1, 2, 3], [4, 5, 6]]);
    col.sub(R).sum().backward();
    T.close(col.grad, [3, 3], 1e-6, 'a [2,1] column stretched across 3 columns collects 3 contributions per row');
  } },
  { step: 'broadcast', name: 'mul sends a gradient to BOTH operands, unbroadcast to each shape', run(m, T) {
    const a = randomTensor(m, [2, 3], 8, { requiresGrad: true });
    const b = m.Tensor.from([2, 3, 4], { requiresGrad: true });
    a.mul(b).sum().backward();
    T.close(a.grad, [2, 3, 4, 2, 3, 4], 1e-6, 'd(a*b)/da = b, broadcast over the rows');
    T.ok(b.grad !== null, 'the right operand of mul must also receive a gradient');
    T.close(b.grad, ops.sum(a, 0).data, 1e-5, 'd(a*b)/db = a, summed over the 2 rows b was broadcast to');
    a.zeroGrad();
    a.mul(2.5).sum().backward();
    T.close(a.grad, new Array(6).fill(2.5), 1e-6, 'multiplying by a plain number scales the gradient');
  } },
  { step: 'broadcast', name: 'squaring by x.mul(x) gives 2x: both operand paths accumulate into the same leaf', run(m, T) {
    const x = m.Tensor.from([1, -2, 3], { requiresGrad: true });
    x.mul(x).sum().backward();
    T.close(x.grad, [2, -4, 6], 1e-6, 'd(x²)/dx = 2x: the gradient through the left operand (x) plus the gradient through the right operand (x) must both land in x.grad');
    x.zeroGrad();
    const y = x.mul(x).mul(x).sum();
    y.backward();
    T.close(x.grad, [3, 12, 27], 1e-5, 'd(x³)/dx = 3x²');
  } },

  // ---------- step 4: nonlinearities and cross-entropy ----------
  { step: 'nonlinear', name: 'exp and log gradients', run(m, T) {
    const x = m.Tensor.from([0, 1, -1, 2.5], { requiresGrad: true });
    x.exp().sum().backward();
    T.close(x.grad, [1, Math.E, 1 / Math.E, Math.exp(2.5)], 1e-5, 'd(exp x)/dx = exp(x), the OUTPUT of the forward pass, not x itself');
    const p = m.Tensor.from([0.5, 1, 4], { requiresGrad: true });
    p.log().mul(m.Tensor.from([1, 2, 3])).sum().backward();
    T.close(p.grad, [2, 2, 0.75], 1e-5, 'd(log x)/dx = 1/x, times the incoming gradient');
    const q = m.Tensor.from([0.5, 1, 4], { requiresGrad: true });
    const num = numericGrad(() => q.exp().log().mul(q).sum(), q);
    q.exp().log().mul(q).sum().backward();
    T.close(q.grad, num, 1e-2, 'chained exp → log → mul disagrees with central differences');
  } },
  { step: 'nonlinear', name: 'relu passes the gradient only where the input is positive', run(m, T) {
    const x = m.Tensor.from([-2, 0, 3, -0.5, 1], { requiresGrad: true });
    x.relu().mul(m.Tensor.from([1, 2, 3, 4, 5])).sum().backward();
    T.close(x.grad, [0, 0, 3, 0, 5], 1e-6, 'relu gradient is the incoming gradient where x > 0 and exactly 0 elsewhere (including at x = 0)');
    T.close(x.relu(), [0, 0, 3, 0, 1], 1e-6, 'forward value must be unchanged');
  } },
  { step: 'nonlinear', name: 'crossEntropy is the mean negative log-softmax of the targets and is stable for huge logits', run(m, T) {
    const logits = m.Tensor.from([[1, 2, 3], [3, 1, 0]], { requiresGrad: true });
    const loss = m.crossEntropy(logits, [2, 0]);
    T.eq(loss.shape, [], 'the loss is a scalar (shape [])');
    const expect = (-(3 - Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3))) - (3 - Math.log(Math.exp(3) + Math.exp(1) + 1))) / 2;
    T.close(loss.item(), expect, 1e-5, 'loss = mean over rows of −log softmax(logits)[target]');
    const big = m.crossEntropy(m.Tensor.from([[1000, 0, 0]], { requiresGrad: true }), [0]);
    T.ok(Number.isFinite(big.item()) && big.item() < 1e-3, `logits of 1000 must not overflow: fuse log-softmax (subtract the row max) instead of log(softmax(x)); got ${big.item()}`);
    const wrong = m.crossEntropy(m.Tensor.from([[1000, 0, 0]], { requiresGrad: true }), [1]);
    T.close(wrong.item(), 1000, 1e-3, 'a confident wrong answer costs about 1000 nats and must stay finite (log(0) would be −Infinity)');
    T.throws(() => m.crossEntropy(logits, [0, 1, 2]), '3 targets for 2 rows of logits must throw');
  } },
  { step: 'nonlinear', name: 'crossEntropy gradient is (softmax − onehot) / N', run(m, T) {
    const logits = randomTensor(m, [4, 5], 9, { requiresGrad: true });
    for (let i = 0; i < logits.data.length; i++) logits.data[i] *= 3;
    const targets = [3, 0, 4, 1];
    m.crossEntropy(logits, targets).backward();
    T.ok(logits.grad !== null, 'logits.grad is still null after backward(): the crossEntropy closure must call accumulate(logits, dLogits)');
    const p = ops.softmax(logits);
    const expect = new Float32Array(20);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 5; j++) expect[i * 5 + j] = (p.data[i * 5 + j] - (j === targets[i] ? 1 : 0)) / 4;
    T.eq(logits.grad.length, 20, 'logits.grad is a flat Float32Array with one entry per logit');
    T.close(logits.grad, expect, 1e-5, 'dL/dlogits = (softmax − onehot) / N with N = 4 rows; without the 1/N the gradient is 4× too large and training with a tuned lr diverges');
    for (let i = 0; i < 4; i++) {
      let rowSum = 0;
      for (let j = 0; j < 5; j++) rowSum += logits.grad[i * 5 + j];
      T.close(rowSum, 0, 1e-6, 'each row of the gradient sums to 0: raising every logit equally changes nothing');
    }
    const num = numericGrad(() => m.crossEntropy(logits, targets), logits);
    T.close(logits.grad, num, 1e-2, 'analytic gradient disagrees with central differences');
    logits.zeroGrad();
    m.crossEntropy(logits, targets).scale(8).backward();
    T.close(logits.grad, expect.map((v) => 8 * v), 1e-4, 'the incoming gradient (8) must multiply the result; do not assume the loss is the root');
  } },

  // ---------- step 5: gradCheck ----------
  { step: 'gradcheck', name: 'passes a correct MLP and restores the inputs it perturbed', run(m, T) {
    const x = randomTensor(m, [3, 4], 10);
    const W1 = randomTensor(m, [4, 6], 11, { requiresGrad: true });
    const b1 = m.Tensor.from([0.1, -0.2, 0.3, 0, 0.5, -0.1], { requiresGrad: true });
    const W2 = randomTensor(m, [6, 3], 12, { requiresGrad: true });
    const before = [W1, b1, W2].map((t) => Float32Array.from(t.data));
    const res = m.gradCheck((W1, b1, W2) => m.crossEntropy(x.matmul(W1).add(b1).relu().matmul(W2), [0, 2, 1]), [W1, b1, W2]);
    T.ok(res.ok === true, `a correct gradient must pass; maxRelErr=${res.maxRelErr}`);
    T.ok(res.maxRelErr < 1e-2 && res.maxRelErr >= 0, `maxRelErr must be a small non-negative number, got ${res.maxRelErr}`);
    T.eq(res.details.length, 24 + 6 + 18, 'one detail entry per element of every input (24 + 6 + 18)');
    for (const d of res.details) T.ok(typeof d.analytic === 'number' && typeof d.numeric === 'number' && typeof d.relErr === 'number', 'each detail needs analytic, numeric and relErr');
    [W1, b1, W2].forEach((t, k) => T.eq(Array.from(t.data), Array.from(before[k]), 'gradCheck must put every perturbed value back exactly'));
  } },
  { step: 'gradcheck', name: 'catches a backward closure that is off by a factor of 2', run(m, T) {
    const x = m.Tensor.from([1, 2, 3], { requiresGrad: true });
    // A hand-built node whose forward is sum(x) but whose backward claims the derivative is 2 instead of 1.
    const buggySum = (t) => {
      let s = 0;
      for (const v of t.data) s += v;
      const out = new m.Tensor({ shape: [], data: Float32Array.of(s) });
      out.requiresGrad = true;
      out._op = 'buggySum';
      out._children = [t];
      out._backward = () => {
        if (t.grad === null) t.grad = new Float32Array(t.data.length);
        for (let i = 0; i < t.grad.length; i++) t.grad[i] += 2 * out.grad[0];
      };
      return out;
    };
    const res = m.gradCheck(buggySum, [x]);
    T.ok(res.ok === false, 'analytic 2 vs numeric 1 must fail the check');
    T.close(res.maxRelErr, 0.5, 1e-2, 'relErr = |2 − 1| / max(1, 2, 1) = 0.5');
    T.eq(res.details.length, 3);
    T.close(res.details[0].analytic, 2, 1e-6);
    T.close(res.details[0].numeric, 1, 1e-2);
  } },
  { step: 'gradcheck', name: 'uses central differences and honours eps and tol', run(m, T) {
    const x = m.Tensor.from([2], { requiresGrad: true });
    const cube = (t) => t.mul(t).mul(t).sum();
    const res = m.gradCheck(cube, [x], { eps: 0.1, tol: 1e-2 });
    T.eq(res.details.length, 1, 'x has one element, so details must have exactly one entry');
    T.close(res.details[0].analytic, 12, 1e-5, 'd(x³)/dx at 2 is 12');
    T.ok(res.ok && res.maxRelErr < 2e-3, `with eps = 0.1 a central difference of x³ at 2 is 12.01 (error 8e-4); a one-sided difference gives 12.61 (error 5%). Got maxRelErr=${res.maxRelErr}`);
    const strict = m.gradCheck(cube, [x], { eps: 0.1, tol: 1e-6 });
    T.ok(strict.ok === false, 'the same error must fail when tol is 1e-6: ok is maxRelErr <= tol');
    T.close(strict.maxRelErr, res.maxRelErr, 1e-9, 'tol changes the verdict, not the measurement');
    T.throws(() => m.gradCheck(cube, [m.Tensor.from([2])]), 'an input without requiresGrad has no analytic gradient to compare: throw');
    const v = m.Tensor.from([1, 2], { requiresGrad: true });
    T.throws(() => m.gradCheck((t) => t.mul(t), [v]), 'fn must return a scalar: throw when it returns a size-2 vector');
  } },

  // ---------- step 6: SGD and linear regression ----------
  { step: 'train', name: 'sgdStep moves each parameter against its gradient and leaves .grad alone', run(m, T) {
    const p = m.Tensor.from([1, 2, 3], { requiresGrad: true });
    p.grad = new Float32Array([10, -20, 0]);
    const q = m.Tensor.from([5, 5], { requiresGrad: true });   // never had a backward pass: grad is null
    m.sgdStep([p, q], 0.1);
    T.close(p.data, [0, 4, 3], 1e-6, 'p -= lr * grad: 1 − 0.1·10 = 0, 2 − 0.1·(−20) = 4, 3 − 0 = 3');
    T.close(p.grad, [10, -20, 0], 1e-6, 'sgdStep must not clear the gradient; zeroGrad is a separate, explicit call');
    T.close(q.data, [5, 5], 1e-6, 'a parameter with grad === null is skipped, not crashed on');
    m.sgdStep([p], 0.5);
    T.close(p.data, [-5, 14, 3], 1e-6, 'a second step applies the (still present) gradient again');
  } },
  { step: 'train', name: 'trainLinear recovers w = 3, b = 2 from noiseless data', run(m, T) {
    const xs = Array.from({ length: 32 }, (_, i) => -1 + (2 * i) / 31);
    const ys = xs.map((x) => 3 * x + 2);
    const r = m.trainLinear(xs, ys, { steps: 200, lr: 0.1 });
    T.eq(r.losses.length, 200, 'one loss per step');
    T.eq(r.ws.length, 200); T.eq(r.bs.length, 200);
    T.ok(Math.abs(r.w - 3) < 0.02, `w should approach 3 (got ${r.w}); check the sign of the update and that zeroGrad runs every step`);
    T.ok(Math.abs(r.b - 2) < 0.02, `b should approach 2 (got ${r.b})`);
    T.ok(r.losses[r.losses.length - 1] < 1e-3, `final MSE should be near 0 on noiseless data, got ${r.losses[r.losses.length - 1]}`);
    T.ok(r.losses[0] > r.losses[10] && r.losses[10] > r.losses[199], 'loss must fall over training');
    T.close(r.ws[199], r.w, 1e-9, 'ws records w after each step, so the last entry is the returned w');
  } },
  { step: 'train', name: 'trainLinear fits noisy data and the loss falls by more than 10×', run(m, T) {
    const next = T.rng(3);
    const xs = Array.from({ length: 64 }, () => next() * 4 - 2);
    const ys = xs.map((x) => 3 * x + 2 + (next() - 0.5) * 0.5);
    const r = m.trainLinear(xs, ys, { steps: 100, lr: 0.05 });
    T.ok(Math.abs(r.w - 3) < 0.15 && Math.abs(r.b - 2) < 0.15, `expected w ≈ 3, b ≈ 2, got w=${r.w.toFixed(3)} b=${r.b.toFixed(3)}`);
    T.ok(r.losses[99] < r.losses[0] / 10, `loss went ${r.losses[0].toFixed(3)} → ${r.losses[99].toFixed(4)}; gradient descent should cut it by more than 10×`);
    T.ok(r.losses.every(Number.isFinite), 'no NaN or Infinity: if gradients accumulate across steps the effective learning rate grows and training diverges');
  } },
];
