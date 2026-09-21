// Tests for lib/tensor.js: every op's backward is checked against central differences, plus the
// bookkeeping rules (accumulation, noGrad, scalar root) and a speed floor for the training loops.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as ops from '../ops.js';
import { rng, now } from '../util.js';
import { Tensor, crossEntropy, gradCheck, noGrad, isGradEnabled, unbroadcast } from '../tensor.js';

/** A trainable tensor of the given shape, filled deterministically from the seeded rng. */
function param(shape, seed, std = 1) {
  return Tensor.randn(shape, rng(seed), std, { requiresGrad: true });
}

/** A fixed, non-trainable tensor; multiplying by it makes the upstream gradient non-uniform. */
function fixed(shape, seed) {
  return Tensor.randn(shape, rng(seed));
}

/** Assert that gradCheck passes, reporting the worst relative error when it does not. */
function checkGrad(name, fn, inputs, opts) {
  const result = gradCheck(fn, inputs, opts);
  assert.ok(result.ok, `${name}: maxRelErr=${result.maxRelErr}`);
  assert.ok(result.details.length > 0, `${name}: nothing was checked`);
}

// ---------- elementwise arithmetic, including broadcasting ----------

test('add/sub/mul/div gradients on equal shapes', () => {
  const a = param([2, 3], 1);
  const b = Tensor.from([[1.3, 2.1, 0.7], [3.2, 1.9, 2.6]], { requiresGrad: true });
  const w = fixed([2, 3], 3);
  checkGrad('add', (x, y) => x.add(y).mul(w).sum(), [a, b]);
  checkGrad('sub', (x, y) => x.sub(y).mul(w).sum(), [a, b]);
  checkGrad('mul', (x, y) => x.mul(y).mul(w).sum(), [a, b]);
  checkGrad('div', (x, y) => x.div(y).mul(w).sum(), [a, b]);
});

test('broadcasting [2,3] + [3] sums the gradient over the leading dim', () => {
  const a = param([2, 3], 4);
  const b = param([3], 5);
  const w = fixed([2, 3], 6);
  checkGrad('add broadcast', (x, y) => x.add(y).mul(w).sum(), [a, b]);
  checkGrad('mul broadcast', (x, y) => x.mul(y).mul(w).sum(), [a, b]);
  checkGrad('sub broadcast', (x, y) => x.sub(y).mul(w).sum(), [a, b]);

  // The analytic gradient of sum(a + b) w.r.t. b is the number of rows, once per column.
  const onlyB = param([3], 7);
  param([2, 3], 8).add(onlyB).sum().backward();
  assert.deepEqual(Array.from(onlyB.grad), [2, 2, 2]);
});

test('broadcasting a size-1 dim sums the gradient back into it', () => {
  const a = param([2, 1], 9);
  const b = param([2, 3], 10);
  const w = fixed([2, 3], 11);
  checkGrad('[2,1] * [2,3]', (x, y) => x.mul(y).mul(w).sum(), [a, b]);
});

test('scalar operands, scale and neg', () => {
  const a = param([2, 3], 12);
  const w = fixed([2, 3], 13);
  checkGrad('mul scalar', (x) => x.mul(2.5).mul(w).sum(), [a]);
  checkGrad('add scalar', (x) => x.add(0.75).mul(w).sum(), [a]);
  checkGrad('div scalar', (x) => x.div(4).mul(w).sum(), [a]);
  checkGrad('scale', (x) => x.scale(-1.5).mul(w).sum(), [a]);
  checkGrad('neg', (x) => x.neg().mul(w).sum(), [a]);
});

test('unbroadcast sums over leading and size-1 dims', () => {
  const g = Float32Array.from([1, 2, 3, 4, 5, 6]);
  assert.deepEqual(Array.from(unbroadcast(g, [2, 3], [3])), [5, 7, 9]);
  assert.deepEqual(Array.from(unbroadcast(g, [2, 3], [2, 1])), [6, 15]);
  assert.deepEqual(Array.from(unbroadcast(g, [2, 3], [2, 3])), [1, 2, 3, 4, 5, 6]);
});

// ---------- linear algebra and shape ops ----------

test('matmul gradients: 2D, batched with a shared 2D B, and fully batched', () => {
  checkGrad('matmul 2D', (x, y) => x.matmul(y).mul(fixed([3, 5], 20)).sum(), [param([3, 4], 21), param([4, 5], 22)]);
  checkGrad(
    'matmul batched, shared B',
    (x, y) => x.matmul(y).mul(fixed([2, 3, 5], 23)).sum(),
    [param([2, 3, 4], 24), param([4, 5], 25)],
  );
  checkGrad(
    'matmul batched, batched B',
    (x, y) => x.matmul(y).mul(fixed([2, 3, 5], 26)).sum(),
    [param([2, 3, 4], 27), param([2, 4, 5], 28)],
  );
});

test('a shared 2D B collects the gradient of every batch element', () => {
  const a = Tensor.ones([3, 2, 2], { requiresGrad: true });
  const b = Tensor.ones([2, 2], { requiresGrad: true });
  a.matmul(b).sum().backward();
  // Each of the 3 batch elements contributes sum over its 2 rows: 3 * 2 = 6 per entry of B.
  assert.deepEqual(Array.from(b.grad), [6, 6, 6, 6]);
});

test('transpose, permute, reshape and slice gradients', () => {
  checkGrad('transpose', (x) => x.transpose().mul(fixed([4, 3], 30)).sum(), [param([3, 4], 31)]);
  checkGrad('permute', (x) => x.permute([0, 2, 1, 3]).mul(fixed([2, 4, 3, 5], 32)).sum(), [param([2, 3, 4, 5], 33)]);
  checkGrad('reshape', (x) => x.reshape([6, -1]).mul(fixed([6, 2], 34)).sum(), [param([3, 4], 35)]);
  checkGrad('slice', (x) => x.slice(1, 1, 3).mul(fixed([2, 2], 36)).sum(), [param([2, 4], 37)]);
});

test('slice scatters its gradient back into the right window', () => {
  const x = param([2, 4], 38);
  x.slice(1, 1, 3).sum().backward();
  assert.deepEqual(Array.from(x.grad), [0, 1, 1, 0, 0, 1, 1, 0]);
});

test('embed gradient is scatter-added into the table', () => {
  checkGrad('embed', (t) => t.embed([[0, 2], [2, 4]]).mul(fixed([2, 2, 3], 40)).sum(), [param([5, 3], 41)]);

  // Repeated ids must accumulate: row 0 twice, row 2 once, row 1 never.
  const table = param([3, 2], 42);
  table.embed([0, 0, 2]).sum().backward();
  assert.deepEqual(Array.from(table.grad), [2, 2, 0, 0, 1, 1]);
});

// ---------- reductions ----------

test('sum and mean gradients over an axis and over everything', () => {
  const a = param([2, 3, 4], 50);
  checkGrad('sum all', (x) => x.sum(), [a]);
  checkGrad('sum axis 1', (x) => x.sum(1).mul(fixed([2, 4], 51)).sum(), [a]);
  checkGrad('sum axis 1 keepDims', (x) => x.sum(1, true).mul(fixed([2, 1, 4], 52)).sum(), [a]);
  checkGrad('mean all', (x) => x.mean(), [a]);
  checkGrad('mean axis -1', (x) => x.mean(-1).mul(fixed([2, 3], 53)).sum(), [a]);
  checkGrad('mean axis 0 keepDims', (x) => x.mean(0, true).mul(fixed([1, 3, 4], 54)).sum(), [a]);
});

// ---------- nonlinearities ----------

test('exp, tanh, relu, gelu and pow gradients', () => {
  const a = Tensor.from([[0.5, -0.7, 1.2], [-1.5, 0.3, 0.9]], { requiresGrad: true });
  const w = fixed([2, 3], 60);
  checkGrad('exp', (x) => x.exp().mul(w).sum(), [a]);
  checkGrad('tanh', (x) => x.tanh().mul(w).sum(), [a]);
  checkGrad('relu', (x) => x.relu().mul(w).sum(), [a]);
  checkGrad('gelu', (x) => x.gelu().mul(w).sum(), [a]);
  checkGrad('pow 3', (x) => x.pow(3).mul(w).sum(), [a]);
});

test('log and sqrt gradients on positive inputs', () => {
  const a = Tensor.from([[0.4, 1.7, 3.1], [2.2, 0.9, 1.1]], { requiresGrad: true });
  const w = fixed([2, 3], 61);
  checkGrad('log', (x) => x.log().mul(w).sum(), [a]);
  checkGrad('sqrt', (x) => x.sqrt().mul(w).sum(), [a]);
  checkGrad('pow 0.5', (x) => x.pow(0.5).mul(w).sum(), [a]);
});

test('gelu matches the tanh-approximation forward from ops', () => {
  const a = Tensor.from([-2, -0.5, 0, 0.5, 2]);
  assert.ok(ops.allClose(a.gelu(), ops.gelu(a)));
});

// ---------- row ops along the last dimension ----------

test('softmax and logSoftmax gradients', () => {
  const a = param([2, 3, 5], 70);
  checkGrad('softmax', (x) => x.softmax().mul(fixed([2, 3, 5], 71)).sum(), [a]);
  checkGrad('logSoftmax', (x) => x.logSoftmax().mul(fixed([2, 3, 5], 72)).sum(), [a]);

  // Softmax rows sum to 1, so the gradient of the row sum is exactly zero.
  const b = param([2, 4], 73);
  b.softmax().sum().backward();
  for (const v of b.grad) assert.ok(Math.abs(v) < 1e-6, `expected ~0, got ${v}`);
});

test('layerNorm gradients for x, gamma and beta', () => {
  const x = param([3, 6], 80);
  const gamma = param([6], 81, 0.5);
  const beta = param([6], 82, 0.5);
  const w = fixed([3, 6], 83);
  checkGrad('layerNorm', (a, g, b) => a.layerNorm(g, b).mul(w).sum(), [x, gamma, beta]);
  checkGrad('layerNorm no affine', (a) => a.layerNorm(null, null).mul(w).sum(), [x]);
  checkGrad('layerNorm [B,T,C]', (a) => a.layerNorm(null, null).mul(fixed([2, 3, 4], 84)).sum(), [param([2, 3, 4], 85)]);
});

test('layerNorm forward matches ops.layerNorm', () => {
  const x = param([3, 6], 86);
  const gamma = param([6], 87);
  const beta = param([6], 88);
  assert.ok(ops.allClose(x.layerNorm(gamma, beta), ops.layerNorm(x, gamma, beta)));
});

test('maskedFill passes gradient only where the mask is non-zero', () => {
  const mask = ops.causalMask(3);
  checkGrad('maskedFill', (x) => x.maskedFill(mask, -1).mul(fixed([3, 3], 90)).sum(), [param([3, 3], 91)]);

  const x = param([3, 3], 92);
  x.maskedFill(mask, -1).sum().backward();
  assert.deepEqual(Array.from(x.grad), [1, 0, 0, 1, 1, 0, 1, 1, 1]);
});

// ---------- cross entropy ----------

test('crossEntropy gradient on [2,3,5] logits with nested targets', () => {
  const targets = [[0, 1, 2], [3, 4, 0]];
  checkGrad('crossEntropy [B,T,V]', (x) => crossEntropy(x, targets), [param([2, 3, 5], 100)]);
  checkGrad('crossEntropy [N,V]', (x) => crossEntropy(x, [2, 0, 1, 4]), [param([4, 5], 101)]);
});

test('crossEntropy on uniform logits equals log(V), and its gradient is (softmax - onehot)/N', () => {
  const logits = Tensor.zeros([2, 4], { requiresGrad: true });
  const loss = crossEntropy(logits, [0, 3]);
  assert.ok(Math.abs(loss.item() - Math.log(4)) < 1e-5);
  loss.backward();
  const p = 0.25 / 2; // softmax probability 0.25, divided by N = 2
  const expected = [p - 0.5, p, p, p, p, p, p, p - 0.5];
  Array.from(logits.grad).forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-6, `grad[${i}]=${v}`));
});

test('crossEntropy rejects a target count that does not match the logits', () => {
  assert.throws(() => crossEntropy(Tensor.zeros([2, 4]), [0, 1, 2]), /do not match/);
  assert.throws(() => crossEntropy(Tensor.zeros([2, 4]), [0, 9]), /out of range/);
});

// ---------- graph bookkeeping ----------

test('a tensor used twice accumulates both gradients', () => {
  const x = Tensor.from([1, 2, 3], { requiresGrad: true });
  x.add(x).sum().backward();
  assert.deepEqual(Array.from(x.grad), [2, 2, 2]);

  const y = Tensor.from([1, 2, 3], { requiresGrad: true });
  y.mul(2).add(y.mul(3)).sum().backward();
  assert.deepEqual(Array.from(y.grad), [5, 5, 5]);

  // Without zeroGrad, a second backward pass adds to what is already there.
  const z = Tensor.from([1, 2], { requiresGrad: true });
  z.sum().backward();
  z.sum().backward();
  assert.deepEqual(Array.from(z.grad), [2, 2]);
  z.zeroGrad();
  assert.equal(z.grad, null);
});

test('intermediate gradients are scratch space, reset by each backward pass', () => {
  const x = Tensor.from([1, 2], { requiresGrad: true });
  const h = x.mul(3);
  h.sum().backward();
  assert.deepEqual(Array.from(h.grad), [1, 1]);
  h.sum().backward();
  assert.deepEqual(Array.from(h.grad), [1, 1]);
  assert.deepEqual(Array.from(x.grad), [6, 6]); // the leaf kept accumulating, the intermediate did not
});

test('noGrad records nothing and restores the previous setting', () => {
  const x = Tensor.ones([2, 2], { requiresGrad: true });
  assert.equal(isGradEnabled(), true);
  const y = noGrad(() => {
    assert.equal(isGradEnabled(), false);
    return x.mul(2).sum();
  });
  assert.equal(isGradEnabled(), true);
  assert.equal(y.requiresGrad, false);
  assert.equal(y._op, '');
  assert.deepEqual(y._children, []);
  assert.equal(y._backward, null);
  assert.equal(y.item(), 8);
  assert.throws(() => y.backward(), /no recorded graph/);
  assert.equal(x.grad, null);
});

test('backward needs a scalar root; detach cuts the graph', () => {
  const x = Tensor.ones([2, 2], { requiresGrad: true });
  assert.throws(() => x.mul(2).backward(), /size 1/);
  const d = x.mul(2).detach();
  assert.equal(d.requiresGrad, false);
  assert.deepEqual(d._children, []);
});

test('op names and children are recorded for inspection', () => {
  const a = Tensor.ones([2], { requiresGrad: true });
  const b = Tensor.ones([2]);
  const c = a.mul(b);
  assert.equal(c._op, 'mul');
  assert.equal(c.requiresGrad, true);
  assert.deepEqual(c._children, [a, b]);
  // Nothing requiring grad means nothing recorded.
  const plain = b.mul(b);
  assert.equal(plain.requiresGrad, false);
  assert.deepEqual(plain._children, []);
});

test('gradCheck returns usable details and rejects bad usage', () => {
  const x = param([2, 3], 110);
  const result = gradCheck((t) => t.mul(t).sum(), [x]);
  assert.equal(result.ok, true);
  assert.ok(result.maxRelErr < 1e-2);
  assert.ok(result.details.every((d) => Number.isFinite(d.numeric) && Number.isFinite(d.analytic)));
  assert.throws(() => gradCheck((t) => t.mul(t), [x]), /scalar/);
  assert.throws(() => gradCheck((t) => t.sum(), [Tensor.ones([2])]), /requiresGrad/);
});

// ---------- speed ----------

test('forward + backward of a depth-4 [16,64]x[64,64] matmul chain is fast', () => {
  const next = rng(123);
  const x = Tensor.randn([16, 64], next);
  const weights = [0, 1, 2, 3].map(() => Tensor.randn([64, 64], next, 0.1, { requiresGrad: true }));
  const run = () => {
    let h = x;
    for (const w of weights) h = h.matmul(w).tanh();
    h.sum().backward();
  };
  run(); // warm up the JIT so the measurement is of steady-state code
  for (const w of weights) w.zeroGrad();
  const start = now();
  run();
  const elapsed = now() - start;
  assert.ok(elapsed < 50, `forward+backward took ${elapsed.toFixed(2)} ms`);
  for (const w of weights) assert.ok(w.grad !== null);
});
