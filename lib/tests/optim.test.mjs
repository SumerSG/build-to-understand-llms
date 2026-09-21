// Tests for lib/optim.js: hand-computed single steps for both optimizers, the clipping and schedule
// helpers, and two end-to-end fits (linear regression with SGD, XOR with AdamW) that must converge.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { rng } from '../util.js';
import { Tensor, crossEntropy } from '../tensor.js';
import { SGD, AdamW, clipGradNorm, cosineWithWarmup } from '../optim.js';

/** A leaf tensor with a gradient set by hand, so a single optimizer step can be checked on paper. */
function paramWithGrad(values, grads) {
  const p = Tensor.from(values, { requiresGrad: true });
  p.grad = Float32Array.from(grads);
  return p;
}

/** Assert two numbers agree to `tol`, reporting both when they do not. */
function close(actual, expected, tol, what) {
  assert.ok(Math.abs(actual - expected) <= tol, `${what}: got ${actual}, expected ${expected}`);
}

// ---------- SGD ----------

test('SGD takes the plain gradient step', () => {
  const p = paramWithGrad([1, -2], [0.5, 0.25]);
  new SGD([p], { lr: 0.1 }).step();
  close(p.data[0], 1 - 0.05, 1e-6, 'p0');
  close(p.data[1], -2 - 0.025, 1e-6, 'p1');
});

test('SGD with momentum builds up velocity', () => {
  const p = paramWithGrad([0], [1]);
  const opt = new SGD([p], { lr: 0.1, momentum: 0.9 });
  opt.step(); // v = 1        -> p = -0.1
  close(p.data[0], -0.1, 1e-6, 'after step 1');
  opt.step(); // v = 0.9 + 1  -> p = -0.29
  close(p.data[0], -0.29, 1e-6, 'after step 2');
});

test('SGD skips parameters without a gradient and zeroGrad clears the rest', () => {
  const withGrad = paramWithGrad([1], [1]);
  const withoutGrad = Tensor.from([5], { requiresGrad: true });
  const opt = new SGD([withGrad, withoutGrad], { lr: 0.5 });
  opt.step();
  close(withGrad.data[0], 0.5, 1e-6, 'updated');
  close(withoutGrad.data[0], 5, 1e-6, 'untouched');
  opt.zeroGrad();
  assert.equal(withGrad.grad, null);
});

// ---------- AdamW ----------

test('AdamW first step matches the hand-computed value', () => {
  // t = 1: mHat = m/(1-b1) = g and vHat = v/(1-b2) = g^2, so the step is lr*g/(|g|+eps) ~ lr*sign(g).
  const p = paramWithGrad([1, -2], [0.5, -0.25]);
  const opt = new AdamW([p], { lr: 0.1, betas: [0.9, 0.95], eps: 1e-8, weightDecay: 0 });
  opt.step();
  assert.equal(opt.t, 1);
  close(p.data[0], 1 - 0.1 * (0.5 / (0.5 + 1e-8)), 1e-6, 'p0');
  close(p.data[1], -2 + 0.1 * (0.25 / (0.25 + 1e-8)), 1e-6, 'p1');
  // The stored moments are the raw (uncorrected) running averages.
  close(opt.m[0][0], 0.05, 1e-7, 'm0');
  close(opt.v[0][0], 0.05 * 0.25, 1e-9, 'v0');
});

test('AdamW weight decay is decoupled: it scales p before the Adam step', () => {
  const p = paramWithGrad([1], [0.5]);
  new AdamW([p], { lr: 0.1, weightDecay: 0.1, eps: 1e-8 }).step();
  const afterDecay = 1 - 0.1 * 0.1 * 1; // 0.99
  close(p.data[0], afterDecay - 0.1 * (0.5 / (0.5 + 1e-8)), 1e-6, 'p0');
});

test('AdamW bias correction makes the second step use the corrected moments', () => {
  const p = paramWithGrad([0], [1]);
  const opt = new AdamW([p], { lr: 0.1, betas: [0.9, 0.95], eps: 1e-8 });
  opt.step();
  p.grad[0] = 1;
  opt.step();
  // With a constant gradient the bias-corrected estimates stay at mHat = 1, vHat = 1 every step.
  close(p.data[0], -0.2, 1e-5, 'after two steps');
  assert.equal(opt.t, 2);
});

// ---------- gradient clipping ----------

test('clipGradNorm returns the total norm and scales only when it is exceeded', () => {
  const a = paramWithGrad([0, 0], [3, 4]);
  const b = paramWithGrad([0], [12]);
  const norm = clipGradNorm([a, b], 6.5); // sqrt(9 + 16 + 144) = 13
  close(norm, 13, 1e-5, 'norm');
  close(a.grad[0], 1.5, 1e-5, 'a0');
  close(a.grad[1], 2, 1e-5, 'a1');
  close(b.grad[0], 6, 1e-5, 'b0');

  const c = paramWithGrad([0, 0], [3, 4]);
  close(clipGradNorm([c], 100), 5, 1e-5, 'norm under the cap');
  assert.deepEqual(Array.from(c.grad), [3, 4]); // unchanged
});

test('clipGradNorm ignores parameters without a gradient', () => {
  const a = paramWithGrad([0], [3]);
  const b = Tensor.from([0], { requiresGrad: true });
  close(clipGradNorm([a, b], 10), 3, 1e-6, 'norm');
});

// ---------- learning-rate schedule ----------

test('cosineWithWarmup at step 0, warmup, midpoint and total', () => {
  const opts = { warmup: 10, total: 100, peak: 1e-3 }; // min defaults to peak/10
  close(cosineWithWarmup(0, opts), 0, 1e-12, 'step 0');
  close(cosineWithWarmup(5, opts), 5e-4, 1e-12, 'halfway through warmup');
  close(cosineWithWarmup(10, opts), 1e-3, 1e-12, 'end of warmup');
  close(cosineWithWarmup(55, opts), 5.5e-4, 1e-9, 'cosine midpoint');
  close(cosineWithWarmup(100, opts), 1e-4, 1e-12, 'total');
  close(cosineWithWarmup(500, opts), 1e-4, 1e-12, 'past total');
});

test('cosineWithWarmup honours an explicit min and decreases monotonically after warmup', () => {
  const opts = { warmup: 4, total: 20, peak: 0.1, min: 0.02 };
  close(cosineWithWarmup(4, opts), 0.1, 1e-12, 'peak');
  close(cosineWithWarmup(20, opts), 0.02, 1e-12, 'min');
  for (let s = 5; s <= 20; s++) {
    assert.ok(cosineWithWarmup(s, opts) <= cosineWithWarmup(s - 1, opts) + 1e-12, `not decreasing at ${s}`);
  }
});

// ---------- end-to-end fits ----------

test('SGD fits y = 3x + 2 to w~3, b~2 within 300 steps', () => {
  const next = rng(7);
  const xs = [], ys = [];
  for (let i = 0; i < 64; i++) {
    const x = next() * 2 - 1;
    xs.push([x]);
    ys.push([3 * x + 2]);
  }
  const X = Tensor.from(xs), Y = Tensor.from(ys);
  const w = Tensor.zeros([1, 1], { requiresGrad: true });
  const b = Tensor.zeros([1], { requiresGrad: true });
  const opt = new SGD([w, b], { lr: 0.1 });
  let loss = Infinity;
  for (let step = 0; step < 300; step++) {
    opt.zeroGrad();
    const out = X.matmul(w).add(b).sub(Y).pow(2).mean();
    out.backward();
    opt.step();
    loss = out.item();
  }
  close(w.item(), 3, 0.1, 'w');
  close(b.item(), 2, 0.1, 'b');
  assert.ok(loss < 1e-3, `final loss ${loss}`);
});

test('SGD with momentum fits the same line', () => {
  const next = rng(11);
  const xs = [], ys = [];
  for (let i = 0; i < 64; i++) {
    const x = next() * 2 - 1;
    xs.push([x]);
    ys.push([3 * x + 2]);
  }
  const X = Tensor.from(xs), Y = Tensor.from(ys);
  const w = Tensor.zeros([1, 1], { requiresGrad: true });
  const b = Tensor.zeros([1], { requiresGrad: true });
  const opt = new SGD([w, b], { lr: 0.02, momentum: 0.9 });
  for (let step = 0; step < 300; step++) {
    opt.zeroGrad();
    const out = X.matmul(w).add(b).sub(Y).pow(2).mean();
    out.backward();
    opt.step();
  }
  close(w.item(), 3, 0.1, 'w');
  close(b.item(), 2, 0.1, 'b');
});

test('AdamW trains a 2-layer MLP on XOR to loss < 0.1', () => {
  const next = rng(3);
  const X = Tensor.from([[0, 0], [0, 1], [1, 0], [1, 1]]);
  const targets = [0, 1, 1, 0];
  const w1 = Tensor.randn([2, 8], next, 1, { requiresGrad: true });
  const b1 = Tensor.zeros([8], { requiresGrad: true });
  const w2 = Tensor.randn([8, 2], next, 1, { requiresGrad: true });
  const b2 = Tensor.zeros([2], { requiresGrad: true });
  const params = [w1, b1, w2, b2];
  const opt = new AdamW(params, { lr: 0.1, weightDecay: 0 });
  let loss = Infinity;
  for (let step = 0; step < 400; step++) {
    opt.zeroGrad();
    const hidden = X.matmul(w1).add(b1).tanh();
    const logits = hidden.matmul(w2).add(b2);
    const out = crossEntropy(logits, targets);
    out.backward();
    clipGradNorm(params, 1);
    opt.step();
    loss = out.item();
  }
  assert.ok(loss < 0.1, `final XOR loss ${loss}`);

  // And the predictions are right: argmax of each row equals the target.
  const logits = X.matmul(w1).add(b1).tanh().matmul(w2).add(b2).toArray();
  logits.forEach((row, i) => assert.equal(row[0] > row[1] ? 0 : 1, targets[i], `row ${i}`));
});
