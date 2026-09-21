// Tests for lib/attention.js — scaled dot-product attention and multi-head attention.
import test from 'node:test';
import assert from 'node:assert/strict';

import * as ops from '../ops.js';
import { Tensor, gradCheck } from '../tensor.js';
import { rng } from '../util.js';
import { attention, MultiHeadAttention } from '../attention.js';

/** A deterministic random Tensor, optionally trainable. */
function randTensor(shape, seed, requiresGrad = false) {
  return new Tensor(ops.randn(shape, rng(seed), 1), { requiresGrad });
}

test('attention: weights rows sum to 1', () => {
  const q = randTensor([1, 2, 5, 4], 1);
  const k = randTensor([1, 2, 5, 4], 2);
  const v = randTensor([1, 2, 5, 4], 3);
  const { weights } = attention(q, k, v);
  assert.deepEqual(weights.shape, [1, 2, 5, 5]);
  for (let row = 0; row < weights.data.length; row += 5) {
    let total = 0;
    for (let j = 0; j < 5; j++) total += weights.data[row + j];
    assert.ok(Math.abs(total - 1) < 1e-5, `row at ${row} sums to ${total}`);
  }
});

test('attention: causal weights are lower-triangular', () => {
  const q = randTensor([1, 2, 6, 4], 4);
  const k = randTensor([1, 2, 6, 4], 5);
  const v = randTensor([1, 2, 6, 4], 6);
  const { weights } = attention(q, k, v, { causal: true });
  const T = 6;
  const rows = weights.data.length / T;
  for (let r = 0; r < rows; r++) {
    const i = r % T; // query position inside its [T,T] block
    for (let j = 0; j < T; j++) {
      const w = weights.data[r * T + j];
      if (j > i) assert.equal(w, 0, `weight (${i},${j}) should be masked, got ${w}`);
      else assert.ok(w > 0, `weight (${i},${j}) should be positive, got ${w}`);
    }
  }
});

test('attention: non-causal attends to every position', () => {
  const q = randTensor([4, 3], 7);
  const k = randTensor([4, 3], 8);
  const v = randTensor([4, 3], 9);
  const { out, weights } = attention(q, k, v, { causal: false });
  assert.deepEqual(out.shape, [4, 3]);
  assert.deepEqual(weights.shape, [4, 4]);
  for (const w of weights.data) assert.ok(w > 0);
});

test('attention: matches a hand-computed 2-position example', () => {
  // dh = 1 so the scale is 1/sqrt(1) = 1 and every dot product is just a product of two numbers.
  const q = Tensor.from([[1], [2]]);
  const k = Tensor.from([[1], [3]]);
  const v = Tensor.from([[10], [20]]);
  const { out, weights } = attention(q, k, v, { causal: true });
  // Row 0 sees only key 0, so its weight is 1 and its output is v0.
  assert.equal(weights.data[0], 1);
  assert.equal(weights.data[1], 0);
  assert.ok(Math.abs(out.data[0] - 10) < 1e-6);
  // Row 1: scores are q1·k0 = 2 and q1·k1 = 6, so softmax([2, 6]) = [1/(1+e⁴), e⁴/(1+e⁴)].
  const p1 = Math.exp(2) / (Math.exp(2) + Math.exp(6));
  assert.ok(Math.abs(weights.data[2] - p1) < 1e-6);
  assert.ok(Math.abs(out.data[1] - (p1 * 10 + (1 - p1) * 20)) < 1e-5);
});

test('attention: the default scale is 1/sqrt(dh)', () => {
  const q = randTensor([1, 1, 4, 16], 10);
  const k = randTensor([1, 1, 4, 16], 11);
  const v = randTensor([1, 1, 4, 16], 12);
  const auto = attention(q, k, v, { causal: false });
  const explicit = attention(q, k, v, { causal: false, scale: 1 / Math.sqrt(16) });
  assert.ok(ops.allClose(auto.weights, explicit.weights, 1e-6));
  const different = attention(q, k, v, { causal: false, scale: 1 });
  assert.ok(!ops.allClose(auto.weights, different.weights, 1e-3));
});

test('attention: causal masking rejects mismatched query and key counts', () => {
  const q = randTensor([3, 2], 13);
  const k = randTensor([5, 2], 14);
  const v = randTensor([5, 2], 15);
  assert.throws(() => attention(q, k, v, { causal: true }), /as many queries as keys/);
  // Without the mask, cross-attention shapes are fine.
  const { out } = attention(q, k, v, { causal: false });
  assert.deepEqual(out.shape, [3, 2]);
});

test('MultiHeadAttention: shapes, parameters and lastWeights', () => {
  const mha = new MultiHeadAttention({ nEmbd: 8, nHead: 2, next: rng(16) });
  const x = randTensor([2, 5, 8], 17);
  const out = mha.forward(x);
  assert.deepEqual(out.shape, [2, 5, 8]);

  const params = mha.parameters();
  assert.equal(params.length, 4);
  assert.deepEqual(params[0].shape, [8, 24]); // qkv weight
  assert.deepEqual(params[1].shape, [24]); // qkv bias
  assert.deepEqual(params[2].shape, [8, 8]); // proj weight
  assert.deepEqual(params[3].shape, [8]); // proj bias
  for (const p of params) assert.equal(p.requiresGrad, true);

  assert.deepEqual(mha.lastWeights.shape, [2, 2, 5, 5]);
  assert.ok(mha.lastWeights.data instanceof Float32Array);
  // lastWeights is a detached copy: writing to it cannot corrupt the graph it came from.
  const before = mha.lastWeights.data[0];
  mha.lastWeights.data[0] = 42;
  const again = mha.forward(x);
  assert.ok(ops.allClose(again, out, 1e-6));
  assert.ok(Math.abs(mha.lastWeights.data[0] - before) < 1e-6);
});

test('MultiHeadAttention: rejects a head count that does not divide nEmbd', () => {
  assert.throws(() => new MultiHeadAttention({ nEmbd: 8, nHead: 3, next: rng(0) }), /divisible/);
});

test('MultiHeadAttention: output at position t ignores later tokens', () => {
  const mha = new MultiHeadAttention({ nEmbd: 8, nHead: 2, next: rng(18) });
  const T = 6;
  const C = 8;
  const x = randTensor([1, T, C], 19);
  const baseline = mha.forward(x);

  // Replace everything after position 2 with different values.
  const changed = new Tensor({ shape: [1, T, C], data: new Float32Array(x.data) });
  const noise = ops.randn([1, T, C], rng(20), 1);
  for (let t = 3; t < T; t++) {
    for (let c = 0; c < C; c++) changed.data[t * C + c] = noise.data[t * C + c];
  }
  const after = mha.forward(changed);

  for (let t = 0; t < T; t++) {
    let maxDiff = 0;
    for (let c = 0; c < C; c++) {
      maxDiff = Math.max(maxDiff, Math.abs(baseline.data[t * C + c] - after.data[t * C + c]));
    }
    if (t <= 2) assert.ok(maxDiff < 1e-6, `position ${t} changed by ${maxDiff}`);
    else assert.ok(maxDiff > 1e-4, `position ${t} should have changed, moved only ${maxDiff}`);
  }
});

test('MultiHeadAttention: gradients pass a numeric check', () => {
  const mha = new MultiHeadAttention({ nEmbd: 4, nHead: 2, next: rng(21) });
  const x = randTensor([1, 3, 4], 22, true);
  // The loss weights the outputs unevenly so no gradient cancels by symmetry.
  const weight = randTensor([1, 3, 4], 23);
  const check = gradCheck((input) => mha.forward(input).mul(weight).sum(), [x, ...mha.parameters()]);
  assert.ok(check.ok, `maxRelErr ${check.maxRelErr}`);
});

test('attention: gradients pass a numeric check on q, k and v', () => {
  const q = randTensor([1, 2, 3, 2], 24, true);
  const k = randTensor([1, 2, 3, 2], 25, true);
  const v = randTensor([1, 2, 3, 2], 26, true);
  const weight = randTensor([1, 2, 3, 2], 27);
  const check = gradCheck(
    (a, b, c) => attention(a, b, c, { causal: true }).out.mul(weight).sum(),
    [q, k, v],
  );
  assert.ok(check.ok, `maxRelErr ${check.maxRelErr}`);
});
