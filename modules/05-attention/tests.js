import * as ops from 'lib/ops.js';
import { Tensor, gradCheck } from 'lib/tensor.js';
import { MultiHeadAttention as RefMHA } from 'lib/attention.js';

/** Gaussian Tensor with a seeded rng; requiresGrad so backward passes can be checked. */
function gauss(T, shape, seed, std = 1, requiresGrad = true) {
  return Tensor.randn(shape, T.rng(seed), std, { requiresGrad });
}

const isTensor = (x) => x instanceof Tensor;
const maxAbs = (arr) => arr.reduce((m, v) => Math.max(m, Math.abs(v)), 0);

/** Explicit double-precision attention on nested arrays [T][dh], the reference the batched tests compare against. */
function slowAttention(q, k, v, causal, scale) {
  const T = q.length, dh = q[0].length, dv = v[0].length;
  const weights = [], out = [];
  for (let i = 0; i < T; i++) {
    const s = [];
    for (let j = 0; j < T; j++) {
      let dot = 0;
      for (let c = 0; c < dh; c++) dot += q[i][c] * k[j][c];
      s.push(causal && j > i ? -Infinity : dot * scale);
    }
    const m = Math.max(...s);
    const e = s.map((x) => Math.exp(x - m));
    const z = e.reduce((a, b) => a + b, 0);
    const w = e.map((x) => x / z);
    weights.push(w);
    const row = new Array(dv).fill(0);
    for (let j = 0; j < T; j++) for (let c = 0; c < dv; c++) row[c] += w[j] * v[j][c];
    out.push(row);
  }
  return { weights, out };
}

/** Overwrite a layer's parameters with larger Gaussians so a gradient check is not trivially small. */
function inflate(T, layer, std = 0.3) {
  const next = T.rng(77);
  for (const p of layer.parameters()) {
    const r = ops.randn(p.shape, next, std);
    p.data.set(r.data);
  }
}

export const tests = [
  // ---------- step 1: scores ----------
  { step: 'scores', name: 'attentionScores is q·kᵀ divided by sqrt(dh) (not the raw dot product, not divided by dh)', run(m, T) {
    const q = Tensor.from([[1, 0], [0, 1]]);
    const k = Tensor.from([[1, 2], [3, 4]]);
    const s = m.attentionScores(q, k);
    T.ok(isTensor(s), 'the scores must be a Tensor from lib/tensor.js so gradients can flow through them later');
    T.shape(s, [2, 2], 'T queries against T keys give a [T, T] score matrix');
    const r = Math.SQRT1_2;
    T.close(s, [[1 * r, 3 * r], [2 * r, 4 * r]], 1e-5, 'score[i][j] = (q_i · k_j) / sqrt(dh) with dh = 2; the raw dot products are [[1,3],[2,4]] and dividing by dh instead of sqrt(dh) halves them');
  } },
  { step: 'scores', name: 'works on batched [B, H, T, dh] tensors and honours an explicit scale', run(m, T) {
    const q = gauss(T, [2, 2, 3, 4], 1), k = gauss(T, [2, 2, 3, 4], 2);
    const raw = m.attentionScores(q, k, { scale: 1 });
    const scaled = m.attentionScores(q, k);
    T.shape(raw, [2, 2, 3, 3], 'batch and head dims are kept; only the last two become [T, T]');
    const Q = q.toArray(), K = k.toArray();
    const expectRaw = Array.from({ length: 2 }, () => Array.from({ length: 2 }, () => Array.from({ length: 3 }, () => new Array(3).fill(0))));
    for (let b = 0; b < 2; b++) for (let h = 0; h < 2; h++) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      let dot = 0;
      for (let c = 0; c < 4; c++) dot += Q[b][h][i][c] * K[b][h][j][c];
      expectRaw[b][h][i][j] = dot;
    }
    T.close(raw, expectRaw, 1e-4, 'with scale: 1 the scores are the plain dot products q_i · k_j, computed per batch element and per head');
    T.close(scaled, expectRaw.map((b) => b.map((h) => h.map((r) => r.map((x) => x / 2)))), 1e-4, 'the default scale is 1/sqrt(dh) = 1/2 for dh = 4');
  } },
  { step: 'scores', name: 'the 1/sqrt(dh) scale keeps the score variance near 1 for dh = 64', run(m, T) {
    const q = gauss(T, [1, 1, 32, 64], 3), k = gauss(T, [1, 1, 32, 64], 4);
    const s = m.attentionScores(q, k);
    T.shape(s, [1, 1, 32, 32], '32 queries against 32 keys');
    const n = s.data.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += s.data[i];
    mean /= n;
    let variance = 0;
    for (let i = 0; i < n; i++) variance += (s.data[i] - mean) ** 2;
    variance /= n;
    T.ok(variance > 0.6 && variance < 1.6, `score variance is ${variance.toFixed(3)}; with unit-variance q and k and dh = 64 it should be about 1. Unscaled scores have variance about 64 (softmax saturates); dividing by dh instead of sqrt(dh) gives about 1/64 (softmax goes flat)`);
  } },
  { step: 'scores', name: 'gradients flow back into q and k through the scores', run(m, T) {
    const q = gauss(T, [3, 4], 5), k = gauss(T, [3, 4], 6);
    m.attentionScores(q, k).sum().backward();
    T.ok(q.grad !== null && k.grad !== null, 'attentionScores must be built from Tensor ops (matmul, transpose, scale) so backward() reaches q and k');
    // d/dq_i of sum_ij (q_i · k_j) / 2 is (sum_j k_j) / 2 for every row i.
    const K = k.toArray();
    const colSum = [0, 1, 2, 3].map((c) => K.reduce((s, row) => s + row[c], 0) / 2);
    T.close(Array.from(q.grad), [...colSum, ...colSum, ...colSum], 1e-4, 'd(sum of scores)/dq_i must be (Σ_j k_j) / sqrt(dh) for every query row (q.grad is flat, 3 rows of 4)');
  } },

  // ---------- step 2: mask ----------
  { step: 'mask', name: 'maskCausal sets exactly the entries with key j > query i to -Infinity', run(m, T) {
    const s = Tensor.from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    const masked = m.maskCausal(s);
    T.ok(isTensor(masked), 'return a Tensor (use maskedFill, which records the graph) rather than a raw object');
    T.eq(T.arr(masked), [[1, -Infinity, -Infinity], [4, 5, -Infinity], [7, 8, 9]], 'the diagonal (a token attending to itself) stays; everything strictly above it becomes -Infinity, not a large negative number and not 0');
    T.eq(T.arr(s), [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'the input scores must not be modified in place');
    const batched = m.maskCausal(Tensor.from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]));
    T.eq(T.arr(batched), [[[1, -Infinity], [3, 4]], [[5, -Infinity], [7, 8]]], 'the same [T, T] mask applies to every batch element and head');
  } },
  { step: 'mask', name: 'attentionWeights: every row sums to 1, future keys get exactly 0, row 0 is one-hot on itself', run(m, T) {
    const s = gauss(T, [1, 2, 4, 4], 7, 2, false);
    const w = m.attentionWeights(s);
    T.shape(w, [1, 2, 4, 4]);
    const W = w.toArray();
    for (let h = 0; h < 2; h++) {
      for (let i = 0; i < 4; i++) {
        const row = W[0][h][i];
        T.close(row.reduce((a, b) => a + b, 0), 1, 1e-5, `row ${i} of head ${h} must sum to 1: masking has to happen before the softmax so the remaining keys are renormalised`);
        for (let j = i + 1; j < 4; j++) T.eq(row[j], 0, `weight[${i}][${j}] must be exactly 0: exp(-Infinity) = 0, so a future key contributes nothing`);
      }
      T.eq(W[0][h][0], [1, 0, 0, 0], 'position 0 may only see itself, so its whole row is [1, 0, 0, 0] whatever the scores are');
    }
    const last = ops.softmax(ops.fromArray([W[0][0][3].map((_, j) => s.data[3 * 4 + j])]));
    T.close(W[0][0][3], Array.from(last.data), 1e-5, 'the last row sees every key, so it is the plain softmax of its scores');
  } },
  { step: 'mask', name: 'causal: false is a plain row softmax, and the weights stay differentiable', run(m, T) {
    const s = gauss(T, [2, 3, 3], 8);
    const w = m.attentionWeights(s, { causal: false });
    T.close(w, ops.softmax(s), 1e-6, 'without the mask every key is visible: softmax over the last dim of the raw scores');
    w.mul(w).sum().backward();
    T.ok(s.grad !== null && maxAbs(s.grad) > 0, 'the weights must be produced by Tensor.softmax so backward() reaches the scores');
    const s2 = gauss(T, [3, 3], 9);
    m.attentionWeights(s2).mul(m.attentionWeights(s2)).sum().backward();
    T.eq(s2.grad[1], 0, 'a masked score (row 0, key 1) receives zero gradient: it cannot influence the output');
    T.eq(s2.grad[2], 0, 'a masked score (row 0, key 2) receives zero gradient');
    T.ok(Math.abs(s2.grad[3]) > 0, 'a visible score (row 1, key 0) does receive gradient');
  } },

  // ---------- step 3: attend ----------
  { step: 'attend', name: 'attention returns { out, weights } with out = weights · v (hand-computed 2-token case)', run(m, T) {
    const q = Tensor.from([[1], [1]]), k = Tensor.from([[0], [0]]), v = Tensor.from([[2], [4]]);
    const r = m.attention(q, k, v);
    T.ok(r && isTensor(r.out) && isTensor(r.weights), 'return an object with Tensor fields `out` and `weights`');
    T.close(r.weights, [[1, 0], [0.5, 0.5]], 1e-6, 'all scores are 0, so the causal rows are [1, 0] and [0.5, 0.5]');
    T.close(r.out, [[2], [3]], 1e-6, 'out[0] = 1·v0 = 2 and out[1] = 0.5·v0 + 0.5·v1 = 3');
    const full = m.attention(q, k, v, { causal: false });
    T.close(full.out, [[3], [3]], 1e-6, 'without the mask both positions average both values: 3');
  } },
  { step: 'attend', name: 'batched [B, H, T, dh] attention matches an explicit double-precision loop', run(m, T) {
    const q = gauss(T, [2, 2, 4, 3], 10), k = gauss(T, [2, 2, 4, 3], 11), v = gauss(T, [2, 2, 4, 3], 12);
    const Q = q.toArray(), K = k.toArray(), V = v.toArray();
    for (const causal of [true, false]) {
      const r = m.attention(q, k, v, { causal });
      T.shape(r.out, [2, 2, 4, 3], 'out has the shape of q: one dh-vector per query');
      T.shape(r.weights, [2, 2, 4, 4], 'weights are [B, H, T, T]');
      for (let b = 0; b < 2; b++) for (let h = 0; h < 2; h++) {
        const e = slowAttention(Q[b][h], K[b][h], V[b][h], causal, 1 / Math.sqrt(3));
        T.close(r.weights.toArray()[b][h], e.weights, 1e-4, `weights for batch ${b}, head ${h} (causal: ${causal}) must equal softmax(q·kᵀ/sqrt(3)) with the mask applied before the softmax`);
        T.close(r.out.toArray()[b][h], e.out, 1e-4, `out for batch ${b}, head ${h} (causal: ${causal}) must be the weight-averaged values`);
      }
    }
  } },
  { step: 'attend', name: 'position 0 returns v[0] exactly under the mask; identical keys give the mean of v without it', run(m, T) {
    const q = gauss(T, [1, 2, 5, 4], 13), k = gauss(T, [1, 2, 5, 4], 14), v = gauss(T, [1, 2, 5, 4], 15);
    const r = m.attention(q, k, v);
    const O = r.out.toArray(), V = v.toArray();
    for (let h = 0; h < 2; h++) T.close(O[0][h][0], V[0][h][0], 1e-6, 'the first token can only attend to itself, so its output is its own value vector');
    const zeroK = Tensor.zeros([1, 2, 5, 4]);
    const flat = m.attention(q, zeroK, v, { causal: false });
    for (let h = 0; h < 2; h++) {
      const mean = [0, 1, 2, 3].map((c) => V[0][h].reduce((s, row) => s + row[c], 0) / 5);
      for (let i = 0; i < 5; i++) T.close(flat.out.toArray()[0][h][i], mean, 1e-5, 'all-equal scores give uniform weights, so every output is the mean of the values');
    }
  } },
  { step: 'attend', name: 'a numerical gradient check passes through attention', run(m, T) {
    const q = gauss(T, [1, 2, 4, 3], 16), k = gauss(T, [1, 2, 4, 3], 17), v = gauss(T, [1, 2, 4, 3], 18);
    const res = gradCheck((a, b, c) => m.attention(a, b, c).out.pow(2).sum(), [q, k, v]);
    T.ok(res.ok, `analytic and numerical gradients disagree (max relative error ${res.maxRelErr}): build attention only from Tensor ops so every step is recorded`);
    T.ok(k.grad !== null && maxAbs(k.grad) > 0 && v.grad !== null && maxAbs(v.grad) > 0, 'the output must depend on k (through the weights) and on v (through the weighted sum), so both receive gradient');
  } },

  // ---------- step 4: heads ----------
  { step: 'heads', name: 'splitHeads sends channels h·dh..(h+1)·dh of every token to head h; mergeHeads inverts it', run(m, T) {
    const x = Tensor.from([[[0, 1, 2, 3], [4, 5, 6, 7]]]); // [B=1, T=2, C=4]
    const s = m.splitHeads(x, 2);
    T.shape(s, [1, 2, 2, 2], 'expected [B, H, T, dh] = [1, 2, 2, 2]');
    T.eq(T.arr(s), [[[[0, 1], [4, 5]], [[2, 3], [6, 7]]]], 'head 0 must hold channels 0–1 of both tokens and head 1 channels 2–3: reshape to [B, T, H, dh] then permute so H comes before T (a bare reshape to [B, H, T, dh] mixes tokens into heads)');
    T.eq(T.arr(m.mergeHeads(s)), T.arr(x), 'mergeHeads(splitHeads(x)) must give x back');
    const heads = Tensor.from([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]]); // [1, H=2, T=2, dh=2]
    T.eq(T.arr(m.mergeHeads(heads)), [[[0, 1, 4, 5], [2, 3, 6, 7]]], 'token t of the merged tensor is head 0\'s row t followed by head 1\'s row t');
    T.shape(m.splitHeads(gauss(T, [3, 5, 12], 19), 4), [3, 4, 5, 3]);
  } },
  { step: 'heads', name: 'MultiHeadAttention: [B, T, C] in and out, 4C² + 4C parameters, causal lastWeights [B, H, T, T]', run(m, T) {
    const C = 8, H = 2;
    const layer = new m.MultiHeadAttention({ nEmbd: C, nHead: H, next: T.rng(20) });
    const params = layer.parameters();
    T.eq(params.length, 4, 'parameters() must list the qkv weight and bias, then the proj weight and bias');
    T.eq(params.reduce((s, p) => s + p.data.length, 0), 4 * C * C + 4 * C, 'qkv is C×3C (+3C bias) and proj is C×C (+C bias): 4C² + 4C numbers in total');
    const x = gauss(T, [2, 5, C], 21);
    const y = layer.forward(x);
    T.shape(y, [2, 5, C], 'the output has the same [B, T, C] shape as the input: attention is a residual-friendly map');
    T.ok(isTensor(y) && y.requiresGrad, 'the output must be a Tensor with a recorded graph so the layer can be trained');
    T.ok(layer.lastWeights && layer.lastWeights.shape, 'forward must store the softmaxed weights in lastWeights (a raw copy is fine)');
    T.shape(layer.lastWeights, [2, H, 5, 5], 'lastWeights is [B, H, T, T]');
    const W = T.arr(layer.lastWeights);
    for (let b = 0; b < 2; b++) for (let h = 0; h < H; h++) for (let i = 0; i < 5; i++) {
      T.close(W[b][h][i].reduce((a, c) => a + c, 0), 1, 1e-5, 'each stored weight row sums to 1');
      for (let j = i + 1; j < 5; j++) T.eq(W[b][h][i][j], 0, 'the layer must be causal: no weight on future keys');
    }
    T.throws(() => layer.forward(gauss(T, [2, 5, C + 1], 22)), 'forward must throw when the channel count does not match nEmbd');
  } },
  { step: 'heads', name: 'matches lib/attention.js exactly when given the same weights', run(m, T) {
    const C = 12, H = 3;
    const ref = new RefMHA({ nEmbd: C, nHead: H, next: T.rng(23) });
    inflate(T, ref, 0.2);
    const layer = new m.MultiHeadAttention({ nEmbd: C, nHead: H, next: T.rng(24) });
    layer.qkv.weight.data.set(ref.qkv.weight.data);
    layer.qkv.bias.data.set(ref.qkv.bias.data);
    layer.proj.weight.data.set(ref.proj.weight.data);
    layer.proj.bias.data.set(ref.proj.bias.data);
    const x = gauss(T, [2, 6, C], 25);
    const y = layer.forward(x), yRef = ref.forward(x);
    T.close(y, yRef, 1e-5, 'with identical qkv and proj weights your layer must reproduce the reference output: check the q/k/v slice order (q first, then k, then v) and that heads are merged back in the same order they were split');
    T.close(layer.lastWeights, ref.lastWeights, 1e-5, 'the per-head weights must match too');
    y.pow(2).sum().backward();
    yRef.pow(2).sum().backward();
    T.close(T.arr(layer.qkv.weight.grad), T.arr(ref.qkv.weight.grad), 1e-3, 'gradients into the qkv weight must match the reference: every op in forward has to be a Tensor op');
  } },

  // ---------- step 5: proof ----------
  { step: 'proof', name: 'causalityProbe reports zero change up to t and a real change after t for a causal layer', run(m, T) {
    const layer = new RefMHA({ nEmbd: 8, nHead: 2, next: T.rng(26) });
    inflate(T, layer, 0.3);
    const x = gauss(T, [2, 7, 8], 27, 1, false);
    const before = Array.from(x.data);
    const r = m.causalityProbe(layer, x, 3, T.rng(28));
    T.ok(r && typeof r.maxBefore === 'number' && typeof r.maxAfter === 'number', 'return { maxBefore, maxAfter } as plain numbers');
    T.eq(r.maxBefore, 0, 'positions 0..3 must not move at all when positions 4..6 are perturbed: causal attention never reads the future, so the difference is exactly 0, not merely small');
    T.ok(r.maxAfter > 1e-4, `positions after t must change (got ${r.maxAfter}): the noise has to be added to positions t+1..T-1 with a non-zero standard deviation`);
    T.eq(Array.from(x.data), before, 'the probe must perturb a copy, not the caller\'s tensor');
  } },
  { step: 'proof', name: 'causalityProbe catches a layer that leaks the future, and perturbs nothing when t is the last position', run(m, T) {
    const leaky = { forward: (x) => x.add(x.mean(1, true)), parameters: () => [] };
    const x = gauss(T, [1, 6, 4], 29, 1, false);
    const r = m.causalityProbe(leaky, x, 2, T.rng(30));
    T.ok(r.maxBefore > 1e-3, `a layer that averages over time leaks future tokens into every position, so maxBefore must be clearly positive (got ${r.maxBefore})`);
    const layer = new RefMHA({ nEmbd: 4, nHead: 1, next: T.rng(31) });
    inflate(T, layer, 0.3);
    const end = m.causalityProbe(layer, x, 5, T.rng(32));
    T.eq(end.maxBefore, 0, 'with t = T-1 there is nothing after t to perturb, so nothing may change: perturb positions strictly greater than t, not t itself');
    T.eq(end.maxAfter, 0, 'with t = T-1 maxAfter is 0 because no position lies after t');
  } },
  { step: 'proof', name: 'gradCheckAttention passes on your MultiHeadAttention and checks x plus every parameter', run(m, T) {
    const layer = new m.MultiHeadAttention({ nEmbd: 8, nHead: 2, next: T.rng(33) });
    inflate(T, layer, 0.3);
    const x = gauss(T, [1, 4, 8], 34);
    const res = m.gradCheckAttention(layer, x);
    T.ok(res && typeof res.ok === 'boolean' && Array.isArray(res.details), 'return the result object of lib/tensor.js gradCheck ({ ok, maxRelErr, details })');
    T.ok(res.ok, `the gradient check failed with max relative error ${res.maxRelErr}: every op in the layer must be a recorded Tensor op`);
    T.ok(res.maxRelErr < 1e-2, 'maxRelErr must be below the 1e-2 float32 tolerance');
    const inputsChecked = new Set(res.details.map((d) => d.input)).size;
    T.eq(inputsChecked, 1 + layer.parameters().length, 'the inputs handed to gradCheck must be x followed by every parameter of the layer (qkv weight, qkv bias, proj weight, proj bias): a check that skips the parameters would miss a broken Linear backward');
  } },
  { step: 'proof', name: 'gradCheckAttention reports failure for a layer whose backward is wrong', run(m, T) {
    // Forward computes x², but the graph only records x · const, so the analytic gradient is x instead of 2x.
    const wrong = { forward: (x) => x.mul(x.detach()), parameters: () => [] };
    const x = gauss(T, [1, 3, 4], 35);
    const res = m.gradCheckAttention(wrong, x);
    T.eq(res.ok, false, 'a layer with a wrong backward must be reported as failing, not silently passed');
    T.ok(res.maxRelErr > 0.1, `the analytic gradient is off by a factor of 2, so maxRelErr should be large (got ${res.maxRelErr})`);
  } },
];
