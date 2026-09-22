import { GPT } from 'lib/gpt.js';
import { Tensor } from 'lib/tensor.js';
import * as ops from 'lib/ops.js';
import { AdamW } from 'lib/optim.js';

const LN2 = Math.log(2);
const softplus = (z) => (z > 30 ? z : Math.log(1 + Math.exp(z))); // -log sigmoid(-z), reference in float64
const btRef = (margins) => margins.reduce((s, m) => s + softplus(-m), 0) / margins.length;

function tinyModel(seed = 3) {
  return new GPT({ vocabSize: 20, blockSize: 16, nLayer: 1, nHead: 2, nEmbd: 16, seed });
}

/** Reference padded batch builder, so steps 2, 4 and 5 do not depend on the worked-example helpers. */
function refPad(examples, padId = 0) {
  const T = Math.max(...examples.map((e) => e.x.length));
  const pad = (arr, v) => arr.concat(new Array(T - arr.length).fill(v));
  return { x: examples.map((e) => pad(e.x, padId)), y: examples.map((e) => pad(e.y, padId)), mask: examples.map((e) => pad(e.mask, 0)), lengths: examples.map((e) => e.x.length) };
}

/** Sum of masked target log-probs, computed independently of the learner's code. */
function refSeqLogProbs(model, batch) {
  const logits = model.forward(batch.x);
  const [B, T, V] = logits.shape;
  const lp = ops.logSoftmax(logits);
  const out = [];
  for (let b = 0; b < B; b++) {
    let s = 0;
    for (let t = 0; t < T; t++) if (batch.mask[b][t]) s += lp.data[(b * T + t) * V + batch.y[b][t]];
    out.push(s);
  }
  return out;
}

/** Three fake preference pairs over a 20-token vocabulary: a shared 3-token prompt, different responses. */
function fakePairs() {
  const ex = (ids, promptLen) => ({ x: ids.slice(0, -1), y: ids.slice(1), mask: ids.slice(1).map((_, i) => (i + 1 >= promptLen ? 1 : 0)) });
  return [
    { chosen: ex([1, 2, 3, 4, 5, 6, 19], 3), rejected: ex([1, 2, 3, 7, 8, 19], 3) },
    { chosen: ex([9, 10, 11, 12, 13, 19], 3), rejected: ex([9, 10, 11, 14, 15, 16, 17, 19], 3) },
    { chosen: ex([2, 4, 6, 8, 19], 3), rejected: ex([2, 4, 6, 3, 19], 3) },
  ];
}

function refLogpsFor(model, pairs) {
  return {
    chosen: refSeqLogProbs(model, refPad(pairs.map((p) => p.chosen))),
    rejected: refSeqLogProbs(model, refPad(pairs.map((p) => p.rejected))),
  };
}

function paramsSnapshot(model) {
  return model.parameters().map((p) => Float32Array.from(p.data));
}

function paramsEqual(a, b, tol = 0) {
  for (let k = 0; k < a.length; k++) for (let i = 0; i < a[k].length; i++) if (Math.abs(a[k][i] - b[k][i]) > tol) return false;
  return true;
}

export const tests = [
  // ---------- step 1 ----------
  { step: 'bradley-terry', name: 'bradleyTerryLoss is the mean of −log σ(chosen − rejected): log 2 at zero margin, hand values elsewhere', run(m, T) {
    const zero = m.bradleyTerryLoss([0], [0]);
    T.ok(zero instanceof Tensor, 'return a scalar Tensor (the reward head and DPO both need to backpropagate through it)');
    T.eq(zero.data.length, 1, 'the loss is one number: the mean over the pairs');
    T.close(zero.item(), LN2, 1e-5, 'with equal scores the model is a coin flip, so −log(1/2) = log 2');
    T.close(m.bradleyTerryLoss([5], [0]).item(), 0.0067153, 1e-5, 'a chosen score 5 above rejected: σ(5) = 0.9933, so the loss is −log 0.9933');
    T.close(m.bradleyTerryLoss([0], [5]).item(), 5.0067153, 1e-5, 'the wrong order costs the whole margin plus a little: −log σ(−5) = 5.0067 (not −5: the loss is never negative)');
    T.close(m.bradleyTerryLoss([1, -1], [0, 0]).item(), btRef([1, -1]), 1e-5, 'several pairs are averaged: mean(0.3133, 1.3133) = 0.8133');
    T.close(m.bradleyTerryLoss(Tensor.from([2, 3]), Tensor.from([1, 5])).item(), btRef([1, -2]), 1e-5, 'Tensor inputs must work too; only the difference chosen − rejected matters');
  } },
  { step: 'bradley-terry', name: 'the gradient pushes the chosen score up and the rejected score down by (1 − σ(margin)) / N, exactly −0.5/N at margin 0', run(m, T) {
    const rc = Tensor.from([0, 1, -2], { requiresGrad: true });
    const rr = Tensor.from([0, 0, 0], { requiresGrad: true });
    m.bradleyTerryLoss(rc, rr).backward();
    T.ok(rc.grad !== null && rr.grad !== null, 'backward() must reach both score tensors');
    const sig = (x) => 1 / (1 + Math.exp(-x));
    const expect = [0, 1, -2].map((d) => -(1 - sig(d)) / 3);
    T.close(Array.from(rc.grad), expect, 1e-5, 'd loss / d chosen = −(1 − σ(margin)) / N: a pair the model already gets right contributes little');
    T.close(Array.from(rr.grad), expect.map((g) => -g), 1e-5, 'd loss / d rejected is the exact opposite: the loss depends only on the difference');
    T.close(rc.grad[0], -0.5 / 3, 1e-6, 'at margin exactly 0 the gradient must be −0.5/N, not 0: DPO starts with policy = reference, so every margin is exactly 0 on step 1, and a relu/abs-based softplus has a zero subgradient there and training never begins');
  } },
  { step: 'bradley-terry', name: 'numerically stable: a margin of −200 gives loss ≈ 200, not Infinity or NaN', run(m, T) {
    const bad = m.bradleyTerryLoss([0], [200]).item();
    T.ok(Number.isFinite(bad), `got ${bad}: log(1 + exp(200)) overflows float32 (exp(88) is already Infinity). Use log σ(m) = m − logsumexp(m, 0), e.g. via logSoftmax over [m, 0]`);
    T.close(bad, 200, 1e-3, '−log σ(−200) is 200 to float precision');
    const good = m.bradleyTerryLoss([200], [0]).item();
    T.ok(Number.isFinite(good) && good >= 0 && good < 1e-6, `−log σ(200) is 0 to float precision, got ${good}`);
    T.close(m.bradleyTerryLoss([-50, 50], [50, -50]).item(), 50, 1e-3, 'mean of 100 and ≈0');
  } },
  { step: 'bradley-terry', name: 'rewardAccuracy is the fraction of pairs whose chosen score is strictly above the rejected one', run(m, T) {
    T.close(m.rewardAccuracy([1, 2, 3], [0, 2, 4]), 1 / 3, 1e-9, 'only the first pair is ranked correctly; a tie (2 vs 2) is not a win');
    T.close(m.rewardAccuracy(Tensor.from([0.5, -1]), Tensor.from([0.1, -2])), 1, 1e-9, 'Tensor inputs, both correct');
    T.close(m.rewardAccuracy(Float32Array.from([0, 0]), Float32Array.from([0, 0])), 0, 1e-9, 'all ties: accuracy 0 (this is what an untrained DPO policy reports, since policy = reference)');
    T.close(m.rewardAccuracy([], []), 0, 1e-9, 'no pairs: 0, not NaN');
  } },

  // ---------- step 2 ----------
  { step: 'seqlogprob', name: 'sequenceLogProbs is the SUM of the masked-in target log-probs per sequence (hand computed with a tiny GPT)', run(m, T) {
    const model = tinyModel(5);
    const batch = { x: [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], y: [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]], mask: [[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]], lengths: [6, 4] };
    const lp = m.sequenceLogProbs(model, batch);
    T.ok(lp instanceof Tensor, 'return a Tensor of shape [B] so DPO can backpropagate through it');
    T.shape(lp, [2], 'one log-probability per sequence');
    const expect = refSeqLogProbs(model, batch);
    T.close(Array.from(lp.data), expect, 1e-4, 'log p(response | prompt) = Σ_t mask[t] · log softmax(logits[t])[y[t]]: a sum over the response tokens, not a mean, and only where mask is 1');
    T.ok(lp.data[0] < 0 && lp.data[1] < 0, 'log-probabilities are negative');
  } },
  { step: 'seqlogprob', name: 'masked-out targets and padding do not change the result; a masked-in target does', run(m, T) {
    const model = tinyModel(6);
    const base = { x: [[3, 4, 5, 6, 7]], y: [[4, 5, 6, 7, 8]], mask: [[0, 0, 1, 1, 1]], lengths: [5] };
    const a = m.sequenceLogProbs(model, base).item();
    const changed = { x: [[3, 4, 5, 6, 7]], y: [[0, 19, 6, 7, 8]], mask: [[0, 0, 1, 1, 1]], lengths: [5] };
    T.close(m.sequenceLogProbs(model, changed).item(), a, 1e-5, 'targets at masked-out (prompt) positions must not contribute');
    const padded = { x: [[3, 4, 5, 6, 7, 0, 0]], y: [[4, 5, 6, 7, 8, 0, 0]], mask: [[0, 0, 1, 1, 1, 0, 0]], lengths: [5] };
    T.close(m.sequenceLogProbs(model, padded).item(), a, 1e-4, 'right-padding with mask 0 must not change the value (the causal mask means earlier positions never see the padding)');
    const other = { x: [[3, 4, 5, 6, 7]], y: [[4, 5, 6, 7, 9]], mask: [[0, 0, 1, 1, 1]], lengths: [5] };
    T.ok(Math.abs(m.sequenceLogProbs(model, other).item() - a) > 1e-4, 'changing a masked-in target must change the log-probability');
    const noMask = { x: [[3, 4, 5, 6, 7]], y: [[4, 5, 6, 7, 8]], mask: [[0, 0, 0, 0, 0]], lengths: [5] };
    T.close(m.sequenceLogProbs(model, noMask).item(), 0, 1e-6, 'with nothing masked in the sum is 0');
  } },
  { step: 'seqlogprob', name: 'the result is attached to the graph: backward() reaches the model parameters', run(m, T) {
    const model = tinyModel(7);
    const batch = { x: [[1, 2, 3, 4], [5, 6, 7, 8]], y: [[2, 3, 4, 5], [6, 7, 8, 9]], mask: [[0, 1, 1, 1], [0, 0, 1, 1]], lengths: [4, 4] };
    const lp = m.sequenceLogProbs(model, batch);
    lp.sum().backward();
    const g = model.wte.weight.grad;
    T.ok(g !== null, 'wte.grad must be filled: compute the log-probs with Tensor ops (logSoftmax, mul, sum), not by reading .data into plain numbers');
    let nonzero = 0;
    for (let i = 0; i < g.length; i++) if (g[i] !== 0) nonzero++;
    T.ok(nonzero > 0, 'the gradient must be non-zero somewhere');
    T.ok(model.blocks[0].attn.qkv.weight.grad !== null, 'the gradient must flow through the whole network, not just the embedding');
  } },

  // ---------- step 3 ----------
  { step: 'dpo', name: 'implicitRewards is β·(log π − log ref); dpoLoss matches hand-computed values', run(m, T) {
    const r = m.implicitRewards([-10, -3], [-11, -3], 0.1);
    T.ok(r instanceof Tensor, 'implicitRewards returns a Tensor (it sits on the graph during training)');
    T.close(Array.from(r.data), [0.1, 0], 1e-6, 'β·(−10 − (−11)) = 0.1 and β·0 = 0');
    T.close(m.dpoLoss([-10], [-12], [-11], [-11], 0.1).item(), 0.5981389, 1e-5, 'margin = 0.1·((−10+11) − (−12+11)) = 0.2, loss = log(1 + e^−0.2) = 0.5981');
    T.close(m.dpoLoss([-10], [-12], [-11], [-11], 1.0).item(), 0.1269280, 1e-5, 'with β = 1 the margin is 2 and the loss 0.1269: β multiplies the log-ratios, it does not divide them');
    T.close(m.dpoLoss([-10], [-12], [-5], [-20], 0.1).item(), 1.5412, 1e-3, 'the reference matters: chosen got 5 nats WORSE than ref, rejected 8 nats BETTER, so the margin is −1.3 and the loss 1.5412 even though the policy prefers chosen in absolute terms');
    T.close(m.dpoLoss([-1, -2], [-1, -2], [-1, -2], [-1, -2], 0.1).item(), LN2, 1e-5, 'policy = reference on every pair: every margin is 0, the loss is log 2');
  } },
  { step: 'dpo', name: 'at policy = reference the gradient moves chosen up and rejected down by β/(2N): DPO can leave the starting point', run(m, T) {
    const pc = Tensor.from([-4, -7], { requiresGrad: true });
    const pr = Tensor.from([-3, -9], { requiresGrad: true });
    const loss = m.dpoLoss(pc, pr, [-4, -7], [-3, -9], 0.1);
    T.close(loss.item(), LN2, 1e-5);
    loss.backward();
    T.ok(pc.grad !== null && pr.grad !== null, 'the policy log-probs must receive gradient (the reference log-probs are constants)');
    T.close(Array.from(pc.grad), [-0.025, -0.025], 1e-6, 'd loss / d log π(chosen) = −β·(1 − σ(0)) / N = −0.1·0.5/2');
    T.close(Array.from(pr.grad), [0.025, 0.025], 1e-6, 'd loss / d log π(rejected) = +β·0.5/2');
    const pc2 = Tensor.from([-4, -7], { requiresGrad: true }), pr2 = Tensor.from([-3, -9], { requiresGrad: true });
    m.dpoLoss(pc2, pr2, [-4, -7], [-3, -9], 0.5).backward();
    T.close(Array.from(pc2.grad), [-0.125, -0.125], 1e-6, 'five times the β, five times the gradient: β sets how hard the policy is pushed away from the reference');
  } },
  { step: 'dpo', name: 'dpoLoss equals the Bradley–Terry loss on the implicit rewards for random inputs', run(m, T) {
    const next = T.rng(31);
    const N = 6;
    const draw = () => Array.from({ length: N }, () => -20 * next());
    const pc = draw(), pr = draw(), rc = draw(), rr = draw();
    for (const beta of [0.05, 0.1, 1]) {
      const margins = pc.map((_, i) => beta * ((pc[i] - rc[i]) - (pr[i] - rr[i])));
      T.close(m.dpoLoss(pc, pr, rc, rr, beta).item(), btRef(margins), 1e-4, `β = ${beta}: −log σ(β[(π_c − ref_c) − (π_r − ref_r)]) averaged over ${N} pairs`);
      T.close(m.dpoLoss(pc, pr, rc, rr, beta).item(), m.bradleyTerryLoss(m.implicitRewards(pc, rc, beta), m.implicitRewards(pr, rr, beta)).item(), 1e-6, 'DPO is Bradley–Terry with r = β·log(π/ref); build it from your step-1 loss');
    }
  } },

  // ---------- step 4 ----------
  { step: 'reward-head', name: 'hiddenStates is the residual stream after the final LayerNorm: times wteᵀ it reproduces model.forward', run(m, T) {
    const model = tinyModel(8);
    const x = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]];
    const h = m.hiddenStates(model, x);
    T.ok(h instanceof Tensor, 'return a Tensor');
    T.shape(h, [2, 5, 16], '[B, T, C]: one C-vector per position, not logits [B, T, V]');
    const logits = model.forward(x);
    T.close(T.arr(h.matmul(model.wte.weight.transpose())), T.arr(logits), 1e-4, 'the tied head reads exactly this tensor: wte + wpe, every block, then lnF (forgetting lnF gives different logits)');
  } },
  { step: 'reward-head', name: 'lastTokenHidden gathers row lengths[b] − 1 of each sequence and routes gradient only to that row', run(m, T) {
    const next = T.rng(9);
    const data = new Float32Array(2 * 5 * 3);
    for (let i = 0; i < data.length; i++) data[i] = next() * 2 - 1;
    const h = new Tensor({ shape: [2, 5, 3], data }, { requiresGrad: true });
    const out = m.lastTokenHidden(h, [3, 5]);
    T.shape(out, [2, 3], '[B, C]');
    const row = (b, t) => Array.from(data.subarray((b * 5 + t) * 3, (b * 5 + t) * 3 + 3));
    T.close(T.arr(out), [row(0, 2), row(1, 4)], 1e-6, 'sequence 0 has 3 real tokens so its last one is at index 2 (not 3, and not T − 1 = 4, which is padding)');
    out.sum().backward();
    const g = h.grad;
    T.ok(g !== null, 'gradient must flow back into the hidden states (use Tensor ops such as mul with a one-hot selector and sum, not .data copies)');
    for (let b = 0; b < 2; b++) for (let t = 0; t < 5; t++) for (let c = 0; c < 3; c++) {
      const want = (b === 0 && t === 2) || (b === 1 && t === 4) ? 1 : 0;
      T.close(g[(b * 5 + t) * 3 + c], want, 1e-6, `gradient at [${b}, ${t}] must be ${want}: only the selected position feeds the reward`);
    }
    T.throws(() => m.lastTokenHidden(h, [0, 5]), 'a length of 0 has no last token and must throw');
    T.throws(() => m.lastTokenHidden(h, [3, 6]), 'a length beyond T must throw');
  } },
  { step: 'reward-head', name: 'trainRewardHead drives the Bradley–Terry loss down and reaches full reward accuracy on separable features', run(m, T) {
    const next = T.rng(12);
    const C = 8, N = 12;
    const v = Array.from({ length: C }, () => next() * 2 - 1);
    const feat = (sign) => Float32Array.from(v, (a) => sign * a + 0.3 * (next() * 2 - 1));
    const features = Array.from({ length: N }, () => ({ chosen: feat(1), rejected: feat(-1) }));
    const head = new m.RewardHead(C, { next: T.rng(2) });
    const before = paramsSnapshot(head);
    const history = m.trainRewardHead(head, features, { steps: 60, lr: 5e-2 });
    T.eq(history.length, 60, 'one record per step');
    T.ok(typeof history[0].loss === 'number' && typeof history[0].accuracy === 'number', 'each record is { loss, accuracy }');
    T.close(history[0].loss, LN2, 0.15, 'a fresh head scores every sequence near 0, so the first loss is close to log 2');
    T.ok(history[59].loss < history[0].loss * 0.5, `the loss must fall well below its start (first ${history[0].loss.toFixed(3)}, last ${history[59].loss.toFixed(3)})`);
    T.eq(history[59].accuracy, 1, 'these features are linearly separable, so the trained head must rank every pair correctly');
    T.ok(!paramsEqual(before, paramsSnapshot(head)), 'the head parameters must actually be updated by an optimizer step');
    for (const p of head.parameters()) T.ok(p.grad === null, 'gradients must be cleared after each step');
  } },

  // ---------- step 5 ----------
  { step: 'dpo-train', name: 'dpoStep: both responses forward, DPO loss, backward, clip, step, zeroGrad; the margin rises when the batch is repeated', run(m, T) {
    const policy = tinyModel(21);
    const pairs = fakePairs();
    const ref = refLogpsFor(policy, pairs);
    const idx = [0, 1];
    const batch = { chosen: refPad(idx.map((i) => pairs[i].chosen)), rejected: refPad(idx.map((i) => pairs[i].rejected)), refChosen: idx.map((i) => ref.chosen[i]), refRejected: idx.map((i) => ref.rejected[i]) };
    const optimizer = new AdamW(policy.parameters(), { lr: 1e-2 });
    const r1 = m.dpoStep(policy, optimizer, batch, { beta: 0.1 });
    T.ok(r1 && typeof r1.loss === 'number' && typeof r1.margin === 'number' && typeof r1.accuracy === 'number', 'return { loss, margin, accuracy }');
    T.close(r1.loss, LN2, 1e-4, 'on the first step the policy IS the reference, so every implicit reward is 0 and the loss is log 2');
    T.close(r1.margin, 0, 1e-5, 'the margin reported by a step is measured on that step\'s forward pass (before the update): 0 here');
    T.eq(r1.accuracy, 0, 'ties are not wins: accuracy 0 before any update');
    T.eq(optimizer.t, 1, 'exactly one optimizer step per call');
    for (const p of policy.parameters()) T.ok(p.grad === null, 'gradients must be cleared after the step');
    const r2 = m.dpoStep(policy, optimizer, batch, { beta: 0.1 });
    const r3 = m.dpoStep(policy, optimizer, batch, { beta: 0.1 });
    T.ok(r3.loss < r2.loss && r2.loss < r1.loss, `repeating a batch at lr 1e-2 must lower the loss (got ${r1.loss.toFixed(4)}, ${r2.loss.toFixed(4)}, ${r3.loss.toFixed(4)})`);
    T.ok(r3.margin > r2.margin && r2.margin > 0, `the implicit reward margin β·[(π_c − ref_c) − (π_r − ref_r)] must grow (got ${r2.margin.toFixed(4)} then ${r3.margin.toFixed(4)}); check the sign of the loss`);
    T.eq(r3.accuracy, 1, 'after two updates both pairs in the batch should be ranked correctly');
  } },
  { step: 'dpo-train', name: 'trainDPO: one record per step, onStep called, deterministic given the seed, the reference untouched and the margin positive on every pair', async run(m, T) {
    const policy = tinyModel(23);
    const reference = tinyModel(23);
    const refBefore = paramsSnapshot(reference);
    const pairs = fakePairs();
    const refLogps = refLogpsFor(reference, pairs);
    const seen = [];
    const history = await m.trainDPO(policy, pairs, refLogps, { steps: 20, beta: 0.1, lr: 1e-2, batchSize: 2, next: T.rng(4), onStep: (step, r) => { seen.push([step, r.loss]); } });
    T.eq(history.length, 20, 'one record per step');
    T.eq(seen.length, 20, 'onStep must be called once per step');
    T.eq(seen.map((s) => s[0]), history.map((_, i) => i), 'onStep receives the step index 0..steps−1');
    T.close(seen.map((s) => s[1]), history.map((r) => r.loss), 1e-9, 'onStep receives the same record that is returned');
    T.close(history[0].loss, LN2, 1e-4, 'step 0 starts at log 2 (policy = reference)');
    T.ok(history[19].loss < 0.3, `20 steps at lr 1e-2 on 3 pairs must bring the loss well below log 2 (got ${history[19].loss.toFixed(3)})`);
    T.ok(paramsEqual(refBefore, paramsSnapshot(reference)), 'the reference must never move: its log-probs are constants computed once');
    const after = refLogpsFor(policy, pairs);
    for (let i = 0; i < pairs.length; i++) {
      const margin = 0.1 * ((after.chosen[i] - refLogps.chosen[i]) - (after.rejected[i] - refLogps.rejected[i]));
      T.ok(margin > 0.2, `pair ${i}: the implicit reward margin after training must be clearly positive (got ${margin.toFixed(3)})`);
    }
    const again = await m.trainDPO(tinyModel(23), pairs, refLogps, { steps: 20, beta: 0.1, lr: 1e-2, batchSize: 2, next: T.rng(4) });
    T.close(again.map((r) => r.loss), history.map((r) => r.loss), 1e-6, 'the same seed must give the same run: the batch indices come from `next` and nothing else');
  } },
  { step: 'dpo-train', name: 'with β = 0 there is no signal: the loss stays at log 2, margins stay 0 and the parameters do not move', async run(m, T) {
    const policy = tinyModel(25);
    const pairs = fakePairs();
    const refLogps = refLogpsFor(policy, pairs);
    const before = paramsSnapshot(policy);
    const history = await m.trainDPO(policy, pairs, refLogps, { steps: 5, beta: 0, lr: 1e-2, batchSize: 2, next: T.rng(5) });
    for (const r of history) {
      T.close(r.loss, LN2, 1e-5, 'β multiplies the log-ratio difference; at β = 0 every margin is 0 and the loss is log 2 forever');
      T.close(r.margin, 0, 1e-9, 'the reported margin must be β times the log-ratio difference');
    }
    T.ok(paramsEqual(before, paramsSnapshot(policy)), 'a zero gradient must leave the parameters exactly where they were (if they moved, β is not being applied to the loss)');
    const moved = await m.trainDPO(policy, pairs, refLogps, { steps: 3, beta: 0.1, lr: 1e-2, batchSize: 2, next: T.rng(5) });
    T.ok(moved[2].loss < LN2 - 1e-3 && !paramsEqual(before, paramsSnapshot(policy)), 'and with β = 0.1 the same setup must train');
  } },
];
