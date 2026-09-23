import { GPT } from 'lib/gpt.js';
import { Tensor, noGrad, crossEntropy, gradCheck } from 'lib/tensor.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { loadModel } from 'lib/infer.js';
import { generate } from 'lib/sampling.js';
import { AdamW, clipGradNorm } from 'lib/optim.js';
import { rng, randInt } from 'lib/util.js';

const FIXTURE_TEXT = 'the cat sat on the mat. the dog ran to the barn. a bird sang in the tree. the cat and the dog sat. one two three four.';

function tinyTokenizer() {
  return BPETokenizer.train(FIXTURE_TEXT, { vocabSize: 48 });
}

function tinyGPT(vocabSize, seed, { blockSize = 16, nEmbd = 16 } = {}) {
  return new GPT({ vocabSize, blockSize, nLayer: 1, nHead: 2, nEmbd, seed });
}

/** A tiny teacher with peaked logits: random init scaled up so its distributions are far from uniform. */
function peakyTeacher(vocabSize, seed, scale = 40) {
  const model = tinyGPT(vocabSize, seed);
  for (let i = 0; i < model.wte.weight.data.length; i++) model.wte.weight.data[i] *= scale;
  return model;
}

function raw(nested) {
  const shape = [];
  let cur = nested;
  while (Array.isArray(cur)) { shape.push(cur.length); cur = cur[0]; }
  return { shape, data: Float32Array.from(nested.flat(Infinity)) };
}

function param(nested) {
  return Tensor.param(raw(nested));
}

function randomRaw(shape, next, scale = 3) {
  const n = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = (next() * 2 - 1) * scale;
  return { shape, data };
}

function softmaxRow(z, T = 1) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp((v - m) / T));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Reference mean forward KL(p_T ‖ q_T) · T² between two raw/Tensor logits of the same shape. */
function refKL(student, teacher, T = 1) {
  const V = student.shape[student.shape.length - 1];
  let total = 0, rows = 0;
  for (let r = 0; r < student.data.length; r += V) {
    const p = softmaxRow(Array.from(teacher.data.subarray(r, r + V)), T);
    const q = softmaxRow(Array.from(student.data.subarray(r, r + V)), T);
    for (let j = 0; j < V; j++) if (p[j] > 0) total += p[j] * Math.log(p[j] / q[j]);
    rows++;
  }
  return (T * T * total) / rows;
}

/** Reference mean reverse KL(q ‖ p) over rows with weight 1. */
function refReverseKL(student, teacher, weights) {
  const V = student.shape[student.shape.length - 1];
  let total = 0, count = 0;
  for (let r = 0, i = 0; r < student.data.length; r += V, i++) {
    if (!weights[i]) continue;
    const p = softmaxRow(Array.from(teacher.data.subarray(r, r + V)));
    const q = softmaxRow(Array.from(student.data.subarray(r, r + V)));
    for (let j = 0; j < V; j++) total += q[j] * Math.log(q[j] / p[j]);
    count++;
  }
  return total / count;
}

function asArray(t) {
  return Array.from(t.data);
}

/** A training pool the way buildPool makes one, from a local teacher, so step 6 does not depend on it. */
function makePool(teacher, windows, blockSize, seed) {
  const next = rng(seed);
  const V = teacher.config.vocabSize;
  const x = [], y = [];
  for (let w = 0; w < windows; w++) {
    const row = [];
    for (let t = 0; t <= blockSize; t++) row.push(Math.floor(next() * V));
    x.push(row.slice(0, -1));
    y.push(row.slice(1));
  }
  const logits = noGrad(() => teacher.forward(x));
  return { x, y, teacher: { shape: logits.shape.slice(), data: Float32Array.from(logits.data) } };
}

/** Rows idx of a raw [N, T, V] tensor (the tests' own copy, so step 6 does not depend on the starter's). */
function rows(t, idx) {
  const per = t.data.length / t.shape[0];
  const data = new Float32Array(idx.length * per);
  idx.forEach((i, k) => data.set(t.data.subarray(i * per, (i + 1) * per), k * per));
  return { shape: [idx.length, ...t.shape.slice(1)], data };
}

function poolKL(student, pool) {
  const logits = noGrad(() => student.forward(pool.x));
  return refKL(logits, pool.teacher, 1);
}

function poolCE(student, pool) {
  return noGrad(() => crossEntropy(student.forward(pool.x), pool.y).item());
}

export const tests = [
  // ---------- step 1: softTargets ----------
  { step: 'soft', name: 'softTargets at T = 1 is plain softmax, row by row, with the input shape', run(m, T) {
    const p = m.softTargets(raw([[1, 2, 3], [0, 0, 0]]), 1);
    T.shape(p, [2, 3], 'softTargets must keep the logits\' shape: one distribution per row');
    T.close(asArray(p), [0.09003, 0.24473, 0.66524, 1 / 3, 1 / 3, 1 / 3], 1e-4,
      'at T = 1 the soft targets are exactly softmax(z); each row is normalised on its own, not the whole tensor');
    const q = m.softTargets(raw([[1, 2, 3]]), 2);
    T.close(asArray(q), [0.18632, 0.30720, 0.50648], 1e-4, 'at T = 2 the result must be softmax(z / 2) = softmax([0.5, 1, 1.5]): divide the logits by T BEFORE exponentiating');
  } },
  { step: 'soft', name: 'high T flattens towards uniform, low T sharpens towards one-hot, huge logits stay finite', run(m, T) {
    const z = raw([[4, 1, -2, 0]]);
    const hot = asArray(m.softTargets(z, 1000));
    T.close(hot, [0.25, 0.25, 0.25, 0.25], 2e-3, 'as T grows every logit / T goes to 0, so the distribution becomes uniform: this is what exposes the "dark knowledge" in the small probabilities');
    const cold = asArray(m.softTargets(z, 0.05));
    T.ok(cold[0] > 0.999, `at T = 0.05 almost all mass must sit on the largest logit, got ${cold[0].toFixed(4)}`);
    const big = asArray(m.softTargets(raw([[1000, 999, 998]]), 0.5));
    T.ok(big.every((v) => Number.isFinite(v)), 'logits of 1000 overflow exp(); subtract the row maximum first');
    T.close(big, softmaxRow([1000, 999, 998], 0.5), 1e-4);
    T.throws(() => m.softTargets(z, 0), 'T = 0 would divide by zero: throw for T <= 0 rather than return NaN');
  } },
  { step: 'soft', name: 'softTargets works on a [B, T, V] Tensor and does not mutate it', run(m, T) {
    const next = T.rng(3);
    const logits = new Tensor(randomRaw([2, 3, 5], next));
    const before = Array.from(logits.data);
    const p = m.softTargets(logits, 3);
    T.shape(p, [2, 3, 5]);
    for (let r = 0; r < 6; r++) {
      const row = Array.from(p.data.subarray(r * 5, r * 5 + 5));
      T.close(row, softmaxRow(before.slice(r * 5, r * 5 + 5), 3), 1e-4, `row ${r} must be softmax(z / T) of its own logits`);
    }
    T.eq(Array.from(logits.data), before, 'the teacher logits must not be modified in place');
  } },

  // ---------- step 2: klDistillLoss ----------
  { step: 'kl', name: 'hand-computed 3-class example at T = 1 and T = 2', run(m, T) {
    const teacher = raw([[2, 1, 0]]);
    const l1 = m.klDistillLoss(param([[0, 1, 2]]), teacher, 1);
    T.ok(l1 instanceof Tensor && l1.size === 1, 'klDistillLoss must return a scalar Tensor so you can call backward() on it');
    T.close(l1.item(), 1.15042, 1e-4, 'p = [0.665, 0.245, 0.090], q = [0.090, 0.245, 0.665]: KL(p ‖ q) = Σ p·log(p/q) = 1.1504');
    const l2 = m.klDistillLoss(param([[0, 1, 2]]), teacher, 2);
    T.close(l2.item(), 1.28063, 1e-4, 'at T = 2, KL(p₂ ‖ q₂) = 0.32016 and the loss is T² · 0.32016 = 1.2806; without the T² factor you get 0.320');
  } },
  { step: 'kl', name: 'zero when the logits match, forward (not reverse) KL, averaged over positions', run(m, T) {
    const next = T.rng(5);
    const same = randomRaw([2, 3, 6], next);
    T.close(m.klDistillLoss(new Tensor({ shape: same.shape, data: Float32Array.from(same.data) }, { requiresGrad: true }), same, 2).item(), 0, 1e-5,
      'a student that already matches the teacher must have zero distillation loss (include the Σ p·log p term)');
    const fwd = m.klDistillLoss(param([[1, 1, 0]]), raw([[3, 0, -1]]), 1).item();
    T.close(fwd, 0.60483, 1e-4, 'the loss is KL(teacher ‖ student) = Σ p_teacher · log(p_teacher / q_student) = 0.6048; the reverse direction would give 0.9369');
    const one = m.klDistillLoss(param([[0, 1, 2]]), raw([[2, 1, 0]]), 1).item();
    const twice = m.klDistillLoss(param([[[0, 1, 2], [0, 1, 2]]]), raw([[[2, 1, 0], [2, 1, 0]]]), 1).item();
    T.close(twice, one, 1e-4, 'two identical positions must give the same loss as one: average over positions, do not sum');
    const s = randomRaw([2, 3, 6], next), t = randomRaw([2, 3, 6], next);
    T.close(m.klDistillLoss(new Tensor(s, { requiresGrad: true }), t, 1.5).item(), refKL(s, t, 1.5), 1e-4, 'random [2, 3, 6] logits must match T² · mean over the 6 positions of KL(p_T ‖ q_T)');
  } },
  { step: 'kl', name: 'gradient is T · (q_T − p_T) / N: the T² keeps its size independent of T', run(m, T) {
    const next = T.rng(8);
    const t = randomRaw([4, 5], next);
    for (const temp of [1, 4]) {
      const s = new Tensor(randomRaw([4, 5], next), { requiresGrad: true });
      m.klDistillLoss(s, t, temp).backward();
      T.ok(s.grad, 'the student logits must receive a gradient: build the loss from Tensor ops on studentLogits');
      const expected = [];
      for (let r = 0; r < 4; r++) {
        const p = softmaxRow(Array.from(t.data.subarray(r * 5, r * 5 + 5)), temp);
        const q = softmaxRow(Array.from(s.data.subarray(r * 5, r * 5 + 5)), temp);
        for (let j = 0; j < 5; j++) expected.push((temp * (q[j] - p[j])) / 4);
      }
      T.close(Array.from(s.grad), expected, 1e-4, `at T = ${temp} the gradient must be T·(q_T − p_T)/N (Hinton et al. 2015); without T² it shrinks by 1/T² and the soft term fades as T grows`);
    }
  } },

  // ---------- step 3: distillLoss ----------
  { step: 'mixed', name: 'the two limits: alpha = 1 is the KL term, alpha = 0 is plain cross-entropy (teacher may be null)', run(m, T) {
    const next = T.rng(12);
    const s = randomRaw([2, 3, 7], next), t = randomRaw([2, 3, 7], next);
    const y = [[1, 4, 6], [0, 2, 3]];
    const kd = m.distillLoss(new Tensor(s, { requiresGrad: true }), t, y, { alpha: 1, T: 2 });
    T.close(kd.item(), refKL(s, t, 2), 1e-4, 'alpha = 1 must be exactly T² · KL at temperature T, with no cross-entropy mixed in');
    const ce = m.distillLoss(new Tensor(s, { requiresGrad: true }), null, y, { alpha: 0, T: 2 });
    T.close(ce.item(), crossEntropy(new Tensor(s), y).item(), 1e-5, 'alpha = 0 must be the ordinary cross-entropy on the hard labels; it must not touch the (null) teacher');
  } },
  { step: 'mixed', name: 'alpha mixes the terms, and the hard-label term uses T = 1', run(m, T) {
    const loss = m.distillLoss(param([[0, 1, 2]]), raw([[2, 1, 0]]), [2], { alpha: 0.3, T: 2 });
    T.close(loss.item(), 0.66951, 1e-4, '0.3 · 1.2806 (T² · KL at T = 2) + 0.7 · 0.4076 (CE at T = 1) = 0.6695; computing the CE on logits / T would give 0.6769');
    const next = T.rng(13);
    const sRaw = randomRaw([3, 5], next), tRaw = randomRaw([3, 5], next), y = [0, 3, 4];
    const grads = [0, 1, 0.25].map((alpha) => {
      const s = new Tensor({ shape: sRaw.shape, data: Float32Array.from(sRaw.data) }, { requiresGrad: true });
      m.distillLoss(s, alpha === 0 ? null : tRaw, y, { alpha, T: 3 }).backward();
      T.ok(s.grad, `alpha = ${alpha}: the loss must be differentiable with respect to the student logits`);
      return Array.from(s.grad);
    });
    T.close(grads[2], grads[0].map((g, i) => 0.75 * g + 0.25 * grads[1][i]), 1e-4, 'the gradient of the mixture must be (1 − alpha)·∇CE + alpha·∇KL');
  } },

  // ---------- step 4: sequenceLevelCorpus ----------
  { step: 'seqkd', name: 'each document is prompt + the teacher\'s sampled continuation, then eos', run(m, T) {
    const tok = tinyTokenizer();
    const teacher = loadModel(peakyTeacher(tok.vocabSize, 21, 4).toJSON());
    const prompts = ['the cat', 'a bird', 'the dog ran'];
    const out = m.sequenceLevelCorpus(teacher, tok, prompts, { maxNewTokens: 8, temperature: 1, next: rng(4) });
    T.ok(out && Array.isArray(out.ids) && Array.isArray(out.texts), 'return { ids, texts }');
    T.eq(out.texts.length, prompts.length, 'one generated document per prompt');
    const ref = rng(4);
    const expected = prompts.map((p) => p + generate(teacher, tok, p, { maxNewTokens: 8, temperature: 1, next: ref }));
    T.eq(out.texts, expected, 'texts[i] must be prompts[i] + generate(teacher, …) with ONE rng threaded through the prompts in order; the teacher, not the data, writes the corpus');
    T.ok(out.texts.some((t, i) => t.length > prompts[i].length), 'at least one continuation must be non-empty');
    const other = m.sequenceLevelCorpus(teacher, tok, prompts, { maxNewTokens: 8, temperature: 1, next: rng(5) });
    T.ok(other.texts.some((t, i) => t !== out.texts[i]), 'at temperature 1 the teacher SAMPLES, so another seed must give another corpus: pass `temperature` and `next` through to generate');
  } },
  { step: 'seqkd', name: 'ids are encode(text) + eos per document; temperature is passed through', run(m, T) {
    const tok = tinyTokenizer();
    const teacher = loadModel(peakyTeacher(tok.vocabSize, 22, 4).toJSON());
    // 'the c' ends mid-word: with rng(8) the continuation starts with letters that BPE merges across the
    // boundary, so encode(prompt) + encode(continuation) is NOT encode(prompt + continuation).
    const prompts = ['the cat', 'one two', 'the c', 'a bi'];
    const out = m.sequenceLevelCorpus(teacher, tok, prompts, { maxNewTokens: 6, temperature: 1, next: rng(8) });
    const expectIds = [];
    for (const text of out.texts) expectIds.push(...tok.encode(text), tok.eos);
    T.eq(out.ids, expectIds, 'the token stream is encode(prompt + continuation) (encode the whole text once: BPE merges can cross the prompt boundary) and an eos separator after each document, so getBatch can sample windows from it like a pre-training corpus');
    T.eq(out.ids.filter((id) => id === tok.eos).length, prompts.length, 'exactly one eos per document');
    const greedy = prompts.map((p) => p + generate(teacher, tok, p, { maxNewTokens: 6, temperature: 0, next: rng(0) }));
    const g1 = m.sequenceLevelCorpus(teacher, tok, prompts, { maxNewTokens: 6, temperature: 0, next: rng(1) });
    const g2 = m.sequenceLevelCorpus(teacher, tok, prompts, { maxNewTokens: 6, temperature: 0, next: rng(2) });
    T.eq(g1.texts, greedy, 'temperature 0 is greedy decoding (Kim & Rush used beam search, the mode-seeking variant): pass `temperature` through to generate');
    T.eq(g2.texts, greedy, 'at temperature 0 the seed must not matter');
  } },

  // ---------- step 5: reverseKL and onPolicyLoss ----------
  { step: 'onpolicy', name: 'reverseKL is KL(student ‖ teacher): hand-computed, zero on a match, mode-seeking', run(m, T) {
    const r = m.reverseKL(param([[1, 1, 0]]), raw([[3, 0, -1]]));
    T.ok(r instanceof Tensor && r.size === 1, 'reverseKL must return a scalar Tensor');
    T.close(r.item(), 0.93693, 1e-4, 'Σ q·log(q/p) with q = student, p = teacher is 0.9369; the forward direction would give 0.6048');
    T.close(m.reverseKL(param([[0.5, -1, 2]]), raw([[0.5, -1, 2]])).item(), 0, 1e-5, 'identical distributions have zero divergence');
    // Teacher bimodal on tokens 0 and 1; student bets everything on token 0.
    const rev = m.reverseKL(param([[3, -1, -1]]), raw([[2, 2, -2]])).item();
    T.close(rev, 0.59561, 1e-4, 'reverse KL of a student that covers one of the two modes is only 0.596 (forward KL would be 1.316): reverse KL tolerates dropping a mode, which is why it is called mode-seeking');
  } },
  { step: 'onpolicy', name: 'reverseKL averages over masked-in positions only, and its gradient is correct', run(m, T) {
    const next = T.rng(31);
    const s = randomRaw([2, 3, 5], next), t = randomRaw([2, 3, 5], next);
    const mask = [[0, 1, 1], [0, 0, 1]];
    const w = mask.flat();
    const st = new Tensor({ shape: s.shape, data: Float32Array.from(s.data) }, { requiresGrad: true });
    const loss = m.reverseKL(st, t, mask);
    T.close(loss.item(), refReverseKL(s, t, w), 1e-4, 'the mean must run over the 3 masked-in positions, not all 6');
    T.close(m.reverseKL(new Tensor(s, { requiresGrad: true }), t, null).item(), refReverseKL(s, t, [1, 1, 1, 1, 1, 1]), 1e-4, 'mask = null means every position counts');
    loss.backward();
    for (let i = 0; i < 6; i++) {
      if (w[i]) continue;
      T.eq(Array.from(st.grad.subarray(i * 5, i * 5 + 5)), [0, 0, 0, 0, 0], `position ${i} is masked out (a prompt token): its gradient must be exactly zero`);
    }
    const gc = gradCheck((x) => m.reverseKL(x, t, mask), [Tensor.param({ shape: s.shape, data: Float32Array.from(s.data) })]);
    T.ok(gc.ok, `gradient check failed (max relative error ${gc.maxRelErr.toExponential(2)}): the gradient must flow through BOTH q and log q`);
  } },
  { step: 'onpolicy', name: 'onPolicyLoss: the student samples, the mask covers the generated tokens, the loss is reverse KL there', run(m, T) {
    const V = 24;
    const student = tinyGPT(V, 41);
    const teacher = peakyTeacher(V, 42);
    const prompts = [[1, 2, 3], [4, 5, 6]];
    const out = m.onPolicyLoss(student, teacher, prompts, { maxNewTokens: 5, temperature: 1, next: rng(7) });
    const ref = rng(7);
    const expectIds = prompts.map((p) => student.generate(p, { maxNewTokens: 5, temperature: 1, next: ref }));
    T.eq(out.ids, expectIds, 'on-policy means the STUDENT writes the continuations: ids[i] = student.generate(prompts[i], …) with one rng threaded in order');
    T.eq(out.mask, [[0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1]], 'x = ids without its last token; position t predicts token t + 1, so the first generated token (index 3) is predicted at position 2 = promptLen − 1');
    const x = expectIds.map((s) => s.slice(0, -1));
    const sl = noGrad(() => student.forward(x));
    const tl = noGrad(() => teacher.forward(x));
    T.close(out.loss.item(), refReverseKL(sl, tl, out.mask.flat()), 1e-4, 'the loss must be the reverse KL between the student\'s and the teacher\'s next-token distributions on the generated positions');
  } },
  { step: 'onpolicy', name: 'onPolicyLoss trains the student and leaves the teacher frozen', run(m, T) {
    const V = 24;
    const student = tinyGPT(V, 51);
    const teacher = peakyTeacher(V, 52);
    const before = teacher.parameters().map((p) => Array.from(p.data));
    const { loss } = m.onPolicyLoss(student, teacher, [[1, 2], [3, 4], [5, 6]], { maxNewTokens: 4, next: rng(3) });
    T.ok(loss instanceof Tensor && loss.size === 1, 'return { loss } as a scalar Tensor');
    loss.backward();
    T.ok(teacher.parameters().every((p) => p.grad === null || p.grad.every((g) => g === 0)), 'the teacher must receive no gradient: compute its logits under noGrad (teacherLogits does this)');
    T.ok(teacher.parameters().every((p, i) => p.data.every((v, j) => v === before[i][j])), 'the teacher\'s weights must not change');
    T.ok(student.wte.weight.grad && student.wte.weight.grad.some((g) => g !== 0), 'the student\'s parameters must receive a gradient');
    T.throws(() => m.onPolicyLoss(student, teacher, [[1, 2], [3]], { maxNewTokens: 2, next: rng(1) }), 'prompts of different lengths cannot share one mask; throw');
    const g = m.onPolicyLoss(student, teacher, [[1, 2], [7, 8]], { maxNewTokens: 5, temperature: 0, next: rng(5) });
    const greedy = [[1, 2], [7, 8]].map((p) => student.generate(p, { maxNewTokens: 5, temperature: 0, next: rng(0) }));
    T.eq(g.ids, greedy, 'pass `temperature` to student.generate: at temperature 0 the student\'s samples are its greedy continuations');
  } },

  // ---------- step 6: the experiment ----------
  { step: 'experiment', name: 'top1Agreement counts positions where the argmaxes match', run(m, T) {
    const s = raw([[[0.1, 0.9, 0], [0.8, 0.1, 0.1], [0, 0, 1]]]);
    const t = raw([[[0, 5, 1], [1, 2, 0], [-1, -1, 3]]]);
    T.close(m.top1Agreement(s, t), 2 / 3, 1e-9, 'positions 0 and 2 agree (argmax 1 and 2), position 1 does not (0 vs 1): 2 of 3');
    T.close(m.top1Agreement(t, t), 1, 1e-9, 'a model agrees with itself everywhere');
    T.close(m.top1Agreement(raw([[1, 1, 0], [0, 2, 2]]), raw([[5, 0, 0], [0, 3, 1]])), 1, 1e-9, 'ties go to the first index, as in argmaxArray');
    T.close(m.top1Agreement(raw([[1, 0], [0, 1], [1, 0], [0, 1]]), raw([[0, 1], [0, 1], [0, 1], [0, 1]])), 0.5, 1e-9, 'return a fraction of positions, not a count');
  } },
  { step: 'experiment', name: 'distillTrain: alpha = 1 pulls the student towards the teacher, alpha = 0 learns the labels', run(m, T) {
    const V = 20;
    const teacher = peakyTeacher(V, 61);
    const pool = makePool(teacher, 12, 8, 62);
    const kdStudent = tinyGPT(V, 63);
    const ceStudent = tinyGPT(V, 63);
    const kl0 = poolKL(kdStudent, pool);
    const ce0 = poolCE(ceStudent, pool);
    const losses = m.distillTrain(kdStudent, pool, { steps: 30, batchSize: 4, lr: 1e-2, alpha: 1, T: 1, next: rng(64) });
    T.ok(losses && typeof losses.then === 'function', 'distillTrain must be async (it awaits onStep so the demo can yield)');
    return losses.then(async (kdLosses) => {
      T.eq(kdLosses.length, 30, 'return one loss per step');
      const ceLosses = await m.distillTrain(ceStudent, { x: pool.x, y: pool.y, teacher: null }, { steps: 30, batchSize: 4, lr: 1e-2, alpha: 0, T: 1, next: rng(64) });
      const kl1 = poolKL(kdStudent, pool), klCE = poolKL(ceStudent, pool);
      T.ok(kl1 < 0.8 * kl0, `after 30 distillation steps the KL to the teacher should fall by at least 20% (from ${kl0.toFixed(3)}, got ${kl1.toFixed(3)}): update the student with the distillation loss`);
      T.ok(kl1 < klCE, `the distilled student must end closer to the teacher than the label-trained one (KL ${kl1.toFixed(3)} vs ${klCE.toFixed(3)}): alpha must select the KL term`);
      T.ok(poolCE(ceStudent, pool) < ce0 - 0.1, 'alpha = 0 with no teacher must still train on the hard labels');
      T.ok(ceLosses.every(Number.isFinite), 'losses must be finite numbers');
    });
  } },
  { step: 'experiment', name: 'distillTrain replays the specified loop: randInt indices, matching teacher rows, alpha and T passed on, clip to 1, AdamW', run(m, T) {
    const V = 20;
    const teacher = peakyTeacher(V, 81);
    const pool = makePool(teacher, 10, 6, 82);
    const opts = { steps: 5, batchSize: 3, lr: 2e-2, alpha: 0.7, T: 3 };
    // The reference loop, written out with the same lib pieces and YOUR distillLoss (tested in step 3).
    const ref = tinyGPT(V, 83);
    const params = ref.parameters();
    const opt = new AdamW(params, { lr: opts.lr, betas: [0.9, 0.95], weightDecay: 0 });
    const next = rng(84);
    const expected = [];
    for (let step = 0; step < opts.steps; step++) {
      const idx = [];
      for (let b = 0; b < opts.batchSize; b++) idx.push(randInt(next, pool.x.length));
      const loss = m.distillLoss(ref.forward(idx.map((i) => pool.x[i])), rows(pool.teacher, idx), idx.map((i) => pool.y[i]), { alpha: opts.alpha, T: opts.T });
      opt.zeroGrad();
      loss.backward();
      clipGradNorm(params, 1.0);
      opt.step();
      expected.push(loss.item());
    }
    const seen = [];
    return m.distillTrain(tinyGPT(V, 83), pool, { ...opts, next: rng(84), onStep: async (step) => { seen.push(step); } }).then((losses) => {
      T.close(losses, expected, 1e-4, 'per-step losses must match the loop in the instructions: indices from randInt(next, N), the cached teacher rows OF THOSE windows, distillLoss with the given alpha and T, zeroGrad, backward, clipGradNorm(params, 1.0), AdamW step (betas [0.9, 0.95], no weight decay)');
      T.eq(seen, [0, 1, 2, 3, 4], 'call (and await) onStep(step, loss) once per step, in order: the demo evaluates and yields from it');
    });
  } },
  { step: 'experiment', name: 'distillTrain is deterministic in `next`; evaluate reports CE and agreement over the whole pool', run(m, T) {
    const V = 20;
    const teacher = peakyTeacher(V, 71);
    const pool = makePool(teacher, 10, 6, 72);
    const run = (seed) => m.distillTrain(tinyGPT(V, 73), pool, { steps: 5, batchSize: 3, lr: 1e-2, alpha: 0.5, T: 2, next: rng(seed) });
    return Promise.all([run(1), run(1), run(2)]).then(([a, b, c]) => {
      T.close(a, b, 1e-6, 'same seed, same losses: every random draw must come from `next`');
      T.ok(a.some((v, i) => Math.abs(v - c[i]) > 1e-6), 'a different seed must draw different windows');
      const student = tinyGPT(V, 74);
      const res = m.evaluate(student, pool);
      const logits = noGrad(() => student.forward(pool.x));
      T.close(res.loss, crossEntropy(logits, pool.y).item(), 1e-4, 'evaluate().loss is the mean cross-entropy on the true next tokens over every position of every window');
      T.close(res.agreement, m.top1Agreement(logits, pool.teacher), 1e-6, 'evaluate().agreement is the top-1 agreement with the pool\'s teacher logits');
      T.eq(m.evaluate(student, { x: pool.x, y: pool.y, teacher: null }).agreement, null, 'with no teacher logits there is no agreement to report: null');
    });
  } },
];
