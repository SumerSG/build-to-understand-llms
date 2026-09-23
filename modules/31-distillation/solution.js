// Module 31 — reference solution: temperature-softened targets, the T²-scaled KL loss, the mixed objective,
// sequence-level and on-policy data paths, and the from-scratch vs distilled experiment.
//
// Conventions (the lab's): logits are [B, T, V] (batch, time, vocabulary) or [N, V] (in shapes T is time; the
// argument named T in the loss functions is the temperature); a "raw" tensor is
// { shape, data: Float32Array }; the teacher is FROZEN, so everything computed from it is a constant raw
// tensor produced inside noGrad and wrapped as a plain (non-grad) Tensor when it meets the student's graph.

import { GPT } from 'lib/gpt.js';
import { Tensor, noGrad, crossEntropy } from 'lib/tensor.js';
import { AdamW, clipGradNorm } from 'lib/optim.js';
import { generate } from 'lib/sampling.js';
import { getBatch } from 'lib/data.js';
import { randInt } from 'lib/util.js';

// ---------- worked examples ----------

/** The student: the same architecture as the checkpoint teacher, cut down to 1 layer of 32 channels. */
export function makeStudent({ vocabSize, blockSize, seed = 0 }) {
  return new GPT({ vocabSize, blockSize, nLayer: 1, nHead: 2, nEmbd: 32, seed });
}

/**
 * The frozen teacher's logits for a batch of windows x (number[][], B×T), as a raw [B, T, V] tensor.
 * noGrad means no graph is recorded, so no gradient can ever reach the teacher's weights.
 */
export function teacherLogits(teacher, x) {
  const logits = noGrad(() => teacher.forward(x));
  return { shape: logits.shape.slice(), data: logits.data };
}

/**
 * Offline distillation: draw `windows` training windows once and cache the teacher's logits for them,
 * so training never runs the teacher again. Returns { x, y, teacher } with teacher a raw [N, T, V].
 */
export function buildPool(teacher, ids, { windows, blockSize, next, chunk = 8 }) {
  const { x, y } = getBatch(ids, { blockSize, batchSize: windows, next });
  const V = teacher.config.vocabSize;
  const data = new Float32Array(windows * blockSize * V);
  for (let start = 0; start < windows; start += chunk) {
    const part = teacherLogits(teacher, x.slice(start, start + chunk));
    data.set(part.data, start * blockSize * V);
  }
  return { x, y, teacher: { shape: [windows, blockSize, V], data } };
}

/** Log-softmax of every row of a raw tensor (a constant: no graph). Used for teacher log-probabilities. */
export function logSoftmaxRows(t) {
  const V = t.shape[t.shape.length - 1];
  const out = new Float32Array(t.data.length);
  for (let r = 0; r < t.data.length; r += V) {
    let max = -Infinity;
    for (let j = 0; j < V; j++) if (t.data[r + j] > max) max = t.data[r + j];
    let z = 0;
    for (let j = 0; j < V; j++) z += Math.exp(t.data[r + j] - max);
    const logZ = max + Math.log(z);
    for (let j = 0; j < V; j++) out[r + j] = t.data[r + j] - logZ;
  }
  return { shape: t.shape.slice(), data: out };
}

// ---------- step 1: temperature-softened targets ----------

/** softmax(z / T) along the last axis. Accepts a raw tensor, a Tensor, or a plain 1-D array of logits. */
export function softTargets(logits, T = 1) {
  if (!(T > 0)) throw new Error(`softTargets: temperature must be > 0, got ${T}`);
  const data = logits.data ?? Float32Array.from(logits);
  const shape = logits.shape ? logits.shape.slice() : [data.length];
  const V = shape[shape.length - 1];
  const out = new Float32Array(data.length);
  for (let r = 0; r < data.length; r += V) {
    let max = -Infinity;
    for (let j = 0; j < V; j++) if (data[r + j] > max) max = data[r + j];
    let z = 0;
    for (let j = 0; j < V; j++) {
      const e = Math.exp((data[r + j] - max) / T);
      out[r + j] = e;
      z += e;
    }
    for (let j = 0; j < V; j++) out[r + j] /= z;
  }
  return { shape, data: out };
}

// ---------- step 2: the distillation loss ----------

/**
 * Forward KL from the teacher's softened distribution p to the student's q, scaled by T² and averaged
 * over positions:  T² · mean_positions Σ_v p(v) · (log p(v) − log q(v)).  Returns a scalar Tensor.
 */
export function klDistillLoss(studentLogits, targetLogits, T = 1) {
  const p = softTargets(targetLogits, T);
  if (p.data.length !== studentLogits.data.length) {
    throw new Error(`klDistillLoss: teacher [${p.shape}] and student [${studentLogits.shape}] differ`);
  }
  const V = p.shape[p.shape.length - 1];
  const positions = p.data.length / V;
  // Σ p log p does not depend on the student: a constant that makes the loss exactly 0 when q = p.
  let plogp = 0;
  for (let i = 0; i < p.data.length; i++) if (p.data[i] > 0) plogp += p.data[i] * Math.log(p.data[i]);
  const logq = studentLogits.scale(1 / T).logSoftmax();
  const crossTerm = logq.mul(new Tensor({ shape: studentLogits.shape.slice(), data: p.data })).sum(); // Σ p log q
  return crossTerm.neg().add(plogp).scale((T * T) / positions);
}

// ---------- step 3: the mixed objective ----------

/**
 * alpha · KL(teacher ‖ student at temperature T) + (1 − alpha) · CE(student, hard labels at T = 1).
 * With alpha = 0 the teacher may be null: that is plain training from the data.
 */
export function distillLoss(studentLogits, targetLogits, targets, { alpha = 0.5, T = 2 } = {}) {
  if (!(alpha >= 0 && alpha <= 1)) throw new Error(`distillLoss: alpha must be in [0, 1], got ${alpha}`);
  if (alpha === 0) return crossEntropy(studentLogits, targets);
  const kl = klDistillLoss(studentLogits, targetLogits, T);
  if (alpha === 1) return kl;
  return kl.scale(alpha).add(crossEntropy(studentLogits, targets).scale(1 - alpha));
}

// ---------- step 4: sequence-level distillation ----------

/**
 * Kim & Rush (2016): let the teacher write the training set. For each prompt, sample a continuation from
 * the teacher (an InferModel from lib/infer.js loadModel) with lib/sampling.js generate, and append
 * encode(prompt + continuation) followed by eos to one token stream. Returns { ids, texts }.
 */
export function sequenceLevelCorpus(teacher, tokenizer, prompts, { maxNewTokens = 32, temperature = 1, next }) {
  if (typeof next !== 'function') throw new Error('sequenceLevelCorpus: pass a seeded rng function as `next`');
  const ids = [];
  const texts = [];
  for (const prompt of prompts) {
    const continuation = generate(teacher, tokenizer, prompt, { maxNewTokens, temperature, next });
    const text = prompt + continuation;
    texts.push(text);
    ids.push(...tokenizer.encode(text), tokenizer.eos);
  }
  return { ids, texts };
}

// ---------- step 5: on-policy distillation ----------

/**
 * Reverse KL, KL(q_student ‖ p_teacher) = Σ_v q(v) · (log q(v) − log p(v)), averaged over the positions
 * where mask is 1 (mask: number[][] B×T, or null for every position). Returns a scalar Tensor.
 */
export function reverseKL(studentLogits, targetLogits, mask = null) {
  const V = studentLogits.shape[studentLogits.shape.length - 1];
  const positions = studentLogits.data.length / V;
  const weights = new Float32Array(positions);
  if (mask === null) weights.fill(1);
  else {
    const flat = mask.flat(Infinity);
    if (flat.length !== positions) throw new Error(`reverseKL: mask has ${flat.length} entries for ${positions} positions`);
    for (let i = 0; i < positions; i++) weights[i] = flat[i];
  }
  let count = 0;
  for (const w of weights) count += w;
  if (count === 0) throw new Error('reverseKL: the mask selects no positions');
  const logp = new Tensor(logSoftmaxRows(targetLogits));
  const logq = studentLogits.logSoftmax();
  const q = logq.exp();
  const perToken = q.mul(logq.sub(logp)); // [.., V]
  const w = new Tensor({ shape: [...studentLogits.shape.slice(0, -1), 1], data: weights });
  return perToken.mul(w).sum().scale(1 / count);
}

/**
 * One on-policy batch (GKD, Agarwal et al. 2023): the STUDENT samples continuations of the prompts, the
 * frozen teacher scores every position of those samples, and the loss is the reverse KL on the positions
 * that predict a generated token. Prompts must share one length. Returns { loss, ids, mask }.
 */
export function onPolicyLoss(student, teacher, promptIds, { maxNewTokens, temperature = 1, next }) {
  const promptLen = promptIds[0].length;
  if (promptIds.some((p) => p.length !== promptLen)) throw new Error('onPolicyLoss: prompts must share one length');
  if (promptLen < 1) throw new Error('onPolicyLoss: prompts need at least one token');
  const ids = promptIds.map((p) => student.generate(p, { maxNewTokens, temperature, next }));
  const x = ids.map((s) => s.slice(0, -1));
  // Position t predicts token t + 1; the first generated token sits at index promptLen.
  const mask = x.map((row) => row.map((_, t) => (t >= promptLen - 1 ? 1 : 0)));
  const target = teacherLogits(teacher, x);
  const loss = reverseKL(student.forward(x), target, mask);
  return { loss, ids, mask };
}

// ---------- step 6: the experiment ----------

/** Fraction of positions where the student's argmax equals the teacher's argmax (first index on ties). */
export function top1Agreement(studentLogits, targetLogits) {
  const V = studentLogits.shape[studentLogits.shape.length - 1];
  const a = studentLogits.data, b = targetLogits.data;
  if (a.length !== b.length) throw new Error('top1Agreement: shapes differ');
  let match = 0, positions = 0;
  for (let r = 0; r < a.length; r += V) {
    let ia = 0, ib = 0;
    for (let j = 1; j < V; j++) {
      if (a[r + j] > a[r + ia]) ia = j;
      if (b[r + j] > b[r + ib]) ib = j;
    }
    if (ia === ib) match++;
    positions++;
  }
  return match / positions;
}

/** Rows `idx` of a raw [N, T, V] tensor, as a raw [idx.length, T, V]. */
function takeRows(t, idx) {
  const per = t.data.length / t.shape[0];
  const data = new Float32Array(idx.length * per);
  idx.forEach((i, k) => data.set(t.data.subarray(i * per, (i + 1) * per), k * per));
  return { shape: [idx.length, ...t.shape.slice(1)], data };
}

/**
 * Train `student` on a pool { x, y, teacher } for `steps` AdamW steps. Each step draws batchSize windows
 * at random (with `next`), computes distillLoss with the cached teacher logits of those windows, clips
 * the gradient norm to 1 and updates. alpha = 0 (teacher may be null) is training from scratch.
 * Returns the per-step losses.
 */
export async function distillTrain(student, pool, { steps, batchSize = 4, lr = 3e-3, alpha = 0.5, T = 2, next, onStep = null }) {
  if (typeof next !== 'function') throw new Error('distillTrain: pass a seeded rng function as `next`');
  if (alpha > 0 && !pool.teacher) throw new Error('distillTrain: alpha > 0 needs teacher logits in the pool');
  const params = student.parameters();
  const optimizer = new AdamW(params, { lr, betas: [0.9, 0.95], weightDecay: 0 });
  const losses = [];
  for (let step = 0; step < steps; step++) {
    const idx = [];
    for (let b = 0; b < batchSize; b++) idx.push(randInt(next, pool.x.length));
    const x = idx.map((i) => pool.x[i]);
    const y = idx.map((i) => pool.y[i]);
    const target = alpha > 0 ? takeRows(pool.teacher, idx) : null;
    const loss = distillLoss(student.forward(x), target, y, { alpha, T });
    optimizer.zeroGrad();
    loss.backward();
    clipGradNorm(params, 1.0);
    optimizer.step();
    losses.push(loss.item());
    if (onStep) await onStep(step, loss.item());
  }
  return losses;
}

/**
 * Held-out report on a pool: mean cross-entropy on the true next tokens, and (when the pool has teacher
 * logits) top-1 agreement with the teacher. Runs under noGrad in chunks of 8 windows.
 */
export function evaluate(model, pool, { chunk = 8 } = {}) {
  let nll = 0, agree = 0, positions = 0;
  noGrad(() => {
    for (let start = 0; start < pool.x.length; start += chunk) {
      const idx = [];
      for (let i = start; i < Math.min(pool.x.length, start + chunk); i++) idx.push(i);
      const logits = model.forward(idx.map((i) => pool.x[i]));
      const n = logits.data.length / logits.shape[logits.shape.length - 1];
      nll += crossEntropy(logits, idx.map((i) => pool.y[i])).item() * n;
      if (pool.teacher) agree += top1Agreement(logits, takeRows(pool.teacher, idx)) * n;
      positions += n;
    }
  });
  return { loss: nll / positions, agreement: pool.teacher ? agree / positions : null };
}
