// Knowledge distillation.
// Conventions: logits are [B, T, V] (batch, time, vocabulary) or [N, V] (in shapes T is time; the argument
// named T in the loss functions is the temperature); a "raw" tensor is
// { shape, data: Float32Array }. The teacher is FROZEN: everything computed from it is a constant raw tensor
// produced inside noGrad, wrapped as a plain (non-grad) Tensor when it meets the student's graph.

import { GPT } from 'lib/gpt.js';
import { Tensor, noGrad, crossEntropy } from 'lib/tensor.js';
import { AdamW, clipGradNorm } from 'lib/optim.js';
import { generate } from 'lib/sampling.js';
import { getBatch } from 'lib/data.js';
import { randInt } from 'lib/util.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

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

/** Rows `idx` of a raw [N, T, V] tensor, as a raw [idx.length, T, V] (you will need it in step 6). */
function takeRows(t, idx) {
  const per = t.data.length / t.shape[0];
  const data = new Float32Array(idx.length * per);
  idx.forEach((i, k) => data.set(t.data.subarray(i * per, (i + 1) * per), k * per));
  return { shape: [idx.length, ...t.shape.slice(1)], data };
}

// ---------- step 1: temperature-softened targets ----------

/** softmax(z / T) along the last axis. Accepts a raw tensor, a Tensor, or a plain 1-D array of logits. */
export function softTargets(logits, T = 1) {
  // TODO: step 1 — throw for T <= 0.
  const data = logits.data ?? Float32Array.from(logits);
  const shape = logits.shape ? logits.shape.slice() : [data.length];
  const V = shape[shape.length - 1];
  const out = new Float32Array(data.length);
  for (let r = 0; r < data.length; r += V) {
    let max = -Infinity;
    for (let j = 0; j < V; j++) if (data[r + j] > max) max = data[r + j];
    // TODO: step 1 — fill out[r .. r + V) with exp((z − max) / T), then divide the row by its sum.
  }
  return { shape, data: out };
}

// ---------- step 2: the distillation loss ----------

/**
 * Forward KL from the teacher's softened distribution p to the student's q, scaled by T² and averaged
 * over positions:  T² · mean_positions Σ_v p(v) · (log p(v) − log q(v)).  Returns a scalar Tensor.
 * studentLogits is a Tensor (with grad); targetLogits, the teacher's logits, is a raw tensor or Tensor (a constant).
 */
export function klDistillLoss(studentLogits, targetLogits, T = 1) {
  // TODO: step 2
  return Tensor.zeros([]);
}

// ---------- step 3: the mixed objective ----------

/**
 * alpha · klDistillLoss(student, teacher, T) + (1 − alpha) · crossEntropy(student, targets) at T = 1.
 * With alpha = 0 the teacher may be null: that is plain training from the data.
 */
export function distillLoss(studentLogits, targetLogits, targets, { alpha = 0.5, T = 2 } = {}) {
  // TODO: step 3
  return Tensor.zeros([]);
}

// ---------- step 4: sequence-level distillation ----------

/**
 * Let the teacher (an InferModel from lib/infer.js loadModel) write the training set: for each prompt,
 * sample a continuation with lib/sampling.js generate, and append encode(prompt + continuation) and then
 * tokenizer.eos to one token stream. Returns { ids: number[], texts: string[] }.
 */
export function sequenceLevelCorpus(teacher, tokenizer, prompts, { maxNewTokens = 32, temperature = 1, next }) {
  // TODO: step 4
  return { ids: [], texts: [] };
}

// ---------- step 5: on-policy distillation ----------

/**
 * Reverse KL, KL(q_student ‖ p_teacher) = Σ_v q(v) · (log q(v) − log p(v)), averaged over the positions
 * where mask is 1 (mask: number[][] B×T, or null for every position). Returns a scalar Tensor.
 */
export function reverseKL(studentLogits, targetLogits, mask = null) {
  // TODO: step 5
  return Tensor.zeros([]);
}

/**
 * One on-policy batch: the STUDENT samples maxNewTokens after each prompt (student.generate), the frozen
 * teacher scores the samples, and the loss is reverseKL on the positions that predict a generated token.
 * Prompts must share one length (throw otherwise). Returns { loss, ids, mask }.
 */
export function onPolicyLoss(student, teacher, promptIds, { maxNewTokens, temperature = 1, next }) {
  // TODO: step 5
  return { loss: Tensor.zeros([]), ids: [], mask: [] };
}

// ---------- step 6: the experiment ----------

/** Fraction of positions where the student's argmax equals the teacher's argmax (first index on ties). */
export function top1Agreement(studentLogits, targetLogits) {
  // TODO: step 6
  return 0;
}

/**
 * Train `student` on a pool { x, y, teacher } for `steps` AdamW steps (lr, betas [0.9, 0.95], no weight
 * decay). Each step draws batchSize window indices with randInt(next, pool.x.length), computes distillLoss
 * with those windows' cached teacher logits (takeRows), clips the gradient norm to 1 and updates.
 * alpha = 0 (teacher may be null) is training from scratch. Awaits onStep(step, loss.item()) if given.
 * Returns the per-step losses (numbers).
 */
export async function distillTrain(student, pool, { steps, batchSize = 4, lr = 3e-3, alpha = 0.5, T = 2, next, onStep = null }) {
  // TODO: step 6
  return [];
}

/**
 * Held-out report on a pool: { loss: mean cross-entropy on the true next tokens over every position,
 * agreement: top1Agreement with pool.teacher over every position, or null when pool.teacher is null }.
 * Run it under noGrad, a few windows at a time.
 */
export function evaluate(model, pool, { chunk = 8 } = {}) {
  // TODO: step 6
  return { loss: 0, agreement: 0 };
}
