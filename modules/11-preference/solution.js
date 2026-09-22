// Module 11 — reference solution: the Bradley–Terry loss, sequence log-probs, the DPO loss, a reward-model
// head on the GPT's final hidden state, and the DPO training loop against a frozen reference.

import { GPT, Linear } from 'lib/gpt.js';
import { Tensor, noGrad } from 'lib/tensor.js';
import { AdamW, clipGradNorm } from 'lib/optim.js';
import { randInt } from 'lib/util.js';

// ---------- shared helpers (worked examples in the starter) ----------

/** Anything score-like (number, number[], Float32Array or Tensor) as a 1-D Tensor without a gradient. */
export function asTensor(x) {
  if (x instanceof Tensor) return x;
  if (typeof x === 'number') return Tensor.from([x]);
  return new Tensor({ shape: [x.length], data: Float32Array.from(x) });
}

/** prompt + '\n' + response + eos as ids, with mask 1 on the response tokens (and the eos) only. */
export function tokenizePair(tokenizer, prompt, response) {
  const promptIds = tokenizer.encode(prompt + '\n');
  const responseIds = tokenizer.encode(response).concat([tokenizer.eos]);
  const ids = promptIds.concat(responseIds);
  const mask = promptIds.map(() => 0).concat(responseIds.map(() => 1));
  return { ids, mask };
}

/** Token stream -> (x, y, mask): y is x shifted left by one and the mask follows y (module 10). */
export function shift({ ids, mask }) {
  return { x: ids.slice(0, -1), y: ids.slice(1), mask: mask.slice(1) };
}

/** One preference pair as two training examples that share a prompt. */
export function buildPair(tokenizer, { prompt, chosen, rejected }) {
  return { chosen: shift(tokenizePair(tokenizer, prompt, chosen)), rejected: shift(tokenizePair(tokenizer, prompt, rejected)) };
}

/** Right-pad examples to a common length with padId (mask 0), so one forward pass can hold all of them. */
export function padBatch(examples, padId = 0) {
  const T = Math.max(...examples.map((e) => e.x.length));
  const pad = (arr, v) => arr.concat(new Array(T - arr.length).fill(v));
  return {
    x: examples.map((e) => pad(e.x, padId)),
    y: examples.map((e) => pad(e.y, padId)),
    mask: examples.map((e) => pad(e.mask, 0)),
    lengths: examples.map((e) => e.x.length),
  };
}

/** A bit-identical copy of a GPT with its own parameter tensors: the frozen reference. */
export function cloneModel(model) {
  const copy = new GPT(model.config);
  const src = model.parameters(), dst = copy.parameters();
  for (let k = 0; k < src.length; k++) dst[k].data.set(src[k].data);
  return copy;
}

// ---------- step 1: the Bradley–Terry loss ----------

export function bradleyTerryLoss(rChosen, rRejected) {
  const margin = asTensor(rChosen).sub(asTensor(rRejected)); // [N]
  const N = margin.shape[0];
  // log σ(m) = m − logsumexp(m, 0) = logSoftmax([m, 0])[0]: stable for any m, exact gradient at m = 0.
  const pair = margin.reshape([N, 1]).mul(Tensor.from([[1, 0]])); // [N, 2] rows are [m, 0]
  const logSigmoid = pair.logSoftmax().slice(1, 0, 1); // [N, 1]
  return logSigmoid.neg().mean();
}

export function rewardAccuracy(rChosen, rRejected) {
  const c = asTensor(rChosen).data, r = asTensor(rRejected).data;
  if (c.length !== r.length) throw new Error(`rewardAccuracy: ${c.length} chosen scores vs ${r.length} rejected`);
  let wins = 0;
  for (let i = 0; i < c.length; i++) if (c[i] > r[i]) wins++;
  return c.length ? wins / c.length : 0;
}

// ---------- step 2: the log-probability of a response ----------

export function sequenceLogProbs(model, batch) {
  const logits = model.forward(batch.x); // [B, T, V]
  const [B, T, V] = logits.shape;
  // pick[b, t, v] = mask[b][t] if v === y[b][t] else 0; sum(logp * pick) is then the masked sum of target log-probs.
  const pick = new Float32Array(B * T * V);
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const target = batch.y[b][t];
      if (!(target >= 0 && target < V)) throw new Error(`sequenceLogProbs: target ${target} out of range for V=${V}`);
      pick[(b * T + t) * V + target] = batch.mask[b][t];
    }
  }
  const logProbs = logits.logSoftmax();
  return logProbs.mul(new Tensor({ shape: [B, T, V], data: pick })).reshape([B, T * V]).sum(1); // [B]
}

// ---------- step 3: the DPO loss ----------

export function implicitRewards(logpPolicy, logpRef, beta) {
  return asTensor(logpPolicy).sub(asTensor(logpRef)).scale(beta);
}

export function dpoLoss(policyChosen, policyRejected, refChosen, refRejected, beta) {
  const rChosen = implicitRewards(policyChosen, refChosen, beta);
  const rRejected = implicitRewards(policyRejected, refRejected, beta);
  return bradleyTerryLoss(rChosen, rRejected);
}

// ---------- step 4: a reward model on the final hidden state ----------

export function hiddenStates(model, x) {
  const T = x[0].length;
  const positions = [];
  for (let t = 0; t < T; t++) positions.push(t);
  let h = model.wte.forward(x).add(model.wpe.forward(positions));
  for (const block of model.blocks) h = block.forward(h);
  return model.lnF.forward(h); // [B, T, C]: what the LM head would read
}

export function lastTokenHidden(h, lengths) {
  const [B, T, C] = h.shape;
  const select = new Float32Array(B * T);
  for (let b = 0; b < B; b++) {
    const n = lengths[b];
    if (!(n >= 1 && n <= T)) throw new Error(`lastTokenHidden: length ${n} is outside 1..${T}`);
    select[b * T + n - 1] = 1;
  }
  return h.mul(new Tensor({ shape: [B, T, 1], data: select })).sum(1); // [B, C]
}

/** Linear(C, 1): one scalar reward per sequence from its last hidden state. */
export class RewardHead {
  constructor(nEmbd, { next } = {}) {
    this.proj = new Linear(nEmbd, 1, { next });
  }

  /** hLast [B, C] -> rewards [B]. */
  forward(hLast) {
    return this.proj.forward(hLast).reshape([-1]);
  }

  parameters() {
    return this.proj.parameters();
  }
}

export function trainRewardHead(head, features, { steps, lr = 1e-2, weightDecay = 0 } = {}) {
  const N = features.length;
  const C = features[0].chosen.length;
  const stack = (key) => {
    const data = new Float32Array(N * C);
    for (let i = 0; i < N; i++) data.set(features[i][key], i * C);
    return new Tensor({ shape: [N, C], data });
  };
  const hChosen = stack('chosen'), hRejected = stack('rejected');
  const optimizer = new AdamW(head.parameters(), { lr, weightDecay });
  const history = [];
  for (let step = 0; step < steps; step++) {
    const rChosen = head.forward(hChosen), rRejected = head.forward(hRejected);
    const loss = bradleyTerryLoss(rChosen, rRejected);
    loss.backward();
    optimizer.step();
    optimizer.zeroGrad();
    history.push({ loss: loss.item(), accuracy: rewardAccuracy(rChosen, rRejected) });
  }
  return history;
}

// ---------- step 5: the DPO training loop ----------

/** The frozen reference's log-probs for every pair, computed once under noGrad (they never change). */
export function referenceLogProbs(reference, pairs, { batchSize = 8, padId = 0 } = {}) {
  const chosen = [], rejected = [];
  noGrad(() => {
    for (let start = 0; start < pairs.length; start += batchSize) {
      const chunk = pairs.slice(start, start + batchSize);
      chosen.push(...sequenceLogProbs(reference, padBatch(chunk.map((p) => p.chosen), padId)).data);
      rejected.push(...sequenceLogProbs(reference, padBatch(chunk.map((p) => p.rejected), padId)).data);
    }
  });
  return { chosen, rejected };
}

/** Per-pair log-ratios, implicit reward margins and reward accuracy of `policy` against the reference. */
export function evaluatePairs(policy, pairs, refLogps, { beta = 0.1, batchSize = 8, padId = 0 } = {}) {
  const { chosen, rejected } = referenceLogProbs(policy, pairs, { batchSize, padId }); // policy log-probs, no graph
  const chosenRatio = chosen.map((lp, i) => lp - refLogps.chosen[i]);
  const rejectedRatio = rejected.map((lp, i) => lp - refLogps.rejected[i]);
  const margins = chosenRatio.map((c, i) => beta * (c - rejectedRatio[i]));
  const meanMargin = margins.reduce((a, b) => a + b, 0) / (margins.length || 1);
  return { chosenRatio, rejectedRatio, margins, meanMargin, accuracy: rewardAccuracy(chosenRatio, rejectedRatio) };
}

export function dpoStep(policy, optimizer, batch, { beta = 0.1, maxGradNorm = 1.0 } = {}) {
  const policyChosen = sequenceLogProbs(policy, batch.chosen);
  const policyRejected = sequenceLogProbs(policy, batch.rejected);
  const loss = dpoLoss(policyChosen, policyRejected, batch.refChosen, batch.refRejected, beta);
  loss.backward();
  const gradNorm = clipGradNorm(policy.parameters(), maxGradNorm);
  optimizer.step();
  optimizer.zeroGrad();
  const { margin, accuracy } = noGrad(() => {
    const rChosen = implicitRewards(policyChosen, batch.refChosen, beta).data;
    const rRejected = implicitRewards(policyRejected, batch.refRejected, beta).data;
    let sum = 0;
    for (let i = 0; i < rChosen.length; i++) sum += rChosen[i] - rRejected[i];
    return { margin: sum / rChosen.length, accuracy: rewardAccuracy(rChosen, rRejected) };
  });
  return { loss: loss.item(), margin, accuracy, gradNorm };
}

export async function trainDPO(policy, pairs, refLogps, { steps, beta = 0.1, lr = 1e-4, batchSize = 2, maxGradNorm = 1.0, weightDecay = 0, padId = 0, next, onStep = null }) {
  if (typeof next !== 'function') throw new Error('trainDPO: pass a seeded rng function as `next`');
  const optimizer = new AdamW(policy.parameters(), { lr, betas: [0.9, 0.95], weightDecay });
  const history = [];
  for (let step = 0; step < steps; step++) {
    const idx = [];
    for (let b = 0; b < batchSize; b++) idx.push(randInt(next, pairs.length));
    const batch = {
      chosen: padBatch(idx.map((i) => pairs[i].chosen), padId),
      rejected: padBatch(idx.map((i) => pairs[i].rejected), padId),
      refChosen: idx.map((i) => refLogps.chosen[i]),
      refRejected: idx.map((i) => refLogps.rejected[i]),
    };
    const result = dpoStep(policy, optimizer, batch, { beta, maxGradNorm });
    history.push(result);
    if (onStep) await onStep(step, result);
  }
  return history;
}
