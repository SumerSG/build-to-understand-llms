// Module 11 — Reward models & DPO.
// A preference "A beats B" becomes a training signal in two ways. The Bradley–Terry loss turns a score
// difference into a logistic loss; a reward model learns those scores from the GPT's final hidden state.
// DPO skips the reward model: the policy's own log-ratio against a frozen reference IS the score.
//
// Conventions (same as module 10): token ids are plain JS arrays, batches are number[][] (B×T) with a
// mask that follows y, all randomness comes through a `next` function from rng(seed), and functions
// return new values rather than mutating inputs (except optimizer steps, which update parameters in place).

import { GPT, Linear } from 'lib/gpt.js';
import { Tensor, noGrad } from 'lib/tensor.js';
import { AdamW, clipGradNorm } from 'lib/optim.js';
import { randInt } from 'lib/util.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/**
 * Anything score-like (number, number[], Float32Array or Tensor) as a 1-D Tensor without a gradient.
 * The loss functions accept plain arrays (hand tests) and Tensors (training) alike because of this.
 */
export function asTensor(x) {
  if (x instanceof Tensor) return x;
  if (typeof x === 'number') return Tensor.from([x]);
  return new Tensor({ shape: [x.length], data: Float32Array.from(x) });
}

/**
 * prompt + '\n' + response + eos as ids, with mask 1 on the response tokens (and the eos) only.
 * This is module 10's tokenizeExample with a plain-text template: the checkpoint has no chat markers,
 * and the preference signal only needs to know which tokens belong to the response.
 */
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

/**
 * Right-pad examples to a common length with padId (mask 0), so one forward pass can hold all of them.
 * `lengths[b]` is the number of real tokens in x[b]; the causal mask means padding never influences them.
 */
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

/** Linear(C, 1): one scalar reward per sequence from its last hidden state (step 4 feeds it). */
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

/**
 * The frozen reference's log-probs for every pair, computed once under noGrad (they never change, so
 * production trainers precompute them too). Depends on your sequenceLogProbs from step 2.
 */
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

/**
 * Per-pair log-ratios log π − log ref, implicit reward margins and reward accuracy of `policy` against
 * the reference (used by the goal demo before and after training). Depends on steps 1 and 2.
 */
export function evaluatePairs(policy, pairs, refLogps, { beta = 0.1, batchSize = 8, padId = 0 } = {}) {
  const { chosen, rejected } = referenceLogProbs(policy, pairs, { batchSize, padId }); // policy log-probs, no graph
  const chosenRatio = chosen.map((lp, i) => lp - refLogps.chosen[i]);
  const rejectedRatio = rejected.map((lp, i) => lp - refLogps.rejected[i]);
  const margins = chosenRatio.map((c, i) => beta * (c - rejectedRatio[i]));
  const meanMargin = margins.reduce((a, b) => a + b, 0) / (margins.length || 1);
  return { chosenRatio, rejectedRatio, margins, meanMargin, accuracy: rewardAccuracy(chosenRatio, rejectedRatio) };
}

// ---------- step 1: the Bradley–Terry loss ----------

/**
 * Mean over pairs of −log σ(rChosen − rRejected). Inputs are 1-D Tensors or plain arrays of N scores
 * (use asTensor). Returns a scalar Tensor. Must stay finite for margins like −200 and have the exact
 * gradient −(1 − σ(m))/N at m = 0 (DPO starts every pair at margin 0).
 */
export function bradleyTerryLoss(rChosen, rRejected) {
  // TODO: step 1
  return asTensor(0);
}

/** Fraction of pairs whose chosen score is strictly greater than the rejected score (0 for no pairs). */
export function rewardAccuracy(rChosen, rRejected) {
  // TODO: step 1
  return 0;
}

// ---------- step 2: the log-probability of a response ----------

/**
 * batch = { x, y, mask } (B×T each, see padBatch). Returns a Tensor [B] whose entry b is
 * Σ_t mask[b][t] · log softmax(logits[b][t])[y[b][t]]: the log-probability of the response given the prompt.
 * Build it from Tensor ops so backward() reaches the model.
 */
export function sequenceLogProbs(model, batch) {
  // TODO: step 2
  return null;
}

// ---------- step 3: the DPO loss ----------

/** β · (log π − log ref), elementwise over a batch: the implicit reward. Returns a Tensor [N]. */
export function implicitRewards(logpPolicy, logpRef, beta) {
  // TODO: step 3
  return null;
}

/**
 * −log σ(β[(π_c − ref_c) − (π_r − ref_r)]) averaged over the batch. policyChosen/policyRejected are
 * Tensors [N] from sequenceLogProbs (on the graph); refChosen/refRejected are constants (arrays or Tensors).
 */
export function dpoLoss(policyChosen, policyRejected, refChosen, refRejected, beta) {
  // TODO: step 3
  return null;
}

// ---------- step 4: a reward model on the final hidden state ----------

/**
 * The GPT forward pass WITHOUT the LM head: token + position embeddings, every block, the final
 * LayerNorm. x is number[][] (B×T). Returns a Tensor [B, T, C]. model.wte, model.wpe, model.blocks
 * and model.lnF are all public (lib/gpt.js).
 */
export function hiddenStates(model, x) {
  // TODO: step 4
  return null;
}

/**
 * h [B, T, C] and lengths [B] -> Tensor [B, C]: row lengths[b] − 1 of each sequence (its last real
 * token). Throw if a length is outside 1..T. Gradient must flow back to exactly that row.
 */
export function lastTokenHidden(h, lengths) {
  // TODO: step 4
  return null;
}

/**
 * Full-batch training of a RewardHead with the Bradley–Terry loss. features is an array of
 * { chosen: Float32Array [C], rejected: Float32Array [C] } (last hidden states, precomputed under noGrad).
 * Returns [{ loss, accuracy }] with one record per step.
 */
export function trainRewardHead(head, features, { steps, lr = 1e-2, weightDecay = 0 } = {}) {
  // TODO: step 4
  return [];
}

// ---------- step 5: the DPO training loop ----------

/**
 * One DPO update. batch = { chosen, rejected (padded batches), refChosen, refRejected (number[]) }.
 * Forward both responses through the policy, DPO loss, backward, clipGradNorm, optimizer step, zeroGrad.
 * Returns { loss, margin, accuracy, gradNorm } where margin and accuracy are measured on this step's
 * forward pass (before the update), from the implicit rewards.
 */
export function dpoStep(policy, optimizer, batch, { beta = 0.1, maxGradNorm = 1.0 } = {}) {
  // TODO: step 5
  return null;
}

/**
 * The loop: AdamW over the policy; each step draws batchSize pair indices with randInt(next, pairs.length),
 * pads the chosen and rejected examples separately, looks up their reference log-probs, and calls dpoStep.
 * Returns the array of step records; awaits onStep(step, record) when given.
 */
export async function trainDPO(policy, pairs, refLogps, { steps, beta = 0.1, lr = 1e-4, batchSize = 2, maxGradNorm = 1.0, weightDecay = 0, padId = 0, next, onStep = null }) {
  // TODO: step 5
  return [];
}
