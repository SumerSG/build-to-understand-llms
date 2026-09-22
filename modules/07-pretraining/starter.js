// Module 07 — The pre-training loop.
//
// You will build the loop that turns a randomly initialised GPT (lib/gpt.js, your module 06) into one
// that writes text like its corpus: sample a batch of token windows, predict every next token at once,
// take an AdamW step on the mean cross-entropy, repeat. The schedule, clipping, validation and sampling
// exist to keep that loop stable and to let you watch it work.
//
// Conventions: token ids are plain JS arrays; a batch is number[][] of shape B×T; every random draw
// comes from a `next` function made by rng(seed) so a run can be reproduced exactly.

import { crossEntropy, noGrad } from 'lib/tensor.js';
import { GPT } from 'lib/gpt.js';
import { rng, randInt } from 'lib/util.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/**
 * Contiguous train/validation split: the first `frac` of the ids train, the rest validate.
 * Contiguous (not shuffled) so that no validation window overlaps a training one.
 */
export function trainValSplit(ids, frac = 0.9) {
  const cut = Math.floor(ids.length * frac);
  return { train: ids.slice(0, cut), val: ids.slice(cut) };
}

/** Build the model from a config object. `seed` makes the initial weights reproducible. */
export function makeModel(config) {
  const { vocabSize, blockSize, nLayer, nHead, nEmbd, seed = 0 } = config;
  return new GPT({ vocabSize, blockSize, nLayer, nHead, nEmbd, seed });
}

// ---------- step 1: batches of shifted windows ----------

/**
 * Draw `batchSize` random windows of `blockSize` tokens from `ids`. x[b] is the window, y[b] is the
 * same window shifted one token to the right, so y[b][t] is the target for the prefix x[b][0..t].
 * Draw each start offset with randInt(next, n). Throw if ids is too short for one window plus its target.
 */
export function getBatch(ids, blockSize, batchSize, next) {
  // TODO: step 1
  return { x: [], y: [] };
}

// ---------- step 2: AdamW ----------

/**
 * Adam with decoupled weight decay (Loshchilov & Hutter 2019).
 * Per parameter tensor keep two buffers: m (running mean of the gradient) and v (running mean of
 * the squared gradient). Each step: decay p, update m and v, bias-correct them with 1 − beta^t,
 * then p −= lr · mHat / (sqrt(vHat) + eps). Skip parameters whose .grad is null.
 */
export class AdamW {
  constructor(params, { lr = 1e-3, betas = [0.9, 0.95], eps = 1e-8, weightDecay = 0 } = {}) {
    this.params = params;
    this.lr = lr;
    this.beta1 = betas[0];
    this.beta2 = betas[1];
    this.eps = eps;
    this.weightDecay = weightDecay;
    this.t = 0;
    // TODO: step 2 — allocate this.m and this.v, one Float32Array per parameter (same length as p.data)
  }

  /** One update for every parameter that has a gradient. */
  step() {
    // TODO: step 2
  }

  /** Clear every parameter's gradient so the next backward pass starts from zero. */
  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }
}

// ---------- step 3: gradient clipping and one training step ----------

/**
 * Global gradient-norm clipping. Compute the L2 norm over ALL gradients together; if it exceeds
 * `maxNorm`, scale every gradient by maxNorm / norm in place. Return the norm before clipping.
 */
export function clipGradNorm(params, maxNorm) {
  // TODO: step 3
  return 0;
}

/**
 * One optimisation step: zero grads, forward, mean cross-entropy over every position, backward,
 * clip, update. Return { loss, gradNorm }: the loss of the batch BEFORE the update (a number) and the
 * pre-clip gradient norm.
 */
export function trainStep(model, optimizer, x, y, { maxGradNorm = 1.0 } = {}) {
  // TODO: step 3
  return { loss: 0, gradNorm: 0 };
}

// ---------- step 4: the learning-rate schedule ----------

/**
 * Linear warmup from 0 to `peak` over `warmup` steps, then a cosine decay to `min` at `total`
 * (and `min` forever after). min defaults to a tenth of the peak.
 */
export function cosineWithWarmup(step, { warmup, total, peak, min = peak / 10 }) {
  // TODO: step 4
  return peak;
}

// ---------- step 5: evaluation ----------

/**
 * Mean cross-entropy over `evalBatches` random batches of `ids`, with no graph recorded. The model is
 * left untouched: no gradients, no updates. Return a plain number.
 */
export function estimateLoss(model, ids, { blockSize, batchSize, evalBatches = 4, next }) {
  // TODO: step 5
  return 0;
}

// ---------- step 6: the run ----------

/**
 * The full pre-training run. config:
 *   { trainIds, valIds, vocabSize, blockSize, nLayer, nHead, nEmbd, seed,
 *     steps, batchSize, lr, warmup, weightDecay, maxGradNorm, evalInterval, evalBatches }
 * Every step records { step, lr, loss, gradNorm } and, every `evalInterval` steps and on the last
 * step, `valLoss`. Await `onStep(record, model)` once per step so a caller can draw progress,
 * sample from the half-trained model, and let the browser breathe (it may be async). Resolve to { model, history, tokensSeen }.
 */
export async function train(config, onStep = null) {
  const {
    trainIds, valIds, blockSize, steps, batchSize, lr, warmup = 0,
    weightDecay = 0.1, maxGradNorm = 1.0, evalInterval = 50, evalBatches = 4, seed = 0,
  } = config;
  const model = makeModel(config);
  const next = rng(seed); // one stream for training batches and validation batches
  const history = [];
  // TODO: step 6
  return { model, history, tokensSeen: 0 };
}

/** Continue `prompt` by `maxNewTokens` tokens and return the whole text (prompt included). */
export function sample(model, tokenizer, prompt, { maxNewTokens = 100, temperature = 1, next }) {
  // TODO: step 6
  return prompt;
}
