// Module 07 — The pre-training loop (reference solution).
//
// Pre-training is one loop: draw a batch of token windows, predict every next token at once, take an
// AdamW step on the mean cross-entropy, repeat. Everything else in this file (the schedule, clipping,
// periodic validation, sampling) exists to keep that loop stable and to let you see what it is doing.

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
 * same window shifted one token to the left (y[b][t] = x[b][t+1]), so y[b][t] is the target for the prefix x[b][0..t].
 * All randomness comes from `next` (an rng function), so a seed reproduces the batch.
 */
export function getBatch(ids, blockSize, batchSize, next) {
  const lastStart = ids.length - blockSize - 1;
  if (lastStart < 0) throw new Error(`getBatch: need at least ${blockSize + 1} ids, got ${ids.length}`);
  const x = [];
  const y = [];
  for (let b = 0; b < batchSize; b++) {
    const start = randInt(next, lastStart + 1);
    x.push(ids.slice(start, start + blockSize));
    y.push(ids.slice(start + 1, start + blockSize + 1));
  }
  return { x, y };
}

// ---------- step 2: AdamW ----------

/**
 * Adam with decoupled weight decay (Loshchilov & Hutter 2019).
 * Per parameter tensor it keeps two buffers: m (running mean of the gradient) and v (running mean of
 * the squared gradient). The update divides m by sqrt(v), so every parameter moves by about `lr`
 * regardless of how large its gradient is.
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
    this.m = params.map((p) => new Float32Array(p.data.length));
    this.v = params.map((p) => new Float32Array(p.data.length));
  }

  /** One update for every parameter that has a gradient. */
  step() {
    this.t += 1;
    const c1 = 1 - Math.pow(this.beta1, this.t);
    const c2 = 1 - Math.pow(this.beta2, this.t);
    const decay = this.lr * this.weightDecay;
    for (let k = 0; k < this.params.length; k++) {
      const p = this.params[k];
      if (!p.grad) continue;
      const data = p.data, g = p.grad, m = this.m[k], v = this.v[k];
      for (let i = 0; i < data.length; i++) {
        // Decoupled: the decay shrinks p on its own and never enters m or v.
        if (decay !== 0) data[i] -= decay * data[i];
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g[i];
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g[i] * g[i];
        const mHat = m[i] / c1;
        const vHat = v[i] / c2;
        data[i] -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }

  /** Clear every parameter's gradient so the next backward pass starts from zero. */
  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }
}

// ---------- step 3: gradient clipping and one training step ----------

/**
 * Global gradient-norm clipping. Computes the L2 norm over ALL gradients together; if it exceeds
 * `maxNorm`, scales every gradient by maxNorm / norm in place. Returns the norm before clipping.
 */
export function clipGradNorm(params, maxNorm) {
  let sumSq = 0;
  for (const p of params) {
    if (!p.grad) continue;
    const g = p.grad;
    for (let i = 0; i < g.length; i++) sumSq += g[i] * g[i];
  }
  const totalNorm = Math.sqrt(sumSq);
  if (totalNorm > maxNorm) {
    const s = maxNorm / totalNorm;
    for (const p of params) {
      if (!p.grad) continue;
      const g = p.grad;
      for (let i = 0; i < g.length; i++) g[i] *= s;
    }
  }
  return totalNorm;
}

/**
 * One optimisation step: zero grads, forward, mean cross-entropy over every position, backward,
 * clip, update. Returns the loss of the batch BEFORE the update and the pre-clip gradient norm.
 */
export function trainStep(model, optimizer, x, y, { maxGradNorm = 1.0 } = {}) {
  optimizer.zeroGrad();
  const logits = model.forward(x); // [B,T,V]
  const loss = crossEntropy(logits, y); // scalar: mean over B*T predictions
  loss.backward();
  const gradNorm = clipGradNorm(model.parameters(), maxGradNorm);
  optimizer.step();
  return { loss: loss.item(), gradNorm };
}

// ---------- step 4: the learning-rate schedule ----------

/**
 * Linear warmup from 0 to `peak` over `warmup` steps, then a cosine decay to `min` at `total`
 * (and `min` forever after). min defaults to a tenth of the peak, as in GPT-3.
 */
export function cosineWithWarmup(step, { warmup, total, peak, min = peak / 10 }) {
  if (step < warmup) return (peak * step) / warmup;
  if (step >= total) return min;
  const progress = (step - warmup) / (total - warmup);
  return min + 0.5 * (peak - min) * (1 + Math.cos(Math.PI * progress));
}

// ---------- step 5: evaluation ----------

/**
 * Mean cross-entropy over `evalBatches` random batches of `ids`, with no graph recorded. The model is
 * left untouched: no gradients, no updates.
 */
export function estimateLoss(model, ids, { blockSize, batchSize, evalBatches = 4, next }) {
  return noGrad(() => {
    let total = 0;
    for (let k = 0; k < evalBatches; k++) {
      const { x, y } = getBatch(ids, blockSize, batchSize, next);
      total += crossEntropy(model.forward(x), y).item();
    }
    return total / evalBatches;
  });
}

// ---------- step 6: the run ----------

/**
 * The full pre-training run. config:
 *   { trainIds, valIds, vocabSize, blockSize, nLayer, nHead, nEmbd, seed,
 *     steps, batchSize, lr, warmup, weightDecay, maxGradNorm, evalInterval, evalBatches }
 * Every step records { step, lr, loss, gradNorm } and, every `evalInterval` steps and on the last
 * step, `valLoss`. `onStep(record, model)` is awaited once per step so a caller can draw progress,
 * sample from the half-trained model, and let the browser breathe (it may be async). Resolves to { model, history, tokensSeen }.
 */
export async function train(config, onStep = null) {
  const {
    trainIds, valIds, blockSize, steps, batchSize, lr, warmup = 0,
    weightDecay = 0.1, maxGradNorm = 1.0, evalInterval = 50, evalBatches = 4, seed = 0,
  } = config;
  const model = makeModel(config);
  const optimizer = new AdamW(model.parameters(), { lr, betas: [0.9, 0.95], weightDecay });
  const next = rng(seed); // one stream for training batches and validation batches
  const history = [];
  for (let step = 0; step < steps; step++) {
    optimizer.lr = cosineWithWarmup(step, { warmup, total: steps, peak: lr });
    const { x, y } = getBatch(trainIds, blockSize, batchSize, next);
    const { loss, gradNorm } = trainStep(model, optimizer, x, y, { maxGradNorm });
    const record = { step, lr: optimizer.lr, loss, gradNorm };
    if (valIds && ((step + 1) % evalInterval === 0 || step === steps - 1)) {
      record.valLoss = estimateLoss(model, valIds, { blockSize, batchSize, evalBatches, next });
    }
    history.push(record);
    if (onStep) await onStep(record, model);
  }
  return { model, history, tokensSeen: steps * batchSize * blockSize };
}

/** Continue `prompt` by `maxNewTokens` tokens and return the whole text (prompt included). */
export function sample(model, tokenizer, prompt, { maxNewTokens = 100, temperature = 1, next }) {
  const ids = tokenizer.encode(prompt);
  const out = model.generate(ids, { maxNewTokens, temperature, next });
  return tokenizer.decode(out);
}
