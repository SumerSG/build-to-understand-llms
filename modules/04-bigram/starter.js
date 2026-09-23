// From counting to learning: the bigram model.
//
// A bigram model is a table: row i holds P(next token | previous token = i). You will build that table two
// ways, by counting and by gradient descent, and everything downstream (perplexity, sampling) works on the
// table without caring which way it was made. Tables are raw tensors { shape: [V, V], data: Float32Array },
// row-major as in the tensors module: entry (i, j) lives at data[i * V + j].

import { randInt } from 'lib/util.js';
import { Tensor, crossEntropy, noGrad } from 'lib/tensor.js';
import { AdamW } from 'lib/optim.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/** A random minibatch of (previous token, next token) pairs drawn from `ids` with the seeded rng `next`. */
export function makeBatch(ids, batchSize, next) {
  const xs = new Array(batchSize);
  const ys = new Array(batchSize);
  for (let b = 0; b < batchSize; b++) {
    const t = randInt(next, ids.length - 1);   // t + 1 must stay inside the array
    xs[b] = ids[t];
    ys[b] = ids[t + 1];
  }
  return { xs, ys };
}

/** Row `i` of a [V, V] table as a Float32Array view (no copy): the distribution over what follows token i. */
export function rowOf(table, i) {
  const V = table.shape[1];
  return table.data.subarray(i * V, (i + 1) * V);
}

// ---------- step 1: counting ----------

/**
 * Count every adjacent pair (ids[t], ids[t+1]) into a [V, V] table: counts[i, j] is how often token j
 * followed token i. n ids contain n - 1 pairs.
 */
export function countBigrams(ids, V) {
  const data = new Float32Array(V * V);
  for (let t = 0; t + 1 < ids.length; t++) {
    // TODO: step 1 — add one to the entry for the pair (ids[t], ids[t + 1])
  }
  return { shape: [V, V], data };
}

/**
 * Turn counts into probabilities row by row with add-alpha smoothing:
 *   P[i, j] = (counts[i, j] + alpha) / (rowTotal[i] + alpha * V)
 * If the denominator is 0 (an unseen row with alpha = 0) fill that row with 1 / V.
 */
export function bigramProbs(counts, alpha = 0) {
  const V = counts.shape[0];
  const out = new Float32Array(V * V);
  for (let i = 0; i < V; i++) {
    let total = 0;
    for (let j = 0; j < V; j++) total += counts.data[i * V + j];
    // TODO: step 1 — fill out[i * V + j] for every j using total, alpha and V
  }
  return { shape: [V, V], data: out };
}

// ---------- step 2: evaluation ----------

/** Mean of -log P(ids[t+1] | ids[t]) over the n - 1 transitions in `ids`. */
export function negLogLikelihood(probs, ids) {
  // TODO: step 2
  return 0;
}

/** Perplexity is exp(NLL): the size of the fair die the model is effectively rolling. */
export function perplexity(nll) {
  // TODO: step 2
  return nll;
}

// ---------- step 3: sampling ----------

/**
 * Draw the next token given `prev`: take row `prev` of the table, apply the temperature
 * (p^(1/T), renormalised; T = 1 leaves the row alone), then invert the cumulative distribution with
 * one uniform draw u = next(). Return the smallest index whose cumulative probability exceeds u.
 */
export function sampleNext(probs, prev, next, temperature = 1) {
  // TODO: step 3
  return 0;
}

/** Sample n tokens in a chain starting from `start` (not included in the output). */
export function generate(probs, start, n, next, temperature = 1) {
  // TODO: step 3
  return [];
}

// ---------- step 4: the neural bigram ----------

/**
 * A neural bigram model is a [V, V] table of LOGITS, W, that autograd can train.
 * Return { V, W } where W is a Tensor with requiresGrad, initialised to small Gaussian noise (std 0.01).
 */
export function initNeural(V, next, std = 0.01) {
  // TODO: step 4
  return { V, W: Tensor.zeros([V, V]) };
}

/** Logits for each id in xs: row xs[i] of W, as a Tensor of shape [xs.length, V] attached to the graph. */
export function neuralLogits(model, xs) {
  // TODO: step 4
  return null;
}

/** Mean cross-entropy of the targets ys under the logits for xs (a scalar Tensor). */
export function neuralLoss(model, xs, ys) {
  // TODO: step 4
  return null;
}

// ---------- step 5: training and the comparison ----------

/** One optimiser step on one batch: zero the grads, compute the loss, backward, step. Return the loss as a number. */
export function trainStep(model, opt, xs, ys) {
  // TODO: step 5
  return NaN;
}

/** Train with AdamW on random minibatches from `ids` for `steps` steps; return the loss of every step. */
export function trainNeural(model, ids, { steps = 200, batchSize = 512, lr = 0.1, next } = {}) {
  // TODO: step 5
  return [];
}

/** The trained model as a probability table: softmax of every row of W, as a raw { shape, data } tensor. */
export function neuralProbs(model) {
  // TODO: step 5
  return null;
}
