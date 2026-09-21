// Module 04 — reference solution: a count-based bigram model and a neural bigram model.
//
// Both models end up as the same object: a raw tensor { shape: [V, V], data: Float32Array } whose row i is
// the probability distribution over the token that follows token i. Everything downstream (perplexity,
// sampling) works on that table and does not care how it was produced.

import { randInt } from 'lib/util.js';
import { Tensor, crossEntropy, noGrad } from 'lib/tensor.js';
import { AdamW } from 'lib/optim.js';

// ---------- worked examples ----------

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

export function countBigrams(ids, V) {
  const data = new Float32Array(V * V);
  for (let t = 0; t + 1 < ids.length; t++) data[ids[t] * V + ids[t + 1]] += 1;
  return { shape: [V, V], data };
}

export function bigramProbs(counts, alpha = 0) {
  const V = counts.shape[0];
  const out = new Float32Array(V * V);
  for (let i = 0; i < V; i++) {
    let total = 0;
    for (let j = 0; j < V; j++) total += counts.data[i * V + j];
    const denom = total + alpha * V;
    for (let j = 0; j < V; j++) {
      out[i * V + j] = denom > 0 ? (counts.data[i * V + j] + alpha) / denom : 1 / V;
    }
  }
  return { shape: [V, V], data: out };
}

// ---------- step 2: evaluation ----------

export function negLogLikelihood(probs, ids) {
  const V = probs.shape[1];
  const n = ids.length - 1;
  let total = 0;
  for (let t = 0; t < n; t++) total -= Math.log(probs.data[ids[t] * V + ids[t + 1]]);
  return total / n;
}

export function perplexity(nll) {
  return Math.exp(nll);
}

// ---------- step 3: sampling ----------

export function sampleNext(probs, prev, next, temperature = 1) {
  const row = rowOf(probs, prev);
  const V = row.length;
  let p = row;
  if (temperature !== 1) {
    p = new Float32Array(V);
    let z = 0;
    for (let j = 0; j < V; j++) { p[j] = Math.pow(row[j], 1 / temperature); z += p[j]; }
    for (let j = 0; j < V; j++) p[j] /= z;
  }
  const u = next();
  let acc = 0;
  for (let j = 0; j < V; j++) {
    acc += p[j];
    if (u < acc) return j;
  }
  return V - 1;   // float rounding can leave the cumulative sum a hair below 1
}

export function generate(probs, start, n, next, temperature = 1) {
  const out = [];
  let prev = start;
  for (let i = 0; i < n; i++) {
    prev = sampleNext(probs, prev, next, temperature);
    out.push(prev);
  }
  return out;
}

// ---------- step 4: the neural bigram ----------

export function initNeural(V, next, std = 0.01) {
  return { V, W: Tensor.randn([V, V], next, std, { requiresGrad: true }) };
}

export function neuralLogits(model, xs) {
  return model.W.embed(xs);
}

export function neuralLoss(model, xs, ys) {
  return crossEntropy(neuralLogits(model, xs), ys);
}

// ---------- step 5: training and the comparison ----------

export function trainStep(model, opt, xs, ys) {
  opt.zeroGrad();
  const loss = neuralLoss(model, xs, ys);
  loss.backward();
  opt.step();
  return loss.item();
}

export function trainNeural(model, ids, { steps = 200, batchSize = 512, lr = 0.1, next } = {}) {
  const opt = new AdamW([model.W], { lr });
  const losses = [];
  for (let s = 0; s < steps; s++) {
    const { xs, ys } = makeBatch(ids, batchSize, next);
    losses.push(trainStep(model, opt, xs, ys));
  }
  return losses;
}

export function neuralProbs(model) {
  const p = noGrad(() => model.W.softmax());
  return { shape: p.shape.slice(), data: p.data };
}
