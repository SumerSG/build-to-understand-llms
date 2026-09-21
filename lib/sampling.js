// lib/sampling.js — the decoding pipeline: logit processors plus the sampler.
// This file is the REFERENCE SOLUTION for module 14 (Decoding & sampling).
//
// A language model does not output a token, it outputs a distribution over every token. Decoding is a
// separate, controllable policy on top of that distribution, and each knob here is a different way of
// cutting off its tail. Every function takes logits (Float32Array or number[]), returns a NEW
// Float32Array, and never mutates its input, so processors compose in a pipeline.
//
// The pipeline order used by sample() is temperature -> top-k -> top-p -> min-p. Temperature comes first
// because it reshapes the probabilities the truncation rules then measure; apply it after top-p and the
// same `p` would keep a different set of tokens.

import { argmaxArray, sampleIndex } from './util.js';
import { newCache, prefill, forwardStep } from './infer.js';

/** Copy any logits container into a fresh Float32Array (the shape every processor returns). */
function copyLogits(logits) {
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = logits[i];
  return out;
}

/** Sharpen (t < 1) or flatten (t > 1) the distribution. t <= 0 collapses it to the argmax. */
export function applyTemperature(logits, t) {
  if (t === 1) return copyLogits(logits);
  const out = new Float32Array(logits.length);
  if (t <= 0) {
    // The limit of t -> 0 is greedy decoding; dividing by zero would give NaN, so take the limit directly.
    out.fill(-Infinity);
    out[argmaxArray(logits)] = 0;
    return out;
  }
  for (let i = 0; i < logits.length; i++) out[i] = logits[i] / t;
  return out;
}

/** Indices of `logits` sorted by value, largest first; ties keep their original order. */
function indicesByLogitDesc(logits) {
  const order = new Array(logits.length);
  for (let i = 0; i < order.length; i++) order[i] = i;
  order.sort((a, b) => (logits[b] - logits[a]) || (a - b));
  return order;
}

/** Keep only the k largest logits; the rest become -Infinity. k <= 0 (or k >= vocab) means no filter. */
export function topKFilter(logits, k) {
  if (k <= 0 || k >= logits.length) return copyLogits(logits);
  const order = indicesByLogitDesc(logits);
  const out = new Float32Array(logits.length);
  out.fill(-Infinity);
  for (let i = 0; i < k; i++) out[order[i]] = logits[order[i]];
  return out;
}

/**
 * Nucleus sampling: sort by probability, then keep the smallest prefix whose mass reaches p. Flat
 * distributions keep many tokens, peaked ones keep few — that is the point of top-p over top-k.
 */
export function topPFilter(logits, p) {
  if (p >= 1 || p <= 0) return copyLogits(logits);
  const probs = softmaxLogits(logits);
  const order = indicesByLogitDesc(logits);
  const out = new Float32Array(logits.length);
  out.fill(-Infinity);
  let mass = 0;
  for (let i = 0; i < order.length; i++) {
    out[order[i]] = logits[order[i]];
    mass += probs[order[i]];
    if (mass >= p) break; // the token that crosses p is kept, so at least one always survives
  }
  return out;
}

/** Min-p: drop every token less than p times as likely as the most likely one. */
export function minPFilter(logits, p) {
  if (p <= 0) return copyLogits(logits);
  const probs = softmaxLogits(logits);
  let maxProb = 0;
  for (let i = 0; i < probs.length; i++) if (probs[i] > maxProb) maxProb = probs[i];
  const floor = p * maxProb;
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = probs[i] < floor ? -Infinity : logits[i];
  return out; // the argmax always passes, so the distribution is never empty
}

/**
 * CTRL-style repetition penalty (Keskar et al. 2019): already-seen tokens get their positive logits
 * divided by `penalty` and their negative logits multiplied by it, which lowers both towards -Infinity.
 */
export function repetitionPenalty(logits, prevIds, penalty) {
  const out = copyLogits(logits);
  if (!prevIds || penalty === 1 || penalty <= 0) return out;
  const seen = new Set(prevIds);
  for (const id of seen) {
    if (id < 0 || id >= out.length) continue;
    out[id] = out[id] > 0 ? out[id] / penalty : out[id] * penalty;
  }
  return out;
}

/** Numerically stable softmax over logits; an all -Infinity input falls back to uniform. */
export function softmaxLogits(logits) {
  const out = new Float32Array(logits.length);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  if (!Number.isFinite(max)) {
    out.fill(1 / logits.length);
    return out;
  }
  let z = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    out[i] = e;
    z += e;
  }
  for (let i = 0; i < out.length; i++) out[i] /= z;
  return out;
}

/** Run the processors in pipeline order and draw one token id with the seeded rng `next`. */
export function sample(logits, { temperature = 1, topK = 0, topP = 1, minP = 0, next } = {}) {
  let processed = applyTemperature(logits, temperature);
  processed = topKFilter(processed, topK);
  processed = topPFilter(processed, topP);
  processed = minPFilter(processed, minP);
  const probs = softmaxLogits(processed);
  return sampleIndex(probs, next());
}

/** The token with the largest logit: sampling with temperature 0. */
export function greedy(logits) {
  return argmaxArray(logits);
}

/**
 * Generate text from an inference model (lib/infer.js loadModel) and a tokenizer: encode the prompt,
 * prefill the KV cache with it, then sample one token at a time, feeding each sampled token back in.
 * Returns ONLY the generated text, decoded; the prompt is not repeated.
 *
 * Extra options beyond maxNewTokens/next/stopAtEos are handed straight to sample(), so temperature,
 * topK, topP and minP work here exactly as they do on a single step.
 */
export function generate(model, tokenizer, prompt, { maxNewTokens = 50, next, stopAtEos = true, ...samplingOpts } = {}) {
  if (typeof next !== 'function') throw new Error('generate: pass a seeded rng function as `next`');
  let ids = tokenizer.encode(prompt);
  // Nothing to condition on: start from the end-of-text token, the way a corpus document begins.
  if (ids.length === 0) {
    if (!(tokenizer.eos >= 0)) throw new Error('generate: empty prompt and the tokenizer has no eos token');
    ids = [tokenizer.eos];
  }
  // The context window is fixed, so a prompt longer than it keeps only its newest tokens.
  if (ids.length > model.config.blockSize) ids = ids.slice(ids.length - model.config.blockSize);
  const cache = newCache(model);
  let logits = prefill(model, cache, ids);

  const generated = [];
  for (let step = 0; step < maxNewTokens; step++) {
    const id = sample(logits, { ...samplingOpts, next });
    if (stopAtEos && id === tokenizer.eos) break;
    generated.push(id);
    // The context window is the hard limit: past blockSize there is no position embedding left to use.
    if (cache.length >= model.config.blockSize) break;
    logits = forwardStep(model, cache, id);
  }
  return tokenizer.decode(generated);
}
