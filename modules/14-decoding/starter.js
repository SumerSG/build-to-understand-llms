// Module 14 — Decoding & sampling.
//
// The model gives you a vector of logits, one per token in the vocabulary. Decoding turns that vector
// into one token id. Everything below is a "logit processor": it takes logits (a Float32Array or a plain
// number[]), returns a NEW Float32Array of the same length, and never mutates its input. A dropped token
// is marked with -Infinity, which softmax turns into probability 0, so processors compose in a pipeline.
// Everything below the "worked examples" line is yours to implement.

import { argmaxArray } from 'lib/util.js';
import { newCache, prefill, forwardStep } from 'lib/infer.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/** Copy any logits container into a fresh Float32Array: the shape every processor returns. */
export function copyLogits(logits) {
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = logits[i];
  return out;
}

/**
 * Numerically stable softmax over logits (subtract the max first, as in module 01). A -Infinity entry
 * gets exp(-Infinity) = 0, so filtered-out tokens get probability 0 and the survivors are renormalised.
 * An all -Infinity input falls back to uniform rather than 0/0 = NaN.
 */
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

/** Indices of `logits` sorted by value, largest first; ties keep their original order. */
export function indicesByLogitDesc(logits) {
  const order = new Array(logits.length);
  for (let i = 0; i < order.length; i++) order[i] = i;
  order.sort((a, b) => (logits[b] - logits[a]) || (a - b));
  return order;
}

// ---------- step 1: temperature and greedy ----------

/**
 * Divide every logit by t. t < 1 sharpens the distribution, t > 1 flattens it, t = 1 is a copy.
 * t <= 0 means greedy: return -Infinity everywhere except 0 at the argmax (never divide by zero).
 */
export function applyTemperature(logits, t) {
  // TODO: step 1
  return copyLogits(logits);
}

/** The id of the largest logit: what sampling at temperature 0 always returns. */
export function greedy(logits) {
  // TODO: step 1
  return 0;
}

// ---------- step 2: top-k ----------

/** Keep only the k largest logits; set every other entry to -Infinity. k <= 0 or k >= length: no filter. */
export function topKFilter(logits, k) {
  // TODO: step 2
  return copyLogits(logits);
}

// ---------- step 3: top-p (nucleus) ----------

/**
 * Sort tokens by probability, walk down accumulating mass, keep the smallest prefix whose mass reaches p,
 * set the rest to -Infinity. The token that crosses p is kept, so at least one always survives.
 * p >= 1 or p <= 0: no filter.
 */
export function topPFilter(logits, p) {
  // TODO: step 3
  return copyLogits(logits);
}

// ---------- step 4: min-p and the penalties ----------

/** Drop every token whose probability is below p times the largest probability. p <= 0: no filter. */
export function minPFilter(logits, p) {
  // TODO: step 4
  return copyLogits(logits);
}

/**
 * CTRL-style repetition penalty: for each id in prevIds, divide its logit by `penalty` if the logit is
 * positive, multiply it by `penalty` if negative (both move it towards -Infinity). penalty = 1: no change.
 * prevIds may be null (no history): return a copy.
 */
export function repetitionPenalty(logits, prevIds, penalty) {
  // TODO: step 4
  return copyLogits(logits);
}

/**
 * OpenAI-style additive penalties: subtract `frequency` times the number of times an id appears in
 * prevIds, plus `presence` once if it appears at all. prevIds may be null (no history): return a copy.
 */
export function frequencyPresencePenalty(logits, prevIds, { frequency = 0, presence = 0 } = {}) {
  // TODO: step 4
  return copyLogits(logits);
}

// ---------- step 5: the pipeline and the sampler ----------

/**
 * Run the processors in order: repetitionPenalty, frequencyPresencePenalty, applyTemperature,
 * topKFilter, topPFilter, minPFilter. Every option has an "off" default.
 */
export function processLogits(logits, {
  prevIds = null, repetitionPenalty: rep = 1, frequencyPenalty = 0, presencePenalty = 0,
  temperature = 1, topK = 0, topP = 1, minP = 0,
} = {}) {
  // TODO: step 5
  return copyLogits(logits);
}

/**
 * Draw one token id: softmax the processed logits, take ONE uniform u = next(), and return the first
 * index whose cumulative probability exceeds u. `opts` are the processLogits options plus `next`.
 */
export function sample(logits, opts = {}) {
  // TODO: step 5
  return 0;
}

// ---------- step 6: generate ----------

/**
 * Autoregressive decoding with a KV cache (lib/infer.js): encode the prompt (empty prompt → [eos]),
 * keep only its newest blockSize tokens, prefill, then repeat: sample a token from the current logits
 * (passing prevIds = prompt + generated so the penalties can see them), stop on eos if stopAtEos, push
 * it, stop if any string in `stop` appears in the decoded text (cut the text there), stop at maxNewTokens
 * or when the cache is full, otherwise forwardStep the token to get the next logits.
 * Returns { text, ids, finishReason } with finishReason one of 'length' | 'stop' | 'eos'.
 */
export function generate(model, tokenizer, prompt, { maxNewTokens = 50, next, stop = [], stopAtEos = true, ...samplingOpts } = {}) {
  // TODO: step 6
  return { text: '', ids: [], finishReason: 'length' };
}
