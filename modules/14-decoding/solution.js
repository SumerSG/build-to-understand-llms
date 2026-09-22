// Module 14 — reference solution: the decoding pipeline (the same code lives in lib/sampling.js).
//
// A language model outputs a distribution over every token, not a token. Decoding is the separate,
// controllable policy that turns that distribution into one id, and every knob below is a different way
// of cutting off the distribution's tail. Every processor takes logits (Float32Array or number[]),
// returns a NEW Float32Array, and never mutates its input, so processors compose in a pipeline.

import { argmaxArray } from 'lib/util.js';
import { newCache, prefill, forwardStep } from 'lib/infer.js';

// ---------- worked examples ----------

/** Copy any logits container into a fresh Float32Array: the shape every processor returns. */
export function copyLogits(logits) {
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = logits[i];
  return out;
}

/** Numerically stable softmax over logits. -Infinity entries get probability 0; all -Infinity → uniform. */
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

export function applyTemperature(logits, t) {
  if (t === 1) return copyLogits(logits);
  const out = new Float32Array(logits.length);
  if (t <= 0) {
    // The limit t → 0 is greedy: dividing by zero would give NaN, so take the limit directly.
    out.fill(-Infinity);
    out[argmaxArray(logits)] = 0;
    return out;
  }
  for (let i = 0; i < logits.length; i++) out[i] = logits[i] / t;
  return out;
}

export function greedy(logits) {
  return argmaxArray(logits);
}

// ---------- step 2: top-k ----------

export function topKFilter(logits, k) {
  if (k <= 0 || k >= logits.length) return copyLogits(logits);
  const order = indicesByLogitDesc(logits);
  const out = new Float32Array(logits.length);
  out.fill(-Infinity);
  for (let i = 0; i < k; i++) out[order[i]] = logits[order[i]];
  return out;
}

// ---------- step 3: top-p (nucleus) ----------

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

// ---------- step 4: min-p and the penalties ----------

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

export function frequencyPresencePenalty(logits, prevIds, { frequency = 0, presence = 0 } = {}) {
  const out = copyLogits(logits);
  if (!prevIds || (frequency === 0 && presence === 0)) return out;
  const counts = new Map();
  for (const id of prevIds) counts.set(id, (counts.get(id) ?? 0) + 1);
  for (const [id, n] of counts) {
    if (id < 0 || id >= out.length) continue;
    out[id] -= frequency * n + presence;
  }
  return out;
}

// ---------- step 5: the pipeline and the sampler ----------

export function processLogits(logits, {
  prevIds = null, repetitionPenalty: rep = 1, frequencyPenalty = 0, presencePenalty = 0,
  temperature = 1, topK = 0, topP = 1, minP = 0,
} = {}) {
  let out = repetitionPenalty(logits, prevIds, rep);
  out = frequencyPresencePenalty(out, prevIds, { frequency: frequencyPenalty, presence: presencePenalty });
  out = applyTemperature(out, temperature);
  out = topKFilter(out, topK);
  out = topPFilter(out, topP);
  out = minPFilter(out, minP);
  return out;
}

export function sample(logits, opts = {}) {
  const { next } = opts;
  if (typeof next !== 'function') throw new Error('sample: pass a seeded rng function as `next`');
  const probs = softmaxLogits(processLogits(logits, opts));
  const u = next(); // exactly one draw per token: determinism depends on it
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (u < acc) return i;
  }
  // Float rounding can leave the cumulative sum a hair below 1: return the last token with mass.
  for (let i = probs.length - 1; i >= 0; i--) if (probs[i] > 0) return i;
  return probs.length - 1;
}

// ---------- step 6: generate ----------

export function generate(model, tokenizer, prompt, { maxNewTokens = 50, next, stop = [], stopAtEos = true, ...samplingOpts } = {}) {
  if (typeof next !== 'function') throw new Error('generate: pass a seeded rng function as `next`');
  let promptIds = tokenizer.encode(prompt);
  if (promptIds.length === 0) {
    if (!(tokenizer.eos >= 0)) throw new Error('generate: empty prompt and the tokenizer has no eos token');
    promptIds = [tokenizer.eos];
  }
  // The context window is fixed, so a prompt longer than it keeps only its newest tokens.
  const { blockSize } = model.config;
  if (promptIds.length > blockSize) promptIds = promptIds.slice(promptIds.length - blockSize);
  const cache = newCache(model);
  let logits = prefill(model, cache, promptIds);

  const ids = [];
  let text = '';
  let finishReason = 'length';
  while (ids.length < maxNewTokens) {
    const id = sample(logits, { ...samplingOpts, prevIds: promptIds.concat(ids), next });
    if (stopAtEos && id === tokenizer.eos) { finishReason = 'eos'; break; }
    ids.push(id);
    text = tokenizer.decode(ids);
    // A stop sequence can straddle token boundaries, so look for it in the decoded text, not in ids.
    let cut = -1;
    for (const s of stop) {
      const at = s.length ? text.indexOf(s) : -1;
      if (at >= 0 && (cut < 0 || at < cut)) cut = at;
    }
    if (cut >= 0) { text = text.slice(0, cut); finishReason = 'stop'; break; }
    if (ids.length >= maxNewTokens) break; // finishReason stays 'length'
    if (cache.length >= blockSize) break; // no position embedding left: the window is the hard limit
    logits = forwardStep(model, cache, id);
  }
  return { text, ids, finishReason };
}
