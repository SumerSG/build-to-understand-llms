// Module 15 — The KV cache: incremental decoding on raw tensors.
// A "model" here is { config, w } from loadModel() in lib/infer.js: config is the GPT config and w maps
// parameter names ('wte.weight', 'blocks.0.attn.qkv.weight', …) to raw tensors { shape, data }.
// Everything runs forward-only on lib/ops.js kernels: no autograd graph, no gradients.
//
// The worked examples (linear … forward) are complete. forward() is the uncached reference: it
// recomputes every position every time. Your job, step by step, is the cached path that gives the
// same logits from one token of work per token.

import * as ops from 'lib/ops.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/** y = x · W + b for raw tensors: the one pattern every projection in the model uses. */
export function linear(x, weight, bias) {
  return ops.add(ops.matmul(x, weight), bias);
}

/** [T, C] -> [H, T, dh]: view the channels as H heads, then put the head axis in front of time. */
export function splitHeads(x, time, nHead, headDim) {
  return ops.permute(ops.reshape(x, [time, nHead, headDim]), [1, 0, 2]);
}

/** [H, T, dh] -> [T, C]: the exact inverse of splitHeads. */
export function mergeHeads(x, time, nEmbd) {
  return ops.reshape(ops.permute(x, [1, 0, 2]), [time, nEmbd]);
}

/** The per-position MLP of block `prefix`: expand to 4C, GELU, project back to C. */
export function mlpForward(w, prefix, x) {
  const hidden = ops.gelu(linear(x, w[`${prefix}.mlp.fc.weight`], w[`${prefix}.mlp.fc.bias`]));
  return linear(hidden, w[`${prefix}.mlp.proj.weight`], w[`${prefix}.mlp.proj.bias`]);
}

/** Token embedding + learned position embedding for `ids` placed at positions from … from + ids.length − 1. */
export function embedTokens(w, ids, from) {
  const positions = ids.map((_, i) => from + i);
  return ops.add(ops.embed(w['wte.weight'], ids), ops.embed(w['wpe.weight'], positions));
}

/** The final LayerNorm and the tied head: [T, C] -> logits [T, V]. */
export function head(w, x) {
  const normalized = ops.layerNorm(x, w['lnF.gamma'], w['lnF.beta']);
  return ops.matmul(normalized, ops.transpose(w['wte.weight']));
}

/** Project a normalised [T, C] activation of block `prefix` to q, k, v, each [H, T, dh]. */
export function projectQKV(w, prefix, normed, time, nHead, headDim) {
  const nEmbd = nHead * headDim;
  const qkv = linear(normed, w[`${prefix}.attn.qkv.weight`], w[`${prefix}.attn.qkv.bias`]);
  return {
    q: splitHeads(ops.slice(qkv, 1, 0, nEmbd), time, nHead, headDim),
    k: splitHeads(ops.slice(qkv, 1, nEmbd, 2 * nEmbd), time, nHead, headDim),
    v: splitHeads(ops.slice(qkv, 1, 2 * nEmbd, 3 * nEmbd), time, nHead, headDim),
  };
}

/**
 * Full recompute, no cache: logits [T, V] for every position of `ids`. Position t attends to
 * positions 0…t through the causal mask. This is the reference the cached path must reproduce.
 */
export function forward(model, ids) {
  const { nLayer, nHead, nEmbd, blockSize } = model.config;
  const w = model.w;
  const time = ids.length;
  if (time === 0) throw new Error('forward: need at least one token');
  if (time > blockSize) throw new Error(`forward: sequence of ${time} exceeds blockSize ${blockSize}`);
  const headDim = nEmbd / nHead;
  const scale = 1 / Math.sqrt(headDim);
  const mask = ops.causalMask(time);

  let x = embedTokens(w, ids, 0); // [T, C]
  for (let layer = 0; layer < nLayer; layer++) {
    const prefix = `blocks.${layer}`;
    const normed = ops.layerNorm(x, w[`${prefix}.ln1.gamma`], w[`${prefix}.ln1.beta`]);
    const { q, k, v } = projectQKV(w, prefix, normed, time, nHead, headDim); // each [H, T, dh]

    const scores = ops.scale(ops.matmul(q, ops.transpose(k)), scale); // [H, T, T]
    const weights = ops.softmax(ops.maskedFill(scores, mask, -Infinity)); // future keys get weight 0
    const attended = mergeHeads(ops.matmul(weights, v), time, nEmbd); // [T, C]
    x = ops.add(x, linear(attended, w[`${prefix}.attn.proj.weight`], w[`${prefix}.attn.proj.bias`]));

    const normed2 = ops.layerNorm(x, w[`${prefix}.ln2.gamma`], w[`${prefix}.ln2.beta`]);
    x = ops.add(x, mlpForward(w, prefix, normed2));
  }
  return head(w, x);
}

// ---------- step 1: one query against stored keys and values ----------

/**
 * Attention for the newest position only. q is [H, 1, dh]; k and v are [H, t, dh] and hold every
 * position up to and including the current one. Returns [H, 1, dh]. No mask: everything stored is
 * in the past by construction.
 */
export function attendOne(q, k, v) {
  const headDim = q.shape[q.shape.length - 1];
  // TODO: step 1 — scores [H, 1, t] = q · kᵀ scaled by 1/sqrt(headDim); softmax over the stored
  // positions (last axis); return the weighted sum of v, shape [H, 1, dh]. Compare with forward() above.
  return null;
}

// ---------- step 2: the cache ----------

/** An empty cache: per layer a [H, 0, dh] key block and value block, plus how many tokens are stored. */
export function newCache(model) {
  // TODO: step 2 — return { k, v, length: 0, maxLength: blockSize } with one ops.zeros([nHead, 0, headDim])
  // per layer in k and in v.
  return null;
}

/**
 * Append the [H, n, dh] keys and values of the newest position(s) to layer `layer`, along the time
 * axis. Returns the layer's full { k, v } after the append (each [H, t + n, dh]). Does not touch
 * cache.length: that advances once per token, after every layer has been appended to.
 */
export function appendKV(cache, layer, k, v) {
  // TODO: step 2 — concatenate along the time axis (axis 1) into cache.k[layer] and cache.v[layer];
  // return { k: cache.k[layer], v: cache.v[layer] }.
  return null;
}

// ---------- step 3: one decode step ----------

/**
 * Decode one token: embed `id` at position cache.length, and in every layer project q, k, v for this
 * one position, append k and v, attend over the whole cache, then the MLP. Returns the logits
 * Float32Array [V] for the NEXT token and advances cache.length by one.
 */
export function forwardStep(model, cache, id) {
  const { nLayer, nHead, nEmbd } = model.config;
  const w = model.w;
  const headDim = nEmbd / nHead;
  const position = cache.length;
  // TODO: step 3 — throw if the cache is full; x = embedTokens(w, [id], position); per layer: ln1,
  // projectQKV(…, 1, nHead, headDim), appendKV, attendOne over the stored k/v, mergeHeads, attn.proj
  // residual, ln2, MLP residual; then cache.length = position + 1 and return head(w, x).data.
  return null;
}

// ---------- step 4: prefill and generation, cached and uncached ----------

/** Run a whole prompt through the cache; return the logits [V] after its last token. */
export function prefill(model, cache, ids) {
  // TODO: step 4 — throw on an empty prompt; run forwardStep for every id; return the last logits.
  return null;
}

/** Index of the largest logit (first on ties). Worked: use it in generateGreedy. */
export function argmaxOf(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  return best;
}

/**
 * Greedy decoding of `maxNewTokens` tokens after `promptIds`, returning only the new ids.
 * cached: prefill once, then one forwardStep per token. Uncached: re-run forward over the whole
 * sequence every step and read the last row. Both must produce the same ids.
 */
export function generateGreedy(model, promptIds, maxNewTokens, { cached = true } = {}) {
  const out = [];
  // TODO: step 4 — cached: prefill once, then argmax → push → forwardStep (skip the step after the last
  // token). Uncached: forward(model, promptIds + out) every step and take the last row's argmax.
  return out;
}

// ---------- step 5: the cost model ----------

/** Trainable scalars of a GPT config; equals new GPT(config).numParams() from lib/gpt.js. */
export function paramCount(config) {
  // TODO: step 5 — V·C + T·C (embeddings) + L·(12C² + 13C) (blocks) + 2C (final LayerNorm).
  return 0;
}

/**
 * FLOPs to produce one more token when `contextLen` tokens (including the new one) are in context.
 * Matmuls: 2 per parameter. Attention: 4·L·T·C (scores 2·L·T·C, weighted values 2·L·T·C). Without a
 * cache the whole context is recomputed, so the total is multiplied by T.
 */
export function flopsPerToken(config, contextLen, { cached = true } = {}) {
  // TODO: step 5
  return 0;
}

/**
 * Bytes the KV cache occupies for one sequence of `contextLen` tokens:
 * 2 (K and V) · nLayer · nKVHead · headDim · contextLen · bytesPerElement.
 * nKVHead defaults to nHead (multi-head attention); GQA models set it lower.
 */
export function cacheBytes(config, contextLen, { bytesPerElement = 4 } = {}) {
  // TODO: step 5 — config.nKVHead defaults to nHead; headDim = nEmbd / nHead (or config.headDim).
  return 0;
}
