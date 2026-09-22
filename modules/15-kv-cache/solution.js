// Module 15 — reference solution: incremental decoding with a KV cache on raw tensors.
// The same architecture as lib/gpt.js, run forward-only on lib/ops.js kernels (no autograd graph).
// lib/infer.js holds the vetted version of this file; module 14 and later import that one.

import * as ops from 'lib/ops.js';

// ---------- worked examples (shared with the starter) ----------

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
  const scores = ops.scale(ops.matmul(q, ops.transpose(k)), 1 / Math.sqrt(headDim)); // [H, 1, t]
  const weights = ops.softmax(scores);
  return ops.matmul(weights, v); // [H, 1, dh]
}

// ---------- step 2: the cache ----------

/** An empty cache: per layer a [H, 0, dh] key block and value block, plus how many tokens are stored. */
export function newCache(model) {
  const { nLayer, nHead, nEmbd, blockSize } = model.config;
  const headDim = nEmbd / nHead;
  const k = [];
  const v = [];
  for (let layer = 0; layer < nLayer; layer++) {
    k.push(ops.zeros([nHead, 0, headDim]));
    v.push(ops.zeros([nHead, 0, headDim]));
  }
  return { k, v, length: 0, maxLength: blockSize };
}

/**
 * Append the [H, n, dh] keys and values of the newest position(s) to layer `layer`, along the time
 * axis. Returns the layer's full { k, v } after the append (each [H, t + n, dh]). Does not touch
 * cache.length: that advances once per token, after every layer has been appended to.
 */
export function appendKV(cache, layer, k, v) {
  cache.k[layer] = ops.concat(cache.k[layer], k, 1);
  cache.v[layer] = ops.concat(cache.v[layer], v, 1);
  return { k: cache.k[layer], v: cache.v[layer] };
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
  if (cache.length >= cache.maxLength) throw new Error(`forwardStep: cache is full at maxLength ${cache.maxLength}`);
  const headDim = nEmbd / nHead;
  const position = cache.length;

  let x = embedTokens(w, [id], position); // [1, C]
  for (let layer = 0; layer < nLayer; layer++) {
    const prefix = `blocks.${layer}`;
    const normed = ops.layerNorm(x, w[`${prefix}.ln1.gamma`], w[`${prefix}.ln1.beta`]);
    const { q, k, v } = projectQKV(w, prefix, normed, 1, nHead, headDim); // each [H, 1, dh]
    const stored = appendKV(cache, layer, k, v); // [H, t + 1, dh]
    const attended = mergeHeads(attendOne(q, stored.k, stored.v), 1, nEmbd); // [1, C]
    x = ops.add(x, linear(attended, w[`${prefix}.attn.proj.weight`], w[`${prefix}.attn.proj.bias`]));

    const normed2 = ops.layerNorm(x, w[`${prefix}.ln2.gamma`], w[`${prefix}.ln2.beta`]);
    x = ops.add(x, mlpForward(w, prefix, normed2));
  }
  cache.length = position + 1;
  return head(w, x).data; // [V]
}

// ---------- step 4: prefill and generation, cached and uncached ----------

/** Run a whole prompt through the cache; return the logits [V] after its last token. */
export function prefill(model, cache, ids) {
  if (ids.length === 0) throw new Error('prefill: need at least one token');
  let logits = null;
  // A real engine prefills in one batched pass (it is compute-bound); token by token is the same result.
  for (const id of ids) logits = forwardStep(model, cache, id);
  return logits;
}

/** Index of the largest logit (first on ties). */
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
  if (cached) {
    const cache = newCache(model);
    let logits = prefill(model, cache, promptIds);
    for (let i = 0; i < maxNewTokens; i++) {
      const id = argmaxOf(logits);
      out.push(id);
      if (i === maxNewTokens - 1) break;
      logits = forwardStep(model, cache, id);
    }
  } else {
    const ids = promptIds.slice();
    for (let i = 0; i < maxNewTokens; i++) {
      const all = forward(model, ids); // [T, V]
      const last = ops.slice(all, 0, ids.length - 1, ids.length).data; // [V]
      const id = argmaxOf(last);
      out.push(id);
      ids.push(id);
    }
  }
  return out;
}

// ---------- step 5: the cost model ----------

/** Trainable scalars of a GPT config; equals new GPT(config).numParams() from lib/gpt.js. */
export function paramCount(config) {
  const { vocabSize, blockSize, nLayer, nEmbd } = config;
  const embeddings = vocabSize * nEmbd + blockSize * nEmbd;
  // Per block: 12C² of weights (qkv 3C², proj C², mlp fc 4C² + proj 4C²), 9C of biases, 4C of LayerNorm.
  const perBlock = 12 * nEmbd * nEmbd + 13 * nEmbd;
  return embeddings + nLayer * perBlock + 2 * nEmbd; // + the final LayerNorm
}

/**
 * FLOPs to produce one more token when `contextLen` tokens (including the new one) are in context.
 * Matmuls: 2 per parameter. Attention: 4·L·T·C (scores 2·L·T·C, weighted values 2·L·T·C). Without a
 * cache the whole context is recomputed, so the total is multiplied by T.
 */
export function flopsPerToken(config, contextLen, { cached = true } = {}) {
  const perToken = 2 * paramCount(config) + 4 * config.nLayer * contextLen * config.nEmbd;
  return cached ? perToken : perToken * contextLen;
}

/**
 * Bytes the KV cache occupies for one sequence of `contextLen` tokens:
 * 2 (K and V) · nLayer · nKVHead · headDim · contextLen · bytesPerElement.
 * nKVHead defaults to nHead (multi-head attention); GQA models set it lower.
 */
export function cacheBytes(config, contextLen, { bytesPerElement = 4 } = {}) {
  const { nLayer, nHead, nEmbd } = config;
  const nKVHead = config.nKVHead ?? nHead;
  const headDim = config.headDim ?? nEmbd / nHead;
  return 2 * nLayer * nKVHead * headDim * contextLen * bytesPerElement;
}
