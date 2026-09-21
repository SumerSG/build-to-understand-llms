// lib/infer.js — forward-only inference on raw tensors, with a KV cache.
// This file is the REFERENCE SOLUTION for module 15 (The KV cache) and the base of the inference track.
//
// Training needs an autograd graph; serving does not. So this file drops lib/tensor.js entirely and runs
// the same architecture as lib/gpt.js directly on lib/ops.js kernels, reading the weights of a
// GPT.toJSON() checkpoint. Same arithmetic, no graph, no gradients — forward() here agrees with
// GPT.forward() to well within 1e-4.
//
// The cache is the point of the file. In causal attention a past token's key and value never change:
// they depend on that token and the tokens before it, and nothing later can reach back. So storing them
// once turns generation from "recompute the whole prefix per token" (forward) into "one query against
// stored keys and values" (forwardStep). The price is memory that grows linearly with context:
// 2 · nLayer · nHead · headDim · T floats.
//
// Honest simplifications (docs/CURRICULUM_BRIEFS.md, 15-kv-cache): batch size 1, float32, no paging
// (module 16), no prefix sharing (module 17), and prefill is just a loop of decode steps.

import * as ops from './ops.js';

/** Turn a GPT.toJSON() object into raw Float32Array weights: { config, w: { name: raw } }. */
export function loadModel(json) {
  const w = {};
  for (const name of Object.keys(json.params)) {
    const saved = json.params[name];
    w[name] = ops.raw(saved.shape, Float32Array.from(saved.data));
  }
  return { config: { ...json.config }, w };
}

/** y = x · W + b for raw tensors, the one pattern every projection in the model uses. */
function linear(x, weight, bias) {
  return ops.add(ops.matmul(x, weight), bias);
}

/** [T, C] -> [H, T, dh]: view the channels as H heads, then put the head axis in front of time. */
function splitHeads(x, time, nHead, headDim) {
  return ops.permute(ops.reshape(x, [time, nHead, headDim]), [1, 0, 2]);
}

/** [H, T, dh] -> [T, C]: the exact inverse of splitHeads. */
function mergeHeads(x, time, nEmbd) {
  return ops.reshape(ops.permute(x, [1, 0, 2]), [time, nEmbd]);
}

/** The per-position MLP: expand to 4C, GELU, project back to C. */
function mlpForward(w, prefix, x) {
  const hidden = ops.gelu(linear(x, w[`${prefix}.mlp.fc.weight`], w[`${prefix}.mlp.fc.bias`]));
  return linear(hidden, w[`${prefix}.mlp.proj.weight`], w[`${prefix}.mlp.proj.bias`]);
}

/** Token embedding + learned position embedding for the ids at positions `from`…`from + ids.length`. */
function embedTokens(w, ids, from) {
  const positions = ids.map((_, i) => from + i);
  return ops.add(ops.embed(w['wte.weight'], ids), ops.embed(w['wpe.weight'], positions));
}

/** The final LayerNorm and the tied head: [T, C] -> logits [T, V]. */
function head(w, x) {
  const normalized = ops.layerNorm(x, w['lnF.gamma'], w['lnF.beta']);
  return ops.matmul(normalized, ops.transpose(w['wte.weight']));
}

/**
 * Full recompute: logits [T, V] for every position of `ids`, with no cache. This is the reference the
 * cached path is checked against, and what a prefill would cost if you did it in one batched pass.
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

  let x = embedTokens(w, ids, 0);
  for (let layer = 0; layer < nLayer; layer++) {
    const prefix = `blocks.${layer}`;
    const normed = ops.layerNorm(x, w[`${prefix}.ln1.gamma`], w[`${prefix}.ln1.beta`]);
    const qkv = linear(normed, w[`${prefix}.attn.qkv.weight`], w[`${prefix}.attn.qkv.bias`]);
    const q = splitHeads(ops.slice(qkv, 1, 0, nEmbd), time, nHead, headDim);
    const k = splitHeads(ops.slice(qkv, 1, nEmbd, 2 * nEmbd), time, nHead, headDim);
    const v = splitHeads(ops.slice(qkv, 1, 2 * nEmbd, 3 * nEmbd), time, nHead, headDim);

    const scores = ops.scale(ops.matmul(q, ops.transpose(k)), scale); // [H,T,T]
    const weights = ops.softmax(ops.maskedFill(scores, mask, -Infinity));
    const attended = mergeHeads(ops.matmul(weights, v), time, nEmbd);
    x = ops.add(x, linear(attended, w[`${prefix}.attn.proj.weight`], w[`${prefix}.attn.proj.bias`]));

    const normed2 = ops.layerNorm(x, w[`${prefix}.ln2.gamma`], w[`${prefix}.ln2.beta`]);
    x = ops.add(x, mlpForward(w, prefix, normed2));
  }
  return head(w, x);
}

/** An empty KV cache: per layer a [H, 0, dh] key and value block that forwardStep appends to. */
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
 * Decode one token: embed it at position cache.length, append its key and value to every layer's cache,
 * attend over everything stored so far, and return the logits [V] for the NEXT token. Cost per token is
 * constant in the prefix length for the matmuls and linear in it only for the attention itself.
 */
export function forwardStep(model, cache, id) {
  const { nLayer, nHead, nEmbd, blockSize } = model.config;
  const w = model.w;
  if (cache.length >= blockSize) throw new Error(`forwardStep: cache is full at blockSize ${blockSize}`);
  const headDim = nEmbd / nHead;
  const scale = 1 / Math.sqrt(headDim);
  const position = cache.length;

  let x = embedTokens(w, [id], position); // [1, C]
  for (let layer = 0; layer < nLayer; layer++) {
    const prefix = `blocks.${layer}`;
    const normed = ops.layerNorm(x, w[`${prefix}.ln1.gamma`], w[`${prefix}.ln1.beta`]);
    const qkv = linear(normed, w[`${prefix}.attn.qkv.weight`], w[`${prefix}.attn.qkv.bias`]);
    const q = splitHeads(ops.slice(qkv, 1, 0, nEmbd), 1, nHead, headDim); // [H,1,dh]
    const k = splitHeads(ops.slice(qkv, 1, nEmbd, 2 * nEmbd), 1, nHead, headDim);
    const v = splitHeads(ops.slice(qkv, 1, 2 * nEmbd, 3 * nEmbd), 1, nHead, headDim);

    // Growing by concat keeps the reference readable; a real engine appends into a preallocated buffer.
    cache.k[layer] = ops.concat(cache.k[layer], k, 1); // [H, t+1, dh]
    cache.v[layer] = ops.concat(cache.v[layer], v, 1);

    // No mask: everything in the cache is in the past by construction, so the one query sees all of it.
    const scores = ops.scale(ops.matmul(q, ops.transpose(cache.k[layer])), scale); // [H,1,t+1]
    const weights = ops.softmax(scores);
    const attended = mergeHeads(ops.matmul(weights, cache.v[layer]), 1, nEmbd);
    x = ops.add(x, linear(attended, w[`${prefix}.attn.proj.weight`], w[`${prefix}.attn.proj.bias`]));

    const normed2 = ops.layerNorm(x, w[`${prefix}.ln2.gamma`], w[`${prefix}.ln2.beta`]);
    x = ops.add(x, mlpForward(w, prefix, normed2));
  }
  cache.length = position + 1;
  return head(w, x).data; // [V]
}

/** Run a whole prompt through the cache and return the logits after its last token. */
export function prefill(model, cache, ids) {
  if (ids.length === 0) throw new Error('prefill: need at least one token');
  let logits = null;
  // A real engine prefills in one batched pass (it is compute-bound); token by token is the same result.
  for (const id of ids) logits = forwardStep(model, cache, id);
  return logits;
}

/** Trainable scalars implied by a config — the closed form behind the 2·params rule of thumb. */
function paramCount(config) {
  const { vocabSize, blockSize, nLayer, nEmbd } = config;
  const embeddings = vocabSize * nEmbd + blockSize * nEmbd;
  // Per block: 12C² of weights (qkv 3C², proj C², mlp 8C²), 9C of biases, 4C of LayerNorm gains/shifts.
  const perBlock = 12 * nEmbd * nEmbd + 13 * nEmbd;
  return embeddings + nLayer * perBlock + 2 * nEmbd; // + the final LayerNorm
}

/**
 * FLOPs to produce one more token with `contextLen` tokens of context. The matmuls cost about 2 FLOPs per
 * parameter per token; attention adds 4·L·T·C on top (scores and the weighted sum of values, each
 * 2·L·T·C). Without a cache every new token re-runs the whole context, hence the extra factor T.
 */
export function flopsPerToken(config, contextLen, { cached = true } = {}) {
  const perToken = 2 * paramCount(config) + 4 * config.nLayer * contextLen * config.nEmbd;
  return cached ? perToken : perToken * contextLen;
}
