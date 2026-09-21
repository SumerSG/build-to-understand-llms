// lib/attention.js — scaled dot-product attention and multi-head attention on autograd Tensors.
// This file is the REFERENCE SOLUTION for module 05 (Attention from scratch).
//
// Attention lets every position build its own representation as a weighted average of the other
// positions' values. The weights come from a similarity score between this position's query and every
// other position's key, so the model decides at run time where to look instead of having the wiring
// baked into its parameters.
//
//   scores  = q · kᵀ / sqrt(dh)      [.., T, T]   how well query i matches key j
//   weights = softmax(scores)        rows sum to 1
//   out     = weights · v            [.., T, dh]  the weighted average of the values
//
// The 1/sqrt(dh) is not cosmetic: a dot product of dh independent unit-variance terms has variance dh,
// and without the scale the softmax of those scores saturates and its gradient vanishes as dh grows.
//
// The causal mask is what makes this a language model: position i may only attend to positions j <= i,
// so the prediction at i cannot see the answer that comes after it. We add it by setting the disallowed
// scores to -Infinity BEFORE the softmax, where exp(-Infinity) = 0 removes them exactly.
//
// Honest simplifications (docs/CURRICULUM_BRIEFS.md, 05-attention): no FlashAttention tiling (the whole
// [T,T] score matrix is materialised), no rotary embeddings (module 06 uses learned positions), no
// grouped-query attention.

import * as ops from './ops.js';
import { Linear } from './layers.js';

/**
 * Scaled dot-product attention. q, k, v are Tensors [B,H,T,dh] (a plain [T,dh] also works);
 * scale defaults to 1/sqrt(dh). Returns the attended values and the softmaxed weights.
 */
export function attention(q, k, v, { causal = true, scale = null } = {}) {
  const dh = q.shape[q.shape.length - 1];
  const queryCount = q.shape[q.shape.length - 2];
  const keyCount = k.shape[k.shape.length - 2];
  const factor = scale === null ? 1 / Math.sqrt(dh) : scale;

  // transpose() swaps the last two dims, so kᵀ is [.., dh, T] and the product is [.., T, T].
  let scores = q.matmul(k.transpose()).scale(factor);
  if (causal) {
    if (queryCount !== keyCount) {
      throw new Error(`attention: causal masking needs as many queries as keys (${queryCount} vs ${keyCount})`);
    }
    // causalMask(T) is 1 where key j <= query i; maskedFill replaces everything else with -Infinity.
    scores = scores.maskedFill(ops.causalMask(keyCount), -Infinity);
  }
  const weights = scores.softmax();
  const out = weights.matmul(v);
  return { out, weights };
}

/**
 * Multi-head attention: run `nHead` independent attentions on `nEmbd / nHead` channels each, then mix
 * their outputs back together. One head can only average one way per position; several heads let the
 * layer look up several different things at once (previous token, matching bracket, subject of the verb).
 */
export class MultiHeadAttention {
  /** One Linear produces q, k and v at once (3·nEmbd columns), a second projects the merged heads back. */
  constructor({ nEmbd, nHead, next }) {
    if (nEmbd % nHead !== 0) throw new Error(`MultiHeadAttention: nEmbd ${nEmbd} is not divisible by nHead ${nHead}`);
    this.nEmbd = nEmbd;
    this.nHead = nHead;
    this.headDim = nEmbd / nHead;
    this.qkv = new Linear(nEmbd, 3 * nEmbd, { next });
    this.proj = new Linear(nEmbd, nEmbd, { next });
    // The last forward's weights as a raw tensor [B,H,T,T], detached, for the heatmap in the demo.
    this.lastWeights = null;
  }

  /** Split [B,T,C] into per-head [B,H,T,dh]: view the channels as H groups, then bring H next to the batch. */
  _splitHeads(x, batch, time) {
    return x.reshape([batch, time, this.nHead, this.headDim]).permute([0, 2, 1, 3]);
  }

  /** x [B,T,C] -> [B,T,C], causal. */
  forward(x) {
    if (x.shape.length !== 3) throw new Error(`MultiHeadAttention.forward: expected [B,T,C], got [${x.shape}]`);
    const [batch, time, channels] = x.shape;
    if (channels !== this.nEmbd) throw new Error(`MultiHeadAttention.forward: expected C=${this.nEmbd}, got ${channels}`);

    const projected = this.qkv.forward(x); // [B,T,3C]
    const q = this._splitHeads(projected.slice(2, 0, channels), batch, time);
    const k = this._splitHeads(projected.slice(2, channels, 2 * channels), batch, time);
    const v = this._splitHeads(projected.slice(2, 2 * channels, 3 * channels), batch, time);

    const { out, weights } = attention(q, k, v, { causal: true });
    this.lastWeights = { shape: weights.shape.slice(), data: new Float32Array(weights.data) };

    // Merge the heads back: [B,H,T,dh] -> [B,T,H,dh] -> [B,T,C], then mix them with the output projection.
    const merged = out.permute([0, 2, 1, 3]).reshape([batch, time, channels]);
    return this.proj.forward(merged);
  }

  /** Every trainable tensor of this layer: the qkv projection then the output projection. */
  parameters() {
    return [...this.qkv.parameters(), ...this.proj.parameters()];
  }
}
