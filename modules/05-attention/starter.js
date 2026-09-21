// Module 05 — Attention from scratch.
//
// You build scaled dot-product attention on autograd Tensors (lib/tensor.js), then wrap it into a causal
// multi-head self-attention layer, then prove two things about it: it never reads the future, and its
// gradients are right. Shapes: a sequence is [B, T, C] (batch, time, channels); per-head tensors are
// [B, H, T, dh] with dh = C / H. Everything below the "worked example" is yours to implement; the TODO
// markers say which step each piece belongs to.

import * as ops from 'lib/ops.js';
import { Tensor, gradCheck, noGrad } from 'lib/tensor.js';
import { Linear } from 'lib/gpt.js';
import { randn } from 'lib/util.js';

// ---------- worked example (done for you; read it, it sets the conventions) ----------

/**
 * Mean entropy (in nats) of the attention rows of each head. weights is a Tensor or raw tensor whose
 * last three dims are [H, T, T] (any leading batch dims). A row that puts all its mass on one key has
 * entropy 0; a row spread evenly over n keys has entropy ln(n). Zero weights contribute 0, so the
 * masked entries of a causal row are simply skipped. The goal demo reports this number per head.
 *
 * Conventions to notice: Tensors and raw tensors both expose `.shape` and a flat `.data` Float32Array in
 * row-major order, so the element [b, h, i, j] lives at ((b·H + h)·T + i)·T + j; the function is pure.
 */
export function entropyPerHead(weights) {
  const nd = weights.shape.length;
  const [H, T] = [weights.shape[nd - 3], weights.shape[nd - 2]];
  const batch = weights.data.length / (H * T * T);
  const totals = new Array(H).fill(0);
  for (let b = 0; b < batch; b++) {
    for (let h = 0; h < H; h++) {
      const base = (b * H + h) * T * T;
      for (let i = 0; i < T; i++) {
        let entropy = 0;
        for (let j = 0; j < T; j++) {
          const w = weights.data[base + i * T + j];
          if (w > 0) entropy -= w * Math.log(w);
        }
        totals[h] += entropy;
      }
    }
  }
  return totals.map((s) => s / (batch * T));
}

// ---------- step 1: similarity scores ----------

/**
 * scores = q · kᵀ · factor, where factor defaults to 1 / sqrt(dh) and dh is the last dim of q.
 * q and k are Tensors [.., T, dh] (any leading batch/head dims); the result is [.., T, T].
 * Tensor.transpose() swaps the last two dims, so k.transpose() is [.., dh, T].
 */
export function attentionScores(q, k, { scale = null } = {}) {
  const dh = q.shape[q.shape.length - 1];
  const factor = scale === null ? 1 : scale; // TODO step 1: the default factor is not 1
  // TODO step 1: multiply q by kᵀ and scale the product by factor
  return q.scale(factor);
}

// ---------- step 2: the causal mask and the softmax ----------

/** Return a copy of scores [.., T, T] with every entry where key j > query i set to -Infinity. */
export function maskCausal(scores) {
  // TODO step 2
  return scores;
}

/** Rows of probabilities over the keys: softmax over the last dim, after masking when causal. */
export function attentionWeights(scores, { causal = true } = {}) {
  // TODO step 2
  return scores;
}

// ---------- step 3: the weighted sum ----------

/**
 * Scaled dot-product attention on Tensors [.., T, dh].
 * Returns { out: Tensor [.., T, dh], weights: Tensor [.., T, T] }.
 */
export function attention(q, k, v, { causal = true, scale = null } = {}) {
  // TODO step 3
  return { out: q, weights: null };
}

// ---------- step 4: multi-head attention ----------

/** [B, T, C] -> [B, H, T, dh]: view the channels as H groups of dh, then move H in front of T. */
export function splitHeads(x, nHead) {
  // TODO step 4
  return x;
}

/** [B, H, T, dh] -> [B, T, C]: the exact inverse of splitHeads. */
export function mergeHeads(x) {
  // TODO step 4
  return x;
}

/**
 * Causal multi-head self-attention. The constructor and parameters() are done: one Linear produces q, k
 * and v for every head at once (3·C columns: q in 0..C, k in C..2C, v in 2C..3C), a second Linear mixes
 * the merged heads back together. You write forward().
 */
export class MultiHeadAttention {
  constructor({ nEmbd, nHead, next }) {
    if (nEmbd % nHead !== 0) throw new Error(`MultiHeadAttention: nEmbd ${nEmbd} is not divisible by nHead ${nHead}`);
    this.nEmbd = nEmbd;
    this.nHead = nHead;
    this.headDim = nEmbd / nHead;
    this.qkv = new Linear(nEmbd, 3 * nEmbd, { next });
    this.proj = new Linear(nEmbd, nEmbd, { next });
    this.lastWeights = null; // raw [B,H,T,T] copy of the weights from the most recent forward (for the heatmap)
  }

  /**
   * x [B, T, C] -> [B, T, C]. Project, slice q/k/v along the last axis (Tensor.slice(axis, start, end)),
   * split heads, attend causally, store a raw copy of the weights in this.lastWeights, merge, project.
   * Throw if x is not 3-D or its channel count is not nEmbd.
   */
  forward(x) {
    // TODO step 4
    return x;
  }

  /** Every trainable tensor: the qkv projection's weight and bias, then the output projection's. */
  parameters() {
    return [...this.qkv.parameters(), ...this.proj.parameters()];
  }
}

// ---------- step 5: prove it (causality probe and gradient check) ----------

/**
 * Perturb every position AFTER t (t+1 .. T-1) of a copy of x with Gaussian noise (randn(next) per
 * element) and run the layer on both versions inside noGrad. Return
 *   { maxBefore, maxAfter }
 * where maxBefore is the largest absolute output change at positions 0..t (exactly 0 for a causal
 * layer) and maxAfter the largest change at positions t+1..T-1 (> 0 when there is anything after t).
 */
export function causalityProbe(layer, x, t, next) {
  // TODO step 5
  return { maxBefore: 0, maxAfter: 0 };
}

/**
 * Numerical gradient check through the whole layer with lib/tensor.js gradCheck: the scalar loss is
 * the sum of squared outputs, and the inputs checked are x (a Tensor with requiresGrad) followed by
 * every parameter of the layer. Return gradCheck's result object ({ ok, maxRelErr, details }).
 */
export function gradCheckAttention(layer, x, opts = {}) {
  // TODO step 5
  return { ok: false, maxRelErr: Infinity, details: [] };
}
