// Module 05 — reference solution. The same layer lives in lib/attention.js; modules 06+ import that one.
//
// Attention in one line: out = softmax(q · kᵀ / sqrt(dh)) · v. Every position builds its output as a
// weighted average of the values at every position it is allowed to see, and the weights come from how
// well its query matches each key. The causal mask (-Infinity above the diagonal, before the softmax) is
// what turns this into a language model: position i cannot read positions j > i.

import * as ops from 'lib/ops.js';
import { Tensor, gradCheck, noGrad } from 'lib/tensor.js';
import { Linear } from 'lib/gpt.js';
import { randn } from 'lib/util.js';

// ---------- worked example: attention entropy (used by the goal demo) ----------

/**
 * Mean entropy (in nats) of the attention rows of each head. weights is a Tensor or raw tensor whose
 * last three dims are [H, T, T] (any leading batch dims). A row that puts all its mass on one key has
 * entropy 0; a row spread evenly over n keys has entropy ln(n). Zero weights contribute 0, so the
 * masked entries of a causal row are simply skipped.
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

/** scores = q · kᵀ · factor, where factor defaults to 1 / sqrt(dh). Shape [.., T, T]. */
export function attentionScores(q, k, { scale = null } = {}) {
  const dh = q.shape[q.shape.length - 1];
  const factor = scale === null ? 1 / Math.sqrt(dh) : scale;
  return q.matmul(k.transpose()).scale(factor);
}

// ---------- step 2: the causal mask and the softmax ----------

/** Set every score with key index j > query index i to -Infinity, so the softmax gives it weight 0. */
export function maskCausal(scores) {
  const T = scores.shape[scores.shape.length - 1];
  const queries = scores.shape[scores.shape.length - 2];
  if (queries !== T) throw new Error(`maskCausal: need a square [T,T] score matrix, got [${queries},${T}]`);
  return scores.maskedFill(ops.causalMask(T), -Infinity);
}

/** Rows of probabilities: softmax over the keys, after masking when causal. */
export function attentionWeights(scores, { causal = true } = {}) {
  return (causal ? maskCausal(scores) : scores).softmax();
}

// ---------- step 3: the weighted sum ----------

/** Scaled dot-product attention on Tensors [.., T, dh]. Returns { out: [.., T, dh], weights: [.., T, T] }. */
export function attention(q, k, v, { causal = true, scale = null } = {}) {
  const weights = attentionWeights(attentionScores(q, k, { scale }), { causal });
  const out = weights.matmul(v);
  return { out, weights };
}

// ---------- step 4: multi-head attention ----------

/** [B, T, C] -> [B, H, T, dh]: view the channels as H groups of dh, then move H in front of T. */
export function splitHeads(x, nHead) {
  const [B, T, C] = x.shape;
  if (C % nHead !== 0) throw new Error(`splitHeads: C=${C} is not divisible by nHead=${nHead}`);
  return x.reshape([B, T, nHead, C / nHead]).permute([0, 2, 1, 3]);
}

/** [B, H, T, dh] -> [B, T, C]: the exact inverse of splitHeads. */
export function mergeHeads(x) {
  const [B, H, T, dh] = x.shape;
  return x.permute([0, 2, 1, 3]).reshape([B, T, H * dh]);
}

/**
 * Causal multi-head self-attention: one Linear makes q, k and v for every head at once, each head
 * attends on its own dh channels, and a second Linear mixes the merged heads back together.
 */
export class MultiHeadAttention {
  constructor({ nEmbd, nHead, next }) {
    if (nEmbd % nHead !== 0) throw new Error(`MultiHeadAttention: nEmbd ${nEmbd} is not divisible by nHead ${nHead}`);
    this.nEmbd = nEmbd;
    this.nHead = nHead;
    this.headDim = nEmbd / nHead;
    this.qkv = new Linear(nEmbd, 3 * nEmbd, { next });
    this.proj = new Linear(nEmbd, nEmbd, { next });
    this.lastWeights = null; // raw [B,H,T,T] from the most recent forward, for the heatmap
  }

  /** x [B, T, C] -> [B, T, C]. */
  forward(x) {
    if (x.shape.length !== 3) throw new Error(`MultiHeadAttention.forward: expected [B,T,C], got [${x.shape}]`);
    const [, , C] = x.shape;
    if (C !== this.nEmbd) throw new Error(`MultiHeadAttention.forward: expected C=${this.nEmbd}, got ${C}`);
    const projected = this.qkv.forward(x); // [B, T, 3C]
    const q = splitHeads(projected.slice(2, 0, C), this.nHead);
    const k = splitHeads(projected.slice(2, C, 2 * C), this.nHead);
    const v = splitHeads(projected.slice(2, 2 * C, 3 * C), this.nHead);
    const { out, weights } = attention(q, k, v, { causal: true });
    this.lastWeights = { shape: weights.shape.slice(), data: new Float32Array(weights.data) };
    return this.proj.forward(mergeHeads(out));
  }

  /** Every trainable tensor: the qkv projection's weight and bias, then the output projection's. */
  parameters() {
    return [...this.qkv.parameters(), ...this.proj.parameters()];
  }
}

// ---------- step 5: prove it (causality probe and gradient check) ----------

/**
 * Perturb every position after t with Gaussian noise and measure how much the layer's output moves.
 * maxBefore: largest absolute change at positions 0..t (must be exactly 0 for a causal layer).
 * maxAfter:  largest absolute change at positions t+1..T-1 (should be > 0: the noise did reach them).
 */
export function causalityProbe(layer, x, t, next) {
  const [B, T, C] = x.shape;
  const base = noGrad(() => layer.forward(x));
  const noisy = new Float32Array(x.data);
  for (let b = 0; b < B; b++) {
    for (let i = t + 1; i < T; i++) {
      for (let c = 0; c < C; c++) noisy[(b * T + i) * C + c] += randn(next);
    }
  }
  const after = noGrad(() => layer.forward(new Tensor({ shape: x.shape.slice(), data: noisy })));
  let maxBefore = 0, maxAfter = 0;
  const width = base.data.length / (B * T);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < T; i++) {
      for (let c = 0; c < width; c++) {
        const idx = (b * T + i) * width + c;
        const diff = Math.abs(after.data[idx] - base.data[idx]);
        if (i <= t) maxBefore = Math.max(maxBefore, diff);
        else maxAfter = Math.max(maxAfter, diff);
      }
    }
  }
  return { maxBefore, maxAfter };
}

/**
 * Numerical gradient check through the whole layer: the scalar loss is the sum of squared outputs, and
 * the inputs checked are x (a Tensor with requiresGrad) followed by every parameter of the layer.
 */
export function gradCheckAttention(layer, x, opts = {}) {
  const inputs = [x, ...layer.parameters()];
  return gradCheck((xIn) => layer.forward(xIn).pow(2).sum(), inputs, opts);
}
