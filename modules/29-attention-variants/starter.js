// Modern attention: RoPE, GQA, MLA and sliding windows.
// Raw tensors are { shape: number[], data: Float32Array }, row-major (lib/ops.js conventions).
// Head tensors are [H, T, dh]: H heads, T time steps, dh channels per head.
// Everything marked TODO is yours; the worked examples set the conventions.

import * as ops from 'lib/ops.js';

// ---------- worked examples (done for you; read them first) ----------

/**
 * Plain multi-head attention, the reference every variant reduces to. q is [H, Tq, dh]; k and v are
 * [H, Tk, dh] with Tk >= Tq. The queries are the LAST Tq positions of the sequence: query i sits at
 * absolute position Tk − Tq + i, so the same function serves a full prefill (Tq = Tk) and a decode step
 * against a cache (Tq = 1). With causal = true, query at position p sees keys 0…p.
 */
export function mha(q, k, v, { causal = true } = {}) {
  const [H, Tq, dh] = q.shape;
  const Tk = k.shape[1];
  if (k.shape[0] !== H || v.shape[0] !== H) throw new Error(`mha: q has ${H} heads but k/v have ${k.shape[0]}/${v.shape[0]}`);
  if (Tk < Tq) throw new Error(`mha: ${Tq} queries but only ${Tk} keys`);
  const scale = 1 / Math.sqrt(dh);
  const out = new Float32Array(H * Tq * dh);
  const scores = new Float64Array(Tk);
  for (let h = 0; h < H; h++) {
    for (let i = 0; i < Tq; i++) {
      const pos = Tk - Tq + i;                  // absolute position of this query
      const last = causal ? pos : Tk - 1;       // the last key it may see
      const qOff = (h * Tq + i) * dh;
      let max = -Infinity;
      for (let j = 0; j <= last; j++) {
        const kOff = (h * Tk + j) * dh;
        let s = 0;
        for (let c = 0; c < dh; c++) s += q.data[qOff + c] * k.data[kOff + c];
        scores[j] = s * scale;
        if (scores[j] > max) max = scores[j];
      }
      let z = 0;
      for (let j = 0; j <= last; j++) { scores[j] = Math.exp(scores[j] - max); z += scores[j]; }
      for (let j = 0; j <= last; j++) {
        const w = scores[j] / z;
        const vOff = (h * Tk + j) * dh;
        for (let c = 0; c < dh; c++) out[qOff + c] += w * v.data[vOff + c];
      }
    }
  }
  return { shape: [H, Tq, dh], data: out };
}

/** [T, H·dh] -> [H, T, dh]: view the channels as H heads and move the head axis to the front. */
export function splitHeads(x, nHead) {
  const [T, C] = x.shape;
  if (C % nHead !== 0) throw new Error(`splitHeads: ${C} channels do not split into ${nHead} heads`);
  return ops.permute(ops.reshape(x, [T, nHead, C / nHead]), [1, 0, 2]);
}

/** Random MLA weights (std 1/sqrt(fan-in)) for the given dims. Nothing here is trained. */
export function initMLA({ nEmbd, nHead, headDim, dLatent, next }) {
  const w = (nIn, nOut) => ops.randn([nIn, nOut], next, 1 / Math.sqrt(nIn));
  return {
    wq: w(nEmbd, nHead * headDim),     // x -> queries, all heads
    wdkv: w(nEmbd, dLatent),           // x -> latent c (the only thing cached)
    wuk: w(dLatent, nHead * headDim),  // c -> keys, all heads
    wuv: w(dLatent, nHead * headDim),  // c -> values, all heads
  };
}

/** An empty latent cache for step 3: zero rows of dLatent floats. mlaDecodeStep appends to cache.c. */
export function newLatentCache(dLatent) {
  return { c: ops.zeros([0, dLatent]) };
}

// ---------- step 1: rotary position embeddings ----------

/** The dh/2 rotation frequencies θ_i = base^(−2i/dh), i = 0 … dh/2 − 1, as a Float64Array. */
export function ropeFrequencies(dh, base = 10000) {
  // TODO: step 1
  return new Float64Array(dh / 2).fill(1);
}

/**
 * Rotate each channel pair (x[2i], x[2i+1]) of every row by the angle positions[t] · θ_i, where t is the
 * row's time step. x is [..., T, dh]; positions has length T. `freqs`, when given, replaces
 * ropeFrequencies(dh, base) (step 5 passes rescaled ones). Returns a new tensor.
 */
export function applyRope(x, positions, { base = 10000, freqs = null } = {}) {
  const dh = x.shape[x.shape.length - 1];
  const T = x.shape[x.shape.length - 2];
  if (positions.length !== T) throw new Error(`applyRope: ${positions.length} positions for ${T} time steps`);
  const theta = freqs ?? ropeFrequencies(dh, base);
  const out = new Float32Array(x.data.length);
  const rows = x.data.length / dh;
  for (let r = 0; r < rows; r++) {
    const p = positions[r % T];    // flat row r of an [..., T, dh] tensor is time step r % T
    const off = r * dh;
    for (let i = 0; i < dh / 2; i++) {
      // TODO: step 1 — rotate the pair (x.data[off + 2i], x.data[off + 2i + 1]) by the angle p · theta[i]
      // and write the result to out[off + 2i], out[off + 2i + 1].
    }
  }
  return { shape: x.shape.slice(), data: out };
}

// ---------- step 2: grouped-query attention ----------

/** Which KV head query head h reads. Throw if nHead is not a multiple of nKVHead. */
export function kvHeadFor(h, nHead, nKVHead) {
  // TODO: step 2
  return h;
}

/** Attention with q [H, Tq, dh] and k, v [Hkv, Tk, dh] (H a multiple of Hkv); same position rules as mha. */
export function gqaAttention(q, k, v, { causal = true } = {}) {
  // TODO: step 2
  return ops.zeros(q.shape);
}

/** KV-cache bytes one token costs: a key and a value per KV head per layer. */
export function kvBytesPerToken({ nLayer, nKVHead, headDim, bytesPerElement = 2 }) {
  // TODO: step 2
  return 0;
}

// ---------- step 3: multi-head latent attention ----------

/** c = x · W_dkv: [T, C] -> [T, dLatent]. The only per-token state MLA stores. */
export function mlaLatent(x, w) {
  // TODO: step 3
  return x;
}

/** Rebuild every head's keys and values from the latent: c [T, dLatent] -> { k, v } each [H, T, dh]. */
export function mlaExpand(c, w, nHead) {
  // TODO: step 3
  return { k: c, v: c };
}

/** Full-sequence MLA: queries from x · W_q, keys and values from the latent of x, causal. Returns [H, T, dh]. */
export function mlaAttention(x, w, { nHead }) {
  // TODO: step 3
  return ops.zeros([nHead, x.shape[0], w.wq.shape[1] / nHead]);
}

/**
 * Decode one token. xRow is [1, C]. Append its latent to cache.c, rebuild K and V for every cached token,
 * and attend this token's queries over them. Returns [H, 1, dh].
 */
export function mlaDecodeStep(xRow, cache, w, { nHead }) {
  // TODO: step 3
  return ops.zeros([nHead, 1, w.wq.shape[1] / nHead]);
}

/** Bytes per token of an MLA cache: one latent (plus the shared RoPE key, dRope) per layer. */
export function mlaBytesPerToken({ nLayer, dLatent, dRope = 0, bytesPerElement = 2 }) {
  // TODO: step 3
  return 0;
}

// ---------- step 4: sliding-window attention and the ring cache ----------

/** Causal attention where the query at position p sees only keys p − window + 1 … p. Shapes as in mha. */
export function slidingWindowAttention(q, k, v, window) {
  // TODO: step 4
  return mha(q, k, v);
}

/**
 * A rolling KV buffer of `window` slots per head (Mistral's rolling buffer cache). Storage this.k and
 * this.v are Float32Arrays laid out [H, window, dh]; the token at position p lives in slot p % window.
 */
export class RingKVCache {
  constructor({ nHead, headDim, window }) {
    this.nHead = nHead;
    this.headDim = headDim;
    this.window = window;
    this.k = new Float32Array(nHead * window * headDim);
    this.v = new Float32Array(nHead * window * headDim);
    this.count = 0; // tokens ever appended
  }

  /** Tokens currently held: min(count, window). */
  get length() {
    // TODO: step 4
    return 0;
  }

  /** Append one token's keys and values, each [H, 1, dh], overwriting the oldest slot once full. */
  append(k, v) {
    // TODO: step 4
  }

  /** Held keys as [H, length, dh], oldest first. */
  keys() {
    // TODO: step 4
    return ops.zeros([this.nHead, 0, this.headDim]);
  }

  /** Held values as [H, length, dh], oldest first. */
  values() {
    // TODO: step 4
    return ops.zeros([this.nHead, 0, this.headDim]);
  }

  /** Bytes the two buffers occupy at the given element size (allocated, not filled). */
  bytes(bytesPerElement = 4) {
    // TODO: step 4
    return 0;
  }
}

// ---------- step 5: stretching RoPE past the training length ----------

/**
 * RoPE frequencies rescaled so a model trained on `trainLen` tokens can run at factor × trainLen.
 * method: 'none' | 'pi' | 'ntk' | 'yarn' (see the step instructions). Returns a Float64Array of dh/2.
 */
export function scaledRopeFrequencies(dh, { base = 10000, factor = 1, method = 'none', trainLen = 0, alpha = 1, beta = 32 } = {}) {
  // TODO: step 5
  return ropeFrequencies(dh, base);
}

/**
 * Mean over pairs of the rotation reached at length `len` that was never reached in training, as a
 * fraction of a full turn. Pair i reached min(trainLen · trainFreqs[i], 2π) in training.
 */
export function unseenRotation(freqs, trainFreqs, trainLen, len) {
  // TODO: step 5
  return 0;
}

/**
 * Done for you (used by the demo): how much rescaling changes the RoPE score between a query and a key at
 * the offsets the model was trained on. For Δ = 0 … maxOffset − 1, score(Δ) = rope(q, Δ) · k; returns the
 * mean of |score_freqs(Δ) − score_trainFreqs(Δ)| / (|q|·|k|). q and k are Float32Arrays of length dh.
 */
export function scoreDrift(q, k, freqs, trainFreqs, maxOffset) {
  const dh = q.length;
  const rep = new Float32Array(maxOffset * dh);
  for (let t = 0; t < maxOffset; t++) rep.set(q, t * dh);
  const x = { shape: [maxOffset, dh], data: rep };
  const positions = Array.from({ length: maxOffset }, (_, t) => t);
  const a = applyRope(x, positions, { freqs });
  const b = applyRope(x, positions, { freqs: trainFreqs });
  let nq = 0, nk = 0;
  for (let c = 0; c < dh; c++) { nq += q[c] * q[c]; nk += k[c] * k[c]; }
  const norm = Math.sqrt(nq * nk);
  let total = 0;
  for (let t = 0; t < maxOffset; t++) {
    let sa = 0, sb = 0;
    for (let c = 0; c < dh; c++) { sa += a.data[t * dh + c] * k[c]; sb += b.data[t * dh + c] * k[c]; }
    total += Math.abs(sa - sb) / norm;
  }
  return total / maxOffset;
}

// ---------- step 6: the cache budget ----------

/**
 * KV-cache cost of one sequence of `contextLen` tokens per variant, at the dims in `config`:
 * { nLayer, nHead, nKVHead, headDim, dLatent, dRope, window, fullAttnEvery = 4, bytesPerElement }.
 * Returns rows { variant, bytesPerToken, tokensHeld, fixedBytes, totalBytes } for 'MHA', 'GQA', 'MQA',
 * 'MLA', 'SWA', 'HYBRID' in that order (SWA = sliding window on top of GQA, as in Mistral 7B; HYBRID =
 * one GQA layer in every fullAttnEvery, the rest recurrent with a constant headDim × headDim state per head).
 */
export function cacheBudget(config, contextLen) {
  // TODO: step 6
  return [];
}
