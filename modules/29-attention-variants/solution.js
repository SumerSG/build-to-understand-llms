// Modern attention: RoPE, GQA, MLA, sliding windows — reference solution. Modern attention variants on raw tensors { shape, data: Float32Array }.
//
// Every variant here is a trade against the KV cache or the position signal:
//   RoPE     puts position into the q·k dot product as a rotation, so scores depend only on m − n.
//   GQA      lets groups of query heads share one key/value head: fewer heads to cache.
//   MLA      caches one low-rank latent per token and rebuilds every head's K and V from it.
//   Sliding  attends only to the last W keys, so a ring buffer of W slots is the whole cache.
// Head tensors are [H, T, dh]: H heads, T time steps, dh channels per head.

import * as ops from 'lib/ops.js';

// ---------- worked examples ----------

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

// ---------- step 1: rotary position embeddings ----------

/** The dh/2 rotation frequencies θ_i = base^(−2i/dh), i = 0 … dh/2 − 1. Pair 0 spins fastest. */
export function ropeFrequencies(dh, base = 10000) {
  if (dh % 2 !== 0) throw new Error(`ropeFrequencies: head dim ${dh} must be even`);
  const theta = new Float64Array(dh / 2);
  for (let i = 0; i < dh / 2; i++) theta[i] = base ** (-2 * i / dh);
  return theta;
}

/**
 * Rotate each channel pair (x[2i], x[2i+1]) of every row by the angle positions[t] · θ_i, where t is the
 * row's time step. x is [..., T, dh]; positions has length T. Returns a new tensor.
 */
export function applyRope(x, positions, { base = 10000, freqs = null } = {}) {
  const dh = x.shape[x.shape.length - 1];
  const T = x.shape[x.shape.length - 2];
  if (positions.length !== T) throw new Error(`applyRope: ${positions.length} positions for ${T} time steps`);
  const theta = freqs ?? ropeFrequencies(dh, base);
  const out = new Float32Array(x.data.length);
  const rows = x.data.length / dh;
  for (let r = 0; r < rows; r++) {
    const p = positions[r % T];
    const off = r * dh;
    for (let i = 0; i < dh / 2; i++) {
      const angle = p * theta[i];
      const cos = Math.cos(angle), sin = Math.sin(angle);
      const x0 = x.data[off + 2 * i], x1 = x.data[off + 2 * i + 1];
      out[off + 2 * i] = x0 * cos - x1 * sin;
      out[off + 2 * i + 1] = x0 * sin + x1 * cos;
    }
  }
  return { shape: x.shape.slice(), data: out };
}

// ---------- step 2: grouped-query attention ----------

/** Which KV head query head h reads: each consecutive group of nHead / nKVHead query heads shares one. */
export function kvHeadFor(h, nHead, nKVHead) {
  if (nHead % nKVHead !== 0) throw new Error(`kvHeadFor: ${nHead} query heads do not divide into ${nKVHead} KV heads`);
  return Math.floor(h / (nHead / nKVHead));
}

/** Attention with q [H, Tq, dh] and k, v [Hkv, Tk, dh]; same position rules as mha. */
export function gqaAttention(q, k, v, { causal = true } = {}) {
  const [H, Tq, dh] = q.shape;
  const [Hkv, Tk] = k.shape;
  if (H % Hkv !== 0) throw new Error(`gqaAttention: ${H} query heads do not divide into ${Hkv} KV heads`);
  const scale = 1 / Math.sqrt(dh);
  const out = new Float32Array(H * Tq * dh);
  const scores = new Float64Array(Tk);
  for (let h = 0; h < H; h++) {
    const g = kvHeadFor(h, H, Hkv);
    for (let i = 0; i < Tq; i++) {
      const pos = Tk - Tq + i;
      const last = causal ? pos : Tk - 1;
      const qOff = (h * Tq + i) * dh;
      let max = -Infinity;
      for (let j = 0; j <= last; j++) {
        const kOff = (g * Tk + j) * dh;
        let s = 0;
        for (let c = 0; c < dh; c++) s += q.data[qOff + c] * k.data[kOff + c];
        scores[j] = s * scale;
        if (scores[j] > max) max = scores[j];
      }
      let z = 0;
      for (let j = 0; j <= last; j++) { scores[j] = Math.exp(scores[j] - max); z += scores[j]; }
      for (let j = 0; j <= last; j++) {
        const w = scores[j] / z;
        const vOff = (g * Tk + j) * dh;
        for (let c = 0; c < dh; c++) out[qOff + c] += w * v.data[vOff + c];
      }
    }
  }
  return { shape: [H, Tq, dh], data: out };
}

/** KV-cache bytes one token costs: a key and a value per KV head per layer. */
export function kvBytesPerToken({ nLayer, nKVHead, headDim, bytesPerElement = 2 }) {
  return 2 * nLayer * nKVHead * headDim * bytesPerElement;
}

// ---------- step 3: multi-head latent attention ----------

/** c = x · W_dkv: [T, C] -> [T, dLatent]. This is the only per-token state MLA stores. */
export function mlaLatent(x, w) {
  return ops.matmul(x, w.wdkv);
}

/** Rebuild every head's keys and values from the latent: c [T, dLatent] -> k, v [H, T, dh]. */
export function mlaExpand(c, w, nHead) {
  return { k: splitHeads(ops.matmul(c, w.wuk), nHead), v: splitHeads(ops.matmul(c, w.wuv), nHead) };
}

/** Full-sequence MLA: queries from x, keys and values from the latent of x, causal. Returns [H, T, dh]. */
export function mlaAttention(x, w, { nHead }) {
  const q = splitHeads(ops.matmul(x, w.wq), nHead);
  const { k, v } = mlaExpand(mlaLatent(x, w), w, nHead);
  return mha(q, k, v, { causal: true });
}

/** An empty latent cache: zero rows of dLatent floats. */
export function newLatentCache(dLatent) {
  return { c: ops.zeros([0, dLatent]) };
}

/**
 * Decode one token. xRow is [1, C]. Append its latent to cache.c, rebuild K and V for every cached token,
 * and attend this token's queries over them. Returns [H, 1, dh].
 */
export function mlaDecodeStep(xRow, cache, w, { nHead }) {
  cache.c = ops.concat(cache.c, mlaLatent(xRow, w), 0);
  const q = splitHeads(ops.matmul(xRow, w.wq), nHead);
  const { k, v } = mlaExpand(cache.c, w, nHead);
  return mha(q, k, v, { causal: true });
}

/** Bytes per token of an MLA cache: one latent (plus the shared RoPE key, if any) per layer. */
export function mlaBytesPerToken({ nLayer, dLatent, dRope = 0, bytesPerElement = 2 }) {
  return nLayer * (dLatent + dRope) * bytesPerElement;
}

// ---------- step 4: sliding-window attention and the ring cache ----------

/** Causal attention where the query at position p sees only keys p − window + 1 … p. Shapes as in mha. */
export function slidingWindowAttention(q, k, v, window) {
  const [H, Tq, dh] = q.shape;
  const Tk = k.shape[1];
  const scale = 1 / Math.sqrt(dh);
  const out = new Float32Array(H * Tq * dh);
  const scores = new Float64Array(Tk);
  for (let h = 0; h < H; h++) {
    for (let i = 0; i < Tq; i++) {
      const pos = Tk - Tq + i;
      const first = Math.max(0, pos - window + 1);
      const qOff = (h * Tq + i) * dh;
      let max = -Infinity;
      for (let j = first; j <= pos; j++) {
        const kOff = (h * Tk + j) * dh;
        let s = 0;
        for (let c = 0; c < dh; c++) s += q.data[qOff + c] * k.data[kOff + c];
        scores[j] = s * scale;
        if (scores[j] > max) max = scores[j];
      }
      let z = 0;
      for (let j = first; j <= pos; j++) { scores[j] = Math.exp(scores[j] - max); z += scores[j]; }
      for (let j = first; j <= pos; j++) {
        const w = scores[j] / z;
        const vOff = (h * Tk + j) * dh;
        for (let c = 0; c < dh; c++) out[qOff + c] += w * v.data[vOff + c];
      }
    }
  }
  return { shape: [H, Tq, dh], data: out };
}

/**
 * A rolling KV buffer of `window` slots per head (Mistral's rolling buffer cache). Storage k and v are
 * Float32Arrays laid out [H, window, dh]; the token at position p lives in slot p % window.
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
    return Math.min(this.count, this.window);
  }

  /** Append one token's keys and values, each [H, 1, dh], overwriting the oldest slot once full. */
  append(k, v) {
    const { nHead: H, headDim: dh, window: W } = this;
    const slot = this.count % W;
    for (let h = 0; h < H; h++) {
      const dst = (h * W + slot) * dh;
      for (let c = 0; c < dh; c++) {
        this.k[dst + c] = k.data[h * dh + c];
        this.v[dst + c] = v.data[h * dh + c];
      }
    }
    this.count++;
  }

  _gather(buf) {
    const { nHead: H, headDim: dh, window: W } = this;
    const n = this.length;
    const start = this.count - n; // position of the oldest held token
    const out = new Float32Array(H * n * dh);
    for (let h = 0; h < H; h++) {
      for (let t = 0; t < n; t++) {
        const src = (h * W + ((start + t) % W)) * dh;
        out.set(buf.subarray(src, src + dh), (h * n + t) * dh);
      }
    }
    return { shape: [H, n, dh], data: out };
  }

  /** Held keys as [H, length, dh], oldest first. */
  keys() { return this._gather(this.k); }

  /** Held values as [H, length, dh], oldest first. */
  values() { return this._gather(this.v); }

  /** Bytes the buffer occupies at the given element size. Constant: it never grows. */
  bytes(bytesPerElement = 4) {
    return 2 * this.nHead * this.window * this.headDim * bytesPerElement;
  }
}

// ---------- step 5: stretching RoPE past the training length ----------

/**
 * RoPE frequencies rescaled so a model trained on `trainLen` tokens can run at factor × trainLen.
 *   'none' θ_i unchanged; 'pi' θ_i / factor (Position Interpolation); 'ntk' base · factor^(dh/(dh−2));
 *   'yarn' per-pair blend: pairs with fewer than alpha turns in trainLen are interpolated, pairs with more
 *   than beta turns are left alone, and a linear ramp in between (YaRN's "NTK-by-parts").
 */
export function scaledRopeFrequencies(dh, { base = 10000, factor = 1, method = 'none', trainLen = 0, alpha = 1, beta = 32 } = {}) {
  const theta = ropeFrequencies(dh, base);
  if (method === 'none' || factor === 1) return theta;
  if (method === 'pi') return theta.map((t) => t / factor);
  if (method === 'ntk') return ropeFrequencies(dh, base * factor ** (dh / (dh - 2)));
  if (method === 'yarn') {
    if (!(trainLen > 0)) throw new Error('scaledRopeFrequencies: yarn needs trainLen');
    return theta.map((t) => {
      const turns = (trainLen * t) / (2 * Math.PI); // full rotations this pair makes over the training length
      const gamma = Math.min(1, Math.max(0, (turns - alpha) / (beta - alpha)));
      return (1 - gamma) * (t / factor) + gamma * t;
    });
  }
  throw new Error(`scaledRopeFrequencies: unknown method "${method}"`);
}

/**
 * How much of the rotation circle each pair reaches at length `len` that it never reached in training,
 * averaged over pairs, as a fraction of a full turn (0 = nothing new, 1 = every pair sees a whole new
 * circle). Pair i reached min(trainLen · trainFreqs[i], 2π) in training and reaches min(len · freqs[i], 2π) now.
 */
export function unseenRotation(freqs, trainFreqs, trainLen, len) {
  const TWO_PI = 2 * Math.PI;
  let total = 0;
  for (let i = 0; i < freqs.length; i++) {
    const seen = Math.min(trainLen * trainFreqs[i], TWO_PI);
    const now = Math.min(len * freqs[i], TWO_PI);
    total += Math.max(0, now - seen) / TWO_PI;
  }
  return total / freqs.length;
}

/**
 * The price of rescaling: how much the RoPE score between a query and a key changes at the offsets the
 * model was trained on. For Δ = 0 … maxOffset − 1, score(Δ) = rope(q, Δ) · k; returns the mean of
 * |score_freqs(Δ) − score_trainFreqs(Δ)| divided by |q|·|k|. q and k are Float32Arrays of length dh.
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
 * KV-cache cost of one sequence of `contextLen` tokens for each variant, at the dims in `config`:
 * { nLayer, nHead, nKVHead, headDim, dLatent, dRope, window, fullAttnEvery = 4, bytesPerElement }.
 * Returns rows { variant, bytesPerToken, tokensHeld, fixedBytes, totalBytes } for MHA, GQA, MQA, MLA,
 * SWA (sliding window on top of GQA, as in Mistral 7B) and HYBRID (one GQA layer in every
 * `fullAttnEvery`, the rest recurrent with a constant headDim × headDim state per head, Qwen3-Next style).
 */
export function cacheBudget(config, contextLen) {
  const { nLayer, nHead, nKVHead, headDim, dLatent, dRope = 0, window, fullAttnEvery = 4, bytesPerElement = 2 } = config;
  const kv = (heads, layers = nLayer) => kvBytesPerToken({ nLayer: layers, nKVHead: heads, headDim, bytesPerElement });
  const row = (variant, bytesPerToken, tokensHeld, fixedBytes = 0) => ({ variant, bytesPerToken, tokensHeld, fixedBytes, totalBytes: bytesPerToken * tokensHeld + fixedBytes });
  const nFull = Math.floor(nLayer / fullAttnEvery);
  const stateBytes = (nLayer - nFull) * nHead * headDim * headDim * bytesPerElement;
  return [
    row('MHA', kv(nHead), contextLen),
    row('GQA', kv(nKVHead), contextLen),
    row('MQA', kv(1), contextLen),
    row('MLA', mlaBytesPerToken({ nLayer, dLatent, dRope, bytesPerElement }), contextLen),
    row('SWA', kv(nKVHead), Math.min(contextLen, window)),
    row('HYBRID', kv(nKVHead, nFull), contextLen, stateBytes),
  ];
}
