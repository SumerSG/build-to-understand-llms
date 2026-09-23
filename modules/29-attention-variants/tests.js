import * as ops from 'lib/ops.js';
import { Tensor } from 'lib/tensor.js';
import { attention } from 'lib/attention.js';

// A random raw tensor with a fixed seed, so every run sees the same numbers.
function rand(T, shape, seed, std = 1) {
  return ops.randn(shape, T.rng(seed), std);
}

// k [Hkv, T, dh] -> [H, T, dh] by giving every query head the KV head the `map` function names.
function expandHeads(k, H, map) {
  const [, Tk, dh] = k.shape;
  const out = new Float32Array(H * Tk * dh);
  for (let h = 0; h < H; h++) out.set(k.data.subarray(map(h) * Tk * dh, (map(h) + 1) * Tk * dh), h * Tk * dh);
  return { shape: [H, Tk, dh], data: out };
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const row = (t, r, dh) => Array.from(t.data.subarray(r * dh, (r + 1) * dh));

const LLAMA3_8B = { nLayer: 32, nHead: 32, nKVHead: 8, headDim: 128, dLatent: 512, dRope: 64, window: 4096, bytesPerElement: 2 };

export const tests = [
  // ---------- step 1: RoPE ----------
  { step: 'rope', name: 'frequencies are base^(-2i/dh), one per channel pair', run(m, T) {
    const f = m.ropeFrequencies(8);
    T.eq(f.length, 4, 'a head of 8 channels has 4 rotation pairs, so 4 frequencies');
    T.close(Array.from(f), [1, 0.1, 0.01, 0.001], 1e-9, 'θ_i = 10000^(-2i/8): pair 0 turns one radian per position, each later pair 10x slower');
    T.close(Array.from(m.ropeFrequencies(4, 100)), [1, 0.1], 1e-9, 'the base argument must be used: 100^(-2i/4)');
  } },
  { step: 'rope', name: 'rotates adjacent pairs (x[2i], x[2i+1]) by position · θ_i', run(m, T) {
    const x = ops.fromArray([[1, 0, 1, 0], [0, 1, 0, 1]]);
    const y = m.applyRope(x, [1, 1], { base: 100 });
    T.shape(y, [2, 4], 'applyRope must keep the shape [T, dh]');
    const c1 = Math.cos(1), s1 = Math.sin(1), c2 = Math.cos(0.1), s2 = Math.sin(0.1);
    T.close(row(y, 0, 4), [c1, s1, c2, s2], 1e-5, 'at position 1 with base 100, (1, 0) in pair 0 rotates by 1 rad to (cos 1, sin 1) and pair 1 by 0.1 rad; pairs are ADJACENT channels (2i, 2i+1), as in RoFormer');
    T.close(row(y, 1, 4), [-s1, c1, -s2, c2], 1e-5, '(0, 1) rotated counter-clockwise by a goes to (−sin a, cos a): out0 = x0·cos − x1·sin, out1 = x0·sin + x1·cos');
  } },
  { step: 'rope', name: 'position 0 is the identity, rotation preserves length, input untouched', run(m, T) {
    const x = rand(T, [3, 16], 1);
    const before = Array.from(x.data);
    T.close(Array.from(m.applyRope(x, [0, 0, 0]).data), before, 1e-6, 'rotating by angle 0 must return the input unchanged');
    const y = m.applyRope(x, [5, 17, 300]);
    for (let r = 0; r < 3; r++) {
      T.close(Math.hypot(...row(y, r, 16)), Math.hypot(...before.slice(r * 16, r * 16 + 16)), 1e-4, 'a rotation never changes a vector\'s length; RoPE moves only the angle');
    }
    T.close(Array.from(x.data), before, 0, 'applyRope must return a new tensor and leave its input unmodified');
  } },
  { step: 'rope', name: 'q_m · k_n depends only on the offset m − n', run(m, T) {
    const q = rand(T, [1, 16], 2), k = rand(T, [1, 16], 3);
    const score = (mp, np) => dot(m.applyRope(q, [mp]).data, m.applyRope(k, [np]).data);
    const base = score(7, 3);
    T.close(score(4, 0), base, 1e-3, 'offset 4 must give the same score at positions (4, 0) as at (7, 3)');
    T.close(score(104, 100), base, 1e-3, 'and at (104, 100): RoPE encodes relative position');
    T.ok(Math.abs(score(9, 3) - base) > 1e-3, 'a different offset (6 instead of 4) must change the score; did you rotate at all?');
  } },
  { step: 'rope', name: 'on [H, T, dh], every head\'s row t uses positions[t]', run(m, T) {
    const x = rand(T, [2, 3, 8], 4);
    const pos = [5, 6, 7];
    const y = m.applyRope(x, pos);
    T.shape(y, [2, 3, 8]);
    for (let h = 0; h < 2; h++) for (let t = 0; t < 3; t++) {
      const single = m.applyRope({ shape: [1, 8], data: x.data.slice((h * 3 + t) * 8, (h * 3 + t + 1) * 8) }, [pos[t]]);
      T.close(row(y, h * 3 + t, 8), Array.from(single.data), 1e-5, `head ${h} time ${t} must be rotated by position ${pos[t]}; the time index of flat row r is r % T, not r`);
      T.close(Math.hypot(...row(y, h * 3 + t, 8)), Math.hypot(...row(x, h * 3 + t, 8)), 1e-4, 'each rotated row keeps its length; an all-zero or unrotated output is wrong');
    }
    T.ok(Math.abs(y.data[0] - x.data[0]) > 1e-4, 'position 5 must actually rotate the first pair (5 rad for pair 0)');
  } },

  // ---------- step 2: GQA ----------
  { step: 'gqa', name: 'kvHeadFor groups consecutive query heads', run(m, T) {
    T.eq([0, 1, 2, 3, 4, 5, 6, 7].map((h) => m.kvHeadFor(h, 8, 2)), [0, 0, 0, 0, 1, 1, 1, 1], 'with 8 query heads and 2 KV heads, heads 0–3 share KV head 0 and heads 4–7 share KV head 1 (h // (H/Hkv), not h % Hkv)');
    T.eq(m.kvHeadFor(5, 8, 8), 5, 'with as many KV heads as query heads (MHA) every head keeps its own');
    T.eq(m.kvHeadFor(7, 8, 1), 0, 'with one KV head (MQA) every query head reads head 0');
    T.throws(() => m.kvHeadFor(0, 6, 4), '6 query heads cannot be split evenly over 4 KV heads; throw');
  } },
  { step: 'gqa', name: 'H_kv = H is exactly multi-head attention (module-05 reference)', run(m, T) {
    const q = rand(T, [4, 6, 8], 10), k = rand(T, [4, 6, 8], 11), v = rand(T, [4, 6, 8], 12);
    const out = m.gqaAttention(q, k, v);
    T.shape(out, [4, 6, 8], 'the output has one row per query head and time step: [H, T, dh]');
    const ref = attention(new Tensor(q), new Tensor(k), new Tensor(v), { causal: true }).out;
    T.close(Array.from(out.data), Array.from(ref.data), 1e-4, 'with one KV head per query head, GQA must equal lib/attention.js attention (softmax(q·kᵀ/sqrt(dh)), causal) exactly');
    const open = m.gqaAttention(q, k, v, { causal: false });
    const refOpen = attention(new Tensor(q), new Tensor(k), new Tensor(v), { causal: false }).out;
    T.close(Array.from(open.data), Array.from(refOpen.data), 1e-4, 'with { causal: false } every query sees every key, as in mha; the option must be honoured, not hard-wired');
  } },
  { step: 'gqa', name: 'H_kv = 1 is multi-query attention; H_kv = 2 shares KV heads by group', run(m, T) {
    const q = rand(T, [4, 5, 8], 20), k1 = rand(T, [1, 5, 8], 21), v1 = rand(T, [1, 5, 8], 22);
    const mqa = m.gqaAttention(q, k1, v1);
    const ref1 = m.mha(q, expandHeads(k1, 4, () => 0), expandHeads(v1, 4, () => 0));
    T.close(Array.from(mqa.data), Array.from(ref1.data), 1e-5, 'one KV head shared by all 4 query heads must equal MHA with that head copied 4 times');
    const k2 = rand(T, [2, 5, 8], 23), v2 = rand(T, [2, 5, 8], 24);
    const g = m.gqaAttention(q, k2, v2);
    const group = (h) => Math.floor(h / 2);
    const ref2 = m.mha(q, expandHeads(k2, 4, group), expandHeads(v2, 4, group));
    T.close(Array.from(g.data), Array.from(ref2.data), 1e-5, 'query heads 0,1 must read KV head 0 and heads 2,3 KV head 1 (Llama\'s repeat_kv); interleaving 0,1,0,1 is a different model');
  } },
  { step: 'gqa', name: 'the full H_kv = 2 pass and a decode query (Tq = 1) both match grouped MHA', run(m, T) {
    const q = rand(T, [4, 5, 8], 30), k = rand(T, [2, 5, 8], 31), v = rand(T, [2, 5, 8], 32);
    const full = m.gqaAttention(q, k, v);
    const group = (h) => Math.floor(h / 2);
    T.close(Array.from(full.data), Array.from(m.mha(q, expandHeads(k, 4, group), expandHeads(v, 4, group)).data), 1e-5, 'the full H_kv = 2 pass (all 5 queries) must equal MHA with each KV head copied to its group of 2 query heads (heads 0,1 → KV 0, heads 2,3 → KV 1); fix this before the decode check below');
    const lastQ = { shape: [4, 1, 8], data: new Float32Array(32) };
    for (let h = 0; h < 4; h++) lastQ.data.set(q.data.subarray((h * 5 + 4) * 8, (h * 5 + 5) * 8), h * 8);
    const step = m.gqaAttention(lastQ, k, v);
    T.shape(step, [4, 1, 8]);
    for (let h = 0; h < 4; h++) T.close(row(step, h, 8), row(full, h * 5 + 4, 8), 1e-5, 'the single query sits at absolute position Tk − 1 and sees every cached key; this is how GQA runs against a KV cache');
    const k2 = ops.clone(k); for (let c = 0; c < 8; c++) k2.data[4 * 8 + c] += 5; // perturb the last key of KV head 0
    const full2 = m.gqaAttention(q, k2, v);
    T.close(row(full2, 0, 8), row(full, 0, 8), 1e-6, 'causal: changing the key at position 4 must not change the output at position 0');
  } },
  { step: 'gqa', name: 'KV bytes per token: Llama-3-8B at exactly 128 KiB', run(m, T) {
    T.eq(m.kvBytesPerToken({ nLayer: 32, nKVHead: 8, headDim: 128, bytesPerElement: 2 }), 131072, '2 (K and V) · 32 layers · 8 KV heads · 128 dims · 2 bytes (bf16) = 131,072 bytes');
    T.eq(m.kvBytesPerToken({ nLayer: 32, nKVHead: 32, headDim: 128 }), 524288, 'Llama-2-7B has 32 KV heads (plain MHA): 4x more per token. bytesPerElement defaults to 2');
  } },

  // ---------- step 3: MLA ----------
  { step: 'mla', name: 'the latent is x · W_dkv and expands to per-head K and V', run(m, T) {
    const w = m.initMLA({ nEmbd: 16, nHead: 4, headDim: 4, dLatent: 6, next: T.rng(40) });
    const x = rand(T, [5, 16], 41);
    const c = m.mlaLatent(x, w);
    T.shape(c, [5, 6], 'one latent of dLatent = 6 floats per token');
    T.close(Array.from(c.data), Array.from(ops.matmul(x, w.wdkv).data), 1e-5, 'c = x · W_dkv');
    const { k, v } = m.mlaExpand(c, w, 4);
    T.shape(k, [4, 5, 4], 'keys rebuilt for every head: [H, T, dh]');
    T.shape(v, [4, 5, 4]);
    T.close(Array.from(k.data), Array.from(m.splitHeads(ops.matmul(c, w.wuk), 4).data), 1e-5, 'k = splitHeads(c · W_uk)');
    T.close(Array.from(v.data), Array.from(m.splitHeads(ops.matmul(c, w.wuv), 4).data), 1e-5, 'v = splitHeads(c · W_uv), not W_uk');
  } },
  { step: 'mla', name: 'MLA is multi-head attention with low-rank K and V projections', run(m, T) {
    const w = m.initMLA({ nEmbd: 16, nHead: 4, headDim: 4, dLatent: 6, next: T.rng(42) });
    const x = rand(T, [6, 16], 43);
    const out = m.mlaAttention(x, w, { nHead: 4 });
    T.shape(out, [4, 6, 4]);
    const q = m.splitHeads(ops.matmul(x, w.wq), 4);
    const k = m.splitHeads(ops.matmul(x, ops.matmul(w.wdkv, w.wuk)), 4);
    const v = m.splitHeads(ops.matmul(x, ops.matmul(w.wdkv, w.wuv)), 4);
    T.close(Array.from(out.data), Array.from(m.mha(q, k, v).data), 1e-4, 'MLA must equal causal MHA whose key weight is W_dkv·W_uk and value weight W_dkv·W_uv (rank 6 instead of 16)');
  } },
  { step: 'mla', name: 'decoding from the latent cache matches recomputation, and the cache holds only latents', run(m, T) {
    const w = m.initMLA({ nEmbd: 16, nHead: 4, headDim: 4, dLatent: 6, next: T.rng(44) });
    const x = rand(T, [7, 16], 45);
    const full = m.mlaAttention(x, w, { nHead: 4 });
    const cache = m.newLatentCache(6);
    for (let t = 0; t < 7; t++) {
      const out = m.mlaDecodeStep(ops.slice(x, 0, t, t + 1), cache, w, { nHead: 4 });
      T.shape(out, [4, 1, 4], 'one decode step returns [H, 1, dh]');
      T.shape(cache.c, [t + 1, 6], `after ${t + 1} decode step(s) cache.c must hold ${t + 1} latents of 6 floats; if it did not grow, assign the concatenated tensor back to cache.c (a local variable is lost when the call returns)`);
      for (let h = 0; h < 4; h++) T.close(row(out, h, 4), row(full, h * 7 + t, 4), 1e-4, `step ${t}, head ${h}: decoding from cached latents must reproduce the full pass`);
    }
    T.shape(cache.c, [7, 6], 'after 7 tokens the cache is 7 latents of 6 floats: no per-head keys or values are stored');
  } },
  { step: 'mla', name: 'MLA bytes per token at DeepSeek-V2 dims', run(m, T) {
    T.eq(m.mlaBytesPerToken({ nLayer: 60, dLatent: 512, dRope: 64, bytesPerElement: 2 }), 69120, '60 layers · (512 latent + 64 shared RoPE key) · 2 bytes = 69,120 bytes; there is no factor 2 for K and V: one latent serves both');
    T.eq(m.mlaBytesPerToken({ nLayer: 2, dLatent: 32 }), 128, 'dRope defaults to 0 and bytesPerElement to 2: 2 · 32 · 2');
  } },

  // ---------- step 4: sliding window ----------
  { step: 'window', name: 'a window at least T long is full attention; a window of 1 returns v', run(m, T) {
    const q = rand(T, [2, 6, 8], 50), k = rand(T, [2, 6, 8], 51), v = rand(T, [2, 6, 8], 52);
    T.close(Array.from(m.slidingWindowAttention(q, k, v, 6).data), Array.from(m.mha(q, k, v).data), 1e-5, 'with window >= T no key is ever out of reach, so the output must equal causal MHA');
    T.close(Array.from(m.slidingWindowAttention(q, k, v, 100).data), Array.from(m.mha(q, k, v).data), 1e-5, 'a window longer than the sequence changes nothing');
    const one = m.slidingWindowAttention(q, k, v, 1);
    T.shape(one, [2, 6, 8]);
    T.close(Array.from(one.data), Array.from(v.data), 1e-5, 'window 1 means each token sees only itself, so the output is its own value; if it is not, your window is off by one');
  } },
  { step: 'window', name: 'window 3 sees exactly keys p−2 … p', run(m, T) {
    const q = rand(T, [2, 7, 4], 53), k = rand(T, [2, 7, 4], 54), v = rand(T, [2, 7, 4], 55);
    const out = m.slidingWindowAttention(q, k, v, 3);
    for (let p = 0; p < 7; p++) {
      const lo = Math.max(0, p - 2);
      const ref = m.mha(ops.slice(q, 1, p, p + 1), ops.slice(k, 1, lo, p + 1), ops.slice(v, 1, lo, p + 1));
      for (let h = 0; h < 2; h++) T.close(row(out, h * 7 + p, 4), row(ref, h, 4), 1e-5, `position ${p} must attend to keys ${lo}…${p} (the last 3, including itself)`);
    }
    const last = m.slidingWindowAttention(ops.slice(q, 1, 6, 7), k, v, 3);
    T.shape(last, [2, 1, 4], 'a decode query against 7 cached keys returns [H, 1, dh]');
    for (let h = 0; h < 2; h++) T.close(row(last, h, 4), row(out, h * 7 + 6, 4), 1e-5, 'a single query against Tk = 7 keys sits at position Tk − Tq + i = 6 and sees keys 4…6, as in mha; treating it as position 0 would see only key 0');
  } },
  { step: 'window', name: 'the ring cache holds the last W tokens, oldest first, in constant memory', run(m, T) {
    const H = 2, dh = 4, W = 4, N = 9;
    const q = rand(T, [H, N, dh], 56), k = rand(T, [H, N, dh], 57), v = rand(T, [H, N, dh], 58);
    const ref = m.slidingWindowAttention(q, k, v, W);
    const ring = new m.RingKVCache({ nHead: H, headDim: dh, window: W });
    const bufLen = ring.k.length;
    const bytes0 = ring.bytes(2);
    T.eq(bytes0, 2 * H * W * dh * 2, 'bytes(2) counts the allocated buffers, 2 · H · W · dh · 2, even before any token arrives');
    T.eq(ring.bytes(), 2 * H * W * dh * 4, 'bytes() defaults to 4 bytes per element (Float32Array)');
    T.eq(bufLen, H * W * dh, 'the buffer is allocated once: H · window · dh floats for keys');
    for (let t = 0; t < N; t++) {
      ring.append(ops.slice(k, 1, t, t + 1), ops.slice(v, 1, t, t + 1));
      const n = Math.min(t + 1, W);
      T.eq(ring.length, n, `after ${t + 1} appends the cache holds min(${t + 1}, ${W}) tokens`);
      const keys = ring.keys();
      T.shape(keys, [H, n, dh]);
      T.close(Array.from(keys.data), Array.from(ops.slice(k, 1, t + 1 - n, t + 1).data), 1e-6, `keys() must return tokens ${t + 1 - n}…${t} in order, oldest first`);
      T.eq(ring.bytes(2), bytes0, `after ${t + 1} appends bytes() must not change: the ring's memory is fixed at construction, not proportional to what it holds`);
      const out = m.mha(ops.slice(q, 1, t, t + 1), keys, ring.values());
      for (let h = 0; h < H; h++) T.close(row(out, h, dh), row(ref, h * N + t, dh), 1e-5, `decoding token ${t} against the ring must match sliding-window attention`);
    }
    T.eq(ring.k.length, bufLen, 'the buffer must never grow');
    T.eq(ring.bytes(2), 2 * H * W * dh * 2, 'bytes(2) = K and V · H · W · dh · 2 bytes, whatever the sequence length');
  } },
  { step: 'window', name: 'token p is written to slot p % W (Mistral\'s rolling buffer)', run(m, T) {
    const H = 2, dh = 3, W = 4;
    const k = rand(T, [H, 6, dh], 59), v = rand(T, [H, 6, dh], 60);
    const ring = new m.RingKVCache({ nHead: H, headDim: dh, window: W });
    for (let t = 0; t < 6; t++) ring.append(ops.slice(k, 1, t, t + 1), ops.slice(v, 1, t, t + 1));
    for (let h = 0; h < H; h++) {
      T.close(Array.from(ring.k.subarray((h * W + 0) * dh, (h * W + 1) * dh)), row(k, h * 6 + 4, dh), 1e-6, 'storage is [H, W, dh]; token 4 overwrote slot 4 % 4 = 0');
      T.close(Array.from(ring.v.subarray((h * W + 1) * dh, (h * W + 2) * dh)), row(v, h * 6 + 5, dh), 1e-6, 'token 5 overwrote slot 1 in the value buffer');
      T.close(Array.from(ring.k.subarray((h * W + 2) * dh, (h * W + 3) * dh)), row(k, h * 6 + 2, dh), 1e-6, 'slot 2 still holds token 2, the oldest one kept');
    }
  } },

  // ---------- step 5: context extension ----------
  { step: 'extend', name: 'PI divides every frequency; NTK raises the base', run(m, T) {
    const theta = Array.from(m.ropeFrequencies(8));
    T.close(Array.from(m.scaledRopeFrequencies(8, { method: 'none', factor: 4 })), theta, 1e-12, "'none' leaves the frequencies alone");
    T.close(Array.from(m.scaledRopeFrequencies(8, { method: 'pi', factor: 4 })), theta.map((t) => t / 4), 1e-12, "'pi' (Position Interpolation) divides every θ_i by the factor, squeezing 4x more positions into the trained angle range");
    const ntk = Array.from(m.scaledRopeFrequencies(8, { method: 'ntk', factor: 4 }));
    const b = 10000 * 4 ** (8 / 6);
    T.close(ntk, [0, 1, 2, 3].map((i) => b ** (-2 * i / 8)), 1e-12, "'ntk' uses base · factor^(dh/(dh−2))");
    T.close(ntk[0], 1, 1e-12, 'NTK leaves the fastest pair untouched (local detail survives)');
    T.close(ntk[3], theta[3] / 4, 1e-12, '…and slows the slowest pair by exactly the factor, like PI');
    T.throws(() => m.scaledRopeFrequencies(8, { method: 'linear', factor: 4 }), 'an unknown method must throw rather than silently return unscaled frequencies');
  } },
  { step: 'extend', name: 'YaRN interpolates slow pairs, keeps fast ones, ramps in between', run(m, T) {
    const theta = m.ropeFrequencies(128);
    const y = m.scaledRopeFrequencies(128, { method: 'yarn', factor: 4, trainLen: 4096 });
    T.eq(y.length, 64);
    T.close(y[0], theta[0], 1e-12, 'pair 0 turns about 650 times over 4096 tokens (> beta = 32): leave it alone');
    T.close(y[63], theta[63] / 4, 1e-15, 'the slowest pair turns less than once (< alpha = 1): interpolate it fully, θ/4');
    const i = 30;
    const turns = 4096 * theta[i] / (2 * Math.PI);
    const g = (turns - 1) / 31;
    T.ok(g > 0 && g < 1, 'pair 30 is in the ramp');
    T.close(y[i], (1 - g) * theta[i] / 4 + g * theta[i], 1e-12, 'in the ramp θ\' = (1 − γ)·θ/factor + γ·θ with γ = (turns − alpha)/(beta − alpha)');
  } },
  { step: 'extend', name: 'unseenRotation measures angles no pair reached in training', run(m, T) {
    const f = [1, 0.01];
    T.close(m.unseenRotation(f, f, 10, 10), 0, 1e-12, 'at the training length nothing is new');
    T.close(m.unseenRotation(f, f, 10, 100), 0.9 / (2 * Math.PI) / 2, 1e-9, 'pair 0 already covered the full circle (10 rad > 2π) so adds 0; pair 1 went from 0.1 rad to 1 rad: 0.9/(2π), averaged over 2 pairs');
    T.close(m.unseenRotation(f.map((t) => t / 10), f, 10, 100), 0, 1e-12, 'interpolated by the same factor as the length, no pair leaves its trained range');
    T.close(m.unseenRotation([0.01], [0.01], 10, 1000), (2 * Math.PI - 0.1) / (2 * Math.PI), 1e-9, 'a pair cannot be more than one full turn out of range: cap each angle at 2π');
    T.close(m.unseenRotation(f, f, 100, 10), 0, 1e-12, 'a context SHORTER than training reaches no new angle: clamp each pair at max(0, now − seen), never subtract');
    T.close(m.unseenRotation([1, 0.001], [1, 0.01], 10, 10), 0, 1e-12, 'a pair slowed below its trained speed reaches less, not negative, unseen rotation');
  } },

  // ---------- step 6: the budget ----------
  { step: 'budget', name: 'cache budget at Llama-3-8B dims and 32k tokens', run(m, T) {
    const rows = m.cacheBudget(LLAMA3_8B, 32768);
    T.eq(rows.map((r) => r.variant), ['MHA', 'GQA', 'MQA', 'MLA', 'SWA', 'HYBRID'], 'one row per variant, in this order');
    const by = Object.fromEntries(rows.map((r) => [r.variant, r]));
    T.eq(by.MHA.bytesPerToken, 524288, 'MHA: 2 · 32 · 32 · 128 · 2');
    T.eq(by.GQA.totalBytes, 4 * 2 ** 30, 'GQA with 8 KV heads: 128 KiB/token · 32,768 tokens = 4 GiB');
    T.eq(by.MQA.bytesPerToken, 16384, 'MQA keeps one KV head');
    T.eq(by.MLA.bytesPerToken, 36864, 'MLA: 32 layers · (512 + 64) · 2 bytes');
    T.eq(by.SWA.tokensHeld, 4096, 'the sliding window holds only the last 4096 tokens');
    T.eq(by.SWA.totalBytes, 512 * 2 ** 20, 'SWA on GQA: 4096 tokens · 128 KiB = 512 MiB');
    for (const r of rows) T.eq(r.totalBytes, r.bytesPerToken * r.tokensHeld + r.fixedBytes, `${r.variant}: totalBytes = bytesPerToken · tokensHeld + fixedBytes`);
    for (const v of ['MHA', 'GQA', 'MQA', 'MLA', 'SWA']) T.eq(by[v].fixedBytes, 0, `${v} has no constant state: fixedBytes 0`);
    T.eq(by.HYBRID.bytesPerToken, 32768, 'HYBRID: only 32 / 4 = 8 layers are full attention with 8 KV heads: 2 · 8 · 8 · 128 · 2 bytes');
    T.eq(by.HYBRID.fixedBytes, 24 * 32 * 128 * 128 * 2, 'HYBRID: the other 24 layers each keep a 128 × 128 state per head for 32 heads, in bf16: 24 MiB');
    T.eq(by.HYBRID.totalBytes, 2 ** 30 + 24 * 2 ** 20, 'HYBRID at 32k tokens: 32 KiB · 32,768 = 1 GiB of cache plus 24 MiB of state');
    const jamba = m.cacheBudget({ ...LLAMA3_8B, fullAttnEvery: 8 }, 1024).find((r) => r.variant === 'HYBRID');
    T.eq(jamba.bytesPerToken, 16384, 'fullAttnEvery = 8 (Jamba\'s 1-in-8 ratio) leaves 4 attention layers');
    T.eq(jamba.fixedBytes, 28 * 32 * 128 * 128 * 2, '…and 28 recurrent layers of state');
  } },
  { step: 'budget', name: 'every cache grows linearly with context except the sliding window', run(m, T) {
    const names = (m.cacheBudget(LLAMA3_8B, 2048) || []).map((r) => r && r.variant);
    for (const vname of ['MHA', 'GQA', 'MQA', 'MLA', 'SWA', 'HYBRID']) {
      T.ok(names.includes(vname), `no row with variant '${vname}' (got variants [${names.map((n) => JSON.stringify(n)).join(', ')}]); rows are looked up by name, upper-case, in the order MHA, GQA, MQA, MLA, SWA, HYBRID`);
    }
    const short = Object.fromEntries(m.cacheBudget(LLAMA3_8B, 2048).map((r) => [r.variant, r.totalBytes]));
    const long = Object.fromEntries(m.cacheBudget(LLAMA3_8B, 65536).map((r) => [r.variant, r.totalBytes]));
    T.eq(short.SWA, short.GQA, 'below the window, sliding-window attention caches exactly what GQA caches');
    for (const vname of ['MHA', 'GQA', 'MQA', 'MLA']) T.eq(long[vname], short[vname] * 32, `${vname}: 32x the context costs 32x the cache`);
    T.eq(long.SWA, m.cacheBudget(LLAMA3_8B, 8192).find((r) => r.variant === 'SWA').totalBytes, 'beyond the window the SWA cache is constant');
    T.eq(long.HYBRID - short.HYBRID, (65536 - 2048) * 32768, 'HYBRID grows only through its 8 attention layers; the recurrent state is the same at 2k and 64k tokens');
  } },
];
