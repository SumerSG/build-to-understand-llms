import { GPT } from 'lib/gpt.js';
import { loadModel, forward as refForward } from 'lib/infer.js';
import * as ops from 'lib/ops.js';

const TINY = { vocabSize: 16, blockSize: 16, nLayer: 2, nHead: 2, nEmbd: 8 };

/**
 * A random-weight model as the learner's functions see it: { config, w }. `boost` scales the position
 * embedding and the qkv projections so that the logits change from position to position (a fresh GPT
 * is initialised so small that greedy decoding repeats one token forever, which would make the
 * cached-equals-uncached test vacuous).
 */
function testModel(cfg = TINY, seed = 1, boost = 1) {
  const model = loadModel(new GPT({ ...cfg, seed }).toJSON());
  if (boost !== 1) {
    for (const name of Object.keys(model.w)) {
      if (name === 'wpe.weight' || name.endsWith('attn.qkv.weight')) {
        const d = model.w[name].data;
        for (let i = 0; i < d.length; i++) d[i] *= boost;
      }
    }
  }
  return model;
}

function randomTensor(shape, next, scale = 1) {
  const t = ops.zeros(shape);
  for (let i = 0; i < t.data.length; i++) t.data[i] = (next() * 2 - 1) * scale;
  return t;
}

/** Plain-loop reference for one query against t keys: softmax(q·kᵀ / sqrt(dh)) · v, per head. */
function attendOneReference(q, k, v, scaled = true) {
  const [H, t, dh] = k.shape;
  const out = ops.zeros([H, 1, dh]);
  const factor = scaled ? 1 / Math.sqrt(dh) : 1;
  for (let h = 0; h < H; h++) {
    const scores = [];
    for (let j = 0; j < t; j++) {
      let s = 0;
      for (let d = 0; d < dh; d++) s += q.data[h * dh + d] * k.data[(h * t + j) * dh + d];
      scores.push(s * factor);
    }
    const m = Math.max(...scores);
    const e = scores.map((s) => Math.exp(s - m));
    const z = e.reduce((a, b) => a + b, 0);
    for (let j = 0; j < t; j++) for (let d = 0; d < dh; d++) out.data[h * dh + d] += (e[j] / z) * v.data[(h * t + j) * dh + d];
  }
  return out;
}

/**
 * A copy of `model` whose position table counts how many rows are looked up. ops.embed reads one row per
 * embedded position with table.data.subarray, so the counter is the number of token-positions pushed
 * through the model: 1 per forwardStep, T per uncached forward over T tokens. This is how the tests tell
 * a real cache from a full recompute that happens to give the same numbers.
 */
function countingModel(model) {
  const counter = { rows: 0 };
  class CountingF32 extends Float32Array {
    subarray(...a) { counter.rows++; return super.subarray(...a); }
  }
  const wpe = model.w['wpe.weight'];
  const data = new CountingF32(wpe.data.length);
  data.set(wpe.data);
  return { model: { ...model, w: { ...model.w, 'wpe.weight': { shape: wpe.shape.slice(), data } } }, counter };
}

/** Layer-0 keys and values for `ids`, [H, T, dh] each, computed directly from the weights (no cache). */
function layer0KV(model, ids) {
  const { nHead, nEmbd } = model.config;
  const w = model.w;
  const dh = nEmbd / nHead, T = ids.length;
  const x = ops.add(ops.embed(w['wte.weight'], ids), ops.embed(w['wpe.weight'], ids.map((_, i) => i)));
  const normed = ops.layerNorm(x, w['blocks.0.ln1.gamma'], w['blocks.0.ln1.beta']);
  const qkv = ops.add(ops.matmul(normed, w['blocks.0.attn.qkv.weight']), w['blocks.0.attn.qkv.bias']);
  const heads = (from) => ops.permute(ops.reshape(ops.slice(qkv, 1, from, from + nEmbd), [T, nHead, dh]), [1, 0, 2]);
  return { k: heads(nEmbd), v: heads(2 * nEmbd) };
}

const lastRow = (logits2d, ids) => Array.from(ops.slice(logits2d, 0, ids.length - 1, ids.length).data);
const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };

export const tests = [
  // ---------- step 1: attendOne ----------
  { step: 'attend', name: 'attendOne is softmax(q·kᵀ / sqrt(dh)) · v for one query over t stored keys', run(m, T) {
    const next = T.rng(11);
    const H = 2, t = 5, dh = 4;
    const q = randomTensor([H, 1, dh], next, 2), k = randomTensor([H, t, dh], next, 2), v = randomTensor([H, t, dh], next);
    const out = m.attendOne(q, k, v);
    T.shape(out, [H, 1, dh], 'one query per head must give one output row per head: [H, 1, dh]');
    const unscaled = attendOneReference(q, k, v, false);
    T.ok(!ops.allClose(out, unscaled, 1e-3), 'scores must be scaled by 1/sqrt(dh) before the softmax, as in every attention layer of the lab; the unscaled result differs');
    T.close(out, attendOneReference(q, k, v), 1e-5, 'the weighted sum of the stored values, with softmax weights over the scaled dot products');
  } },
  { step: 'attend', name: 'a single stored key gets weight 1, and identical keys average the values (weights must sum to 1)', run(m, T) {
    const next = T.rng(12);
    const H = 2, dh = 4;
    const q = randomTensor([H, 1, dh], next, 3);
    const v1 = randomTensor([H, 1, dh], next);
    T.close(m.attendOne(q, randomTensor([H, 1, dh], next, 3), v1), v1, 1e-6, 'with one key the softmax has one entry, so the output is exactly that value regardless of q');
    const t = 3;
    const v = randomTensor([H, t, dh], next);
    const mean = ops.zeros([H, 1, dh]);
    for (let h = 0; h < H; h++) for (let d = 0; d < dh; d++) {
      let s = 0;
      for (let j = 0; j < t; j++) s += v.data[(h * t + j) * dh + d];
      mean.data[h * dh + d] = s / t;
    }
    T.close(m.attendOne(q, ops.zeros([H, t, dh]), v), mean, 1e-6, 'all-zero keys give equal scores, so the output must be the MEAN of the values: softmax weights sum to 1, they are not raw exponentials');
  } },

  // ---------- step 2: the cache ----------
  { step: 'cache', name: 'newCache has an empty [H, 0, dh] key block and value block per layer, length 0 and maxLength = blockSize', run(m, T) {
    const model = testModel();
    const cache = m.newCache(model);
    T.ok(cache && Array.isArray(cache.k) && Array.isArray(cache.v), 'the cache is { k: raw[], v: raw[], length, maxLength }');
    T.eq(cache.k.length, TINY.nLayer, 'one key block per layer: every layer has its own keys');
    T.eq(cache.v.length, TINY.nLayer, 'one value block per layer');
    for (let l = 0; l < TINY.nLayer; l++) {
      T.shape(cache.k[l], [2, 0, 4], `layer ${l} keys start as [nHead, 0, headDim] with headDim = nEmbd / nHead = 4`);
      T.shape(cache.v[l], [2, 0, 4], `layer ${l} values start as [nHead, 0, headDim]`);
    }
    T.eq(cache.length, 0, 'no tokens stored yet');
    T.eq(cache.maxLength, TINY.blockSize, 'the position table has blockSize rows, so the cache can never hold more');
    const other = m.newCache(testModel({ ...TINY, nLayer: 3, nHead: 4 }));
    T.eq(other.k.length, 3, 'the number of blocks must follow the model config, not a constant');
    T.shape(other.k[0], [4, 0, 2], 'headDim follows nEmbd / nHead');
  } },
  { step: 'cache', name: 'appendKV grows the time axis in arrival order, leaves other layers alone and does not advance length', run(m, T) {
    const next = T.rng(21);
    const model = testModel();
    const cache = m.newCache(model);
    const k1 = randomTensor([2, 1, 4], next), v1 = randomTensor([2, 1, 4], next);
    const k2 = randomTensor([2, 1, 4], next), v2 = randomTensor([2, 1, 4], next);
    m.appendKV(cache, 1, k1, v1);
    const ret = m.appendKV(cache, 1, k2, v2);
    T.shape(cache.k[1], [2, 2, 4], 'two appended positions give [H, 2, dh]: concatenate along the TIME axis (axis 1), not the head axis');
    T.shape(cache.v[1], [2, 2, 4], 'values grow exactly like keys');
    for (let h = 0; h < 2; h++) {
      T.close(Array.from(cache.k[1].data.subarray((h * 2 + 0) * 4, (h * 2 + 1) * 4)), Array.from(k1.data.subarray(h * 4, h * 4 + 4)), 1e-6, `head ${h}, time 0 must hold the first appended key`);
      T.close(Array.from(cache.k[1].data.subarray((h * 2 + 1) * 4, (h * 2 + 2) * 4)), Array.from(k2.data.subarray(h * 4, h * 4 + 4)), 1e-6, `head ${h}, time 1 must hold the second appended key: order is arrival order`);
      T.close(Array.from(cache.v[1].data.subarray((h * 2 + 1) * 4, (h * 2 + 2) * 4)), Array.from(v2.data.subarray(h * 4, h * 4 + 4)), 1e-6, `head ${h}, time 1 of the values`);
    }
    T.shape(cache.k[0], [2, 0, 4], 'appending to layer 1 must not touch layer 0');
    T.eq(cache.length, 0, 'appendKV must not advance cache.length: it is bumped once per token by forwardStep, after every layer');
    T.ok(ret && ret.k && ret.v && ret.k.shape && ret.k.shape[1] === 2, 'return the layer\'s full { k, v } after the append so the caller can attend over it');
    m.appendKV(cache, 1, randomTensor([2, 2, 4], next), randomTensor([2, 2, 4], next));
    T.shape(cache.k[1], [2, 4, 4], 'a block of 2 positions appended to 2 stored ones gives 4');
  } },

  // ---------- step 3: forwardStep ----------
  { step: 'step', name: 'forwardStep reproduces the uncached forward at every position within 1e-4', run(m, T) {
    const model = testModel(TINY, 3, 8);
    const ids = [1, 5, 3, 9, 2, 7, 3];
    const full = refForward(model, ids); // [T, V] from lib/infer.js
    const cache = m.newCache(model);
    for (let t = 0; t < ids.length; t++) {
      const logits = m.forwardStep(model, cache, ids[t]);
      T.ok(logits instanceof Float32Array && logits.length === TINY.vocabSize, `forwardStep must return a Float32Array of length vocabSize = ${TINY.vocabSize} (the logits for the next token)`);
      T.close(Array.from(logits), Array.from(ops.slice(full, 0, t, t + 1).data), 1e-4, `logits after token ${t} must equal row ${t} of forward(model, ids): the cache changes the cost, never the answer`);
      T.eq(cache.length, t + 1, 'cache.length must advance by exactly one per decoded token');
    }
    for (let l = 0; l < TINY.nLayer; l++) T.shape(cache.k[l], [2, ids.length, 4], `layer ${l} must hold one key per token seen`);
  } },
  { step: 'step', name: 'forwardStep does one position of work, stores the real keys and values, and reads them back', run(m, T) {
    const base = testModel(TINY, 6, 8);
    const { model, counter } = countingModel(base);
    const ids = [3, 8, 1, 14, 6];
    const cache = m.newCache(model);
    for (let t = 0; t < ids.length; t++) {
      const before = counter.rows;
      m.forwardStep(model, cache, ids[t]);
      T.ok(counter.rows - before <= 1, `forwardStep for token ${t} embedded ${counter.rows - before} positions: a decode step embeds ONE token at position cache.length, it must not re-run the whole sequence through forward()`);
    }
    const ref = layer0KV(base, ids);
    T.close(cache.k[0], ref.k, 1e-5, 'cache.k[0] must hold the layer-0 keys of every token so far, [H, T, dh] in arrival order (not placeholders)');
    T.close(cache.v[0], ref.v, 1e-5, 'cache.v[0] must hold the layer-0 values of every token so far');

    // Poison the stored values of a second, identical cache: a step that really reads the cache must change.
    const clean = m.newCache(model), poisoned = m.newCache(model);
    for (const id of ids) { m.forwardStep(model, clean, id); m.forwardStep(model, poisoned, id); }
    for (let l = 0; l < TINY.nLayer; l++) poisoned.v[l] = ops.zeros(poisoned.v[l].shape);
    const a = m.forwardStep(model, clean, 2), b = m.forwardStep(model, poisoned, 2);
    let diff = 0;
    for (let i = 0; i < a.length; i++) diff = Math.max(diff, Math.abs(a[i] - b[i]));
    T.ok(diff > 1e-3, 'zeroing the stored values in cache.v did not change the next logits: forwardStep must attend over the k/v stored in the cache (what appendKV returns), not recompute them from the token ids');
  } },
  { step: 'step', name: 'position and history matter: the same token at a new position gives new logits, and a full cache throws', run(m, T) {
    const model = testModel({ ...TINY, blockSize: 4 }, 4, 8);
    const cache = m.newCache(model);
    const first = Array.from(m.forwardStep(model, cache, 3));
    const second = Array.from(m.forwardStep(model, cache, 3));
    let diff = 0;
    for (let i = 0; i < first.length; i++) diff = Math.max(diff, Math.abs(first[i] - second[i]));
    T.ok(diff > 1e-3, 'the same token id at position 1 must not give the logits it gave at position 0: embed it at position cache.length and attend over the earlier key too');
    m.forwardStep(model, cache, 1);
    m.forwardStep(model, cache, 2);
    T.eq(cache.length, 4);
    T.throws(() => m.forwardStep(model, cache, 0), 'a 5th token cannot be placed: there is no position embedding for it, so forwardStep must throw when cache.length >= maxLength');
  } },

  // ---------- step 4: prefill and generation ----------
  { step: 'prefill', name: 'prefill returns the logits after the last prompt token and leaves the whole prompt in the cache', run(m, T) {
    const model = testModel(TINY, 5, 8);
    const ids = [4, 4, 1, 12, 7];
    const cache = m.newCache(model);
    const logits = m.prefill(model, cache, ids);
    T.ok(logits instanceof Float32Array && logits.length === TINY.vocabSize, 'prefill returns one Float32Array [V]: the distribution for the first generated token');
    T.close(Array.from(logits), lastRow(refForward(model, ids), ids), 1e-4, 'must equal the LAST row of the uncached forward over the prompt');
    T.eq(cache.length, ids.length, 'after prefill the cache holds every prompt token');
    T.shape(cache.k[0], [2, ids.length, 4], 'every layer stores a key per prompt token');
    T.throws(() => m.prefill(model, m.newCache(model), []), 'an empty prompt has no last token to return logits for; throw rather than return null');
  } },
  { step: 'prefill', name: 'generateGreedy: the cached and uncached paths produce identical tokens, matching an independent greedy loop', run(m, T) {
    const model = testModel(TINY, 2, 12);
    const prompt = [2, 9];
    const n = 10;
    const ref = [];
    const ids = prompt.slice();
    for (let i = 0; i < n; i++) { const id = argmax(lastRow(refForward(model, ids), ids)); ref.push(id); ids.push(id); }
    T.ok(new Set(ref).size >= 3, 'test model sanity: greedy output should use several distinct tokens');
    const cached = m.generateGreedy(model, prompt, n, { cached: true });
    const uncached = m.generateGreedy(model, prompt, n, { cached: false });
    T.eq(cached.length, n, 'return exactly maxNewTokens new ids (the prompt is not part of the result)');
    T.eq(uncached, ref, 'uncached path: take the argmax of the LAST row of forward over prompt + everything generated so far');
    T.eq(cached, ref, 'cached path: prefill the prompt once, then one forwardStep per generated token, feeding back the argmax; it must match token for token');
    T.eq(prompt, [2, 9], 'the prompt array must not be mutated');
  } },
  { step: 'prefill', name: 'generateGreedy: the cached path pushes each position through the model once; the uncached path recomputes', run(m, T) {
    const { model, counter } = countingModel(testModel(TINY, 2, 12));
    const prompt = [2, 9, 4];
    const n = 6;
    counter.rows = 0;
    m.generateGreedy(model, prompt, n, { cached: true });
    const cachedRows = counter.rows;
    T.ok(cachedRows <= prompt.length + n - 1, `the cached path embedded ${cachedRows} positions; prefill embeds the ${prompt.length} prompt tokens once and each forwardStep embeds one new token, so at most ${prompt.length} + ${n} − 1 = ${prompt.length + n - 1}. More means it re-runs forward() over the sequence instead of using the cache`);
    counter.rows = 0;
    m.generateGreedy(model, prompt, n, { cached: false });
    let expected = 0;
    for (let i = 0; i < n; i++) expected += prompt.length + i;
    T.eq(counter.rows, expected, `the uncached path must call forward over prompt + generated-so-far once per new token: ${prompt.length} + ${prompt.length + 1} + … = ${expected} positions embedded in total`);
  } },
  { step: 'prefill', name: 'the cached path fits exactly at the cache limit (no wasted step after the last token) and throws one token past it', run(m, T) {
    const model = testModel({ ...TINY, blockSize: 6 }, 10, 12);
    const out = m.generateGreedy(model, [1, 2, 3], 4, { cached: true });
    T.eq(out.length, 4, 'a 3-token prompt plus 4 new tokens: the 4th is chosen from the logits of position 5, the last slot, so no 7th position is needed and this must not throw (do not run a forwardStep after the final token)');
    T.eq(out, m.generateGreedy(model, [1, 2, 3], 4, { cached: false }), 'the uncached path agrees at the boundary too');
    T.throws(() => m.generateGreedy(model, [1, 2, 3], 5, { cached: true }), 'a 5th new token would need logits from a 7th position, which does not exist: forwardStep must throw and generateGreedy must let it propagate');
  } },

  // ---------- step 5: the cost model ----------
  { step: 'cost', name: 'paramCount equals GPT.numParams() from lib/gpt.js for two configs', run(m, T) {
    T.eq(m.paramCount(TINY), new GPT({ ...TINY, seed: 0 }).numParams(), 'V·C + B·C + L·(12C² + 13C) + 2C with B = blockSize (the wpe rows, not a context length): embeddings, per-block weights, biases and LayerNorms, then the final LayerNorm (the head is tied to wte, so it is not counted twice)');
    const bigger = { vocabSize: 256, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 64 };
    T.eq(m.paramCount(bigger), new GPT({ ...bigger, seed: 0 }).numParams(), 'the checkpoint config (256 vocab, 64 wide, 2 layers) must count correctly too');
  } },
  { step: 'cost', name: 'flopsPerToken: cached = 2N + 4·L·T·C and grows linearly with T; uncached is T times that', run(m, T) {
    const N = new GPT({ ...TINY, seed: 0 }).numParams();
    for (const t of [1, 8, 16]) {
      T.eq(m.flopsPerToken(TINY, t, { cached: true }), 2 * N + 4 * TINY.nLayer * t * TINY.nEmbd, `cached cost at context ${t}: 2 FLOPs per parameter plus 4·L·T·C for the attention over T stored keys`);
      T.eq(m.flopsPerToken(TINY, t, { cached: false }), t * (2 * N + 4 * TINY.nLayer * t * TINY.nEmbd), `uncached cost at context ${t}: the whole context is recomputed, so multiply by T`);
    }
    T.eq(m.flopsPerToken(TINY, 5), m.flopsPerToken(TINY, 5, { cached: true }), 'cached is the default');
    const ratio = m.flopsPerToken(TINY, 16, { cached: false }) / m.flopsPerToken(TINY, 16, { cached: true });
    T.close(ratio, 16, 1e-9, 'at context 16 the uncached path does 16x the work of the cached one');
  } },
  { step: 'cost', name: 'cacheBytes is 2·L·nKVHead·headDim·T·bytes, and Llama-3-8B (32 layers, 8 KV heads, headDim 128, bf16) needs 131072 bytes per token', run(m, T) {
    T.eq(m.cacheBytes(TINY, 1), 2 * 2 * 2 * 4 * 1 * 4, 'tiny config, 1 token, float32: 2 (K and V) · 2 layers · 2 heads · 4 headDim · 4 bytes');
    T.eq(m.cacheBytes(TINY, 10), 10 * m.cacheBytes(TINY, 1), 'linear in the context length');
    T.eq(m.cacheBytes(TINY, 10, { bytesPerElement: 2 }), m.cacheBytes(TINY, 10) / 2, 'bf16 halves it');
    const llama = { nLayer: 32, nHead: 32, nKVHead: 8, nEmbd: 4096 };
    T.eq(m.cacheBytes(llama, 1, { bytesPerElement: 2 }), 131072, 'with GQA only nKVHead = 8 heads of keys and values are stored, not all 32 query heads');
    T.eq(m.cacheBytes(llama, 131072, { bytesPerElement: 2 }), 17179869184, '128k tokens of context is 16 GiB of cache for a single sequence');
    T.eq(m.cacheBytes({ nLayer: 32, nHead: 32, nEmbd: 4096 }, 1, { bytesPerElement: 2 }), 524288, 'without nKVHead every query head has its own keys and values (MHA): 4x the GQA figure');
  } },
];
