// app/chat-worker.js — runs the lab model for the Chat playground: tokenizer + inference model +
// KV cache reused across turns + the sampling pipeline. Streams tokens back to the page.
import { rewriteImports } from './rewrite.js';

let lib = null;        // { infer, sampling, tokenizer: BPETokenizer instance, data }
let model = null;      // InferModel
let cache = null;      // KV cache reused across turns
let cachedIds = [];    // token ids currently represented in `cache`
let base = null;

async function loadLib(b) {
  base = b;
  const [infer, sampling, tok, data] = await Promise.all([
    import(`${b}/lib/infer.js`), import(`${b}/lib/sampling.js`), import(`${b}/lib/tokenizer.js`), import(`${b}/lib/data.js`),
  ]);
  lib = { infer, sampling, tokmod: tok, data };
}

async function loadCheckpoint(modelJson, tokJson) {
  model = lib.infer.loadModel(modelJson);
  lib.tokenizer = lib.tokmod.BPETokenizer.fromJSON(tokJson);
  cache = null;
  cachedIds = [];
  const params = Object.values(model.w).reduce((s, t) => s + t.data.length, 0);
  return { config: model.config, params, vocab: lib.tokenizer.vocabSize };
}

/** Keep only the first n positions of a KV cache: every layer's [H, T, headDim] keys and values. */
function truncateCache(c, n) {
  for (const kv of [c.k, c.v]) {
    for (let layer = 0; layer < kv.length; layer++) {
      const [H, T, hd] = kv[layer].shape;
      const data = new Float32Array(H * n * hd);
      for (let h = 0; h < H; h++) data.set(kv[layer].data.subarray(h * T * hd, h * T * hd + n * hd), h * n * hd);
      kv[layer] = { shape: [H, n, hd], data };
    }
  }
  c.length = n;
}

/** Longest common prefix between the cached ids and the new prompt ids. */
function sharedPrefix(a, b) {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return i;
}

async function generate({ messages, opts }) {
  const { infer, sampling, tokenizer, data } = lib;
  const rng = (await import(`${base}/lib/util.js`)).rng(opts.seed ?? 1);
  const prompt = data.formatChat(messages);
  let ids = tokenizer.encode(prompt);
  const blockSize = model.config.blockSize;
  // Keep the prompt inside the context window (drop from the front, keeping the newest turns).
  const maxPrompt = Math.max(8, blockSize - Math.min(opts.maxNewTokens, blockSize - 8));
  let trimmed = false;
  if (ids.length > maxPrompt) { ids = ids.slice(ids.length - maxPrompt); trimmed = true; }
  // Reuse the KV cache for the shared prefix; prefill only the new suffix. The cache may hold more than
  // the shared prefix (the reply as generated can tokenize differently from the reply as re-encoded), so
  // keep the matching part and drop the rest. Dropping old turns shifts every position, so nothing matches.
  let reused = 0;
  if (cache && cachedIds.length) {
    reused = Math.min(sharedPrefix(cachedIds, ids), ids.length - 1);
    if (reused <= 0) { cache = null; cachedIds = []; reused = 0; }
    else if (reused < cachedIds.length) { truncateCache(cache, reused); cachedIds = cachedIds.slice(0, reused); }
  }
  if (!cache) cache = infer.newCache(model);
  const t0 = performance.now();
  let logits = null;
  for (let i = reused; i < ids.length; i++) {
    logits = infer.forwardStep(model, cache, ids[i]);
  }
  cachedIds = ids.slice();
  if (!logits) logits = infer.forwardStep(model, cache, ids[ids.length - 1]);
  const prefillMs = performance.now() - t0;
  self.postMessage({ type: 'prefill', promptTokens: ids.length, reusedTokens: reused, ms: prefillMs });
  if (trimmed) self.postMessage({ type: 'note', text: `the conversation is longer than the ${blockSize}-token context window, so the oldest tokens were dropped; every position shifted, so the cache was rebuilt` });
  const out = [];
  const t1 = performance.now();
  const endId = tokenizer.encode(data.CHAT.end)[0];
  for (let n = 0; n < opts.maxNewTokens; n++) {
    if (cachedIds.length >= blockSize) { self.postMessage({ type: 'note', text: 'context window full' }); break; }
    const id = sampling.sample(logits, { temperature: opts.temperature, topK: opts.topK, topP: opts.topP, next: rng });
    if (id === tokenizer.eos || id === endId) break;
    out.push(id);
    cachedIds.push(id);
    self.postMessage({ type: 'token', text: tokenizer.decode([id]) });
    logits = infer.forwardStep(model, cache, id);
    if (n % 4 === 0) await new Promise((r) => setTimeout(r, 0));
  }
  self.postMessage({ type: 'done', text: tokenizer.decode(out), tokens: out.length, ms: performance.now() - t1 });
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      await loadLib(msg.base);
      const [m, t] = await Promise.all([
        fetch(`${msg.base}/lib/checkpoints/tiny-gpt.json`).then((r) => r.json()),
        fetch(`${msg.base}/lib/checkpoints/tokenizer.json`).then((r) => r.json()),
      ]);
      self.postMessage({ type: 'ready', info: await loadCheckpoint(m, t) });
    } else if (msg.type === 'load') {
      self.postMessage({ type: 'ready', info: await loadCheckpoint(msg.model, msg.tokenizer || (await fetch(`${base}/lib/checkpoints/tokenizer.json`).then((r) => r.json()))) });
    } else if (msg.type === 'generate') {
      await generate(msg);
    } else if (msg.type === 'reset') {
      cache = null; cachedIds = [];
      self.postMessage({ type: 'reset-done' });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err && err.message ? err.message : String(err) });
  }
};

// rewriteImports is imported so this worker shares the same module graph as the sandbox; unused directly.
void rewriteImports;
