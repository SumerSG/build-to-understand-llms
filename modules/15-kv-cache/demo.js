// Module 15 demo — generate from the pre-trained checkpoint twice, with and without YOUR cache, timing
// every token; confirm the two paths agree token for token; then size the cache for real models.
// If the checkpoint cannot be loaded (mid-retrain, or a browser without JSON imports) the demo falls back
// to a random-weight model: the text is then noise, the mechanics and the equality check are the same.
import { loadModel } from 'lib/infer.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { GPT } from 'lib/gpt.js';
import { toyCorpus } from 'lib/data.js';
import * as ops from 'lib/ops.js';
import { now } from 'lib/util.js';

const PROMPT = 'The cat';

async function loadCheckpoint(lab) {
  try {
    const json = (await import('lib/checkpoints/tiny-gpt.json', { with: { type: 'json' } })).default;
    const tok = (await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default;
    if (!json || !json.params || !json.config) throw new Error('checkpoint has no params');
    return { model: loadModel(json), tokenizer: BPETokenizer.fromJSON(tok), trained: true };
  } catch (e) {
    lab.log(`Checkpoint not available (${String(e.message).split(' imported from')[0].slice(0, 120)}); using a random-weight model instead.`);
    const tokenizer = BPETokenizer.train(toyCorpus(200, 1), { vocabSize: 256 });
    const gpt = new GPT({ vocabSize: tokenizer.vocabSize, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 64, seed: 3 });
    return { model: loadModel(gpt.toJSON()), tokenizer, trained: false };
  }
}

const fmtBytes = (b) => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(1)} GiB` : b >= 2 ** 20 ? `${(b / 2 ** 20).toFixed(1)} MiB` : `${(b / 1024).toFixed(1)} KiB`);
const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
const show = (text) => text.replace(/\n/g, ' ⏎ ').trim();

export default async function demo(m, lab) {
  const { model, tokenizer, trained } = await loadCheckpoint(lab);
  const cfg = model.config;
  const promptIds = tokenizer.encode(PROMPT);
  const NEW = cfg.blockSize - promptIds.length; // every slot the position table allows
  lab.log(`Model: ${cfg.nLayer} layers, ${cfg.nHead} heads, ${cfg.nEmbd} wide, vocab ${cfg.vocabSize}, context ${cfg.blockSize}${trained ? ' (pre-trained checkpoint)' : ' (random weights)'}.`);
  lab.log(`Prompt "${PROMPT}" is ${promptIds.length} token(s); generating ${NEW} tokens greedily, twice.`);

  // ---------- 0. warm-up: the first calls to any function pay for JIT compilation ----------
  // Without this, that one-off cost lands in whichever phase runs first (the uncached loop, and the
  // cached prefill, which is the first use of forwardStep) and makes a 2-token prefill look ~10x dearer
  // per token than a decode step. In this toy they are the same work. Run both paths once, untimed.
  {
    const scratch = m.newCache(model);
    let warm = m.prefill(model, scratch, promptIds);
    for (let i = 0; i < 4 && scratch.length < cfg.blockSize; i++) warm = m.forwardStep(model, scratch, m.argmaxOf(warm));
    m.forward(model, promptIds);
  }

  // ---------- 1. uncached: recompute the whole sequence for every token ----------
  const uncachedMs = [];
  const uncachedIds = [];
  let ids = promptIds.slice();
  let t0 = now();
  for (let i = 0; i < NEW; i++) {
    const ts = now();
    const all = m.forward(model, ids); // [T, V]
    const id = m.argmaxOf(ops.slice(all, 0, ids.length - 1, ids.length).data);
    uncachedMs.push(now() - ts);
    uncachedIds.push(id);
    ids.push(id);
    if (i % 8 === 7) { lab.progress(0.5 * (i + 1) / NEW, `uncached token ${i + 1}/${NEW}`); await lab.tick(); }
  }
  const uncachedTotal = now() - t0;

  // ---------- 2. cached: prefill once, then one step per token ----------
  const cachedMs = [];
  const cachedIds = [];
  const cache = m.newCache(model);
  t0 = now();
  let logits = m.prefill(model, cache, promptIds);
  const prefillMs = now() - t0;
  for (let i = 0; i < NEW; i++) {
    const id = m.argmaxOf(logits);
    cachedIds.push(id);
    if (i === NEW - 1) { cachedMs.push(0); break; } // the last token needs no step of its own
    const ts = now();
    logits = m.forwardStep(model, cache, id);
    cachedMs.push(now() - ts);
    if (i % 8 === 7) { lab.progress(0.5 + 0.5 * (i + 1) / NEW, `cached token ${i + 1}/${NEW}`); await lab.tick(); }
  }
  const cachedTotal = now() - t0;

  // ---------- 3. equality: the cache must change the cost, never the answer ----------
  const same = cachedIds.length === uncachedIds.length && cachedIds.every((id, i) => id === uncachedIds[i]);
  lab.check(same, `cached and uncached greedy decoding diverged at token ${cachedIds.findIndex((id, i) => id !== uncachedIds[i])}`);
  const viaGenerate = m.generateGreedy(model, promptIds, NEW, { cached: true });
  lab.check(viaGenerate.every((id, i) => id === uncachedIds[i]), 'generateGreedy(cached) disagrees with the uncached loop');
  const text = tokenizer.decode([...promptIds, ...cachedIds]);
  lab.md(`Both paths produced the same **${NEW}** tokens: "${show(text).slice(0, 160)}${text.length > 160 ? '…' : ''}"`);

  // ---------- 4. per-token latency ----------
  const x = Array.from({ length: NEW }, (_, i) => promptIds.length + i + 1); // context length when the token was produced
  lab.plot({
    title: 'Per-token latency: full recompute vs your KV cache',
    x, series: [{ name: 'uncached forward()', values: uncachedMs }, { name: 'cached forwardStep()', values: cachedMs }],
    xlabel: 'tokens in context', ylabel: 'ms per token',
  });
  const flopsUncached = x.map((t) => m.flopsPerToken(cfg, t, { cached: false }));
  const flopsCached = x.map((t) => m.flopsPerToken(cfg, t, { cached: true }));
  lab.plot({
    title: 'FLOPs per generated token (your cost model)',
    x, series: [{ name: 'uncached', values: flopsUncached }, { name: 'cached', values: flopsCached }],
    xlabel: 'tokens in context', ylabel: 'FLOPs', yscale: 'log',
  });
  const flopRatio = flopsUncached.at(-1) / flopsCached.at(-1);
  const meanUncached = uncachedMs.reduce((a, b) => a + b, 0) / NEW;
  const meanCached = cachedMs.slice(0, -1).reduce((a, b) => a + b, 0) / (NEW - 1);

  // ---------- 5. cache size vs context, for this model and for real ones ----------
  const contexts = [64, 1024, 8192, 32768, 131072];
  const models = [
    { name: `tiny-gpt (this)`, cfg: { nLayer: cfg.nLayer, nHead: cfg.nHead, nEmbd: cfg.nEmbd }, bytes: 4 },
    { name: 'GPT-2 small', cfg: { nLayer: 12, nHead: 12, nEmbd: 768 }, bytes: 2 },
    { name: 'Llama-3-8B (GQA)', cfg: { nLayer: 32, nHead: 32, nKVHead: 8, nEmbd: 4096 }, bytes: 2 },
    { name: 'Llama-3-70B (GQA)', cfg: { nLayer: 80, nHead: 64, nKVHead: 8, nEmbd: 8192 }, bytes: 2 },
  ];
  const sizes = models.map((r) => contexts.map((t) => m.cacheBytes(r.cfg, t, { bytesPerElement: r.bytes })));
  lab.heatmap({
    title: 'KV cache per sequence, log10(bytes): rows are models, columns are context lengths',
    rows: sizes.map((r) => r.map((b) => +Math.log10(b).toFixed(2))),
    rowLabels: models.map((r) => r.name), colLabels: contexts.map((t) => (t >= 1024 ? `${t / 1024}k` : String(t))),
  });
  lab.table({
    title: 'The same table in bytes (bf16 for the real models, float32 for this one)',
    columns: ['model', 'bytes / token', ...contexts.map((t) => (t >= 1024 ? `${t / 1024}k ctx` : `${t} ctx`))],
    rows: models.map((r, i) => [r.name, fmtBytes(sizes[i][0] / contexts[0]), ...sizes[i].map(fmtBytes)]),
  });
  const ourCache = m.cacheBytes(cfg, cfg.blockSize);
  const weightBytes = m.paramCount(cfg) * 4;

  const speedup = uncachedTotal / cachedTotal;
  lab.done(`Your cache reproduced full recomputation **token for token** (${NEW} of ${NEW} tokens identical). Uncached generation took **${uncachedTotal.toFixed(0)} ms** (${meanUncached.toFixed(2)} ms per token, rising with context); cached took **${cachedTotal.toFixed(0)} ms** (${prefillMs.toFixed(1)} ms prefill of ${promptIds.length} token(s), ${(prefillMs / promptIds.length).toFixed(2)} ms each, + ${meanCached.toFixed(2)} ms per decode step), a **${speedup.toFixed(1)}×** speedup. Your cost model says the last token cost ${fmtInt(flopsUncached.at(-1))} FLOPs uncached versus ${fmtInt(flopsCached.at(-1))} cached (**${flopRatio.toFixed(0)}×**). The full ${cfg.blockSize}-token cache of this model is ${fmtBytes(ourCache)} against ${fmtBytes(weightBytes)} of weights; a Llama-3-8B-shaped cache is ${fmtBytes(sizes[2][0] / contexts[0])} per token and **${fmtBytes(sizes[2][4])}** at 128k context.`);
}
