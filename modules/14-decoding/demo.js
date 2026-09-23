// Decoding & sampling demo — one prompt, one set of logits, five decoding policies. The model is the pre-trained
// checkpoint (lib/checkpoints/tiny-gpt.json); YOUR processors, sampler and generate loop do the decoding.
// If the checkpoint cannot be loaded (mid-retrain, or a browser without JSON imports) the demo falls back
// to a random-weight model so every step still runs; the text is then noise, the mechanics are the same.
import { loadModel, newCache, prefill, forwardStep } from 'lib/infer.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { GPT } from 'lib/gpt.js';
import { toyCorpus } from 'lib/data.js';
import { rng } from 'lib/util.js';

const PROMPT = 'The cat';
const NEW_TOKENS = 60;
const SETTINGS = [
  { name: 'greedy (t = 0)', opts: { temperature: 0 }, bars: true },
  { name: 'greedy + repetition penalty 1.3', opts: { temperature: 0, repetitionPenalty: 1.3 }, bars: false },
  { name: 't = 1, no cut', opts: { temperature: 1 }, bars: true },
  { name: 't = 0.7, top-p 0.9', opts: { temperature: 0.7, topP: 0.9 }, bars: true },
  { name: 't = 1.5, min-p 0.1', opts: { temperature: 1.5, minP: 0.1 }, bars: true },
];

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

/** Shannon entropy in bits: log2 of the "effective number of equally likely tokens". */
function entropyBits(probs) {
  let h = 0;
  for (let i = 0; i < probs.length; i++) if (probs[i] > 0) h -= probs[i] * Math.log2(probs[i]);
  return h;
}

const countFinite = (arr) => { let n = 0; for (let i = 0; i < arr.length; i++) if (Number.isFinite(arr[i])) n++; return n; };

/** Fraction of 4-grams of token ids that already occurred earlier in the same output: a repetition score. */
function repeatedNgramRate(ids, n = 4) {
  if (ids.length < n + 1) return 0;
  const seen = new Set();
  let repeats = 0, total = 0;
  for (let i = 0; i + n <= ids.length; i++) {
    const key = ids.slice(i, i + n).join(',');
    total++;
    if (seen.has(key)) repeats++;
    else seen.add(key);
  }
  return repeats / total;
}

const show = (text) => text.replace(/\n/g, ' ⏎ ').trim();

export default async function demo(m, lab) {
  const { model, tokenizer, trained } = await loadCheckpoint(lab);
  const V = model.config.vocabSize;
  lab.log(`Model: ${model.config.nLayer} layers, ${model.config.nEmbd} wide, vocab ${V}, context ${model.config.blockSize}${trained ? ' (pre-trained checkpoint)' : ' (random weights)'}.`);

  // ---------- 1. one set of logits, five policies ----------
  const promptIds = tokenizer.encode(PROMPT);
  const cache = newCache(model);
  const logits = prefill(model, cache, promptIds);
  const raw = m.softmaxLogits(logits);
  const top10 = Array.from(raw.keys()).sort((a, b) => raw[b] - raw[a]).slice(0, 10);
  const labels = top10.map((id) => JSON.stringify(tokenizer.decode([id])).slice(1, -1).replace(/ /g, '␣'));
  lab.md(`After the prompt **"${PROMPT}"** the model's raw next-token distribution puts **${(100 * raw[top10[0]]).toFixed(1)}%** on \`${labels[0]}\` and **${(100 * raw[top10[9]]).toFixed(2)}%** on the tenth candidate; its entropy is **${entropyBits(raw).toFixed(2)} bits**. Every chart below is the same vector after one of your pipelines.`);

  const firstStep = [];
  for (const s of SETTINGS) {
    const processed = m.processLogits(logits, { ...s.opts, prevIds: promptIds });
    const probs = m.softmaxLogits(processed);
    const H = entropyBits(probs);
    const kept = countFinite(processed);
    firstStep.push({ H, kept });
    if (s.bars) {
      lab.bar({ title: `${s.name}: top-10 tokens, ${kept} of ${V} kept, entropy ${H.toFixed(2)} bits`, labels, values: top10.map((id) => +probs[id].toFixed(4)) });
    }
    await lab.tick();
  }

  // ---------- 2. sixty tokens under each policy ----------
  const runs = [];
  for (let i = 0; i < SETTINGS.length; i++) {
    const s = SETTINGS[i];
    const out = m.generate(model, tokenizer, PROMPT, { ...s.opts, maxNewTokens: NEW_TOKENS, next: rng(14) });
    runs.push({ ...s, out, repeat: repeatedNgramRate(out.ids) });
    lab.progress((i + 1) / SETTINGS.length, s.name);
    await lab.tick();
  }
  lab.table({
    title: `${NEW_TOKENS} tokens from "${PROMPT}" under each policy (seed 14)`,
    columns: ['policy', 'entropy at step 1 (bits)', 'tokens kept at step 1', 'repeated 4-grams', 'finish', 'text'],
    rows: runs.map((r, i) => [r.name, firstStep[i].H.toFixed(2), firstStep[i].kept, `${(100 * r.repeat).toFixed(0)}%`, r.out.finishReason, show(r.out.text)]),
  });

  // ---------- 3. how big is the nucleus along a generation? ----------
  const sizes = { 'top-p 0.9': [], 'min-p 0.1': [], 'top-k 10': [] };
  const entropies = [];
  {
    const c = newCache(model);
    let l = prefill(model, c, promptIds);
    const next = rng(14);
    const ids = [];
    for (let step = 0; step < NEW_TOKENS; step++) {
      sizes['top-p 0.9'].push(countFinite(m.topPFilter(l, 0.9)));
      sizes['min-p 0.1'].push(countFinite(m.minPFilter(l, 0.1)));
      sizes['top-k 10'].push(countFinite(m.topKFilter(l, 10)));
      entropies.push(entropyBits(m.softmaxLogits(l)));
      const id = m.sample(l, { temperature: 1, next });
      ids.push(id);
      if (c.length >= model.config.blockSize) break;
      l = forwardStep(model, c, id);
      if (step % 10 === 9) await lab.tick();
    }
  }
  const steps = sizes['top-p 0.9'].map((_, i) => i);
  lab.plot({
    title: 'Tokens kept per step along a t = 1 generation: mass-based cuts adapt, top-k does not',
    x: steps,
    series: Object.entries(sizes).map(([name, values]) => ({ name, values })),
    xlabel: 'decode step', ylabel: 'tokens kept',
  });
  lab.plot({ title: 'Entropy of the raw next-token distribution per step', x: steps, series: [{ name: 'entropy (bits)', values: entropies }], xlabel: 'decode step', ylabel: 'bits' });
  const meanP = sizes['top-p 0.9'].reduce((a, b) => a + b, 0) / steps.length;
  const maxP = Math.max(...sizes['top-p 0.9']);
  const minP = Math.min(...sizes['top-p 0.9']);

  // ---------- 4. does the sampler match the maths? ----------
  const checkOpts = { temperature: 0.7, topP: 0.9 };
  const expected = m.softmaxLogits(m.processLogits(logits, checkOpts));
  const N = 5000;
  const counts = new Float64Array(V);
  const next = rng(99);
  for (let i = 0; i < N; i++) {
    counts[m.sample(logits, { ...checkOpts, next })]++;
    if (i % 1000 === 999) await lab.tick();
  }
  let maxGap = 0, outside = 0;
  for (let i = 0; i < V; i++) {
    maxGap = Math.max(maxGap, Math.abs(counts[i] / N - expected[i]));
    if (expected[i] === 0 && counts[i] > 0) outside += counts[i];
  }
  lab.check(outside === 0, `${outside} draws landed on tokens the pipeline had filtered out`);
  lab.check(maxGap < 0.03, `empirical frequencies differ from softmax(processed logits) by ${maxGap.toFixed(3)}`);

  const greedy = runs[0], penalised = runs[1];
  lab.done(
    `From the same logits your five policies kept **${firstStep.map((f) => f.kept).join(' / ')}** of ${V} tokens at step 1 with entropies **${firstStep.map((f) => f.H.toFixed(2)).join(' / ')} bits** (${SETTINGS.map((s) => s.name).join('; ')}). ` +
    `Over ${NEW_TOKENS} tokens greedy repeated **${(100 * greedy.repeat).toFixed(0)}%** of its 4-grams; the 1.3 repetition penalty brought that to **${(100 * penalised.repeat).toFixed(0)}%**, and the sampled policies ${runs.slice(2).map((r) => `${(100 * r.repeat).toFixed(0)}%`).join(', ')}. ` +
    `Along a t = 1 generation top-p 0.9 kept between **${minP} and ${maxP}** tokens (mean ${meanP.toFixed(1)}) while top-k held 10. ` +
    `${N.toLocaleString()} draws from your sampler at t = 0.7 / top-p 0.9 matched the processed distribution to within **${maxGap.toFixed(3)}** and never hit a filtered token.`,
  );
}
