// Long-context evaluation.
// A "model" here is a function (context, question, next) → answer string, where `next` is a seeded rng.
// Lengths are counted in words (see countTokens); depths are fractions in [0, 1] of the way into the filler.
// Everything below the worked examples marked TODO is yours. The worked examples set the conventions:
// pure functions, all randomness from `next`, numbers (not booleans) from graders.
import { rng, randInt, shuffle, meanArray } from 'lib/util.js';
import { toyCorpus } from 'lib/data.js';

// ---------- worked examples and fixtures (done for you; read them first) ----------

/** Filler sentences: the toy grammar from lib/data.js, one sentence per entry, 7.5 words on average. */
export const FILLER = toyCorpus(3000, 33).split('\n');

/** Keys for needles. None of these words occurs in the toy grammar, so filler can never answer a question. */
export const CITIES = [
  'Lisbon', 'Nairobi', 'Oslo', 'Lima', 'Hanoi', 'Quito', 'Dakar', 'Tbilisi', 'Perth', 'Bergen',
  'Cusco', 'Accra', 'Tallinn', 'Kyoto', 'Porto', 'Riga', 'Sapporo', 'Tunis', 'Valletta', 'Windhoek',
];

/** Length of a text in "tokens". We count words; a BPE tokenizer gives roughly 1.3 tokens per English word. */
export function countTokens(text) {
  return String(text).split(/\s+/).filter(Boolean).length;
}

/** True if `word` occurs in `text` as a whole word (case-insensitive). */
export function containsWord(text, word) {
  const escaped = String(word).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(^|[^A-Za-z0-9])${escaped}($|[^A-Za-z0-9])`, 'i').test(String(text));
}

/** A needle: one fact with a key and a 4-digit value. `avoid` lists values already used in this task. */
export function makeNeedle(key, next, avoid = []) {
  let value;
  do value = String(1000 + randInt(next, 9000)); while (avoid.includes(value));
  return { key, value, sentence: `The secret code for ${key} is ${value}.` };
}

/** The eval harness module's percentile bootstrap: resample the scores B times and read off the 2.5% and 97.5% means. */
export function bootstrapCI(scores, { B = 200, alpha = 0.05, next = rng(0) } = {}) {
  const n = scores.length;
  if (!n) throw new Error('bootstrapCI: need at least one score');
  const means = new Float64Array(B);
  for (let b = 0; b < B; b++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += scores[randInt(next, n)];
    means[b] = s / n;
  }
  means.sort();
  return { mean: meanArray(scores), lo: means[Math.floor((alpha / 2) * (B - 1))], hi: means[Math.ceil((1 - alpha / 2) * (B - 1))] };
}

/**
 * One eval item, built on your buildHaystack. Three RULER-style kinds (Hsieh et al. 2024):
 *   'niah'       one needle at `depth`; answer its value.
 *   'multikey'   the target needle at `depth` plus 3 distractor needles for OTHER cities at random depths.
 *   'multivalue' 3 needles for the SAME city (first at `depth`, the others at random depths); answer all values.
 * Returns { kind, context, question, answers, distractors, positions, tokens }.
 */
export function makeTask(kind, { length, depth, next, filler = FILLER }) {
  const keys = shuffle(next, CITIES.slice());
  const used = [];
  const needle = (key) => { const n = makeNeedle(key, next, used); used.push(n.value); return n; };
  let needles, depths, question, answers, distractors;
  if (kind === 'niah') {
    needles = [needle(keys[0])];
    depths = [depth];
    question = `What is the secret code for ${keys[0]}?`;
    answers = [needles[0].value];
    distractors = [];
  } else if (kind === 'multikey') {
    needles = [needle(keys[0]), needle(keys[1]), needle(keys[2]), needle(keys[3])];
    depths = [depth, next(), next(), next()];
    question = `What is the secret code for ${keys[0]}?`;
    answers = [needles[0].value];
    distractors = needles.slice(1).map((n) => n.value);
  } else if (kind === 'multivalue') {
    needles = [needle(keys[0]), needle(keys[0]), needle(keys[0])];
    depths = [depth, next(), next()];
    question = `What are all the secret codes for ${keys[0]}?`;
    answers = needles.map((n) => n.value);
    distractors = [];
  } else {
    throw new Error(`makeTask: unknown kind "${kind}" (use niah, multikey or multivalue)`);
  }
  const hay = buildHaystack({ length, needles, depths, next, filler });
  return { kind, context: hay.text, question, answers, distractors, positions: hay.positions, tokens: hay.tokens };
}

/**
 * A synthetic long-context "model" whose retrieval fades with distance, so the eval has something to find.
 * For a fact that starts `pos` words into the visible context and ends `dist` words before the question:
 *   recency = exp(-(dist / reach')²)            reach' = reach / (1 + interference · otherKeys)
 *   primacy = sink · exp(-pos / sinkSpan)        the first tokens stay visible (attention sinks)
 *   p(recall) = 1 - (1 - recency) · (1 - primacy)
 * Context beyond `window` words is cut from the left, as lib/sampling.js's generate() does at blockSize.
 * A missed fact is replaced by a distractor's value half the time (a confident wrong answer), else "not found".
 */
export function makeFadingModel({ window = 8192, reach = 2500, sink = 0.7, sinkSpan = 300, interference = 0.2 } = {}) {
  const NEEDLE = /^The secret code for (\w+) is (\d+)\.$/;
  return function fadingModel(context, question, next) {
    const sentences = String(context).split(/(?<=[.?!])\s+/);
    const facts = [];
    let pos = 0;
    for (const s of sentences) {
      const m = NEEDLE.exec(s);
      const len = countTokens(s);
      if (m) facts.push({ key: m[1], value: m[2], start: pos, end: pos + len });
      pos += len;
    }
    const total = pos;
    const cut = Math.max(0, total - window);
    const seen = facts.filter((f) => f.start >= cut).map((f) => ({ ...f, start: f.start - cut, end: f.end - cut }));
    const visible = total - cut;
    const q = /secret codes? for (\w+)/.exec(String(question));
    if (!q) return 'I do not understand the question.';
    const key = q[1];
    const others = new Set(seen.filter((f) => f.key !== key).map((f) => f.key)).size;
    const r = reach / (1 + interference * others);
    const pRecall = (f) => {
      const recency = Math.exp(-(((visible - f.end) / r) ** 2));
      const primacy = sink * Math.exp(-f.start / sinkSpan);
      return 1 - (1 - recency) * (1 - primacy);
    };
    const targets = seen.filter((f) => f.key === key);
    if (/\ball\b/i.test(String(question))) {
      const got = targets.filter((f) => next() < pRecall(f)).map((f) => f.value);
      return got.length ? `The secret codes for ${key} are ${got.join(', ')}.` : `I could not find any secret codes for ${key}.`;
    }
    for (const f of targets) if (next() < pRecall(f)) return `The secret code for ${key} is ${f.value}.`;
    const wrong = seen.filter((f) => f.key !== key);
    if (wrong.length && next() < 0.5) {
      const w = wrong.reduce((a, b) => (pRecall(b) > pRecall(a) ? b : a));
      return `The secret code for ${key} is ${w.value}.`;
    }
    return `I could not find the secret code for ${key}.`;
  };
}

/** Llama-3-8B shapes from Meta's model card: 32 layers, d_model 4096, 8 KV heads of dim 128 (GQA). */
export const LLAMA3_8B = { name: 'Llama-3-8B', params: 8.03e9, layers: 32, dModel: 4096, nKvHeads: 8, headDim: 128 };

// ---------- step 1: the haystack ----------

/**
 * Build one haystack: filler sentences with each needle planted at its own depth.
 *   length   target size in words, needles included
 *   needles  [{ key, value, sentence }] (see makeNeedle)
 *   depths   one number in [0, 1] per needle: 0 = before all filler, 1 = after all filler
 *   next     seeded rng; filler starts at sentence randInt(next, filler.length) and wraps around
 * Returns { sentences, text: sentences.join(' '), positions (index of needle k in sentences), tokens }.
 */
export function buildHaystack({ length, needles, depths, next, filler = FILLER }) {
  if (needles.length !== depths.length) throw new Error('buildHaystack: one depth per needle');
  const needleTokens = needles.reduce((s, n) => s + countTokens(n.sentence), 0);
  const target = length - needleTokens;
  // TODO: step 1 — collect filler sentences until their words reach `target`, skipping any sentence that
  // mentions a needle's key or value as a whole word (throw after filler.length skips in a row).
  // TODO: step 1 — insert each needle at the sentence boundary nearest depth · (filler words).
  return { sentences: [], text: '', positions: [], tokens: 0 };
}

// ---------- step 2: graders ----------

/** The eval harness module's normalisation, trimmed to what codes need: lowercase, trim, collapse spaces, one trailing period, thousands commas. */
export function normalizeAnswer(s) {
  let t = String(s).toLowerCase().trim().replace(/\s+/g, ' ');
  t = t.replace(/\.$/, '').trim();
  t = t.replace(/(\d),(?=\d{3}\b)/g, '$1');
  return t;
}

/** 1 if answer and reference normalise to the same string, else 0. */
export function exactMatch(answer, reference) {
  // TODO: step 2
  return 0;
}

/** 1 if the normalised reference occurs as a whole word inside the normalised answer, else 0. */
export function fuzzyMatch(answer, reference) {
  // TODO: step 2
  return 0;
}

/** Fraction of `references` that fuzzy-match inside `answer` (0 when there are none). */
export function recallFraction(answer, references) {
  // TODO: step 2
  return 0;
}

/** Score one model answer for a task from makeTask: multivalue → recallFraction; otherwise fuzzy, and 0 if any distractor appears. */
export function gradeTask(task, answer) {
  // TODO: step 2
  return 0;
}

// ---------- step 3: the grid runner ----------

/**
 * Evaluate `model` on a lengths × depths grid with `trials` items per cell.
 * Returns { kind, lengths, depths, trials, acc, lo, hi } where acc/lo/hi are [lengths.length][depths.length].
 */
export function contextEval(model, { lengths, depths, trials = 10, kind = 'niah', seed = 0, B = 200, filler = FILLER }) {
  // TODO: step 3
  return { kind, lengths: lengths.slice(), depths: depths.slice(), trials, acc: [], lo: [], hi: [] };
}

// ---------- step 4: effective context and the depth curve ----------

/** Mean accuracy over depths, one number per length. */
export function lengthCurve(result) {
  // TODO: step 4
  return [];
}

/** The longest evaluated length such that it and every shorter length average >= threshold; 0 if none. */
export function effectiveContext(result, threshold = 0.8) {
  // TODO: step 4
  return 0;
}

/** Accuracy per depth at one evaluated length (a copy). Throw if `length` was not evaluated. */
export function depthCurve(result, length) {
  // TODO: step 4
  return [];
}

/** Mean of the two edge depths minus the mean of the interior depths. Throw if fewer than 3 depths. */
export function middleDrop(curve) {
  // TODO: step 4
  return 0;
}

// ---------- step 5: the price of a long prompt ----------

/** FLOPs to prefill T tokens: the KV cache module's per-token cost 2N + 4·L·t·C summed over t = 1…T. */
export function prefillFlops(cfg, T) {
  // TODO: step 5
  return 0;
}

/** KV-cache bytes for T tokens: 2 (K and V) · layers · nKvHeads · headDim · T · bytesPerElement. */
export function kvCacheBytes(cfg, T, bytesPerElement = 2) {
  // TODO: step 5
  return 0;
}

/** { tokens, flops, attnShare, seconds, kvBytes } for one prompt of T tokens. */
export function longPromptCost(cfg, T, { peakFlops = 989e12, mfu = 0.5, bytesPerElement = 2 } = {}) {
  // TODO: step 5
  return { tokens: T, flops: 0, attnShare: 0, seconds: 0, kvBytes: 0 };
}
