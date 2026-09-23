// Long-context evaluation: reference solution.
// A "model" here is a function (context, question, next) → answer string, where `next` is a seeded rng.
// Lengths are counted in words (see countTokens); depths are fractions in [0, 1] of the way into the filler.
import { rng, randInt, shuffle, meanArray } from 'lib/util.js';
import { toyCorpus } from 'lib/data.js';

// ---------- worked examples and fixtures (identical in starter.js) ----------

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

export function buildHaystack({ length, needles, depths, next, filler = FILLER }) {
  if (needles.length !== depths.length) throw new Error('buildHaystack: one depth per needle');
  const needleTokens = needles.reduce((s, n) => s + countTokens(n.sentence), 0);
  const target = length - needleTokens;
  const accidental = (s) => needles.some((n) => containsWord(s, n.value) || containsWord(s, n.key));
  const lowered = needles.flatMap((n) => [n.value.toLowerCase(), n.key.toLowerCase()]);
  const maybe = (s) => { const l = s.toLowerCase(); return lowered.some((w) => l.includes(w)); };
  const body = [];
  // `skipped` counts skips in a row: a full lap of the filler with nothing usable means there is none.
  let tokens = 0, i = randInt(next, filler.length), skipped = 0;
  while (tokens < target) {
    const s = filler[i % filler.length];
    i++;
    if (maybe(s) && accidental(s)) {
      if (++skipped >= filler.length) throw new Error('buildHaystack: every filler sentence contains a needle');
      continue;
    }
    skipped = 0;
    body.push(s);
    tokens += countTokens(s);
  }
  const F = body.length;
  // cum[i] = filler words before sentence i. Each needle goes at the boundary nearest depth · tokens.
  const cum = [0];
  for (const s of body) cum.push(cum[cum.length - 1] + countTokens(s));
  const at = depths.map((d) => {
    const want = Math.min(1, Math.max(0, d)) * tokens;
    let best = 0;
    for (let b = 1; b <= F; b++) if (Math.abs(cum[b] - want) < Math.abs(cum[best] - want)) best = b;
    return best;
  });
  const sentences = [];
  const positions = new Array(needles.length);
  for (let idx = 0; idx <= F; idx++) {
    for (let k = 0; k < needles.length; k++) {
      if (at[k] === idx) { positions[k] = sentences.length; sentences.push(needles[k].sentence); }
    }
    if (idx < F) sentences.push(body[idx]);
  }
  const text = sentences.join(' ');
  return { sentences, text, positions, tokens: tokens + needleTokens };
}

// ---------- step 2: graders ----------

export function normalizeAnswer(s) {
  let t = String(s).toLowerCase().trim().replace(/\s+/g, ' ');
  t = t.replace(/\.$/, '').trim();
  t = t.replace(/(\d),(?=\d{3}\b)/g, '$1');
  return t;
}

export function exactMatch(answer, reference) {
  return normalizeAnswer(answer) === normalizeAnswer(reference) ? 1 : 0;
}

export function fuzzyMatch(answer, reference) {
  return containsWord(normalizeAnswer(answer), normalizeAnswer(reference)) ? 1 : 0;
}

export function recallFraction(answer, references) {
  if (!references.length) return 0;
  let hit = 0;
  for (const r of references) hit += fuzzyMatch(answer, r);
  return hit / references.length;
}

export function gradeTask(task, answer) {
  if (task.kind === 'multivalue') return recallFraction(answer, task.answers);
  if (task.distractors.some((d) => fuzzyMatch(answer, d))) return 0;
  return fuzzyMatch(answer, task.answers[0]);
}

// ---------- step 3: the grid runner ----------

export function contextEval(model, { lengths, depths, trials = 10, kind = 'niah', seed = 0, B = 200, filler = FILLER }) {
  const next = rng(seed);
  const boot = rng(seed + 7919);
  const acc = [], lo = [], hi = [];
  for (const length of lengths) {
    const accRow = [], loRow = [], hiRow = [];
    for (const depth of depths) {
      const scores = [];
      for (let t = 0; t < trials; t++) {
        const task = makeTask(kind, { length, depth, next, filler });
        scores.push(gradeTask(task, model(task.context, task.question, next)));
      }
      const ci = bootstrapCI(scores, { B, next: boot });
      accRow.push(ci.mean); loRow.push(ci.lo); hiRow.push(ci.hi);
    }
    acc.push(accRow); lo.push(loRow); hi.push(hiRow);
  }
  return { kind, lengths: lengths.slice(), depths: depths.slice(), trials, acc, lo, hi };
}

// ---------- step 4: effective context and the depth curve ----------

export function lengthCurve(result) {
  return result.acc.map((row) => meanArray(row));
}

export function effectiveContext(result, threshold = 0.8) {
  const curve = lengthCurve(result);
  let best = 0;
  for (let i = 0; i < curve.length; i++) {
    if (curve[i] < threshold) break;
    best = result.lengths[i];
  }
  return best;
}

export function depthCurve(result, length) {
  const i = result.lengths.indexOf(length);
  if (i < 0) throw new Error(`depthCurve: length ${length} was not evaluated (have ${result.lengths.join(', ')})`);
  return result.acc[i].slice();
}

export function middleDrop(curve) {
  if (curve.length < 3) throw new Error('middleDrop: need at least 3 depths (two edges and a middle)');
  const edges = (curve[0] + curve[curve.length - 1]) / 2;
  return edges - meanArray(curve.slice(1, -1));
}

// ---------- step 5: the price of a long prompt ----------

export function prefillFlops(cfg, T) {
  return 2 * cfg.params * T + 2 * cfg.layers * cfg.dModel * T * (T + 1);
}

export function kvCacheBytes(cfg, T, bytesPerElement = 2) {
  return 2 * cfg.layers * cfg.nKvHeads * cfg.headDim * T * bytesPerElement;
}

export function longPromptCost(cfg, T, { peakFlops = 989e12, mfu = 0.5, bytesPerElement = 2 } = {}) {
  const flops = prefillFlops(cfg, T);
  const attnFlops = flops - 2 * cfg.params * T;
  return {
    tokens: T,
    flops,
    attnShare: attnFlops / flops,
    seconds: flops / (peakFlops * mfu),
    kvBytes: kvCacheBytes(cfg, T, bytesPerElement),
  };
}
