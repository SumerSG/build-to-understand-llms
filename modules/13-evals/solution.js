// Module 13 — An eval harness: reference solution.
// Graders are pure functions (answer, reference, question) → score in [0, 1].
// A task is { id, question, answer }. A model is (question, next) → string, where `next` is a seeded rng.
import { rng, randInt, choice, meanArray } from 'lib/util.js';

// ---------- worked examples and fixtures (identical in starter.js) ----------

/** Lowercase word/number tokens; punctuation is dropped. Used by the n-gram contamination check. */
export function wordTokens(text) {
  return String(text).toLowerCase().match(/[a-z0-9]+/g) || [];
}

const NAMES = ['Maya', 'Omar', 'Lena', 'Ravi', 'Ines', 'Kofi', 'Sara', 'Theo'];
const ITEMS = ['apples', 'pens', 'marbles', 'stickers', 'coins', 'books', 'cards', 'shells'];

/** n GSM8K-style one-step word problems with integer answers. Deterministic for a given seed. */
export function makeTasks(n, seed = 13) {
  const next = rng(seed);
  const tasks = [];
  for (let i = 0; i < n; i++) {
    const name = choice(next, NAMES), item = choice(next, ITEMS);
    const kind = i % 5;
    let question, value;
    if (kind === 0) {
      const a = 10 + randInt(next, 90), b = 10 + randInt(next, 90);
      question = `${name} has ${a} ${item}. A friend gives ${name} ${b} more ${item}. How many ${item} does ${name} have now?`;
      value = a + b;
    } else if (kind === 1) {
      const a = 50 + randInt(next, 50), b = 10 + randInt(next, 40);
      question = `${name} had ${a} ${item} and gave ${b} of them away. How many ${item} does ${name} have left?`;
      value = a - b;
    } else if (kind === 2) {
      const a = 2 + randInt(next, 98), b = 2 + randInt(next, 11);
      question = `There are ${a} boxes with ${b} ${item} in each box. How many ${item} are there altogether?`;
      value = a * b;
    } else if (kind === 3) {
      const b = 2 + randInt(next, 11), a = 2 + randInt(next, 98);
      question = `${name} shares ${a * b} ${item} equally among ${b} friends. How many ${item} does each friend get?`;
      value = a;
    } else {
      const a = 20 + randInt(next, 80), b = 2 + randInt(next, 11);
      question = `A shop sold ${a} ${item} every day for ${b} days. How many ${item} did the shop sell in total?`;
      value = a * b;
    }
    tasks.push({ id: `t${String(i).padStart(2, '0')}`, question, answer: String(value) });
  }
  return tasks;
}

const NUMBER_RE = /-?\d+(?:,\d{3})*(?:\.\d+)?/g;

/**
 * A scripted stand-in for an LLM judge. It returns free text, not a number, the way a real
 * judge does, and it has two deliberate flaws: it accepts any answer that *mentions* the reference
 * number, and it is impressed by long answers (>= 15 words) that contain any number at all.
 */
export function scriptedJudge(question, answer, reference) {
  const nums = (String(answer).match(NUMBER_RE) || []).map((x) => x.replace(/,/g, ''));
  const ref = String(reference).replace(/,/g, '');
  const long = wordTokens(answer).length >= 15;
  if (nums.includes(ref)) return 'Verdict: CORRECT. The response states the reference value.';
  if (long && nums.length) return 'Verdict: CORRECT. The response reasons carefully and reaches a numeric answer.';
  return 'Verdict: INCORRECT. The response does not state the reference value.';
}

/**
 * A scripted pairwise judge: returns 'A' or 'B'. It scores each side (reference mentioned: +2,
 * long: +1) and, on a tie, picks A — the position bias that real judges show (Zheng et al. 2023).
 */
export function scriptedPairwiseJudge(question, a, b, reference) {
  const ref = String(reference).replace(/,/g, '');
  const score = (ans) => {
    const nums = (String(ans).match(NUMBER_RE) || []).map((x) => x.replace(/,/g, ''));
    return (nums.includes(ref) ? 2 : 0) + (wordTokens(ans).length >= 15 ? 1 : 0);
  };
  return score(b) > score(a) ? 'B' : 'A';
}

/** Markdown table for the report returned by runEval. */
export function formatReport(report) {
  const lines = ['| metric | estimate | 95% CI | pass^k |', '|---|---|---|---|'];
  for (const m of report.metrics) {
    lines.push(`| pass@${m.k} | ${m.value.toFixed(3)} | [${m.lo.toFixed(3)}, ${m.hi.toFixed(3)}] | ${m.passPowK.toFixed(3)} |`);
  }
  return lines.join('\n');
}

// ---------- step 1: graders ----------

export function normalizeAnswer(s) {
  let t = String(s).toLowerCase().trim().replace(/\s+/g, ' ');
  t = t.replace(/\.$/, '').trim();
  t = t.replace(/(\d),(?=\d{3}\b)/g, '$1');
  t = t.replace(/\b(\d+)\.(\d*[1-9])?0+\b/g, (_, i, f) => (f ? `${i}.${f}` : i));
  return t;
}

export function extractFinal(text) {
  const s = String(text);
  const i = s.lastIndexOf('####');
  if (i >= 0) return s.slice(i + 4).trim();
  const nums = s.match(NUMBER_RE);
  if (nums) return nums[nums.length - 1];
  return s.trim();
}

export function exactMatch(answer, reference) {
  return normalizeAnswer(answer) === normalizeAnswer(reference) ? 1 : 0;
}

export function regexMatch(answer, pattern) {
  // Rebuild a RegExp without g/y: those flags make .test() stateful through lastIndex.
  const re = pattern instanceof RegExp ? new RegExp(pattern.source, pattern.flags.replace(/[gy]/g, '')) : new RegExp(pattern, 'i');
  return re.test(String(answer)) ? 1 : 0;
}

// ---------- step 2: pass@k ----------

export function logChoose(n, k) {
  if (k < 0 || k > n) return -Infinity;
  let s = 0;
  for (let i = 0; i < k; i++) s += Math.log(n - i) - Math.log(i + 1);
  return s;
}

export function passAtK(n, c, k) {
  if (k < 1 || k > n) throw new Error(`passAtK: k=${k} must be in 1..n=${n}`);
  if (c < 0 || c > n) throw new Error(`passAtK: c=${c} must be in 0..n=${n}`);
  if (n - c < k) return 1;
  return 1 - Math.exp(logChoose(n - c, k) - logChoose(n, k));
}

export function passPowK(n, c, k) {
  if (k < 1 || k > n) throw new Error(`passPowK: k=${k} must be in 1..n=${n}`);
  if (c < 0 || c > n) throw new Error(`passPowK: c=${c} must be in 0..n=${n}`);
  if (c < k) return 0;
  return Math.exp(logChoose(c, k) - logChoose(n, k));
}

// ---------- step 3: bootstrap confidence intervals ----------

export function bootstrapCI(scores, { B = 1000, alpha = 0.05, next = rng(0) } = {}) {
  const n = scores.length;
  if (!n) throw new Error('bootstrapCI: need at least one score');
  const mean = meanArray(scores);
  const means = new Float64Array(B);
  for (let b = 0; b < B; b++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += scores[randInt(next, n)];
    means[b] = s / n;
  }
  let se = 0;
  const mb = meanArray(means);
  for (let b = 0; b < B; b++) se += (means[b] - mb) ** 2;
  se = Math.sqrt(se / B);
  means.sort();
  const lo = means[Math.floor((alpha / 2) * (B - 1))];
  const hi = means[Math.ceil((1 - alpha / 2) * (B - 1))];
  return { mean, lo, hi, se };
}

// ---------- step 4: judges ----------

export function judgeGrader(judge) {
  return (answer, reference, question) => (/\bcorrect\b/i.test(String(judge(question, answer, reference))) ? 1 : 0);
}

export function agreementRate(records, graderA, graderB) {
  if (!records.length) return 1;
  let agree = 0;
  for (const r of records) {
    const a = graderA(r.answer, r.reference, r.question) >= 0.5;
    const b = graderB(r.answer, r.reference, r.question) >= 0.5;
    if (a === b) agree++;
  }
  return agree / records.length;
}

export function positionBias(pairwise, pairs) {
  let consistent = 0, firstWins = 0, decided = 0;
  for (const p of pairs) {
    const v1 = pairwise(p.question, p.a, p.b, p.reference);
    const v2 = pairwise(p.question, p.b, p.a, p.reference);
    if ((v1 === 'A' && v2 === 'B') || (v1 === 'B' && v2 === 'A') || (v1 === 'tie' && v2 === 'tie')) consistent++;
    for (const v of [v1, v2]) {
      if (v === 'tie') continue;
      decided++;
      if (v === 'A') firstWins++;
    }
  }
  return {
    consistency: pairs.length ? consistent / pairs.length : 1,
    firstWinRate: decided ? firstWins / decided : 0.5,
  };
}

// ---------- step 5: contamination and the runner ----------

export function ngrams(text, n) {
  const w = wordTokens(text);
  const out = new Set();
  for (let i = 0; i + n <= w.length; i++) out.add(w.slice(i, i + n).join(' '));
  return out;
}

export function contamination(tasks, corpus, n = 13) {
  const seen = ngrams(corpus, n);
  const flagged = [];
  for (const t of tasks) {
    for (const g of ngrams(t.question, n)) {
      if (seen.has(g)) { flagged.push(t.id); break; }
    }
  }
  return { rate: tasks.length ? flagged.length / tasks.length : 0, flagged, n };
}

export function runEval({ tasks, model, grader, n = 10, ks = [1, 5, 10], next = rng(0), B = 1000 }) {
  for (const k of ks) if (k > n) throw new Error(`runEval: pass@${k} needs at least ${k} samples per task, got n=${n}`);
  const results = [];
  for (const t of tasks) {
    let c = 0;
    const answers = [];
    for (let i = 0; i < n; i++) {
      const a = model(t.question, next);
      answers.push(a);
      if (grader(a, t.answer, t.question) >= 0.5) c++;
    }
    results.push({ id: t.id, question: t.question, reference: t.answer, n, c, answers });
  }
  const metrics = ks.map((k) => {
    const per = results.map((r) => passAtK(r.n, r.c, k));
    const ci = bootstrapCI(per, { B, next });
    const passPow = meanArray(results.map((r) => passPowK(r.n, r.c, k)));
    return { k, value: ci.mean, lo: ci.lo, hi: ci.hi, se: ci.se, passPowK: passPow };
  });
  return { nTasks: tasks.length, n, results, metrics };
}
