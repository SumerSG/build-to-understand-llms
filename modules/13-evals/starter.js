// Module 13 — An eval harness.
// Conventions used throughout this file:
//   a task   is { id, question, answer }          (answer is the reference, as a string)
//   a grader is (answer, reference, question) → score in [0, 1]   (pure: no state, no randomness)
//   a model  is (question, next) → string          (next is a seeded rng from lib/util.js)
// Everything below the "worked examples" section is yours to implement.
import { rng, randInt, choice, meanArray } from 'lib/util.js';

// ---------- worked examples and fixtures (done for you; read them, they set the conventions) ----------

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

/** Matches integers and decimals, with optional thousands separators: 42, -3.5, 1,188. */
const NUMBER_RE = /-?\d+(?:,\d{3})*(?:\.\d+)?/g;

/**
 * A scripted stand-in for an LLM judge. It returns free text, not a number, the way a real
 * judge does, and it has two deliberate flaws: it accepts any answer that *mentions* the reference
 * number, and it is impressed by long answers (>= 15 words) that contain any number at all.
 * You do not fix the judge; in step 4 you measure it.
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

/**
 * Markdown table for the report returned by runEval (step 5). It documents the report shape:
 * report.metrics is an array of { k, value, lo, hi, se, passPowK }.
 */
export function formatReport(report) {
  const lines = ['| metric | estimate | 95% CI | pass^k |', '|---|---|---|---|'];
  for (const m of report.metrics) {
    lines.push(`| pass@${m.k} | ${m.value.toFixed(3)} | [${m.lo.toFixed(3)}, ${m.hi.toFixed(3)}] | ${m.passPowK.toFixed(3)} |`);
  }
  return lines.join('\n');
}

// ---------- step 1: graders ----------

/**
 * Canonical form of an answer string so that "42." and " 42" and "42.0" and "1,000"/"1000" compare equal:
 * lowercase; trim; collapse whitespace runs to one space; drop one trailing period;
 * remove thousands separators (a comma after a digit, followed by exactly three digits and then a word
 * boundary: 1,188 -> 1188 but 12,3456 stays); drop trailing zeros
 * after a decimal point (42.0 -> 42, 3.50 -> 3.5) but never an interior zero (10.05 stays).
 * The first two rules are done; add the rest.
 */
export function normalizeAnswer(s) {
  let t = String(s).toLowerCase().trim().replace(/\s+/g, ' ');
  // TODO: step 1 — trailing period, thousands separators, trailing decimal zeros
  return t;
}

/**
 * The part of a model response that is the final answer: everything after the LAST "####"
 * (the GSM8K convention) if present; otherwise the last number in the text (NUMBER_RE helps);
 * otherwise the trimmed text.
 */
export function extractFinal(text) {
  const s = String(text);
  const i = s.lastIndexOf('####');
  if (i >= 0) return s.slice(i + 4).trim();
  // TODO: step 1 — last number, else the trimmed text
  return s;
}

/** 1 if the normalised answer equals the normalised reference, else 0. Always a number, never a boolean. */
export function exactMatch(answer, reference) {
  // TODO: step 1
  return 0;
}

/**
 * 1 if `pattern` (a RegExp, or a string compiled case-insensitively) matches anywhere in the answer, else 0.
 * A grader must be pure: a RegExp with the g or y flag keeps `lastIndex` between .test() calls, so rebuild
 * it without those flags (pattern.source, pattern.flags) before testing.
 */
export function regexMatch(answer, pattern) {
  // TODO: step 1
  return 0;
}

// ---------- step 2: pass@k ----------

/** log of the binomial coefficient C(n, k); -Infinity when k < 0 or k > n. Never compute C itself. */
export function logChoose(n, k) {
  // TODO: step 2
  return 0;
}

/**
 * Unbiased pass@k from n samples of which c are correct (Chen et al. 2021):
 *   1 - C(n-c, k) / C(n, k)
 * Throw if k is not in 1..n or c is not in 0..n. Return exactly 1 when n - c < k.
 */
export function passAtK(n, c, k) {
  // TODO: step 2
  return 0;
}

/**
 * Unbiased pass^k (all k draws, without replacement from your n samples, correct): C(c, k) / C(n, k).
 * Same validation as passAtK (throw if k is not in 1..n or c is not in 0..n). Return 0 when c < k.
 */
export function passPowK(n, c, k) {
  // TODO: step 2
  return 0;
}

// ---------- step 3: bootstrap confidence intervals ----------

/**
 * Percentile-bootstrap confidence interval for the mean of `scores` (one score per task).
 * Draw B resamples of size n with replacement using `next` (randInt(next, n) picks an index),
 * record each resample's mean, sort them, and return
 *   { mean, lo, hi, se }
 * where mean is the plain mean of scores, lo/hi are the sorted resample means at positions
 * floor((alpha/2)*(B-1)) and ceil((1-alpha/2)*(B-1)), and se is the population standard deviation of the resample means (divide by B).
 */
export function bootstrapCI(scores, { B = 1000, alpha = 0.05, next = rng(0) } = {}) {
  // TODO: step 3
  return { mean: meanArray(scores), lo: 0, hi: 1, se: 0 };
}

// ---------- step 4: judges ----------

/**
 * Turn a free-text judge, judge(question, answer, reference) → string, into a grader
 * (answer, reference, question) → 0|1. Score 1 iff the verdict contains the whole word
 * "correct" (any case). "INCORRECT" must score 0.
 */
export function judgeGrader(judge) {
  // TODO: step 4
  return () => 0;
}

/** Fraction of records ({ question, answer, reference }) on which the two graders give the same verdict (score >= 0.5). 1 for no records. */
export function agreementRate(records, graderA, graderB) {
  // TODO: step 4
  return 0;
}

/**
 * Position-bias check for a pairwise judge pairwise(question, a, b, reference) → 'A' | 'B' | 'tie'.
 * Judge every pair ({ question, a, b, reference }) in both orders. Return
 *   { consistency, firstWinRate }
 * consistency: fraction of pairs whose two verdicts mirror each other (A then B, B then A, or tie both times);
 * firstWinRate: of the calls that picked a side, the fraction that picked 'A' (ties are excluded from
 *   both the numerator and the denominator, so 0.5 always means "no preference for the first slot").
 * No pairs: consistency 1. No decided calls: firstWinRate 0.5.
 */
export function positionBias(pairwise, pairs) {
  // TODO: step 4
  return { consistency: 0, firstWinRate: 0 };
}

// ---------- step 5: contamination and the runner ----------

/** Set of all consecutive n-word sequences in `text` (tokens from wordTokens, joined by single spaces). */
export function ngrams(text, n) {
  // TODO: step 5
  return new Set();
}

/**
 * Which tasks share at least one n-gram with `corpus` (a single string of training text)? Return { rate, flagged, n } where flagged is the
 * array of task ids in task order and rate = flagged.length / tasks.length (0 for no tasks).
 */
export function contamination(tasks, corpus, n = 13) {
  // TODO: step 5
  return { rate: 0, flagged: [], n };
}

/**
 * The eval runner. For every task, sample n answers from model(question, next), grade each with
 * grader(answer, task.answer, task.question) and count c = number with score >= 0.5. Then for each k in ks
 * compute per-task passAtK(n, c, k), average across tasks and bootstrap a CI over the per-task values.
 * Throw if any k > n. Return
 *   { nTasks, n, results: [{ id, question, reference, n, c, answers }], metrics: [{ k, value, lo, hi, se, passPowK }] }
 */
export function runEval({ tasks, model, grader, n = 10, ks = [1, 5, 10], next = rng(0), B = 1000 }) {
  // TODO: step 5
  return { nTasks: tasks.length, n, results: [], metrics: ks.map((k) => ({ k, value: 0, lo: 0, hi: 0, se: 0, passPowK: 0 })) };
}
