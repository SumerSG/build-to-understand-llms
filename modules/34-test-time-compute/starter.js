// Module 34 — Reasoning & test-time compute.
// Everything above the "step 1" line is done for you: read it, it sets the conventions. Below it,
// each TODO names the step it belongs to.
//
// Symbols used throughout:
//   s     = the number of steps in a chain (3–6)
//   e     = the chain's per-step error rate: the probability the reasoner slips on any one step
//   δᵢ    = the chain's slip at step i: a wrong step i lands exactly δᵢ away from the right value
//   N     = how many traces you sample for one problem
//   λ, σ  = the reward model's length bias and noise scale
//
// A chain is a problem:     { id, start, ops: [{ op: '+'|'-', arg }], values, answer, steps, errorRate, slips }
// A step is what the reasoner writes for one line of working:     { value, tokens }
// A trace is a full attempt: { steps: step[], answer, tokens }     (tokens includes the answer line)

import { rng, randn, randInt, choice, hash32 } from 'lib/util.js';

// ---------- worked examples, constants and given helpers (done for you) ----------

export const BASE_TOKENS = 8;    // every step writes about this many tokens ("51 − 17 = 34" plus framing)
export const HEDGE_MAX = 4;      // plus 0..HEDGE_MAX tokens of hedging, uniform
export const DETOUR_TOKENS = 6;  // a slipped step spends this many extra tokens going astray
export const ANSWER_TOKENS = 4;  // the closing "Final answer: 34"
export const SLIPS = [1, 10];    // the size of the reasoner's habitual mistakes: an off-by-one, a dropped carry

/** Apply one operation of a chain to a value. */
export function applyOp(value, { op, arg }) {
  if (op === '+') return value + arg;
  if (op === '-') return value - arg;
  throw new Error(`unknown op ${op}`);
}

/**
 * n arithmetic chains of 3–6 steps. Each has its own error rate e, uniform in [0.03, 0.40], so chains
 * range from easy to hard, and its own slips: step i always goes wrong the same way (by δᵢ, which is
 * ±1 or ±10 with one sign per chain), so wrong answers are not random noise. They cluster.
 */
export function makeChains(n, seed = 34) {
  const next = rng(seed);
  const chains = [];
  for (let id = 0; id < n; id++) {
    const steps = 3 + randInt(next, 4);
    const start = 10 + randInt(next, 90);
    const ops = [];
    const values = [];
    let v = start;
    for (let i = 0; i < steps; i++) {
      const o = { op: next() < 0.5 ? '+' : '-', arg: 1 + randInt(next, 49) };
      ops.push(o);
      v = applyOp(v, o);
      values.push(v);
    }
    const u = next();
    const errorRate = 0.03 + 0.37 * u;
    const sign = next() < 0.5 ? -1 : 1;
    const slips = ops.map(() => sign * choice(next, SLIPS));
    chains.push({ id, start, ops, values, answer: v, steps, errorRate, slips });
  }
  return chains;
}

/** Close a list of steps into a trace: the answer is the last step's value; tokens include the answer line. */
export function finishTrace(steps) {
  let tokens = ANSWER_TOKENS;
  for (const s of steps) tokens += s.tokens;
  return { steps, answer: steps.length ? steps[steps.length - 1].value : null, tokens };
}

/** Module 12's verifier, specialised to numeric answers: reward 1 when the final answer is right, else 0. */
export function verify(chain, trace) {
  return trace && trace.answer === chain.answer ? 1 : 0;
}

/** Module 13's unbiased pass@k: the chance that k of n samples, c of them correct, include a correct one. */
export function passAtK(n, c, k) {
  if (k < 1 || k > n) throw new Error(`passAtK: k=${k} must be in 1..n=${n}`);
  if (n - c < k) return 1;
  let p = 1;
  for (let i = 0; i < k; i++) p *= (n - c - i) / (n - i);
  return 1 - p;
}

/**
 * Module 15's KV-cache size for one sequence: 2 (K and V) · nLayer · nKVHead · headDim · tokens · bytes.
 * nKVHead defaults to nHead (plain multi-head attention); GQA models such as Llama 3 set it lower.
 */
export function cacheBytes(config, contextLen, { bytesPerElement = 4 } = {}) {
  const nKVHead = config.nKVHead ?? config.nHead;
  const headDim = config.headDim ?? config.nEmbd / config.nHead;
  return 2 * config.nLayer * nKVHead * headDim * contextLen * bytesPerElement;
}

/**
 * A reward model's error on one specific input, as a standard-normal number. A real reward model is a
 * deterministic function of its input, so its mistakes are too: the same (tag, chain, steps) always
 * gets the same draw, and a prefix gets the same draw however it is later extended.
 */
export function rmNoise(tag, chain, steps) {
  let key = `${tag}|${chain.id}|`;
  for (const s of steps) key += `${s.value}:${s.tokens},`;
  return randn(rng(hash32(key)));
}

// ---------- step 1: the reasoner and the step checker ----------

/**
 * The reasoner writes the next step after `prefix` (an array of steps already written).
 * Draw u = next(); the step slips when u < chain.errorRate. The right value is applyOp(previous value,
 * chain.ops[i]) where the previous value is the prefix's last value (or chain.start); a slip adds
 * chain.slips[i]. Then draw the hedge: tokens = BASE_TOKENS + randInt(next, HEDGE_MAX + 1), plus
 * DETOUR_TOKENS on a slip.
 */
export function sampleStep(chain, prefix, next) {
  // Completion problem: finding the step index and the value to continue from is written for you.
  const i = prefix.length;
  if (i >= chain.steps) throw new Error(`sampleStep: chain ${chain.id} has only ${chain.steps} steps`);
  const prev = i === 0 ? chain.start : prefix[i - 1].value;
  // TODO: step 1. Three lines: the right value of step i (applyOp on prev), whether it slips (one draw
  // of next() against chain.errorRate), and its token count (a second draw for the hedge). Then return
  // { value, tokens }, adding chain.slips[i] to the value on a slip.
  return { value: prev, tokens: 0 };
}

/** A full attempt: sample chain.steps steps in order, each continuing from the steps before it. */
export function sampleTrace(chain, next) {
  // TODO: step 1. Grow a list of steps with sampleStep, then close it with finishTrace.
  return finishTrace([]);
}

/**
 * The program verifier at step level: for each step, is it the correct operation applied to the value
 * the trace itself had just before it? A step that computes correctly from a wrong input is still true.
 * Returns an array of booleans, one per step in trace.steps (which may be a partial trace).
 */
export function checkSteps(chain, trace) {
  // TODO: step 1
  return trace.steps.map(() => true);
}

// ---------- step 2: self-consistency ----------

/**
 * The most common answer. null/undefined answers are abstentions and are not counted; if every answer
 * abstains, return null. Ties go to the answer whose first occurrence comes earliest.
 * majorityVote([3, 5, 5, 3, 5]) === 5;  majorityVote([7, 3, 3, 7]) === 7
 */
export function majorityVote(answers) {
  // TODO: step 2
  return answers.length ? answers[0] : null;
}

// ---------- step 3: best-of-N and outcome reward models ----------

/**
 * The answer with the largest total score: each answer's scores are summed over every sample that gave it.
 * Same abstention and tie rules as majorityVote.  weightedVote([1, 2, 2], [0.5, 0.3, 0.4]) === 2
 */
export function weightedVote(answers, scores) {
  // TODO: step 3
  return null;
}

/** The trace with the highest scorer(trace); the earliest wins a tie. */
export function bestOfN(traces, scorer) {
  // TODO: step 3
  return traces[0];
}

/**
 * Extra tokens per step beyond BASE_TOKENS, the answer line excluded: the surface feature the outcome
 * reward model latched onto. A trace whose steps average 10 tokens has verbosity 2.
 */
export function verbosity(trace) {
  // TODO: step 3
  return 0;
}

/** Outcome reward model: verify + λ·verbosity + σ·rmNoise('orm', chain, trace.steps), λ = 0.3, σ = 0.1 by default. */
export function ormScore(chain, trace, { lambda = 0.3, sigma = 0.1 } = {}) {
  // TODO: step 3
  return verify(chain, trace);
}

// ---------- step 4: process reward models and step-level beam search ----------

/**
 * Per-step PRM scores: 1 for a correct step, 0 for a wrong one (checkSteps), plus
 * σ·rmNoise('prm', chain, the steps up to and including that one) when sigma is not 0.
 */
export function stepScores(chain, trace, { sigma = 0 } = {}) {
  // TODO: step 4
  return trace.steps.map(() => 1);
}

/** PRM score of a (possibly partial) trace: the minimum step score, 1 for an empty trace. */
export function prmScore(chain, trace, opts = {}) {
  // TODO: step 4
  return 1;
}

/**
 * Step-level beam search. Start from one empty prefix. At each of `depth` levels, call
 * expand(prefix) on every prefix in the beam (each call returns an array of longer prefixes), score all
 * the candidates with prm(prefix), and keep the `beamWidth` highest (a stable sort: earlier candidates
 * win ties). Return { best, beam, sampled } where sampled counts every candidate generated.
 */
export function stepBeamSearch({ beamWidth, depth, expand, prm }) {
  // TODO: step 4
  return { best: [], beam: [[]], sampled: 0 };
}

// ---------- step 5: thinking budgets and the cost of test-time compute ----------

/**
 * Budget forcing: keep whole steps while the steps so far plus the answer line fit in maxTokens. If
 * every step fits, return the trace unchanged (truncated: false). Otherwise stop and force an answer:
 * the last kept step's value (null if none), tokens = kept steps + ANSWER_TOKENS, truncated: true.
 * Never modify the trace you are given.
 */
export function withBudget(trace, maxTokens) {
  // TODO: step 5
  return { ...trace, truncated: false };
}

/** Module 16's defaults, approximately a Llama-3-8B-class model in bf16 on one H100. */
export const LLAMA3_8B = { nLayer: 32, nHead: 32, nKVHead: 8, headDim: 128, nEmbd: 4096 };
export const DEFAULT_COST = { model: LLAMA3_8B, bytesPerElement: 2, tFixed: 0.005, tPerToken: 0.00005, gpus: 1 };

/**
 * Serving cost of one strategy on one problem. strategy = { samples, traceTokens, promptTokens = 0 }:
 * `samples` sequences decode together as one batch, each writing `traceTokens` tokens, after a single
 * shared prefill of the prompt (skipped when promptTokens is 0). Returns
 *   decodeTokens = samples · traceTokens
 *   decodeSteps  = traceTokens                       (one batched iteration per token position)
 *   peakKVBytes  = cacheBytes(prompt) + samples · cacheBytes(traceTokens)   (prompt KV shared once)
 *   seconds      = [tFixed + promptTokens·tPerToken] + decodeSteps · (tFixed + samples·tPerToken)
 *   gpuSeconds   = seconds · gpus
 * `config` fields you leave out fall back to DEFAULT_COST.
 */
export function ttcCost(strategy, config = DEFAULT_COST) {
  // TODO: step 5
  return { decodeTokens: 0, decodeSteps: 0, peakKVBytes: 0, seconds: 0, gpuSeconds: 0 };
}
