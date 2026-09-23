// Speculative decoding (Leviathan et al. 2023; Chen et al. 2023).
//
// Symbols used throughout:
//   p  = the TARGET model's next-token distribution (the one whose samples you want)
//   q  = the DRAFT model's next-token distribution (cheap, usually right, never trusted)
//   x  = a token id the draft proposed by sampling x ~ q
//   u  = a uniform random number in [0, 1) from the seeded rng
//   K  = how many tokens the draft guesses before the target checks them
//   α  = acceptance rate: the probability a drafted token survives verification
//
// A "model" is a function model(ids, n) that returns n probability vectors: the next-token
// distributions after the last n positions of ids (row j is the distribution over the token that
// follows ids[ids.length - n + j]). The draft is called with n = 1; the target with n = K + 1, which is
// the single teacher-forced verify pass. Distributions are plain arrays of numbers that sum to 1.

import { sampleIndex } from 'lib/util.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/** Draw an index from a probability vector using one uniform u in [0, 1): the inverse-CDF walk of the decoding module. */
export function sampleFrom(probs, u) {
  return sampleIndex(probs, u);
}

/**
 * The simplest possible model: a first-order Markov chain. P[a] is the distribution over the token that
 * follows token a. This is the model interface every function below uses: model(ids, n) -> rows.
 */
export function markovModel(P) {
  return function model(ids, n) {
    if (n > ids.length) throw new Error(`markovModel: asked for ${n} rows but only ${ids.length} tokens of context`);
    const rows = [];
    for (let j = ids.length - n; j < ids.length; j++) rows.push(P[ids[j]].slice());
    return rows;
  };
}

// ---------- step 1: the acceptance rule ----------

/**
 * Probability of keeping drafted token x: min(1, p[x] / q[x]).
 * A token the draft could not have drawn (q[x] = 0) is kept (return 1) rather than dividing by zero.
 */
export function acceptProb(p, q, x) {
  if (q[x] <= 0) return 1;
  // TODO: step 1 — the ratio, clipped to 1
  return 1;
}

/** Keep x when the uniform u falls below its acceptance probability. */
export function shouldAccept(p, q, x, u) {
  // TODO: step 1
  return true;
}

// ---------- step 2: the residual distribution ----------

/**
 * norm(max(0, p − q)): the target mass the draft under-proposed, renormalised to sum to 1.
 * When p and q are identical there is nothing left over; return a copy of p.
 */
export function residual(p, q) {
  // TODO: step 2
  return Array.from(p);
}

/**
 * One position, end to end: propose x ~ q, accept with min(1, p/q), otherwise draw from the residual.
 * `next` is the seeded rng. Returns { token, accepted }. The token is distributed exactly as p.
 */
export function speculativeSampleOne(p, q, next) {
  // TODO: step 2
  return { token: 0, accepted: true };
}

// ---------- step 3: draft K, verify once ----------

/**
 * Draft K tokens one at a time from `draft`, verify all of them with ONE call to `target` (n = K + 1 rows),
 * accept the longest prefix the rule allows, then emit one more token: the residual sample at the first
 * rejection, or the "bonus" token from the target's last row when nothing was rejected.
 * Returns { tokens, accepted } with tokens.length === accepted + 1. Do not mutate ctx.
 */
export function speculativeStep(target, draft, ctx, K, next) {
  // TODO: step 3
  return { tokens: [], accepted: 0 };
}

// ---------- step 4: acceptance rate and tokens per step ----------

/** The analytic acceptance rate for one position: Σ min(p, q), which equals 1 − total variation distance. */
export function acceptanceRate(p, q) {
  // TODO: step 4
  return 0;
}

/** Expected tokens per step when each drafted token is accepted independently with probability alpha: (1 − alpha^(K+1)) / (1 − alpha). */
export function expectedTokensPerStep(alpha, K) {
  // TODO: step 4 (mind alpha = 1)
  return 1;
}

/**
 * Generate maxNewTokens tokens after `prompt` with repeated speculative steps and measure what happened.
 * Returns { tokens, steps, drafted, examined, accepted, alpha, tokensPerStep, runLengths }:
 *   drafted   = steps × K
 *   examined  = drafted tokens that verification actually reached (a rejection discards the rest of that step)
 *   alpha     = accepted / examined
 *   tokensPerStep = 1 + accepted / steps
 *   runLengths[i] = number of steps that accepted exactly i tokens, i = 0..K
 */
export function generateSpeculative(target, draft, prompt, { K = 4, maxNewTokens = 64, next } = {}) {
  if (typeof next !== 'function') throw new Error('generateSpeculative: pass a seeded rng function as `next`');
  // TODO: step 4
  return { tokens: [], steps: 0, drafted: 0, examined: 0, accepted: 0, alpha: 0, tokensPerStep: 1, runLengths: new Array(K + 1).fill(0) };
}

// ---------- step 5: the speedup model ----------

/**
 * Speedup over plain decoding. One target decode step costs 1. Each drafted token costs c (draft time /
 * target time). Verifying K + 1 tokens in one pass costs 1 + rho * K: rho = 0 when the pass is memory-bound
 * (the extra tokens ride along for free), rho = 1 when it is compute-bound (each token costs a full step).
 */
export function speedup({ alpha, K, c, rho = 0 }) {
  // TODO: step 5
  return 1;
}

/** The K in 1..maxK with the largest modelled speedup (the smallest K on a tie). */
export function bestK({ alpha, c, rho = 0, maxK = 16 }) {
  // TODO: step 5
  return 1;
}
