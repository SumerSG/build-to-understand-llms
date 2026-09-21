// Module 18 — reference solution: speculative decoding (Leviathan et al. 2023; Chen et al. 2023).
//
// Symbols used throughout:
//   p  = the TARGET model's next-token distribution (the one whose samples you want)
//   q  = the DRAFT model's next-token distribution (cheap, usually right, never trusted)
//   x  = a token id the draft proposed by sampling x ~ q
//   u  = a uniform random number in [0, 1) from the seeded rng
//   K  = how many tokens the draft guesses before the target checks them
//   α  = acceptance rate: the probability a drafted token survives verification
// A "model" is a function model(ids, n) that returns n probability vectors: the next-token
// distributions after the last n positions of ids (row j is the distribution over the token that
// follows ids[ids.length - n + j]). The draft is called with n = 1; the target with n = K + 1, which is
// the single teacher-forced verify pass.

import { sampleIndex } from 'lib/util.js';

// ---------- worked examples ----------

/** Draw an index from a probability vector using one uniform u in [0, 1): the inverse-CDF rule of module 14. */
export function sampleFrom(probs, u) {
  return sampleIndex(probs, u);
}

/**
 * The simplest possible model: a first-order Markov chain. P[a] is the distribution over the token that
 * follows token a. The model interface is the one every function below uses: model(ids, n) -> rows.
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

/** Probability of keeping drafted token x: min(1, p[x] / q[x]). A token the draft could not have drawn (q[x] = 0) is kept. */
export function acceptProb(p, q, x) {
  if (q[x] <= 0) return 1;
  return Math.min(1, p[x] / q[x]);
}

/** Keep x when the uniform u falls below its acceptance probability. */
export function shouldAccept(p, q, x, u) {
  return u < acceptProb(p, q, x);
}

// ---------- step 2: the residual distribution ----------

/** norm(max(0, p - q)): where the draft under-proposed relative to the target. Equal p and q have no residual; return p. */
export function residual(p, q) {
  const out = new Array(p.length);
  let z = 0;
  for (let i = 0; i < p.length; i++) {
    out[i] = Math.max(0, p[i] - q[i]);
    z += out[i];
  }
  if (z <= 0) return Array.from(p);
  for (let i = 0; i < out.length; i++) out[i] /= z;
  return out;
}

/** One position, end to end: propose x ~ q, accept with min(1, p/q), otherwise draw from the residual. The result is distributed exactly as p. */
export function speculativeSampleOne(p, q, next) {
  const x = sampleFrom(q, next());
  if (shouldAccept(p, q, x, next())) return { token: x, accepted: true };
  return { token: sampleFrom(residual(p, q), next()), accepted: false };
}

// ---------- step 3: draft K, verify once ----------

/**
 * Draft K tokens one at a time from `draft`, verify all of them with ONE call to `target` (n = K + 1 rows),
 * accept the longest prefix the rule allows, then emit one more token: the residual sample at the first
 * rejection, or the "bonus" token from the target's last row when nothing was rejected.
 * Returns { tokens, accepted } with tokens.length === accepted + 1.
 */
export function speculativeStep(target, draft, ctx, K, next) {
  const drafted = [];
  const qs = [];
  for (let i = 0; i < K; i++) {
    const q = draft(ctx.concat(drafted), 1)[0];
    drafted.push(sampleFrom(q, next()));
    qs.push(q);
  }
  const ps = target(ctx.concat(drafted), K + 1);
  const tokens = [];
  for (let i = 0; i < K; i++) {
    if (shouldAccept(ps[i], qs[i], drafted[i], next())) {
      tokens.push(drafted[i]);
    } else {
      tokens.push(sampleFrom(residual(ps[i], qs[i]), next()));
      return { tokens, accepted: i };
    }
  }
  tokens.push(sampleFrom(ps[K], next()));
  return { tokens, accepted: K };
}

// ---------- step 4: acceptance rate and tokens per step ----------

/** The analytic acceptance rate for one position: Σ min(p, q) = 1 - total variation distance between p and q. */
export function acceptanceRate(p, q) {
  let a = 0;
  for (let i = 0; i < p.length; i++) a += Math.min(p[i], q[i]);
  return a;
}

/** Expected tokens per step when every drafted token is accepted independently with probability alpha: (1 - alpha^(K+1)) / (1 - alpha). */
export function expectedTokensPerStep(alpha, K) {
  if (alpha >= 1) return K + 1;
  return (1 - Math.pow(alpha, K + 1)) / (1 - alpha);
}

/**
 * Generate maxNewTokens tokens after `prompt` with repeated speculative steps and measure what happened.
 * alpha is accepted / examined, where a drafted token is "examined" only if verification reached it
 * (tokens after a rejection are discarded unexamined). runLengths[i] counts the steps that accepted exactly i tokens.
 */
export function generateSpeculative(target, draft, prompt, { K = 4, maxNewTokens = 64, next } = {}) {
  if (typeof next !== 'function') throw new Error('generateSpeculative: pass a seeded rng function as `next`');
  let ctx = prompt.slice();
  const tokens = [];
  const runLengths = new Array(K + 1).fill(0);
  let steps = 0, accepted = 0, examined = 0;
  while (tokens.length < maxNewTokens) {
    const r = speculativeStep(target, draft, ctx, K, next);
    steps++;
    accepted += r.accepted;
    examined += r.accepted < K ? r.accepted + 1 : K;
    runLengths[r.accepted]++;
    for (const t of r.tokens) tokens.push(t);
    ctx = ctx.concat(r.tokens);
  }
  return {
    tokens: tokens.slice(0, maxNewTokens),
    steps,
    drafted: steps * K,
    examined,
    accepted,
    alpha: examined ? accepted / examined : 0,
    tokensPerStep: 1 + accepted / steps,
    runLengths,
  };
}

// ---------- step 5: the speedup model ----------

/**
 * Speedup over plain decoding. One target decode step costs 1. Each drafted token costs c (draft time / target
 * time). Verifying K + 1 tokens in one pass costs 1 + rho * K: rho = 0 when the pass is memory-bound (the
 * extra tokens ride along for free), rho = 1 when it is compute-bound (each token costs a full step).
 */
export function speedup({ alpha, K, c, rho = 0 }) {
  return expectedTokensPerStep(alpha, K) / (1 + K * c + rho * K);
}

/** The K in 1..maxK with the largest modelled speedup (the smallest K on a tie). */
export function bestK({ alpha, c, rho = 0, maxK = 16 }) {
  let best = 1, bestValue = -Infinity;
  for (let K = 1; K <= maxK; K++) {
    const s = speedup({ alpha, K, c, rho });
    if (s > bestValue) { best = K; bestValue = s; }
  }
  return best;
}
