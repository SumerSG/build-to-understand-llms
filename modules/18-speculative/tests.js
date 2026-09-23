import { sampleIndex } from 'lib/util.js';

// ---------- helpers shared by the tests ----------

/** A Markov "model" whose every row is the same distribution: model(ids, n) returns n copies of `row`. */
function constantModel(row) {
  return (ids, n) => Array.from({ length: n }, () => row.slice());
}

/** A Markov transition table where token a is always followed by (a + 1) % V, for a chain with no randomness. */
function successorTable(V) {
  return Array.from({ length: V }, (_, a) => Array.from({ length: V }, (_, b) => (b === (a + 1) % V ? 1 : 0)));
}

/** Empirical frequency of each id in `samples` over a vocabulary of size V. */
function histogram(samples, V) {
  const h = new Array(V).fill(0);
  for (const s of samples) h[s] += 1 / samples.length;
  return h;
}

export const tests = [
  // ---------- step 1: the acceptance rule ----------
  { step: 'accept', name: 'acceptProb is min(1, p[x] / q[x])', run(m, T) {
    const p = [0.5, 0.3, 0.2], q = [0.25, 0.5, 0.25];
    T.close(m.acceptProb(p, q, 0), 1, 1e-9, 'p[0] / q[0] = 2 must be clipped to 1: a token the target likes MORE than the draft is always kept');
    T.close(m.acceptProb(p, q, 1), 0.6, 1e-9, 'p[1] / q[1] = 0.3 / 0.5: a token the draft over-proposed is kept only with probability p/q');
    T.close(m.acceptProb(p, q, 2), 0.8, 1e-9, 'p[2] / q[2] = 0.2 / 0.25');
    T.close(m.acceptProb([0.5, 0.5], [1, 0], 1), 1, 1e-9, 'q[x] = 0: the draft could not have proposed x, so define the ratio as 1 rather than dividing by zero');
  } },
  { step: 'accept', name: 'shouldAccept keeps x exactly when u < min(1, p/q)', run(m, T) {
    const p = [0.5, 0.3, 0.2], q = [0.25, 0.5, 0.25];
    T.eq(m.shouldAccept(p, q, 1, 0.59), true, 'u = 0.59 is below the 0.6 threshold for token 1, so it is kept');
    T.eq(m.shouldAccept(p, q, 1, 0.61), false, 'u = 0.61 is above the 0.6 threshold for token 1, so it is rejected');
    T.eq(m.shouldAccept(p, q, 0, 0.999), true, 'token 0 has acceptance probability 1: every u in [0, 1) keeps it');
    T.eq(m.shouldAccept(p, q, 2, 0.0), true, 'u = 0 is below every positive threshold');
    T.eq(m.shouldAccept([0, 1], [0.5, 0.5], 0, 0), false, 'p[x] = 0 gives acceptance probability 0, and the target must never emit a token it gives zero probability, not even at u = 0: compare with a strict <, not <=');
  } },
  { step: 'accept', name: 'over many trials the KEPT tokens are distributed as min(p, q), not as q', run(m, T) {
    // x ~ q, keep with min(1, p/q): P(keep x) = q[x] * min(1, p[x]/q[x]) = min(p[x], q[x]).
    const p = [0.1, 0.45, 0.45], q = [0.7, 0.2, 0.1];
    const next = T.rng(11);
    const N = 6000;
    const kept = [0, 0, 0];
    for (let i = 0; i < N; i++) {
      const x = sampleIndex(q, next());
      if (m.shouldAccept(p, q, x, next())) kept[x] += 1 / N;
    }
    T.close(kept, [0.1, 0.2, 0.1], 0.03, 'fraction of trials that kept each token should be min(p, q) per token: always accepting gives q = [0.7, 0.2, 0.1], a deterministic p > q rule gives [0, 0.2, 0.1]');
    const total = kept[0] + kept[1] + kept[2];
    T.close(total, 0.4, 0.03, 'the overall acceptance rate must be Σ min(p, q) = 0.4: this is the α that sets your speedup');
  } },

  // ---------- step 2: the residual distribution ----------
  { step: 'residual', name: 'residual is max(0, p − q) renormalised, on a hand-checkable example', run(m, T) {
    const r = m.residual([0.5, 0.3, 0.2], [0.2, 0.1, 0.7]);
    T.close(r, [0.6, 0.4, 0], 1e-9, 'max(0, p − q) = [0.3, 0.2, 0] sums to 0.5, so the residual is [0.6, 0.4, 0]; forgetting the clip gives a negative entry, forgetting to renormalise gives [0.3, 0.2, 0]');
    T.close(r[0] + r[1] + r[2], 1, 1e-9, 'the residual is a probability distribution: it must sum to 1');
    T.eq(r[2], 0, 'where the draft proposed MORE mass than the target (q ≥ p) the residual must be exactly 0: those tokens were already over-represented');
  } },
  { step: 'residual', name: 'identical p and q leave no residual: return p itself, never NaN', run(m, T) {
    const p = [0.2, 0.5, 0.3];
    const r = m.residual(p, [0.2, 0.5, 0.3]);
    T.ok(r.every((v) => Number.isFinite(v)), 'max(0, p − q) is all zeros here; dividing by its sum gives NaN unless you handle the case');
    T.close(r, p, 1e-9, 'with p = q every token is accepted, so the residual is never drawn from; returning p keeps the function total');
  } },
  { step: 'residual', name: 'speculativeSampleOne draws one fresh uniform for the proposal, one for the acceptance test and one for the residual, in that order', run(m, T) {
    const p = [0.1, 0.45, 0.45], q = [0.9, 0.05, 0.05];
    // A scripted rng: hands out exactly these uniforms and complains if asked for more.
    const script = (us) => { let i = 0; return () => { if (i >= us.length) throw new Error(`speculativeSampleOne drew more than the ${us.length} uniforms this case needs`); return us[i++]; }; };
    T.eq(m.speculativeSampleOne(p, q, script([0.05, 0.5, 0.7])), { token: 2, accepted: false },
      'u = 0.05 proposes token 0; its acceptance probability is 0.1 / 0.9 ≈ 0.11 and the SECOND uniform 0.5 is above it, so it is rejected and the THIRD uniform 0.7 draws token 2 from the residual [0, 0.5, 0.5]. Reusing the proposal uniform for the acceptance test (0.05 < 0.11) would keep token 0: the two draws become correlated and the output is no longer exactly p');
    T.eq(m.speculativeSampleOne(p, q, script([0.05, 0.08])), { token: 0, accepted: true },
      'u = 0.05 proposes token 0 and 0.08 < 0.11 keeps it; an accepted proposal uses exactly two uniforms (the residual is only drawn on rejection)');
    T.eq(m.speculativeSampleOne(p, q, script([0.93, 0.99])), { token: 1, accepted: true },
      'u = 0.93 proposes token 1 (cumulative q is 0.9, 0.95, 1); min(1, 0.45 / 0.05) = 1, so every u keeps it');
  } },
  { step: 'residual', name: 'speculativeSampleOne reproduces the TARGET distribution even from a bad draft', run(m, T) {
    // The draft puts 90% on token 0; the target puts 10% there. The output must still be exactly p.
    const p = [0.1, 0.45, 0.45], q = [0.9, 0.05, 0.05];
    const next = T.rng(5);
    const N = 6000;
    const tokens = [];
    let accepted = 0;
    for (let i = 0; i < N; i++) {
      const r = m.speculativeSampleOne(p, q, next);
      T.ok(Number.isInteger(r.token) && typeof r.accepted === 'boolean', 'return { token, accepted }');
      tokens.push(r.token);
      if (r.accepted) accepted += 1 / N;
    }
    T.close(histogram(tokens, 3), p, 0.03, 'the empirical distribution must match p: sampling p on rejection instead of the residual gives about [0.18, 0.41, 0.41]; skipping renormalisation over-weights the last token');
    T.close(accepted, 0.2, 0.03, 'the fraction accepted must be Σ min(p, q) = 0.2');
  } },

  // ---------- step 3: draft K, verify once ----------
  { step: 'verify', name: 'draft is called K times with n = 1; target exactly once with n = K + 1', run(m, T) {
    const P = successorTable(6);
    const draftCalls = [], targetCalls = [];
    const draft = (ids, n) => { draftCalls.push([ids.length, n]); return m.markovModel(P)(ids, n); };
    const target = (ids, n) => { targetCalls.push([ids.length, n]); return m.markovModel(P)(ids, n); };
    const ctx = [0, 1, 2];
    m.speculativeStep(target, draft, ctx, 3, T.rng(1));
    T.eq(draftCalls.map((c) => c[1]), [1, 1, 1], 'the draft is autoregressive: K calls, one next-token row each');
    T.eq(draftCalls.map((c) => c[0]), [3, 4, 5], 'each draft call sees the context plus the tokens drafted so far');
    T.eq(targetCalls.length, 1, 'verification is ONE teacher-forced pass over all K drafted tokens; calling the target per token forfeits the whole speedup');
    T.eq(targetCalls[0], [6, 4], 'the target sees ctx + K drafted tokens and returns K + 1 rows: one to check each draft and one for the bonus token');
    T.eq(ctx, [0, 1, 2], 'the caller\'s context must not be mutated');
  } },
  { step: 'verify', name: 'a perfect draft is accepted in full and the step emits K + 1 tokens, the last one the bonus', run(m, T) {
    const model = m.markovModel(successorTable(8));
    const r = m.speculativeStep(model, model, [3], 4, T.rng(2));
    T.eq(r.accepted, 4, 'when draft and target agree exactly, min(1, p/q) = 1 for every drafted token');
    T.eq(r.tokens, [4, 5, 6, 7, 0], 'four accepted drafts (4, 5, 6, 7) plus the bonus token 0 sampled from the target\'s (K + 1)-th row: a step always returns accepted + 1 tokens');
  } },
  { step: 'verify', name: 'a rejection truncates the drafts: tokens.length === accepted + 1 and the replacement comes from the residual', run(m, T) {
    const V = 5;
    const target = m.markovModel(successorTable(V));            // a -> a + 1
    const Pd = successorTable(V);
    Pd[2] = [1, 0, 0, 0, 0];                                     // the draft goes 2 -> 0 where the target goes 2 -> 3
    const draft = m.markovModel(Pd);
    let r = m.speculativeStep(target, draft, [1], 4, T.rng(3));
    T.eq(r.accepted, 1, 'the draft proposes 2 (agreed), then 0 (the target gives it zero probability, so p/q = 0 and it is rejected)');
    T.eq(r.tokens, [2, 3], 'after the rejection at position 1 the residual is the target row itself (3); drafts beyond the rejection are discarded, so no bonus token');
    r = m.speculativeStep(target, draft, [2], 4, T.rng(4));
    T.eq(r, { tokens: [3], accepted: 0 }, 'a rejection at position 0 yields exactly one token, drawn from the residual');
  } },
  { step: 'verify', name: 'the first emitted token is distributed as the target, whatever the draft proposes', run(m, T) {
    const p = [0.2, 0.5, 0.3], q = [0.7, 0.2, 0.1];
    const target = constantModel(p), draft = constantModel(q);
    const next = T.rng(9);
    const firsts = [];
    for (let i = 0; i < 3000; i++) firsts.push(m.speculativeStep(target, draft, [0], 3, next).tokens[0]);
    T.close(histogram(firsts, 3), p, 0.04, 'position 0 is a speculativeSampleOne with the target\'s first verify row: its output must be p, not q');
  } },

  { step: 'verify', name: 'the bonus token is drawn from the target\'s last row, not the draft\'s', run(m, T) {
    // K = 1 with constant models: a step that accepts its one draft emits a second token, the bonus.
    const p = [0.2, 0.5, 0.3], q = [0.7, 0.2, 0.1];
    const target = constantModel(p), draft = constantModel(q);
    const next = T.rng(13);
    const bonus = [];
    for (let i = 0; i < 4000; i++) {
      const r = m.speculativeStep(target, draft, [0], 1, next);
      T.eq(r.tokens.length, r.accepted + 1, 'a step emits accepted + 1 tokens');
      if (r.accepted === 1) bonus.push(r.tokens[1]);
    }
    T.ok(bonus.length > 1700 && bonus.length < 2300, `Σ min(p, q) = 0.5, so about half of 4000 K = 1 steps should accept their draft; got ${bonus.length}`);
    T.close(histogram(bonus, 3), p, 0.04, 'the bonus is a free draw from row K of the verify pass, so it is distributed as p; drawing it from the draft gives about q = [0.7, 0.2, 0.1]');
  } },

  // ---------- step 4: acceptance rate and tokens per step ----------
  { step: 'measure', name: 'acceptanceRate is Σ min(p, q), i.e. 1 − total variation distance', run(m, T) {
    T.close(m.acceptanceRate([0.5, 0.3, 0.2], [0.2, 0.5, 0.3]), 0.7, 1e-9, 'min per token: 0.2 + 0.3 + 0.2');
    T.close(m.acceptanceRate([0.2, 0.5, 0.3], [0.2, 0.5, 0.3]), 1, 1e-9, 'identical distributions: everything is accepted');
    T.close(m.acceptanceRate([1, 0], [0, 1]), 0, 1e-9, 'disjoint support: nothing is ever accepted');
    T.close(m.acceptanceRate([0.1, 0.45, 0.45], [0.9, 0.05, 0.05]), 0.2, 1e-9, 'the same 0.2 the step-2 test measured empirically');
  } },
  { step: 'measure', name: 'expectedTokensPerStep is (1 − α^(K+1)) / (1 − α) with the right limits', run(m, T) {
    T.close(m.expectedTokensPerStep(0, 4), 1, 1e-9, 'α = 0: every step still emits one token (the residual sample)');
    T.close(m.expectedTokensPerStep(1, 4), 5, 1e-9, 'α = 1: all K drafts plus the bonus token, and no division by zero');
    T.close(m.expectedTokensPerStep(0.5, 3), 1.875, 1e-9, '1 + 0.5 + 0.25 + 0.125: the sum of α^i for i = 0..K');
    T.close(m.expectedTokensPerStep(0.8, 4), 3.3616, 1e-9, '(1 − 0.8^5) / 0.2');
    T.ok(m.expectedTokensPerStep(0.8, 8) > m.expectedTokensPerStep(0.8, 4), 'more drafts can only add expected tokens (with diminishing returns)');
  } },
  { step: 'measure', name: 'generateSpeculative measures α as accepted / EXAMINED and emits exactly maxNewTokens tokens', run(m, T) {
    // Σ min(p, q) = 0.5 in every state, so the true α is 0.5 regardless of the context.
    const target = constantModel([0.5, 0.5, 0, 0]);
    const draft = constantModel([0.25, 0.25, 0.25, 0.25]);
    const r = m.generateSpeculative(target, draft, [0], { K: 4, maxNewTokens: 600, next: T.rng(21) });
    T.eq(r.tokens.length, 600, 'return exactly maxNewTokens tokens; the last step may overshoot and must be trimmed');
    T.ok(r.tokens.every((t) => t === 0 || t === 1), 'every emitted token must be one the TARGET could produce (tokens 2 and 3 have zero target probability); a drafted token that slipped through unverified would show up here');
    T.close(r.alpha, 0.5, 0.05, 'α = accepted / examined ≈ 0.5: dividing by drafted (steps × K) instead gives about 0.23 because tokens after a rejection were never examined');
    T.close(r.tokensPerStep, 1.9375, 0.15, 'measured tokens per step ≈ (1 − 0.5^5) / 0.5 = 1.9375');
    T.close(r.tokensPerStep, 1 + r.accepted / r.steps, 1e-9, 'tokensPerStep is 1 + accepted / steps: every step emits one extra token');
    T.eq(r.runLengths.length, 5, 'runLengths[i] for i = 0..K counts steps that accepted exactly i tokens');
    T.eq(r.runLengths.reduce((a, b) => a + b, 0), r.steps, 'the run-length histogram must account for every step');
    T.eq(r.drafted, r.steps * 4, 'drafted = steps × K');
  } },
  { step: 'measure', name: 'a perfect draft gives α = 1 and every step accepts all K', run(m, T) {
    const model = m.markovModel(successorTable(7));
    const r = m.generateSpeculative(model, model, [0], { K: 3, maxNewTokens: 40, next: T.rng(2) });
    T.close(r.alpha, 1, 1e-9, 'draft = target: min(1, p/q) = 1 for every token');
    T.eq(r.runLengths, [0, 0, 0, r.steps], 'every step accepted exactly K = 3 tokens');
    T.eq(r.steps, 10, '40 tokens at 4 per step is 10 steps');
    T.eq(r.tokens.slice(0, 8), [1, 2, 3, 4, 5, 6, 0, 1], 'the emitted sequence follows the chain a -> a + 1 mod 7');
  } },

  // ---------- step 5: the speedup model ----------
  { step: 'speedup', name: 'speedup divides expected tokens per step by the cost of one speculative step', run(m, T) {
    T.close(m.speedup({ alpha: 0.8, K: 4, c: 0, rho: 0 }), 3.3616, 1e-6, 'free draft, memory-bound verify: the speedup IS the expected tokens per step');
    T.close(m.speedup({ alpha: 0.8, K: 4, c: 0.1, rho: 0 }), 3.3616 / 1.4, 1e-6, 'K drafts at 0.1 each cost 0.4 on top of the one verify pass');
    T.close(m.speedup({ alpha: 0.8, K: 4, c: 0.1 }), 3.3616 / 1.4, 1e-6, 'rho defaults to 0 (memory-bound)');
    T.close(m.speedup({ alpha: 0.8, K: 4, c: 0.1, rho: 1 }), 3.3616 / 5.4, 1e-6, 'compute-bound (rho = 1): the K extra verified tokens each cost a full step, and speculation is SLOWER than plain decoding');
    T.ok(m.speedup({ alpha: 0.2, K: 4, c: 0.2, rho: 0 }) < 1, 'a poor draft (α = 0.2) at c = 0.2 loses: 1.25 tokens per step for 1.8 units of time');
  } },
  { step: 'speedup', name: 'bestK is the K in 1..maxK with the largest modelled speedup', run(m, T) {
    const brute = ({ alpha, c, rho = 0, maxK = 16 }) => {
      let best = 1, bv = -Infinity;
      for (let K = 1; K <= maxK; K++) {
        const s = m.expectedTokensPerStep(alpha, K) / (1 + K * c + rho * K);
        if (s > bv) { best = K; bv = s; }
      }
      return best;
    };
    for (const cfg of [{ alpha: 0.9, c: 0.02 }, { alpha: 0.5, c: 0.02 }, { alpha: 0.3, c: 0.3 }, { alpha: 0.95, c: 0.05, rho: 0.1 }, { alpha: 0.7, c: 0.1, maxK: 3 }]) {
      T.eq(m.bestK(cfg), brute(cfg), `bestK(${JSON.stringify(cfg)}) must match a brute-force search over 1..maxK`);
    }
    T.ok(m.bestK({ alpha: 0.9, c: 0.02 }) > m.bestK({ alpha: 0.5, c: 0.02 }), 'a better draft justifies a longer draft window: bestK must grow with α');
    T.eq(m.bestK({ alpha: 0.3, c: 0.3 }), 1, 'an expensive, poor draft: only K = 1 is worth it');
    T.eq(m.bestK({ alpha: 0, c: 0 }), 1, 'α = 0 and c = 0 give exactly 1× at every K: on a tie return the SMALLEST K (replace only on a strictly greater speedup)');
    T.ok(m.bestK({ alpha: 0.7, c: 0.1, maxK: 3 }) <= 3, 'never exceed maxK');
  } },
];
