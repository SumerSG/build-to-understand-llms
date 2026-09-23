import { rng, randInt } from 'lib/util.js';

// ---------- fixtures: the tests carry their own copy of the reasoner so that a bug in step 1 cannot
// make the later steps fail for the wrong reason ----------

const BASE = 8, HEDGE = 4, DETOUR = 6, ANSWER = 4;

function chainOf(id, start, ops, errorRate, slips) {
  const values = [];
  let v = start;
  for (const [op, arg] of ops) { v = op === '+' ? v + arg : v - arg; values.push(v); }
  return { id, start, ops: ops.map(([op, arg]) => ({ op, arg })), values, answer: v, steps: ops.length, errorRate, slips };
}

// A 3-step chain: 10 → 15 → 12 → 19.
const small = (errorRate = 0, slips = [1, 1, 1], id = 900) => chainOf(id, 10, [['+', 5], ['-', 3], ['+', 7]], errorRate, slips);

function sixStep(id, errorRate, next) {
  const ops = Array.from({ length: 6 }, () => [next() < 0.5 ? '+' : '-', 1 + randInt(next, 49)]);
  const sign = next() < 0.5 ? -1 : 1;
  return chainOf(id, 10 + randInt(next, 90), ops, errorRate, ops.map(() => sign * (next() < 0.5 ? 1 : 10)));
}

function refStep(chain, prefix, next) {
  const i = prefix.length;
  const prev = i === 0 ? chain.start : prefix[i - 1].value;
  const o = chain.ops[i];
  const right = o.op === '+' ? prev + o.arg : prev - o.arg;
  const slipped = next() < chain.errorRate;
  const tokens = BASE + randInt(next, HEDGE + 1) + (slipped ? DETOUR : 0);
  return { value: slipped ? right + chain.slips[i] : right, tokens };
}

function finish(steps) {
  return { steps, answer: steps.length ? steps[steps.length - 1].value : null, tokens: ANSWER + steps.reduce((s, x) => s + x.tokens, 0) };
}

function refTrace(chain, next) {
  const steps = [];
  while (steps.length < chain.steps) steps.push(refStep(chain, steps, next));
  return finish(steps);
}

const trace = (values, tokens) => finish(values.map((value, i) => ({ value, tokens: tokens[i] })));

function* subsets(n, k, start = 0, acc = []) {
  if (acc.length === k) { yield acc.slice(); return; }
  for (let i = start; i < n; i++) { acc.push(i); yield* subsets(n, k, i + 1, acc); acc.pop(); }
}

export const tests = [
  // ---------- step 1: the reasoner ----------
  { step: 'reasoner', name: 'sampleStep continues from the prefix and slips by the chain\'s own δ', run(m, T) {
    const next = T.rng(1);
    const c = small(0);
    const s0 = m.sampleStep(c, [], next);
    T.ok(s0 && typeof s0 === 'object', `sampleStep must return a step object { value, tokens }, got ${JSON.stringify(s0)}`);
    T.eq(s0.value, 15, 'with errorRate 0 the first step is start (10) + 5 = 15');
    T.ok(s0.tokens >= BASE && s0.tokens <= BASE + HEDGE, `a correct step costs BASE_TOKENS + 0..HEDGE_MAX tokens (8..12), got ${s0.tokens}`);
    T.eq(m.sampleStep(c, [{ value: 20, tokens: 8 }], next).value, 17,
      'step 2 must continue from the prefix\'s own last value (20 − 3 = 17), not the chain\'s true value 15: that is how one slip carries forward');
    const bad = small(1, [10, 10, 10]);
    const s1 = m.sampleStep(bad, [], next);
    T.eq(s1.value, 25, 'with errorRate 1 every step slips: 15 + slips[0] (10) = 25');
    T.ok(s1.tokens >= BASE + DETOUR && s1.tokens <= BASE + HEDGE + DETOUR, `a slipped step spends DETOUR_TOKENS more (14..18), got ${s1.tokens}`);
    const neg = small(1, [1, -10, 1]);
    T.eq(m.sampleStep(neg, [{ value: 15, tokens: 8 }], next).value, 2, 'step i slips by slips[i]: 15 − 3 + (−10) = 2');
    T.throws(() => m.sampleStep(c, [{ value: 15, tokens: 8 }, { value: 12, tokens: 8 }, { value: 19, tokens: 8 }], next), 'a prefix that already has chain.steps steps has no next step: throw');
    const seen = (chain) => [...new Set(Array.from({ length: 300 }, () => m.sampleStep(chain, [], next).tokens))].sort((a, b) => a - b);
    T.eq(seen(c), [8, 9, 10, 11, 12], 'over 300 correct steps every hedge 0..HEDGE_MAX should appear: randInt(next, HEDGE_MAX + 1) returns 0..4, randInt(next, HEDGE_MAX) would never write 12 tokens');
    T.eq(seen(bad), [14, 15, 16, 17, 18], 'over 300 slipped steps the token count should cover BASE_TOKENS + 0..HEDGE_MAX + DETOUR_TOKENS = 14..18');
  } },
  { step: 'reasoner', name: 'sampleTrace keeps the books: steps, answer, tokens, determinism', run(m, T) {
    const c = small(0);
    const t = m.sampleTrace(c, T.rng(2));
    T.eq(t.steps.length, 3, 'a trace has exactly chain.steps steps');
    T.eq(t.steps.map((s) => s.value), [15, 12, 19], 'with errorRate 0 the trace reproduces chain.values');
    T.eq(t.answer, 19, 'the answer is the last step\'s value');
    T.eq(t.tokens, t.steps.reduce((s, x) => s + x.tokens, 0) + ANSWER, 'tokens = the steps\' tokens + ANSWER_TOKENS (use finishTrace)');
    const h = small(0.5, [1, 10, 1]);
    T.eq(m.sampleTrace(h, T.rng(9)), m.sampleTrace(h, T.rng(9)), 'the same seed must give the same trace: draw randomness only from next');
    T.eq(m.sampleTrace(h, T.rng(9)), refTrace(h, T.rng(9)),
      'with the same seed your trace should match the reference draw for draw: per step, one next() for the slip first, then one randInt(next, HEDGE_MAX + 1) for the hedge. Another order gives the right statistics but not the demo numbers quoted in the concept');
  } },
  { step: 'reasoner', name: 'per-sample accuracy is (1 − e)^s', run(m, T) {
    for (const [e, ops] of [[0.2, [['+', 5], ['-', 3], ['+', 7], ['+', 11], ['-', 20]]], [0.1, [['+', 5], ['-', 3], ['+', 7]]]]) {
      const c = chainOf(901, 10, ops, e, ops.map(() => 1));
      const next = T.rng(3);
      let right = 0;
      const n = 4000;
      for (let i = 0; i < n; i++) right += m.sampleTrace(c, next).answer === c.answer ? 1 : 0;
      const expect = (1 - e) ** ops.length;
      T.close(right / n, expect, 0.03, `with e = ${e} and ${ops.length} steps a trace is right only if no step slips: (1 − ${e})^${ops.length} = ${expect.toFixed(3)}. `
        + 'About 1 − e means a slip is not carried into later steps; about 1 − e·s or a fixed rate means the slip is drawn once per trace instead of once per step.');
    }
  } },
  { step: 'reasoner', name: 'checkSteps judges each step against the trace\'s own previous value', run(m, T) {
    const c = small(0, [10, 10, 10]);
    T.eq(m.checkSteps(c, trace([15, 12, 19], [8, 8, 8])), [true, true, true], 'a fully correct trace');
    T.eq(m.checkSteps(c, trace([15, 22, 29], [8, 14, 8])), [true, false, true],
      'step 2 slipped (15 − 3 should be 12, not 22), but step 3 computed 22 + 7 = 29 correctly. Comparing with chain.values would mark step 3 wrong too; the step checker checks the operation, not the running total');
    T.eq(m.checkSteps(c, trace([25], [14])), [false], 'a partial trace is checked step by step as far as it goes');
    const h = chainOf(902, 40, [['+', 5], ['-', 3], ['+', 7], ['-', 9]], 0.4, [1, 10, 1, 10]);
    const next = T.rng(4);
    for (let i = 0; i < 200; i++) {
      const t = m.sampleTrace(h, next);
      const ok = m.checkSteps(h, t);
      const drift = ok.reduce((s, good, j) => s + (good ? 0 : h.slips[j]), 0);
      T.eq(t.answer - h.answer, drift, 'wrong answers cluster: the final answer is off by exactly the sum of slips[i] over the steps checkSteps marks false');
    }
  } },

  // ---------- step 2: self-consistency ----------
  { step: 'vote', name: 'majorityVote on hand-made answer lists', run(m, T) {
    T.eq(m.majorityVote([3, 5, 5, 3, 5]), 5, 'the most common answer wins');
    T.eq(m.majorityVote([2, 9, 9]), 9, 'the answer need not be the first one sampled');
    T.eq(m.majorityVote([7, 3, 3, 7]), 7, 'a tie goes to the answer that appeared first, so the vote is deterministic');
    T.eq(m.majorityVote([4, 6, 6, 4, 4, 6]), 4, 'a tie goes to the answer that appeared first');
    T.eq(m.majorityVote([null, 4, null, undefined]), 4, 'null/undefined answers are abstentions and are not counted');
    T.eq(m.majorityVote([]), null, 'no answers: null');
    T.eq(m.majorityVote([null, null]), null, 'only abstentions: null');
    T.eq(m.majorityVote([0, 5, 0]), 0, '0 is an answer, not an abstention (chains can end at 0): test for null/undefined explicitly, not with !a');
    T.eq(m.majorityVote([-3, 2, -3, 2, 2]), 2, 'negative answers are counted like any other');
    const a = [1, 2, 2];
    m.majorityVote(a);
    T.eq(a, [1, 2, 2], 'majorityVote must not modify its input');
  } },
  { step: 'vote', name: 'self-consistency converges to the most common answer, right or wrong', run(m, T) {
    // Answers drawn from the reasoner's own distribution: correct + (sum of slips) with every slip +1.
    const draw = (e, s, next) => { let k = 0; for (let i = 0; i < s; i++) if (next() < e) k++; return 100 + k; };
    const acc = (e, s, N, next, trials = 400) => {
      let right = 0;
      for (let t = 0; t < trials; t++) right += m.majorityVote(Array.from({ length: N }, () => draw(e, s, next))) === 100 ? 1 : 0;
      return right / trials;
    };
    const next = T.rng(5);
    const easy1 = acc(0.1, 3, 1, next), easy31 = acc(0.1, 3, 31, next);
    T.ok(easy31 > 0.97 && easy31 > easy1 + 0.2, `easy chain (e = 0.1, 3 steps; right answer 73% of samples): voting over 31 samples should be nearly always right, got ${easy1.toFixed(2)} at N = 1 and ${easy31.toFixed(2)} at N = 31`);
    const hard1 = acc(0.4, 3, 1, next), hard31 = acc(0.4, 3, 31, next);
    T.ok(hard1 > 0.15 && hard31 < 0.05, `hard chain (e = 0.4): the right answer has probability 0.216 but "off by one" has 0.432, so voting over 31 samples should almost never be right; got ${hard1.toFixed(2)} at N = 1 and ${hard31.toFixed(2)} at N = 31. More samples make self-consistency worse here`);
  } },

  // ---------- step 3: best-of-N and outcome reward models ----------
  { step: 'bestofn', name: 'bestOfN and weightedVote pick the highest score, earliest on ties', run(m, T) {
    const ts = [{ answer: 1 }, { answer: 2 }, { answer: 3 }, { answer: 4 }];
    const sc = new Map([[ts[0], 0.2], [ts[1], 0.9], [ts[2], 0.9], [ts[3], -1]]);
    T.ok(m.bestOfN(ts, (t) => sc.get(t)) === ts[1], 'bestOfN returns the trace object with the highest score; the earliest of two equal scores wins');
    T.ok(m.bestOfN([ts[3]], (t) => sc.get(t)) === ts[3], 'with one candidate, return it even when its score is negative');
    T.ok(m.bestOfN(ts, (t) => -sc.get(t)) === ts[3], 'bestOfN must use the scorer, not the position');
    T.eq(m.weightedVote([1, 2, 2], [0.9, 0.3, 0.4]), 1, 'weights add per answer: 1 has 0.9, 2 has 0.7');
    T.eq(m.weightedVote([1, 2, 2], [0.5, 0.3, 0.4]), 2, 'weights add per answer: 2 has 0.7, 1 has 0.5; the largest single score is not the rule');
    T.eq(m.weightedVote([5, 6, 6, 5], [0.5, 0.25, 0.25, 0]), 5, 'a tie (0.5 each) goes to the answer that appeared first');
    T.eq(m.weightedVote([null, 8], [5, 1]), 8, 'abstentions carry no weight');
    T.eq(m.weightedVote([3, 4, 4], [-0.2, -0.5, -0.4]), 3, 'totals can be negative (ORM noise): 3 has −0.2, 4 has −0.9. Start the best total at −Infinity, not 0');
    T.eq(m.weightedVote([0, 7], [1, 0.5]), 0, '0 is an answer, not an abstention');
    T.ok(m.bestOfN(ts, (t) => sc.get(t) - 5) === ts[1], 'when every score is negative, bestOfN still returns the highest (−4.1, the second trace), not the first');
  } },
  { step: 'bestofn', name: 'ormScore = verify + λ·verbosity + σ·noise', run(m, T) {
    const c = small(0);
    const good = trace([15, 12, 19], [8, 10, 12]);      // 30 step tokens over 3 steps: verbosity 10 − 8 = 2
    const bad = trace([15, 22, 29], [9, 16, 11]);       // 36 / 3 − 8 = 4
    T.close(m.ormScore(c, good, { lambda: 0, sigma: 0 }), 1, 1e-9, 'with λ = σ = 0 the ORM is the verifier: 1 for a right answer');
    T.close(m.ormScore(c, bad, { lambda: 0, sigma: 0 }), 0, 1e-9, 'with λ = σ = 0 the ORM is the verifier: 0 for a wrong answer');
    T.close(m.ormScore(c, good, { lambda: 0.5, sigma: 0 }), 1 + 0.5 * 2, 1e-9, 'verbosity is extra tokens PER STEP beyond BASE_TOKENS, answer line excluded: (34 − 4)/3 − 8 = 2');
    T.close(m.ormScore(c, bad, { lambda: 0.5, sigma: 0 }), 0.5 * 4, 1e-9, 'verbosity of the slipped trace: (40 − 4)/3 − 8 = 4');
    T.close(m.ormScore(c, bad, { lambda: 0, sigma: 0.7 }), 0.7 * m.rmNoise('orm', c, bad.steps), 1e-9, 'the noise term is σ · rmNoise(\'orm\', chain, trace.steps)');
    T.close(m.ormScore(c, good), 1 + 0.3 * 2 + 0.1 * m.rmNoise('orm', c, good.steps), 1e-9, 'defaults: λ = 0.3, σ = 0.1');
  } },
  { step: 'bestofn', name: 'with the oracle, best-of-N accuracy is exactly pass@N', run(m, T) {
    const c = small(0);
    const right = trace([15, 12, 19], [8, 8, 8]);
    const wrong = (d) => trace([15, 12, 19 + d], [8, 8, 14]);
    const pool = [wrong(1), right, wrong(10), wrong(1), right, wrong(2), wrong(1), right];   // n = 8, c = 3
    const oracle = (t) => m.verify(c, t);
    for (let N = 1; N <= 5; N++) {
      let hits = 0, total = 0;
      for (const idx of subsets(pool.length, N)) { hits += m.verify(c, m.bestOfN(idx.map((i) => pool[i]), oracle)); total++; }
      T.close(hits / total, m.passAtK(8, 3, N), 1e-9, `averaged over all ${total} ways to pick ${N} of the 8 traces (3 correct), best-of-${N} with a sound verifier is right exactly when a correct trace is among them: pass@${N} = ${m.passAtK(8, 3, N).toFixed(4)}`);
    }
  } },
  { step: 'bestofn', name: 'best-of-N against the default ORM peaks, then falls (overoptimisation)', run(m, T) {
    const chains = m.makeChains(200, 1);
    const Ns = [1, 2, 4, 8, 16, 64];
    const acc = Ns.map(() => 0);
    const next = T.rng(8);
    for (const c of chains) {
      const pool = Array.from({ length: 64 }, () => refTrace(c, next));
      Ns.forEach((N, j) => { acc[j] += m.verify(c, m.bestOfN(pool.slice(0, N), (t) => m.ormScore(c, t))) / chains.length; });
    }
    const peak = Math.max(...acc.slice(1, 5));
    const msg = `accuracy at N = ${Ns.join(', ')}: ${acc.map((a) => a.toFixed(3)).join(', ')}.`;
    T.ok(peak > acc[0] + 0.1, `${msg} Moderate N should beat N = 1 by more than 0.1: the ORM mostly prefers right answers`);
    T.ok(acc[5] < peak - 0.05, `${msg} At N = 64 accuracy should fall at least 0.05 below its peak: with enough samples the ORM finds long, wrong traces its length bias overrates`);
  } },

  // ---------- step 4: process reward models and step-level beam search ----------
  { step: 'process', name: 'the PRM scores every step and aggregates with the minimum', run(m, T) {
    const c = small(0, [10, 10, 10]);
    const t = trace([15, 22, 29], [8, 14, 8]);
    T.eq(m.stepScores(c, t), [1, 0, 1], 'noise-free step scores: 1 for a correct step, 0 for a wrong one (use checkSteps)');
    T.close(m.prmScore(c, t), 0, 1e-9, 'one bad step sinks the trace: min(1, 0, 1) = 0. The mean (0.67) would rank it above a trace that is merely long');
    T.close(m.prmScore(c, trace([15, 12, 19], [8, 8, 8])), 1, 1e-9, 'an all-correct trace scores 1');
    T.close(m.prmScore(c, { steps: [] }), 1, 1e-9, 'an empty prefix has made no mistake yet: score 1');
    T.close(m.prmScore(c, { steps: t.steps.slice(0, 1) }), 1, 1e-9, 'a partial trace is scored on the steps it has');
    const sig = 0.5;
    const want = [1, 0, 1].map((ok, i) => ok + sig * m.rmNoise('prm', c, t.steps.slice(0, i + 1)));
    T.close(m.stepScores(c, t, { sigma: sig }), want, 1e-9, 'with noise, step i scores ok + σ · rmNoise(\'prm\', chain, steps.slice(0, i + 1)): the noise depends only on the prefix, so beam search sees consistent scores');
    T.close(m.prmScore(c, t, { sigma: sig }), Math.min(...want), 1e-9, 'prmScore is the minimum of the noisy step scores');
  } },
  { step: 'process', name: 'stepBeamSearch keeps the best beamWidth prefixes at every depth', run(m, T) {
    const expand = (p) => [0, 1, 2].map((d) => [...p, d]);
    const sum = (p) => p.reduce((s, x) => s + x, 0);
    const r = m.stepBeamSearch({ beamWidth: 2, depth: 3, expand, prm: sum });
    T.eq(r.best, [2, 2, 2], 'with prm = sum of digits the best prefix is [2, 2, 2]');
    T.eq(r.beam, [[2, 2, 2], [2, 2, 1]], 'the beam holds the top beamWidth candidates; [2, 2, 1] and [2, 1, 2] tie at 5 and the one generated first wins');
    T.eq(r.sampled, 3 + 6 + 6, 'sampled counts every candidate: 3 from the single empty root, then 2 prefixes × 3 children at each later depth');
    // A trap: [0] looks worse than [1], but [0, 1] is the best prefix of all. Greedy (width 1) misses it.
    const trap = (p) => sum(p) + (p[0] === 0 && p[1] === 1 ? 10 : 0);
    const e2 = (p) => [0, 1].map((d) => [...p, d]);
    T.eq(m.stepBeamSearch({ beamWidth: 1, depth: 2, expand: e2, prm: trap }).best, [1, 1], 'width 1 is greedy: it commits to [1] and never sees [0, 1]');
    T.eq(m.stepBeamSearch({ beamWidth: 2, depth: 2, expand: e2, prm: trap }).best, [0, 1], 'width 2 keeps [0] alive long enough to find [0, 1]');
  } },
  { step: 'process', name: 'with a noise-free PRM, beam search beats best-of-N at equal sampled steps', run(m, T) {
    const next = T.rng(11);
    const chains = Array.from({ length: 80 }, (_, i) => sixStep(1000 + i, 0.35, next));
    let beamRight = 0, bonRight = 0, beamSampled = 0;
    for (const c of chains) {
      const prm = (p) => m.prmScore(c, { steps: p });
      const r = m.stepBeamSearch({ beamWidth: 2, depth: 6, expand: (p) => Array.from({ length: 4 }, () => [...p, refStep(c, p, next)]), prm });
      T.ok(r && Array.isArray(r.best), 'stepBeamSearch must return { best, beam, sampled } with best an array of steps');
      beamRight += finish(r.best).answer === c.answer ? 1 : 0;
      beamSampled += r.sampled;
      const N = Math.max(1, Math.floor(r.sampled / 6));   // the same number of sampled steps, spent on whole traces
      const pool = Array.from({ length: N }, () => refTrace(c, next));
      bonRight += m.bestOfN(pool, (t) => prm(t.steps)).answer === c.answer ? 1 : 0;
    }
    T.eq(beamSampled, 80 * (4 + 5 * 8), 'width 2 × 4 children: 4 candidates at depth 1, then 8 at each of the 5 later depths');
    const a = beamRight / chains.length, b = bonRight / chains.length;
    T.ok(a > 0.9 && a > b + 0.2, `6-step chains with e = 0.35 (a whole trace is right 7.5% of the time). Beam search should reach > 0.9 and beat best-of-7 by more than 0.2; got beam ${a.toFixed(2)}, best-of-N ${b.toFixed(2)}. Step-level search throws a slip away where it happens instead of wasting the rest of the trace on it`);
  } },

  // ---------- step 5: budgets and cost ----------
  { step: 'budget', name: 'withBudget keeps whole steps that fit with the answer line, then forces an answer', run(m, T) {
    const t = trace([15, 12, 19], [10, 12, 9]);   // 31 step tokens + 4 = 35
    const before = JSON.stringify(t);
    const full = m.withBudget(t, 35);
    T.eq(full.answer, 19, 'a budget of exactly 35 fits the whole trace (10 + 12 + 9 + 4): use <=, not <');
    T.eq(full.truncated, false, 'an untouched trace reports truncated: false');
    T.eq(full.steps.length, 3, 'nothing was cut');
    const cut = m.withBudget(t, 34);
    T.eq([cut.steps.length, cut.answer, cut.tokens, cut.truncated], [2, 12, 26, true],
      'at 34 the third step no longer fits with the answer line: keep 2 steps, force the answer 12 (the last kept value), tokens 10 + 12 + 4 = 26');
    T.eq([m.withBudget(t, 14).steps.length, m.withBudget(t, 14).answer], [1, 15], 'at 14: one step (10) plus the answer line (4)');
    const none = m.withBudget(t, 13);
    T.eq([none.steps.length, none.answer, none.tokens, none.truncated], [0, null, 4, true], 'at 13 not even one step fits: no answer, only the answer line is spent');
    T.eq(JSON.stringify(t), before, 'withBudget must not modify the trace it is given');
  } },
  { step: 'budget', name: 'ttcCost matches the KV formula of the KV cache module and the iteration model of the continuous batching module', run(m, T) {
    const cfg = { model: { nLayer: 32, nHead: 32, nKVHead: 8, headDim: 128, nEmbd: 4096 }, bytesPerElement: 2, tFixed: 0.005, tPerToken: 0.00005, gpus: 1 };
    const perTok = 2 * 32 * 8 * 128 * 2;            // 131,072 bytes: K and V, 32 layers, 8 KV heads of 128, bf16
    const par = m.ttcCost({ samples: 8, traceTokens: 1000, promptTokens: 200 }, cfg);
    T.eq(par.decodeTokens, 8000, '8 samples × 1000 tokens');
    T.eq(par.decodeSteps, 1000, 'the 8 samples decode as one batch: one iteration per token position');
    T.eq(par.peakKVBytes, perTok * (200 + 8 * 1000), 'the prompt\'s KV is stored once and shared; each sample adds its own 1000 tokens. With nHead instead of nKVHead the answer is 4× too big');
    T.close(par.seconds, (0.005 + 200 * 0.00005) + 1000 * (0.005 + 8 * 0.00005), 1e-9, 'one prefill iteration (tFixed + 200·tPerToken) plus 1000 decode iterations of tFixed + 8·tPerToken = 5.415 s');
    T.close(par.gpuSeconds, par.seconds, 1e-9, 'on one GPU, GPU-seconds = seconds');
    const seq = m.ttcCost({ samples: 1, traceTokens: 8000, promptTokens: 200 }, cfg);
    T.close(seq.seconds, 0.015 + 8000 * 0.00505, 1e-9, 'one sequence thinking 8× longer: 8000 iterations of tFixed + tPerToken = 40.4 s, 7.5× the parallel cost for the same 8000 decode tokens');
    T.eq(seq.peakKVBytes, par.peakKVBytes, 'the same number of decode tokens holds the same KV bytes, whether in one long trace or eight short ones');
    T.close(m.ttcCost({ samples: 4, traceTokens: 100 }, { ...cfg, gpus: 2 }).gpuSeconds, 2 * 100 * (0.005 + 4 * 0.00005), 1e-9, 'with no prompt there is no prefill iteration; on 2 GPUs GPU-seconds is 2 × seconds');
    let partial = null;
    try { partial = m.ttcCost({ samples: 4, traceTokens: 100 }, { gpus: 2 }); } catch (e) { T.fail(`ttcCost({ samples: 4, traceTokens: 100 }, { gpus: 2 }) threw "${e.message}": a config that sets only some fields must fall back to DEFAULT_COST for the rest ({ ...DEFAULT_COST, ...config })`); }
    T.ok(partial && Number.isFinite(partial.gpuSeconds), 'a config that sets only gpus must fall back to DEFAULT_COST for every other field: merge with { ...DEFAULT_COST, ...config }');
    T.close([partial.gpuSeconds, partial.peakKVBytes], [2 * 100 * (0.005 + 4 * 0.00005), 4 * 100 * perTok], 1e-9, 'with { gpus: 2 } the model, bytes and timings come from DEFAULT_COST');
  } },
];
