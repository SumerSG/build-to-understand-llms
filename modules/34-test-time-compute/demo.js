import { rng } from 'lib/util.js';

// Goal demo: spend more inference compute on 300 reasoning problems in four different ways, and watch
// which selector turns that compute into accuracy and what each one costs to serve.

const NS = [1, 2, 4, 8, 16, 32, 64];
// Beam shapes whose sampled step count roughly matches N whole traces: k children per prefix, width w.
// Sampled steps = k + (s − 1)·w·k versus N·s for best-of-N.
const BEAM = { 1: [1, 1], 2: [1, 2], 4: [2, 2], 8: [2, 4], 16: [4, 4], 32: [4, 8], 64: [8, 8] };
const PRM_SIGMA = 0.5;     // the learned PRM is noisy too: each step score is ok + 0.5·noise
const PROMPT_TOKENS = 60;  // a short word problem
const BUSY = 300;          // a busy server decodes all 300 problems' samples in one batch

/**
 * GPU-seconds per problem on a busy server: all BUSY problems' sequences decode in one batch (your
 * ttcCost), so the fixed cost of each iteration is shared. A reward model the size of the policy
 * reads `scoredTokens` tokens per problem, which costs about tPerToken each (a batched prefill).
 */
function serverCost(m, samples, traceTokens, scoredTokens = 0) {
  const c = m.ttcCost({ samples: samples * BUSY, traceTokens: Math.round(traceTokens), promptTokens: PROMPT_TOKENS * BUSY });
  return c.gpuSeconds / BUSY + scoredTokens * m.DEFAULT_COST.tPerToken;
}

const mean = (xs) => xs.reduce((s, x) => s + x, 0) / Math.max(1, xs.length);
const pct = (x) => `${(100 * x).toFixed(1)}%`;

export default async function demo(m, lab) {
  const chains = m.makeChains(300, 34);
  const next = rng(3434);
  lab.log(`${chains.length} chains of 3–6 steps; per-step error rates from 3% to 40%. Sampling 64 traces each…`);

  // 1. The pool: 64 independent traces per chain, and a sanity check against (1 − e)^s.
  const pools = [];
  let single = 0, predicted = 0;
  for (let i = 0; i < chains.length; i++) {
    const c = chains[i];
    const pool = Array.from({ length: 64 }, () => m.sampleTrace(c, next));
    pools.push(pool);
    single += pool.filter((t) => m.verify(c, t)).length / 64 / chains.length;
    predicted += (1 - c.errorRate) ** c.steps / chains.length;
    if (i % 50 === 0) { lab.progress(0.2 * i / chains.length, 'sampling'); await lab.tick(); }
  }
  lab.check(Math.abs(single - predicted) < 0.03, `per-sample accuracy ${single.toFixed(3)} should be close to the mean of (1 − e)^s, ${predicted.toFixed(3)}`);
  const traceTokens = mean(pools.flat().map((t) => t.tokens));

  // 2. Parallel sampling with four selectors, at N = 1 … 64.
  const acc = { vote: [], ormBest: [], ormVote: [], oracle: [], beam: [] };
  const tokens = { vote: [], beam: [] };
  for (const N of NS) {
    let vote = 0, ormBest = 0, ormVote = 0, oracle = 0, tok = 0;
    for (let i = 0; i < chains.length; i++) {
      const c = chains[i], pool = pools[i].slice(0, N);
      const answers = pool.map((t) => t.answer);
      vote += m.majorityVote(answers) === c.answer ? 1 : 0;
      ormBest += m.verify(c, m.bestOfN(pool, (t) => m.ormScore(c, t)));
      ormVote += m.weightedVote(answers, pool.map((t) => m.ormScore(c, t))) === c.answer ? 1 : 0;
      oracle += m.verify(c, m.bestOfN(pool, (t) => m.verify(c, t)));
      tok += pool.reduce((s, t) => s + t.tokens, 0);
    }
    acc.vote.push(vote / chains.length); acc.ormBest.push(ormBest / chains.length);
    acc.ormVote.push(ormVote / chains.length); acc.oracle.push(oracle / chains.length);
    tokens.vote.push(tok / chains.length);
    await lab.tick();
  }
  lab.progress(0.4, 'beam search');

  // 3. Process supervision: step-level beam search against a noisy PRM, at a matched number of sampled steps.
  const beamCost = [];
  for (const N of NS) {
    const [w, k] = BEAM[N];
    let right = 0, tok = 0, gpuS = 0;
    for (let i = 0; i < chains.length; i++) {
      const c = chains[i];
      let spent = 0;
      const expand = (p) => Array.from({ length: k }, () => { const s = m.sampleStep(c, p, next); spent += s.tokens; return [...p, s]; });
      const r = m.stepBeamSearch({ beamWidth: w, depth: c.steps, expand, prm: (p) => m.prmScore(c, { steps: p }, { sigma: PRM_SIGMA }) });
      const t = m.finishTrace(r.best);
      right += m.verify(c, t);
      tok += spent + m.ANSWER_TOKENS;
      if (i % 60 === 0) await lab.tick();
    }
    acc.beam.push(right / chains.length);
    tokens.beam.push(tok / chains.length);
    // Every level decodes w·k candidate steps side by side, so each problem keeps w·k sequences in the
    // batch for the length of one trace; the PRM then reads every sampled token once.
    beamCost.push(serverCost(m, w * k, tok / chains.length / (w * k), tok / chains.length));
    lab.progress(0.4 + 0.4 * (NS.indexOf(N) + 1) / NS.length, `beam search, N ≈ ${N}`);
  }

  const x = NS.map((n) => Math.log2(n));
  lab.plot({ title: 'Accuracy vs samples per problem (x = log₂ N, so 0 … 6 means N = 1 … 64)', x,
    series: [
      { name: 'best-of-N, oracle verifier (= pass@N)', values: acc.oracle },
      { name: 'beam search, noisy PRM (matched steps)', values: acc.beam },
      { name: 'majority vote (self-consistency)', values: acc.vote },
      { name: 'ORM-weighted vote', values: acc.ormVote },
      { name: 'best-of-N, ORM with length bias', values: acc.ormBest },
    ], xlabel: 'log₂ N', ylabel: 'accuracy' });
  const ormPeak = Math.max(...acc.ormBest);
  const ormPeakN = NS[acc.ormBest.indexOf(ormPeak)];

  // 4. Cost: GPU-seconds per problem from your ttcCost, and the accuracy each strategy reaches within a budget.
  // Each problem's prompt is prefilled once and its KV shared by that problem's N samples.
  const parCost = NS.map((N) => serverCost(m, N, traceTokens));
  const rmCost = NS.map((N, j) => serverCost(m, N, traceTokens, tokens.vote[j]));   // plus an ORM pass over every trace
  const strategies = [
    { name: 'majority vote', cost: parCost, acc: acc.vote },
    { name: 'best-of-N (ORM)', cost: rmCost, acc: acc.ormBest },
    { name: 'ORM-weighted vote', cost: rmCost, acc: acc.ormVote },
    { name: 'beam search (PRM)', cost: beamCost, acc: acc.beam },
    { name: 'best-of-N (oracle)', cost: parCost, acc: acc.oracle },
  ];
  const all = [...parCost, ...rmCost, ...beamCost];
  const lo = Math.log2(Math.min(...all)), hi = Math.log2(Math.max(...all));
  const grid = Array.from({ length: 25 }, (_, i) => lo + (hi - lo) * i / 24);
  const within = (s, budget) => {                  // the best accuracy this strategy reaches at cost ≤ budget
    let best = NaN;
    s.cost.forEach((c, i) => { if (c <= budget * (1 + 1e-9) && !(s.acc[i] <= best)) best = s.acc[i]; });
    return best;
  };
  lab.plot({ title: 'Accuracy reachable within a GPU-second budget per problem (x = log₂ seconds)', x: grid.map((g) => +g.toFixed(2)),
    series: strategies.map((s) => ({ name: s.name, values: grid.map((g) => within(s, 2 ** g)) })),
    xlabel: 'log₂ GPU-seconds per problem', ylabel: 'accuracy' });

  const budget = rmCost[NS.indexOf(8)];             // what eight samples and an ORM pass over them cost
  const atBudget = strategies.map((s) => within(s, budget));
  lab.bar({ title: `Accuracy within ${(1000 * budget).toFixed(1)} GPU-ms per problem on a busy server (the cost of best-of-8 with an ORM)`,
    labels: strategies.map((s) => s.name), values: atBudget.map((a) => (Number.isFinite(a) ? a : 0)) });

  // 5. Thinking budgets: one sample, and a vote over 8, with every trace cut to maxTokens.
  const budgets = [16, 24, 32, 40, 48, 56, 64, 72, 80];
  const oneAcc = [], voteAcc = [];
  for (const b of budgets) {
    let one = 0, v8 = 0;
    for (let i = 0; i < chains.length; i++) {
      const c = chains[i], cut = pools[i].slice(0, 8).map((t) => m.withBudget(t, b));
      one += m.verify(c, cut[0]);
      v8 += m.majorityVote(cut.map((t) => t.answer)) === c.answer ? 1 : 0;
    }
    oneAcc.push(one / chains.length); voteAcc.push(v8 / chains.length);
  }
  lab.plot({ title: 'Budget forcing: accuracy when every trace is cut to maxTokens', x: budgets,
    series: [{ name: 'one sample', values: oneAcc }, { name: 'majority vote over 8', values: voteAcc }],
    xlabel: 'maxTokens per trace', ylabel: 'accuracy' });
  lab.progress(0.9, 'cost table');

  // 6. The serving bill, per problem, at N = 16, and at the scale of real reasoning traces.
  const i16 = NS.indexOf(16);
  const rows = [];
  const row = (name, strat, accuracy) => {
    const c = m.ttcCost(strat);
    rows.push([name, strat.samples, strat.traceTokens, c.decodeTokens, +(c.peakKVBytes / 2 ** 20).toFixed(1), +c.gpuSeconds.toFixed(3), accuracy]);
    return c;
  };
  const T = Math.round(traceTokens);
  row('one sample', { samples: 1, traceTokens: T, promptTokens: PROMPT_TOKENS }, pct(acc.vote[0]));
  row('vote / best-of-16', { samples: 16, traceTokens: T, promptTokens: PROMPT_TOKENS }, `${pct(acc.vote[i16])} / ${pct(acc.ormBest[i16])} (ORM)`);
  row('beam 4 × 4', { samples: 16, traceTokens: T, promptTokens: PROMPT_TOKENS }, pct(acc.beam[i16]));
  const longOne = row('long trace (o1-scale)', { samples: 1, traceTokens: 8000, promptTokens: 500 }, 'n/a (scale)');
  const longPar = row('16 × long traces', { samples: 16, traceTokens: 8000, promptTokens: 500 }, 'n/a (scale)');
  lab.table({ title: 'Serving one problem alone (Llama-3-8B-like dims, bf16 KV, the iteration model from the continuous batching module): sixteen sequences cost barely more than one',
    columns: ['strategy', 'sequences', 'tokens each', 'decode tokens', 'peak KV (MiB)', 'GPU-seconds', 'accuracy'], rows });

  const reach = strategies.map((s) => {             // the cheapest cost at which each strategy reaches 80%
    const ok = s.cost.filter((c, i) => s.acc[i] >= 0.8);
    return ok.length ? `${s.name} ${(1000 * Math.min(...ok)).toFixed(1)} GPU-ms` : `${s.name} never`;
  });
  const practical = strategies.filter((s) => !s.name.includes('oracle'));
  const practicalAcc = practical.map((s) => within(s, budget)).map((a) => (Number.isFinite(a) ? a : -1));
  const bestAtBudget = practical[practicalAcc.indexOf(Math.max(...practicalAcc))];
  lab.done([
    `Per-sample accuracy on 300 chains is **${pct(single)}** (the mean of \`(1 − e)^s\` predicts ${pct(predicted)}).`,
    `At N = 64: the oracle verifier reaches **${pct(acc.oracle.at(-1))}** (pass@64), majority vote **${pct(acc.vote.at(-1))}**, ORM-weighted vote ${pct(acc.ormVote.at(-1))}, and beam search with a noisy PRM **${pct(acc.beam.at(-1))}**.`,
    `Best-of-N against the length-biased ORM peaks at **${pct(ormPeak)} at N = ${ormPeakN}**, then falls to **${pct(acc.ormBest.at(-1))}** at N = 64: more samples give the reward model more long, wrong traces to prefer.`,
    `Within ${(1000 * budget).toFixed(1)} GPU-ms per problem on a busy server (best-of-8 with an ORM), the best selector short of the oracle is **${bestAtBudget.name}** at ${pct(Math.max(...practicalAcc))}; the oracle reaches ${pct(atBudget.at(-1))}.`,
    `Cheapest cost to reach 80% accuracy: ${reach.join(', ')}.`,
    `A budget of 40 tokens per trace cuts single-sample accuracy to ${pct(oneAcc[budgets.indexOf(40)])}.`,
    `At real reasoning scale, one 8,000-token trace costs **${longOne.gpuSeconds.toFixed(1)} GPU-s** and ${(longOne.peakKVBytes / 2 ** 30).toFixed(2)} GiB of KV; sixteen in parallel cost ${longPar.gpuSeconds.toFixed(1)} GPU-s and ${(longPar.peakKVBytes / 2 ** 30).toFixed(1)} GiB.`,
  ].join(' '));
}
