// Goal demo for module 18: the trained checkpoint (lib/checkpoints/tiny-gpt.json) is the TARGET; two drafts
// propose tokens for it — a bigram table counted from the corpus (cheap, often wrong) and the checkpoint
// itself at a lower temperature (expensive, usually right). YOUR acceptance rule, residual, verify step,
// measurements and speedup model do all of the decoding.

import { rng, softmaxArray, now } from 'lib/util.js';
import { loadModel, forward } from 'lib/infer.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { CORPUS } from 'lib/data.js';

export default async function demo(m, lab) {
  // ---------- 1. the guarantee, on a distribution you can read ----------
  const p = [0.10, 0.45, 0.45];   // target
  const q = [0.90, 0.05, 0.05];   // a draft that is confidently wrong
  const trials = 20000;
  const next0 = rng(1);
  const hist = [0, 0, 0], naive = [0, 0, 0];
  let accepted1 = 0;
  for (let i = 0; i < trials; i++) {
    const r = m.speculativeSampleOne(p, q, next0);
    hist[r.token] += 1 / trials;
    if (r.accepted) accepted1++;
    naive[m.sampleFrom(q, next0())] += 1 / trials;   // "trust the draft": what you get without verification
  }
  lab.table({
    title: `One position, ${trials.toLocaleString()} trials: your draft-then-verify output vs the target`,
    columns: ['token', 'target p', 'draft q', 'trust the draft', 'speculative (yours)', 'residual norm(max(0, p − q))'],
    rows: p.map((_, t) => [t, p[t], q[t], +naive[t].toFixed(3), +hist[t].toFixed(3), +m.residual(p, q)[t].toFixed(3)]),
  });
  const alpha1 = accepted1 / trials;
  lab.log(`accepted ${(100 * alpha1).toFixed(1)}% of draft proposals; analytic Σ min(p, q) = ${m.acceptanceRate(p, q).toFixed(3)}`);
  await lab.tick();

  // ---------- 2. the real target: the checkpoint ----------
  const json = (await import('lib/checkpoints/tiny-gpt.json', { with: { type: 'json' } })).default;
  const tokJson = (await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default;
  const model = loadModel(json);
  const tok = BPETokenizer.fromJSON(tokJson);
  const V = model.config.vocabSize, B = model.config.blockSize;
  let forwardCalls = 0;

  /** The checkpoint as a model(ids, n): one full forward over the last blockSize tokens, softmax of the last n rows. */
  function gptModel(temperature) {
    return (ids, n) => {
      const win = ids.slice(-B);
      forwardCalls++;
      const logits = forward(model, win);
      const T = win.length;
      const rows = [];
      for (let j = T - n; j < T; j++) rows.push(softmaxArray(Array.from(logits.data.subarray(j * V, (j + 1) * V)), temperature));
      return rows;
    };
  }
  const target = gptModel(1);

  /** A bigram draft: count which token follows which in the corpus, add-0.05 smoothing. Costs nothing per token. */
  const corpusIds = tok.encode(CORPUS);
  const counts = Array.from({ length: V }, () => new Float64Array(V));
  for (let i = 0; i + 1 < corpusIds.length; i++) counts[corpusIds[i]][corpusIds[i + 1]]++;
  const P = counts.map((row) => {
    const out = new Array(V);
    let z = 0;
    for (let j = 0; j < V; j++) { out[j] = row[j] + 0.05; z += out[j]; }
    return out.map((v) => v / z);
  });
  const bigram = m.markovModel(P);
  const selfDraft = gptModel(0.5);   // the target itself, sharpened: a proxy for "a smaller model of the same family"

  const prompt = tok.encode('The painter');
  const N = 200;          // tokens per speculative run with the bigram draft
  const NPLAIN = N;       // same length as the speculative runs: forward() is cheaper while the window is still filling up
  const NSELF = 120;      // shorter runs for the self-draft (K extra forwards per step) and the temperature sweep keep the demo inside its time budget

  // plain decoding, for the wall-clock baseline (one target forward per token)
  forwardCalls = 0;
  let t0 = now();
  {
    let ctx = prompt.slice();
    const nx = rng(3);
    for (let i = 0; i < NPLAIN; i++) {
      ctx.push(m.sampleFrom(target(ctx, 1)[0], nx()));
      if (i % 40 === 0) { lab.progress(0.1 + 0.1 * (i / NPLAIN), 'plain decoding'); await lab.tick(); }
    }
  }
  const plainMsPerTok = (now() - t0) / NPLAIN;
  const plainForwards = forwardCalls;

  // ---------- 3. K sweep with the bigram draft ----------
  const Ks = [1, 2, 4, 6];
  const sweep = [];
  for (let i = 0; i < Ks.length; i++) {
    forwardCalls = 0;
    t0 = now();
    const r = m.generateSpeculative(target, bigram, prompt, { K: Ks[i], maxNewTokens: N, next: rng(3) });
    sweep.push({ K: Ks[i], r, ms: now() - t0, forwards: forwardCalls });
    lab.progress(0.2 + 0.4 * ((i + 1) / Ks.length), `bigram draft, K = ${Ks[i]}`);
    await lab.tick();
  }
  const alphaBigram = sweep.find((s) => s.K === 4).r.alpha;
  lab.plot({
    title: `Tokens per verify step vs K (bigram draft, measured α at K = 4 is ${alphaBigram.toFixed(2)})`,
    x: Ks,
    series: [
      { name: 'measured', values: sweep.map((s) => +s.r.tokensPerStep.toFixed(3)) },
      { name: 'model (1 − α^(K+1)) / (1 − α)', values: Ks.map((K) => +m.expectedTokensPerStep(alphaBigram, K).toFixed(3)) },
      { name: 'ceiling K + 1', values: Ks.map((K) => K + 1) },
    ],
    xlabel: 'K (tokens drafted per step)',
    ylabel: 'tokens emitted per target pass',
  });

  // ---------- 4. the self-draft and a temperature sweep on the target ----------
  forwardCalls = 0;
  t0 = now();
  const selfRun = m.generateSpeculative(target, selfDraft, prompt, { K: 4, maxNewTokens: NSELF, next: rng(3) });
  const selfMs = now() - t0, selfForwards = forwardCalls;
  lab.progress(0.7, 'self-draft');
  await lab.tick();
  const temps = [0.5, 1, 2];
  const tempRuns = [];
  for (let i = 0; i < temps.length; i++) {
    tempRuns.push(temps[i] === 1 ? sweep.find((s) => s.K === 4).r : m.generateSpeculative(gptModel(temps[i]), bigram, prompt, { K: 4, maxNewTokens: NSELF, next: rng(3) }));
    lab.progress(0.75 + 0.2 * ((i + 1) / temps.length), `target temperature ${temps[i]}`);
    await lab.tick();
  }
  lab.bar({
    title: `Acceptance rate α of the bigram draft vs the TARGET's sampling temperature (K = 4; ${NSELF}–${N} tokens per run, so differences of a few points are noise)`,
    labels: temps.map((t) => `T = ${t}`),
    values: tempRuns.map((r) => +r.alpha.toFixed(3)),
  });

  // ---------- 5. where the drafts fail: per-position acceptance and run lengths ----------
  const K4 = sweep.find((s) => s.K === 4).r;
  const perPosition = (run, K) => {
    // examined at position i = steps that accepted >= i; accepted at i = steps that accepted >= i + 1
    const reached = (i) => run.runLengths.slice(i).reduce((a, b) => a + b, 0);
    return Array.from({ length: K }, (_, i) => (reached(i) ? +(reached(i + 1) / reached(i)).toFixed(3) : 0));
  };
  lab.plot({
    title: 'Acceptance rate by draft position (K = 4; later positions are reached by fewer steps, so they are noisier)',
    x: [1, 2, 3, 4],
    series: [
      { name: 'bigram draft', values: perPosition(K4, 4) },
      { name: 'self-draft (T = 0.5)', values: perPosition(selfRun, 4) },
    ],
    xlabel: 'position within the draft',
    ylabel: 'fraction accepted, given verification reached it',
  });
  lab.bar({
    title: 'Accepted run lengths per step, bigram draft, K = 4 (0 = first guess rejected)',
    labels: K4.runLengths.map((_, i) => `${i} accepted`),
    values: K4.runLengths,
  });

  // ---------- 6. the speedup model vs what this toy actually measured ----------
  const cBigram = 0.001, cSelf = 1;   // draft cost relative to one target pass: a table lookup vs the same model
  const rows = [
    ['bigram, K = 4', K4.alpha, K4.tokensPerStep, m.speedup({ alpha: K4.alpha, K: 4, c: cBigram }), m.speedup({ alpha: K4.alpha, K: 4, c: cBigram, rho: 1 }), m.bestK({ alpha: K4.alpha, c: cBigram })],
    ['self-draft T = 0.5, K = 4', selfRun.alpha, selfRun.tokensPerStep, m.speedup({ alpha: selfRun.alpha, K: 4, c: cSelf }), m.speedup({ alpha: selfRun.alpha, K: 4, c: cSelf, rho: 1 }), m.bestK({ alpha: selfRun.alpha, c: cSelf })],
  ];
  lab.table({
    title: 'Measured α and the speedup model (c = draft cost / target cost; rho = 0 memory-bound, rho = 1 compute-bound)',
    columns: ['draft', 'α', 'tokens / step', 'speedup, rho = 0', 'speedup, rho = 1', 'best K (rho = 0)'],
    rows: rows.map((r) => [r[0], +r[1].toFixed(3), +r[2].toFixed(2), +r[3].toFixed(2), +r[4].toFixed(2), r[5]]),
  });
  lab.table({
    title: 'Wall clock in this JavaScript toy (forward() recomputes the whole window, so a verify pass costs the same as a decode step)',
    columns: ['method', 'tokens', 'target forward passes', 'draft forward passes', 'ms per token', 'speedup vs plain'],
    rows: [
      ['plain decoding', NPLAIN, plainForwards, 0, +plainMsPerTok.toFixed(1), 1],
      ...sweep.map((s) => [`bigram draft, K = ${s.K}`, N, s.r.steps, 0, +(s.ms / N).toFixed(1), +(plainMsPerTok / (s.ms / N)).toFixed(2)]),
      ['self-draft T = 0.5, K = 4', NSELF, selfRun.steps, selfForwards - selfRun.steps, +(selfMs / NSELF).toFixed(1), +(plainMsPerTok / (selfMs / NSELF)).toFixed(2)],
    ],
  });

  const bestSweep = sweep.reduce((a, b) => (b.r.tokensPerStep > a.r.tokensPerStep ? b : a));
  const text = tok.decode(K4.tokens).replace(/\s+/g, ' ').trim().slice(0, 120);
  lab.done(`On the synthetic pair, your sampler put **${(100 * hist[0]).toFixed(1)}% / ${(100 * hist[1]).toFixed(1)}% / ${(100 * hist[2]).toFixed(1)}%** on the three tokens against a target of 10% / 45% / 45%, while trusting the draft would have given ${(100 * naive[0]).toFixed(0)}% on token 0; the acceptance rate was ${(100 * alpha1).toFixed(1)}% against the analytic ${(100 * m.acceptanceRate(p, q)).toFixed(0)}%.

Against the checkpoint, the bigram draft was accepted **${(100 * K4.alpha).toFixed(1)}%** of the time at K = 4, giving **${K4.tokensPerStep.toFixed(2)} tokens per target pass** (model: ${m.expectedTokensPerStep(K4.alpha, 4).toFixed(2)}); the best K in the sweep was ${bestSweep.K} at ${bestSweep.r.tokensPerStep.toFixed(2)}, while the independent-acceptance model, which assumes every position is accepted at the same rate, would pick K = ${rows[0][5]}. The sharpened self-draft reached α = **${(100 * selfRun.alpha).toFixed(1)}%** and ${selfRun.tokensPerStep.toFixed(2)} tokens per pass, but at c = 1 the model gives it only **${rows[1][3].toFixed(2)}×**, against **${rows[0][3].toFixed(2)}×** for the almost-free bigram draft. With the same bigram draft, α was ${(100 * tempRuns[0].alpha).toFixed(1)}% / ${(100 * tempRuns[1].alpha).toFixed(1)}% / ${(100 * tempRuns[2].alpha).toFixed(1)}% at target temperature 0.5 / 1 / 2: α = Σ min(p, q) peaks where the target's shape is closest to the draft's, and moves whichever way the temperature pulls p away from it. In a compute-bound regime (rho = 1) both drafts model at under 1× (${rows[0][4].toFixed(2)}× and ${rows[1][4].toFixed(2)}×): speculation only pays where verification is nearly free.

Plain decoding cost ${plainMsPerTok.toFixed(1)} ms per token (one target pass each) and the bigram draft at K = 4 cost ${(sweep.find((s) => s.K === 4).ms / N).toFixed(1)} ms per token (${K4.steps} target passes for ${N} tokens), a wall-clock **${(plainMsPerTok / (sweep.find((s) => s.K === 4).ms / N)).toFixed(2)}×**. Sample of the verified output: "${text}"`);
}
