// Goal demo for module 13 — run a full eval with your harness and read the error bars.
import { rng, hash32, meanArray } from 'lib/util.js';

export default async function demo(m, lab) {
  // ---- the "model under test": a scripted stand-in with a per-task success rate ----
  const tasks = m.makeTasks(40, 13);
  const answerOf = new Map(tasks.map((t) => [t.question, t.answer]));
  // Each task gets a fixed latent difficulty in [0.15, 0.75] derived from its text, so the
  // model is genuinely better at some tasks than others — as a real model is.
  const rateOf = (q) => 0.15 + 0.6 * ((hash32(q) % 1000) / 1000);
  function model(question, next) {
    const ref = answerOf.get(question);
    const correct = next() < rateOf(question);
    const shown = correct ? ref : String(Number(ref) + (next() < 0.5 ? 1 : -1) * (1 + Math.floor(next() * 9)));
    if (next() < 0.6) {
      // Verbose chain of thought: the final number is there, but not on its own.
      return `Let me work through this step by step. I read the quantities in the question, combine them in the order the problem describes, and check the result once more before answering. That gives ${shown}. #### ${shown}`;
    }
    return `${shown}`;
  }

  const strict = (a, ref) => m.exactMatch(a, ref);
  const extract = (a, ref) => m.exactMatch(m.extractFinal(a), ref);
  const judge = m.judgeGrader(m.scriptedJudge);

  // ---- run the eval in chunks so the page stays responsive ----
  lab.log(`Evaluating ${tasks.length} tasks, 10 samples each = ${tasks.length * 10} model calls.`);
  const next = rng(7);
  const CHUNK = 8, N = 10, KS = [1, 5, 10];
  const results = [];
  for (let i = 0; i < tasks.length; i += CHUNK) {
    const part = m.runEval({ tasks: tasks.slice(i, i + CHUNK), model, grader: extract, n: N, ks: KS, next, B: 50 });
    results.push(...part.results);
    lab.progress((i + CHUNK) / tasks.length, `graded ${Math.min(i + CHUNK, tasks.length)}/${tasks.length} tasks`);
    await lab.tick();
  }

  // ---- metrics with bootstrap confidence intervals, from your estimators ----
  const boot = rng(101);
  const metrics = KS.map((k) => {
    const per = results.map((r) => m.passAtK(r.n, r.c, k));
    const ci = m.bootstrapCI(per, { B: 1000, next: boot });
    return { k, value: ci.mean, lo: ci.lo, hi: ci.hi, se: ci.se, passPowK: meanArray(results.map((r) => m.passPowK(r.n, r.c, k))) };
  });
  lab.md('**Report (your `formatReport`):**\n\n' + m.formatReport({ metrics }));
  lab.table({
    title: `pass@k over ${tasks.length} tasks, n = ${N} samples per task`,
    columns: ['k', 'pass@k', '95% CI low', '95% CI high', 'CI width', 'pass^k'],
    rows: metrics.map((x) => [x.k, +x.value.toFixed(3), +x.lo.toFixed(3), +x.hi.toFixed(3), +(x.hi - x.lo).toFixed(3), +x.passPowK.toFixed(4)]),
  });

  // ---- the grader is half the eval: three graders, the same sampled answers ----
  const records = [];
  for (const r of results) for (const a of r.answers) records.push({ question: r.question, answer: a, reference: r.reference });
  const graders = [['exact match (raw)', strict], ['exact match after extract', extract], ['LLM judge (scripted)', judge]];
  const accuracy = [];
  for (const [, g] of graders) {
    let s = 0;
    for (const rec of records) s += g(rec.answer, rec.reference, rec.question) >= 0.5 ? 1 : 0;
    accuracy.push(s / records.length);
    await lab.tick();
  }
  lab.bar({
    title: `Same ${records.length} responses, three graders: accuracy`,
    labels: graders.map((g) => g[0]),
    values: accuracy.map((a) => +(100 * a).toFixed(1)),
  });
  const agreeJudge = m.agreementRate(records, extract, judge);
  const agreeStrict = m.agreementRate(records, extract, strict);
  lab.bar({
    title: 'Agreement with "exact match after extract" (%)',
    labels: ['raw exact match', 'LLM judge'],
    values: [+(100 * agreeStrict).toFixed(1), +(100 * agreeJudge).toFixed(1)],
  });

  // ---- error bars shrink like 1/sqrt(tasks) ----
  const per1 = results.map((r) => m.passAtK(r.n, r.c, 1));
  const sizes = [5, 10, 20, 40];
  const widths = [];
  for (const nT of sizes) {
    const ci = m.bootstrapCI(per1.slice(0, nT), { B: 800, next: rng(3) });
    widths.push(ci.hi - ci.lo);
    await lab.tick();
  }
  const ref = sizes.map((nT) => widths[0] * Math.sqrt(sizes[0] / nT));
  lab.plot({
    title: '95% interval width for pass@1 vs number of tasks',
    x: sizes,
    series: [{ name: 'bootstrap width', values: widths.map((w) => +w.toFixed(4)) }, { name: 'width at 5 tasks / sqrt(tasks/5)', values: ref.map((w) => +w.toFixed(4)) }],
    xlabel: 'tasks evaluated',
    ylabel: 'CI width (absolute)',
  });

  // ---- judge position bias ----
  const pairs = results.slice(0, 20).map((r) => ({ question: r.question, a: r.answers[0], b: r.answers[1], reference: r.reference }));
  const bias = m.positionBias(m.scriptedPairwiseJudge, pairs);
  lab.bar({
    title: 'Pairwise judge, each pair scored in both orders (%)',
    labels: ['verdicts consistent under swap', 'first slot wins', 'fair value for first slot'],
    values: [+(100 * bias.consistency).toFixed(1), +(100 * bias.firstWinRate).toFixed(1), 50],
  });

  // ---- contamination ----
  const leaked = [tasks[1], tasks[4], tasks[9], tasks[17], tasks[31]];
  const corpus = [
    'Assorted web text about shops, boxes and friends sharing things, of no particular value.',
    ...leaked.map((t) => t.question),
    'More ordinary text that happens to mention apples and marbles and coins.',
  ].join(' ');
  const contam = m.contamination(tasks, corpus, 13);
  const clean = results.filter((r) => !contam.flagged.includes(r.id));
  const cleanPass1 = meanArray(clean.map((r) => m.passAtK(r.n, r.c, 1)));
  lab.table({
    title: '13-gram overlap between eval prompts and the "training corpus"',
    columns: ['subset', 'tasks', 'pass@1'],
    rows: [
      ['all', results.length, +metrics[0].value.toFixed(3)],
      ['flagged as contaminated', contam.flagged.length, +meanArray(results.filter((r) => contam.flagged.includes(r.id)).map((r) => m.passAtK(r.n, r.c, 1))).toFixed(3)],
      ['clean', clean.length, +cleanPass1.toFixed(3)],
    ],
  });

  const p1 = metrics[0], p10 = metrics[2];
  lab.done(`Your harness evaluated **${tasks.length} tasks x ${N} samples = ${records.length} responses**.

**pass@1 = ${p1.value.toFixed(3)}**, 95% CI [${p1.lo.toFixed(3)}, ${p1.hi.toFixed(3)}] — a width of ${(p1.hi - p1.lo).toFixed(3)}, so any "improvement" smaller than about ${((p1.hi - p1.lo) * 100).toFixed(0)} points on this eval is noise. **pass@10 = ${p10.value.toFixed(3)}** while **pass^10 = ${p10.passPowK.toFixed(4)}**: the same model looks near-perfect if you may retry and nearly useless if every attempt must succeed.

The grader moved the score more than any of that: raw exact match scored **${(100 * accuracy[0]).toFixed(1)}%**, exact match after extracting the final answer **${(100 * accuracy[1]).toFixed(1)}%**, and the scripted judge **${(100 * accuracy[2]).toFixed(1)}%** — a spread of **${(100 * (Math.max(...accuracy) - Math.min(...accuracy))).toFixed(1)} points** on identical responses. The judge agreed with extract-then-match on only **${(100 * agreeJudge).toFixed(1)}%** of responses, and the pairwise judge was consistent under swapping on **${(100 * bias.consistency).toFixed(0)}%** of pairs while picking the first slot **${(100 * bias.firstWinRate).toFixed(0)}%** of the time (50% would be fair).

Finally, **${contam.flagged.length} of ${tasks.length} prompts** shared a 13-gram with the corpus; dropping them changes pass@1 from ${p1.value.toFixed(3)} to **${cleanPass1.toFixed(3)}**. Widening the eval from 5 to 40 tasks shrank the pass@1 interval from ${widths[0].toFixed(3)} to ${widths[3].toFixed(3)}, a factor of **${(widths[0] / widths[3]).toFixed(2)}x** against the ${Math.sqrt(8).toFixed(2)}x that 1/sqrt(tasks) predicts.`);
}
