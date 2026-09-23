import { crossEntropy } from 'lib/tensor.js';
import { GPT } from 'lib/gpt.js';
import { AdamW, clipGradNorm } from 'lib/optim.js';
import { CORPUS, getBatch } from 'lib/data.js';
import { CharTokenizer } from 'lib/tokenizer.js';
import { rng } from 'lib/util.js';

// The goal: a 4-expert, top-2 MoE GPT against a dense GPT with the same ACTIVE feed-forward width,
// trained on the same batches, plus a short run without the balancing loss to show collapse.
export default async function demo(m, lab) {
  const tokenizer = new CharTokenizer(CORPUS);
  const ids = tokenizer.encode(CORPUS);
  const C = 32, E = 4, K = 2, H = 64;          // dense MLP hidden is 4C = 128 = K · H
  const cfg = { vocabSize: tokenizer.vocabSize, blockSize: 32, nLayer: 1, nHead: 4, nEmbd: C, seed: 3 };
  const moeOpts = { nExperts: E, k: K, hidden: H, capacityFactor: 1.25 };
  const batchSize = 6, steps = 150, ablationSteps = 60, auxCoef = 0.01;
  const N = batchSize * cfg.blockSize;

  // Accounting first: is your countParams consistent with the layer you built?
  const acct = m.countParams({ nEmbd: C, hidden: H, nExperts: E, k: K });
  const probe = m.moeGPT(cfg, moeOpts, { next: rng(9) });
  const layerParams = probe.blocks[0].moe.parameters().reduce((s, p) => s + p.size, 0);
  lab.check(acct.total === layerParams, `countParams total ${acct.total} must equal the MoE layer's ${layerParams} parameters`);
  const denseFFN = 2 * C * 4 * C + 4 * C + C;
  lab.log(`corpus ${CORPUS.length} chars, vocab ${tokenizer.vocabSize}; ${N} tokens per step, capacity ${m.expertCapacity(N, E, K, moeOpts.capacityFactor)} slots per expert`);
  lab.log(`feed-forward parameters: dense ${denseFFN}; MoE total ${acct.total}, active ${acct.active} (${(acct.total / acct.active).toFixed(2)}× more stored per FLOP spent)`);

  async function train(label, model, { moe, aux, nSteps, done, of }) {
    const opt = new AdamW(model.parameters(), { lr: 5e-3, weightDecay: 0 });
    const next = rng(5);                         // same batches for every run
    const losses = [], auxes = [], util = [], drops = [];
    for (let s = 0; s < nSteps; s++) {
      const { x, y } = getBatch(ids, { blockSize: cfg.blockSize, batchSize, next });
      let loss = crossEntropy(model.forward(x), y);
      losses.push(loss.item());
      if (moe) {
        const a = model.auxLoss();
        auxes.push(a.item());
        const st = model.blocks[0].moe.lastStats;
        util.push(st.counts.map((c) => c / st.assignments));
        drops.push(st.dropped / st.assignments);
        if (aux) loss = loss.add(a.scale(auxCoef));
      }
      opt.zeroGrad();
      loss.backward();
      clipGradNorm(model.parameters(), 1.0);
      opt.step();
      if (s % 5 === 0) { lab.progress((done + s) / of, `${label}: step ${s} loss ${losses[s].toFixed(3)}`); await lab.tick(); }
    }
    return { losses, auxes, util, drops };
  }

  const total = 2 * steps + ablationSteps;
  const t0 = performance.now();
  const dense = await train('dense', new GPT(cfg), { moe: false, nSteps: steps, done: 0, of: total });
  const moe = await train('MoE + aux', m.moeGPT(cfg, moeOpts, { next: rng(9) }), { moe: true, aux: true, nSteps: steps, done: steps, of: total });
  const noAux = await train('MoE, no aux', m.moeGPT(cfg, moeOpts, { next: rng(9) }), { moe: true, aux: false, nSteps: ablationSteps, done: 2 * steps, of: total });
  const seconds = (performance.now() - t0) / 1000;

  const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
  const ema = (v, b = 0.9) => { let s = v[0]; return v.map((x) => (s = b * s + (1 - b) * x)); };
  lab.plot({
    title: 'Training loss (EMA 0.9): dense MLP vs 4-expert top-2 MoE at equal active width',
    series: [{ name: `dense (${denseFFN} FFN params)`, values: ema(dense.losses) }, { name: `MoE (${acct.total} total, ${acct.active} active)`, values: ema(moe.losses) }],
    xlabel: 'step', ylabel: 'cross-entropy (nats/char)',
  });

  const bucketed = (util, width) => {
    const rows = [];
    for (let e = 0; e < E; e++) {
      const r = [];
      for (let s = 0; s < util.length; s += width) r.push(mean(util.slice(s, s + width).map((u) => u[e])));
      rows.push(r);
    }
    return rows;
  };
  const colLabels = (n, width) => Array.from({ length: Math.ceil(n / width) }, (_, i) => String(i * width));
  lab.heatmap({
    title: 'With the balancing loss: share of routed assignments per expert (0.25 = balanced, 0.5 = the most top-2 allows), every 10 steps',
    rows: bucketed(moe.util, 10), rowLabels: Array.from({ length: E }, (_, e) => `expert ${e}`), colLabels: colLabels(steps, 10), min: 0, max: 0.5,
  });
  lab.heatmap({
    title: 'Without it: the same model and batches, every 5 steps',
    rows: bucketed(noAux.util, 5), rowLabels: Array.from({ length: E }, (_, e) => `expert ${e}`), colLabels: colLabels(ablationSteps, 5), min: 0, max: 0.5,
  });
  lab.plot({
    title: 'Load-balancing loss E·Σ f·P (1 = balanced; 2 = top-2 collapsed onto two experts)',
    series: [{ name: 'with aux loss (coef 0.01)', values: moe.auxes.slice(0, ablationSteps) }, { name: 'without', values: noAux.auxes }],
    xlabel: 'step', ylabel: 'aux',
  });
  lab.bar({
    title: 'Feed-forward parameters: stored vs used per token',
    labels: ['dense MLP', 'MoE total (stored)', 'MoE active (per token)'],
    values: [denseFFN, acct.total, acct.active],
  });

  const tail = 15;
  const denseEnd = mean(dense.losses.slice(-tail)), moeEnd = mean(moe.losses.slice(-tail));
  const auxStart = moe.auxes[0], auxEnd = mean(moe.auxes.slice(-tail));
  const noAuxEnd = mean(noAux.auxes.slice(-tail)), withAuxSame = mean(moe.auxes.slice(ablationSteps - tail, ablationSteps));
  const dropEarly = mean(moe.drops.slice(0, tail)), dropLate = mean(moe.drops.slice(-tail)), dropNoAux = mean(noAux.drops.slice(-tail));
  const lastShare = (u) => { const last = u.slice(-tail); return Array.from({ length: E }, (_, e) => mean(last.map((r) => r[e]))); };
  const shareAux = lastShare(moe.util), shareNo = lastShare(noAux.util);
  lab.table({
    title: `Summary (means over the last ${tail} steps of each run)`,
    columns: ['run', 'steps', 'LM loss', 'aux loss', 'drop rate', 'busiest expert share', 'idlest expert share'],
    rows: [
      ['dense', steps, +denseEnd.toFixed(3), '-', '-', '-', '-'],
      ['MoE + aux', steps, +moeEnd.toFixed(3), +auxEnd.toFixed(3), `${(100 * dropLate).toFixed(1)}%`, +Math.max(...shareAux).toFixed(3), +Math.min(...shareAux).toFixed(3)],
      ['MoE, no aux', ablationSteps, +mean(noAux.losses.slice(-tail)).toFixed(3), +noAuxEnd.toFixed(3), `${(100 * dropNoAux).toFixed(1)}%`, +Math.max(...shareNo).toFixed(3), +Math.min(...shareNo).toFixed(3)],
    ],
  });

  const diff = moeEnd - denseEnd;
  const verdict = Math.abs(diff) < 0.05
    ? `within ${Math.abs(diff).toFixed(3)} nats of each other: at this scale the extra stored parameters barely matter`
    : diff < 0 ? `the MoE ahead by ${(-diff).toFixed(3)} nats` : `the dense block ahead by ${diff.toFixed(3)} nats, which is common at this tiny scale and short run`;
  lab.done(`You trained three models in ${seconds.toFixed(1)} s on ${N} tokens per step. After ${steps} steps the dense block reached loss **${denseEnd.toFixed(3)}** and your 4-expert top-2 MoE **${moeEnd.toFixed(3)}** (${verdict}), while storing **${acct.total}** feed-forward parameters against the dense ${denseFFN} and using **${acct.active}** per token (${acct.flopsPerToken} FLOPs). With the balancing loss, aux went from ${auxStart.toFixed(3)} to **${auxEnd.toFixed(3)}** (1 is perfect), the busiest expert took ${(100 * Math.max(...shareAux)).toFixed(1)}% of assignments, and the capacity factor of 1.25 dropped ${(100 * dropEarly).toFixed(1)}% of assignments early and **${(100 * dropLate).toFixed(1)}%** late. Without it, aux reached **${noAuxEnd.toFixed(3)}** after ${ablationSteps} steps (versus ${withAuxSame.toFixed(3)} with it at the same point), the idlest expert got ${(100 * Math.min(...shareNo)).toFixed(1)}% of assignments, and ${(100 * dropNoAux).toFixed(1)}% were dropped.`);
}
