// Bigram model demo: build the count-based and the neural bigram model on the character-level corpus,
// compare their perplexities, watch the neural table converge towards the counts, and sample from both.
import { CORPUS, interleavedSplit } from 'lib/data.js';
import { CharTokenizer } from 'lib/tokenizer.js';
import { rng } from 'lib/util.js';
import { AdamW } from 'lib/optim.js';

const show = (s) => s.replace(/ /g, '␣').replace(/\n/g, '⏎');
const fmtPpl = (x) => (Number.isFinite(x) ? x.toFixed(2) : '∞');

export default async function demo(m, lab) {
  const tok = new CharTokenizer(CORPUS);
  const V = tok.vocabSize;
  const ids = tok.encode(CORPUS);
  const { train, val } = interleavedSplit(ids);
  lab.log(`Corpus: ${ids.length} characters, V = ${V} distinct. Train ${train.length} ids, validation ${val.length} ids.`);

  // 1. The count model at several smoothing strengths.
  const counts = m.countBigrams(train, V);
  lab.check(counts.shape[0] === V && counts.shape[1] === V, `countBigrams must return a [${V}, ${V}] table`);
  const alphas = [0, 0.001, 0.01, 0.1, 1, 10];
  const countRows = alphas.map((alpha) => {
    const p = m.bigramProbs(counts, alpha);
    return { alpha, p, train: m.perplexity(m.negLogLikelihood(p, train)), val: m.perplexity(m.negLogLikelihood(p, val)) };
  });
  const finite = countRows.filter((r) => Number.isFinite(r.val));
  const best = finite.reduce((a, b) => (b.val < a.val ? b : a));
  lab.bar({ title: 'Count model: validation perplexity vs smoothing alpha', labels: finite.map((r) => `α=${r.alpha}`), values: finite.map((r) => +r.val.toFixed(3)) });
  const unseen = val.filter((x, i) => i > 0 && counts.data[val[i - 1] * V + x] === 0).length;
  lab.md(`With **alpha = 0** the validation perplexity is **${fmtPpl(countRows[0].val)}**: ${unseen} of the ${val.length - 1} held-out transitions never occurred in the training text, and a single probability-zero event makes the whole score infinite. Any alpha > 0 fixes that; the best of the values tried is **alpha = ${best.alpha}** (validation perplexity ${best.val.toFixed(2)}, versus ${countRows[0].train.toFixed(2)} on the training text at alpha = 0, the best any bigram can do there).`);
  await lab.tick();

  // 2. The neural model, trained step by step so the loss curve can be plotted against the count model.
  const steps = 600, batchSize = 1024, lr = 0.05, evalEvery = 25;
  const model = m.initNeural(V, rng(1));
  const opt = new AdamW([model.W], { lr });
  const batchRng = rng(2);
  const losses = [];
  const valNll = [];
  const countTrainNll = m.negLogLikelihood(countRows[0].p, train);
  const t0 = performance.now();
  for (let s = 0; s < steps; s++) {
    const { xs, ys } = m.makeBatch(train, batchSize, batchRng);
    losses.push(m.trainStep(model, opt, xs, ys));
    if (s % evalEvery === 0 || s === steps - 1) {
      valNll.push(m.negLogLikelihood(m.neuralProbs(model), val));
      lab.progress(s / steps, `step ${s}: loss ${losses[s].toFixed(3)}`);
      await lab.tick();
    } else {
      valNll.push(null);
    }
  }
  const trainMs = performance.now() - t0;
  lab.check(losses.every(Number.isFinite), 'training produced a non-finite loss');
  lab.plot({
    title: 'Neural bigram: cross-entropy per step (dashed floor: the count model\'s NLL on the whole training text; single minibatches can dip below it)',
    series: [
      { name: 'neural, minibatch loss', values: losses },
      { name: 'neural, validation NLL', values: valNll },
      { name: 'count model (α=0), training NLL', values: losses.map(() => countTrainNll) },
    ],
    xlabel: 'step', ylabel: 'mean −log P(next | prev)',
  });

  // 3. Did gradient descent recover the counts?
  const neural = m.neuralProbs(model);
  const countProbs = countRows[0].p;
  const rowTotals = Array.from({ length: V }, (_, i) => { let s = 0; for (let j = 0; j < V; j++) s += counts.data[i * V + j]; return s; });
  let maxDiff = 0, sumDiff = 0, nFrequent = 0;
  for (let i = 0; i < V; i++) {
    if (rowTotals[i] < 200) continue;
    nFrequent++;
    for (let j = 0; j < V; j++) {
      const d = Math.abs(neural.data[i * V + j] - countProbs.data[i * V + j]);
      sumDiff += d;
      if (d > maxDiff) maxDiff = d;
    }
  }
  const meanDiff = sumDiff / (nFrequent * V);
  const block = [' ', ...'abcdefghijklmnopqrstuvwxyz'].map((ch) => tok.stoi.get(ch)).filter((i) => i !== undefined);
  const labels = block.map((i) => show(tok.itos[i]));
  const sub = (t) => block.map((i) => block.map((j) => +t.data[i * V + j].toFixed(3)));
  lab.heatmap({ title: 'Count model P(next | prev), alpha = 0: rows = previous character, columns = next', rows: sub(countProbs), rowLabels: labels, colLabels: labels, min: 0, max: 0.5 });
  lab.heatmap({ title: `Neural model softmax(W) after ${steps} steps: the same table, learned by gradient descent`, rows: sub(neural), rowLabels: labels, colLabels: labels, min: 0, max: 0.5 });
  await lab.tick();

  // 4. Perplexities side by side.
  const uniform = { shape: [V, V], data: new Float32Array(V * V).fill(1 / V) };
  const neuralTrain = m.perplexity(m.negLogLikelihood(neural, train));
  const neuralVal = m.perplexity(m.negLogLikelihood(neural, val));
  const pplRows = [
    ['uniform (no model)', fmtPpl(m.perplexity(m.negLogLikelihood(uniform, train))), fmtPpl(m.perplexity(m.negLogLikelihood(uniform, val)))],
    ...countRows.map((r) => [`count, alpha = ${r.alpha}`, fmtPpl(r.train), fmtPpl(r.val)]),
    [`neural, ${steps} AdamW steps`, fmtPpl(neuralTrain), fmtPpl(neuralVal)],
  ];
  lab.table({ title: 'Perplexity (exp of mean NLL per character)', columns: ['model', 'train', 'validation'], rows: pplRows });

  // 5. Text from each model, plus the effect of temperature.
  const start = tok.stoi.get('\n');
  const sampleFrom = (p, temperature = 1, n = 200) => tok.decode(m.generate(p, start, n, rng(3), temperature));
  const countText = sampleFrom(best.p);
  const neuralText = sampleFrom(neural);
  lab.md(`**200 characters from the count model (alpha = ${best.alpha}):**\n\n\`\`\`\n${countText}\n\`\`\`\n\n**200 characters from the neural model:**\n\n\`\`\`\n${neuralText}\n\`\`\``);
  lab.table({
    title: 'Neural model, 80 characters at three temperatures (same seed)',
    columns: ['temperature', 'sample'],
    rows: [0.5, 1, 2].map((t) => [t, show(sampleFrom(neural, t, 80))]),
  });

  lab.done(`Two bigram models of ${V}×${V} = ${V * V} numbers each. Counting gives training perplexity **${countRows[0].train.toFixed(2)}** at alpha = 0 and validation perplexity **${fmtPpl(countRows[0].val)}** (${unseen} unseen transitions); with alpha = ${best.alpha} validation perplexity is **${best.val.toFixed(2)}**. The neural model reached training perplexity **${neuralTrain.toFixed(2)}** and validation perplexity **${neuralVal.toFixed(2)}** after ${steps} steps in ${(trainMs / 1000).toFixed(1)} s, and over the ${nFrequent} rows with at least 200 counts its probabilities differ from the count table by **${meanDiff.toFixed(4)}** on average (max ${maxDiff.toFixed(3)}): gradient descent on cross-entropy rediscovered the counts. Both models write text that looks like letters but not like words, which is exactly what one character of context can buy.`);
}
