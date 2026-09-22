import { GPT } from 'lib/gpt.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { PREFERENCES } from 'lib/data.js';
import { noGrad } from 'lib/tensor.js';
import { rng } from 'lib/util.js';

// The goal: on the pre-trained checkpoint, (1) train YOUR reward head with YOUR Bradley–Terry loss and
// read its reward accuracy on pairs it never saw, then (2) run YOUR DPO loop against a frozen reference
// and watch the implicit reward margin rise from exactly 0.
export default async function demo(m, lab) {
  const checkpoint = (await import('lib/checkpoints/tiny-gpt.json', { with: { type: 'json' } })).default;
  const tokenizerJson = (await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default;
  const tokenizer = BPETokenizer.fromJSON(tokenizerJson);
  const policy = GPT.fromJSON(checkpoint);
  const { blockSize, nEmbd } = policy.config;
  const beta = 0.1;
  const mean = (arr) => arr.reduce((a, b) => a + b, 0) / (arr.length || 1);

  // 1. The data: every preference pair that fits the context window, split into train and held-out.
  const all = PREFERENCES.map((p) => ({ prompt: p.prompt, chosenText: p.chosen, rejectedText: p.rejected, ...m.buildPair(tokenizer, p) }))
    .filter((p) => p.chosen.x.length < blockSize && p.rejected.x.length < blockSize);
  const heldOut = all.filter((_, i) => i % 4 === 3);
  const train = all.filter((_, i) => i % 4 !== 3);
  const tokens = (e) => e.x.length + 1;
  lab.log(`${all.length} of ${PREFERENCES.length} pairs fit blockSize ${blockSize}; ${train.length} for training, ${heldOut.length} held out. Mean length: chosen ${mean(all.map((p) => tokens(p.chosen))).toFixed(1)} tokens, rejected ${mean(all.map((p) => tokens(p.rejected))).toFixed(1)}`);

  // 2. A reward model: a linear head on the frozen checkpoint's final hidden state (steps 1 and 4).
  const featuresOf = (pairs) => noGrad(() => pairs.map((p) => {
    const one = (ex) => {
      const batch = m.padBatch([ex]);
      return Float32Array.from(m.lastTokenHidden(m.hiddenStates(policy, batch.x), batch.lengths).data);
    };
    return { chosen: one(p.chosen), rejected: one(p.rejected) };
  }));
  lab.progress(0.02, 'reward head: extracting features');
  await lab.tick();
  const trainFeatures = featuresOf(train);
  const heldOutFeatures = featuresOf(heldOut);
  const head = new m.RewardHead(nEmbd, { next: rng(3) });
  const headAccuracy = (features) => noGrad(() => {
    const C = nEmbd, N = features.length;
    const stack = (key) => { const data = new Float32Array(N * C); for (let i = 0; i < N; i++) data.set(features[i][key], i * C); return m.asTensor(data).reshape([N, C]); };
    return m.rewardAccuracy(head.forward(stack('chosen')), head.forward(stack('rejected')));
  });
  const rmHistory = m.trainRewardHead(head, trainFeatures, { steps: 300, lr: 1e-2, weightDecay: 0.01 });
  const rmAfter = { train: headAccuracy(trainFeatures), heldOut: headAccuracy(heldOutFeatures) };
  lab.log(`reward head (Linear(${nEmbd}, 1) on the frozen body, 300 full-batch steps): Bradley–Terry loss ${rmHistory[0].loss.toFixed(3)} -> ${rmHistory[rmHistory.length - 1].loss.toFixed(3)}; reward accuracy ${(100 * rmAfter.train).toFixed(0)}% on the ${train.length} training pairs, ${(100 * rmAfter.heldOut).toFixed(0)}% on the ${heldOut.length} held-out pairs (one pair is ${(100 / heldOut.length).toFixed(0)} points, so read that number as a rough estimate)`);
  lab.plot({
    title: 'Reward head: Bradley–Terry loss and training-pair accuracy',
    x: rmHistory.map((_, i) => i),
    series: [{ name: 'loss', values: rmHistory.map((r) => r.loss) }, { name: 'accuracy (train pairs)', values: rmHistory.map((r) => r.accuracy) }],
    xlabel: 'step', ylabel: 'loss / accuracy',
  });
  lab.progress(0.08, 'reward head trained');
  await lab.tick();

  // 3. DPO: a frozen reference (a copy of the policy), its log-probs computed once, margins before training.
  const reference = m.cloneModel(policy);
  const refTrain = m.referenceLogProbs(reference, train);
  const refHeld = m.referenceLogProbs(reference, heldOut);
  const before = m.evaluatePairs(policy, train, refTrain, { beta });
  const beforeHeld = m.evaluatePairs(policy, heldOut, refHeld, { beta });
  lab.log(`before DPO: mean implicit reward margin ${before.meanMargin.toFixed(4)} on training pairs (policy = reference, so every margin is exactly 0), reward accuracy ${(100 * before.accuracy).toFixed(0)}%. Reference log p(chosen) averages ${mean(refTrain.chosen).toFixed(1)} nats, log p(rejected) ${mean(refTrain.rejected).toFixed(1)}`);
  lab.progress(0.12, 'reference log-probs done');
  await lab.tick();

  // 4. The DPO loop (steps 2, 3 and 5).
  const config = { steps: 150, beta, lr: 1e-4, batchSize: 2, maxGradNorm: 1.0 };
  lab.log(`DPO: ${config.steps} steps, ${config.batchSize} pairs per step, β = ${beta}, AdamW lr ${config.lr}, clip ${config.maxGradNorm}`);
  const t0 = performance.now();
  const history = await m.trainDPO(policy, train, refTrain, {
    ...config,
    next: rng(11),
    onStep: async (step, r) => {
      if (step % 3 === 0) { lab.progress(0.12 + 0.8 * ((step + 1) / config.steps), `DPO step ${step}: loss ${r.loss.toFixed(3)}, margin ${r.margin.toFixed(2)}`); await lab.tick(); }
      if (step % 30 === 0) lab.log(`step ${step}: loss ${r.loss.toFixed(3)}, batch margin ${r.margin.toFixed(3)}, batch accuracy ${r.accuracy.toFixed(2)}, grad norm ${r.gradNorm.toFixed(2)}`);
    },
  });
  const seconds = (performance.now() - t0) / 1000;
  const smooth = (arr, w = 10) => arr.map((_, i) => mean(arr.slice(Math.max(0, i - w + 1), i + 1)));
  lab.plot({
    title: 'DPO loss (−log σ of the batch margin) and 10-step mean',
    x: history.map((_, i) => i),
    series: [{ name: 'loss', values: history.map((r) => r.loss) }, { name: '10-step mean', values: smooth(history.map((r) => r.loss)) }],
    xlabel: 'step', ylabel: 'loss (nats)',
  });
  lab.plot({
    title: 'Implicit reward margin β[(π_c − ref_c) − (π_r − ref_r)] on each step\'s batch',
    x: history.map((_, i) => i),
    series: [{ name: 'batch margin', values: history.map((r) => r.margin) }, { name: '10-step mean', values: smooth(history.map((r) => r.margin)) }],
    xlabel: 'step', ylabel: 'margin',
  });

  // 5. After: the same pairs, the same reference.
  const after = m.evaluatePairs(policy, train, refTrain, { beta });
  const afterHeld = m.evaluatePairs(policy, heldOut, refHeld, { beta });
  lab.bar({
    title: 'Reward accuracy (fraction of pairs ranked chosen > rejected)',
    labels: ['RM head, train', 'RM head, held-out', 'DPO before, train', 'DPO after, train', 'DPO after, held-out'],
    values: [rmAfter.train, rmAfter.heldOut, before.accuracy, after.accuracy, afterHeld.accuracy],
  });
  // `all` spread the PREFERENCES fields in, so each pair still carries its prompt and the two texts.
  const row = (pairs, ev, i, tag) => [pairs[i].prompt, pairs[i].chosenText, pairs[i].rejectedText, ev.chosenRatio[i].toFixed(2), ev.rejectedRatio[i].toFixed(2), ev.margins[i].toFixed(3), tag];
  const rows = [];
  for (const i of [0, 3, 7, 12, 20]) if (i < train.length) rows.push(row(train, after, i, 'train'));
  for (const i of [0, 4]) if (i < heldOut.length) rows.push(row(heldOut, afterHeld, i, 'held-out'));
  lab.table({
    title: 'After DPO: log-ratios log π − log ref per response (before training every entry was 0.00) and the implicit reward margin β·(difference)',
    columns: ['prompt', 'chosen', 'rejected', 'log-ratio chosen', 'log-ratio rejected', 'margin', 'split'],
    rows,
  });
  const meanChosen = mean(after.chosenRatio), meanRejected = mean(after.rejectedRatio);
  const upChosen = after.chosenRatio.filter((r) => r > 0).length;
  lab.log(`after DPO: mean log-ratio of chosen ${meanChosen >= 0 ? '+' : ''}${meanChosen.toFixed(2)} nats (${upChosen} of ${train.length} went up), of rejected ${meanRejected >= 0 ? '+' : ''}${meanRejected.toFixed(2)} nats; mean margin ${after.meanMargin.toFixed(3)} on training pairs, ${afterHeld.meanMargin.toFixed(3)} on held-out pairs`);

  lab.done(`Your Bradley–Terry loss trained a \`Linear(${nEmbd}, 1)\` reward head on the frozen checkpoint to **${(100 * rmAfter.train).toFixed(0)}%** reward accuracy on the ${train.length} training pairs and **${(100 * rmAfter.heldOut).toFixed(0)}%** on ${heldOut.length} held-out pairs (chance is 50%). Your DPO loop then ran **${config.steps} steps** (β = ${beta}, ${config.batchSize} pairs per step) in ${seconds.toFixed(1)} s: the loss fell from **${history[0].loss.toFixed(3)}** (log 2, because the policy started as the reference) to a 10-step mean of **${mean(history.slice(-10).map((r) => r.loss)).toFixed(3)}**, and the mean implicit reward margin on the training pairs rose from **${before.meanMargin.toFixed(3)}** to **${after.meanMargin.toFixed(3)}** (reward accuracy ${(100 * before.accuracy).toFixed(0)}% → **${(100 * after.accuracy).toFixed(0)}%**; held-out margin ${afterHeld.meanMargin.toFixed(3)}, accuracy **${(100 * afterHeld.accuracy).toFixed(0)}%**). The margin was built by moving chosen responses **${meanChosen >= 0 ? '+' : ''}${meanChosen.toFixed(2)} nats** and rejected responses **${meanRejected >= 0 ? '+' : ''}${meanRejected.toFixed(2)} nats** relative to the reference: DPO only constrains the difference, which is why production runs watch both log-ratios, not just the margin.`);
}
