import { GPT } from 'lib/gpt.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { loadModel } from 'lib/infer.js';
import { CORPUS, getBatch, interleavedSplit } from 'lib/data.js';
import { rng } from 'lib/util.js';

// The goal: three identical 1-layer students, the same initial weights, the same number of steps. One learns
// from the corpus labels, one from the frozen checkpoint's soft targets (YOUR KL loss and mixed objective),
// one from text the teacher wrote (YOUR sequence-level corpus). Then the teacher grades each student's own
// samples with YOUR on-policy reverse KL.
export default async function demo(m, lab) {
  const checkpoint = (await import('lib/checkpoints/tiny-gpt.json', { with: { type: 'json' } })).default;
  const tokenizerJson = (await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default;
  const teacher = GPT.fromJSON(checkpoint);
  const teacherInfer = loadModel(checkpoint);
  const tok = BPETokenizer.fromJSON(tokenizerJson);
  const V = teacher.config.vocabSize;
  const blockSize = 32;
  const steps = 150;
  const evalEvery = 25;
  const probe = m.makeStudent({ vocabSize: V, blockSize: teacher.config.blockSize, seed: 3 });
  lab.log(`teacher: ${teacher.config.nLayer} layers × ${teacher.config.nEmbd} channels, ${teacher.numParams().toLocaleString()} parameters (frozen)`);
  lab.log(`student: ${probe.config.nLayer} layer × ${probe.config.nEmbd} channels, ${probe.numParams().toLocaleString()} parameters (${(teacher.numParams() / probe.numParams()).toFixed(1)}× smaller)`);

  // 1. Dark knowledge (step 1): the teacher's next-token distribution at one position, softened.
  const ids = tok.encode(CORPUS);
  const { train, val } = interleavedSplit(ids);
  const contextText = 'The teacher is green beside the';
  const context = tok.encode(contextText);
  const lastLogits = m.teacherLogits(teacher, [context]);
  const row = { shape: [V], data: lastLogits.data.subarray((context.length - 1) * V, context.length * V) };
  const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => row.data[b] - row.data[a]).slice(0, 10);
  const temps = [0.5, 1, 2, 4];
  const softened = temps.map((T) => { const p = m.softTargets(row, T); return order.map((i) => p.data[i]); });
  lab.heatmap({
    title: `Teacher after "${contextText}": softTargets at four temperatures (its top 10 tokens)`,
    rows: softened, rowLabels: temps.map((T) => `T = ${T}`), colLabels: order.map((i) => JSON.stringify(tok.decode([i]))), min: 0, max: 1,
  });
  const topToken = JSON.stringify(tok.decode([order[0]]));
  const secondToken = JSON.stringify(tok.decode([order[1]]));
  const offTop = (T) => 1 - softened[temps.indexOf(T)][0]; // mass on every token except the teacher's favourite
  const second = (T) => softened[temps.indexOf(T)][1];

  // 2. Offline teacher logits for the training windows and the held-out windows.
  lab.log('caching the teacher\'s logits for 96 training windows and 16 held-out windows…');
  const pool = m.buildPool(teacher, train, { windows: 96, blockSize, next: rng(1) });
  await lab.tick();
  const valPool = m.buildPool(teacher, val, { windows: 16, blockSize, next: rng(2) });
  const teacherVal = m.evaluate(teacher, valPool);
  lab.log(`teacher on held-out windows: ${teacherVal.loss.toFixed(3)} nats/token`);
  await lab.tick();

  // 3. Sequence-level data (step 4): the teacher writes a corpus of about the same size from 64 short prompts.
  const prompts = pool.x.slice(0, 64).map((r) => tok.decode(r.slice(0, 4)));
  const seq = m.sequenceLevelCorpus(teacherInfer, tok, prompts, { maxNewTokens: 56, temperature: 1, next: rng(3) });
  const seqBatch = getBatch(seq.ids, { blockSize, batchSize: 96, next: rng(4) });
  lab.log(`teacher-written corpus: ${seq.ids.length} tokens from ${prompts.length} prompts, e.g. ${JSON.stringify(seq.texts[0].slice(0, 70))}`);
  await lab.tick();

  // 4. Three students, same init, same steps (steps 3 and 6).
  const arms = [
    { name: 'from scratch (labels)', pool: { x: pool.x, y: pool.y, teacher: null }, alpha: 0, T: 1 },
    { name: 'logit KD (alpha 0.5, T 2)', pool, alpha: 0.5, T: 2 },
    { name: 'sequence-level KD (teacher text)', pool: { x: seqBatch.x, y: seqBatch.y, teacher: null }, alpha: 0, T: 1 },
  ];
  const xs = [];
  for (let s = 0; s <= steps; s += evalEvery) xs.push(s);
  for (const [k, arm] of arms.entries()) {
    const student = m.makeStudent({ vocabSize: V, blockSize: teacher.config.blockSize, seed: 3 });
    arm.student = student;
    arm.curve = [m.evaluate(student, valPool).loss];
    const t0 = performance.now();
    arm.losses = await m.distillTrain(student, arm.pool, {
      steps, batchSize: 4, lr: 3e-3, alpha: arm.alpha, T: arm.T, next: rng(5),
      onStep: async (step) => {
        if ((step + 1) % evalEvery === 0) arm.curve.push(m.evaluate(student, valPool).loss);
        if (step % 5 === 0) { lab.progress((k * steps + step + 1) / (arms.length * steps), `${arm.name}: step ${step}`); await lab.tick(); }
      },
    });
    arm.seconds = (performance.now() - t0) / 1000;
    arm.final = m.evaluate(student, valPool);
    lab.log(`${arm.name}: held-out CE ${arm.final.loss.toFixed(3)}, top-1 agreement ${(100 * arm.final.agreement).toFixed(1)}% (${arm.seconds.toFixed(1)} s)`);
  }
  lab.plot({
    title: 'Held-out cross-entropy on real text (nats/token) at equal steps',
    x: xs,
    series: [...arms.map((a) => ({ name: a.name, values: a.curve })), { name: 'teacher (frozen)', values: xs.map(() => teacherVal.loss) }],
    xlabel: 'training step (batch of 4 × 32 tokens)', ylabel: 'cross-entropy',
  });
  lab.bar({
    title: 'Top-1 agreement with the teacher on held-out positions (%)',
    labels: arms.map((a) => a.name),
    values: arms.map((a) => +(100 * a.final.agreement).toFixed(1)),
  });

  // 5. On-policy view (step 5): the teacher grades what each student actually writes.
  const promptIds = valPool.x.slice(0, 8).map((r) => r.slice(0, 4));
  for (const arm of arms) arm.onPolicy = m.onPolicyLoss(arm.student, teacher, promptIds, { maxNewTokens: 24, next: rng(11) }).loss.item();
  await lab.tick();

  // 6. What each model writes, greedily, from two prompts.
  const samplePrompts = ['The teacher', 'Where did'];
  const greedy = (model, text) => tok.decode(model.generate(tok.encode(text), { maxNewTokens: 14, temperature: 0, next: rng(0) }).slice(tok.encode(text).length)).replace(/\n/g, ' / ');
  lab.table({
    title: 'Greedy continuations (after 150 steps) and the teacher\'s reverse KL on each student\'s own samples',
    columns: ['model', `"${samplePrompts[0]}…"`, `"${samplePrompts[1]}…"`, 'held-out CE', 'on-policy reverse KL'],
    rows: [
      ['teacher', ...samplePrompts.map((p) => greedy(teacher, p)), teacherVal.loss.toFixed(3), '0 (by definition)'],
      ...arms.map((a) => [a.name, ...samplePrompts.map((p) => greedy(a.student, p)), a.final.loss.toFixed(3), a.onPolicy.toFixed(3)]),
    ],
  });

  const [scratch, kd, sl] = arms;
  const gap = scratch.final.loss - kd.final.loss;
  // Steps the scratch student needed to reach the distilled student's step-50 held-out loss, if it ever did.
  const target = kd.curve[2];
  const reachIdx = scratch.curve.findIndex((v) => v <= target);
  const reach = reachIdx < 0 ? `never reached it in ${steps} steps` : `needed ${xs[reachIdx]} steps to reach it`;
  lab.done(`Same student (${probe.numParams().toLocaleString()} parameters, ${(teacher.numParams() / probe.numParams()).toFixed(1)}× smaller than the teacher), same init, **${steps} steps** each (scratch and logit KD on the same 96 windows of real text, sequence-level on 96 windows of the ${seq.ids.length}-token corpus the teacher wrote). Held-out cross-entropy: from scratch **${scratch.final.loss.toFixed(3)}**, logit KD **${kd.final.loss.toFixed(3)}** (a gap of **${gap.toFixed(3)} nats/token** in favour of distillation), sequence-level KD ${sl.final.loss.toFixed(3)}; the teacher sits at ${teacherVal.loss.toFixed(3)}. The distilled student was at ${target.toFixed(3)} after 50 steps; the scratch student ${reach}. Top-1 agreement with the teacher: ${(100 * scratch.final.agreement).toFixed(1)}% → **${(100 * kd.final.agreement).toFixed(1)}%** with KD (${(100 * sl.final.agreement).toFixed(1)}% sequence-level). The teacher's on-policy reverse KL on each student's own samples: ${scratch.onPolicy.toFixed(2)} / ${kd.onPolicy.toFixed(2)} / ${sl.onPolicy.toFixed(2)} nats. After "${contextText}" the teacher gives ${topToken} ${(100 * (1 - offTop(1))).toFixed(1)}% at T = 1; softened to T = 4, everything else carries ${(100 * offTop(4)).toFixed(0)}% of the mass (runner-up ${secondToken}: ${(100 * second(1)).toFixed(1)}% → ${(100 * second(4)).toFixed(1)}%). That ranking of the alternatives is the information per token a one-hot label never carries.`);
}
