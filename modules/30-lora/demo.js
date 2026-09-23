import { GPT } from 'lib/gpt.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { INSTRUCTIONS } from 'lib/data.js';
import { noGrad } from 'lib/tensor.js';
import { rng } from 'lib/util.js';

// The goal: fine-tune the checkpoint twice on the same instruction data for the same number of steps,
// once through YOUR adapters and once in full, then merge the adapters away and show nothing changed.
export default async function demo(m, lab) {
  const checkpoint = (await import('lib/checkpoints/tiny-gpt.json', { with: { type: 'json' } })).default;
  const tokenizerJson = (await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default;
  const tokenizer = BPETokenizer.fromJSON(tokenizerJson);
  const examples = INSTRUCTIONS.map(({ prompt, response }) => m.encodeExample(tokenizer, prompt, response));

  // Masked loss over every example at once (one padded batch), without building a graph.
  const evalLoss = (model) => noGrad(() => {
    const len = Math.max(...examples.map((e) => e.ids.length));
    const x = [], y = [], mask = [];
    for (const e of examples) {
      const ids = e.ids.concat(new Array(len - e.ids.length).fill(0));
      const mk = e.mask.concat(new Array(len - e.mask.length).fill(0));
      x.push(ids.slice(0, -1)); y.push(ids.slice(1)); mask.push(mk.slice(1));
    }
    return m.maskedLoss(model.forward(x), y, mask).item();
  });
  const prompts = ['Say hello.', 'Name a color.', 'Count to three.', 'Name a fruit.', 'What is two plus two?'];
  const show = (s) => (s.length ? s : '(empty)');

  const base = GPT.fromJSON(checkpoint);
  const baseLoss = evalLoss(base);
  // The checkpoint has never seen this format, so show its raw greedy continuation (newlines escaped).
  const before = prompts.map((p) => {
    const ids = tokenizer.encode(m.formatPrompt(p));
    const out = base.generate(ids, { maxNewTokens: 10, topK: 1, next: rng(0) });
    return tokenizer.decode(out.slice(ids.length)).replace(/\n/g, '\\n');
  });
  lab.log(`checkpoint: ${base.numParams().toLocaleString()} parameters; masked loss on the ${examples.length} instruction pairs before fine-tuning: ${baseLoss.toFixed(3)}`);

  const steps = 100, batchSize = 2;
  const onStep = (name, offset) => async (step, loss) => {
    if (step % 5 === 0) { lab.progress((offset + step + 1) / (2 * steps), `${name} step ${step} loss ${loss.toFixed(2)}`); await lab.tick(); }
    if (step % 25 === 0) lab.log(`${name} step ${step}: loss ${loss.toFixed(3)}`);
  };

  // 1. LoRA: r = 8, alpha = 16, on every linear layer.
  const loraModel = GPT.fromJSON(checkpoint);
  const wrapped = m.applyLora(loraModel, { rank: 8, alpha: 16, targets: m.ALL_LINEAR, next: rng(3) });
  const loraCount = m.countParams(loraModel);
  lab.log(`LoRA r=8, alpha=16 on ${wrapped.length} linear layers: ${loraCount.trainable.toLocaleString()} trainable of ${loraCount.total.toLocaleString()} (${loraCount.percent.toFixed(2)}%)`);
  const t0 = performance.now();
  const lora = await m.finetune(loraModel, examples, { steps, lr: 3e-3, batchSize, next: rng(11), onStep: onStep('LoRA', 0) });
  const loraSeconds = (performance.now() - t0) / 1000;
  const loraOptState = lora.optimizer.m.reduce((s, a) => s + a.length, 0) + lora.optimizer.v.reduce((s, a) => s + a.length, 0);
  const loraLoss = evalLoss(loraModel);
  const loraReplies = prompts.map((p) => m.reply(loraModel, tokenizer, p));

  // 2. Merge the adapters into the weights and check the model did not change.
  const probe = [tokenizer.encode(m.formatPrompt(INSTRUCTIONS[0].prompt) + ' ' + INSTRUCTIONS[0].response)];
  const logitsBefore = noGrad(() => Array.from(loraModel.forward(probe).data));
  const merged = m.mergeLora(loraModel);
  const logitsAfter = noGrad(() => Array.from(loraModel.forward(probe).data));
  let maxDiff = 0;
  for (let i = 0; i < logitsBefore.length; i++) maxDiff = Math.max(maxDiff, Math.abs(logitsBefore[i] - logitsAfter[i]));
  lab.check(maxDiff < 1e-4, `merged logits differ from the adapted model by ${maxDiff}; merge must compute W + (alpha/r)·A·B`);
  lab.check(loraModel.numParams() === base.numParams(), 'the merged model must have exactly the base parameter count');
  const mergedReplies = prompts.map((p) => m.reply(loraModel, tokenizer, p));
  const sameReplies = mergedReplies.every((r, i) => r === loraReplies[i]);
  lab.log(`merged ${merged} adapters: max |logit difference| ${maxDiff.toExponential(2)}, ${loraModel.numParams().toLocaleString()} parameters, replies ${sameReplies ? 'identical' : 'DIFFERENT'}`);

  // 3. Full fine-tuning: same data, same steps, same batches, every parameter trainable.
  const fullModel = GPT.fromJSON(checkpoint);
  const fullCount = m.countParams(fullModel);
  const t1 = performance.now();
  const full = await m.finetune(fullModel, examples, { steps, lr: 1e-3, batchSize, next: rng(11), onStep: onStep('full', steps) });
  const fullSeconds = (performance.now() - t1) / 1000;
  const fullOptState = full.optimizer.m.reduce((s, a) => s + a.length, 0) + full.optimizer.v.reduce((s, a) => s + a.length, 0);
  const fullLoss = evalLoss(fullModel);
  const fullReplies = prompts.map((p) => m.reply(fullModel, tokenizer, p));

  // Visuals.
  const smooth = (xs) => xs.map((_, i) => { const w = xs.slice(Math.max(0, i - 9), i + 1); return w.reduce((a, b) => a + b, 0) / w.length; });
  lab.plot({
    title: 'Masked training loss (10-step mean), same data and batches; lr 3e-3 for LoRA, 1e-3 for full',
    x: lora.losses.map((_, i) => i),
    series: [{ name: `LoRA r=8 (${loraCount.percent.toFixed(1)}% trainable)`, values: smooth(lora.losses) }, { name: 'full fine-tuning (100%)', values: smooth(full.losses) }],
    xlabel: 'step', ylabel: 'loss (nats per response token)',
  });
  lab.bar({ title: 'Trainable parameters', labels: ['LoRA r=8', 'full fine-tuning'], values: [loraCount.trainable, fullCount.trainable] });
  const loraMem = m.trainingMemory(loraCount), fullMem = m.trainingMemory(fullCount);
  lab.bar({
    title: 'Gradient + optimizer bytes (mixed-precision AdamW accounting: 14 B per trainable parameter)',
    labels: ['LoRA r=8', 'full fine-tuning'],
    values: [loraMem.grads + loraMem.optimizer, fullMem.grads + fullMem.optimizer],
  });
  lab.table({
    title: 'Memory accounting (bytes) and what this run actually allocated',
    columns: ['run', 'trainable', 'weights (2 B each)', 'grads (2 B)', 'optimizer (12 B)', 'total', 'AdamW m+v values allocated here'],
    rows: [
      ['LoRA r=8', loraCount.trainable, loraMem.weights, loraMem.grads, loraMem.optimizer, loraMem.total, loraOptState],
      ['full', fullCount.trainable, fullMem.weights, fullMem.grads, fullMem.optimizer, fullMem.total, fullOptState],
    ],
  });
  lab.table({
    title: 'Greedy replies to "User: <prompt>\\nBot:"',
    columns: ['prompt', 'checkpoint (raw continuation)', 'LoRA (merged)', 'full fine-tuning'],
    rows: prompts.map((p, i) => [p + (INSTRUCTIONS.some((e) => e.prompt === p) ? '' : ' (not in training set)'), show(before[i]), show(mergedReplies[i]), show(fullReplies[i])]),
  });

  const recovered = (100 * (baseLoss - loraLoss)) / (baseLoss - fullLoss);
  lab.log(`time: LoRA ${loraSeconds.toFixed(1)} s, full ${fullSeconds.toFixed(1)} s for ${steps} steps each`);
  lab.done(
    `LoRA trained **${loraCount.trainable.toLocaleString()}** of ${loraCount.total.toLocaleString()} parameters (**${loraCount.percent.toFixed(1)}%**) ` +
    `and allocated ${loraOptState.toLocaleString()} AdamW values against ${fullOptState.toLocaleString()} for full fine-tuning. ` +
    `Masked loss on the ${examples.length} pairs went ${baseLoss.toFixed(2)} → **${loraLoss.toFixed(2)}** with LoRA and → **${fullLoss.toFixed(2)}** with full fine-tuning ` +
    `after ${steps} steps each, so LoRA recovered **${recovered.toFixed(0)}%** of the full fine-tuning improvement. ` +
    `Merging ${merged} adapters changed the logits by at most ${maxDiff.toExponential(1)} and left ${loraModel.numParams().toLocaleString()} parameters, the base count. ` +
    `At d = 64 a rank-8 adapter is large relative to the model; on a 7B model the same recipe trains about 0.3% and needs approximately ${(m.trainingMemory({ total: 6.74e9 + 2e7, trainable: 2e7 }).total / 1e9).toFixed(1)} GB of state against ${(m.trainingMemory({ total: 6.74e9, trainable: 6.74e9 }).total / 1e9).toFixed(0)} GB.`,
  );
}
