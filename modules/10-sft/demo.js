import { GPT } from 'lib/gpt.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { INSTRUCTIONS, CHAT } from 'lib/data.js';
import { rng } from 'lib/util.js';

// The goal: take the pre-trained checkpoint, which has never seen a chat marker, and watch YOUR template,
// mask, packing, embedding resize and loop turn it into a model that answers inside the template and stops.
export default async function demo(m, lab) {
  const checkpoint = (await import('lib/checkpoints/tiny-gpt.json', { with: { type: 'json' } })).default;
  const tokenizerJson = (await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default;
  const base = GPT.fromJSON(checkpoint);
  const baseTokenizer = BPETokenizer.fromJSON(tokenizerJson);

  // 1. Tokens for the template (step 1) and a model that has rows for them (step 5).
  const tokenizer = m.addChatTokens(baseTokenizer);
  const ids = m.markerIds(tokenizer);
  lab.log(`tokenizer: ${baseTokenizer.vocabSize} -> ${tokenizer.vocabSize} ids; markers ${JSON.stringify(ids)}`);
  const model = m.resizeEmbeddings(base, tokenizer.vocabSize);
  lab.log(`model: ${base.numParams().toLocaleString()} -> ${model.numParams().toLocaleString()} parameters (${(tokenizer.vocabSize - baseTokenizer.vocabSize) * base.config.nEmbd} new embedding values)`);

  // 2. The dataset (steps 2 and 4): every INSTRUCTIONS pair, masked to the assistant, packed into windows.
  const examples = INSTRUCTIONS.map(({ prompt, response }) => m.tokenizeExample(tokenizer, prompt, response));
  const totalTokens = examples.reduce((s, e) => s + e.ids.length, 0);
  const assistantTokens = examples.reduce((s, e) => s + e.mask.reduce((a, b) => a + b, 0), 0);
  const packs = m.packExamples(examples, { blockSize: model.config.blockSize, eos: tokenizer.eos });
  const packedPositions = packs.length * model.config.blockSize;
  // Count the packed TARGET positions (y) by what they are: assistant, prompt/marker, or separator/padding.
  let packedIn = 0, packedPrompt = 0, packedFill = 0;
  for (const p of packs) for (let t = 0; t < p.y.length; t++) {
    if (p.mask[t]) packedIn++;
    else if (p.y[t] === tokenizer.eos) packedFill++;
    else packedPrompt++;
  }
  lab.log(`${examples.length} examples, ${totalTokens} tokens of which ${assistantTokens} are assistant tokens; packed into ${packs.length} windows of ${model.config.blockSize}`);

  // Show the first example the way the model sees it: what is predicted from what, and whether it counts.
  const first = m.buildExample(tokenizer, INSTRUCTIONS[0].prompt, INSTRUCTIONS[0].response);
  const show = (text) => text.replace(/\n/g, '\\n');
  lab.table({
    title: `One example as (x, y, mask): "${INSTRUCTIONS[0].prompt}" -> "${INSTRUCTIONS[0].response}"`,
    columns: ['t', 'x[t] (input)', 'y[t] (target)', 'mask[t]'],
    rows: first.x.map((x, t) => [t, show(tokenizer.decode([x])), show(tokenizer.decode([first.y[t]])), first.mask[t]]),
  });
  lab.bar({
    title: 'Positions in the packed dataset: only the masked-in ones produce gradient',
    labels: ['assistant (mask 1)', 'prompt + markers (mask 0)', 'separators + padding (mask 0)'],
    values: [packedIn, packedPrompt, packedFill],
  });

  // 3. Before: the base model has no idea what the markers mean.
  const prompts = ['Say hello.', 'Name a color.', 'Count to three.', 'What color is the sky?', 'Name a fruit.'];
  const heldOut = new Set(prompts.filter((p) => !INSTRUCTIONS.some((ex) => ex.prompt === p)));
  const before = prompts.map((p) => m.generateChat(model, tokenizer, p, { maxNewTokens: 20 }));
  for (let i = 0; i < prompts.length; i++) lab.log(`before  ${prompts[i]}  ->  ${JSON.stringify(show(before[i].text))}${before[i].ended ? '' : '  (never emitted <|end|>)'}`);

  // 4. Fine-tune (steps 3 and 6).
  const config = { steps: 150, lr: 1e-3, batchSize: 2, weightDecay: 0.1, maxGradNorm: 1.0 };
  lab.log(`fine-tuning: ${config.steps} steps, batch ${config.batchSize} x ${model.config.blockSize}, lr ${config.lr} (pre-training peaked at 3e-3), AdamW wd ${config.weightDecay}`);
  const t0 = performance.now();
  const losses = await m.finetune(model, packs, {
    ...config,
    next: rng(1),
    onStep: async (step, loss) => {
      if (step % 5 === 0) { lab.progress((step + 1) / config.steps, `step ${step} masked loss ${loss.toFixed(2)}`); await lab.tick(); }
      if (step % 30 === 0) lab.log(`step ${step}: masked loss ${loss.toFixed(3)}`);
    },
  });
  const seconds = (performance.now() - t0) / 1000;
  const smooth = losses.map((_, i) => { const w = losses.slice(Math.max(0, i - 9), i + 1); return w.reduce((a, b) => a + b, 0) / w.length; });
  lab.plot({
    title: 'Masked cross-entropy (assistant tokens only), nats per token',
    x: losses.map((_, i) => i), series: [{ name: 'per step', values: losses }, { name: '10-step mean', values: smooth }],
    xlabel: 'step', ylabel: 'loss',
  });

  // 5. After: the same prompts through the same template.
  const after = prompts.map((p) => m.generateChat(model, tokenizer, p, { maxNewTokens: 20 }));
  lab.table({
    title: 'Greedy completions of <|user|>prompt<|end|><|assistant|> before and after SFT',
    columns: ['prompt', 'before SFT', 'after SFT', 'emitted <|end|>?'],
    rows: prompts.map((p, i) => [p + (heldOut.has(p) ? ' (not in the training set)' : ''), show(before[i].text), show(after[i].text), after[i].ended ? `yes, after ${after[i].tokens} tokens` : 'no']),
  });
  for (let i = 0; i < prompts.length; i++) lab.log(`after   ${prompts[i]}  ->  ${JSON.stringify(show(after[i].text))}`);

  const firstLoss = losses.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
  const lastLoss = losses.slice(-10).reduce((a, b) => a + b, 0) / 10;
  const endedBefore = before.filter((g) => g.ended).length;
  const endedAfter = after.filter((g) => g.ended).length;
  const tokensSeen = config.steps * config.batchSize * model.config.blockSize;
  lab.done(`Your SFT loop fine-tuned the ${model.numParams().toLocaleString()}-parameter checkpoint for **${config.steps} steps** (${tokensSeen.toLocaleString()} window positions, ${Math.round((packedIn / packedPositions) * 100)}% of them masked in) in ${seconds.toFixed(1)} s. The masked loss fell from **${firstLoss.toFixed(3)}** to **${lastLoss.toFixed(3)}** nats per assistant token. Before SFT, ${endedBefore} of ${prompts.length} completions emitted \`${CHAT.end}\`; after, **${endedAfter} of ${prompts.length}** did. "${prompts[0]}" went from "${show(before[0].text).slice(0, 40)}" to "${show(after[0].text)}"; the held-out prompt "${[...heldOut][0]}" gave "${show(after[prompts.indexOf([...heldOut][0])].text)}", which shows the format was learned even where the content was not.`);
}
