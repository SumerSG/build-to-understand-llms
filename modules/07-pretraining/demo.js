import { CharTokenizer } from 'lib/tokenizer.js';
import { toyCorpus } from 'lib/data.js';
import { rng } from 'lib/util.js';

// The goal: watch your loop turn a random GPT into one that writes sentences like the corpus.
export default async function demo(m, lab) {
  const text = toyCorpus(800, 1);
  const tokenizer = new CharTokenizer(text);
  const ids = tokenizer.encode(text);
  const { train: trainIds, val: valIds } = m.trainValSplit(ids, 0.9);

  const config = {
    trainIds, valIds, vocabSize: tokenizer.vocabSize,
    blockSize: 32, nLayer: 1, nHead: 2, nEmbd: 32, seed: 1,
    steps: 350, batchSize: 8, lr: 1e-2, warmup: 35, weightDecay: 0.1, maxGradNorm: 1.0,
    evalInterval: 50, evalBatches: 4,
  };
  const tokensPerStep = config.batchSize * config.blockSize;
  lab.log(`corpus: ${text.length} characters, vocab ${tokenizer.vocabSize}, train ${trainIds.length} / val ${valIds.length} tokens`);
  lab.log(`model: ${config.nLayer} layer, ${config.nEmbd} dims, ${config.nHead} heads, block ${config.blockSize}; ${tokensPerStep} tokens per step, ${config.steps} steps`);

  // What an untrained model writes (the same seed the run will use, so this is literally step 0).
  const prompt = 'The ';
  const sampleAt = (model, seed) => m.sample(model, tokenizer, prompt, { maxNewTokens: 90, temperature: 0.8, next: rng(seed) }).replace(/\n/g, ' / ');
  const samples = [{ step: 0, text: sampleAt(m.makeModel(config), 11) }];
  lab.log(`step 0 sample: ${samples[0].text}`);

  const steps = [], trainLoss = [], valLoss = [], lrs = [], gradNorms = [];
  const mid = Math.floor(config.steps / 2);
  const t0 = performance.now();
  const run = await m.train(config, async (r, model) => {
    steps.push(r.step); trainLoss.push(r.loss); lrs.push(r.lr); gradNorms.push(r.gradNorm);
    valLoss.push(r.valLoss === undefined ? NaN : r.valLoss);
    if (r.valLoss !== undefined) lab.log(`step ${r.step}: train ${r.loss.toFixed(3)}  val ${r.valLoss.toFixed(3)}  lr ${r.lr.toExponential(2)}  |g| ${r.gradNorm.toFixed(2)}`);
    if (r.step === mid) { samples.push({ step: r.step, text: sampleAt(model, 11) }); lab.log(`step ${r.step} sample: ${samples.at(-1).text}`); }
    if (r.step % 5 === 0) { lab.progress((r.step + 1) / config.steps, `step ${r.step} loss ${r.loss.toFixed(2)}`); await lab.tick(); }
  });
  const seconds = (performance.now() - t0) / 1000;
  const tokensPerSec = run.tokensSeen / seconds;
  samples.push({ step: config.steps, text: sampleAt(run.model, 11) });
  lab.log(`step ${config.steps} sample: ${samples.at(-1).text}`);

  lab.plot({
    title: 'Loss (nats per character): the heartbeat of every training run',
    x: steps, series: [{ name: 'train (per batch)', values: trainLoss }, { name: 'validation (4 batches)', values: valLoss }],
    xlabel: 'step', ylabel: 'cross-entropy',
  });
  lab.plot({ title: 'Learning rate: linear warmup, then cosine to peak/10', x: steps, series: [{ name: 'lr', values: lrs }], xlabel: 'step', ylabel: 'lr' });
  lab.plot({ title: 'Gradient norm before clipping (clipped to 1.0)', x: steps, series: [{ name: '|g|', values: gradNorms }], xlabel: 'step', ylabel: 'L2 norm' });
  lab.table({ title: `Samples from "${prompt}" at temperature 0.8`, columns: ['step', 'sample'], rows: samples.map((s) => [s.step, s.text]) });

  const first = trainLoss.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
  const last = trainLoss.slice(-10).reduce((a, b) => a + b, 0) / 10;
  const finalVal = valLoss.filter(Number.isFinite).at(-1);
  const clipped = gradNorms.filter((g) => g > config.maxGradNorm).length;
  const uniform = Math.log(tokenizer.vocabSize);
  const gap = finalVal - last;
  const verdict = gap < 0.2
    ? `close to the train loss (gap ${gap.toFixed(2)}), so the model is undertrained rather than overfitting: more steps would still help`
    : `well above the train loss (gap ${gap.toFixed(2)}): the model has started to memorise its training windows`;
  lab.done(`Your loop trained a ${run.model.numParams().toLocaleString()}-parameter GPT for **${config.steps} steps** on **${run.tokensSeen.toLocaleString()} tokens** in ${seconds.toFixed(1)} s (**${Math.round(tokensPerSec).toLocaleString()} tokens/s**). Train loss, averaged over the first and last 10 steps, fell from ${first.toFixed(3)} (uniform guessing is ln ${tokenizer.vocabSize} = ${uniform.toFixed(3)}) to **${last.toFixed(3)}** nats/char; final validation loss **${finalVal.toFixed(3)}**, ${verdict}. Clipping fired on ${clipped} of ${config.steps} steps. Final sample: "${samples.at(-1).text.slice(0, 80)}…". Compare it with the step-0 sample: the characters now form word-length chunks separated by spaces, sentences end in punctuation, and some real words from the corpus appear, but most words are still invented. That is what about 1.5 nats/char looks like; more steps, more layers or a wider model push the loss down and turn more of the chunks into real words.`);
}
