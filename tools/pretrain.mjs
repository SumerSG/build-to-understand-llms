// tools/pretrain.mjs — train the checkpoint the inference modules load: lib/checkpoints/tiny-gpt.json
// and the BPE tokenizer that goes with it, lib/checkpoints/tokenizer.json.
//
// This is a Node script (it writes files), but every line of model code it calls lives in lib/ and runs
// unchanged in the browser. It is the same loop module 07 asks the learner to write, only longer:
//
//   batch -> forward -> crossEntropy -> backward -> clip -> AdamW step -> zeroGrad
//
// Usage:
//   node tools/pretrain.mjs                 # 1500 steps from scratch
//   node tools/pretrain.mjs 300             # 300 steps
//   node tools/pretrain.mjs 2000 --resume   # continue from the saved checkpoint

import { mkdirSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { CORPUS, getBatch, trainValSplit } from '../lib/data.js';
import { BPETokenizer } from '../lib/tokenizer.js';
import { GPT } from '../lib/gpt.js';
import { crossEntropy, noGrad } from '../lib/tensor.js';
import { AdamW, clipGradNorm, cosineWithWarmup } from '../lib/optim.js';
import { loadModel } from '../lib/infer.js';
import { generate } from '../lib/sampling.js';
import { rng, fmt, now } from '../lib/util.js';

const CHECKPOINT_DIR = new URL('../lib/checkpoints/', import.meta.url);
const TOKENIZER_PATH = fileURLToPath(new URL('tokenizer.json', CHECKPOINT_DIR));
const MODEL_PATH = fileURLToPath(new URL('tiny-gpt.json', CHECKPOINT_DIR));

const CONFIG = { vocabSize: 256, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 64, seed: 0 };
const TRAIN = {
  batchSize: 16,
  lr: 3e-3,
  warmup: 50,
  weightDecay: 0.1,
  maxGradNorm: 1.0,
  evalBatches: 8,
  logEvery: 50,
  sampleEvery: 250,
  saveEvery: 250,
};

/** Read the command line: an optional step count and an optional --resume flag. */
function parseArgs(argv) {
  let steps = 1500;
  let resume = false;
  for (const arg of argv) {
    if (arg === '--resume') resume = true;
    else if (/^--steps=\d+$/.test(arg)) steps = Number(arg.slice('--steps='.length));
    else if (/^\d+$/.test(arg)) steps = Number(arg);
    else throw new Error(`pretrain: unknown argument '${arg}'`);
  }
  return { steps, resume };
}

/** Train the BPE tokenizer on the corpus, or load the saved one when resuming. */
function prepareTokenizer(resume) {
  if (resume && existsSync(TOKENIZER_PATH)) {
    console.log('tokenizer: reusing', TOKENIZER_PATH);
    return BPETokenizer.fromJSON(JSON.parse(readFileSync(TOKENIZER_PATH, 'utf8')));
  }
  const started = now();
  const tokenizer = BPETokenizer.train(CORPUS, { vocabSize: CONFIG.vocabSize });
  writeFileSync(TOKENIZER_PATH, JSON.stringify(tokenizer.toJSON()));
  console.log(`tokenizer: trained vocab ${tokenizer.vocabSize} in ${fmt(now() - started)} ms -> ${TOKENIZER_PATH}`);
  return tokenizer;
}

/** The loss of predicting every token from its corpus frequency alone: the bar the model has to beat. */
function unigramEntropy(ids, vocabSize) {
  const counts = new Float64Array(vocabSize);
  for (const id of ids) counts[id] += 1;
  let entropy = 0;
  for (const count of counts) {
    if (count === 0) continue;
    const p = count / ids.length;
    entropy -= p * Math.log(p);
  }
  return entropy;
}

/** Mean loss over a few fixed batches, with no graph recorded (evaluation must not allocate gradients). */
function estimateLoss(model, ids, seed) {
  return noGrad(() => {
    const next = rng(seed);
    let total = 0;
    for (let i = 0; i < TRAIN.evalBatches; i++) {
      const { x, y } = getBatch(ids, { blockSize: CONFIG.blockSize, batchSize: TRAIN.batchSize, next });
      total += crossEntropy(model.forward(x), y).item();
    }
    return total / TRAIN.evalBatches;
  });
}

/** Write the model plus a little training metadata (extra keys are ignored by GPT.fromJSON/loadModel). */
function saveCheckpoint(model, meta) {
  writeFileSync(MODEL_PATH, JSON.stringify({ ...model.toJSON(), meta }));
}

/** Generate a short sample from the model as it stands, through the same inference path the modules use. */
function sampleText(model, tokenizer, prompt, seed) {
  const inferModel = loadModel(model.toJSON());
  return generate(inferModel, tokenizer, prompt, {
    maxNewTokens: 48,
    temperature: 0.8,
    topK: 20,
    next: rng(seed),
  });
}

function main() {
  const { steps, resume } = parseArgs(process.argv.slice(2));
  mkdirSync(fileURLToPath(CHECKPOINT_DIR), { recursive: true });

  const tokenizer = prepareTokenizer(resume);
  const ids = tokenizer.encode(CORPUS);
  const { train, val } = trainValSplit(ids, 0.9);
  const baseline = unigramEntropy(train, tokenizer.vocabSize);
  console.log(
    `data: ${CORPUS.length} chars -> ${ids.length} tokens ` +
      `(${(ids.length / CORPUS.length).toFixed(3)} tokens/char), train ${train.length}, val ${val.length}`,
  );
  console.log(`baseline: unigram entropy ${baseline.toFixed(4)} nats/token (the loss to beat)`);

  let model;
  let startStep = 0;
  if (resume && existsSync(MODEL_PATH)) {
    const saved = JSON.parse(readFileSync(MODEL_PATH, 'utf8'));
    model = GPT.fromJSON(saved);
    startStep = saved.meta?.step ?? 0;
    console.log(`model: resumed from ${MODEL_PATH} at step ${startStep} (optimizer state restarts)`);
  } else {
    model = new GPT(CONFIG);
    console.log(`model: fresh ${fmt(model.numParams())} parameters (${model.numParams()})`);
  }

  const optimizer = new AdamW(model.parameters(), {
    lr: TRAIN.lr,
    betas: [0.9, 0.95],
    weightDecay: TRAIN.weightDecay,
  });
  const batchRng = rng(1234 + startStep);
  const total = startStep + steps;
  const started = now();
  let lastLoss = NaN;

  for (let step = startStep; step < total; step++) {
    // The schedule is defined over the whole run, so a resumed run picks it up where it left off.
    optimizer.lr = cosineWithWarmup(step, { warmup: TRAIN.warmup, total, peak: TRAIN.lr });

    const { x, y } = getBatch(train, { blockSize: CONFIG.blockSize, batchSize: TRAIN.batchSize, next: batchRng });
    const loss = crossEntropy(model.forward(x), y);
    loss.backward();
    const gradNorm = clipGradNorm(model.parameters(), TRAIN.maxGradNorm);
    optimizer.step();
    optimizer.zeroGrad();
    lastLoss = loss.item();

    if (step % TRAIN.logEvery === 0 || step === total - 1) {
      const elapsed = (now() - started) / 1000;
      const valLoss = estimateLoss(model, val, 99);
      console.log(
        `step ${String(step).padStart(5)} | train ${lastLoss.toFixed(4)} | val ${valLoss.toFixed(4)} | ` +
          `lr ${optimizer.lr.toExponential(2)} | |g| ${gradNorm.toFixed(2)} | ` +
          `${((step - startStep + 1) / elapsed).toFixed(2)} steps/s`,
      );
    }
    if (step > startStep && step % TRAIN.sampleEvery === 0) {
      console.log(`sample @ ${step}: ${JSON.stringify(sampleText(model, tokenizer, 'The ', step))}`);
    }
    if (step > startStep && step % TRAIN.saveEvery === 0) {
      saveCheckpoint(model, { step: step + 1, trainLoss: lastLoss, config: CONFIG });
    }
  }

  const trainLoss = estimateLoss(model, train, 7);
  const valLoss = estimateLoss(model, val, 99);
  saveCheckpoint(model, { step: total, trainLoss, valLoss, baseline, config: CONFIG });

  const wallClock = (now() - started) / 1000;
  const tokensSeen = steps * TRAIN.batchSize * CONFIG.blockSize;
  console.log('---');
  console.log(`done: ${steps} steps in ${wallClock.toFixed(1)} s (${(steps / wallClock).toFixed(2)} steps/s, ` +
    `${fmt(tokensSeen / wallClock)} tokens/s)`);
  console.log(`final: train ${trainLoss.toFixed(4)} | val ${valLoss.toFixed(4)} | unigram baseline ${baseline.toFixed(4)}`);
  console.log(`sample: ${JSON.stringify(sampleText(model, tokenizer, 'The ', 4242))}`);
  console.log(`saved: ${MODEL_PATH} (${(readFileSync(MODEL_PATH).length / 1e6).toFixed(2)} MB)`);
}

main();
