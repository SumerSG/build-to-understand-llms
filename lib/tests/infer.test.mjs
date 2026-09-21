// Tests for lib/infer.js (forward-only inference and the KV cache) and generate() from lib/sampling.js.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import * as ops from '../ops.js';
import { rng } from '../util.js';
import { GPT } from '../gpt.js';
import { BPETokenizer } from '../tokenizer.js';
import { CORPUS, trainValSplit } from '../data.js';
import { generate } from '../sampling.js';
import { loadModel, forward, newCache, forwardStep, prefill, flopsPerToken } from '../infer.js';

const CONFIG = { vocabSize: 23, blockSize: 16, nLayer: 2, nHead: 2, nEmbd: 8, seed: 5 };

/** A small trained-looking model plus its raw-tensor inference twin. */
function buildPair(config = CONFIG) {
  const model = new GPT(config);
  return { model, inferModel: loadModel(model.toJSON()) };
}

/** Read a checkpoint that tools/pretrain.mjs wrote next to lib/. */
function readCheckpoint(name) {
  return JSON.parse(readFileSync(new URL(`../checkpoints/${name}`, import.meta.url), 'utf8'));
}

test('loadModel: config and raw Float32Array weights', () => {
  const { model, inferModel } = buildPair();
  assert.deepEqual(inferModel.config, CONFIG);
  assert.deepEqual(Object.keys(inferModel.w), model.namedParameters().map(([name]) => name));
  assert.ok(inferModel.w['wte.weight'].data instanceof Float32Array);
  assert.deepEqual(inferModel.w['wte.weight'].shape, [CONFIG.vocabSize, CONFIG.nEmbd]);
});

test('infer.forward matches GPT.forward at every position', () => {
  const { model, inferModel } = buildPair();
  const ids = [3, 1, 4, 1, 5, 9, 2, 6];
  const raw = forward(inferModel, ids);
  const reference = model.forward([ids]);
  assert.deepEqual(raw.shape, [ids.length, CONFIG.vocabSize]);
  let maxDiff = 0;
  for (let i = 0; i < raw.data.length; i++) maxDiff = Math.max(maxDiff, Math.abs(raw.data[i] - reference.data[i]));
  assert.ok(maxDiff < 1e-4, `largest logit difference ${maxDiff}`);
});

test('infer.forward rejects an empty or over-long sequence', () => {
  const { inferModel } = buildPair();
  assert.throws(() => forward(inferModel, []), /at least one token/);
  assert.throws(() => forward(inferModel, new Array(CONFIG.blockSize + 1).fill(1)), /blockSize/);
});

test('newCache starts empty and grows one position per step', () => {
  const { inferModel } = buildPair();
  const cache = newCache(inferModel);
  assert.equal(cache.length, 0);
  assert.equal(cache.k.length, CONFIG.nLayer);
  assert.equal(cache.v.length, CONFIG.nLayer);
  assert.deepEqual(cache.k[0].shape, [CONFIG.nHead, 0, CONFIG.nEmbd / CONFIG.nHead]);

  for (let t = 1; t <= 3; t++) {
    forwardStep(inferModel, cache, t);
    assert.equal(cache.length, t);
    for (let layer = 0; layer < CONFIG.nLayer; layer++) {
      assert.deepEqual(cache.k[layer].shape, [CONFIG.nHead, t, CONFIG.nEmbd / CONFIG.nHead]);
      assert.deepEqual(cache.v[layer].shape, [CONFIG.nHead, t, CONFIG.nEmbd / CONFIG.nHead]);
    }
  }
});

test('forwardStep reproduces forward at every position', () => {
  const { inferModel } = buildPair();
  const ids = [7, 7, 2, 0, 11, 3, 4, 4, 9, 1];
  const full = forward(inferModel, ids);
  const cache = newCache(inferModel);
  let maxDiff = 0;
  for (let t = 0; t < ids.length; t++) {
    const stepLogits = forwardStep(inferModel, cache, ids[t]);
    assert.equal(stepLogits.length, CONFIG.vocabSize);
    for (let j = 0; j < CONFIG.vocabSize; j++) {
      maxDiff = Math.max(maxDiff, Math.abs(stepLogits[j] - full.data[t * CONFIG.vocabSize + j]));
    }
  }
  assert.ok(maxDiff < 1e-4, `cached decoding drifted by ${maxDiff}`);
});

test('prefill returns the logits of the last prompt token', () => {
  const { inferModel } = buildPair();
  const ids = [2, 5, 8, 1];
  const full = forward(inferModel, ids);
  const cache = newCache(inferModel);
  const last = prefill(inferModel, cache, ids);
  assert.equal(cache.length, ids.length);
  const offset = (ids.length - 1) * CONFIG.vocabSize;
  for (let j = 0; j < CONFIG.vocabSize; j++) {
    assert.ok(Math.abs(last[j] - full.data[offset + j]) < 1e-4);
  }
  assert.throws(() => prefill(inferModel, newCache(inferModel), []), /at least one token/);
});

test('the cache refuses to grow past blockSize', () => {
  const { inferModel } = buildPair();
  const cache = newCache(inferModel);
  for (let t = 0; t < CONFIG.blockSize; t++) forwardStep(inferModel, cache, 1);
  assert.equal(cache.length, CONFIG.blockSize);
  assert.throws(() => forwardStep(inferModel, cache, 1), /cache is full/);
});

test('flopsPerToken: cached decoding is far cheaper than full recompute', () => {
  const config = { vocabSize: 256, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 64 };
  const cached = flopsPerToken(config, 64, { cached: true });
  const uncached = flopsPerToken(config, 64, { cached: false });
  assert.ok(cached < uncached, `${cached} should be below ${uncached}`);
  assert.equal(uncached, cached * 64);
  // 2·params for the matmuls plus 4·L·T·C for attention: 2·120576 + 4·2·64·64 = 273920.
  assert.equal(cached, 2 * 120_576 + 4 * 2 * 64 * 64);
  // The attention term is small here but grows with context, while the matmul term does not.
  assert.ok(flopsPerToken(config, 1, { cached: true }) < cached);
  assert.equal(flopsPerToken(config, 1, { cached: false }), flopsPerToken(config, 1, { cached: true }));
});

test('generate: returns only the generated text, and stops at eos', () => {
  const tokenizer = BPETokenizer.train('the cat sat on the mat. the dog sat on the log.', { vocabSize: 40 });
  const config = { vocabSize: tokenizer.vocabSize, blockSize: 16, nLayer: 1, nHead: 2, nEmbd: 8, seed: 6 };
  const model = new GPT(config);
  // Force the model to predict eos everywhere: a zero gain makes the final LayerNorm output beta, so the
  // logits are beta · wteᵀ; with beta = e₀ that is just column 0 of the embedding table.
  model.lnF.gamma.data.fill(0);
  model.lnF.beta.data.fill(0);
  model.lnF.beta.data[0] = 1;
  for (let v = 0; v < config.vocabSize; v++) model.wte.weight.data[v * config.nEmbd] = 0;
  model.wte.weight.data[tokenizer.eos * config.nEmbd] = 10;
  const inferModel = loadModel(model.toJSON());

  const stopped = generate(inferModel, tokenizer, 'the cat', { maxNewTokens: 5, temperature: 0, next: rng(1) });
  assert.equal(stopped, '');

  const unstopped = generate(inferModel, tokenizer, 'the cat', {
    maxNewTokens: 3,
    temperature: 0,
    stopAtEos: false,
    next: rng(1),
  });
  assert.equal(unstopped, '<|endoftext|>'.repeat(3));
});

test('generate: deterministic given a seed, and bounded by maxNewTokens and blockSize', () => {
  const tokenizer = BPETokenizer.train(CORPUS.slice(0, 4000), { vocabSize: 120 });
  const config = { vocabSize: tokenizer.vocabSize, blockSize: 16, nLayer: 1, nHead: 2, nEmbd: 8, seed: 7 };
  const inferModel = loadModel(new GPT(config).toJSON());
  const prompt = 'the ';

  const a = generate(inferModel, tokenizer, prompt, { maxNewTokens: 8, temperature: 1, topK: 5, next: rng(3) });
  const b = generate(inferModel, tokenizer, prompt, { maxNewTokens: 8, temperature: 1, topK: 5, next: rng(3) });
  assert.equal(typeof a, 'string');
  assert.equal(a, b);
  assert.ok(a.length > 0);
  assert.ok(!a.startsWith(prompt), 'the prompt must not be repeated in the output');

  // Asking for more tokens than the context window holds stops cleanly instead of throwing.
  const long = generate(inferModel, tokenizer, prompt, { maxNewTokens: 100, temperature: 1, next: rng(4) });
  assert.equal(typeof long, 'string');

  // A prompt longer than the context window keeps its newest tokens rather than throwing.
  const longPrompt = 'the quick brown fox jumps over the lazy dog. '.repeat(4);
  assert.ok(tokenizer.encode(longPrompt).length > config.blockSize);
  const fromLong = generate(inferModel, tokenizer, longPrompt, { maxNewTokens: 4, temperature: 1, next: rng(5) });
  assert.equal(typeof fromLong, 'string');
  assert.throws(() => generate(inferModel, tokenizer, prompt, { maxNewTokens: 2 }), /seeded rng/);
});

test('the trained checkpoint beats the unigram baseline on held-out text', () => {
  const tokenizer = BPETokenizer.fromJSON(readCheckpoint('tokenizer.json'));
  const inferModel = loadModel(readCheckpoint('tiny-gpt.json'));
  assert.equal(inferModel.config.vocabSize, tokenizer.vocabSize);

  const ids = tokenizer.encode(CORPUS);
  const { train, val } = trainValSplit(ids, 0.9);

  // Baseline: predict every token from its corpus frequency alone. Its loss is the unigram entropy.
  const counts = new Float64Array(tokenizer.vocabSize);
  for (const id of train) counts[id] += 1;
  let unigramEntropy = 0;
  for (const count of counts) {
    if (count === 0) continue;
    const p = count / train.length;
    unigramEntropy -= p * Math.log(p);
  }

  // A fixed slice of the validation split: the first few full windows, scored with the uncached forward.
  const blockSize = inferModel.config.blockSize;
  const windows = 4;
  let totalLoss = 0;
  let totalTokens = 0;
  for (let w = 0; w < windows; w++) {
    const window = val.slice(w * blockSize, w * blockSize + blockSize + 1);
    const logits = forward(inferModel, window.slice(0, blockSize));
    const logProbs = ops.logSoftmax(logits);
    for (let t = 0; t < blockSize; t++) {
      totalLoss -= logProbs.data[t * inferModel.config.vocabSize + window[t + 1]];
      totalTokens += 1;
    }
  }
  const valLoss = totalLoss / totalTokens;
  assert.ok(Number.isFinite(valLoss));
  assert.ok(valLoss < unigramEntropy - 0.5, `val loss ${valLoss.toFixed(3)} vs unigram ${unigramEntropy.toFixed(3)}`);
});

test('the trained checkpoint generates corpus-like text', () => {
  const tokenizer = BPETokenizer.fromJSON(readCheckpoint('tokenizer.json'));
  const inferModel = loadModel(readCheckpoint('tiny-gpt.json'));
  const text = generate(inferModel, tokenizer, 'The ', { maxNewTokens: 40, temperature: 0.8, topK: 20, next: rng(9) });
  assert.equal(typeof text, 'string');
  assert.ok(text.length > 10, `generated only ${JSON.stringify(text)}`);
  // Every character it produces should be one the corpus uses.
  const corpusChars = new Set(CORPUS);
  for (const ch of text) assert.ok(corpusChars.has(ch), `unexpected character ${JSON.stringify(ch)}`);
});
