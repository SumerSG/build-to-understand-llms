// Tests for lib/gpt.js — the layers, the model, the parameter bookkeeping and the checkpoint format.
import test from 'node:test';
import assert from 'node:assert/strict';

import * as ops from '../ops.js';
import { Tensor, crossEntropy } from '../tensor.js';
import { AdamW, clipGradNorm } from '../optim.js';
import { rng } from '../util.js';
import { GPT, Block, MLP, Linear, Embedding, LayerNorm, paramNames } from '../gpt.js';

/** The parameter count of a config, worked out on paper rather than by walking the model. */
function countParams({ vocabSize, blockSize, nLayer, nEmbd }) {
  const embeddings = vocabSize * nEmbd + blockSize * nEmbd;
  const perBlock =
    2 * nEmbd + // ln1: gamma + beta
    (nEmbd * 3 * nEmbd + 3 * nEmbd) + // qkv projection
    (nEmbd * nEmbd + nEmbd) + // attention output projection
    2 * nEmbd + // ln2
    (nEmbd * 4 * nEmbd + 4 * nEmbd) + // mlp fc
    (4 * nEmbd * nEmbd + nEmbd); // mlp proj
  return embeddings + nLayer * perBlock + 2 * nEmbd; // + the final LayerNorm
}

const LAB_CONFIG = { vocabSize: 256, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 64, seed: 0 };
const TINY_CONFIG = { vocabSize: 17, blockSize: 12, nLayer: 2, nHead: 2, nEmbd: 8, seed: 1 };

test('Linear: y = x·W + b with the expected shapes', () => {
  const layer = new Linear(3, 2, { next: rng(1) });
  assert.deepEqual(layer.weight.shape, [3, 2]);
  assert.deepEqual(layer.bias.shape, [2]);
  for (const b of layer.bias.data) assert.equal(b, 0);

  const x = Tensor.from([[1, 2, 3], [4, 5, 6]]);
  const y = layer.forward(x);
  assert.deepEqual(y.shape, [2, 2]);
  const expected = ops.add(ops.matmul(x, layer.weight), layer.bias);
  assert.ok(ops.allClose(y, expected, 1e-6));
  assert.equal(layer.parameters().length, 2);

  const noBias = new Linear(3, 2, { bias: false, next: rng(1) });
  assert.equal(noBias.bias, null);
  assert.equal(noBias.parameters().length, 1);
});

test('Linear and Embedding refuse to initialise without a seeded rng', () => {
  assert.throws(() => new Linear(2, 2), /seeded rng/);
  assert.throws(() => new Embedding(2, 2), /seeded rng/);
});

test('Embedding: ids become rows, LayerNorm: rows become zero-mean and unit-variance', () => {
  const table = new Embedding(5, 4, { next: rng(2) });
  const rows = table.forward([[0, 3], [1, 1]]);
  assert.deepEqual(rows.shape, [2, 2, 4]);
  for (let c = 0; c < 4; c++) assert.equal(rows.data[c], table.weight.data[c]);

  const norm = new LayerNorm(4);
  const y = norm.forward(Tensor.from([[1, 2, 3, 10]]));
  let mean = 0;
  for (const value of y.data) mean += value;
  mean /= 4;
  let variance = 0;
  for (const value of y.data) variance += (value - mean) ** 2;
  variance /= 4;
  assert.ok(Math.abs(mean) < 1e-5, `mean ${mean}`);
  assert.ok(Math.abs(variance - 1) < 1e-3, `variance ${variance}`);
  assert.equal(norm.parameters().length, 2);
});

test('MLP and Block keep the residual-stream shape', () => {
  const mlp = new MLP(8, { next: rng(3) });
  assert.deepEqual(mlp.fc.weight.shape, [8, 32]);
  assert.deepEqual(mlp.proj.weight.shape, [32, 8]);
  const x = new Tensor(ops.randn([2, 3, 8], rng(4), 1));
  assert.deepEqual(mlp.forward(x).shape, [2, 3, 8]);

  const block = new Block({ nEmbd: 8, nHead: 2 }, { next: rng(5) });
  assert.deepEqual(block.forward(x).shape, [2, 3, 8]);
  assert.equal(block.parameters().length, 12);
});

test('GPT.forward: logits are [B, T, V]', () => {
  const model = new GPT(TINY_CONFIG);
  const logits = model.forward([[1, 2, 3, 4], [5, 6, 7, 8]]);
  assert.deepEqual(logits.shape, [2, 4, TINY_CONFIG.vocabSize]);
  for (const value of logits.data) assert.ok(Number.isFinite(value));
  assert.throws(() => model.forward([new Array(TINY_CONFIG.blockSize + 1).fill(0)]), /blockSize/);
  assert.throws(() => model.forward([1, 2, 3]), /number\[\]\[\]/);
});

test('GPT.numParams matches the closed form for three configs', () => {
  const configs = [
    TINY_CONFIG,
    LAB_CONFIG,
    { vocabSize: 100, blockSize: 32, nLayer: 3, nHead: 4, nEmbd: 16, seed: 7 },
  ];
  for (const config of configs) {
    const model = new GPT(config);
    assert.equal(model.numParams(), countParams(config), `config with nEmbd ${config.nEmbd}`);
  }
  // The lab model: 2 blocks of 12·64² plus the embeddings, so a bit over 100k parameters.
  assert.equal(new GPT(LAB_CONFIG).numParams(), 120_576);
});

test('paramNames: one stable name per parameter, in parameters() order', () => {
  const model = new GPT(TINY_CONFIG);
  const names = paramNames(model);
  const params = model.parameters();
  assert.equal(names.length, params.length);
  assert.equal(new Set(names).size, names.length);
  assert.equal(names[0], 'wte.weight');
  assert.equal(names[1], 'wpe.weight');
  assert.equal(names[names.length - 2], 'lnF.gamma');
  assert.equal(names[names.length - 1], 'lnF.beta');
  assert.ok(names.includes('blocks.0.attn.qkv.weight'));
  assert.ok(names.includes('blocks.1.mlp.proj.bias'));
  // The names line up with the tensors themselves, not just in count.
  const named = model.namedParameters();
  named.forEach(([name, tensor], i) => {
    assert.equal(name, names[i]);
    assert.equal(tensor, params[i]);
  });
  assert.equal(named.find(([name]) => name === 'blocks.0.ln1.gamma')[1], model.blocks[0].ln1.gamma);
});

test('GPT: the LM head is tied to the token embedding', () => {
  const model = new GPT(TINY_CONFIG);
  // A separate head would add V·C parameters; tying means wte appears exactly once.
  assert.equal(paramNames(model).filter((n) => n.startsWith('wte')).length, 1);
  const logits = model.forward([[1, 2]]);
  const hidden = model.lnF.forward(
    model.blocks.reduce((x, b) => b.forward(x), model.wte.forward([[1, 2]]).add(model.wpe.forward([0, 1]))),
  );
  const expected = ops.matmul(hidden, ops.transpose(model.wte.weight));
  assert.ok(ops.allClose(logits, expected, 1e-5));
});

test('GPT.forward is causal: a later token cannot change earlier logits', () => {
  const model = new GPT(TINY_CONFIG);
  const V = TINY_CONFIG.vocabSize;
  const base = model.forward([[1, 2, 3, 4]]);
  const changed = model.forward([[1, 2, 3, 9]]);
  for (let t = 0; t < 3; t++) {
    for (let j = 0; j < V; j++) {
      const diff = Math.abs(base.data[t * V + j] - changed.data[t * V + j]);
      assert.ok(diff < 1e-6, `position ${t} moved by ${diff}`);
    }
  }
  let lastDiff = 0;
  for (let j = 0; j < V; j++) lastDiff = Math.max(lastDiff, Math.abs(base.data[3 * V + j] - changed.data[3 * V + j]));
  assert.ok(lastDiff > 1e-4, 'the last position should react to its own token');
});

test('GPT: backward reaches every parameter', () => {
  const model = new GPT(TINY_CONFIG);
  const logits = model.forward([[1, 2, 3, 4], [5, 6, 7, 8]]);
  const loss = crossEntropy(logits, [[2, 3, 4, 5], [6, 7, 8, 9]]);
  assert.ok(Number.isFinite(loss.item()));
  loss.backward();
  const names = paramNames(model);
  model.parameters().forEach((p, i) => {
    assert.ok(p.grad !== null, `${names[i]} got no gradient`);
    let sumSquares = 0;
    for (const g of p.grad) {
      assert.ok(Number.isFinite(g), `${names[i]} has a non-finite gradient`);
      sumSquares += g * g;
    }
    assert.ok(sumSquares > 0, `${names[i]} has an all-zero gradient`);
  });
});

test('GPT: toJSON/fromJSON round-trips to the same logits', () => {
  const model = new GPT(TINY_CONFIG);
  const json = JSON.parse(JSON.stringify(model.toJSON()));
  assert.deepEqual(json.config, TINY_CONFIG);
  assert.deepEqual(Object.keys(json.params), paramNames(model));

  const restored = GPT.fromJSON(json);
  const ids = [[1, 2, 3, 4, 5]];
  const before = model.forward(ids);
  const after = restored.forward(ids);
  // toJSON keeps 6 significant digits, so this is equality up to that rounding, not bit equality.
  assert.ok(ops.allClose(before, after, 1e-4));
  // A second round trip changes nothing at all.
  assert.deepEqual(restored.toJSON().params['lnF.gamma'].data, json.params['lnF.gamma'].data);

  delete json.params['lnF.beta'];
  assert.throws(() => GPT.fromJSON(json), /missing parameter/);
});

test('GPT: the lab checkpoint format stays under 2.5 MB', () => {
  const model = new GPT(LAB_CONFIG);
  const bytes = JSON.stringify(model.toJSON()).length;
  assert.ok(bytes < 2.5e6, `checkpoint would be ${(bytes / 1e6).toFixed(2)} MB`);
});

test('GPT.generate: appends tokens deterministically and crops to blockSize', () => {
  const model = new GPT(TINY_CONFIG);
  const prompt = [1, 2, 3];
  const a = model.generate(prompt, { maxNewTokens: 5, temperature: 1, next: rng(11) });
  const b = model.generate(prompt, { maxNewTokens: 5, temperature: 1, next: rng(11) });
  assert.equal(a.length, prompt.length + 5);
  assert.deepEqual(a, b);
  assert.deepEqual(a.slice(0, 3), prompt);
  for (const id of a) assert.ok(id >= 0 && id < TINY_CONFIG.vocabSize);
  // The prompt is not modified, and generating past blockSize keeps working (the context is cropped).
  assert.deepEqual(prompt, [1, 2, 3]);
  const long = model.generate(prompt, { maxNewTokens: TINY_CONFIG.blockSize + 4, next: rng(12) });
  assert.equal(long.length, prompt.length + TINY_CONFIG.blockSize + 4);

  // topK = 1 is greedy, so the same prompt always continues the same way.
  const greedyA = model.generate(prompt, { maxNewTokens: 4, topK: 1, next: rng(13) });
  const greedyB = model.generate(prompt, { maxNewTokens: 4, topK: 1, next: rng(14) });
  assert.deepEqual(greedyA, greedyB);
});

test('GPT: a few AdamW steps drive the loss down on one repeated batch', () => {
  const config = { vocabSize: 16, blockSize: 8, nLayer: 2, nHead: 2, nEmbd: 16, seed: 2 };
  const model = new GPT(config);
  const x = [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]];
  const y = [[2, 3, 4, 5, 6, 7, 8, 9], [7, 6, 5, 4, 3, 2, 1, 0]];
  const optimizer = new AdamW(model.parameters(), { lr: 3e-3, weightDecay: 0.01 });

  const first = crossEntropy(model.forward(x), y).item();
  let last = first;
  for (let step = 0; step < 40; step++) {
    optimizer.zeroGrad();
    const loss = crossEntropy(model.forward(x), y);
    loss.backward();
    clipGradNorm(model.parameters(), 1.0);
    optimizer.step();
    last = loss.item();
  }
  assert.ok(last < first - 0.5, `loss went from ${first} to ${last}`);
});
