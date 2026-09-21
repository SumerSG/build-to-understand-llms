import * as ops from 'lib/ops.js';
import { Tensor, crossEntropy } from 'lib/tensor.js';
import { GPT as RefGPT } from 'lib/gpt.js';
import { MultiHeadAttention } from 'lib/attention.js';

/** A Tensor of the given shape with entries uniform in [-1, 1), from a seeded rng. */
function randomInput(T, shape, seed) {
  const next = T.rng(seed);
  const t = Tensor.zeros(shape);
  for (let i = 0; i < t.data.length; i++) t.data[i] = next() * 2 - 1;
  return t;
}

/** Mean and standard deviation of a Float32Array. */
function stats(data) {
  let mean = 0;
  for (let i = 0; i < data.length; i++) mean += data[i];
  mean /= data.length;
  let variance = 0;
  for (let i = 0; i < data.length; i++) variance += (data[i] - mean) ** 2;
  return { mean, std: Math.sqrt(variance / data.length) };
}

/** Copy every parameter of a lib/gpt.js model into the learner's model, by parameters() order. */
function copyWeights(from, to, T) {
  const src = from.parameters(), dst = to.parameters();
  T.eq(dst.length, src.length, `parameters() must list ${src.length} tensors in the order wte, wpe, blocks[i] (ln1, attn.qkv, attn.proj, ln2, mlp.fc, mlp.proj), lnF`);
  for (let i = 0; i < src.length; i++) {
    T.eq(dst[i].shape, src[i].shape, `parameter ${i} has the wrong shape: the order must be wte, wpe, then per block ln1 (gamma, beta), attn (qkv weight, bias, proj weight, bias), ln2, mlp (fc weight, bias, proj weight, bias), then lnF`);
    dst[i].data.set(src[i].data);
  }
}

const SMALL = { vocabSize: 64, blockSize: 8, nLayer: 2, nHead: 2, nEmbd: 16, seed: 3 };

export const tests = [
  // ---------- step 1: Linear and Embedding ----------
  { step: 'layers', name: 'Linear computes x · W + b with W [nIn, nOut] and a zero bias', run(m, T) {
    const lin = new m.Linear(3, 2, { next: T.rng(1) });
    T.shape(lin.weight, [3, 2], 'weight must be [nIn, nOut] so that x [.., nIn] · W gives [.., nOut]');
    T.shape(lin.bias, [2]);
    T.close(T.arr(lin.bias), [0, 0], 1e-9, 'the bias starts at zero (GPT-2 convention)');
    const x = Tensor.from([[1, 2, 3], [0, -1, 0.5]]);
    const expected = ops.matmul(x, lin.weight);
    T.close(lin.forward(x), expected, 1e-6, 'with a zero bias the output must equal the raw matmul');
    lin.bias.data.set([0.5, -0.25]);
    T.close(lin.forward(x), ops.add(expected, lin.bias), 1e-6, 'the bias is added to every row');
    const noBias = new m.Linear(3, 2, { bias: false, next: T.rng(1) });
    T.eq(noBias.bias, null, 'bias: false must store null (Llama-style layers have no biases)');
    T.eq(noBias.parameters().length, 1, 'parameters() lists only the weight when there is no bias');
    T.eq(lin.parameters().length, 2, 'parameters() lists weight and bias');
    T.ok(lin.parameters()[0] === lin.weight && lin.parameters()[1] === lin.bias, 'parameters() order: weight, then bias');
  } },
  { step: 'layers', name: 'Linear initialises with std 0.02 and works on [B,T,C] inputs', run(m, T) {
    const lin = new m.Linear(64, 96, { next: T.rng(2) });
    const { mean, std } = stats(lin.weight.data);
    T.ok(Math.abs(mean) < 0.005, `weights must be centred on zero (mean ${mean.toFixed(4)})`);
    T.ok(Math.abs(std - 0.02) < 0.003, `weight std should be 0.02, got ${std.toFixed(4)}: with std 1 the residual stream explodes after a few blocks`);
    const custom = new m.Linear(64, 96, { next: T.rng(2), std: 0.1 });
    T.ok(Math.abs(stats(custom.weight.data).std - 0.1) < 0.01, 'the std option must be honoured');
    const x = randomInput(T, [2, 5, 64], 4);
    const y = lin.forward(x);
    T.shape(y, [2, 5, 96], 'a 2-D weight is shared across every leading dimension: [B,T,nIn] -> [B,T,nOut]');
    T.close(y, ops.matmul(x, lin.weight), 1e-6);
  } },
  { step: 'layers', name: 'Embedding picks rows, keeps the id shape, and its table is a trainable leaf', run(m, T) {
    const emb = new m.Embedding(5, 4, { next: T.rng(3) });
    T.shape(emb.weight, [5, 4], 'the table is [n, d]');
    T.eq(emb.n, 5); T.eq(emb.d, 4);
    const out = emb.forward([[1, 3], [0, 0]]);
    T.shape(out, [2, 2, 4], 'ids B×T -> [B,T,d]');
    const rows = T.arr(emb.weight);
    T.close(T.arr(out), [[rows[1], rows[3]], [rows[0], rows[0]]], 1e-7, 'each output vector must be the row of the table at that id');
    T.eq(emb.parameters().length, 1, 'an embedding has one parameter tensor and no bias');
    T.ok(emb.weight.requiresGrad === true, 'the table must be created with Tensor.param so module 07 can train it');
    const big = new m.Embedding(256, 64, { next: T.rng(5) });
    T.ok(Math.abs(stats(big.weight.data).std - 0.02) < 0.003, 'the table is initialised with std 0.02, like Linear');
    // Gradients must reach the table: a raw ops.embed would silently break training.
    emb.forward([[2, 2]]).sum().backward();
    T.ok(emb.weight.grad !== null, 'backward() must reach the table: use Tensor methods, not raw ops');
    T.close(emb.weight.grad[2 * 4], 2, 1e-6, 'row 2 was used twice, so its gradient is the sum of both uses');
  } },

  // ---------- step 2: token + position embeddings ----------
  { step: 'embed', name: 'embedInputs adds wte[ids] and wpe[position] with the position table broadcast over the batch', run(m, T) {
    const next = T.rng(7);
    const wte = new m.Embedding(10, 4, { next });
    const wpe = new m.Embedding(6, 4, { next });
    const ids = [[3, 1, 4], [1, 5, 9]];
    const out = m.embedInputs(wte, wpe, ids);
    T.shape(out, [2, 3, 4], 'B×T ids -> [B,T,C]');
    const expected = ops.add(ops.embed(wte.weight, ids), ops.embed(wpe.weight, [0, 1, 2]));
    T.close(out, expected, 1e-6, 'every position t gets wte[id] + wpe[t]; the same wpe rows are used for both batch rows');
  } },
  { step: 'embed', name: 'the same token at two positions gets two different vectors (that difference is the position)', run(m, T) {
    const next = T.rng(8);
    const wte = new m.Embedding(10, 4, { next });
    const wpe = new m.Embedding(6, 4, { next });
    const out = T.arr(m.embedInputs(wte, wpe, [[7, 2, 7]]))[0];
    const rows = T.arr(wpe.weight);
    const diff = out[0].map((v, j) => v - out[2][j]);
    const expectedDiff = rows[0].map((v, j) => v - rows[2][j]);
    T.close(diff, expectedDiff, 1e-6, 'token 7 at position 0 minus token 7 at position 2 must equal wpe[0] - wpe[2]; without position embeddings attention could not tell them apart');
    T.ok(diff.some((v) => Math.abs(v) > 1e-6), 'the two occurrences of token 7 must differ');
  } },
  { step: 'embed', name: 'a sequence longer than the block size throws, and gradients reach both tables', run(m, T) {
    const next = T.rng(9);
    const wte = new m.Embedding(10, 4, { next });
    const wpe = new m.Embedding(3, 4, { next });
    T.throws(() => m.embedInputs(wte, wpe, [[1, 2, 3, 4]]), 'T=4 exceeds blockSize=3: there is no row for position 3, so this must throw rather than read garbage');
    m.embedInputs(wte, wpe, [[1, 2, 3]]); // exactly blockSize is fine
    m.embedInputs(wte, wpe, [[1, 2, 3]]).sum().backward();
    T.ok(wte.weight.grad !== null && wpe.weight.grad !== null, 'both tables must receive gradients');
    T.close(wpe.weight.grad.subarray(0, 4), [1, 1, 1, 1], 1e-6, 'position 0 was used once with a sum() loss, so its gradient row is all ones');
  } },

  // ---------- step 3: MLP and Block ----------
  { step: 'block', name: 'MLP is fc (C -> 4C) → GELU → proj (4C -> C)', run(m, T) {
    const mlp = new m.MLP(8, { next: T.rng(10) });
    T.shape(mlp.fc.weight, [8, 32], 'the hidden layer is 4× wider than the residual stream');
    T.shape(mlp.proj.weight, [32, 8]);
    const x = randomInput(T, [2, 3, 8], 11);
    for (let i = 0; i < x.data.length; i++) x.data[i] *= 40; // large inputs so GELU and ReLU disagree clearly
    const hidden = ops.add(ops.matmul(x, mlp.fc.weight), mlp.fc.bias);
    const expected = ops.add(ops.matmul(ops.gelu(hidden), mlp.proj.weight), mlp.proj.bias);
    const y = mlp.forward(x);
    T.shape(y, [2, 3, 8]);
    T.close(y, expected, 1e-5, 'must be proj(gelu(fc(x))): GELU, not ReLU and not tanh, is what GPT-2 uses');
    T.eq(mlp.parameters().length, 4, 'fc.weight, fc.bias, proj.weight, proj.bias');
    let count = 0; for (const p of mlp.parameters()) count += p.size;
    T.eq(count, 8 * 64 + 5 * 8, 'an MLP holds 8C² weights plus 5C biases');
  } },
  { step: 'block', name: 'Block is pre-LN: with zeroed output projections it is exactly the identity', run(m, T) {
    const block = new m.Block({ nEmbd: 8, nHead: 2 }, { next: T.rng(12) });
    T.ok(block.attn instanceof MultiHeadAttention, 'attn must be the MultiHeadAttention from lib/attention.js');
    T.eq(block.attn.nHead, 2, 'the block must pass cfg.nHead to the attention layer');
    block.attn.proj.weight.data.fill(0);
    block.mlp.proj.weight.data.fill(0);
    const x = randomInput(T, [1, 4, 8], 13);
    for (let i = 0; i < x.data.length; i++) x.data[i] = x.data[i] * 3 + 2; // mean 2, std ~1.7: LayerNorm would change it
    T.close(block.forward(x), x, 1e-6, 'when both branches output zero the block must return x unchanged: post-LN (LayerNorm on the sum) would normalise x instead, and a missing residual would return 0');
  } },
  { step: 'block', name: 'Block computes x + attn(ln1(x)), then + mlp(ln2(·)), and lists its parameters in order', run(m, T) {
    const block = new m.Block({ nEmbd: 16, nHead: 4 }, { next: T.rng(14) });
    const x = randomInput(T, [2, 5, 16], 15);
    const h = x.add(block.attn.forward(block.ln1.forward(x)));
    const expected = h.add(block.mlp.forward(block.ln2.forward(h)));
    T.close(block.forward(x), expected, 1e-5, 'the attention branch reads ln1(x) and the MLP branch reads ln2 of the UPDATED stream, not of the original x');
    const params = block.parameters();
    T.eq(params.length, 12, 'ln1 (2) + attn (4) + ln2 (2) + mlp (4) = 12 tensors');
    T.ok(params[0] === block.ln1.gamma && params[2] === block.attn.qkv.weight && params[6] === block.ln2.gamma && params[8] === block.mlp.fc.weight, 'order: ln1, attn, ln2, mlp (what the checkpoints and lib/gpt.js paramNames() expect)');
    let count = 0; for (const p of params) count += p.size;
    T.eq(count, 12 * 256 + 13 * 16, 'a block holds 12C² weights + 13C biases and LayerNorm gains/shifts');
  } },

  // ---------- step 4: the GPT ----------
  { step: 'gpt', name: 'forward maps B×T ids to logits [B,T,V]; wte, wpe, blocks and lnF have the configured shapes', run(m, T) {
    const model = new m.GPT(SMALL);
    T.eq(model.config.nLayer, 2);
    T.shape(model.wte.weight, [64, 16], 'wte is [vocabSize, nEmbd]');
    T.shape(model.wpe.weight, [8, 16], 'wpe is [blockSize, nEmbd]');
    T.eq(model.blocks.length, 2, 'one Block per layer');
    T.shape(model.lnF.gamma, [16]);
    const logits = model.forward([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]);
    T.shape(logits, [2, 5, 64], 'one row of vocabSize scores per position');
    T.ok(Number.isFinite(logits.data[0]), 'logits must be finite');
    T.throws(() => model.forward([[1, 2, 3, 4, 5, 6, 7, 8, 9]]), 'T=9 > blockSize=8 must throw');
  } },
  { step: 'gpt', name: 'with the reference weights copied in, logits match lib/gpt.js (tied head, final LayerNorm, block order)', run(m, T) {
    const ref = new RefGPT(SMALL);
    const mine = new m.GPT({ ...SMALL, seed: 99 });
    copyWeights(ref, mine, T);
    const ids = [[5, 3, 3, 8, 1, 0], [2, 2, 2, 2, 2, 2]];
    T.close(mine.forward(ids), ref.forward(ids), 1e-4, 'same weights must give the same logits: check lnF is applied before the head and that logits = x · wteᵀ (the transpose of the token table)');
    // The tied head means the token table is BOTH the input lookup and the output classifier.
    T.eq(mine.parameters().length, 2 + 12 * 2 + 2, 'no separate LM head: parameters() has wte, wpe, 12 per block, lnF');
    T.eq(mine.numParams(), 64 * 16 + 8 * 16 + 2 * (12 * 256 + 13 * 16) + 2 * 16, 'numParams must equal V·C + T·C + L·(12C² + 13C) + 2C');
  } },
  { step: 'gpt', name: 'the stack is causal and every parameter receives a gradient from the cross-entropy loss', run(m, T) {
    const model = new m.GPT({ ...SMALL, seed: 5 });
    const a = model.forward([[1, 2, 3, 4, 5, 6]]);
    const b = model.forward([[1, 2, 3, 9, 5, 6]]);
    const V = 64;
    T.close(a.data.subarray(0, 3 * V), b.data.subarray(0, 3 * V), 1e-6, 'changing token 3 must not change the logits at positions 0–2: the stack must stay causal');
    let changed = 0;
    for (let i = 3 * V; i < 4 * V; i++) if (Math.abs(a.data[i] - b.data[i]) > 1e-6) changed++;
    T.ok(changed > 0, 'changing token 3 must change the logits at position 3');
    const loss = crossEntropy(model.forward([[1, 2, 3, 4], [4, 3, 2, 1]]), [[2, 3, 4, 5], [3, 2, 1, 0]]);
    T.ok(Number.isFinite(loss.item()), 'loss must be finite');
    T.ok(Math.abs(loss.item() - Math.log(64)) < 0.5, `at init the loss should be near ln(V) = ${Math.log(64).toFixed(3)} (got ${loss.item().toFixed(3)}); std 0.02 init keeps the logits small`);
    loss.backward();
    const params = model.parameters();
    for (let i = 0; i < params.length; i++) {
      const g = params[i].grad;
      T.ok(g !== null, `parameter ${i} has no gradient: every tensor listed by parameters() must be on the graph`);
      let nonZero = false; for (let j = 0; j < g.length; j++) if (g[j] !== 0) { nonZero = true; break; }
      T.ok(nonZero, `parameter ${i} has an all-zero gradient: it is not connected to the loss`);
    }
    const twin = new m.GPT({ ...SMALL, seed: 5 });
    T.close(twin.forward([[1, 2, 3]]), model.forward([[1, 2, 3]]), 1e-7, 'the same seed must build the same model');
    const other = new m.GPT({ ...SMALL, seed: 6 });
    let differs = false;
    const x = other.forward([[1, 2, 3]]), y = model.forward([[1, 2, 3]]);
    for (let i = 0; i < x.data.length; i++) if (Math.abs(x.data[i] - y.data[i]) > 1e-6) { differs = true; break; }
    T.ok(differs, 'a different seed must build a different model');
  } },

  // ---------- step 5: the parameter and compute budget ----------
  { step: 'count', name: 'countParams(config) equals the built model\'s numParams() for three configs', run(m, T) {
    const configs = [
      { vocabSize: 64, blockSize: 8, nLayer: 2, nHead: 2, nEmbd: 16 },
      { vocabSize: 256, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 64 },   // the lab checkpoint
      { vocabSize: 33, blockSize: 5, nLayer: 3, nHead: 1, nEmbd: 12 },    // odd sizes catch a 12C² that forgot the 13C
    ];
    for (const cfg of configs) {
      const model = new RefGPT({ ...cfg, seed: 0 });
      T.eq(m.countParams(cfg), model.numParams(), `countParams(${JSON.stringify(cfg)}) must equal the built model exactly: V·C + T·C + L·(12C² + 13C) + 2C`);
    }
  } },
  { step: 'count', name: 'GPT-2 small comes out at 124,439,808 and the breakdown sums to the total', run(m, T) {
    const gpt2 = { vocabSize: 50257, blockSize: 1024, nLayer: 12, nHead: 12, nEmbd: 768 };
    T.eq(m.countParams(gpt2), 124439808, 'this is the published GPT-2 (124M) parameter count; the 12·L·C² approximation alone gives 84,934,656');
    const b = m.paramBreakdown(gpt2);
    T.eq(b.tokenEmbedding, 50257 * 768, 'tokenEmbedding = V·C');
    T.eq(b.positionEmbedding, 1024 * 768, 'positionEmbedding = T·C');
    T.eq(b.attention, 12 * (4 * 768 * 768 + 4 * 768), 'attention = L·(4C² + 4C): qkv 3C² + proj C² plus their biases');
    T.eq(b.mlp, 12 * (8 * 768 * 768 + 5 * 768), 'mlp = L·(8C² + 5C): fc 4C² + proj 4C² plus their biases');
    T.eq(b.layerNorm, 12 * 4 * 768 + 2 * 768, 'layerNorm = 2C per LayerNorm, two per block plus the final one');
    T.eq(b.tokenEmbedding + b.positionEmbedding + b.attention + b.mlp + b.layerNorm, b.total, 'the components must sum to total');
    T.eq(b.total, m.countParams(gpt2));
  } },
  { step: 'count', name: 'flopsPerToken is 2·params plus an attention term that grows linearly with the context', run(m, T) {
    const lab = { vocabSize: 256, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 64 };
    const params = m.countParams(lab);
    T.eq(m.flopsPerToken(lab, 64), 2 * params + 4 * 2 * 64 * 64, '2 FLOPs per parameter (multiply + add) plus 4·L·T·C for q·kᵀ and weights·v');
    T.eq(m.flopsPerToken(lab, 64) - m.flopsPerToken(lab, 32), 4 * 2 * 32 * 64, 'only the attention term depends on the context length');
    T.eq(m.flopsPerToken(lab, 1), 2 * params + 4 * 2 * 64, 'with one token in context the cost is almost exactly 2·params');
  } },
];
