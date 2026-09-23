import { Tensor, noGrad } from 'lib/tensor.js';
import { GPT, Linear, paramNames } from 'lib/gpt.js';
import { AdamW, clipGradNorm } from 'lib/optim.js';
import { rng } from 'lib/util.js';

const SMALL = { vocabSize: 16, blockSize: 8, nLayer: 2, nHead: 2, nEmbd: 8, seed: 1 };
const LAB = { vocabSize: 256, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 64, seed: 0 };

/** A Tensor of the given shape filled with uniform values in [-1, 1). */
function randTensor(shape, seed, scale = 1) {
  const next = rng(seed);
  let n = 1;
  for (const d of shape) n *= d;
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = (next() * 2 - 1) * scale;
  return new Tensor({ shape: shape.slice(), data });
}

/** A Linear with non-zero weight and bias so that mistakes in either show up. */
function makeBase(nIn, nOut, seed) {
  const base = new Linear(nIn, nOut, { next: rng(seed), std: 0.5 });
  const next = rng(seed + 100);
  for (let i = 0; i < base.bias.data.length; i++) base.bias.data[i] = next() - 0.5;
  return base;
}

/** Reference: x·W + b + s·(x·A)·B with plain loops over the last axis. */
function reference(x, W, b, A, B, s) {
  const [nIn, nOut] = W.shape;
  const r = A.shape[1];
  const rows = x.data.length / nIn;
  const out = new Float64Array(rows * nOut);
  for (let i = 0; i < rows; i++) {
    const h = new Float64Array(r);
    for (let k = 0; k < nIn; k++) for (let q = 0; q < r; q++) h[q] += x.data[i * nIn + k] * A.data[k * r + q];
    for (let j = 0; j < nOut; j++) {
      let v = b ? b.data[j] : 0;
      for (let k = 0; k < nIn; k++) v += x.data[i * nIn + k] * W.data[k * nOut + j];
      for (let q = 0; q < r; q++) v += s * h[q] * B.data[q * nOut + j];
      out[i * nOut + j] = v;
    }
  }
  return Array.from(out);
}

function randomizeB(layer, seed, scale = 0.3) {
  const next = rng(seed);
  for (let i = 0; i < layer.B.data.length; i++) layer.B.data[i] = (next() * 2 - 1) * scale;
}

function randomIds(next, B, T, V) {
  const ids = [];
  for (let b = 0; b < B; b++) { const row = []; for (let t = 0; t < T; t++) row.push(Math.floor(next() * V)); ids.push(row); }
  return ids;
}

/** Tiny synthetic SFT set: prompt tokens (mask 0) followed by a response that is a fixed function of them. */
function syntheticExamples(n, seed) {
  const next = rng(seed);
  const out = [];
  for (let i = 0; i < n; i++) {
    const a = 1 + Math.floor(next() * 6), b = 1 + Math.floor(next() * 6);
    out.push({ ids: [a, b, 0, a + 8, 15], mask: [0, 0, 0, 1, 1] });
  }
  return out;
}

/**
 * A fresh GPT with a large embedding table. The LM head is tied to wte, so with the default std-0.02
 * table (and wte frozen) no adapter could move the logits much; a trained checkpoint's table is large.
 */
function pretrainedLike(seed) {
  const model = new GPT(SMALL);
  const next = rng(seed);
  const w = model.wte.weight.data;
  for (let i = 0; i < w.length; i++) w[i] = next() * 2 - 1;
  return model;
}

const mean = (xs) => xs.reduce((s, v) => s + v, 0) / xs.length;

export const tests = [
  // ---------- step 1: forward ----------
  { step: 'forward', name: 'A is [nIn, r] and random, B is [r, nOut] and zero, scaling is alpha / r', run(m, T) {
    const layer = new m.LoRALinear(makeBase(6, 10, 1), { rank: 3, alpha: 6, next: rng(2) });
    T.ok(layer.A && layer.B, 'the constructor must create this.A and this.B (step 1)');
    T.eq(layer.A.shape, [6, 3], 'A projects the input down: shape [nIn, rank] in the lab\'s [nIn, nOut] layout');
    T.eq(layer.B.shape, [3, 10], 'B projects back up: shape [rank, nOut]');
    T.ok(Array.from(layer.B.data).every((v) => v === 0), 'B must start at exactly zero so the wrapped layer starts equal to the base layer');
    const nonzero = Array.from(layer.A.data).filter((v) => v !== 0).length;
    T.ok(nonzero === 18 && Array.from(layer.A.data).every(Number.isFinite), 'A must be random (Gaussian): if A and B were both zero, neither would ever receive a gradient');
    T.ok(layer.A.requiresGrad && layer.B.requiresGrad, 'A and B are trainable leaves: create them with Tensor.param');
    T.close(layer.scaling, 2, 1e-12, 'scaling must be alpha / rank = 6 / 3 = 2');
    // A constant A (e.g. every entry 1/sqrt(nIn)) has identical columns: the adapter would be rank 1 forever.
    const big = new m.LoRALinear(makeBase(64, 8, 50), { rank: 32, alpha: 32, next: rng(51) });
    const a = Array.from(big.A.data);
    const mu = mean(a), sd = Math.sqrt(mean(a.map((v) => (v - mu) ** 2)));
    T.close(sd, 1 / 8, 0.015, `A's entries must have standard deviation 1 / sqrt(nIn) = 0.125 for nIn = 64 (ops.randn(shape, next, std)); got ${sd.toFixed(4)}. A constant A makes every column of x·A identical, so the adapter can never exceed rank 1`);
    T.ok(Math.abs(mu) < 0.02, `A must be zero-mean Gaussian noise; its entries average ${mu.toFixed(4)}`);
  } },
  { step: 'forward', name: 'with B = 0 the wrapped layer equals the base layer exactly', run(m, T) {
    const base = makeBase(6, 10, 3);
    const layer = new m.LoRALinear(base, { rank: 4, alpha: 8, next: rng(4) });
    const x = randTensor([5, 6], 5);
    const y = layer.forward(x);
    T.shape(y, [5, 10], 'forward maps [..., nIn] to [..., nOut]');
    T.close(Array.from(y.data), Array.from(base.forward(x).data), 1e-6, 'at initialisation the adapter contributes (alpha/r)·(x·A)·0 = 0');
  } },
  { step: 'forward', name: 'forward adds (alpha/r)·(x·A)·B on top of x·W + b', run(m, T) {
    const base = makeBase(6, 10, 6);
    const layer = new m.LoRALinear(base, { rank: 3, alpha: 12, next: rng(7) });
    T.ok(layer.B, 'the constructor must create this.B (step 1)');
    randomizeB(layer, 8, 0.5);
    const x = randTensor([4, 6], 9);
    const expect = reference(x, base.weight, base.bias, layer.A, layer.B, 12 / 3);
    const unscaled = reference(x, base.weight, base.bias, layer.A, layer.B, 1);
    const got = Array.from(layer.forward(x).data);
    const offUnscaled = Math.max(...got.map((v, i) => Math.abs(v - unscaled[i])));
    T.ok(offUnscaled > 1e-3, 'your output matches the UNSCALED update x·A·B; multiply the adapter path by alpha / rank (here 4)');
    T.close(got, expect, 1e-4, 'expected x·W + b + (alpha/r)·(x·A)·B with alpha/r = 4');
  } },
  { step: 'forward', name: 'forward computes (x·A)·B and never builds the [nIn, nOut] product A·B', run(m, T) {
    const layer = new m.LoRALinear(makeBase(6, 10, 52), { rank: 3, alpha: 3, next: rng(53) });
    T.ok(layer.A && layer.B, 'the constructor must create this.A and this.B (step 1)');
    const x = randTensor([4, 6], 54);
    const original = Tensor.prototype.matmul;
    const lefts = [];
    Tensor.prototype.matmul = function (o) { lefts.push(this); return original.call(this, o); };
    try { layer.forward(x); } finally { Tensor.prototype.matmul = original; }
    T.ok(!lefts.includes(layer.A), 'forward multiplied A by B: that materialises a full [nIn, nOut] matrix on every call (nIn·r·nOut multiply-adds, the whole cost LoRA avoids). Compute x.matmul(this.A) first, then multiply that by this.B');
    T.ok(lefts.includes(x), 'forward must multiply the input x itself by W (through base.forward) and by A');
  } },
  { step: 'forward', name: 'forward works on [B, T, nIn] activations and keeps the leading dims', run(m, T) {
    const base = makeBase(8, 5, 10);
    const layer = new m.LoRALinear(base, { rank: 2, alpha: 2, next: rng(11) });
    T.ok(layer.B, 'the constructor must create this.B (step 1)');
    randomizeB(layer, 12, 0.5);
    const x = randTensor([2, 3, 8], 13);
    const y = layer.forward(x);
    T.shape(y, [2, 3, 5], 'a [B,T,C] input must give a [B,T,nOut] output, as the GPT feeds it');
    T.close(Array.from(y.data), reference(x, base.weight, base.bias, layer.A, layer.B, 1), 1e-4, 'expected x·W + b + (x·A)·B for alpha = rank');
  } },

  // ---------- step 2: freeze ----------
  { step: 'freeze', name: 'parameters() is exactly [A, B] and the base weight and bias are frozen', run(m, T) {
    const base = makeBase(6, 4, 14);
    const layer = new m.LoRALinear(base, { rank: 2, alpha: 2, next: rng(15) });
    const params = layer.parameters();
    T.eq(params.length, 2, 'only the two adapter matrices are trainable; returning the base tensors is full fine-tuning');
    T.ok(params[0] === layer.A && params[1] === layer.B, 'parameters() must return [this.A, this.B] (the same Tensor objects, so the optimizer updates them)');
    T.ok(base.weight.requiresGrad === false, 'the base weight must have requiresGrad = false so autograd never computes or stores its gradient');
    T.ok(base.bias.requiresGrad === false, 'freeze the bias too (LoRA\'s default bias="none"): otherwise it trains and must be shipped with the adapter');
  } },
  { step: 'freeze', name: 'after backward, W has no gradient, B has one, and A\'s is zero (because B = 0)', run(m, T) {
    const base = makeBase(6, 4, 16);
    const layer = new m.LoRALinear(base, { rank: 2, alpha: 2, next: rng(17) });
    const x = randTensor([3, 6], 18);
    layer.forward(x).mul(randTensor([3, 4], 19)).sum().backward();
    T.ok(base.weight.grad === null, 'a frozen weight must not receive a gradient: set base.weight.requiresGrad = false in the constructor');
    T.ok(layer.B.grad && Array.from(layer.B.grad).some((v) => v !== 0), 'B must receive a gradient: dL/dB = s·(x·A)ᵀ·g is non-zero because A is random');
    T.ok(!layer.A.grad || Array.from(layer.A.grad).every((v) => v === 0), 'at initialisation dL/dA = s·xᵀ·(g·Bᵀ) must be exactly zero because B = 0');
  } },
  { step: 'freeze', name: 'three AdamW steps move A and B and leave W and b bit-for-bit unchanged', run(m, T) {
    const base = makeBase(6, 4, 20);
    const W0 = Array.from(base.weight.data), b0 = Array.from(base.bias.data);
    const layer = new m.LoRALinear(base, { rank: 2, alpha: 2, next: rng(21) });
    const A0 = Array.from(layer.A.data);
    const opt = new AdamW(layer.parameters(), { lr: 0.05 });
    const x = randTensor([3, 6], 22), target = randTensor([3, 4], 23);
    for (let s = 0; s < 3; s++) {
      layer.forward(x).sub(target).pow(2).mean().backward();
      opt.step();
      opt.zeroGrad();
    }
    T.eq(Array.from(base.weight.data), W0, 'the base weight changed: it must be frozen and absent from parameters()');
    T.eq(Array.from(base.bias.data), b0, 'the base bias changed: it must be frozen and absent from parameters()');
    T.ok(Array.from(layer.B.data).some((v) => v !== 0), 'B did not move: it must be in parameters() with requiresGrad');
    T.ok(Array.from(layer.A.data).some((v, i) => v !== A0[i]), 'A did not move after 3 steps: once B is non-zero, A receives gradient too');
  } },

  // ---------- step 3: apply ----------
  { step: 'apply', name: 'applyLora wraps exactly the targeted Linears and leaves the model\'s output unchanged', run(m, T) {
    const model = new GPT(SMALL);
    const ids = randomIds(rng(24), 2, 6, SMALL.vocabSize);
    const before = noGrad(() => Array.from(model.forward(ids).data));
    const wrapped = m.applyLora(model, { rank: 2, alpha: 4, targets: ['attn.qkv', 'attn.proj'], next: rng(25) });
    T.ok(Array.isArray(wrapped), 'applyLora returns [{ name, layer }] for every wrapped Linear');
    T.eq(wrapped.map((w) => w.name), ['blocks.0.attn.qkv', 'blocks.0.attn.proj', 'blocks.1.attn.qkv', 'blocks.1.attn.proj'],
      'with targets [attn.qkv, attn.proj] and 2 layers, four Linears are wrapped, in paramNames order');
    for (const b of model.blocks) {
      T.ok(b.attn.qkv instanceof m.LoRALinear && b.attn.proj instanceof m.LoRALinear, 'the model must now hold LoRALinear objects at the targeted paths (use setModule)');
      T.ok(!(b.mlp.fc instanceof m.LoRALinear) && !(b.mlp.proj instanceof m.LoRALinear), 'the MLP was not targeted and must stay a plain Linear');
    }
    T.eq(paramNames(model).length, 2 + SMALL.nLayer * 12 + 2, 'paramNames still lists every base tensor (LoRALinear exposes .weight and .bias)');
    const after = noGrad(() => Array.from(model.forward(ids).data));
    for (const { layer } of wrapped) {
      T.ok(layer instanceof m.LoRALinear && layer.rank === 2, 'applyLora must pass `rank` through to every LoRALinear');
      T.close(layer.scaling, 2, 1e-12, 'applyLora must pass `alpha` through: with alpha 4 and rank 2 every adapter\'s scaling is 2 (dropping alpha silently falls back to alpha = rank)');
    }
    T.close(after, before, 1e-5, 'B = 0, so wrapping must not change the logits');
  } },
  { step: 'apply', name: 'after applyLora only the adapters are trainable (embeddings and LayerNorms frozen too)', run(m, T) {
    const model = new GPT(SMALL);
    m.applyLora(model, { rank: 2, alpha: 4, targets: ['attn.qkv', 'mlp.proj'], next: rng(26) });
    T.ok(model.parameters().every((p) => p.requiresGrad === false), 'every base tensor (wte, wpe, LayerNorm gamma/beta, un-targeted Linears) must be frozen');
    const trainable = m.trainableParameters(model);
    T.eq(trainable.length, 2 * 2 * SMALL.nLayer, 'two targets × two layers × (A, B) = 8 trainable tensors');
    const adapters = m.loraLayers(model).flatMap(({ layer }) => [layer.A, layer.B]);
    T.ok(trainable.every((p) => adapters.includes(p)), 'trainableParameters must return the adapters\' A and B tensors themselves');
  } },
  { step: 'apply', name: 'countParams: r = 4 on all linear layers of the lab config is 8,192 of 128,768 (about 6.4%)', run(m, T) {
    const full = new GPT(LAB);
    const c0 = m.countParams(full);
    T.eq([c0.trainable, c0.total], [120576, 120576], 'an un-adapted model is 100% trainable: that is full fine-tuning');
    const model = new GPT(LAB);
    m.applyLora(model, { rank: 4, alpha: 8, next: rng(27) });
    const c = m.countParams(model);
    T.eq(c.trainable, 8192, 'per layer: qkv 4·(64+192) + proj 4·(64+64) + fc 4·(64+256) + proj 4·(256+64) = 4,096; two layers = 8,192');
    T.eq(c.total, 128768, 'total = 120,576 frozen base parameters + 8,192 adapter parameters');
    T.close(c.percent, (100 * 8192) / 128768, 1e-6, 'percent = 100 · trainable / total');
  } },

  // ---------- step 4: merge ----------
  { step: 'merge', name: 'merge() returns a plain Linear with W + (alpha/r)·A·B and does not modify the base', run(m, T) {
    const base = makeBase(6, 5, 28);
    const W0 = Array.from(base.weight.data);
    const layer = new m.LoRALinear(base, { rank: 2, alpha: 6, next: rng(29) });
    randomizeB(layer, 30, 0.5);
    const merged = layer.merge();
    T.ok(merged instanceof Linear && !(merged instanceof m.LoRALinear), 'merge() must return a plain Linear (no adapter left)');
    T.ok(merged.weight !== base.weight, 'build a new weight tensor; keeping the base intact lets you unmerge or swap adapters');
    T.eq(Array.from(base.weight.data), W0, 'merge() must not modify the base weight in place');
    const expect = [];
    for (let i = 0; i < 6; i++) for (let j = 0; j < 5; j++) {
      let d = 0;
      for (let q = 0; q < 2; q++) d += layer.A.data[i * 2 + q] * layer.B.data[q * 5 + j];
      expect.push(W0[i * 5 + j] + 3 * d);
    }
    T.close(Array.from(merged.weight.data), expect, 1e-5, 'merged weight must be W + (alpha/r)·A·B with alpha/r = 3 ([nIn, r]·[r, nOut] = [nIn, nOut], no transpose)');
    T.close(Array.from(merged.bias.data), Array.from(base.bias.data), 0, 'the bias carries over unchanged');
    T.ok(merged.bias !== base.bias, 'copy the bias into the new Linear\'s own bias tensor (.data.set): sharing the frozen base tensor leaves the merged layer with a bias that has requiresGrad = false');
  } },
  { step: 'merge', name: 'mergeLora leaves the logits unchanged (within 1e-5) and removes every adapter', run(m, T) {
    const model = new GPT(SMALL);
    const n0 = model.numParams();
    const wrapped = m.applyLora(model, { rank: 2, alpha: 8, next: rng(31) });
    wrapped.forEach(({ layer }, k) => randomizeB(layer, 40 + k, 0.5));
    const ids = randomIds(rng(32), 2, 7, SMALL.vocabSize);
    const before = noGrad(() => Array.from(model.forward(ids).data));
    const count = m.mergeLora(model);
    T.eq(count, 4 * SMALL.nLayer, 'mergeLora returns how many adapters it merged (4 linears × 2 layers)');
    T.eq(m.loraLayers(model).length, 0, 'after merging no LoRALinear may remain in the model');
    T.eq(model.numParams(), n0, 'the merged model has exactly the base parameter count: the adapter costs nothing at inference');
    const after = noGrad(() => Array.from(model.forward(ids).data));
    T.close(after, before, 1e-5, 'x·(W + s·A·B) must equal x·W + s·(x·A)·B: logits must match within 1e-5');
  } },

  // ---------- step 5: finetune ----------
  { step: 'finetune', name: 'trainingMemory: 2 bytes per frozen parameter, 16 per trainable one', run(m, T) {
    const full = m.trainingMemory({ total: 120576, trainable: 120576 });
    T.ok(full && full.total === 16 * 120576, 'full fine-tuning of the lab model: 2 (bf16 weight) + 2 (bf16 grad) + 12 (fp32 master, m, v) = 16 bytes per parameter');
    T.eq([full.weights, full.grads, full.optimizer], [2 * 120576, 2 * 120576, 12 * 120576], 'expected { weights: 2·total, grads: 2·trainable, optimizer: 12·trainable }');
    // Llama-2-7B, r = 8 on all seven linear projections: ≈ 20.0M adapter parameters.
    const lora = m.trainingMemory({ total: 6.74e9 + 19988480, trainable: 19988480 });
    T.close(lora.total / 1e9, (2 * (6.74e9 + 19988480) + 14 * 19988480) / 1e9, 1e-6, 'LoRA on a 7B model: frozen weights at 2 bytes plus 16 bytes per adapter parameter ≈ 13.8 GB, against ≈ 108 GB for full fine-tuning');
    T.ok(lora.optimizer === 12 * 19988480, 'optimizer state exists only for trainable parameters');
  } },
  { step: 'finetune', name: 'LoRA fine-tuning lowers the loss, leaves every base tensor untouched, and keeps AdamW state only for the adapters', async run(m, T) {
    const model = pretrainedLike(9);
    const snapshot = model.parameters().map((p) => Array.from(p.data));
    m.applyLora(model, { rank: 2, alpha: 4, next: rng(33) });
    const examples = syntheticExamples(24, 34);
    const res = await m.finetune(model, examples, { steps: 40, lr: 1e-2, batchSize: 4, next: rng(35) });
    T.ok(res && Array.isArray(res.losses) && res.losses.length === 40, 'finetune returns { losses, optimizer } with one loss per step');
    T.ok(mean(res.losses.slice(-5)) < mean(res.losses.slice(0, 5)) - 1, `loss should fall by more than 1 nat in 40 steps; went ${mean(res.losses.slice(0, 5)).toFixed(3)} -> ${mean(res.losses.slice(-5)).toFixed(3)}. Does the optimizer hold the adapters? model.parameters() does not include A and B; trainableParameters(model) does`);
    model.parameters().forEach((p, k) => T.eq(Array.from(p.data), snapshot[k], `base tensor ${paramNames(model)[k]} changed: only the adapters may train`));
    const { trainable } = m.countParams(model);
    const stateSize = res.optimizer.m.reduce((s, a) => s + a.length, 0);
    T.eq(stateSize, trainable, `AdamW must be built on trainableParameters(model) only: its first-moment buffers hold ${stateSize} values, the adapters have ${trainable}`);
  } },
  { step: 'finetune', name: 'finetune runs the specified loop: same batches, clipping at maxGradNorm, AdamW (0.9, 0.95), fresh gradients every step', async run(m, T) {
    const build = () => { const model = pretrainedLike(12); m.applyLora(model, { rank: 2, alpha: 4, next: rng(60) }); return model; };
    const examples = syntheticExamples(16, 61);
    const opts = { steps: 6, lr: 2e-2, batchSize: 3, maxGradNorm: 0.05 };
    const res = await m.finetune(build(), examples, { ...opts, next: rng(62) });
    T.ok(res && Array.isArray(res.losses) && res.losses.length === opts.steps, 'finetune returns { losses, optimizer } with one loss per step');
    // The reference loop from the instructions, on an identical adapted model.
    const ref = build();
    const params = m.trainableParameters(ref);
    const optimizer = new AdamW(params, { lr: opts.lr, betas: [0.9, 0.95], weightDecay: 0 });
    const next = rng(62), want = [];
    for (let s = 0; s < opts.steps; s++) {
      const batch = m.makeBatch(examples, { batchSize: opts.batchSize, next });
      const loss = m.maskedLoss(ref.forward(batch.x), batch.y, batch.mask);
      loss.backward();
      clipGradNorm(params, opts.maxGradNorm);
      optimizer.step();
      optimizer.zeroGrad();
      want.push(loss.item());
    }
    T.close(res.losses[0], want[0], 1e-4, 'the first loss differs from the reference: draw each batch with makeBatch(examples, { batchSize, next }) and score it with maskedLoss(model.forward(batch.x), batch.y, batch.mask)');
    T.close(res.losses, want, 1e-4, 'later losses differ from the reference loop: clip trainableParameters(model) to `maxGradNorm` after backward, use AdamW with betas [0.9, 0.95] and no weight decay, and call zeroGrad after every step (otherwise gradients accumulate across steps)');
  } },
  { step: 'finetune', name: 'the same loop on an un-adapted model is full fine-tuning: every tensor moves', async run(m, T) {
    const model = new GPT(SMALL);
    const snapshot = model.parameters().map((p) => Array.from(p.data));
    const res = await m.finetune(model, syntheticExamples(24, 36), { steps: 20, lr: 1e-2, batchSize: 4, next: rng(37) });
    T.ok(res && res.losses && res.losses.length === 20, 'finetune returns { losses, optimizer }');
    T.ok(mean(res.losses.slice(-5)) < mean(res.losses.slice(0, 5)), 'loss should fall during full fine-tuning');
    const moved = model.parameters().filter((p, k) => p.data.some((v, i) => v !== snapshot[k][i])).length;
    T.ok(moved >= model.parameters().length - 1, `full fine-tuning should update (almost) every tensor; ${moved} of ${model.parameters().length} moved`);
    T.eq(res.optimizer.m.reduce((s, a) => s + a.length, 0), model.numParams(), 'with no adapters, AdamW state covers every parameter');
  } },
];
