// Module 30 — reference solution: LoRA adapters, freezing, targeting, merging, and the fine-tuning loop.
// Layout convention (the lab's, from lib/layers.js): a Linear stores W with shape [nIn, nOut] and computes
// y = x · W + b. The adapter is A [nIn, r] (down-projection, Gaussian) and B [r, nOut] (up-projection,
// zeros), so the update it represents is ΔW = (alpha / r) · A · B, a matrix of shape [nIn, nOut] and rank ≤ r.

import * as ops from 'lib/ops.js';
import { Tensor, noGrad } from 'lib/tensor.js';
import { Linear, paramNames } from 'lib/gpt.js';
import { AdamW, clipGradNorm } from 'lib/optim.js';
import { randInt, argmaxArray } from 'lib/util.js';

// ---------- worked examples: the SFT plumbing from module 10, trimmed to what this module needs ----------

/** The plain-text chat format used here (the checkpoint's vocabulary has no special chat markers). */
export function formatPrompt(prompt) {
  return `User: ${prompt}\nBot:`;
}

/** One example as token ids plus a loss mask that is 1 only on the response (and its closing newline). */
export function encodeExample(tokenizer, prompt, response) {
  const promptIds = tokenizer.encode(formatPrompt(prompt));
  const responseIds = tokenizer.encode(` ${response}\n`);
  return {
    ids: promptIds.concat(responseIds),
    mask: promptIds.map(() => 0).concat(responseIds.map(() => 1)),
  };
}

/** A batch of `batchSize` examples drawn with replacement, right-padded to a common length, shifted by one. */
export function makeBatch(examples, { batchSize, next, pad = 0 }) {
  const picked = [];
  for (let b = 0; b < batchSize; b++) picked.push(examples[randInt(next, examples.length)]);
  const len = Math.max(...picked.map((e) => e.ids.length));
  const x = [], y = [], mask = [];
  for (const e of picked) {
    const ids = e.ids.concat(new Array(len - e.ids.length).fill(pad));
    const m = e.mask.concat(new Array(len - e.mask.length).fill(0));
    x.push(ids.slice(0, -1));
    y.push(ids.slice(1));
    mask.push(m.slice(1));
  }
  return { x, y, mask };
}

/** Mean cross-entropy over the positions where mask is 1 (module 10's assistant-only loss). */
export function maskedLoss(logits, y, mask) {
  const V = logits.shape[logits.shape.length - 1];
  const targets = y.flat(Infinity);
  const weights = mask.flat(Infinity);
  let count = 0;
  for (const w of weights) count += w;
  if (count === 0) throw new Error('maskedLoss: the mask selects no positions');
  const pick = new Float32Array(targets.length * V);
  for (let i = 0; i < targets.length; i++) pick[i * V + targets[i]] = weights[i];
  const picked = logits.logSoftmax().mul(new Tensor({ shape: logits.shape.slice(), data: pick })).sum();
  return picked.scale(-1 / count);
}

/** Greedy reply to `prompt`, stopping at the first newline. Runs without building a graph. */
export function reply(model, tokenizer, prompt, { maxNewTokens = 16 } = {}) {
  const promptIds = tokenizer.encode(formatPrompt(prompt));
  const V = model.config.vocabSize;
  const out = [];
  noGrad(() => {
    for (let i = 0; i < maxNewTokens; i++) {
      const context = promptIds.concat(out).slice(-model.config.blockSize);
      const logits = model.forward([context]);
      out.push(argmaxArray(logits.data.subarray((context.length - 1) * V, context.length * V)));
      if (tokenizer.decode(out).includes('\n')) break;
    }
  });
  return tokenizer.decode(out).split('\n')[0].trim();
}

/** Find a sub-module by dotted path: getModule(model, 'blocks.0.attn.qkv') === model.blocks[0].attn.qkv. */
export function getModule(model, path) {
  let obj = model;
  for (const key of path.split('.')) obj = obj[key];
  return obj;
}

/** Replace the sub-module at a dotted path with `value`. */
export function setModule(model, path, value) {
  const keys = path.split('.');
  const parent = getModule(model, keys.slice(0, -1).join('.'));
  parent[keys[keys.length - 1]] = value;
}

// ---------- steps 1 and 2: the adapter ----------

export class LoRALinear {
  /**
   * Wrap `base` (a Linear) with a rank-`rank` adapter. `alpha` sets the scaling alpha / rank; `next` is a
   * seeded rng for A's Gaussian initialisation.
   */
  constructor(base, { rank, alpha = rank, next }) {
    if (typeof next !== 'function') throw new Error('LoRALinear: pass a seeded rng function as `next`');
    this.base = base;
    this.nIn = base.nIn;
    this.nOut = base.nOut;
    this.rank = rank;
    this.alpha = alpha;
    // step 1: the adapter itself
    this.A = Tensor.param(ops.randn([this.nIn, rank], next, 1 / Math.sqrt(this.nIn)));
    this.B = Tensor.param(ops.zeros([rank, this.nOut]));
    this.scaling = alpha / rank;
    // step 2: the base layer is frozen
    base.weight.requiresGrad = false;
    if (base.bias) base.bias.requiresGrad = false;
  }

  /** The base layer's tensors, so lib/gpt.js (namedParameters, toJSON) still finds them by name. */
  get weight() { return this.base.weight; }
  get bias() { return this.base.bias; }

  /** x [..., nIn] -> [..., nOut]:  x·W + b + (alpha/r) · (x·A)·B. ΔW is never materialised. */
  forward(x) {
    return this.base.forward(x).add(x.matmul(this.A).matmul(this.B).scale(this.scaling));
  }

  /** Only the adapter trains. */
  parameters() {
    return [this.A, this.B];
  }

  // ---------- step 4: merge ----------

  /** A fresh plain Linear whose weight is W + (alpha/r)·A·B and whose bias is a copy of b. */
  merge() {
    const merged = new Linear(this.nIn, this.nOut, { bias: this.base.bias !== null, next: () => 0.5 });
    const delta = ops.matmul(this.A, this.B); // [nIn, nOut]
    const W = this.base.weight.data;
    for (let i = 0; i < W.length; i++) merged.weight.data[i] = W[i] + this.scaling * delta.data[i];
    if (this.base.bias) merged.bias.data.set(this.base.bias.data);
    return merged;
  }
}

// ---------- step 3: put adapters into the GPT ----------

export const ALL_LINEAR = ['attn.qkv', 'attn.proj', 'mlp.fc', 'mlp.proj'];

/**
 * Freeze every parameter of `model`, then wrap each Linear whose path ends with one of `targets` in a
 * LoRALinear. Returns [{ name, layer }] in paramNames order, where name is the module path.
 */
export function applyLora(model, { rank, alpha = rank, targets = ALL_LINEAR, next }) {
  for (const p of model.parameters()) p.requiresGrad = false;
  const wrapped = [];
  for (const name of paramNames(model)) {
    if (!name.endsWith('.weight')) continue;
    const path = name.slice(0, -'.weight'.length);
    if (!targets.some((t) => path.endsWith('.' + t))) continue;
    const layer = new LoRALinear(getModule(model, path), { rank, alpha, next });
    setModule(model, path, layer);
    wrapped.push({ name: path, layer });
  }
  return wrapped;
}

/** Every LoRALinear inside `model`, as [{ name, layer }]. */
export function loraLayers(model) {
  const found = [];
  for (const name of paramNames(model)) {
    if (!name.endsWith('.weight')) continue;
    const path = name.slice(0, -'.weight'.length);
    const layer = getModule(model, path);
    if (layer instanceof LoRALinear) found.push({ name: path, layer });
  }
  return found;
}

/** The tensors an optimizer should own: adapter A and B, plus any model parameter still marked requiresGrad. */
export function trainableParameters(model) {
  const params = model.parameters().filter((p) => p.requiresGrad);
  for (const { layer } of loraLayers(model)) params.push(...layer.parameters());
  return params;
}

/** { trainable, total, percent }: total counts the frozen base and the adapters. */
export function countParams(model) {
  let total = model.numParams();
  for (const { layer } of loraLayers(model)) for (const p of layer.parameters()) total += p.size;
  let trainable = 0;
  for (const p of trainableParameters(model)) trainable += p.size;
  return { trainable, total, percent: (100 * trainable) / total };
}

// ---------- step 4: merge every adapter away ----------

/** Replace every LoRALinear in `model` with its merged Linear. Returns how many were merged. */
export function mergeLora(model) {
  const layers = loraLayers(model);
  for (const { name, layer } of layers) setModule(model, name, layer.merge());
  return layers.length;
}

// ---------- step 5: what training costs, and the loop ----------

/**
 * Persistent training memory in bytes under mixed-precision AdamW (the ZeRO accounting from module 08):
 * bf16 weights for every parameter (2 B), bf16 gradients for trainable ones (2 B), and an fp32 master copy
 * plus two fp32 moments for trainable ones (12 B). Frozen: 2 B each. Trainable: 16 B each.
 */
export function trainingMemory({ total, trainable }) {
  const weights = 2 * total;
  const grads = 2 * trainable;
  const optimizer = 12 * trainable;
  return { weights, grads, optimizer, total: weights + grads + optimizer };
}

/**
 * Fine-tune `model` on encoded `examples` for `steps` AdamW steps over trainableParameters(model) only.
 * Returns { losses, optimizer }, where losses holds one number (loss.item()) per step.
 * @param onStep optional async (step: number, loss: number) => void, called after every step;
 *   step counts from 0 and loss is loss.item(), a plain number, not the Tensor.
 */
export async function finetune(model, examples, { steps, lr = 1e-3, batchSize = 4, maxGradNorm = 1.0, next, onStep = null }) {
  if (typeof next !== 'function') throw new Error('finetune: pass a seeded rng function as `next`');
  const params = trainableParameters(model);
  const optimizer = new AdamW(params, { lr, betas: [0.9, 0.95], weightDecay: 0 });
  const losses = [];
  for (let step = 0; step < steps; step++) {
    const batch = makeBatch(examples, { batchSize, next });
    const loss = maskedLoss(model.forward(batch.x), batch.y, batch.mask);
    loss.backward();
    clipGradNorm(params, maxGradNorm);
    optimizer.step();
    optimizer.zeroGrad();
    losses.push(loss.item());
    if (onStep) await onStep(step, loss.item());
  }
  return { losses, optimizer };
}
