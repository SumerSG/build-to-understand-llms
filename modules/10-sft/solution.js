// Supervised fine-tuning — reference solution: chat template, assistant-only loss masking, packing, and the SFT loop.

import { CHAT } from 'lib/data.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { GPT, paramNames } from 'lib/gpt.js';
import { Tensor, noGrad } from 'lib/tensor.js';
import { AdamW, clipGradNorm } from 'lib/optim.js';
import { randInt, argmaxArray } from 'lib/util.js';

/** The four chat markers, in the order they are appended to the vocabulary. */
export const MARKERS = [CHAT.system, CHAT.user, CHAT.assistant, CHAT.end];

// ---------- step 1: the chat template and the tokens for it ----------

export function formatChat(messages) {
  let out = '';
  for (const message of messages) {
    const marker = CHAT[message.role] ?? CHAT.user;
    out += marker + message.content + CHAT.end;
  }
  const last = messages[messages.length - 1];
  if (last && last.role === 'user') out += CHAT.assistant;
  return out;
}

export function addChatTokens(tokenizer) {
  const missing = MARKERS.filter((marker) => !tokenizer.vocab.includes(marker));
  return new BPETokenizer({
    vocab: [...tokenizer.vocab, ...missing],
    merges: tokenizer.merges,
    specials: [...tokenizer.specials, ...missing],
  });
}

/** The id of each marker under `tokenizer`: { system, user, assistant, end }. */
export function markerIds(tokenizer) {
  const ids = {};
  for (const role of Object.keys(CHAT)) ids[role] = tokenizer.encode(CHAT[role])[0];
  return ids;
}

// ---------- step 2: one SFT example and its loss mask ----------

export function tokenizeExample(tokenizer, prompt, response) {
  const promptIds = tokenizer.encode(formatChat([{ role: 'user', content: prompt }]));
  const responseIds = tokenizer.encode(response + CHAT.end);
  const ids = promptIds.concat(responseIds);
  const mask = promptIds.map(() => 0).concat(responseIds.map(() => 1));
  return { ids, mask };
}

/** Token stream -> training pair: x is every token but the last, y is every token but the first, mask follows y. */
export function shift({ ids, mask }) {
  return { x: ids.slice(0, -1), y: ids.slice(1), mask: mask.slice(1) };
}

export function buildExample(tokenizer, prompt, response) {
  return shift(tokenizeExample(tokenizer, prompt, response));
}

// ---------- step 3: cross-entropy that only counts masked-in positions ----------

export function maskedCrossEntropy(logits, y, mask) {
  const V = logits.shape[logits.shape.length - 1];
  const targets = y.flat(Infinity);
  const weights = mask.flat(Infinity);
  const N = targets.length;
  if (N * V !== logits.data.length) throw new Error(`maskedCrossEntropy: ${N} targets do not match logits [${logits.shape}]`);
  if (weights.length !== N) throw new Error(`maskedCrossEntropy: mask has ${weights.length} entries, targets ${N}`);
  let count = 0;
  for (let i = 0; i < N; i++) count += weights[i];
  if (count === 0) throw new Error('maskedCrossEntropy: the mask selects no positions, so the loss is undefined');
  // A one-hot of the target at every position, already multiplied by the mask: sum(lp * pick) is sum(mask * logp(target)).
  const pick = new Float32Array(N * V);
  for (let i = 0; i < N; i++) {
    if (!(targets[i] >= 0 && targets[i] < V)) throw new Error(`maskedCrossEntropy: target ${targets[i]} out of range for V=${V}`);
    pick[i * V + targets[i]] = weights[i];
  }
  const picked = logits.logSoftmax().mul(new Tensor({ shape: logits.shape.slice(), data: pick })).sum();
  return picked.scale(-1 / count);
}

// ---------- step 4: packing several examples into one window ----------

export function packExamples(examples, { blockSize, eos }) {
  const packs = [];
  let ids = [];
  let mask = [];
  const flush = () => {
    if (ids.length === 0) return;
    while (ids.length < blockSize + 1) { ids.push(eos); mask.push(0); }
    packs.push(shift({ ids, mask }));
    ids = [];
    mask = [];
  };
  for (const example of examples) {
    if (example.ids.length > blockSize) {
      throw new Error(`packExamples: an example of ${example.ids.length} tokens does not fit blockSize ${blockSize}`);
    }
    // The example plus its separator must fit in the blockSize + 1 tokens a window holds.
    if (ids.length + example.ids.length + 1 > blockSize + 1) flush();
    for (const id of example.ids) ids.push(id);
    ids.push(eos);
    for (const m of example.mask) mask.push(m);
    mask.push(0);
  }
  flush();
  return packs;
}

// ---------- step 5: give the checkpoint rows for the new tokens ----------

export function resizeEmbeddings(model, newVocabSize) {
  const oldV = model.config.vocabSize;
  const d = model.config.nEmbd;
  if (newVocabSize < oldV) throw new Error(`resizeEmbeddings: cannot shrink the vocabulary from ${oldV} to ${newVocabSize}`);
  const grown = new GPT({ ...model.config, vocabSize: newVocabSize });
  const names = paramNames(model);
  const src = model.parameters();
  const dst = grown.parameters();
  for (let k = 0; k < names.length; k++) {
    if (names[k] !== 'wte.weight') { dst[k].data.set(src[k].data); continue; }
    const old = src[k].data;
    const table = dst[k].data;
    table.set(old); // rows 0..oldV-1 keep their trained values
    const mean = new Float32Array(d);
    for (let i = 0; i < oldV; i++) for (let j = 0; j < d; j++) mean[j] += old[i * d + j] / oldV;
    for (let i = oldV; i < newVocabSize; i++) table.set(mean, i * d);
  }
  return grown;
}

// ---------- step 6: the fine-tuning loop ----------

/** A batch of `batchSize` packs drawn at random (with replacement) from `packs`. */
export function sampleBatch(packs, batchSize, next) {
  const x = [], y = [], mask = [];
  for (let b = 0; b < batchSize; b++) {
    const pack = packs[randInt(next, packs.length)];
    x.push(pack.x);
    y.push(pack.y);
    mask.push(pack.mask);
  }
  return { x, y, mask };
}

export function sftStep(model, optimizer, batch, { maxGradNorm = 1.0 } = {}) {
  const logits = model.forward(batch.x);
  const loss = maskedCrossEntropy(logits, batch.y, batch.mask);
  loss.backward();
  const gradNorm = clipGradNorm(model.parameters(), maxGradNorm);
  optimizer.step();
  optimizer.zeroGrad();
  return { loss: loss.item(), gradNorm };
}

export async function finetune(model, packs, { steps, lr = 3e-4, batchSize = 4, weightDecay = 0.1, maxGradNorm = 1.0, next, onStep = null }) {
  if (typeof next !== 'function') throw new Error('finetune: pass a seeded rng function as `next`');
  const optimizer = new AdamW(model.parameters(), { lr, betas: [0.9, 0.95], weightDecay });
  const losses = [];
  for (let step = 0; step < steps; step++) {
    const batch = sampleBatch(packs, batchSize, next);
    const { loss } = sftStep(model, optimizer, batch, { maxGradNorm });
    losses.push(loss);
    if (onStep) await onStep(step, loss);
  }
  return losses;
}

/**
 * Greedy chat completion: render the prompt with the template, then take the argmax token until the
 * model emits the end marker or maxNewTokens is reached. Returns the decoded reply and whether it ended.
 */
export function generateChat(model, tokenizer, prompt, { maxNewTokens = 24 } = {}) {
  const endId = markerIds(tokenizer).end;
  const promptIds = tokenizer.encode(formatChat([{ role: 'user', content: prompt }]));
  const V = model.config.vocabSize;
  const out = [];
  let ended = false;
  noGrad(() => {
    for (let i = 0; i < maxNewTokens; i++) {
      const context = promptIds.concat(out).slice(-model.config.blockSize);
      const logits = model.forward([context]);
      const last = logits.data.subarray((context.length - 1) * V, context.length * V);
      const id = argmaxArray(last);
      if (id === endId) { ended = true; return; }
      out.push(id);
    }
  });
  return { text: tokenizer.decode(out), tokens: out.length, ended };
}
