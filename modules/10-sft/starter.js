// Module 10 — Supervised fine-tuning.
// You turn the pre-trained checkpoint into a model that follows a chat template: render the template,
// build (x, y, mask) examples whose loss counts only the assistant's tokens, pack several examples into
// one window, grow the embedding table for the new marker tokens, and run the fine-tuning loop.
//
// Conventions (same as module 07): token ids are plain JS arrays, batches are number[][] (B×T), all
// randomness comes through a `next` function from rng(seed), and every function returns new values
// rather than mutating its inputs (except the optimizer step, which updates parameters in place).

import { CHAT } from 'lib/data.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { GPT, paramNames } from 'lib/gpt.js';
import { Tensor, noGrad } from 'lib/tensor.js';
import { AdamW, clipGradNorm } from 'lib/optim.js';
import { randInt, argmaxArray } from 'lib/util.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/** The four chat markers, in the order they are appended to the vocabulary. */
export const MARKERS = [CHAT.system, CHAT.user, CHAT.assistant, CHAT.end];

/**
 * The id of each marker under `tokenizer`: { system, user, assistant, end }.
 * A marker is a special token, so encode() returns exactly one id for it.
 */
export function markerIds(tokenizer) {
  const ids = {};
  for (const role of Object.keys(CHAT)) ids[role] = tokenizer.encode(CHAT[role])[0];
  return ids;
}

/**
 * Token stream -> training pair. x is every token but the last, y is every token but the first, so
 * y[t] is the token the model must predict after reading x[0..t]. The mask is sliced exactly like y,
 * because it says which TARGETS count, and targets live in y.
 */
export function shift({ ids, mask }) {
  return { x: ids.slice(0, -1), y: ids.slice(1), mask: mask.slice(1) };
}

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

/**
 * Greedy chat completion (used by the goal demo): render the prompt with the template, then take the
 * argmax token until the model emits the end marker or maxNewTokens is reached.
 * Returns { text, tokens, ended }. Note that it depends on your formatChat from step 1.
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

// ---------- step 1: the chat template and the tokens for it ----------

/**
 * Render messages [{ role, content }] as one string: for each message, its role's marker from CHAT
 * (unknown roles use CHAT.user), then the content, then CHAT.end. If the LAST message is from the
 * user, append CHAT.assistant so the model continues as the assistant.
 * formatChat([{ role: 'user', content: 'Hi' }]) === '<|user|>Hi<|end|><|assistant|>'
 */
export function formatChat(messages) {
  let out = '';
  for (const message of messages) {
    // TODO: step 1 — append marker + content + end marker
  }
  // TODO: step 1 — open the assistant turn when the conversation ends with a user message
  return out;
}

/**
 * A NEW BPETokenizer that knows the four MARKERS as special tokens (single ids, appended after the
 * existing vocabulary so no existing id moves). Do not mutate `tokenizer`; adding markers that are
 * already present must not add them twice.
 */
export function addChatTokens(tokenizer) {
  // TODO: step 1
  return tokenizer;
}

// ---------- step 2: one SFT example and its loss mask ----------

/**
 * Tokenize one (prompt, response) pair into { ids, mask }: ids is the rendered chat for the user
 * prompt followed by the response and the end marker; mask[i] is 1 if ids[i] is an assistant token
 * (the response and its end marker) and 0 for every prompt token, markers included.
 */
export function tokenizeExample(tokenizer, prompt, response) {
  // TODO: step 2
  return { ids: [], mask: [] };
}

/** tokenizeExample, then shift: { x, y, mask } ready for the model, with the mask aligned to y. */
export function buildExample(tokenizer, prompt, response) {
  // TODO: step 2
  return { x: [], y: [], mask: [] };
}

// ---------- step 3: cross-entropy that only counts masked-in positions ----------

/**
 * logits: Tensor [B, T, V] (or [N, V]); y and mask: nested int arrays matching logits' leading dims.
 * Return a scalar Tensor: sum over positions of mask * nll / sum(mask), where nll is
 * -logSoftmax(logits)[target]. Build it from Tensor ops so backward() flows through it.
 * Throw if the mask selects no positions (the mean would be 0/0).
 */
export function maskedCrossEntropy(logits, y, mask) {
  // TODO: step 3
  return logits.sum();
}

// ---------- step 4: packing several examples into one window ----------

/**
 * Greedily concatenate examples ({ ids, mask } from tokenizeExample) in order, each followed by one
 * `eos` id with mask 0, into windows of blockSize + 1 tokens; pad the last window with eos (mask 0)
 * and return shift() of every window, so each pack is { x, y, mask } with exactly blockSize entries.
 * Start a new window when the next example plus its separator would not fit. Throw if a single
 * example is longer than blockSize.
 */
export function packExamples(examples, { blockSize, eos }) {
  // TODO: step 4
  return [];
}

// ---------- step 5: give the checkpoint rows for the new tokens ----------

/**
 * A NEW GPT with config.vocabSize = newVocabSize whose parameters are copied from `model`, except
 * that wte.weight gains rows: rows 0..oldV-1 are copied and every new row is the column-wise mean
 * of the old rows. Throw if newVocabSize < the old vocabulary size. Do not modify `model`.
 * paramNames(model) gives the name of each entry of model.parameters(), in the same order.
 */
export function resizeEmbeddings(model, newVocabSize) {
  // TODO: step 5
  return model;
}

// ---------- step 6: the fine-tuning loop ----------

/**
 * One SFT step on batch { x, y, mask }: forward, maskedCrossEntropy, backward, clipGradNorm,
 * optimizer.step(), optimizer.zeroGrad(). Return { loss: number, gradNorm: number }.
 */
export function sftStep(model, optimizer, batch, { maxGradNorm = 1.0 } = {}) {
  // TODO: step 6
  return { loss: NaN, gradNorm: NaN };
}

/**
 * Fine-tune `model` on `packs` for `steps` steps with AdamW(lr, betas [0.9, 0.95], weightDecay).
 * Each step draws sampleBatch(packs, batchSize, next) and calls sftStep. Push every loss into an
 * array, `await onStep(step, loss)` when onStep is given, and return the array of losses.
 */
export async function finetune(model, packs, { steps, lr = 3e-4, batchSize = 4, weightDecay = 0.1, maxGradNorm = 1.0, next, onStep = null }) {
  if (typeof next !== 'function') throw new Error('finetune: pass a seeded rng function as `next`');
  // TODO: step 6
  return [];
}
