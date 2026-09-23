// Module 27 — Capstone: chat with your own model.
//
// Nothing in this file is a new idea. Every function is plumbing between layers you built earlier:
// the tokenizer (03), the inference model and its KV cache (15), prefix reuse (17), the sampler (14),
// the tool registry (20) and the context budget (21). The chat product is the composition.
//
// Everything above the "step 1" line is done for you: read it first, it fixes the template, the
// vocabulary, the calculator and the shape of a conversation's state. Below it are five TODOs.
// Conventions: token ids are plain arrays of ints; a KV cache is lib/infer.js newCache(model), and
// `session = { cache, ids }` always satisfies session.cache.length === session.ids.length.

import { BPETokenizer } from 'lib/tokenizer.js';
import * as ops from 'lib/ops.js';
import { newCache, prefill, forwardStep } from 'lib/infer.js';
import { sample } from 'lib/sampling.js';
import { ToolRegistry } from 'lib/harness.js';
import { rng, randn, now } from 'lib/util.js';

// ---------- worked examples (done for you; they set the conventions) ----------

/** The chat template: one marker per role and one end-of-message marker (lib/data.js CHAT plus a tool role). */
export const CHAT_TEMPLATE = {
  system: '<|system|>',
  user: '<|user|>',
  assistant: '<|assistant|>',
  tool: '<|tool|>',
  end: '<|end|>',
};

/** The tags around a tool call: <tool_call>calc(17*23)</tool_call>. */
export const TOOL_OPEN = '<tool_call>';
export const TOOL_CLOSE = '</tool_call>';

/**
 * Tokens the lab's BPE vocabulary never saw: the markers, the tool tags, and three arithmetic operators
 * (the pre-training corpus has no `+`, `*` or `/`, so they would encode to <|unk|>).
 */
export const EXTRA_TOKENS = [...Object.values(CHAT_TEMPLATE), TOOL_OPEN, TOOL_CLOSE, '+', '*', '/'];

/**
 * A copy of `base` (a BPETokenizer) in which every EXTRA_TOKENS entry is one special token with its own id.
 * New ids are appended after the existing ones, so every id the model already knows keeps its meaning.
 */
export function chatTokenizer(base) {
  const json = base.toJSON();
  const added = EXTRA_TOKENS.filter((t) => !json.vocab.includes(t));
  return BPETokenizer.fromJSON({
    ...json,
    vocab: [...json.vocab, ...added],
    specials: [...json.specials, ...EXTRA_TOKENS.filter((t) => !json.specials.includes(t))],
  });
}

/**
 * A copy of an inference model (lib/infer.js loadModel) whose token table has `vocabSize` rows. Existing
 * rows are copied; new rows are small seeded Gaussians (std 0.02, GPT-2's init). The head is tied to the
 * table, so the new tokens get logits too — near zero, because nothing was ever trained on them.
 */
export function extendVocab(model, vocabSize, { seed = 0, std = 0.02 } = {}) {
  const old = model.w['wte.weight'];
  const [rows, width] = old.shape;
  if (vocabSize < rows) throw new Error(`extendVocab: ${vocabSize} is smaller than the current ${rows} rows`);
  const data = new Float32Array(vocabSize * width);
  data.set(old.data);
  const next = rng(seed);
  for (let i = rows * width; i < data.length; i++) data[i] = randn(next) * std;
  return {
    config: { ...model.config, vocabSize },
    w: { ...model.w, 'wte.weight': { shape: [vocabSize, width], data } },
  };
}

/** One message in the template: marker + content + end marker. */
export function renderMessage(message) {
  const marker = CHAT_TEMPLATE[message.role];
  if (marker === undefined || message.role === 'end') throw new Error(`renderMessage: unknown role "${message.role}"`);
  return marker + message.content + CHAT_TEMPLATE.end;
}

/**
 * A whole prompt. Unless the last message is the assistant's, the assistant's turn is opened at the end,
 * followed by `prefill` (text the assistant's reply is forced to start with).
 */
export function renderChat(messages, prefill = '') {
  let text = messages.map(renderMessage).join('');
  const last = messages[messages.length - 1];
  if (!last || last.role !== 'assistant') text += CHAT_TEMPLATE.assistant + prefill;
  return text;
}

/** Length of the longest common prefix of two id arrays. */
export function sharedPrefixLength(a, b) {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return i;
}

/**
 * The calculator tool: + - * /, unary minus and parentheses over decimal numbers, by recursive descent.
 * No eval: the string came from a model, and model output is untrusted input. Throws on anything else.
 */
export function calculate(expression) {
  const tokens = String(expression).match(/\d+(?:\.\d+)?|[-+*/()]|\S/g) ?? [];
  let at = 0;
  const peek = () => tokens[at];
  const take = () => tokens[at++];
  function expr() {
    let value = term();
    while (peek() === '+' || peek() === '-') value = take() === '+' ? value + term() : value - term();
    return value;
  }
  function term() {
    let value = factor();
    while (peek() === '*' || peek() === '/') {
      if (take() === '*') value *= factor();
      else {
        const divisor = factor();
        if (divisor === 0) throw new Error('division by zero');
        value /= divisor;
      }
    }
    return value;
  }
  function factor() {
    const t = take();
    if (t === '-') return -factor();
    if (t === '(') {
      const value = expr();
      if (take() !== ')') throw new Error('missing )');
      return value;
    }
    if (t !== undefined && /^\d/.test(t)) return Number(t);
    throw new Error(`unexpected ${t === undefined ? 'end of expression' : `"${t}"`}`);
  }
  const value = expr();
  if (at < tokens.length) throw new Error(`unexpected "${tokens[at]}"`);
  return Number(value.toPrecision(12));
}

/** The harness's tools (module 20's ToolRegistry): one calculator, `calc`, taking a single string argument. */
export function makeTools() {
  return new ToolRegistry().register('calc', {
    description: 'Evaluate an arithmetic expression with + - * / and parentheses.',
    parameters: { type: 'object', properties: { input: { type: 'string' } }, required: ['input'] },
    handler: ({ input }) => String(calculate(input)),
  });
}

/**
 * A new conversation. `session` is what the KV cache currently holds: `ids[i]` is the token at cache
 * position i, so `session.cache.length === session.ids.length` at all times. `stopIds` are the tokens that
 * end a reply: the end marker, end-of-text, and every role marker (a reply must never open a new turn).
 */
export function createChat({ model, tokenizer, tools = makeTools(), system = null }) {
  const markerId = (text) => tokenizer.encode(text)[0];
  const stopIds = [CHAT_TEMPLATE.end, CHAT_TEMPLATE.system, CHAT_TEMPLATE.user, CHAT_TEMPLATE.assistant, CHAT_TEMPLATE.tool].map(markerId);
  if (tokenizer.eos >= 0) stopIds.push(tokenizer.eos);
  return {
    model,
    tokenizer,
    tools,
    history: system ? [{ role: 'system', content: system }] : [],
    session: { cache: newCache(model), ids: [] },
    stopIds,
  };
}

/** The transcript as markdown, one line per message (tool results shown as such). */
export function renderTranscript(history) {
  return history
    .map((m) => `**${m.role}${m.role === 'tool' ? ` (${m.name})` : ''}:** ${m.content.replace(/\n/g, ' ') || '(empty)'}`)
    .join('\n\n');
}

// ---------- step 1: prompt assembly under a token budget ----------

/**
 * Render `history` (plus the opened assistant turn and `prefill`) and encode it. While it costs more than
 * `budget` tokens, drop the OLDEST turn and re-encode. A turn is a user message plus everything after it up
 * to the next user message (the assistant reply, tool calls and tool results travel together: a tool result
 * without the call that produced it is an orphan). System messages are never dropped, and neither is the
 * current (last) turn: if the budget still cannot be met, throw. Does not modify `history`.
 * Returns { ids, text, messages (the ones kept), dropped (how many messages were removed) }.
 */
export function buildPrompt(tokenizer, history, { budget, prefill = '' } = {}) {
  // TODO: step 1 — render with renderChat, encode, and drop whole oldest turns until it fits.
  return { ids: [], text: '', messages: history.slice(), dropped: 0 };
}

// ---------- step 2: prefill with prefix reuse ----------

/** Keep only the first `n` positions of every layer's keys and values. Mutates and returns `cache`. */
export function truncateCache(cache, n) {
  // TODO: step 2 — slice every layer's keys and values to positions 0..n-1 (ops.slice on axis 1).
  return cache;
}

/**
 * Make `session` hold exactly `ids` and return the logits after the last one. Positions shared with what
 * the cache already holds are reused; the cache is cut back to the shared prefix and only the rest is
 * prefilled. At least one token is always run, because its logits are the output.
 * Returns { logits, reused, prefilled }.
 */
export function syncCache(model, session, ids) {
  // TODO: step 2 — shared prefix, cut the cache back, prefill only the rest, record session.ids.
  return { logits: new Float32Array(model.config.vocabSize), reused: 0, prefilled: 0 };
}

// ---------- step 3: decoding until a stop token ----------

/**
 * Sample up to `maxNewTokens` tokens starting from `logits`. Each iteration: stop with 'length' if the
 * budget is spent, with 'context' if the cache is full; otherwise sample; a token in `stopIds` ends the
 * reply with 'stop' and is NOT kept; any other token is appended to session.ids and run through the cache.
 * Returns { text, ids, stopReason }.
 */
export function decodeReply(model, tokenizer, session, logits, { maxNewTokens = 16, stopIds = [], next, temperature = 1, topK = 0, topP = 1 } = {}) {
  // TODO: step 3 — the sample / check / append / forwardStep loop.
  return { text: '', ids: [], stopReason: 'length' };
}

// ---------- step 4: tool calls ----------

/**
 * Every complete <tool_call>name(argument)</tool_call> in `text`, in order, as { name, input }.
 * `input` is everything between the first '(' and the last ')', trimmed. Blocks without a closing tag or
 * without the name(...) shape are skipped.
 */
export function findToolCalls(text) {
  // TODO: step 4 — scan for TOOL_OPEN … TOOL_CLOSE blocks and match name(argument) inside each.
  return [];
}

/** Run every call in `text` through the registry; one { role: 'tool', name, content } message per call. */
export async function runToolCalls(text, tools) {
  // TODO: step 4 — one tools.call(name, { input }) per call, wrapped as a tool message.
  return [];
}

// ---------- step 5: the chat function ----------

/**
 * One user turn. Appends the user message, then runs rounds: build the prompt under the budget
 * (blockSize − maxNewTokens), sync the cache, decode (skipped when `prefill` already contains a complete
 * tool call), append the assistant message (prefill + generated text), and run its tool calls. A round
 * that produced tool results is followed by another round, up to `maxToolRounds` extra rounds.
 * Returns { reply, stats }:
 *   stats.rounds[i] = { promptTokens, cachedTokens, prefilledTokens, generatedTokens, dropped, stopReason }
 *   stats.promptTokens / cachedTokens / prefilledTokens / generatedTokens = sums over rounds,
 *   stats.dropped = the largest per-round dropped, stats.toolCalls = calls executed,
 *   stats.stopReason = the last round's, stats.ms = wall-clock time of the whole turn (util.now()).
 * A forced round (the prefill already holds a complete call) has generatedTokens 0 and stopReason 'forced'.
 */
export async function chat(state, userMessage, { maxNewTokens = 12, prefill = '', maxToolRounds = 2, next, temperature = 1, topK = 0, topP = 1 } = {}) {
  // TODO: step 5 — wire steps 1–4 together: push the user message, then rounds of
  // buildPrompt → syncCache → decodeReply → push the reply → runToolCalls.
  return { reply: '', stats: { rounds: [] } };
}
