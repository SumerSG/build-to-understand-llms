// Context management & retrieval.
// The context window is a scarce, ORDERED budget of tokens. Everything you build here is a policy for
// spending it: count what each message costs, trim what is bloated, replace old turns by a summary,
// and fetch back from a note store only what the current turn needs.
//
// A message is a plain object { role, content, name? } with role 'system' | 'user' | 'assistant' | 'tool'
// (the same array the agent loop from the agent loop module builds). A tokenizer is anything with encode(str) → number[]
// and decode(ids) → string: the demo uses the lab's BPE tokenizer, the tests use a one-id-per-word one
// so every count can be checked by hand.
//
// Everything above the "step 1" line is done for you; read it, it sets the conventions.
// Functions never modify the arrays or messages they are given: they return new ones.

// ---------- worked examples and constants (done for you) ----------

/**
 * Per-message overhead in tokens: the role marker and end-of-message framing (`<|user|>` … `<|end|>` in
 * lib/data.js's template) cost tokens on the wire, not just the text. OpenAI's token-counting cookbook
 * uses 3 or 4 such tokens per chat message depending on the model; the exact figure depends on the template.
 */
export const TOKENS_PER_MESSAGE = 4;

/**
 * Text that replaces the middle of a truncated tool result. A constant, so its token cost is fixed
 * (13 tokens on the lab BPE; square brackets are avoided because the toy vocabulary has none).
 */
export const TRUNCATION_MARKER = '\n(... cut ...)\n';

/**
 * Number of tokens `text` costs under `tokenizer`. Missing or empty text costs 0.
 * This is the only place the tokenizer is called for counting: every budget decision goes through it.
 */
export function countTokens(tokenizer, text) {
  if (typeof text !== 'string' || text.length === 0) return 0;
  return tokenizer.encode(text).length;
}

/**
 * Lowercase word terms for BM25 (step 4). This is NOT the BPE tokenizer: retrieval matches whole words.
 * tokenizeTerms('The deploy key, slot 7!') → ['the', 'deploy', 'key', 'slot', '7']
 */
export function tokenizeTerms(text) {
  return String(text ?? '').toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

// ---------- step 1: counting the cost of a context ----------

/**
 * Tokens one message costs: TOKENS_PER_MESSAGE, plus its content, plus its `name` if it has one
 * (a tool message carries the tool's name into the context too). Use countTokens for both strings.
 */
export function messageTokens(tokenizer, msg) {
  // TODO: step 1
  return 0;
}

/** Tokens a whole message array costs: the sum of messageTokens over it. */
export function contextTokens(tokenizer, messages) {
  // TODO: step 1
  return 0;
}

/**
 * Does the array fit the budget? Return `{ tokens, budget, remaining, fits }` where
 * remaining = budget − tokens (negative when over) and fits is true when tokens <= budget.
 */
export function budgetReport(tokenizer, messages, budget) {
  // TODO: step 1
  return { tokens: 0, budget, remaining: budget, fits: true };
}

// ---------- step 2: truncation policies ----------

/**
 * Cut `text` down to at most `maxTokens` tokens INCLUDING the marker, keeping the head and the tail:
 * keep = maxTokens − (marker tokens); the head gets ceil(keep / 2) tokens and the tail the rest,
 * with TRUNCATION_MARKER in between. Text that already fits is returned unchanged.
 */
export function truncateText(tokenizer, text, maxTokens) {
  // TODO: step 2
  return text;
}

/**
 * A new array in which every message with role 'tool' whose content costs more than `maxToolTokens`
 * is replaced by a copy with truncated content. Every other message is returned as the same object.
 */
export function truncateToolResults(tokenizer, messages, maxToolTokens) {
  // TODO: step 2
  return messages.slice();
}

/**
 * Drop the OLDEST non-system messages until the array fits `budget`, and no more than that.
 * Messages with role 'system' are never dropped, whatever the budget: the result may still be
 * over budget if the system messages alone exceed it.
 */
export function dropOldest(tokenizer, messages, budget) {
  // TODO: step 2
  return messages.slice();
}

// ---------- step 3: compaction ----------

/**
 * The scripted summariser: every line of the form `remember: <fact>` (prefix at the start of the
 * line, any case, surrounding whitespace ignored, non-empty fact) yields the fact. Duplicates are
 * dropped, first occurrence wins. Missing text yields [].
 */
export function extractFacts(text) {
  // TODO: step 3
  return [];
}

/**
 * Replace every non-system message older than the last `keepLast` of them (and any earlier summary)
 * by ONE summary message, placed where the span began:
 *   { role: 'system', summary: true, count, content: 'Summary of <count> earlier messages.\nremember: …' }
 * `count` is how many original messages it stands for (an old summary contributes its own count).
 * Plain system messages that are not summaries are left where they are.
 */
export function compact(messages, { keepLast = 4 } = {}) {
  // TODO: step 3
  return messages.slice();
}

// ---------- step 4: BM25 ----------

/** Okapi BM25 over a list of short text notes, with the usual k1 = 1.5 and b = 0.75. */
export class BM25Index {
  constructor(docs, { k1 = 1.5, b = 0.75 } = {}) {
    this.docs = docs.slice();
    this.k1 = k1;
    this.b = b;
    this.tf = [];          // one Map term → count per document
    this.df = new Map();   // term → number of documents containing it
    this.lengths = [];     // number of terms in each document
    this.avgdl = 0;        // average document length in terms
    // TODO: step 4 — fill tf, df, lengths and avgdl using tokenizeTerms
  }

  /** `ln(1 + (N − df + 0.5) / (df + 0.5))` where N is the number of documents. */
  idf(term) {
    // TODO: step 4
    return 0;
  }

  /** BM25 score of document `i` for `query` (a string). */
  score(query, i) {
    // TODO: step 4
    return 0;
  }

  /** The `k` best documents with a positive score: `[{ index, score }]`, best first, ties by index. */
  search(query, k = 3) {
    // TODO: step 4
    return [];
  }
}

// ---------- step 5: the context manager ----------

/**
 * Owns the full history and a note store, and on every turn assembles the array the model will see.
 * Policies: 'none' (raw history), 'drop' (oldest-first), 'compact' (summary, then oldest-first as a
 * last resort). `topK > 0` turns on BM25 retrieval from the note store; `memory` writes every
 * `remember:` fact that passes through add() into the store.
 */
export class ContextManager {
  constructor({ tokenizer, budget, system, policy = 'compact', keepLast = 4, maxToolTokens = 60, topK = 0, memory = true, notes = [] }) {
    this.tokenizer = tokenizer;
    this.budget = budget;
    this.system = { role: 'system', content: system };
    this.policy = policy;
    this.keepLast = keepLast;
    this.maxToolTokens = maxToolTokens;
    this.topK = topK;
    this.memory = memory;
    this.history = [];   // every message ever added, never trimmed: assemble() returns a view
    this.notes = [];     // the note store BM25 searches
    this.index = null;   // a BM25Index over this.notes, rebuilt lazily when null
    for (const n of notes) this.remember(n);
  }

  /** Add a note to the store (deduplicated). The BM25 index is rebuilt lazily on the next retrieve. */
  remember(note) {
    if (this.notes.includes(note)) return this;
    this.notes.push(note);
    this.index = null;
    return this;
  }

  /** Append a message to the history; with `memory`, harvest its `remember:` facts into the store. */
  add(msg) {
    this.history.push(msg);
    if (this.memory) for (const f of extractFacts(msg.content)) this.remember(f);
    return this;
  }

  /** The top-k notes for `query` as strings (empty when topK is 0, the store is empty, or nothing matches). */
  retrieve(query) {
    // TODO: step 5
    return [];
  }

  /**
   * Build the context for the next model call: `{ messages, tokens, retrieved }`.
   *   messages  = [system prompt, …managed history…] with, when notes were retrieved, one message
   *               { role: 'user', retrieved: true, content: 'Relevant notes:\n- note\n- note' }
   *               placed immediately before the newest history message.
   *   tokens    = contextTokens of exactly those messages.
   *   retrieved = the note strings that were injected.
   * The query is the content of the newest user message. Unless the policy is 'none', the result
   * must fit `budget`, and the retrieval message is paid for BEFORE the history is trimmed.
   */
  assemble() {
    // TODO: step 5
    return { messages: [], tokens: 0, retrieved: [] };
  }
}

// ---------- step 6: dense and hybrid retrieval ----------

/**
 * Mean of the rows `ids` of an embedding table: a [V, C] tensor with a flat row-major `data`, such as a
 * lib/gpt.js GPT's `model.wte.weight` (row `id` starts at `data[id * C]`). A repeated id counts every
 * time it occurs. No ids → C zeros. Return a new array of length C; never modify the table.
 */
export function meanPool(table, ids) {
  // TODO: step 6
  return new Float64Array(table.shape[1]);
}

/** Cosine similarity `a·b / (|a| |b|)`; 0 when either vector is all zeros. */
export function cosine(a, b) {
  // TODO: step 6
  return 0;
}

/**
 * The `k` documents whose vectors have the highest cosine with `queryVector`: `[{ index, score }]`,
 * best first, ties by index. No score threshold: nearest neighbours always exist.
 */
export function denseSearch(docVectors, queryVector, k = 3) {
  // TODO: step 6
  return [];
}

/**
 * Reciprocal rank fusion: each ranking is a list of hits `{ index, … }`, best first. A document scores
 * `Σ 1 / (k + rank)` over the rankings it appears in, with rank counted from 1 (the hits' own scores
 * are ignored). Return every document that appears anywhere as `[{ index, score }]`, best first,
 * ties by index.
 */
export function reciprocalRankFusion(rankings, { k = 60 } = {}) {
  // TODO: step 6
  return [];
}
