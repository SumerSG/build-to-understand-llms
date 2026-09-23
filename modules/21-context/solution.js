// Module 21 — Context management & retrieval: reference solution.
//
// The context window is a scarce, ORDERED budget of tokens. Everything here is a policy for spending
// it: count what each message costs, trim what is bloated, replace old turns by a summary, and fetch
// back from a note store only what the current turn needs. The policy, not the model, decides what
// the model can know on a given turn.

// ---------- constants and worked examples ----------

/**
 * Per-message overhead in tokens: the role marker and end-of-message framing (`<|user|>` … `<|end|>` in
 * lib/data.js's template) cost tokens on the wire, not just the text. OpenAI's cookbook counts
 * approximately 4 such tokens per chat message; the exact figure depends on the template.
 */
export const TOKENS_PER_MESSAGE = 4;

/**
 * Text that replaces the middle of a truncated tool result. A constant, so its token cost is fixed
 * (13 tokens on the lab BPE; square brackets are avoided because the toy vocabulary has none).
 */
export const TRUNCATION_MARKER = '\n(... cut ...)\n';

/** Number of tokens `text` costs under `tokenizer` (anything with `encode(str) → number[]`). */
export function countTokens(tokenizer, text) {
  if (typeof text !== 'string' || text.length === 0) return 0;
  return tokenizer.encode(text).length;
}

/** Lowercase word terms for BM25: `tokenizeTerms('The deploy key, slot 7!') → ['the','deploy','key','slot','7']`. */
export function tokenizeTerms(text) {
  return String(text ?? '').toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

// ---------- step 1: counting the cost of a context ----------

/** Tokens one message costs: the per-message overhead plus its content (plus its tool name, if any). */
export function messageTokens(tokenizer, msg) {
  return TOKENS_PER_MESSAGE + countTokens(tokenizer, msg.content) + countTokens(tokenizer, msg.name);
}

/** Tokens a whole message array costs. */
export function contextTokens(tokenizer, messages) {
  let total = 0;
  for (const msg of messages) total += messageTokens(tokenizer, msg);
  return total;
}

/** Does the array fit the budget? `{ tokens, budget, remaining, fits }` — remaining is negative when over. */
export function budgetReport(tokenizer, messages, budget) {
  const tokens = contextTokens(tokenizer, messages);
  return { tokens, budget, remaining: budget - tokens, fits: tokens <= budget };
}

// ---------- step 2: truncation policies ----------

/**
 * Cut `text` down to at most `maxTokens` tokens INCLUDING the marker, keeping the head and the tail
 * (the first half of the kept tokens and the last half) with TRUNCATION_MARKER in between.
 * Text that already fits is returned unchanged.
 */
export function truncateText(tokenizer, text, maxTokens) {
  const ids = tokenizer.encode(text);
  if (ids.length <= maxTokens) return text;
  const keep = Math.max(0, maxTokens - countTokens(tokenizer, TRUNCATION_MARKER));
  const head = Math.ceil(keep / 2);
  const tail = keep - head;
  const headText = tokenizer.decode(ids.slice(0, head));
  const tailText = tail > 0 ? tokenizer.decode(ids.slice(ids.length - tail)) : '';
  return headText + TRUNCATION_MARKER + tailText;
}

/** A copy of `messages` in which every tool result longer than `maxToolTokens` is truncated. */
export function truncateToolResults(tokenizer, messages, maxToolTokens) {
  return messages.map((msg) => {
    if (msg.role !== 'tool' || countTokens(tokenizer, msg.content) <= maxToolTokens) return msg;
    return { ...msg, content: truncateText(tokenizer, msg.content, maxToolTokens) };
  });
}

/**
 * Drop the OLDEST non-system messages until the array fits `budget`. System messages are never
 * dropped, whatever the budget: the array that comes back may still be over budget if the system
 * messages alone exceed it. Returns a new array; the input is not modified.
 */
export function dropOldest(tokenizer, messages, budget) {
  const out = messages.slice();
  let tokens = contextTokens(tokenizer, out);
  let i = 0;
  while (tokens > budget && i < out.length) {
    if (out[i].role === 'system') { i++; continue; }
    tokens -= messageTokens(tokenizer, out[i]);
    out.splice(i, 1);
  }
  return out;
}

// ---------- step 3: compaction ----------

/**
 * The scripted summariser: every line of the form `remember: <fact>` (any case, surrounding
 * whitespace ignored) yields the fact; duplicates are dropped, first occurrence wins.
 */
export function extractFacts(text) {
  const facts = [];
  const seen = new Set();
  for (const line of String(text ?? '').split('\n')) {
    const m = /^\s*remember:\s*(.+?)\s*$/i.exec(line);
    if (!m) continue;
    if (seen.has(m[1])) continue;
    seen.add(m[1]);
    facts.push(m[1]);
  }
  return facts;
}

/**
 * Replace every message older than the last `keepLast` non-system messages (and any earlier summary)
 * by ONE pinned summary message that carries the facts forward in `remember:` form, so a later
 * compaction reads them again. Returns a new array; the input is not modified.
 */
export function compact(messages, { keepLast = 4 } = {}) {
  const candidates = [];
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role !== 'system' || messages[i].summary) candidates.push(i);
  }
  const nonSummary = candidates.filter((i) => !messages[i].summary);
  const cutoff = nonSummary.length - keepLast;
  if (cutoff <= 0) return messages.slice();
  const lastSpanIndex = nonSummary[cutoff - 1];
  const span = candidates.filter((i) => i <= lastSpanIndex);
  const facts = extractFacts(span.map((i) => messages[i].content).join('\n'));
  let count = 0; // messages this summary stands for, including those an earlier summary stood for
  for (const i of span) count += messages[i].summary ? messages[i].count : 1;
  let content = `Summary of ${count} earlier messages.`;
  if (facts.length) content += '\n' + facts.map((f) => `remember: ${f}`).join('\n');
  const summary = { role: 'system', content, summary: true, count };
  const spanSet = new Set(span);
  const out = [];
  for (let i = 0; i < messages.length; i++) {
    if (i === span[0]) out.push(summary);
    if (!spanSet.has(i)) out.push(messages[i]);
  }
  return out;
}

// ---------- step 4: BM25 ----------

/** Okapi BM25 over a list of short text notes. Scores are computed per document, then ranked. */
export class BM25Index {
  constructor(docs, { k1 = 1.5, b = 0.75 } = {}) {
    this.docs = docs.slice();
    this.k1 = k1;
    this.b = b;
    this.tf = [];          // one Map term → count per document
    this.df = new Map();   // term → number of documents containing it
    this.lengths = [];
    let total = 0;
    for (const doc of this.docs) {
      const terms = tokenizeTerms(doc);
      const counts = new Map();
      for (const t of terms) counts.set(t, (counts.get(t) ?? 0) + 1);
      for (const t of counts.keys()) this.df.set(t, (this.df.get(t) ?? 0) + 1);
      this.tf.push(counts);
      this.lengths.push(terms.length);
      total += terms.length;
    }
    this.avgdl = this.docs.length ? total / this.docs.length : 0;
  }

  /** `ln(1 + (N − df + 0.5) / (df + 0.5))`: rare terms score high, terms in every document near zero. */
  idf(term) {
    const n = this.docs.length;
    const df = this.df.get(term) ?? 0;
    return Math.log(1 + (n - df + 0.5) / (df + 0.5));
  }

  /** BM25 score of document `i` for `query` (a string). */
  score(query, i) {
    const counts = this.tf[i];
    const norm = 1 - this.b + this.b * (this.lengths[i] / this.avgdl);
    let s = 0;
    for (const term of tokenizeTerms(query)) {
      const tf = counts.get(term) ?? 0;
      if (tf === 0) continue;
      s += this.idf(term) * (tf * (this.k1 + 1)) / (tf + this.k1 * norm);
    }
    return s;
  }

  /** The `k` best documents with a positive score: `[{ index, score }]`, best first, ties by index. */
  search(query, k = 3) {
    const hits = [];
    for (let i = 0; i < this.docs.length; i++) {
      const score = this.score(query, i);
      if (score > 0) hits.push({ index: i, score });
    }
    hits.sort((a, b) => b.score - a.score || a.index - b.index);
    return hits.slice(0, k);
  }
}

// ---------- step 5: the context manager ----------

/**
 * Owns the full history and a note store, and on every turn assembles the array the model will see:
 *   [system] + managed history (with a retrieval message just before the latest turn), under `budget`.
 * Policies: 'none' (raw history), 'drop' (oldest-first), 'compact' (summary + oldest-first as a last
 * resort). `topK > 0` turns on BM25 retrieval from the note store; `memory` writes every `remember:`
 * fact it sees into the store.
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
    this.history = [];
    this.notes = [];
    this.index = null;
    for (const n of notes) this.remember(n);
  }

  /** Add a note to the store (deduplicated). The BM25 index is rebuilt lazily on the next assemble. */
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

  /** The top-k notes for `query` as strings (empty when retrieval is off or nothing matches). */
  retrieve(query) {
    if (!this.topK || !this.notes.length) return [];
    if (!this.index) this.index = new BM25Index(this.notes);
    return this.index.search(query, this.topK).map((h) => this.notes[h.index]);
  }

  /** Build the context for the next model call: `{ messages, tokens, retrieved }`. */
  assemble() {
    const tok = this.tokenizer;
    let history = this.history.slice();
    const lastUser = [...history].reverse().find((m) => m.role === 'user');
    const retrieved = this.retrieve(lastUser ? lastUser.content : '');
    const fixed = [this.system];
    let retrievalMsg = null;
    if (retrieved.length) {
      retrievalMsg = { role: 'user', content: 'Relevant notes:\n' + retrieved.map((n) => `- ${n}`).join('\n'), retrieved: true };
      fixed.push(retrievalMsg);
    }
    if (this.policy !== 'none') {
      history = truncateToolResults(tok, history, this.maxToolTokens);
      const historyBudget = this.budget - contextTokens(tok, fixed);
      if (this.policy === 'compact' && contextTokens(tok, history) > historyBudget) {
        history = compact(history, { keepLast: this.keepLast });
      }
      history = dropOldest(tok, history, historyBudget);
    }
    const messages = [this.system];
    for (let i = 0; i < history.length; i++) {
      if (retrievalMsg && i === history.length - 1) messages.push(retrievalMsg);
      messages.push(history[i]);
    }
    if (retrievalMsg && history.length === 0) messages.push(retrievalMsg);
    return { messages, tokens: contextTokens(tok, messages), retrieved };
  }
}
