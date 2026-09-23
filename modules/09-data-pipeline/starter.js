// The pre-training data pipeline.
// A document is { id, text, domain }. Every stage is a pure function that returns what it kept,
// what it removed, and why, so the final report is a set of counts rather than a feeling.
// Everything below the "worked examples" line is yours to implement; TODO markers name the step.

import { rng, shuffle, hash32 } from 'lib/util.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/** Whitespace-separated words, punctuation still attached: words("Hi, there!") → ["Hi,", "there!"]. */
export function words(text) {
  return String(text).split(/\s+/).filter((w) => w.length > 0);
}

/** Non-empty lines with surrounding whitespace removed. */
export function lines(text) {
  return String(text).split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
}

/** Number of words in a document's text; the default size used by the mixer. */
export function wordCount(doc) {
  return words(doc.text).length;
}

/**
 * Read one document back out of the shards using its index entry { shard, offset, length }.
 * A document may cross a shard boundary, so keep walking into the next shard until
 * `length` tokens have been read. This is what a data loader does at training time.
 */
export function readDoc(shards, entry) {
  const out = [];
  let s = entry.shard, o = entry.offset;
  while (out.length < entry.length) {
    if (o >= shards[s].length) { s++; o = 0; }
    out.push(shards[s][o++]);
  }
  return out;
}

/** numHashes seeded linear hash functions h_i(x) = (a_i * x + b_i) mod 2^32, a_i odd. (Used in step 3.) */
export function hashFamily(numHashes = 32, seed = 1) {
  const next = rng(seed);
  const fam = [];
  for (let i = 0; i < numHashes; i++) {
    const a = (Math.floor(next() * 4294967296) | 1) >>> 0;   // odd, so the map x -> a*x is a bijection mod 2^32
    const b = Math.floor(next() * 4294967296) >>> 0;
    fam.push({ a, b });
  }
  return fam;
}

/** Attach token ids to each document: { ...doc, ids }. Independent per document, so shardable across workers. (Used in step 5.) */
export function tokenizeDocs(docs, tokenizer) {
  return docs.map((d) => ({ ...d, ids: tokenizer.encode(d.text) }));
}

// ---------- step 1: quality filters ----------

export const QUALITY_DEFAULTS = {
  minWords: 20,                 // Gopher: 50; the lab's documents are shorter
  maxWords: 5000,               // Gopher: 100,000
  minMeanWordLength: 3,         // Gopher: 3
  maxMeanWordLength: 10,        // Gopher: 10
  maxSymbolRatio: 0.1,          // Gopher: hashes or ellipses per word above 0.1
  minTerminalFraction: 0.3,     // C4 keeps lines that end in terminal punctuation; we require 30% of lines
  maxDuplicateLineFraction: 0.3, // Gopher: documents with > 30% duplicate lines
  boilerplate: ['lorem ipsum', 'javascript', 'cookie policy', 'terms of service', 'privacy policy'], // C4-style
};

/** A line "ends in terminal punctuation" if its last non-space character is one of these. */
const TERMINAL_RE = /[.!?"'”’)]\s*$/;

/**
 * The first quality rule the text violates, as a short reason string, or null if it passes all of them.
 * Rules, in order: too_short, too_long, word_length, symbol_ratio, terminal_punct, repeated_lines, boilerplate.
 * The first two rules are done; complete the other five in the same style.
 */
export function qualityReason(text, cfg = QUALITY_DEFAULTS) {
  const ws = words(text);
  const n = ws.length;
  if (n < cfg.minWords) return 'too_short';
  if (n > cfg.maxWords) return 'too_long';
  // TODO: step 1 — word_length (mean characters per word outside [min, max]),
  //   symbol_ratio (count of '#', '…' and '...' divided by n, above maxSymbolRatio),
  //   terminal_punct (fraction of lines matching TERMINAL_RE below minTerminalFraction),
  //   repeated_lines ((lines − distinct lines) / lines above maxDuplicateLineFraction),
  //   boilerplate (lowercased text contains any cfg.boilerplate phrase).
  return null;
}

/** Split documents into { kept, removed: [{ id, reason }] }. */
export function qualityFilter(docs, cfg = QUALITY_DEFAULTS) {
  // TODO: step 1
  return { kept: docs, removed: [] };
}

// ---------- step 2: exact deduplication ----------

/** Lowercase, drop everything that is not a letter or digit, collapse whitespace. */
export function normalizeText(text) {
  // TODO: step 2
  return String(text);
}

/** 32-bit fingerprint of the normalised text (FNV-1a from lib/util.js). */
export function docHash(text) {
  // TODO: step 2
  return 0;
}

/** Keep the first document with each fingerprint: { kept, removed: [{ id, dupOf }] }. */
export function exactDedup(docs) {
  // TODO: step 2
  return { kept: docs, removed: [] };
}

// ---------- step 3: near-deduplication with MinHash ----------

/** The set of k-word shingles of the normalised text. Fewer than k words → one shingle of all of them. */
export function shingles(text, k = 5) {
  // TODO: step 3
  return new Set();
}

/** Exact Jaccard similarity |A ∩ B| / |A ∪ B| of two sets. */
export function jaccard(a, b) {
  // TODO: step 3
  return 0;
}

/** MinHash signature: for each function in hashFamily(numHashes, seed), the smallest hash over the set's shingles. */
export function minhash(shingleSet, numHashes = 32, seed = 1) {
  // TODO: step 3
  return new Array(numHashes).fill(0);
}

/** Fraction of positions where two signatures agree: an unbiased estimate of the Jaccard similarity. */
export function estimateJaccard(sigA, sigB) {
  // TODO: step 3
  return 0;
}

/**
 * Remove documents whose estimated Jaccard similarity to an earlier kept document is >= threshold.
 * Returns { kept, removed: [{ id, nearOf, estimate }], pairs: [{ a, b, estimate, jaccard }] } where
 * `pairs` lists every pair with estimate >= minReport (removed or not), with the exact Jaccard for comparison.
 */
export function nearDedup(docs, { k = 5, numHashes = 32, threshold = 0.8, seed = 1, minReport = 0.25 } = {}) {
  // TODO: step 3
  return { kept: docs, removed: [], pairs: [] };
}

// ---------- step 4: domain mixing with epoch caps ----------

/**
 * Interleave documents so the running share of each domain tracks `weights` (Megatron-LM's blending rule:
 * always draw from the domain furthest below its target). A domain is drawn at most maxEpochs times over,
 * each epoch in a fresh seeded shuffle; once exhausted, its weight is redistributed to the others.
 * Returns { order: docId[], total, report: { [domain]: { docs, draws, epochs, size, share, weight } } }.
 */
export function mixDomains(docs, { weights, budget = Infinity, maxEpochs = 1, seed = 1, size = wordCount } = {}) {
  // TODO: step 4
  return { order: docs.map((d) => d.id), total: 0, report: {} };
}

// ---------- step 5: shards and the whole pipeline ----------

/**
 * Concatenate the documents in `order` (each followed by one `eos` token) and cut the stream into
 * shards of exactly shardSize tokens (the last may be shorter). The index records where each
 * document starts: { docId, shard, offset, length } with length = the document's own token count.
 */
export function packShards(docs, order, { shardSize = 1024, eos } = {}) {
  // TODO: step 5
  return { shards: [], index: [] };
}

/**
 * The whole pipeline: quality → exact dedup → near dedup → tokenise → mix → shard.
 * config = { tokenizer, mix: { weights, ... }, quality?, near?, shardSize? }; tokenizer and mix.weights are required.
 * Returns { shards, index, tokens, report: [{ stage, in, out, removed }], removed: [{ id, stage, reason }], pairs, mix }.
 */
export function runPipeline(docs, config) {
  // TODO: step 5
  return { shards: [], index: [], tokens: 0, report: [], removed: [], pairs: [], mix: null };
}
