// The pre-training data pipeline — reference solution: the pre-training data pipeline.
// A document is { id, text, domain }. Every stage is a pure function that returns what it kept,
// what it removed, and why, so the final report is a set of counts rather than a feeling.

import { rng, shuffle, hash32 } from 'lib/util.js';

// ---------- worked examples (shared conventions) ----------

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

const TERMINAL_RE = /[.!?"'”’)]\s*$/;

/**
 * The first quality rule the text violates, as a short reason string, or null if it passes all of them.
 * Rules, in order: too_short, too_long, word_length, symbol_ratio, terminal_punct, repeated_lines, boilerplate.
 */
export function qualityReason(text, cfg = QUALITY_DEFAULTS) {
  const ws = words(text);
  const n = ws.length;
  if (n < cfg.minWords) return 'too_short';
  if (n > cfg.maxWords) return 'too_long';
  let chars = 0;
  for (const w of ws) chars += w.length;
  const meanLen = chars / n;
  if (meanLen < cfg.minMeanWordLength || meanLen > cfg.maxMeanWordLength) return 'word_length';
  const symbols = (String(text).match(/#|…|\.\.\./g) || []).length;
  if (symbols / n > cfg.maxSymbolRatio) return 'symbol_ratio';
  const ls = lines(text);
  let terminal = 0;
  for (const l of ls) if (TERMINAL_RE.test(l)) terminal++;
  if (terminal / ls.length < cfg.minTerminalFraction) return 'terminal_punct';
  const distinct = new Set(ls).size;
  if ((ls.length - distinct) / ls.length > cfg.maxDuplicateLineFraction) return 'repeated_lines';
  const lower = String(text).toLowerCase();
  for (const phrase of cfg.boilerplate) if (lower.includes(phrase)) return 'boilerplate';
  return null;
}

/** Split documents into { kept, removed: [{ id, reason }] }. */
export function qualityFilter(docs, cfg = QUALITY_DEFAULTS) {
  const kept = [], removed = [];
  for (const doc of docs) {
    const reason = qualityReason(doc.text, cfg);
    if (reason === null) kept.push(doc);
    else removed.push({ id: doc.id, reason });
  }
  return { kept, removed };
}

// ---------- step 2: exact deduplication ----------

/** Lowercase, drop everything that is not a letter or digit, collapse whitespace. */
export function normalizeText(text) {
  return String(text).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

/** 32-bit fingerprint of the normalised text (FNV-1a from lib/util.js). */
export function docHash(text) {
  return hash32(normalizeText(text));
}

/** Keep the first document with each fingerprint: { kept, removed: [{ id, dupOf }] }. */
export function exactDedup(docs) {
  const seen = new Map();
  const kept = [], removed = [];
  for (const doc of docs) {
    const h = docHash(doc.text);
    const first = seen.get(h);
    if (first === undefined) { seen.set(h, doc.id); kept.push(doc); }
    else removed.push({ id: doc.id, dupOf: first });
  }
  return { kept, removed };
}

// ---------- step 3: near-deduplication with MinHash ----------

/** The set of k-word shingles of the normalised text. Fewer than k words → one shingle of all of them. */
export function shingles(text, k = 5) {
  const ws = normalizeText(text).split(' ').filter((w) => w.length > 0);
  const out = new Set();
  if (ws.length === 0) return out;
  if (ws.length <= k) { out.add(ws.join(' ')); return out; }
  for (let i = 0; i + k <= ws.length; i++) out.add(ws.slice(i, i + k).join(' '));
  return out;
}

/** Exact Jaccard similarity |A ∩ B| / |A ∪ B| of two sets. */
export function jaccard(a, b) {
  if (a.size === 0 && b.size === 0) return 1;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  return inter / (a.size + b.size - inter);
}

/** numHashes seeded linear hash functions h_i(x) = (a_i * x + b_i) mod 2^32, a_i odd. */
export function hashFamily(numHashes = 32, seed = 1) {
  const next = rng(seed);
  const fam = [];
  for (let i = 0; i < numHashes; i++) {
    const a = (Math.floor(next() * 4294967296) | 1) >>> 0;
    const b = Math.floor(next() * 4294967296) >>> 0;
    fam.push({ a, b });
  }
  return fam;
}

/** MinHash signature: for each hash function, the smallest hash over the set's shingles. */
export function minhash(shingleSet, numHashes = 32, seed = 1) {
  const fam = hashFamily(numHashes, seed);
  const sig = new Array(numHashes).fill(4294967295);
  for (const s of shingleSet) {
    const h = hash32(s);
    for (let i = 0; i < numHashes; i++) {
      const v = (Math.imul(fam[i].a, h) + fam[i].b) >>> 0;
      if (v < sig[i]) sig[i] = v;
    }
  }
  return sig;
}

/** Fraction of positions where two signatures agree: an unbiased estimate of the Jaccard similarity. */
export function estimateJaccard(sigA, sigB) {
  if (sigA.length !== sigB.length) throw new Error(`signature lengths differ: ${sigA.length} vs ${sigB.length}`);
  let same = 0;
  for (let i = 0; i < sigA.length; i++) if (sigA[i] === sigB[i]) same++;
  return same / sigA.length;
}

/**
 * Remove documents whose estimated Jaccard similarity to an earlier kept document is >= threshold.
 * Returns { kept, removed: [{ id, nearOf, estimate }], pairs: [{ a, b, estimate, jaccard }] } where
 * `pairs` lists every pair with estimate >= minReport (removed or not), with the exact Jaccard for comparison.
 */
export function nearDedup(docs, { k = 5, numHashes = 32, threshold = 0.8, seed = 1, minReport = 0.25 } = {}) {
  const sets = docs.map((d) => shingles(d.text, k));
  const sigs = sets.map((s) => minhash(s, numHashes, seed));
  const kept = [], removed = [], pairs = [];
  const keptIdx = [];
  for (let j = 0; j < docs.length; j++) {
    let hit = null;
    for (const i of keptIdx) {
      const est = estimateJaccard(sigs[i], sigs[j]);
      if (est >= minReport) pairs.push({ a: docs[i].id, b: docs[j].id, estimate: est, jaccard: jaccard(sets[i], sets[j]) });
      if (est >= threshold && hit === null) hit = { id: docs[j].id, nearOf: docs[i].id, estimate: est };
    }
    if (hit === null) { kept.push(docs[j]); keptIdx.push(j); }
    else removed.push(hit);
  }
  pairs.sort((x, y) => y.estimate - x.estimate || y.jaccard - x.jaccard);
  return { kept, removed, pairs };
}

// ---------- step 4: domain mixing with epoch caps ----------

/**
 * Interleave documents so the running share of each domain tracks `weights` (Megatron-LM's blending rule:
 * always draw from the domain furthest below its target). A domain is drawn at most maxEpochs times over,
 * each epoch in a fresh seeded shuffle; once exhausted, its weight is redistributed to the others.
 * Returns { order: docId[], total, report: { [domain]: { docs, draws, epochs, size, share, weight } } }.
 */
export function mixDomains(docs, { weights, budget = Infinity, maxEpochs = 1, seed = 1, size = wordCount } = {}) {
  if (!weights || typeof weights !== 'object') throw new Error('mixDomains: weights {domain: number} are required');
  const next = rng(seed);
  const state = [];
  for (const domain of Object.keys(weights)) {
    if (!(weights[domain] > 0)) continue;
    const mine = docs.filter((d) => d.domain === domain);
    if (mine.length === 0) continue;
    state.push({ domain, weight: weights[domain], docs: mine, queue: [], ptr: 0, draws: 0, epochs: 0, size: 0 });
  }
  const order = [];
  let total = 0;
  while (total < budget) {
    const active = state.filter((s) => s.draws < maxEpochs * s.docs.length);
    if (active.length === 0) break;
    let sumW = 0;
    for (const s of active) sumW += s.weight;
    let best = null, bestDeficit = -Infinity;
    for (const s of active) {
      const deficit = (s.weight / sumW) * total - s.size;
      if (deficit > bestDeficit) { bestDeficit = deficit; best = s; }
    }
    if (best.ptr >= best.queue.length) { best.queue = shuffle(next, best.docs.slice()); best.ptr = 0; best.epochs++; }
    const doc = best.queue[best.ptr++];
    const n = size(doc);
    best.draws++;
    best.size += n;
    total += n;
    order.push(doc.id);
  }
  const report = {};
  for (const s of state) {
    report[s.domain] = { docs: s.docs.length, draws: s.draws, epochs: s.draws / s.docs.length, size: s.size, share: total > 0 ? s.size / total : 0, weight: s.weight };
  }
  return { order, total, report };
}

// ---------- step 5: tokenise, shard, and the whole pipeline ----------

/** Attach token ids to each document: { ...doc, ids }. Independent per document, so shardable across workers. */
export function tokenizeDocs(docs, tokenizer) {
  return docs.map((d) => ({ ...d, ids: tokenizer.encode(d.text) }));
}

/**
 * Concatenate the documents in `order` (each followed by one `eos` token) and cut the stream into
 * shards of exactly shardSize tokens (the last may be shorter). The index records where each
 * document starts: { docId, shard, offset, length } with length = the document's own token count.
 */
export function packShards(docs, order, { shardSize = 1024, eos } = {}) {
  if (!(shardSize > 0)) throw new Error('packShards: shardSize must be positive');
  if (typeof eos !== 'number') throw new Error('packShards: an eos token id is required');
  const byId = new Map(docs.map((d) => [d.id, d]));
  const shards = [];
  const index = [];
  let cur = [];
  const push = (t) => { cur.push(t); if (cur.length === shardSize) { shards.push(cur); cur = []; } };
  for (const id of order) {
    const doc = byId.get(id);
    if (!doc) throw new Error(`packShards: unknown document id ${id}`);
    index.push({ docId: id, shard: shards.length, offset: cur.length, length: doc.ids.length });
    for (const t of doc.ids) push(t);
    push(eos);
  }
  if (cur.length > 0) shards.push(cur);
  return { shards, index };
}

/**
 * The whole pipeline: quality → exact dedup → near dedup → tokenise → mix → shard.
 * Returns { shards, index, tokens, report, removed, pairs, mix }.
 */
export function runPipeline(docs, config) {
  const { tokenizer, quality = QUALITY_DEFAULTS, near = {}, mix = {}, shardSize = 1024 } = config || {};
  if (!tokenizer) throw new Error('runPipeline: config.tokenizer is required');
  const report = [];
  const removed = [];
  const q = qualityFilter(docs, quality);
  report.push({ stage: 'quality', in: docs.length, out: q.kept.length, removed: q.removed.length });
  for (const r of q.removed) removed.push({ id: r.id, stage: 'quality', reason: r.reason });
  const e = exactDedup(q.kept);
  report.push({ stage: 'exact-dedup', in: q.kept.length, out: e.kept.length, removed: e.removed.length });
  for (const r of e.removed) removed.push({ id: r.id, stage: 'exact-dedup', reason: `duplicate of ${r.dupOf}` });
  const nd = nearDedup(e.kept, near);
  report.push({ stage: 'near-dedup', in: e.kept.length, out: nd.kept.length, removed: nd.removed.length });
  for (const r of nd.removed) removed.push({ id: r.id, stage: 'near-dedup', reason: `near-duplicate of ${r.nearOf} (est. Jaccard ${r.estimate.toFixed(2)})` });
  const toks = tokenizeDocs(nd.kept, tokenizer);
  const mixed = mixDomains(toks, { ...mix, size: (d) => d.ids.length });
  const { shards, index } = packShards(toks, mixed.order, { shardSize, eos: tokenizer.eos });
  let tokens = 0;
  for (const s of shards) tokens += s.length;
  report.push({ stage: 'mix+shard', in: toks.length, out: index.length, removed: 0 });
  return { shards, index, tokens, report, removed, pairs: nd.pairs, mix: mixed };
}
