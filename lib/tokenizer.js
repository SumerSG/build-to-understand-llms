// lib/tokenizer.js — character-level and byte-pair-encoding tokenizers.
// This file is the REFERENCE SOLUTION for module 03 (Tokenizer).
//
// A tokenizer is a lossless compression scheme with a fixed dictionary: `encode` turns text into a list
// of integer ids, `decode` turns the ids back into exactly the same text. CharTokenizer uses one id per
// character (tiny vocabulary, long sequences). BPETokenizer learns a vocabulary: it starts from single
// characters and repeatedly merges the most frequent adjacent pair of symbols, trading a bigger
// vocabulary for shorter sequences.
//
// Honest simplifications (see docs/CURRICULUM_BRIEFS.md, 03-tokenizer): the base vocabulary is the set of
// characters in the training text, not the 256 bytes real byte-level BPE uses, so a character that never
// appeared in training has no id of its own — it encodes to the dedicated '<|unk|>' id and decodes back as
// the literal text '<|unk|>' (that one case is not round-trip exact, and it is the only one).

/** GPT-2-style pre-tokenisation: words, numbers and punctuation runs, each keeping a leading space. */
export const PRETOKEN_RE = / ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+/g;

/** The symbol every character outside the trained vocabulary encodes to. */
export const UNK = '<|unk|>';

// Pair keys are 'left\u0001right'. U+0001 is a control character that never occurs in normal text, so it
// cannot be confused with the symbols themselves.
const PAIR_SEP = '\u0001';

/** Split text into pre-tokens; merges are only ever learned or applied inside one pre-token. */
export function pretokenize(text) {
  return text.match(PRETOKEN_RE) ?? [];
}

/** One id per distinct character of the training text; the simplest tokenizer that round-trips. */
export class CharTokenizer {
  /** Build the vocabulary from `text`: its distinct characters, sorted so training is deterministic. */
  constructor(text = '') {
    this.itos = [...new Set(text)].sort();
    this.stoi = new Map(this.itos.map((ch, i) => [ch, i]));
  }

  get vocabSize() {
    return this.itos.length;
  }

  /** Text -> ids. Characters outside the vocabulary are skipped (this tokenizer has no unknown id). */
  encode(str) {
    const ids = [];
    for (const ch of str) {
      const id = this.stoi.get(ch);
      if (id !== undefined) ids.push(id);
    }
    return ids;
  }

  /** Ids -> text. Ids outside the vocabulary are skipped. */
  decode(ids) {
    let out = '';
    for (const id of ids) {
      const ch = this.itos[id];
      if (ch !== undefined) out += ch;
    }
    return out;
  }

  /** Plain-object form, safe to JSON.stringify into a checkpoint. */
  toJSON() {
    return { type: 'char', itos: this.itos.slice() };
  }

  /** Rebuild a CharTokenizer from toJSON() output (no re-scan of the training text). */
  static fromJSON(o) {
    const tokenizer = new CharTokenizer('');
    tokenizer.itos = o.itos.slice();
    tokenizer.stoi = new Map(tokenizer.itos.map((ch, i) => [ch, i]));
    return tokenizer;
  }
}

/** Add `delta` (may be negative) to the count of every adjacent pair in `symbols`. */
function addPairs(pairCounts, symbols, delta) {
  for (let i = 0; i + 1 < symbols.length; i++) {
    const key = symbols[i] + PAIR_SEP + symbols[i + 1];
    const n = (pairCounts.get(key) ?? 0) + delta;
    if (n > 0) pairCounts.set(key, n);
    else pairCounts.delete(key);
  }
}

/** The most frequent pair, ties broken by the order the pairs were first counted. */
function bestPair(pairCounts) {
  let bestKey = null;
  let bestCount = 0;
  for (const [key, count] of pairCounts) {
    if (count > bestCount) {
      bestKey = key;
      bestCount = count;
    }
  }
  if (bestKey === null) return null;
  const at = bestKey.indexOf(PAIR_SEP);
  return { a: bestKey.slice(0, at), b: bestKey.slice(at + 1), count: bestCount };
}

/** True if `a` is immediately followed by `b` anywhere in `symbols`. */
function hasPair(symbols, a, b) {
  for (let i = 0; i + 1 < symbols.length; i++) {
    if (symbols[i] === a && symbols[i + 1] === b) return true;
  }
  return false;
}

/** Replace every occurrence of the adjacent pair (a, b) in `symbols` by the single symbol `merged`. */
function mergePair(symbols, a, b, merged) {
  const out = [];
  for (let i = 0; i < symbols.length; i++) {
    if (i + 1 < symbols.length && symbols[i] === a && symbols[i + 1] === b) {
      out.push(merged);
      i++;
    } else {
      out.push(symbols[i]);
    }
  }
  return out;
}

/** Escape a string so it can be dropped into a RegExp as a literal. */
function escapeRegExp(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Byte-pair encoding: a learned vocabulary of merged character sequences. */
export class BPETokenizer {
  /** Build from an already-learned vocabulary; use BPETokenizer.train() to learn one. */
  constructor({ vocab = [], merges = [], specials = [] } = {}) {
    this.vocab = vocab.slice();
    this.merges = merges.map(([a, b]) => [a, b]);
    this.specials = specials.slice();
    if (!this.vocab.includes(UNK)) this.vocab.push(UNK); // appended, so existing ids keep their meaning
    this._index();
  }

  /** (Re)build the lookup tables derived from vocab/merges/specials. */
  _index() {
    this.stoi = new Map(this.vocab.map((symbol, i) => [symbol, i]));
    // Merge order is merge priority: rank 0 was learned first and is applied first.
    this.ranks = new Map(this.merges.map(([a, b], i) => [a + PAIR_SEP + b, i]));
    this.unkId = this.stoi.get(UNK);
    this.eos = this.specials.length ? this.stoi.get(this.specials[0]) : -1;
    // Longest first, so '<|endoftext|>' wins over a special that is a prefix of it.
    const byLength = this.specials.slice().sort((a, b) => b.length - a.length);
    this._specialRe = byLength.length ? new RegExp(`(${byLength.map(escapeRegExp).join('|')})`) : null;
    this._cache = new Map(); // pre-token -> ids, so a repeated word is only merged once
  }

  get vocabSize() {
    return this.vocab.length;
  }

  /**
   * Learn merges from `text`: base vocabulary = its distinct characters, then greedily merge the most
   * frequent adjacent pair until the vocabulary is full or no pair occurs twice. Special tokens are
   * appended last, so specials[0] (the end-of-text token) keeps a stable id near the top of the range.
   */
  static train(text, { vocabSize = 256, specials = ['<|endoftext|>'] } = {}) {
    const vocab = [...new Set(text)].sort();
    vocab.push(UNK);
    const merges = [];

    // Identical pre-tokens are merged identically, so count each distinct one once with a multiplicity.
    const occurrences = new Map();
    for (const pre of pretokenize(text)) occurrences.set(pre, (occurrences.get(pre) ?? 0) + 1);
    const words = [];
    const freq = [];
    for (const [pre, n] of occurrences) {
      words.push([...pre]);
      freq.push(n);
    }

    const pairCounts = new Map();
    for (let w = 0; w < words.length; w++) addPairs(pairCounts, words[w], freq[w]);

    const budget = vocabSize - specials.length;
    while (vocab.length < budget) {
      const best = bestPair(pairCounts);
      if (best === null || best.count < 2) break;
      const merged = best.a + best.b;
      // Update only the words that contain the pair, and only their own pair counts: this keeps the
      // whole training loop linear in the corpus per merge instead of re-counting everything.
      for (let w = 0; w < words.length; w++) {
        if (!hasPair(words[w], best.a, best.b)) continue;
        addPairs(pairCounts, words[w], -freq[w]);
        words[w] = mergePair(words[w], best.a, best.b, merged);
        addPairs(pairCounts, words[w], freq[w]);
      }
      merges.push([best.a, best.b]);
      vocab.push(merged);
    }
    return new BPETokenizer({ vocab: [...vocab, ...specials], merges, specials });
  }

  /** Split a string into alternating ordinary text and whole special tokens. */
  _splitSpecials(str) {
    if (this._specialRe === null) return [{ text: str, special: false }];
    // String.split with one capture group keeps the separators: odd indices are the specials.
    const parts = str.split(this._specialRe);
    const out = [];
    for (let i = 0; i < parts.length; i++) {
      if (parts[i] !== '') out.push({ text: parts[i], special: i % 2 === 1 });
    }
    return out;
  }

  /** Apply the learned merges to one pre-token, lowest rank (learned earliest) first. */
  _encodePretoken(pre) {
    const cached = this._cache.get(pre);
    if (cached !== undefined) return cached;
    let symbols = [...pre];
    while (symbols.length > 1) {
      let bestRank = Infinity;
      let bestAt = -1;
      for (let i = 0; i + 1 < symbols.length; i++) {
        const rank = this.ranks.get(symbols[i] + PAIR_SEP + symbols[i + 1]);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestAt = i;
        }
      }
      if (bestAt < 0) break;
      const a = symbols[bestAt];
      const b = symbols[bestAt + 1];
      symbols = mergePair(symbols, a, b, a + b);
    }
    const ids = symbols.map((symbol) => this.stoi.get(symbol) ?? this.unkId);
    if (this._cache.size > 100000) this._cache.clear(); // a cap so long-running pages cannot grow forever
    this._cache.set(pre, ids);
    return ids;
  }

  /** Text -> ids. Special tokens in the text are matched first and become one id each. */
  encode(str) {
    const ids = [];
    for (const chunk of this._splitSpecials(str)) {
      if (chunk.special) {
        ids.push(this.stoi.get(chunk.text) ?? this.unkId);
        continue;
      }
      for (const pre of pretokenize(chunk.text)) {
        const part = this._encodePretoken(pre);
        for (const id of part) ids.push(id);
      }
    }
    return ids;
  }

  /** Ids -> text: symbols concatenated. Special ids decode to their literal text; unknown ids are skipped. */
  decode(ids) {
    let out = '';
    for (const id of ids) {
      const symbol = this.vocab[id];
      if (symbol !== undefined) out += symbol;
    }
    return out;
  }

  /** Plain-object form, safe to JSON.stringify into lib/checkpoints/tokenizer.json. */
  toJSON() {
    return {
      type: 'bpe',
      vocab: this.vocab.slice(),
      merges: this.merges.map(([a, b]) => [a, b]),
      specials: this.specials.slice(),
    };
  }

  /** Rebuild a BPETokenizer from toJSON() output (no retraining). */
  static fromJSON(o) {
    return new BPETokenizer({ vocab: o.vocab, merges: o.merges, specials: o.specials });
  }
}
