// Module 03 — A BPE tokenizer.
// A tokenizer is a lossless codec: encode(text) -> ids, decode(ids) -> the same text. BPE learns its
// vocabulary by repeatedly merging the most frequent adjacent pair of symbols inside pre-tokens.
// Symbols are strings (a character at first, longer after merges); ids are their index in `vocab`.

// ---------- step 1: pre-tokenisation ----------

/**
 * The pre-tokenisation pattern. Merges are only ever learned or applied INSIDE one pre-token, so this
 * regex decides what can ever become a token: a word with its leading space, a run of digits, a run of
 * punctuation, or a run of whitespace.
 */
export const PRETOKEN_RE = /\S+|\s+/g; // TODO: step 1 — this naive split is wrong; see the instructions

/** Split text into pre-tokens whose concatenation is exactly the text. Done for you: note the null guard. */
export function pretokenize(text) {
  return text.match(PRETOKEN_RE) ?? [];
}

// ---------- worked examples (done for you; they set the conventions) ----------

/** The symbol every character outside the trained vocabulary encodes to. */
export const UNK = '<|unk|>';

/** Joins the two halves of a pair key. U+0001 is a control character that never occurs in normal text. */
export const PAIR_SEP = '\u0001';

/** Map key for the adjacent pair (a, b): pairKey('a', 'b') and pairKey('ab', '') are different keys. */
export function pairKey(a, b) {
  return a + PAIR_SEP + b;
}

/** Inverse of pairKey: 'a\u0001b' -> ['a', 'b']. */
export function splitKey(key) {
  const at = key.indexOf(PAIR_SEP);
  return [key.slice(0, at), key.slice(at + 1)];
}

/** Escape a string so it can be dropped into a RegExp as a literal. */
function escapeRegExp(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Split a string into alternating ordinary text and whole special tokens: [{ text, special }].
 * Longest special first, so '<|endoftext|>' beats a special that is a prefix of it.
 */
export function splitSpecials(str, specials) {
  if (!specials.length) return [{ text: str, special: false }];
  const byLength = specials.slice().sort((x, y) => y.length - x.length);
  const re = new RegExp(`(${byLength.map(escapeRegExp).join('|')})`);
  const parts = str.split(re); // one capture group: odd indices are the specials
  const out = [];
  for (let i = 0; i < parts.length; i++) {
    if (parts[i] !== '') out.push({ text: parts[i], special: i % 2 === 1 });
  }
  return out;
}

// ---------- step 2: pair counting ----------

/**
 * Count adjacent symbol pairs inside each word. `words` is an array of symbol arrays (one per distinct
 * pre-token); `freqs[w]` (default 1) is how many times word w occurs. Returns Map<pairKey, count>.
 */
export function countPairs(words, freqs = null) {
  // TODO: step 2
  return new Map();
}

// ---------- step 3: the merge loop ----------

/** The most frequent pair as { a, b, count }; ties go to the pair inserted first; null if the Map is empty. */
export function bestPair(counts) {
  // TODO: step 3
  return null;
}

/** A new array in which every non-overlapping occurrence of (a, b), scanned left to right, is replaced by a + b. */
export function mergePair(symbols, a, b) {
  // TODO: step 3
  return symbols.slice();
}

// ---------- step 4: encode and decode ----------

/**
 * Apply learned merges to one symbol array: repeatedly merge the adjacent pair with the LOWEST rank
 * (learned earliest) until no adjacent pair is in `ranks` (a Map<pairKey, rank>).
 */
export function applyMerges(symbols, ranks) {
  // TODO: step 4
  return symbols.slice();
}

/** Byte-pair encoding over characters: a learned vocabulary of merged character sequences. */
export class BPETokenizer {
  /** Build from an already-learned vocabulary; use BPETokenizer.train() to learn one. Done for you. */
  constructor({ vocab = [], merges = [], specials = [] } = {}) {
    this.vocab = vocab.slice();                       // id -> symbol
    this.merges = merges.map(([a, b]) => [a, b]);     // in the order they were learned
    this.specials = specials.slice();
    if (!this.vocab.includes(UNK)) this.vocab.push(UNK);
    this.stoi = new Map(this.vocab.map((symbol, i) => [symbol, i]));           // symbol -> id
    this.ranks = new Map(this.merges.map(([a, b], i) => [pairKey(a, b), i]));  // pair -> merge priority
    this.unkId = this.stoi.get(UNK);
    this.eos = this.specials.length ? this.stoi.get(this.specials[0]) : -1;
  }

  get vocabSize() {
    return this.vocab.length;
  }

  /**
   * Learn merges from `text`. Base vocabulary = its distinct characters (sorted) plus UNK; then merge the
   * most frequent pair until the vocabulary has vocabSize - specials.length entries or no pair occurs
   * twice; finally append the specials. The setup is written; the loop is yours.
   */
  static train(text, { vocabSize = 256, specials = ['<|endoftext|>'] } = {}) {
    const vocab = [...new Set(text)].sort();
    vocab.push(UNK);
    const merges = [];

    // Identical pre-tokens are merged identically, so keep each distinct one once with its frequency.
    const occurrences = new Map();
    for (const pre of pretokenize(text)) occurrences.set(pre, (occurrences.get(pre) ?? 0) + 1);
    const words = [];
    const freqs = [];
    for (const [pre, n] of occurrences) {
      words.push([...pre]);
      freqs.push(n);
    }

    const budget = vocabSize - specials.length;
    // TODO: step 3 — while vocab.length < budget: count pairs, pick the best, stop if none occurs
    // twice, merge it in every word, record it in `merges` and push a+b onto `vocab`.

    return new BPETokenizer({ vocab: [...vocab, ...specials], merges, specials });
  }

  /** Text -> ids. Specials are matched first (one id each); each pre-token is merged by rank, then looked up. */
  encode(str) {
    // TODO: step 4
    return [];
  }

  /** Ids -> text: the symbols concatenated. Ids outside the vocabulary are skipped. */
  decode(ids) {
    // TODO: step 4
    return '';
  }

  /** Plain-object form, safe to JSON.stringify. Done for you. */
  toJSON() {
    return { type: 'bpe', vocab: this.vocab.slice(), merges: this.merges.map(([a, b]) => [a, b]), specials: this.specials.slice() };
  }

  /** Rebuild from toJSON() output without retraining. Done for you. */
  static fromJSON(o) {
    return new BPETokenizer({ vocab: o.vocab, merges: o.merges, specials: o.specials });
  }
}

// ---------- step 5: round-trip and compression metrics ----------

/** Compression ratio: characters of `text` per token after encoding. 1.0 is a character tokenizer. */
export function charsPerToken(tokenizer, text) {
  // TODO: step 5
  return 0;
}

/** True when decode(encode(text)) reproduces `text` exactly. */
export function roundTripExact(tokenizer, text) {
  // TODO: step 5
  return false;
}

/** Train one tokenizer per vocabulary size on `text` and measure each: [{ vocabSize, tokens, charsPerToken }]. */
export function compressionCurve(text, vocabSizes) {
  // TODO: step 5
  return [];
}
