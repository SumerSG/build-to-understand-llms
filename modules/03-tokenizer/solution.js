// Module 03 — reference solution. The same algorithm lives in lib/tokenizer.js, where the merge loop
// updates pair counts incrementally instead of re-counting after every merge.

/** GPT-2-style pre-tokenisation: words, numbers and punctuation runs (each with an optional leading space), or whitespace runs. */
export const PRETOKEN_RE = / ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+/g;

/** The symbol every character outside the trained vocabulary encodes to. */
export const UNK = '<|unk|>';

/** Joins the two halves of a pair key. U+0001 is a control character that never occurs in normal text. */
export const PAIR_SEP = '\u0001';

/** Map key for the adjacent pair (a, b). */
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

/** Split text into pre-tokens whose concatenation is exactly the text. Merges never cross pre-token boundaries. */
export function pretokenize(text) {
  return text.match(PRETOKEN_RE) ?? [];
}

/**
 * Count adjacent symbol pairs inside each word. `words` is an array of symbol arrays; `freqs[w]`
 * (default 1) is how many times word w occurs, so identical pre-tokens can be counted once.
 */
export function countPairs(words, freqs = null) {
  const counts = new Map();
  for (let w = 0; w < words.length; w++) {
    const symbols = words[w];
    const n = freqs ? freqs[w] : 1;
    for (let i = 0; i + 1 < symbols.length; i++) {
      const key = pairKey(symbols[i], symbols[i + 1]);
      counts.set(key, (counts.get(key) ?? 0) + n);
    }
  }
  return counts;
}

/** The most frequent pair as { a, b, count }, ties broken by insertion order (first seen wins); null if the Map is empty. */
export function bestPair(counts) {
  let bestKey = null;
  let bestCount = 0;
  for (const [key, count] of counts) {
    if (count > bestCount) {
      bestKey = key;
      bestCount = count;
    }
  }
  if (bestKey === null) return null;
  const [a, b] = splitKey(bestKey);
  return { a, b, count: bestCount };
}

/** A new array in which every non-overlapping occurrence of (a, b), scanned left to right, is replaced by a + b. */
export function mergePair(symbols, a, b) {
  const merged = a + b;
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

/** Apply learned merges to one symbol array: always the adjacent pair with the lowest rank first, until none applies. */
export function applyMerges(symbols, ranks) {
  let out = symbols.slice();
  while (out.length > 1) {
    let bestRank = Infinity;
    let bestAt = -1;
    for (let i = 0; i + 1 < out.length; i++) {
      const rank = ranks.get(pairKey(out[i], out[i + 1]));
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestAt = i;
      }
    }
    if (bestAt < 0) break;
    out = mergePair(out, out[bestAt], out[bestAt + 1]);
  }
  return out;
}

/** Split a string into alternating ordinary text and whole special tokens: [{ text, special }]. */
export function splitSpecials(str, specials) {
  if (!specials.length) return [{ text: str, special: false }];
  const byLength = specials.slice().sort((x, y) => y.length - x.length); // longest first
  const re = new RegExp(`(${byLength.map(escapeRegExp).join('|')})`);
  const parts = str.split(re); // one capture group: odd indices are the specials
  const out = [];
  for (let i = 0; i < parts.length; i++) {
    if (parts[i] !== '') out.push({ text: parts[i], special: i % 2 === 1 });
  }
  return out;
}

/** Byte-pair encoding over characters: a learned vocabulary of merged character sequences. */
export class BPETokenizer {
  constructor({ vocab = [], merges = [], specials = [] } = {}) {
    this.vocab = vocab.slice();
    this.merges = merges.map(([a, b]) => [a, b]);
    this.specials = specials.slice();
    if (!this.vocab.includes(UNK)) this.vocab.push(UNK);
    this.stoi = new Map(this.vocab.map((symbol, i) => [symbol, i]));
    this.ranks = new Map(this.merges.map(([a, b], i) => [pairKey(a, b), i]));
    this.unkId = this.stoi.get(UNK);
    this.eos = this.specials.length ? this.stoi.get(this.specials[0]) : -1;
  }

  get vocabSize() {
    return this.vocab.length;
  }

  static train(text, { vocabSize = 256, specials = ['<|endoftext|>'] } = {}) {
    const vocab = [...new Set(text)].sort();
    vocab.push(UNK);
    const merges = [];

    const occurrences = new Map();
    for (const pre of pretokenize(text)) occurrences.set(pre, (occurrences.get(pre) ?? 0) + 1);
    const words = [];
    const freqs = [];
    for (const [pre, n] of occurrences) {
      words.push([...pre]);
      freqs.push(n);
    }

    const budget = vocabSize - specials.length;
    while (vocab.length < budget) {
      const best = bestPair(countPairs(words, freqs));
      if (best === null || best.count < 2) break;
      for (let w = 0; w < words.length; w++) words[w] = mergePair(words[w], best.a, best.b);
      merges.push([best.a, best.b]);
      vocab.push(best.a + best.b);
    }
    return new BPETokenizer({ vocab: [...vocab, ...specials], merges, specials });
  }

  encode(str) {
    const ids = [];
    for (const chunk of splitSpecials(str, this.specials)) {
      if (chunk.special) {
        ids.push(this.stoi.get(chunk.text));
        continue;
      }
      for (const pre of pretokenize(chunk.text)) {
        for (const symbol of applyMerges([...pre], this.ranks)) ids.push(this.stoi.get(symbol) ?? this.unkId);
      }
    }
    return ids;
  }

  decode(ids) {
    let out = '';
    for (const id of ids) {
      const symbol = this.vocab[id];
      if (symbol !== undefined) out += symbol;
    }
    return out;
  }

  toJSON() {
    return { type: 'bpe', vocab: this.vocab.slice(), merges: this.merges.map(([a, b]) => [a, b]), specials: this.specials.slice() };
  }

  static fromJSON(o) {
    return new BPETokenizer({ vocab: o.vocab, merges: o.merges, specials: o.specials });
  }
}

/** Compression ratio: characters of `text` per token after encoding. 1.0 is a character tokenizer. */
export function charsPerToken(tokenizer, text) {
  const ids = tokenizer.encode(text);
  return ids.length ? text.length / ids.length : 0;
}

/** True when decode(encode(text)) reproduces `text` exactly. */
export function roundTripExact(tokenizer, text) {
  return tokenizer.decode(tokenizer.encode(text)) === text;
}

/** Train one tokenizer per vocabulary size on `text` and measure each: [{ vocabSize, tokens, charsPerToken }]. */
export function compressionCurve(text, vocabSizes) {
  return vocabSizes.map((vocabSize) => {
    const tokenizer = BPETokenizer.train(text, { vocabSize });
    const tokens = tokenizer.encode(text).length;
    return { vocabSize, tokens, charsPerToken: charsPerToken(tokenizer, text) };
  });
}
