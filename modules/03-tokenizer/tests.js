// Module 03 tests. Every text here is a literal so results are exactly reproducible.

const LOW = 'low low low lower lowest';          // Sennrich et al. (2016)'s worked example
const CATS = 'the cat sat on the mat. the cat ate the rat. the rat sat on the cat.';
const PARA = 'Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: '
  + 'once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, '
  + '"and what is the use of a book," thought Alice "without pictures or conversations?" '
  + 'So she was considering in her own mind, as well as she could, for the hot day made her feel very sleepy and stupid, '
  + 'whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, '
  + 'when suddenly a White Rabbit with pink eyes ran close by her. It was 12 minutes past 4 by her watch, or 1234 seconds past.';

export const tests = [
  // ---------- step 1: pre-tokenisation ----------
  { step: 'pretokenize', name: 'words keep their leading space; digits and punctuation are separate pre-tokens', run(m, T) {
    T.eq(m.pretokenize('Hello world'), ['Hello', ' world'], 'the space belongs to the word that follows it (GPT-2 convention)');
    T.eq(m.pretokenize('I have 12 cats!'), ['I', ' have', ' 12', ' cats', '!'], 'a number is its own pre-token and punctuation never joins a word');
  } },
  { step: 'pretokenize', name: 'letters, digits and punctuation runs split from each other; whitespace runs stay whole', run(m, T) {
    T.eq(m.pretokenize('abc123'), ['abc', '123'], 'a digit run must not merge into the letters before it');
    T.eq(m.pretokenize("don't"), ['don', "'", 't'], 'an apostrophe is punctuation, so it splits the word');
    T.eq(m.pretokenize('well-lit!!'), ['well', '-', 'lit', '!!'], 'a run of punctuation is one pre-token');
    T.eq(m.pretokenize('a -- b.'), ['a', ' --', ' b', '.'], 'punctuation takes an optional leading space too, exactly like words and numbers');
    T.eq(m.pretokenize('x  y'), ['x', '  ', 'y'], 'two spaces: the run of whitespace is one pre-token, and "y" gets no leading space');
    T.eq(m.pretokenize('a\n\nb'), ['a', '\n\n', 'b'], 'newlines are whitespace');
  } },
  { step: 'pretokenize', name: 'is lossless (pre-tokens concatenate back to the text) and returns [] for an empty string', run(m, T) {
    const text = 'Ms. O\'Neil paid $1,250.50 for 3 tickets -- "really?"\n\tYes, 100%!';
    const pre = m.pretokenize(text);
    T.ok(pre.length > 10, `expected many pre-tokens for a mixed string, got ${pre.length}`);
    T.eq(pre.join(''), text, 'every character must land in exactly one pre-token, or decode can never be exact');
    T.eq(m.pretokenize(''), [], 'String.match returns null when nothing matches; return an empty array instead');
  } },

  // ---------- step 2: pair counting ----------
  { step: 'pairs', name: 'counts every adjacent pair inside a word, including overlapping occurrences', run(m, T) {
    const c = m.countPairs([['l', 'o', 'w']]);
    T.eq(c.size, 2, 'a 3-symbol word has exactly 2 adjacent pairs');
    T.eq(c.get(m.pairKey('l', 'o')), 1, 'the pair (l, o) occurs once');
    T.eq(c.get(m.pairKey('o', 'w')), 1, 'the pair (o, w) occurs once');
    T.eq(m.countPairs([['a', 'a', 'a']]).get(m.pairKey('a', 'a')), 2, '"aaa" contains (a, a) twice: positions 0-1 and 1-2');
  } },
  { step: 'pairs', name: 'weights each word by its frequency', run(m, T) {
    const c = m.countPairs([['l', 'o', 'w'], ['l', 'o']], [3, 2]);
    T.eq(c.get(m.pairKey('l', 'o')), 5, '(l, o) appears in a word seen 3 times and in a word seen 2 times: 3 + 2');
    T.eq(c.get(m.pairKey('o', 'w')), 3, '(o, w) only appears in the first word, which occurs 3 times');
    T.eq(m.countPairs([['a', 'b']]).get(m.pairKey('a', 'b')), 1, 'without a freqs array every word counts once');
  } },
  { step: 'pairs', name: 'never pairs the last symbol of one word with the first of the next; keys use PAIR_SEP', run(m, T) {
    const c = m.countPairs([['a', 'b'], ['c', 'd']]);
    T.eq(c.get(m.pairKey('b', 'c')), undefined, '(b, c) straddles two pre-tokens and must not be counted: merges never cross pre-token boundaries');
    T.eq(c.size, 2);
    for (const key of c.keys()) T.ok(key.includes(m.PAIR_SEP), `key ${JSON.stringify(key)} must be built with pairKey so "ab"+"c" and "a"+"bc" stay distinct`);
    T.eq(m.countPairs([['a']]).size, 0, 'a one-symbol word has no pairs');
    T.eq(m.countPairs([]).size, 0);
  } },

  // ---------- step 3: the merge loop ----------
  { step: 'train', name: 'bestPair returns the most frequent pair, ties to the first-inserted key, null when empty', run(m, T) {
    const counts = new Map([[m.pairKey('a', 'b'), 2], [m.pairKey('c', 'd'), 5], [m.pairKey('e', 'f'), 5]]);
    T.eq(m.bestPair(counts), { a: 'c', b: 'd', count: 5 }, 'count 5 beats 2; (c, d) was inserted before (e, f) so it wins the tie');
    T.eq(m.bestPair(new Map()), null, 'an empty Map has no best pair');
    T.eq(m.bestPair(new Map([[m.pairKey('x', 'y'), 1]])), { a: 'x', b: 'y', count: 1 });
  } },
  { step: 'train', name: 'mergePair replaces non-overlapping occurrences left to right without mutating its input', run(m, T) {
    const s = ['a', 'b', 'c', 'a', 'b'];
    T.eq(m.mergePair(s, 'a', 'b'), ['ab', 'c', 'ab']);
    T.eq(s, ['a', 'b', 'c', 'a', 'b'], 'the input array must not be modified');
    T.eq(m.mergePair(['a', 'a', 'a'], 'a', 'a'), ['aa', 'a'], 'after merging positions 0-1 the "a" at position 1 is used up, so only one merge happens');
    T.eq(m.mergePair(['ab', 'c'], 'a', 'b'), ['ab', 'c'], 'only whole symbols match, never substrings of a symbol');
  } },
  { step: 'train', name: 'learns lo, low, ␣low, ␣lowe on "low low low lower lowest" and stops when no pair repeats', run(m, T) {
    const tok = m.BPETokenizer.train(LOW, { vocabSize: 20 });
    T.eq(tok.merges, [['l', 'o'], ['lo', 'w'], [' ', 'low'], [' low', 'e']],
      '(l,o) and (o,w) both occur 5 times; (l,o) is seen first. Then (lo,w)=5, (␣,low)=4, (␣low,e)=2; every remaining pair occurs once, so training stops');
    T.eq(tok.vocabSize, 14, '8 characters + <|unk|> + 4 merges + 1 special = 14, fewer than the 20 requested because no pair occurs twice any more');
    T.eq(tok.vocab.slice(0, 9), [' ', 'e', 'l', 'o', 'r', 's', 't', 'w', '<|unk|>'], 'ids 0-7 are the sorted distinct characters, then <|unk|>; merged symbols come after');
    T.eq(tok.vocab[tok.vocab.length - 1], '<|endoftext|>', 'specials are appended last');
    T.eq(tok.eos, 13, 'eos is the id of specials[0]');
    T.ok(tok.vocab.includes(m.UNK), 'the vocabulary must contain the <|unk|> symbol so unseen characters have an id');
  } },
  { step: 'train', name: 'fills the vocabulary to exactly vocabSize, records merges in order, and counts pairs inside pre-tokens only', run(m, T) {
    const tok = m.BPETokenizer.train(CATS, { vocabSize: 24 });
    T.eq(tok.vocabSize, 24, 'with plenty of repeated pairs the vocabulary must reach exactly vocabSize (specials included)');
    T.eq(tok.merges.length, 10, '12 characters + <|unk|> = 13 base symbols, plus 10 merges, plus 1 special = 24');
    for (let i = 0; i < tok.merges.length; i++) {
      T.eq(tok.vocab[13 + i], tok.merges[i][0] + tok.merges[i][1], `vocab entry ${13 + i} must be the ${i}-th merge's concatenation`);
    }
    const words = m.pretokenize(CATS).map((p) => [...p]);
    const best = m.bestPair(m.countPairs(words));
    T.eq(m.countPairs(words).get(m.pairKey(...tok.merges[0])), best.count, 'the first merge must be a most-frequent pair of the initial character pairs');
    let replay = m.pretokenize(CATS).map((p) => [...p]);
    for (const [a, b] of tok.merges) {
      const counts = m.countPairs(replay);
      const seen = counts.get(m.pairKey(a, b)) ?? 0;
      T.eq(seen, m.bestPair(counts).count, `replaying the merges in order on the pre-tokens, merge [${JSON.stringify(a)}, ${JSON.stringify(b)}] must be a most-frequent pair at the moment it is learned, but it occurs ${seen} times there`);
      replay = replay.map((w) => m.mergePair(w, a, b));
    }
    const ab = m.BPETokenizer.train('a b a b a b', { vocabSize: 20 });
    T.eq(ab.merges[0], [' ', 'b'], '(␣,b) occurs 3 times inside pre-tokens; (a,␣) also occurs 3 times but crosses a pre-token boundary and must not be counted');
  } },
  { step: 'train', name: 'each merge replaces every occurrence of the pair in a word, not just the first', run(m, T) {
    const tok = m.BPETokenizer.train('abab abab abab', { vocabSize: 20 });
    T.eq(tok.merges, [['a', 'b'], ['ab', 'ab'], [' ', 'abab']],
      '(a,b) occurs twice inside every "abab": after merge 0 each word must be [ab, ab] (not [ab, a, b]), so (ab,ab)=3 is next, then (␣,abab)=2');
    const replayWord = (word) => tok.merges.reduce((w, [a, b]) => m.mergePair(w, a, b), [...word]);
    T.eq(replayWord(' abab'), [' abab'], 'applying the learned merges in order with mergePair must turn the training word " abab" into one symbol');
    T.ok(tok.vocab.includes('abab') && tok.vocab.includes(' abab'), 'the vocabulary must contain the merged symbols "abab" and " abab"');
  } },

  // ---------- step 4: encode and decode ----------
  { step: 'codec', name: 'applyMerges applies the lowest-rank merge first, not the leftmost', run(m, T) {
    const ranks = new Map([[m.pairKey('b', 'c'), 0], [m.pairKey('a', 'b'), 1]]);
    T.eq(m.applyMerges(['a', 'b', 'c'], ranks), ['a', 'bc'], '(b,c) was learned first (rank 0), so it must win over the leftmost pair (a,b)');
    const ranks2 = new Map([[m.pairKey('a', 'b'), 0], [m.pairKey('ab', 'c'), 1]]);
    T.eq(m.applyMerges(['a', 'b', 'c', 'a', 'b'], ranks2), ['abc', 'ab'], 'merged symbols can take part in later merges');
    T.eq(m.applyMerges(['x', 'y'], new Map()), ['x', 'y'], 'with no applicable merge the symbols are unchanged');
    T.eq(m.applyMerges([], ranks), [], 'an empty pre-token stays empty');
  } },
  { step: 'codec', name: 'encode reproduces the training segmentation and decode inverts it', run(m, T) {
    const tok = m.BPETokenizer.train(LOW, { vocabSize: 20 });
    const id = (s) => tok.stoi.get(s);
    T.eq(tok.encode('low lower'), [id('low'), id(' lowe'), id('r')], '"low" is one symbol; " lower" becomes " lowe" + "r" because (␣low,e) was learned and (␣lowe,r) was not');
    T.eq(tok.decode([id('low'), id(' lowe'), id('r')]), 'low lower', 'decode concatenates the symbols');
    T.eq(tok.decode(tok.encode(LOW)), LOW, 'encode then decode must give back the text exactly');
    const cats = m.BPETokenizer.train(CATS, { vocabSize: 24 });
    T.eq(cats.decode(cats.encode(CATS)), CATS, 'the training text must round-trip exactly');
    const ab = m.BPETokenizer.train('abab abab abab', { vocabSize: 20 });
    T.eq(ab.encode('abab abab').map((i) => ab.vocab[i]), ['abab', ' abab'], 'the training segmentation is one token per word: encode must reproduce it');
    T.eq(tok.encode(''), [], 'empty text encodes to no ids');
  } },
  { step: 'codec', name: 'special tokens are matched first and become one id; unknown characters get unkId', run(m, T) {
    const tok = m.BPETokenizer.train(LOW, { vocabSize: 20 });
    const low = tok.stoi.get('low');
    T.eq(tok.encode('low<|endoftext|>low'), [low, tok.eos, low], 'the special must be recognised before pre-tokenisation, or it would be split into "<", "|", "endoftext", ...');
    T.eq(tok.decode([low, tok.eos, low]), 'low<|endoftext|>low');
    T.eq(tok.encode('¿'), [tok.unkId], 'a character never seen in training has no id of its own and must map to unkId (real byte-level BPE has no such case)');
    T.eq(tok.decode([low, 9999, -1, low]), 'lowlow', 'ids outside the vocabulary are skipped, not decoded as "undefined"');
    const noSpecials = m.BPETokenizer.train(LOW, { vocabSize: 20, specials: [] });
    T.eq(noSpecials.decode(noSpecials.encode('lowest')), 'lowest', 'a tokenizer without specials must still work');
  } },
  { step: 'codec', name: 'encode replays merges by rank; it is not a greedy longest match against the vocabulary', run(m, T) {
    const tok = new m.BPETokenizer({ vocab: ['a', 'b', 'c', 'bc', 'ab'], merges: [['b', 'c'], ['a', 'b']] });
    const id = (s) => tok.stoi.get(s);
    T.eq(tok.encode('abc'), [id('a'), id('bc')], '(b,c) has rank 0, so "abc" is a|bc. Longest-match-first would give ab|c, a segmentation training never produced');
    T.eq(tok.encode('abcab'), [id('a'), id('bc'), id('ab')], 'after (b,c) fires there is no (a,b) left in "a bc"; the trailing "ab" still merges');
    const big = m.BPETokenizer.train(PARA, { vocabSize: 120 });
    const expected = [];
    for (const pre of m.pretokenize(PARA)) for (const s of m.applyMerges([...pre], big.ranks)) expected.push(big.stoi.get(s));
    T.eq(big.encode(PARA), expected, 'encode must be exactly: pretokenize, split into characters, applyMerges by rank, look up ids');
  } },
  { step: 'codec', name: 'merges are applied inside pre-tokens only', run(m, T) {
    const tok = new m.BPETokenizer({ vocab: ['!', 'a', 'c', 's', 't', 's!'], merges: [['s', '!']] });
    const id = (s) => tok.stoi.get(s);
    T.eq(tok.encode('cats!'), [id('c'), id('a'), id('t'), id('s'), id('!')], '"cats" and "!" are different pre-tokens, so the (s,!) merge must not fire even though it is in the merge table');
    T.eq(tok.decode(tok.encode('cats!')), 'cats!');
  } },

  // ---------- step 5: metrics ----------
  { step: 'metrics', name: 'charsPerToken is characters divided by tokens', run(m, T) {
    const tok = m.BPETokenizer.train(LOW, { vocabSize: 20 });
    T.close(m.charsPerToken(tok, 'low lower'), 3, 1e-9, '9 characters became 3 tokens (low, ␣lowe, r): 9 / 3 = 3, not tokens / chars');
    const chars = m.BPETokenizer.train(LOW, { vocabSize: 10 });
    T.eq(chars.merges.length, 0, 'vocabSize 10 leaves no room for merges: 8 characters + <|unk|> + 1 special');
    T.close(m.charsPerToken(chars, LOW), 1, 1e-9, 'with no merges every token is one character');
    const big = m.BPETokenizer.train(PARA, { vocabSize: 120 });
    T.close(m.charsPerToken(big, PARA), PARA.length / big.encode(PARA).length, 1e-9);
  } },
  { step: 'metrics', name: 'roundTripExact is true for in-vocabulary text and false when a character is unknown', run(m, T) {
    const tok = m.BPETokenizer.train(LOW, { vocabSize: 20 });
    T.eq(m.roundTripExact(tok, 'lower lowest'), true, 'every character is in the vocabulary, so decode(encode(text)) === text');
    T.eq(m.roundTripExact(tok, 'low ¿'), false, '"¿" encodes to <|unk|>, which decodes as the literal text "<|unk|>", so the round trip is not exact');
    T.eq(m.roundTripExact(tok, ''), true);
  } },
  { step: 'metrics', name: 'compressionCurve trains one tokenizer per size; the ratio never decreases with vocabulary', run(m, T) {
    const curve = m.compressionCurve(LOW, [10, 12, 14]);
    T.eq(curve.length, 3, 'one entry per requested vocabulary size');
    T.eq(curve.map((c) => c.vocabSize), [10, 12, 14]);
    T.eq(curve.map((c) => c.tokens), [24, 14, 8], 'tokens needed for the 24-character text at each size');
    T.close(curve.map((c) => c.charsPerToken), [1, 24 / 14, 3], 1e-9);
    const big = m.compressionCurve(PARA, [80, 120, 160]);
    for (let i = 1; i < big.length; i++) {
      T.ok(big[i].charsPerToken >= big[i - 1].charsPerToken, `a larger vocabulary can only merge more: ${big[i - 1].charsPerToken} at ${big[i - 1].vocabSize} vs ${big[i].charsPerToken} at ${big[i].vocabSize}`);
    }
    T.ok(big[2].tokens < big[0].tokens, 'more merges must mean fewer tokens');
  } },
];
