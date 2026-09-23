export default {
  id: '03-tokenizer',
  title: 'A BPE tokenizer',
  track: 'foundations',
  minutes: 90,
  threshold: 'A tokenizer is a learned, lossless compression scheme: it greedily merges the most frequent adjacent pair until the vocabulary is full, trading sequence length for vocabulary size.',
  goal: 'A byte-pair-encoding tokenizer trained on the corpus that round-trips text exactly and compresses it, with a plot of compression vs vocabulary size and a table of where the token boundaries fall.',
  prereqs: ['00-hello-lab', '01-tensors', '02-autograd'],
  recall: [
    { q: 'In module 00, of the *distinct* words in a passage of English prose, roughly what fraction appeared exactly once?', options: ['Almost none', 'Roughly half', 'Nearly all'], answer: 1,
      why: 'That long tail of once-only words is why no fixed word vocabulary can cover real text, and why this module builds a subword tokenizer instead.' },
    { q: 'Module 00 stored word counts in a `Map` rather than a plain object because…', options: ['Maps iterate faster', 'Word keys such as "constructor" collide with object internals', 'Objects cannot hold numbers'], answer: 1,
      why: 'You will count adjacent pairs in a Map keyed by strings here; a Map also iterates in insertion order, which this module uses to break ties.' },
    { q: 'Module 01 stored a tensor as one flat array plus a shape. A tensor of shape `[V, d]` (for example an embedding table with one row per token) holds how many numbers?', options: ['V + d', 'V × d', 'd²'], answer: 1,
      why: 'The vocabulary size V you choose here sets the size of that table: GPT-2 small spends 50,257 × 768 ≈ 38.6M of its 124M parameters on it.' },
    { q: 'In module 02, what did `backward()` do to a leaf tensor\'s `.grad` that already held a value?', options: ['Replaced it', 'Added the new gradient to it', 'Set it to 1'], answer: 1,
      why: 'Gradients accumulate, which is why the optimiser calls zeroGrad() before each backward pass. Unrelated to tokenizers, but worth keeping fresh for module 04.' },
    { q: 'In JavaScript, what does `"abc".match(/x/g)` return?', options: ['`[]`', '`null`', '`undefined`'], answer: 1,
      why: 'A global match with no hits returns null, not an empty array. `pretokenize` guards with `?? []` for exactly this reason.' },
  ],
  review: [
    { q: 'Which pair does BPE training merge next?', options: ['The pair whose merged symbol would be longest', 'The most frequent adjacent pair inside pre-tokens', 'A random pair'], answer: 1,
      why: 'Greedy frequency is the whole algorithm (Gage 1994; Sennrich et al. 2016). Each merge saves one token per occurrence, so the most frequent pair saves the most.' },
    { q: 'On the lab corpus, doubling the vocabulary from 256 to 512 reduces the token count by roughly…', options: ['50%', '28%', '5%'], answer: 1,
      why: '18,546 → 13,328 tokens. Frequencies follow Zipf, so the early merges save many tokens each and the later ones very few: diminishing returns.' },
    { q: 'At encode time, merges are applied in the order they were learned (by rank) rather than left to right because…', options: ['It is faster', 'Only that order reproduces the segmentation seen during training, so the model sees the same tokens', 'Left to right would fail to terminate'], answer: 1,
      why: 'A different order gives different tokens for the same text. Every real implementation (GPT-2, tiktoken, HF tokenizers) keeps a rank per merge and applies the lowest rank first.' },
    { q: 'Why can no input ever be out-of-vocabulary for GPT-2\'s tokenizer?', options: ['Its vocabulary contains every Unicode character', 'Its base vocabulary is the 256 byte values, and any text is a byte sequence', 'It replaces unknown words with a placeholder'], answer: 1,
      why: 'Byte-level BPE starts from 256 bytes; unusual text just gets more tokens. This module starts from the characters of the corpus instead, which is why it needs an <|unk|> id.' },
    { q: 'A language model often cannot say how many letters are in a word or reliably add two 7-digit numbers because…', options: ['It sees token ids, and the tokenizer decided where the chunk boundaries fall', 'Its context is too short', 'Attention cannot represent counting'], answer: 0,
      why: 'The model never sees characters. If "strawberry" is three tokens the letters are hidden inside them; if digits are chunked arbitrarily, place value is hidden too.' },
  ],
  concept: `
:::plain
A language model does not read letters or whole words: it reads tokens, which are chunks of text such as a common word, a piece of a rarer word, or a punctuation mark. A tokenizer is the part that cuts text into those chunks and gives each chunk a number, and the one you build here learns its chunks by repeatedly gluing together the pair of pieces that occurs most often in a sample of text. The model needs it because it can only choose from a fixed list of possible next pieces, and chunks are the middle ground between very long runs of single letters and an endless list of words. At work, tokens are the unit that hosted model services bill by, and the context window (the most text a model can take in at once) is also measured in tokens. OpenAI's rule of thumb is roughly four characters of ordinary English per token, and many other languages and long numbers need more tokens for the same content. Tokens also explain some odd failures: a model can miscount the letters in a word because it saw the word as a few chunks, never as separate letters.
:::

## Why not characters, and why not words

A language model predicts the next *token*, so the first design decision is what a token is. Both obvious choices fail.

**Characters** make sequences long. The lab corpus is 42,899 characters, so a character model needs 42,899 positions to read it; attention cost grows with the square of sequence length (module 05), and each generated token costs a forward pass.

**Words** make the vocabulary open. You measured this in module 00: about half the distinct words in a passage occur once. Any fixed word list meets unseen words (names, typos, code) on day one.

**Subwords** sit between the two. Common words become one token, rare words several pieces. The dominant algorithm for choosing the pieces is **byte-pair encoding** (BPE), a 1994 compression trick (Gage) that Sennrich, Haddow and Birch applied to translation in 2016 and that GPT-2 made standard.

## The algorithm

BPE training is a loop with one rule:

1. Start with a base vocabulary (here: every distinct character of the training text).
2. Count every adjacent pair of symbols.
3. Merge the most frequent pair into a new symbol, add it to the vocabulary, record the merge.
4. Repeat until the vocabulary has \`vocabSize\` entries or no pair occurs twice.

Each merge removes one token per occurrence of the pair, so the most frequent pair is the greedy best trade of one vocabulary slot for length. The recorded merge list *is* the tokenizer: to encode you replay the merges in the order they were learned (their **rank**); to decode you concatenate the symbols. Every symbol is a literal piece of text, so decoding is exact: a tokenizer is a **lossless compression scheme with a learned dictionary**.

:::predict
The most common English word is "the". On the lab corpus, which two characters will your tokenizer merge first?
---
\`h\` + \`e\` → \`he\` (1,785 occurrences), ahead of \`␣t\` (1,393) and \`t\` + \`h\` (1,136). \`he\` is inside "the", "The", "she", "her" and "when"; \`␣t\` misses every "␣The" (capital T is another character) and every word after a line break. Frequency, not intuition, decides.
:::

## Pre-tokenisation decides what can become a token

Before counting, the text is split into **pre-tokens** with a regular expression; merges are only counted and applied inside a pre-token. GPT-2's pattern separates words, numbers, punctuation runs and whitespace, and attaches the leading space to the word that follows it, so \`␣the\` and \`the\` are different tokens. You will write a simplified four-alternative version in step 1. Without this boundary, BPE would happily learn tokens such as \`s.␣The\`, which compress the training corpus and generalise to nothing.

## The trade-off, in numbers

A bigger vocabulary buys shorter sequences with a bigger model. GPT-2's vocabulary has 50,257 entries: 256 bytes, 50,000 merges and one \`<|endoftext|>\` token. Meta's Llama 3 report gives 128,256 (tiktoken-style byte-level BPE); DeepSeek-V3's report lists 128K. The embedding table is \`V × d\` numbers (module 01), and the output softmax is over V classes at every position, so the vocabulary is paid for at both ends of the model.

:::predict
Your tokenizer at vocabulary 256 needs 18,546 tokens for the corpus. If you double the vocabulary to 512, roughly how many tokens will it need?
---
About 13,300: a 28% reduction, not a halving. Pair frequencies follow Zipf (module 00): the first merges each remove thousands of tokens, the 400th a handful.
:::

## The tokenizer decides what the model can count

The model sees ids, not characters. If "strawberry" is \`str|aw|berry\`, the number of r's is nowhere in its input, so letter-counting questions fail. Digits are worse: GPT-2 keeps a whole digit run as one pre-token and merges inside it by frequency, so two numbers of the same length can split at different places (one as \`123|45|67\`, another as \`1|234|567\`), hiding place value. Later tokenizers (GPT-4's cl100k, Llama 3) cap number pre-tokens at three digits with a \`\\p{N}{1,3}\` rule. Even the direction matters: Singh and Strouse (2024, "Tokenization counts") found GPT-3.5 and GPT-4 add large numbers markedly more accurately when digits are grouped right to left (as commas do) than in the default left-to-right chunks. Whitespace has the same trap: because the space belongs to the *next* word, a prompt that ends in a trailing space forces the model into tokens it rarely saw in training.

## The alternative: Unigram

BPE builds bottom-up. SentencePiece's **Unigram** model (Kudo 2018) works top-down.

:::deeper Going deeper: how Unigram works and which models use it
Start from a large candidate vocabulary, fit a probability to each piece, and repeatedly prune the pieces whose removal least hurts the likelihood of the corpus. T5 and ALBERT use Unigram; Llama 1 and 2 used SentencePiece's BPE mode; Llama 3 moved to tiktoken-style byte-level BPE.
:::

## Where this toy differs from production

Your base vocabulary is the characters of the training text, so an unseen character encodes to \`<|unk|>\`; GPT-2 starts from the 256 byte values, so nothing is ever unknown and the round trip is exact for any input.

:::deeper Going deeper: how production tokenizers are engineered
Your regex knows only ASCII letters and digits, where GPT-2's uses Unicode categories (\`\\p{L}\`, \`\\p{N}\`) and handles contractions such as \`'s\` and \`'re\`. Your merge loop re-counts every pair after every merge, fine for 43 KB of text; \`lib/tokenizer.js\` updates counts incrementally, and Hugging Face's \`tokenizers\` trainer keeps pairs in a heap so gigabytes stay tractable. \`tiktoken\` encodes in Rust with the same lowest-rank-first rule over bytes that you are about to write.
:::
`,
  steps: [
    {
      id: 'pretokenize',
      title: 'Pre-tokenise with a GPT-2-style regex',
      instructions: `
Replace the placeholder \`PRETOKEN_RE\` so that \`pretokenize(text)\` (already written; note the \`?? []\` guard) splits text into pre-tokens of four kinds, tried in this order:

1. a run of ASCII letters, with an optional single leading space (\`"Hello"\`, \`" world"\`);
2. a run of digits, with an optional leading space (\`" 12"\`);
3. a run of characters that are neither whitespace, ASCII letters nor digits (punctuation and symbols), with an optional leading space (\`"!"\`, \`"--"\`), so \`"!abc"\` is two pre-tokens, not one;
4. a run of whitespace (\`"\\n\\n"\`, \`"  "\`).

The regex needs the \`g\` flag so \`match\` returns every hit. The pre-tokens must concatenate back to the input exactly, because decoding can only be lossless if pre-tokenisation is.

\`\`\`js
pretokenize('I have 12 cats!')   // → ['I', ' have', ' 12', ' cats', '!']
pretokenize('abc123')            // → ['abc', '123']
\`\`\`

Why this matters: merges are only ever learned or applied inside one pre-token, so this pattern decides what can ever become a token. The starter's \`/\\S+|\\s+/g\` glues \`cats!\` together and strips the leading space from \`world\`; the tests show you both failures.
`,
      predict: { question: 'With the finished regex, what does `pretokenize("x  y")` return (two spaces between x and y)?', answer: '`["x", "  ", "y"]`. The letter alternative allows only *one* optional leading space, so the two-space run matches the whitespace alternative and "y" arrives with no space attached. GPT-2 uses `\\s+(?!\\S)` to leave the last space for the next word; this simplified regex does not.' },
      hints: [
        'At each position, a regex alternation `A|B|C` tries its alternatives left to right and keeps the first that matches. Which four kinds of pre-token do you need, and which of them may start with a space?',
        'Four alternatives in the order listed: optional space + letters, optional space + digits, optional space + a run of characters that are none of whitespace/letters/digits (a negated class `[^…]`), then a whitespace run. "Optional single space" is ` ?`. Give the literal the `g` flag so `match` returns every hit.',
        'The shape is `/ ?[A-Za-z]+| ?\\d+| ?[^…]+|\\s+/g`. Fill the negated class with the three kinds of character a punctuation character is *not*.',
      ],
    },
    {
      id: 'pairs',
      title: 'Count adjacent pairs',
      instructions: `
Implement \`countPairs(words, freqs = null)\`. \`words\` is an array of *symbol arrays* (one per distinct pre-token, e.g. \`[['l','o','w'], ['l','o']]\`), and \`freqs[w]\` is how many times word \`w\` occurred in the text (treat a missing \`freqs\` as all ones). Return a \`Map\` from \`pairKey(a, b)\` to the total count of the adjacent pair \`(a, b)\`.

\`\`\`js
countPairs([['l','o','w'], ['l','o']], [3, 2]).get(pairKey('l','o'))   // → 5
\`\`\`

Two things the tests check: pairs are counted *inside* a word only (the last symbol of one word and the first of the next are not a pair), and overlapping occurrences all count (\`['a','a','a']\` contains \`(a, a)\` twice).

Why the key is a joined string: a \`Map\` compares arrays by identity, so \`['a','b']\` would never hit. \`pairKey\` joins the two symbols with \`PAIR_SEP\` (U+0001), a character that never occurs in text, so \`("ab", "c")\` and \`("a", "bc")\` stay distinct. This is the same counting table as module 00's \`countFrequencies\`, with pairs instead of words.
`,
      hints: [
        'Module 00\'s `countFrequencies` is the template: a Map, a read with a default, a write back. What is the thing being counted now, how much does each occurrence add, and where must a scan over one word stop so it never reads past its end?',
        'Loop over words by index `w` (you need `w` to read `freqs[w]`; no `freqs` means weight 1). Inside, loop over positions `i` while `i + 1 < length`, build the key with `pairKey` from the symbols at `i` and `i + 1`, and add the word\'s weight to that key\'s count. Never look across two words.',
        'The inner update is `counts.set(key, (counts.get(key) ?? 0) + n)` with `n = freqs ? freqs[w] : 1`; the two loops around it and the `return` are yours.',
      ],
    },
    {
      id: 'train',
      title: 'The merge loop',
      instructions: `
Three pieces, in dependency order.

\`bestPair(counts)\`: the entry with the largest count, returned as \`{ a, b, count }\` (use \`splitKey\`). Ties go to the pair that was inserted into the Map first, which \`Map\` iteration order gives you for free. Return \`null\` for an empty Map.

\`mergePair(symbols, a, b)\`: a *new* array in which every occurrence of \`a\` immediately followed by \`b\` is replaced by the single symbol \`a + b\`, scanning left to right without overlap: \`['a','a','a']\` with \`(a, a)\` becomes \`['aa', 'a']\`. Do not mutate the input.

The loop in \`BPETokenizer.train\`: the setup (base vocabulary of sorted distinct characters plus \`UNK\`, one entry per distinct pre-token with its frequency) is written. While \`vocab.length < budget\`: count pairs, take the best, stop if there is none or its count is below 2 (a pair seen once saves nothing), merge it in every word, push \`[a, b]\` onto \`merges\` and \`a + b\` onto \`vocab\`. The specials are appended after the loop, so the final vocabulary has exactly \`vocabSize\` entries when the text has enough repeated pairs.

\`\`\`js
BPETokenizer.train('low low low lower lowest', { vocabSize: 20 }).merges
// → [['l','o'], ['lo','w'], [' ','low'], [' low','e']]   (then no pair occurs twice)
\`\`\`
`,
      predict: { question: 'In "low low low lower lowest" the pairs (l,o) and (o,w) both occur 5 times. Which one is merged first, and why?', answer: '(l,o): it is the first pair encountered when scanning the first word "low", so it is inserted into the Map first, and `bestPair` only replaces the current best on a strictly larger count. Tie-breaking must be deterministic or two runs on the same text would produce different tokenizers.' },
      hints: [
        'Three questions to answer before coding. When you scan a Map in insertion order, which comparison keeps the *first* of several equal counts? After mergePair matches a pair at position i, which symbol may not take part in another match? And what two conditions end training?',
        'bestPair: one pass, replace the current best only on a strictly larger count, then `splitKey` the winner. mergePair: walk the input once, pushing `a + b` and stepping past both symbols on a match, otherwise pushing the symbol. The loop: count pairs with the frequencies, take the best, stop if there is none or it occurs fewer than 2 times, merge it in every word, then record it.',
        'Pseudo-code for the loop: `while vocab.length < budget: best ← bestPair(countPairs(words, freqs)); if no best or best.count < 2: stop; every words[w] ← mergePair(words[w], best.a, best.b); append [a, b] to merges and a + b to vocab`. bestPair and mergePair are yours.',
      ],
    },
    {
      id: 'codec',
      title: 'Encode and decode',
      instructions: `
\`applyMerges(symbols, ranks)\`: given one pre-token as a symbol array and \`ranks\` (a \`Map\` from \`pairKey\` to merge rank, built for you in the constructor as \`this.ranks\`), repeatedly find the adjacent pair with the *lowest* rank, merge it with \`mergePair\`, and stop when no adjacent pair is in the Map. Lowest rank first, not leftmost first: the training loop applied merge 0 to the whole corpus before merge 1 existed, and only the same order reproduces the same segmentation.

\`encode(str)\`: use \`splitSpecials(str, this.specials)\` (written for you) to pull whole special tokens out first; a special chunk becomes its single id from \`this.stoi\`. Each ordinary chunk is pre-tokenised, each pre-token is split into characters (\`[...pre]\`), merged with \`applyMerges\`, and each resulting symbol is looked up in \`this.stoi\`; a symbol that is missing (a character never seen in training) becomes \`this.unkId\`.

\`decode(ids)\`: concatenate \`this.vocab[id]\` for each id, skipping ids outside the vocabulary.

\`\`\`js
const tok = BPETokenizer.train('low low low lower lowest', { vocabSize: 20 });
tok.encode('low lower')            // → ids of 'low', ' lowe', 'r'
tok.decode(tok.encode('low lower')) // → 'low lower'
\`\`\`
`,
      predict: { question: 'The merge table is [["b","c"], ["a","b"]] (so (b,c) has rank 0). What does applyMerges(["a","b","c"]) return?', answer: '`["a", "bc"]`. Rank 0 wins even though (a,b) is further left. A leftmost-first implementation would return `["ab", "c"]`, a segmentation the model never saw in training.' },
      hints: [
        'Two separate ideas. applyMerges must pick the adjacent pair with the *lowest rank*, not the first one it sees, and must keep going after each merge because merged symbols can merge again. encode must pull special tokens out before anything else touches the text. What does `ranks.get(...)` return for a pair that was never learned?',
        'applyMerges: repeat { scan every adjacent pair, remember the position with the smallest defined rank; if there is none, stop; otherwise merge that pair with mergePair }. encode: for each chunk of `splitSpecials(str, this.specials)`, a special pushes its one id; ordinary text is pre-tokenised, each pre-token split into characters, merged, and each symbol looked up with a fallback to `this.unkId`. decode: concatenate the vocab entries, skipping undefined ones.',
        'The scan body is `const r = ranks.get(pairKey(out[i], out[i + 1])); if (r !== undefined && r < bestRank) { bestRank = r; bestAt = i; }`, starting from `bestRank = Infinity, bestAt = -1`; `bestAt < 0` after the scan means done. The encode lookup is `this.stoi.get(symbol) ?? this.unkId`.',
      ],
    },
    {
      id: 'metrics',
      title: 'Round-trip and compression metrics',
      instructions: `
Three small functions that turn your tokenizer into numbers you can plot.

\`charsPerToken(tokenizer, text)\`: \`text.length\` divided by the number of ids \`encode\` produces (return 0 for empty text). A character tokenizer scores exactly 1; GPT-2's byte-level tokenizer reaches roughly 4 characters per token on English. Real systems quote bytes per token; with a character-level base vocabulary, characters are the honest unit here.

\`roundTripExact(tokenizer, text)\`: \`true\` when \`decode(encode(text)) === text\`. It is false only when \`text\` contains a character the tokenizer never saw, which decodes as the literal string \`<|unk|>\`.

\`compressionCurve(text, vocabSizes)\`: train one tokenizer per size on \`text\` and return \`[{ vocabSize, tokens, charsPerToken }]\` in the same order. The demo plots this curve for the corpus.
`,
      hints: [
        'All three are short wrappers around `encode`, `decode` and `BPETokenizer.train`. Which count goes on top of the chars-per-token fraction so a character tokenizer scores exactly 1 and better compression scores higher, and what happens when there are no tokens?',
        'charsPerToken: text length divided by the number of ids (0 when there are none). roundTripExact: compare `decode(encode(text))` with `text` using `===`. compressionCurve: map over `vocabSizes`, training a fresh tokenizer for each size and measuring it on the same text.',
        'compressionCurve has the shape `vocabSizes.map((vocabSize) => { const tokenizer = BPETokenizer.train(text, { vocabSize }); … return { vocabSize, tokens, charsPerToken: … }; })`.',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why a BPE tokenizer is a compression scheme, what it trades for what, and why the merge list alone is enough to encode text you have never seen.',
    'Your compression curve flattens as the vocabulary grows. Connect that shape to Zipf\'s law from module 00, and say what it implies about choosing a vocabulary size for a model that must also learn a `V × d` embedding table.',
    'Give one concrete task a model could fail because of tokenization rather than because of its weights, and say what change to the tokenizer (not the model) would fix it.',
  ],
  stretch: [
    'Make the base vocabulary the 256 byte values (encode the text with `TextEncoder`, treat each byte as a symbol) so nothing is ever <|unk|> and the round trip is exact for any input. This is byte-level BPE as in GPT-2 and tiktoken.',
    'Replace the full re-count in the training loop with incremental updates: after a merge, subtract the old pairs of only the words that changed and add their new pairs, as `lib/tokenizer.js` does; then time both on the corpus. Hugging Face\'s `tokenizers` goes further with a heap of pair counts.',
    'Add Llama 3\'s digit rule to your regex (numbers in groups of at most three digits, `\\p{N}{1,3}` with the `u` flag) and GPT-2\'s contraction rules (`\'s`, `\'t`, `\'re`, …), then check how `1234567` and `don\'t` tokenize.',
    'Implement a Unigram trainer in the style of SentencePiece (Kudo 2018): seed the vocabulary with all frequent substrings, run EM to fit piece probabilities, prune the 20% whose removal least reduces corpus likelihood, repeat until the target size. Compare its chars-per-token with your BPE at the same vocabulary size.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
