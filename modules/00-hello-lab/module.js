export default {
  id: '00-hello-lab',
  title: "Zipf's law: learn the lab loop",
  track: 'foundations',
  minutes: 20,
  threshold: 'Word frequency falls roughly as 1/rank, so a handful of words cover most of any text while a long tail of rare words never runs out, which is why language models use subword vocabularies instead of a list of words.',
  goal: 'A word-frequency counter that plots Zipf\'s law on a real passage, built through the Recall → Concept → Build → Goal → Reflect loop you will use in every module.',
  prereqs: [],
  recall: [
    { q: 'In this lab, when does a module count as complete?', options: ['When you have read the concept section', 'When all build steps pass their tests and the goal demo has run on your code', 'When you have opened every hint'], answer: 1,
      why: 'Mastery learning: the evidence of understanding is working code, not time spent reading.' },
    { q: 'Why do the hints stay locked until you have checked a step at least once?', options: ['To save bandwidth', 'Because attempting first, even unsuccessfully, improves later learning (productive failure)', 'Hints are only for the last step'], answer: 1,
      why: 'Kapur (2008): students who first wrestled with a problem unaided and were taught afterwards transferred what they learned better than students given structured help from the start.' },
    { q: 'In JavaScript, what does `map.get(key)` return when the key is absent from a `Map`?', options: ['null', '0', 'undefined'], answer: 2,
      why: 'You will write `(counts.get(w) || 0) + 1` a lot; the `|| 0` is there because a missing key gives `undefined`.' },
    { q: 'Which of these sorts an array of numbers in descending order?', options: ['`arr.sort()`', '`arr.sort((a, b) => b - a)`', '`arr.reverse()`'], answer: 1,
      why: 'Without a comparator, `sort` compares as strings ("10" < "9"). A comparator returning `b - a` puts larger numbers first.' },
  ],
  review: [
    { q: "Zipf's law says the frequency of the word with rank r is roughly proportional to…", options: ['r', '1 / r', 'log r'], answer: 1, why: 'Frequency ≈ f₁ / r: the second most common word appears about half as often as the first, the tenth about a tenth as often.' },
    { q: 'On a log–log plot, a Zipf distribution looks like…', options: ['A straight line with slope about −1', 'A parabola', 'A horizontal line'], answer: 0, why: 'log f = log f₁ − log r, so the points fall near a line of slope −1.' },
    { q: 'What fraction of the *distinct* words in a passage typically appear only once?', options: ['Almost none', 'Roughly half, often more in a short text', 'All of them'], answer: 1, why: 'The demo passage (641 words) has 63% once-only words. This long tail of hapax legomena is why a word vocabulary is never "complete" and why subword tokenizers exist.' },
    { q: 'The demo anchors the ideal Zipf line at the count of the rank-1 word. Where does rank 1 sit relative to that line?', options: ['Exactly on it, by construction', 'Always above it', 'Always below it'], answer: 0, why: '`zipfPredicted(topCount, n)[0]` is `topCount / 1 = topCount`, so the first point matches whatever the data are. Only ranks 2 and up test the law.' },
  ],
  concept: `
## Why start with counting words?

Every language model is, at bottom, a machine for predicting the next token. Before touching a neural network it is worth looking at what the raw statistics of text look like, because they shape every design decision downstream: how tokenizers are built, why vocabularies have tens of thousands of entries, why a tiny model can look fluent on common phrases and fall apart on rare ones.

The striking empirical fact is **Zipf's law**: if you rank words by frequency, the word at rank \`r\` appears about \`f(1) / r\` times, where \`f(1)\` is the count of the most common word. "the" shows up a lot; the second most common word about half as often; the hundredth about a hundredth as often. Take logs of both sides: \`log f(r) = log f(1) − 1 · log r\`. So on a log–log plot (log count against log rank) an ideal Zipf distribution is a straight line with slope −1. Large corpora of millions of words come close to this; Piantadosi (2014, "Zipf's word frequency law in natural language: a critical review and future directions") documents how and where real text deviates.

:::predict
Take a paragraph of ordinary English prose with about 500 words. Of the *distinct* words in it, what fraction do you expect to appear exactly once?
---
Roughly half, and often more in a short text: the demo's 641-word passage has 262 distinct words, and 164 of them (63%) appear exactly once. This long tail of once-only words (linguists call them *hapax legomena*) is the practical reason no fixed word vocabulary can cover real text, and the reason module 03 builds a *subword* tokenizer instead of a word one.
:::

## The loop you will use in every module

This module exists to teach the lab's rhythm on a problem you already understand:

1. **Recall.** A few questions about earlier material (here: about the lab itself and basic JavaScript). Pulling a fact out of memory strengthens it far more than re-reading it.
2. **Concept.** A short read like this one, with *predict* cards. Write your prediction *before* revealing; being wrong on paper is where the learning is.
3. **Build.** The Build tab has a starter file on the right and step instructions on the left. Each step has tests. Press **Check this step** (or Ctrl+Enter). Hints unlock only after your first attempt, one rung at a time.
4. **Goal.** The Goal tab runs a demo on *your* code and draws the result. The module is not done until it works.
5. **Reflect.** Explain the threshold concept in your own words. Optional stretch goals point at the real systems.

## What you will build

Five small functions:

- \`tokenizeWords(text)\`: split text into lowercase words.
- \`countFrequencies(words)\`: a \`Map\` from word to count.
- \`topK(counts, k)\`: the k most frequent words, ties broken alphabetically.
- \`zipfPredicted(topCount, n)\`: what a perfect Zipf distribution would predict for ranks 1…n.
- \`zipfLogError(actual, predicted)\`: one number for how far the real counts are from that prediction.

Then the demo runs your functions on the first ~640 words of *Alice's Adventures in Wonderland*, plots the real counts and the ideal Zipf line on log–log axes, and fits a straight line to the real points so you can read off their actual slope.

:::predict
The demo's ideal line is anchored at the real count of the rank-1 word ("the", 28 times). Will the real counts for ranks 2–40 fall *above* the line, *below* it, or *on* it? And will the fitted slope be steeper or flatter than −1?
---
Rank 1 sits exactly on the line by construction, since \`zipfPredicted(28, n)[0] = 28 / 1\`. Every other rank sits *above* it, by a factor of about 2 to 5: "to" appears 27 times where the line predicts 14, "she" 21 times against 9.3. The fitted slope is about −0.7, *flatter* than −1, and the mean log-error is about 1.36 (an average factor of e^1.36 ≈ 3.9). A 641-word passage is far too short to show the large-corpus law cleanly: "the" is unusually rare here (28 of 641 words, about 4%, versus roughly 7% in the million-word Brown corpus), and many words tie at small counts. Zipf is a statement about large samples; one page of one book is a noisy draw from it.
:::

## Toy versus production

Everything here is deliberately small. Real tokenizers do not split on a regular expression over the letters a–z: GPT-2 and Llama 3 run byte-level BPE, so accented letters, emoji, code and digits all become tokens, and their pre-tokenizer splits contractions (\`"Alice's"\` becomes \`"Alice"\` + \`"'s"\`) instead of keeping them whole. Careful Zipf analyses use millions of words and fit the exponent by maximum likelihood rather than anchoring a line at the single top count, which lets one noisy word (here, "the") set the whole prediction. What carries over unchanged is the shape: a short head of very frequent items and a long tail that never ends.

A note on conventions used throughout the lab: functions are pure (they return new values rather than mutating inputs), randomness always comes from a seeded generator so tests are reproducible, and each step adds exactly one idea.
`,
  steps: [
    {
      id: 'tokenize',
      title: 'Split text into words',
      instructions: `
Implement \`tokenizeWords(text)\`: return an array of the words in \`text\`, lowercased.

A "word" is a maximal run of the letters \`a\`–\`z\` (after lowercasing) and apostrophes, so \`"Alice's"\` is one word and \`"well-lit"\` is two. Punctuation, digits and everything else are dropped. If the text has no letters at all, return an empty array \`[]\` (not \`null\`).

The starter file already contains a finished helper, \`normalize(text)\`, which lowercases and collapses whitespace. Read it: it shows the style used throughout the lab (a short doc comment, a pure function, no globals). Your function may use it.

\`\`\`js
tokenizeWords("Alice was beginning to get very tired.")
// → ['alice', 'was', 'beginning', 'to', 'get', 'very', 'tired']
\`\`\`
`,
      hints: [
        'Regular expressions can find every match in a string at once. Which character class means "a letter"?',
        'Lowercase first (the worked `normalize` helper already does this), then ask for every maximal run of characters drawn from one character class that contains the lowercase letters and the apostrophe. `String.prototype.match` with the `g` flag returns all matches at once.',
        'Shape: `const found = normalize(text).match(/[ ... ]+/g);` Fill in the class. Then handle the case where `match` finds nothing: it returns `null`, and the tests expect `[]`.',
      ],
    },
    {
      id: 'count',
      title: 'Count frequencies',
      instructions: `
Implement \`countFrequencies(words)\`: return a \`Map\` whose keys are words and whose values are how many times each word appears in the array.

\`\`\`js
countFrequencies(['a', 'b', 'a'])   // → Map { 'a' => 2, 'b' => 1 }
\`\`\`

Why a \`Map\` and not a plain object? Word keys like \`"constructor"\` or \`"__proto__"\` collide with object internals; a \`Map\` has no such surprises. You will meet exactly this kind of counting table again when you train the BPE tokenizer in module 03 (counting adjacent pairs) and the bigram model in module 04.
`,
      predict: { question: 'If `words` has 1,000 entries and 300 distinct words, how many keys does the returned Map have?', answer: '300: one key per distinct word. The counts sum to 1,000.' },
      hints: [
        'Loop over the words once. For each word, read the current count (missing keys give `undefined`) and write back count + 1.',
        'Start from an empty `Map`. A missing key reads as `undefined`, so turn that into 0 before adding 1, and write the new count back with `set`.',
        'Loop shape: `for (const w of words) counts.set(w, /* current count of w, or 0 if absent */ + 1);` Then return the map.',
      ],
    },
    {
      id: 'topk',
      title: 'Rank the words',
      instructions: `
Implement \`topK(counts, k)\`: return an array of \`[word, count]\` pairs for the \`k\` most frequent words, most frequent first. Break ties alphabetically (\`"apple"\` before \`"banana"\`) so the result is deterministic. If there are fewer than \`k\` words, return them all.

\`\`\`js
topK(new Map([['b', 2], ['a', 2], ['c', 5]]), 2)   // → [['c', 5], ['a', 2]]
\`\`\`

Compare counts as *numbers*: \`10\` must rank above \`9\`, which a string comparison gets backwards. Deterministic ordering matters more than it looks: every test in this lab compares against a fixed expected value, and later modules (top-k sampling in module 14, the batching scheduler in module 16) depend on the same discipline.
`,
      hints: [
        'A `Map` can be turned into an array of `[key, value]` pairs with `Array.from(counts)` or `[...counts.entries()]`.',
        'Sort with a comparator that first compares counts descending, and only when counts are equal compares the words with `localeCompare` or `<`.',
        'Shape: `[...counts].sort((x, y) => /* numeric count difference, larger first */ || /* word order, -1 or 1 */).slice(/* … */)`. The `||` falls through to the word comparison only when the count difference is 0.',
      ],
    },
    {
      id: 'zipf',
      title: 'The Zipf prediction',
      instructions: `
Implement \`zipfPredicted(topCount, n)\`: return an array of length \`n\` where element \`i\` (0-based) is the count Zipf's law predicts for rank \`i + 1\`, namely \`topCount / (i + 1)\`.

\`\`\`js
zipfPredicted(100, 4)   // → [100, 50, 33.333…, 25]
\`\`\`

Also implement \`zipfLogError(actual, predicted)\` (two arrays of positive numbers, same length): the mean absolute difference between the natural logs \`Math.log(actual[i])\` and \`Math.log(predicted[i])\` over all \`i\`. An error of 0.69 (= \`log 2\`) means the real counts are off by a factor of 2 on average, above or below. This single number tells you how far a passage is from ideal Zipf behaviour; it is the kind of summary statistic you will compute in every later demo.
`,
      hints: [
        'The rank of index i is i + 1. Build the array with a loop or with `Array.from({ length: n }, (_, i) => …)`.',
        'For the error: take `Math.log` of each pair, subtract, take `Math.abs`, then average over the number of pairs.',
        '`zipfPredicted`: `Array.from({ length: n }, (_, i) => /* count at rank i + 1 */)`. `zipfLogError`: accumulate `Math.abs(/* difference of the two natural logs */)` in a loop, then divide by the number of pairs. The absolute value matters: a point above the line and one below must not cancel.',
      ],
    },
  ],
  reflection: [
    'In two or three sentences, explain what Zipf\'s law says and why it makes a fixed word-level vocabulary impractical for a language model.',
    'The demo passage has a fitted slope of about −0.7, not −1. Explain in your own words why one page of text can disagree with a law about word frequencies, and what you would need to see the law clearly.',
    'Which part of the loop (recall, predict, build, goal, reflect) felt least natural? That is usually the part with the most to give.',
  ],
  stretch: [
    'Run your counter on a much larger text (a whole novel) and fit the log–log slope as the demo does. Does it move from the passage\'s ≈ −0.7 towards −1? Piantadosi (2014, "Zipf\'s word frequency law in natural language: a critical review and future directions") explains why ranking and counting words on the same sample distorts the curve, and splits the corpus in two to avoid it.',
    'Count *character bigrams* instead of words. Do they obey Zipf too? This is the first step towards the BPE tokenizer in module 03.',
    'GPT-2\'s tokenizer has 50,257 entries; Llama 3\'s has 128,256. On a large text, use your `topK` to measure what fraction of all word occurrences the top 1,000 words cover, and how many distinct words it would take to cover 99%. That gap is the case for subword tokens.',
  ],
  timeouts: { tests: 15000, demo: 30000 },
};
