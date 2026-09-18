export default {
  id: '00-hello-lab',
  title: "Zipf's law: learn the lab loop",
  track: 'foundations',
  minutes: 20,
  threshold: 'A handful of words carry most of the text: word frequency falls roughly as 1/rank, which is why tokenizers and models spend most of their capacity on a small vocabulary.',
  goal: 'A word-frequency counter that plots Zipf\'s law on a real passage, built through the Recall → Concept → Build → Goal → Reflect loop you will use in every module.',
  prereqs: [],
  recall: [
    { q: 'In this lab, when does a module count as complete?', options: ['When you have read the concept section', 'When all build steps pass their tests and the goal demo has run on your code', 'When you have opened every hint'], answer: 1,
      why: 'Mastery learning: the evidence of understanding is working code, not time spent reading.' },
    { q: 'Why do the hints stay locked until you have checked a step at least once?', options: ['To save bandwidth', 'Because attempting first, even unsuccessfully, improves later learning (productive failure)', 'Hints are only for the last step'], answer: 1,
      why: 'Kapur (2008): learners who struggle before instruction end up with deeper understanding than those who receive help immediately.' },
    { q: 'In JavaScript, what does `map.get(key)` return when the key is absent from a `Map`?', options: ['null', '0', 'undefined'], answer: 2,
      why: 'You will write `(counts.get(w) || 0) + 1` a lot; the `|| 0` is there because a missing key gives `undefined`.' },
    { q: 'Which of these sorts an array of numbers in descending order?', options: ['`arr.sort()`', '`arr.sort((a, b) => b - a)`', '`arr.reverse()`'], answer: 1,
      why: 'Without a comparator, `sort` compares as strings ("10" < "9"). A comparator returning `b - a` puts larger numbers first.' },
  ],
  review: [
    { q: "Zipf's law says the frequency of the word with rank r is roughly proportional to…", options: ['r', '1 / r', 'log r'], answer: 1, why: 'Frequency ≈ f₁ / r: the second most common word appears about half as often as the first, the tenth about a tenth as often.' },
    { q: 'On a log–log plot, a Zipf distribution looks like…', options: ['A straight line with slope about −1', 'A parabola', 'A horizontal line'], answer: 0, why: 'log f = log f₁ − log r, so the points fall near a line of slope −1.' },
    { q: 'What fraction of the *distinct* words in a passage typically appear only once?', options: ['Almost none', 'Roughly half', 'All of them'], answer: 1, why: 'The long tail of hapax legomena is why vocabularies are never "complete" and why subword tokenizers exist.' },
  ],
  concept: `
## Why start with counting words?

Every language model is, at bottom, a machine for predicting the next token. Before touching a neural network it is worth looking at what the raw statistics of text look like, because they shape every design decision downstream: how tokenizers are built, why vocabularies have tens of thousands of entries, why a tiny model can look fluent on common phrases and fall apart on rare ones.

The striking empirical fact is **Zipf's law**: if you rank words by frequency, the word at rank \`r\` appears about \`f(1) / r\` times, where \`f(1)\` is the count of the most common word. "the" shows up a lot; the second most common word about half as often; the hundredth about a hundredth as often. On a log–log plot the points fall near a straight line with slope roughly −1.

:::predict
Take a paragraph of ordinary English prose with about 500 words. Of the *distinct* words in it, what fraction do you expect to appear exactly once?
---
Usually around half. This long tail of once-only words (linguists call them *hapax legomena*) is the practical reason no fixed word vocabulary can cover real text, and the reason module 03 builds a *subword* tokenizer instead of a word one.
:::

## The loop you will use in every module

This module exists to teach the lab's rhythm on a problem you already understand:

1. **Recall.** A few questions about earlier material (here: about the lab itself and basic JavaScript). Pulling a fact out of memory strengthens it far more than re-reading it.
2. **Concept.** A short read like this one, with *predict* cards. Write your prediction *before* revealing; being wrong on paper is where the learning is.
3. **Build.** The Build tab has a starter file on the right and step instructions on the left. Each step has tests. Press **Check this step** (or Ctrl+Enter). Hints unlock only after your first attempt, one rung at a time.
4. **Goal.** The Goal tab runs a demo on *your* code and draws the result. The module is not done until it works.
5. **Reflect.** Explain the threshold concept in your own words. Optional stretch goals point at the real systems.

## What you will build

Four small functions:

- \`tokenizeWords(text)\`: split text into lowercase words.
- \`countFrequencies(words)\`: a \`Map\` from word to count.
- \`topK(counts, k)\`: the k most frequent words, ties broken alphabetically.
- \`zipfPredicted(topCount, n)\`: what a perfect Zipf distribution would predict for ranks 1…n.

Then the demo compares the real counts against the Zipf prediction on a passage from *Alice's Adventures in Wonderland* and plots both on a log scale.

:::predict
Will the real counts fall *above* or *below* the ideal Zipf line for the very top ranks (1–3)?
---
Typically the top one or two words are slightly *below* the ideal line and the middle ranks run a little above it; short passages are noisy. What matters is the overall slope near −1, which you will see in the demo.
:::

A note on conventions used throughout the lab: functions are pure (they return new values rather than mutating inputs), randomness always comes from a seeded generator so tests are reproducible, and each step adds exactly one idea.
`,
  steps: [
    {
      id: 'tokenize',
      title: 'Split text into words',
      instructions: `
Implement \`tokenizeWords(text)\`: return an array of the words in \`text\`, lowercased.

A "word" is a maximal run of letters and apostrophes, so \`"Alice's"\` is one word and \`"well-lit"\` is two. Punctuation and digits are dropped.

The starter file already contains a finished helper, \`normalize(text)\`, which lowercases and collapses whitespace. Read it: it shows the style used throughout the lab (a short doc comment, a pure function, no globals). Your function may use it.

\`\`\`js
tokenizeWords("Alice was beginning to get very tired.")
// → ['alice', 'was', 'beginning', 'to', 'get', 'very', 'tired']
\`\`\`
`,
      hints: [
        'Regular expressions can find every match in a string at once. Which character class means "a letter"?',
        'Lowercase the text first, then use `text.match(/[a-z\']+/g)`. Remember that `match` with the `g` flag returns `null` when nothing matches.',
        'Something close to: `return (normalize(text).match(/[a-z\']+/g) || []);`',
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
        'Use `counts.get(w) || 0` to treat a missing key as zero, then `counts.set(w, that + 1)`.',
        '`const counts = new Map(); for (const w of words) counts.set(w, (counts.get(w) || 0) + 1); return counts;`',
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

Deterministic ordering matters more than it looks: every test in this lab compares against a fixed expected value, and later modules (top-k sampling in module 14, the batching scheduler in module 16) depend on the same discipline.
`,
      hints: [
        'A `Map` can be turned into an array of `[key, value]` pairs with `Array.from(counts)` or `[...counts.entries()]`.',
        'Sort with a comparator that first compares counts descending, and only when counts are equal compares the words with `localeCompare` or `<`.',
        '`return [...counts].sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : 1)).slice(0, k);`',
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

Also implement \`zipfLogError(actual, predicted)\`: the mean absolute difference between \`log(actual[i])\` and \`log(predicted[i])\` over all \`i\`. This single number tells you how far a passage is from ideal Zipf behaviour; it is the kind of summary statistic you will compute in every later demo.
`,
      hints: [
        'The rank of index i is i + 1. Build the array with a loop or with `Array.from({ length: n }, (_, i) => …)`.',
        'For the error: take `Math.log` of each pair, subtract, take `Math.abs`, then average over the number of pairs.',
        '`zipfPredicted`: `Array.from({ length: n }, (_, i) => topCount / (i + 1))`. `zipfLogError`: sum `Math.abs(Math.log(actual[i]) - Math.log(predicted[i]))` over i and divide by `actual.length`.',
      ],
    },
  ],
  reflection: [
    'In two or three sentences, explain what Zipf\'s law says and why it makes a fixed word-level vocabulary impractical for a language model.',
    'Which part of the loop (recall, predict, build, goal, reflect) felt least natural? That is usually the part with the most to give.',
  ],
  stretch: [
    'Plot rank against frequency on a log–log scale for a much larger text and estimate the slope. Real corpora give a slope slightly steeper than −1 (see Piantadosi 2014, "Zipf\'s word frequency law in natural language").',
    'Count *character bigrams* instead of words. Do they obey Zipf too? This is the first step towards the BPE tokenizer in module 03.',
    'GPT-2\'s tokenizer has 50,257 entries; Llama 3\'s has 128,256. Using your `topK`, estimate what fraction of the tokens in a passage would be covered by the top 1,000 words.',
  ],
  timeouts: { tests: 15000, demo: 30000 },
};
