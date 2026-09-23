export default {
  id: '00-hello-lab',
  title: "Zipf's law: learn the lab loop",
  track: 'foundations',
  minutes: 45,
  threshold: 'Word frequency falls roughly as 1/rank, so a handful of words cover most of any text while a long tail of rare words never runs out, which is why language models use subword vocabularies instead of a list of words.',
  goal: 'A word-frequency counter that plots Zipf\'s law on a real passage, built through the Recall → Concept → Build → Goal → Reflect loop you will use in every module.',
  prereqs: ['35-javascript'],
  recall: [
    { q: 'From module 35: what does `counts.get(word)` return when `word` is not a key of the `Map` called `counts`?', options: ['`null`', '`0`', '`undefined`'], answer: 2,
      why: 'A missing key gives `undefined`. That is why counting code writes `(counts.get(w) || 0) + 1`: `||` hands back its right-hand side (here 0) when the left-hand side is `undefined`, `null`, `0` or `""`. You write exactly this in step 2.' },
    { q: 'From module 35: which loop visits the *values* stored in the array `words`?', options: ['`for (const w in words)`', '`for (const w of words)`', '`for (const w = 0; w < words; w++)`'], answer: 1,
      why: '`for...of` gives the values. `for...in` gives the positions, as the strings "0", "1", "2", …, which is a classic trap for anyone coming from Python, where `for w in words` means values. Step 2 loops over words with `for...of`.' },
    { q: 'From module 35: `for (let i = 0; i < 3; i++)` runs its body with `i` equal to…', options: ['0, 1, 2', '1, 2, 3', '0, 1, 2, 3'], answer: 0,
      why: 'Start at 0, stop before reaching 3, add 1 each time. Arrays count positions from 0, so this is the loop that visits every position of an array of length 3. Steps 4 and 5 use it.' },
    { q: 'From module 35: after `const pair = ["alice", 12];`, what is `pair[1]`?', options: ['`"alice"`', '`12`', '`undefined`'], answer: 1,
      why: 'Position 0 holds `"alice"`, position 1 holds `12`. In step 3 every entry is a `[word, count]` pair like this one, so `pair[0]` is the word and `pair[1]` the count.' },
    { q: 'From module 35: what does `[3, 10, 9].sort((a, b) => b - a)` return?', options: ['`[10, 9, 3]`', '`[3, 9, 10]`', '`[10, 3, 9]`'], answer: 0,
      why: 'The comparator returns a negative number when `a` should come first. `b - a` is negative when `a` is larger, so larger numbers go first. Without a comparator, `sort` compares numbers as text, and "10" comes before "3" and "9", which is the third option.' },
  ],
  review: [
    { q: "Zipf's law says the frequency of the word with rank r is roughly proportional to…", options: ['r', '1 / r', 'log r'], answer: 1, why: 'Frequency ≈ f₁ / r: the second most common word appears about half as often as the first, the tenth about a tenth as often.' },
    { q: 'On a log–log plot, a Zipf distribution looks like…', options: ['A straight line with slope about −1', 'A parabola', 'A horizontal line'], answer: 0, why: 'log f = log f₁ − log r, so the points fall near a line of slope −1.' },
    { q: 'What fraction of the *distinct* words in a passage typically appear only once?', options: ['Almost none', 'Roughly half, often more in a short text', 'All of them'], answer: 1, why: 'The demo passage (641 words) has 63% once-only words. This long tail of hapax legomena is why a word vocabulary is never "complete" and why subword tokenizers exist.' },
    { q: 'The demo anchors the ideal Zipf line at the count of the rank-1 word. Where does rank 1 sit relative to that line?', options: ['Exactly on it, by construction', 'Always above it', 'Always below it'], answer: 0, why: '`zipfPredicted(topCount, n)[0]` is `topCount / 1 = topCount`, so the first point matches whatever the data are. Only ranks 2 and up test the law.' },
  ],
  concept: `
:::plain
A language model such as ChatGPT reads and writes text in small pieces called tokens: a token is a whole word, a piece of a word, or a punctuation mark. Before a model can learn anything it needs a fixed list of the tokens it knows, called its vocabulary. In this module you count the words on one page of a book and meet a pattern called Zipf's law: a few words are used all the time, and a huge number of words turn up only once. That long tail of rare words is why models build their vocabulary from word pieces rather than whole words. Along the way you learn the rhythm every module in this lab follows: recall, read and predict, build until the tests pass, run a demo, then explain it in your own words.
:::

## Why start with counting words?

Every language model is, at bottom, a machine for predicting the next token. Before touching a neural network (the kind of model, built from many multiplications of numbers, that later modules construct), it is worth looking at what the raw statistics of text look like, because they shape every design decision downstream: how tokenizers (the programs that cut text into tokens) are built, why vocabularies have tens of thousands of entries, and why a tiny model can look fluent on common phrases and fall apart on rare ones.

The striking fact is **Zipf's law**: if you rank words by how often they occur, the word at rank \`r\` appears about \`f(1) / r\` times, where \`f(1)\` is the count of the most common word. "the" shows up a lot; the second most common word about half as often; the hundredth about a hundredth as often.

To see the law as a picture, use logarithms. The base-10 logarithm counts powers of ten: \`log10(10) = 1\`, \`log10(100) = 2\`, \`log10(1000) = 3\`. A **log–log plot** puts the log of the count on one axis and the log of the rank on the other, so each equal step along an axis means "ten times bigger". Taking logs of Zipf's law gives \`log f(r) = log f(1) − 1 · log r\`, which is the equation of a straight line with **slope** −1: for every step of one to the right (ten times the rank) the line drops by one (a tenth of the count). Large collections of text (a *corpus*, plural *corpora*) with millions of words come close to that line.

:::predict
Take a paragraph of ordinary English prose with about 500 words. Of the *distinct* words in it, what fraction do you expect to appear exactly once?
---
Roughly half, and often more in a short text: the demo's 641-word passage has 262 distinct words, and 164 of them (63%) appear exactly once. This long tail of once-only words (linguists call them *hapax legomena*) is the practical reason no fixed word vocabulary can cover real text, and the reason module 03 builds a *subword* tokenizer instead of a word one.
:::

## The loop you will use in every module

This module exists to teach the lab's rhythm on a problem you already understand:

1. **Recall.** A few questions about earlier material (here: the JavaScript from module 35). Pulling a fact out of memory strengthens it far more than re-reading it, and a wrong answer tells you what to look up.
2. **Concept.** A short read like this one, with *predict* cards. Write your prediction *before* revealing; being wrong on paper is where the learning is.
3. **Build.** The Build tab has step instructions on the left and the starter file in an editor on the right. You replace the placeholder in each function marked \`TODO\` with your own code, then press **Check this step** (or Ctrl+Enter) to run that step's tests. Hints unlock only after your first attempt, one rung at a time.
4. **Goal.** The Goal tab runs a demo on *your* code and draws the result. The module is not done until it works.
5. **Reflect.** Explain the threshold concept (the one idea the module is built around) in your own words. Optional stretch goals point at the real systems.

## What you will build

Five small functions, one per build step:

- \`tokenizeWords(text)\`: split text into lowercase words.
- \`countFrequencies(words)\`: a \`Map\` from word to count.
- \`topK(counts, k)\`: the k most frequent words, ties broken alphabetically.
- \`zipfPredicted(topCount, n)\`: what a perfect Zipf distribution would predict for ranks 1…n.
- \`zipfLogError(actual, predicted)\`: one number for how far the real counts are from that prediction.

Then the demo runs your functions on a fixed 641-word excerpt from the opening of *Alice's Adventures in Wonderland*, plots the real counts and the ideal Zipf line on log–log axes, and fits a straight line through the real points (the best-fitting line, found by the standard "least squares" method) so you can read off their actual slope.

:::predict
The demo's ideal line is anchored at the real count of the rank-1 word ("the", 28 times). Will the real counts for ranks 2–40 fall *above* the line, *below* it, or *on* it? And will the fitted slope be steeper or flatter than −1?
---
Rank 1 sits exactly on the line by construction, since \`zipfPredicted(28, n)[0] = 28 / 1\`. Every other rank sits *above* it, by a factor of about 2 to 5: "to" appears 27 times where the line predicts 14, "she" 21 times against 9.3. The fitted slope is about −0.7, *flatter* than −1, and the mean log-error is about 1.36 (an average factor of e^1.36 ≈ 3.9). A 641-word passage is far too short to show the large-corpus law cleanly: "the" is unusually rare here (28 of 641 words, about 4%, versus roughly 7% in the million-word Brown corpus), and many words tie at small counts. Zipf is a statement about large samples; one page of one book is a noisy draw from it.
:::

## Two conventions used throughout the lab

Functions are **pure**: they compute a result from their inputs and return it, without changing the inputs or anything outside the function. And randomness always comes from a **seeded generator**, a random-number source that produces the same sequence every time it starts from the same seed, so every test gives the same answer on every run. Each build step adds one idea.

:::deeper Going deeper: how real tokenizers and real Zipf studies differ
Everything here is deliberately small. Real tokenizers do not split on a pattern over the letters a–z: GPT-2 and Llama 3 run byte-level BPE (module 03), so accented letters, emoji, code and digits all become tokens, and their pre-tokenizer splits contractions (\`"Alice's"\` becomes \`"Alice"\` + \`"'s"\`) instead of keeping them whole. Careful Zipf analyses use millions of words and fit the exponent by maximum likelihood rather than anchoring a line at the single top count, which lets one noisy word (here, "the") set the whole prediction. Piantadosi (2014, "Zipf's word frequency law in natural language: a critical review and future directions") documents how and where real text deviates from the law. (\`lib/data.js\` also carries an abridged copy of the same chapter inside \`PROSE\`, but its wording differs, so counting 641 words of it gives 272 distinct words and "the" 29 times, not the numbers quoted here.) What carries over unchanged is the shape: a short head of very frequent items and a long tail that never ends.
:::
`,
  steps: [
    {
      id: 'tokenize',
      title: 'Split text into words',
      instructions: `
**How the Build screen works.** The editor on the right holds the whole starter file. Each function you write contains a \`// TODO: step N\` comment and a placeholder line such as \`return [];\`. Replace the placeholder with your own code, between the function's \`{\` and \`}\`, and leave the other functions alone for now. Press **Check this step** (or Ctrl+Enter): the results appear under the editor, each failing test saying what it expected and what your code gave. Anything you \`console.log\` appears there too.

Implement \`tokenizeWords(text)\`: return an array of the words in \`text\`, lowercased.

A "word" is a maximal run of the letters \`a\`–\`z\` (after lowercasing) and apostrophes, so \`"Alice's"\` is one word and \`"well-lit"\` is two. Apostrophes at the edge of a run stay too (\`"'tis"\` stays \`"'tis"\`); that toy rule is harmless here because the demo passage quotes with double quotes. Punctuation, digits and everything else are dropped. If the text has no letters at all, return an empty array \`[]\` (not \`null\`).

\`\`\`js
tokenizeWords("Alice was beginning to get very tired.")
// → ['alice', 'was', 'beginning', 'to', 'get', 'very', 'tired']
\`\`\`

**The three regular-expression pieces you need.** A regular expression is a pattern for finding text, written between two slashes.

- \`[...]\` is a *character class*: it matches any **one** character listed inside the brackets. A dash makes a range, so \`[a-z]\` is any lowercase letter and \`[a-z0-9]\` is any lowercase letter or digit. Most other characters, including the apostrophe, simply stand for themselves inside the brackets.
- \`+\` after something means "one or more of it in a row", so \`[a-z]+\` matches a whole run of letters such as \`cat\`, not just \`c\`.
- The \`g\` flag after the closing slash means "find every match, not only the first".

Module 35 used \`/[a-z]/g\` to find single letters; the \`+\` is what turns that into whole runs. Put together: \`"a cat, 2 dogs".match(/[a-z]+/g)\` returns \`['a', 'cat', 'dogs']\`. When nothing matches, \`match\` returns \`null\` instead of an empty array; \`found || []\` (module 35's \`||\` fallback) turns that \`null\` into \`[]\`.

The starter file already contains a finished helper, \`normalize(text)\`, which lowercases and collapses whitespace. Read it: it shows the style used throughout the lab (a short doc comment, a pure function, no globals). Your function may use it.
`,
      hints: [
        'Which characters may appear inside a word? All of them go inside the brackets of one character class, and the pattern should match runs of them.',
        'Lowercase first (the worked `normalize` helper already does this), then ask `match` for every run of one or more characters from a class that holds the lowercase letters and the apostrophe. Store the result in a variable, because it can be `null`.',
        '```js\nconst found = normalize(text).match(/[ /* the letter range and the apostrophe */ ]+/g);\nreturn found || [];\n```',
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

You need three pieces from module 35: create the empty table with \`const counts = new Map();\`, loop over the words with \`for (const w of words)\` (\`of\` gives the words themselves; \`in\` would give their positions "0", "1", …), and read and write entries with \`counts.get(w)\` and \`counts.set(w, value)\`. Remember that \`get\` returns \`undefined\` for a word you have not seen yet. If you know Python, resist \`counts[w] = ...\`: on a Map that sets a property on the object and leaves the Map itself empty.

Why a \`Map\` and not a plain object? Word keys like \`"constructor"\` or \`"__proto__"\` collide with the built-in properties every object has; a \`Map\` has no such surprises. You will meet exactly this kind of counting table again when you train the BPE tokenizer in module 03 (counting adjacent pairs) and the bigram model in module 04.
`,
      predict: { question: 'If `words` has 1,000 entries and 300 distinct words, how many keys does the returned Map have?', answer: '300: one key per distinct word. The counts sum to 1,000.' },
      hints: [
        'Think of a tally sheet: one pass over the array, and for each word you bump that word\'s running total. Which structure holds "word → running total"?',
        'Create an empty `Map` before the loop. Inside the loop, read the word\'s current count, turn a missing count (`undefined`) into 0, add 1, and write the new count back with `set`. After the loop, return the map.',
        '```js\nconst counts = new Map();\nfor (const w of words) {\n  counts.set(w, /* the current count of w, or 0 if w is not in the map yet */ + 1);\n}\nreturn counts;\n```',
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

The pieces:

- \`[...counts]\` turns the Map into an array of pairs. Each pair is \`[word, count]\`, so for a pair \`x\`, \`x[0]\` is the word and \`x[1]\` is the count.
- \`pairs.sort((x, y) => …)\` sorts the array in place. The arrow function, the *comparator*, receives two **pairs** \`x\` and \`y\` and returns a number: negative to put \`x\` first, positive to put \`y\` first, 0 for a tie. Subtracting the pairs themselves (\`y - x\`) gives \`NaN\`, which leaves the order unchanged; subtract their counts.
- \`x[0].localeCompare(y[0])\` compares two words alphabetically: negative when \`x[0]\` comes first.
- \`a || b\` gives \`b\` when \`a\` is 0, so \`countDifference || wordOrder\` uses the word order only for tied counts.
- \`pairs.slice(0, k)\` returns the first \`k\` entries (all of them if there are fewer).

Compare counts as *numbers*: \`10\` must rank above \`9\`, which a text comparison gets backwards. Deterministic ordering matters more than it looks: every test in this lab compares against a fixed expected value, and later modules (top-k sampling in module 14, the batching scheduler in module 16) depend on the same discipline.
`,
      hints: [
        'Which of the two counts should win, and what sign must the comparator return to put the larger count first?',
        'Turn the Map into an array of pairs, sort it with a comparator that returns the count difference (larger count first) and falls through to the word comparison when that difference is 0, then keep the first k pairs.',
        '```js\nconst pairs = [...counts];              // [[word, count], …]\npairs.sort((x, y) => /* count difference from x[1] and y[1], larger count first */ || x[0].localeCompare(y[0]));\nreturn pairs.slice(0, k);\n```',
      ],
    },
    {
      id: 'zipf',
      title: 'The Zipf prediction',
      instructions: `
Implement \`zipfPredicted(topCount, n)\`: return an array of length \`n\` where element \`i\` (counting from 0) is the count Zipf's law predicts for rank \`i + 1\`, namely \`topCount / (i + 1)\`. For \`n = 0\` return an empty array \`[]\`.

\`\`\`js
zipfPredicted(100, 4)   // → [100, 50, 33.333…, 25]
\`\`\`

The plain way to build an array of length \`n\` is a counting loop that pushes one value per position:

\`\`\`js
const out = [];
for (let i = 0; i < n; i++) {
  out.push(i * 10);        // the value for position i
}
// with n = 3, out is [0, 10, 20]
\`\`\`

A shorter form you will see in later modules does the same in one line: \`Array.from({ length: n }, (_, i) => i * 10)\`. \`Array.from\` calls the arrow function once for each position and passes the position as the second argument; \`_\` is simply a name for the first argument, which is not needed here. Use whichever you find clearer.

Do not round: the values are plain JavaScript numbers, fractions included. The demo draws this array as the ideal straight line on its log–log plot.
`,
      hints: [
        'Element 0 is rank 1, element 1 is rank 2. What is the rank of element i, and what does Zipf\'s law predict for it?',
        'Start from an empty array and loop i from 0 while i < n, pushing topCount divided by the rank of element i, which is i + 1. When n is 0 the loop body never runs, so you return [] without any special case.',
        '```js\nconst out = [];\nfor (let i = 0; i < n; i++) out.push(/* topCount over the rank of element i */);\nreturn out;\n```',
      ],
    },
    {
      id: 'logerr',
      title: 'Measure the distance from Zipf',
      instructions: `
Implement \`zipfLogError(actual, predicted)\`. Both arguments are arrays of positive numbers with the same length. Return one number: the mean (the average) over all positions \`i\` of the absolute difference between the natural logs, \`|Math.log(actual[i]) − Math.log(predicted[i])|\`. \`Math.abs(x)\` gives the absolute value.

\`\`\`js
zipfLogError([10], [5])                // → Math.log(2) ≈ 0.693
zipfLogError([64, 32], [64, 32])       // → 0
\`\`\`

You need the same position \`i\` in both arrays, so loop over positions with \`for (let i = 0; i < actual.length; i++)\`, not over values with \`for (const v of actual)\`.

Use the natural log (\`Math.log\`, not \`Math.log10\`). Why logs? A count twice the prediction and a count half the prediction are equally wrong by a factor of 2, and in log space both are \`log 2\` away. The absolute value keeps a point above the line and a point below it from cancelling. An error of 0.69 (= \`log 2\`) means the real counts are off by a factor of 2 on average. This single number tells you how far a passage is from ideal Zipf behaviour; it is the kind of summary statistic you will compute in every later demo.
`,
      hints: [
        'For each pair, how far apart are the two numbers as a *ratio*? Logs turn ratios into differences.',
        'Walk both arrays together by position. For each position take the difference of the two natural logs, make it non-negative, and add it to a running sum. Divide the sum by the number of positions.',
        '```js\nlet s = 0;\nfor (let i = 0; i < actual.length; i++) s += /* non-negative difference of the two natural logs at position i */;\nreturn s / actual.length;\n```',
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
    'Count *character bigrams* instead of words. Do they obey Zipf too? This is the first step towards the BPE tokenizer in module 03: GPT-2\'s byte-level BPE (which OpenAI\'s tiktoken library runs as its `gpt2` encoding) chose its first merge from counts of adjacent pairs like these, taken over bytes rather than characters and only inside pre-tokens (roughly, words), then re-counted pairs of the merged symbols before each later merge.',
    'GPT-2\'s tokenizer has 50,257 entries; Llama 3\'s has 128,256. On a large text, use your `topK` to measure what fraction of all word occurrences the top 1,000 words cover, and how many distinct words it would take to cover 99%. That gap is the case for subword tokens.',
  ],
  timeouts: { tests: 15000, demo: 30000 },
};
