export default {
  id: '35-javascript',
  title: 'JavaScript for this lab',
  track: 'foundations',
  minutes: 90,
  threshold: 'Code does exactly what its rules say, not what looks reasonable (two equal-looking arrays are not ===, for...in gives positions, a missing Map key is undefined), so you make progress by predicting a result, running it, and letting the test message show where your guess and the rules part ways.',
  goal: 'A letter counter built from your own small functions: it counts and ranks the letters of a famous sentence, draws them as a bar chart, and stores them in a { shape, data } table, using exactly the pieces of JavaScript the next modules rely on.',
  prereqs: [],
  recall: [
    { q: 'This is the first module, so these questions are about everyday ideas the module builds on. You keep a tally of votes on paper. A name comes up that you have not written down yet. What is its tally before you add the new mark?',
      options: ['1', '0: nothing written down counts as zero', 'You cannot know'], answer: 1,
      why: 'In step 4 a JavaScript Map plays the tally sheet. Asking it for a name it has never seen gives `undefined` ("nothing written down"), and you will write `|| 0` to turn that into the 0 you just used without thinking.' },
    { q: 'An everyday idea again: in a building whose floors are numbered from the ground floor, 0, how is the top floor of a 5-storey building numbered?',
      options: ['5', '4', '6'], answer: 1,
      why: 'Five floors numbered 0, 1, 2, 3, 4. JavaScript numbers the positions in a list the same way: a list of length 5 has positions 0 to 4, so the last one is at `length - 1`.' },
    { q: 'Another everyday idea: two shopping lists on two separate pieces of paper say exactly the same things. Are they the same piece of paper?',
      options: ['Yes, because they say the same things', 'No: equal contents, but two different pieces of paper', 'Only if they were written by the same person'], answer: 1,
      why: 'Step 3 turns on exactly this. In JavaScript `[2, 3] === [2, 3]` is `false`: `===` on two arrays asks "is this the same piece of paper?", so to compare contents you check the length and each position yourself.' },
  ],
  review: [
    { q: 'With `const xs = [\'a\', \'b\']`, what does `for (const x in xs) console.log(x)` print?', options: ["'a' then 'b'", "'0' then '1' (positions, as text)", 'Nothing: it is an error'], answer: 1, why: 'for...in walks the positions of an array as text. Use for...of for the values, or a counting loop for numeric positions.' },
    { q: 'What is `[2, 3] === [2, 3]` in JavaScript?', options: ['true', 'false', 'undefined'], answer: 1, why: 'Two array literals make two separate arrays, and === on arrays asks whether they are the very same array. Compare lengths and entries (sameShape) to compare contents.' },
    { q: 'What does `new Map().get(\'q\')` give, and so what does `(new Map().get(\'q\') || 0) + 1` give?', options: ['0, then 1', 'undefined, then 1', 'null, then NaN'], answer: 1, why: 'A missing key reads as undefined. `undefined || 0` is 0, so the count starts at 1. Without the `|| 0` you would get undefined + 1 = NaN.' },
    { q: 'Which comparator sorts `[letter, count]` pairs with the largest count first?', options: ['`(a, b) => b - a`', '`(a, b) => b[1] - a[1]`', '`(a, b) => a[1] - b[1]`'], answer: 1, why: 'a and b are whole pairs, so the counts are a[1] and b[1]. A positive result puts b first, so b[1] - a[1] puts larger counts first. b - a on two arrays is NaN.' },
    { q: 'What is `3 ^ 2` in JavaScript?', options: ['9', '1, because ^ is not a power in JavaScript', 'An error'], answer: 1, why: '^ is a bit operation on whole numbers, not "to the power of", and it gives no error, just a wrong number. Square with `x * x` or `x ** 2`.' },
    { q: 'What is `Math.exp(1000)` in JavaScript?', options: ['A very large but ordinary number', '`Infinity`', 'An error is thrown'], answer: 1, why: 'JavaScript numbers top out near 1.8 × 10^308 (about e^709.8); beyond that you get Infinity rather than an error. Module 01 meets this when it computes softmax.' },
  ],
  concept: `
:::plain
Every other module in this lab asks you to write small pieces of JavaScript, the programming language built into web browsers. This module teaches exactly the parts they use, and nothing more, by having you write eleven short functions: little named recipes that take some values in and give one value back. Together they count the letters in a famous sentence, put them in order and draw a chart. You do not need to have programmed before. If you know a little Python or spreadsheet formulas, a translation table below maps what you know onto JavaScript.
:::

## What you will build

Six steps, each with one idea:

1. **Values and functions.** \`label(letter, count)\` makes the text \`'e: 3'\`; \`surprise(count, total)\` turns a count into a "how surprising" number.
2. **Arrays, loops and arithmetic.** \`range(n)\`, \`sum(numbers)\`, \`sumOfSquares(numbers)\` and \`firstK(items, k)\`.
3. **Decisions and equality.** \`sameShape(a, b)\`, which answers a question that \`===\` gets wrong.
4. **Counting with a Map.** \`countLetters(text)\`, a tally sheet from letter to count.
5. **Sorting.** \`rankCounts(counts)\`, largest count first.
6. **Objects, typed arrays and errors.** \`makeTable\` and \`cell\`: a small table of numbers stored the way every later module stores them.

The Goal tab then runs your functions on the opening of *A Tale of Two Cities*, draws the ten most common letters, and prints your table. Module 00 does the same with words and finds a law about them; this module gives you the tools for it.

## How the Build screen works

The **Build** tab shows the step's instructions on one side and your code file on the other. Each function has a \`// TODO\` comment (a line starting with \`//\` is a note that JavaScript ignores). Type your code between that function's curly braces \`{ }\`, replacing the TODO and the placeholder \`return\` line. Press **Check this step** (or Ctrl+Enter).

Each test prints a line. Green passed. Red failed, and says what it **expected** and what your code **got**: \`expected 'e: 3' but got ''\` means your function gave back empty text where \`'e: 3'\` was wanted. When JavaScript itself stops, the message names the problem instead: \`total is not defined\` means you used the name \`total\` before creating it. Hints unlock after your first check.

To look at a value while you work, add a line like \`console.log(counts);\` inside your function. What it prints appears under the test results. Delete it when you are done; it does no harm if you forget.

The word \`export\` in front of \`function\` makes a function visible to the tests and the demo. Keep it.

## A first look at the pieces

A **value** is a piece of data: a number like \`3\` or \`0.25\`, a piece of text (a **string**) like \`'e'\`, or \`true\`/\`false\`. You give a value a name with \`const\` (the name will not be reassigned) or \`let\` (it will change, like a running total). A **function** is a named recipe: \`function share(count, total) { return count / total; }\` takes two inputs and \`return\` hands back the answer. You **call** it by writing \`share(3, 12)\`, which gives \`0.25\`.

An **array** is a list, \`[3, 1, 2]\`, with positions counted from 0. A **loop** repeats some code once per item. JavaScript has two loops that look alike and behave differently:

:::predict
\`const scores = [10, 20];\` Then \`for (const x in scores) console.log(x);\`. What gets printed: \`10\` and \`20\`, or something else?
---
\`'0'\` and \`'1'\`: the **positions**, and as text, not numbers. \`for...in\` walks positions. To get the values you write \`for (const x of scores)\`, with **of**. Python's \`for x in scores\` gives values, so Python users hit this every time. Step 2 uses \`of\`.
:::

JavaScript also has some special numbers you will meet the moment a formula goes wrong:

:::predict
What do you expect from each: \`Math.exp(1000)\`, \`Math.log(0)\`, \`0 / 0\`? Does JavaScript stop with an error?
---
No error. \`Math.exp(1000)\` (e to the power 1000) is too big to store, so it is \`Infinity\`. \`Math.log(0)\` is \`-Infinity\`. \`0 / 0\` is \`NaN\`, "not a number", and any arithmetic touching \`NaN\` stays \`NaN\`; even \`NaN === NaN\` is \`false\`. So when a test says it got \`NaN\`, look for arithmetic on something that was not a number, often \`undefined\` from reading a position or key that does not exist.
:::

## Python and spreadsheet equivalents

| Idea | JavaScript | Python | Spreadsheet |
|---|---|---|---|
| name a value | \`const total = 10;\` | \`total = 10\` | a named cell |
| text with a value inside | \`e: \${n}\` between backticks | \`f"e: {n}"\` | \`="e: "&A1\` |
| a list and its length | \`[3, 1, 2]\`, \`xs.length\` | \`[3, 1, 2]\`, \`len(xs)\` | a column, \`COUNT\` |
| positions 0 to n − 1 | \`for (let i = 0; i < n; i++)\` | \`for i in range(n)\` | row numbers |
| each value | \`for (const x of xs)\` | \`for x in xs\` | |
| first k items | \`xs.slice(0, k)\` | \`xs[:k]\` | |
| equal? | \`a === b\` | \`a == b\` | \`=A1=B1\` |
| tally table | \`new Map()\`, \`get\`, \`set\` | \`dict\`, \`d.get(k, 0)\` | \`COUNTIF\` |
| sort by count, largest first | \`pairs.sort((a, b) => b[1] - a[1])\` | \`sorted(pairs, key=lambda p: -p[1])\` | Sort Z to A |
| x squared | \`x * x\` or \`x ** 2\` (never \`x ^ 2\`) | \`x ** 2\` | \`=A1^2\` |
| natural log, e to the x | \`Math.log(x)\`, \`Math.exp(x)\` | \`math.log\`, \`math.exp\` | \`LN\`, \`EXP\` |
| stop with an error | \`throw new Error('msg')\` | \`raise ValueError('msg')\` | \`#VALUE!\` |

Three warnings. \`==\` exists in JavaScript but has odd conversion rules, so this lab always uses \`===\` and \`!==\`. Arrays are never \`===\` to another array, however equal they look (step 3). And \`^\` is not "to the power of" as it is in a spreadsheet: \`3 ^ 2\` is 1, with no error, so write \`x * x\` or \`x ** 2\` (step 2).

## Toy versus production

This is a deliberately small slice of JavaScript. Later modules add a few things with comments where they appear: classes (module 02), \`import\`, and the spread \`...\` in more places. Real projects also use tools you will not need here (TypeScript, packages, a terminal). And a letter count is a toy: the tokenizers in real models such as GPT-2 and Llama 3 work on bytes and learned pieces of words (module 03), not single letters. What carries over unchanged is the style: small functions that take values in, return a value, and are checked by tests.
`,
  steps: [
    {
      id: 'values',
      title: 'Values, text and your first functions',
      instructions: `
Read the worked example \`share(count, total)\` at the top of the file first. It shows the shape of every function in this lab:

\`\`\`js
export function share(count, total) {   // name, then the inputs in brackets
  const fraction = count / total;       // work, naming results with const
  return fraction;                      // hand one value back
}
\`\`\`

\`share(3, 12)\` **calls** it with \`count = 3\` and \`total = 12\` and gives back \`0.25\`. Your functions can call it too.

**1. \`label(letter, count)\`** returns a piece of text such as \`'e: 3'\`. Text in quotes is called a **string**. To put values *inside* a string, use a **template string**: write it between backticks (the key left of 1 on most keyboards) instead of quotes, and put each value in a \`\${ }\` slot. Ordinary quotes do not fill slots, and Python's \`f"..."\` does not exist in JavaScript (the file stops loading with "Unexpected string"):

\`\`\`js
const letter = 'e';
\`\${letter}!\`        // → 'e!'           backticks: the slot is filled
'\${letter}!'        // → '\${letter}!'   ordinary quotes: kept exactly as typed
\`\`\`

\`\`\`js
label('e', 3)      // → 'e: 3'
label('the', 12)   // → 'the: 12'
\`\`\`

**2. \`surprise(count, total)\`** returns how surprising something is when it makes up \`share(count, total)\` of all cases: minus the natural log of the share, \`-Math.log(share(count, total))\`. \`Math.log\` is JavaScript's natural logarithm (spreadsheet \`LN\`). You do not need to know logs well: the log of 1 is 0, and the smaller the share, the larger the surprise. \`Math.exp\` undoes \`Math.log\`: \`Math.exp(-surprise(3, 12))\` gives the share 0.25 back.

\`\`\`js
surprise(1, 2)    // → 0.693…  (a coin landing heads)
surprise(3, 12)   // → 1.386…  (rarer, so more surprising)
surprise(5, 5)    // → 0       (always happens: no surprise)
surprise(0, 10)   // → Infinity (never happened: Math.log(0) is -Infinity)
\`\`\`

Why this exists: a language model's training loss (module 04 onwards) is exactly this, the average surprise at the real next character, measured in these units, called **nats**.

Names to avoid: \`var\`, \`function\`, \`return\`, \`new\`, \`const\`, \`let\` and a few other words are part of the language, so a variable cannot be called that (write \`variance\`, not \`var\`).
`,
      hints: [
        'Each function is one `return` line. For label, which kind of quote lets you put values inside the text? For surprise, the worked `share` function already computes the fraction you need.',
        'label: backticks around the whole text, with `${letter}` and `${count}` as slots and `: ` (colon, space) between them. surprise: call `share(count, total)`, take `Math.log` of it, and put a minus sign in front.',
        'Replace each placeholder `return` line. Fill in the two gaps marked `/* ? */`:\n\n```js\nreturn `${letter}: ${/* ? */}`;                     // in label\nreturn -Math.log(share(/* ? */, /* ? */));         // in surprise\n```',
      ],
    },
    {
      id: 'loops',
      title: 'Arrays, loops and arithmetic',
      instructions: `
An **array** is a list of values in square brackets: \`const xs = [5, 6, 7];\`. Positions start at 0, so \`xs[0]\` is 5 and \`xs[2]\` is 7. \`xs.length\` is 3, and the last item is always \`xs[xs.length - 1]\`. \`xs.push(8)\` adds 8 to the end.

A **counting loop** repeats code for i = 0, 1, 2, …:

\`\`\`js
for (let i = 0; i < n; i++) {
  // this runs with i = 0, then 1, ..., up to n - 1
}
\`\`\`

Read it as three parts: start with \`let i = 0\`; keep going while \`i < n\`; after each round do \`i++\` (add 1 to i). A **for...of loop** hands you each value in turn: \`for (const x of xs) { ... }\`. Avoid \`for...in\` on arrays: it hands you the positions as text (\`'0'\`, \`'1'\`), which is almost never what you want. \`total += x\` is short for \`total = total + x\`. Always put \`const\` or \`let\` before a loop's name, as in \`for (const x of xs)\`: without it, \`for (x of xs)\` stops with \`x is not defined\`.

**Arithmetic.** \`+\`, \`-\`, \`*\` and \`/\` work as on a calculator. To square a number write \`x * x\`, or \`x ** 2\` (two stars mean "to the power of"). **JavaScript has no \`^\` for powers.** \`^\` means something else (a bit operation on whole numbers), so \`3 ^ 2\` is 1, not 9, and JavaScript gives no error: it silently gives wrong numbers. Spreadsheets and maths books write powers with \`^\`, so this slip is easy to make when you copy a formula.

\`\`\`js
3 * 3      // → 9
3 ** 2     // → 9
3 ^ 2      // → 1   (not a power!)
\`\`\`

Write four functions:

\`\`\`js
range(4)                    // → [0, 1, 2, 3]   (like Python's range; range(0) is [])
sum([2, 3, 5])              // → 10             (sum([]) is 0)
sumOfSquares([1, 2, 3])     // → 14             (1 + 4 + 9)
firstK([5, 6, 7, 8], 2)     // → [5, 6]
\`\`\`

- \`range(n)\`: the starter creates an empty array \`out\`. Use a counting loop to \`push\` each \`i\` onto it.
- \`sum(numbers)\`: the starter creates \`let total = 0\`. Use a for...of loop to add each value to it. It must also work on a \`Float32Array\` (a list that holds only numbers, step 6), which for...of walks the same way.
- \`sumOfSquares(numbers)\`: the same loop as \`sum\`, adding \`x * x\` instead of \`x\`.
- \`firstK(items, k)\`: \`items.slice(0, k)\` makes a **new** array from positions 0 up to, but not including, k, and leaves \`items\` unchanged. Asking for more than there are simply gives them all.

Why these exist: counting loops run over every position of every table in the lab, sums turn into averages and losses, a sum of squares is the heart of the "spread" (variance) that module 01 computes, and \`slice(0, k)\` is how module 00 keeps the top k words.
`,
      hints: [
        'range needs the counting loop (you want the positions 0 to n - 1 themselves). sum and sumOfSquares need for...of (you want the values). firstK needs no loop at all.',
        'range: loop i from 0 while i < n and push i onto out. sum: for each value x of numbers, add x to total. sumOfSquares: the same, adding x times x (not x ^ 2). All three then return the variable the starter already created. firstK: return a slice from 0 to k.',
        'range: `for (let i = 0; i < n; i++) out.push(/* ? */);` then `return out;`. sum: `for (const x of numbers) total += /* ? */;` then `return total;`. sumOfSquares: the same loop with `total += /* x squared, written with * */;`. firstK: `return items.slice(/* start */, /* end */);`.',
      ],
    },
    {
      id: 'compare',
      title: 'Decisions and equality',
      instructions: `
Programs decide with **if**:

\`\`\`js
if (count === 0) {
  return 'never';
} else {
  return 'seen';
}
\`\`\`

\`===\` asks "equal?", \`!==\` asks "different?", and \`<\`, \`>\`, \`<=\`, \`>=\` compare numbers. Each gives \`true\` or \`false\`. A \`return\` inside an \`if\` ends the function right there, which is handy: you can \`return false\` the moment you find a difference.

Implement \`sameShape(a, b)\`: \`true\` when the arrays \`a\` and \`b\` hold the same numbers in the same order, otherwise \`false\`.

\`\`\`js
sameShape([2, 3], [2, 3])      // → true
sameShape([2, 3], [3, 2])      // → false
sameShape([2, 3], [2, 3, 1])   // → false
\`\`\`

The starter's first try is \`return a === b;\`. Press **Check** before changing it and read the red message. Then write the real comparison: if the lengths differ, the answer is \`false\`; otherwise walk the positions with a counting loop and return \`false\` at the first position where \`a[i] !== b[i]\`; if the loop finishes, return \`true\`. The \`return true\` goes **after** the loop's closing \`}\`, not inside it: inside, it would end the function after checking only position 0, so \`[2, 3]\` and \`[2, 4]\` would count as the same.

Why this exists: module 01 stores tables of numbers with a **shape** such as \`[2, 3]\` (2 rows, 3 columns), and before adding two tables it must check their shapes match. \`a.shape === b.shape\` is always \`false\` there, for the reason you are about to see.
`,
      predict: { question: 'What is `[2, 3] === [2, 3]` in JavaScript, and why?', answer: '`false`. Each `[...]` makes a new, separate array. For arrays (and for objects and Maps), `===` asks "is this the very same array?", like asking whether two shopping lists are the same piece of paper. Numbers and strings do compare by value: `2 === 2` and `\'e\' === \'e\'` are true. So to compare two arrays, you compare their lengths and then each position.' },
      hints: [
        'Press Check on the starter first. Two arrays with the same numbers are still two separate arrays; which things can you compare with === safely? Single numbers can.',
        'First compare a.length with b.length and return false if they differ. Then loop over every position i and return false as soon as a[i] and b[i] differ. If nothing differed, return true.',
        'Shape: `if (a.length !== b.length) return false;` then `for (let i = 0; i < a.length; i++) { if (/* a[i] and b[i] differ */) return false; }` and finally `return true;`.',
      ],
    },
    {
      id: 'count',
      title: 'Counting with a Map (and a first taste of patterns)',
      instructions: `
Implement \`countLetters(text)\`: a **Map** from each letter a–z in \`text\` to how many times it appears, ignoring upper/lower case. Anything that is not a letter (spaces, digits, punctuation) is skipped.

\`\`\`js
countLetters('banana')   // → Map { 'b' => 1, 'a' => 3, 'n' => 2 }
countLetters('2024!')    // → an empty Map
\`\`\`

**A Map** is a tally sheet: it stores values under keys. \`const counts = new Map();\` makes an empty one. \`counts.set('a', 3)\` writes 3 under \`'a'\`; \`counts.get('a')\` reads it back. Reading a key that was never set gives \`undefined\` ("nothing written down"). \`x || 0\` means "x, or 0 if x is empty", where empty includes \`undefined\`. (Python users: \`counts[letter] = ...\` does not write into a Map; it quietly sticks a property on the Map object and the Map stays empty. Always use \`set\` and \`get\`.) So the count so far is \`counts.get(letter) || 0\`, and the new count is that plus 1:

\`\`\`js
counts.set(letter, (counts.get(letter) || 0) + 1);
\`\`\`

**Finding the letters** uses a **regular expression**, a small pattern language for text, written between slashes. \`/[a-z]/g\` means: \`[a-z]\` any one character from a to z, and \`g\` ("global") find every match, not just the first. \`text.toLowerCase()\` makes a lower-case copy, and \`.match(pattern)\` returns an array of all the matches, or \`null\` (nothing at all) when there are none. Since you cannot loop over \`null\`, write \`|| []\` after it to use an empty array instead:

\`\`\`js
'Hi, Bo!'.toLowerCase().match(/[a-z]/g) || []    // → ['h', 'i', 'b', 'o']
'2024!'.toLowerCase().match(/[a-z]/g) || []      // → []   (match gave null)
\`\`\`

The same idea finds whole words: \`/[a-z']+/g\` matches runs of one or more (\`+\`) letters or apostrophes, so \`"alice's cat"\` gives \`["alice's", 'cat']\`. That is the pattern module 00 uses.

So: make the array of letters, then loop over it with for...of, updating \`counts\` for each letter, and return \`counts\`.

Why this exists: this exact counting pattern appears in module 00 (words), module 03 (pairs of symbols for the tokenizer) and module 04 (which character follows which).
`,
      predict: { question: 'With `const counts = new Map();`, what is `counts.get(\'q\') + 1`?', answer: '`NaN`. `counts.get(\'q\')` is `undefined` because nothing was stored under \'q\', and `undefined + 1` is not a number. That is why the count so far is written `counts.get(letter) || 0`: `undefined || 0` is 0, so a first sighting becomes 0 + 1 = 1.' },
      hints: [
        'Two parts: turn the text into an array of single lower-case letters (the step shows the exact line), then walk that array updating the tally. What should the tally do with a letter it has never seen?',
        'Make `letters` from `text.toLowerCase().match(/[a-z]/g)`, falling back to `[]` when match gives null. Then for each letter of letters, set its count in `counts` to its current count (0 if missing) plus 1. Return counts.',
        'Shape: `const letters = text.toLowerCase().match(/[a-z]/g) || [];` then `for (const letter of letters) counts.set(letter, (/* count so far, or 0 */) + 1);` then `return counts;`.',
      ],
    },
    {
      id: 'rank',
      title: 'Sorting with a function you pass in',
      instructions: `
Implement \`rankCounts(counts)\`: take a Map like the one from \`countLetters\` and return an array of \`[letter, count]\` **pairs**, largest count first. Letters with equal counts keep the order they had in the Map.

\`\`\`js
rankCounts(new Map([['c', 9], ['a', 2], ['b', 10]]))   // → [['b', 10], ['c', 9], ['a', 2]]
\`\`\`

The starter already does \`const pairs = [...counts];\`. The three dots **spread** the Map into a new array of pairs, \`[['c', 9], ['a', 2], ['b', 10]]\`. Each pair is itself a small array: for \`const pair = ['c', 9]\`, \`pair[0]\` is the letter \`'c'\` and \`pair[1]\` is the count \`9\`.

**Sorting.** \`pairs.sort(compare)\` rearranges \`pairs\` using a function *you pass in*. \`sort\` calls it with two items, \`a\` and \`b\`, and looks at the number it returns: negative means "a goes first", positive means "b goes first", 0 means "leave them as they are". Without a comparator, \`sort\` compares everything as text, where \`'10'\` comes before \`'9'\`.

A short way to write a small function is an **arrow function**: \`(a, b) => b[1] - a[1]\` means "a function of \`a\` and \`b\` that returns \`b[1] - a[1]\`", the same as \`function (a, b) { return b[1] - a[1]; }\`. When b's count is larger the result is positive, so b goes first: largest first. Note it is \`b[1] - a[1]\`, not \`b - a\`: \`a\` and \`b\` are whole pairs, and subtracting two arrays gives \`NaN\`.

\`sort\` rearranges the array in place and also returns it, and since \`pairs\` is a new array the Map is untouched. \`sort\` is **stable**: when your comparator returns 0, equal items keep their order.

Why this exists: module 00 ranks words this way (and adds a tie-break so equal counts go alphabetically), and module 14 sorts a model's candidate next tokens by probability.
`,
      hints: [
        'Press Check with the unsorted starter first. Each item being sorted is a pair such as [\'c\', 9]. Which part of the pair should decide the order?',
        'Call `pairs.sort(...)` with an arrow function of two pairs, a and b, that returns b\'s count minus a\'s count (so a bigger b gives a positive number and goes first). Then return pairs.',
        'Shape: `pairs.sort((a, b) => /* b\'s count */ - /* a\'s count */);` then `return pairs;`. The count of a pair is at position 1.',
      ],
    },
    {
      id: 'table',
      title: 'Objects, typed arrays and errors: a { shape, data } table',
      instructions: `
Every later module stores a table of numbers as two things kept together: its **shape** (\`[rows, cols]\`) and its numbers laid out row after row in one flat list, called \`data\`. Here you build that.

**Objects.** An object groups named values: \`const t = { shape: [2, 3], data: someNumbers };\`, and \`t.shape\` reads one back. When a variable already has the same name as the field, \`{ shape, data }\` is short for \`{ shape: shape, data: data }\`. The reverse, \`const { shape, data } = t;\`, is **destructuring**: it is the same as \`const shape = t.shape; const data = t.data;\`.

**Float32Array.** A list that holds only numbers, stored compactly, the way machine-learning code stores them. \`new Float32Array(6)\` is six zeros; \`Float32Array.from([1, 2, 3])\` copies an ordinary array into one. It keeps about 7 significant digits, so 0.1 becomes 0.10000000149011612. For...of, \`[i]\` and \`.length\` work on it as on an array.

**Errors.** \`throw new Error('message')\` stops the function immediately and reports the message to whoever called it (in the lab, it shows up in red). Use it when the inputs make no sense, instead of returning something wrong.

**1. \`makeTable(rows, cols, values)\`**: if \`values.length\` is not \`rows * cols\`, throw an Error that says so. Otherwise return \`{ shape, data }\` with \`shape = [rows, cols]\` and \`data = Float32Array.from(values)\`.

**2. \`cell(t, row, col)\`**: the number at that row and column (both from 0). Row 0 fills positions 0 to cols − 1, row 1 starts at position cols, and so on, so (row, col) is at position \`row * cols + col\`, where cols is \`shape[1]\`:

\`\`\`
makeTable(2, 3, [1, 2, 3, 4, 5, 6])      data: [1, 2, 3, 4, 5, 6]    row 0 = 1 2 3
cell(t, 1, 2)  → data[1 * 3 + 2] = 6                                 row 1 = 4 5 6
\`\`\`

\`\`\`js
cell(makeTable(2, 3, [1, 2, 3, 4, 5, 6]), 1, 2)   // → 6
makeTable(2, 2, [1, 2, 3])                        // throws: a 2 x 2 table needs 4 values
\`\`\`

Why this exists: this \`{ shape, data }\` object with a \`Float32Array\` inside is exactly what module 01 calls a tensor, and \`cell\` is the two-dimensional case of the position arithmetic you will write there for any number of dimensions.
`,
      hints: [
        'makeTable has two jobs: refuse bad input, then build and return an object. cell has one: work out which position of the flat data list holds (row, col). How many numbers come before the start of row 1? Of row 2?',
        'makeTable: if values.length !== rows * cols, throw new Error with a message; otherwise make shape and data and return { shape, data }. cell: unpack shape and data from t, then read data at row times the number of columns (shape[1]) plus col.',
        'Shapes to fill in (the gaps are marked `/* ? */`):\n\n```js\n// makeTable\nif (values.length !== rows * cols) throw new Error(`a ${rows} x ${cols} table needs ${rows * cols} values`);\nconst shape = [rows, cols];\nconst data = Float32Array.from(/* ? */);\nreturn { shape, data };\n\n// cell\nconst { shape, data } = t;\nreturn data[/* ? */ * shape[1] + /* ? */];\n```',
      ],
    },
  ],
  reflection: [
    'In your own words, why does `[2, 3] === [2, 3]` give false while `2 === 2` gives true, and how does your sameShape get around it?',
    'Pick one red test message you saw in this module. What did you expect your code to do, what did it actually do, and which rule of JavaScript explained the difference?',
    'Explain to someone who knows Python (or spreadsheets) the difference between for...of, for...in and the counting loop, with one example of when you would use each.',
  ],
  stretch: [
    'Change a copy of countLetters to count words with `/[a-z\']+/g` and run it on a paragraph of your choice. This is where module 00 starts. Real tokenizers split text first with a larger pattern of the same kind: GPT-2\'s, in OpenAI\'s tiktoken library, also separates numbers, punctuation and contractions such as "\'s".',
    'Count pairs of neighbouring letters ("th", "he", ...) with a Map keyed by the two-letter string, and rank them with rankCounts. The most common pair is what a byte-pair-encoding tokenizer (the kind GPT-2 and Llama 3 use, built in module 03) would merge into a single token first.',
    'Extend makeTable and cell to three dimensions, shape `[depth, rows, cols]`: what is the position of `(d, row, col)`? NumPy and PyTorch store tensors exactly like this; PyTorch\'s `tensor.stride()` prints the numbers you just worked out.',
  ],
  timeouts: { tests: 15000, demo: 30000 },
};
