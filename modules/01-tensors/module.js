export default {
  id: '01-tensors',
  title: 'Tensors from scratch',
  track: 'foundations',
  minutes: 150,
  threshold: 'A tensor is a flat array plus a shape; every "matrix" operation is a loop over that flat array whose index arithmetic is fixed by the shape.',
  goal: 'A tiny tensor library (indexing, transpose, matmul, broadcasting, softmax, layernorm) whose outputs match the reference within tolerance, benchmarked and visualised by the goal demo.',
  prereqs: ['00-hello-lab'],
  recall: [
    { q: 'In module 00, why did `countFrequencies` return a `Map` rather than a plain object?', options: ['Maps are faster for numbers', 'Word keys such as "constructor" collide with object internals', 'Objects cannot hold strings'], answer: 1, why: 'A Map has no prototype keys to collide with; you will use the same pattern for pair counts in the BPE tokenizer.' },
    { q: 'Zipf\'s law (module 00) says the frequency at rank r is roughly proportional to…', options: ['1 / r', 'r', 'r²'], answer: 0, why: 'Frequency ≈ f(1)/r. Keep this in mind when the tokenizer module decides which pairs to merge first.' },
    { q: 'With the counting loops from modules 35 and 00, how many times does the body of `for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) { … }` run?', options: ['5', '6', '3'], answer: 1,
      why: 'The inner loop runs 3 times for each of the 2 values of i: 2 × 3 = 6. A grid with 2 rows and 3 columns has 6 cells, and this pair of loops visits each cell once, which is exactly how every function in this module walks a tensor.' },
    { q: 'From module 35: what does `new Float32Array(6)` contain right after creation?', options: ['Six `undefined` values', 'Six zeros', 'Six random values'], answer: 1, why: 'A `Float32Array` holds only numbers, each stored in 32 bits (about 7 significant digits), which is how models store their numbers. It starts filled with zeros, which is why adding up a matrix product into a fresh `Float32Array` works without filling it first.' },
    { q: 'From module 35: what is `Math.exp(1000)` in JavaScript?', options: ['A very large finite number', '`Infinity`', '`NaN`'], answer: 1, why: 'JavaScript numbers overflow above about 1.8e308 (exp(709.8)). This is why softmax must subtract the row maximum first.' },
  ],
  review: [
    { q: 'For a row-major tensor of shape [2, 3, 4], the flat offset of index [1, 2, 3] is…', options: ['9', '23', '18'], answer: 1, why: '1·(3·4) + 2·4 + 3 = 12 + 8 + 3 = 23. The last axis has stride 1, the middle axis stride 4, the first axis stride 12.' },
    { q: 'Why does softmax subtract the row maximum before exponentiating?', options: ['To make the result sum to one', 'To avoid overflow; the result is mathematically identical', 'To speed up the loop'], answer: 1, why: 'exp(x − m)/Σexp(x − m) equals exp(x)/Σexp(x) exactly, but never overflows because the largest exponent is 0.' },
    { q: 'For C[i, j] = Σₚ A[i, p] · B[p, j] with row-major storage, which loop order lets the inner loop of matmul stream through contiguous memory?', options: ['i, j, p (one dot product per output)', 'i, p, j (scale row p of B, accumulate into row i of C)', 'p, j, i'], answer: 1, why: 'With i-p-j, both B[p, :] and C[i, :] are contiguous rows, so the inner loop is a streaming "scale and add" (an AXPY). i-j-p reads B down a column with stride m.' },
    { q: 'LayerNorm normalises…', options: ['Each column across the batch', 'Each row (the last axis) independently', 'The whole tensor'], answer: 1, why: 'Each token\'s feature vector is normalised on its own, so a single sequence can be processed without batch statistics — unlike BatchNorm.' },
  ],
  concept: `
:::plain
Inside a language model, every token (module 00) becomes a list of a few hundred or a few thousand numbers that describes it. A list of numbers is called a vector. Stack the vectors for all the tokens in a sentence and you get a grid of numbers, a matrix, like a block of cells in a spreadsheet. A tensor is the general name for such a grid with any number of dimensions: a list (one dimension), a table (two), a stack of tables (three) and so on. Almost all the work a model does is multiplying these grids together, plus a few simple row-by-row steps: one turns scores into probabilities, another keeps the numbers in a sensible range. In this module you build all of them yourself, starting from one plain list of numbers.
:::

## What a tensor actually is

Every numerical library represents a tensor the same way: **one flat block of numbers plus a shape**. The **shape** lists the size of each dimension: \`[2, 3]\` means 2 rows of 3 columns. There is no "2-D array" in memory. A matrix with shape \`[2, 3]\` is six numbers in a row, and the shape tells you how to read them:

\`\`\`
data:  [a b c d e f]      shape [2, 3]     row 0 = a b c
                                            row 1 = d e f
\`\`\`

The convention used here (and by NumPy and PyTorch, the Python libraries most models are written with) is **row-major**: row 0 comes first, then row 1, so the last index changes fastest as you walk the flat array. Element \`[i, j]\` of an \`[n, m]\` matrix lives at flat position (its **offset**) \`i * m + j\`: skip \`i\` whole rows of \`m\` numbers each, then move \`j\` along. For three dimensions \`[a, b, c]\` the offset of \`[i, j, k]\` is \`i * b * c + j * c + k\`. The multipliers (\`b*c\`, \`c\`, \`1\`) are the **strides**: how many flat positions you skip when that index goes up by one.

:::predict
A tensor has shape \`[4, 5, 6]\`. What is the flat offset of element \`[2, 3, 4]\`? Work it out before revealing.
---
\`2 * (5*6) + 3 * 6 + 4 = 60 + 18 + 4 = 82\`. The strides are \`[30, 6, 1]\`.
:::

Once you see that shape is just bookkeeping, several operations that sound impressive become loops:

- **Transpose** swaps rows and columns: element \`[i, j]\` becomes element \`[j, i]\`, so a \`[2, 3]\` matrix becomes \`[3, 2]\`. Your version copies each number to its new offset.
- **Reshape** reads the same flat numbers with a different shape (six numbers as \`[2, 3]\`, \`[3, 2]\` or \`[6]\`). The data does not move at all.
- **Broadcasting** applies a smaller tensor to every row of a bigger one. The everyday case is adding a **bias**, a list of \`d\` learned numbers, to every token's vector of length \`d\`: the bias is re-read for each row, at column \`i % d\` for flat position \`i\` (\`%\` is the remainder after division).

:::deeper Going deeper: views and strides in PyTorch
PyTorch does not even copy for a transpose: \`x.T\` swaps the two strides and returns a *view* of the same memory, which is why a transposed tensor reports \`is_contiguous() == False\`. Reshape of a contiguous tensor only recomputes the strides, which is why \`view\` is free, and why \`reshape\` has to fall back to a copy when its input is a non-contiguous view such as a transpose. Broadcasting is implemented the same way: the smaller tensor is read with a stride of zero along the repeated axis, so no copy of the bias is ever made.
:::

## Matmul: the operation everything else is made of

A model's layers work by multiplying the token vectors by matrices of learned numbers (its **weights**). In a transformer, the design behind GPT that you build in module 06, well over 90% of the arithmetic is matrix multiplication ("matmul").

Here is one, worked by hand:

\`\`\`
    A            B           C = A · B
 [ 1  2 ]     [ 5  6 ]      [ 19  22 ]
 [ 3  4 ]  ×  [ 7  8 ]  =   [ 43  50 ]
\`\`\`

Each cell of the result sits where **one row of A** crosses **one column of B**: multiply the two lists pair by pair and add the products. Laid out like a spreadsheet:

| C = A · B | column 0 of B is [5, 7] | column 1 of B is [6, 8] |
|---|---|---|
| **row 0 of A is [1, 2]** | 1·5 + 2·7 = **19** | 1·6 + 2·8 = **22** |
| **row 1 of A is [3, 4]** | 3·5 + 4·7 = **43** | 3·6 + 4·8 = **50** |

Cell \`C[1, 0]\` (row 1, column 0) uses row 1 of A, \`[3, 4]\`, and column 0 of B, \`[5, 7]\`: \`3·5 + 4·7 = 43\`. A spreadsheet does exactly this with \`=MMULT(A1:B2, D1:E2)\`. As a formula, for \`A\` of shape \`[n, k]\` and \`B\` of shape \`[k, m]\`, where \`Σₚ\` means "add up over every \`p\` from 0 to \`k − 1\`":

\`\`\`
C[i, j] = Σₚ A[i, p] · B[p, j]
\`\`\`

The row of A and the column of B must have the same length \`k\` (the "inner dimensions" must match), and \`C\` has shape \`[n, m]\`. There is a second way to read the same arithmetic, which is the one your code will use: **row i of C is a weighted sum of the rows of B**, with the numbers in row i of A as the weights. Row 0 of C is \`1 · [5, 6] + 2 · [7, 8] = [5 + 14, 6 + 16] = [19, 22]\`.

The definition is three nested loops, over \`i\`, \`j\` and \`p\`, and the loops can be nested in any order: every order gives the same answer, but not the same speed. In the \`i, j, p\` order the innermost loop walks *down a column* of \`B\`, jumping \`m\` positions through memory at every step. In the \`i, p, j\` order (the weighted-sum reading) it walks *along a row* of \`B\` and a row of \`C\`, reading neighbours. A computer fetches memory in chunks, so reading neighbours is cheaper, and for large matrices the \`i, p, j\` order wins. The goal demo times your loop against both orders so you can see where that starts to matter on your machine. **Memory layout decides speed as much as arithmetic does.**

:::deeper Going deeper: caches, registers and why the crossover moves
Every 64-byte cache line the \`i, p, j\` loop fetches is used in full. In C on matrices too big for the cache this reordering alone is commonly worth 2× or more; in JavaScript the picture depends on size. While \`B\` fits in the CPU's fast caches the two orders run within noise of each other (the \`i, j, p\` loop keeps its running sum in a register, which offsets its strided reads); once \`B\` outgrows them, \`i, p, j\` pulls ahead. GPUs go much further (tiling into shared memory, tensor cores), and module 23 revisits this.
:::

:::predict
Multiplying two \`n × n\` matrices takes \`2·n³\` floating-point operations, or FLOPs (one multiply and one add for each term of each sum; a "floating-point" number is a number with a fractional part). Roughly how many GFLOP/s (billions of FLOPs per second) do you expect plain single-threaded JavaScript to reach? For scale, NVIDIA's H100, a data-centre GPU used to train and serve large models, is rated at approximately 1,000,000 GFLOP/s.
---
Roughly 0.3–3 GFLOP/s for a straightforward typed-array loop, depending on the CPU and the browser; a small cloud VM sits at the low end. That is about six orders of magnitude (a factor of 300,000 to 3,000,000) below an H100, and it is the reason serious training and inference run on accelerators. You will measure yours in the goal demo.
:::

## Softmax and why it needs care

At its last step a language model produces one score for every token in its vocabulary; these raw scores are called **logits**. **Softmax** turns a row of scores into probabilities: positive numbers that add up to 1, with larger scores getting larger shares. It uses \`exp(x)\`, which is \`eˣ\` (\`e ≈ 2.718\`), always positive and fast-growing: \`p_j = exp(x_j) / Σₖ exp(x_k)\`, each exponential divided by the row's total. A GPT uses softmax at the output, to turn logits into next-token probabilities, and inside every attention layer (module 05).

The naïve formula fails on real logits because \`exp(1000)\` is too large for a JavaScript number and becomes \`Infinity\`, and \`Infinity / Infinity\` is \`NaN\` ("not a number"). The fix is to subtract the row's maximum from every score first; the answer is mathematically identical because the same factor cancels above and below the division line, and now the largest exponential is \`exp(0) = 1\`. Every production kernel does this.

Dividing the logits by a **temperature** \`T\` before softmax controls how peaked the result is: \`T < 1\` stretches the gaps between logits and concentrates the probability on the largest, \`T > 1\` shrinks them and flattens the distribution toward uniform. The goal demo shows this on one row; module 04 uses it when you sample text, and module 14 studies it with the other decoding controls.

## LayerNorm

As numbers pass through dozens of layers they can drift to be very large or very small, which makes training unstable. **LayerNorm** (layer normalisation) resets each token's vector to an average of 0 and a spread of 1, then lets the model rescale it with learned numbers. For one row of length \`d\` it computes the **mean** \`mu\` (the average) and the **variance** (the average squared distance from the mean, dividing by \`d\`), subtracts the mean, divides by the square root of the variance, then multiplies feature \`j\` by a learned scale \`gamma[j]\` and adds a learned shift \`beta[j]\`. A small \`eps\` (\`1e-5\`, that is 0.00001) is added to the variance *inside* the square root so a row whose numbers are all equal (variance 0) does not divide by zero. GPT-2 applies LayerNorm before every attention and MLP sub-block, and you will use it in module 06.

:::deeper Going deeper: what real tensor libraries add
Real tensor libraries store an explicit stride per axis, so transpose, slicing and broadcasting are zero-copy views; they support in-place ops and many dtypes (number formats: fp32, bf16, fp16, int8, fp8). Their CPU matmul calls a tuned BLAS library (OpenBLAS, Intel MKL, Apple Accelerate) that blocks for the cache, uses SIMD instructions (one instruction applied to several numbers at once) and every core, and typically runs one to two orders of magnitude faster than a plain loop on the same machine; on a GPU it calls cuBLAS or a fused kernel. The H100 figure above is NVIDIA's datasheet number for dense bf16 on its tensor cores (approximately 989 TFLOP/s). LayerNorm is from Ba, Kiros & Hinton (2016); its variance divides by \`d\` (biased), as in PyTorch's \`nn.LayerNorm\`. Here every op allocates a fresh contiguous \`Float32Array\`, runs on one thread, and supports only the layouts the lab needs. That is a feature: the whole library fits in your head, which is the point of building it.
:::
`,
  steps: [
    {
      id: 'indexing',
      title: 'Row-major indexing and transpose',
      instructions: `
Two functions. A tensor here is an object \`{ shape, data }\`: \`shape\` is an ordinary array such as \`[2, 3]\`, and \`data\` is a \`Float32Array\`, an array that holds only numbers (each stored in 32 bits, which keeps about 7 significant digits) and starts filled with zeros.

\`offset(shape, indices)\` returns a number: the flat offset of a multi-index such as \`[1, 2]\`. This is a completion problem: the starter already has the loop over axes, walking from the last axis to the first, and the running \`stride\`; you fill in its two-line body. The last axis has stride 1, and each earlier axis has a stride equal to the product of the sizes after it. The offset is the sum of \`index * stride\` over the axes.

\`transpose(a)\`: for a 2-D tensor of shape \`[n, m]\` return a new tensor of shape \`[m, n]\` where \`out[j, i] = a[i, j]\`. Do not modify the input: return a new object, do not assign to \`a.shape\` or \`a.data\`, and write into a freshly allocated \`Float32Array\` (a transpose cannot be done safely in place, because it overwrites values it has not read yet). This one you write from scratch, using the offset rule for both shapes. It starts and ends like this:

\`\`\`js
const [n, m] = a.shape;                 // same as: const n = a.shape[0]; const m = a.shape[1];
const out = new Float32Array(n * m);    // n * m zeros, one slot per element
// … a loop over i < n with a loop over j < m inside it, copying a[i, j] to position [j, i] of out …
return { shape: [m, n], data: out };
\`\`\`

The worked examples above the TODO line (\`raw\`, \`fromArray\`, \`toArray\`) show how tensors are created and inspected. Use \`fromArray\` and \`toArray\` freely when you experiment with \`console.log\`.
`,
      predict: { question: 'For shape [2, 3] and indices [1, 2], what offset should your function return?', answer: '5. Row 1 begins at offset 3 (one full row of 3), plus column 2.' },
      hints: [
        'In a [2, 3] matrix, how many flat slots do you skip when the column index goes up by 1? When the row index goes up by 1? Now ask the same question for the middle axis of a [2, 3, 4] tensor.',
        'Inside the loop, axis d contributes its index times the current stride; after that, the stride for the next axis to the left is the current stride times the size of axis d. For transpose, element [i, j] of the input sits at the row-major offset for shape [n, m], which is i * m + j, and it must land at the row-major offset of [j, i] for shape [m, n].',
        '```js\n// offset, inside the loop: add axis d\'s contribution, then update the stride\noff += indices[d] * stride;\n/* … then multiply stride by shape[d] … */\n\n// transpose\nconst [n, m] = a.shape;\nconst out = new Float32Array(n * m);\nfor (let i = 0; i < n; i++) {\n  for (let j = 0; j < m; j++) out[/* offset of [j, i] in an [m, n] matrix */] = a.data[i * m + j];\n}\nreturn { shape: [m, n], data: out };\n```',
      ],
    },
    {
      id: 'matmul',
      title: 'Matrix multiply',
      instructions: `
Implement \`matmul(a, b)\` for 2-D tensors: \`[n, k] × [k, m] → [n, m]\`, with \`C[i, j] = Σₚ A[i, p] · B[p, j]\`. In the code, \`A\` is the tensor \`a\` and its numbers are in \`a.data\`; \`B\` is \`b\` and its numbers are in \`b.data\`. So \`A[i, p]\` is \`a.data[i * k + p]\` and \`B[p, j]\` is \`b.data[p * m + j]\`.

Check the example from the concept before you code: \`[[1, 2], [3, 4]] × [[5, 6], [7, 8]]\` must give \`[[19, 22], [43, 50]]\`, and row 0 of the result is \`1 · [5, 6] + 2 · [7, 8]\`.

If the inner dimensions differ, stop with an error. \`throw\` ends the function immediately and reports the message to whoever called it:

\`\`\`js
if (a.shape[1] !== b.shape[0]) throw new Error(\`matmul: inner dimensions differ: [\${a.shape}] x [\${b.shape}]\`);
\`\`\`

Use the \`i, p, j\` loop order (outer over rows of A, then over the shared dimension p, inner over columns of B): for each row \`i\` and each \`p\`, read the weight \`A[i, p]\` once, then add weight × row \`p\` of B into row \`i\` of C. The inner loop then streams through neighbouring memory. Start from a fresh \`Float32Array(n * m)\`, which is all zeros, and add into it with \`+=\`. Return a new tensor \`{ shape: [n, m], data: out }\` and leave both inputs unchanged. The test times a 32×32 multiply (budget 250 ms) and then a 128×128 one (budget 2000 ms); a triple loop over typed arrays takes a few milliseconds, but calling \`toArray\` or \`offset\` inside the loop blows the first budget.
`,
      predict: { question: 'The goal demo times your matmul against an i-p-j loop and an i-j-p loop. Which of those two wins at n = 64, and which at n = 512?', answer: 'At n = 64 the two are within measurement noise (sometimes i-j-p is slightly ahead): a 16 KB matrix sits in the L1 cache, so strided reads are cheap and i-j-p keeps its sum in a register. At n = 512 each matrix is 1 MB, the column walk through B misses the cache on almost every step, and i-p-j pulls ahead. On a small cloud VM we measured a tie from n = 64 to 256 and i-p-j about 1.2–1.8× faster at n = 512 (more in C).' },
      hints: [
        'Work the concept\'s 2×2 example row by row: which numbers from `a.data` and `b.data` make row 0 of the result, and in what order would a loop read them?',
        'Row i of C is a weighted sum of the rows of B: row p of B gets weight A[i, p]. So loop over i, then over p; read the weight a.data[i * k + p] once, then loop over j adding weight times b.data at [p, j] into out at [i, j]. Check the inner dimensions and throw before the loops.',
        '```js\nconst [n, k] = a.shape;\nconst m = b.shape[1];\n/* … the inner-dimension check from the instructions … */\nconst out = new Float32Array(n * m);\nfor (let i = 0; i < n; i++) {\n  for (let p = 0; p < k; p++) {\n    const av = a.data[i * k + p];            // the weight A[i, p]\n    for (let j = 0; j < m; j++) out[/* offset of [i, j] */] += av * b.data[/* offset of [p, j] */];\n  }\n}\nreturn { shape: [n, m], data: out };\n```',
      ],
    },
    {
      id: 'broadcast',
      title: 'Elementwise ops with broadcasting',
      instructions: `
Implement \`add(a, b)\` and \`mul(a, b)\`, which work element by element ("elementwise"). The second argument may be:

1. a tensor with exactly the same shape as \`a\`: element \`i\` of the result combines \`a.data[i]\` and \`b.data[i]\`;
2. a plain number (applied to every element): \`typeof b === 'number'\` tells you this case;
3. a 1-D tensor whose length equals the last dimension of \`a\`, applied to every row (this is how a bias is added to every token's vector). Flat position \`i\` of \`a\` is in column \`i % d\`, where \`d\` is the last dimension of \`a\`.

Any other shape must throw an \`Error\` (for example, a \`[3, 2]\` tensor onto a \`[2, 3]\` one: same number of elements, different shape). Both return a new tensor with \`a\`'s shape. Never change the inputs.

**Comparing shapes.** In JavaScript, \`[2, 3] === [2, 3]\` and \`[2, 3] == [2, 3]\` are both \`false\`: arrays compare by *identity* (are these the very same array object?), not by contents. So compare shapes entry by entry, as your \`sameShape\` did in module 35: the lengths must match and every entry must match, for example \`s.length === t.length && s.every((v, i) => v === t[i])\`.

**One helper for both.** \`add\` and \`mul\` differ only in the arithmetic, so write one private helper \`binary(a, b, fn)\` that takes the arithmetic as a function argument: \`fn\` is any function of two numbers, such as \`(x, y) => x + y\`, and the helper calls \`fn(a.data[i], …)\` for each element. Then the body of \`add\` is \`return binary(a, b, (x, y) => x + y);\` and \`mul\` is the same with \`*\`. (Full NumPy-style broadcasting is in \`lib/ops.js\` if you are curious.)
`,
      hints: [
        'For each flat position i of the output, ask: which element of b belongs there? The answer differs for a number, a same-shape tensor and a row vector.',
        'Decide the case once, before the loop: b is a number; b has the same shape as a (same length and every entry equal); or b is 1-D with length d, the last dimension of a. In the row-vector case, element i of a is in column i % d, so that is the index into b.data. Anything else: throw.',
        '```js\nfunction binary(a, b, fn) {\n  const out = new Float32Array(a.data.length);\n  const d = a.shape[a.shape.length - 1];\n  if (typeof b === \'number\') {\n    for (let i = 0; i < out.length; i++) out[i] = fn(a.data[i], b);\n  } else if (/* same shape: same length and every entry equal */) {\n    for (let i = 0; i < out.length; i++) out[i] = fn(a.data[i], b.data[i]);\n  } else if (/* b is 1-D and its length is d */) {\n    for (let i = 0; i < out.length; i++) out[i] = fn(a.data[i], b.data[/* the column of i */]);\n  } else {\n    throw new Error(`cannot broadcast [${b.shape}] onto [${a.shape}]`);\n  }\n  return { shape: a.shape.slice(), data: out };\n}\n// inside the existing add:  return binary(a, b, (x, y) => x + y);\n```',
      ],
    },
    {
      id: 'rowops',
      title: 'Row-wise reductions and softmax',
      instructions: `
Three functions that operate along the **last axis**, treating the tensor as \`rows × d\` where \`d = a.shape[a.shape.length - 1]\` (the size of the last axis):

- \`sum(a)\`: returns a tensor \`{ shape: a.shape.slice(0, -1), data }\` (not a bare \`Float32Array\` or array; contrast \`argmax\` below), each entry the sum of a row. \`a.shape.slice(0, -1)\` is the shape without its last entry.
- \`argmax(a)\`: a plain array holding, for each row, the position of the largest value in that row (the first position on ties). It is how a model picks its single most likely next token.
- \`softmax(a)\`: same shape as \`a\`; each row becomes \`exp(x − max) / Σ exp(x − max)\`, with \`Math.exp\` for \`exp\`.

The number of rows is \`a.data.length / d\`; row \`r\` occupies offsets \`r*d … r*d + d − 1\`. Each row is its own distribution, with its own maximum and its own sum: shifting by the whole tensor's maximum makes a row far below it underflow to \`0 / 0 = NaN\`. Softmax must not produce NaN for logits like \`[1000, 1000, 999]\` or \`[-1000, 0, 1000]\`.
`,
      predict: { question: 'What does `softmax([1000, 1000, 999])` return if you forget to subtract the max?', answer: '`[NaN, NaN, NaN]`: exp(1000) is Infinity, and Infinity / Infinity is NaN. With the max subtracted the exponents are 0, 0, −1 and the result is about [0.422, 0.422, 0.155].' },
      hints: [
        'All three share the same skeleton: `for (let r = 0; r < rows; r++) { const base = r * d; … a loop over j from 0 to d − 1 reading a.data[base + j] … }`. sum and argmax store one result per row, at index r; softmax writes a whole row, at base + j.',
        'Softmax per row takes three passes over that row: find its max, write exp(x − max) into the output while adding those values into a running total, then divide the row by the total. Only the max works as the shift: it makes the largest exponent exactly 0, so nothing can overflow.',
        '```js\n// softmax\nconst d = a.shape[a.shape.length - 1];\nconst rows = a.data.length / d;\nconst out = new Float32Array(a.data.length);\nfor (let r = 0; r < rows; r++) {\n  const base = r * d;\n  let mx = -Infinity;\n  for (let j = 0; j < d; j++) mx = Math.max(mx, a.data[base + j]);\n  let z = 0;\n  for (let j = 0; j < d; j++) { out[base + j] = Math.exp(/* the shifted value */); z += out[base + j]; }\n  for (let j = 0; j < d; j++) out[base + j] /= z;\n}\nreturn { shape: a.shape.slice(), data: out };\n```',
      ],
    },
    {
      id: 'layernorm',
      title: 'Layer normalisation',
      instructions: `
Implement \`layerNorm(a, gamma = null, beta = null, eps = 1e-5)\` along the last axis. For every row:

\`\`\`
mu       = mean(row)                        // add the row up, divide by d
variance = mean((row - mu)^2)               // biased: divide by d, not d - 1
y        = (row - mu) / sqrt(variance + eps) * gamma + beta
\`\`\`

(Do not name a JavaScript variable \`var\`: it is a reserved word, and the whole file stops loading. \`variance\` is fine.)

\`gamma\` and \`beta\` are 1-D tensors of length \`d\` or \`null\` (meaning 1 and 0); feature \`j\` of every row uses \`gamma.data[j]\` and \`beta.data[j]\`. Honour the \`eps\` argument and add it inside the square root (\`Math.sqrt\`). A constant row has zero variance; \`eps\` keeps the division finite. Return a new tensor with \`a\`'s shape.
`,
      hints: [
        'What two numbers do you need about a row before you can write any of its outputs? Each needs its own pass over the row.',
        'Pass 1 computes the mean mu; pass 2 the mean of squared differences from mu (divide by d); pass 3 writes each output. Work out `inv = 1 / Math.sqrt(variance + eps)` once per row and multiply by it, rather than dividing d times.',
        '```js\n// inside a loop over rows r, with base = r * d:\nlet mu = 0;\nfor (let j = 0; j < d; j++) mu += a.data[base + j];\nmu /= d;\nlet variance = 0;\n/* … pass 2: add up (a.data[base + j] - mu) squared over the row, then divide by d … */\nconst inv = 1 / Math.sqrt(variance + eps);\nfor (let j = 0; j < d; j++) {\n  const g = gamma ? gamma.data[j] : 1, sh = beta ? beta.data[j] : 0;\n  out[base + j] = (a.data[base + j] - mu) * inv * g + sh;\n}\n```',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why a tensor "is" a flat array plus a shape, and what transpose and reshape each do to the data and to the shape.',
    'Your matmul reached some number of GFLOP/s in the demo. Where does the time go, and what would a GPU change first: the arithmetic or the memory traffic?',
    'Why must softmax subtract the maximum, and why does that not change the answer?',
  ],
  stretch: [
    'Extend `matmul` to batched inputs `[B, n, k] × [B, k, m]`, which attention needs (module 05). `lib/ops.js` shows one way. In PyTorch the same operation is `torch.bmm`, which on NVIDIA GPUs typically hands the work to a cuBLAS batched GEMM routine.',
    'Implement full NumPy broadcasting for `add`/`mul` using stride-0 tricks, then compare against `lib/ops.js` on random shapes.',
    'Block (tile) your matmul so each 32×32 tile of C is computed from tiles of A and B, and measure the GFLOP/s change; this is the first idea behind cuBLAS and the tiled kernels in module 23.',
    'Implement `layerNorm` without the second pass by using `E[x²] − E[x]²`, then find an input where that formula loses precision in float32 (Welford\'s algorithm is the fix).',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
