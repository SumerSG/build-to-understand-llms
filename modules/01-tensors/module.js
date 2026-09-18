export default {
  id: '01-tensors',
  title: 'Tensors from scratch',
  track: 'foundations',
  minutes: 90,
  threshold: 'A tensor is a flat array plus a shape; every "matrix" operation is a loop over that flat array whose index arithmetic is fixed by the shape.',
  goal: 'A tiny tensor library (indexing, transpose, matmul, broadcasting, softmax, layernorm) whose outputs match the reference within tolerance, benchmarked and visualised by the goal demo.',
  prereqs: ['00-hello-lab'],
  recall: [
    { q: 'In module 00, why did `countFrequencies` return a `Map` rather than a plain object?', options: ['Maps are faster for numbers', 'Word keys such as "constructor" collide with object internals', 'Objects cannot hold strings'], answer: 1, why: 'A Map has no prototype keys to collide with; you will use the same pattern for pair counts in the BPE tokenizer.' },
    { q: 'What does `new Float32Array(6)` contain right after creation?', options: ['Six `undefined` values', 'Six zeros', 'Six random values'], answer: 1, why: 'Typed arrays are zero-initialised, which is why accumulating a matmul into a fresh Float32Array works without an explicit fill.' },
    { q: 'Zipf\'s law (module 00) says the frequency at rank r is roughly proportional to…', options: ['1 / r', 'r', 'r²'], answer: 0, why: 'Frequency ≈ f(1)/r. Keep this in mind when the tokenizer module decides which pairs to merge first.' },
    { q: 'What is `Math.exp(1000)` in JavaScript?', options: ['A very large finite number', '`Infinity`', '`NaN`'], answer: 1, why: 'Doubles overflow above about 1.8e308 (exp(709.8)). This is why softmax must subtract the row maximum first.' },
  ],
  review: [
    { q: 'For a row-major tensor of shape [2, 3, 4], the flat offset of index [1, 2, 3] is…', options: ['9', '23', '18'], answer: 1, why: '1·(3·4) + 2·4 + 3 = 12 + 8 + 3 = 23. The last axis has stride 1, the middle axis stride 4, the first axis stride 12.' },
    { q: 'Why does softmax subtract the row maximum before exponentiating?', options: ['To make the result sum to one', 'To avoid overflow; the result is mathematically identical', 'To speed up the loop'], answer: 1, why: 'exp(x − m)/Σexp(x − m) equals exp(x)/Σexp(x) exactly, but never overflows because the largest exponent is 0.' },
    { q: 'Which loop order lets the inner loop of matmul stream through contiguous memory?', options: ['i, j, k (dot product per output)', 'i, k, j (scale row of B, accumulate into row of C)', 'k, j, i'], answer: 1, why: 'With i-k-j, both B[k, :] and C[i, :] are contiguous rows, so the inner loop is a fast AXPY. i-j-k reads B column-wise with stride m.' },
    { q: 'LayerNorm normalises…', options: ['Each column across the batch', 'Each row (the last axis) independently', 'The whole tensor'], answer: 1, why: 'Each token\'s feature vector is normalised on its own, so a single sequence can be processed without batch statistics — unlike BatchNorm.' },
  ],
  concept: `
## What a tensor actually is

Every framework, from NumPy to PyTorch to the kernels inside a GPU, represents an n-dimensional array the same way: **one flat block of numbers plus a shape**. There is no "2-D array" in memory. A matrix with shape \`[2, 3]\` is six floats in a row, and the shape tells you how to read them:

\`\`\`
data:  [a b c d e f]      shape [2, 3]     row 0 = a b c
                                            row 1 = d e f
\`\`\`

The convention used here (and by C, NumPy and PyTorch by default) is **row-major**: the last index varies fastest. Element \`[i, j]\` of a \`[n, m]\` matrix lives at flat offset \`i * m + j\`. For three dimensions \`[a, b, c]\` the offset of \`[i, j, k]\` is \`i * b * c + j * c + k\`. The multipliers (\`b*c\`, \`c\`, \`1\`) are the **strides**.

:::predict
A tensor has shape \`[4, 5, 6]\`. What is the flat offset of element \`[2, 3, 4]\`? Work it out before revealing.
---
\`2 * (5*6) + 3 * 6 + 4 = 60 + 18 + 4 = 82\`. The strides are \`[30, 6, 1]\`.
:::

Once you see that shape is just bookkeeping, several things that look like magic become loops:

- **Transpose** copies each element to the offset it would have with the axes swapped. Nothing is "rotated"; the numbers are reordered.
- **Reshape** does not touch the data at all. It only changes the shape (this is why it is free in PyTorch and why \`view\` exists).
- **Broadcasting** reads the smaller tensor with a stride of zero along the axis being repeated, so a bias vector of length \`d\` is "added to every row" by using \`j % d\` as its index.

## Matmul: the operation everything else is made of

In a transformer, well over 90% of the arithmetic is matrix multiplication. The definition is a triple loop: \`C[i, j] = Σₖ A[i, k] · B[k, j]\`. The interesting part is the **order** of the loops. Written as \`i, j, k\` the inner loop reads \`B[k, j]\` with a stride of \`m\`, jumping through memory. Written as \`i, k, j\`, the inner loop reads a contiguous row of \`B\` and writes a contiguous row of \`C\`; on a CPU this alone is often 2–5× faster because of cache lines. GPUs go much further (tiling into shared memory, tensor cores), and module 23 revisits this, but the idea starts here: **memory layout decides speed as much as arithmetic does.**

:::predict
Your matmul does \`2·n³\` floating-point operations for two \`n × n\` matrices. Roughly how many GFLOP/s do you expect plain JavaScript to reach on a laptop? An H100 GPU reaches about 1,000,000 GFLOP/s (1 PFLOP/s) in bf16.
---
Typically 0.5–3 GFLOP/s for a straightforward typed-array loop. That is five to six orders of magnitude below a modern GPU, which is the whole reason data centers exist. You will measure yours in the goal demo.
:::

## Softmax and why it needs care

Softmax turns a row of scores into probabilities: \`p_j = exp(x_j) / Σ exp(x_k)\`. It appears twice in every transformer layer (in attention) and once at the output. The naïve formula fails on real logits because \`exp(1000)\` overflows to Infinity and \`Infinity / Infinity\` is NaN. The fix is to subtract the row maximum first; the result is mathematically identical because the constant cancels in the ratio. Every production kernel does this.

## LayerNorm

LayerNorm (Ba, Kiros & Hinton 2016) normalises each token's feature vector to mean 0 and variance 1, then applies a learned scale \`gamma\` and shift \`beta\`. It keeps activations in a sane range as depth grows; GPT-2 applies it before every attention and MLP block ("pre-LN"), and you will use it in module 06. The variance uses \`1/d\` (biased), and a small \`eps\` (\`1e-5\`) inside the square root prevents division by zero on constant rows.

## What is deliberately left out

Real tensor libraries support arbitrary strides, views, in-place ops and dozens of dtypes. Here every op allocates a fresh \`Float32Array\` and works only on the layouts the lab needs. That is a feature: the whole library fits in your head, which is the point of building it.
`,
  steps: [
    {
      id: 'indexing',
      title: 'Row-major indexing and transpose',
      instructions: `
Two functions.

\`offset(shape, indices)\`: the flat offset of a multi-index. Compute the strides from the shape (last axis has stride 1, each earlier axis multiplies by the sizes after it) and sum \`index * stride\`.

\`transpose(a)\`: for a 2-D tensor of shape \`[n, m]\` return a new tensor of shape \`[m, n]\` where \`out[j, i] = a[i, j]\`. Do not modify the input; allocate a fresh \`Float32Array\`.

The worked examples above the TODO line (\`raw\`, \`fromArray\`, \`toArray\`) show how tensors are created and inspected. Use \`fromArray\` and \`toArray\` freely when you experiment with \`console.log\`.
`,
      predict: { question: 'For shape [2, 3] and indices [1, 2], what offset should your function return?', answer: '5. Row 1 begins at offset 3 (one full row of 3), plus column 2.' },
      hints: [
        'Walk the axes from last to first, keeping a running stride. The last axis contributes `indices[last] * 1`.',
        'Loop `d` from `shape.length - 1` down to 0: `off += indices[d] * stride; stride *= shape[d]`. For transpose, loop over `i` and `j` and write `out[j * n + i] = a.data[i * m + j]`.',
        'Transpose: `const [n, m] = a.shape; const out = new Float32Array(n * m); for (i…) for (j…) out[j * n + i] = a.data[i * m + j]; return { shape: [m, n], data: out };`',
      ],
    },
    {
      id: 'matmul',
      title: 'Matrix multiply',
      instructions: `
Implement \`matmul(a, b)\` for 2-D tensors: \`[n, k] × [k, m] → [n, m]\`, with \`C[i, j] = Σₚ A[i, p] · B[p, j]\`. Throw an \`Error\` if \`a.shape[1] !== b.shape[0]\`.

Use the \`i, p, j\` loop order (outer over rows of A, then over the shared dimension, inner over columns of B) so the inner loop streams through contiguous memory. The test times a 128×128 multiply; a triple loop over typed arrays passes comfortably, but reading \`a.data\` through \`toArray\` inside the loop will not.
`,
      hints: [
        'Start from a zero-filled `Float32Array(n * m)` and accumulate into it; typed arrays start at zero.',
        'For each row i and each p in 0..k: read `av = A[i * k + p]` once, then for each j add `av * B[p * m + j]` into `out[i * m + j]`.',
        '`for (i) for (p) { const av = A[i*k+p]; for (j) out[i*m+j] += av * B[p*m+j]; }` then `return { shape: [n, m], data: out }`.',
      ],
    },
    {
      id: 'broadcast',
      title: 'Elementwise ops with broadcasting',
      instructions: `
Implement \`add(a, b)\` and \`mul(a, b)\`. The second argument may be:

1. a tensor with the same shape as \`a\` (elementwise),
2. a plain number (applied to every element), or
3. a 1-D tensor whose length equals the last dimension of \`a\`, applied to every row (this is how a bias is added to every token's vector).

Any other shape must throw. Never mutate the inputs. A single private helper that takes the arithmetic as a function, \`binary(a, b, fn)\`, keeps the two exports to one line each; the reference does exactly that. Full NumPy-style broadcasting is in \`lib/ops.js\` if you are curious.
`,
      hints: [
        'Three cases: typeof b is "number"; b.data.length equals a.data.length; b is 1-D with b.shape[0] equal to the last dim of a. Decide the case once, before the loop.',
        'For the row-vector case, index b with `i % d` where `d` is the last dimension: element i of the flat array is in column `i % d`.',
        '`function binary(a, b, fn) { const out = new Float32Array(a.data.length); const d = a.shape.at(-1); if (typeof b === "number") … else if (b.data.length === a.data.length) … else if (b.shape.length === 1 && b.shape[0] === d) out[i] = fn(a.data[i], b.data[i % d]) else throw …; return { shape: a.shape.slice(), data: out }; }`',
      ],
    },
    {
      id: 'rowops',
      title: 'Row-wise reductions and softmax',
      instructions: `
Three functions that operate along the **last axis**, treating the tensor as \`rows × d\` where \`d = shape.at(-1)\`:

- \`sum(a)\`: returns shape \`a.shape.slice(0, -1)\`, each entry the sum of a row.
- \`argmax(a)\`: a plain array of the index of the largest value in each row (first index on ties).
- \`softmax(a)\`: same shape as \`a\`; each row becomes \`exp(x − max) / Σ exp(x − max)\`.

The number of rows is \`a.data.length / d\`; row \`r\` occupies offsets \`r*d … r*d + d − 1\`. Softmax must not produce NaN for logits like \`[1000, 1000, 999]\`.
`,
      predict: { question: 'What does `softmax([1000, 1000, 999])` return if you forget to subtract the max?', answer: '`[NaN, NaN, NaN]`: exp(1000) is Infinity, and Infinity / Infinity is NaN. With the max subtracted the exponents are 0, 0, −1 and the result is about [0.422, 0.422, 0.155].' },
      hints: [
        'All three share the same skeleton: `for (r = 0; r < data.length; r += d) { … loop j in 0..d over data[r + j] … }`.',
        'Softmax per row: find the max, compute exp(x − max) into the output while summing, then divide the row by the sum.',
        '`let m = -Infinity; for (j) m = Math.max(m, x[r+j]); let z = 0; for (j) { out[r+j] = Math.exp(x[r+j] - m); z += out[r+j]; } for (j) out[r+j] /= z;`',
      ],
    },
    {
      id: 'layernorm',
      title: 'Layer normalisation',
      instructions: `
Implement \`layerNorm(a, gamma = null, beta = null, eps = 1e-5)\` along the last axis:

\`\`\`
mu  = mean(row)
var = mean((row - mu)^2)          // biased: divide by d
y   = (row - mu) / sqrt(var + eps) * gamma + beta
\`\`\`

\`gamma\` and \`beta\` are 1-D tensors of length \`d\` or \`null\` (meaning 1 and 0). A constant row has zero variance; \`eps\` keeps the division finite.
`,
      hints: [
        'Two passes per row: one for the mean, one for the variance. Then a third to write the output.',
        'Precompute `inv = 1 / Math.sqrt(v + eps)` once per row and multiply, rather than dividing d times.',
        '`for (j) out[r+j] = (x[r+j] - mu) * inv * (gamma ? gamma.data[j] : 1) + (beta ? beta.data[j] : 0);`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why a tensor "is" a flat array plus a shape, and what transpose and reshape each do to the data and to the shape.',
    'Your matmul reached some number of GFLOP/s in the demo. Where does the time go, and what would a GPU change first: the arithmetic or the memory traffic?',
    'Why must softmax subtract the maximum, and why does that not change the answer?',
  ],
  stretch: [
    'Extend `matmul` to batched inputs `[B, n, k] × [B, k, m]`, which attention needs (module 05). `lib/ops.js` shows one way.',
    'Implement full NumPy broadcasting for `add`/`mul` using stride-0 tricks, then compare against `lib/ops.js` on random shapes.',
    'Block (tile) your matmul so each 32×32 tile of C is computed from tiles of A and B, and measure the GFLOP/s change; this is the first idea behind cuBLAS and the tiled kernels in module 23.',
    'Implement `layerNorm` without the second pass by using `E[x²] − E[x]²`, then find an input where that formula loses precision in float32 (Welford\'s algorithm is the fix).',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
