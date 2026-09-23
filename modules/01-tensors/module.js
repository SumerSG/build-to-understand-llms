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
    { q: 'For C[i, j] = Σₚ A[i, p] · B[p, j] with row-major storage, which loop order lets the inner loop of matmul stream through contiguous memory?', options: ['i, j, p (one dot product per output)', 'i, p, j (scale row p of B, accumulate into row i of C)', 'p, j, i'], answer: 1, why: 'With i-p-j, both B[p, :] and C[i, :] are contiguous rows, so the inner loop is a streaming "scale and add" (an AXPY). i-j-p reads B down a column with stride m.' },
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

- **Transpose** is a statement about index arithmetic: element \`[i, j]\` becomes element \`[j, i]\`. Your version copies each number to the offset it has with the axes swapped. PyTorch does not even copy: \`x.T\` swaps the two strides and returns a *view* of the same memory, which is why a transposed tensor reports \`is_contiguous() == False\`.
- **Reshape** of a contiguous tensor does not touch the data at all; it only changes the shape and recomputes the strides. That is why \`view\` is free in PyTorch, and why \`reshape\` has to fall back to a copy when the input is a non-contiguous view such as a transpose.
- **Broadcasting** reads the smaller tensor with a stride of zero along the axis being repeated, so a bias vector of length \`d\` is "added to every row" by using \`j % d\` as its index.

## Matmul: the operation everything else is made of

In a transformer, well over 90% of the arithmetic is matrix multiplication: the attention projections, the attention scores and the MLP are all matmuls, while softmax, LayerNorm and the activation touch each number only a handful of times. For \`A\` of shape \`[n, k]\` and \`B\` of shape \`[k, m]\`, the product \`C\` has shape \`[n, m]\` and the definition is a triple loop: \`C[i, j] = Σₚ A[i, p] · B[p, j]\`, summing over the shared index \`p\` from 0 to \`k − 1\`. The interesting part is the **order** of the loops. Written as \`i, j, p\` the inner loop reads \`B[p, j]\` down a column, jumping \`m\` floats through memory at every step. Written as \`i, p, j\`, the inner loop reads a contiguous row of \`B\` and writes a contiguous row of \`C\`, so every 64-byte cache line fetched is used in full. In C on matrices too big for the cache this reordering alone is commonly worth 2× or more; in JavaScript the picture depends on size. While \`B\` fits in the CPU's fast caches the two orders run within noise of each other (the \`i, j, p\` loop keeps its running sum in a register, which offsets its strided reads); once \`B\` outgrows them, \`i, p, j\` pulls ahead. On a small cloud VM we measured a tie at \`n = 64\`–\`256\` and \`i, p, j\` about 1.2–1.8× faster at \`n = 512\`. The goal demo times your loop against the \`i, j, p\` order at sizes up to 512 so you can see where the crossover falls on your machine. GPUs go much further (tiling into shared memory, tensor cores) and module 23 revisits this, but the idea starts here: **memory layout decides speed as much as arithmetic does.**

:::predict
Your matmul does \`2·n³\` floating-point operations for two \`n × n\` matrices (one multiply and one add per term of each sum). Roughly how many GFLOP/s (billions of floating-point operations per second) do you expect plain single-threaded JavaScript to reach? For scale, NVIDIA's H100 SXM datasheet lists approximately 989 TFLOP/s of dense bf16 tensor-core throughput, about 1,000,000 GFLOP/s.
---
Roughly 0.3–3 GFLOP/s for a straightforward typed-array loop, depending on the CPU and the browser; a small cloud VM sits at the low end. That is about six orders of magnitude (a factor of 300,000 to 3,000,000) below an H100, and it is the reason serious training and inference run on accelerators. You will measure yours in the goal demo.
:::

## Softmax and why it needs care

Softmax turns a row of scores \`x\` into probabilities: \`p_j = exp(x_j) / Σₖ exp(x_k)\`. In a GPT it appears once in every attention layer (over each query's row of scores, for every head) and once at the output (over the vocabulary, to turn logits into next-token probabilities). The naïve formula fails on real logits because \`exp(1000)\` overflows to Infinity and \`Infinity / Infinity\` is NaN. The fix is to subtract the row maximum first; the result is mathematically identical because the constant cancels in the ratio. Every production kernel does this.

## LayerNorm

LayerNorm (Ba, Kiros & Hinton 2016) normalises each token's feature vector (length \`d\`) to mean 0 and variance 1, then applies a learned per-feature scale \`gamma\` and shift \`beta\`, both of length \`d\`. It keeps activations in a sane range as depth grows; GPT-2 applies it at the input of every attention and MLP sub-block ("pre-LN") plus once after the last block, and you will use it in module 06. The variance divides by \`d\` (biased, as in PyTorch's \`nn.LayerNorm\`), and a small \`eps\` (\`1e-5\`) added *inside* the square root keeps the division finite on constant rows.

## What is deliberately left out

Real tensor libraries store an explicit stride per axis, so transpose, slicing and broadcasting are zero-copy views; they support in-place ops and many dtypes (fp32, bf16, fp16, int8, fp8). Their CPU matmul calls a tuned BLAS library (OpenBLAS, Intel MKL, Apple Accelerate) that blocks for the cache, uses SIMD instructions and every core, and typically runs one to two orders of magnitude faster than a plain loop on the same machine; on a GPU it calls cuBLAS or a fused kernel. Here every op allocates a fresh contiguous \`Float32Array\`, runs on one thread, and supports only the layouts the lab needs. That is a feature: the whole library fits in your head, which is the point of building it.
`,
  steps: [
    {
      id: 'indexing',
      title: 'Row-major indexing and transpose',
      instructions: `
Two functions.

\`offset(shape, indices)\` returns a number: the flat offset of a multi-index. This is a completion problem: the starter already has the loop over axes and the running \`stride\`; you fill in its two-line body. The last axis has stride 1, and each earlier axis has a stride equal to the product of the sizes after it. The offset is the sum of \`index * stride\` over the axes.

\`transpose(a)\`: for a 2-D tensor of shape \`[n, m]\` return a new tensor \`{ shape: [m, n], data }\` where \`out[j, i] = a[i, j]\`. Do not modify the input; allocate a fresh \`Float32Array\`. This one you write from scratch, using the offset rule for both shapes.

The worked examples above the TODO line (\`raw\`, \`fromArray\`, \`toArray\`) show how tensors are created and inspected. Use \`fromArray\` and \`toArray\` freely when you experiment with \`console.log\`.
`,
      predict: { question: 'For shape [2, 3] and indices [1, 2], what offset should your function return?', answer: '5. Row 1 begins at offset 3 (one full row of 3), plus column 2.' },
      hints: [
        'In a [2, 3] matrix, how many flat slots do you skip when the column index goes up by 1? When the row index goes up by 1? Now ask the same question for the middle axis of a [2, 3, 4] tensor.',
        'Inside the loop, axis d contributes its index times the current stride; after that, the stride for the next axis to the left is the current stride times the size of axis d. For transpose, element [i, j] of the input sits at the row-major offset for shape [n, m], and it must land at the row-major offset of [j, i] for shape [m, n].',
        'Offset body: `off += indices[d] * stride;` then grow `stride` by `shape[d]`. Transpose inner assignment: `out[/* offset of [j, i] in an [m, n] matrix */] = a.data[i * m + j];` inside a double loop over i < n and j < m.',
      ],
    },
    {
      id: 'matmul',
      title: 'Matrix multiply',
      instructions: `
Implement \`matmul(a, b)\` for 2-D tensors: \`[n, k] × [k, m] → [n, m]\`, with \`C[i, j] = Σₚ A[i, p] · B[p, j]\`. Throw an \`Error\` if \`a.shape[1] !== b.shape[0]\`.

Use the \`i, p, j\` loop order (outer over rows of A, then over the shared dimension, inner over columns of B) so the inner loop streams through contiguous memory. The test times a 128×128 multiply; a triple loop over typed arrays passes comfortably, but reading \`a.data\` through \`toArray\` inside the loop will not.
`,
      predict: { question: 'The goal demo times your i-p-j matmul against an i-j-p loop. Which wins at n = 64, and which at n = 512?', answer: 'At n = 64 the two are within measurement noise (sometimes i-j-p is slightly ahead): a 16 KB matrix sits in the L1 cache, so strided reads are cheap and i-j-p keeps its sum in a register. At n = 512 each matrix is 1 MB, the column walk through B misses the cache on almost every step, and i-p-j is typically 1.2–2× faster in JavaScript (more in C).' },
      hints: [
        'Start from a zero-filled `Float32Array(n * m)` and accumulate into it; typed arrays start at zero.',
        'Think of row i of C as a weighted sum of the rows of B: row p of B gets weight A[i, p]. So for each row i and each p in 0..k, read the weight A[i, p] once, then add weight times row p of B into row i of C, element by element.',
        '`const av = A[i * k + p];` then an inner loop over j that accumulates `av * B[/* offset of [p, j] */]` into `out[/* offset of [i, j] */]`. Check the inner dimensions before the loops.',
      ],
    },
    {
      id: 'broadcast',
      title: 'Elementwise ops with broadcasting',
      instructions: `
Implement \`add(a, b)\` and \`mul(a, b)\`. The second argument may be:

1. a tensor with exactly the same shape as \`a\` (elementwise; compare the shape arrays, not just the element counts),
2. a plain number (applied to every element), or
3. a 1-D tensor whose length equals the last dimension of \`a\`, applied to every row (this is how a bias is added to every token's vector).

Any other shape must throw an \`Error\` (for example, a \`[3, 2]\` tensor onto a \`[2, 3]\` one: same element count, different shape). Both return a new tensor with \`a\`'s shape. Never mutate the inputs. A single private helper that takes the arithmetic as a function, \`binary(a, b, fn)\`, keeps the two exports to one line each; the reference does exactly that. Full NumPy-style broadcasting is in \`lib/ops.js\` if you are curious.
`,
      hints: [
        'For each flat position i of the output, ask: which element of b belongs there? The answer differs for a number, a same-shape tensor and a row vector.',
        'Decide the case once, before the loop: typeof b is "number"; b has the same shape as a (same length and every axis equal); or b is 1-D with length d, the last dimension of a. Element i of a flat row-major array is in column `i % d`, so that is the index into a row vector.',
        '`function binary(a, b, fn)`: allocate `out`, set `d = a.shape.at(-1)`, then a three-way `if` whose branches each run one loop `out[i] = fn(a.data[i], /* b, b.data[i] or b.data[i % d] */)`, and a final `else throw`. Export `add = (a, b) => binary(a, b, (x, y) => x + y)` and the same for mul.',
      ],
    },
    {
      id: 'rowops',
      title: 'Row-wise reductions and softmax',
      instructions: `
Three functions that operate along the **last axis**, treating the tensor as \`rows × d\` where \`d = shape.at(-1)\`:

- \`sum(a)\`: returns a tensor of shape \`a.shape.slice(0, -1)\`, each entry the sum of a row.
- \`argmax(a)\`: a plain array of the index of the largest value in each row (first index on ties).
- \`softmax(a)\`: same shape as \`a\`; each row becomes \`exp(x − max) / Σ exp(x − max)\`.

The number of rows is \`a.data.length / d\`; row \`r\` occupies offsets \`r*d … r*d + d − 1\`. Each row is its own distribution, with its own maximum and its own sum. Softmax must not produce NaN for logits like \`[1000, 1000, 999]\` or \`[-1000, 0, 1000]\`.
`,
      predict: { question: 'What does `softmax([1000, 1000, 999])` return if you forget to subtract the max?', answer: '`[NaN, NaN, NaN]`: exp(1000) is Infinity, and Infinity / Infinity is NaN. With the max subtracted the exponents are 0, 0, −1 and the result is about [0.422, 0.422, 0.155].' },
      hints: [
        'All three share the same skeleton: `for (r = 0; r < data.length; r += d) { … loop j in 0..d over data[r + j] … }`.',
        'Softmax per row takes three passes over that row: find its max, write exp(x − max) into the output while accumulating the sum, then divide the row by the sum. Only the max works as the shift: it makes the largest exponent exactly 0, so nothing can overflow.',
        'Inside the row loop: `let mx = -Infinity;` pass 1 updates `mx`; pass 2 sets `out[r + j] = Math.exp(/* shifted value */)` and adds it to `z`; pass 3 divides. Reset `mx` and `z` for every row.',
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

\`gamma\` and \`beta\` are 1-D tensors of length \`d\` or \`null\` (meaning 1 and 0); feature \`j\` of every row uses \`gamma[j]\` and \`beta[j]\`. Honour the \`eps\` argument and add it inside the square root. A constant row has zero variance; \`eps\` keeps the division finite. Return a new tensor with \`a\`'s shape.
`,
      hints: [
        'What two numbers do you need about a row before you can write any of its outputs? Each needs its own pass over the row.',
        'Pass 1 computes the mean mu; pass 2 the mean of squared deviations from mu (divide by d); pass 3 writes each output. Precompute `inv = 1 / Math.sqrt(v + eps)` once per row and multiply, rather than dividing d times.',
        'Pass 3: `out[r + j] = (x[r + j] - mu) * inv * g + b;` where `g` is `gamma.data[j]` or 1 when gamma is null, and `b` likewise from beta. Index gamma and beta by the column j, not the flat offset.',
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
