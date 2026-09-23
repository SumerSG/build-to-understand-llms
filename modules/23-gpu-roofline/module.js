export default {
  id: '23-gpu-roofline',
  title: 'GPUs, memory bandwidth & the roofline',
  track: 'systems',
  minutes: 90,
  threshold: 'Every kernel is limited either by arithmetic or by memory traffic, and the ratio FLOPs-per-byte decides which; the roofline tells you in advance which fix can help any kernel: more reuse per byte (batching, tiling, fusion, lower precision) below the ridge, faster arithmetic above it.',
  goal: 'A roofline calculator and a tiled matmul: you put every matmul of a Llama-3-8B decode and prefill step on an H100 roofline, compute the batch size that reaches the ridge, measure your own naive-versus-tiled GFLOP/s in the page, and rebuild attention with FlashAttention\'s online softmax.',
  prereqs: ['01-tensors', '08-scaling', '15-kv-cache'],
  recall: [
    { q: 'In module 08 you counted a forward pass as roughly how many FLOPs per parameter per token?',
      options: ['1', '2', '6'], answer: 1,
      why: 'Inference is about 2 FLOPs per parameter per token (multiply and add); 6 is the training figure that adds the backward pass. This module divides that count by the bytes the same pass moves.' },
    { q: 'What does the KV cache from module 15 remove from each decode step?',
      options: ['The attention softmax', 'Recomputing keys and values for all previous tokens', 'The output projection'], answer: 1,
      why: 'With the cache, a decode step multiplies a single row of activations by every weight matrix. Those [1, K] x [K, N] matmuls are exactly the memory-bound shapes you analyse here.' },
    { q: 'In module 01, why was the `i, k, j` matmul loop order faster than `i, j, k`?',
      options: ['It does fewer multiplications', 'Its inner loop reads contiguous memory instead of striding down a column', 'It uses less memory'], answer: 1,
      why: 'Both orders do the same 2*n^3 FLOPs. Only the memory access pattern differs, and here you will measure what that is worth in GFLOP/s.' },
    { q: 'Module 19 stored weights in int4 instead of bf16. What does that do to the bytes moved per token?',
      options: ['Divides them by 4', 'Leaves them unchanged', 'Multiplies them by 2'], answer: 0,
      why: 'int4 is 0.5 bytes per parameter against 2 for bf16. Because decoding is memory-bound, dividing the bytes by 4 is close to dividing the decode time by 4 — quantisation is a latency technique, not only a capacity one.' },
  ],
  review: [
    { q: 'NVIDIA lists the H100 SXM at approximately 989 TFLOP/s dense bf16 and 3.35 TB/s of HBM3 bandwidth. Its ridge point is therefore about…',
      options: ['3 FLOP/byte', '295 FLOP/byte', '3350 FLOP/byte'], answer: 1,
      why: '989e12 / 3.35e12 = 295.2 FLOP/byte. Any kernel below that intensity cannot reach peak no matter how well it is written.' },
    { q: 'A batch-1 decode matmul has an arithmetic intensity of about 1 FLOP/byte. What fraction of the H100 bf16 peak can it reach?',
      options: ['About 0.34%', 'About 34%', '100% if the kernel is written well'], answer: 0,
      why: '1 / 295.2 = 0.34%. The kernel is reading 33 MB of weights to do 33 MFLOP; the tensor cores are idle almost the whole time.' },
    { q: 'Why does increasing the batch size raise arithmetic intensity for a weight matmul?',
      options: ['The weight bytes are read once and reused for every row of the batch', 'Larger batches need fewer FLOPs per token', 'The GPU clocks higher'], answer: 0,
      why: 'FLOPs scale with M while the dominant byte count, the [K, N] weight matrix, does not. That is the entire economic argument for continuous batching (module 16).' },
    { q: 'Tiling a matmul with blockSize B multiplies its arithmetic intensity by roughly…',
      options: ['B', 'B squared', 'nothing; it only reorders the loops'], answer: 0,
      why: 'Each tile loaded into fast memory is reused B times before eviction, so slow-memory traffic drops from about 2n^3 element reads to 2n^3/B.' },
    { q: 'FlashAttention is faster than a naive attention kernel mainly because it…',
      options: ['Does fewer FLOPs', 'Never writes the [seqLen, seqLen] score matrix to HBM', 'Uses a lower precision'], answer: 1,
      why: 'It does the same 4 * seqLen^2 * headDim FLOPs per head. It keeps tiles of Q, K and V in SRAM and recomputes the softmax online, so the quadratic term disappears from the byte count, not from the FLOP count.' },
  ],
  concept: `
:::plain
A GPU, the chip that runs language models, can be slow for two different reasons: it can run out of arithmetic speed, or it can sit waiting for numbers to arrive from its memory. The roofline model is a simple way to tell which: count how many calculations a task does for each byte it has to fetch, and compare that ratio with the chip's own ratio of calculation speed to memory speed. When a model writes text for a single user, each new word needs the whole model read from memory but only a little arithmetic, so the chip mostly waits, using well under one percent of its calculating power in this module's example. That one fact explains much of how LLMs are served: why providers process many users' requests together, why storing a model in smaller numbers can make it faster, and why reading your prompt costs less per token (word or piece of a word) than writing the answer.
:::

## Two roofs over every kernel

A GPU can do two things: arithmetic and moving bytes. An H100 SXM does approximately 989 TFLOP/s of dense bf16 matrix maths (NVIDIA's H100 datasheet) and reads approximately 3.35 TB/s from its HBM3 memory. A kernel that does \`flops\` floating-point operations and moves \`bytes\` bytes therefore cannot finish faster than \`flops / peakFlops\`, and cannot finish faster than \`bytes / bandwidth\`. If the two overlap perfectly — and a well-written kernel prefetches while it computes — the time is the larger of the two:

\`\`\`
time = max(flops / peakFlops, bytes / bandwidth)
\`\`\`

That is the whole roofline model (Williams, Waterman & Patterson, 2009). Rearranged as achieved FLOP/s against **arithmetic intensity** \`intensity = flops / bytes\`, it becomes two straight lines: a sloped memory roof \`intensity * bandwidth\` and a flat compute roof \`peakFlops\`. They meet at the **ridge point** \`peakFlops / bandwidth\`, which for the H100 is \`989e12 / 3.35e12 = 295\` FLOP/byte. A kernel below 295 FLOP/byte is **memory-bound**: the tensor cores wait for HBM. Above it, it is **compute-bound**.

:::deeper Going deeper: newer GPUs
Per GPU, NVIDIA's DGX B200 figures give approximately 2.25 PFLOP/s dense bf16 and 8 TB/s of HBM3e: a ridge of about 281 FLOP/byte, almost the H100's. At FP4, approximately 9 PFLOP/s dense on the same bandwidth, the ridge is 4 times higher, so even more of inference sits on the memory side.
:::

:::predict
A decode step multiplies one token's activations, a \`[1, 4096]\` row, by a \`[4096, 4096]\` bf16 weight matrix. That is 33.6 MFLOP. How many bytes must move, and what intensity does that give?
---
The weights alone are \`4096 * 4096 * 2 = 33.55 MB\`; the input and output rows add 16 KB. So about 33.6 MB for 33.6 MFLOP: an intensity of almost exactly **1 FLOP/byte**. Against a ridge of 295 that is 0.34% of peak: for every nanosecond the tensor cores work, they wait about 300. The accelerator is behaving like a memory controller.
:::

## The same maths, a thousand times better

Now feed 2048 tokens through that same matrix — a prefill, or a batch of 2048 decode requests. The FLOPs multiply by 2048; the weight bytes do not change. Intensity rises to about 1024 FLOP/byte, comfortably above the ridge, and the kernel reaches peak. Nothing about the arithmetic changed. Only the reuse did.

This is why every serving system you have built toward behaves as it does. Continuous batching (module 16) exists to push intensity up the slope. Speculative decoding (module 18) works because verifying four guessed tokens costs one weight read instead of four. Quantisation (module 19) helps decode latency because the memory roof moves, not because int4 arithmetic is faster. And prefill and decode have such different profiles that vLLM and SGLang schedule them separately, and disaggregated serving runs them on different machines entirely.

:::predict
An H100 holds Llama-3-8B in bf16, which is 16.06 GB of weights. Every generated token must read all of them. What is the ceiling on tokens per second for a *single* stream, ignoring attention and overheads?
---
\`3.35e12 / 16.06e9 = 209\` tokens per second, and that is an upper bound no kernel can beat. The compute roof for the same model sits at \`989e12 / (2 * 8.03e9) = 61,600\` tokens/s — 295 times higher. Batching is not an optimisation here; it is the only way to use the hardware you bought.
:::

## The hierarchy under the roof

"Bandwidth" is not one number. On an H100 each streaming multiprocessor has approximately 256 KiB of registers and up to 228 KiB (233,472 bytes) of shared memory (SRAM), backed by a 50 MB L2 cache, then 80 GB of HBM3, then host DRAM across PCIe at roughly 64 GB/s. Aggregate shared-memory bandwidth is roughly an order of magnitude above HBM. Every optimisation in this module is the same move: **keep data in a faster level and reuse it there.** Tiling a matmul into \`blockSize x blockSize\` blocks reuses each loaded tile \`blockSize\` times, cutting slow-memory traffic by that same factor. Kernel fusion avoids a round trip to HBM between two elementwise ops. FlashAttention (Dao et al., 2022) tiles attention so that the score matrix, which is \`[seqLen, seqLen]\` for a sequence of \`seqLen\` tokens, is never written to HBM at all: same FLOPs, about 33 times fewer bytes at 4096 tokens (headDim 128, bf16). The enabling trick is the **online softmax**: process keys a block at a time, keep a running max and sum, and rescale the accumulated output whenever a block raises the max (step 5).

## Where this toy differs from production

:::deeper Going deeper: how engineers measure real kernels
You will benchmark JavaScript on a CPU, and JavaScript reaches roughly 1 GFLOP/s — about a million times below an H100's tensor cores, and far below your CPU's own memory roof, so our tiling experiment measures cache behaviour, not the HBM wall. There are no tensor cores, no warps, no asynchronous copies, and no way to see the effect of fp8. The cost model is also optimistic: it assumes perfect overlap, perfect caching of each tile, and it ignores kernel launch overhead, wave quantisation and the fact that real kernels reach 60–80% of the roofline. Production engineers measure the real thing with Nsight Compute and report **MFU** (model FLOPs utilisation) against the datasheet peak. What survives the simplification is the ordering: which ops are memory-bound, and by how much.
:::
`,
  steps: [
    {
      id: 'roofline',
      title: 'The roofline model',
      instructions: `
Four small functions. Every later step is built from them.

- \`arithmeticIntensity(flops, bytes)\` — FLOPs per byte. Throw an \`Error\` if \`bytes\` is not greater than 0; an op that moves no bytes has no intensity, and returning \`Infinity\` would silently poison every calculation downstream.
- \`ridgePoint(peakFlops, bandwidth)\` — the intensity where the roofs cross, in FLOP/byte.
- \`attainable(intensity, peakFlops, bandwidth)\` — the best FLOP/s achievable at that intensity: the *lower* of the flat compute roof and the sloped memory roof \`intensity * bandwidth\`.
- \`kernelTime(flops, bytes, peakFlops, bandwidth)\` — seconds, assuming compute and traffic overlap perfectly.

The tests check that \`flops / kernelTime\` equals \`attainable(intensity)\` exactly, which is only true if \`kernelTime\` takes the maximum of the two times rather than their sum.
`,
      hints: [
        'Take a kernel that does 1 GFLOP and moves 1 GB, on the H100 row. Work out how long the arithmetic alone would take and how long the traffic alone would take. Which of those two numbers do you actually wait for — and why would the answer change if the hardware could not do both at once?',
        'attainable is `Math.min` of two quantities; kernelTime is `Math.max` of two quantities. Check that the units work: bytes / (bytes/s) is seconds, and intensity (FLOP/byte) times bandwidth (bytes/s) is FLOP/s.',
        '`if (!(bytes > 0)) throw new Error(...)` also rejects `NaN` and negative values, which `bytes === 0` does not.',
      ],
    },
    {
      id: 'opcost',
      title: 'What one matmul costs',
      instructions: `
\`matmulCost(M, K, N, bytesPerElement = 2)\` returns \`{ flops, bytes, intensity }\` for \`[M, K] x [K, N] -> [M, N]\`:

- \`flops = 2 * M * K * N\` — one multiply and one add for each of the \`M * K * N\` terms.
- \`bytes = (M*K + K*N + M*N) * bytesPerElement\` — read both inputs, write the output, each exactly once. This is the *best case*: it assumes every element is touched once, which is what a well-tiled kernel achieves.

Throw if any dimension is not greater than 0.

\`analyseOp(op, hw)\` takes \`{ name, flops, bytes }\` and a row of \`HARDWARE\` and returns
\`{ name, flops, bytes, intensity, bound, seconds, attainedFlops, fractionOfPeak }\`, where \`bound\` is
\`'memory'\` when the intensity is strictly below the device's ridge point and \`'compute'\` otherwise,
and \`fractionOfPeak\` is \`attainedFlops / hw.flops\`. Note that \`bound\` depends on the device: the tests
run the same op on an H100 and on a CPU and expect different answers.
`,
      predict: {
        question: 'A decode matmul is [1, 4096] x [4096, 4096] and a prefill matmul is [2048, 4096] x [4096, 4096]. The prefill does 2048 times the FLOPs. How many times the bytes does it move?',
        answer: 'About twice. The weights (33.55 MB) dominate both and are read once either way; prefill adds 2048 rows of input and output, roughly 33.5 MB more. So 2048x the work for 2x the traffic — intensity rises from 1.0 to 1024 FLOP/byte.',
      },
      hints: [
        'Write the three matrices out: A is M by K, B is K by N, C is M by N. Which of the three dominates when M is 1?',
        'Reuse step 1 inside step 2: intensity is `arithmeticIntensity(flops, bytes)`, seconds is `kernelTime(...)`, attained is `attainable(...)`. Do not re-derive them.',
        '`const ridge = ridgePoint(hw.flops, hw.bandwidth); const bound = intensity < ridge ? "memory" : "compute";` and `fractionOfPeak = attainable(intensity, hw.flops, hw.bandwidth) / hw.flops`.',
      ],
    },
    {
      id: 'batching',
      title: 'The batch size that reaches the ridge',
      instructions: `
\`minBatchForCompute(hw, { K, N, bytesPerElement = 2 })\` returns the batch size \`M\` at which
\`[M, K] x [K, N]\` first reaches the ridge point. Set the intensity equal to the ridge \`R\` and solve for \`M\`:

\`\`\`
2*M*K*N = R * (M*K + K*N + M*N) * b        // b = bytesPerElement
M * (2*K*N - R*b*(K + N)) = R*b*K*N
\`\`\`

If the bracket is 0 or negative, no batch size is ever enough: return \`Infinity\`. Return the exact real
number, not a rounded one — a test feeds your answer back into \`matmulCost\` and checks the intensity is
the ridge point to four decimals.

\`decodeThroughput(hw, { params, bytesPerParam = 2, batch = 1 })\` returns
\`{ memoryBound, computeBound, tokensPerSecond, bound }\` in tokens per second:

- \`memoryBound = batch * hw.bandwidth / (params * bytesPerParam)\` — every step reads all the weights once, and all \`batch\` sequences in the step share that one read.
- \`computeBound = hw.flops / (2 * params)\` — about 2 FLOPs per parameter per token (module 08). Both the FLOPs and the tokens scale with the batch, so this ceiling does not move.
- \`tokensPerSecond\` is the smaller; \`bound\` names which one won (call a tie \`'memory'\`). Throw if \`batch\` is below 1.

You will end this step holding two crossover batches for the same H100 and the same 4096-wide bf16
weights — about 345 and about 295 — and both are right. \`minBatchForCompute\` counts the activation bytes
(\`M*K\` in and \`M*N\` out) that grow with the batch, so it needs a larger \`M\` to clear the ridge.
\`decodeThroughput\` counts only the weight stream, whose intensity is exactly \`2*M / bytesPerParam\`; that
reaches the ridge at \`ridge * bytesPerParam / 2 = 295.2\`, which is the rule of thumb you will hear quoted.
The gap of about 50 between them is the activations.
`,
      hints: [
        'For minBatchForCompute the algebra is one line of rearrangement; do it on paper first and keep `R * bytesPerElement` as a single variable.',
        'Guard the division: compute the denominator `2*K*N - rb*(K + N)` first, return Infinity when it is not positive, and only then divide.',
        'decodeThroughput: `const memoryBound = (batch * hw.bandwidth) / (params * bytesPerParam); const computeBound = hw.flops / (2 * params);` then `Math.min` and a comparison for `bound`.',
      ],
    },
    {
      id: 'tiling',
      title: 'Tiling: buying intensity with loop order',
      instructions: `
\`tiledMatmul(A, B, n, blockSize = 32)\` computes the same product as the worked \`naiveMatmul\`, but
over blocks: six loops, an outer \`i0, j0, k0\` over tile origins in steps of \`blockSize\` and an inner
\`i, j, k\` inside the current tile. \`A\` and \`B\` are flat \`Float32Array\`s of \`n * n\`, row-major.
Because a tile of \`C\` is accumulated across several \`k0\` tiles, the inner loop must *add* to what is
already in \`C\`, not overwrite it. \`blockSize\` need not divide \`n\`: clamp each tile's end with
\`Math.min\`. Throw if \`blockSize\` is below 1 — a block size of 0 would loop forever.

\`blockTraffic(n, blockSize, bytesPerElement = 4)\` returns the bytes such a kernel reads from the slow
level: there are \`(n/blockSize)^3\` tile products, each reading one tile of \`A\` and one of \`B\`
(\`blockSize^2\` elements each), and \`C\` is written once:
\`(2*n^3/blockSize + n^2) * bytesPerElement\`. At \`blockSize = 1\` this reduces to the naive count,
which is how you check the formula. Throw if \`blockSize\` is below 1 here too: the formula divides by
it, and a block of 0 describes no kernel at all.
`,
      predict: {
        question: 'Your tiled matmul does exactly the same 2*n^3 multiply-adds as the naive one. Will it be faster in JavaScript at n = 256, and at n = 1024?',
        answer: 'At n = 256 the whole of B is 256 KB and already fits in cache, so tiling buys little and the extra loop bookkeeping often makes it slower. At n = 1024, B is 4 MB and the naive dot-product order re-reads it column by column; in the goal demo the tiled version typically runs about 1.1x to 2x faster there, though the exact figure depends on your cache sizes and varies from run to run, because a single JavaScript timing is noisy. The shape of the curve is the lesson: tiling pays only once the working set stops fitting in the fast level.',
      },
      hints: [
        'Start from `naiveMatmul` and wrap its three loops in three outer loops over tile origins; the body changes only in its bounds.',
        'Each inner loop runs from the tile origin to `Math.min(origin + blockSize, n)`. Accumulate into a local `let s = C[i * n + j]` and store it back after the k loop so you do not re-read C on every multiply.',
        'One tile bound looks like `const iMax = Math.min(i0 + blockSize, n);` — write the other two the same way. The innermost `k` loop must start from what is already in `C[i*n+j]`, not from 0, because the tile at `k0 = 0` only contributed part of the dot product; the tiles at later `k0` add the rest.',
      ],
    },
    {
      id: 'hierarchy',
      title: 'SRAM, HBM and why FlashAttention wins',
      instructions: `
\`attentionCost(seqLen, headDim, { bytesPerElement = 2, flash = false })\` returns
\`{ flops, bytes, intensity }\` for **one attention head**:

- \`flops = 4 * seqLen^2 * headDim\` — \`2 * seqLen^2 * headDim\` for \`Q·Kᵀ\` and the same again for \`P·V\`. The softmax itself is not counted.
- \`bytes\` with \`flash: false\` — \`Q\`, \`K\`, \`V\` and \`O\` each move once (\`4 * seqLen * headDim\` elements) and the \`[seqLen, seqLen]\` score matrix moves four times: write the scores, read them for softmax, write the probabilities, read them for the second matmul. That is \`4 * seqLen^2\` elements.
- \`bytes\` with \`flash: true\` — the score tiles never leave SRAM, so only the \`4 * seqLen * headDim\` term survives.

Multiply the element count by \`bytesPerElement\` at the end.

\`flashBlockSize(sramBytes, headDim, bytesPerElement = 2)\` returns the largest block of rows whose four
tiles (\`Q\`, \`K\`, \`V\` and the running output \`O\`, each \`blockSize x headDim\`) fit in \`sramBytes\`.
Floor it, and never return less than 1.

Now build the kernel that byte count describes. \`tiledAttention(q, k, v, blockSize, { causal = true } = {})\`
computes the same output as module 05's attention (\`lib/attention.js\`) for one head, the way FlashAttention
does (Dao et al., 2022): it never holds more than \`blockSize\` scores for a query at once. \`q\` is a raw tensor
\`{ shape: [Tq, dh], data }\` of \`Tq\` query rows with head dimension \`dh\`; \`k\` and \`v\` are
\`{ shape: [Tk, dh], data }\` for \`Tk\` keys; all are row-major. Return \`{ shape: [Tq, dh], data }\` with a
\`Float32Array\`. The score of query \`i\` against key \`j\` is \`s = q_i · k_j / sqrt(dh)\`. With \`causal\`
(the default, as in lib/attention.js) query \`i\` sees keys \`j <= i\` only. Throw if \`blockSize\` is below 1.

For each query row keep three things: the running max \`m\` of the scores seen so far (start at \`-Infinity\`),
the running sum \`l\` of \`exp(s - m)\`, and an accumulator \`acc\` of \`dh\` values holding the sum of
\`exp(s - m) * v_j\`. Walk the keys in blocks of \`blockSize\` rows. When a block raises the max from
\`mOld\` to \`mNew\`, every term already in \`l\` and \`acc\` was measured against the old max. Multiplying both
by \`exp(mOld - mNew)\` corrects all of them at once, because \`exp(s - mOld) * exp(mOld - mNew) = exp(s - mNew)\`.
After the last block the output row is \`acc / l\`. Read elements by index (\`k.data[j * dh + c]\`): one test
passes plain arrays so that it can watch the order in which you read \`K\` and \`V\`.

This is what \`attentionCost(seqLen, headDim, { flash: true })\` was pricing: the only score state is
\`blockSize\` numbers per query row, small enough for SRAM, so the \`4 * seqLen^2\` term never reaches HBM.
The byte model is still optimistic. A real kernel cannot hold every row it needs in SRAM at once, so it
re-reads some inputs from HBM once per block: FlashAttention-2 gives each block of query rows to one thread
block, which streams all of \`K\` and \`V\`. The FlashAttention paper's HBM count, which grows as
\`seqLen^2 * headDim^2 / sramSize\`, includes re-reads like these; \`attentionCost\` leaves them out.
`,
      hints: [
        'For attentionCost, write the element counts before you multiply by bytesPerElement: flash keeps the linear term and drops the quadratic one. For tiledAttention, suppose you have added up `exp(s - 3)` over a first block of keys and the next block contains a score of 5. What single number, multiplied into everything you already added, turns each `exp(s - 3)` into `exp(s - 5)`?',
        'flashBlockSize: four tiles of `blockSize * headDim` elements must fit in `sramBytes`; floor, and never go below 1. tiledAttention: for each query row set `m = -Infinity`, `l = 0` and `acc` to dh zeros. For each block of keys, compute its scores and their max, set `mNew = Math.max(m, blockMax)`, multiply `l` and every `acc[c]` by `exp(m - mNew)`, add each `exp(s - mNew)` to `l` and `exp(s - mNew) * v_j` to `acc`, then set `m = mNew`. The output row is `acc / l`. Under causal masking the key loop for row i stops at i + 1.',
        'The block update inside tiledAttention, with the rescale left for you:\n\n```\nconst mNew = Math.max(m, blockMax);\n// ... rescale l and every acc[c] onto mNew here ...\nfor (let j = j0; j < jMax; j++) {\n  const p = Math.exp(s[j - j0] - mNew);\n  l += p;\n  for (let c = 0; c < dh; c++) acc[c] += p * v.data[j * dh + c];\n}\nm = mNew;\n```\n\nOn the first block `m` is `-Infinity`, so the rescale factor is `exp(-Infinity) = 0`, which is harmless because `l` and `acc` are still 0.',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why a batch-1 decode step on an H100 reaches under 1% of the chip\'s peak FLOP/s, and why writing a better kernel cannot fix it.',
    'You are asked to halve the time-to-first-token of a serving system and, separately, to double its tokens-per-second at high load. Using the roofline, say which knob you would reach for in each case and why they are different problems.',
    'Tiling, kernel fusion and FlashAttention are three names for one idea. State that idea in a sentence, then say what plays the role of the "fast level" in each.',
  ],
  stretch: [
    'Add an fp8 row to the analysis: the H100 datasheet lists approximately 1979 TFLOP/s dense fp8, double the bf16 figure, with no change in bandwidth. Recompute the ridge point and the batch needed to reach it, and explain why DeepSeek-V3 reports training in fp8 as a bandwidth win as much as a FLOPs win.',
    'Model chunked prefill as vLLM and SGLang implement it: split an 8192-token prefill into chunks of 512 and interleave them with decode steps of batch 64, then compare the intensity and the time of the mixed batch with running the two phases separately.',
    'Extend `attentionCost` to grouped-query attention with `nKvHeads` smaller than `nHeads` (Llama-3-8B uses 8 and 32) and to a KV cache read during decode, then show why GQA is a bandwidth optimisation for decoding rather than a quality one (you built GQA itself in module 29; here you only count its bytes).',
    'Autotune your tiled matmul the way Triton and CUTLASS autotune real kernels: sweep `blockSize` over 8, 16, 32, 64, 128 at several `n`, plot GFLOP/s, and see whether the best block size matches the one your L2 cache size predicts.',
  ],
  timeouts: { tests: 20000, demo: 90000 },
};
