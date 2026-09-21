export default {
  id: '19-quantization',
  title: 'Quantisation',
  track: 'inference',
  minutes: 90,
  threshold: 'Quantisation replaces each float with a small integer times a shared scale; the error is set by the largest value in the group that shares the scale, so smaller groups and outlier handling are what make 4-bit weights usable.',
  goal: 'Absmax, per-channel and group-wise int8/int4 quantisers (symmetric and with zero points), a weight-only quantised matmul, error metrics and a memory calculator for weights and KV cache; the demo quantises the trained checkpoint, plots error against group size, measures next-token agreement with fp32 over 100 positions, and shows what one outlier does to each scheme.',
  prereqs: ['01-tensors', '15-kv-cache'],
  recall: [
    { q: 'A `Float32Array` of 1,000,000 elements (module 01) occupies how many bytes?',
      options: ['1,000,000', '4,000,000', '8,000,000'], answer: 1,
      why: 'Four bytes per float32. Production models ship in bf16 (2 bytes); this module gets the same weights down to 4.125 bits, about half a byte.' },
    { q: 'In module 15, one token of KV cache for a model with `nLayer` layers, `nHead` heads and head dimension `dh` costs how many floats?',
      options: ['`nLayer * nHead * dh`', '`2 * nLayer * nHead * dh`', '`nLayer * nHead * dh * T`'], answer: 1,
      why: 'A key and a value per head per layer. Llama-3-8B stores 2 * 32 * 8 * 128 = 65,536 values per token, which at 2 bytes each is 128 KB; the KV calculator you build here uses exactly this formula.' },
    { q: 'Module 08 counted a forward pass at roughly how many FLOPs per parameter per token?',
      options: ['1', '2', '6'], answer: 1,
      why: 'One multiply and one add per weight. Decode at batch 1 therefore reads every weight once to do only 2 FLOPs with it, which is why the byte count of the weights, not the arithmetic, sets decode speed.' },
    { q: 'Why does a decode step in module 15 run so much faster with the cache than without?',
      options: ['The matmuls are skipped', 'Each new token attends over stored keys and values instead of recomputing them for the whole prefix', 'The vocabulary is smaller'], answer: 1,
      why: 'With the cache, each step multiplies one token\'s activations by every weight matrix. That is a [1, K] x [K, N] product, the shape whose cost is dominated by reading the K * N weights.' },
  ],
  review: [
    { q: 'Symmetric absmax int8 sets the scale to…',
      options: ['`mean|x| / 127`', '`max|x| / 127`', '`(max − min) / 255`'], answer: 1,
      why: 'The largest magnitude must map to the largest code, 127. Everything else is rounded to a multiple of that scale, so one outlier coarsens the grid for every value that shares the scale.' },
    { q: 'Quantising a [4096, 4096] weight in int4 with groups of 128 and fp16 scales costs how many bits per parameter?',
      options: ['4', '4.125', '4.5'], answer: 1,
      why: '4 bits of code plus one 16-bit scale shared by 128 values: 4 + 16/128 = 4.125. Groups of 32 with a 4-bit zero point would be 4 + 20/32 = 4.625.' },
    { q: 'Why does weight-only int4 speed up decoding at small batch even though the arithmetic is still done in fp16?',
      options: ['int4 multiplies are faster', 'Decode is memory-bound: reading a quarter of the bytes per token is what saves the time', 'The KV cache shrinks'], answer: 1,
      why: 'Each decode step reads every weight once and does 2 FLOPs with it. Cutting the bytes from 2 per parameter to about 0.5 cuts the dominant cost; the dequantisation happens inside the kernel while the GPU would otherwise be waiting on memory.' },
    { q: 'An asymmetric scheme adds a zero point so that…',
      options: ['Negative numbers can be represented', 'All 2^bits codes cover the actual [min, max] of the group instead of [−max|x|, +max|x|]', 'The scale can be an integer'], answer: 1,
      why: 'For a group of values in [2, 3], symmetric int4 spends its 16 codes on [−3, 3] and uses 3 of them; asymmetric spends all 16 on [2, 3], a step 6x finer.' },
    { q: 'LLM.int8 (Dettmers et al. 2022) found that above roughly 6.7B parameters, a few activation channels…',
      options: ['Become exactly zero', 'Reach magnitudes about 20x the rest and destroy per-tensor quantisation unless kept in fp16', 'Stop mattering'], answer: 1,
      why: 'Outlier feature dimensions set the absmax scale for everything else. LLM.int8 keeps those columns in fp16 and quantises the rest; AWQ instead rescales the salient channels before quantising.' },
  ],
  concept: `
## One integer, one shared scale

Store a float \`x\` as an integer code \`q\` and a float \`scale\` shared with its neighbours: \`x ≈ q · scale\`. With 8 bits the codes run from −128 to 127; with 4 bits from −8 to 7. The **absmax** rule picks \`scale = max|x| / qmax\`, where \`qmax\` is the largest code (127 or 7), so the biggest value lands exactly on the last code and every other value is rounded to the nearest multiple of \`scale\`. Rounding to the nearest grid point bounds the error of each value by \`scale / 2\`. That one inequality is the whole subject: the error of *every* value in a group is decided by the *largest* value in that group.

A weight matrix of a trained transformer is mostly small numbers, roughly Gaussian with standard deviation 0.02, with a few entries ten or twenty times larger. One scale for the whole tensor lets those entries set a grid so coarse that most weights round to zero or one step. Giving each row (an output channel) its own scale, **per-channel** quantisation, isolates a big row from the rest. Giving each run of 32, 64 or 128 values along a row its own scale, **group-wise** quantisation, isolates a big value from all but its 31–127 neighbours. The price is storing more scales: int4 with groups of 128 and 16-bit scales costs \`4 + 16/128 = 4.125\` bits per weight.

:::predict
A row holds 128 weights drawn from N(0, 0.02) and one weight of 0.5. In symmetric int4 with one scale for the row, what fraction of the 128 small weights round to code 0?
---
The scale is \`0.5 / 7 ≈ 0.071\`, so any |x| below 0.036 rounds to zero: about 93% of a N(0, 0.02) sample (|x| < 1.8σ). Almost the whole row becomes zero, and the layer's output changes direction. With groups of 32 only the outlier's own group suffers.
:::

## Symmetric or with a zero point

Symmetric schemes waste codes when a group is lopsided. **Asymmetric** quantisation maps the group's actual \`[min, max]\` onto \`[0, 2^bits − 1]\` with \`scale = (max − min) / (2^bits − 1)\` and an integer **zero point** \`zero = round(−min / scale)\`, so \`x ≈ (q − zero) · scale\`. For weights, which are nearly symmetric around zero, the gain is small; for activations and KV cache entries, which are often all positive after a GELU or strongly skewed, it matters. GPTQ and AWQ checkpoints on Hugging Face store a 4-bit zero point per group; KIVI quantises keys per channel and values per token asymmetrically at 2 bits.

## Why 4-bit weights work at all

At batch size 1, a decode step multiplies one row of activations by every weight matrix in the model: about 2 FLOPs per weight, and one read of every weight. NVIDIA's H100 datasheet lists approximately 3.35 TB/s of HBM3 bandwidth against approximately 989 TFLOP/s of bf16 arithmetic, so a kernel at 1 FLOP per byte spends over 99% of its time waiting on memory (module 23 makes this precise). Reading 0.5 bytes per weight instead of 2 cuts that wait by about 4x. This is **weight-only** quantisation: the codes are unpacked to fp16 inside the matmul kernel, one group at a time, and the arithmetic stays in fp16. It needs a matched kernel: vLLM's Marlin and ExLlama kernels, and \`bitsandbytes\` for NF4, do the dequantisation in registers. Without such a kernel a framework dequantises the whole matrix to fp16 first, keeps the memory saving on disk and loses it on the GPU.

:::predict
Llama-3-8B has 8.03 billion parameters. Roughly how many bytes does its weight set occupy in bf16, in int8, and in int4 with groups of 128 and 16-bit scales? Does 70B (70.6 billion parameters) fit on one 80 GB H100 in int4?
---
16.06 GB, 8.03 GB and 4.14 GB (4.125 bits each). Llama-3-70B in int4 g128 is 36.4 GB, so it fits on one 80 GB H100 with room for KV cache; in bf16 it needs 141 GB and at least two GPUs.
:::

## Rounding better than nearest

Nearest rounding is not the best you can do. **GPTQ** (Frantar et al. 2022) quantises a weight matrix one column at a time and, after rounding each column, adjusts the not-yet-quantised columns to compensate for the error, using the inverse of the Hessian \`H = 2 · X · Xᵀ\` estimated from a small **calibration set** (typically 128 sequences of 2,048 tokens from C4). **AWQ** (Lin et al. 2023) observes that about 1% of channels carry most of the activation magnitude, scales those weight channels up before rounding and the activations correspondingly down, and searches the scaling exponent on calibration data. **LLM.int8** (Dettmers et al. 2022) found that models above roughly 6.7B parameters develop a handful of activation channels with magnitudes about 20x the rest, and keeps those columns in fp16 while quantising everything else to int8 per row and per column. **NF4** (QLoRA, Dettmers et al. 2023) replaces the uniform 16-level grid with 16 quantiles of a normal distribution, in blocks of 64, and quantises the scales themselves to 8 bits.

Typical cost, measured as WikiText-2 perplexity: approximately 0.1–0.3 points for 7B–70B models in int4 with groups of 128 under GPTQ or AWQ (their papers report Llama-2-7B going from 5.47 to about 5.6), rising steeply at 3 bits and below, and larger models tolerate quantisation better than small ones. Hopper GPUs add hardware **fp8** (E4M3 for weights and activations, E5M2 for gradients) at approximately twice the bf16 tensor-core rate, used by DeepSeek-V3 for training and by vLLM and TensorRT-LLM for serving; fp8 needs no groups because the exponent gives every value its own scale. The **KV cache** quantises too: vLLM's \`kv_cache_dtype="fp8"\` halves it, and KIVI reaches 2 bits.

## Where this toy differs from production

Your int4 codes occupy one byte each in an \`Int8Array\`; real kernels pack two per byte and reorder them for the tensor cores. Your kernel dequantises on a CPU in JavaScript, so you will see no speed-up, only the memory arithmetic and the error; the speed-up exists only when a fused GPU kernel does the same loop. Your quantiser rounds to nearest; GPTQ and AWQ would be 2–3 steps of extra code on top of what you build. And the checkpoint is a 2-layer, 64-dimensional model without the outlier features that make quantising a 70B model hard, so the demo injects an outlier by hand to show the mechanism.
`,
  steps: [
    {
      id: 'absmax',
      title: 'Absmax symmetric int8',
      instructions: `
Complete \`quantizeAbsmax(x, bits = 8)\` and write \`dequantizeAbsmax(qt)\`.

\`x\` is a raw tensor, a \`Float32Array\` or a plain array (the worked \`flat\` helper handles all three). With \`{ qmin, qmax } = qrange(bits)\` (127 and −128 for int8, 7 and −8 for int4) the rule is:

\`\`\`
scale = max|x| / qmax          (use 1 when max|x| is 0)
q[i]  = clamp(round(x[i] / scale), qmin, qmax)
\`\`\`

Return \`{ q: Int8Array, scale, bits }\`. \`dequantizeAbsmax\` returns a \`Float32Array\` of \`q[i] * scale\`. The starter already finds \`amax\` for you.

The round-trip error of every value is at most \`scale / 2\`; the tests check that bound and that int4 has roughly 256x the mean squared error of int8.
`,
      predict: { question: 'You quantise 1,000 samples from N(0, 0.5) to int8 absmax. Which value decides the scale, and what does that make the error bound for the 990 values near zero?', answer: 'The single largest |x|, around 1.6 for 1,000 Gaussian samples. scale ≈ 1.6 / 127 ≈ 0.0126, so every value is within 0.0063 of its reconstruction. One outlier of 16 would make that bound ten times worse for everyone.' },
      hints: [
        'The codes are integers; `Math.round` rounds to nearest. Read `qmin` and `qmax` from `qrange(bits)` rather than hard-coding 127.',
        'scale = amax > 0 ? amax / qmax : 1. Then loop once more over the data: divide by scale, round, clamp into [qmin, qmax], store in the Int8Array.',
        '`for (i) q[i] = clamp(Math.round(data[i] / scale), qmin, qmax);` and for dequantize `out[i] = qt.q[i] * qt.scale`.',
      ],
    },
    {
      id: 'perchannel',
      title: 'One scale per channel',
      instructions: `
Write \`quantizePerChannel(w, bits = 8)\` and the general \`dequantize(qw)\`.

\`w\` is a raw \`[rows, cols]\` tensor. Apply the absmax rule of step 1 to each row separately and return the quantised-matrix struct from the top of the file: \`{ shape, q, scales, zeros: null, bits, groupSize: cols }\`, with \`scales[r]\` the scale of row \`r\` and \`q\` laid out exactly like \`w.data\`.

\`dequantize\` must handle the general struct, not only per-channel: a row is split into \`nGroups = cols / groupSize\` groups, and value \`[r, c]\` uses \`scales[r * nGroups + floor(c / groupSize)]\`. If \`qw.zeros\` is present, subtract \`zeros[same index]\` from the code before scaling. Per-channel is the case \`groupSize === cols\`; step 3 will produce smaller groups without changing this function.

The third test injects one row 100x larger than the others and checks that the small rows are more than 100x more accurate than under a single tensor-wide scale.
`,
      hints: [
        'Row r occupies offsets r*cols … r*cols + cols − 1. Find its amax, compute its scale, then quantise just those values.',
        'In dequantize, loop over rows, then groups g, then the groupSize values in the group; base = r*cols + g*groupSize. Read the scale (and zero) once per group.',
        '`const idx = r * nGroups + g; const s = qw.scales[idx]; const z = qw.zeros ? qw.zeros[idx] : 0; for (k) out[base + k] = (qw.q[base + k] - z) * s;`',
      ],
    },
    {
      id: 'groups',
      title: 'Group-wise int4, with and without a zero point',
      instructions: `
Write \`quantizeGroups(w, { bits = 4, groupSize = 64, symmetric = true })\`.

Throw if \`cols % groupSize !== 0\`. For each row and each group of \`groupSize\` consecutive values find \`min\` and \`max\`, then:

- **symmetric**: \`scale = max(|min|, |max|) / qmax\`, \`zero = 0\`, and \`zeros\` is \`null\`;
- **asymmetric**: \`{ qmin, qmax } = qrange(bits, false)\` is \`[0, 2^bits − 1]\`, \`scale = (max − min) / (qmax − qmin)\`, \`zero = round(−min / scale)\`, stored in \`zeros[r * nGroups + g]\`.

Either way \`q = clamp(round(x / scale) + zero, qmin, qmax)\`, and \`scale\` falls back to 1 when the group is constant. Return the same struct as step 2 with the given \`groupSize\`.

The tests check that error falls strictly as the group shrinks on a matrix with outliers, and that an all-positive group in \`[2, 3]\` is more than 4x more accurate asymmetrically than symmetrically. Your \`dequantize\` from step 2 should already handle both.
`,
      predict: { question: 'A [16, 256] matrix has four outliers per row at 25x the typical weight. Going from groupSize 256 (per-channel) to 16, does the int4 error drop by about 2x, 4x, or more than 10x?', answer: 'More than 10x. With per-channel scales every one of the 256 values in a row is graded on the outlier\'s scale; with groups of 16, only the 4 groups that contain an outlier are, and the other 12 use a grid about 25x finer.' },
      hints: [
        'The group loop is the row loop of step 2 with one more level: for each row, for each group g, base = r*cols + g*groupSize. Scale index is r*nGroups + g.',
        'Compute lo and hi in one pass over the group. Symmetric: amax = max(|lo|, |hi|). Asymmetric: range = hi − lo; scale = range / (qmax − qmin); zero = Math.round(−lo / scale).',
        '`for (k) q[base + k] = clamp(Math.round(w.data[base + k] / scale) + zero, qmin, qmax);` with zero = 0 in the symmetric case and `zeros[r * nGroups + g] = zero` in the asymmetric one.',
      ],
    },
    {
      id: 'qmatmul',
      title: 'Weight-only quantised matmul',
      instructions: `
Write \`quantizeWeight(w, opts)\` and \`quantizedMatmul(x, qw)\`.

A linear layer in this lab computes \`y = x · W\` with \`W\` of shape \`[K, N]\` (\`K\` input channels, \`N\` output channels). A decode kernel wants each output channel's weights contiguous, with groups along \`K\`, so \`quantizeWeight\` transposes \`W\` to \`[N, K]\` (\`ops.transpose\`) and calls \`quantizeGroups\` on that. This is the layout GPTQ, AWQ and vLLM's Marlin kernel use.

\`quantizedMatmul(x, qw)\` takes \`x\` as raw \`[T, K]\` and returns raw \`[T, N]\`. For each \`t\` and each output channel \`n\`, walk the groups of row \`n\`: accumulate \`Σ x[t, k] · (q[n, k] − zero)\` over the group's \`K\` indices, multiply that partial sum by the group's scale, and add it to the output. The float weight is never materialised; one scale multiply per group is what a fused GPU kernel does in registers. Throw if \`x.shape[1] !== qw.shape[1]\`.

The tests compare your kernel against \`ops.matmul(x, transpose(dequantize(qw)))\` for a symmetric int8 weight and an asymmetric int4 one, and check relative error against fp32 (under 1% for int8 per-channel, under 10% for int4 g32).
`,
      hints: [
        'quantizeWeight is one line once you see the layout. For the matmul, index x as x.data[t*K + k] and the codes as qw.q[n*K + k]; the scale index for row n, group g is n*nGroups + g.',
        'Four nested loops: t, n, g, k (k from g*G to (g+1)*G). Keep a `part` sum inside the g loop and do `acc += scale * part` after it.',
        '`let acc = 0; for (g) { const s = qw.scales[n*nGroups+g], z = qw.zeros ? qw.zeros[n*nGroups+g] : 0; let part = 0; for (k = g*G; k < (g+1)*G; k++) part += x.data[t*K+k] * (qw.q[n*K+k] - z); acc += s * part; } out[t*N+n] = acc;`',
      ],
    },
    {
      id: 'memory',
      title: 'The memory calculator',
      instructions: `
Write \`weightBytes(params, { bits, groupSize = Infinity, scaleBits = 16, zeroBits = 0 })\`, \`bitsPerParam(opts)\` and \`kvCacheBytes(modelCfg, { contextLen, batch = 1, bits = 16 })\`.

Weights cost \`params · bits / 8\` bytes for the codes plus one scale and (optionally) one zero point per group: \`(params / groupSize) · (scaleBits + zeroBits) / 8\`. A per-tensor scale (\`groupSize = Infinity\`) adds nothing measurable. \`bitsPerParam\` is the same quantity per weight, in bits: int4 g128 with fp16 scales is 4.125; int4 g32 with fp16 scales and 4-bit zeros is 4.625.

KV cache: a key and a value per layer per KV head per position, so \`2 · nLayer · nKvHeads · headDim · contextLen · batch · bits / 8\`. The exported \`LLAMA3_8B\` and \`LLAMA3_70B\` constants carry the Llama 3 configs (Meta, 2024): 32 and 80 layers, 8 KV heads each thanks to grouped-query attention, head dimension 128. Llama-3-8B at 8,192 tokens in fp16 is exactly 1 GiB; without GQA (32 KV heads) it would be 4 GiB.
`,
      predict: { question: 'Llama-3-8B, batch 64, context 8,192, fp16 KV cache. Does the cache or the bf16 weight set take more memory?', answer: 'The cache: 64 GiB against 16 GB of weights. This is why KV-cache quantisation (fp8 in vLLM, 2-bit in KIVI) and paged allocation (module 16) matter as much as weight quantisation for throughput.' },
      hints: [
        'Two terms for weights: codes and overhead. Guard the overhead with `groupSize === Infinity ? 0 : …`.',
        'bitsPerParam(opts) = weightBytes(1, opts) * 8 reuses the formula with a single parameter.',
        '`return 2 * nLayer * nKvHeads * headDim * contextLen * batch * (bits / 8);`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why one weight of 0.5 in a row of 128 weights near 0.02 ruins symmetric int4 for the whole row, and how group size, zero points and AWQ-style channel scaling each address it differently.',
    'Weight-only int4 does no int4 arithmetic. Say exactly where the decode time is saved and why the same trick buys little for a 2,048-token prefill.',
    'Your demo measured top-1 agreement between quantised and fp32 next-token predictions. Why is that a stricter test than perplexity for some errors and a looser one for others?',
  ],
  stretch: [
    'Implement GPTQ\'s error feedback for one weight matrix: quantise column by column and add each column\'s rounding error, weighted by the inverse Hessian from calibration activations, into the remaining columns. Compare its error with nearest rounding at int4 g128 (Frantar et al. 2022; the reference implementation is in AutoGPTQ).',
    'Implement AWQ\'s scaling search: for each input channel compute the mean activation magnitude on calibration text, scale the weight rows by `s = mean^α` and the activations by `1/s`, and grid-search α in [0, 1] to minimise the output error (Lin et al. 2023; the kernels live in vLLM and llm-awq).',
    'Replace the uniform int4 grid with NF4\'s 16 normal-distribution quantiles in blocks of 64 and quantise the scales to 8 bits, as QLoRA does in `bitsandbytes`. Measure whether the quantile grid beats uniform int4 on this checkpoint\'s Gaussian-looking weights.',
    'Quantise the KV cache from module 15 instead of the weights: keys per channel, values per token, asymmetric, at 4 and 2 bits, and measure next-token agreement as context grows (KIVI, Liu et al. 2024; vLLM\'s `kv_cache_dtype="fp8"`).',
  ],
  timeouts: { tests: 20000, demo: 90000 },
};
