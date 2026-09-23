export default {
  id: '19-quantization',
  title: 'Quantisation',
  track: 'inference',
  minutes: 105,
  threshold: 'Quantisation replaces each float with a small integer times a shared scale; the error is set by the largest value in the group that shares the scale, so smaller groups and outlier handling are what make 4-bit weights usable.',
  goal: 'Absmax, per-channel and group-wise int8/int4 quantisers (symmetric and with zero points), a weight-only quantised matmul, error metrics, a memory calculator for weights and KV cache, and round-to-nearest-even into bf16, fp16, fp8 and FP4 with MXFP4 and NVFP4 block scales; the demo quantises the trained checkpoint, plots error against group size, measures next-token agreement with fp32 over 100 positions, shows what one outlier does to each scheme, and compares MXFP4 and NVFP4 with int4 g128.',
  prereqs: ['01-tensors', '08-scaling', '15-kv-cache'],
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
      options: ['Become exactly zero', 'Reach magnitudes up to about 20x the rest and wreck int8 quantisation of the other channels unless kept in fp16', 'Stop mattering'], answer: 1,
      why: 'Outlier feature dimensions set the absmax scale for everything else. LLM.int8 keeps those columns in fp16 and quantises the rest; AWQ instead rescales the salient channels before quantising.' },
  ],
  concept: `
:::plain
Quantisation stores a model's numbers in fewer bits (binary digits), say 4 instead of 16. The model then fits on fewer or cheaper chips and often answers faster, since writing each word mostly means reading the model from memory. The price is some accuracy: usually small at 4 bits, worse below. An "int4", "fp8" or "4-bit" model makes this trade.
:::

## One integer, one shared scale

Store a float \`x\` as an integer code \`q\` and a float \`scale\` shared with its neighbours: \`x ≈ q · scale\`. Codes run −128…127 at 8 bits and −8…7 at 4 bits. The **absmax** rule picks \`scale = max|x| / qmax\`, where \`qmax\` is the largest code (127 or 7), so the biggest value lands on the last code and every other value rounds to the nearest multiple of \`scale\`, an error of at most \`scale / 2\`. That is the whole subject: the error of *every* value in a group is decided by the *largest* value in that group.

A trained weight matrix is mostly small numbers, roughly N(0, 0.02), with a few entries ten or twenty times larger. One scale for the whole tensor lets those entries set a grid so coarse that most weights round to zero. One scale per row (an output channel), **per-channel** quantisation, isolates a big row from the rest. One scale per run of 32, 64 or 128 values along a row, **group-wise** quantisation, isolates a big value from all but its neighbours. Scales cost bits: int4 with groups of 128 and 16-bit scales is \`4 + 16/128 = 4.125\` bits per weight.

:::predict
A row holds 128 weights drawn from N(0, 0.02) and one weight of 0.5. In symmetric int4 with one scale for the row, what fraction of the 128 small weights round to code 0?
---
The scale is \`0.5 / 7 ≈ 0.071\`, so any |x| below 0.036 rounds to zero: about 93% of a N(0, 0.02) sample (|x| < 1.8σ). With groups of 32 only the outlier's own group suffers.
:::

## Symmetric or with a zero point

Symmetric schemes waste codes on lopsided groups. **Asymmetric** quantisation maps the group's actual \`[min, max]\` onto \`[0, 2^bits − 1]\` with \`scale = (max − min) / (2^bits − 1)\` and an integer **zero point** \`zero = round(−min / scale)\`, the code that stands for 0.0, so \`x ≈ (q − zero) · scale\`. For weights, nearly symmetric around zero, the gain is small; for skewed tensors it is large: GELU outputs never go below about −0.17, and some key channels sit on a large constant offset (which is why KIVI quantises keys per channel, asymmetrically). GPTQ and AWQ checkpoints store the zero point in 4 bits per group, so it must lie in \`[0, 15]\`; GPTQ ensures that by widening each group's range to include 0. Your quantiser leaves the zero point unconstrained, which lets the step-3 test's all-positive group \`[2, 3]\` use all 16 codes.

## Why 4-bit weights work at all

At batch size 1 a decode step reads every weight once and does about 2 FLOPs with it. NVIDIA's H100 SXM datasheet lists about 3.35 TB/s of HBM3 bandwidth and about 989 TFLOP/s of dense bf16, so a kernel doing 1 FLOP per byte spends over 99% of its time waiting on memory (module 15; module 23 draws it on the roofline). Reading 0.5 bytes per weight instead of 2 cuts that wait by about 4x. This is **weight-only** quantisation: the matmul kernel unpacks the codes to fp16 group by group and computes in fp16. It needs a matched kernel (vLLM ships Marlin and ExLlamaV2 for GPTQ and AWQ checkpoints; \`bitsandbytes\` handles NF4), or the whole matrix is dequantised first and the saving exists only on disk. Prefill is compute-bound and gains little.

:::predict
Llama-3-8B has 8.03 billion parameters. How many bytes are its weights in bf16, int8, and int4 with groups of 128 and 16-bit scales? Does 70B (70.6 billion parameters) fit on one 80 GB H100 in int4?
---
16.06 GB, 8.03 GB and 4.14 GB (4.125 bits each). Llama-3-70B in int4 g128 is 36.4 GB, so it fits one 80 GB H100; in bf16 it needs 141 GB and two GPUs.
:::

## Rounding better than nearest

**GPTQ** (Frantar et al. 2022) quantises a matrix column by column and nudges the remaining columns to cancel each rounding error, weighted by the inverse Hessian \`H = 2 · X · Xᵀ\` from a small **calibration set** (typically 128 sequences). **AWQ** (Lin et al. 2023) finds that about 1% of input channels carry most of the activation magnitude, scales those weight channels up before rounding (and the activations down to match) by a factor tuned on calibration data. **LLM.int8** (Dettmers et al. 2022) found that models above roughly 6.7B parameters develop a few activation channels about 20x larger than the rest, and keeps those columns in fp16, the rest in int8. **NF4** (QLoRA, Dettmers et al. 2023) replaces the uniform grid with 16 quantiles of a normal distribution in blocks of 64, with 8-bit scales.

:::deeper Going deeper: quality cost and fp8 hardware
Typical cost, as WikiText-2 perplexity: approximately 0.1–0.3 points for 7B–70B models in int4 g128 under GPTQ or AWQ (AWQ reports Llama-2-7B going from 5.47 to about 5.6), rising steeply below 4 bits; larger models tolerate it better. Hopper GPUs add hardware **fp8** (conventionally E4M3 for weights and activations, E5M2 for gradients) at approximately twice the dense bf16 rate; vLLM and TensorRT-LLM serve in it and DeepSeek-V3 trains in it. Each fp8 value carries its own exponent, so it tolerates spread inside a tensor far better than an integer grid, but it still needs scales to fit a tensor into E4M3's range: one per tensor in most serving recipes, one per 1x128 activation tile and per 128x128 weight block in DeepSeek-V3, which uses E4M3 throughout. The **KV cache** quantises too: vLLM's \`kv_cache_dtype="fp8"\` halves it.
:::

## Floats with fewer bits, and scales per block

A float stores a sign, an exponent field \`e\` of \`E\` bits and a mantissa \`m\` of \`M\` bits, and means \`x = ±1.m · 2^(e − bias)\`, where \`bias\` is a fixed offset that lets the exponent go negative. Every power-of-two interval holds \`2^M\` evenly spaced values, so the rounding error is relative: a small value gets a small step. **bf16** (8 exponent bits, 7 mantissa bits) keeps fp32's range, up to about 3.4e38, and gives up precision; **fp16** (5 and 10) keeps three more mantissa bits but tops out at 65504, and its smallest subnormal is about 6e-8, which is why fp16 training scales the loss up to keep small gradients from flushing to zero and bf16 training usually does not. fp8 comes as **E4M3** (largest value 448, no infinities) and **E5M2** (largest 57344). The 4-bit **E2M1** has fifteen values: 0 and ±{0.5, 1, 1.5, 2, 3, 4, 6}.

Four bits of float cannot span a weight matrix alone, so they carry a shared scale, as your integers do. The Open Compute Project's **MX** (microscaling) specification (2023) gives every block of 32 elements one power-of-two scale stored as an 8-bit exponent: MXFP4 costs \`4 + 8/32 = 4.25\` bits per weight. NVIDIA's **NVFP4** uses blocks of 16 with an E4M3 scale per block and one fp32 scale per tensor: \`4 + 8/16 = 4.5\` bits. This is step 3's lesson built into the format: an outlier can coarsen only its own 16 or 32 neighbours. NVIDIA says its Blackwell GPUs apply these block scales in hardware, inside FP4 tensor cores. OpenAI released gpt-oss (2025) with its mixture-of-experts weights in MXFP4, which it says lets the 120B model run on one 80 GB GPU.

## Where this toy differs from production

Your int4 codes occupy one byte each in an \`Int8Array\`; real kernels pack two per byte and reorder them for the tensor cores. Your kernel dequantises on a CPU in JavaScript: you will see the memory arithmetic and the error but no speed-up, which needs a fused GPU kernel. Your quantiser rounds to nearest, with no GPTQ or AWQ correction. Your \`fpRound\` returns a JavaScript double that happens to lie on the format's grid; hardware stores the bit pattern, two FP4 values to a byte. The checkpoint is a 2-layer, 64-wide model without a 70B model's outlier features, so the demo plants one by hand.
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

The round-trip error of every value is at most \`scale / 2\`; the tests check that bound and that int4, whose step is \`max|x| / 7\` against int8's \`max|x| / 127\`, has roughly \`(127/7)² ≈ 330\` times the mean squared error of int8.
`,
      predict: { question: 'You quantise 1,000 samples from N(0, 0.5) to int8 absmax. Which value decides the scale, and what does that make the error bound for the 990 values near zero?', answer: 'The single largest |x|, around 1.6 for 1,000 Gaussian samples. scale ≈ 1.6 / 127 ≈ 0.0126, so every value is within 0.0063 of its reconstruction. One outlier of 16 would make that bound ten times worse for everyone.' },
      hints: [
        'The codes are integers; `Math.round` rounds to nearest. Read `qmin` and `qmax` from `qrange(bits)` rather than hard-coding 127.',
        'scale = amax > 0 ? amax / qmax : 1. Then loop once more over the data: divide by scale, round, clamp into [qmin, qmax], store in the Int8Array.',
        '`const scale = amax > 0 ? amax / qmax : 1; for (let i = 0; i < data.length; i++) q[i] = clamp(/* x[i] in units of scale, rounded */, qmin, qmax);` and dequantize is the one-line inverse of that expression.',
      ],
    },
    {
      id: 'perchannel',
      title: 'One scale per channel',
      instructions: `
Write \`quantizePerChannel(w, bits = 8)\` and the general \`dequantize(qw)\`.

\`w\` is a raw \`[rows, cols]\` tensor. Apply the absmax rule of step 1, with \`qrange(bits)\` for whatever \`bits\` is passed (the tests and the demo also call it with 4), to each row separately and return the quantised-matrix struct from the top of the file: \`{ shape, q, scales, zeros: null, bits, groupSize: cols }\`, with \`scales[r]\` the scale of row \`r\` and \`q\` laid out exactly like \`w.data\`.

\`dequantize\` must handle the general struct, not only per-channel: a row is split into \`nGroups = cols / groupSize\` groups, and value \`[r, c]\` uses \`scales[r * nGroups + floor(c / groupSize)]\`. If \`qw.zeros\` is present, subtract \`zeros[same index]\` from the code before scaling. Per-channel is the case \`groupSize === cols\`; step 3 will produce smaller groups without changing this function.

The third test injects one row 100x larger than the others and checks that the small rows are more than 100x more accurate than under a single tensor-wide scale.
`,
      hints: [
        'Row r occupies offsets r*cols … r*cols + cols − 1. Find its amax, compute its scale, then quantise just those values.',
        'In dequantize, loop over rows, then groups g, then the groupSize values in the group; base = r*cols + g*groupSize. Read the scale (and zero) once per group.',
        '`const idx = r * nGroups + g; const s = qw.scales[idx]; const z = qw.zeros ? qw.zeros[idx] : 0; for (let k = 0; k < G; k++) out[base + k] = /* the code, zero-corrected, times s */;`',
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

Either way \`q = clamp(round(x / scale) + zero, qmin, qmax)\` (the zero point is itself an integer code, so round it), and \`scale\` falls back to 1 when the group is constant. Return the same struct as step 2 with the given \`groupSize\`.

Asymmetric codes are unsigned, \`[0, 2^bits − 1]\`, and the \`Int8Array\` store holds only \`[−128, 127]\`: at 8 bits a code of 200 would silently wrap to −56. Throw if \`symmetric\` is false and \`bits > 7\`. The fallback scale of 1 is a toy rule: it is exact for an all-zero group, but an asymmetric group that is constant at 0.3 gets \`zero = round(−0.3) = 0\` and comes back as 0. Real libraries store such a group exactly; the tests do not exercise it.

The tests check that error falls strictly as the group shrinks on a matrix with outliers, that an all-positive group in \`[2, 3]\` is more than 4x more accurate asymmetrically than symmetrically, and that asymmetric 8-bit throws. Your \`dequantize\` from step 2 should already handle both.
`,
      predict: { question: 'Each row of a [16, 256] matrix holds weights from N(0, 0.02) plus four outliers of ±0.5 (25x the typical weight) in different places. Going from groupSize 256 (one scale per row) to 16, does the symmetric int4 MSE drop by about 2x, about 4x, or more than 10x?', answer: 'About 4x (3.9x on a seeded sample). With one scale per row the step is 0.5 / 7 ≈ 0.07, so nearly every small weight rounds to 0 and the MSE is close to the weights\' own variance. With groups of 16, the 4 groups that hold an outlier (a quarter of the row) still lose nearly all their small weights; only the other 12 get a fine grid. The groups that contain an outlier set the floor, which is why production recipes pair small groups with outlier handling.' },
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

A linear layer in this lab computes \`y = x · W\` with \`W\` of shape \`[K, N]\` (\`K\` input channels, \`N\` output channels). A decode kernel wants each output channel's weights contiguous, with groups along \`K\`, so \`quantizeWeight\` transposes \`W\` to \`[N, K]\` (\`ops.transpose\`) and calls \`quantizeGroups\` on that. This is the layout GPTQ, AWQ and vLLM's Marlin kernel use. \`opts\` passes straight through, so per-channel int8 is \`quantizeWeight(w, { bits: 8, groupSize: K })\`; with no options you get \`quantizeGroups\`'s default, int4 g64, which throws unless \`K\` is divisible by 64.

\`quantizedMatmul(x, qw)\` takes \`x\` as raw \`[T, K]\` and returns raw \`[T, N]\`. For each \`t\` and each output channel \`n\`, walk the groups of row \`n\`: accumulate \`Σ x[t, k] · (q[n, k] − zero)\` over the group's \`K\` indices, multiply that partial sum by the group's scale, and add it to the output. The float weight is never materialised; one scale multiply per group is what a fused GPU kernel does in registers. Throw if \`x.shape[1] !== qw.shape[1]\`.

The tests compare your kernel against \`ops.matmul(x, transpose(dequantize(qw)))\` for a symmetric int8 weight and an asymmetric int4 one, and check relative error against fp32 (under 1% for int8 per-channel, under 10% for int4 g32).
`,
      hints: [
        'quantizeWeight is one line once you see the layout. For the matmul, index x as x.data[t*K + k] and the codes as qw.q[n*K + k]; the scale index for row n, group g is n*nGroups + g.',
        'Four nested loops: t, n, g, k (k from g*G to (g+1)*G). Keep a `part` sum inside the g loop and do `acc += scale * part` after it.',
        '`let acc = 0; for (g) { const s = qw.scales[n*nGroups+g], z = qw.zeros ? qw.zeros[n*nGroups+g] : 0; let part = 0; for (k = g*G; k < (g+1)*G; k++) part += /* x[t, k] times the zero-corrected code q[n, k] */; acc += s * part; } out[t*N+n] = acc;`',
      ],
    },
    {
      id: 'memory',
      title: 'The memory calculator',
      instructions: `
Write \`weightBytes(params, { bits, groupSize = Infinity, scaleBits = 16, zeroBits = 0 })\`, \`bitsPerParam(opts)\` and \`kvCacheBytes(modelCfg, { contextLen, batch = 1, bits = 16 })\`.

Weights cost \`params · bits / 8\` bytes for the codes plus one scale and (optionally) one zero point per group: \`(params / groupSize) · (scaleBits + zeroBits) / 8\`. A per-tensor scale (\`groupSize = Infinity\`) adds nothing measurable. \`bitsPerParam\` is the same quantity per weight, in bits: int4 g128 with fp16 scales is 4.125; int4 g32 with fp16 scales and 4-bit zeros is 4.625.

KV cache: a key and a value per layer per KV head per position, so \`2 · nLayer · nKvHeads · headDim · contextLen · batch · bits / 8\`. The exported \`LLAMA3_8B\` and \`LLAMA3_70B\` constants carry the Llama 3 configs (Meta, 2024): 32 and 80 layers, 8 KV heads each thanks to grouped-query attention, head dimension 128. Llama-3-8B at 8,192 tokens in fp16 is exactly 1 GiB; without GQA (32 KV heads) it would be 4 GiB.

The same weight formula prices the block-float formats of step 6: MXFP4 is \`{ bits: 4, groupSize: 32, scaleBits: 8 }\`, 4.25 bits per weight, and NVFP4 is \`{ bits: 4, groupSize: 16, scaleBits: 8 }\`, 4.5 bits (its single fp32 scale per tensor is negligible).
`,
      predict: { question: 'Llama-3-8B, batch 64, context 8,192, fp16 KV cache. Does the cache or the bf16 weight set take more memory?', answer: 'The cache: 64 GiB against 16 GB of weights. This is why KV-cache quantisation (fp8 in vLLM, 2-bit in KIVI) and paged allocation (module 16) matter as much as weight quantisation for throughput.' },
      hints: [
        'Everything here is counting. For weights: how many groups are there, and how many bits does each carry on top of its codes? For the cache: what is stored per token, per layer, per sequence?',
        'Weights: the codes take params · bits / 8 bytes; there are params / groupSize groups, each carrying scaleBits + zeroBits bits, and a per-tensor scale (groupSize Infinity) contributes 0. bitsPerParam is the byte count of one parameter, in bits. KV: a key and a value vector of headDim entries per KV head, per layer, per position, per sequence.',
        '`const codes = (params * bits) / 8; const overhead = groupSize === Infinity ? 0 : /* number of groups */ * /* bytes of scale + zero per group */; return codes + overhead;` bitsPerParam can call weightBytes with params = 1; kvCacheBytes is one product of the factors in hint 2, times bytes per value.',
      ],
    },
    {
      id: 'floats',
      title: 'Floating-point formats and microscaling',
      instructions: `
Write \`fpRound(x, fmt)\`, \`mxQuantize(x, { block = 32, elem = 'e2m1' })\` and \`nvfp4Quantize(x)\`.

\`fmt\` is \`{ exp, man, bias, max, saturate }\`: \`exp\` exponent bits, \`man\` mantissa bits, \`bias\` the exponent offset (default \`2^(exp−1) − 1\`), \`max\` the largest finite value (default \`(2 − 2^−man) · 2^(2^exp − 2 − bias)\`, the IEEE rule that reserves the top exponent for infinity and NaN) and \`saturate\`, whether overflow clamps to \`±max\` instead of returning \`±Infinity\`. The exported \`FORMATS\` holds bf16, fp16, fp8 E4M3 and E5M2, and FP4 E2M1. E4M3 and E2M1 have no infinities, so they set \`max\` (448 and 6) and saturate. With \`a = |x|\`:

\`\`\`
e   = floor(log2 a)                    the binade of a: 2^e ≤ a < 2^(e+1)
if 2^e > a: e = e − 1                  Math.log2 can round up to an integer just below a power of two,
if 2^(e+1) ≤ a: e = e + 1              so check the binade and correct by one
e   = max(e, 1 − bias)                 below 2^(1−bias) the format is subnormal: the spacing stops shrinking
ulp = 2^(e − man)                      the spacing of representable values near a
r   = roundHalfEven(a / ulp) · ulp
if r > max: r = saturate ? max : Infinity
return sign(x) · r                     (0 and NaN come back unchanged)
\`\`\`

A tie goes to the even multiple of \`ulp\`, the one whose last mantissa bit is 0. At 1.0 bf16's \`ulp\` is \`2^−7\`, so \`1 + 2^−8\` sits exactly halfway between 1 and \`1 + 2^−7\` and rounds to 1. \`Math.round\` sends every tie up, which is wrong here.

\`mxQuantize\` follows the OCP MX specification v1.0 (2023). For each block of \`block\` consecutive values (the last may be shorter), let \`amax\` be the largest \`|x|\`. The block's scale is the power of two \`2^(floor(log2 amax) − emax)\`, with \`floor(log2 amax)\` corrected exactly as in \`fpRound\` (a small helper serves both), where \`emax = floor(log2 FORMATS[elem].max)\` is the exponent of the element format's largest power of two (2 for E2M1, whose largest is 4). An all-zero block gets scale 1. Each element is \`fpRound(x / scale, FORMATS[elem])\`. Return \`{ elems: Float32Array, scales: Float32Array, block }\`.

\`nvfp4Quantize(x)\` uses blocks of 16 and two levels of scale. \`tensorScale = max|x| / (6 · 448)\` (1 if \`x\` is all zero). Each block's scale is \`s = fpRound(amax / 6 / tensorScale, FORMATS.e4m3)\`, and each element is \`fpRound(x / (s · tensorScale), FORMATS.e2m1)\`, or 0 if \`s\` is 0. Return \`{ elems, scales, tensorScale, block: 16 }\`. Dividing by \`6 · 448\` gives the block that holds the tensor's largest value a scale of exactly 448, the top of E4M3's range. The worked \`dequantizeBlocks\` turns either result back into floats.

The tests check ties to even in bf16, overflow to \`Infinity\` at 65520 in fp16, saturation at 448 in E4M3 and subnormals, the exact E2M1 grid, power-of-two MX scales, E4M3-representable NVFP4 scales, and that both beat int4 g128 on weights with scattered outliers.
`,
      predict: { question: 'An MXFP4 block of 32 has largest |x| = 7.5. What scale does the MX rule pick, and what happens to the 7.5?', answer: 'floor(log2 7.5) = 2, so the scale is 2^(2 − 2) = 1. The element 7.5 / 1 = 7.5 rounds to 8, which E2M1 lacks, so it saturates to 6: an error of 1.5 on the block\'s largest value. Any block whose largest |x| / scale lands in [7, 8) saturates this way, because the floor rule can leave that ratio anywhere in [4, 8) while E2M1 stops at 6. An E4M3 scale can be 7.5 / 6 = 1.25 exactly, which is one reason NVFP4 is more accurate.' },
      hints: [
        'Work with the magnitude and put the sign back at the end; return 0 and NaN unchanged. What remains is two questions: what is the spacing of the grid near |x|, and which multiple of it is nearest? In the block quantisers fpRound does all the rounding; you only choose the scales.',
        'fpRound: e = Math.floor(Math.log2(a)), corrected by one if 2^e > a or 2^(e+1) ≤ a (log2 is not always exact), then raised to at least 1 − bias; ulp = 2^(e − man). Split a / ulp into floor f and remainder d: d > 0.5 rounds up, d < 0.5 down, d = 0.5 goes to whichever of f and f + 1 is even. Then apply max and saturate. mxQuantize: per block, amax gives an exponent, and the scale is 2 to that exponent minus emax. nvfp4Quantize: one pass for the tensor amax, then per block an E4M3 scale and E2M1 elements.',
        '`const f = Math.floor(v), d = v - f; const n = d > 0.5 ? f + 1 : d < 0.5 ? f : /* the even one of f, f + 1 */; let r = n * ulp; if (r > max) r = /* clamp or overflow */;` and in mxQuantize `scales[b] = 2 ** (/* exponent of amax */ - Math.floor(Math.log2(FORMATS[elem].max)));`',
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
