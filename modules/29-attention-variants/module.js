export default {
  id: '29-attention-variants',
  title: 'Modern attention: RoPE, GQA, MLA, sliding windows',
  track: 'inference',
  minutes: 105,
  threshold: 'Every modern attention variant is a trade against the KV cache and the position signal: RoPE puts position into the dot product so it caches cleanly and can be stretched, GQA and MLA shrink what each token must store, and sliding windows bound it, each with a measurable cost in what the model can attend to.',
  goal: 'Rotary positions, grouped-query and multi-head latent attention, and a sliding-window ring cache, each proven equivalent to plain attention where it should be, with the KV-cache bytes each one saves: the demo charts the cache of every variant (and a 1-in-4 hybrid) at Llama-3-8B dims, shows RoPE scores depending only on relative offset, and measures what context-extension scaling costs and buys.',
  prereqs: ['05-attention', '06-transformer', '15-kv-cache', '16-batching'],
  recall: [
    { q: 'In module 06, what sets the hard context limit of your GPT?', options: ['The number of layers', 'The learned position table `wpe` has exactly `blockSize` rows, so there is no vector for position `blockSize`', 'The softmax overflows at long lengths'], answer: 1,
      why: 'A learned table cannot say anything about a position it never stored. RoPE replaces the table with a rotation computed from the position, so every position has an encoding; whether the model can use unseen ones is the question step 5 measures.' },
    { q: 'In module 15, why can the keys and values of earlier tokens be cached and reused?', options: ['Because the weights are frozen', 'Because causal attention lets position t depend only on positions 0…t, so nothing later changes them', 'Because keys are cheaper than queries'], answer: 1,
      why: 'Every variant here keeps that property. RoPE matters because it rotates k by its OWN position only, so a cached key stays valid; a scheme that re-encoded old keys relative to the newest token would break the cache.' },
    { q: 'Module 15 sized the KV cache per token as…', options: ['`nLayer · nHead · headDim` floats', '`2 · nLayer · nHead · headDim` floats (a key and a value per head per layer)', '`nEmbd²` floats'], answer: 1,
      why: 'That formula has three knobs a model designer can turn: the number of KV heads (GQA, MQA), what is stored per token (MLA\'s latent), and how many tokens are kept (sliding windows). This module turns each one.' },
    { q: 'In module 05, why are attention scores divided by `sqrt(dh)`?', options: ['To keep them positive', 'A dot product of dh unit-variance terms has variance dh; the scale brings it back to 1 so the softmax does not saturate', 'To make each row sum to 1'], answer: 1,
      why: 'RoPE is a rotation, so it never changes the length of q or k and the same 1/sqrt(dh) still applies. Your worked `mha` uses it unchanged.' },
    { q: 'Why is decoding at small batch sizes limited by memory bandwidth (modules 15 and 16)?', options: ['Softmax is slow on GPUs', 'Each step reads every weight and the whole KV cache from memory to do a few FLOPs per byte', 'The tokenizer runs on the CPU'], answer: 1,
      why: 'At long context the cache read rivals the weight read, so a 4x smaller cache is close to a 4x faster attention step as well as 4x more sequences per GPU.' },
  ],
  review: [
    { q: 'With RoPE, you add 100 to the position of every query and every key. The attention scores…', options: ['Grow by a factor depending on 100', 'Do not change: q_m · k_n depends only on m − n', 'Become random'], answer: 1,
      why: 'Rotating both vectors by an extra 100·θ_i in every pair leaves the angle between them unchanged. That is the relative-position property your step-1 tests check.' },
    { q: 'Llama-3-8B has 32 query heads and 8 KV heads. Which KV head does query head 13 read?', options: ['13 % 8 = 5', 'floor(13 / 4) = 3', '13'], answer: 1,
      why: 'Consecutive groups of H / Hkv = 4 query heads share a KV head: heads 12–15 read KV head 3. This is Llama\'s `repeat_kv` layout; a modulo mapping would be a different (incompatible) model.' },
    { q: 'Why does MLA cache fewer bytes per token than GQA at DeepSeek-V2 dims?', options: ['It stores keys in int4', 'It stores one 512-float latent plus a 64-float shared RoPE key per layer and rebuilds every head\'s K and V from the latent', 'It drops the value vectors'], answer: 1,
      why: '576 numbers per layer versus 2 · 8 · 128 = 2048 for 8-head GQA, about 3.6x fewer; DeepSeek report MLA equals GQA with approximately 2.25 groups while matching or beating MHA quality in their ablations.' },
    { q: 'Mistral 7B uses a 4096-token sliding window. At 32k tokens of context its KV cache is, relative to full attention with the same heads…', options: ['The same', 'About 8x smaller: only the last 4096 tokens are held', 'About 2x smaller'], answer: 1,
      why: '32,768 / 4096 = 8, the figure the Mistral 7B paper reports. The cost: a single layer cannot see further than 4096 tokens back; information travels further only by hopping through layers.' },
    { q: 'Position Interpolation extends a 4k model to 16k by dividing every RoPE frequency by 4. What does it cost?', options: ['Nothing', 'Neighbouring positions become 4x harder to tell apart, because the fastest pair now turns 0.25 rad per token instead of 1', 'The KV cache grows 4x per token'], answer: 1,
      why: 'Your scoreDrift shows PI changing short-range scores the most. NTK-aware scaling and YaRN keep the fast pairs and stretch only the slow ones, which is why they need less fine-tuning.' },
  ],
  concept: `
:::plain
Attention is the part of a language model that lets each word look back at earlier words. The original 2017 design has two costs that matter in use: it learns word order from a fixed table of positions, so it cannot read past the length that table covers, and it stores a lot of memory for every word of the conversation (the KV cache from module 15). This module builds the fixes used in today's open models: marking position by rotating numbers so that only the distance between two words matters, letting many parts of the model share one stored copy, compressing what is stored, and letting most words look back only over a recent window. Each exists to shrink that per-word memory or to let a model read longer inputs than it was trained on. These choices are a large part of why two models of similar size can need several times more or less memory for the same long conversation, which shapes how many users a server can hold and what long context costs.
:::

## What the 2017 recipe spends

Two properties of the module-05/06 design decide its serving cost. Position enters once, as a learned vector added to the input, so the model knows nothing beyond its table. And every token stores a key and a value for every head in every layer (module 15): \`2 · nLayer · nHead · headDim\` numbers per token. At Llama-2-7B dims (32 layers, 32 heads, head dimension \`dh = 128\`, bf16) that is 512 KiB per token, so 4 GiB for one 8k-token sequence. Every variant below changes one of those two things.

## RoPE: position as a rotation

Rotary position embeddings (Su et al. 2021, RoFormer) drop the position table. Split the \`dh\` channels of each query and key into \`dh/2\` pairs \`(x[2i], x[2i+1])\` and rotate pair \`i\` by the angle \`p · θ_i\`, where \`p\` is the token's position and \`θ_i = base^(−2i/dh)\` (base 10000 in RoFormer and Llama 2; Llama 3 uses 500,000). Pair 0 turns one radian per token; the last barely moves.

Rotating \`q\` by angle \`a\` and \`k\` by \`b\` changes their dot product only through \`a − b\`, so \`q_m · k_n\` depends only on the offset \`m − n\`. Each key is rotated once, by its own position, and cached unchanged.

:::predict
You shift every query and key position by 1000. How many attention scores change?
---
None. Each pair of both vectors rotates by an extra \`1000 · θ_i\`, and the angle between them is unchanged. With a learned \`wpe\` table, the same shift would change every input vector.
:::

:::deeper Going deeper: ALiBi, the other route
**ALiBi** (Press, Smith & Lewis 2021; BLOOM, MPT) takes the other route: no rotation, just a penalty \`−slope · (i − j)\` added to each score, a fixed slope per head. It extrapolates well, but most recent open models use RoPE.
:::

## GQA and MQA: fewer KV heads

Shazeer (2019) proposed **multi-query attention** (MQA) because decoding is bound by reading the cache: all query heads share one key and one value head. **Grouped-query attention** (Ainslie et al. 2023) is the middle ground: \`H\` query heads in groups of \`H / H_kv\`, each group sharing one KV head. \`H_kv = H\` is plain multi-head attention; \`H_kv = 1\` is MQA. Llama 2 70B and every Llama 3 model use 8 KV heads, which cuts Llama-3-8B's cache to \`2 · 32 · 8 · 128 · 2\` bytes, exactly 128 KiB (131,072 bytes) per token.

:::predict
Llama-3-70B has 80 layers, 64 query heads, 8 KV heads and \`dh = 128\`. How much KV cache does one bf16 token take?
---
\`2 · 80 · 8 · 128 · 2 = 327,680\` bytes, 320 KiB. The 64 query heads do not appear: queries are never cached.
:::

## MLA: cache a latent, rebuild the heads

DeepSeek-V2 (2024) introduced **multi-head latent attention**. Each token's hidden state is projected down to a latent \`c = x · W_dkv\` of \`d_c = 512\` numbers (\`4 · dh\`), and every head's key and value are rebuilt from it: \`k = c · W_uk\`, \`v = c · W_uv\`. Only \`c\` is cached. Mathematically this is multi-head attention whose key and value weights have rank at most \`d_c\`. RoPE does not survive that factorisation, so DeepSeek adds a separate 64-dimensional (\`dh/2\`) rotated key shared by all heads: 576 cached numbers per layer, about \`4.5 · dh\`. That is 3.6x fewer than 8-KV-head GQA (2048) and 57x fewer than 128-head MHA (32,768); the paper's Table 1 equates it to GQA with about 2.25 groups, with quality matching or beating MHA in its ablations. (Its headline "93.3% smaller than DeepSeek 67B" also counts KV quantisation, not MLA alone.)

## Sliding windows: bound the cache

Mistral 7B (2023) lets each token attend to only the previous \`W = 4096\` tokens. The cache becomes a **rolling buffer**: token \`p\` goes to slot \`p mod W\`, overwriting the token that left the window. The paper reports an 8x smaller cache at 32k tokens. Information can still hop further through layers (32 × 4096, about 131k tokens in principle), but no single layer looks back past the window.

**Hybrids** go further: most layers are linear-attention or state-space layers (Mamba-2, Gated DeltaNet) whose recurrent state is a fixed-size matrix however long the sequence, and only a few are full attention: 1 in 8 in Jamba, 1 in 4 in Qwen3-Next. The cache grows a quarter or an eighth as fast, plus a constant. **DeepSeek Sparse Attention** (DeepSeek-V3.2) keeps the full cache but adds a light indexer that picks the top 2048 past tokens each query attends to, cutting compute, not memory.

## Stretching RoPE past training length

Beyond the training length, slow pairs reach angles the model never saw. **Position Interpolation** (Chen et al. 2023) divides every \`θ_i\` by the stretch factor \`s\`: no angle is new, but adjacent tokens become \`s\` times harder to tell apart. **NTK-aware** scaling (a 2023 community proposal) raises the base to \`base · s^(dh/(dh−2))\`, stretching slow pairs and leaving the fastest alone. **YaRN** (Peng et al. 2023) decides per pair: pairs that finish many turns within the training length keep their frequency, pairs that finish less than one are fully interpolated, with a ramp in between. It also rescales attention temperature, which you skip. Llama 3.1 and DeepSeek-V3 ship a per-frequency scheme of this kind; Qwen2.5 documents YaRN for its long-context mode.

## What is different in production

:::deeper Going deeper: how real engines run these variants
No weights are trained here. The variants run on random weights at the lab checkpoint's dims (byte counts at Llama-3-8B dims), so you verify arithmetic and memory, not quality. Real engines fuse RoPE into kernels, never materialise the \`[T, T]\` scores, page the ring buffer (module 16), and run MLA with \`W_uk\` absorbed into the query projection (so full keys are never rebuilt) and a decoupled RoPE key. Your hybrid row assumes a \`dh × dh\` state per head; real recurrent states differ in shape and precision. \`unseenRotation\` and \`scoreDrift\` are cheap diagnostics, not the perplexity measurements the papers report.
:::
`,
  steps: [
    {
      id: 'rope',
      title: 'Rotary position embeddings',
      instructions: `
Two functions.

\`ropeFrequencies(dh, base = 10000)\`: a \`Float64Array\` of \`dh/2\` frequencies, \`θ_i = base^(−2i/dh)\` for \`i = 0 … dh/2 − 1\`. With \`dh = 8\` that is \`[1, 0.1, 0.01, 0.001]\`.

\`applyRope(x, positions, { base, freqs })\`: \`x\` is \`[..., T, dh]\` and \`positions\` has one entry per time step. For every row at time step \`t\` with \`p = positions[t]\`, rotate each pair \`(x0, x1) = (x[2i], x[2i+1])\` by \`a = p · θ_i\`:

\`\`\`
out[2i]   = x0 · cos(a) − x1 · sin(a)
out[2i+1] = x0 · sin(a) + x1 · cos(a)
\`\`\`

The loop skeleton is in the starter; you fill in \`ropeFrequencies\` and the two lines inside the pair loop. Pairs are adjacent channels, as in RoFormer and Meta's Llama code. Hugging Face's \`rotate_half\` pairs channel \`i\` with \`i + dh/2\` instead: the same idea with a different layout, so their weights and yours cannot be swapped.

The tests check the property that makes RoPE worth having: \`applyRope(q, [m]) · applyRope(k, [n])\` depends only on \`m − n\`.
`,
      predict: { question: 'With dh = 16 and base 10000, a key sits at position 3 and the query at position 7. If you move the key to position 4, which frequency pair\'s relative rotation angle changes the most, and which the least?', answer: 'Pair 0, the fastest: its angle difference moves by θ_0 = 1 radian, while the last pair (θ_7 = 10000^(−14/16) ≈ 3e−4) barely moves. How much each pair\'s score term changes also depends on the lengths of that pair in q and k, but the angle is what position controls. Fast pairs resolve nearby positions; slow pairs carry long-range position information.' },
      hints: [
        'A rotation by angle a sends (1, 0) to (cos a, sin a) and (0, 1) to (−sin a, cos a). Everything else follows by linearity.',
        'Frequencies: loop i from 0 to dh/2 − 1 and use `base ** (-2 * i / dh)`. In applyRope, compute the angle once per pair, take its cos and sin, read the two old values into locals BEFORE writing either output.',
        '`const angle = p * theta[i]; const cos = Math.cos(angle), sin = Math.sin(angle);`\n`const x0 = x.data[off + 2 * i], x1 = x.data[off + 2 * i + 1];`\n`out[off + 2 * i] = x0 * cos - x1 * sin;`\n`out[off + 2 * i + 1] = /* the other rotated component */;`',
      ],
    },
    {
      id: 'gqa',
      title: 'Grouped-query attention',
      instructions: `
\`kvHeadFor(h, nHead, nKVHead)\`: the KV head query head \`h\` reads. Consecutive groups of \`nHead / nKVHead\` query heads share one: \`h // (nHead / nKVHead)\`. Throw if \`nHead\` is not a multiple of \`nKVHead\`.

\`gqaAttention(q, k, v, { causal })\`: \`q\` is \`[H, Tq, dh]\`, \`k\` and \`v\` are \`[H_kv, Tk, dh]\`. Same arithmetic and position rules as the worked \`mha\` (query \`i\` sits at position \`Tk − Tq + i\`), except that query head \`h\` reads KV head \`kvHeadFor(h, H, H_kv)\`. Start by copying \`mha\`, then make two changes: replace its equal-head check (\`k\` and \`v\` now have \`H_kv\` heads, so keep only a check that \`H\` is a multiple of \`H_kv\`, which \`kvHeadFor\` does for you), and change how you compute the key and value offsets.

\`kvBytesPerToken({ nLayer, nKVHead, headDim, bytesPerElement = 2 })\`: \`2 · nLayer · nKVHead · headDim · bytesPerElement\`.

The tests compare \`H_kv = H\` against the module-05 attention in \`lib/attention.js\`, \`H_kv = 1\` against MHA with one KV head copied to every query head (MQA), and \`H_kv = 2\` against MHA with each KV head copied to its group.
`,
      predict: { question: 'Llama-3-8B has 32 query heads and 8 KV heads. How much smaller is its KV cache than a Llama-2-7B-style model with 32 KV heads at the same layers and head size?', answer: '4x: 128 KiB per token versus 512 KiB. The query projection is unchanged; only the K and V projections (and the cache) shrink.' },
      hints: [
        'In mha the key offset is `(h * Tk + j) * dh`. Which part of that names the head, and which head should it name now?',
        'Compute `g = kvHeadFor(h, H, Hkv)` once per query head, outside the time loops. Use `g` for the key and value offsets; keep `h` for the query and output offsets. Tk now comes from `k.shape[1]` and Hkv from `k.shape[0]`.',
        'Inside the head loop: `const g = kvHeadFor(h, H, Hkv);` … `const kOff = /* which head, which key, times dh */;` … `const vOff = (g * Tk + j) * dh;`. kvHeadFor itself: check divisibility, then `Math.floor(h / (nHead / nKVHead))`.',
      ],
    },
    {
      id: 'mla',
      title: 'Multi-head latent attention',
      instructions: `
The weights come from the worked \`initMLA\`: \`wq [C, H·dh]\`, \`wdkv [C, d_c]\`, \`wuk [d_c, H·dh]\`, \`wuv [d_c, H·dh]\`. Here \`C\` is the model width and \`d_c\` the latent size.

- \`mlaLatent(x, w)\`: \`c = x · wdkv\`, \`[T, C] → [T, d_c]\`.
- \`mlaExpand(c, w, nHead)\`: \`{ k: splitHeads(c · wuk), v: splitHeads(c · wuv) }\`, each \`[H, T, dh]\`.
- \`mlaAttention(x, w, { nHead })\`: queries \`splitHeads(x · wq)\`, keys and values from the expanded latent of \`x\`, then causal \`mha\`.
- \`mlaDecodeStep(xRow, cache, w, { nHead })\`: \`xRow\` is one token \`[1, C]\`. Append its latent to \`cache.c\` (use \`ops.concat\` along axis 0), rebuild K and V from all of \`cache.c\`, and attend this token's queries over them. Returns \`[H, 1, dh]\`.
- \`mlaBytesPerToken({ nLayer, dLatent, dRope = 0, bytesPerElement = 2 })\`: \`nLayer · (dLatent + dRope) · bytesPerElement\`. There is no factor 2: one latent serves both K and V.

The cache holds \`d_c\` floats per token instead of \`2 · H · dh\`. The tests check that MLA is exactly MHA with the low-rank key weight \`wdkv · wuk\`, and that decoding from the latent cache reproduces the full pass token by token.
`,
      hints: [
        'Every piece is a matmul followed by `splitHeads`; `lib/ops.js` has `matmul`, `concat` and `slice`. The worked `mha` does the attention.',
        'mlaAttention: q from x·wq, c from mlaLatent, {k, v} from mlaExpand(c), then `mha(q, k, v, { causal: true })`. mlaDecodeStep: grow `cache.c` first, then the same three lines with xRow as the only query; mha places a single query at the last position.',
        'mlaDecodeStep in outline: `cache.c = ops.concat(cache.c, /* this token\'s latent, [1, d_c] */, 0);`, then queries from xRow alone, K and V from `/* every row of cache.c */`, then the worked mha. mlaExpand: `splitHeads(ops.matmul(c, /* which weight? */), nHead)`, once for K and once for V. mlaBytesPerToken: one latent plus one RoPE key per layer, times the element size.',
      ],
    },
    {
      id: 'window',
      title: 'Sliding windows and the ring cache',
      instructions: `
\`slidingWindowAttention(q, k, v, window)\`: causal attention (shapes and positions as in \`mha\`: with \`Tq < Tk\` the queries are the last \`Tq\` positions, \`p = Tk − Tq + i\`) where the query at position \`p\` sees only keys \`p − window + 1 … p\`: \`window\` keys including itself. With \`window = 1\` each token sees only itself.

\`RingKVCache\`: the constructor is written. \`this.k\` and \`this.v\` are \`Float32Array\`s laid out \`[H, window, dh]\` and allocated once. Implement:

- \`length\`: tokens currently held, \`min(count, window)\`.
- \`append(k, v)\`: \`k\`, \`v\` are \`[H, 1, dh]\`. Write head \`h\` into slot \`count % window\` of head \`h\`'s block, then increment \`count\`.
- \`keys()\`, \`values()\`: the held tokens as \`[H, length, dh]\`, **oldest first**. The oldest held token has position \`count − length\`.
- \`bytes(bytesPerElement = 4)\`: what the two buffers occupy, \`2 · H · window · dh · bytesPerElement\`: what is allocated, not what is filled, so the same whether 0, 1 or 1,000,000 tokens have passed through.

The tests stream tokens through the ring, attend each one with \`mha\` over \`keys()\` and \`values()\`, and compare against your \`slidingWindowAttention\`.
`,
      predict: { question: 'Window 4, and you have appended tokens 0…9. Which slots hold which tokens, and in what order must keys() return them?', answer: 'Slots 0,1,2,3 hold tokens 8,9,6,7 (token p is in slot p % 4). keys() returns 6,7,8,9: start at position count − length = 6 and read slot (6 + t) % 4 for t = 0…3.' },
      hints: [
        'Only one number changes in the attention loops: the first key a query may see. The ring is the same idea in memory: position p always lives at p % W.',
        'In the attention, `first = Math.max(0, pos - window + 1)` and every `j` loop runs from `first` to `pos`. In the ring, `append` computes `slot = count % W` and copies dh floats per head to `(h * W + slot) * dh`; the gather reads slot `(start + t) % W` for t in 0…length−1 with `start = count − length`.',
        'Gather: `for (h…) for (t…) { const src = (h * W + (/* slot of the t-th oldest token */)) * dh; out.set(buf.subarray(src, src + dh), (h * n + t) * dh); }` with `const n = this.length, start = this.count - n;` and `out = new Float32Array(H * n * dh)` set up first, then `return { shape: [H, n, dh], data: out }`.',
      ],
    },
    {
      id: 'extend',
      title: 'Stretching RoPE past the training length',
      instructions: `
\`scaledRopeFrequencies(dh, { base, factor, method, trainLen, alpha = 1, beta = 32 })\` returns \`dh/2\` frequencies. Let \`θ = ropeFrequencies(dh, base)\` and \`s = factor\`:

| method | frequencies |
|---|---|
| \`'none'\` | \`θ_i\` |
| \`'pi'\` | \`θ_i / s\` (Position Interpolation) |
| \`'ntk'\` | \`ropeFrequencies(dh, base · s^(dh/(dh−2)))\` |
| \`'yarn'\` | \`(1 − γ_i) · θ_i / s + γ_i · θ_i\` |

For YaRN, \`turns_i = trainLen · θ_i / (2π)\` (the paper's \`r = L / λ\`, with wavelength \`λ_i = 2π / θ_i\`) is how many full circles pair \`i\` makes over the training length, and \`γ_i = clamp((turns_i − alpha) / (beta − alpha), 0, 1)\`. Throw on an unknown method.

\`unseenRotation(freqs, trainFreqs, trainLen, len)\`: for each pair, training reached the angle \`seen = min(trainLen · trainFreqs[i], 2π)\` and length \`len\` reaches \`now = min(len · freqs[i], 2π)\`. Return the mean over pairs of \`max(0, now − seen) / 2π\`. Zero means no pair sees an angle it was not trained on; a length shorter than training also gives zero, never a negative number.

\`scoreDrift\` is done for you in the starter: it measures the other side of the trade, how much rescaling changes scores at offsets the model was trained on.
`,
      hints: [
        'Only YaRN needs a per-pair decision. For the others, map over θ or build new frequencies from a new base.',
        'YaRN: for each θ_i compute turns, then γ clamped to [0, 1], then blend the interpolated `θ/s` and the original `θ`. Pairs with many turns (fast) get γ = 1 and stay; pairs with less than one turn get γ = 0 and are divided by s.',
        '`return theta.map((t) => { const turns = trainLen * t / (2 * Math.PI); const gamma = /* clamp (turns − alpha)/(beta − alpha) */; return (1 - gamma) * (t / factor) + gamma * t; });`. unseenRotation: `total += Math.max(0, now - seen) / (2 * Math.PI)`, then divide by the number of pairs.',
      ],
    },
    {
      id: 'budget',
      title: 'The cache budget',
      instructions: `
\`cacheBudget(config, contextLen)\`: one row per variant, in the order MHA, GQA, MQA, MLA, SWA, HYBRID, each \`{ variant, bytesPerToken, tokensHeld, fixedBytes, totalBytes }\` with \`totalBytes = bytesPerToken · tokensHeld + fixedBytes\`. \`config\` is \`{ nLayer, nHead, nKVHead, headDim, dLatent, dRope, window, fullAttnEvery = 4, bytesPerElement }\`.

| variant | bytesPerToken | tokensHeld |
|---|---|---|
| MHA | \`kvBytesPerToken\` with \`nKVHead = nHead\` | \`contextLen\` |
| GQA | \`kvBytesPerToken\` with \`nKVHead\` | \`contextLen\` |
| MQA | \`kvBytesPerToken\` with one KV head | \`contextLen\` |
| MLA | \`mlaBytesPerToken\` | \`contextLen\` |
| SWA | as GQA (Mistral 7B uses both) | \`min(contextLen, window)\` |
| HYBRID | \`kvBytesPerToken\` with \`nKVHead\`, but only \`nFull = Math.floor(nLayer / fullAttnEvery)\` layers | \`contextLen\` |

\`fixedBytes\` is 0 for every row except HYBRID. There the other \`nLayer − nFull\` layers are recurrent (Qwen3-Next style: 3 Gated DeltaNet layers per full-attention layer), and each keeps one \`headDim × headDim\` state matrix per head whatever the context length: \`fixedBytes = (nLayer − nFull) · nHead · headDim² · bytesPerElement\`. At Llama-3-8B dims that is 24 MiB, the same at 1k tokens as at 1M.

Reuse your step-2 and step-3 functions. The demo turns this table into a bar chart at Llama-3-8B dims.
`,
      hints: [
        'Four of the six rows differ only in bytes per token; one differs in how many tokens it keeps; one pays a constant on top of a smaller per-token cost.',
        'Write a small helper `row(variant, bytesPerToken, tokensHeld, fixedBytes = 0)` that fills in totalBytes, and a helper `kv(heads, layers = nLayer)` that calls kvBytesPerToken with the config\'s headDim and bytesPerElement. The hybrid row pays kv for the nFull attention layers per token, plus one state per head in each recurrent layer.',
        '`const nFull = Math.floor(nLayer / fullAttnEvery);` then `return [row(\'MHA\', kv(nHead), contextLen), row(\'GQA\', /* … */), /* MQA, MLA */, row(\'SWA\', kv(nKVHead), /* tokens a window keeps */), row(\'HYBRID\', kv(nKVHead, nFull), contextLen, /* recurrent layers × heads × dh² × bytes */)];`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why RoPE makes the attention score depend only on the offset m − n, and why that is exactly what lets a cached key stay valid forever.',
    'GQA, MLA and sliding windows all shrink the KV cache. For each, name what the model gives up to get the saving, and which one you would pick for a 1M-token context and why.',
    'Position Interpolation, NTK-aware scaling and YaRN trade two costs your demo measured. Describe the trade in your own words using the numbers you saw.',
  ],
  stretch: [
    'Implement MLA\'s weight absorption: fold `W_uk` into the query so scores are computed directly against the cached latents (`q · (c · W_uk)ᵀ = (q · W_ukᵀ) · cᵀ`), and add DeepSeek-V2\'s decoupled 64-dimensional RoPE key. Check the result equals your `mlaDecodeStep`. See the DeepSeek-V2 paper and the MLA kernels in vLLM and SGLang.',
    'Add ALiBi (Press et al. 2021, used in BLOOM and MPT): add `−slope_h · (i − j)` to each score, with slopes `2^(−8h/H)`, and compare how its scores behave beyond the training length against your unscaled RoPE.',
    'Combine the ring buffer with module 16\'s paged allocator: a sliding-window sequence can free a whole block once every token in it has left the window, as vLLM does for Mistral and Gemma 2\'s local layers.',
    'Implement Llama 3.1\'s `rope_type: "llama3"` frequency scaling (factor 8, low/high frequency factors 1 and 4, original context 8192, from the Hugging Face transformers source) and compare its frequencies with your YaRN ramp.',
  ],
  timeouts: { tests: 20000, demo: 120000 },
};
