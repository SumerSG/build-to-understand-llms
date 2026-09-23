export default {
  id: '06-transformer',
  title: 'The GPT architecture',
  track: 'transformer',
  minutes: 120,
  threshold: 'A GPT is a stack of identical residual blocks that alternately mix information across positions (attention) and transform each position on its own (MLP); everything else is plumbing.',
  goal: 'A complete GPT (embeddings, pre-LN blocks, MLP, tied LM head) with correct shapes and a parameter-count formula that matches the built model exactly, whose forward pass on 64 random tokens the demo charts component by component.',
  prereqs: ['02-autograd', '05-attention'],
  recall: [
    { q: 'In module 01, LayerNorm normalises…', options: ['Each column across the batch', 'Each row (one position\'s feature vector) on its own', 'The whole tensor at once'], answer: 1,
      why: 'Every position is normalised independently, which is why a transformer block can be applied to a single sequence of any length without batch statistics.' },
    { q: 'In module 02, a tensor that is used twice in the graph ends up with a gradient equal to…', options: ['The gradient from its last use', 'The sum of the gradients from both uses', 'The average of the two'], answer: 1,
      why: 'The chain rule sums over paths. The tied token table in this module is used at the input and at the output, and its gradient is the sum of both contributions.' },
    { q: 'In module 05, `MultiHeadAttention({ nEmbd: C, nHead })` holds which trainable layers?', options: ['One Linear per head', 'A qkv Linear(C, 3C) and a proj Linear(C, C)', 'A single Linear(C, C)'], answer: 1,
      why: 'That is 3C² + C² = 4C² weights per attention layer, the first term of this module\'s parameter budget.' },
    { q: 'In module 04, a model that spreads its probability uniformly over V tokens has a cross-entropy loss of…', options: ['`ln(V)`', '`V`', '`1 / V`'], answer: 0,
      why: 'Negative log of 1/V. A freshly initialised GPT should sit near ln(V); if it starts far above, the initial weights are too large.' },
  ],
  review: [
    { q: 'How many weights (ignoring biases) does one transformer block hold, in terms of the channel width C?', options: ['4C²', '8C²', '12C²'], answer: 2,
      why: 'Attention: qkv 3C² + proj C² = 4C². MLP: 4C² up + 4C² down = 8C². Together 12C², so a model is roughly 12·L·C² plus the V·C token table.' },
    { q: 'In a pre-LN block whose attention and MLP branches output zero, the block returns…', options: ['`x` unchanged', '`LayerNorm(x)`', 'Zero'], answer: 0,
      why: 'The LayerNorms sit inside the branches, so the residual path from input to output is the identity. In post-LN the normalisation wraps the sum and the identity path is gone; that is why deep post-LN stacks need careful warmup.' },
    { q: 'Weight tying means the logits are computed as…', options: ['`x · W_head` with a fresh [C, V] matrix', '`x · wteᵀ`, reusing the token embedding table', '`softmax(x)`'], answer: 1,
      why: 'The same [V, C] table reads tokens in and scores tokens out, saving V·C parameters (38.6M in GPT-2 small, 31% of the model).' },
    { q: 'Which part of a GPT lets position 7 use information from position 3?', options: ['The MLP', 'Attention', 'LayerNorm'], answer: 1,
      why: 'Only attention mixes across positions. The MLP and both LayerNorms act on each position\'s vector alone.' },
    { q: 'The FLOPs to process one token with a model of P parameters and a short context are approximately…', options: ['`P`', '`2·P`', '`P²`'], answer: 1,
      why: 'Every weight takes part in one multiply-add. The attention products add 4·L·T·C; next to the 24·L·C² of the block matmuls that is T/(6C), so it is about 17% at T = C and equals the matmul cost at T = 6C.' },
  ],
  concept: `
## The residual stream is the whole design

Take the input sequence, look up one vector per token, and call the resulting \`[B, T, C]\` tensor **x** (batch, time, channels). A GPT never replaces x. Each of its L blocks reads x, computes an update, and *adds* it back:

\`\`\`
x = x + attn(ln1(x))    // mix information ACROSS positions
x = x + mlp(ln2(x))     // transform EACH position on its own
\`\`\`

Everything a GPT does is one of these two moves. Attention (module 05) is the only place where position 7 can read from position 3; the MLP is a per-position function, applied identically to every token's vector. Because every block only *adds* to x, there is an unbroken identity path from the embeddings to the logits: the **residual stream**. Gradients travel down it without passing through any weight, which is why a 96-layer stack trains at all.

The LayerNorms sit *inside* the branches (**pre-LN**, as in GPT-2 and nearly every large model since). The original transformer put them on the sum, \`x = ln(x + attn(x))\` (**post-LN**), which breaks the identity path: at initialisation the output is a normalised mix, not x, and deep post-LN stacks need learning-rate warmup to avoid diverging (Xiong et al. 2020 measure this). Your tests check the pre-LN property directly: zero both branches and the block must return x exactly.

## Plumbing: embeddings, final norm, tied head

Below the blocks, two lookup tables: \`wte\` (\`[V, C]\`, one row per token) and \`wpe\` (\`[T_max, C]\`, one row per position). The input is \`wte[ids] + wpe[0..T-1]\`. That table is the only reason a GPT has a context limit: there is no row for position \`blockSize\`. Above the blocks, a final LayerNorm, then the head. GPT-2 **ties** the head to the token table: \`logits = x · wteᵀ\`, so the vector that represents a token on the way in is the vector it is scored against on the way out. This costs zero new parameters, and the table's gradient is the sum of its two uses (module 02).

:::predict
GPT-2 small has V = 50,257, T = 1,024, L = 12, C = 768. Roughly what fraction of its 124M parameters is the token embedding table?
---
About 31%: 50,257 × 768 = 38,597,376. Without tying, the head would add the same again. This is why "12·L·C²" alone (84.9M here) undercounts a small model by a third.
:::

## The parameter budget

Count what you build. A \`Linear(nIn, nOut)\` holds \`nIn·nOut + nOut\` scalars. Per block:

| component | weights | biases + norms |
|---|---|---|
| attention: qkv \`C→3C\`, proj \`C→C\` | \`4C²\` | \`4C\` |
| MLP: fc \`C→4C\`, proj \`4C→C\` | \`8C²\` | \`5C\` |
| ln1, ln2 | | \`4C\` |

So \`12C² + 13C\` per block, and the model is \`V·C + T·C + L·(12C² + 13C) + 2C\`. For GPT-2 small that is exactly **124,439,808**, the published figure: token table 38.6M (31%), positions 0.8M, attention 28.3M (23%), MLP 56.7M (46%), LayerNorm 38 thousand. Two thirds of every block is the MLP.

Now Llama-3-8B (V = 128,256, C = 4,096, L = 32): the embedding is 525.3M and, because the head is *untied*, the output matrix is another 525.3M. Attention uses grouped-query attention with 8 key/value heads for 32 query heads, so q and o are \`C×C\` (16.8M each) but k and v are \`C×1024\` (4.2M each): 41.9M per layer. The MLP is SwiGLU with three matrices of \`4096 × 14,336\`: 176.2M per layer. Per layer 218.1M; times 32 is 6.98B; plus the two tables gives **8.03B**. The shape of the budget is the same as GPT-2's: the MLP dominates, attention is a fifth, and the tables matter only when V·C is comparable to L·C².

:::predict
You double the context length of a model from 1,024 to 2,048 tokens. Which components gain parameters?
---
Only \`wpe\` (\`T·C\`): 0.8M more for GPT-2 small. Attention's weights are \`[C, 3C]\` and \`[C, C]\`, independent of T. What grows with T is the *compute*: the score matrix costs \`4·L·T·C\` FLOPs per token on top of the \`2·params\` for the matmuls, and the KV cache (module 15) grows with T too.
:::

## From parameters to FLOPs

Every weight matrix is used in one multiply-add per weight per token, so a forward pass costs about \`2·params\` FLOPs per token (slightly high: the \`wpe\` lookup and the LayerNorm gains are not multiply-adds), plus \`4·L·T·C\` for \`q·kᵀ\` and \`weights·v\`. For the lab model (V = 256, T = 64, L = 2, C = 64) that is 2 × 120,576 + 32,768 ≈ 274 thousand FLOPs per token; for Llama-3-8B at a 4,096-token context it is roughly 16 GFLOPs + 2 GFLOPs. Module 08 builds the 6·N·D training rule on top of this.

## Where the toy differs from production

Your model is GPT-2's architecture at 1/1000 scale; the differences are mostly what Llama-style models changed later: **RMSNorm** instead of LayerNorm (no mean subtraction, no beta), **SwiGLU** instead of GELU with a 4× hidden (three matrices; Llama 1 and Llama 2 7B/13B use a hidden width of about 8C/3 ≈ 2.7C so the MLP still costs about 8C², and Llama 3 8B widens it to 3.5C = 14,336), **rotary position embeddings** applied inside attention instead of a learned \`wpe\` table, **no biases**, grouped-query attention, and an untied head. None of these change the threshold idea: a residual stream, and blocks that alternately mix across positions and transform within them. Production code also fuses LayerNorm, GELU and the residual add into single kernels, and FlashAttention-style kernels never materialise the \`[T, T]\` score matrix; your version runs plain JavaScript loops, one op at a time.
`,
  steps: [
    {
      id: 'layers',
      title: 'Linear and Embedding',
      instructions: `
Two layers, following the conventions of the worked \`LayerNorm\` above.

\`new Linear(nIn, nOut, { bias = true, next, std = 0.02 })\`: \`weight\` is a trainable \`[nIn, nOut]\` Tensor drawn from \`Tensor.randn(shape, next, std)\`; \`bias\` is a trainable zero \`[nOut]\` Tensor, or \`null\` when \`bias\` is false. \`forward(x)\` returns \`x · W + b\` for any \`x [..., nIn]\` (a 2-D weight is shared over every leading dimension by \`Tensor.matmul\`). \`parameters()\` returns \`[weight, bias]\` or \`[weight]\`.

\`new Embedding(n, d, { next, std = 0.02 })\`: a trainable \`[n, d]\` table, same init. \`forward(ids)\` picks rows with \`Tensor.embed\` (ids may be nested, so \`[[B×T]]\` gives \`[B, T, d]\`). \`parameters()\` returns \`[weight]\`.

The std of 0.02 is GPT-2's choice (nanoGPT keeps it). With std 1 the logits at initialisation are in the hundreds and the first loss is far above \`ln(V)\`; the tests check both the std and that gradients reach the weight, bias and table, which they cannot if you use raw \`ops\` instead of Tensor methods.
`,
      hints: [
        'Look at how LayerNorm above creates its parameters: `Tensor.param(...)` around a raw-shaped Tensor. Linear needs one Gaussian leaf (`Tensor.randn([nIn, nOut], next, std)`) and one zero leaf (`Tensor.zeros([nOut])`).',
        'Constructor: weight = param(randn), bias = `bias ? param(zeros([nOut])) : null`. forward: one `matmul`, then `.add(this.bias)` only when the bias exists; Tensor.add broadcasts `[nOut]` over the leading dims. Embedding.forward is a single call: `this.weight.embed(ids)`.',
        '`this.weight = Tensor.param(Tensor.randn([nIn, nOut], next, std));`\n`this.bias = bias ? Tensor.param(Tensor.zeros([nOut])) : null;`\n`forward(x) { const y = /* x times the weight */; return this.bias === null ? y : y.add(this.bias); }`\n`parameters() { return this.bias === null ? [this.weight] : [this.weight, this.bias]; }`',
      ],
    },
    {
      id: 'embed',
      title: 'Token + position embeddings',
      instructions: `
Implement \`embedInputs(wte, wpe, ids)\`: \`ids\` is \`number[][]\` of shape B×T; return \`wte[ids] + wpe[0..T-1]\` as a Tensor \`[B, T, C]\`.

Attention is permutation-invariant: without \`wpe\`, "dog bites man" and "man bites dog" would produce the same set of vectors. The position table gives each slot its own learned offset. It has exactly \`blockSize\` rows (\`wpe.n\`), so throw an \`Error\` when \`T > wpe.n\`; this is the context limit of every learned-position GPT. The position lookup is \`[T, C]\` and broadcasts over the batch when added to \`[B, T, C]\`.
`,
      predict: { question: 'Two batch rows contain the same token id at position 0. Before any block runs, are their position-0 vectors equal?', answer: 'Yes: both are wte[id] + wpe[0]. Only the blocks (through attention over the other tokens) can make them differ.' },
      hints: [
        'Two lookups and one add. The position ids are not in `ids`: they are 0, 1, …, T−1 for every batch row, with T = `ids[0].length`.',
        'Build `positions = [0..T-1]`, check `T <= wpe.n` first, then `wte.forward(ids)` is `[B,T,C]` and `wpe.forward(positions)` is `[T,C]`; `Tensor.add` broadcasts the second over B.',
        '`const time = ids[0].length;`\n`if (time > wpe.n) throw new Error(\`sequence of ${time} exceeds blockSize ${wpe.n}\`);`\n`const positions = []; for (let t = 0; t < time; t++) positions.push(t);`\n`return /* token lookup */.add(/* position lookup */);`',
      ],
    },
    {
      id: 'block',
      title: 'The MLP and the pre-LN block',
      instructions: `
\`new MLP(nEmbd, { next })\`: \`fc = Linear(C, 4C)\`, \`proj = Linear(4C, C)\`; \`forward(x)\` is \`proj(gelu(fc(x)))\` (Tensor has \`.gelu()\`); \`parameters()\` is fc's then proj's. GELU is a smooth ReLU, \`x·Φ(x)\` with Φ the standard normal CDF (Hendrycks & Gimpel 2016), and \`.gelu()\` computes the tanh approximation of it that GPT-2 uses; the MLP needs some nonlinearity, or its two Linears would collapse into a single Linear. The 4× widening is GPT-2's ratio, and it is where two thirds of every block's parameters live.

\`new Block({ nEmbd, nHead }, { next })\`: \`ln1\`, \`attn = new MultiHeadAttention({ nEmbd, nHead, next })\` from \`lib/attention.js\`, \`ln2\`, \`mlp\`, constructed **in that order** so the seeded initialisation is reproducible (the tests compare your initial weights with \`lib/gpt.js\`'s Block built from the same seed). \`forward(x)\` is the two lines from the file header, \`x [B, T, C] → [B, T, C]\`; the MLP branch reads the stream *after* the attention update. \`parameters()\` returns ln1, attn, ln2, mlp in order (this is the order \`lib/gpt.js\` and the checkpoints use).

Why pre-LN: the tests zero both output projections and require the block to return \`x\` unchanged. They also give ln1 and ln2 different gains, so the MLP branch must use ln2, not ln1 again. Post-LN would return a normalised \`x\`, and a missing residual would return zero. That clean identity path is what lets gradients reach the first block of a deep stack.
`,
      hints: [
        'Each of the two header lines is one `Tensor.add`. The LayerNorm goes on the *input* of each branch, never on the sum, and the second branch reads the updated stream.',
        'MLP.forward is three calls chained inside out: fc, then GELU on its output, then proj. Block.forward: `h = x + attn(ln1(x))`, then `return h + mlp(ln2(h))`. parameters(): spread the four sub-layers with `...` in the order ln1, attn, ln2, mlp.',
        '`forward(x) {`\n`  const h = /* x plus the attention branch applied to ln1(x) */;`\n`  return h.add(this.mlp.forward(this.ln2.forward(h)));`\n`}`\n`parameters() { return [...this.ln1.parameters(), ...this.attn.parameters(), ...this.ln2.parameters(), ...this.mlp.parameters()]; }`',
      ],
    },
    {
      id: 'gpt',
      title: 'Stack, final LayerNorm, tied head',
      instructions: `
\`new GPT({ vocabSize, blockSize, nLayer, nHead, nEmbd, seed = 0 })\`: with the single \`next = rng(seed)\` already in the constructor, build \`wte = Embedding(vocabSize, nEmbd)\`, \`wpe = Embedding(blockSize, nEmbd)\`, then \`nLayer\` Blocks into \`this.blocks\`, in that order. \`lnF\` is already there.

\`forward(ids)\`: \`embedInputs\` → every block in turn → \`lnF\` → \`logits = x · wteᵀ\`, a Tensor \`[B, T, V]\`. Use \`this.wte.weight.transpose()\`; do **not** create a separate head matrix. \`parameters()\` returns wte, wpe, each block's parameters, lnF, in that order. \`numParams()\` sums \`p.size\`.

The tests copy the weights of \`lib/gpt.js\` into your model by \`parameters()\` order and require identical logits, check that changing token 3 leaves the logits at positions 0–2 untouched, and run \`crossEntropy(...).backward()\` to confirm every parameter is connected to the loss. Module 07 will train exactly this object.
`,
      predict: { question: 'Before any training, with std-0.02 weights and V = 64, roughly what cross-entropy loss should a forward pass give?', answer: 'Close to ln(64) ≈ 4.16. The logits are small, so the softmax is nearly uniform. A loss of 20 or more at init means the initial weights are too large (std 1 instead of 0.02).' },
      hints: [
        'Construct wte, wpe, then the blocks, all from the one `next`; the forward is embedInputs → blocks → lnF → head, and the head adds no parameter of its own.',
        'The head is one matmul of the normalised stream `[B,T,C]` with the token table turned on its side, `[C,V]`; no new Tensor.param. parameters(): start with wte and wpe, push each block\'s `parameters()`, finish with lnF. numParams: loop over parameters() and add `p.size`.',
        '`forward(ids) {`\n`  let x = embedInputs(this.wte, this.wpe, ids);`\n`  for (const block of this.blocks) x = block.forward(x);`\n`  x = this.lnF.forward(x);`\n`  return /* x scored against the transposed token table */;`\n`}`',
      ],
    },
    {
      id: 'count',
      title: 'The parameter and compute budget in closed form',
      instructions: `
Three pure functions of a config object \`{ vocabSize, blockSize, nLayer, nHead, nEmbd }\` (call them V, T, L, C):

- \`paramBreakdown(config)\` → \`{ tokenEmbedding, positionEmbedding, attention, mlp, layerNorm, total }\`, each an exact count for the model you just built, biases and LayerNorm gains included. Use the table in the concept section.
- \`countParams(config)\` → the total, \`V·C + T·C + L·(12C² + 13C) + 2C\`.
- \`flopsPerToken(config, contextLen)\` → \`2·countParams + 4·L·contextLen·C\`.

The tests compare \`countParams\` with \`numParams()\` of the reference model for three configs, and require GPT-2 small to come out at exactly 124,439,808. Being exact matters: an approximation that is 30% off on a small model is what leads people to believe "parameters ≈ 12LC²" applies to models where the vocabulary table dominates.
`,
      hints: [
        'Count what you built: two tables; per block two Linears in attention (C→3C and C→C, with biases), two in the MLP (C→4C and 4C→C, with biases), two LayerNorms; plus the final LayerNorm. A Linear(nIn, nOut) holds nIn·nOut + nOut scalars and a LayerNorm(C) holds 2C.',
        'attention per block: 3C² + 3C + C² + C = 4C² + 4C. mlp per block: 4C² + 4C + 4C² + C = 8C² + 5C. layerNorm: 4C per block plus 2C for lnF. Multiply the per-block terms by L, add V·C and T·C. FLOPs: two per parameter plus 4·L·T·C.',
        '`const tokenEmbedding = vocabSize * C;`\n`const positionEmbedding = blockSize * C;`\n`const attention = nLayer * (4 * C * C + 4 * C);`\n`const mlp = /* L times the fc and proj weights and biases */;`\n`const layerNorm = nLayer * 4 * C + 2 * C;`\n`return { tokenEmbedding, positionEmbedding, attention, mlp, layerNorm, total: /* their sum */ };`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why "residual stream" is a better mental model of a GPT than "a sequence of layers", and what each of the two block updates is allowed to see.',
    'Your countParams says two thirds of every block is the MLP, yet attention gets most of the attention in explanations. Using the FLOPs formula, say when the attention term stops being negligible, and what that implies for long contexts.',
    'The head is tied to the token table. Write down one argument for tying (GPT-2) and one for untying (Llama 3), and say which you would choose for a 256-token vocabulary and why.',
  ],
  stretch: [
    'Replace the learned `wpe` with rotary position embeddings applied to q and k inside attention (Su et al. 2021), as GPT-J, GPT-NeoX, Llama and most open models since do; check that `countParams` drops by exactly T·C. Module 29 later builds RoPE properly.',
    'Swap `LayerNorm` for RMSNorm and the GELU MLP for SwiGLU with a hidden width of `round(8C/3)`, the Llama 1 and Llama 2 7B/13B block (Llama 2 70B and Llama 3 8B use 3.5C). Recompute the per-block budget and confirm it against `numParams()`.',
    'Add `generate(ids, { maxNewTokens, temperature, next })` that recomputes the full prefix at each step inside `noGrad`, as `lib/gpt.js` does; time it against sequence length, then read module 15 to see what the KV cache in vLLM and llama.cpp removes.',
    'Scale the output projections by `1/sqrt(2L)` at initialisation, as GPT-2 does for the residual branches, and measure the per-block growth of the residual stream norm the demo plots, before and after.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
