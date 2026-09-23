export default {
  id: '05-attention',
  title: 'Attention from scratch',
  track: 'transformer',
  minutes: 90,
  threshold: 'Attention lets each position build its representation as a weighted average of other positions\' values, with weights computed from query–key similarity; the causal mask is what makes it a language model.',
  goal: 'A causal multi-head self-attention layer that matches reference outputs and renders its own attention heatmap.',
  prereqs: ['01-tensors', '02-autograd', '04-bigram'],
  recall: [
    { q: 'In module 01, what does `reshape` do to a tensor\'s data?', options: ['Copies it into the new layout', 'Nothing; only the shape changes', 'Transposes it'], answer: 1,
      why: 'Reshape is free because row-major data does not move. Splitting channels into heads is a reshape followed by a permute, and only the permute copies.' },
    { q: 'Over which axis does `softmax` normalise in lib/ops.js and lib/tensor.js?', options: ['The first axis', 'The last axis', 'Every element together'], answer: 1,
      why: 'Attention scores are [.., T, T] with keys on the last axis, so a row-wise softmax gives each query a distribution over keys.' },
    { q: 'In module 02, a tensor that is used twice in the graph receives its gradient how?', options: ['Only from the last use', 'Summed from both uses (+=)', 'Averaged over the uses'], answer: 1,
      why: 'One input x feeds q, k and v through a single projection, so x.grad is the sum of three contributions. Accumulation is what makes that correct.' },
    { q: 'Why does `gradCheck` in lib/tensor.js use eps ≈ 1e-3 and a tolerance of 1e-2 rather than 1e-7 and 1e-6?', options: ['To run faster', 'Because the data is float32, whose rounding noise swamps a smaller step', 'Because attention is not differentiable'], answer: 1,
      why: 'Float32 has about 7 significant digits; a central difference with eps = 1e-7 would be pure rounding noise. You use gradCheck on the whole layer in step 5.' },
    { q: 'How much context did the bigram model of module 04 use to predict the next token?', options: ['The whole sequence', 'Exactly one previous token', 'A fixed window of 8 tokens'], answer: 1,
      why: 'A bigram sees one token. Attention is the mechanism that lets position t read any of the t tokens before it, with weights it chooses per input.' },
  ],
  review: [
    { q: 'Why are the scores divided by `sqrt(dh)`?', options: ['To keep them positive', 'Because a dot product of dh unit-variance terms has variance dh; the scale returns it to 1 so the softmax does not saturate', 'To make the weights sum to 1'], answer: 1,
      why: 'Var(Σ q_c·k_c) = dh for independent unit-variance entries. With dh = 64 the unscaled scores have std 8, the softmax is nearly one-hot, and its gradient is nearly zero.' },
    { q: 'Where does the causal mask go, and with what value?', options: ['After the softmax, multiplying by 0', 'Before the softmax, setting future scores to -Infinity', 'On the values v, zeroing future rows'], answer: 1,
      why: 'exp(-Infinity) = 0, so the future keys drop out and the remaining weights still sum to 1. Zeroing after the softmax would leave rows that do not sum to 1.' },
    { q: 'How many weights does a multi-head attention layer with model width C have (ignoring biases)?', options: ['C²', '4C²', '4C² · H'], answer: 1,
      why: 'W_q, W_k, W_v and W_o are each C×C; the heads share the same 4C² parameters by slicing them, they do not multiply them.' },
    { q: 'How does the cost of the score matmul grow with sequence length T?', options: ['Linearly in T', 'As T²·C', 'As T·C²'], answer: 1,
      why: 'Every query is compared with every key: T×T scores, each a dot product of length dh, over H heads, gives 2·T²·C FLOPs. That quadratic term is why long context is expensive and why the KV cache and FlashAttention exist.' },
    { q: 'What is row 0 of a causal attention weight matrix, whatever q and k are?', options: ['Uniform over all keys', '[1, 0, 0, …, 0]', 'All zeros'], answer: 1,
      why: 'Position 0 may only see itself; one visible key with the rest at -Infinity gives a single weight of 1.' },
  ],
  concept: `
## A soft dictionary lookup

Notation, used throughout: a batch of \`B\` sequences of \`T\` tokens, each token a vector of \`C\` channels, is a tensor \`[B, T, C]\`. Attention splits \`C\` into \`H\` heads of \`dh = C / H\` channels, so per-head tensors are \`[B, H, T, dh]\`.

Every position \`i\` produces three vectors from its input \`x_i\` by linear maps: a **query** \`q_i = x_i · W_q\`, a **key** \`k_i = x_i · W_k\` and a **value** \`v_i = x_i · W_v\`. Think of a dictionary: keys are what each position advertises, values are what it hands over, and the query is what position \`i\` is looking for. A hard lookup returns the value of the one matching key. Attention does the soft version: it scores every key by the dot product \`q_i · k_j\`, turns the scores into weights with a softmax, and returns the weighted average of all the values:

\`\`\`
score[i, j] = (q_i · k_j) / sqrt(dh)
w[i, :]     = softmax(score[i, :])          // T weights that sum to 1
out_i       = Σ_j w[i, j] · v_j
\`\`\`

Nothing about *which* positions matter is baked into the parameters: the same \`W_q\`, \`W_k\`, \`W_v\` produce a different weight matrix for every input, so the model decides at run time where each token looks. Because every step is a matmul or a softmax, the routing itself is differentiable: the gradient tells \`W_q\` and \`W_k\` where to look, not just what to return.

## Why divide by sqrt(dh)

:::predict
Suppose the entries of \`q_i\` and \`k_j\` are independent with mean 0 and variance 1, and \`dh = 64\`. What is the variance of the raw dot product \`q_i · k_j\`?
---
64: a sum of 64 independent unit-variance terms has variance 64, so the scores have standard deviation 8. Softmax of numbers that spread over tens of units is essentially one-hot, and the gradient through a one-hot softmax is nearly zero. Dividing by \`sqrt(64) = 8\` brings the variance back to 1. Vaswani et al. (2017) introduced the scale for exactly this reason.
:::

## The mask is the language model

A language model predicts token \`t + 1\` from tokens \`0 … t\`. If position \`t\` could attend to position \`t + 1\`, the prediction would copy the answer. The fix is a **causal mask**: before the softmax, every score with key index \`j > i\` is set to \`-Infinity\`. Since \`exp(-Infinity) = 0\`, those keys contribute nothing and the remaining weights still sum to 1. As a matrix, the mask is upper-triangular \`-Infinity\` above the diagonal; the diagonal itself stays, because a token may always read itself.

:::predict
With the causal mask applied, what is row 0 of the weight matrix, for any \`q\` and \`k\`?
---
\`[1, 0, 0, …, 0]\`. Position 0 has exactly one visible key (itself), so the softmax of \`[s, -Infinity, …]\` is 1 at index 0 and 0 elsewhere.
:::

## Why several heads

One head produces one weighted average per position, so it can express one kind of look-up at a time. Splitting \`C\` channels into \`H\` heads of \`dh\` channels runs \`H\` independent look-ups at once, each on its own slice of the channels. The split is a \`reshape\` from \`[B, T, C]\` to \`[B, T, H, dh]\` followed by a \`permute\` to \`[B, H, T, dh]\`; merging is the inverse, and an output projection \`W_o\` (\`C × C\`) then mixes the heads.

So \`W_q\`, \`W_k\`, \`W_v\` and \`W_o\` are each \`C × C\`: a layer has \`4C²\` weights (plus \`4C\` biases), and the heads slice those parameters rather than adding any. In your implementation, as in GPT-2, the first three live in one fused \`Linear(C, 3C)\`. GPT-2 small (Radford et al. 2019) uses exactly this layer with \`C = 768\` and \`H = 12\`, so \`dh = 64\` and about 2.4 million attention weights per layer.

## What it costs

Per layer, the four projections cost about \`8·T·C²\` FLOPs (2 FLOPs per multiply-add, four \`C × C\` matrices, \`T\` tokens). The score matmul and the weighted sum of values cost about \`4·T²·C\` between them. The projections grow linearly in \`T\`; the attention term grows quadratically and overtakes them once \`T > 2C\`. For GPT-2 small at \`T = 1024\` the quadratic term is already about 40% of the attention layer's FLOPs (the MLP, module 06, adds more linear work); at \`T = 32,768\` it is about 95%. Memory too: this module materialises the full \`[B, H, T, T]\` weight matrix, which at \`T = 32,768\` would be \`32,768² × 4\` bytes = 4 GiB per head in float32. Two later ideas fight this: the KV cache (module 15) stores \`k\` and \`v\` so that generating one more token costs \`O(T)\` instead of \`O(T²)\`, and FlashAttention (Dao et al. 2022) computes the softmax in tiles so the \`[T, T]\` matrix never exists in memory.

## Where this toy differs from production

Your layer has the same maths and weight layout as GPT-2, and the demo loads the lab's pre-trained checkpoint into it. But: it materialises the whole score matrix (no FlashAttention tiling); it has no notion of position (GPT-2-style learned position embeddings arrive in module 06; rotary embeddings, as in Llama, arrive in module 29); every head has its own keys and values (Llama 3 uses grouped-query attention, where several query heads share one key/value head to shrink the KV cache; module 29 builds it); there is no dropout; and everything is float32.
`,
  steps: [
    {
      id: 'scores',
      title: 'Similarity scores',
      instructions: `
Implement \`attentionScores(q, k, { scale = null } = {})\`.

\`q\` and \`k\` are Tensors whose last two dims are \`[T, dh]\` (any leading batch and head dims). Return \`q · kᵀ\` multiplied by \`scale\`, which defaults to \`1 / sqrt(dh)\`, as a Tensor of shape \`[.., T, T]\`. Entry \`[i, j]\` says how well query \`i\` matches key \`j\`.

Use Tensor methods only (\`matmul\`, \`transpose\`, \`scale\`) so the graph is recorded: step 5 differentiates through this. \`Tensor.transpose()\` swaps the last two dims, so \`k.transpose()\` is \`[.., dh, T]\` and the batched \`matmul\` does the rest.

The starter has the skeleton in place; the default factor and the product are yours.
`,
      predict: { question: 'With unit-variance q and k and dh = 64, what variance do the scores have before and after the scale?', answer: 'About 64 before (a sum of 64 unit-variance products) and about 1 after (dividing by sqrt(64) divides the variance by 64). The third test measures exactly this.' },
      hints: [
        'The scale is not 1 / dh. Which power of dh turns a variance of dh back into 1?',
        'Two things: when no scale is given, default the factor to one over the square root of dh (the last dim of q). Then take the batched matrix product of q with k whose last two dims are swapped, and multiply every entry by the factor. Every operation must be a Tensor method so backward() can reach q and k.',
        '`const factor = scale === null ? ‹default› : scale;` then `return q.matmul(‹kᵀ›).‹multiply by factor›;`',
      ],
    },
    {
      id: 'mask',
      title: 'The causal mask and the softmax',
      instructions: `
Two functions.

\`maskCausal(scores)\`: return a new Tensor equal to \`scores\` (\`[.., T, T]\`) except that every entry with key index \`j > i\` is \`-Infinity\`. Do not modify the input. \`ops.causalMask(T)\` builds a raw \`[T, T]\` tensor with 1 where \`j ≤ i\` and 0 elsewhere, and \`Tensor.maskedFill(mask, value)\` writes \`value\` wherever the mask is 0 and broadcasts the \`[T, T]\` mask over the leading dims; you may also build the mask yourself.

\`attentionWeights(scores, { causal = true } = {})\`: apply \`maskCausal\` when \`causal\`, then softmax over the last dim. Every row must sum to 1, and masked keys must get exactly 0.

Keep both differentiable (\`maskedFill\` and \`softmax\` are Tensor ops); the tests check that a masked score receives zero gradient and a visible one does not.
`,
      hints: [
        'The mask is applied to the scores, before the softmax. What does exp(-Infinity) give?',
        'maskCausal: read T from the last dim of the scores, build a [T, T] mask that keeps j ≤ i, and fill every other entry with minus infinity using a Tensor op that returns a new Tensor. attentionWeights: choose masked or unmasked scores depending on the flag, then take the softmax over the last dim.',
        '`const T = scores.shape[‹last›]; return scores.maskedFill(ops.causalMask(T), ‹fill value›);` and `const s = causal ? ‹…› : scores; return s.‹…›();`',
      ],
    },
    {
      id: 'attend',
      title: 'The weighted sum',
      instructions: `
Implement \`attention(q, k, v, { causal = true, scale = null } = {})\` on Tensors \`[.., T, dh]\` and return \`{ out, weights }\`:

- \`weights = attentionWeights(attentionScores(q, k, { scale }), { causal })\`, shape \`[.., T, T]\`;
- \`out = weights · v\`, shape \`[.., T, dh]\`: row \`i\` of \`out\` is the weighted average of the value rows, with the weights from row \`i\`.

This is the whole of scaled dot-product attention. \`out\` has the shape of \`q\`, which is what lets it be added back to the residual stream in module 06. Under the causal mask \`out_0\` equals \`v_0\` exactly; without it, all-equal scores give the plain mean of the values.
`,
      hints: [
        'You have already written both halves. The only new operation is a matmul between the weights and v.',
        'Compose: scores from step 1, passing the caller\'s scale through; weights from step 2, passing the caller\'s causal flag through; then the product of the [.., T, T] weights with the [.., T, dh] values. Return both in an object.',
        '`const weights = attentionWeights(attentionScores(‹…›), ‹…›); const out = ‹weights times v›; return { out, weights };`',
      ],
    },
    {
      id: 'heads',
      title: 'Multi-head attention',
      instructions: `
Three pieces.

\`splitHeads(x, nHead)\`: \`[B, T, C] → [B, H, T, dh]\` with \`dh = C / nHead\`. Head \`h\` gets channels \`h·dh … (h+1)·dh\` of every token. Reshape to \`[B, T, H, dh]\` first, then \`permute([0, 2, 1, 3])\`. A bare reshape to \`[B, H, T, dh]\` has the right shape and the wrong contents (it slices tokens into heads); the first test catches it.

\`mergeHeads(x)\`: \`[B, H, T, dh] → [B, T, C]\`, the exact inverse: permute back, then reshape.

\`MultiHeadAttention.forward(x)\` for \`x\` of shape \`[B, T, C]\`. The constructor already built \`this.qkv\` (\`Linear(C, 3C)\`) and \`this.proj\` (\`Linear(C, C)\`). Throw if \`x\` is not 3-D or \`C !== this.nEmbd\`. Then: \`projected = this.qkv.forward(x)\` is \`[B, T, 3C]\`; slice it along axis 2 into \`q\` (\`0…C\`), \`k\` (\`C…2C\`) and \`v\` (\`2C…3C\`) with \`Tensor.slice(2, start, end)\`; split each into heads; call your \`attention\` with \`causal: true\`; store a raw copy of the weights in \`this.lastWeights\` (\`{ shape, data: new Float32Array(weights.data) }\`); merge the heads; return \`this.proj.forward(merged)\`.

The layer has \`4C² + 4C\` parameters and must match \`lib/attention.js\` to within 1e-5 when given the same weights, which the third test does by copying them in.
`,
      predict: { question: 'For C = 8 and H = 2, how many parameters does the layer have, biases included?', answer: '4·64 + 4·8 = 288: the qkv Linear is 8×24 + 24 and proj is 8×8 + 8. The heads add nothing.' },
      hints: [
        'Reshape is free and permute copies; you need both. For forward, write the shapes of every intermediate as a comment before you write the code: [B,T,3C] → three [B,T,C] → three [B,H,T,dh] → out [B,H,T,dh] → [B,T,C] → [B,T,C].',
        'splitHeads: view the last dim as H groups of dh (a reshape to [B, T, H, dh]), then swap the T and H axes with a permute. mergeHeads: the same axis swap undoes itself, then a reshape flattens H and dh back into C. forward: validate the shape, project once, slice the projection three times along axis 2 (q, then k, then v), split each, attend causally, save a raw copy of the weights, merge the output, project again.',
        '`splitHeads`: `x.reshape([B, T, ‹H›, ‹dh›]).permute([0, ‹…›, ‹…›, 3])`. `forward`: `const projected = this.qkv.forward(x);  const q = splitHeads(projected.slice(2, ‹start›, ‹end›), this.nHead);  /* k, v likewise */  const { out, weights } = attention(q, k, v, { causal: true });  this.lastWeights = { shape: weights.shape.slice(), data: new Float32Array(weights.data) };  return ‹project the merged heads›;`',
      ],
    },
    {
      id: 'proof',
      title: 'Prove it: causality and gradients',
      instructions: `
Two checks that every attention implementation should ship with.

\`causalityProbe(layer, x, t, next)\`: run \`layer.forward(x)\` (inside \`noGrad\`), then make a copy of \`x.data\`, add \`randn(next)\` to every channel of every position **after** \`t\` (positions \`t+1 … T-1\`, all batch elements), run the layer again on the perturbed copy, and return \`{ maxBefore, maxAfter }\`: the largest absolute output change over positions \`0 … t\`, and over positions \`t+1 … T-1\`. For a causal layer \`maxBefore\` is exactly 0, not merely small: nothing at or before \`t\` can read the perturbed positions. \`maxAfter\` should be positive whenever there is anything after \`t\`; it is your evidence that the perturbation was real. Do not modify the caller's \`x\`. The tests run your probe on a real layer, on an identity layer that records what it was given, and on a layer with an off-by-one mask where position \`t\` reads \`t + 1\`: only a probe whose \`maxBefore\` includes position \`t\` catches that one.

\`gradCheckAttention(layer, x, opts = {})\`: call \`gradCheck\` from \`lib/tensor.js\` with inputs \`[x, ...layer.parameters()]\` (\`x\` must have \`requiresGrad\`) and the scalar loss \`layer.forward(x).pow(2).sum()\`, and return its result. Because the parameters are the layer's own Tensors, perturbing them in place changes what \`forward\` computes, so one call checks the input gradient and every parameter gradient at once. A check that only passes \`x\` would miss a broken \`Linear\` backward; the test counts the inputs covered.
`,
      hints: [
        'The probe compares two forward passes on inputs that differ only after t. Which positions may legitimately differ, and which must be identical to the last bit?',
        'Probe: copy the raw data, add noise to every channel of positions t+1 … T-1 in every batch element, wrap both forward passes in noGrad, then walk the outputs and send each absolute difference to maxBefore when its position is at most t (t itself included) and to maxAfter otherwise. Gradient check: hand gradCheck a function of the input that returns the scalar loss, together with the list of tensors to check; the function can ignore its arguments beyond the first because the layer already holds its parameters.',
        '`const noisy = new Float32Array(x.data); for (b …) for (let i = ‹first perturbed position›; i < T; i++) for (c …) noisy[(b * T + i) * C + c] += randn(next);` then compare `noGrad(() => layer.forward(x))` with the forward on `new Tensor({ shape: x.shape.slice(), data: noisy })`. And `return gradCheck((xIn) => ‹scalar loss›, [x, ‹…›], opts);`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague, without the word "attention", what out_i = Σ_j w[i, j]·v_j computes and where the w[i, j] come from. Then explain why the same parameters can route information differently for different inputs.',
    'The mask is one line of code. Argue from the maths (not the code) why removing it turns a language model into something that cannot be trained by next-token prediction.',
    'Your layer stores a [B, H, T, T] weight matrix. Write down its size in bytes for the lab model (T = 64, H = 4) and for a 128k-context model with 32 heads, and say which of the two later ideas (KV cache, FlashAttention) addresses which cost.',
  ],
  stretch: [
    'Add grouped-query attention: keep H query heads but only G < H key/value heads, each shared by H / G query heads (Llama 3 uses H = 32, G = 8 for the 8B model). Count the parameters and the KV-cache size before and after. Module 29 builds this properly; treat this as a preview.',
    'Implement the attention forward as a single-pass "online softmax" that never materialises the [T, T] matrix: for each query, stream over keys keeping a running max, a running sum of exponentials and a running weighted sum of values, rescaling the last two whenever the max grows. This is the core of FlashAttention (Dao et al. 2022); compare outputs with your attention() to 1e-5.',
    'Add rotary position embeddings (RoPE, Su et al. 2021), as in Llama: rotate pairs of q and k channels by an angle proportional to the position before the score matmul, and verify that the score depends only on the relative distance i − j. Module 29 builds this properly; treat this as a preview.',
    'Load lib/checkpoints/tiny-gpt.json, run your layer on a sentence, and find the head whose weights most often peak on the previous token. Previous-token heads are the first half of the induction-head circuit described by Olsson et al. (2022), and Wang et al. (2022) located two of them in GPT-2 small in their indirect-object-identification circuit.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
