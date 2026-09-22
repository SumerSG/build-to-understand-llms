export default {
  id: '15-kv-cache',
  title: 'The KV cache',
  track: 'inference',
  minutes: 90,
  threshold: 'In causal attention a past token\'s keys and values never change, so store them once and each new token costs one query against the cache instead of a full recompute; the price is memory that grows with context.',
  goal: 'Incremental decoding with a key/value cache that produces identical tokens to full recomputation while doing a fraction of the FLOPs: newCache, forwardStep, prefill, a greedy generator that runs both ways, a FLOP and byte cost model, and a measured speedup on the checkpoint.',
  prereqs: ['05-attention', '06-transformer', '14-decoding'],
  recall: [
    { q: 'In module 05, which scores does the causal mask set to −Infinity before the softmax?', options: ['Scores between a query and every key', 'Scores between a query and keys at LATER positions', 'Scores on the diagonal'], answer: 1,
      why: 'A query at position t sees keys 0…t only. That one rule is why a past token\'s key and value never need to change once computed: nothing later can reach back into them.' },
    { q: 'In module 06, why is the language-model head not counted separately in `numParams()`?', options: ['It has no parameters', 'It is tied to the token embedding `wte`, so logits = x · wteᵀ reuses the same matrix', 'It is folded into the final LayerNorm'], answer: 1,
      why: 'You will write the same closed-form count here (`paramCount`) and check it against `GPT.numParams()`; the tied head is the classic place to double-count.' },
    { q: 'From modules 06 and 08: processing one token through a model with N parameters costs approximately how many FLOPs in the forward pass?', options: ['N', '2N', '6N'], answer: 1,
      why: 'One multiply and one add per weight. 6N is the training figure (forward plus a backward that is twice as expensive). Serving pays 2N per token, for every token, forever.' },
    { q: 'In module 14, `generate` called `prefill(model, cache, promptIds)` from lib/infer.js once. What did it return?', options: ['The logits after the LAST prompt token', 'Logits for every prompt position', 'The prompt embedded as vectors'], answer: 0,
      why: 'One Float32Array of vocabSize logits: the distribution for the first generated token. This module is where you build that function yourself.' },
    { q: 'In module 01, `reshape` did not touch the data. What about `concat` in lib/ops.js?', options: ['Also free: it only changes the shape', 'It allocates a new Float32Array and copies both inputs into it', 'It mutates the first argument in place'], answer: 1,
      why: 'Your cache grows by concat, so every decode step copies the whole cache. That is fine here and is exactly what production engines avoid with preallocated, paged buffers (module 16).' },
  ],
  review: [
    { q: 'Why can the keys and values of earlier tokens be stored and reused for every later decode step?', options: ['Because the weights are frozen', 'Because causal attention lets position t depend only on positions 0…t, so nothing later changes them', 'Because keys and values are cheap to compute'], answer: 1,
      why: 'Frozen weights are necessary but not sufficient: without the causal mask, a new token would change every earlier position\'s attention output, and with it every later layer\'s keys and values.' },
    { q: 'Your cost model gives cached = `2N + 4·L·T·C` FLOPs per token at context T. The uncached cost is…', options: ['The same', 'T times the cached cost', '2 times the cached cost'], answer: 1,
      why: 'Without a cache the whole T-token context goes through the model again to produce one token. Over a generation of n tokens that sums to about n²/2 token-forwards instead of n.' },
    { q: 'A Llama-3-8B-shaped model (32 layers, 8 KV heads, head dim 128, bf16) stores approximately how much KV cache per token?', options: ['8 KB', '128 KB', '2 MB'], answer: 1,
      why: '2 · 32 · 8 · 128 · 2 bytes = 131,072 bytes. At 128k tokens of context that is 16 GiB for one sequence, the same as the model\'s weights.' },
    { q: 'Why is the decode phase memory-bandwidth-bound at batch size 1?', options: ['Because softmax is slow', 'Because every weight is read from HBM once to do only one token\'s worth of arithmetic on it', 'Because the cache is on the CPU'], answer: 1,
      why: 'Approximately 16 GB of weights at approximately 3.35 TB/s (NVIDIA H100 datasheet) is about 5 ms, while 2N = 16 GFLOP takes tens of microseconds. Batching (module 16) exists to amortise that read.' },
    { q: 'Grouped-query attention (Llama 3, Mistral) shrinks the KV cache by…', options: ['Storing keys in int8', 'Sharing one key/value head among a group of query heads, so only nKVHead heads are stored', 'Dropping the value tensor'], answer: 1,
      why: 'Llama-3-8B has 32 query heads but 8 KV heads: a 4× smaller cache with almost no quality loss (Ainslie et al. 2023). Multi-query attention (Shazeer 2019) is the limit with one KV head.' },
  ],
  concept: `
## What the uncached model wastes

To produce token 11, the uncached \`forward\` of module 14 embeds all ten earlier tokens, runs them through every layer, and reads one row of the result: the logits at the last position. The other nine rows are thrown away. Token 12 repeats all of it, plus one more token. Generating \`n\` tokens costs about \`n²/2\` token-forwards instead of \`n\`.

The waste is avoidable because of the causal mask. In every layer, the key and value of position \`t\` are linear functions of the residual stream at position \`t\`, which depends only on positions \`0…t\`. Nothing that arrives later can change them. So once computed, \`k\` and \`v\` are final: store them. The **KV cache** is that store: per layer, a key block and a value block of shape \`[H, T, dh]\` (\`H\` heads, \`T\` tokens so far, \`dh = C / H\` channels per head) that grows by one row per token.

With the cache, decoding a new token means: embed it at position \`T\`, in each layer compute its own \`q, k, v\` (a \`[1, C] × [C, 3C]\` matmul), append \`k\` and \`v\`, and attend the one query over all \`T + 1\` stored keys. No mask is needed: everything in the cache is in the past by construction.

:::predict
With the cache, how does the cost of producing one more token depend on the context length \`T\`? Think about the matmuls against the weights separately from the attention itself.
---
The weight matmuls do not depend on \`T\`: one token goes through them whatever the context, about \`2N\` FLOPs for \`N\` parameters. Only the attention grows: one query against \`T\` keys, then \`T\` values, \`4·L·T·C\` FLOPs across \`L\` layers. Per-token cost is \`2N + 4·L·T·C\`, linear in \`T\` with a large constant. Without the cache it is \`T\` times that.
:::

## Two phases: prefill and decode

**Prefill** pushes the whole prompt through the model. All \`T\` prompt positions are independent given the causal mask, so a real engine does it in one batched pass: \`[T, C]\` activations against every weight matrix, lots of arithmetic per byte of weight read. Prefill is **compute-bound**, and its attention term is quadratic in the prompt length.

**Decode** produces one token per step. Every weight matrix is read once from GPU memory to do a single row's worth of arithmetic on it. For a Llama-3-8B-shaped model in bf16 that is approximately 16 GB of weights per token; NVIDIA's H100 datasheet lists approximately 3.35 TB/s of HBM3 bandwidth, so the read alone takes about 5 ms, while the \`2N ≈ 16 GFLOP\` of arithmetic would take tens of microseconds at the datasheet's roughly 990 TFLOP/s of dense bf16. Decode at batch size 1 is **memory-bandwidth-bound** by a factor of a few hundred. Module 16 amortises the weight read across a batch; module 23's roofline model says which regime you are in.

## The arithmetic of the cache

For one sequence of \`T\` tokens:

\`bytes = 2 · nLayer · nKVHead · headDim · T · bytesPerElement\`

The leading 2 is keys plus values. For Llama-3-8B (32 layers, 8 key/value heads, head dim 128, bf16): \`2 · 32 · 8 · 128 · 2 = 131,072\` bytes, 128 KiB per token. At its 128k-token context that is 16 GiB for a single sequence, as much as the weights. A serving GPU holds many sequences, so the cache, not the weights, limits batch size.

Why 8 heads when the model has 32? **Grouped-query attention** (GQA, Ainslie et al. 2023) shares one key/value head among a group of query heads; Llama 3 uses groups of 4, so the cache is 4× smaller. **Multi-query attention** (MQA, Shazeer 2019) is the limit with one shared head. DeepSeek-V2 and V3 cache a compressed latent vector per token instead (multi-head latent attention). Each is a cache-size decision first.

:::predict
Llama-3-70B has 80 layers, 8 key/value heads and head dim 128, served in bf16. How many bytes of cache per token, and how much at 128k context?
---
\`2 · 80 · 8 · 128 · 2 = 327,680\` bytes, 320 KiB per token, so 40 GiB at 128k tokens. Without GQA (64 key/value heads) it would be 320 GiB per sequence, more than any single GPU holds.
:::

## What context length costs

Memory grows **linearly** with \`T\` per sequence. Prefill compute grows **quadratically**, because each of \`T\` queries attends to up to \`T\` keys. Decode compute per token grows linearly in \`T\` but is dominated by the fixed \`2N\` until the context is very long; that is most of why input tokens are priced below output tokens.

Two consequences are whole modules. Reserving the worst-case cache per request up front wastes most of a GPU's memory; **paged attention** (module 16) allocates it in fixed blocks. The cache of a shared prompt prefix is identical across requests; **prefix caching** (module 17) reuses it.

## Where this toy differs from production

Batch size 1 and float32 throughout. The cache grows by \`concat\`, which copies the whole block every step; real engines write into a preallocated buffer, paged in 16-token blocks (vLLM). \`prefill\` here is a loop of decode steps: the same numbers as the batched pass, none of its efficiency. This model has one key/value head per query head (no GQA) and learned absolute position embeddings; Llama-family models apply RoPE to \`q\` and \`k\` before caching, so stored keys carry their position inside them. None of this changes the invariant you are about to test: cached and uncached decoding must give the same logits at every position, to within float32 noise.
`,
  steps: [
    {
      id: 'attend',
      title: 'One query against the stored keys and values',
      instructions: `
Read the worked examples first. \`forward(model, ids)\` is the complete uncached model on raw tensors: embeddings, per layer \`ln1 → q,k,v → causal attention → proj\` and \`ln2 → MLP\` with residual adds, then the final LayerNorm and the tied head. It is your reference for everything that follows, and its helpers (\`projectQKV\`, \`mergeHeads\`, \`mlpForward\`, \`embedTokens\`, \`head\`) are yours to call.

Implement \`attendOne(q, k, v)\`: attention for the **newest position only**. \`q\` is \`[H, 1, dh]\`; \`k\` and \`v\` are \`[H, t, dh]\` and hold every position up to and including the current one. Return \`[H, 1, dh]\`:

\`\`\`
scores  = q · kᵀ / sqrt(dh)        // [H, 1, t]
weights = softmax(scores)          // over the t stored positions
out     = weights · v              // [H, 1, dh]
\`\`\`

There is no mask. In \`forward\` the mask hides keys at later positions; here every stored key is at an earlier or equal position, so the mask would be all ones. Compare this line by line with the attention inside \`forward\`: it is the same computation for one row of \`scores\`.
`,
      predict: { question: 'With exactly one key stored (t = 1), what does attendOne return, whatever q is?', answer: 'Exactly v. A softmax over one entry is [1], so the output is 1 · v. This is the first decode step of every sequence: the first token attends only to itself.' },
      hints: [
        'Every operation is one call in lib/ops.js: matmul, transpose (swaps the last two axes, so kᵀ is [H, dh, t]), scale, softmax (last axis). The head dimension is the last entry of q.shape.',
        'Batched matmul over the head axis does all heads at once: `matmul(q, transpose(k))` is [H, 1, t]. Scale it by `1 / Math.sqrt(headDim)`, softmax along the last axis, then `matmul(weights, v)`.',
        '`const scores = ops.scale(ops.matmul(q, ops.transpose(k)), ‹factor›); const weights = ‹softmax of scores›; return ops.matmul(weights, v);`',
      ],
    },
    {
      id: 'cache',
      title: 'The cache data structure',
      instructions: `
Two functions.

\`newCache(model)\` returns \`{ k, v, length: 0, maxLength: blockSize }\`. \`k\` and \`v\` are arrays with one raw tensor per layer, each of shape \`[nHead, 0, headDim]\` (\`headDim = nEmbd / nHead\`): a block with no positions yet. \`length\` is how many tokens are stored; \`maxLength\` is the model's \`blockSize\`, because the position table \`wpe\` has that many rows and no token can be placed beyond it.

\`appendKV(cache, layer, k, v)\` appends the \`[H, n, dh]\` keys and values of the newest position(s) to that layer's blocks along the **time axis** (axis 1), stores the results back into \`cache.k[layer]\` and \`cache.v[layer]\`, and returns \`{ k, v }\`, the layer's full blocks after the append, so the caller can attend over them. It must not touch \`cache.length\`: the count advances once per token, after every layer has been appended to, and that is \`forwardStep\`'s job.

\`ops.concat(a, b, axis)\` does the growing. It copies; module 16 replaces it with a preallocated, paged buffer.
`,
      hints: [
        'A loop over layers pushing `ops.zeros([nHead, 0, headDim])` into two arrays. Read nLayer, nHead, nEmbd and blockSize from model.config.',
        'For appendKV, the shape you want afterwards is [H, t + n, dh]: heads unchanged, time longer, channels unchanged. That is a concat along axis 1; along axis 0 you would get [2H, …], which is the classic mistake.',
        '`cache.k[layer] = ops.concat(cache.k[layer], k, ‹axis›); cache.v[layer] = ‹same for v›; return { k: cache.k[layer], v: cache.v[layer] };`',
      ],
    },
    {
      id: 'step',
      title: 'One decode step',
      instructions: `
Implement \`forwardStep(model, cache, id)\`: decode one token and return its logits as a \`Float32Array\` of length \`vocabSize\` (the distribution for the **next** token), advancing \`cache.length\` by one.

Mirror \`forward\`, but for a single position \`position = cache.length\`:

1. Throw if \`cache.length >= cache.maxLength\`: there is no position embedding for that slot.
2. \`x = embedTokens(w, [id], position)\`, shape \`[1, C]\`.
3. For each layer: \`ln1\`; \`projectQKV(w, prefix, normed, 1, nHead, headDim)\` for this position's \`q, k, v\` (\`[H, 1, dh]\` each); \`appendKV\`; \`attendOne\` over the returned blocks; \`mergeHeads(…, 1, nEmbd)\`; the \`attn.proj\` residual; \`ln2\` and the MLP residual, exactly as in \`forward\`.
4. \`cache.length = position + 1\`; return \`head(w, x).data\`.

The test feeds a sequence one token at a time and checks that every step's logits equal the corresponding row of \`forward\` within \`1e-4\`. The order of the residual adds and LayerNorms matters; copy it from the reference rather than from memory.
`,
      predict: { question: 'You feed token id 3 twice in a row into a fresh cache. Are the two logit vectors the same?', answer: 'No. The second copy is embedded at position 1 (a different position embedding) and attends over two stored keys instead of one. Identical logits would mean the position or the cache is being ignored.' },
      hints: [
        'Start by copying the body of forward and deleting what a single position does not need: the mask, causalMask, and the time argument becomes 1. The scores/softmax/values lines become one call to attendOne.',
        'Per layer: `normed = layerNorm(x, ln1)`; `{ q, k, v } = projectQKV(w, prefix, normed, 1, nHead, headDim)`; `stored = appendKV(cache, layer, k, v)`; `attended = mergeHeads(attendOne(q, stored.k, stored.v), 1, nEmbd)`; `x = add(x, linear(attended, proj.weight, proj.bias))`; then ln2 and mlpForward with a second residual add.',
        '`let x = embedTokens(w, [id], position); for (…layers…) { const normed = ops.layerNorm(x, w[`${prefix}.ln1.gamma`], w[`${prefix}.ln1.beta`]); const { q, k, v } = projectQKV(w, prefix, normed, 1, nHead, headDim); ‹append, attend, merge, proj residual›; const normed2 = …; x = ops.add(x, mlpForward(w, prefix, normed2)); } cache.length = position + 1; return head(w, x).data;`',
      ],
    },
    {
      id: 'prefill',
      title: 'Prefill and generation, both ways',
      instructions: `
\`prefill(model, cache, ids)\`: run every prompt token through \`forwardStep\` and return the last logits. Throw on an empty prompt. This is the toy's prefill phase; a real engine runs the prompt as one batched pass because that is compute-efficient, but the numbers are identical.

\`generateGreedy(model, promptIds, maxNewTokens, { cached = true })\`: return an array of exactly \`maxNewTokens\` new ids (not including the prompt), always picking \`argmaxOf(logits)\`, which is provided.

- **cached**: \`prefill\` once, then loop: argmax, push, \`forwardStep\` on the pushed id. Do not run a \`forwardStep\` after the final token: its logits would never be used, and at the cache limit it would throw for nothing.
- **uncached**: loop: \`forward\` over \`promptIds\` plus everything generated so far, take the last row (\`ops.slice(all, 0, T − 1, T).data\`), argmax, push.

Do not mutate \`promptIds\`. The test compares both paths against an independent greedy loop written with lib/infer.js, so it catches the case where your two paths agree with each other but not with the reference.
`,
      hints: [
        'The uncached path is what module 14 would have done without lib/infer.js: recompute, read the last row, repeat. The cached path is the same loop with forwardStep replacing forward.',
        'Cached: `let logits = prefill(model, cache, promptIds); for (i < n) { const id = argmaxOf(logits); out.push(id); if (i === n − 1) break; logits = forwardStep(model, cache, id); }`.',
        'Uncached: `const ids = promptIds.slice(); for (i < n) { const all = forward(model, ids); const last = ops.slice(all, 0, ‹start›, ‹end›).data; const id = argmaxOf(last); out.push(id); ids.push(id); }`',
      ],
    },
    {
      id: 'cost',
      title: 'The cost model: FLOPs and bytes',
      instructions: `
Three functions of a config, no model needed.

\`paramCount(config)\`: \`V·C + T·C\` for the token and position embeddings, \`L · (12C² + 13C)\` for the blocks (qkv \`3C²\`, proj \`C²\`, MLP \`8C²\`; \`9C\` of biases; \`4C\` of LayerNorm gains and shifts), plus \`2C\` for the final LayerNorm. The head is tied to \`wte\`. Must equal \`new GPT(config).numParams()\`.

\`flopsPerToken(config, contextLen, { cached = true })\`: cached is \`2 · paramCount + 4 · L · T · C\` (two FLOPs per parameter, plus the query-key scores and the weighted values, \`2·L·T·C\` each). Uncached is that times \`T\`, because the whole context is recomputed. This is the rule of thumb behind the \`2N\` in Kaplan et al. (2020); it counts matmuls only.

\`cacheBytes(config, contextLen, { bytesPerElement = 4 })\`: \`2 · nLayer · nKVHead · headDim · T · bytesPerElement\`, where \`nKVHead = config.nKVHead ?? nHead\` and \`headDim = config.headDim ?? nEmbd / nHead\`, so the same function sizes this model (float32, no GQA) and Llama-3-8B (bf16, 8 KV heads out of 32).
`,
      hints: [
        'Write paramCount as three named terms and add them; the test message spells the formula out. flopsPerToken and cacheBytes are each one expression once you have the symbols.',
        'flops: `perToken = 2 * paramCount(config) + 4 * nLayer * contextLen * nEmbd`; return `cached ? perToken : perToken * contextLen`. bytes: read nKVHead with `??` so a plain config still works.',
        '`const perBlock = 12 * nEmbd * nEmbd + 13 * nEmbd; return ‹embeddings› + nLayer * perBlock + 2 * nEmbd;` and `return 2 * nLayer * nKVHead * headDim * contextLen * bytesPerElement;`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why the keys and values of earlier tokens can be cached but the query cannot, and what property of the architecture the cache depends on. What would break if attention were bidirectional?',
    'Your demo measured a speedup from the cache. Using your own cost model, say where the remaining per-token time goes at batch size 1 on a GPU, and why a bigger batch makes each token cheaper without making the cache smaller.',
    'A colleague proposes doubling the context window of a served model from 64k to 128k. Using cacheBytes, describe what changes in memory per request, in prefill cost, and in decode cost per token, and which of the three is the real constraint.',
  ],
  stretch: [
    'Replace concat with a preallocated `[H, blockSize, dh]` buffer per layer written in place at row `cache.length`, and measure the per-token latency change at long context. This is the difference between HF transformers\' `DynamicCache` and `StaticCache`, and the reason vLLM writes into fixed blocks.',
    'Add grouped-query attention to the toy: give the config `nKVHead < nHead`, project only that many key/value heads, and repeat each one across its group of queries. Check `cacheBytes` shrinks by `nHead / nKVHead` as it does in Llama 3 (32 query heads, 8 KV heads).',
    'Implement a batched prefill: run the worked `forward`-style pass over the whole prompt once, copy every layer\'s `[H, T, dh]` keys and values into the cache, and confirm the logits match the token-by-token prefill. Then time both on a 60-token prompt. vLLM and SGLang schedule prefill and decode as different kinds of work for exactly this reason.',
    'Add a sliding window: keep only the last `W` positions in each layer\'s cache (Mistral 7B uses W = 4096) and measure how the logits drift from the full-context model as the sequence passes W. Then read how StreamingLLM keeps the first few "attention sink" tokens as well.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
