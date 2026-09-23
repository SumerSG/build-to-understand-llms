export default {
  id: '16-batching',
  title: 'Continuous batching & paged attention',
  track: 'inference',
  minutes: 105,
  threshold: 'A serving engine is a scheduler: every decode iteration it decides which requests share the GPU, and paging the KV cache into fixed blocks is what lets it pack memory tightly enough to keep that batch large.',
  goal: 'A continuous-batching scheduler with a paged KV block allocator; you measure throughput, latency and memory fragmentation, and compare static batching, continuous batching and continuous batching with paging on the same trace.',
  prereqs: ['14-decoding', '15-kv-cache'],
  recall: [
    { q: 'In module 15, what did the KV cache let you skip on every decode step?', options: ['Sampling the next token', 'Recomputing the keys and values of every earlier token', 'The final layer norm'], answer: 1, why: 'The cache turns decoding into one forward pass over one token. That single-token step is exactly the iteration this module schedules.' },
    { q: 'How does one request\'s KV cache grow while it generates?', options: ['It stays fixed', 'Linearly, one entry per token', 'Quadratically in the sequence length'], answer: 1, why: 'One key and one value per layer per token. Linear growth is why the allocator can hand out memory one block at a time instead of reserving the worst case up front.' },
    { q: 'From module 08: generating one token with a model of N parameters costs approximately how many FLOPs?', options: ['2N', '6N', 'N²'], answer: 0, why: 'Forward pass only. That is the per-token term in this module\'s iteration cost model; the fixed term is reading the weights out of HBM.' },
    { q: 'Module 13 printed a bootstrap confidence interval next to each accuracy. What was it for?', options: ['Making the score look rigorous', 'Showing how much of the number is sampling noise from a finite set', 'Speeding up the eval'], answer: 1, why: 'Same discipline as serving: a mean latency hides the tail, which is why you will report p50 and p95 time-to-first-token rather than an average.' },
    { q: 'In module 14, how many tokens does one sampling step produce per sequence?', options: ['Exactly one', 'As many as the temperature allows', 'One per attention head'], answer: 0, why: 'One iteration, one token per running request. Everything in this module follows from that: throughput can only come from running more requests per iteration.' },
  ],
  review: [
    { q: 'An iteration costs `tFixed + tPerToken · tokens` with tFixed = 5 ms and tPerToken = 50 us. Going from a batch of 1 to a batch of 32 decodes changes the cost per token from…', options: ['5.05 ms to 5.05 ms', '5.05 ms to 0.21 ms', '5.05 ms to 1.6 ms'], answer: 1, why: 'One iteration costs 5.05 ms at batch 1 and 6.6 ms at batch 32, so the per-token cost falls about 24x. Throughput comes from amortising the fixed term, not from making the arithmetic cheaper.' },
    { q: 'In static batching, a request that needs 1 token sharing a batch with one that needs 500 finishes after…', options: ['1 iteration', '500 iterations, when the whole batch returns', '250 iterations on average'], answer: 1, why: 'Head-of-line blocking. The slot is not released and the response is not returned until the longest member of the batch is done — which is exactly what iteration-level scheduling (Orca, Yu et al. 2022) fixed.' },
    { q: 'Why cut the KV cache into 16-token blocks instead of giving each request one contiguous slab?', options: ['Blocks make attention arithmetic cheaper', 'A slab must be sized for the longest possible output, so most of it is never used', 'Blocks reduce the number of layers'], answer: 1, why: 'The vLLM paper (Kwon et al. 2023) measured that earlier systems used only about 20-38% of their KV memory for real token state. Blocks bound the waste to the tail of one block per sequence.' },
    { q: 'Your scheduler cannot grow a running request because no block is free. Preemption by recomputation means…', options: ['Copying that request\'s blocks to host memory and back', 'Freeing its blocks and re-running its prompt plus its generated tokens when it is readmitted', 'Dropping the request with an error'], answer: 1, why: 'Recompute trades compute for memory and needs no PCIe transfer; swapping to host memory is the other option vLLM implements. Both are invisible to the user except as latency.' },
    { q: 'Two requests send the identical 40-token prompt and blockSize is 16. How many blocks can they share?', options: ['3', '2', '0'], answer: 1, why: 'Only the two full blocks. The third block holds 8 prompt tokens and will be written with each request\'s own generated tokens, so each request gets its own. vLLM\'s automatic prefix caching likewise shares only full blocks; true copy-on-write (vLLM\'s parallel sampling) shares the partial block too and copies it at the first write.' },
  ],
  concept: `
## The engine is a scheduler

A serving engine looks like a model and behaves like an operating system. The model is fixed; what varies, thousands of times a second, is **which requests share the GPU next** and **who keeps memory**. Those two decisions set your throughput, your latency and your bill.

One iteration runs the model once over a batch. Each running request contributes one token (a decode step) or its whole prompt (a prefill step, module 15), and the iteration costs approximately

\`t = tFixed + tPerToken · (tokens in the batch)\`

\`tFixed\` does not care how big the batch is: the weights must be read out of HBM whatever happens. For an 8B model in bf16 that is approximately 16 GB at the approximately 3.35 TB/s of HBM3 bandwidth in NVIDIA's H100 datasheet, about 5 ms. \`tPerToken\` is the arithmetic: approximately \`2N\` FLOPs per token (module 08), about 46 us at an assumed 350 TFLOP/s of achieved bf16 throughput, which the simulator rounds to 50 us. **The fixed term is 100 times the per-token term**, and that ratio drives everything in this module.

:::predict
With \`tFixed = 5 ms\` and \`tPerToken = 50 us\`, what does one decode iteration cost at batch 1, and at batch 32? What is the cost per token in each case?
---
Batch 1: \`5 + 0.05 = 5.05 ms\`, so 5.05 ms per token. Batch 32: \`5 + 1.6 = 6.6 ms\`, so 0.21 ms per token — about **24x cheaper per token** for 30% more time per step. Throughput comes from amortising \`tFixed\`, and it keeps rising until the per-token term dominates (compute-bound) or you run out of requests to put in the batch.
:::

## Static batching wastes the tail

The obvious way to batch is to collect N requests, run them together until all finish, then take the next N. Output lengths are heavily skewed — most requests want a sentence, a few want an essay — so a batch runs at the pace of its longest member. A request that finished at iteration 3 keeps its slot, is padded, costs compute, and its answer is not returned until iteration 500. That is **head-of-line blocking**.

Orca (Yu et al. 2022) replaced it with **iteration-level scheduling**: after every iteration, retire whoever finished and admit whoever is waiting. Nothing waits for a batch boundary, because there are none. vLLM, TensorRT-LLM ("in-flight batching") and SGLang all work this way.

## Memory is the real constraint

Continuous batching only helps if you can hold the batch. A Llama-3-8B-shaped model with grouped-query attention stores approximately 128 KB of KV per token; an 80 GB H100 holding approximately 16 GB of weights has room for roughly 500,000 tokens — about 250 requests of 2,000 tokens. Reserve a contiguous slab per request, sized for the longest output it *might* produce, and you waste most of that: the vLLM paper (Kwon et al. 2023) measured existing systems using only **20-38% of KV memory for actual token state**.

PagedAttention borrows the fix from virtual memory. Cut the cache into fixed **blocks** (vLLM's default is 16 tokens), give each request a **block table** mapping logical positions to physical blocks, and allocate one block at a time. Waste drops to the unused tail of one block per sequence, and because blocks are indirect, two requests with the same prefix can point at the same physical block under a reference count, copying only when one writes (copy-on-write).

:::predict
Two requests arrive with the identical 40-token system prompt, blockSize 16. How many blocks does the second one have to allocate?
---
One. Blocks 0 and 1 are full and identical, so the second request retains them (refcount 2). Block 2 holds 8 prompt tokens and 8 empty slots that each request will fill with its *own* generated tokens, so it must be copied. Prefix sharing saves whole blocks, never partial ones.
:::

## Scheduling under pressure

With memory finite the scheduler needs two more policies. **Admission**: a waiting request joins only if its prompt's blocks are free — taking a smaller one first instead would starve the head of the queue. **Preemption**: if a running request cannot grow, evict someone. The vLLM paper evicts the most recently arrived request and either swaps its blocks to host memory or throws them away and recomputes them on readmission (vLLM's current V1 engine keeps only recomputation). Either way the work is redone, so preemption is a latency spike, not an error.

You watch four numbers, not one: **TTFT** (time to first token, dominated by queueing and prefill), **TPOT** or inter-token latency (dominated by batch size), throughput, and **goodput** — requests served *within* their latency targets. Raising the batch raises throughput and TPOT together, so serving is a dial between them. Chunked prefill (Sarathi-Serve, Agrawal et al. 2024) splits a long prompt across iterations, because one big prefill stalls everyone else's decode.

## What this simulator is not

You are building the scheduler, not the kernels. The cost model is linear in batched tokens, so attention's quadratic cost in context length, FlashAttention tiling and CPU overhead never appear; a real iteration also gets slower as sequences lengthen. Arrivals are synthetic, not production traffic, and your simulator knows each output length in advance — a real scheduler cannot, which is why length prediction is an open research topic. Sharing here matches whole prompts by key; real prefix caching finds the longest shared prefix, with a radix tree over token ids in SGLang's RadixAttention or a table of chained block hashes in vLLM, and evicts LRU, which is module 17. And a real engine runs on many GPUs with tensor parallelism (module 24), which changes \`tFixed\` but no idea in this module.
`,
  steps: [
    {
      id: 'cost',
      title: 'The iteration cost model',
      instructions: `
Three small functions fix the physics of the simulator. A **request state** is the object \`newRequestState\` builds (read it in the starter): \`{ id, arrival, promptLen, outputLen, prefilled, generated, firstToken, restarts, blocks }\`.

\`tokensThisIteration(s)\` — how many tokens this request puts into the current iteration. A prefilled request decodes exactly **one** token. A request that has not been prefilled must process \`promptLen + generated\` tokens: its prompt, plus any tokens it had already produced before it was preempted and lost its cache (step 5). For a fresh request \`generated\` is 0, so this is just the prompt.

\`iterationSeconds(batch, cfg)\` — \`cfg.tFixed + cfg.tPerToken · (total tokens in the batch)\`. \`tFixed\` is paid **once per iteration**, not once per request: that is the whole reason batching works. An empty batch still costs \`tFixed\`.

\`kvTokens(s)\` — how many KV entries the request currently owns: 0 before its prefill, and \`promptLen + generated − 1\` afterwards. Check the off-by-one: right after prefill, \`generated\` is 1 and the cache holds exactly the prompt.
`,
      predict: { question: 'A batch holds one request prefilling a 512-token prompt and 31 requests decoding. With tFixed = 5 ms and tPerToken = 50 us, how long is that iteration, and how long would it be without the prefill?', answer: '`5 + 0.05 × 543 = 32.2 ms` with the prefill, `5 + 0.05 × 31 = 6.6 ms` without. One long prefill makes every decoding request wait five times longer for its next token — the stall that chunked prefill (Sarathi-Serve) was invented to remove.' },
      hints: [
        'Two of the three functions are a single expression. The only real decision is what a request that has been preempted owes: its KV cache is gone, so what has to be pushed through the model again before it can emit its next token?',
        'Write `tokensThisIteration` first and call it from the loop in `iterationSeconds`; then `kvTokens` is the same quantity seen from the cache\'s side. Test yourself on one case: a request with promptLen 50 that has generated 1 token owns 50 entries, not 51.',
        '`tokensThisIteration`: `return s.prefilled ? 1 : /* prompt plus what it must replay */;`. `iterationSeconds`: sum `tokensThisIteration` over the batch, then add `cfg.tFixed` once, outside the loop. `kvTokens` has the same `s.prefilled ? … : 0` shape; derive the expression from the two checkpoints in the instructions (50 entries after prefill, one more per decode).',
      ],
    },
    {
      id: 'static',
      title: 'Static batching, and what it wastes',
      instructions: `
\`runStatic(requests, cfg)\` — the baseline every serving paper compares against. A \`request\` is \`{ id, arrival, promptLen, outputLen, key }\`, with \`arrival\` in seconds.

The loop, with \`clock\` starting at 0:

1. If nothing has arrived yet, jump the clock to the next arrival (the engine idles).
2. Fill a batch with up to \`cfg.maxBatch\` requests that have already arrived, in arrival order.
3. Run **one prefill iteration** for the whole batch. Advance the clock by \`iterationSeconds\`. Every member now has \`prefilled = true\`, \`generated = 1\` and \`firstToken = clock\`: the prefill emits the first token.
4. Run decode iterations until the **longest** member has produced \`outputLen\` tokens. Finished members keep their slot and are padded — they still contribute one token to \`iterationSeconds\` every iteration. That waste is the point of the exercise.
5. When the batch is done, record every member with \`end = clock\` — the whole batch returns together.

Return \`summarize('static', records, { makespan, iterations, batchTokens })\`, where \`makespan\` is the final clock, \`iterations\` counts prefill plus decode iterations, and \`batchTokens\` is the total number of tokens the model processed, padding included.

A record is \`{ id, arrival, firstToken, end, promptLen, outputLen, restarts }\`; \`summarize\` turns those into TTFT, TPOT and throughput.
`,
      hints: [
        'Two nested loops: an outer one over batches and an inner one over decode iterations. Ask what has to be true before the outer loop can start its next pass.',
        'Sort the requests by arrival once, then walk them with an index. The number of decode iterations for a batch is `max(outputLen) − 1`, because the prefill already produced one token for everyone.',
        'Inner loop: `for (let step = 1; step < longest; step++) { clock += iterationSeconds(batch, cfg); iterations++; for (const s of batch) if (s.generated < s.outputLen) s.generated++; }` — note that the cost uses the whole batch, including the requests that are already done.',
      ],
    },
    {
      id: 'continuous',
      title: 'Continuous batching',
      instructions: `
\`runContinuous(requests, cfg)\` — the same simulator with the batch boundary deleted. Per iteration:

1. Admit arrived requests, in arrival order, while \`running.length < cfg.maxBatch\`.
2. If nothing is running, jump the clock to the next arrival and continue.
3. Cost the iteration with \`iterationSeconds(running, cfg)\` — a newly admitted request contributes its whole prompt to the same iteration as everyone else's decode step — and advance the clock.
4. Update every running request: a request that was not prefilled becomes \`prefilled\`, sets \`firstToken = clock\` and \`generated = 1\`; the others do \`generated++\`.
5. Retire every request with \`generated >= outputLen\`: record it with \`end = clock\` and remove it from \`running\` **this iteration**, so the slot is free for the next one.

Return \`summarize('continuous', records, { makespan, iterations, batchTokens })\`.

A test checks that \`batchTokens\` equals exactly \`sum(promptLen) + sum(outputLen − 1)\`. There is no padding left to pay for: every token the model processes is a prompt token or a token someone actually wanted. This is Orca's iteration-level scheduling (Yu et al. 2022) in twenty lines.
`,
      predict: { question: 'Static and continuous batching run the same trace and do the same number of iterations in some cases. Which trace makes them identical, and why?', answer: 'Any trace where every request has the same outputLen, or where maxBatch is 1. Padding only exists when members of a batch finish at different times, and with one slot there is nothing to share. The gap grows with the spread of output lengths, which is why the tests use a distribution where 80% want 4 tokens and 20% want 120.' },
      hints: [
        'Delete the inner loop of `runStatic` and move admission and retirement inside the single remaining loop. What is the stopping condition now that batches no longer exist?',
        'Loop while there are unarrived requests or anything is running. Remove finished requests by walking `running` backwards so splicing does not skip an element.',
        '`while (i < arrivals.length || running.length) { admit…; if (!running.length) { clock = arrivals[i].arrival; continue; } clock += iterationSeconds(running, cfg); …; for (let k = running.length - 1; k >= 0; k--) if (running[k].generated >= running[k].outputLen) { records.push(…); running.splice(k, 1); } }`',
      ],
    },
    {
      id: 'allocator',
      title: 'The paged block allocator',
      instructions: `
The class skeleton is in the starter: \`numBlocks\` blocks of \`blockSize\` tokens, a \`freeList\` of block ids and a \`refs\` array of reference counts (you use \`refs\` in step 6; for now a live block has count 1). Fill in four methods and one function.

\`blocksNeeded(nTokens)\` — \`ceil(nTokens / blockSize)\`. Zero tokens need zero blocks; one token needs one whole block.

\`allocate(nTokens)\` — take that many blocks off the free list, set each count to 1, and return the **block table**, a plain array of block ids. If there are not enough, return \`null\` and **take nothing**: a half-allocated request leaks the blocks it did get, forever. Take blocks from the front of the free list so allocation is deterministic.

\`appendToken(table, lenBefore)\` — make room for one more KV entry in a table that currently holds \`lenBefore\` entries. If \`lenBefore < table.length · blockSize\` the block you already hold has room: return \`true\` and allocate nothing. Otherwise take one block, push it onto \`table\` and return \`true\` — or return \`false\` if the pool is empty, leaving \`table\` untouched so the caller can preempt someone and try again.

\`free(table)\` — drop one reference from each block; when a count reaches 0 push the block back on the free list. Throw on a block that is already free: a double free hands one block to two requests, and a silent one is a bug you will never find.

\`internalFragmentation(lengths, blockSize)\` — given the sequence lengths resident in the cache, return \`{ tokens, slots, wasted, fraction }\` where \`slots\` is \`sum(ceil(L / blockSize) · blockSize)\`. Run it with \`blockSize = 1024\` and you have modelled the contiguous-slab allocator vLLM replaced.
`,
      hints: [
        'The free list is a queue of block ids. Every method is a few lines; the care is all in the failure paths — what must be true of the allocator after a failed allocate?',
        'Check the count you need against `freeList.length` *before* you take anything. In `appendToken`, the table already tells you how many slots you own: `table.length * blockSize`. Compare that with `lenBefore`.',
        '`allocate`: `const need = this.blocksNeeded(nTokens); if (need > this.freeList.length) return null;` then shift `need` ids into a new array. `free`: `for (const b of table) { if (this.refs[b] <= 0) throw new Error(...); if (--this.refs[b] === 0) this.freeList.push(b); }`',
      ],
    },
    {
      id: 'paged',
      title: 'Admission control and preemption',
      instructions: `
\`runPaged(requests, cfg)\` — \`runContinuous\` with a real memory budget. Start from your step-3 loop and add an allocator built from \`cfg.numBlocks\` and \`cfg.blockSize\`.

**Guard first.** If any request needs more than \`numBlocks · blockSize\` KV slots (\`promptLen + outputLen − 1\`), throw: it can never run, and preempting the only running request to make room for itself would loop forever. Real engines reject such a request at admission.

**Queues.** Keep a \`waiting\` array. Each iteration, move every request whose \`arrival <= clock\` onto the end of it, then admit from the front while \`running.length < cfg.maxBatch\`: a request needs \`allocate(promptLen + generated)\` — enough for the KV its prefill will write — and if that returns \`null\`, **stop admitting**. Do not skip ahead to a smaller request: first-come-first-served is what keeps the head of the queue from starving.

**Growth and preemption.** Before costing the iteration, every prefilled running request needs room for one more entry: \`appendToken(s.blocks, kvTokens(s))\`. When that fails, preempt the **most recently admitted** running request (the last element of \`running\`): free its blocks, set \`prefilled = false\`, keep its \`generated\` count, increment \`restarts\`, and \`unshift\` it onto the front of \`waiting\` so it is readmitted first. If the victim is the request you were trying to grow, it simply leaves the batch; otherwise retry the append. This is preemption by **recomputation** — when it comes back it re-processes \`promptLen + generated\` tokens, which is what your step-1 \`tokensThisIteration\` already returns.

**Extra metrics.** Alongside the step-3 fields report \`preemptions\`, \`recomputedTokens\` (tokens processed by re-prefills of requests with \`generated > 0\`), \`peakBlocks\` (the maximum of \`usedCount\`) and \`wastedFraction\` (1 minus the ratio of total KV tokens to total allocated slots, summed over iterations).

**Re-prefill after preemption.** When a readmitted request's prefill iteration finishes, it emits its *next* token: \`generated++\`, not \`generated = 1\`, and \`firstToken\` keeps its original value — the user already has that token. Copying the step-3 update line unchanged gets both wrong; a test checks the exact timeline.

With \`numBlocks\` large this must reproduce \`runContinuous\` exactly, iteration for iteration — a test checks it.
`,
      hints: [
        'The only new question each iteration is "does everyone fit?", asked twice: once at admission and once for growth. Everything else is your step-3 loop.',
        'Handle growth in a `while (k < running.length)` loop rather than a `for` loop, because preempting removes an element and you may need to retry the same index. Each preemption shrinks `running`, so it terminates.',
        '`const victim = running[running.length - 1]; running.pop(); alloc.free(victim.blocks); victim.blocks = []; victim.prefilled = false; victim.restarts++; preemptions++; waiting.unshift(victim);` — then, if `victim` was the request at index `k`, do not advance `k` (the array is now shorter); otherwise retry the append at the same `k`.',
      ],
    },
    {
      id: 'prefix',
      title: 'Prefix sharing with copy-on-write',
      instructions: `
Two requests with the identical prompt compute the identical keys and values for it. Storing both is pure waste. This step shares only exact-match whole prompts; module 17 later turns it into a radix tree over blocks of tokens with eviction, so requests that share only a prefix can share it too. That is close to SGLang's RadixAttention; vLLM gets the same longest-prefix match from a hash table of chained block hashes.

First finish \`BlockAllocator.retain(blocks)\`: add one to each block's reference count, throwing if a block is not currently allocated. Your \`free\` from step 4 already returns a block only when its count reaches 0, so the two methods together are the whole mechanism.

\`allocateShared(alloc, cache, key, nTokens)\` — \`cache\` is a \`Map\` from a prompt key to the array of **full** blocks already resident for that prompt.

1. \`total = blocksNeeded(nTokens)\`, \`full = floor(nTokens / blockSize)\`. Only full blocks can be shared: the last, partially filled block will be written with each request's own generated tokens, so it is copied (copy-on-write).
2. \`shared\` is \`full\` if the key is cached, else 0. \`fresh = total − shared\`.
3. If \`fresh\` blocks do not fit, return \`null\` and change nothing — including the cache.
4. Otherwise \`retain\` the shared blocks, allocate the fresh ones, and return \`{ blocks, sharedBlocks, newBlocks }\` with \`blocks\` in order: shared blocks first, then fresh.
5. On a miss, store this prompt's first \`full\` blocks in the cache and \`retain\` them once more — the cache itself holds a reference, so the prefix survives after the request that created it has finished. A real engine bounds that with LRU eviction (module 17).

\`allocate(fresh * alloc.blockSize)\` asks for exactly \`fresh\` blocks, and \`allocate(0)\` returns an empty array.
`,
      hints: [
        'Work out the three counts — total, shareable, new — before you touch the allocator, so the out-of-memory case can return before anything has changed.',
        'A 40-token prompt with blockSize 16 is 3 blocks, of which 2 are full. The second request pays for 1 block instead of 3. A 32-token prompt is 2 full blocks and the second request pays for nothing at all.',
        '`const cached = cache.get(key); const shared = cached ? Math.min(cached.length, full) : 0; const fresh = total - shared; if (alloc.freeCount < fresh) return null;` then retain the shared blocks, allocate the fresh ones, and build the table shared-first. On a miss, the cache entry is a second owner of the first `full` blocks, so it needs its own reference.',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why continuous batching raises throughput without making a single matrix multiply faster. Use the numbers your demo printed.',
    'Your scheduler makes two decisions each iteration: who runs, and who keeps memory. Describe a workload where the first decision dominates the result, and one where the second does — and say which metric (throughput, p95 TTFT, TPOT) you would watch to tell them apart.',
    'Paging the KV cache into 16-token blocks makes attention read through a block table instead of a contiguous array — an indirection that costs something in the kernel. Explain why it pays for itself anyway, in terms of the batch size it buys.',
  ],
  stretch: [
    'Add chunked prefill: cap the tokens per iteration at `maxBatchedTokens` and split a long prompt across several iterations, so a 4,000-token prefill never stalls everyone\'s decode. Measure TPOT before and after. This is Sarathi-Serve (Agrawal et al. 2024); chunked prefill is now on by default in vLLM\'s V1 engine.',
    'Replace recompute preemption with swapping: pay a PCIe transfer for the victim\'s blocks (approximately 64 GB/s per direction over PCIe 5 x16) instead of re-prefilling them, and find the sequence length where each policy wins. The vLLM paper implements and compares both.',
    'Replace the exact-key prefix cache with a radix tree over token ids so requests that share only a prefix can share blocks, and add LRU eviction of the cached blocks. That is SGLang\'s RadixAttention, and it is where module 17 goes.',
    'Split the engine in two: one instance runs prefills, another runs decodes, and the KV blocks are transferred between them. Measure what that does to p95 TTFT and to TPOT separately — this is disaggregated serving (DistServe, Zhong et al. 2024), and it is in module 26.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
