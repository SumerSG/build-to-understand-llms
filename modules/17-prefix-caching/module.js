export default {
  id: '17-prefix-caching',
  title: 'Prefix caching & prompt caching',
  track: 'inference',
  minutes: 90,
  threshold: 'Two requests that begin with the same tokens have identical KV state for those tokens, so the serving system can store KV by prefix instead of by request — and the data structure that makes "longest shared prefix" cheap is a radix tree over blocks of tokens.',
  goal: 'A radix-tree prefix cache with hash-chained blocks, reference counting and LRU eviction, plus a traffic simulator and a prompt-caching cost model; you measure hit rate against cache capacity and turn that hit rate into a bill and a time-to-first-token.',
  prereqs: ['05-attention', '15-kv-cache', '16-batching'],
  recall: [
    { q: 'In causal self-attention (module 05), the key and value vectors for token `i` are computed from…',
      options: ['Every token in the sequence, including later ones', 'The hidden state at position `i`, which depends only on tokens `0..i`', 'Tokens `i` and `i+1`'], answer: 1,
      why: 'Each layer projects its own hidden state at position `i` into a key and a value, and under the causal mask that hidden state depends only on tokens `0..i`. This is exactly why a shared prefix has shared KV: nothing later can change it.' },
    { q: 'In module 15 you cached keys and values so that decoding step `t` did not recompute them. What made that cache valid?',
      options: ['The keys and values for earlier tokens never change once computed', 'The model is deterministic given a seed', 'Float32 rounding is stable'], answer: 0,
      why: 'Causal masking freezes the KV of every earlier position. Prefix caching is the same observation applied across requests instead of across steps.' },
    { q: 'Module 16 allocated KV memory in fixed-size blocks rather than one contiguous buffer per sequence. The main reason was…',
      options: ['Blocks are faster to index', 'Blocks stop internal fragmentation and let sequences share physical pages', 'Blocks reduce the number of FLOPs'], answer: 1,
      why: 'Sharing physical blocks between sequences is the mechanism prefix caching rides on: a cache hit is just pointing a new request at blocks that already exist.' },
    { q: 'Your BPE tokenizer from module 03 encodes the same string twice. The two token id arrays are…',
      options: ['Identical', 'Identical up to the first merge', 'Different, because merges are sampled'], answer: 0,
      why: 'Encoding is deterministic, so identical prompt text gives identical token ids — the precondition for hashing tokens as a cache key. Change the tokenizer and every cached entry is void.' },
  ],
  review: [
    { q: 'With a block size of 16 tokens, how many blocks of a 33-token prompt can be cached?',
      options: ['3', '2', '33'], answer: 1,
      why: 'Only complete blocks have a stable identity. Two full blocks cover tokens 0–31; token 32 sits in a partial block and is recomputed on every request.' },
    { q: 'Why is a block\'s hash chained to the hash of the block before it, rather than computed from the block\'s tokens alone?',
      options: ['To make hashes shorter', 'So that a hash match implies the whole prefix matches, not just that one block', 'To avoid collisions between tokens'], answer: 1,
      why: 'Without chaining, the same 16 tokens appearing in two different contexts would share a hash, and a request could be handed KV computed from a different prefix — silently wrong output.' },
    { q: 'Which node of the radix tree is eligible for eviction?',
      options: ['Any node whose lastUsed is oldest', 'A childless node with a reference count of zero', 'The root'], answer: 1,
      why: 'A parent is a prefix of its children, so evicting it first would orphan longer cached prefixes; a non-zero reference count means a running request is reading those KV blocks right now.' },
    { q: 'With a cache-write multiplier of 1.25x and a cache-read multiplier of 0.1x on the base input price, prompt caching starts saving money above a hit rate of about…',
      options: ['0%', '22%', '50%'], answer: 1,
      why: '(1.25 − 1) / (1.25 − 0.1) ≈ 0.217. Below that, the premium you pay writing blocks into the cache outweighs the discount you collect reading them back.' },
    { q: 'You must inject the current timestamp into every request. Where does it hurt the hit rate least?',
      options: ['At the very start of the system prompt', 'Immediately after the system prompt', 'At the end, after the user turn'], answer: 2,
      why: 'Caching matches prefixes. Anything that changes per request invalidates everything after it, so volatile content goes last — measured in the demo as 0% versus 78% hit rate on identical tokens.' },
  ],
  concept: `
## Two requests, one prefill

Serving a prompt costs two different things. **Prefill** runs the whole prompt through the model at once and is compute-bound; **decode** then emits one token at a time from the KV cache of module 15. A 4,000-token system prompt with a 20-token question spends almost all its money on prefill — and the next request carrying that system prompt redoes the work for an identical result.

Identical because of causality: in module 05 the key and value vectors for position \`i\` came from the hidden state at \`i\`, which depends only on tokens \`0..i\`. So if two requests agree on their first \`n\` tokens, their KV agrees exactly on those \`n\` positions, layer for layer. KV is a function of the prefix, so cache it by prefix.

:::predict
Two requests share a 4,000-token system prompt and differ in their last 20 tokens. What fraction of the second request's *prefill* can be skipped, and of its *decode*?
---
About 99.5% of the prefill (4,000 of 4,020 tokens) and none of the decode, which produces tokens whose KV nobody has computed. Prefix caching is a time-to-first-token optimisation, not a throughput-per-output-token one.
:::

## Blocks, and why hashes chain

Module 16 allocated KV in fixed-size blocks — 16 tokens each in vLLM's default. Prefix caching reuses that granularity: a block is shared entirely or not at all, and a partial block at the end of a prompt has no stable identity, so it is recomputed every time. A 33-token prompt has exactly two cacheable blocks.

vLLM's *automatic prefix caching* gives each block an identity with a hash chain: \`h_i = hash(h_{i-1}, tokens_of_block_i)\`, with \`h_{-1} = 0\`. The chain is the whole trick. Hashing a block's tokens alone would collide whenever the same 16 tokens appeared in two contexts, and the server would hand a request KV built from someone else's prefix. With chaining, matching \`h_i\` means the entire prefix through block \`i\` matched, so a hash-table lookup is a correct prefix test. Step 1 builds that table.

## The radix tree

A hash table answers "is this exact prefix cached?". A server wants the *longest* cached prefix of a prompt, and wants to see which prefixes are shared so it can evict the unpopular ones. A radix tree — a trie whose nodes hold runs of blocks rather than single blocks — answers both in one walk. When a request diverges inside a node you split it: the shared head stays one node and each branch gets a tail. This is the structure of SGLang's **RadixAttention** (Zheng et al., 2024), with one difference: SGLang's edges hold token ids (a page of one token by default), so it compares tokens directly and never hashes; yours holds block hashes, which is closer to how vLLM names blocks.

Eviction needs two rules. Only **childless** nodes may go, because a parent is a prefix of its children and freeing it would orphan a longer cached prefix. And a node with a non-zero **reference count** may never go, because a decoding request is reading those exact KV blocks; step 3 pins a path on \`acquire\` and unpins on \`release\`, four requests at a time — the continuous-batching window of module 16.

:::predict
Your simulator runs 500 requests across 50 interleaved chats with a capacity of 32 blocks — too small for even one conversation. Do you expect a hit rate near zero?
---
No: about 33%. Four requests are pinned in flight at any moment, so their shared system prompt is un-evictable and the hottest blocks survive by accident. Capacity then moves the hit rate smoothly to 86% at 2,048 blocks.
:::

## What the hit rate is worth

Hosted APIs expose the same mechanism as **prompt caching**: you mark a cache breakpoint, cached tokens are billed at a discount, tokens written into the cache carry a premium, entries expire after a short time-to-live, and prefixes below a minimum length are not cached at all. Step 5 models it with example multipliers — write at 1.25x the base input price, read at 0.1x — giving a break-even: caching pays only above a hit rate of \`(cacheWrite - base) / (cacheWrite - cacheRead)\`, about 22% here. Below that you pay a write premium for blocks nobody reads back.

The model in step 5 simplifies one thing: it bills every uncached prompt token at the write price. A real API writes only the tokens up to the breakpoint you mark and bills anything after it at the base price, so its cold requests cost a little less than yours.

Two rules fall out. **Put stable content first** — system prompt, tools, few-shot examples — and volatile content (timestamps, session ids, per-request documents) last; the demo measures 0% against 78% on identical tokens, ordered differently. And **any edit invalidates everything after it**: change one token in block 0 of a 96-token prompt and all six blocks are gone.

## What a cached entry is bound to

KV is not portable: it is bound to the exact weights, the numerical precision and any LoRA adapter, so a model update or a switch from bf16 to fp8 voids the whole cache. vLLM mixes extra keys such as the LoRA adapter id into each block hash for exactly this reason. A shared tree is also a fairness and privacy surface — one tenant's traffic evicts another's hot prefix, and the timing gap between hit and miss reveals whether someone else recently sent the same prompt. Production systems partition by tenant (vLLM accepts a per-request *cache salt* mixed into the first block's hash, so differently salted requests can never share a block) or accept that side channel deliberately.

## Where this toy differs from production

Yours stores a 32-bit number per block; a real cache stores \`2 x layers x kv_heads x head_dim\` numbers per token in GPU HBM, so capacity is gigabytes and eviction pressure is constant. A 32-bit hash also collides by the birthday bound after tens of thousands of distinct blocks, which a busy server passes in seconds; vLLM uses a much wider hash (SHA-256 is an option) so collisions are negligible rather than detected, and SGLang avoids the question by comparing tokens. vLLM does not build a tree at all: it keeps step 1's hash table plus a reference count per block and an LRU queue of free blocks. SGLang keeps the tree with a lock count per node, as you do. Both can offload evicted blocks to host memory rather than dropping them. Your simulator is serial: two requests never prefill at once and contend for free blocks. What transfers exactly is the shape — block granularity, chained identity, longest-prefix match, leaves-only LRU under reference counts, and hit rate to bill.
`,
  steps: [
    {
      id: 'blocks',
      title: 'Block hashes and an exact-prefix cache',
      instructions: `
Two pieces.

\`blockHashes(tokens, blockSize = BLOCK_SIZE)\`: the chained hash of every **complete** block of \`tokens\`, in order. Element \`i\` must depend on tokens \`0 .. (i+1)*blockSize - 1\` and nothing else. The worked helpers above the TODO line do the two hard parts for you: \`toBlocks\` drops the incomplete tail, and \`hashChain(prevHash, block)\` combines the running hash with one block. Start the chain at \`0\`.

\`HashCache\`: a flat cache holding one \`Map\` entry per cached block, keyed by that chained hash.

- \`insert(tokens)\` stores every block and returns how many of them were **not** already cached.
- \`lookup(tokens)\` returns \`{ blocks, tokens }\` for the longest cached prefix — walk the hashes from the start and **stop at the first miss**, because a cached block whose predecessor is missing is unusable.
- \`size()\` is already written.

This is vLLM's automatic prefix caching in miniature. It is enough to answer "is this exact prefix cached?", which is what step 2 then improves on.
`,
      predict: { question: 'You cache a 96-token prompt (six blocks). You then look up the same prompt with token 90 changed. How many blocks hit?', answer: 'Five. Token 90 is inside block 5 (tokens 80–95), so blocks 0–4 are untouched and only the last one misses. Change token 3 instead and you get zero: the chain carries the edit into every later block.' },
      hints: [
        'The chain is a fold: one running value that each block updates. What should that value be before the first block?',
        'For `blockHashes`: `let h = 0;` then for each block from `toBlocks`, set `h = hashChain(h, block)` and push the new `h`. For `lookup`: compute the hashes, then count how many leading entries `this.blocks.has(...)`.',
        '`lookup`: `const hashes = blockHashes(tokens, this.blockSize); let n = 0; while (/* n is in range and block n is cached */) n++;` then report `n` in blocks and in tokens. `insert` walks the same hashes without the early stop and counts the ones it had to `set`.',
      ],
    },
    {
      id: 'radix',
      title: 'The radix tree: longest-prefix match and node splitting',
      instructions: `
Fill in \`match\` and \`insert\` on \`PrefixTree\`. A node is \`{ key, children, parent, refs, lastUsed }\` (see \`makeNode\`), where \`key\` is a run of block hashes the node owns and \`children\` is a \`Map\` from the **first hash of a child's key** to that child. The root has an empty key. \`size()\`, \`nodeCount()\` and \`touch()\` are written for you.

\`match(hashes)\` returns \`{ blocks, node }\`: \`blocks\` is the length of the longest prefix of \`hashes\` present in the tree, and \`node\` is the deepest node fully covered by that match (the root if nothing matched). Note that \`blocks\` can run part-way into one more node. Call \`this.touch(node)\` before returning so the LRU clock in step 3 sees the access.

\`insert(hashes)\` inserts the whole sequence and returns the node at depth \`hashes.length\`. Walk down as \`match\` does. Three cases:

1. no child starts with \`hashes[i]\` — hang a new leaf holding \`hashes.slice(i)\` and add its length to \`this.blocks\`;
2. the child's key matches entirely — descend and continue;
3. the child's key matches for \`c\` blocks and then diverges — **split** it into a head of length \`c\` and a tail holding the rest, then continue from the head. The head covers part of the path the old node covered, so give it the child's \`refs\` and \`lastUsed\`. Step 3 walks every ancestor of a pinned node when it releases it, so a head created with \`refs = 0\` under a pinned path drops to -1 on release and can never be evicted.

Keep \`this.blocks\` equal to the total number of block hashes stored. A split moves blocks between nodes but does not change that total.
`,
      hints: [
        'Both functions share the same inner loop: look up `node.children.get(hashes[i])`, then count how many of the child\'s key entries agree with `hashes` starting at `i`.',
        'Let `c` be that count. If `c === child.key.length` you consumed the whole node: `node = child; i += c;` and loop. Otherwise you have diverged — for `match` you are done and the answer is `i + c`; for `insert` you must split at `c`.',
        'A split creates one new node and rewires three links: `const head = makeNode(child.key.slice(0, c), node); node.children.set(head.key[0], head); child.key = child.key.slice(c); child.parent = head; head.children.set(child.key[0], child);` — keep `child` as the *tail* so a handle someone else already holds still points at the deepest node of its path. Copy `refs` and `lastUsed` onto the head.',
      ],
    },
    {
      id: 'evict',
      title: 'Reference counting and LRU eviction',
      instructions: `
Three more methods, and the cache becomes bounded.

\`acquire(hashes)\`: insert the sequence, then increment \`refs\` on **every node from the end node up to the root** and \`touch\` the path. Return the end node. A caller holds this node while its request is decoding.

\`release(node)\`: decrement \`refs\` along the same root-ward walk.

\`evict(capacityBlocks)\`: while \`this.size() > capacityBlocks\`, find the evictable node with the smallest \`lastUsed\` and remove it, then repeat. Return the total blocks freed. A node is evictable only when it is **not the root**, has **no children**, and has \`refs === 0\`. If no such node exists, stop and return what you freed — an all-pinned tree must not loop forever.

Two invariants to hold on to. A parent is a prefix of its children, so evicting parents first would orphan longer cached prefixes; that is why only leaves go. And \`refs > 0\` means a running request is attending over those exact KV blocks, so evicting them would corrupt its output — vLLM and SGLang both refuse for the same reason.
`,
      predict: { question: 'A tree holds [1,2] (pinned by a running request) and [3,4] (idle and older). You call `evict(0)`. What happens?', answer: 'Two blocks are freed — the [3,4] leaf — and `evict` returns 2 with the tree still at 2 blocks, above its capacity of 0. A second call returns 0 rather than spinning: the cache is simply oversubscribed until the pinned request finishes.' },
      hints: [
        'To remove a node you need its parent and the key its parent filed it under. Which entry of `node.key` is that?',
        'Each round: walk the whole tree collecting candidates (`node !== this.root && node.children.size === 0 && node.refs === 0`), keep the one with the smallest `lastUsed`, then `victim.parent.children.delete(victim.key[0])` and subtract `victim.key.length` from `this.blocks`.',
        'The loop shape is `while (this.blocks > capacityBlocks) { const victim = /* LRU evictable leaf */; if (!victim) break; … }`. Reuse the stack traversal from `nodeCount()` to find candidates; the tree is small enough that rescanning each round is fine.',
      ],
    },
    {
      id: 'traffic',
      title: 'Realistic traffic and the hit-rate curve',
      instructions: `
\`makeWorkload(options)\` builds a deterministic request stream. Every conversation opens with the **same** \`systemTokens\` system prompt and one of \`fewShotVariants\` few-shot bundles of \`fewShotTokens\` each; then each turn appends \`userTokens\` of user text, and between turns the assistant's \`replyTokens\` are appended too. So turn \`t\`'s prompt is a strict token prefix of turn \`t+1\`'s. Return requests as \`{ id, conv, turn, tokens, outputTokens }\` in **arrival order**, with conversations **interleaved** — pick a random still-live conversation for each next request, so that a small cache is forced to evict between a conversation's turns. Use the worked \`randomTokens(next, n, vocab)\` helper and the single \`next = rng(seed)\` already in the file; that one generator is the only source of randomness. \`id\` is the arrival index (0, 1, 2, …), \`turn\` counts from 0 within a conversation, \`outputTokens\` is \`replyTokens\`, and each conversation draws its few-shot variant once, with \`randInt(next, fewShotVariants)\`. The tests check determinism for a fixed seed, the token counts per turn and the prefix property; they do not compare against one exact token stream, so the order of your draws is yours to choose.

\`simulate(requests, { capacityBlocks, blockSize, concurrency })\` replays the stream against one \`PrefixTree\`. For each request: hash it, \`match\` it (**before** inserting, or every request hits itself), \`acquire\` it and push the node onto an in-flight list, release the oldest while the list is longer than \`concurrency\`, then \`evict(capacityBlocks)\`. Release whatever is still in flight at the end, and do not evict again after that. Report \`sizeBlocks\` and \`nodes\` from the tree as it stands after that final release, so a tight cache can end well above \`capacityBlocks\`: the last \`concurrency\` requests were pinned when the final \`evict\` ran. Return the fields listed in the starter's stub, with \`hitRate = hitBlocks / totalBlocks\` and \`cachedTokens = hitBlocks * blockSize\`.
`,
      hints: [
        'Keep one growing `prompt` array per conversation. The request\'s tokens are `prompt ++ user`; afterwards the conversation\'s prompt becomes `prompt ++ user ++ reply`.',
        'For interleaving, hold an array of live conversations and repeatedly `randInt(next, live.length)` to choose one; splice a conversation out once it has produced all its turns. For `simulate`, an array used as a queue (`push` / `shift`) is all the in-flight bookkeeping you need.',
        'Per request, in pseudo-code: `hashes ← blockHashes(tokens, blockSize)`; `hit ← tree.match(hashes).blocks`; `queue.push(tree.acquire(hashes))`; while the queue is longer than `concurrency`, `release` its front; `evicted += tree.evict(capacityBlocks)`; add `hit`, `hashes.length` and `tokens.length` to the running totals. After the loop, release whatever is left.',
      ],
    },
    {
      id: 'cost',
      title: 'The cost and latency model',
      instructions: `
Turn a hit rate into money and milliseconds. All three functions are arithmetic; getting the *shape* right is the point.

\`requestCost(promptTokens, cachedTokens, outputTokens, pricing)\`: cached prompt tokens are billed at \`pricing.cacheRead\`, the remaining prompt tokens at \`pricing.cacheWrite\` (processing them also writes them into the cache), output tokens at \`pricing.output\`. Throw if \`cachedTokens > promptTokens\`. Notice what this says about a cold request: with nothing cached you pay **more** than the \`baselineCost\` worked example, which bills prompt tokens at \`pricing.base\`.

\`breakEvenHitRate(pricing)\`: solve \`h * cacheRead + (1 - h) * cacheWrite = base\` for \`h\`. Return 0 rather than a negative number when the cache is free to write. You may assume \`cacheWrite > cacheRead\`, as in every real price list; otherwise the equation has no solution.

\`ttftMs(promptTokens, cachedTokens, { prefillTokensPerSecond, overheadMs })\`: a fixed overhead plus the time to prefill only the tokens that were **not** served from cache. Time is in milliseconds and the rate is in tokens per second.
`,
      hints: [
        'Write the cost as a sum of three terms, one per token category, before simplifying anything.',
        'For break-even, expand: `h*read + write - h*write = base`, so `h*(read - write) = base - write`, so `h = (write - base) / (write - read)`. Clamp at 0.',
        'For `ttftMs`, the tokens still to prefill are `Math.max(0, promptTokens - cachedTokens)`; the prefill time in ms is that count divided by the rate, times 1000.',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why a cache hit on a prefix is safe — what property of causal attention guarantees that the cached keys and values are exactly the ones this new request would have computed?',
    'Your cache reached a high hit rate on this traffic. Describe a realistic workload where a prefix cache is nearly useless, and say what about it defeats the mechanism.',
    'You are asked to add "the current time is HH:MM" to a product\'s system prompt. Write the two-sentence reply you would send, using a number from your own run.',
  ],
  stretch: [
    'Add block-level offload: instead of dropping an evicted leaf, move it to a slower second tier with a fixed transfer cost, and measure whether the recovered hits pay for the transfers. vLLM\'s CPU offloading and LMCache do exactly this.',
    'Partition the tree per tenant and compare total hit rate against one shared tree on traffic where one tenant sends 90% of the requests. This is the fairness question SGLang and vLLM both face in multi-tenant deployments.',
    'Replace LRU with LFU or with a "keep the deepest shared subtree" policy that scores a node by how many leaves hang below it, and see which wins on the demo traffic. SGLang\'s scheduler goes further and *reorders* the queue so that requests sharing a prefix run together.',
    'Add prefix-aware routing across N replicas: hash the first few blocks of a prompt to pick a replica, so the same prefix lands on the same machine. Measure aggregate hit rate against round-robin. Module 26 builds this at cluster scale, and it is what production vLLM and SGLang routers do.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
