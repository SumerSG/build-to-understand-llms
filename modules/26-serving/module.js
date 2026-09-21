export default {
  id: '26-serving',
  title: 'Serving at scale',
  track: 'systems',
  minutes: 105,
  threshold: 'At cluster scale the wins are routing and pooling: send each request to the replica that already holds its prefix, and stop making compute-bound prefill share a GPU with memory-bound decode.',
  goal: 'A cluster simulator with prefix-aware routing, disaggregated prefill/decode pools, a queue-depth autoscaler and an SLO report, which meets TTFT and TPOT SLOs on traffic that a round-robin cluster of the same size fails.',
  prereqs: ['16-batching', '17-prefix-caching', '25-cluster'],
  recall: [
    {
      q: 'In module 16, what does a continuous-batching scheduler decide at every single iteration?',
      options: ['How many tokens each request may generate', 'Which requests are in the batch right now', 'Which GPU runs the model'],
      answer: 1,
      why: 'Iteration-level scheduling (Orca) is the engine inside one replica. This module never changes that decision; it decides which engine the request reaches in the first place.',
    },
    {
      q: 'A prefix-cache hit (module 17) saves work in which phase?',
      options: ['Prefill only', 'Decode only', 'Both equally'],
      answer: 0,
      why: 'The hit supplies keys and values for tokens that would otherwise be recomputed by prefill. Decode still costs one iteration per token, which is why a cache hit moves TTFT and barely moves TPOT.',
    },
    {
      q: 'Module 23 priced a kernel as `max(flops / peakFlops, bytes / bandwidth)`. Which side is a single decode step of an 8B model on?',
      options: ['Compute-bound', 'Memory-bound: it reads all the weights to produce one token', 'Neither, it is latency-bound on the PCIe bus'],
      answer: 1,
      why: 'Reading 16 GB of bf16 weights at approximately 3.35 TB/s (NVIDIA H100 datasheet) costs about 4.8 ms whatever the batch size. That is exactly the `tFixed` term of the cost model you are about to use.',
    },
    {
      q: 'Module 24 priced one message as `latency + bytes / bandwidth`. What does that model say about moving a KV cache between two machines?',
      options: ['It is free if the link is fast enough', 'There is a fixed setup cost plus a term linear in the bytes moved', 'The cost depends only on the number of messages'],
      answer: 1,
      why: 'You will reuse exactly this alpha-beta form for `kvTransferSeconds`: a setup cost plus `tokens * bytesPerToken / bandwidth`.',
    },
    {
      q: 'From module 15: roughly how much KV cache does one token of Llama-3-8B occupy in bf16?',
      options: ['Approximately 2 KB', 'Approximately 128 KB', 'Approximately 8 MB'],
      answer: 1,
      why: '32 layers x 8 KV heads x 128 dims x 2 (K and V) x 2 bytes = 131,072 bytes. That number decides whether moving a KV cache between pools is cheap or ruinous.',
    },
  ],
  review: [
    {
      q: 'Why does a cluster of 8 replicas with cache-aware routing get a higher prefix-cache hit rate than the same 8 replicas with round-robin?',
      options: ['It has more cache memory', 'Each prefix lands on one or two replicas instead of all eight, so each replica\'s LRU holds fewer distinct prefixes', 'Round-robin evicts the cache on purpose'],
      answer: 1,
      why: 'Total cache capacity is identical. Round-robin spreads every prefix over every replica, so each LRU thrashes over all of them; affinity concentrates the working set.',
    },
    {
      q: 'Why does a hash ring beat `hash(key) % n` for routing by prefix?',
      options: ['It is faster to compute', 'When n changes, the ring moves only the keys that lived on the replica that appeared or disappeared', 'It guarantees a perfectly even split'],
      answer: 1,
      why: 'Going from 8 replicas to 9, modulo remaps about 8 keys in 9; a ring remaps about 1 in 9. With an autoscaler changing n every few seconds, modulo would throw away the caches on every scaling event.',
    },
    {
      q: 'In a colocated replica, what does one 2,600-token prefill step do to the requests already decoding there?',
      options: ['Nothing; they run on other streaming multiprocessors', 'They all wait about 135 ms for their next token', 'They are evicted and restarted'],
      answer: 1,
      why: 'One iteration is one step for the whole replica. The prefill occupies it, so every in-flight decode sees a 135 ms gap. That single fact is the argument for disaggregation (DistServe, Splitwise) and for chunked prefill.',
    },
    {
      q: 'Goodput, as defined by DistServe (Zhong et al. 2024), is…',
      options: ['Tokens per second', 'Requests per second that meet every latency SLO', 'GPU utilisation'],
      answer: 1,
      why: 'A cluster can have excellent throughput while half its users see a 3-second first token. Goodput refuses to count work that failed the contract, which is why it is the metric the whole module optimises.',
    },
    {
      q: 'Why does the autoscaler in this module scale up immediately but scale down only when a whole window of ticks agrees?',
      options: ['To save on API calls', 'Because a replica takes approximately 30 s to become useful, so an early scale-down is expensive and a late scale-up is worse', 'Because scaling down is not allowed while requests are running'],
      answer: 1,
      why: 'The control loop has dead time equal to the cold start. Asymmetric hysteresis (Kubernetes HPA calls it downscale stabilisation) is what stops a sawtooth from forming around that delay.',
    },
  ],
  concept: `
## One machine is an engine; many machines are a placement problem

In module 16 you built the engine: a scheduler that decides, at every iteration, which requests share one GPU. This module is the layer above it. You have \`R\` replicas, each running that engine with its own prefix cache (module 17), and a load balancer in front of them. Nothing here changes what a replica does inside an iteration. Everything here changes **which replica a request reaches** and **what kind of work that replica is allowed to do**.

The cost model is the one from module 16, and it comes straight from the roofline of module 23: one iteration takes \`tFixed + tPerToken * (tokens in the batch)\`, with \`tFixed\` = 5 ms (reading 16 GB of bf16 weights at approximately 3.35 TB/s, NVIDIA H100 datasheet) and \`tPerToken\` = 50 us. A decode step for 32 requests moves 32 tokens and costs 6.6 ms. A prefill step for one 2,600-token prompt moves 2,600 tokens and costs 135 ms — twenty times as long, on the same replica, in the same queue.

## Decision 1: send the request to the replica that already has its prefix

Prefix caches are **per replica**. A cache hit is worth the entire prefix: 2,600 tokens of prefill that you do not pay for. But the hit only happens if the request lands on the replica that served the same prefix before, and round-robin guarantees that it usually does not. With 16 system prompts, 8 replicas and room for 4 prefixes each, round-robin makes every replica see all 16, so every LRU thrashes.

:::predict
Total cache capacity is the same either way: 8 replicas x 4 prefix slots = 32 slots for 16 prefixes. So why does affinity routing raise the hit rate at all?
---
Because capacity is not the constraint, *locality* is. Round-robin spreads each prefix across all 8 replicas, so each replica's 4 slots must cover all 16 prefixes and evict constantly. Affinity sends each prefix to one or two replicas, so each replica covers 2-4 prefixes and keeps them. In the goal demo this takes the hit rate from approximately 45% to approximately 75%, and median TTFT from 0.15 s to 0.04 s, with no extra hardware.
:::

This is what the SGLang router and the vLLM production stack router do: pick the replica with the longest cached prefix match, unless that replica is much busier than the others, in which case fall back to the least loaded. The escape valve matters — pure affinity turns the popular prefix's replica into everyone's queue. Your \`overloadFactor\` of 1.5 is that valve.

For a prefix nobody has served yet there is nothing to follow, so you need a deterministic *home*: a **hash ring**. Each replica gets 64 virtual nodes at \`hash32(id + '#' + v)\`; a key goes to the first point at or after \`hash32(key)\`. The reason for the ring rather than \`hash(key) % n\` is the autoscaler: modulo remaps almost every key when \`n\` changes, and every remapped key is a cold cache.

## Decision 2: stop making prefill and decode share a GPU

Prefill is compute-bound and decode is memory-bound, and putting them in one queue makes each one the other's tail latency. On a colocated replica, one 135 ms prefill step freezes every decode in flight: the user who was streaming at 100 tokens/s sees a 135 ms gap. TTFT and TPOT are separate SLOs with separate victims, and colocation makes them fight.

Disaggregation gives each phase its own pool. A request is prefilled in the prefill pool, its KV cache is shipped to a decode replica, and it streams there. This is DistServe (Zhong et al. 2024), Splitwise (Patel et al. 2024) and Mooncake (Qin et al. 2024), which serves Kimi this way with a shared KV store across the fleet. The price is the transfer, and it is the alpha-beta model of module 24: \`setup + tokens * bytesPerToken / bandwidth\`.

:::predict
A 4,000-token prompt on Llama-3-8B: its KV cache is 4,000 x 128 KB = 512 MB. Over a 25 GB/s RDMA link that is about 20 ms. Is shipping it cheaper than recomputing it on the decode replica?
---
Much cheaper. Recomputing means prefilling 4,000 tokens again: \`2 * 8e9 * 4000\` = 64 TFLOP, about 160 ms at 40% of an H100's approximately 989 TFLOP/s bf16 peak. Transfer wins by roughly 8x, and the gap widens with prompt length because both scale linearly but the constants differ. This is also why Mooncake keeps KV in a pool and reuses it across requests.
:::

## Goodput, admission control and the bill

Throughput hides failure: a cluster can move plenty of tokens per second while a third of its users wait three seconds for the first one. **Goodput** — requests per second that meet *every* SLO — refuses to count work that broke the contract, and it is the metric to optimise (Zhong et al. 2024). Tail matters more than mean here: p95 and p99 TTFT are what a user actually experiences on a bad day. Past saturation the honest move is **admission control**: reject or queue new requests rather than let everyone miss the SLO, which is what \`maxQueue\` backpressure in your driver does in miniature.

Capacity itself is a control problem. You scale on queue depth, but a replica takes approximately 30-60 s to pull weights and warm up, so the loop has dead time and needs asymmetric hysteresis: up at once, down only when a whole window agrees (Kubernetes HPA calls this downscale stabilisation). The end metric is neither latency nor GPUs but **dollars per million output tokens** = \`GPU-seconds / 3600 * $ per GPU-hour / (tokens / 1e6)\`.

One paragraph on the case where a fleet serves many models: with LoRA adapters, hundreds of fine-tunes share one base model's weights and differ by a few tens of megabytes, so a single replica can multiplex them (S-LoRA, Punica) and the router must be adapter-aware as well as prefix-aware — an adapter miss costs a load, not a recompute. For fully separate models the routing problem becomes a bin-packing problem over GPUs, and idle models get evicted the way idle prefixes do.

## Where this toy differs from production

The cost model is linear and the GPU is a single number: no kernels, no paging, no KV capacity limit (module 16 modelled that one), no chunked prefill, no speculative decoding, no quantisation. Prefixes are opaque strings standing in for "the first 64 token ids", where a real router hashes token blocks or walks a radix tree. The network is uncontended and every transfer gets the full 25 GB/s. The autoscaler sees perfect instantaneous metrics; real ones see a 15-60 s scrape delay on top of the cold start. There are no failures, no multi-tenancy, no priorities, and one model. Read the numbers it prints as comparisons between configurations, never as capacity plans.
`,
  steps: [
    {
      id: 'cost',
      title: 'The replica cost model',
      instructions: `
The driver \`runCluster\` is written for you. It needs four numbers from you.

\`tokensThisIteration(s)\` is **done** as a worked example: a request that has not been prefilled contributes its uncached prompt (\`promptLen - cached\`), and a request that is decoding contributes exactly 1 token. Read it; the other three follow the same shape.

\`iterationSeconds(batch, cfg)\`: seconds for one iteration over an array of request states — \`cfg.tFixed\` once, plus \`cfg.tPerToken\` for every token in the batch. An empty batch still costs \`tFixed\`.

\`remainingWork(s)\`: tokens of work this request still owes — its uncached prompt tokens if it has not been prefilled yet, plus \`outputLen - generated\` tokens it has still to produce.

\`replicaLoad(r)\`: the sum of \`remainingWork\` over \`r.queue\` and \`r.running\`. This one number is the signal every router in step 2 balances on, so it must count work *owed*, not work done.
`,
      predict: {
        question: 'One replica, `maxBatch` 32. Which costs more: one iteration that decodes 32 requests, or one iteration that prefills a single 2,600-token prompt?',
        answer: 'The prefill, by about 20x: 5 ms + 32 x 50 us = 6.6 ms against 5 ms + 2,600 x 50 us = 135 ms. Every decode sharing that replica waits the full 135 ms for its next token, which is the entire argument of step 3.',
      },
      hints: [
        'Every one of these is a loop that adds up `tokensThisIteration` or its parts. None is longer than four lines.',
        'For `remainingWork`, ask what the replica has left to compute for this request: prefill tokens only if `prefilled` is false, plus the output tokens not yet generated. A finished request owes 0.',
        '`iterationSeconds`: `let tokens = 0; for (const s of batch) tokens += tokensThisIteration(s); return cfg.tFixed + cfg.tPerToken * tokens;` — and `replicaLoad` is the same shape over `r.queue` then `r.running`.',
      ],
    },
    {
      id: 'routing',
      title: 'Where does this request go?',
      instructions: `
Three functions. \`candidates\` is always an array of replica objects the balancer may use right now (ready, not draining, right role), and \`chooseReplica\` returns an **index into that array**, not a replica id.

\`buildRing(ids, { vnodes })\`: a consistent-hashing ring. For each replica id, place \`vnodes\` points at \`hash32(\\\`\${id}#\${v}\\\`)\` for \`v\` in \`0..vnodes-1\`, and return them as \`[{ hash, id }, ...]\` sorted by \`hash\` ascending. Throw if \`ids\` is empty.

\`pickOnRing(ring, key)\`: return the \`id\` of the first ring point whose hash is \`>= hash32(String(key))\`, wrapping to \`ring[0]\` if there is none. Binary search, since the ring is sorted. Throw on an empty ring.

\`chooseReplica(policy, candidates, req, ctx)\` with \`ctx = { dispatched, ring, cfg }\`:

- \`'round-robin'\`: \`ctx.dispatched % candidates.length\`, ignoring load entirely.
- \`'least-loaded'\`: the smallest \`replicaLoad\`, first index on ties.
- \`'cache-aware'\`: among candidates whose \`cache\` Map already has \`req.prefix\`, take the least loaded. Keep it **unless** its load exceeds \`cfg.overloadFactor\` times the mean load of the candidates. If no candidate holds the prefix (or the holder is overloaded), try the ring home \`pickOnRing(ctx.ring, req.prefix)\` under the same overload test, and otherwise fall back to least-loaded.

Throw on an empty candidate list and on an unknown policy name: a balancer that silently invents a destination is worse than one that stops.
`,
      hints: [
        'Compute the loads once into an array at the top and reuse it; both the mean and the argmin come from that array.',
        'The cache-aware policy is three ordered attempts with the same guard: a warm replica, then the ring home, then least-loaded. The guard is `load <= cfg.overloadFactor * mean`.',
        'Ring lookup: `let lo = 0, hi = ring.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (ring[mid].hash < h) lo = mid + 1; else hi = mid; } return ring[lo % ring.length].id;`',
      ],
    },
    {
      id: 'disagg',
      title: 'Two pools: prefill and decode',
      instructions: `
When \`runCluster\` is given \`{ disaggregate: true, pools }\`, a request is routed to the **prefill pool**, runs one prefill iteration there (which produces its first token), is then shipped to a **decode replica**, and streams the rest there. You supply the price of the shipping and the split of the fleet.

\`kvTransferSeconds(tokens, cfg)\`: \`cfg.kvSetupSeconds + tokens * cfg.kvBytesPerToken / cfg.kvBandwidth\`. With the defaults that is 0.5 ms plus 128 KB per token at 25 GB/s. Throw if \`tokens\` is negative.

\`planPools(requests, nReplicas, cfg)\`: return \`{ prefill, decode, prefillWork, decodeWork }\`. Estimate the seconds of work each phase implies over the whole trace — prefill is \`promptLen * cfg.tPerToken\` per request; decode is \`(outputLen - 1)\` iterations, each costing \`cfg.tPerToken\` for the token plus its share \`cfg.tFixed / cfg.maxBatch\` of the fixed cost — then split the replicas in that proportion, rounding to whole machines. Neither pool may be empty, the two must sum to \`nReplicas\`, and fewer than 2 replicas must throw.

Once both work, the step's third test compares a colocated run against a disaggregated one on the same trace: \`stallSeconds\` (decode time lost to prefill steps) must drop to exactly zero, and p95 TPOT with it.
`,
      predict: {
        question: 'The demo trace is 93% prefill work by these estimates, so `planPools` puts 3 of 4 replicas in the prefill pool. What does the single decode replica do that the 3 prefill replicas cannot?',
        answer: 'It runs decode iterations that are never interrupted. 2,600 requests x 57 remaining tokens is only about 150k tokens of decode, which one replica clears at roughly 4,800 tokens/s — the scarce resource is not decode capacity but an *uninterrupted* decode stream. That is why p95 TPOT falls from 16.5 ms to 6.7 ms while nothing about the hardware changed.',
      },
      hints: [
        '`kvTransferSeconds` is one line and mirrors `commTime` from module 24: a fixed alpha plus bytes over bandwidth.',
        'For `planPools`, accumulate the two work totals in one pass over `requests`, then `Math.round(prefillWork / (prefillWork + decodeWork) * nReplicas)`.',
        'Clamp after rounding: `prefill = Math.max(1, Math.min(nReplicas - 1, prefill))`, then `decode = nReplicas - prefill`. Handle the degenerate zero-work trace so you never return `NaN`.',
      ],
    },
    {
      id: 'autoscale',
      title: 'A controller with a 30-second dead time',
      instructions: `
Every \`cfg.scaleIntervalSeconds\` the driver takes a snapshot \`{ pendingTokens, ready, total }\` — \`pendingTokens\` is all outstanding work in the cluster, \`total\` counts replicas alive including ones still booting — and asks you twice.

\`queueTarget(snap, cfg)\`: the raw signal. \`Math.ceil(pendingTokens / cfg.targetQueueTokens)\`, clamped into \`[cfg.minReplicas, cfg.maxReplicas]\`. Round up: a partial replica serves nobody.

\`autoscaleTarget(snap, cfg)\` where \`snap = { total, raw, window }\` and \`window\` holds the raw targets of the last \`cfg.stabilizationTicks\` ticks, most recent last:

- if \`raw >= total\`, return \`raw\` — scale up immediately, a backlog is an emergency;
- otherwise take the largest value in the window; if that is still \`>= total\`, hold at \`total\`;
- otherwise shrink by at most \`cfg.scaleDownStep\` replicas: \`Math.max(stabilized, total - cfg.scaleDownStep)\`.

\`gpuSeconds(spans, endTime)\`: sum \`end - start\` over spans, treating \`end === null\` as \`endTime\` and ignoring spans of zero or negative length. A replica is billed from the moment it is created, including the approximately 30 s it spends loading weights and serving nobody.
`,
      hints: [
        'All three are short. The only subtlety in `autoscaleTarget` is that three different values can come out: the raw target, the current total, and one step down.',
        'Write the three branches in the order given. The window is a plain array; its maximum is what decides whether a scale-down is allowed at all.',
        'Test the middle branch by hand: `total = 8, raw = 2, window = [8, 7, 2, 2]` must return 8 — 20 seconds ago the cluster needed all 8 and a replica you kill now costs 30 s to get back.',
      ],
    },
    {
      id: 'slo',
      title: 'Goodput, attainment and the bill',
      instructions: `
The report is what the whole simulator exists to produce. \`percentile(values, p)\` is given.

\`sloReport(records, slo)\` over finished records \`{ arrival, firstToken, end, outputLen, ... }\`, with \`slo = { ttft, tpot }\` in seconds. For each record, \`TTFT = firstToken - arrival\` and, when \`outputLen > 1\`, \`TPOT = (end - firstToken) / (outputLen - 1)\` (a one-token answer has no inter-token interval, so it contributes to TTFT statistics only). Return:

\`\`\`
{ n, ttftP50, ttftP95, tpotP50, tpotP95, met, attainment, goodput, throughput, outputTokens, makespan }
\`\`\`

\`met\` counts records meeting **both** SLOs, \`attainment\` is \`met / n\`, \`makespan\` is the last completion minus the first arrival, \`goodput\` is \`met / makespan\` requests per second, and \`throughput\` is \`outputTokens / makespan\`. Empty input returns zeros, never \`NaN\` — the demo calls this once per 10-second window and some windows are empty.

\`costPerMillionTokens(gpuSec, outputTokens, dollarsPerGpuHour)\`: \`(gpuSec / 3600) * dollarsPerGpuHour / (outputTokens / 1e6)\`. Throw on zero tokens rather than returning \`Infinity\`.
`,
      hints: [
        'One pass over `records` can collect the TTFT list, the TPOT list, the met count, the token total and the makespan bounds.',
        'Keep the two SLO checks together in that pass: a record counts only if it passed both, which is the entire difference between goodput and throughput.',
        'Guard the empty case at the top and return the zero-filled object, otherwise `Math.max()` over an empty list gives `-Infinity` and the window plot fills with holes.',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why cache-aware routing raises the hit rate when it adds no cache memory at all, and what single number in your router stops it from overloading the popular replica.',
    'Your disaggregated run reached 100% SLO attainment on the same 4 GPUs that the round-robin run failed on. Walk through what a request does in each configuration, in order, and name the step where the colocated one loses.',
    'The autoscaler cost more GPU-seconds than the fixed cluster and attained less. Under what traffic shape would that verdict flip, and which of its constants (cold start, tick interval, stabilisation window, target queue depth) would you change first?',
  ],
  stretch: [
    'Add chunked prefill: cap a prefill at `maxBatchTokens` per iteration and let the remainder mix with decoding requests in the same batch. This is what vLLM does by default now, and it is the alternative to disaggregation — compare the two on the same trace.',
    'Scale the two pools independently, each with its own queue-depth signal, as DistServe and Splitwise do, and see whether the planner beats the static `planPools` split under the diurnal hump.',
    'Give each replica a KV-block budget as in module 16, so the decode pool can run out of memory and preempt; then measure whether disaggregation still wins when the decode pool is memory-limited rather than compute-limited.',
    'Make the router adapter-aware: give each request a LoRA adapter id as well as a prefix, charge an adapter load on a miss, and route on both keys — the problem S-LoRA and Punica solve when one base model serves hundreds of fine-tunes.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
