export default {
  id: '25-cluster',
  title: 'Nodes, interconnects & clusters',
  track: 'systems',
  minutes: 105,
  threshold: 'A cluster is not a pile of GPUs but a bandwidth hierarchy — memory, then NVLink, then the network — and the fastest parallel layout is the one that puts the chattiest communication on the fastest link.',
  goal: 'A topology-aware placement planner (NVLink inside a node, InfiniBand across nodes, alpha–beta collective cost model, MoE all-to-all) that picks the fastest parallel layout.',
  prereqs: ['28-moe', '23-gpu-roofline', '24-parallelism'],
  recall: [
    {
      q: 'Module 24 priced a ring all-reduce of `B` bytes over `n` GPUs. How many bytes does each GPU send?',
      options: ['`B`', '`2(n−1)/n · B`', '`n · B`'],
      answer: 1,
      why: 'n−1 reduce-scatter steps plus n−1 all-gather steps, each moving one chunk of B/n. This module reuses that formula unchanged and only changes which link you charge it to.',
    },
    {
      q: 'In module 24, why was tensor parallelism kept inside one node?',
      options: ['It needs more memory than the other splits', 'It all-reduces the activation four times per layer, so it needs the fastest link in the machine', 'The optimizer state cannot cross a node boundary'],
      answer: 1,
      why: 'Four all-reduces per layer times 80 layers times every micro-batch. Here you will compute the exact penalty for letting that group straddle a node boundary, instead of taking it on trust.',
    },
    {
      q: 'Module 23 priced a kernel as `max(flops / peakFlops, bytes / bandwidth)`. An all-reduce does one addition per element. Which term wins?',
      options: ['The FLOP term', 'The byte term, by a very large margin', 'They are usually balanced'],
      answer: 1,
      why: 'A collective has arithmetic intensity near zero, so its cost is entirely traffic. That is why every function in this module counts bytes and bandwidth and ignores the arithmetic inside the reduction.',
    },
    {
      q: 'From module 08, how many FLOPs does training cost per parameter per token?',
      options: ['2', '6', '12'],
      answer: 1,
      why: 'Approximately 2 forward and 4 backward. Every step time here starts as `6·params·tokens / (peakFlops · mfu)` divided by the GPU count; the interesting part is what gets added on top.',
    },
    {
      q: 'In module 24, what did ZeRO stage 1 shard across the data-parallel ranks?',
      options: ['The parameters', 'The gradients', 'The optimizer state (the 12 bytes per parameter)'],
      answer: 2,
      why: 'The planner here uses exactly that memory model — bf16 weights and gradients replicated over the tensor and pipeline shards, fp32 AdamW state divided by dp — to decide which layouts fit in 80 GB.',
    },
  ],
  review: [
    {
      q: 'Ordered fastest to slowest, the four rungs of the hierarchy in this module are…',
      options: ['NVLink, HBM, InfiniBand leaf, spine', 'HBM, NVLink, InfiniBand leaf, spine', 'HBM, InfiniBand leaf, NVLink, spine'],
      answer: 1,
      why: 'Approximately 3.35 TB/s of HBM3 on the package, ~450 GB/s per direction over NVLink 4, ~50 GB/s over one NDR 400 port, and less than that once the spine is oversubscribed. The first three rungs are each roughly an order of magnitude apart.',
    },
    {
      q: 'A hierarchical all-reduce over 16 nodes of 8 GPUs puts how much of a `B`-byte buffer on the inter-node link?',
      options: ['`B`', '`B/8`', '`B/128`'],
      answer: 1,
      why: 'The in-node reduce-scatter leaves each GPU owning 1/8 of the buffer, and only those chunks cross the network. Dividing the slow-link traffic by the GPUs per node is the entire reason the schedule has two levels.',
    },
    {
      q: 'Under the rank order `rank = (ppIndex·dp + dpIndex)·tp + tpIndex`, a tensor-parallel group is…',
      options: ['`tp` consecutive ranks', '`tp` ranks spaced `dp` apart', '`tp` ranks spaced `pp` apart'],
      answer: 0,
      why: 'Consecutive, so with tp ≤ 8 the group lands inside one node and its four all-reduces per layer stay on NVLink. The data-parallel group is spaced `tp` apart and is allowed to cross the network.',
    },
    {
      q: 'Your planner picks tp = 8 over tp = 16 for a 70B model on 128 H100s mainly because…',
      options: ['tp = 16 does not fit in 80 GB', 'tp = 16 makes a group of 16 straddle two nodes, so every tensor-parallel all-reduce drops to InfiniBand', 'tp = 16 leaves too few data-parallel replicas'],
      answer: 1,
      why: 'tp = 16 uses LESS memory per GPU, and still loses by about a third of the step time. The only thing that changed is which link the chattiest collective runs on.',
    },
    {
      q: 'Node-limited routing in a mixture of experts bounds…',
      options: ['How many experts a token uses', 'How many distinct nodes a token\'s experts may live on', 'How many tokens an expert accepts'],
      answer: 1,
      why: 'DeepSeek-V3 keeps top-8 routing but forbids the router from spreading those 8 experts over more than 4 nodes, because the inter-node volume scales with the node count, not with topK. The per-expert cap is the capacity factor, a different knob.',
    },
  ],
  concept: `
## The machine is a hierarchy, not a pile

Buy 128 H100s and you do not get 128 equal peers. You get 16 boxes. Inside one box — an NVIDIA HGX or DGX H100 baseboard — eight SXM modules hang off four NVSwitches, and any GPU can reach any other at approximately 900 GB/s bidirectional, so roughly 450 GB/s in each direction (NVIDIA's H100 datasheet). The box also holds eight ConnectX-7 InfiniBand NDR network cards, one per GPU, each carrying 400 Gb/s, which is 50 GB/s (NVIDIA's DGX H100 and ConnectX-7 datasheets). For comparison, the PCIe Gen5 x16 slot the GPU sits in carries approximately 64 GB/s each way, so NVLink is roughly 7× PCIe. Those cards are how the box talks to the other fifteen.

Write the four numbers in a column, with one newer rung for comparison, and the shape of every decision in this module appears:

| rung | approximate one-way bandwidth | how far |
|------|------------------------------|---------|
| HBM3 on the package | 3,350 GB/s | inside one GPU |
| NVLink 4 through NVSwitch | 450 GB/s | inside one node (8 GPUs) |
| NVLink 5 through NVLink Switch (GB200 NVL72, not in our model) | 900 GB/s (1.8 TB/s bidirectional) | inside one 72-GPU rack |
| InfiniBand NDR 400, one leaf hop | 50 GB/s | inside one pod |
| the same port through a spine we make 2:1 oversubscribed | 25 GB/s | anywhere |

The three H100 rungs are each roughly an order of magnitude apart (3,350 → 450 → 50); the spine row is our modelling choice, not a datasheet number. Large training fabrics are often built non-blocking, but real ones still lose bandwidth to congestion once traffic leaves a leaf switch, and halving it makes the planner's choices visible. Fabrics are built **rail-optimised**: GPU *k* of every node connects to leaf switch *k* (NVIDIA's DGX SuperPOD reference architecture groups 32 nodes per such unit), so a collective in which every rank talks to the same slot on other nodes stays on one rail and does not need the spine. Our model collapses that into two inter-node tiers, "same pod" and "anywhere".

The ladder is not frozen at H100. NVIDIA lists its Blackwell GPUs (B200 and GB200, shipping in volume from 2025) at approximately 2.25 to 2.5 PFLOP/s dense bf16 and 8 TB/s of HBM3e each, a bit more than double the H100 on both counts, so module 23's ridge point barely moves: \`2.25e15 / 8e12 ≈ 281\` FLOP/byte against 295. The bigger change is the NVLink rung. A GB200 NVL72 rack joins 72 GPUs through NVLink Switch trays at approximately 1.8 TB/s bidirectional per GPU (NVIDIA's GB200 NVL72 page), so the fast domain becomes a rack instead of a box. Every constant in your code stays H100; the method carries over unchanged.

:::predict
An all-reduce of a 16 GB gradient buffer over 128 GPUs. Schedule A rings through all 128 ranks on the inter-node link. Schedule B reduce-scatters inside each node over NVLink, all-reduces the resulting chunks across the 16 nodes, then all-gathers inside each node. How much faster is B?
---
About 6× with this module's constants (213 ms against 1,276 ms). In B each GPU puts only \`bytes/8\` on the slow link — the chunk it owns after the in-node reduce-scatter — and the ring across nodes is 16 ranks instead of 128. The eight GPUs of a node run eight such cross-node rings at the same time, one per slot, each on its own NIC, so no single card carries the whole buffer. You will measure your own number in the demo.
:::

## Why tensor parallelism is eight wide

Megatron-LM communicates four all-reduces of the full \`[tokens, dModel]\` activation per transformer layer. For a 70B model that is 320 all-reduces per micro-batch. Data parallelism communicates one gradient all-reduce per *step*, and pipeline parallelism one activation per stage boundary. So the chattiest axis must get the fastest link, and the fastest link stops at the edge of the NVLink domain, 8 GPUs on an HGX H100 node, so **tp = 8** here. A GB200 NVL72 rack stretches that domain to 72 GPUs, and NVIDIA's own NVL72 write-ups spend much of the extra room on wide expert parallelism for serving mixtures of experts rather than on wider tensor parallelism: a tensor-parallel all-reduce moves about the same bytes per GPU whatever \`tp\` is, while each GPU's share of the matmuls shrinks as \`1/tp\`.

That is not a rule you memorise — it falls out of the arithmetic. In the demo your planner will find that tp = 16 uses *less* memory per GPU and is still about a third slower, purely because a group of 16 consecutive ranks straddles two nodes.

Rank order is what makes this work. Megatron-LM numbers ranks with tensor parallelism innermost: \`rank = (ppIndex·dp + dpIndex)·tp + tpIndex\`. Tensor-parallel groups are therefore consecutive ids, data-parallel groups are spaced \`tp\` apart, and pipeline groups \`tp·dp\` apart.

## Mixtures of experts move tokens, not gradients

Module 28 built the router and the experts; here they live on different GPUs. A mixture of experts replaces the dense MLP with \`E\` experts and routes each token to \`k\` of them. DeepSeek-V3 (671B total parameters, 37B activated, 256 routed experts, top-8, hidden size 7168) spreads the experts across nodes, so every MoE layer needs two **all-to-all** exchanges: dispatch the tokens and combine the results.

The naive volume is \`tokens · k · hidden · bytes\`, twice. DeepSeek-V3's trick is to send **one copy per destination node** and fan it out over NVLink inside that node, and then to cap the router at four nodes per token ("node-limited routing"). With 8 nodes of experts, top-8 routing touches approximately \`8·(1 − (7/8)^8) = 5.25\` distinct nodes; the cap takes that to 4.

:::predict
Node-limited routing cuts the nodes per token from 5.25 to 4. Which traffic drops — the NVLink traffic, the inter-node traffic, or both?
---
Only the inter-node traffic: it falls to 4/5.25 ≈ 76% of its old value, a 24% cut. The token still visits 8 experts, so the number of copies delivered over NVLink inside the destination nodes is unchanged. You are moving traffic down the hierarchy, not removing it.
:::

## Power, and things breaking

An H100 SXM board is rated at approximately 700 W (NVIDIA's H100 datasheet). A whole DGX H100 — eight GPUs plus CPUs, NICs, NVSwitches, storage and fans — is rated at approximately 10.2 kW maximum (NVIDIA's DGX H100 datasheet), about 1.8× the GPU boards alone. Multiply by a power usage effectiveness (facility power over IT power) near 1.3 for cooling and conversion losses, and 10,000 GPUs draw roughly 16 MW — the reason new clusters are sited next to substations.

At that size, failures are continuous. The Llama 3 paper (Grattafiori et al. 2024) reports **466 job interruptions over a 54-day snapshot** on up to 16,384 H100s: 47 planned (maintenance) and **419 unexpected**, and roughly 78% of the unexpected ones were attributed to confirmed or suspected hardware issues, GPUs above all. 419 in 54 days is about 7.8 a day; if you charge every one to a GPU, that is one failure per GPU every ~50,700 hours (5.8 years). A tiny per-device rate times 16,384 devices is a failure every three hours. A synchronous step is a single point of failure: one dead GPU stops all 16,384. The defence is checkpointing, priced by Young's formula: checkpoint every \`sqrt(2 · MTBF · checkpointTime)\` seconds and lose \`sqrt(2 · checkpointTime / MTBF)\` of your wall clock to writes and redone work.

## Where this toy stops

Every link here is a single alpha–beta number. Real fabrics have congestion, adaptive routing, SHARP in-network reduction on the switch and per-rail asymmetry; NCCL picks among ring, tree and double-binary-tree algorithms by message size and topology. Our "pod" and "cluster" tiers stand in for a real fat tree with measurable oversubscription, and we assume perfectly balanced expert routing and independent failures. Our reliability model treats every interruption as an independent, memoryless failure with no restart cost; real jobs also pay minutes to reschedule, reload and warm up. The hierarchy and the arithmetic of routing traffic onto it are real; every constant is approximate.
`,
  steps: [
    {
      id: 'topology',
      title: 'The bandwidth hierarchy as data',
      instructions: `
A GPU is an integer id. The topology turns that id into a place in the machine, and a pair of places into a link.

\`location(gpu, topo)\` — return \`{ gpu, node, pod, slot }\`. The starter has the validation and \`node\` already; finish \`pod\` and \`slot\`. A node is a contiguous block of \`topo.gpusPerNode\` ids, a pod a contiguous block of \`topo.gpusPerNode * topo.nodesPerPod\`, and \`slot\` is the position inside the node. Throw on a negative or non-integer id.

\`linkBetween(a, b, topo)\` — return one of \`topo.links\`: \`self\` when the ids are equal, \`node\` when they share a node, \`pod\` when they share a pod, \`cluster\` otherwise.

\`slowestLink(gpus, topo)\` — given an array of ids, the slowest link any pair among them must use. A group of one uses \`topo.links.self\`; an empty group throws. Use \`TIERS\` to compare two links: \`TIERS.indexOf(link.tier)\` is how far apart the pair is.

Read the \`CLUSTER\` and link constants above the worked-examples line first. They are the whole hierarchy, with the datasheet each number came from.
`,
      predict: {
        question: 'In the real `CLUSTER` (8 GPUs per node, 8 nodes per pod), which link do GPUs 7 and 8 use?',
        answer: 'The pod link. GPU 7 is the last of node 0 and GPU 8 the first of node 1, so they are adjacent in numbering and a full order of magnitude apart in bandwidth — approximately 450 GB/s becomes approximately 50 GB/s.',
      },
      hints: [
        'Two integer divisions and one remainder give you node, pod and slot. `linkBetween` then compares those fields from most local to least.',
        'For `slowestLink`, the tiers form a hierarchy: two GPUs in the same node are always in the same pod. So the worst pair always contains the first GPU, and one pass comparing `gpus[0]` against every other id is enough.',
        '`location`: `node: Math.floor(gpu / topo.gpusPerNode)`, and the pod is the same idea with a bigger block. `slowestLink`: start from `let worst = topo.links.self;`, loop, and replace `worst` whenever `TIERS.indexOf(link.tier)` is larger. Do not stop at the ends of the array: in `[0, 4, 1]` the far GPU is in the middle.',
      ],
    },
    {
      id: 'collective',
      title: 'Routing a collective onto the hierarchy',
      instructions: `
You built \`ringAllReduceTime(bytes, n, link)\` in module 24; it is given above. Now decide which link it runs on.

\`flatAllReduceTime(bytes, gpus, topo)\` — one ring through every GPU in the group, every hop charged at the group's slowest link. Two lines.

\`hierarchicalAllReduce(bytes, gpus, topo)\` — the two-level schedule, in three sequential phases. With \`g\` GPUs of the group on each node:

1. reduce-scatter inside each node over \`topo.links.node\`: \`g − 1\` steps of \`bytes/g\`;
2. all-reduce those \`bytes/g\` chunks across the nodes over the slowest link between the nodes. The \`g\` GPUs of a node each run their own cross-node ring at the same time (one per slot, each on its own NIC), so the phase costs one ring of \`bytes/g\` over \`nodes\` ranks. Find the link from one representative GPU per node;
3. all-gather inside each node — the mirror of phase 1.

Skip a phase with nothing to do (\`g = 1\`, or a single node). Throw if the group does not hold the same number of GPUs on every node it touches, because then there is no symmetric schedule. Return \`{ nodes, gpusPerNode, crossLink, phases: [{ name, bytes, link, time }], time }\`.

\`allReduceTime(bytes, gpus, topo)\` — the better of the two. A group of one costs nothing.

Check as you go: inside a single node the hierarchical schedule must come out **exactly equal** to the plain ring. Phases 1 and 3 are the ring's two halves.
`,
      predict: {
        question: 'Before you write it: for a group entirely inside one node, will `hierarchicalAllReduce` be faster than, slower than, or equal to `ringAllReduceTime`?',
        answer: 'Exactly equal. A ring all-reduce IS a reduce-scatter followed by an all-gather: 2(g−1) steps of bytes/g. With one node phase 2 disappears and the two remaining phases are the ring.',
      },
      hints: [
        'Group the ids by `location(g, topo).node` into a Map first. Everything else follows from the number of nodes and the number of GPUs on each.',
        'Phase 1 and 3 are each `(g - 1) * commTime(bytes / g, topo.links.node)` — sequential steps, not a ring, because the chunk never returns. Phase 2 is `ringAllReduceTime(bytes / g, nodes.length, crossLink)` where `crossLink = slowestLink(oneGpuPerNode, topo)`.',
        '`const chunk = bytes / g; const phases = []; if (g > 1) phases.push({ name: "reduce-scatter in node", bytes: chunk, link: nodeLink, time: (g - 1) * commTime(chunk, nodeLink) }); if (nodes.length > 1) phases.push({ … ringAllReduceTime(chunk, nodes.length, crossLink) }); if (g > 1) phases.push({ /* the all-gather, same time as phase 1 */ });` then sum the times.',
      ],
    },
    {
      id: 'placement',
      title: 'Which GPUs are in which group',
      instructions: `
A layout \`{ tp, pp, dp }\` says how many ways each axis splits. Where those ranks *land* is what decides the cost. Megatron-LM's rank order puts tensor parallelism innermost:

\`\`\`
rank = (ppIndex * dp + dpIndex) * tp + tpIndex
\`\`\`

\`groupFor(kind, rank, layout)\` — the GPU ids that share the \`'tp'\`, \`'dp'\` or \`'pp'\` group of \`rank\`, in ascending order, including \`rank\` itself. Invert the formula to get the three indices, then vary one of them. Throw on an unknown kind or a rank outside \`[0, tp*pp*dp)\`.

\`groupLink(kind, layout, topo)\` — the slowest link used by **any** group of that kind. Groups of one kind all have the same shape, but where a group starts decides whether it straddles a boundary, so walk every distinct group rather than only the one containing rank 0.

This is the step where the threshold concept becomes code: \`groupLink('tp', { tp: 8, pp: 1, dp: 16 }, CLUSTER)\` must come back as the node link, and \`tp: 16\` must come back as the pod link.
`,
      hints: [
        'Inverting the formula: `tpIndex = rank % tp`, `dpIndex = Math.floor(rank / tp) % dp`, `ppIndex = Math.floor(rank / (tp * dp))`. Write a small helper `at(t, d, p)` that rebuilds a rank from three indices.',
        "For 'tp' vary `tpIndex` over `0..tp-1` and keep the other two; for 'dp' vary `dpIndex`; for 'pp' vary `ppIndex`. For `groupLink`, loop over every rank, take its group, and skip the ones you have already seen — a group's smallest member is a unique key for it.",
        '`const at = (t, d, p) => (p * dp + d) * tp + t;` then for kind "dp": `for (let i = 0; i < dp; i++) out.push(at(tpIndex, i, ppIndex));` and `return out.sort((a, b) => a - b);`',
      ],
    },
    {
      id: 'planner',
      title: 'Pricing and choosing a layout',
      instructions: `
\`layoutStepTime({ model, gpus, layout, topo, microBatchSeqs, memoryCap })\` — one training step, with every collective priced on the link its group actually lands on. With \`m = model.batchSeqs / (dp * microBatchSeqs)\` micro-batches, \`microTokens = microBatchSeqs * model.seqLen\`, \`layersPerStage = model.layers / pp\`, \`shardParams = model.params / (tp * pp)\` and \`actBytes = BYTES.params * microTokens * model.dModel\`:

| term | formula |
|------|---------|
| \`compute\` | \`6 * model.params * tokens / (topo.gpu.flops * topo.gpu.mfu * gpus)\` |
| \`tpComm\` | \`m * layersPerStage * 4 * allReduceTime(actBytes, groupFor('tp', 0, layout), topo)\`, zero if \`tp = 1\` |
| \`bubble\` | \`(compute + tpComm) * (pp - 1) / m\` |
| \`ppComm\` | \`m * 2 * (pp - 1) * commTime(actBytes, groupLink('pp', layout, topo))\`, zero if \`pp = 1\` |
| \`dpComm\` | \`allReduceTime(BYTES.grads * shardParams, groupFor('dp', 0, layout), topo)\`, zero if \`dp = 1\` |
| \`dpExposed\` | \`max(0, dpComm - topo.overlap * compute)\` |

\`stepTime\` is the sum of \`compute\`, \`tpComm\`, \`bubble\`, \`ppComm\` and \`dpExposed\`. Also return \`m\`, \`tokensPerSec\` (the global batch over the step time), \`links\` (the \`groupLink\` of each axis), \`memory\` (call the given \`memoryPerGpu\`) and \`fits\`.

Throw unless \`tp*pp*dp === gpus\`, \`pp\` divides \`model.layers\`, \`tp\` divides both \`model.dModel\` and \`model.dFF\`, and \`dp * microBatchSeqs\` divides \`model.batchSeqs\`. A planner that silently returns a number for an illegal layout will recommend one.

\`rankLayouts({ model, gpus, topo, microBatchSeqs, memoryCap })\` — every legal layout, priced, sorted with the ones that fit first and fastest first inside each group. For both functions \`topo\` defaults to \`CLUSTER\`, \`microBatchSeqs\` to 1 and \`memoryCap\` to \`topo.gpu.memory\` (the starter's signatures already say so); \`fits\` is \`memory.total <= memoryCap\`. For Llama-3-70B on 128 GPUs there are exactly 30 legal layouts.

Note the difference from module 24: \`ppComm\` is charged **per micro-batch**, because every micro-batch crosses every stage boundary in both directions.
`,
      predict: {
        question: 'On 128 H100s, `tp = 16` needs about 36 GB per GPU and `tp = 8` about 64 GB. Which one will your planner pick, and why?',
        answer: 'tp = 8. tp = 16 is about a third slower (roughly 54 s against 41 s per step) even though it uses barely half the memory. A group of 16 consecutive ranks spans two nodes, so all 4 tensor-parallel all-reduces per layer drop from approximately 450 GB/s to approximately 50 GB/s. Memory decides what is *possible*; the link decides what is *fast*.',
      },
      hints: [
        'Build it term by term and print each one for a layout you can check by hand — one GPU, `tp = pp = dp = 1`, should give `stepTime === compute` exactly.',
        'Do the validation first, before any arithmetic, so a bad layout cannot produce a number. For `rankLayouts`, loop `tp` over the divisors of `gpus`, then `pp` over the divisors of `gpus / tp`, and let `dp` be what is left.',
        'The sort is two-keyed: `out.sort((a, b) => (a.fits === b.fits ? a.stepTime - b.stepTime : a.fits ? -1 : 1))`.',
      ],
    },
    {
      id: 'moe',
      title: 'The mixture-of-experts all-to-all',
      instructions: `
A mixture of experts routes each token to \`topK\` of \`E\` experts. When the experts live on different GPUs, every MoE layer costs two all-to-all exchanges: dispatch the token vectors and combine the results.

\`expectedNodesPerToken(topK, nodes, maxNodes)\` — with experts spread uniformly, each of the \`topK\` choices lands on a given node with probability \`1/nodes\`, so the expected number of *distinct* nodes touched is \`nodes * (1 - (1 - 1/nodes)^topK)\`. Cap it at \`topK\`, at \`nodes\`, and at \`maxNodes\` (default \`Infinity\`) — DeepSeek-V3's node-limited routing. Throw if \`topK\` or \`nodes\` is below 1.

\`moeAllToAll({ tokensPerGpu, topK, dModel, gpus, topo, bytesPerElement, capacityFactor, maxNodes })\` — \`gpus\` is the expert-parallel group; \`topo\` defaults to \`CLUSTER\`, \`bytesPerElement\` to 2, \`capacityFactor\` to 1 and \`maxNodes\` to \`Infinity\`. Throw on an empty group. With \`perToken = 2 * dModel * bytesPerElement * capacityFactor\` bytes per copy (the 2 is dispatch plus combine):

- \`intraNodeBytes = tokensPerGpu * perToken * topK\` — every copy is delivered over NVLink;
- \`interNodeBytes = tokensPerGpu * perToken * nodesPerToken * (nodes - 1) / nodes\` — **one copy per destination node**, which then fans out inside it, and of the nodes a token reaches a fraction \`(nodes−1)/nodes\` is remote.

A switch delivers all pairs at once, so charge each phase one \`commTime\` of this GPU's own outgoing bytes: the inter-node phase on the slowest link between the nodes' representatives (zero when there is only one node), the intra-node phase on \`topo.links.node\`. Return \`{ nodes, nodesPerToken, interNodeBytes, intraNodeBytes, interNodeTime, intraNodeTime, time }\`.
`,
      hints: [
        'Count the distinct nodes in `gpus` with a Set of `location(g, topo).node` before anything else. Everything downstream is that count.',
        'For the cross-node link, pick one GPU per distinct node and hand that list to `slowestLink`. The probability formula is one line: `nodes * (1 - Math.pow(1 - 1 / nodes, topK))`, then `Math.min` with the three caps.',
        '`const perToken = 2 * dModel * bytesPerElement * capacityFactor; const interNodeBytes = tokensPerGpu * perToken * nodesPerToken * (nodes - 1) / nodes; const interNodeTime = nodes > 1 ? commTime(interNodeBytes, crossLink) : 0;`',
      ],
    },
    {
      id: 'reliability',
      title: 'Failures and how often to checkpoint',
      instructions: `
Three small functions that decide how much of a long run you actually keep.

\`failuresPerDay(gpus, mtbfHours)\` — failures are independent, so the rates add. Throw on a negative GPU count or a non-positive MTBF.

\`youngInterval(mtbfSeconds, checkpointSeconds)\` — Young's optimal checkpoint interval, \`sqrt(2 * mtbf * checkpointTime)\`, where \`mtbf\` is the mean time between failures of the **whole job**, not of one GPU. Throw on a non-positive MTBF or a negative checkpoint time.

\`wastedFraction({ mtbfSeconds, checkpointSeconds, intervalSeconds })\` — the fraction of wall clock that makes no progress: \`checkpointSeconds / intervalSeconds\` spent writing, plus \`intervalSeconds / (2 * mtbfSeconds)\` of work redone, since a failure on average lands halfway through an interval. Throw on a non-positive interval or MTBF.

The tests scan intervals either side of \`youngInterval\` and require that none of them wastes less. If one does, your two formulas disagree.
`,
      hints: [
        'Two of these three are one-liners. The only thing to be careful about is units: `failuresPerDay` takes hours, the other two take seconds.',
        'The job MTBF is the reciprocal of the failure RATE: if the cluster loses `f` GPUs a day, the job survives `86400 / f` seconds on average. That is the number to feed `youngInterval`.',
        'To see that the optimum is right, differentiate `c/i + i/(2·M)` with respect to `i`: `-c/i² + 1/(2M) = 0` gives `i = sqrt(2Mc)`, and substituting back gives a waste of exactly `sqrt(2c/M)`.',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why tensor parallelism is almost always exactly 8 wide, using your own numbers from the demo rather than the rule of thumb. What would have to change about the hardware for the answer to become 16?',
    'The hierarchical all-reduce moves the same result with a different schedule. Say in your own words which link carries how many bytes in each of its three phases, and why that division is where the speedup comes from.',
    'Your planner ranked one layout first on step time and rejected others on memory. Describe a situation where you would deliberately choose a slower layout, and what number you would have to look at to justify it.',
  ],
  stretch: [
    'Add rail-optimised routing: let GPU `k` of every node reach GPU `k` of any other node on its own rail at full leaf bandwidth, while a collective that mixes slots pays the spine. Check whether your planner then prefers a different data-parallel group spacing. This is the topology NCCL detects and Megatron-LM assumes.',
    'Model NVIDIA SHARP, which performs the reduction inside the InfiniBand switch: the cross-node phase becomes a single upload plus a single download rather than a `2(n−1)`-step ring. Re-run the 128-GPU plan and report how much of the step it saves.',
    'Add expert parallelism as a fourth axis to `rankLayouts` for DeepSeek-V3, using your `moeAllToAll` as its communication term, and find the largest expert-parallel degree that still beats keeping all 256 experts on one node.',
    'Replace the single-failure model with elastic training: DeepSpeed and Ray Train can continue on the surviving nodes after reconfiguring. Model the reconfiguration time as a second kind of checkpoint and find the cluster size at which elasticity beats restarting.',
  ],
  timeouts: { tests: 20000, demo: 90000 },
};
