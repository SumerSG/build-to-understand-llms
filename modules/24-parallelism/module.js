export default {
  id: '24-parallelism',
  title: 'Data, tensor & pipeline parallelism',
  track: 'systems',
  minutes: 105,
  threshold: 'Once a model or its training state no longer fits on one GPU you must split something — the batch, the matrices, or the layers — and each split buys memory at the price of one specific communication pattern whose cost you can compute.',
  goal: 'A simulator for ring all-reduce, tensor-parallel sharding, pipeline bubbles and ZeRO memory that reports step time and memory per GPU for any (dp, tp, pp, zero) layout, and a planner that picks the fastest layout fitting in 80 GB.',
  prereqs: ['08-scaling', '23-gpu-roofline'],
  recall: [
    {
      q: 'From module 08, how many FLOPs does training cost per parameter per token?',
      options: ['2', '6', '12'],
      answer: 1,
      why: 'Approximately 2 in the forward pass and 4 in the backward. Every step time in this module starts as `6·params·tokens / (peakFlops · mfu)` and is then divided by the GPU count.',
    },
    {
      q: 'Module 08 counted one training step of a bf16 model with fp32 AdamW at roughly how many bytes per parameter?',
      options: ['2', '6', '16'],
      answer: 2,
      why: '2 for the bf16 weights, 2 for the bf16 gradients, and 12 for the fp32 master copy plus the two Adam moments. ZeRO is nothing but a rule for which of those 16 bytes each rank keeps.',
    },
    {
      q: 'In the matmul of module 01, `C = A · B` with `A` of shape `[n, k]` and `B` of shape `[k, m]`: if you keep only some of the COLUMNS of `B`, what do you get?',
      options: ['Nothing usable without the other columns', 'The matching columns of `C`, complete and correct', 'A partial sum of all of `C`'],
      answer: 1,
      why: 'Each output column depends on exactly one column of B. That is why Megatron splits the first matrix of each pair by column: no communication is needed to finish those columns.',
    },
    {
      q: 'Module 23 priced a kernel as `max(flops / peakFlops, bytes / bandwidth)`. An all-reduce does one addition per element. Which term wins?',
      options: ['The FLOP term', 'The byte term, by a very large margin', 'They are usually balanced'],
      answer: 1,
      why: 'A collective has an arithmetic intensity near zero, so its cost is entirely traffic. That is why this whole module models communication with bytes and bandwidth and ignores the arithmetic inside the reduction.',
    },
  ],
  review: [
    {
      q: 'In a ring all-reduce over n GPUs, how many bytes does each GPU send for a buffer of `B` bytes?',
      options: ['`B`', '`2(n−1)/n · B`', '`n · B`'],
      answer: 1,
      why: 'n−1 reduce-scatter steps and n−1 all-gather steps, each moving one chunk of `B/n`. The factor approaches 2 and never exceeds it, which is why ring all-reduce barely gets more expensive as the cluster grows.',
    },
    {
      q: 'In Megatron-style tensor parallelism, how many all-reduces does one transformer layer cost in the FORWARD pass?',
      options: ['0', '2', '4'],
      answer: 1,
      why: 'One after the attention output projection and one after the MLP output projection — each row-parallel matmul leaves every rank with a partial sum. The backward pass costs two more, so four per layer per step.',
    },
    {
      q: 'A pipeline with `p` stages and `m` micro-batches wastes what fraction of the step in the bubble?',
      options: ['`(p−1)/p`', '`(p−1)/m`', '`p/m`'],
      answer: 1,
      why: 'Filling and draining costs p−1 stage-slots out of the m slots of useful work. More micro-batches amortise the same fill and drain, which is why real runs use many more micro-batches than stages.',
    },
    {
      q: 'Which ZeRO stage shards the parameters themselves across data-parallel ranks?',
      options: ['Stage 1', 'Stage 2', 'Stage 3'],
      answer: 2,
      why: 'Stage 1 shards the optimizer state, stage 2 also the gradients, stage 3 also the parameters. Stage 3 is what PyTorch FSDP implements, and it costs 1.5× the communication of a plain all-reduce.',
    },
    {
      q: 'Why is tensor parallelism kept inside a single node while data and pipeline parallelism cross nodes?',
      options: ['Because tensor parallelism needs more memory', 'Because it all-reduces the activation several times per layer, so it needs the fastest link in the machine', 'Because NVLink cannot carry gradients'],
      answer: 1,
      why: 'Tensor parallelism communicates four times per layer — hundreds of times per step. Pipeline parallelism sends one activation per stage boundary and data parallelism one gradient per step, so both tolerate the slower inter-node link.',
    },
  ],
  concept: `
## Three things you can split

A Llama-3-70B training step needs about 1.1 TB of persistent state: 70 billion parameters at the 16 bytes per parameter you counted in module 08. An H100 has 80 GB. The step must be spread over many GPUs, and there are exactly three things you can cut.

**Split the batch.** Every GPU keeps a complete copy of the model and processes a different slice of the tokens. The gradients must then be averaged, which is one **all-reduce** of the whole gradient buffer per step. This is *data parallelism*, and it is the only one of the three that does not reduce the memory a single GPU needs for weights.

**Split the matrices.** Every GPU keeps a slice of each weight matrix and they cooperate on every matmul. This is *tensor parallelism*, introduced for transformers by Megatron-LM (Shoeybi et al. 2019). The trick is to pair the splits: the first matrix of each pair (the fused QKV projection, the MLP up-projection) is split by **columns**, so each rank computes complete columns of the output with no communication; the second (the two output projections) is split by **rows**, so each rank holds a partial sum and the ranks finish with one all-reduce. Two all-reduces per layer forward, two more backward.

**Split the layers.** Stage 0 holds layers 0–19, stage 1 holds 20–39, and an activation is passed forward at each boundary. This is *pipeline parallelism* (GPipe, Huang et al. 2019). Its problem is idleness: while stage 0 works on the first micro-batch, the other stages have nothing to do.

:::predict
You cut a batch into \`m\` micro-batches and push them through \`p\` pipeline stages. Ignoring communication, what fraction of the step do the stages spend idle?
---
\`(p−1)/m\`. Filling and draining overlap into p−1 wasted slots out of m useful ones. With p=16 and m=16 you waste 94% of the machine; with m=512, 3%. 1F1B (PipeDream, Narayanan et al. 2019) does not change this fraction — it changes *memory*, by starting each micro-batch's backward pass as early as possible so a stage holds \`min(p, m)\` micro-batches of activations instead of all \`m\`.
:::

## What each split costs on the wire

Every collective in this module is priced with the same two numbers: a fixed **latency** per message and a **bandwidth** in bytes per second. One message of \`B\` bytes costs \`latency + B/bandwidth\`. A ring all-reduce over \`n\` GPUs is \`2(n−1)\` such messages, each carrying one chunk of \`B/n\`, so each GPU sends \`2(n−1)/n · B\` — a factor that climbs towards 2 and stops. That is the whole reason the ring is the default: doubling the cluster does not double the gradient traffic per GPU.

It does double the *number of steps*, though, and each step pays the latency again. So a small buffer over a large ring is pure latency (1 KB over 64 GPUs on InfiniBand is 1.9 ms of latency and 20 nanoseconds of transfer), while a large one is pure bandwidth. Gradient bucketing in PyTorch DDP exists to keep messages in the second regime and to start the all-reduce on early buckets while the backward pass is still running — the \`overlap\` term in your simulator.

:::predict
Tensor parallelism all-reduces the activation four times per layer. Data parallelism all-reduces the gradient once per step. For an 80-layer model with 64 micro-batches, which moves more bytes, and by how much?
---
Tensor parallelism, by a wide margin: 4 × 80 × 64 = 20,480 collectives per step against 1. Each is smaller (an activation, not the whole gradient), but the count is what matters. That is why tensor parallelism stays inside one node, on NVLink 4 at approximately 450 GB/s per GPU, while data and pipeline parallelism may cross InfiniBand at approximately 50 GB/s.
:::

## ZeRO: paying for memory with communication

ZeRO (Rajbhandari et al. 2020) asks why every data-parallel rank stores all 16 bytes per parameter when it only ever *updates* 1/dp of them. Stage 1 shards the 12-byte optimizer state, stage 2 also the 2-byte gradients, stage 3 also the 2-byte parameters, gathering each shard just before it is needed. Stage 3 is PyTorch's FSDP. It costs 1.5× the communication of plain data parallelism, usually hidden behind compute, so you turn ZeRO up until the model fits.

Llama 3 was trained with tensor parallelism of 8 (one node), pipeline parallelism of 16, and data parallelism across the rest of approximately 16,000 H100s (Grattafiori et al., "The Llama 3 Herd of Models", 2024). Two further splits exist that this module does not model: **sequence (context) parallelism**, which cuts the sequence axis to shrink activation memory and the long-context attention cost, and **expert parallelism**, which places different mixture-of-experts experts on different GPUs and pays an all-to-all — that one is module 25.

## Where this toy differs from production

Your simulator is an arithmetic model, not a measurement. It assumes every GPU runs at exactly \`mfu\` of peak, that collectives achieve the full link bandwidth, that the network is uncontended, and that a single \`overlap\` fraction captures all compute/communication overlap. It ignores activation recomputation (roughly 30% more compute for a large drop in activation memory, used in essentially every real run), sequence parallelism, optimizer offload to host memory, and the extra embedding and loss weights on the first and last pipeline stages. Compare the layouts it prints against each other; do not read them as wall-clock predictions.
`,
  steps: [
    {
      id: 'ring',
      title: 'The alpha–beta model and the ring all-reduce',
      instructions: `
Two functions, both built on the worked \`commTime(bytes, link)\` above.

\`ringAllReduceBytes(bytes, n)\`: how many bytes ONE GPU sends during a ring all-reduce of a \`bytes\`-sized buffer over \`n\` GPUs. The ring runs \`n − 1\` reduce-scatter steps followed by \`n − 1\` all-gather steps, and each step moves one chunk of \`bytes / n\`. With \`n = 1\` the answer is 0. Throw an \`Error\` if \`n < 1\`.

\`ringAllReduceTime(bytes, n, link)\`: the wall-clock time of that collective. The steps are **sequential** — a GPU cannot send chunk 2 before it has received and reduced chunk 1 — so the time is the number of steps times the cost of one step, and one step is a \`commTime\` call on a chunk. Return 0 for \`n = 1\`; throw if \`n < 1\`.

Both are one line. The point is which quantity goes inside \`commTime\`: the chunk, not the buffer.
`,
      predict: {
        question: 'Take a 1 KB buffer and a 1 GB buffer, both all-reduced over 64 GPUs on InfiniBand (15 µs latency, 50 GB/s). What is the ratio of their times? What would it be if the latency term did not exist?',
        answer: 'Approximately 22×, not 1,000,000×. The 1 KB case is 2·63·15 µs = 1.89 ms of pure latency plus 20 ns of transfer; the 1 GB case is 1.89 ms of latency plus about 39 ms of transfer. Without the latency term the ratio would be exactly 1,000,000. Small collectives are latency-bound, which is why frameworks fuse gradients into buckets of tens of megabytes.',
      },
      hints: [
        'Write down the two phases separately first: how many steps is reduce-scatter, how many is all-gather, and how big is the chunk each step moves?',
        '`2(n − 1)` steps in total, each moving `bytes / n`. For the byte count that is `2(n−1)·bytes/n`; for the time it is `2(n−1)` multiplied by `commTime(bytes / n, link)`, because each step pays the latency again.',
        '`if (!(n >= 1)) throw new Error(...); if (n === 1) return 0; return 2 * (n - 1) * commTime(bytes / n, link);` — and the byte version is the same shape without the link.',
      ],
    },
    {
      id: 'dp',
      title: 'Data parallelism and the overlap that hides it',
      instructions: `
\`dataParallelStep({ params, tokensPerGpu, dp, gpu, link, overlap = 0 })\` returns \`{ compute, comm, exposed, step }\` in seconds.

- \`compute\` is \`computeTime(params, tokensPerGpu, gpu)\` — one replica's share of the arithmetic.
- \`comm\` is a ring all-reduce of the gradient. Gradients are bf16, so the buffer is \`BYTES.grads * params\` bytes, all-reduced over \`dp\` ranks on \`link\`.
- \`exposed\` is the part of \`comm\` that the learner cannot hide: the all-reduce may run in the background during at most \`overlap * compute\` seconds of the step, so \`exposed = comm − min(comm, overlap * compute)\`. Taking the \`min\` matters — without it a fast all-reduce would produce a negative, and a step shorter than the arithmetic it performs.
- \`step\` is \`compute + exposed\`.

\`dpEfficiency(cfg, dp)\` then reports \`compute / step\` for a config \`cfg\` that has every field except \`dp\`. It is 1.0 when the all-reduce is fully hidden and falls towards 0 as gradients start to dominate.
`,
      hints: [
        'Only one line of this is subtle. Ask yourself what `exposed` should be when the all-reduce is faster than the window it is allowed to hide in.',
        'Compute `hidden = Math.min(comm, overlap * compute)` first, then `exposed = comm - hidden`. `dpEfficiency` just spreads `cfg` and the given `dp` into `dataParallelStep` and divides.',
        '`const hidden = Math.min(comm, overlap * compute);` then `exposed = comm - hidden` and `step = compute + exposed`. `dpEfficiency` spreads `{ ...cfg, dp }` into `dataParallelStep` and returns `compute / step` from the result.',
      ],
    },
    {
      id: 'tp',
      title: 'Tensor parallelism: column then row',
      instructions: `
\`tpShards({ dModel, dFF }, tp)\` describes what ONE GPU holds of one transformer layer. Return exactly four entries, in this order:

| name | full shape | split | shard shape |
|------|-----------|-------|-------------|
| \`attn.qkv\` | \`[dModel, 3·dModel]\` | \`'column'\` | \`[dModel, 3·dModel/tp]\` |
| \`attn.proj\` | \`[dModel, dModel]\` | \`'row'\` | \`[dModel/tp, dModel]\` |
| \`mlp.fc\` | \`[dModel, dFF]\` | \`'column'\` | \`[dModel, dFF/tp]\` |
| \`mlp.proj\` | \`[dFF, dModel]\` | \`'row'\` | \`[dFF/tp, dModel]\` |

Each entry is \`{ name, split, shape }\`. A **column** split divides the output dimension (the second axis), a **row** split divides the input dimension (the first). Throw an \`Error\` if \`tp < 1\` or if \`dModel\` or \`dFF\` is not divisible by \`tp\`. A test checks that the four shards add up to exactly \`1/tp\` of the layer's weights — no replication, nothing lost.

\`tpLayerComm({ dModel, tokens }, tp, link)\` returns \`{ allReduces, bytesPerAllReduce, time }\` for one layer on a micro-batch of \`tokens\` tokens. There are 4 all-reduces when \`tp > 1\` (two forward, two backward) and 0 when \`tp === 1\`. Each carries the **whole** bf16 \`[tokens, dModel]\` activation — it is a sum over ranks, so the payload does not shrink with \`tp\`. Price it with your \`ringAllReduceTime\`.
`,
      predict: {
        question: 'Why does the column-then-row pairing avoid a communication between the two matmuls of the MLP?',
        answer: 'The column split gives rank i complete columns of the hidden activation — the columns it needs, because GeLU is elementwise. The row split of the down-projection expects exactly those rows of its input. So rank i can run both matmuls on data it already has, and only the final sum needs the ranks to talk. Splitting the other way round would need an all-gather in the middle as well.',
      },
      hints: [
        'Write the four entries out literally. The only question per entry is which axis gets divided by `tp`, and the split name tells you.',
        'Column-parallel: divide `shape[1]`. Row-parallel: divide `shape[0]`. For `tpLayerComm`, the byte count is `BYTES.params * tokens * dModel` — independent of `tp` — and the time is `allReduces * ringAllReduceTime(bytes, tp, link)`.',
        '`return [{ name: "attn.qkv", split: "column", shape: [dModel, 3 * dModel / tp] }, { name: "attn.proj", split: "row", shape: [dModel / tp, dModel] }, …];` with the divisibility check before it.',
      ],
    },
    {
      id: 'pp',
      title: 'Pipeline parallelism and the bubble',
      instructions: `
\`pipelineBubble(p, m)\` returns the idle fraction \`(p − 1) / m\` for \`p\` stages and \`m\` micro-batches. Throw if either is below 1.

\`pipelineStep({ modelTime, p, m, activationBytesPerMicroBatch = 0 })\` returns \`{ stageTime, bubble, step, activationBytes1F1B, activationBytesGPipe }\`.

- \`modelTime\` is the seconds ONE worker would need to push the whole batch through ALL \`p\` stages, so \`stageTime = modelTime / p\`.
- \`step = stageTime * (1 + bubble)\`: a stage does its own work and then waits out the bubble.
- \`activationBytes1F1B = min(p, m) * activationBytesPerMicroBatch\`. Under the 1F1B schedule a stage starts a backward pass as soon as one is available, so at most \`min(p, m)\` micro-batches are in flight on any one stage.
- \`activationBytesGPipe = m * activationBytesPerMicroBatch\`: GPipe runs every forward before any backward, so a stage holds all \`m\`.

That last pair is the whole reason 1F1B replaced GPipe. The bubble is identical; the memory is not.
`,
      hints: [
        'Four of the five fields are one expression each. Get `stageTime` right first — read the definition of `modelTime` twice.',
        '`stageTime = modelTime / p`, `bubble = pipelineBubble(p, m)`, `step = stageTime * (1 + bubble)`. The two activation figures differ only in `Math.min(p, m)` versus `m`.',
        'The return is `{ stageTime, bubble, step: stageTime * (1 + bubble), activationBytes1F1B: …, activationBytesGPipe: … }`, where the two activation fields multiply `activationBytesPerMicroBatch` by `Math.min(p, m)` and by `m` respectively.',
      ],
    },
    {
      id: 'zero',
      title: 'ZeRO and FSDP memory',
      instructions: `
\`zeroMemoryPerGpu(params, dp, stage)\` returns \`{ params, grads, optimizer, total }\` in bytes per GPU, using the per-parameter costs in the \`BYTES\` constant: \`params\` 2, \`grads\` 2, \`optimizer\` 12.

Each stage divides one more of those three by \`dp\`:

| stage | params | grads | optimizer | bytes/param at dp = 64 |
|-------|--------|-------|-----------|------------------------|
| 0 | replicated | replicated | replicated | 16 |
| 1 | replicated | replicated | **/dp** | 4.19 |
| 2 | replicated | **/dp** | **/dp** | 2.22 |
| 3 | **/dp** | **/dp** | **/dp** | 0.25 |

Throw an \`Error\` if \`stage\` is not one of 0, 1, 2, 3, or if \`dp < 1\`. Note that at \`dp = 1\` all four stages must give the same answer: there is nobody to shard with.

Activations are **not** part of this. They are a separate term, and step 6 is where you will find out that they, not the weights, are usually what fails to fit.
`,
      hints: [
        'Each of the three lines has the same shape: `BYTES.x * params` divided by either `dp` or 1. What decides which?',
        'A comparison: the optimizer state is sharded when `stage >= 1`, the gradients when `stage >= 2`, the parameters when `stage >= 3`. Write `(stage >= 1 ? dp : 1)` as the divisor.',
        '`const optimizer = BYTES.optimizer * params / (stage >= 1 ? dp : 1);` and the same pattern for `grads` (>= 2) and `params` (>= 3); `total` is their sum.',
      ],
    },
    {
      id: 'planner',
      title: 'The planner: combine DP × TP × PP and choose',
      instructions: `
\`plan({ model, gpus, strategy, hardware = HARDWARE })\` simulates one step under \`strategy = { dp, tp, pp, zero = 1, microBatchSeqs = 1 }\`.

**Validate first**, throwing an \`Error\` with a message naming the offending number, when: \`dp·tp·pp !== gpus\`; \`tp > hardware.gpusPerNode\`; \`model.layers % pp !== 0\`; \`model.batchSeqs % (dp * microBatchSeqs) !== 0\`. Also call \`tpShards(model, tp)\` so an unshardable \`dModel\` or \`dFF\` throws too.

**Then derive:**

\`\`\`
m               = model.batchSeqs / (dp * microBatchSeqs)   // micro-batches per pipeline per step
microTokens     = microBatchSeqs * model.seqLen
tokens          = model.batchSeqs * model.seqLen            // global batch
layersPerStage  = model.layers / pp
shardParams     = model.params / (tp * pp)                  // what one GPU owns before ZeRO
dpLink          = tp * dp <= gpusPerNode ? intraNode : interNode
ppLink          = gpus   <= gpusPerNode ? intraNode : interNode
\`\`\`

**Time.** \`compute\` is the whole batch's \`computeTime\` divided by \`gpus\`. \`tpComm\` is \`m * layersPerStage\` calls to \`tpLayerComm\` on \`intraNode\` (tensor parallelism is always intra-node). Feed \`(compute + tpComm) * pp\` into \`pipelineStep\` as \`modelTime\`, with \`activationBytesPerMicroBatch = ACT_BYTES_PER_TOKEN_LAYER_DIM * microTokens * model.dModel * layersPerStage / tp\`. Add \`ppComm\`, which is \`2(pp − 1)\` boundary crossings of one bf16 \`[microTokens, dModel]\` activation on \`ppLink\` (0 when \`pp === 1\`), and the exposed part of \`dpComm\`, a ring all-reduce of \`BYTES.grads * shardParams\` bytes — multiplied by \`ZERO3_COMM_FACTOR\` when \`zero === 3\` — over \`dp\` ranks on \`dpLink\`, hidden by up to \`hardware.overlap * compute\`.

**Memory.** \`zeroMemoryPerGpu(shardParams, dp, zero).total\` plus \`pipelineStep\`'s \`activationBytes1F1B\`.

Return \`{ dp, tp, pp, zero, m, stepTime, memoryPerGpu, fits, tokensPerSec, breakdown, memory }\` with \`fits = memoryPerGpu <= hardware.gpu.memory\`, \`tokensPerSec = tokens / stepTime\`, \`breakdown = { compute, tpComm, bubble, ppComm, dpComm, dpExposed }\` (where \`bubble\` is the seconds the bubble adds, not the fraction), and \`memory = { params, grads, optimizer, activations }\`.

\`enumerateStrategies(model, gpus, hardware)\` returns every legal \`{ dp, tp, pp, zero }\`: \`tp\` from 1 to \`min(gpus, gpusPerNode)\` dividing \`gpus\`, \`dModel\` and \`dFF\`; \`pp\` dividing \`gpus/tp\` and \`model.layers\`; \`dp = gpus/(tp·pp)\` dividing \`model.batchSeqs\`; all four \`zero\` stages.

\`bestPlan({ model, gpus, hardware, memoryCap = hardware.gpu.memory })\` returns the plan with the smallest \`stepTime\` among those with \`memoryPerGpu <= memoryCap\`, breaking ties towards less memory, or \`null\` if none fits.
`,
      predict: {
        question: 'For Llama-3-70B on 64 H100s, which layout do you expect to be fastest: pure data parallelism (dp=64), or something with tensor parallelism? And which do you expect to fit in 80 GB?',
        answer: 'Pure data parallelism is the fastest — it has no tensor-parallel all-reduces and its single gradient all-reduce hides behind a 70-second step — but it needs about 200 GB per GPU, almost all of it activations, and does not fit. The planner ends up at tp=4, which quarters the activation memory to about 63 GB and costs roughly 7% more step time. Tensor parallelism is not a speed-up; it is the tax you pay to run at all.',
      },
      hints: [
        'Build it in the order the return value is written: validate, derive the six quantities in the table, then time, then memory. Every line calls something you already wrote.',
        'The one place to be careful is `modelTime`. `compute` is already one GPU\'s share of the whole cluster\'s work, which means it is already a stage\'s work — so `modelTime`, defined as all `pp` stages, is `(compute + tpComm) * pp`. For `bestPlan`, loop over `enumerateStrategies`, call `plan`, skip anything over `memoryCap`, keep the minimum.',
        '`const pipe = pipelineStep({ modelTime: (compute + tpComm) * pp, p: pp, m, activationBytesPerMicroBatch: actPerMicro }); const stepTime = pipe.step + ppComm + dpExposed;` and `breakdown.bubble = pipe.step - (compute + tpComm)`.',
      ],
    },
  ],
  reflection: [
    'A colleague says "we are out of memory, let us add more GPUs with data parallelism". Explain in your own words why that does not help, and what each of the other two splits would actually free.',
    'Tensor parallelism is confined to one node and pipeline parallelism is not. Trace that rule back to the two communication patterns: what is sent, how big it is, and how many times per step.',
    'Your planner reports a step time and a memory figure for every layout. Name three things it assumes that a real cluster would violate, and say which direction each one would move the reported number.',
  ],
  stretch: [
    'Add activation recomputation: a flag that multiplies `compute` by about 1.33 and divides the activation term by `layersPerStage` (only the layer boundaries are stored). Megatron-LM calls this selective recompute; see Korthikanti et al. 2022. Then rerun `bestPlan` and see whether tensor parallelism is still needed.',
    'Add sequence parallelism: split the sequence axis across the same `tp` ranks in the LayerNorm and dropout regions, which turns two of the four all-reduces per layer into a reduce-scatter plus an all-gather of the same total volume, and divides the remaining activation memory by `tp`. This is section 3 of the same Megatron-LM paper.',
    'Replace the ring all-reduce with a two-level one: reduce-scatter inside the node over NVLink, all-reduce across nodes over InfiniBand, all-gather inside the node again. This is what NCCL actually does on a multi-node cluster; compare your two models at dp = 1024.',
    'Give `enumerateStrategies` a `microBatchSeqs` axis (1, 2, 4, 8) and see what the planner chooses. Larger micro-batches cut the number of tensor-parallel collectives but raise activation memory and the bubble — this is the trade-off DeepSpeed\'s autotuner searches.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
