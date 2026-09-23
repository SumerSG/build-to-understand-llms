export default {
  id: '08-scaling',
  title: 'Scaling laws & the arithmetic of compute',
  track: 'transformer',
  minutes: 75,
  threshold: 'Training compute is approximately 6 × parameters × tokens, and for a fixed compute budget there is one best split between model size and data — so the cost, duration and hardware of a training run are a few multiplications, not a mystery.',
  goal: 'A run planner that turns model size, tokens and hardware into FLOPs, memory, and wall-clock time, and finds the compute-optimal model for a budget.',
  prereqs: ['06-transformer', '07-pretraining'],
  recall: [
    { q: 'In module 06 you wrote a parameter-count formula for a GPT. Which part of a 175B-parameter model dominates the count?', options: ['The token and position embeddings', 'The per-layer attention and MLP weights', 'The final LayerNorm'], answer: 1, why: 'Embeddings are `V·d` once; the blocks are `12·d²` per layer times many layers. That is why the 6ND rule can ignore everything but the block weights and still be close.' },
    { q: 'In module 07 your training step called backward() before the AdamW step. What does backward compute for each weight matrix?', options: ['Only the gradient of the loss with respect to the weights', 'A gradient with respect to the weights and one with respect to the layer input', 'The forward activations again'], answer: 1, why: 'The weight gradient feeds the optimizer; the input gradient is what flows on to the previous layer. That is two matmuls where forward did one — 2 FLOPs forward + 4 backward = the 6 in 6ND.' },
    { q: 'AdamW in module 07 keeps how many running buffers per parameter?', options: ['None', 'One (momentum)', 'Two (first and second moment)'], answer: 2, why: 'The `m` and `v` buffers are kept in fp32 — 8 bytes per parameter — which is half the memory bill of a bf16 training run.' },
    { q: 'Your matmul in module 01 does `2·n³` FLOPs for two `n × n` matrices. Why the factor of 2?', options: ['Two passes over memory', 'Each output element costs n multiplies and n adds', 'Float32 counts double'], answer: 1, why: 'One multiply-add is two floating-point operations. The same convention makes a forward pass 2 FLOPs per parameter per token.' },
    { q: 'In module 05, what shape is the attention score matrix for one head over a sequence of T tokens?', options: ['[T, d]', '[T, T]', '[d, d]'], answer: 1, why: 'One score per pair of positions. Those T×T matrices per head are exactly the activation term you will price in step 3 — and the one FlashAttention refuses to store.' },
  ],
  review: [
    { q: 'A 30B-parameter model trained on 2T tokens costs how many FLOPs?', options: ['1.2e23', '3.6e23', '6.0e22'], answer: 1, why: '6 × 3e10 × 2e12 = 3.6e23. Every training-cost question is this one multiplication.' },
    { q: 'You are told a run reached 45% MFU. What does that mean?', options: ['45% of the GPUs were busy', 'The run did 45% of the FLOPs the chips could do at their datasheet peak', '45% of the memory was used'], answer: 1, why: 'MFU is useful-model-FLOPs divided by peak-FLOPs-seconds. It is why a nominal 989 TFLOP/s H100 plans at roughly 400 TFLOP/s.' },
    { q: 'Training a 70B model in bf16 with AdamW needs approximately how much HBM for persistent state, before activations?', options: ['140 GB', '560 GB', '1.1 TB'], answer: 2, why: '16 bytes per parameter (2 weights + 2 grads + 8 Adam + 4 fp32 master) × 7e10 = 1.12 TB — fourteen 80 GB H100s just to hold it.' },
    { q: 'At a fixed compute budget C, plotting predicted loss against model size N gives…', options: ['A line falling with N', 'A U-shape with a single minimum', 'A flat line'], answer: 1, why: 'Small N leaves the `A/N^alpha` term large; large N starves the model of tokens because `D = C/6N` shrinks. The minimum is the compute-optimal model.' },
    { q: 'Llama-3 8B saw approximately 1875 tokens per parameter instead of about 20. What does that buy?', options: ['A lower training bill', 'A model that is cheaper to serve at the same quality', 'A larger context window'], answer: 1, why: 'Inference costs 2N per token forever. Spending extra training compute to hit a given loss with a smaller N pays back once you serve enough tokens.' },
  ],
  concept: `
## Six multiplications decide a training run

A transformer training run looks like an engineering epic and prices like arithmetic. Fix two numbers — \`N\`, the parameter count, and \`D\`, the number of training tokens — and almost everything follows.

Start with one parameter and one token. In the forward pass that parameter is used in one multiply-add: **2 FLOPs**. The backward pass computes two gradients, one with respect to the layer input and one with respect to the weight, each another multiply-add: **4 FLOPs**. Total **6 FLOPs per parameter per token**, so a run costs

\`C = 6 · N · D\`

This is the estimate used in Kaplan et al. 2020 and in every scaling paper since. It ignores embeddings, LayerNorm, the softmax, and the attention terms that grow with context length; for the billion-parameter models priced in this module those are a few percent. For the toy GPTs of modules 06–07 the embeddings are a sizeable fraction: 6ND is an estimate for large runs, not a law. Inference is the forward pass only: \`2 · N\` FLOPs per token generated.

:::predict
GPT-3 is approximately 175 billion parameters trained on approximately 300 billion tokens (Brown et al. 2020). How many FLOPs is that, and how long would one NVIDIA H100 take? NVIDIA's H100 datasheet lists approximately 989 TFLOP/s of dense bf16 throughput.
---
\`6 × 1.75e11 × 3e11 = 3.15e23\` FLOPs (the paper reports 3.14e23). At the full 989e12 FLOP/s that is 3.2e8 seconds, about **10 years** on one chip — and no run reaches the datasheet peak, so read on.
:::

## MFU: the number between the datasheet and reality

No run does 989 TFLOP/s. Weights have to be fetched from HBM, gradients have to cross the network, the pipeline has bubbles. **Model FLOPs utilisation (MFU)** is the useful model FLOPs divided by what the chips could have done at peak in the same time. Large dense runs report roughly 30–50%: the PaLM paper (Chowdhery et al. 2022) reports 46.2% on TPU v4, and Meta's Llama 3 report describes 38–43% on H100s. Plan at 0.4 and you will rarely be embarrassed.

\`seconds = C / (gpus · peakFlops · mfu)\`

Price it at an assumed 2 US dollars per H100-hour and GPT-3's 221,000 H100-hours is roughly 0.44 million dollars of compute. Published estimates of the actual 2020 run, on V100s, are approximately 4 million dollars — roughly an order of magnitude more for the same FLOPs, which is the story of this decade in one line.

## Memory, not FLOPs, is what forces a cluster

FLOPs you can wait for. Memory you cannot. In bf16 mixed precision with AdamW, every parameter carries:

| item | bytes |
|---|---|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| Adam first and second moments (fp32) | 8 |
| fp32 master weights | 4 |
| **total** | **16** |

That accounting is from the ZeRO paper (Rajbhandari et al. 2020). A 405B model is therefore approximately 6.5 TB of persistent state — 81 H100s just to hold it, before a single activation is stored. Activations add the Megatron-LM estimate (Korthikanti et al. 2022) of \`s·b·h·(34 + 5·a·s/h)\` bytes per layer, where \`s\` is sequence length, \`b\` batch, \`h\` hidden size and \`a\` heads; the \`5·a·s²·b\` part is the attention score matrices, which FlashAttention never writes down. This is why module 24 exists: parallelism is a memory problem first.

## Kaplan, then Chinchilla

Kaplan et al. 2020 fit loss against \`N\` and \`D\` and concluded that when compute grows you should grow the model much faster than the data. Hoffmann et al. 2022 re-ran the experiment with the learning-rate schedule matched to each run's length and found the opposite: **N and D should grow together**, roughly **20 tokens per parameter**. Their 70B Chinchilla beat the 280B Gopher trained on the same compute. The fit is

\`L(N, D) = E + A / N^alpha + B / D^beta\`

\`E\` is the irreducible loss of the data itself, \`A/N^alpha\` is what you lose by being small, \`B/D^beta\` is what you lose by not reading enough. Minimise it subject to \`6ND = C\` and the answer is closed-form. One catch: the constants printed in the paper do not reproduce its own 20:1 rule. Plugged into that closed form they give about 78 tokens per parameter at 1e23 FLOPs and about 93 at Gopher's budget. Besiroglu et al. 2024 could not reproduce those constants from the paper's own data and published a re-fit, which gives about 19. You will implement both and watch them disagree; the demo uses the re-fit.

:::predict
If 20 tokens per parameter is optimal, why did Meta train an 8B model on approximately 15 trillion tokens — 1875 tokens per parameter?
---
Because Chinchilla optimises the **training** bill only. Serving costs \`2N\` FLOPs per token forever. Over-training a small model to the quality of a bigger one costs extra compute once and saves on every request after; you will compute the break-even point in step 5.
:::

## Where this toy is wrong

These are fits, not laws. The constants come from dense decoder-only models on one data distribution; mixture-of-experts models such as DeepSeek-V3 break the \`6ND\` rule (only active parameters count), repeated data breaks the \`D\` term (Muennighoff et al. 2023), and none of this predicts whether a model can do arithmetic. The planner also assumes perfect linear scaling across GPUs, no failures and no restarts — Meta reports 419 unexpected interruptions in a 54-day snapshot of Llama-3-405B pre-training (466 counting planned maintenance). Treat its output as the right order of magnitude, which is exactly what it is good for.
`,
  steps: [
    {
      id: 'flops',
      title: 'The 6ND rule',
      instructions: `
Two one-line functions, and the reasoning behind them is the whole module.

\`trainingFlops(N, D)\` returns \`6 · N · D\`: 2 FLOPs forward and 4 backward per parameter per token. Throw an \`Error\` if \`N\` or \`D\` is not positive — a run with no parameters is a bug, not a free lunch, and a silent \`0\` will propagate into every later number.

\`inferenceFlops(N, tokens = 1)\` returns \`2 · N · tokens\`: inference is the forward pass only.

The worked code above the TODO line (\`PRECISION_BYTES\`, \`GPUS\`, \`formatDuration\`) sets the conventions: constants are exported so the tests and the demo can reuse them, and hardware figures carry their source in a comment.
`,
      predict: { question: 'Your model has 1e9 parameters and you train it on 1e10 tokens. How many FLOPs, and what fraction of that is the backward pass?', answer: '6e19 FLOPs. Four sixths — about 67% — is backward: two of the six FLOPs are the forward multiply-add, four are the two backward matmuls.' },
      hints: [
        'Both functions are a single multiplication. The work is in the argument check, and in being sure which constant belongs where: 2 for a forward pass, 6 for forward plus backward.',
        'Guard first, then return. `if (!(N > 0) || !(D > 0)) throw new Error(...)`. Writing the condition as `!(N > 0)` rather than `N <= 0` also rejects `NaN`, which otherwise sails through and poisons the whole plan.',
        'The error message should name the offending values, e.g. `` throw new Error(`trainingFlops: N and D must be positive, got N=${N}, D=${D}`) ``, then `return 6 * N * D`.',
      ],
    },
    {
      id: 'wallclock',
      title: 'From FLOPs to days and dollars',
      instructions: `
\`wallClockSeconds(flops, { gpus, peakFlops, mfu })\` returns \`flops / (gpus · peakFlops · mfu)\`.

\`mfu\` — model FLOPs utilisation — is the fraction of the chips' datasheet peak your run actually achieves, and it is a fraction in \`(0, 1]\`. Throw if it is outside that range: passing \`40\` when you meant \`0.4\` is the single most common mistake with this formula, and it makes a nine-day run look like thirteen minutes. Also throw if \`gpus < 1\` or \`peakFlops <= 0\`.

\`gpuHours(flops, { peakFlops, mfu })\` is the same work priced as one chip's time in hours. It must not depend on how many GPUs share the job — wall-clock × GPU count is invariant — so implement it in terms of \`wallClockSeconds\` with \`gpus: 1\` and divide by 3600.

This model assumes perfect linear scaling. Real clusters do not scale linearly; module 24 is where that assumption is paid for.
`,
      predict: { question: 'GPT-3 is 3.15e23 FLOPs. On 1024 H100s (approximately 989 TFLOP/s each per NVIDIA\'s datasheet) at 40% MFU, how many days?', answer: 'About 9 days: 3.15e23 / (1024 × 989e12 × 0.4) = 777,600 seconds. The same run on 1024 A100s at 312 TFLOP/s would take about 29 days.' },
      hints: [
        'One division. The three factors that multiply together in the denominator are chips, per-chip peak, and the fraction of that peak you actually reach.',
        'Validate `mfu` in a small helper you call from both functions, so the rule lives in one place. `gpuHours` should call `wallClockSeconds({ gpus: 1, ... })` rather than repeating the division.',
        '`if (!(mfu > 0 && mfu <= 1)) throw new Error(...)`; then `return flops / (gpus * peakFlops * mfu)`. `gpuHours` is `wallClockSeconds(flops, { gpus: 1, peakFlops, mfu }) / 3600`.',
      ],
    },
    {
      id: 'memory',
      title: 'The memory bill, and the planner',
      instructions: `
\`trainingMemory(N, { precision = 'bf16', optimizer = 'adamw' })\` returns
\`{ weights, grads, optimizerStates, masterWeights, total, bytesPerParam }\` in bytes. Weights and gradients each cost \`PRECISION_BYTES[precision]\` per parameter; the optimizer costs \`OPTIMIZER_BYTES[optimizer]\`; every precision other than \`fp32\` also keeps a 4-byte fp32 master copy that the update is applied to. Throw on a name that is not in either table — returning \`NaN\` would quietly poison the plan.

\`activationBytes({ batch, seq, hidden, layers, heads, flashAttention = false, bytesPerValue = 2 })\` uses the Megatron-LM estimate: per layer, \`seq · batch · hidden · 34\` bytes of saved tensors plus \`5 · heads · seq² · batch\` bytes of attention scores, at 2 bytes per value. Write it as \`layers · (17 · seq · batch · hidden + 2.5 · heads · seq² · batch) · bytesPerValue\` so the \`bytesPerValue\` factor applies to both terms. With \`flashAttention: true\` the score term is zero, because the scores are recomputed in the backward pass instead of stored.

\`planRun({ params, tokens, gpus, gpuFlops, gpuMemoryBytes, mfu, precision, optimizer, pricePerGpuHour })\` is the artifact: it calls the three functions you already wrote and returns \`{ flops, seconds, days, gpuHours, dollars, memory, minGpusForState, tokensPerParam }\`. \`minGpusForState\` is \`Math.ceil(memory.total / gpuMemoryBytes)\` — how many chips it takes to *hold* the run, ignoring activations entirely. Do not re-derive any arithmetic here.
`,
      hints: [
        'Four line items make up `total`. Only one of them depends on a condition: whether a separate fp32 master copy is needed at all.',
        'Look the byte counts up in the exported tables and check for `undefined` before using them, so a typo throws instead of producing NaN. `masterWeights` is `precision === "fp32" ? 0 : 4 * N`.',
        '`const pb = PRECISION_BYTES[precision]; if (pb === undefined) throw ...; const weights = pb * N, grads = pb * N, optimizerStates = ob * N, masterWeights = ...; return { weights, grads, optimizerStates, masterWeights, total, bytesPerParam: total / N }`. In `planRun`, `days` is `seconds / 86400` and `dollars` is `gpuHours × pricePerGpuHour`.',
      ],
    },
    {
      id: 'chinchilla',
      title: 'The Chinchilla frontier',
      instructions: `
\`scalingLoss(N, D, fit = CHINCHILLA)\` returns \`E + A / N^alpha + B / D^beta\`, reading the five constants off \`fit\` so the same code works for the paper's numbers and for the Besiroglu et al. 2024 re-fit.

\`chinchillaOptimal(C, fit)\` returns the \`{ N, D, loss, tokensPerParam }\` that minimise that loss subject to \`6 · N · D = C\`. Substituting \`D = C / (6N)\` turns it into a one-variable minimisation; setting the derivative to zero gives a closed form for \`N\`, and \`D\` follows from the constraint.

You may instead search \`N\` numerically (ternary search on the log of \`N\`, or a fine grid then a refinement) — the tests only require you to land within 3% of the true minimiser and to spend the budget exactly. Either way, \`6 · N · D\` must equal \`C\`, and \`loss\` must be \`scalingLoss(N, D, fit)\` at the point you return.

Sanity check while you work: for a symmetric fit (\`A = B\`, \`alpha = beta\`) the optimum must split the budget evenly, \`N = D = sqrt(C / 6)\`.

Throw an \`Error\` if \`C\` is not positive, as \`trainingFlops\` does.

Numbers to check against: at \`C = 1e23\`, \`CHINCHILLA\` (the paper's printed constants, and the default) gives about 14.6B parameters and 78 tokens per parameter; \`CHINCHILLA_REFIT\` gives about 29.5B parameters and 19 tokens per parameter. If you get 78 with the default fit, your closed form is right. The printed constants are what disagree with the paper's 20:1 rule.
`,
      hints: [
        'The constraint removes one variable. Write the loss as a function of N alone, with D replaced by C / (6N), and look at where its slope is zero.',
        'Differentiating `A·N^(-alpha) + B·(C/6N)^(-beta)` in N and setting it to zero gives `alpha·A·N^(-alpha-1) = beta·B·(6/C)^beta · N^(beta-1)`, which rearranges to a single power of N. If the algebra is unappealing, ternary-search `log10(N)` between 4 and 15 — the function is unimodal.',
        'Closed form: `const G = Math.pow((alpha * A) / (beta * B), 1 / (alpha + beta)); const N = G * Math.pow(C / 6, beta / (alpha + beta)); const D = C / (6 * N);` then return the loss and `D / N` alongside.',
      ],
    },
    {
      id: 'overtrain',
      title: 'Over-training, and why Llama-3-8B saw 15T tokens',
      instructions: `
Three functions that answer one question: is it worth over-training a small model?

\`tokensForLoss(N, targetLoss, fit)\` inverts \`scalingLoss\` in \`D\`. Solve \`B / D^beta = targetLoss − E − A / N^alpha\` for \`D\`. When that right-hand side is not positive the target is below what a model of that size can ever reach, however much data it sees — return \`Infinity\`, not a negative or \`NaN\`.

\`optimalForLoss(targetLoss, fit)\` returns the cheapest run that reaches the target: \`{ N, D, flops, loss }\`. Loss on the compute-optimal frontier falls monotonically as the budget \`C\` grows, so bisect on \`log10(C)\` — 100 iterations between 1e10 and 1e40 FLOPs is instant and exact enough — and return \`chinchillaOptimal\` at the budget you land on. Throw if \`targetLoss\` is at or below \`fit.E\`.

\`overtrainingAnalysis({ N, D, inferenceTokens, fit })\` compares an actual run against that optimum:

- \`loss\` — what the actual run reaches.
- \`trainingFlops\` — \`6·N·D\` of the actual run, so the caller does not have to recompute it.
- \`optimal\` — the \`optimalForLoss\` result for that same loss.
- \`extraTrainingFlops\` — how much training compute the actual run spent beyond the optimal budget, clamped at 0.
- \`savingPerInferenceToken\` — \`inferenceFlops(optimal.N) − inferenceFlops(N)\`, positive when the actual model is the smaller one.
- \`breakEvenInferenceTokens\` — extra training divided by saving per token, or \`Infinity\` when the saving is not positive.
- \`lifetimeFlops\` and \`lifetimeFlopsOptimal\` — training plus \`2 · N · inferenceTokens\` for each.

Run it on Llama-3-8B (\`N = 8e9\`, \`D = 15e12\`) and you get the argument Meta made out loud in the Llama 3 report: the extra training compute is recovered after roughly 1e13 served tokens.

The fit changes the size of the optimal model but not that conclusion. With the default \`CHINCHILLA\` constants the equal-loss optimum is about 26B parameters, Llama-3-8B spends about 2.0x its training compute, and break-even is near 9.9e12 tokens. With \`CHINCHILLA_REFIT\`, which the tests and the demo use here, the optimum is about 34B parameters, the factor is about 5.5x, and break-even is near 1.1e13.
`,
      predict: { question: 'Is an over-trained 8B model ever the wrong choice? When?', answer: 'When you serve few tokens. With `inferenceTokens = 0` the over-trained run is strictly worse — it spent more compute for the same loss. The trade only pays past the break-even point, which is why research checkpoints and production endpoints are trained differently.' },
      hints: [
        'Inverting `E + A/N^alpha + B/D^beta = L` in D is two rearrangements and one fractional power. Check the sign of what is left for the data term before taking that power.',
        'For `optimalForLoss`, bisect on the exponent rather than on C itself: budgets span thirty orders of magnitude, so a linear bisection on C would need far more iterations than one on log10(C).',
        '`let lo = 10, hi = 40; for (i < 100) { const mid = (lo + hi) / 2; if (chinchillaOptimal(10 ** mid, fit).loss > targetLoss) lo = mid; else hi = mid; }` then return `chinchillaOptimal(10 ** ((lo + hi) / 2), fit)` with its flops.',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why training costs 6 FLOPs per parameter per token while inference costs 2, and what has to be true of the architecture for that rule to hold.',
    'You are given 1e24 FLOPs. Walk through how you would choose N and D, then say which of your planner\'s assumptions (MFU, linear scaling, the fit constants) you would least trust, and what you would measure first to check it.',
    'Your planner says a 405B model needs at least 81 H100s just to hold its training state. Explain in your own words why memory, rather than FLOPs, is what forces a run onto a cluster — and what you would shard first.',
  ],
  stretch: [
    'Add mixture-of-experts accounting: DeepSeek-V3 has approximately 671B total parameters but activates approximately 37B per token, so the 6ND rule uses the active count while the memory bill uses the total. Add an `activeParams` option to `planRun` and re-price it.',
    'Add ZeRO stages to `trainingMemory` as DeepSpeed defines them: stage 1 shards optimizer states across data-parallel ranks, stage 2 also gradients, stage 3 also weights. Take a `dataParallel` count and report bytes per GPU.',
    'Add activation checkpointing: full recomputation reduces the stored activations to roughly `2 · seq · batch · hidden` bytes per layer at the cost of one extra forward pass (about 33% more training FLOPs). Find the batch size where checkpointing becomes the cheaper way to fit the run.',
    'Fit your own scaling law: train the module 07 GPT at four sizes on four token counts, fit `E + A/N^alpha + B/D^beta` by least squares on the log of the loss, and compare your exponents with Hoffmann et al. 2022. This is the experiment, in miniature.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
