export default {
  id: '28-moe',
  title: 'Mixture of experts',
  track: 'transformer',
  minutes: 105,
  threshold: 'A model can have far more parameters than any token touches: routing each token to a few experts decouples capacity (total parameters) from cost (active parameters per token), and the whole design problem becomes keeping the experts evenly used.',
  goal: 'A sparse mixture-of-experts layer (top-k router, expert MLPs, load-balancing loss, capacity and dropped tokens) swapped into your GPT and compared with a dense block at equal active parameters.',
  prereqs: ['02-autograd', '06-transformer', '07-pretraining', '08-scaling'],
  recall: [
    { q: 'In the module-06 GPT block with width C, how are the 12·C² parameters split?', options: ['6C² attention, 6C² MLP', '4C² attention, 8C² MLP', '8C² attention, 4C² MLP'], answer: 1,
      why: 'qkv (3C²) plus proj (C²) is 4C²; the MLP widens to 4C and back, 8C². Two thirds of every block sits in the MLP, which is exactly the part a mixture of experts multiplies.' },
    { q: 'For a dense model with P parameters, the forward pass costs roughly how many FLOPs per token (module 08)?', options: ['P', '2·P', '6·P'], answer: 1,
      why: 'Each weight is used once per token in a multiply and an add. 6·P is forward plus backward during training. In an MoE, P becomes the ACTIVE parameter count, not the total.' },
    { q: 'In module 02, you build a new Tensor from numbers copied out of `p.data` and use it in the loss. What gradient reaches `p`?', options: ['The same as if you had used p directly', 'None: the copy has no recorded parent, so the graph stops there', 'Half of it'], answer: 1,
      why: 'Autograd differentiates only the operations it recorded. In this module the router learns only if the gates are built from probs with Tensor ops; copying their values out cuts the router off from the loss.' },
    { q: 'In module 07 your training loss becomes `lm + 0.01 · aux` before `backward()`. What gradient does each weight receive?', options: ['Only the gradient of lm; aux is just logged', 'The gradient of lm plus 0.01 times the gradient of aux', 'The gradient of whichever term is larger'], answer: 1,
      why: 'Backward is linear: the gradient of a weighted sum is the weighted sum of the gradients. That is how the load-balancing loss steers the router without swamping the language-model signal.' },
  ],
  review: [
    { q: 'Mixtral 8x7B has 8 experts per layer and routes each token to 2. Approximately how many parameters does one token use?', options: ['About 47B, all of them', 'About 13B', 'About 14B = 2 × 7B'], answer: 1,
      why: 'Mistral reports approximately 47B total and 13B active. Only the expert MLPs are replicated; attention, embeddings and norms are shared, so neither total (not 8 × 7 = 56B) nor active (not 2 × 7 = 14B) is a simple multiple.' },
    { q: 'The Switch load-balancing loss E · Σ f_i · P_i equals 1 when…', options: ['One expert takes every token', 'Tokens and router probability are spread evenly, f_i = P_i = 1/E', 'The router is untrained'], answer: 1,
      why: 'E · E · (1/E)(1/E) = 1. With every token and all the probability on one expert it rises to E. The loss is minimised by spreading, which is the point.' },
    { q: 'You raise the capacity factor from 1.0 to 2.0. What happens?', options: ['Fewer tokens are dropped, but each expert reserves twice the slots (memory and padded compute)', 'More tokens are dropped', 'The router learns faster'], answer: 0,
      why: 'Capacity = ceil(factor · N · k / E). A larger buffer absorbs imbalance but costs memory and wasted compute on padding in fixed-shape kernels. Switch Transformer used factors around 1.0–1.25.' },
    { q: 'With k = 1 and gates renormalised over the chosen expert, what gradient does the language-model loss send to the router?', options: ['The same as with k = 2', 'None: the single gate is always exactly 1', 'A doubled gradient'], answer: 1,
      why: 'p / p = 1 regardless of the logits, so the output does not depend on the router weights. Switch Transformer multiplies by the raw probability p for top-1 so the router still learns from the task loss.' },
    { q: 'DeepSeek-V3\'s "auxiliary-loss-free" balancing works by…', options: ['Dropping tokens until experts are even', 'Adding a per-expert bias to the scores used for top-k selection only, nudged down for overloaded experts and up for idle ones', 'Using top-1 routing'], answer: 1,
      why: 'The bias changes which experts are chosen but not the gate values that weight the outputs, so balancing no longer competes with the language-model gradient.' },
  ],
  concept: `
## Capacity is not cost

In the GPT from module 06, every token passes through every weight. Parameters (what it can store) and FLOPs per token (what it costs) are tied: about 2 FLOPs per parameter (module 08). A **mixture of experts** (MoE) breaks that link. Replace the block's MLP with \`E\` independent MLPs, the **experts**, plus a small **router** that sends each token to only \`k\` of them. Total parameters grow roughly \`E\`-fold; the work per token grows only \`k\`-fold.

Shazeer et al. (2017) introduced the sparsely gated MoE layer between LSTM layers. GShard (2020) and Switch Transformer (Fedus, Zoph & Shazeer, 2021) brought it to transformers; Switch routes each token to one expert and reports up to approximately 7× faster pre-training than a dense T5-Base at equal FLOPs per token. Google's GLaM (approximately 1.2T parameters, 64 experts) and xAI's Grok-1 (approximately 314B, 8 experts) route each token to 2. Mixtral 8x7B (Mistral AI, December 2023) uses 2 of 8 experts, approximately 13B of its 47B parameters per token. DeepSeek-V3 (2024) has 256 small routed experts plus one always-on **shared expert** per layer, picks 8, and activates approximately 37B of 671B parameters.

## The router

The router is a single matrix \`W_g\` of shape \`[C, E]\` (\`C\` is the model width). For a token vector \`x\`, \`probs = softmax(x · W_g)\`. Keep the \`k\` largest, then **renormalise** them to sum to 1 (Mixtral does this); these are the **gates**. The layer's output is \`Σ over chosen e of gate_e · expert_e(x)\`. If \`k = E\` nothing is discarded and you have a dense, softmax-weighted ensemble.

Top-k selection has no gradient; the router learns only through the gate values that multiply the expert outputs.

:::predict
Set \`k = 1\` and renormalise the gates. What gradient does the language-model loss send to \`W_g\`?
---
None. The single gate is \`p / p = 1\` whatever the logits are, so the output does not depend on the router. Switch Transformer avoids this by multiplying the top-1 expert's output by the raw probability \`p\`. With \`k = 2\`, the ratio between the two chosen experts still carries a gradient.
:::

## Dispatch and combine

Running experts token by token costs \`N · k\` tiny matrix products for \`N\` tokens. Instead, **dispatch**: group the tokens by expert, run each expert once on its batch, multiply by the gates, and **combine** (scatter) the results back. The worked \`naiveMoE\` is the per-token definition your \`dispatchCombine\` must match.

## Collapse and the balancing loss

Whichever expert happens to get more tokens early trains faster, gets better, and attracts more tokens. Left alone the router **collapses** onto a few experts; the rest are dead weight. Switch Transformer adds an auxiliary loss, \`aux = E · Σ_i f_i · P_i\`, where \`f_i\` is the fraction of routing assignments that went to expert \`i\` and \`P_i\` is the mean router probability for expert \`i\` over the batch. \`f\` is a count and has no gradient; the gradient flows through \`P\` and pushes probability away from the overloaded experts. It joins the language-model loss with a small coefficient (Switch used 0.01).

:::predict
With \`E = 4\`, what is \`aux\` when routing is perfectly balanced, and what is it when every token goes to expert 0 with probability 1?
---
Balanced: \`f_i = P_i = 1/4\`, so \`4 · 4 · (1/16) = 1\`. Collapsed: \`f_0 = P_0 = 1\`, so \`4 · 1 = 4 = E\`. In practice the loss sits between 1 and \`E\`; the demo shows yours staying near 1 with the loss on, and climbing without it.
:::

DeepSeek-V3 instead uses **auxiliary-loss-free** balancing: a per-expert bias, added only to the scores used for selection, is lowered after each step for overloaded experts and raised for idle ones. (It keeps a much smaller sequence-level balance loss as a safety net.)

## Capacity and dropped tokens

A fixed-shape kernel needs a fixed buffer per expert. The **capacity** is \`ceil(factor · N · k / E)\`: the balanced share times a **capacity factor**. Tokens past an expert's capacity are **dropped** there; a token dropped by all its experts reaches the next layer through the residual connection alone. A factor of \`E\` never drops anything; Switch used factors around 1.0–1.25. DeepSeek-V3 reports dropping no tokens at all.

## Why MoE is a memory and communication problem

FLOPs fall, bytes do not. At inference every expert must be resident, because the next batch may route anywhere: Mixtral needs memory for 47B parameters while doing the arithmetic of 13B. At the batch sizes servers typically run, decoding is limited by how fast the weights can be read from memory, not by arithmetic (module 15, later on the path, works through why, and module 23 draws it on the roofline). In a large decode batch nearly every expert is hit, so each step still reads nearly all 47B parameters: the FLOPs of a 13B model, the memory traffic of a 47B one. Once the experts no longer fit on one GPU they are sharded (**expert parallelism**), and each MoE layer costs two all-to-all exchanges, one to send every token to the GPUs holding its experts and one to bring the results back; module 25, later on the path, prices them. That traffic is why DeepSeek-V3 caps each token at 4 nodes.

## Where the toy differs from production

Your dispatch is a JavaScript loop that gathers rows with \`embed\` and scatters them back with a one-hot matmul, on 4 experts of width 64. Production kernels **permute** the tokens (sort them by expert id), run all experts in one **grouped GEMM** (MegaBlocks, vLLM's fused MoE kernel, CUTLASS grouped GEMM), then un-permute, with experts spread across GPUs. Real experts are SwiGLU MLPs in bf16 or fp8, across dozens of layers. At our scale of one layer and 150 steps, MoE does not reliably beat the dense block: the advantage reported in the papers needs many more tokens per expert than a browser can train on.
`,
  steps: [
    {
      id: 'router',
      title: 'The router: softmax, top-k, renormalised gates',
      instructions: `
Two pieces.

\`topKIndices(row, k)\`: the indices of the \`k\` largest values of \`row\` (a plain array or a \`Float32Array\`), **largest first**, ties going to the **lower index**. Deterministic tie-breaking matters: the tests compare your routing against theirs.

\`Router.forward(x)\` for \`x\` a Tensor \`[N, C]\` returns \`{ probs, topIdx, gates }\`:

- \`probs\`: Tensor \`[N, E]\`, the full \`softmax(x · W_g)\` over all E experts (step 3's balancing loss needs the probability on the experts that were not chosen too). \`this.gate\` is a bias-free \`Linear(C, E)\`; use its \`forward\` and Tensor \`softmax()\` (last axis).
- \`topIdx\`: \`number[][]\`, for each token \`topKIndices\` of its probs row with \`this.k\`.
- \`gates\`: Tensor \`[N, E]\`, zero outside each token's top-k and \`p_e / Σ_{chosen} p\` inside.

\`gates\` must stay on the autograd graph, since it is the router's only source of gradient from the language-model loss. Build a constant 0/1 mask as a raw tensor (\`ops.zeros([N, E])\`, set the chosen entries to 1), wrap it in \`new Tensor(mask)\`, multiply, and divide by the row sum (\`sum(-1, true)\` keeps the \`[N, 1]\` shape for broadcasting).
`,
      predict: { question: 'With k = E, what will `gates` equal?', answer: 'Exactly `probs`: the mask is all ones and each row already sums to 1. A mixture of experts with k = E is a dense, softmax-weighted ensemble, and the test checks precisely that.' },
      hints: [
        'Selection (which experts) is plain JavaScript on numbers; weighting (how much) must be Tensor ops so gradients reach `W_g`. Keep the two separate.',
        'For `topKIndices`: make the list `[0, 1, …, len−1]`, sort it by value descending with index ascending as the tie-breaker, and slice the first k. For `forward`: compute probs, loop over tokens reading `probs.data.subarray(t*E, (t+1)*E)`, record the top-k and set those mask entries to 1.',
        '`idx.sort((a, b) => /* value descending, then index ascending */);` … then in forward: `const kept = probs.mul(new Tensor(mask)); const gates = /* divide kept by its row sum, keepDims */;`',
      ],
    },
    {
      id: 'dispatch',
      title: 'Dispatch and combine',
      instructions: `
\`groupByExpert(topIdx, nExperts)\` returns an array of \`E\` lists: \`lists[e]\` holds the token indices routed to expert \`e\`, in increasing token order. An expert with no tokens gets an empty list.

\`dispatchCombine(x, gates, lists, experts)\` with \`x\` \`[N, C]\`, \`gates\` \`[N, E]\`: for every expert with a non-empty list,

1. **gather** its tokens: \`gatherRows(x, rows)\` gives \`[n_e, C]\`;
2. run the expert **once** on that batch;
3. multiply each output row by that token's gate for this expert: gather the gate column \`gates.slice(1, e, e + 1)\` (shape \`[N, 1]\`) at the same rows, giving \`[n_e, 1]\`, and multiply (it broadcasts);
4. **scatter** back with \`scatterRows(y, rows, N)\` (\`[N, C]\`, zero elsewhere) and add into the running total.

Return a Tensor \`[N, C]\` (all zeros if no expert has tokens). The test checks the result against the worked \`naiveMoE\`, counts expert calls, and checks that gradients reach both \`x\` and the router.
`,
      predict: { question: 'With N = 12 tokens, E = 4 experts and k = 2, how many rows do the experts process in total, and how many expert calls does dispatch make at most?', answer: '24 rows (N·k) in at most 4 calls, one per expert. The per-token loop makes 24 calls of one row each. Same arithmetic, far fewer and larger matrix products, which is what a GPU wants.' },
      hints: [
        'The worked helpers already do the hard autograd parts: `gatherRows` is an embedding lookup and `scatterRows` is a one-hot matmul. Your job is the loop around them.',
        'Loop e over experts; skip empty lists; y = expert(gathered rows); g = gathered gate column; placed = scatter(y · g); out = out ? out.add(placed) : placed.',
        '`const y = experts[e].forward(gatherRows(x, rows)); const g = gatherRows(gates.slice(1, e, e + 1), rows); const placed = /* scatter y.mul(g) back to N rows */;`',
      ],
    },
    {
      id: 'balance',
      title: 'The load-balancing loss',
      instructions: `
\`routingFractions(topIdx, nExperts)\` returns \`f\`, a plain array: \`f[i]\` is the number of assignments to expert \`i\` divided by the total number of assignments \`N · k\`, so \`f\` sums to 1.

\`loadBalanceLoss(probs, topIdx)\` returns the scalar Tensor \`E · Σ_i f_i · P_i\`, where \`P = probs.mean(0)\` (shape \`[E]\`, the mean router probability per expert) and \`E = probs.shape[1]\`. Wrap \`f\` in a constant Tensor; the gradient flows only through \`P\`.

Dividing \`f\` by \`N · k\` (not \`N\`) is our convention so that a perfectly balanced router scores exactly 1 for any \`k\`. Hugging Face's Mixtral implementation divides by \`N\`, so its balanced value is \`k\`; the gradient direction is the same.
`,
      predict: { question: 'Expert 0 receives every token. Which way does the gradient of this loss push expert 0\'s router logits?', answer: 'Down. dLoss/dP_0 = E · f_0 is the largest of the E partial derivatives, and softmax passes that on as a positive gradient on logit 0 and negative gradients on the others, so gradient descent shifts probability toward the idle experts.' },
      hints: [
        'Two different averages: `f` averages the hard routing decisions (counts), `P` averages the soft probabilities. The loss is their dot product, scaled by E.',
        'Count assignments into an array of E zeros, divide each by the total. For the loss: `P = probs.mean(0)`, multiply elementwise by a Tensor made from f, sum, scale by E.',
        '`const f = new Tensor(ops.raw([E], routingFractions(topIdx, E))); return /* mean over tokens */.mul(f).sum().scale(E);`',
      ],
    },
    {
      id: 'capacity',
      title: 'Capacity factor and dropped tokens',
      instructions: `
\`expertCapacity(nTokens, nExperts, k, factor)\` returns \`ceil(factor · nTokens · k / nExperts)\`: the balanced share of the \`N · k\` assignments, times the capacity factor, rounded **up**.

\`applyCapacity(lists, capacity)\` returns \`{ kept, dropped }\`. \`kept[e]\` is the first \`capacity\` entries of \`lists[e]\` (earlier tokens have priority, as in Switch Transformer). \`dropped\` is the total number of assignments cut off. Do not modify \`lists\`: the layer still reports the pre-capacity counts.

A dropped assignment simply produces no output from that expert. \`dispatchCombine\` already handles that, because the token is no longer in the expert's list.
`,
      hints: [
        'The maximum any one expert can receive is N (a token picks an expert at most once), so a factor of E, giving N·k slots, can never overflow.',
        'Capacity is one `Math.ceil`. For applyCapacity, `map` over the lists with `slice(0, capacity)` (which copies) and add `max(0, length − capacity)` to a counter.',
        '`const kept = lists.map((rows) => { dropped += /* overflow of this list */; return rows.slice(0, capacity); });`',
      ],
    },
    {
      id: 'moe',
      title: 'The MoE layer and its accounting',
      instructions: `
\`MoE.forward(x)\` assembles everything. \`x\` is \`[..., C]\` (the block passes \`[B, T, C]\`):

1. flatten to \`[N, C]\` with \`reshape\` (\`N = x.size / C\`);
2. \`this.router.forward\`, then \`groupByExpert\`;
3. capacity: if \`this.capacityFactor\` is finite use \`expertCapacity(N, E, k, factor)\`, otherwise \`N\` (no limit); apply it;
4. set \`this.lastAux = loadBalanceLoss(probs, topIdx)\` and \`this.lastStats = { counts, dropped, assignments: N · k, capacity }\` where \`counts[e]\` is the number routed to expert \`e\` **before** capacity;
5. \`dispatchCombine\` with the kept lists, then reshape back to \`x.shape\`.

\`countParams({ nEmbd, hidden, nExperts, k })\` accounts for one MoE layer: an expert has \`2·C·H + H + C\` parameters, the router \`C·E\`. Return \`{ total, active, flopsPerToken }\` with \`total = E·expert + router\`, \`active = k·expert + router\` and \`flopsPerToken = 2·active\`.

The worked \`MoEBlock\` and \`moeGPT\` below your code swap this layer into the lib GPT, collect its parameters and sum \`lastAux\` over layers. The demo trains it.
`,
      hints: [
        'Every piece already exists; this step is wiring, plus remembering that statistics must be recorded before the capacity cut hides them.',
        'shape → flat → route → lists → capacity (or N) → applyCapacity → lastAux, lastStats → dispatchCombine(flat, gates, kept, this.experts) → reshape(shape). For countParams write `expert` and `router` first.',
        '`const cap = Number.isFinite(this.capacityFactor) ? expertCapacity(N, this.nExperts, this.k, this.capacityFactor) : N; const { kept, dropped } = applyCapacity(lists, cap); …`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague how Mixtral can have approximately 47B parameters yet cost about as much per token as a 13B dense model, and what that does NOT make cheaper when you serve it.',
    'Describe expert collapse as a feedback loop, and explain how the loss E · Σ f_i · P_i breaks it even though f has no gradient. What did the no-aux run in the demo show?',
    'The capacity factor trades dropped tokens against wasted memory and compute. Which side would you favour for training, and which for serving with a latency target, and why?',
  ],
  stretch: [
    'Replace the auxiliary loss with DeepSeek-V3\'s bias-based balancing: keep a per-expert bias added to the scores for top-k selection only, and after each step subtract γ from overloaded experts and add γ to idle ones. Compare utilisation and loss with the Switch loss.',
    'Add a DeepSeekMoE-style shared expert that every token passes through, alongside finer-grained routed experts (e.g. 8 experts of width C with top-4 instead of 4 of width 2C with top-2), at the same active parameters.',
    'Implement expert-choice routing (Zhou et al., Google, 2022): each expert picks its top-capacity tokens instead of tokens picking experts. Load is balanced by construction; what happens to tokens nobody picks, and why is this awkward for autoregressive decoding?',
    'Replace the one-hot scatter with a permutation: sort the N·k assignments by expert id, run the experts on contiguous slices, and un-permute. This is the data layout behind MegaBlocks and vLLM\'s fused MoE kernel; measure the speedup in the demo.',
  ],
  timeouts: { tests: 20000, demo: 120000 },
};
