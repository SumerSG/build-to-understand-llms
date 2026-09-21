export default {
  id: '04-bigram',
  title: 'From counting to learning: the bigram model',
  track: 'foundations',
  minutes: 90,
  threshold: 'A language model is a table of conditional probabilities over the next token; "training" replaces counting with gradient descent on the same objective (cross-entropy = negative log-likelihood), which is what lets it generalise when the table is too big to count.',
  goal: 'A count-based and a neural bigram language model on the character-level corpus; you measure perplexity for both, watch the neural table converge to the counts, and sample text from each.',
  prereqs: ['01-tensors', '02-autograd', '03-tokenizer'],
  recall: [
    { q: 'The `CharTokenizer` from module 03 builds its vocabulary from…', options: ['The 256 byte values', 'The sorted distinct characters of the training text', 'A fixed list of English letters'], answer: 1,
      why: 'On the lab corpus that is 71 characters, so both bigram tables in this module are 71 × 71 and every id is a row index.' },
    { q: 'In module 01, entry (i, j) of a row-major `[V, V]` table lives at flat offset…', options: ['`i + j`', '`i * V + j`', '`j * V + i`'], answer: 1,
      why: 'Every count, probability and logit lookup in this module is this one offset; getting it backwards transposes the model.' },
    { q: 'In module 02, `crossEntropy(logits, targets)` returned…', options: ['The sum of −log p over the targets', 'The mean of −log p over the targets', 'The product of the target probabilities'], answer: 1,
      why: 'Mean negative log-likelihood is the training loss of the neural bigram, and perplexity is exp of exactly that number.' },
    { q: 'After `loss.backward()` in module 02, a leaf tensor\'s `.grad` holds…', options: ['Only the most recent gradient', 'The sum of every gradient since it was last zeroed', 'A copy of its data'], answer: 1,
      why: 'Gradients accumulate with +=, so a training step must zero them before the next backward pass or it will step on stale gradients.' },
    { q: 'Module 00 measured that the most frequent word appears roughly how many times more often than the word at rank 10?', options: ['About 10×', 'About 2×', 'About 100×'], answer: 0,
      why: 'Zipf: frequency ≈ f(1)/rank. The same skew means a few rows of the bigram table hold thousands of counts and many hold almost none, which is where smoothing decides the held-out score.' },
  ],
  review: [
    { q: 'A perplexity of 20 on held-out text means…', options: ['The model is right one time in 20', 'Per token, the model is as uncertain as a fair 20-sided die', 'The loss is 20'], answer: 1,
      why: 'Perplexity is exp(mean −log p). A uniform choice among 20 options has −log p = log 20 at every step, so exp gives 20 back. It is a geometric-mean "effective branching factor".' },
    { q: 'With alpha = 0, a single held-out transition that never occurred in training makes the perplexity…', options: ['Slightly larger', 'Infinite', 'Unchanged'], answer: 1,
      why: 'That transition has probability exactly 0, −log 0 = ∞, and one infinite term makes the mean infinite. On the lab corpus 31 of 4,095 validation transitions are unseen.' },
    { q: 'Trained to convergence on every pair of the training ids, the neural bigram\'s `softmax(W)` equals…', options: ['The count table normalised row by row (alpha = 0)', 'A uniform table', 'The add-one smoothed table'], answer: 0,
      why: 'Both minimise the same mean −log p, and for a free table that minimum is unique: row i must equal the empirical distribution of what followed i. The demo measures the gap at about 0.001 per entry.' },
    { q: 'The reason to bother with the neural bigram, given that counting is exact and faster, is that…', options: ['It reaches a lower training perplexity', 'It needs no tokenizer', 'The same loss and gradient machinery still works when the context is too rich to count'], answer: 2,
      why: 'A count table needs one row per distinct context; with 1,024 tokens of context there are more contexts than atoms. `logits = W[x]` becomes `logits = f(context) · Wᵀ` and nothing else changes; module 05 starts building f.' },
    { q: 'As alpha grows without bound, the add-alpha table approaches…', options: ['All zeros', 'The uniform table, so perplexity approaches V', 'The unsmoothed counts'], answer: 1,
      why: 'Each entry tends to alpha / (alpha · V) = 1/V. Smoothing interpolates between the data (alpha = 0) and the prior (alpha → ∞); on the lab corpus the best alpha is about 0.1.' },
  ],
  concept: `
## A language model is a table

Fix a tokenizer with \`V\` tokens. By the chain rule a language model only has to supply \`P(next | everything so far)\`. A **bigram** model keeps one token of context, \`P(next | previous)\`: a table with \`V\` rows and \`V\` columns, row \`i\` being a distribution over what follows token \`i\`. With the lab's \`CharTokenizer\`, \`V = 71\`, so the whole model is 5,041 numbers. Andrej Karpathy's *makemore* starts here too.

Perplexity reads entries out of the table and sampling walks its rows; neither cares whether it came from counting or gradient descent, so you build it both ways.

## Maximum likelihood is counting

Let \`N[i, j]\` be how often token \`j\` followed token \`i\` in the training text. The log-likelihood of that text under a table \`P\` is \`Σ log P[xₜ, xₜ₊₁]\` over its \`n − 1\` transitions, and the table that maximises it is \`P[i, j] = N[i, j] / Σⱼ N[i, j]\`. Counting *is* maximum-likelihood training, in closed form. Step 1 builds it.

:::predict
The corpus has 71 distinct characters. What perplexity does a uniform table (every entry \`1/71\`) get on it, and roughly what will the count table get? Commit to two numbers.
---
Uniform: exactly 71, because every transition scores \`log 71\`. The count table reaches about 8.5 on the training text: one character of context cuts the effective choice from 71 ways to about 8.5.
:::

## Perplexity: the size of the model's die

The mean negative log-likelihood (NLL) per transition is \`−(1/(n−1)) Σ log P[xₜ, xₜ₊₁]\`; **perplexity** is \`exp(NLL)\`. Spreading probability evenly over \`k\` options at every step gives NLL \`log k\` and perplexity \`k\`, so "perplexity 20" means the model is as uncertain, on average, as a fair 20-sided die. It is a geometric mean, so one confident mistake (a transition at \`1e-6\` costs \`13.8\` nats against a typical 2) dominates. And it is per token *of the tokenizer used*: per-character and per-BPE-token numbers are not comparable. Step 2 keeps NLL and perplexity as separate functions because the NLL is the loss every later training run plots.

## Smoothing is a prior

A zero count claims a transition is impossible. The validation split contains 31 transitions the training split never shows, and one is enough to make the score infinite. **Add-alpha smoothing** pretends every cell was seen \`alpha\` extra times: \`P[i, j] = (N[i, j] + alpha) / (Σⱼ N[i, j] + alpha · V)\`. Formally these are pseudo-counts from a Dirichlet prior; informally, \`alpha\` sets how much you trust the data over uniform ignorance (\`alpha → ∞\` gives the uniform table). \`alpha = 1\` (Laplace) is far too much here; the demo finds the best value near 0.1.

:::predict
You evaluate the alpha = 0 count table on the validation text. What number comes back?
---
\`Infinity\`. One unseen transition has \`P = 0\` and \`−log 0 = ∞\`. The fix is not a bigger corpus (there is always a next unseen pair) but a prior.
:::

Before 2012 the production form of this idea, **Kneser–Ney smoothing** over 5-gram counts (Chen & Goodman 1999), sat inside every speech recogniser and translation system, at Google over roughly two trillion tokens (Brants et al. 2007). Counting scaled; the context did not.

## Replacing counting with gradient descent

Now build the same table a second way. Keep a \`[V, V]\` matrix of **logits** \`W\`, take \`logits = W[x]\` (row \`x\`, an embedding lookup; equivalently \`onehot(x) · W\`), softmax the row, and minimise the mean cross-entropy against the next token, which is the NLL of step 2 written as a loss. The gradient for row \`i\` is \`(softmax(W[i]) − onehot(target)) / N\`, averaged over every time \`i\` was the context; it is zero exactly when \`softmax(W[i])\` equals the empirical distribution of what followed \`i\`. Gradient descent has one place to stop, and it is the count table. Steps 4 and 5 build and train it; the demo overlays the two heatmaps.

:::predict
\`W\` starts as Gaussian noise with standard deviation 0.01. What is the first loss value, before any training step?
---
About \`log 71 ≈ 4.26\`. Near-zero logits make every row nearly uniform, and uniform scores \`log V\` on anything. Every loss curve in this lab starts near \`log V\`: a sanity check worth keeping.
:::

Why take the slow road to the same table? Because \`W[x]\` is the only part that assumes one token of context. Replace the lookup by any differentiable \`h = f(context)\` and write \`logits = h · Wᵀ\`: loss, gradient and optimiser are untouched. That expression is GPT's output layer, with \`f\` a stack of transformer blocks and \`W\` the tied token-embedding table. A count table for 1,024 tokens of context would need \`V^1024\` rows; gradients do not care. The neural version also smooths implicitly: after 600 steps unseen transitions keep small non-zero probabilities, so its validation perplexity is finite without any alpha, though slightly worse than the best-smoothed counts because nobody tuned that implicit alpha.

## Where this toy differs from production

The vocabulary is 71 characters; GPT-2 uses 50,257 byte-level BPE tokens, so its output table alone is \`50,257 × 768\` numbers and its perplexity is per token. Classical n-gram toolkits (SRILM, KenLM) stored billions of 5-gram counts with back-off, not a dense table. The neural model here is a bare table trained with AdamW for 600 minibatch steps; a real run (module 07) adds a schedule, clipping, validation and checkpoints. And a bigram, counted or learned, writes text that looks like letters, not words: that is what one character of context buys; the rest of the curriculum closes the gap.
`,
  steps: [
    {
      id: 'count',
      title: 'Count bigrams and normalise with add-alpha smoothing',
      instructions: `
Two functions, both returning a raw tensor \`{ shape: [V, V], data: Float32Array }\`.

\`countBigrams(ids, V)\`: \`data[i * V + j]\` is how many times token \`j\` followed token \`i\` in \`ids\`. An array of \`n\` ids has \`n − 1\` transitions; do not wrap around from the last id to the first. The loop is written; complete its body.

\`bigramProbs(counts, alpha = 0)\`: each row becomes a probability distribution with add-alpha smoothing:

\`\`\`
P[i, j] = (counts[i, j] + alpha) / (rowTotal[i] + alpha * V)
\`\`\`

If the denominator is 0 (a row that was never seen and \`alpha = 0\`), fill that row with \`1 / V\` so the table stays a valid distribution. Do not modify \`counts\`. The row-total loop is written; complete the normalisation.

Why smoothing lives here and not in evaluation: the table *is* the model. Every consumer of it (perplexity, sampling, the heatmap in the demo) should see probabilities that already reflect the prior, so the choice of \`alpha\` is made once, at training time, exactly as a real n-gram toolkit does.
`,
      hints: [
        'One transition is one increment of one cell. The row is the previous token, the column the next; the flat-offset rule from module 01 turns (row, column) into an index into `data`.',
        'Inside the loop: add 1 to `data[ids[t] * V + ids[t + 1]]`. For the probabilities, compute `denom = total + alpha * V` once per row, then each entry is `(count + alpha) / denom`, with a guard that writes `1 / V` when `denom` is 0.',
        '`const denom = total + alpha * V; for (let j = 0; j < V; j++) out[i * V + j] = denom > 0 ? (… + alpha) / denom : 1 / V;` — the elided term is the count in cell (i, j) of the input table.',
      ],
    },
    {
      id: 'evaluate',
      title: 'Negative log-likelihood and perplexity',
      instructions: `
\`negLogLikelihood(probs, ids)\`: the mean of \`−log P[ids[t], ids[t + 1]]\` over the \`n − 1\` transitions in \`ids\`, where \`P\` is a \`[V, V]\` probability table from step 1 (or, later, from the neural model). Use the natural log. A transition with probability 0 contributes \`Infinity\`, and the correct answer is then \`Infinity\`: do not clamp or skip it. The tests check that alpha = 0 on held-out text gives exactly that.

\`perplexity(nll)\`: \`exp(nll)\`.

Two functions rather than one because the NLL is what every training loop in this lab plots (it is the cross-entropy loss), and perplexity is how every paper reports it. Keeping both in view stops you from confusing a loss of 2.1 with a perplexity of 2.1.
`,
      predict: { question: 'What perplexity does a table that gives probability 1 to the right next token at every transition get?', answer: '1: every term is −log 1 = 0, the mean is 0, and exp(0) = 1. Perplexity is bounded below by 1 and above by nothing.' },
      hints: [
        'The sum runs over transitions, not ids: `n` ids give `n − 1` pairs. The probability of a transition sits at `probs.data[prev * V + next]`, with `V = probs.shape[1]`.',
        'Loop `t` from 0 while `t + 1 < ids.length`, subtract `Math.log(p)` from a running total, and divide by `ids.length - 1` at the end. `Math.log(0)` is `-Infinity`, which is the right answer, not a bug to work around. Perplexity is one call to `Math.exp`.',
        '`let total = 0; for (let t = 0; t + 1 < ids.length; t++) total -= Math.log(probs.data[…]); return total / (ids.length - 1);` — the elided index is `ids[t] * V + ids[t + 1]`.',
      ],
    },
    {
      id: 'sample',
      title: 'Sampling from a row, with temperature',
      instructions: `
\`sampleNext(probs, prev, next, temperature = 1)\`: draw one token id from row \`prev\` of the table. \`next\` is a seeded rng function from \`lib/util.js\` (\`rng(seed)\`), so draw \`u = next()\` **once** and invert the cumulative distribution: walk the row, accumulating probabilities, and return the first index whose running sum exceeds \`u\` (\`u < acc\`). If float rounding leaves the sum a hair below 1, return \`V − 1\`.

Temperature reshapes the row before the draw: \`p[j] ∝ row[j]^(1/T)\`, renormalised to sum to 1. \`T = 1\` leaves the row alone; \`T → 0\` concentrates on the most likely token; \`T → ∞\` flattens towards uniform. (Applied to probabilities this is the same as \`softmax(logits / T)\` applied to logits, which is how module 14 will do it.)

\`generate(probs, start, n, next, temperature = 1)\`: sample \`n\` tokens in a chain, each conditioned on the previous one, starting from \`start\`; return the \`n\` new ids (not \`start\`).

The worked helper \`rowOf(table, i)\` gives you the row as a typed-array view. \`lib/util.js\` has a \`sampleIndex(probs, u)\` you can compare against, but write the loop yourself: it is the same three lines that top-k, top-p and speculative decoding will all build on.
`,
      hints: [
        'One uniform draw picks a token. Picture the row as segments laid end to end on [0, 1): the draw lands in exactly one segment. Temperature changes the segment widths before the draw, never the draw itself.',
        'When `temperature !== 1`, copy the row into a fresh Float32Array with `Math.pow(row[j], 1 / temperature)` and divide by its sum. Then `acc += p[j]; if (u < acc) return j;`. `generate` is a loop that feeds each sample back in as `prev` and pushes it.',
        '`const u = next(); let acc = 0; for (let j = 0; j < V; j++) { acc += p[j]; if (…) return j; } return V - 1;` — the elided condition compares `u` with the running sum, strictly.',
      ],
    },
    {
      id: 'neural',
      title: 'The neural bigram: a table of logits',
      instructions: `
The neural model is one parameter, a \`[V, V]\` \`Tensor\` of logits \`W\`, and it produces the same kind of \`[V, V]\` probability table as step 1 once you take a softmax of every row.

\`initNeural(V, next, std = 0.01)\`: return \`{ V, W }\` where \`W\` is Gaussian noise with the given standard deviation and \`requiresGrad = true\`. Small noise means every row starts close to uniform, so the first loss is close to \`log V\`.

\`neuralLogits(model, xs)\`: for an array of context ids \`xs\`, the logits are the rows \`W[xs[i]]\`, as a \`Tensor\` of shape \`[xs.length, V]\`. This is an embedding lookup, and it must stay attached to the autograd graph so that gradients flow back into the rows that were used. \`Tensor.prototype.embed(ids)\` from \`lib/tensor.js\` does exactly that (its backward scatter-adds into \`W.grad\`); \`ops.embed\` from module 01 would silently break the chain.

\`neuralLoss(model, xs, ys)\`: the mean cross-entropy of the targets \`ys\` under those logits, as a scalar \`Tensor\`. \`crossEntropy(logits, targets)\` from \`lib/tensor.js\` computes \`mean(−log softmax(logits)[target])\` with a fused, stable log-softmax. Compare it with step 2: it is the same number, with the table replaced by \`softmax(W)\`.

Why the lookup is a matmul in disguise: \`W[x]\` equals \`onehot(x) · W\`, a \`[1, V] × [V, V]\` product. GPT's output layer is \`h · Wᵀ\` with \`h\` a learned vector instead of a one-hot; the one-hot is the special case where the context is a single token and nothing is learned about it.
`,
      hints: [
        'The whole model is one Tensor. The row lookup you wrote as a raw op in module 01 has an autograd twin on `Tensor` that records which rows were read, so backward can add gradient into just those rows.',
        '`Tensor.randn(shape, next, std, opts)` builds the parameter when `opts` marks it trainable; `model.W.embed(xs)` gives `[xs.length, V]` logits; `crossEntropy(logits, ys)` gives the mean NLL as a scalar Tensor.',
        '`initNeural`: `return { V, W: Tensor.randn([V, V], next, std, { … }) };` with the option that makes W a trainable leaf elided. `neuralLoss`: `return crossEntropy(neuralLogits(model, xs), ys);`.',
      ],
    },
    {
      id: 'train',
      title: 'Train it and compare with the counts',
      instructions: `
\`trainStep(model, opt, xs, ys)\`: one optimiser step on one batch. Zero the gradients, compute \`neuralLoss\`, call \`backward()\`, call \`opt.step()\`, and return the loss as a plain number (\`loss.item()\`). The order matters: gradients accumulate (module 02), so a step that forgets to zero them steps on the sum of every gradient so far; the tests compare two steps against a reference that zeroes correctly.

\`trainNeural(model, ids, { steps = 200, batchSize = 512, lr = 0.1, next })\`: build one \`AdamW\` over \`[model.W]\` (from \`lib/optim.js\`), then for each step draw a batch with the worked helper \`makeBatch(ids, batchSize, next)\` and call \`trainStep\`. Return the array of per-step losses.

\`neuralProbs(model)\`: the trained model as a probability table, \`softmax\` of every row of \`W\`, returned as a raw \`{ shape, data }\` tensor so that \`negLogLikelihood\`, \`perplexity\` and \`generate\` from the earlier steps accept it unchanged. Wrap the softmax in \`noGrad\` so evaluation does not record a graph.

The last test trains on every pair of a small corpus for 300 full-batch steps and checks that \`neuralProbs\` matches \`bigramProbs(counts, 0)\` to within 0.02 in every cell. That is the threshold concept of the module made concrete: gradient descent on cross-entropy converges to the count table, because both are maximising the same likelihood.
`,
      predict: { question: 'In the demo the count model with alpha = 0.1 scores about 8.9 validation perplexity. Will the neural model, after 600 minibatch steps and no explicit smoothing, score lower or higher?', answer: 'Higher, about 9.3. Finite training leaves the unseen transitions with tiny but non-zero probabilities, which behaves like a very small alpha (under-smoothed), and minibatch noise keeps the frequent rows slightly off the counts. On the training text the two are within 0.1 perplexity of each other.' },
      hints: [
        'The four lines of the training loop from module 02, in order: clear gradients, forward, backward, step. Build the optimiser once, outside the step loop; AdamW keeps running moment estimates that must survive across steps.',
        'Zero the gradients with `opt.zeroGrad()`, compute the loss, call `backward()`, call `opt.step()`, and return `loss.item()`. In `trainNeural`, `new AdamW([model.W], { lr })` once, then `makeBatch` and `trainStep` per step, pushing each loss. For the table, `noGrad(() => model.W.softmax())` and copy out `shape` and `data`.',
        '`const opt = new AdamW([model.W], { lr }); const losses = []; for (let s = 0; s < steps; s++) { const { xs, ys } = makeBatch(ids, batchSize, next); losses.push(…); } return losses;` — the elided call is your `trainStep`. `neuralProbs`: `const p = noGrad(() => model.W.…()); return { shape: p.shape.slice(), data: p.data };`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why the trained neural bigram ends up equal to the count table, and what changes, and what does not, when the context becomes eight tokens instead of one.',
    'The count model scored about 8.9 perplexity per character on validation text. Say concretely what that number means, and why it cannot be compared with a perplexity reported for GPT-2.',
    'Add-alpha smoothing and stopping gradient descent early both keep unseen transitions from having probability zero. In what sense are they the same idea, and which one would you rather tune?',
  ],
  stretch: [
    'Build a trigram count model as a `[V, V, V]` table (71³ ≈ 358k cells) and measure how many validation contexts are unseen. That data sparsity is what Kneser–Ney smoothing (Chen & Goodman 1999; the default in KenLM and SRILM) was designed around.',
    'Replace `W.embed(xs)` by an explicit one-hot matmul, `onehot(xs) · W`, using `Tensor.matmul`, and check that logits and gradients are identical. GPT-2 ties its output projection to the embedding table (`logits = h · wteᵀ`, Press & Wolf 2017), which is this identity run in reverse.',
    'Implement Kneser–Ney (absolute discounting plus continuation counts) for the bigram and compare validation perplexity with the best add-alpha value; this was the state of the art in language modelling until neural models overtook it around 2012.',
    'Train the neural bigram with `SGD` instead of `AdamW` at several learning rates and overlay the loss curves. Rows for rare tokens receive tiny gradients under SGD; per-parameter normalisation is why Adam-family optimisers (Kingma & Ba 2015; Loshchilov & Hutter 2019) are the default for every LLM training run.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
