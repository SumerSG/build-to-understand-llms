export default {
  id: '07-pretraining',
  title: 'The pre-training loop',
  track: 'transformer',
  minutes: 120,
  threshold: 'Pre-training is one loop: sample a batch of token windows, predict every next token at once, take an AdamW step on the mean cross-entropy, repeat; every trick (schedules, clipping, mixed precision) exists to keep that loop stable and efficient.',
  goal: 'A training loop (batches, AdamW, warmup+cosine schedule, gradient clipping, eval) that trains a small GPT in your browser until it generates recognisable text.',
  prereqs: ['02-autograd', '04-bigram', '06-transformer'],
  recall: [
    { q: 'In module 02, what does `loss.backward()` do to a leaf tensor\'s `.grad` when it already holds a value?', options: ['Overwrites it', 'Adds to it (accumulates)', 'Throws an error'], answer: 1,
      why: 'Gradients accumulate with `+=` so a tensor used twice receives both contributions. That is why every training step must start by zeroing them; forgetting it is the most common training-loop bug.' },
    { q: 'Module 04 measured perplexity as `exp(mean NLL)`. A model whose mean cross-entropy is 2.0 nats per character has a perplexity of about…', options: ['2', '7.4', '100'], answer: 1,
      why: 'e² ≈ 7.39: on average the model is as uncertain as a fair choice among 7.4 characters. Uniform guessing over a 46-character vocabulary is ln 46 ≈ 3.83 nats, perplexity 46.' },
    { q: 'In module 06, `GPT.forward(ids)` with `ids` of shape B×T returns logits of shape…', options: ['[B, V]: one prediction per sequence', '[B, T, V]: one prediction per position', '[T, V]'], answer: 1,
      why: 'The causal mask means position t only sees tokens 0..t, so every position\'s logits are a valid next-token prediction. One forward pass therefore yields B·T training examples.' },
    { q: 'Why can a sequence longer than `blockSize` not be fed to the module-06 GPT?', options: ['The attention matmul would be too slow', 'The position-embedding table `wpe` has exactly `blockSize` rows', 'The vocabulary would overflow'], answer: 1,
      why: 'Learned position embeddings are a lookup table with one row per slot. This is why `getBatch` cuts the corpus into windows of exactly `blockSize` tokens.' },
  ],
  review: [
    { q: 'In `getBatch`, why is `y` the window `x` shifted right by one token?', options: ['To make the batch twice as large', 'Because the causal transformer predicts every position at once, so the target at position t is the token at t+1', 'So that x and y have different lengths'], answer: 1,
      why: 'One forward pass over a B×T window gives B·T next-token predictions; the targets for all of them are simply the same text shifted by one. This is the trick that makes pre-training efficient.' },
    { q: 'What does AdamW\'s division by `sqrt(v)` achieve?', options: ['It normalises the loss', 'Every parameter moves by roughly `lr` per step, whatever the scale of its gradient', 'It makes the update exactly the negative gradient'], answer: 1,
      why: 'Rare-token embedding rows get tiny gradients and MLP weights get large ones; per-parameter scaling by the running RMS of the gradient lets one learning rate serve all of them. The `m` and `v` buffers cost 8 extra bytes per parameter in fp32.' },
    { q: 'Why do LLM runs warm the learning rate up from zero?', options: ['To save compute', 'Adam\'s moment estimates come from only a few noisy gradients at first, so full-size steps early on point in bad directions', 'Because the loss is undefined at step 0'], answer: 1,
      why: 'Bias correction fixes the magnitude of `m` and `v` but not their variance; a few hundred small steps let the estimates settle before the peak learning rate is applied.' },
    { q: 'Gradient clipping at `maxNorm` scales…', options: ['Each gradient element to at most maxNorm', 'Each parameter tensor\'s gradient separately', 'All gradients together, so the global L2 norm becomes maxNorm'], answer: 2,
      why: 'One global scale factor keeps the direction of the update unchanged and only bounds its length. Per-element or per-tensor clipping would change the direction.' },
    { q: 'Over the last 1,000 steps a run\'s train loss kept falling to 1.5 while its validation loss turned upward from 2.6 to 3.1. The model is…', options: ['Undertrained', 'Overfitting: memorising training windows', 'Perfectly tuned'], answer: 1,
      why: 'Train falling while validation rises, with a large and growing gap, means the model predicts windows it has seen much better than fresh text. More data (or fewer epochs over the same data) fixes it; more steps would widen the gap. Undertrained looks different: both curves still falling, close together.' },
  ],
  concept: `
## One loop

Every large language model was made by the same loop, and you already own every piece of it: \`backward()\` from module 02, cross-entropy from module 04, a GPT whose \`forward\` returns one prediction per position from module 06. Pre-training is:

\`\`\`
repeat:
  x, y   = a batch of token windows, y shifted one token right
  loss   = mean cross-entropy over every position of forward(x) against y
  loss.backward();  clip the gradients;  optimizer.step()
\`\`\`

Everything else in this module is a guard rail around that loop. The plot of loss against steps is the heartbeat of every training run; when it stops falling, or spikes, someone is paged.

## Predict every position at once

A window of \`T\` tokens is not one training example but \`T\` of them. Attention is causal, so position \`t\` of the logits only saw tokens \`0..t\` and is a legitimate prediction of token \`t+1\`; the targets for the whole window are the text shifted by one. With batch size \`B\` and block size \`T\`, one forward and backward pass scores \`B·T\` predictions, and the loss is their mean in nats per token.

:::predict
Your vocabulary has 46 characters. What loss do you expect at step 0, before any training, and what does a loss of 1.0 nats per character mean?
---
About \`ln(46) = 3.83\`: a freshly initialised model spreads probability nearly uniformly. A loss of 1.0 nats per character means the model is, on average, as uncertain as a fair choice among \`e¹ ≈ 2.7\` characters, or 1.44 bits per character. Well-trained character models reach approximately 1 bit per character on English (the enwik8 benchmark), so 1.0 nats is respectable for a toy.
:::

## Why AdamW

Plain gradient descent uses one step size for every parameter, but the gradients reaching a rare token's embedding row are tiny and sparse while those at an MLP weight are large and dense. Adam (Kingma & Ba 2015) keeps two running averages per parameter, \`m\` of the gradient and \`v\` of its square, and updates by \`lr · m / (sqrt(v) + eps)\`: each parameter moves by roughly \`lr\` per step regardless of its gradient scale. Both averages start at zero and are biased low early on; dividing by \`1 − beta^t\` (bias correction) fixes that, which is why your first step moves each weight by almost exactly \`lr\` (\`eps\` keeps it a hair under).

The W is weight decay done *decoupled* (Loshchilov & Hutter 2019): instead of adding \`wd · p\` to the gradient, where Adam's normalisation would rescale it away, the decay shrinks the weight directly, \`p −= lr · wd · p\`. Llama 2 (Touvron et al. 2023) reports AdamW with \`betas = [0.9, 0.95]\`, weight decay 0.1 and gradient clipping at 1.0; those are this module's defaults.

## Why warm up, why decay, why clip

Bias correction fixes the *size* of Adam's early estimates but not their *noise*: after five steps \`v\` is an average of five squared gradients, and a full-size step divided by it can point anywhere. Warmup ramps \`lr\` linearly from 0 over a few hundred steps so the estimates settle first. GPT-3 (Brown et al. 2020) warmed up over the first 375 million tokens, then followed a cosine decay to 10% of the peak; the same shape, \`cosineWithWarmup\`, is in nearly every open training recipe since.

Even with warmup, an occasional batch produces a gradient ten times larger than usual, and one such step can undo hours of progress. Clipping the *global* norm of all gradients to \`maxNorm\` bounds the length of the update while keeping its direction. Log the norm before clipping: a run whose gradient norm creeps upward is about to diverge.

:::predict
You run the same 350-step config twice, once with clipping at 1.0 and once without. Do you expect a different final loss?
---
Only slightly, on a tiny stable model. In the demo the pre-clip norm exceeds 1.0 on about half the steps (163 of 350) but only 15 steps exceed 1.5 and none exceeds 2.3, so each clipped update shrinks by a modest factor; with clipping switched off the same seed ends within about 0.05 nats of the clipped run (slightly lower, in fact). Clipping is insurance, not a speed-up. On a 70B-parameter run (Llama 2 70B used a peak learning rate of 1.5e-4 with clipping at 1.0), the rare spikes it catches are what separate a finished run from a restart from checkpoint.
:::

## Tokens, not epochs

The unit of a batch in an LLM is tokens: GPT-3 175B used approximately 3.2 million tokens per step (Brown et al. 2020, Table 2.1). Your demo uses 256. Most pre-training corpora are seen roughly once, so \`tokensSeen = steps · batchSize · blockSize\`, not epochs, is the number that scaling laws (module 08) reason about, and tokens per second is the throughput that decides what a run costs.

The training loss is measured on batches the model is updating on; the validation loss on held-out text under \`noGrad\`, with no graph and no update. Their gap is the diagnostic. Undertrained: both falling, close together, more steps help. Overfitting: train keeps falling, validation rises, the model is memorising its windows. LLMs at scale are almost always undertrained; the toy here overfits within minutes if you shrink the corpus.

## What is missing here, and what production adds

This loop runs in fp32 on one thread of one CPU. Real runs add **mixed precision** (bf16 matmuls with fp32 master weights and optimizer state, 16 bytes per parameter in total), **gradient checkpointing** (recompute activations during backward instead of storing them), **checkpoint and resume** (model, optimizer buffers and rng state saved every few hundred steps so a crash costs minutes, not days), and a **data loader** that shards trillions of tokens across thousands of workers (module 09). Nothing here is distributed; module 24 splits this same loop across GPUs. The loop itself does not change.
`,
  steps: [
    {
      id: 'batch',
      title: 'Batches of shifted windows',
      instructions: `
Implement \`getBatch(ids, blockSize, batchSize, next)\` returning \`{ x, y }\`, two arrays of \`batchSize\` rows. Each row of \`x\` is a contiguous window of \`blockSize\` token ids drawn from a random start offset; the matching row of \`y\` is the same window shifted one token to the right, so \`y[b][t]\` is the next token after \`x[b][t]\`.

Draw each row's start offset with its own call \`randInt(next, lastStart + 1)\` (already imported), one call per row in row order, so a seed reproduces the batch exactly (the tests replay the same draws). The last valid start \`lastStart\` is \`ids.length − blockSize − 1\`, because the target of the final position needs one more token. Throw an \`Error\` if the corpus is too short for that.

The worked examples above the TODO (\`trainValSplit\`, \`makeModel\`) show the conventions: ids are plain arrays, \`slice\` copies, config objects are destructured.
`,
      predict: { question: 'For ids [5, 6, 7, 8, 9] and blockSize 3, which start offsets are valid?', answer: '0 and 1 only. Start 1 gives x = [6, 7, 8], y = [7, 8, 9]. Start 2 would need y to end at ids[5], which does not exist: lastStart = 5 − 3 − 1 = 1.' },
      hints: [
        'A window at start s occupies ids[s .. s+blockSize−1]; its targets occupy ids[s+1 .. s+blockSize]. Which is the largest s for which the targets still exist?',
        'Compute `lastStart = ids.length − blockSize − 1` once, throw if it is negative, then loop batchSize times: `start = randInt(next, lastStart + 1)`, push `ids.slice(start, start + blockSize)` into x and the slice shifted by one into y.',
        '`for (let b = 0; b < batchSize; b++) { const start = randInt(next, lastStart + 1); x.push(ids.slice(start, start + blockSize)); y.push(ids.slice(…)); }` — the elided slice starts one later and ends one later.',
      ],
    },
    {
      id: 'adamw',
      title: 'AdamW',
      instructions: `
Complete the \`AdamW\` class. In the constructor allocate \`this.m\` and \`this.v\`: one \`Float32Array\` of length \`p.data.length\` per parameter, both zero. Then implement \`step()\`:

\`\`\`
t += 1
for each parameter p with a gradient g:
  p -= lr * weightDecay * p            // decoupled decay, applied to p directly
  m = beta1 * m + (1 - beta1) * g
  v = beta2 * v + (1 - beta2) * g²
  mHat = m / (1 - beta1^t);  vHat = v / (1 - beta2^t)
  p -= lr * mHat / (sqrt(vHat) + eps)
\`\`\`

All updates are in place on \`p.data\`, elementwise. Skip parameters whose \`.grad\` is \`null\`. \`zeroGrad\` is written for you. The tests compare you against \`lib/optim.js\` over several steps, so the buffers must persist between calls.
`,
      predict: { question: 'With lr 0.1 and a gradient of 1000 on one weight and 0.001 on another, how far does each move on the first step?', answer: 'Both move by 0.1. After bias correction mHat = g and vHat = g², so the update is lr · g / |g| = lr · sign(g). This per-parameter normalisation is the reason one learning rate can serve embeddings and MLP weights alike.' },
      hints: [
        'Keep the two buffers per parameter as `this.m[k]` and `this.v[k]`, indexed the same way as `this.params[k]`. `step()` is two nested loops: over parameters, then over elements.',
        'Compute the two bias corrections `1 − beta1**t` and `1 − beta2**t` once per step, outside the loops. Inside: decay first, then update m and v, then divide by the corrections, then update the weight.',
        '`for (let i = 0; i < data.length; i++) { if (decay !== 0) data[i] -= decay * data[i]; m[i] = this.beta1 * m[i] + (1 - this.beta1) * g[i]; v[i] = …; data[i] -= (this.lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + this.eps); }` — fill in the v update, which uses g[i] squared.',
      ],
    },
    {
      id: 'trainstep',
      title: 'Gradient clipping and one training step',
      instructions: `
Two functions.

\`clipGradNorm(params, maxNorm)\`: compute the L2 norm over **all** gradients of all parameters together (\`sqrt\` of the sum of every squared gradient element). If it exceeds \`maxNorm\`, multiply every gradient in place by \`maxNorm / norm\`. Return the norm from **before** clipping. Skip parameters with no gradient.

\`trainStep(model, optimizer, x, y, { maxGradNorm = 1.0 })\`: one iteration of the loop. Zero the gradients, run \`model.forward(x)\` (logits \`[B, T, V]\`), take \`crossEntropy(logits, y)\` (a scalar Tensor; it is already the mean over every position), call \`backward()\`, clip with \`model.parameters()\`, call \`optimizer.step()\`. Return \`{ loss, gradNorm }\` where \`loss\` is a plain number from \`.item()\` and describes the batch **before** the update.

The tests hand you the reference optimizer, so this step does not depend on your step-2 AdamW being right.
`,
      hints: [
        'Clipping is global: one scale factor for every gradient, computed from one sum of squares across all parameters. Per-tensor clipping would change the direction of the update, and the tests check for it.',
        'The order of the training step is fixed: zeroGrad, forward, loss, backward, clip, step. Reading the loss with `.item()` before the optimizer step is safe; the number does not change afterwards, but the graph is gone.',
        'For `clipGradNorm`: `let sumSq = 0; for (const p of params) { if (!p.grad) continue; for (const g of p.grad) sumSq += g * g; } const norm = Math.sqrt(sumSq);` then, only if `norm > maxNorm`, a second pass multiplies every element by the same factor. `trainStep` is six calls in the order of hint 2; turn the loss Tensor into a number with `.item()`.',
      ],
    },
    {
      id: 'schedule',
      title: 'Warmup and cosine decay',
      instructions: `
Implement \`cosineWithWarmup(step, { warmup, total, peak, min = peak / 10 })\`:

- for \`step < warmup\`: linear from 0 to \`peak\`, so \`peak · step / warmup\`;
- for \`step >= total\`: \`min\`;
- otherwise, with \`progress = (step − warmup) / (total − warmup)\` running from 0 to 1: \`min + 0.5 · (peak − min) · (1 + cos(π · progress))\`.

The tests check the exact values at step 0, mid-warmup, the peak, a quarter of the way through the decay (where a cosine differs from a straight line), the end and beyond. In \`train\` you will write the result into \`optimizer.lr\` before every step.
`,
      hints: [
        'Three cases in order: warmup, past the end, cosine in between. Return early from each so the arithmetic of the next case never runs on the wrong range.',
        'The cosine term `0.5 · (1 + cos(π · progress))` runs from 1 at progress 0 to 0 at progress 1; multiply it by the distance from min to peak and add min.',
        '`if (step < warmup) return (peak * step) / warmup;` handles the first case (and with `warmup = 0` the division is never reached). Then `if (step >= total) return min;`. The last line combines `progress` from the instructions with the cosine term of hint 2.',
      ],
    },
    {
      id: 'eval',
      title: 'Validation loss',
      instructions: `
Implement \`estimateLoss(model, ids, { blockSize, batchSize, evalBatches = 4, next })\`: draw \`evalBatches\` batches from \`ids\` with your \`getBatch\`, compute the cross-entropy of each with \`model.forward\`, and return the mean as a plain number.

The whole thing must run inside \`noGrad(() => …)\` from \`lib/tensor.js\` so that no autograd graph is recorded: evaluation is forward-only, must not call \`backward()\`, and must not change the model. The test spies on \`model.forward\` to check that gradient recording is off while it runs, and checks that no parameter has a gradient afterwards.
`,
      hints: [
        'One batch of validation loss is as noisy as one batch of training loss; the average of several is what you plot. Draw the batches from the `next` you were given so a seed reproduces the number.',
        '`noGrad(fn)` runs fn with recording switched off and returns whatever fn returns, and switches recording back on even if fn throws. Put the loop inside it and return the mean from inside.',
        '`return noGrad(() => { let total = 0; for (let k = 0; k < evalBatches; k++) { const { x, y } = getBatch(ids, blockSize, batchSize, next); total += …; } return total / evalBatches; });` — the elided term is the cross-entropy of `model.forward(x)` against `y`, as a number.',
      ],
    },
    {
      id: 'train',
      title: 'The run, and sampling from it',
      instructions: `
Two functions that assemble everything.

\`train(config, onStep)\` is \`async\`. The skeleton already builds the model and an rng. Build one \`AdamW\` over \`model.parameters()\` with \`{ lr, betas: [0.9, 0.95], weightDecay }\`. Then for \`step\` in \`0 .. steps−1\`:

1. set \`optimizer.lr = cosineWithWarmup(step, { warmup, total: steps, peak: lr })\`;
2. draw a batch from \`trainIds\` with \`next\`, call \`trainStep\`;
3. build \`record = { step, lr, loss, gradNorm }\`; every \`evalInterval\` steps (that is, when \`(step + 1) % evalInterval === 0\`) and on the last step, add \`record.valLoss = estimateLoss(model, valIds, …)\` using the same \`next\`;
4. push the record to \`history\` and \`await onStep(record, model)\` if a callback was given.

Resolve to \`{ model, history, tokensSeen }\` with \`tokensSeen = steps · batchSize · blockSize\`.

The tests replay your run against a reference loop built from these exact rules and compare the loss, gradient norm and validation loss at every step, so pass \`maxGradNorm\` to \`trainStep\`, use \`valIds\` (not \`trainIds\`) for evaluation, and draw validation batches from the same \`next\` right after that step's training batch.

\`sample(model, tokenizer, prompt, { maxNewTokens, temperature, next })\`: encode the prompt, call \`model.generate(ids, { maxNewTokens, temperature, next })\` (from module 06; it runs under \`noGrad\`), decode the result and return the whole string, prompt included.
`,
      hints: [
        'Build the optimizer once, before the loop; its m and v buffers must persist. The schedule is a property write on the optimizer each step, not a new optimizer.',
        'The eval condition has two parts joined by `||`: `(step + 1) % evalInterval === 0` and `step === steps − 1`. Guard it with `valIds &&` so a run without a validation split still works. The callback may be async, so `await` it.',
        '`for (let step = 0; step < steps; step++) { optimizer.lr = cosineWithWarmup(…); const { x, y } = getBatch(trainIds, blockSize, batchSize, next); const { loss, gradNorm } = trainStep(…); const record = { … }; if (valIds && (…)) record.valLoss = estimateLoss(…); history.push(record); if (onStep) await onStep(record, model); }` — build the optimizer before the loop and return after it.',
      ],
    },
  ],
  reflection: [
    'Write the pre-training loop from memory in five lines, then explain to a colleague what each of warmup, cosine decay, gradient clipping and weight decay protects against, and which of the four you would drop first for a tiny model.',
    'Your demo saw about 90,000 tokens; GPT-3 saw 300 billion. Which parts of your `train` function would survive that scale unchanged, and which would have to be replaced by mixed precision, checkpointing, sharded data loading or multi-GPU parallelism?',
    'The train loss is measured on batches the model is updating on and the validation loss on held-out text. Describe what the two curves look like when a model is undertrained, when it is overfitting, and when the learning rate is too high.',
  ],
  stretch: [
    'Add `saveCheckpoint` / `loadCheckpoint` that serialise the model (`GPT.toJSON`), the AdamW buffers, `t` and the step number, and show that a run interrupted at step 150 and resumed reproduces the uninterrupted run exactly; this is what every Megatron-LM and DeepSpeed job does every few hundred steps.',
    'Simulate mixed precision: round every parameter to bf16 (8 exponent bits, 7 explicit mantissa bits: the top 16 bits of the fp32 pattern, rounded to nearest) after each step while keeping an fp32 master copy inside the optimizer, and compare loss curves; this is the fp32-master-weights scheme in NVIDIA Apex and PyTorch AMP.',
    'Implement gradient accumulation: run `k` micro-batches, summing gradients, before one optimizer step, and confirm the loss curve matches a run with batch size `k × batchSize`; this is how Llama-scale runs reach millions of tokens per step on limited memory.',
    'Replace the contiguous split with `interleavedSplit` from `lib/data.js`, train on the full `CORPUS` (toy sentences plus Shakespeare), and explain why the validation curve changes; GPT-3\'s data mixing (Brown et al. 2020, Table 2.2) is the same question at scale.',
  ],
  timeouts: { tests: 20000, demo: 120000 },
};
