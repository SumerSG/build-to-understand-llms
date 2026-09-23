export default {
  id: '31-distillation',
  title: 'Knowledge distillation',
  track: 'posttraining',
  minutes: 90,
  threshold: 'A teacher\'s full next-token distribution carries far more information per token than the single correct label, so a smaller student that matches it learns faster and ends closer to the teacher than one trained on the data alone.',
  goal: 'A distillation trainer (temperature-softened targets, a T²-scaled KL loss, a mixed objective, a sequence-level data path) that trains a 1-layer student to track the frozen checkpoint teacher faster than training from scratch, with held-out loss curves, top-1 agreement and sample generations for the three students (scratch, logit KD, sequence-level KD). You also build the on-policy reverse-KL loss; the demo uses it to score each student on its own samples, and training with it is a stretch exercise.',
  prereqs: ['07-pretraining', '10-sft', '30-lora'],
  recall: [
    { q: 'In module 07, `sample` passed a `temperature` to `model.generate`, which divides the logits by it before the softmax. What does temperature 4 do to the distribution?', options: ['Sharpens it towards the argmax', 'Flattens it towards uniform while keeping the ranking', 'Changes which token is most likely'], answer: 1,
      why: 'Dividing every logit by the same positive number keeps their order but shrinks the gaps, so exp() spreads the mass out. Distillation uses exactly this knob on the teacher, for a different purpose: to make the small probabilities large enough to learn from.' },
    { q: 'A freshly initialised model over a 256-token vocabulary (module 04 and module 07) has a cross-entropy of roughly…', options: ['0 nats', 'ln 256 ≈ 5.55 nats', '256 nats'], answer: 1,
      why: 'Near-uniform predictions give −ln(1/256) = ln 256 ≈ 5.55 nats per token. Every student in the demo starts at 5.55; the teacher checkpoint sits near 2.7 on held-out text.' },
    { q: 'In module 10, the SFT loss divides the summed per-token loss by…', options: ['B · T, all positions', 'The number of masked-in positions', 'The batch size'], answer: 1,
      why: 'A mean over the positions that count keeps the loss in nats per counted token. Your `reverseKL` in step 5 uses the same rule with a mask over the generated tokens.' },
    { q: 'In module 10, the SFT loss is cross-entropy against the one token in the data at each position. Which of the model\'s predicted probabilities does that loss read directly?', options: ['Only the probability of the target token', 'The whole distribution', 'The top-k probabilities'], answer: 0,
      why: 'With a one-hot target, cross-entropy is −log p(target); the other tokens enter only through the softmax normaliser. Distillation replaces the one-hot target with the teacher\'s whole distribution.' },
    { q: 'In module 02, what does running a forward pass inside `noGrad` change?', options: ['The outputs', 'No graph is recorded, so no gradient can flow back into those weights', 'Weights are rounded to float16'], answer: 1,
      why: 'The teacher must stay frozen. Computing its logits under `noGrad` makes them constants in the student\'s graph, and it saves the memory the graph would take.' },
  ],
  review: [
    { q: 'Why is the KL term multiplied by T²?', options: ['To convert nats to bits', 'Its gradient with respect to the student logits scales as 1/T², so T² keeps its size comparable to the hard-label term as T changes', 'To make the loss positive'], answer: 1,
      why: 'd KL(p_T ‖ q_T) / dz = (q_T − p_T) / T, and for large T the difference q_T − p_T itself shrinks like 1/T. Hinton et al. (2015) multiply by T² so changing T does not silently change the weighting of the two terms.' },
    { q: 'A teacher puts 50% on "cat" and 50% on "dog". A student can only put its mass on one of them. Which divergence is SMALLER for the student that picks "cat" alone?', options: ['Forward KL(teacher ‖ student)', 'Reverse KL(student ‖ teacher)', 'They are equal'], answer: 1,
      why: 'Reverse KL only looks where the student puts mass, and there the teacher agrees, so it costs about log 2. Forward KL looks where the TEACHER puts mass; the student\'s near-zero "dog" makes it huge. Forward KL is mode-covering, reverse KL is mode-seeking.' },
    { q: 'What does on-policy distillation (GKD) fix that logit distillation on a fixed dataset does not?', options: ['It removes the need for a teacher', 'The student is trained on the prefixes it will actually produce at inference, so its own mistakes get teacher feedback', 'It makes each step cheaper'], answer: 1,
      why: 'Off-policy the student only ever sees data or teacher prefixes; at generation time it conditions on its own tokens and drifts (exposure bias). Sampling from the student and letting the teacher score those samples trains on exactly the states that matter.' },
    { q: 'You can call a frontier model only through an API that returns text. Which distillation can you do?', options: ['Logit KD with the KL loss', 'Sequence-level KD: fine-tune on the text it generates', 'Neither'], answer: 1,
      why: 'Logit KD needs the teacher\'s full distribution over a shared vocabulary. Text is enough for sequence-level KD, which is what Alpaca, the DeepSeek-R1 distilled models and most "synthetic data" pipelines do.' },
    { q: 'In the demo, why can the scratch and distilled students be compared fairly?', options: ['They have different sizes', 'Same architecture, same initial weights, same windows, same seeds and the same number of steps; only the loss differs', 'The distilled one trains longer'], answer: 1,
      why: 'Controlled comparison: the only change is where the targets come from, so the held-out gap is the effect of the soft targets, not of more data or compute.' },
  ],
  concept: `
## One label versus a whole distribution

In module 07 every position had a single target: the next token in the corpus, a one-hot distribution. The trained model outputs a full distribution over 256 tokens. After "The teacher is green beside the" it gives " road" about 97%, " garden" 0.6%, and a long tail of near-zeros. That ranking (garden is a place, "r" is not) is knowledge extracted from the whole corpus. Hinton, Vinyals & Dean (2015) called it **dark knowledge**: train a small **student** to match the **teacher**'s distribution instead of, or as well as, the label.

## Temperature makes the tail visible

At 97% versus 0.6%, the alternatives barely register. Dividing the teacher's logits \`z\` by a temperature \`T > 1\` before the softmax, \`p_T = softmax(z / T)\`, keeps the ranking and inflates the small probabilities. The student is softened the same way, \`q_T = softmax(s / T)\`, and the loss is the KL divergence between them, where \`v\` runs over the vocabulary:

\`\`\`
L_KD = T² · mean over positions of Σ_v p_T(v) · (log p_T(v) − log q_T(v))
\`\`\`

The gradient of the KL with respect to the student's logits \`s\` is \`(q_T − p_T) / T\`, and for large \`T\` the difference itself shrinks like \`1/T\`. The \`T²\` factor cancels that, so changing \`T\` does not silently reweight the loss. It is usually mixed, with a weight \`alpha\` between 0 and 1, with the ordinary label loss (cross-entropy, CE, on the label \`y\`) at \`T = 1\`:

\`\`\`
L = alpha · L_KD + (1 − alpha) · CE(s, y)
\`\`\`

:::predict
At \`T = 1\` the teacher gives " road" 97% after "The teacher is green beside the". Roughly how much does " road" get at \`T = 4\`?
---
About 5.5%. At \`T = 4\` the remaining 255 tokens share about 95% of the mass, and the runner-up " garden" rises from 0.6% to about 1.6%. The ranking is unchanged; the student can now see it.
:::

## Forward KL, reverse KL, and whose samples you train on

\`KL(p ‖ q)\` (**forward**) averages over where the *teacher* puts mass: a student that ignores a plausible teacher token pays heavily, so it spreads out to cover every mode (**mode-covering**). \`KL(q ‖ p)\` (**reverse**) averages over where the *student* puts mass, so it may drop a mode and commit to another (**mode-seeking**). A small student cannot represent everything its teacher does: forward KL makes it hedge, reverse KL makes it pick.

The training sequences can come from three places:

- **Logit KD** on a fixed corpus: the teacher's distribution at every position of real text. This is Hinton's recipe, and Google reports it for Gemma 2 2B and 9B. It needs the teacher's logits over a shared vocabulary.
- **Sequence-level KD** (Kim & Rush 2016, who used beam search): the teacher *writes* the training set and the student fine-tunes on it with plain cross-entropy. It is the only option when the teacher is an API that returns text: Alpaca (52k examples written by an OpenAI model) and the DeepSeek-R1 distilled Qwen and Llama models (about 800k samples curated with R1) work this way, and it is the "synthetic data" branch of the module 09 pipeline.
- **On-policy KD** (GKD, Agarwal et al. 2023; MiniLLM, Gu et al. 2023, with reverse KL): the *student* samples and the teacher scores every token. The student learns on the prefixes it will actually produce, mistakes included, which fixes the train/inference mismatch (exposure bias) of the other two. Qwen3's report describes off-policy then on-policy distillation for its small models.

:::predict
Three identical students train for 150 steps: on corpus labels, on the teacher's soft targets over the same windows (\`alpha = 0.5, T = 2\`), and on the same amount of teacher-written text. Rank their held-out cross-entropy on real text.
---
Logit KD is best (about 3.67 nats), then scratch (about 3.90), then sequence-level (about 4.18). A sampled teacher token carries no more information than a real one, and this tiny teacher writes noisier text than the corpus, so sequence-level KD loses on real text here while still agreeing with the teacher more often than scratch. Its advantages are volume and API access.
:::

## Where you will meet it

DistilBERT (Sanh et al. 2019) halved BERT's depth and, by the authors' count, kept about 97% of its GLUE score while running about 60% faster. MiniLM (Wang et al. 2020) distills attention distributions rather than outputs. Meta says Llama 3.2 1B and 3B used logits from Llama 3.1 8B and 70B as token-level targets. Speculative decoding, which module 18 builds later, accepts a small draft model's token with probability \`min(1, p/q)\` (p the target's probability, q the draft's), so drafts are often distilled from their target (DistillSpec, Zhou et al. 2023). The contrast is TinyLlama, a 1.1B model trained from scratch on about 3 trillion tokens: the expensive way to get a small model.

## Where the toy differs from production

Your teacher has about 120k parameters and your student 23k; production teachers have tens to hundreds of billions. You cache full 256-way logits for 96 windows; with 128k-token vocabularies and trillions of training tokens that cannot be stored, so real pipelines run the teacher alongside the student or keep only the top-k logits per position. Your runs are 150 steps with one seed: the gaps are indicative. Real on-policy distillation samples with a serving engine such as vLLM.
`,
  steps: [
    {
      id: 'soft',
      title: 'Temperature-softened targets',
      instructions: `
Complete \`softTargets(logits, T = 1)\`: return \`softmax(z / T)\` along the last axis, as \`{ shape, data: Float32Array }\` with the input's shape.

- \`logits\` may be a raw tensor, a \`Tensor\`, or a plain 1-D array; the skeleton already reads all three.
- Each row of length \`V\` is normalised on its own (a \`[B, time, V]\` input is \`B · time\` separate distributions; \`T\` in this module always means temperature).
- Subtract the row maximum before \`exp\`; teacher logits can be large, and at \`T < 1\` they get larger.
- Throw for \`T <= 0\`. (The sampler in \`lib/sampling.js\` treats \`T = 0\` as greedy decoding, the limit as \`T\` shrinks to 0; here a zero temperature has no meaning as a training target.)
- Do not modify the input.

This one function is used for the teacher's targets (step 2) and, in the demo, to show how temperature exposes the tail.
`,
      predict: { question: 'For logits `[1, 2, 3]`, what is `softTargets` at `T = 2`?', answer: '`softmax([0.5, 1, 1.5])` ≈ `[0.186, 0.307, 0.506]`, against `[0.090, 0.245, 0.665]` at `T = 1`. The order is the same; the gaps shrink.' },
      hints: [
        'Temperature is applied to the logits, before the exponential, not to the probabilities afterwards.',
        'Per row: you already have `max`. Fill each `out[r + j]` with `exp((z − max) / T)` while summing, then divide the row by the sum. Check `T` at the top.',
        '`let z = 0; for (let j = 0; j < V; j++) { const e = Math.exp(/* … */); out[r + j] = e; z += e; }` then a second loop dividing by `z`.',
      ],
    },
    {
      id: 'kl',
      title: 'The distillation loss',
      instructions: `
Implement \`klDistillLoss(studentLogits, targetLogits, T = 1)\`, returning a **scalar Tensor**:

\`\`\`
p = softTargets(targetLogits, T)                  // constant: the frozen teacher
log q = logSoftmax(studentLogits / T)             // Tensor ops: this is where the gradient flows
loss = T² · (1 / N) · Σ_positions Σ_v p · (log p − log q)
\`\`\`

\`N\` is the number of positions (\`data.length / V\`). \`studentLogits\` is a \`Tensor\` with gradient; \`targetLogits\` (the teacher's) is a raw tensor or \`Tensor\` of the same shape.

The \`Σ p · log p\` part does not depend on the student, so compute it as a plain number and add it. It makes the loss exactly 0 when the student matches, which is what makes the value readable. The \`Σ p · log q\` part must be Tensor operations: \`studentLogits.scale(1 / T).logSoftmax()\`, multiplied by \`new Tensor(p)\`, summed.

The tests check a 3-class example by hand: student logits \`[[0, 1, 2]]\`, teacher logits \`[[2, 1, 0]]\` give 1.1504 at \`T = 1\` and 1.2806 at \`T = 2\` (0.3202 if you forget the \`T²\`). They also check that duplicating a position does not change the loss (mean, not sum), and that the gradient equals \`T · (q_T − p_T) / N\`.
`,
      predict: { question: 'At `T = 4`, how much smaller would the gradient be if you forgot the `T²`?', answer: '16 times smaller. The gradient of the plain KL is `(q_T − p_T) / T`; with the `T²` it becomes `T · (q_T − p_T)`. Without the factor, raising `T` quietly shifts the balance of the mixed loss towards the hard labels.' },
      hints: [
        'Split the KL into a constant part (`Σ p log p`, a number) and a part that depends on the student (`−Σ p log q`, a Tensor).',
        'Soften both sides with the same `T`. Only the cross term `−Σ p log q` needs a gradient, so it is the only part built from Tensor ops; `Σ p log p` is a plain number from the loop over `p.data`. Add them, then apply the `T²` and the `1 / N` last, to the combined scalar.',
        '`const logq = studentLogits.scale(1 / T).logSoftmax(); const cross = logq.mul(new Tensor({ shape: studentLogits.shape.slice(), data: p.data })).sum(); return /* combine cross, plogp and T * T / positions */;`',
      ],
    },
    {
      id: 'mixed',
      title: 'Mixing soft and hard targets',
      instructions: `
Implement \`distillLoss(studentLogits, targetLogits, targets, { alpha = 0.5, T = 2 })\`:

\`\`\`
loss = alpha · klDistillLoss(studentLogits, targetLogits, T) + (1 − alpha) · crossEntropy(studentLogits, targets)
\`\`\`

- The cross-entropy (from \`lib/tensor.js\`) is on the **unsoftened** logits (\`T = 1\`): the labels are one-hot, so there is nothing to soften, and at inference the student runs at \`T = 1\`.
- \`alpha = 0\` is plain training on the data; then \`targetLogits\` may be \`null\`, and you must not touch it. \`alpha = 1\` is pure distillation.
- Throw if \`alpha\` is outside \`[0, 1]\`.

Hinton et al. found a small weight on the hard term helps; DistilBERT mixes its distillation loss with the masked-language-model loss in the same way. In the demo, \`alpha = 0\` with no teacher is the from-scratch student, so both demo students go through this one function.
`,
      hints: [
        'Two terms, each already written: your `klDistillLoss` and lib\'s `crossEntropy`. The only decisions are the weights and the edge cases.',
        'Validate alpha. Handle `alpha === 0` before anything reads the teacher: that case is the cross-entropy alone. Otherwise compute the KL term; when alpha is 1 it is the whole loss, and in between you weight the two scalar Tensors and add them.',
        '`if (alpha === 0) return crossEntropy(studentLogits, targets); const kl = klDistillLoss(studentLogits, targetLogits, T); /* alpha === 1: kl alone; otherwise scale each term and add */`',
      ],
    },
    {
      id: 'seqkd',
      title: 'Sequence-level distillation: the teacher writes the data',
      instructions: `
Implement \`sequenceLevelCorpus(teacher, tokenizer, prompts, { maxNewTokens = 32, temperature = 1, next })\`.

\`teacher\` is an inference model (\`loadModel\` from \`lib/infer.js\`), \`prompts\` an array of strings. For each prompt, in order:

1. \`continuation = generate(teacher, tokenizer, prompt, { maxNewTokens, temperature, next })\` (from \`lib/sampling.js\`; it samples one token at a time from the model's logits divided by \`temperature\`, as module 07's \`sample\` did, and returns only the new text, stopping at \`eos\`).
2. \`text = prompt + continuation\`; push it to \`texts\`.
3. Append \`tokenizer.encode(text)\` and then \`tokenizer.eos\` to \`ids\`.

Return \`{ ids, texts }\`. \`ids\` is one token stream you can cut into training windows with \`getBatch\`, exactly like a pre-training corpus, and the student trains on it with ordinary cross-entropy (\`distillLoss\` with \`alpha = 0\`).

Use the one \`next\` for every prompt, so the corpus is reproducible from a single seed. Pass \`temperature\` through: \`0\` is greedy, the mode-seeking choice Kim & Rush approximated with beam search; \`1\` samples the teacher's distribution.

This is the pipeline behind "distilling" an API model: no logits, only text.
`,
      hints: [
        'This is a loop over the prompts with three lines in the body. Everything hard is already in `lib/sampling.js`.',
        'For each prompt call `generate(teacher, tokenizer, prompt, { maxNewTokens, temperature, next })`, concatenate it to the prompt, record the text, and push the encoded ids followed by `tokenizer.eos`.',
        '`for (const prompt of prompts) { const text = prompt + generate(teacher, tokenizer, prompt, /* the options */); /* record text; append its ids, then the separator */ }`. Encode the whole text once: BPE merges can cross the prompt boundary.',
      ],
    },
    {
      id: 'onpolicy',
      title: 'On-policy distillation: the student writes, the teacher grades',
      instructions: `
Two functions.

\`reverseKL(studentLogits, targetLogits, mask = null)\`: a scalar Tensor,

\`\`\`
mean over positions with mask = 1 of Σ_v q(v) · (log q(v) − log p(v))
q = softmax(studentLogits) (Tensor, with gradient)     p = softmax(targetLogits) (constant)
\`\`\`

\`mask\` is \`number[][]\` (B×T) or \`null\` (every position counts). Divide by the number of masked-in positions, and throw if there are none. Use \`logSoftmaxRows\` (worked example) for \`log p\`. The gradient must flow through **both** \`q\` and \`log q\`: build \`logq = studentLogits.logSoftmax()\` and \`q = logq.exp()\`. A \`[B, T, 1]\` Tensor of mask weights broadcasts over the vocabulary.

\`onPolicyLoss(student, teacher, promptIds, { maxNewTokens, temperature = 1, next })\`: both models are \`GPT\`s, \`promptIds\` is \`number[][]\` with every prompt the same length (throw otherwise).

1. \`ids[i] = student.generate(promptIds[i], { maxNewTokens, temperature, next })\`. The **student** samples. \`GPT.generate\` returns the prompt followed by the \`maxNewTokens\` new ids (unlike \`lib/sampling.js\` \`generate\` in step 4, which returns only the new text), so each \`ids[i]\` has length \`promptLen + maxNewTokens\`.
2. \`x = ids.map(s => s.slice(0, -1))\`. Position \`t\` predicts token \`t + 1\`, so \`mask[i][t] = 1\` exactly when \`t >= promptLen − 1\` (the positions that predict a generated token).
3. Teacher logits with \`teacherLogits(teacher, x)\` (under \`noGrad\`: the teacher stays frozen); student logits with \`student.forward(x)\`. Each row of \`x\` is scored in one forward pass, so \`promptLen + maxNewTokens − 1\` must be at most both models' \`blockSize\`; \`generate\` itself slides its window and never complains, but \`forward\` throws on a longer sequence.
4. Return \`{ loss: reverseKL(studentLogits, teacher's, mask), ids, mask }\`.
`,
      predict: { question: 'The teacher splits 50/50 between two tokens; the student puts 96% on the first and 4% on the second. Which is larger, forward or reverse KL?', answer: 'Forward KL, `0.5·ln(0.5/0.96) + 0.5·ln(0.5/0.04)` ≈ 0.94 nats, is almost twice reverse KL, `0.96·ln(0.96/0.5) + 0.04·ln(0.04/0.5)` ≈ 0.53. Reverse KL weights each token by the student\'s probability, so the neglected second mode enters with weight 0.04; forward KL weights it by the teacher\'s 0.5 and charges `ln 12.5` for it.' },
      hints: [
        'Reverse KL swaps the roles: the weights of the average are now the student\'s probabilities q, and those depend on the parameters too.',
        'reverseKL: `logq = s.logSoftmax()`, `q = logq.exp()`, `perToken = q.mul(logq.sub(new Tensor(logSoftmaxRows(t))))`, multiply by a `[..., 1]` weight Tensor, sum, divide by the count. onPolicyLoss: generate with the student, drop the last token for `x`, build the mask from `promptLen − 1`.',
        '`const mask = x.map((row) => row.map((_, t) => /* 1 on generated-token predictions */)); const loss = reverseKL(student.forward(x), teacherLogits(teacher, x), mask);`',
      ],
    },
    {
      id: 'experiment',
      title: 'The experiment: scratch versus distilled at equal steps',
      instructions: `
Three functions for the experiment the demo runs. A **pool** is \`{ x, y, teacher }\`: \`x\` and \`y\` are \`number[][]\` windows (\`y\` shifted by one), \`teacher\` is a raw \`[N, T, V]\` tensor of cached teacher logits (from \`buildPool\`) or \`null\`.

\`top1Agreement(studentLogits, targetLogits)\`: the fraction of positions where the two argmaxes match (first index on ties). Production papers report it next to loss because it is what a greedy decoder, or a speculative-decoding verifier, sees.

\`async distillTrain(student, pool, { steps, batchSize = 4, lr = 3e-3, alpha = 0.5, T = 2, next, onStep = null })\`: create \`AdamW(student.parameters(), { lr, betas: [0.9, 0.95], weightDecay: 0 })\`, then each step:

1. Draw \`batchSize\` indices with \`randInt(next, pool.x.length)\`, and gather \`x\`, \`y\` and, if \`alpha > 0\`, the cached teacher rows with \`takeRows(pool.teacher, idx)\`.
2. \`loss = distillLoss(student.forward(x), target, y, { alpha, T })\`; \`zeroGrad\`, \`backward\`, \`clipGradNorm(params, 1.0)\`, \`step\`.
3. Record \`loss.item()\` and, if given, \`await onStep(step, loss.item())\` (the demo uses it to evaluate and to yield).

Return the losses. The teacher never runs here: offline distillation reads cached logits, so both students cost the same per step.

\`evaluate(model, pool)\`: under \`noGrad\`, a few windows at a time, return \`{ loss, agreement }\`: the mean cross-entropy on \`y\` over every position, and \`top1Agreement\` with \`pool.teacher\` over every position (or \`null\` without teacher logits). Weight each chunk by its number of positions.
`,
      hints: [
        'Agreement is two argmaxes per row of length V. Training is the module 07 loop with `distillLoss` as the loss and cached teacher rows as the extra input.',
        'For `distillTrain`, the one easy mistake is drawing indices with anything other than `next` (the tests replay seeds). For `evaluate`, accumulate `loss × positions` and `agreement × positions` per chunk and divide by the total at the end.',
        '`const idx = []; for (let b = 0; b < batchSize; b++) idx.push(randInt(next, pool.x.length)); const target = alpha > 0 ? takeRows(pool.teacher, idx) : null; const loss = /* distillLoss on student.forward(x) */; optimizer.zeroGrad(); loss.backward(); clipGradNorm(params, 1.0); optimizer.step();`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why matching a teacher\'s distribution can beat training on the true labels, when the labels are what you are ultimately graded on. Use the "beside the road / garden" example or one of your own.',
    'In your demo, sequence-level KD lost to plain labels on held-out text but agreed with the teacher more often. Why can both be true, and when would you still choose sequence-level KD?',
    'Forward KL makes a small student hedge across everything the teacher considers plausible; reverse KL makes it commit. For a chat model that must never produce one very bad answer, which would you pick, and what would you lose?',
  ],
  stretch: [
    'Implement top-k logit distillation: store only the teacher\'s 8 largest logits per position (renormalised) and measure how much of the held-out gain survives. This is how large-vocabulary pipelines keep the storage feasible when training small models such as Gemma 2 2B from a larger teacher.',
    'Add the generalised Jensen–Shannon divergence from GKD (Agarwal et al. 2023), which interpolates between forward and reverse KL with a weight beta, and run a short on-policy phase after logit KD with `onPolicyLoss`. Compare the teacher\'s reverse KL on the student\'s own samples before and after.',
    'Once you have done module 18, distil a draft model for it: train a student on the checkpoint\'s soft targets, then measure the speculative-decoding acceptance rate with the distilled draft versus a from-scratch draft of the same size (DistillSpec, Zhou et al. 2023).',
    'Replace sampling with greedy (temperature 0) in `sequenceLevelCorpus`, the mode-seeking choice Kim & Rush approximated with beam search, and compare held-out loss and agreement against temperature-1 samples.',
  ],
  timeouts: { tests: 20000, demo: 180000 },
};
