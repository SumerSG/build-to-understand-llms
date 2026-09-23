export default {
  id: '18-speculative',
  title: 'Speculative decoding',
  track: 'inference',
  minutes: 90,
  threshold: 'Decoding is memory-bound, so verifying K guessed tokens in one forward pass costs about the same as generating one; if the guesses are usually right you get several tokens per step, and the rejection-sampling rule keeps the output distribution exactly the target\'s.',
  goal: 'Draft-and-verify decoding with the rejection-sampling acceptance rule that provably preserves the target distribution, with measured speedup: a cheap draft proposes K tokens, one target pass checks them, and the demo shows the acceptance rate, tokens per step and where speculation stops paying.',
  prereqs: ['14-decoding', '15-kv-cache', '16-batching'],
  recall: [
    { q: 'In module 14, `sample` drew one uniform `u` and walked the cumulative probabilities. It returned…', options: ['The index with the largest probability', 'The first index whose cumulative probability exceeds u', 'A random index, ignoring the probabilities'], answer: 1,
      why: 'That inverse-CDF walk is `sampleIndex` in lib/util.js. Every draw in this module (proposal, residual, bonus token) goes through it with one uniform, which is what makes the tests reproducible.' },
    { q: 'From module 15: at batch size 1, a decode step of a Llama-3-8B-shaped model in bf16 on an H100 is limited by…', options: ['The 16 GFLOP of arithmetic', 'Reading approximately 16 GB of weights at approximately 3.35 TB/s', 'The softmax over the vocabulary'], answer: 1,
      why: 'The weight read takes about 5 ms, the arithmetic tens of microseconds. Feeding K + 1 tokens through the same pass adds arithmetic but not weight traffic, so the pass costs about the same. That slack is what speculation spends.' },
    { q: 'In module 07, one forward pass over a training sequence produced…', options: ['One logit vector for the last position only', 'A logit vector for every position, each predicting the token that follows it', 'A single scalar loss and nothing else'], answer: 1,
      why: 'Teacher forcing: feed the whole sequence, read a next-token distribution at every position. Verification here is exactly that pass over the drafted tokens, reading K + 1 distributions at once.' },
    { q: 'In module 16, the per-token cost of a decode iteration fell from 5.05 ms to about 0.21 ms when the batch grew from 1 to 32 because…', options: ['The arithmetic got cheaper', 'The fixed weight-read term was amortised over 32 sequences', 'The KV cache was smaller'], answer: 1,
      why: 'Batching and speculation spend the same slack: both fill the arithmetic the weight read leaves idle. Once a large batch has filled it, extra speculative tokens cost real time, which is the `rho` term in your speedup model.' },
    { q: 'In module 14, dividing the logits by a temperature above 1 before the softmax…', options: ['Sharpens the distribution towards the argmax', 'Flattens it, moving mass into the tail', 'Leaves the probabilities unchanged'], answer: 1,
      why: 'The acceptance rate is `Σ min(p, q)`, so anything that reshapes the target `p` without reshaping the draft `q` changes how often drafts are accepted. The demo sweeps the target temperature to show it.' },
  ],
  review: [
    { q: 'The draft proposes token x with q[x] = 0.5; the target gives it p[x] = 0.3. The verify step keeps x with probability…', options: ['0.3', '0.6', '1'], answer: 1,
      why: 'min(1, p/q) = 0.3 / 0.5. Kept mass is q · p/q = p for over-proposed tokens and q for under-proposed ones, i.e. min(p, q) per token: exactly the mass the target and draft agree on.' },
    { q: 'After a rejection, the replacement token is drawn from…', options: ['The target distribution p', 'The draft distribution q', 'norm(max(0, p − q)), the target mass the draft under-proposed'], answer: 2,
      why: 'Accepted tokens already carry min(p, q). Only the residual p − min(p, q) = max(0, p − q) is missing; drawing it on rejection makes the total exactly p. Drawing from p instead double-counts where the draft was right.' },
    { q: 'With acceptance rate α = 0.8 and K = 4, the expected number of tokens per verify pass is…', options: ['4.0', '3.36', '5.0'], answer: 1,
      why: '(1 − 0.8^5) / (1 − 0.8) = 3.36: one token is guaranteed (residual or bonus), each further draft survives with probability 0.8 given the previous one did. 5 is the ceiling K + 1.' },
    { q: 'Speculative decoding helps least when…', options: ['The batch is small and the GPU is waiting on memory', 'The batch is large and the GPU is already compute-bound', 'The draft is a tiny model of the same family'], answer: 1,
      why: 'The trick spends idle arithmetic. When a large batch has already filled it, the K extra verified tokens per sequence cost real time (rho → 1 in your model) and the speedup can drop below 1. That is why serving engines treat speculation as a latency optimisation for small batches, not a throughput one.' },
    { q: 'Why does `generateSpeculative` compute α as accepted / examined rather than accepted / drafted?', options: ['They are always equal', 'Tokens drafted after the first rejection were never checked, so they say nothing about the draft\'s quality', 'Drafted is unknown'], answer: 1,
      why: 'A rejection at position i discards positions i + 1 … K − 1 unverified. Dividing by drafted underestimates α (about 0.23 instead of 0.5 in the test) and would make you pick the wrong K.' },
  ],
  concept: `
:::plain
A language model normally writes one token (a word or a piece of a word) per step, and each step is slow mainly because the chip has to read the whole model out of memory, not because of the arithmetic. Checking several proposed tokens in one step costs about the same as writing one, so speculative decoding lets a much cheaper helper, called the draft, guess the next few tokens and then has the real model check all of the guesses in a single step. Guesses the real model agrees with are kept, and at the first disagreement the real model supplies its own token instead, so one slow step can produce several tokens. A careful acceptance rule makes the output follow exactly the probabilities the real model would have used on its own, so quality does not change, only speed. For people who use LLMs at work, this is a common way providers make answers stream faster, and it helps most when a server is lightly loaded; under heavy load it can even slow things down.
:::

## The slack in a decode step

At small batch, one decode step reads every weight once and does almost no arithmetic per byte read. For a Llama-3-8B-shaped model in bf16 on an H100 that is approximately 16 GB through approximately 3.35 TB/s (NVIDIA's datasheet figure), about 5 ms, while the \`2N\` FLOPs take tens of microseconds. **A step that verifies K + 1 tokens costs about the same as a step that generates one**, because the weights are still read once.

But you do not have K + 1 tokens to verify: each depends on the one before it. Speculative decoding (Leviathan, Kalman & Matias 2023; Chen et al. 2023) takes the guesses from something cheaper, the **draft**, and checks them all with one pass of the real model, the **target**. That pass is the teacher-forced pass of module 07 run on the KV cache of module 15: feed \`ctx + K drafted tokens\`, read \`K + 1\` next-token distributions, roll the cache back on rejection.

## The acceptance rule, and why it is exact

Let \`p\` be the target's next-token distribution, \`q\` the draft's, and \`x ~ q\` the draft's proposal:

\`\`\`
keep x with probability  min(1, p[x] / q[x])
on rejection, draw from  residual = norm(max(0, p − q))
\`\`\`

The probability that \`x\` is proposed and kept is \`q[x] · min(1, p[x]/q[x]) = min(p[x], q[x])\`: accepted tokens carry exactly the mass on which \`p\` and \`q\` agree. The rejection branch draws precisely what is missing, \`max(0, p − q)\` scaled to sum to 1, and the total is \`p\` for every token. A bad draft only makes you reject more often.

:::predict
The target has \`p = [0.1, 0.45, 0.45]\` and the draft \`q = [0.9, 0.05, 0.05]\`. Over many trials, what fraction of proposals are accepted, and what is the output distribution?
---
Acceptance is \`Σ min(p, q) = 0.2\`. The output is still exactly \`[0.1, 0.45, 0.45]\`: the 80% of trials that reject draw from the residual \`norm([0, 0.4, 0.4]) = [0, 0.5, 0.5]\`, which tops up tokens 1 and 2.
:::

The per-position acceptance rate \`α = Σ min(p, q)\` equals \`1 − TV(p, q)\` and is the only number about the draft that matters. It moves whenever \`p\` is reshaped without \`q\`: Leviathan et al. report lower acceptance at temperature 1 than at 0, with draft and target sampled at the same temperature. The direction is not fixed: α is highest where the target's shape is closest to the draft's, so a flat draft agrees more with a hotter target. The demo sweeps the target's temperature with the draft held fixed and shows exactly that.

## Draft K, verify once

A step drafts K tokens autoregressively from \`q\`, runs the target once over all of them, and walks the drafts in order applying the rule. At the first rejection it draws the residual and stops: the later drafts were conditioned on a token that no longer exists. If all K survive, the target's \`(K + 1)\`-th row is a free extra draw, the **bonus token**.

If each draft survived independently with probability α, the expected tokens per step would be \`(1 − α^(K+1)) / (1 − α)\`. Real acceptances are not independent (easy stretches of text and hard ones come in runs, and α varies with the position in the draft), so treat it as a model to check against measurement, which the demo does.

:::predict
With α = 0.8, going from K = 4 to K = 8 raises the expected tokens per step from 3.36 to what, and at what cost?
---
To \`(1 − 0.8^9) / 0.2 = 4.33\`: one more token per step for twice the drafting and twice the verified tokens. Past a few positions \`α^K\` is small and extra drafts are mostly discarded, which is why deployed systems draft a handful of tokens, not dozens.
:::

## The cost model

Let one target step cost 1, one draft token cost \`c\`, and the verify pass cost \`1 + rho · K\`, where \`rho = 0\` means the extra tokens are free (memory-bound) and \`rho = 1\` means each costs a full step (compute-bound):

\`\`\`
speedup = expectedTokensPerStep(α, K) / (1 + K·c + rho·K)
\`\`\`

Three lessons. The draft must be cheap: at \`c = 1\` even α = 0.8 cannot reach 1×. It must agree with the target: at α = 0.2 and \`c = 0.2\` no K beats 1×. And the pass must have slack: at large batch the GPU is already compute-bound (module 16 filled that slack), \`rho\` heads to 1, and speculation slows you down. Serving engines therefore treat it as a latency optimisation for small batches; vLLM, for example, has offered a setting that switches speculation off once the batch passes a threshold.

## Where drafts come from

:::deeper Going deeper: the kinds of draft real systems use
- **A smaller model of the same family**, which needs a matching tokenizer and a second set of weights in memory.
- **n-gram / prompt lookup** (Saxena 2023; \`prompt_lookup_num_tokens\` in Hugging Face \`generate\`): propose what followed the last few tokens earlier in the context. Free, and strong on code.
- **Self-speculation**: Medusa (Cai et al. 2024) adds heads predicting tokens 2, 3, 4 ahead from the target's hidden state; EAGLE (Li et al. 2024) predicts the next feature vector instead and accepts more; DeepSeek-V3's multi-token-prediction module plays the same role. No second model; \`c\` is a small fraction of a step.
:::

## Where this toy differs from production

:::deeper Going deeper: how production systems verify drafts
Your target calls \`forward()\` over the whole window for every verify pass, so a pass costs the same as a decode step for a different reason than on a GPU: it is recompute-bound, not bandwidth-bound. A production pass runs on the KV cache, rolls it back on rejection, batches drafts across sequences, and verifies a *tree* of drafts (Medusa, EAGLE, SpecInfer) with a custom attention mask rather than one chain. And α on a 2-layer, 64-dimensional checkpoint says nothing about α on a 70-billion-parameter model; the rule, the residual and the cost model carry over unchanged.
:::
`,
  steps: [
    {
      id: 'accept',
      title: 'The acceptance rule for one token',
      instructions: `
Two small functions. \`p\` and \`q\` are plain arrays of probabilities over the same vocabulary, \`x\` is a token id the draft proposed, \`u\` is a uniform number in \`[0, 1)\`.

\`acceptProb(p, q, x)\`: return \`min(1, p[x] / q[x])\`. The skeleton already handles \`q[x] = 0\` (a token the draft could never have proposed) by returning 1, so the division is safe; you fill in the ratio.

\`shouldAccept(p, q, x, u)\`: return \`true\` exactly when \`u < acceptProb(p, q, x)\`.

Why this rule and not, say, "keep x if the target likes it at least as much as the draft"? Because kept mass per token must be \`q[x] · acceptProb = min(p[x], q[x])\`: the deterministic rule keeps \`q[x]\` where \`p ≥ q\` and nothing elsewhere, which is not a scaled copy of anything. The third test checks the kept histogram over 6,000 trials against \`min(p, q)\`.
`,
      predict: { question: 'With p = [0.5, 0.3, 0.2] and q = [0.25, 0.5, 0.25], what is acceptProb for tokens 0, 1 and 2?', answer: '1, 0.6 and 0.8. Token 0 has p/q = 2, clipped to 1: the target wants it more than the draft offers it. Tokens 1 and 2 were over-proposed and are thinned by 0.3/0.5 and 0.2/0.25.' },
      hints: [
        'The ratio p[x] / q[x] can exceed 1 when the target likes x more than the draft; a probability cannot, so clip it.',
        'acceptProb clips a ratio with `Math.min`. shouldAccept reuses acceptProb and compares the uniform against it with a strict `<`: an acceptance probability of 1 then keeps every u in [0, 1), and a probability of 0 keeps none, not even u = 0.',
        '`acceptProb`: `return Math.min(1, /* how much the target likes x relative to the draft */);`  `shouldAccept`: `return u < /* the acceptance probability of x */;`',
      ],
    },
    {
      id: 'residual',
      title: 'The residual distribution on rejection',
      instructions: `
\`residual(p, q)\`: return a new array \`norm(max(0, p − q))\`: clip every entry of \`p − q\` at zero, then divide by the sum so it is a distribution. If the sum is zero (\`p\` and \`q\` identical) return a copy of \`p\`; a rejection cannot happen then, but the function must still be total and must not return NaN.

\`speculativeSampleOne(p, q, next)\`: one position end to end. Propose \`x = sampleFrom(q, next())\`; if \`shouldAccept(p, q, x, next())\` return \`{ token: x, accepted: true }\`; otherwise return \`{ token: sampleFrom(residual(p, q), next()), accepted: false }\`. Draw a fresh uniform for each of the three (proposal, then acceptance, then residual, in that order, and the residual one only on rejection). A test feeds scripted uniforms and checks the exact result; reusing the proposal uniform for the acceptance test correlates the two draws and quietly biases the output away from \`p\`.

Work the 3-token case by hand once: \`p = [0.5, 0.3, 0.2]\`, \`q = [0.2, 0.1, 0.7]\`. Accepted mass is \`min(p, q) = [0.2, 0.1, 0.2]\`, total 0.5. The missing mass is \`[0.3, 0.2, 0]\`; scaled to 1 it is \`[0.6, 0.4, 0]\`, and drawing it on the 50% of rejections gives \`[0.2 + 0.3, 0.1 + 0.2, 0.2 + 0] = p\`. Token 2, which the draft over-proposed, gets no top-up at all.
`,
      hints: [
        'Two passes over the vocabulary: one to compute the clipped differences and their sum, one to divide by the sum.',
        'Clip with `Math.max(0, p[i] - q[i])`. Guard `z <= 0` before dividing. In speculativeSampleOne the residual is only computed on the rejection branch.',
        '`const out = new Array(p.length); let z = 0; for (i) { out[i] = /* clipped difference */; z += out[i]; } if (z <= 0) return Array.from(p); for (i) out[i] /= z; return out;`',
      ],
    },
    {
      id: 'verify',
      title: 'Draft K tokens, verify with one target pass',
      instructions: `
\`speculativeStep(target, draft, ctx, K, next)\`. Both models follow the interface of \`markovModel\`: \`model(ids, n)\` returns \`n\` distributions, one for the token after each of the last \`n\` positions of \`ids\`.

1. Draft: for \`i\` in \`0..K−1\`, call \`draft(ctx + drafted so far, 1)\`, sample a token with \`sampleFrom(q_i, next())\` from its row \`q_i\`, and remember both the token and the row. All K draft uniforms are drawn before any acceptance uniform, because drafting finishes before the verify pass starts.
2. Verify: call \`target(ctx + all K drafted tokens, K + 1)\` **once**. Row \`i\` is \`p_i\`, the target's distribution at the position the draft filled with token \`i\`; row \`K\` is the distribution after the last drafted token.
3. Walk \`i = 0..K−1\`: if \`shouldAccept(p_i, q_i, x_i, next())\` push \`x_i\`; otherwise push a sample from \`residual(p_i, q_i)\` and return \`{ tokens, accepted: i }\`.
4. If nothing was rejected, push a sample from row \`K\` (the bonus token) and return \`{ tokens, accepted: K }\`.

The result always has \`tokens.length === accepted + 1\`. Do not mutate \`ctx\`. The first test spies on both models: K draft calls with \`n = 1\`, exactly one target call with \`n = K + 1\`. Calling the target once per drafted token would still be correct and would throw away the entire point of the module. Another test checks that the bonus token is distributed as the target's last row, not the draft's.
`,
      predict: { question: 'The draft proposes 4 tokens; the second one is rejected. How many tokens does the step emit, and where does the last one come from?', answer: 'Two: the accepted first draft and a residual sample at position 1. Drafts 3 and 4 were conditioned on the rejected token and are discarded; there is no bonus token because the pass did not reach row K.' },
      hints: [
        'Keep two parallel arrays while drafting: the tokens and the rows they were drawn from. You need q_i again at verification time.',
        'After `const ps = target(ctx.concat(drafted), K + 1)`, loop i over the drafts; on the first rejection push the residual sample and return immediately with accepted = i. Fall through to the bonus draw only if the loop finishes.',
        '`for (let i = 0; i < K; i++) { if (shouldAccept(ps[i], qs[i], drafted[i], next())) tokens.push(drafted[i]); else { tokens.push(/* residual sample from ps[i], qs[i] */); return { tokens, accepted: i }; } } tokens.push(sampleFrom(ps[K], next())); return { tokens, accepted: K };`',
      ],
    },
    {
      id: 'measure',
      title: 'Acceptance rate and tokens per step',
      instructions: `
Three functions.

\`acceptanceRate(p, q)\`: the analytic per-position acceptance rate \`Σ_i min(p[i], q[i])\`. This equals \`1 − TV(p, q)\`; it is 1 for identical distributions and 0 for disjoint ones.

\`expectedTokensPerStep(alpha, K)\`: \`(1 − alpha^(K+1)) / (1 − alpha)\`, the sum \`1 + α + … + α^K\`. Return \`K + 1\` when \`alpha >= 1\` rather than dividing by zero.

\`generateSpeculative(target, draft, prompt, { K, maxNewTokens, next })\`: repeat \`speculativeStep\` until at least \`maxNewTokens\` tokens have been emitted, appending each step's tokens to the context. Return \`{ tokens, steps, drafted, examined, accepted, alpha, tokensPerStep, runLengths }\` where \`tokens\` is trimmed to exactly \`maxNewTokens\`, \`drafted = steps × K\`, \`examined\` counts only the drafted tokens verification reached (\`accepted + 1\` on a step that rejected, \`K\` on a step that accepted everything), \`alpha = accepted / examined\`, \`tokensPerStep = 1 + accepted / steps\`, and \`runLengths[i]\` is the number of steps that accepted exactly \`i\` tokens for \`i = 0..K\`. With \`maxNewTokens = 0\` no step runs; return \`alpha = 0\` and \`tokensPerStep = 1\` (the starter's defaults) rather than dividing zero by zero.

The distinction between \`examined\` and \`drafted\` is the whole of the measurement: tokens after a rejection were never checked and carry no information about the draft.
`,
      hints: [
        'One loop, one counter per field. Each step returns `accepted`; from it you know how many tokens were emitted (accepted + 1) and how many drafts were examined.',
        'examined per step: `r.accepted < K ? r.accepted + 1 : K`. Increment `runLengths[r.accepted]`. Extend the context with `r.tokens` before the next step.',
        '`while (tokens.length < maxNewTokens) { const r = speculativeStep(target, draft, ctx, K, next); steps++; accepted += r.accepted; examined += /* reached */; runLengths[r.accepted]++; tokens.push(...r.tokens); ctx = ctx.concat(r.tokens); }` then `alpha: accepted / examined, tokensPerStep: 1 + accepted / steps, tokens: tokens.slice(0, maxNewTokens)`.',
      ],
    },
    {
      id: 'speedup',
      title: 'The speedup model and the best K',
      instructions: `
\`speedup({ alpha, K, c, rho = 0 })\`: \`expectedTokensPerStep(alpha, K) / (1 + K·c + rho·K)\`. One plain target step costs 1. Each drafted token costs \`c\` (draft time over target time: about 0.001 for a table lookup, about 0.1 for a 10× smaller model, 1 for the target itself). The verify pass costs \`1 + rho·K\`: \`rho = 0\` when it is memory-bound and the extra tokens are free, \`rho = 1\` when it is compute-bound and each extra token costs a full step's worth of time.

\`bestK({ alpha, c, rho = 0, maxK = 16 })\`: the \`K\` in \`1..maxK\` with the largest modelled speedup; the smallest such \`K\` on a tie.

Check the three regimes yourself before running the tests: α = 0.8, K = 4, c = 0.1 gives 2.4× when memory-bound and 0.62× when compute-bound; α = 0.2 with c = 0.2 never beats 1× at any K (K = 1 ties at exactly 1×). This is the kind of model an engine needs to decide whether speculation pays at a given batch size, and it is why speculative decoding is a latency optimisation, not a throughput one.
`,
      hints: [
        'The numerator you already have from step 4. The denominator is time per step in units of one plain decode step: the drafts, the pass, and the pass\'s extra tokens if they are not free.',
        'bestK is a linear scan: track the best value seen and replace only on a strictly greater speedup, so ties keep the smaller K.',
        '`let best = 1, bestValue = -Infinity; for (let K = 1; K <= maxK; K++) { const s = /* speedup at this K */; if (s > bestValue) { best = K; bestValue = s; } } return best;`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why a step that verifies five tokens costs about the same as a step that generates one, and name the regime in which that stops being true.',
    'Walk through the acceptance rule and the residual for a 3-token vocabulary and show, in your own words, why the output is exactly the target distribution no matter how bad the draft is.',
    'The demo measured α for a bigram draft and for a sharpened copy of the target. Which would you deploy, and what would change your answer: the batch size, the temperature, or the cost of the draft?',
  ],
  stretch: [
    'Implement prompt-lookup drafting (Saxena 2023; `prompt_lookup_num_tokens` in Hugging Face `generate`): find the last 3 tokens earlier in the context and propose the tokens that followed them. Measure α on a prompt that repeats a passage.',
    'Add a KV cache to the target (lib/infer.js `forwardStep`) and roll it back to the last accepted token on rejection, as vLLM and TensorRT-LLM do; count the forward FLOPs saved versus the recompute used in the demo.',
    'Verify a tree of drafts instead of a chain: draft the top-2 tokens at each of 3 positions, build the causal mask that lets one pass score every path, and accept the longest surviving path. This is the mechanism in Medusa (Cai et al. 2024), EAGLE (Li et al. 2024) and SpecInfer (Miao et al. 2023).',
    'Batch it: run 8 sequences through the target at once with K = 4 and let `rho` rise with batch size in your speedup model. Find the batch at which speculation stops paying: the crossover a serving engine has to tune around. (The vLLM batch-size cutoff mentioned in the lesson is a hand-set version of this crossover.)',
  ],
  timeouts: { tests: 20000, demo: 120000 },
};
