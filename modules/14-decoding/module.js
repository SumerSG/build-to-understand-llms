export default {
  id: '14-decoding',
  title: 'Decoding & sampling',
  track: 'inference',
  minutes: 75,
  threshold: 'The model outputs a distribution, not a token; decoding is a separate, controllable policy over that distribution, and every knob is a different way of cutting off its tail.',
  goal: 'A logit-processor pipeline (temperature, top-k, top-p, min-p, repetition and frequency/presence penalties) plus a seeded sampler and a `generate` loop with stop sequences, whose sampled frequencies match the maths and which decodes the pre-trained checkpoint under four settings in the goal demo.',
  prereqs: ['01-tensors', '03-tokenizer', '04-bigram', '06-transformer'],
  recall: [
    { q: 'In module 01 you subtracted the row maximum inside softmax. What does adding the same constant to every logit do to the softmax output?', options: ['Nothing: softmax is shift-invariant', 'It scales every probability by exp(c)', 'It sharpens the distribution'], answer: 0,
      why: 'exp(x + c) / Σ exp(x + c) = exp(x) / Σ exp(x). Dividing by a temperature is a scale, not a shift, which is why it does change the distribution.' },
    { q: 'In module 04, `sampleNext` drew `u = next()` exactly once and then walked the cumulative sum. Why once?', options: ['A second draw would be slower', 'Every extra draw shifts the random stream that later tokens see, so results stop being reproducible across implementations', 'The rng only allows one call per token'], answer: 1,
      why: 'One uniform per token is the contract that makes "same seed, same output" hold between your sampler, the reference, and anything else in the lab.' },
    { q: 'In module 03, what does `BPETokenizer.decode(ids)` do with each id?', options: ['Looks up a byte and re-encodes UTF-8', 'Concatenates that id\'s vocabulary string onto the output', 'Inserts a space after each token'], answer: 1,
      why: 'Because decode is plain concatenation, a stop string such as "\\n\\n" or "<|end|>" can be split across two ids. You will check stop sequences on the decoded text, not on ids.' },
    { q: 'In module 13, why must a reported pass@k state the sampling temperature?', options: ['Temperature changes the grader', 'At temperature 0 every one of the n samples is identical, so pass@k collapses to pass@1; the diversity pass@k measures comes from the decoding policy', 'It does not; pass@k is independent of temperature'], answer: 1,
      why: 'Decoding is part of the measured system. This module is where that policy gets built.' },
  ],
  review: [
    { q: 'On a peaked distribution (one token at 95%), top-k with k = 40 and top-p with p = 0.9 keep, respectively…', options: ['40 tokens and 1 token', '1 token and 40 tokens', 'The same set'], answer: 0,
      why: 'Top-k keeps k tokens no matter how little mass they hold; top-p stops as soon as the cumulative mass reaches p, so a confident step keeps only the confident token.' },
    { q: 'Why are the penalties applied before temperature, and temperature before top-p?', options: ['It is faster', 'Each stage reshapes what the next stage measures: a penalty can change which tokens are in the nucleus, and temperature changes how much mass the same p covers', 'The order does not matter because the operations commute'], answer: 1,
      why: 'They do not commute. HF `generate`, vLLM and SGLang all fix the order: penalties, then temperature, then truncation. Applying top-p on raw logits and then cooling gives a different distribution.' },
    { q: 'Min-p with p = 0.1 drops every token whose probability is below…', options: ['0.1', '0.1 times the largest probability', 'The 10th largest probability'], answer: 1,
      why: 'The threshold is relative to the mode, so it tightens when the model is confident and loosens when it is not — which is why it stays sane at temperatures where top-p lets the tail in.' },
    { q: 'Why does greedy decoding tend to loop ("The painter is quiet in the garden. The painter is quiet in the garden.")?', options: ['The KV cache is corrupted', 'Once a phrase has been emitted, its continuation becomes the most likely token again, and argmax has no way to break the cycle', 'The model was undertrained'], answer: 1,
      why: 'Holtzman et al. 2019 showed that maximising likelihood produces text with a very different (more repetitive, lower-surprise) statistical profile than human text. Sampling or a repetition penalty breaks the loop.' },
    { q: 'A production server returns different text for the same prompt and seed at batch size 1 and batch size 8. The most likely cause is…', options: ['The seed is ignored', 'The logits differ by around 1e-6 because GPU kernels reduce in a different order for different batch shapes; one token near a sampling boundary flips and everything after it diverges', 'Top-p is non-deterministic'], answer: 1,
      why: 'Float addition is not associative. Same seed only implies same output when the arithmetic that produced the logits is bit-identical, which continuous batching does not guarantee.' },
  ],
  concept: `
## The model does not choose a token

Every model from modules 04–06 ends the same way: \`V\` logits, one per vocabulary entry, which softmax turns into a distribution over the next token. Nothing in the network picks one. That is the job of a separate program, the **decoder**, and you can change it without touching a weight. Here it is a pipeline of **logit processors**: each takes logits, returns new logits, and marks a dropped token with \`-Infinity\` (probability 0 after softmax).

## Why not just take the most likely token?

Greedy decoding, \`argmax\` at every step, loops: once "The painter is quiet in the garden." is in the context, its continuation is the most likely token again and argmax cannot break out. Holtzman et al. 2019 measured this on GPT-2: likelihood-maximising text is far more repetitive and less surprising than human text.

Sampling from the full distribution fixes the loops and introduces the opposite failure. A 50,000-token vocabulary has a long **tail**: thousands of tokens, each nearly impossible, that together carry real mass, often 5–10% of a step. Sample from all of it and every twentieth token is one the model would never have chosen, and it becomes context for the next. The tail is where errors live; decoding is mostly about where to cut it.

## Temperature: a sharpening operator

Temperature divides every logit by \`t\` before softmax: \`p_i ∝ exp(l_i / t)\`. \`t < 1\` widens the gaps and sharpens the distribution; \`t > 1\` flattens it. The limits are the two policies above: \`t → 0\` is greedy, \`t → ∞\` is uniform. Temperature moves mass between head and tail but never removes the tail.

:::predict
The logits \`[2, 1, 0, -1]\` give probabilities \`[0.64, 0.24, 0.09, 0.03]\`. At temperature \`0.5\`, what is the probability of the first token, roughly?
---
About \`0.87\`. Dividing by 0.5 doubles every gap: the logits become \`[4, 2, 0, -2]\` and \`e⁴ / (e⁴ + e² + 1 + e⁻²) ≈ 0.867\`. The tail token fell from 0.03 to 0.002 but is still there.
:::

## Cutting the tail: top-k, top-p, min-p

**Top-k** (Fan et al. 2018) keeps the \`k\` largest logits. One sort, one threshold, and a blind spot: \`k\` is fixed while the shape of the distribution is not. When the model is certain, \`k = 40\` keeps 39 near-impossible tokens; when it is undecided among 200 continuations, it throws 160 away.

**Top-p**, nucleus sampling (Holtzman et al. 2019, "The Curious Case of Neural Text Degeneration"), thresholds on mass instead: sort by probability, accumulate, keep the smallest prefix whose mass reaches \`p\`. A peaked step keeps one token, a flat step many.

:::predict
100 tokens are exactly equally likely. How many does top-p with \`p = 0.9\` keep, and how many does top-k with \`k = 10\`? Now one token holds 99.9% of the mass: same two questions.
---
Flat: top-p keeps 90 (float rounding can make it 91), top-k keeps 10. Peaked: top-p keeps 1, top-k still keeps 10. Top-p adapts to the shape; top-k does not.
:::

**Min-p** (Nguyen et al. 2024, "Turning Up the Heat") thresholds relative to the mode: drop every token below \`p × max(prob)\`. Its point is high temperature: at \`t = 1.5\` top-p lets the flattened tail in, while min-p's floor tracks the mode.

## Penalties, and why order matters

Two families discourage repetition. The **repetition penalty** of CTRL (Keskar et al. 2019) divides a seen token's logit by \`penalty\` if positive and multiplies it if negative, so both move towards \`-Infinity\`. The OpenAI-style **frequency** and **presence** penalties are additive: subtract \`frequency × count\` and \`presence × [seen at all]\`. Typical values are 1.1–1.3 and 0.1–0.5.

Your pipeline runs penalties, then temperature, then top-k, top-p and min-p, as HF \`generate\` and vLLM do. The order matters because each stage changes what the next one measures: a penalty can push a token out of the top-k, and cooling the logits shrinks the set the same \`p\` covers. The tests check it.

## Beam search, and why chat models skip it

Beam search keeps the \`b\` partial sequences with the highest total log-probability. It wins on translation and summarisation, where one right answer exists, and loses on open-ended text, where the most probable long sequence is bland and repetitive: greedy's failure at \`b\` times the compute and KV cache. Chat models sample with a tail cut.

## Seeds and determinism

Your sampler draws one uniform per token, so the same seed gives the same text; a test checks it. Production servers accept a \`seed\` and still warn that outputs may differ. Under continuous batching your request shares a batch with strangers; GPU kernels reduce in a different order for different batch shapes, float addition is not associative, and the logits differ at the \`1e-6\` level. Rarely, the uniform lands within \`1e-6\` of a cumulative-probability boundary, one token flips, and everything after it has a different history. Same seed, different output, both correct.

## Where this toy differs from production

You run one sequence at a time on a 256-token vocabulary with plain loops over a \`Float32Array\`, and top-p sorts the whole vector every step. A serving engine samples a whole batch in one fused GPU kernel, uses a partial sort for top-k, approximates top-p on a 128,000-token vocabulary where a full sort per step is too slow, and checks stop sequences incrementally in the detokenizer. The arithmetic is what you write here; only the batching and the kernels change.
`,
  steps: [
    {
      id: 'temperature',
      title: 'Temperature and greedy',
      instructions: `
Two functions. The worked examples above the TODO line (\`copyLogits\`, \`softmaxLogits\`, \`indicesByLogitDesc\`) set the conventions: every processor returns a **new** \`Float32Array\`, never touches its input, and marks a dropped token with \`-Infinity\`.

\`applyTemperature(logits, t)\`: return \`logits / t\` elementwise. \`t = 1\` is a plain copy. For \`t <= 0\` do not divide: return the limit directly, \`-Infinity\` everywhere except \`0\` at the argmax, so \`softmaxLogits\` of the result is exactly one-hot and nothing becomes \`NaN\`.

\`greedy(logits)\`: the index of the largest logit, first index on ties. It must work on all-negative logits, so do not start the search from a running maximum of 0.

Why a separate function for something \`argmax\` already does: greedy is the policy every other setting is compared against in the demo, and it is what \`temperature: 0\` must reduce to.
`,
      predict: { question: 'What is `softmaxLogits(applyTemperature([2, 1, 0, -1], 1e6))`?', answer: 'About `[0.25, 0.25, 0.25, 0.25]`: dividing by a huge t makes every logit ≈ 0, and softmax of equal logits is uniform. The tests check both limits.' },
      hints: [
        'Three cases, decided before the loop: t equals 1 (copy), t is at most 0 (one-hot on the argmax), otherwise divide. `argmaxArray` from lib/util.js is already imported.',
        'For the t <= 0 case: allocate the output, `fill(-Infinity)`, then set the argmax position to 0. For greedy, keep `best = 0` and compare `logits[i] > logits[best]` so negatives and ties behave.',
        '`const out = new Float32Array(logits.length); if (t <= 0) { out.fill(-Infinity); /* one line: put 0 at the argmax */ return out; } for (let i = 0; i < logits.length; i++) out[i] = /* … */; return out;`',
      ],
    },
    {
      id: 'topk',
      title: 'Top-k',
      instructions: `
Implement \`topKFilter(logits, k)\`: keep the \`k\` largest logits with their values unchanged and set every other entry to \`-Infinity\`. \`k <= 0\` or \`k >= logits.length\` means "off": return a copy.

A dropped token must be \`-Infinity\`, not \`0\`: a logit of 0 still has probability \`exp(0) / Z\` after softmax. Ties keep the earlier index, which \`indicesByLogitDesc\` already guarantees.

This is the cheapest tail cut (a sort and a threshold) and the one GPT-2's original sampling code used (\`top_k = 40\`). Its weakness, a fixed count on a distribution whose shape changes every step, is what the next step fixes; one test makes that visible.
`,
      hints: [
        'You need the indices of the k largest values, not the values themselves: `indicesByLogitDesc(logits)` gives you all indices in descending order.',
        'Allocate the output filled with -Infinity, then copy the logit at each of the first k indices of the sorted order.',
        '`/* guard the "off" cases first */ const order = indicesByLogitDesc(logits); const out = new Float32Array(logits.length); out.fill(-Infinity); for (let i = 0; i < k; i++) /* copy the logit at order[i] */; return out;`',
      ],
    },
    {
      id: 'topp',
      title: 'Top-p (nucleus)',
      instructions: `
Implement \`topPFilter(logits, p)\`: compute probabilities with \`softmaxLogits\`, walk the tokens in descending probability, accumulate the mass, and keep every token up to and **including** the one at which the cumulative mass first reaches \`p\` (\`mass >= p\`). Set the rest to \`-Infinity\`. \`p >= 1\` or \`p <= 0\` means "off".

Two details the tests check: the crossing token is kept, so at least one token always survives even at \`p = 0.01\`; and the walk is in probability order, not vocabulary order. Kept logits keep their values; \`softmaxLogits\` afterwards renormalises over the survivors.

Top-p is applied to whatever logits it is given. In the pipeline that is the temperature-scaled logits, so the same \`p\` keeps fewer tokens after cooling; the third test shows this.
`,
      predict: { question: 'Four tokens with probabilities 0.5, 0.3, 0.15, 0.05 and p = 0.75. How many survive?', answer: 'Two. 0.5 < 0.75, so the second token is needed; 0.5 + 0.3 = 0.8 ≥ 0.75, so the walk stops there. The token that crosses p is part of the nucleus.' },
      hints: [
        'Softmax first, then sort indices by logit (same order as by probability). The cumulative sum is over the sorted probabilities.',
        'Loop over the sorted indices: copy the logit into the output, add its probability to `mass`, and `break` as soon as `mass >= p`. Writing before checking is what keeps the crossing token.',
        '`const probs = softmaxLogits(logits); const order = indicesByLogitDesc(logits); const out = new Float32Array(logits.length); out.fill(-Infinity); let mass = 0; for (const i of order) { out[i] = logits[i]; /* one line: add to mass and break when it reaches p */ } return out;`',
      ],
    },
    {
      id: 'penalties',
      title: 'Min-p and the penalties',
      instructions: `
Three processors, each a few lines.

\`minPFilter(logits, p)\`: compute probabilities, find the largest, and set to \`-Infinity\` every token whose probability is below \`p × maxProb\`. The threshold is relative to the mode, not absolute: with \`p = 0.2\` and a top probability of 0.5, the floor is 0.1. \`p <= 0\` is off. The argmax always passes its own floor, so the result is never empty.

\`repetitionPenalty(logits, prevIds, penalty)\`: for every distinct id in \`prevIds\` (use a \`Set\`; ids outside the vocabulary are ignored), divide the logit by \`penalty\` if it is positive and multiply it by \`penalty\` if it is negative. Both cases push the token towards \`-Infinity\`; a plain subtraction or a plain division gets one sign wrong. \`penalty = 1\` is off. This is the rule from CTRL (Keskar et al. 2019) and the one \`repetition_penalty\` implements in HF and vLLM.

\`frequencyPresencePenalty(logits, prevIds, { frequency, presence })\`: count how many times each id occurs in \`prevIds\`, then subtract \`frequency × count + presence\` from each seen id's logit. Presence is paid once per distinct id, frequency once per occurrence. These are the OpenAI API's \`frequency_penalty\` and \`presence_penalty\`.
`,
      hints: [
        'Min-p: the floor is `p * maxProb`, computed once; then a single pass comparing `probs[i] < floor`. Repetition: iterate the Set, and the sign test is on the current logit value.',
        'For the additive penalties, build a `Map` from id to count in one pass over prevIds, then one pass over the map: `out[id] -= frequency * n + presence`.',
        'Repetition penalty core: `const out = copyLogits(logits); for (const id of new Set(prevIds)) { if (id < 0 || id >= out.length) continue; out[id] = /* positive ? divide : multiply */; } return out;`',
      ],
    },
    {
      id: 'sample',
      title: 'The pipeline and the sampler',
      instructions: `
\`processLogits(logits, opts)\` runs the processors in this order and returns the processed logits:

1. \`repetitionPenalty\` with \`opts.prevIds\` and \`opts.repetitionPenalty\` (default 1)
2. \`frequencyPresencePenalty\` with \`opts.frequencyPenalty\` and \`opts.presencePenalty\` (default 0)
3. \`applyTemperature\` with \`opts.temperature\` (default 1)
4. \`topKFilter\` with \`opts.topK\` (default 0), then \`topPFilter\` with \`opts.topP\` (default 1), then \`minPFilter\` with \`opts.minP\` (default 0)

Every default is that processor's "off" value, so \`processLogits(logits)\` is a copy. The order is the one HF \`generate\` and vLLM use, and the first test checks two consequences: a penalty applied before top-k can change which tokens survive, and temperature applied before top-p changes how many.

\`sample(logits, opts)\`: \`softmaxLogits(processLogits(logits, opts))\`, then draw **one** uniform \`u = opts.next()\` and return the first index whose cumulative probability exceeds \`u\` (the inverse-CDF walk from module 04). If rounding leaves the cumulative sum a hair below 1, return the last token with non-zero probability. Throw if \`opts.next\` is not a function.

The empirical test takes 20,000 draws and compares the histogram with \`softmax(processed)\`: a missing temperature, a missing renormalisation, or sampling from the unprocessed logits all show up as a gap of several points.
`,
      predict: { question: 'You apply top-p 0.9 first and temperature 0.3 second, instead of the pipeline order. Do more or fewer tokens survive than in the correct order?', answer: 'More. On `[2, 1, 0, -1]` top-p at temperature 1 keeps three tokens; cooling afterwards cannot remove any. In the correct order the cooled logits put 0.965 on the first token and top-p keeps one.' },
      hints: [
        'processLogits is six calls, each feeding the previous result. Destructure the options with defaults in the signature (the starter already does) so an absent option is "off".',
        'sample: `const probs = softmaxLogits(processLogits(logits, opts)); const u = next();` then `acc += probs[i]; if (u < acc) return i;` exactly as in module 04.',
        '`let out = repetitionPenalty(logits, prevIds, rep); out = frequencyPresencePenalty(out, prevIds, { frequency: frequencyPenalty, presence: presencePenalty }); /* temperature, then the three truncations */ return out;`',
      ],
    },
    {
      id: 'generate',
      title: 'The decode loop with stop sequences',
      instructions: `
Implement \`generate(model, tokenizer, prompt, opts)\` on top of \`lib/infer.js\` (\`newCache\`, \`prefill\`, \`forwardStep\` are imported). Options: \`maxNewTokens\` (default 50), \`next\` (required), \`stop\` (array of strings, default \`[]\`), \`stopAtEos\` (default true); every other option is passed through to \`sample\`. Return \`{ text, ids, finishReason }\` where \`finishReason\` is \`'length'\`, \`'stop'\` or \`'eos'\`.

The loop:

1. \`promptIds = tokenizer.encode(prompt)\`. If empty, use \`[tokenizer.eos]\` (throw if the tokenizer has no eos). Keep only the newest \`model.config.blockSize\` ids: the context window is fixed.
2. \`cache = newCache(model)\`; \`logits = prefill(model, cache, promptIds)\`.
3. While fewer than \`maxNewTokens\` ids have been generated: \`id = sample(logits, { ...samplingOpts, prevIds: promptIds.concat(ids), next })\`. If \`stopAtEos\` and \`id === tokenizer.eos\`, finish with \`'eos'\` without pushing it. Push the id and decode the generated ids to \`text\`. If any stop string occurs in \`text\`, cut \`text\` at the earliest occurrence and finish with \`'stop'\` (the \`ids\` keep everything sampled). If the cache is full (\`cache.length >= blockSize\`), finish with \`'length'\`. Otherwise \`logits = forwardStep(model, cache, id)\`.

Check stop sequences on the decoded **text**, not on ids: a two-character stop can straddle two tokens, and the test makes sure it is still found. Passing \`prevIds\` is what lets the penalties see the history; the first test rebuilds the loop with your own \`sample\` and requires identical ids.
`,
      hints: [
        'Write the loop first without stop sequences or eos and get the "identical to the reference loop" test passing; then add the two early exits.',
        'Keep `text = tokenizer.decode(ids)` up to date each step. For the stop check, loop over `stop`, use `text.indexOf(s)`, and remember the smallest non-negative index so the earliest stop wins.',
        '`while (ids.length < maxNewTokens) { const id = sample(logits, { ...samplingOpts, prevIds: promptIds.concat(ids), next }); if (stopAtEos && id === tokenizer.eos) { finishReason = "eos"; break; } ids.push(id); text = tokenizer.decode(ids); /* stop-sequence check: cut text, set "stop", break */ if (ids.length >= maxNewTokens || cache.length >= blockSize) break; logits = forwardStep(model, cache, id); }`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why "the model chose a token" is the wrong mental model, and what changes about debugging a chat product once you see decoding as a policy you own.',
    'Top-k, top-p and min-p all cut the tail. Describe a distribution shape where each one gives the best behaviour and one where it gives the worst.',
    'Your sampler is bit-for-bit reproducible under a seed. List the assumptions that made that true and which of them a GPU serving engine with continuous batching breaks.',
  ],
  stretch: [
    'Add beam search (`beamSearch(model, tokenizer, prompt, { beams, maxNewTokens })`) that keeps the `b` best partial sequences by total log-probability, and compare its output with greedy and top-p on the checkpoint; then read why HF `generate` applies a length penalty to it and why open-ended chat models leave it off.',
    'Implement the `no_repeat_ngram_size` processor from HF `generate`: ban any token that would complete an n-gram already present in the context. Measure how it changes the repetition rate of greedy decoding on the checkpoint.',
    'Return per-token log-probabilities from `generate` the way the OpenAI API\'s `logprobs` and vLLM\'s `SamplingParams(logprobs=…)` do, and compute the total log-probability of the chosen sequence under each setting.',
    'Read vLLM\'s sampler (`vllm/v1/sample/sampler.py`) and list its processor order next to yours; then find where it applies top-p on a 128k vocabulary and why a full sort per step is avoided.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
