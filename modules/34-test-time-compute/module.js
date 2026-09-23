export default {
  id: '34-test-time-compute',
  title: 'Reasoning & test-time compute',
  track: 'inference',
  minutes: 105,
  threshold: 'Once a model sometimes solves a problem, sampling more and choosing better turns inference compute into accuracy, and the selector sets the ceiling: majority vote converges to the model\'s most common answer, best-of-N against a learned reward model eventually picks the reward model\'s favourite mistakes, and only a sound verifier tracks pass@k.',
  goal: 'Self-consistency voting, best-of-N with outcome and process reward models, step-level beam search and thinking budgets on a verifiable reasoning task, with the accuracy each buys per GPU-second.',
  prereqs: ['12-rlvr', '13-evals', '15-kv-cache', '16-batching'],
  recall: [
    { q: 'Module 13\'s unbiased estimator: from n = 10 samples of which c = 2 are correct, what is pass@5?', options: ['0.2', 'About 0.78', '1.0'], answer: 1,
      why: '`1 − C(8, 5) / C(10, 5) = 1 − 56 / 252 ≈ 0.78`: the chance that five draws without replacement include at least one of the two correct samples. In this module it is exactly the accuracy of best-of-5 with a perfect verifier.' },
    { q: 'In module 12\'s GRPO, a group of G samples for one prompt all get reward 0. What advantage does each sample get?', options: ['−1 each', '0 each: the group mean is 0 and nothing is above or below it', 'The KL penalty alone'], answer: 1,
      why: 'Advantages are rewards minus the group mean, over the group\'s standard deviation. A prompt the policy never solves teaches nothing, so RLVR only sharpens what sampling already sometimes finds: the same pass@k ceiling that bounds best-of-N here.' },
    { q: 'Module 15\'s KV-cache formula `2 · nLayer · nKVHead · headDim · bytes` per token, for 32 layers, 8 KV heads of 128 dimensions in bf16, gives…', options: ['About 32 KiB per token', 'About 128 KiB per token', 'About 1 MiB per token'], answer: 1,
      why: '`2 · 32 · 8 · 128 · 2 = 131,072` bytes. An 8,000-token reasoning trace therefore holds about 1 GB of KV cache, for one sequence.' },
    { q: 'Module 16 models a decode iteration as `tFixed + tPerToken · tokens` with tFixed = 5 ms and tPerToken = 50 µs. Decoding 16 sequences side by side instead of 1 makes each iteration…', options: ['16 times slower', 'About 15% slower (5.8 ms instead of 5.05 ms)', 'Exactly as fast'], answer: 1,
      why: 'Decode is memory-bound: reading the weights (tFixed) dominates, and extra sequences in the batch add only their arithmetic. That is why sampling 16 answers in parallel costs so little extra latency.' },
    { q: 'Module 11\'s Bradley–Terry reward model is trained so that `P(chosen ≻ rejected) = sigmoid(r_chosen − r_rejected)`. What does that training guarantee about the reward on responses unlike any in the training pairs?', options: ['It is calibrated', 'Nothing: the score is only constrained where the data were', 'It is always lower'], answer: 1,
      why: 'A reward model is a function fitted on finite pairs. Push hard enough on it (many samples, or RL) and you find inputs where its score and the truth disagree, which is reward-model overoptimisation.' },
  ],
  review: [
    { q: 'A chain has per-step error rate e = 0.4 over 3 steps, and every slip adds +1. What does majority vote do as N grows?', options: ['It converges to the right answer', 'It converges to "off by one", which has probability 0.432 against 0.216 for the right answer', 'It stays at 0.216'], answer: 1,
      why: 'Voting converges to the mode of the model\'s answer distribution. When a wrong answer is the mode, more samples make the vote more reliably wrong.' },
    { q: 'Best-of-N with a sound verifier on a pool where c of n samples are correct has expected accuracy…', options: ['c / n, whatever N is', 'pass@N = `1 − C(n − c, N) / C(n, N)`', 'Always 1'], answer: 1,
      why: 'A perfect verifier picks a correct trace whenever one is among the N chosen, which is the definition of pass@N. It is the ceiling for every selector.' },
    { q: 'Why does the process reward model aggregate its step scores with the minimum rather than the mean?', options: ['It is cheaper to compute', 'One wrong step makes the whole trace wrong, so the weakest step should decide the score', 'The mean is not defined for partial traces'], answer: 1,
      why: 'With the mean, a long trace with one bad step can outscore a short correct one. Lightman et al. (2023) score a solution by the product of step-correctness probabilities, which a single bad step also drives towards zero.' },
    { q: 'You have a fixed decode budget of 8,000 tokens per problem on one GPU. Why is 8 parallel traces of 1,000 tokens much faster than one trace of 8,000?', options: ['Parallel traces need less KV cache', 'The 8 traces decode as one batch: 1,000 memory-bound iterations instead of 8,000', 'Short traces use a smaller model'], answer: 1,
      why: 'Each decode iteration pays tFixed once however many sequences are in the batch. The KV bytes are the same either way (8,000 tokens held), but the serial trace pays tFixed 8,000 times.' },
    { q: 'Best-of-N against the length-biased ORM gets worse beyond some N. What is happening?', options: ['The sampler gets worse', 'More samples contain more long, wrong traces, and the ORM\'s spurious preference for length ranks one of them first', 'Ties are broken badly'], answer: 1,
      why: 'Selection is optimisation: the harder you select on a proxy, the more you pick its errors (Gao et al. 2022). The oracle has no such errors, so its curve only rises.' },
  ],
  concept: `
:::plain
Test-time compute means letting a model do more work on each question as it answers, instead of only making the model bigger or training it longer. There are two basic ways: have it write several independent attempts and pick one, or let a single attempt reason step by step for longer, which is what "reasoning" or "thinking" models such as OpenAI's o1 and DeepSeek-R1 do. Either way something must choose the final answer, such as a majority vote or a separate scoring model, and this module shows that a flawed scorer can make results worse the more attempts it sees. The extra thinking is not free: each thinking token (a word or a piece of a word) is generated like any other output token, and providers such as OpenAI bill hidden reasoning tokens as output tokens. That is why a "reasoning effort" or thinking-budget setting trades answer quality against cost and waiting time.
:::

## A second axis for scaling

Module 08 scaled *training* compute. Here you spend compute *at inference*, per problem. Snell et al. (2024) showed that, spent well, test-time compute can beat a model about 14 times larger on problems the small model already sometimes solves. OpenAI's o1 (2024) and DeepSeek-R1 (2025) made the idea standard: they write long reasoning traces before answering, and o1's report plots accuracy rising with both training and thinking compute.

There are two ways to spend it. **Parallel**: sample N independent traces and choose one. **Sequential**: let one trace think longer. Either way, a **selector** must turn many tokens into one answer; it is what this module is about.

Your testbed is a scripted reasoner on arithmetic chains such as \`51 → +17 → −9 → +30\`. Each chain has its own per-step error rate \`e\` (3% to 40%) and \`s\` steps (3 to 6). A slip at step \`i\` always lands the same distance \`δᵢ\` from the right value (±1 or ±10), and later steps compute correctly from the wrong value. So a trace is right with probability \`(1 − e)^s\`, and wrong answers **cluster** as real models' do.

## Selectors

**Self-consistency** (Wang et al. 2022): sample N traces, return the most common final answer. It needs no reward model, but it converges to the *mode* of the model's answer distribution.

:::predict
A chain has \`e = 0.4\`, 3 steps, and every slip adds +1. The right answer has probability \`0.6³ = 0.216\`; "off by one" has \`3 · 0.4 · 0.6² = 0.432\`. What happens to majority-vote accuracy as N goes from 1 to 31?
---
It falls from about 0.22 to nearly 0: the vote converges to the mode, and here the mode is wrong. Across the demo's 300 chains the net gain is modest: from 42% at N = 1 to 60% at N = 64.
:::

**Best-of-N with a verifier** (Cobbe et al. 2021, which also introduced GSM8K): score each trace, keep the highest. With a *sound* verifier, like module 12's program checker, best-of-N is right whenever any sample is right, so its accuracy is exactly module 13's pass@N. That is the ceiling. It is also what RLVR (module 12) trains against: RL sharpens the policy toward what sampling already finds, and in R1 the traces grew longer as training went on.

Most problems have no program checker, so you train a **reward model**. An **outcome reward model** (ORM) scores the finished trace. A **process reward model** (PRM) scores every step (Uesato et al. 2022; Lightman et al. 2023, "Let's Verify Step by Step", trained on PRM800K, about 800,000 human step labels). Selecting with their PRM, best-of-N solved about 78% of a representative MATH subset, ahead of their ORM.

A learned reward model is a proxy with its own errors. Gao et al. (2022) measured **reward-model overoptimisation**: as you select or train harder against a proxy, the true reward rises, peaks and falls. Your ORM prefers longer traces, a documented bias of real reward models (Singhal et al. 2023 found that much of the reward gain in RLHF comes from longer outputs), and slipped steps are longer.

:::predict
Best-of-N against that ORM is right about 42% of the time at N = 1. What does it do at N = 64?
---
In the demo it peaks at about 58% near N = 8 and falls to about 46% at N = 64. With 64 samples there is almost always a long, wrong trace whose length bonus beats a correct trace's margin. The oracle, over the same pools, climbs to 100%.
:::

## Searching over steps

A PRM can score *partial* traces, so you can search: keep the best few prefixes, extend each by one step, and repeat. This is **step-level beam search**. It throws a bad step away where it happens, instead of paying for the whole wrong trace. With a good PRM it beats best-of-N at the same number of sampled tokens. Tree search such as MCTS wins at games, but the R1 report lists both PRMs and MCTS among its *unsuccessful* attempts. Step labels were costly, learned PRMs got reward-hacked, and a language step has far more continuations than a Go move.

## Budgets and the bill

**Budget forcing** caps thinking. s1 (Muennighoff et al. 2025) stops a trace at a token limit and forces the answer, or appends "Wait" to make the model keep going. Commercial APIs expose the same control as a "reasoning effort" setting or a thinking-token budget.

Every thinking token is a decode step (module 15). Its KV cache grows with the trace, about 128 KiB per token for a Llama-3-8B-like model in bf16, so about 1 GB for an 8,000-token trace. Decode is memory-bound (module 16), so N parallel samples share each iteration's fixed cost, while one long trace pays that cost once per token. Providers such as OpenAI bill reasoning tokens as output tokens even when they do not show them to you.

## Toy versus production

The reasoner is scripted, not trained: its errors are independent per step and drawn at a fixed rate, and its "reward models" are the exact checker plus a length bias and Gaussian noise. The curves show how selection works and what it costs, not the numbers of any real model.

:::deeper Going deeper: how real models and searches differ
Real models' errors correlate across samples, real PRMs are much noisier than this one, real traces run to thousands of tokens rather than about 55, and real beam search shares prefix KV (module 17) and pays for PRM forward passes.
:::
`,
  steps: [
    {
      id: 'reasoner',
      title: 'The task and the reasoner',
      instructions: `
The worked examples at the top of the starter give you the task. \`makeChains(n, seed)\` builds chains: \`{ start, ops, values, answer, steps, errorRate, slips }\`. \`finishTrace(steps)\` closes a list of steps into a trace \`{ steps, answer, tokens }\`, and \`verify(chain, trace)\` is module 12's reward (1 if the final answer is right). Read them first.

Build the reasoner and the step checker:

- \`sampleStep(chain, prefix, next)\` writes step \`i = prefix.length\`, continuing from the prefix's last value (or \`chain.start\`). This is a completion problem: the index and the previous value are done for you. Draw \`next()\` **once**: the step slips when that draw is \`< chain.errorRate\`, and a slip adds \`chain.slips[i]\` to the right value. Then draw the hedge: \`tokens = BASE_TOKENS + randInt(next, HEDGE_MAX + 1)\`, plus \`DETOUR_TOKENS\` on a slip. Keep that order (slip draw first, hedge draw second): a test replays your trace against the reference with the same seed. Throw if the chain has no step \`i\`.
- \`sampleTrace(chain, next)\` samples \`chain.steps\` steps in order and closes them with \`finishTrace\`.
- \`checkSteps(chain, trace)\` returns one boolean per step: is it the right operation applied to **the trace's own** previous value? A step that computes \`22 + 7 = 29\` correctly after an earlier slip is a correct step. This is the step-level verifier the PRM is built from in step 4.

Because each slip carries forward and the same step always slips the same way, a trace is right with probability \`(1 − e)^s\`, and a wrong answer is off by exactly the sum of its slips.
`,
      predict: { question: 'A 5-step chain has e = 0.2. What fraction of sampled traces reach the right answer?', answer: '`0.8⁵ ≈ 0.33`. A trace is right only if all five steps are, and a slip is never repaired later. A per-step error rate that sounds low still sinks two traces in three.' },
      hints: [
        'Where does step i start from: the chain\'s true value, or whatever the trace wrote before it? Which one makes a slip carry forward?',
        'Compute the right value with applyOp on prev and chain.ops[i]. Decide the slip with a single draw of next(), then make the second draw for the hedge. sampleTrace is a while loop that pushes sampleStep(chain, steps, next) until it has chain.steps steps. checkSteps walks the steps carrying prev, starting from chain.start.',
        '`const right = applyOp(prev, chain.ops[i]); const slipped = next() < chain.errorRate;` then `tokens = BASE_TOKENS + randInt(next, HEDGE_MAX + 1) + /* detour if slipped */`. In checkSteps: `out.push(s.value === applyOp(prev, chain.ops[i])); prev = /* which value? */;`',
      ],
    },
    {
      id: 'vote',
      title: 'Self-consistency',
      instructions: `
Implement \`majorityVote(answers)\`: return the most common answer.

- \`null\` and \`undefined\` are **abstentions** (a trace cut off before it answered, in step 5). They are not counted. \`0\` and negative numbers are real answers: some chains end there. If every answer abstains, or the list is empty, return \`null\`.
- Ties go to the answer whose **first occurrence comes earliest**, so the vote is deterministic: \`majorityVote([7, 3, 3, 7]) === 7\`.
- Do not modify the input.

The test also checks the threshold behaviour. On an easy chain, voting over 31 samples is almost always right. On a chain where "off by one" is more likely than the right answer, voting over 31 samples is almost never right.
`,
      hints: [
        'You need a count per distinct answer and a way to remember which answer appeared first. Which built-in keeps keys in insertion order?',
        'Count into a Map (skipping null and undefined), then scan the Map\'s entries in insertion order and keep an answer only if its count is strictly greater than the best so far. Strictly greater is what makes the earliest answer win a tie.',
        '`for (const a of answers) { if (/* an abstention? test null and undefined, not falsiness */) continue; counts.set(a, /* old count or 0, plus 1 */); }` Then one pass over `counts` that keeps a running best answer and best count, starting from `null` and `-Infinity`.',
      ],
    },
    {
      id: 'bestofn',
      title: 'Best-of-N and outcome reward models',
      instructions: `
Four functions:

- \`weightedVote(answers, scores)\`: like \`majorityVote\`, but each sample adds its score to its answer's total. The answer with the largest total wins, even when every total is negative. Same abstention and tie rules. (\`majorityVote\` is \`weightedVote\` with every score 1, so you may rewrite it that way.)
- \`bestOfN(traces, scorer)\`: the trace (the object itself) with the largest \`scorer(trace)\`; the earliest wins a tie; a single negative-scored trace is still returned.
- \`verbosity(trace)\`: extra tokens per step beyond \`BASE_TOKENS\`, with the answer line excluded: \`(tokens − ANSWER_TOKENS) / steps − BASE_TOKENS\`.
- \`ormScore(chain, trace, { lambda = 0.3, sigma = 0.1 })\`: the outcome reward model, \`verify + λ·verbosity + σ·rmNoise('orm', chain, trace.steps)\`. \`rmNoise\` is given. It is a fixed pseudo-random error for each input, as a real reward model's errors are fixed functions of their input.

A slipped step writes \`DETOUR_TOKENS\` extra tokens, so this ORM overrates wrong traces. Two tests show the consequence. With the oracle \`verify\` as scorer, best-of-N averaged over every subset of a pool is exactly \`passAtK(n, c, N)\`. With the ORM, accuracy rises, peaks and falls as N grows.
`,
      predict: { question: 'A 3-step trace with one slip has 6 more tokens than the same trace without it. By how much does that raise its ORM score at λ = 0.3, compared with the 1 point a correct answer earns?', answer: '6 extra tokens over 3 steps is +2 verbosity, so +0.6. One slip does not outweigh correctness on average, but hedging varies by up to 4 tokens a step. Among 64 samples there is usually a wrong trace whose total bonus beats the right one\'s. Two slips (+1.2) beat it outright.' },
      hints: [
        'bestOfN is an argmax over a list with a custom key. weightedVote is majorityVote where each sample adds its score instead of 1.',
        'Keep bestScore starting at -Infinity and replace the current best only when a score is strictly greater. That makes the earliest trace win ties and returns a lone negative-scored trace. For verbosity, subtract the answer line before dividing by the number of steps.',
        '`const s = scorer(t); if (s > bestScore) { best = t; bestScore = s; }`. ORM: `return verify(chain, trace) + lambda * verbosity(trace) + sigma * /* the given noise helper, tag \'orm\' */;`',
      ],
    },
    {
      id: 'process',
      title: 'Process reward models and step-level beam search',
      instructions: `
The PRM scores **every step**, so it can judge a partial trace:

- \`stepScores(chain, trace, { sigma = 0 })\`: for step \`i\`, 1 if \`checkSteps\` says it is correct, else 0, plus \`σ · rmNoise('prm', chain, trace.steps.slice(0, i + 1))\` when \`sigma\` is not 0. The noise depends only on the prefix up to that step, so a prefix keeps its score however it is later extended.
- \`prmScore(chain, trace, opts)\`: the **minimum** of the step scores, and 1 for an empty trace, which has made no mistake yet.
- \`stepBeamSearch({ beamWidth, depth, expand, prm })\`: start from one empty prefix \`[]\`. At each of \`depth\` levels, call \`expand(prefix)\` on every prefix in the beam; each call returns an array of longer prefixes. Score every candidate with \`prm(candidate)\` and keep the \`beamWidth\` best. Sort stably, so earlier candidates win ties. Return \`{ best, beam, sampled }\`, where \`sampled\` counts every candidate generated.

\`stepBeamSearch\` knows nothing about chains: prefixes are opaque arrays, and the tests drive it with digits as well as reasoning steps. The last test runs it on 6-step chains with \`e = 0.35\`, where a whole trace is right only 7.5% of the time, and compares it with best-of-N at the same number of sampled steps.
`,
      predict: { question: 'Beam width 2, four children per prefix, depth 6. How many steps does beam search sample, and how many whole 6-step traces is that for best-of-N?', answer: '4 at depth 1 (one root), then 2 × 4 = 8 at each of the 5 later depths: 44 steps, the budget of 7 whole traces. Beam search spends them where they matter, and reaches over 90% where best-of-7 with the same PRM stays near 40%.' },
      hints: [
        'Two things make search over steps work: a score for a prefix that never goes up when a bad step is added, and a beam that keeps several prefixes alive so one lucky-looking step does not commit you.',
        'stepScores maps checkSteps to 1/0 and adds the prefix noise. For beam search, loop depth times: gather every (candidate, score, index) from expanding each beam prefix, sort by score descending and then index ascending, slice the first beamWidth, and add the number of candidates to sampled.',
        '`for (const p of beam) for (const c of expand(p)) cands.push({ c, s: prm(c), i: cands.length });` then `cands.sort((a, b) => /* score desc, then index asc */); beam = cands.slice(0, beamWidth).map((x) => x.c);`',
      ],
    },
    {
      id: 'budget',
      title: 'Thinking budgets and the serving bill',
      instructions: `
- \`withBudget(trace, maxTokens)\`: budget forcing. Keep whole steps while the steps kept so far **plus the answer line** (\`ANSWER_TOKENS\`) fit in \`maxTokens\` (fitting exactly counts). If every step fits, return a copy of the trace with \`truncated: false\`. Otherwise stop at the first step that does not fit and force an answer: \`finishTrace(kept)\` with \`truncated: true\`. Its answer is the last kept value, or \`null\` if nothing fit. Do not modify the input.
- \`ttcCost({ samples, traceTokens, promptTokens = 0 }, config)\`: the serving cost of \`samples\` sequences decoded together, each writing \`traceTokens\` tokens after one shared prefill:

\`\`\`
decodeTokens = samples · traceTokens
decodeSteps  = traceTokens                        (one batched iteration per position)
peakKVBytes  = cacheBytes(prompt) + samples · cacheBytes(traceTokens)
seconds      = [tFixed + promptTokens · tPerToken] + decodeSteps · (tFixed + samples · tPerToken)
gpuSeconds   = seconds · gpus
\`\`\`

The bracketed prefill term is paid only when \`promptTokens > 0\`. \`cacheBytes\` is module 15's formula (given); pass \`bytesPerElement\` from the config. Fields missing from \`config\` fall back to \`DEFAULT_COST\`, which holds approximately Llama-3-8B dimensions (32 layers, 8 KV heads of 128) in bf16 and module 16's \`tFixed = 5 ms\`, \`tPerToken = 50 µs\`.
`,
      predict: { question: 'With the default config and a 200-token prompt: how long do 8 parallel traces of 1,000 tokens take, and how long does one trace of 8,000 tokens take? Which holds more KV cache at its peak?', answer: 'Parallel: `0.015 + 1000 · (0.005 + 8 · 0.00005) = 5.415 s`. Serial: `0.015 + 8000 · 0.00505 = 40.4 s`, about 7.5 times longer. Peak KV is identical, `131,072 · 8,200` bytes (about 1.07 GB), because both hold 8,000 decoded tokens. Thinking longer costs latency; thinking wider reaches the same peak KV eight times sooner, so the memory must be free at once.' },
      hints: [
        'withBudget is a running total that stops at the first step that would overflow once the answer line is added. ttcCost is arithmetic: write each line of the formula block as one line of code.',
        'In withBudget, keep a `used` count of step tokens and break when `used + step.tokens + ANSWER_TOKENS > maxTokens`. Compare the kept count with the trace\'s step count to tell whether you truncated. In ttcCost, merge the config over DEFAULT_COST first, and count the prompt\'s KV once and each sample\'s trace KV separately.',
        '`const cfg = { ...DEFAULT_COST, ...config };` then `peakKVBytes = cacheBytes(cfg.model, promptTokens, opt) + samples * /* one sample\'s trace KV */;` and `seconds = prefill + decodeSteps * (/* one batched iteration */);` with `prefill = promptTokens > 0 ? /* one prefill iteration */ : 0`.',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why majority vote, best-of-N against a learned reward model and best-of-N against a program verifier have different ceilings as N grows. Where does each converge, and why?',
    'Your beam search beat best-of-N at equal sampled tokens with a PRM that was the exact step checker plus noise. Which property of that PRM did the result depend on most, and why might a real learned PRM (the R1 report\'s "unsuccessful attempt") not have it?',
    'A product team asks for "more reasoning" on a hard task. Using your `ttcCost`, explain when you would spend the budget on more parallel samples and when on longer traces, and what each does to latency, KV memory and the bill.',
  ],
  stretch: [
    'Replace beam search with Monte Carlo tree search: PUCT selection as in AlphaZero, with rollouts to the end of the chain scored by `verify`. Compare it with beam search at equal sampled steps, then read the "Unsuccessful Attempts" section of the DeepSeek-R1 report on why MCTS did not scale for them.',
    'Build a PRM without step labels, as Math-Shepherd (Wang et al. 2024) does: estimate each prefix\'s score as the fraction of rollouts from it that reach the right answer. Compare the cost in sampled tokens with the noise-free PRM\'s accuracy.',
    'Beam candidates share their prefixes. Account for that with module 17\'s radix-tree prefix cache, the idea behind SGLang\'s RadixAttention and fork primitive and vLLM\'s automatic prefix caching, and recompute beam search\'s peak KV bytes in `ttcCost`.',
    'Implement s1-style budget *extension*: when a trace ends early, append a "Wait" step that re-checks the last step with some probability of catching a slip. Plot accuracy against thinking tokens, and compare with spending the same tokens on parallel samples.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
