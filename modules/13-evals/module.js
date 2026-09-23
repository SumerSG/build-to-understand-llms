export default {
  id: '13-evals',
  title: 'An eval harness',
  track: 'posttraining',
  minutes: 75,
  threshold: 'An eval is a measurement instrument: its grader, its sample size and its contamination decide what the number means far more than the number itself does.',
  goal: 'An eval runner with exact-match, regex and judge graders, the unbiased pass@k estimator and bootstrap confidence intervals, producing a report table with real error bars.',
  prereqs: ['09-data-pipeline', '10-sft', '11-preference', '12-rlvr'],
  recall: [
    { q: 'In module 12 (GRPO), where did the reward for a rollout come from?', options: ['A learned reward model', 'A verifier that checked the final answer against a known reference', 'The log-probability of the rollout'], answer: 1,
      why: 'RL with verifiable rewards uses exactly the graders you are about to build. A sloppy grader in module 12 is a sloppy reward signal, not just a sloppy report.' },
    { q: 'In module 12 you sampled a group of G rollouts per prompt rather than one. Why?', options: ['To save memory', 'To estimate the mean and spread of the reward for that prompt from repeated samples', 'Because the model is deterministic'], answer: 1,
      why: 'Same idea here: n samples per task let you estimate pass@k instead of one lucky or unlucky draw.' },
    { q: 'In module 11, the Bradley–Terry model turned pairwise preferences into a scalar reward. What does that tell you about a pairwise LLM judge?', options: ['Its verdicts are absolute scores', 'Its verdicts are comparisons, so a ranking is only as good as the comparisons are consistent', 'It cannot be used for evaluation'], answer: 1,
      why: 'Chatbot Arena-style Elo is Bradley–Terry over judge or human votes; if the comparisons flip when you swap the order, the ranking is measuring the harness, not the models.' },
    { q: 'In module 10 (SFT), what exactly did the chat template control?', options: ['The learning rate', 'Which tokens the model saw as context and which ones the loss was computed on', 'The vocabulary size'], answer: 1,
      why: 'Evals feed prompts through the same template. Changing the template changes the score, which is why prompt sensitivity is a real measurement problem.' },
    { q: 'In module 09 you deduplicated the pre-training corpus with n-gram overlap. Which idea reappears in this module?', options: ['Sharding', 'Near-duplicate detection by shared n-grams — here between eval prompts and the training corpus', 'Token packing'], answer: 1,
      why: 'Contamination detection is deduplication pointed at your test set instead of your training set.' },
  ],
  review: [
    { q: 'Why is pass@k estimated as `1 - C(n-c, k) / C(n, k)` rather than `1 - (1 - c/n)^k`?', options: ['It is faster to compute', 'The first is unbiased for k samples drawn without replacement from the n you took; the second plugs in an estimate and is biased', 'They are the same number'], answer: 1,
      why: 'Chen et al. 2021 (the HumanEval paper) introduced the combinatorial form exactly because the plug-in estimator is biased. With n=10, c=3, k=5 the two differ by 0.085 (0.917 vs 0.832).' },
    { q: 'You compute `C(1000, 500)` from factorials in float64. What happens?', options: ['It is exact', 'The intermediate products overflow to Infinity, and Infinity/Infinity is NaN', 'It rounds to zero'], answer: 1,
      why: 'C(1000, 500) itself is about 2.7e299, inside float64\'s range of about 1.8e308, but 1000! is about 4e2567, so the factorial route hits Infinity long before the division. Summing `log(n-i) - log(i+1)` and exponentiating the difference stays finite.' },
    { q: 'You run a 200-task eval and get 71%. Your colleague runs the same eval on the same model and gets 68%. The most likely explanation is…', options: ['One of you has a bug', 'Sampling noise: the 95% interval for 200 tasks at p=0.7 is roughly plus or minus 6 points', 'The model changed'], answer: 1,
      why: 'The standard error is sqrt(0.7*0.3/200) = 0.032, so a 3-point gap is well inside one interval. This is Miller 2024, "Adding error bars to evals": report the interval or the comparison is meaningless.' },
    { q: 'A judge grades answer A before answer B, then you swap them and it flips its verdict on 30% of pairs. What does that measure?', options: ['That the answers are equally good', 'Position bias: the judge is partly scoring the slot, not the content', 'That the judge is well calibrated'], answer: 1,
      why: 'Zheng et al. 2023 (MT-Bench) measured this and made two-order judging standard. Your `positionBias` reports consistency, the fraction of pairs whose two verdicts mirror each other.' },
    { q: 'A benchmark prompt shares a verbatim 13-gram with the training corpus. What is the right conclusion?', options: ['The model is definitely cheating', 'The task is flagged: its score no longer measures generalisation, so report contaminated and clean subsets separately', 'Lower n to 5 and re-check'], answer: 1,
      why: 'Overlap is evidence about the measurement, not proof about the model. Sainz et al. 2023 argue for reporting contamination alongside the score; GPT-3, GPT-4 and Llama 3 all published overlap analyses, each with a different overlap rule.' },
  ],
  concept: `
## The number is not the measurement

By module 12 you can watch a reward go up; this module decides whether that number means anything. An eval has four parts, and only one is the model:

1. **Tasks** — the questions, and how their prompts are formatted.
2. **A grader** — a function from a model response to a score.
3. **A sampling budget** — how many responses per task, at what temperature.
4. **A report** — the statistic and its uncertainty.

Get the other three wrong and the score moves more than most model upgrades move it.

Four benchmarks, measuring different things. **MMLU** (Hendrycks et al. 2020): 57 subjects of multiple-choice questions, scored by letter match, so it measures knowledge plus format compliance. **GSM8K** (Cobbe et al. 2021): 8.5k grade-school word problems whose reference answer follows a \`####\` marker, scored by extracting the final number — so it measures arithmetic *and* your extraction regex. **HumanEval** (Chen et al. 2021): 164 Python functions scored by running unit tests, an execution grader, the least ambiguous kind. **SWE-bench** (Jimenez et al. 2023): produce a patch that makes a real repository's tests pass, which measures the whole agent harness.

## The grader is half the eval

A model that answers \`1,188.\` when the reference is \`1188\` is right. Strict string comparison calls it wrong, and accuracy drops. So graders normalise (lowercase, trim, collapse whitespace, drop a trailing period, strip thousands separators, drop trailing decimal zeros) and then *extract*: the text after the last \`####\`, or failing that the last number.

Each rule is a judgement call that changes the score, so the harness is code, versioned with the model: "GSM8K 92.3" belongs to a specific harness commit.

:::predict
You evaluate the same model with two graders: strict exact match, and exact match after extracting the last number. On 100 chain-of-thought responses, roughly how far apart will the two scores be?
---
Usually 10 to 40 points: chain-of-thought responses rarely end with a bare number, so strict match scores near zero. Two reasonable graders routinely differ by more than two model generations do.
:::

## pass@k, and why the obvious formula is wrong

You sample \`n\` responses for a task and \`c\` are correct. How often do \`k\` samples contain a correct one? The tempting answer, \`1 - (1 - c/n)^k\`, is biased: it treats the estimated rate \`c/n\` as the truth. Chen et al. 2021 use the unbiased combinatorial form \`passAtK(n, c, k) = 1 - C(n-c, k) / C(n, k)\` — the fraction of the \`C(n, k)\` subsets of your own samples that contain a success. Compute it in log space: \`C(1000, 500)\` is about \`2.7e299\`, inside float64's range, but the factorial route is not — \`1000!\` is about \`4e2567\`, so the intermediate products hit \`Infinity\` and \`Infinity / Infinity\` is \`NaN\`.

Its twin is \`pass^k = C(c, k) / C(n, k)\`: the chance that *all* \`k\` draws are correct. pass@k fits when you can verify and retry (code with tests); pass^k fits when one failure is fatal (an agent taking an irreversible action). A 70%-per-attempt model has pass@10 of about 1.0 and pass^10 of about 0.03.

:::predict
A model solves each task with probability 0.5. You report pass@1 = 0.50. What does pass@10 look like, and does it tell you the model got better?
---
About 0.999. The model did not change; the measurement did: "at least one of ten attempts". Reporting pass@k without k, n and the temperature is meaningless — and pass@k rises with temperature even as pass@1 falls.
:::

## Error bars, judges, contamination

Two runs of a 200-task eval differ by a few points from sampling alone. The bootstrap (Efron 1979) gives the interval without any distributional assumption: resample the per-task scores with replacement \`B\` times, take each resample's mean, and read off the 2.5th and 97.5th percentiles. The width shrinks like \`1/sqrt(tasks)\`, so quadrupling the task count halves it. Miller 2024, "Adding error bars to evals", argues this should be standard.

When no exact answer exists, labs use an **LLM judge**. Judges are cheap and biased: toward longer answers (verbosity), toward whichever answer came first (position bias — Zheng et al. 2023 found flips on a large fraction of pairs), and toward their own outputs (self-preference). You will measure the judge, not fix it: agreement with exact match, and swapping the two answers.

Finally, **contamination**: if an eval prompt appears verbatim in the training corpus, the score measures memorisation. The standard check is n-gram overlap, but n is a choice: GPT-3 (Brown et al. 2020) used 13-grams, Llama 3 used 8-grams plus a per-dataset threshold on the fraction of contaminated tokens, and GPT-4 sampled 50-character substrings. Your default \`n = 13\` is GPT-3's choice, not a constant of nature. Sainz et al. 2023 argue for publishing the overlap rate next to the score.

## Where this toy differs from production

Your graders run on strings, in one process, on 40 procedurally generated word problems, against a scripted "model" that is a seeded random function rather than a network. A real harness — lm-evaluation-harness, HELM, OpenAI's evals, SWE-bench's Docker runner — sandboxes untrusted model-written code, retries API failures, caches responses by prompt hash, records template, model version, temperature and harness commit with every run, and shards thousands of tasks across machines. The statistics are identical; the plumbing is an order of magnitude larger.
`,
  steps: [
    {
      id: 'graders',
      title: 'Graders as pure functions',
      instructions: `
A **grader** is \`(answer, reference, question) → score in [0, 1]\`. Pure: no state, no randomness, no I/O. Four functions.

\`normalizeAnswer(s)\`: the canonical form of an answer string. Lowercase, trim and collapse whitespace runs (done for you), then drop **one** trailing period, remove thousands separators, and drop trailing zeros after a decimal point so \`42.0 → 42\` and \`3.50 → 3.5\` — but \`10.05\` must stay \`10.05\`.

A thousands separator is a comma that sits after a digit and is followed by exactly three digits and then a word boundary (end of string, space, punctuation — not another digit or letter). So \`1,188 → 1188\` and \`1,234,567 → 1234567\`, but \`12,3456\` stays \`12,3456\`. Remove only the comma; keep the digits on both sides.

\`extractFinal(text)\`: the part of a response that is the final answer. Everything after the **last** \`####\` if present (the GSM8K convention, done for you); otherwise the last number in the text; otherwise the trimmed text. \`NUMBER_RE\` at the top of the file matches integers, decimals and \`1,188\`.

\`exactMatch(answer, reference)\`: \`1\` if the two normalise to the same string, else \`0\`. Return numbers, not booleans — graders are averaged.

\`regexMatch(answer, pattern)\`: \`1\` if \`pattern\` matches anywhere in the answer. \`pattern\` may already be a \`RegExp\`; a string is compiled case-insensitively. A grader must give the same answer on every call, and a \`RegExp\` with the \`g\` or \`y\` flag is stateful: \`.test()\` advances its \`lastIndex\`, so the same input alternates \`1, 0, 1\`. Rebuild such a pattern from \`pattern.source\` and \`pattern.flags\` with \`g\` and \`y\` removed before testing. Do not mutate the caller's \`RegExp\`.

These four decide the score more than the model does, so they are step 1.
`,
      predict: { question: 'Before you write it: what does `normalizeAnswer("12,3456")` have to return, and why is that not the same rule as "remove all commas"?', answer: '`12,3456` unchanged. A thousands separator is a comma after a digit, followed by exactly three digits and then a word boundary (not another digit or letter). `12,3456` is not a formatted number, so blindly stripping commas would silently turn two different strings into the same one.' },
      hints: [
        'Each rule is one `String.prototype.replace`, and the order you apply them in matters. Which rule has to run before you can tell whether `42.0.` has trailing decimal zeros? For `regexMatch`, what state does a `RegExp` object carry between calls?',
        'Strip the trailing period first, with the literal `/\\.$/`. For `regexMatch`, rebuild a `RegExp` pattern from its `source` and its `flags` minus `g` and `y`. Thousands separators: replace a comma that is followed by exactly three digits and then a word boundary. Lookahead lets you match the comma without consuming the digits. Trailing decimal zeros: match `digits . digits` and rewrite the fractional part, keeping it only if something non-zero remains.',
        'Separators: `t = t.replace(/(\\d),(?=\\d{3}\\b)/g, "$1")`. Trailing zeros: a pattern with groups for the integer part, an optional fractional part that ends in a non-zero digit, and a run of zeros; the replacer returns `i` alone when that optional group is empty. For `extractFinal`, `s.match(NUMBER_RE)` gives every number in order — which one do you want?',
      ],
    },
    {
      id: 'passk',
      title: 'The unbiased pass@k estimator',
      instructions: `
Three functions, all combinatorial, all computed in log space.

\`logChoose(n, k)\`: \`log C(n, k)\`, as a sum of \`log(n - i) - log(i + 1)\` for \`i\` in \`0..k-1\`. Return \`-Infinity\` when \`k < 0\` or \`k > n\`. Never compute \`C(n, k)\` itself: \`C(1000, 500)\` is about \`2.7e299\` and the intermediate products overflow to \`Infinity\`.

\`passAtK(n, c, k)\`: \`1 - C(n - c, k) / C(n, k)\` (Chen et al. 2021), computed as \`1 - exp(logChoose(n - c, k) - logChoose(n, k))\`. Throw an \`Error\` if \`k\` is outside \`1..n\` or \`c\` outside \`0..n\`. Return exactly \`1\` when \`n - c < k\`, since then every \`k\`-subset must contain a success.

\`passPowK(n, c, k)\`: \`C(c, k) / C(n, k)\` — all \`k\` draws correct. Same validation; \`0\` when \`c < k\`.

The tests check you against brute-force enumeration of every \`k\`-subset, and against \`n = 1000\`, where a naive implementation returns \`NaN\`.
`,
      predict: { question: 'With n = 10, c = 3, k = 5, the unbiased estimator gives 0.917 and the plug-in formula `1 - (1 - c/n)^k` gives 0.832. Which is larger, and why does the gap shrink as n grows?', answer: 'The unbiased one is larger here. Drawing without replacement from your own 10 samples guarantees you see 5 distinct draws, which finds a success more often than 5 independent coin flips at rate 0.3. As n grows, sampling without replacement from a large pool approaches independent sampling, so the two converge.' },
      hints: [
        'The ratio `C(n-c, k) / C(n, k)` is the probability that a random k-subset is made entirely of failures. Subtract it from 1.',
        'Compute the logarithm of each binomial with a loop that adds one term per draw, then exponentiate the difference of two such logs instead of dividing two huge numbers. Handle the impossible cases (`n - c < k`, `c < k`) before touching logs, so `-Infinity - -Infinity` never becomes `NaN`.',
        'Sketch of `logChoose`: guard `k < 0 || k > n` and return `-Infinity`; otherwise start `s = 0` and, for `i` from `0` to `k - 1`, add one log term that compares the shrinking pool of remaining items with the draw count — `C(n, k) = (n/1)·((n-1)/2)·…`. Then `passAtK` is validation, an early `return 1` when `n - c < k`, and one line of the form `1 - Math.exp(/* log of the all-failure count */ - /* log of the total count */)`. `passPowK` has the same shape with the success count on top.',
      ],
    },
    {
      id: 'bootstrap',
      title: 'Bootstrap confidence intervals',
      instructions: `
\`bootstrapCI(scores, { B = 1000, alpha = 0.05, next = rng(0) })\` where \`scores\` is one number per task.

Draw \`B\` resamples of size \`n = scores.length\` **with replacement** using \`randInt(next, n)\` for each index. Record each resample's mean. Then return:

- \`mean\`: the plain mean of \`scores\` (not of the resamples),
- \`lo\`, \`hi\`: the sorted resample means at positions \`floor((alpha/2) * (B - 1))\` and \`ceil((1 - alpha/2) * (B - 1))\`,
- \`se\`: the standard deviation of the \`B\` resample means, as a population standard deviation (divide the summed squared deviations by \`B\`, not \`B - 1\`).

That is the percentile bootstrap (Efron 1979). It assumes nothing about the distribution of the scores, which matters because per-task pass@k values are not normal — they pile up at 0 and 1.

All randomness goes through the supplied \`next\`, so the same seed gives the same interval. The tests check that \`se\` is near \`sqrt(p(1-p)/n)\` and that 16× more tasks shrinks the interval about 4×.
`,
      hints: [
        'One outer loop over `b` in `0..B-1`, one inner loop over `i` in `0..n-1` that accumulates `scores[randInt(next, n)]`. Divide by n once per resample.',
        'Store the B means in a `Float64Array`, compute `se` from them before sorting (or after — sorting does not change the variance), then sort ascending and index the percentiles. `Float64Array.prototype.sort` is numeric by default, unlike `Array.prototype.sort`.',
        'Shape: `const means = new Float64Array(B);` then the two loops fill `means[b]` with one resample mean each. Next, `se` is `Math.sqrt` of the average squared distance of `means[b]` from their own mean. Finally sort and pick two entries: both indices scale a tail probability by `B - 1` (not `B`, or the top index runs off the end), rounding the lower one down and the upper one up.',
      ],
    },
    {
      id: 'judge',
      title: 'An LLM judge, and measuring its biases',
      instructions: `
The file already contains \`scriptedJudge(question, answer, reference) → string\` and \`scriptedPairwiseJudge(question, a, b, reference) → 'A' | 'B'\`. They stand in for a model call. Read them: both have deliberate flaws (the pointwise one is impressed by any answer of 15+ words containing a number; the pairwise one breaks ties by picking A). You will not fix them. You will measure them.

\`judgeGrader(judge)\`: return a grader \`(answer, reference, question) → 0 | 1\` that calls \`judge(question, answer, reference)\` — note the argument order flip — and scores \`1\` iff the verdict text contains the **whole word** \`correct\`, any case. \`INCORRECT\` contains the letters \`correct\`; a substring test scores it 1 and silently inverts your eval.

\`agreementRate(records, graderA, graderB)\`: the fraction of \`records\` (each \`{ question, answer, reference }\`) on which the two graders give the same verdict, treating \`score >= 0.5\` as "correct". With no records there is nothing to disagree on: return \`1\`, the same convention \`positionBias\` uses for \`consistency\`.

\`positionBias(pairwise, pairs)\`: judge every pair \`{ question, a, b, reference }\` twice, once as \`(a, b)\` and once as \`(b, a)\`. Return \`{ consistency, firstWinRate }\`: \`consistency\` is the fraction of pairs whose two verdicts mirror each other (\`A\` then \`B\`, \`B\` then \`A\`, or \`tie\` both times), and \`firstWinRate\` is the fraction of the calls that picked a side which picked \`A\` — ties are excluded from both the numerator and the denominator, so \`0.5\` always means "no preference for the first slot"; return \`0.5\` when no call picked a side. With no pairs, \`consistency\` is \`1\`. A fair judge scores \`consistency = 1\` and \`firstWinRate = 0.5\`.
`,
      hints: [
        'For the whole-word test, a regex with word boundaries does it in one line; think about what `\\b` matches around the letters of `incorrect`.',
        '`/\\bcorrect\\b/i.test(verdict)` — in `incorrect` there is no word boundary before the `c`. `agreementRate` is one loop comparing two booleans. `positionBias` calls the judge twice per pair and counts two different things in the same loop.',
        'Per pair, `v1 = pairwise(q, a, b, ref)` and `v2 = pairwise(q, b, a, ref)`. A pair is consistent in exactly three mirrored cases: A-then-B, B-then-A, and tie-then-tie. Then, for each of the two verdicts, skip ties and bump a `decided` counter, bumping a `firstWins` counter too when the verdict is `"A"`. Guard both divisions for empty inputs.',
      ],
    },
    {
      id: 'runner',
      title: 'Contamination and the runner',
      instructions: `
\`ngrams(text, n)\`: the \`Set\` of all consecutive \`n\`-word sequences in \`text\`, using \`wordTokens\` (already written: lowercase, punctuation dropped) and joining with single spaces. A 5-word text has 3 trigrams; text shorter than \`n\` has none.

\`contamination(tasks, corpus, n = 13)\`: which tasks share at least one \`n\`-gram with \`corpus\`? \`corpus\` is a single string (the training text, possibly many documents joined by spaces); build its n-gram set with \`ngrams(corpus, n)\` and compare each task's \`ngrams(task.question, n)\` against it. Return \`{ rate, flagged, n }\`, where \`flagged\` is the array of task ids in task order and \`rate\` is \`flagged.length / tasks.length\` (\`0\` for no tasks). Build the corpus n-gram set **once**, outside the task loop — the whole point of the \`Set\` is that each lookup is O(1).

\`runEval({ tasks, model, grader, n = 10, ks = [1, 5, 10], next, B = 1000 })\`: the runner. For each task, sample \`n\` answers from \`model(task.question, next)\`, grade each with \`grader(answer, task.answer, task.question)\`, and count \`c\`, the number scoring \`>= 0.5\`. Throw before doing any work if any \`k > n\`. Return:

\`\`\`
{ nTasks, n,
  results: [{ id, question, reference, n, c, answers }],
  metrics: [{ k, value, lo, hi, se, passPowK }] }
\`\`\`

For each \`k\`: the per-task values are \`passAtK(n, c, k)\`; \`value\`, \`lo\`, \`hi\`, \`se\` come from \`bootstrapCI\` over those per-task values, and \`passPowK\` is their mean under \`passPowK(n, c, k)\`. Keep the raw \`answers\` — the demo re-grades them with a different grader without re-running the model, which is exactly what a cached real harness does.
`,
      hints: [
        'Validate `ks` against `n` first, then two nested loops (tasks, then samples), then a `map` over `ks`. `formatReport` at the top of the file documents the exact metric field names.',
        '`bootstrapCI(perTask, { B, next })` returns `{ mean, lo, hi, se }` — `value` is its `mean`, so pass@1 automatically comes out as the mean over tasks of `c/n`. Reuse the same `next` for sampling and for bootstrapping; one seed, one reproducible run.',
        'For each `k`, build three things: the array `per` of per-task `passAtK(r.n, r.c, k)` values; `ci`, the result of bootstrapping `per` with the shared `next`; and the mean over tasks of `passPowK(r.n, r.c, k)` (`meanArray` is imported). The metric object copies `k`, then `ci.mean` into `value`, then `lo`, `hi`, `se`, then that mean into `passPowK`.',
      ],
    },
  ],
  reflection: [
    'Your demo grades the same sampled answers with three graders and gets three different accuracies. Explain to a colleague, without using the word "bug", why all three numbers are honest and what you would have to publish alongside a score for it to be reproducible.',
    'You are asked to decide whether model B is better than model A. B scores 2 points higher on a 100-task eval. Write out what you would check before answering, in the order you would check it, and say what each check rules out.',
    'pass@k and pass^k are computed from the same n and c. Describe one product where you would report pass@10 and one where reporting it would be dishonest, and say what makes the difference.',
  ],
  stretch: [
    'Add an execution grader: tasks carry a snippet of JavaScript and a list of assertions, and the grader runs the model\'s code against them. This is how HumanEval and SWE-bench score — and it is why they need a sandbox that your `new Function` version does not have.',
    'Implement the paired bootstrap for a model comparison: resample tasks once and recompute *both* models\' means on the same resample, then report the interval for the difference. Because task difficulty cancels, the paired interval is far narrower than the difference of two independent intervals; this is the comparison lm-evaluation-harness and Miller 2024 recommend.',
    'Add prompt-sensitivity measurement: run the same tasks through three prompt templates (bare question, few-shot, chat template from module 10) and report the spread. Published MMLU numbers move by several points on template alone, which is why HELM fixes the template as part of the benchmark.',
    'Replace the scripted judge with a real one: the small GPT you trained in module 07, prompted to answer CORRECT or INCORRECT, and re-measure agreement and position bias. Compare the drop against the agreement numbers Zheng et al. 2023 report for GPT-4 as a judge against human votes on MT-Bench.',
    'Measure over-refusal: build 20 benign prompts that only look risky ("how do I kill a Python process") and 20 genuinely harmful ones, grade each response as refusal or compliance with a scripted judge, and report the refusal rate on the benign set and the compliance rate on the harmful set (the two sides XSTest, Röttger et al. 2023, measures), each with your bootstrap CI. With 20 prompts per set, check how wide those intervals are before you claim one model refuses less than another.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
