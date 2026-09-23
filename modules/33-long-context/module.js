export default {
  id: '33-long-context',
  title: 'Long-context evaluation',
  track: 'inference',
  minutes: 90,
  threshold: 'A model\'s advertised context window and its effective context are different numbers; only a controlled retrieval test at many lengths and depths tells you where a model stops being able to use what you gave it.',
  goal: 'A needle-in-a-haystack and retrieval eval suite over lengths and depths, an effective-context estimator, and the compute and memory price of long prompts: the demo draws an accuracy heatmap and depth curves for a model that advertises 8,192 words, reports how much of that it really uses, and prices 4k–128k prompts at Llama-3-8B dims.',
  prereqs: ['13-evals', '15-kv-cache', '17-prefix-caching', '29-attention-variants'],
  recall: [
    { q: 'In module 13, why did every eval number come with a bootstrap interval?', options: ['To make the table look precise', 'Because a score from n items is a sample: with 20 items at 70% the 95% interval is roughly plus or minus 20 points', 'Because the grader is random'], answer: 1,
      why: 'This module runs 20 trials per (length, depth) cell. A cell at 0.7 and one at 0.8 are not distinguishable at that n, which is why you estimate effective context from row means and report the intervals.' },
    { q: 'Module 15 sized the Llama-3-8B KV cache (32 layers, 8 KV heads, head dim 128, bf16) at approximately…', options: ['8 KB per token', '128 KiB per token', '2 MB per token'], answer: 1,
      why: '2 · 32 · 8 · 128 · 2 bytes = 131,072. Your step 5 multiplies it by the prompt length: 16 GiB for one 128k-token sequence.' },
    { q: 'In module 29, what does Position Interpolation do to extend a RoPE model from 4k to 16k tokens?', options: ['Adds new learned position rows', 'Divides every rotary frequency by the stretch factor 4, so no angle is new but neighbours are harder to tell apart', 'Removes positional information'], answer: 1,
      why: 'Context extension makes long positions representable. Whether the model can then retrieve from them is a separate question, and it is exactly what this module measures.' },
    { q: 'In module 17, what does a prefix-cache hit save?', options: ['Tokenizing the prompt', 'The prefill compute for the shared prefix, whose keys and values are reused', 'Decode time for every output token'], answer: 1,
      why: 'A long document that stays fixed across questions is the best case for prefix caching: step 5 shows prefill of a 128k prompt costs petaFLOPs, and a cache hit skips it.' },
    { q: 'What does lib/sampling.js\'s `generate` do with a prompt longer than the model\'s blockSize?', options: ['Throws an error', 'Keeps only the newest blockSize tokens, silently dropping the start', 'Compresses it'], answer: 1,
      why: 'Left truncation is why the lab checkpoint scores zero on a needle placed before its 64-token window, and why the synthetic model in this module cuts from the left beyond its window.' },
  ],
  review: [
    { q: 'A model card says "128k context". What does that number guarantee?', options: ['That retrieval works at any depth up to 128k', 'Only that 128k tokens fit through the forward pass; how much the model can use is an empirical question', 'That perplexity is flat up to 128k'], answer: 1,
      why: 'RULER (Hsieh et al. 2024) found that only about half of the models it tested that claim 32k or more stay above its threshold at 32k. The window is an upper bound; the effective context is a measurement.' },
    { q: 'Accuracy by length is 0.95, 0.90, 0.60, 0.85 at 1k, 2k, 4k, 8k. With threshold 0.8, the effective context is…', options: ['8k', '2k', '4k'], answer: 1,
      why: 'Effective context is the longest length before the first failure. Claiming 8k because one longer cell recovered would promise retrieval at 4k that the model does not deliver.' },
    { q: 'Why do RULER\'s multi-key and multi-value variants give a shorter effective context than the single needle?', options: ['They use longer haystacks', 'Distractors test whether the model finds the right key, and multiple values test whether it finds all of them; a lucky or shallow match no longer passes', 'They use a stricter tokenizer'], answer: 1,
      why: 'In the reference run of the demo, multikey\'s effective context was half the single needle\'s. A single needle with an unusual sentence is the easiest possible retrieval, which is why vanilla needle tests saturate near 100% for many recent models.' },
    { q: 'Why is perplexity on long documents a weak test of long-context ability?', options: ['It cannot be computed beyond 4k', 'Most next tokens are predictable from nearby text, so perplexity can stay flat while the model ignores everything far away', 'It is always higher for long documents'], answer: 1,
      why: 'A model that only uses the last 2k tokens still predicts most tokens of a book well. Retrieval, aggregation and reasoning tasks force the answer to depend on a specific distant token.' },
    { q: 'Doubling a Llama-3-8B prompt from 64k to 128k tokens multiplies prefill compute by roughly…', options: ['2×', '3×', '4×'], answer: 1,
      why: 'Prefill is `2·N·T + 2·L·C·T·(T+1)`: the weight term doubles, the attention term quadruples. At 64k attention is about half the work, so the total grows about 3× (your step-5 test checks 2.5–4).' },
  ],
  concept: `
## Two numbers that get confused

A model's **context window** is how many tokens fit through its forward pass: the size of its position encoding and of the KV cache you can afford. Its **effective context** is the longest input from which it still reliably *uses* information. The first is a design choice written on the model card. The second you have to measure, and it is usually smaller.

This module builds the instrument. You plant a fact at a controlled position in filler text, ask for it back, and repeat over a grid of lengths and depths.

## Needle in a haystack

Greg Kamradt's 2023 "needle in a haystack" test put one out-of-place sentence (in the original, a remark about the best thing to do in San Francisco) at a chosen **depth** (0% = the start, 100% = just before the question) inside Paul Graham essays of a chosen **length**, then asked for it. The result is a heatmap: rows are lengths, columns are depths, colour is accuracy. You will build every part: the haystack generator (step 1), the graders (step 2), and the grid runner with bootstrap intervals from module 13 (step 3).

Two details decide whether the test measures anything. The filler must not contain the answer by accident, or the model can pass without retrieving. And the needle must really sit at the requested depth, measured in tokens rather than sentences.

:::predict
A model advertises a 128k window. On a single needle it scores 100% at every length and depth. Would you conclude its effective context is 128k?
---
No. A single, distinctive needle is the easiest retrieval there is: the needle does not look like the filler, so almost any attention to it succeeds. RULER (Hsieh et al. 2024) added harder variants: several needles with **different keys** (retrieve the right one), several **values under one key** (retrieve all), variable tracking and aggregation. Models that were near-perfect on the vanilla needle lost a lot of accuracy on these, and RULER reports that only about half of the models it tested that claim 32k or more stay above its threshold (Llama-2-7B's score at 4k, 85.6%) at 32k.
:::

## Lost in the middle, attention sinks and position scaling

Liu et al. (2023), "Lost in the Middle", moved the relevant document through a list of 20 retrieved documents and found a **U-shaped** curve: accuracy is highest when the evidence is at the start or the end and lowest in the middle. For GPT-3.5-Turbo the middle was, in some settings, below its closed-book score. Two mechanisms are usually cited. Recency: causal attention to nearby tokens is easy to learn, because most next tokens depend on them. Primacy: the first tokens receive disproportionate attention even when they carry no content. Xiao et al. (2023) named these **attention sinks** and showed in StreamingLLM that keeping the first 4 tokens in the cache is enough to keep a sliding-window model stable.

Module 29's RoPE scaling (Position Interpolation, NTK-aware, YaRN) makes long positions *representable*; it does not make them *used*. Llama 3.1, Qwen and DeepSeek-V3 all combine scaling with long-context training data. Your eval is how you tell whether that worked.

:::predict
Your synthetic model has a window of 8,192 words. Before you run the demo: at 4,096 words, which needle depth do you expect to score worst, and which best?
---
Best at 100% depth (the needle sits right before the question). Worst around 25%: far from the question, but past the few hundred words that the sink term keeps visible. 0% recovers somewhat because of the sink. The curve is U-shaped but lopsided towards recency, which is also what Liu et al. measured.
:::

## Why not perplexity?

Perplexity on a long book barely changes if the model ignores everything more than 2k tokens back, because most next tokens are predictable from nearby text. Long-context claims need tasks whose answer depends on a specific distant token (**retrieval**), on many of them (**aggregation**), or on combining them (**reasoning**). LongBench (Bai et al. 2023) and InfiniteBench (Zhang et al. 2024) collect realistic tasks of this kind (QA over long documents, summarisation, code), averaging several thousand words and over 100k tokens respectively; synthetic tests like yours are cheaper and let you control length and depth exactly.

## What a long prompt costs

Every token of a long prompt is prefilled (module 23 calls this the compute-bound phase), and every token's keys and values sit in the KV cache (module 15). Summing module 15's per-token cost over positions 1 to T gives prefill FLOPs \`2·N·T + 2·L·C·T·(T+1)\`, where \`N\` is the parameter count, \`L\` the number of layers and \`C\` the model width. The first term is linear in T, the second quadratic, so attention becomes most of the work at long lengths. For Llama-3-8B, 128k tokens is approximately 6.6 PFLOPs, about 13 s on one H100 at 50% of the roughly 989 TFLOP/s dense bf16 in NVIDIA's datasheet, and 16 GiB of cache. Prefix caching (module 17) is how providers avoid paying that twice for the same document.

## Where the toy differs from production

The lab checkpoint has a **64-token** window, and its tokenizer spends most of that on one needle sentence, so it cannot show any of these curves; the demo probes it only to show the window's hard edge. The curves come from a **synthetic model** whose recall is a formula with a distance fade, a sink term and key interference. It makes the eval's mechanics testable; it is not a claim about any real model. Real evals count tokenizer tokens, not words (Llama 3's tokenizer produces roughly 1.3 tokens per English word), use realistic filler, generate answers with the actual model, and run thousands of items per length.
`,
  steps: [
    {
      id: 'haystack',
      title: 'The haystack generator',
      instructions: `
Implement \`buildHaystack({ length, needles, depths, next, filler = FILLER })\`. Each needle is \`{ key, value, sentence }\` (see the worked \`makeNeedle\`), and \`depths[k]\` in \`[0, 1]\` says where needle \`k\` goes.

1. **Budget.** The needles count toward \`length\`, so the filler target is \`length − (words in all needle sentences)\`. The skeleton computes it for you.
2. **Filler.** Start at sentence \`randInt(next, filler.length)\` and walk forward, wrapping around with \`i % filler.length\`. Append sentences until the filler word count reaches the target (stop at the first sentence that makes it \`>=\` target).
3. **No accidental needles.** Skip any filler sentence that mentions a needle's key or value as a *whole word* (use the worked \`containsWord\`). If you skip \`filler.length\` sentences **in a row** you have gone a full lap without finding usable filler, so throw: there is no valid haystack. (Count consecutive skips, not total skips: a long haystack wraps around the filler many times and meets the same few bad sentences on every lap.)
4. **Placement.** Let \`F\` be the filler word count. Needle \`k\` goes at the sentence boundary whose preceding filler word count is nearest \`depths[k] · F\`, so depth 0 is before all filler and depth 1 after it. Place by words, not by sentence index: sentences vary from 5 to 10 words, and an index-based depth drifts by more than a sentence.

Return \`{ sentences, text: sentences.join(' '), positions, tokens }\`, where \`positions[k]\` is the index of needle \`k\` in \`sentences\` (in the order of \`needles\`, not of depth) and \`tokens\` is the word count of \`text\`.

The worked \`makeTask\` above your code builds the three RULER-style items (single needle, multi-key, multi-value) on top of your function.
`,
      predict: { question: 'The filler sentences average 7.5 words. If you placed the needle at sentence index round(depth · sentences) instead of by word count, how far off could a 1,500-word haystack be at depth 0.1?', answer: 'About two sentences. Sentences run from 5 to 10 words, and the ones that happen to open a haystack need not average 7.5, so the first 10% of sentences can hold noticeably fewer or more than 10% of the words: with the test\'s seed, index placement puts 135 filler words before the needle instead of about 150, and the test allows 10. Kamradt\'s original script also placed needles by token fraction, then backed up to the previous sentence end.' },
      hints: [
        'Three phases: collect filler (with a skip rule), compute where each needle goes, then build the final sentence list in one pass.',
        'With S filler sentences holding F words, keep an array `cum` of length S+1 where `cum[b]` is the filler word count before filler sentence b (`cum[0] = 0`, `cum[S] = F`). For each depth pick the b minimising `|cum[b] − depth · F|`. Then walk b = 0…S: first push any needle whose boundary is b (recording its index in `positions`), then push filler sentence b (if b < S).',
        'The collection loop: `while (tokens < target) { const s = filler[i % filler.length]; i++; if (/* s mentions a key or value */) { if (++skipped >= filler.length) throw …; continue; } skipped = 0; body.push(s); tokens += countTokens(s); }`',
      ],
    },
    {
      id: 'graders',
      title: 'Exact and fuzzy retrieval graders',
      instructions: `
Four graders, all returning numbers in \`[0, 1]\`:

- \`exactMatch(answer, reference)\`: \`1\` if \`normalizeAnswer\` (module 13's rules, given) makes them identical.
- \`fuzzyMatch(answer, reference)\`: \`1\` if the normalised reference appears as a **whole word** in the normalised answer. Models answer in sentences ("The secret code for Oslo is 4,821."), so exact match alone would score a correct retrieval as wrong. "Whole word" matters: \`48210\` is not \`4821\`.
- \`recallFraction(answer, references)\`: the share of references that fuzzy-match; \`0\` for an empty list.
- \`gradeTask(task, answer)\`: for \`task.kind === 'multivalue'\`, \`recallFraction(answer, task.answers)\`. Otherwise \`fuzzyMatch(answer, task.answers[0])\`, **but 0 if any of \`task.distractors\` also appears**. Without that rule a model that lists every number in the context passes the multi-key task without retrieving anything.
`,
      hints: [
        'Build everything on `normalizeAnswer` and the worked `containsWord`; none of these needs more than a few lines.',
        '`fuzzyMatch` is `containsWord(normalizeAnswer(answer), normalizeAnswer(reference))` turned into a number. `recallFraction` sums `fuzzyMatch` over the references and divides by their count.',
        'For gradeTask: `if (task.kind === "multivalue") return recallFraction(…); if (task.distractors.some(/* d appears in answer */)) return 0; return …;`',
      ],
    },
    {
      id: 'runner',
      title: 'The grid runner',
      instructions: `
Implement \`contextEval(model, { lengths, depths, trials = 10, kind = 'niah', seed = 0, B = 200, filler = FILLER })\`.

For every length (outer loop) and every depth (inner loop), run \`trials\` items: \`makeTask(kind, { length, depth, next, filler })\`, call \`model(task.context, task.question, next)\`, and score it with \`gradeTask\`. Then summarise the cell with the worked \`bootstrapCI(scores, { B, next: boot })\`.

Use two seeded generators: \`next = rng(seed)\` for items and model calls, \`boot = rng(seed + 7919)\` for the bootstrap, so changing \`B\` does not change which items you drew.

Return \`{ kind, lengths, depths, trials, acc, lo, hi }\`, where \`acc[i][j]\` is the mean score at \`lengths[i]\`, \`depths[j]\` and \`lo\`/\`hi\` are its 95% interval.
`,
      predict: { question: 'With 20 trials per cell, a cell reads 0.70. Roughly how wide is its 95% bootstrap interval?', answer: 'About plus or minus 0.2 (the standard error is sqrt(0.7 · 0.3 / 20) ≈ 0.10). Neighbouring cells that differ by 0.1 are within noise, so read the heatmap for its shape and compute effective context from row means, which pool all depths.' },
      hints: [
        'Three nested loops (length, depth, trial) and one bootstrap per cell. Build each row of `acc`, `lo` and `hi` as you go.',
        'Create `next` and `boot` once, before the loops, so the whole grid is one deterministic stream. The model gets the same `next`.',
        'Inside the depth loop: `const scores = []; for (…trials) { const task = makeTask(kind, { length, depth, next, filler }); scores.push(/* grade the model\'s answer */); } const ci = bootstrapCI(scores, { B, next: boot });` then push `ci.mean`, `ci.lo`, `ci.hi`.',
      ],
    },
    {
      id: 'effective',
      title: 'Effective context and the depth curve',
      instructions: `
Four small functions that turn a grid into claims:

- \`lengthCurve(result)\`: mean accuracy over depths, one number per length.
- \`effectiveContext(result, threshold = 0.8)\`: the longest length such that **it and every shorter length** have \`lengthCurve >= threshold\`; \`0\` if even the first fails. Stop at the first failure: a longer length that happens to recover is not usable context.
- \`depthCurve(result, length)\`: a copy of the accuracy row at \`length\`; throw if it was not evaluated.
- \`middleDrop(curve)\`: the mean of the first and last entries minus the mean of the interior entries; throw for fewer than 3 depths. Positive means lost in the middle.

The last test runs your whole pipeline on a small fading model with a 1,000-word window and checks that the effective context comes out at 100 to 400 words.
`,
      hints: [
        'effectiveContext is a scan from the shortest length that remembers the last passing length and breaks on the first failure.',
        '`lengthCurve` maps each row of `result.acc` to its mean. `depthCurve` finds the row with `result.lengths.indexOf(length)` and returns `.slice()`.',
        '`let best = 0; for (let i = 0; i < curve.length; i++) { if (/* fails */) break; best = result.lengths[i]; } return best;`',
      ],
    },
    {
      id: 'cost',
      title: 'The price of a long prompt',
      instructions: `
Cost functions for a config \`{ params, layers, dModel, nKvHeads, headDim }\` (\`LLAMA3_8B\` is given):

- \`prefillFlops(cfg, T)\`: module 15's per-token cost \`2·N + 4·L·t·C\` summed over positions \`t = 1…T\`, where \`N = params\`, \`L = layers\`, \`C = dModel\`. In closed form: \`2·N·T + 2·L·C·T·(T+1)\`.
- \`kvCacheBytes(cfg, T, bytesPerElement = 2)\`: \`2 · layers · nKvHeads · headDim · T · bytesPerElement\`. Use the KV heads, not the query heads: that is grouped-query attention's saving (module 29).
- \`longPromptCost(cfg, T, { peakFlops = 989e12, mfu = 0.5, bytesPerElement = 2 })\`: \`{ tokens: T, flops, attnShare, seconds, kvBytes }\` where \`attnShare\` is the attention part of \`flops\` divided by \`flops\` and \`seconds = flops / (peakFlops · mfu)\`. MFU (model FLOPs utilisation) is the fraction of peak you actually reach.
`,
      predict: { question: 'At what prompt length does attention become half of Llama-3-8B\'s prefill compute?', answer: 'Around 61k tokens. Attention is `2·L·C·T²` ≈ `262,144·T²` and the weights are `2·N·T` ≈ `1.6e10·T`; they are equal at T ≈ 61,000. Below a few thousand tokens attention is noise; at 128k it is about two thirds of the work.' },
      hints: [
        'The sum of t from 1 to T is T·(T+1)/2; the rest is bookkeeping.',
        'The attention part of prefill is everything except `2·N·T`. Compute `flops` once and derive `attnShare` and `seconds` from it.',
        '`const flops = prefillFlops(cfg, T); const attn = flops - 2 * cfg.params * T; return { tokens: T, flops, attnShare: …, seconds: …, kvBytes: kvCacheBytes(cfg, T, bytesPerElement) };`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why "this model has a 128k context window" and "this model can use 128k tokens" are different claims, using the numbers your demo produced for the three task kinds.',
    'Your multi-value task showed almost no middle drop, while the single needle showed a large one. What about the task design causes that, and what does it tell you about reading a single depth curve?',
    'A team wants to put a 100k-token manual in every request instead of building retrieval (module 21). Using your cost model and your effective-context measurement, argue both sides.',
  ],
  stretch: [
    'Add RULER\'s variable-tracking task (Hsieh et al. 2024): plant a chain `X1 = 4821`, `X2 = X1`, `X3 = X2` … at scattered depths and ask which variables equal 4821. It needs multi-hop retrieval, and effective context drops further.',
    'Wrap an OpenAI-compatible endpoint (vLLM or SGLang serving Llama-3.1-8B-Instruct) as `model(context, question)`, count lengths with its real tokenizer, and reproduce the heatmap at 4k to 128k tokens. Then enable prefix caching and measure how much of the cost from step 5 disappears for repeated haystacks.',
    'Make effective context conservative: bootstrap the row mean at each length (resampling all items across depths) and require the lower bound, not the point estimate, to clear the threshold. Compare against RULER\'s single-threshold rule.',
    'Give `makeFadingModel` a StreamingLLM-style cache (4 sink tokens plus a sliding window, Xiao et al. 2023) and show that the needle heatmap gets a hard edge at the window while the sink keeps depth 0 alive.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
