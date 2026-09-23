export default {
  id: '09-data-pipeline',
  title: 'The pre-training data pipeline',
  track: 'transformer',
  minutes: 90,
  threshold: 'Model quality is decided as much by what you remove from the corpus as by what you keep: filtering, deduplication and mixing are modelling decisions, and every one of them is a measurable count.',
  goal: 'A pipeline that filters, deduplicates, mixes and shards raw documents into token shards, with a report of what was kept and why: the demo runs it on 60-odd real and synthetic documents and charts what each stage removed.',
  prereqs: ['03-tokenizer', '07-pretraining', '08-scaling'],
  recall: [
    { q: 'In module 08, the compute-optimal (Chinchilla) rule says a model of N parameters should see roughly how many tokens?', options: ['About N', 'About 20 N', 'About 1000 N'], answer: 1,
      why: 'Hoffmann et al. 2022 fit about 20 tokens per parameter. A 70B model wants 1.4T tokens, which is why the size of the *cleaned* corpus, not the raw crawl, bounds the model you can train.' },
    { q: 'In module 07, a run shows train loss 1.5 and validation loss 3.1, both still falling. What is happening?', options: ['The learning rate is too low', 'The model is memorising training windows (overfitting)', 'Gradient clipping is too aggressive'], answer: 1,
      why: 'A growing train/val gap means the model predicts windows it has seen much better than fresh text. Duplicated documents cause exactly this: the model sees the same window many times per epoch without you noticing.' },
    { q: 'In module 03, what does `BPETokenizer.encode` do when the text contains the string `<|endoftext|>`?', options: ['Splits it into character tokens like any other text', 'Emits the single special id `tokenizer.eos`', 'Throws, because specials may not appear in text'], answer: 1,
      why: 'Specials are matched before pre-tokenisation and become one id each. This module writes `tokenizer.eos` between documents itself, so the boundary is always exactly one token.' },
    { q: 'In module 07, `getBatch` returns `y` as `x` shifted right by one token because…', options: ['The causal transformer predicts every position at once, so the target at t is the token at t+1', 'The first token has no embedding', 'AdamW requires targets of the same length'], answer: 0,
      why: 'One forward pass over a B×T window gives B·T predictions. A shard of packed tokens with `eos` separators is precisely the stream `getBatch` cuts windows from.' },
  ],
  review: [
    { q: 'Two documents have MinHash signatures of 32 values that agree in 24 positions. The estimated Jaccard similarity is…', options: ['24', '0.75', '0.25'], answer: 1,
      why: 'The estimate is the *fraction* of agreeing positions, 24/32. Each hash function agrees with probability exactly J, so the fraction is an unbiased estimate with standard deviation `sqrt(J(1-J)/k)`, about 0.08 for k = 32.' },
    { q: 'Exact deduplication hashes the *normalised* text (lowercased, punctuation removed). Which pair does it still keep as two documents?', options: ['"Hello, World!" and "hello world"', 'A paragraph and the same paragraph with one word changed', 'A paragraph and the same paragraph with tabs instead of spaces'], answer: 1,
      why: 'Normalisation absorbs case, punctuation and whitespace, but one changed word changes the hash entirely. Catching that is what MinHash near-deduplication is for.' },
    { q: 'You set mixing weights `{ web: 0.5, books: 0.5 }`. The mixer makes the two domains equal in…', options: ['Number of documents drawn', 'Tokens (size) contributed', 'Number of epochs'], answer: 1,
      why: 'Weights are shares of the token stream, because tokens are what the model trains on. Books documents are longer, so they are drawn less often for the same share.' },
    { q: 'A high-quality domain runs out of documents before the token budget is met. With `maxEpochs: 2`, the pipeline…', options: ['Stops the whole run', 'Repeats that domain up to twice, then gives its weight to the remaining domains', 'Repeats it indefinitely'], answer: 1,
      why: 'Muennighoff et al. 2023 found that repeating scarce data up to about 4 epochs costs little; beyond that the returns shrink fast. A cap makes the repeat count explicit instead of accidental.' },
    { q: 'What does the shard index entry `{ docId, shard, offset, length }` let a data loader do?', options: ['Skip the `eos` token', 'Find any document without decoding the stream, even one that crosses a shard boundary', 'Recompute the tokenizer'], answer: 1,
      why: '`readDoc` walks from `(shard, offset)` for `length` tokens, moving into the next shard when one ends. That is how contamination checks and per-source loss curves find a document again after sharding.' },
  ],
  concept: `
## The corpus is a model decision

A web crawl is not a dataset: most of Common Crawl is menus, boilerplate, spam, and the same page a hundred times over. Every pre-training corpus (C4, MassiveText, RefinedWeb, FineWeb, DCLM) is the output of a *pipeline* that decides, document by document, what the model will never see. DCLM (Li et al. 2024) held compute fixed, changed only the filtering, and moved downstream accuracy by several points.

Your pipeline has five stages, in production order:

1. **Quality filters**: rules that reject a document with a reason.
2. **Exact deduplication**: hash the normalised text; keep the first copy.
3. **Near-deduplication**: MinHash over word shingles with a Jaccard threshold.
4. **Mixing**: interleave domains by weight, capping how often a scarce domain repeats.
5. **Tokenise and shard**: cut the token stream into fixed-size shards with an index.

Each stage is a pure function returning \`{ kept, removed }\`, so the report is a table of counts. That is the threshold idea: **a filter is a hypothesis about what helps the model, and its effect is a number you can read off.**

## Quality heuristics (C4, Gopher)

C4 (Raffel et al. 2020) kept only lines ending in terminal punctuation and dropped pages containing "lorem ipsum" or a curly brace. Gopher (Rae et al. 2021) added document-level rules: 50 to 100,000 words, mean word length between 3 and 10 characters, a symbol-to-word ratio (hashes, ellipses) below 0.1, and no more than 30% duplicate lines. Your \`qualityReason\` uses smaller word counts because the lab's documents are short. RefinedWeb and FineWeb reuse most of them; DCLM adds a learned classifier.

:::predict
The default rules require at least 30% of lines to end in \`.\`, \`!\`, \`?\` or a closing quote. The demo feeds Shakespeare's sonnets, one verse line per text line. Do they survive?
---
No. Sonnet lines end in commas and semicolons: two or three of fourteen pass, so the rule removes all six sonnets (and keeps four of Blake's six "Tyger" stanzas). C4's line rule is known to delete verse, dialogue and code; whether that is a bug is a modelling decision the report makes visible.
:::

## Deduplication

Lee et al. 2021 ("Deduplicating Training Data Makes Language Models Better") found a 61-word sentence repeated over 60,000 times in C4, and that a model trained on deduplicated data emitted memorised text ten times less often. Exact duplicates are cheap: normalise, hash (\`hash32\` from \`lib/util.js\`, FNV-1a), keep the first document per hash.

Near-duplicates (the same article under a different header) need a similarity. The **Jaccard similarity** of two sets is \`|A ∩ B| / |A ∪ B|\`. Represent each document by its set of 5-word **shingles** (every window of five consecutive words), so documents that share text share shingles. Exact Jaccard for every pair is \`O(n²)\` set intersections. **MinHash** (Broder 1997) replaces each set by a signature: for each of \`k\` hash functions, the minimum hash value over the set. Two sets share a minimum under one hash function with probability exactly \`J\`, so the fraction of agreeing positions is an unbiased estimate of \`J\` with standard deviation \`sqrt(J(1-J)/k)\`.

:::predict
With \`k = 32\` hash functions and a true Jaccard similarity of 0.8, roughly how far off is the estimate typically?
---
\`sqrt(0.8 × 0.2 / 32) ≈ 0.07\`, so an 0.8 threshold sometimes keeps a pair at true similarity 0.85 and removes one at 0.75. Production uses more hashes (FineWeb: 112 in 14 buckets of 8) and accepts the remaining noise.
:::

Your \`nearDedup\` compares each document against every earlier survivor, still quadratic. At scale, **locality-sensitive hashing** splits each signature into \`b\` bands of \`r\` rows and compares only documents that share a band bucket; a pair at similarity \`J\` collides with probability \`1 − (1 − J^r)^b\` (the first stretch goal).

## Mixing and epochs

A clean corpus is still several datasets. The Pile (Gao et al. 2020) set weights by hand (Pile-CC 18%, PubMed Central 14%, Books3 12%, …); DoReMi (Xie et al. 2023) learns them with a small proxy model. Your \`mixDomains\` uses Megatron-LM's blending rule: at every draw, pick the domain whose running share of tokens is furthest below its target. Weights are shares of *tokens*, not documents.

Scarce, high-quality sources (Wikipedia, textbooks) run out before the budget does. Muennighoff et al. 2023 ("Scaling Data-Constrained Language Models") found that up to about 4 epochs of repeated data is nearly as good as fresh data; beyond that, returns fall off quickly. A \`maxEpochs\` cap makes the repeat count explicit; an exhausted domain's weight goes to the others.

## Tokenising and sharding

Tokenisation is independent per document, so it is embarrassingly parallel: production jobs spread documents over thousands of workers, each writing its own shards. Your \`packShards\` writes one \`eos\` after every document, cuts the stream into shards of exactly \`shardSize\` tokens, and writes an index \`{ docId, shard, offset, length }\`. The index makes **contamination checks** possible: finding the document that contains a leaked benchmark question.

## Where the toy differs from production

Your pipeline holds every document in memory, compares MinHash signatures pairwise instead of through LSH buckets, uses 32 hash functions, and skips language identification, URL blocklists, PII scrubbing and the learned classifiers of DCLM and FineWeb-Edu. Production shards are files of 100M or more tokens on object storage, addressed by \`(shard, offset)\`. The stage boundaries, reason strings and counts are the same.
`,
  steps: [
    {
      id: 'quality',
      title: 'Quality filters with a reason',
      instructions: `
Complete \`qualityReason(text, cfg)\`: return the name of the **first** rule the text violates, or \`null\` if it passes all of them. The two word-count rules are done. Add, in this order:

- \`word_length\`: mean characters per word (total characters of the words divided by the word count) below \`cfg.minMeanWordLength\` or above \`cfg.maxMeanWordLength\`.
- \`symbol_ratio\`: the number of \`#\`, \`…\` and \`...\` occurrences divided by the word count is above \`cfg.maxSymbolRatio\`.
- \`terminal_punct\`: the fraction of lines (from \`lines(text)\`) matching \`TERMINAL_RE\` is below \`cfg.minTerminalFraction\`.
- \`repeated_lines\`: \`(lines − distinct lines) / lines\` is above \`cfg.maxDuplicateLineFraction\`.
- \`boilerplate\`: the lowercased text contains any phrase in \`cfg.boilerplate\`.

Then write \`qualityFilter(docs, cfg)\`: \`{ kept, removed }\` where \`removed\` holds \`{ id, reason }\`. Pass \`cfg\` through; the tests change thresholds.

These are Gopher's rules (Rae et al. 2021, Appendix A) with smaller word counts. The reason string is not decoration: the report at the end of the pipeline is built from it.
`,
      predict: { question: 'A document of 24 lines like "Menu item number 7", none ending in punctuation. Which reason fires first: terminal_punct or repeated_lines?', answer: 'terminal_punct. The rules run in order and 0 of 24 lines end in terminal punctuation; the lines are all different, so repeated_lines would not fire anyway.' },
      hints: [
        'Each rule is two or three lines: compute one statistic from `ws` or `lines(text)`, compare it with the config, return the reason string. Order matters because only the first violated rule is reported.',
        'Symbols: `(text.match(/#|…|\\.\\.\\./g) || []).length`. Distinct lines: `new Set(ls).size`. Terminal lines: count `ls` entries for which `TERMINAL_RE.test(l)`. Boilerplate: `text.toLowerCase().includes(phrase)` for each phrase.',
        '`const ls = lines(text); let terminal = 0; for (const l of ls) if (TERMINAL_RE.test(l)) terminal++; if (/* fraction below cfg.minTerminalFraction */) return \'terminal_punct\'; const distinct = new Set(ls).size; if ((ls.length - distinct) / ls.length > cfg.maxDuplicateLineFraction) return \'repeated_lines\';`',
      ],
    },
    {
      id: 'exact',
      title: 'Exact deduplication by hash',
      instructions: `
Three functions.

\`normalizeText(text)\`: lowercase, replace every run of characters that is not a letter or digit with a single space, trim. \`"  Hello,   World! "\` becomes \`"hello world"\`.

\`docHash(text)\`: \`hash32(normalizeText(text))\`, the 32-bit FNV-1a fingerprint from \`lib/util.js\`.

\`exactDedup(docs)\`: keep the first document for each fingerprint, in input order; \`removed\` holds \`{ id, dupOf }\` naming the surviving copy.

A 32-bit hash has a collision probability of about \`n² / 2³³\` for \`n\` documents, negligible here; production uses 64-bit or 128-bit hashes for the same reason. Normalising first is what turns "the same page with different capitalisation" into an exact duplicate.
`,
      hints: [
        'One regular expression does the normalisation: a character class for "not a letter or digit", with the global flag, replaced by a space.',
        'Keep a `Map` from hash to the id of the first document that had it. For each document: if the hash is new, record it and keep the document; otherwise push `{ id, dupOf: firstId }`.',
        '`const seen = new Map(); for (const doc of docs) { const h = docHash(doc.text); const first = seen.get(h); if (first === undefined) { /* record and keep */ } else removed.push({ id: doc.id, dupOf: first }); }`',
      ],
    },
    {
      id: 'minhash',
      title: 'Near-deduplication with MinHash',
      instructions: `
Five functions. \`hashFamily(numHashes, seed)\` is already written: it returns \`numHashes\` pairs \`{ a, b }\` defining \`h_i(x) = (a_i · x + b_i) mod 2³²\`.

- \`shingles(text, k = 5)\`: a \`Set\` of every window of \`k\` consecutive words of \`normalizeText(text)\`, joined by single spaces. Fewer than \`k\` words give one shingle of all of them; empty text gives an empty set.
- \`jaccard(a, b)\`: exact \`|a ∩ b| / |a ∪ b|\` for two sets (1 if both are empty).
- \`minhash(shingleSet, numHashes = 32, seed = 1)\`: for each hash function \`i\`, the minimum over all shingles \`s\` of \`(Math.imul(a_i, hash32(s)) + b_i) >>> 0\`. Start every position at \`4294967295\` (the largest 32-bit value).
- \`estimateJaccard(sigA, sigB)\`: the fraction of positions where the signatures agree; throw if the lengths differ.
- \`nearDedup(docs, { k, numHashes, threshold, seed, minReport })\`: compare each document's signature against every earlier **kept** document; remove it on the first estimate \`>= threshold\`, recording \`{ id, nearOf, estimate }\`. Also collect \`pairs\`: every pair with estimate \`>= minReport\`, as \`{ a, b, estimate, jaccard }\` with the exact Jaccard alongside, sorted by estimate descending.

The signature is built inside \`minhash\` from the family, so the same seed always gives the same signature; that is what makes the tests reproducible.
`,
      predict: { question: 'Two documents that are exact copies: what is estimateJaccard of their signatures?', answer: 'Exactly 1: identical sets give identical minima under every hash function. Near-dedup therefore also removes exact duplicates, which is why the exact pass runs first (it is far cheaper).' },
      hints: [
        'Shingles are the sliding windows of the word array: index `i` from 0 while `i + k <= words.length`. For the signature, the `>>> 0` keeps the arithmetic in unsigned 32-bit range after `Math.imul` wraps.',
        'minhash: `fam = hashFamily(numHashes, seed)`; `sig` filled with 4294967295; for each shingle compute `h = hash32(s)` once, then for each `i` compute `v` and keep the smaller of `sig[i]` and `v`. nearDedup: precompute all shingle sets and signatures, keep a list of kept indices, and loop over it for each new document.',
        '`for (const s of shingleSet) { const h = hash32(s); for (let i = 0; i < numHashes; i++) { const v = /* (a_i * h + b_i) mod 2^32 */; if (v < sig[i]) sig[i] = v; } }` and in nearDedup: `for (const i of keptIdx) { const est = estimateJaccard(sigs[i], sigs[j]); if (est >= minReport) pairs.push({ a: docs[i].id, b: docs[j].id, estimate: est, jaccard: jaccard(sets[i], sets[j]) }); if (est >= threshold && hit === null) hit = { id: docs[j].id, nearOf: docs[i].id, estimate: est }; }`',
      ],
    },
    {
      id: 'mix',
      title: 'Domain mixing with epoch caps',
      instructions: `
Implement \`mixDomains(docs, { weights, budget = Infinity, maxEpochs = 1, seed = 1, size = wordCount })\`. Throw if \`weights\` is missing.

For each domain with a positive weight and at least one document, keep a state \`{ domain, weight, docs, queue, ptr, draws, epochs, size }\`. Then loop while \`total < budget\`:

1. The **active** domains are those with \`draws < maxEpochs × docs.length\`. Stop when none is active.
2. Normalise the weights over the active domains only, so an exhausted domain's weight goes to the others.
3. Pick the active domain with the largest deficit \`(weight / sumW) × total − size\` (Megatron-LM's blending rule: the domain furthest below its target share).
4. If its queue is used up, refill it with \`shuffle(next, docs.slice())\` from a single \`rng(seed)\` and count an epoch. Draw the next document, add \`size(doc)\` to the domain's size and to \`total\`, push its id to \`order\`.

Return \`{ order, total, report }\` where \`report[domain] = { docs, draws, epochs: draws / docs, size, share: size / total, weight }\`.

\`size\` defaults to the word count; the pipeline in step 5 passes the token count instead. Weights are shares of size, not of document counts.
`,
      hints: [
        'Two loops, one inside the other: the outer runs until the budget is met or nothing is active; the inner scans the active domains for the largest deficit. Sizes matter: a domain of long documents must be drawn less often for the same share.',
        'Deficit of domain d: `target_d × total − size_d` where `target_d = weight_d / (sum of active weights)`. With `total = 0` every deficit is 0, so the first draw goes to whichever domain you check first; from then on the rule self-corrects.',
        '`while (total < budget) { const active = state.filter((s) => s.draws < maxEpochs * s.docs.length); if (active.length === 0) break; let sumW = 0; for (const s of active) sumW += s.weight; let best = null, bestDeficit = -Infinity; for (const s of active) { const deficit = /* target share times total, minus size */; if (deficit > bestDeficit) { bestDeficit = deficit; best = s; } } if (best.ptr >= best.queue.length) { best.queue = shuffle(next, best.docs.slice()); best.ptr = 0; best.epochs++; } const doc = best.queue[best.ptr++]; … }`',
      ],
    },
    {
      id: 'shards',
      title: 'Token shards, an index, and the whole pipeline',
      instructions: `
\`tokenizeDocs(docs, tokenizer)\` is written: it attaches \`ids\` to each document.

Implement \`packShards(docs, order, { shardSize = 1024, eos })\`: walk \`order\` (document ids, possibly repeated), and for each one append its \`ids\` followed by one \`eos\` to a running shard; whenever the shard reaches exactly \`shardSize\` tokens, push it and start a new one. Before appending a document, record \`{ docId, shard, offset, length }\` in the index, where \`shard\` and \`offset\` say where its first token lands and \`length\` is its own token count (without the \`eos\`). Push a non-empty final shard. Throw on an unknown id, a non-positive \`shardSize\`, or a missing \`eos\`.

Then \`runPipeline(docs, config)\` with \`config = { tokenizer, quality?, near?, mix?, shardSize? }\`: run \`qualityFilter\`, \`exactDedup\`, \`nearDedup(kept, near)\`, \`tokenizeDocs\`, \`mixDomains(toks, { ...mix, size: d => d.ids.length })\`, \`packShards(toks, order, { shardSize, eos: tokenizer.eos })\`. Return \`{ shards, index, tokens, report, removed, pairs, mix }\`: \`report\` is one row \`{ stage, in, out, removed }\` per stage (\`quality\`, \`exact-dedup\`, \`near-dedup\`, \`mix+shard\`), \`removed\` is one \`{ id, stage, reason }\` per removed document, \`tokens\` is the total over all shards, and \`pairs\` comes from \`nearDedup\`.

\`readDoc(shards, entry)\` (written) reads a document back through the index; the tests use it to check that nothing was lost across shard boundaries.
`,
      hints: [
        'Keep one current shard array and a `push(token)` helper that appends and, when the length hits `shardSize`, moves the array into `shards` and starts a fresh one. The index entry for a document is `{ shard: shards.length, offset: cur.length }` at the moment before its first token is pushed.',
        'runPipeline is bookkeeping: after each stage push `{ stage, in, out, removed }` and copy the stage\'s `removed` entries into the flat list with a `stage` field and a reason string (`duplicate of <id>`, `near-duplicate of <id>`). The mixer must measure size in tokens, so pass `size: (d) => d.ids.length`.',
        '`const push = (t) => { cur.push(t); if (cur.length === shardSize) { shards.push(cur); cur = []; } }; for (const id of order) { const doc = byId.get(id); if (!doc) throw new Error(…); index.push({ docId: id, shard: shards.length, offset: cur.length, length: doc.ids.length }); /* push every id, then the eos */ } if (cur.length > 0) shards.push(cur);`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why "which documents to delete" is a modelling decision rather than a cleaning chore, using two rules from your pipeline whose effect you can read off the report.',
    'The MinHash estimate of Jaccard similarity is unbiased but noisy. Why is a noisy similarity with a threshold still better than no near-deduplication, and what would you change first: more hash functions or a different threshold?',
    'Your mixer caps each domain at `maxEpochs`. Walk through what happens to a 1% high-quality domain with weight 0.3 as the budget grows, and connect it to what module 07 taught about the train/validation gap.',
  ],
  stretch: [
    'Replace the pairwise signature comparison in `nearDedup` with locality-sensitive hashing: split each signature into `b` bands of `r` rows, bucket by band, compare only bucket-mates, and plot the probability curve `1 − (1 − J^r)^b` against the pairs your pipeline found. This is how RefinedWeb and FineWeb (via the datatrove library) deduplicate billions of documents.',
    'Add a contamination check: take 20 questions from `MATH_TASKS` in `lib/data.js`, plant three of them inside documents, and use the shingle sets plus the shard index to report which shard and offset each leaked question sits at. Every serious eval report (GPT-4, Llama 3) includes a table like this.',
    'Train a tiny "quality classifier": label the demo\'s documents by domain, fit a bag-of-words logistic regression with the ops from module 01, and use its probability as a filter score. DCLM and FineWeb-Edu do this with fastText and a small Llama-based annotator, respectively.',
    'Implement DoReMi\'s idea in miniature: train the bigram model from module 04 on each domain, measure its loss per domain, and reweight domains towards the ones where the loss gap is largest. Compare the resulting weights with the hand-set Pile weights.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
