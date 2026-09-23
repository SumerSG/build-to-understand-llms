export default {
  id: '21-context',
  title: 'Context management & retrieval',
  track: 'harness',
  minutes: 90,
  threshold: 'The context window is a scarce, ordered budget of tokens, and managing it is a retrieval and compression problem (what to keep, what to summarise, what to fetch back): the policy, not the model, decides what the model can know on a given turn.',
  goal: 'A context manager with token budgeting, truncation, compaction and BM25 retrieval that keeps a long conversation under budget without losing key facts; the demo runs a 60-turn conversation under an 800-token budget and shows how many late questions each policy can still answer.',
  prereqs: ['03-tokenizer', '17-prefix-caching', '20-agent-loop'],
  recall: [
    { q: 'Module 20: an agent calls one tool on each of its first 8 turns (about 2,000 tokens per result) and answers on turn 9. Why does the provider read roughly 70,000 tool-result tokens across the 9 calls rather than 16,000?',
      options: ['Tool results are tokenized twice', 'The whole transcript is re-sent on every turn, so the cost is the sum of a growing prefix', 'The system prompt is repeated after every tool call'], answer: 1,
      why: 'Call k re-reads the k − 1 results already in the transcript: (0 + 1 + … + 8) · 2,000 = 72,000. Every model call reads the entire message array. That is why the size of the array you assemble on each turn, which is what this module controls, is the cost driver of a long-running agent.' },
    { q: 'Module 17: you must add a per-request block of text (a timestamp, retrieved notes) to every call. Where does it hurt the prefix-cache hit rate least?',
      options: ['At the very start, before the system prompt', 'As late as possible, after the stable content', 'It makes no difference'], answer: 1,
      why: 'A cached prefix stops at the first differing token. Anything that changes per request should come after everything that does not, which is why this module injects retrieved notes just before the newest message.' },
    { q: 'Module 03: the same English sentence is encoded by a 256-symbol tokenizer and by a 4,096-symbol BPE tokenizer trained on similar text. Which produces fewer tokens?',
      options: ['The 256-symbol one', 'The 4,096-symbol one', 'They produce the same number'], answer: 1,
      why: 'More merges mean longer tokens. Token counts are a property of the tokenizer, so a budget has to be measured with the tokenizer the model actually uses, not with a characters-per-token guess.' },
    { q: 'Module 15: the KV cache for one sequence grows how with the number of tokens in the context?',
      options: ['Constant', 'Linearly', 'Quadratically'], answer: 1,
      why: 'Every token stores one key and one value per layer. A longer context costs memory on the server as well as money and attention quality, which is part of why providers cap context length.' },
    { q: 'Module 32: in the InfoNCE loss of your CLIP head, a batch holds B image-caption pairs. Where do the negatives for an image come from?',
      options: ['A separate set of hand-labelled mismatches', 'The other B − 1 captions in the same batch', 'Random noise vectors'], answer: 1,
      why: 'Every other pair in the batch is a negative, which is why CLIP trained on batches of 32,768. Text embedding models for retrieval are trained the same way on (query, passage) pairs, and step 6 ranks notes with the kind of vectors that loss produces.' },
  ],
  review: [
    { q: 'Why does `dropOldest` skip messages with role "system" even when the result stays over budget?',
      options: ['System messages are free', 'The system prompt carries the instructions and pinned summaries; a policy that can delete them silently changes what the model is', 'System messages are always short'], answer: 1,
      why: 'Over budget is reported and can be handled (bigger budget, shorter system prompt). Losing the instructions is invisible and produces a model that behaves differently without any error.' },
    { q: 'In BM25 with k1 = 1.5, a term that appears 100 times in a note contributes at most…',
      options: ['100 × idf', '(k1 + 1) × idf = 2.5 × idf', 'idf'], answer: 1,
      why: 'The term-frequency factor `tf·(k1+1) / (tf + k1·norm)` saturates at `k1 + 1`. Repeating a word does not make a note endlessly more relevant, which is the main thing BM25 fixes over raw counting.' },
    { q: '`assemble()` subtracts the cost of the retrieval message from the budget BEFORE trimming the history because…',
      options: ['Retrieval is cheaper that way', 'Otherwise the trimmed history fills the whole budget and adding the notes pushes the final array over it', 'BM25 needs the full history'], answer: 1,
      why: 'Budgets compose by reservation: fixed parts (system prompt, retrieved notes) are paid for first, and the remaining tokens go to the part that can shrink.' },
    { q: 'A term that occurs in every note of a store of N = 50 notes has idf `ln(1 + (N − df + 0.5)/(df + 0.5))` of about…',
      options: ['0.01', '1', '3.9'], answer: 0,
      why: '`ln(1 + 0.5/50.5) ≈ 0.0099`. Words that appear everywhere ("the", "is") are almost worthless as evidence; rare words ("vault", "heron") carry the score.' },
    { q: 'Your agent drops the oldest non-system message on every turn to stay under budget. What happens to a provider\'s prefix cache?',
      options: ['Nothing, the cache is per-token', 'It misses after the system prompt on every turn, because the array no longer starts with the same messages', 'It hits more often because the context is shorter'], answer: 1,
      why: 'A sliding window changes the second message every turn. Compaction that replaces a span only occasionally, and appends otherwise, keeps a stable prefix between compactions.' },
  ],
  concept: `
## A window is not a memory

Every model call sees exactly one array of messages, and that array has a hard limit: the **context window**, measured in tokens. Claude and GPT-4-class models advertise windows of approximately 128,000 to 1,000,000 tokens (per their providers' documentation). An agent fills it fast: module 20 showed that each tool result is re-sent on every later turn. The window is a **budget**, and it is **ordered**: position matters as much as presence.

Two effects shrink it further. Liu et al. (2023, "Lost in the Middle") showed that models answer questions best when the relevant passage is near the start or the end of the context, and noticeably worse when it sits in the middle (module 33 measured this curve on a synthetic model). Chroma's 2025 "Context Rot" report found that accuracy on simple retrieval tasks drops as input length grows, well before the advertised limit. The **effective context** is shorter than the nominal one.

:::predict
Your demo conversation runs 60 turns with a 24-line log dump every fifth turn. Unmanaged, how large do you expect the final context to be, and what share of it is raw tool output?
---
On the solution it reaches approximately 11,100 BPE tokens, about 14× the 800-token budget, and roughly a third of that is tool output. In real coding agents the share is often higher: file contents and command output dominate the transcript, which is why truncating tool results is the first policy every harness applies.
:::

## Four policies

Production harnesses combine them:

1. **Count.** Cost is \`TOKENS_PER_MESSAGE + tokens(content)\` per message, measured with the real tokenizer. The per-message term is the role framing; OpenAI's token-counting cookbook uses 3 or 4 per chat message, by model.
2. **Truncate.** Cut long tool results to a cap, keeping the **head and the tail** (errors and summaries tend to be at the end of logs) with a marker in between so the model knows something was removed.
3. **Drop oldest.** A sliding window: remove the oldest non-system message until the array fits. Cheap, and it forgets everything older than the window.
4. **Compact.** Replace a span of old turns by one summary message. Claude Code's auto-compact and similar features in other agents do this with an LLM call when the window is nearly full. Yours uses a scripted summariser that keeps every \`remember:\` line, so you can test that facts survive.

## Retrieval: fetch back what you dropped

Compaction keeps facts in the window, paying for them every turn. The alternative is **retrieval-augmented generation (RAG)**: keep facts in an external store and, each turn, fetch only the few relevant to the latest message. Documents are split into **chunks** (a few hundred tokens each) so a hit returns a passage, not a manual; your notes are one-fact chunks.

You will rank notes with **BM25** (Robertson et al., Okapi, 1990s). For a query term with document frequency \`df\` among \`N\` notes, and a note of length \`len\` where the term appears \`tf\` times:

\`\`\`
idf   = ln(1 + (N - df + 0.5) / (df + 0.5))
norm  = 1 - b + b * len / avgdl
score = Σ over query terms of  idf * tf * (k1 + 1) / (tf + k1 * norm)      k1 = 1.5, b = 0.75
\`\`\`

\`avgdl\` is the average note length. Rare terms weigh more, repeats saturate (k1), long notes are discounted (b).

:::predict
The store holds "the deploy key is in vault slot 7" and "the build server is called atlas". The user asks "which slot holds the key?". Which note does BM25 return, and would it find the right note for "where are my credentials?"
---
The deploy-key note: "slot" and "key" are rare terms that match. (The atlas note scores a little too, through "the", whose idf is small.) The second query shares no word with any note, so every score is 0 and nothing comes back. That is the lexical gap dense embeddings close, and why hybrid retrieval exists.
:::

## Dense and hybrid retrieval

A **dense retriever** maps each text to one vector and ranks notes by cosine similarity to the query's vector, so it can match "credentials" to "key" if its embedding model has learned they are related. DPR (Karpukhin et al. 2020) and Contriever (Izacard et al. 2021) train that model contrastively, with an InfoNCE loss like your CLIP head's in module 32: each query must pick its own passage out of many negatives, such as the other passages in its batch. Dense is not strictly better: on the BEIR benchmark (Thakur et al. 2021) plain BM25 beat many dense retrievers on unseen domains.

At scale, stores do not score every vector; they search an **approximate nearest-neighbour** index such as HNSW (a layered proximity graph, Malkov and Yashunin 2016) or FAISS's IVF-PQ (probe only the nearest k-means clusters, over product-quantised vectors). **Hybrid search** merges the BM25 and dense rankings by **reciprocal rank fusion** (step 6); Elasticsearch and Vespa both offer it. A **cross-encoder reranker** (Nogueira and Cho 2019) then reads the query and each candidate passage together in one transformer: more accurate than comparing two vectors, but one forward pass per candidate, so it only reorders a shortlist.

## Ordering and caching

Module 17 showed that a provider's prefix cache only hits on an identical prefix, so **stable content goes first** (system prompt, pinned summary) and volatile content last. Retrieved notes change every turn, so your manager injects them just before the newest message: the end of the context, which is also where "lost in the middle" says the model reads best. A sliding window, by contrast, changes the second message on every turn and defeats the cache; compaction changes the prefix only when it runs.

## Where this toy differs from production

Your summariser is a regular expression that keeps tagged facts perfectly; a real one is a model call that paraphrases, drops details it judged unimportant, and costs its own tokens and latency. Your notes are one-line strings and your index is rebuilt from scratch; production stores chunk documents and keep inverted and ANN indexes updated incrementally. The step 6 tests write the embedding rows by hand and pool them without context; a real embedding model is a trained transformer whose token vectors depend on their neighbours. Your 800-token budget and small BPE make absolute counts unlike any commercial model. The policies, the ordering rules and the failure modes are the same.
`,
  steps: [
    {
      id: 'count',
      title: 'Count the cost of a context',
      instructions: `
Every decision in this module is arithmetic on token counts, so they must be exact. \`countTokens(tokenizer, text)\` is done for you: it is the only place the tokenizer is called.

- \`messageTokens(tokenizer, msg)\`: \`TOKENS_PER_MESSAGE\` plus the tokens of \`msg.content\` plus the tokens of \`msg.name\` if present (a tool message carries its tool name into the context).
- \`contextTokens(tokenizer, messages)\`: the sum over the array. An empty array costs 0.
- \`budgetReport(tokenizer, messages, budget)\`: \`{ tokens, budget, remaining, fits }\` with \`remaining = budget − tokens\` (negative when over) and \`fits = tokens <= budget\`.

The tests use a tokenizer with one id per word so you can check every number by hand, and a character tokenizer to make sure you call the tokenizer you are given rather than guessing.
`,
      predict: { question: 'A message with empty content: what does it cost?', answer: '`TOKENS_PER_MESSAGE` (4). The role marker and end-of-message framing are sent even when there is no text. Across hundreds of short messages this overhead is a real share of the budget.' },
      hints: [
        '`countTokens` already returns 0 for missing or empty text, so you can call it on `msg.name` without checking whether it exists.',
        'A message costs three things added together: the fixed framing overhead and the token counts of its two strings. A context costs the sum of its messages. The report needs the count only once; the other three fields follow from it and the budget.',
        '`const tokens = contextTokens(tokenizer, messages); return { tokens, budget, remaining: …, fits: … };` — remember that exactly on budget still fits.',
      ],
    },
    {
      id: 'truncate',
      title: 'Truncation policies',
      instructions: `
Three functions; none of them may modify its input.

**\`truncateText(tokenizer, text, maxTokens)\`**: if the text already fits, return it unchanged. Otherwise keep \`keep = maxTokens − tokens(TRUNCATION_MARKER)\` tokens of the original: the first \`ceil(keep / 2)\` and the last \`keep − ceil(keep / 2)\`, decoded back to text, with \`TRUNCATION_MARKER\` between them. The result must cost exactly \`maxTokens\` on the test tokenizer. Head and tail both matter: a log's command line is at the top, its error at the bottom.

**\`truncateToolResults(tokenizer, messages, maxToolTokens)\`**: a new array where each message with role \`'tool'\` whose content costs more than \`maxToolTokens\` is replaced by a copy (\`{ ...msg, content }\`) with truncated content. Every other message is the same object.

**\`dropOldest(tokenizer, messages, budget)\`**: remove the oldest non-system messages, one at a time, until the array fits, and not one more. Messages with role \`'system'\` are **never** removed; if they alone exceed the budget, return them and let the caller see the overflow.
`,
      predict: { question: 'The system prompt costs 34 tokens and the budget is 10. What should dropOldest return?', answer: 'Just the system prompt, still over budget at 34. Being over budget is visible and fixable; silently deleting the instructions is neither.' },
      hints: [
        'The budget is in tokens, so a cut measured in characters will not land on exactly maxTokens: in which representation of the text can you count and cut at the same time? For dropOldest, what do you need to know after each removal, and can you get it without recounting everything?',
        'For dropOldest: copy the array, compute its cost, then walk an index from the front. A system message advances the index; any other message is spliced out and its cost subtracted. Stop as soon as the total fits.',
        '`const ids = tokenizer.encode(text); if (ids.length <= maxTokens) return text; const keep = …; const head = Math.ceil(keep / 2); return tokenizer.decode(ids.slice(0, head)) + TRUNCATION_MARKER + /* the last keep − head ids */;` — watch the case where the tail length is 0, since `ids.slice(-0)` is the whole array.',
      ],
    },
    {
      id: 'compact',
      title: 'Compaction into a summary',
      instructions: `
**\`extractFacts(text)\`** is the scripted summariser: every line that starts (after optional whitespace) with \`remember:\` in any case yields the rest of the line, trimmed and non-empty. Drop duplicates, keep first-occurrence order, return \`[]\` for missing text.

**\`compact(messages, { keepLast = 4 })\`**: take the non-system messages; all but the last \`keepLast\` of them form the span to compact, together with any earlier summary message (\`summary: true\`) that sits before them. Replace the span by **one** message placed where the span began:

\`\`\`
{ role: 'system', summary: true, count, content: 'Summary of <count> earlier messages.\\nremember: fact\\nremember: fact' }
\`\`\`

\`count\` is how many original messages the summary stands for; an old summary contributes its own \`count\`. The role \`'system'\` pins it (your dropOldest will never remove it), and writing facts back as \`remember:\` lines means the next compaction reads them again. If nothing is older than \`keepLast\`, return a copy unchanged.
`,
      predict: { question: 'You compact once, the conversation continues, and you compact again. If your second compaction ignored the first summary, what would the model lose?', answer: 'Every fact from before the first compaction. Folding the old summary into the new one, with its facts and its count, is what makes compaction repeatable. Agents that forget after their second compaction usually have exactly this bug.' },
      hints: [
        'Collect indices first, then build the output. The candidates for the span are the non-system messages plus any summaries; the last keepLast non-summary messages are kept.',
        'Find the index of the last message to compact (the `(n − keepLast)`-th non-summary candidate). The span is every candidate at or before it. Extract facts from the joined contents of the span (old summaries included), sum the counts, then rebuild the array, emitting the summary at the first span index and skipping span members.',
        '`/^\\s*remember:\\s*(.+?)\\s*$/i` matches a fact line. For the count: `for (const i of span) count += messages[i].summary ? /* its count */ : 1;` and the output loop is `if (i === span[0]) out.push(summary); if (!spanSet.has(i)) out.push(messages[i]);`.',
      ],
    },
    {
      id: 'bm25',
      title: 'BM25 from scratch',
      instructions: `
Fill in \`BM25Index\`. Use \`tokenizeTerms\` (lowercase words, done for you) for both documents and queries.

- **constructor**: for each document store a \`Map\` of term counts in \`this.tf\`, its length in terms in \`this.lengths\`, count each distinct term once per document in \`this.df\`, and set \`this.avgdl\`.
- **\`idf(term)\`**: \`ln(1 + (N − df + 0.5) / (df + 0.5))\`, with \`df = 0\` for unseen terms. The \`1 +\` keeps it positive even for terms in every document (Lucene uses this form).
- **\`score(query, i)\`**: sum over the query's terms of \`idf · tf · (k1 + 1) / (tf + k1 · (1 − b + b · len / avgdl))\`; terms with \`tf = 0\` add nothing.
- **\`search(query, k)\`**: \`[{ index, score }]\` for the \`k\` best documents with score > 0, best first, equal scores in index order.

The tests compare against scores computed by hand, and check that a longer note scores lower for the same tf, that \`b = 0\` removes that effect, and that tf saturates below \`(k1 + 1) · idf\`.
`,
      hints: [
        'Build the statistics once in the constructor; score then only does lookups. The document frequency counts documents, not occurrences, so iterate over each document\'s distinct terms when updating df.',
        'Per document: counts = new Map; for each term increment it; then for each key of counts increment df. For score: compute the length normaliser once per document, then loop over query terms, skip tf = 0, and add the per-term contribution.',
        '`const norm = 1 - this.b + this.b * (this.lengths[i] / this.avgdl); for (const term of tokenizeTerms(query)) { const tf = counts.get(term) ?? 0; if (tf === 0) continue; s += /* idf times the saturating tf factor */; }` and search sorts with `(a, b) => b.score - a.score || a.index - b.index`.',
      ],
    },
    {
      id: 'assemble',
      title: 'Assemble the context each turn',
      instructions: `
Two methods of \`ContextManager\` remain; \`remember\` and \`add\` are done for you.

**\`retrieve(query)\`**: \`[]\` when \`topK\` is 0 or the store is empty; otherwise build \`this.index = new BM25Index(this.notes)\` if it is null, search with \`topK\`, and return the note strings.

**\`assemble()\`** returns \`{ messages, tokens, retrieved }\`:

1. Query = content of the newest user message in the history. \`retrieved = this.retrieve(query)\`; if non-empty, build \`{ role: 'user', retrieved: true, content: 'Relevant notes:\\n- note\\n- note' }\`.
2. Unless the policy is \`'none'\`: truncate tool results to \`maxToolTokens\`; reserve the system prompt and the retrieval message from the budget; with \`'compact'\`, compact the history if it does not fit what remains; then \`dropOldest\` to what remains (compaction's last resort).
3. \`messages\` = system prompt, then the managed history with the retrieval message inserted **immediately before the newest history message**. \`tokens\` = \`contextTokens(messages)\`.

\`this.history\` is never modified: assemble is a view computed from it on every call.
`,
      predict: { question: 'With policy "drop", topK = 0 and a fact stated on turn 2 of 50, can the model answer a question about that fact on turn 50?', answer: 'No. The fact left the window around the time the budget first filled, and nothing brings it back. With topK = 2 the same fact was harvested into the note store by add(), and BM25 returns it when the question shares its words. Retrieval turns "forgotten" into "fetched on demand".' },
      hints: [
        'Separate the fixed parts (system prompt, retrieval message) from the part that can shrink (the history). Only the history goes through truncateToolResults, compact and dropOldest.',
        'historyBudget = budget − contextTokens([system, retrievalMsg?]). Compact only if the truncated history exceeds historyBudget; always finish with dropOldest(history, historyBudget). Then build the output array, inserting the retrieval message when you reach the last history index.',
        '`const fixed = [this.system]; if (retrievalMsg) fixed.push(retrievalMsg); … const historyBudget = /* reserve the fixed parts */; … for (let i = 0; i < history.length; i++) { if (retrievalMsg && i === history.length - 1) messages.push(retrievalMsg); messages.push(history[i]); }` — and handle an empty history.',
      ],
    },
    {
      id: 'hybrid',
      title: 'Dense and hybrid retrieval',
      instructions: `
BM25 cannot match "credentials" to "key". A dense retriever can, if its embedding model puts the two words near each other. Build the four pieces, then fuse the two rankings.

- **\`meanPool(table, ids)\`**: a text's vector is the mean of its tokens' rows in a \`[V, C]\` table with \`shape\` and flat row-major \`data\`, such as \`model.wte.weight\` of a \`lib/gpt.js\` GPT. Row \`id\` starts at \`data[id * C]\`. A repeated id counts every time; no ids gives \`C\` zeros.
- **\`cosine(a, b)\`**: \`a·b / (|a| |b|)\`, where \`|a|\` is the Euclidean length; 0 when either vector is all zeros.
- **\`denseSearch(docVectors, queryVector, k = 3)\`**: \`[{ index, score }]\` for the \`k\` highest cosines, best first, ties by index. Unlike BM25's \`search\`, keep zero and negative scores: a nearest neighbour always exists.
- **\`reciprocalRankFusion(rankings, { k = 60 } = {})\`**: \`rankings\` is a list of hit lists, each best first. Document \`d\` scores \`Σ 1 / (k + rank_d)\` over the lists it appears in, with \`rank_d\` counted from 1; the hits' own scores are ignored. Return every document that appears in any list as \`[{ index, score }]\`, best first, ties by index.

RRF fuses ranks rather than scores because BM25 scores are unbounded and cosines lie in [−1, 1]: added together, whichever scale is larger would decide. The constant \`k = 60\` is the one Cormack et al. (2009) used. It keeps first place from dominating: a note ranked second by both retrievers (\`2/62\`) beats one ranked first by only one of them (\`1/61\`).

The tests write the embedding rows by hand as a stand-in for a contrastively trained model, so that "credentials", "key", "password" and "passphrase" share one direction. On their query BM25 ranks the deploy-key note second, behind a shorter note that also says "vault"; dense retrieval ranks it second, behind the password note; the fused ranking puts it first.
`,
      predict: { question: 'With k = 60, one note is first in the BM25 list and absent from the dense list; another is tenth in both. Which does RRF rank higher?', answer: 'The one that is tenth in both: 2/70 ≈ 0.029 against 1/61 ≈ 0.016. Agreement between two different retrievers counts for more than one retriever\'s favourite. For k below 8 the answer flips, because 1/(k + 1) then exceeds 2/(k + 10) (they tie at k = 8; at k = 0 it is 1/1 against 2/10). A large k stops one retriever\'s favourite from outvoting agreement.' },
      hints: [
        'Each function is a few lines. For the fusion, ask what you must accumulate per document across all the lists, and what a hit\'s position in its list tells you that its score does not.',
        'meanPool: a Float64Array of C zeros; add row id for each id, then divide by ids.length. cosine: one loop that accumulates a·b, |a|² and |b|². denseSearch: score every document, sort with the same comparator as BM25 search, slice to k. Fusion: a Map from index to score; for each ranking and each position r counted from 0, add 1 / (k + r + 1); then turn the Map into hits and sort.',
        '`for (const ranking of rankings) ranking.forEach((hit, r) => scores.set(hit.index, (scores.get(hit.index) ?? 0) + /* 1 / (k + the 1-based rank) */)); const fused = [...scores].map(([index, score]) => ({ index, score }));` and for cosine: `if (na === 0 || nb === 0) return 0; return /* dot over the product of the two lengths */;`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why a model with a 200,000-token window still needs a context manager. Use "effective context", cost per turn and tool-result bloat in your answer.',
    'In the demo, compaction and BM25 memory both kept all 10 late questions answerable, yet compaction held all 5 facts in every context and retrieval only the 2 it fetched. What does each one cost per turn, what can each one lose, and which would you choose for an agent that runs for a day?',
    'Your manager decides what the model can know on every turn. Describe one failure a user would blame on "the model" that is really a context-policy decision.',
  ],
  stretch: [
    'Replace the scripted summariser with a prompt to the lab\'s tiny GPT (module 14\'s sampler), then measure how many facts survive. This is the gap between your compact() and Claude Code\'s LLM-written auto-compact summaries.',
    'Train the embeddings instead of writing them: fit a lib/gpt.js embedding table with module 32\'s InfoNCE loss on (question, fact) pairs, plug `denseSearch` and `reciprocalRankFusion` into `ContextManager.retrieve`, and count how many of the demo\'s ten late questions BM25, dense and hybrid retrieval each answer. DPR, Contriever and E5 are this recipe at scale.',
    'Make compaction cache-aware: compact in large steps only when the budget is nearly full, and count how many turns keep an identical prefix, using your prefix cache from module 17. Compare against the sliding window.',
    'Implement MemGPT-style (Packer et al. 2023) memory tools: let the scripted model call `memory_write` and `memory_search` through your module 20 harness instead of harvesting `remember:` lines automatically.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
