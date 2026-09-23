export default {
  id: '22-structured-output',
  title: 'Structured outputs & constrained decoding',
  track: 'harness',
  minutes: 90,
  threshold: 'You can guarantee a format without any training by masking, at every step, every token that could not continue a valid output: the grammar becomes a filter on the logits.',
  goal: 'A grammar-constrained sampler that masks logits token by token so the model can only emit JSON matching a schema: a story model that has never seen JSON, and even a model with random logits, produce schema-valid objects every time.',
  prereqs: ['03-tokenizer', '14-decoding', '15-kv-cache', '20-agent-loop'],
  recall: [
    { q: 'Module 14: a logit is set to -Infinity before `softmaxLogits`. What probability does that token get?', options: ['A tiny positive number such as 1e-38', 'Exactly 0', 'NaN, which the sampler skips'], answer: 1,
      why: '`exp(-Infinity - max)` is exactly 0 in floating point, and the stable softmax subtracts a finite max, so the token can never be drawn. That is the whole mechanism of this module.' },
    { q: 'Module 03: the pre-tokenizer pattern ` ?[^\\sA-Za-z\\d]+` treats a run of punctuation such as `",` as…', options: ['Separate characters that can never merge', 'One pre-token, so BPE can learn it as a single token', 'Whitespace, which is dropped'], answer: 1,
      why: 'Punctuation runs stay together, so real vocabularies contain tokens like `",` or `"}` that close a string AND do something else. That is why a grammar has to be checked character by character inside each token.' },
    { q: 'Module 15: what is stored in the KV cache after a decode step?', options: ['The logits of every candidate token', 'The key and value vectors of the token that was actually appended', 'The probabilities after sampling'], answer: 1,
      why: 'The cache depends only on the tokens chosen. A mask changes which token is chosen, never what the forward pass computes, so constrained decoding needs no change to the cache.' },
    { q: 'Module 20: `parseToolCalls` met a `<tool_call>` block whose body failed `JSON.parse`. It…', options: ['Threw and stopped the agent', 'Skipped the block and kept scanning', 'Repaired the JSON by adding quotes'], answer: 1,
      why: 'Skipping keeps the harness alive but silently loses the model\'s intent. Constrained decoding removes this failure at its source: the malformed block can no longer be generated.' },
  ],
  review: [
    { q: 'Why is the end-of-sequence token masked until the automaton accepts?', options: ['To make outputs longer', 'Stopping early would leave truncated JSON, such as an object with a required key missing', 'EOS has no logit'], answer: 1,
      why: 'Validity needs two things: every emitted token keeps the text completable, and generation may end only at a complete value. Masking EOS until `accepts(state)` enforces the second.' },
    { q: 'You are inside the name string and "age" is still required. The single token `"}` is…', options: ['Allowed: its first character closes the string legally', 'Masked: the `}` would close the object while a required key is missing', 'Allowed only at temperature 0'], answer: 1,
      why: 'A token is allowed only if EVERY character keeps the automaton alive. `"` is fine, `}` is not, so the whole token is masked.' },
    { q: 'Constrained decoding with a schema guarantees…', options: ['That the output parses and matches the schema', 'That the values are true', 'That the output is what the unconstrained model would most likely have written'], answer: 0,
      why: 'The mask enforces form, not content: the demo produces negative ages and nonsense names. It also reshapes the distribution: where the model put under 1% of its mass on legal tokens, the grammar did the choosing.' },
    { q: 'Speculative decoding (module 18) is combined with a grammar. The verify step must…', options: ['Ignore the grammar: the target model is always right', 'Apply the mask at every drafted position, advancing the automaton along the draft, and reject any draft token the grammar forbids', 'Run the grammar only on the final text'], answer: 1,
      why: 'The distribution being sampled is now the masked target. Verifying against the unmasked target would accept tokens the grammar forbids, and the guarantee would be lost.' },
    { q: 'Why does the mask cache in step 5 hit about 99% of the time in the demo?', options: ['Masks are approximations', 'A mask depends only on the automaton state, and generation visits a few dozen distinct states thousands of times', 'The model repeats itself'], answer: 1,
      why: 'That observation is the basis of compile-ahead engines: Outlines precomputes state → allowed tokens for a finite-state machine, and XGrammar precomputes most of the mask for its pushdown automaton.' },
  ],
  concept: `
## Why constrain at all

In module 20 the harness parsed tool calls with \`JSON.parse\`, and a malformed block was simply skipped. A model that writes \`{"age": thirty}\` crashes nothing, but its intent is gone. Fine-tuning and retries help; there is also a way to make invalid output **impossible**, for any model, with no training.

At every step the model produces logits over a vocabulary of \`V\` tokens. Given the text so far, some of those tokens can never lead to a valid output. Set their logits to \`-Infinity\`: softmax gives them probability exactly 0, and the sampler from module 14 picks among the rest as usual. If at least one token always survives, and the end-of-sequence token (EOS) survives only once the text is complete, then **every** output is valid by construction. The grammar has become a filter on the logits.

## From schema to automaton

The filter must answer "can this prefix still become valid?" many times per token, so you compile the schema into a **character-level automaton**: a start state, \`step(state, ch)\` that returns the next state or \`null\` when \`ch\` makes the prefix hopeless, and \`accepts(state)\` for "complete". For \`{"age":42}\` the object machine reads \`{\`, matches the key text \`"age":\`, hands \`4\` and \`2\` to an integer machine, and on \`}\` checks that the integer is complete and no required key is missing.

:::predict
You attach the mask to a "model" whose logits are pure random numbers and draw 100 objects. What fraction parse and match the schema?
---
All 100. Validity comes from the mask, not from the model; the model only chooses among tokens the grammar allows. The step 4 tests check exactly this.
:::

## Token boundaries make it tricky

Models emit tokens, not characters. A BPE token such as \`",\` closes a string and writes a comma; \`" the"\` starts with a space; \`tru\` stops in the middle of a literal. The rule you will implement: **a token is allowed if every one of its characters keeps the automaton alive**, so you simulate each token's characters from the current state. Consequences: a token may end mid-value, which is fine; the constrained text is often tokenized differently from how the tokenizer would split the same string, so the model reads sequences it rarely saw in training; and if the vocabulary lacks a character the grammar needs, you reach a dead end: the lab tokenizer has no \`{\` or \`}\`, so the demo appends them as two extra tokens with a very low logit. Byte-level BPE, as in GPT-2 and Llama 3, covers all 256 bytes and never has this problem.

:::predict
The text so far is \`{"name":"ab\` and "age" is a required key. Is the single token \`",\` allowed? Is \`"}\`?
---
\`",\` is allowed: the quote closes the name and the comma leads to the next key. \`"}\` is masked: its quote is fine, but \`}\` would close the object with "age" missing, and one bad character sinks the whole token.
:::

## The cost, and the compile-ahead trick

A naive mask scans all \`V\` tokens at every step. With 258 tokens that is cheap; with Llama 3's 128,256, times every character of every token, it becomes a per-step cost that engines work hard to hide. But the mask depends **only on the automaton state**, not on the model or the exact prefix. Generation revisits a handful of states thousands of times, so you cache masks by state (step 5). Outlines (Willard & Louf, 2023) takes this to its limit: it compiles a regex or JSON Schema into a finite-state machine and precomputes, for every state, the set of allowed tokens. Nested JSON needs a stack (a pushdown automaton) whose states cannot all be listed, so XGrammar (Dong et al., 2024) precomputes the tokens whose validity ignores the stack and checks the few others at runtime. llguidance, the Rust engine behind Microsoft's Guidance, computes masks lazily with a lexer and an Earley parser. vLLM and SGLang can use XGrammar, llguidance or Outlines as the structured-output backend.

The mask does not touch the KV cache: it acts on logits after the forward pass. It does constrain speculative decoding (module 18): the verify step must apply the mask at every drafted position and reject a draft token the grammar forbids. APIs expose two strengths: **JSON mode** promises parseable JSON of any shape, while **schema mode** (OpenAI's Structured Outputs, for example) compiles your JSON Schema into a grammar like the one you will build.

## What the guarantee does not cover

The mask enforces form, not content. The demo's story model writes names like "My mar my te" and negative ages, because the schema said "integer", not "age". Masking also changes the distribution: renormalising over legal tokens step by step is not sampling from the model conditioned on a valid output. Where the model put less than 1% of its probability on legal tokens, the grammar made the choice. Finally, a \`max_tokens\` cut-off can still truncate JSON. Here \`maxLength\` and \`maxDigits\` bound the length of every value, so the longest legal object is finite (124 characters for the demo schema, when every name character is a \`\\uXXXX\` escape) and a \`maxTokens\` at least that large always lets generation finish.

## Where the toy differs from production

Your grammar covers compact JSON with keys in a fixed order: no whitespace outside strings, no arrays, no floats, no \`minimum\` or \`pattern\`, and \`maxDigits\` is a lab stand-in for JSON Schema's \`maximum\`. Each token is simulated character by character in JavaScript. Production engines work on bytes, accept full JSON Schema or general context-free grammars (EBNF, Lark), allow flexible whitespace, and apply precomputed bitmasks to a whole batch of logits on the GPU.
`,
  steps: [
    {
      id: 'values',
      title: 'String and integer machines',
      instructions: `
A **machine** is \`{ start, step(state, ch), accepts(state) }\`. \`step\` returns a NEW state, or \`null\` as soon as \`ch\` means the text can never become valid. It never looks ahead and never mutates \`state\`. The starter's worked \`literalMachine\` (for booleans and enums) shows the pattern, and \`advance\` and \`matches\` run a machine over a string.

Implement two machines:

- \`stringMachine(maxLength)\`: a JSON string, quotes included, of at most \`maxLength\` characters. The starter handles the \`'open'\` phase. Fill in \`'body'\` (a quote closes; control characters below \`0x20\` are illegal; when \`n === maxLength\` only the quote is left; a backslash starts an escape), \`'esc'\` (one of \`"\` \`\\\` \`/\` \`b\` \`f\` \`n\` \`r\` \`t\`, or \`u\` followed by exactly four hex digits) and \`'hex'\`. An escape counts as **one** character.
- \`integerMachine(maxDigits)\`: \`-?(0|[1-9][0-9]*)\` with at most \`maxDigits\` digits. \`-0\` is valid JSON; \`007\`, \`+3\` and a lone \`-\` are not.

Why the caps? They make every value finite, so a sampler driven by random logits cannot write a 10,000-character name, and every generation terminates.
`,
      predict: { question: 'After reading only "-", what should the integer machine return from step, and what should accepts say?', answer: 'step returns a live state (a digit can still follow), and accepts returns false (a lone minus is not a number). "Alive" and "accepting" are different questions, and the mask in step 3 needs the first one.' },
      hints: [
        'Treat each phase as the question "which characters may come next here?". Return null the moment the answer is "none"; never peek at later characters.',
        'String body: check the closing quote first (it is legal even when the string is full), then control characters, then the length cap, then a backslash, then "anything else is one more character". Integer: from start or minus, "0" goes to a phase that nothing may follow and 1–9 go to digits; digits accept more digits while d < maxDigits; accepts is true in the zero and digits phases.',
        'The body phase, with one line left for you:\n```js\nif (p === \'body\') {\n  if (ch === \'"\') return { p: \'done\', n, h: 0 };\n  if (ch.charCodeAt(0) < 0x20) return null;\n  if (n >= maxLength) return null;\n  if (ch === \'\\\\\') return /* enter the escape phase: n does not grow yet */;\n  return { p: \'body\', n: n + 1, h: 0 };\n}\n```',
      ],
    },
    {
      id: 'object',
      title: 'The object grammar',
      instructions: `
Implement \`compileSchema(schema)\` for \`{ type: 'object', properties: { key: spec, … }, required: [...] }\`. It returns a machine for **compact** JSON (no whitespace outside strings) with keys in declaration order:

- Build one value machine per property with \`compileValue\` (this also makes nested objects work) and one key literal per property, \`JSON.stringify(key) + ':'\`, e.g. \`"age":\`.
- After \`{\` or \`,\` the next key may be any later key up to and including the first **required** one; optional keys may be skipped, required keys may not.
- Inside a value, try that property's machine first. Only if it rejects the character **and** the value is complete may the character be \`,\` (if another key can still come) or \`}\` (if no required key remains).
- \`{}\` is legal only when nothing is required; \`}\` straight after \`,\` (a trailing comma) never is. Nothing may follow the closing \`}\`.

A suggested state is \`{ p, last, cur, text, comma, sub }\`: the phase (\`'open' | 'key' | 'value' | 'done'\`), the index of the last property written, the property being written, the key text typed so far, whether you just read a comma, and the value machine's own state. Keep it plain data; step 5 will use \`JSON.stringify(state)\` as a cache key.
`,
      predict: { question: 'The text is {"age":4 and the next character is "2". Which machine decides, and what happens on "," instead?', answer: 'The integer machine accepts "2", so the object machine just stores the new sub-state. On "," the integer machine returns null; because "4" is a complete integer, the object machine takes the comma itself and moves to the key phase. Delegate first, then handle structure.' },
      hints: [
        'The object machine is a supervisor. It handles only "{", the key text, "," and "}"; every character inside a value goes to that property\'s machine, whose state you store in the object state.',
        'Precompute keys, value machines and key literals. Write two helpers: candidates(last), the later keys up to and including the first required one, and canClose(last), true if no required key comes after last. Phases: open → key (match the typed text against the candidate literals; an exact match switches to value) → value → key or done. Record whether you came from a comma so a trailing "}" is refused.',
        'The value phase:\n```js\nif (p === \'value\') {\n  const inner = machines[cur].step(sub, ch);\n  if (inner !== null) return { ...state, sub: inner };\n  if (!machines[cur].accepts(sub)) return null;\n  if (ch === \',\' && candidates(cur).length > 0) return { p: \'key\', last: cur, cur: -1, text: \'\', comma: true, sub: null };\n  if (ch === \'}\' && /* … */) return { p: \'done\', /* … */ };\n  return null;\n}\n```',
      ],
    },
    {
      id: 'mask',
      title: 'From characters to a token mask',
      instructions: `
Implement \`maskForState(machine, state, vocab, eos)\`: a \`Uint8Array\` of length \`vocab.length\` where \`mask[id] = 1\` if the token may come next.

- A token \`vocab[id]\` is allowed if **every one of its characters** keeps the machine alive from \`state\`. \`advance\` does this for you.
- The EOS token (\`id === eos\`) is allowed exactly when \`machine.accepts(state)\`. Its string does not matter.
- Empty strings (special tokens with no text) are never allowed: they advance nothing, so a sampler could pick them forever.

The tests use a vocabulary with every printable ASCII character plus tokens that span JSON syntax (\`{"\`, \`",\`, \`"}\`, \`tru\`, \`"gold"\`, \`0,\`) and check that at least one token is allowed at every state along random walks. The demo uses the 256-token BPE vocabulary of the lab checkpoint plus two appended tokens, \`{\` and \`}\`, which the story tokenizer never learned.
`,
      hints: [
        'You already have `advance(machine, state, text)`. What does its return value tell you about a token?',
        'Loop over every id. Handle eos first (allowed iff the machine accepts), skip empty strings (leave 0), and for everything else the entry is 1 exactly when advancing by the whole token string does not return null.',
        '```js\nconst mask = new Uint8Array(vocab.length);\nfor (let id = 0; id < vocab.length; id++) {\n  if (id === eos) { /* allowed only when the machine accepts */ continue; }\n  if (!vocab[id]) continue;\n  /* 1 if the whole token string keeps the machine alive from state */\n}\nreturn mask;\n```',
      ],
    },
    {
      id: 'sample',
      title: 'Constrained sampling',
      instructions: `
Two functions.

\`applyMask(logits, mask)\`: a new \`Float32Array\` in which every token with \`mask[i] === 0\` has logit \`-Infinity\` and the rest are unchanged. Do not modify the input.

\`constrainedGenerate(model, machine, vocab, { eos, next, temperature = 1, maxTokens = 256, masker = null })\`:

1. Start from \`machine.start\` with no ids.
2. Each step: \`logits = model(ids)\` (the model sees the ids generated so far), mask them with \`masker(state)\` if one is given, otherwise \`maskForState\`, and draw with \`sample(masked, { temperature, next })\` from \`lib/sampling.js\`.
3. If the id is \`eos\`, return \`{ text, ids, finished: true }\`. Otherwise advance the state by \`vocab[id]\`, push the id and append its text.
4. If \`maxTokens\` runs out, return \`{ text, ids, finished: false }\`.

The tests run a model with random logits 100 times and parse every output, use a hostile model that prefers \`}\` and EOS everywhere, check that each model call receives exactly the ids generated before it, and draw 2,000 times from a two-token grammar to check that the frequencies match the softmax of the masked logits at temperature 1 and 0.5 (so argmax, or a dropped \`temperature\`, fails).
`,
      predict: { question: 'The hostile test model gives "}" logit 10, EOS 9, " the" 8 and "0," 6, and everything else -5. What do you expect the name and the age to be?', answer: 'Inside the name string "}", " the" and "0," are all legal, so the name is made of them, mostly "}" characters. After "age": the only one of the favourites that is legal is "0,", so the age is 0. The mask removes illegal choices; among the legal ones the model\'s preferences still decide.' },
      hints: [
        'Masking is just "logit becomes -Infinity". The softmax inside `sample` turns that into probability 0, so you never need to renormalise yourself.',
        'Choose the mask function once: `masker ?? (s => maskForState(machine, s, vocab, eos))`. Then loop up to maxTokens: logits, mask, sample, check eos, advance, push.',
        '```js\nfor (let t = 0; t < maxTokens; t++) {\n  const id = sample(applyMask(model(ids), maskOf(state)), { temperature, next });\n  /* … stop on eos … */\n  state = advance(machine, state, vocab[id]);\n  ids.push(id); text += vocab[id];\n}\n```',
      ],
    },
    {
      id: 'cache',
      title: 'Masks cached by state',
      instructions: `
A mask depends only on the automaton state. Implement \`cachedMasker(machine, vocab, eos)\`, which returns a function \`masker(state) → mask\` that:

- keys a \`Map\` on \`JSON.stringify(state)\`, so two different prefixes that reach equal states share one entry (\`{"name":"a\` and \`{"name":"b\` do);
- on a miss computes \`maskForState\` and stores it; on a hit returns the stored mask without stepping the machine at all (a test counts \`step\` calls);
- carries \`masker.stats = { calls, misses, tokenScans }\`: every call increments \`calls\`; a miss increments \`misses\` and adds \`vocab.length\` to \`tokenScans\`.

Pass it as \`masker\` to \`constrainedGenerate\`. The demo reports lookups, misses and the token scans the cache avoided, \`(calls − misses) × V\`. This is the idea behind Outlines' precomputed index, done lazily.
`,
      predict: { question: 'The demo runs 120 constrained generations of about 55 tokens each with a 258-token vocabulary. Roughly how many distinct states will the cache see: about 50, about 5,000, or about 500,000?', answer: 'Under a hundred (88 in the reference run), against about 7,600 lookups (the demo also replays each checkpoint run through the masker to measure probability mass): a hit rate near 99%. String states carry their length, integer states their digit count, and everything else is a few key and literal positions. Precomputing all of them ahead of time is feasible, which is exactly what compile-ahead engines do.' },
      hints: [
        'What is a mask a function of? Not the model, not the exact prefix: only the state. Equal states give equal masks.',
        'Create a Map and a stats object once, outside the returned function. Inside: count the call, build the key with JSON.stringify(state), and on a miss count it, add vocab.length to tokenScans, compute and store. Attach stats to the function object before returning it.',
        '```js\nconst masker = (state) => {\n  stats.calls++;\n  const key = JSON.stringify(state);\n  let mask = cache.get(key);\n  if (mask === undefined) { /* … count the miss and the scans, compute, store … */ }\n  return mask;\n};\nmasker.stats = stats;\n```',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague how masking logits can guarantee schema-valid JSON from a model that was never trained on JSON, and name the two conditions (about which tokens survive, and about EOS) that the guarantee needs.',
    'In the demo the checkpoint put under 1% of its probability on legal tokens at some positions. What does the output at those positions tell you about the model, and why is "it parses" not the same as "it is right"?',
    'Why can one BPE token be legal in one state and illegal in another even though its text never changes? Use a token such as `",` or `"}` in your answer.',
  ],
  stretch: [
    'Add arrays (`{ type: "array", items: spec, maxItems }`) and optional whitespace. You now need a stack of open containers: a pushdown automaton, which is how XGrammar and llguidance handle nested JSON.',
    'Implement jump-forward decoding: when the mask allows exactly one token, or the machine has a single forced continuation such as `,"member":`, append it without calling the model. SGLang does this with a compressed finite-state machine and reports large speedups on JSON-heavy workloads.',
    'Precompute masks for every reachable state before generation starts, as Outlines does: breadth-first search from `machine.start` over single characters, then build the state → allowed-tokens index. Compare the one-off build time with the cached masker.',
    'Combine your masker with speculative decoding from module 18: advance the automaton along the draft, mask both draft and target distributions at each position, and check that the acceptance rule still preserves the masked target distribution. Serving engines such as vLLM and SGLang face exactly this when both features are switched on.',
  ],
  timeouts: { tests: 20000, demo: 90000 },
};
