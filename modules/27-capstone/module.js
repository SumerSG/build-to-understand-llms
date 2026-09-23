export default {
  id: '27-capstone',
  title: 'Capstone: chat with your own model',
  track: 'capstone',
  minutes: 120,
  threshold: 'A chat product is the composition of every layer built in this lab; nothing new appears at the top, only the plumbing that connects tokenizer, model, KV cache, decoding policy, tools and context budget.',
  goal: 'Tokenizer + model + KV cache + sampler + harness wired together into a chat that runs in the page on a model trained in this lab: `chat(state, userMessage)` returns the reply plus per-turn stats (prompt, cached, prefilled and generated tokens, tool calls, latency), and the goal demo holds a four-turn conversation with one calculator call on the lab checkpoint.',
  prereqs: ['03-tokenizer', '10-sft', '14-decoding', '15-kv-cache', '17-prefix-caching', '20-agent-loop', '21-context'],
  recall: [
    { q: 'In module 15, why is it valid to keep the keys and values of past tokens instead of recomputing them for every new token?',
      options: ['They are approximately constant, and the error is small', 'Causal attention means a past token\'s key and value depend only on it and the tokens before it, so nothing later can change them', 'The cache stores the logits, which never change'], answer: 1,
      why: 'That same fact is what lets you keep the cache across chat turns: whatever prefix the new prompt shares with the cached ids has exactly the same keys and values.' },
    { q: 'In `sample` from lib/sampling.js (module 14), in what order are the logit processors applied?',
      options: ['top-k, top-p, temperature, min-p', 'temperature, top-k, top-p, min-p', 'min-p, top-p, top-k, temperature'], answer: 1,
      why: 'Temperature reshapes the distribution that the truncation rules then measure. You will call the same `sample` from your decode loop, and temperature 0 is its greedy limit.' },
    { q: 'Module 17 measured prefix caching with a timestamp injected into every request. Where did the timestamp hurt the hit rate least?',
      options: ['At the start of the system prompt', 'At the end, after the user turn', 'It makes no difference'], answer: 1,
      why: 'Reuse stops at the first token that differs. In this module the context budget plays the timestamp\'s role: dropping an old turn changes everything after the system prompt.' },
    { q: 'In module 20\'s `ToolRegistry.call`, what happens when the model asks for a tool that does not exist?',
      options: ['It throws and the agent loop stops with "error"', 'It returns an "Error: unknown tool …" string that goes back to the model', 'It silently returns an empty string'], answer: 1,
      why: 'Tool failures are data the model can read and recover from, not crashes. Your `runToolCalls` relies on this: it never needs a try/catch.' },
    { q: 'Module 21\'s `dropOldest` removes old messages to meet a budget. Which messages does it never remove?',
      options: ['Tool results', 'System messages', 'The first user message'], answer: 1,
      why: 'The system prompt defines the assistant. Your `buildPrompt` keeps that rule and adds one: it removes whole turns, so a tool result never loses the call that produced it.' },
  ],
  review: [
    { q: 'On turn 2, what lets your pipeline skip most of the prompt\'s prefill?',
      options: ['The prompt of turn 2 begins with the exact token ids already in the cache (turn 1\'s prompt and reply)', 'The model remembers turn 1 in its weights', 'The sampler caches its random numbers'], answer: 0,
      why: 'Reuse is the longest common prefix of the cached ids and the new ids. In the demo turn 2 reused 29 of its 43 prompt tokens; the rest is the new user message.' },
    { q: 'The whole prompt is already in the cache. Why does `syncCache` still run one token?',
      options: ['To warm up the model', 'Because the output of a prefill is the logits after the last token, and those are not stored in a KV cache', 'Because the cache is always one token short'], answer: 1,
      why: 'A KV cache stores keys and values, not logits. Cutting back by one and re-running the last token produces the logits the decoder needs, for the cost of a single decode step.' },
    { q: 'Why is the prompt budget `blockSize − maxNewTokens` rather than `blockSize`?',
      options: ['To leave room for the reply: with learned position embeddings there is no position after blockSize', 'To make prompts shorter and faster', 'Because the tokenizer adds hidden tokens'], answer: 0,
      why: 'A prompt that filled the window would leave the reply no positions at all; your decode loop would stop at once with "context".' },
    { q: 'By turn 4 the budget has dropped every earlier turn (8 messages). How much of turn 4\'s prompt can come from the cache?',
      options: ['Almost all of it, since the text is the same', 'Only the system prompt and the marker after it', 'None of it'], answer: 1,
      why: 'Removing a turn shifts every later token to a new position. The shared prefix ends right after the system prompt: 10 tokens in the demo.' },
    { q: 'Why does `buildPrompt` drop a whole turn instead of one message at a time?',
      options: ['It is faster', 'A tool result or reply without the question and call that produced it misleads the model, and real APIs reject a tool result with no matching call', 'Messages cannot be removed individually from a list'], answer: 1,
      why: 'Context management is about keeping the context coherent as well as short.' },
  ],
  concept: `
## Where each module sits on the request path

A message sent to a production assistant passes through a fixed sequence of stages. Each is listed with the module where you built it:

1. **Client → API gateway.** Authentication, rate limits, billing and safety checks.
2. **Chat template and context manager.** The message list becomes one string with role markers, trimmed to a token budget (modules 10 and 21).
3. **Tokenizer.** String becomes ids (module 03).
4. **Router.** Picks a replica, preferably one whose cache already holds this conversation's prefix (modules 17 and 26).
5. **Prefill.** The prompt runs through the model once, filling the KV cache (modules 06 and 15). Only the part the cache does not already hold needs to run.
6. **Decode.** One token at a time: logits, then the decoding policy (temperature, top-k, top-p), then a sampled id, then that id goes back into the cache (module 14). The loop ends at a stop token or a length limit.
7. **Harness.** If the reply contains a tool call, the harness runs the tool, appends the result and goes back to step 2 (module 20).
8. **Detokenize and stream** tokens to the client.

You write stages 2, 5, 6 and 7 and connect them. None is new; the work is in the joins: what one stage hands to the next, and which invariant must hold between them.

## The invariant that makes turns cheap

A conversation's state is its message list (\`history\`) plus a **session**: a KV cache and the ids it holds, with \`session.cache.length === session.ids.length\` at all times. On a new turn you render and encode the whole history again, compare the new ids with \`session.ids\`, keep the longest common prefix, cut the cache back to it and prefill only the rest. Module 17 made this argument across requests; here it holds across turns.

:::predict
Turn 1 is 19 prompt tokens plus a 10-token reply. Turn 2 appends \`<|end|>\`, a new user message and \`<|assistant|>\`, 43 tokens in all. How many of them can come from the cache?
---
The demo measures 29: all 19 prompt tokens and all 10 reply tokens, because re-encoding the reply text gave back exactly the sampled ids. That is not guaranteed: a sampler can emit a token sequence BPE would never produce itself, which is one reason SGLang's radix cache keys on token ids rather than text. The 14 new tokens are prefilled.
:::

## The joins

- **Template.** The lab's BPE vocabulary has never seen \`<\` or \`|\`, so \`<|user|>\` would cost 7 tokens, four of them \`<|unk|>\`, and every marker would share the same four. \`chatTokenizer\` registers the markers, the tool tags and \`+ * /\` as single tokens, and \`extendVocab\` gives the model new embedding rows for them, as module 10 did, but untrained.
- **Budget.** The prompt may use \`blockSize − maxNewTokens\` tokens, which leaves room for the reply. Over budget, the oldest whole turn goes, so a tool result never loses its call.
- **Stop.** Every role marker and end-of-text stop the reply and are kept neither in it nor in the cache; the next turn's template writes \`<|end|>\` back and prefills it.
- **Tools.** \`<tool_call>calc(17*23)</tool_call>\` is found, run through module 20's registry and appended as a \`tool\` message; then the model runs again with the result in context.

:::predict
By turn 4 the budget has removed every earlier turn: turns 1, 2 and 3, 8 messages in all (the calculator turn alone is 4). How many tokens of turn 4's prompt are cache hits?
---
10: the system prompt (9 tokens) and the \`<|user|>\` marker after it. Everything after the system prompt now sits at different positions, so it all has to be prefilled again. This is why production systems summarise or drop context in large, infrequent steps. Trimming a little on every turn would throw away the cache every time.
:::

## What your model can and cannot do

The checkpoint is the 2-layer, 64-dimension model pre-trained in this lab on a 43 KB corpus with a 64-token window. It was never fine-tuned on the chat markers or on tool calls, so its replies are corpus-like fragments, not answers, and they end at the length limit because it almost never produces \`<|end|>\`. The calculator turn uses a **prefill**: the harness writes the start of the assistant's reply (here, the whole call), much as Anthropic's API has let a client prefill the assistant turn, or as a forced \`tool_choice\` does. That exercises every join; it does not show that the model can decide to call a tool, which module 10's fine-tuning would have to teach.

## Where the toy differs from production

Production context windows are 128K tokens (Llama 3.1) rather than 64. Prefill is one batched pass, not a loop of decode steps. KV lives in paged blocks shared across requests (vLLM's default block is 16 tokens), not in one growing array per conversation. Chat templates are Jinja templates shipped with the tokenizer (Hugging Face's \`chat_template\`). Tool calls are JSON (OpenAI function calling) or Python-style calls (Llama 3.2), and are often enforced with constrained decoding (module 22). This pipeline also leaves out whole stages that products need: safety classifiers such as Llama Guard on input and output, token streaming over server-sent events, tracing and latency metrics (time to first token, time per output token), billing per token, and retries. Each would be another stage on the same path.
`,
  steps: [
    {
      id: 'prompt',
      title: 'Prompt assembly under a token budget',
      instructions: `
Implement \`buildPrompt(tokenizer, history, { budget, prefill = '' })\` → \`{ ids, text, messages, dropped }\`.

The worked function \`renderChat(messages, prefill)\` already renders a message list in the template and opens the assistant turn. Your job is the budget (module 21):

1. Render a **copy** of \`history\` with \`renderChat(messages, prefill)\` and encode it. The prefill counts: its tokens take cache positions like any other. If \`ids.length <= budget\`, you are done.
2. Otherwise drop the **oldest turn**: the first non-system message, which is a user message, together with every message after it up to (not including) the next \`user\` message. That takes the assistant reply, any tool calls and any tool results with it. Re-render, re-encode, repeat.
3. System messages are never dropped, and neither is the current (last) turn. If only those are left and the prompt is still too long, \`throw\`.

\`dropped\` counts the **messages** removed; \`messages\` is the list that was kept. Never modify \`history\`: it is the transcript, and dropping only affects what the model sees.

Why whole turns? A tool result without the call that produced it, or an answer without its question, misleads the model. Real APIs refuse a tool result that has no matching call.
`,
      predict: { question: 'History: system, then three turns of about 12 tokens each, then a new user message. The budget fits the system prompt plus two turns. Which messages survive?', answer: 'The system prompt, the most recent complete turn and the new user message. The two oldest turns are dropped whole, because dropping only the oldest one still leaves the prompt over budget.' },
      hints: [
        'You never edit the rendered string. You edit a copy of the message list and render it again. What is the smallest unit you can remove without leaving a reply that answers nothing?',
        'Loop while `ids.length > budget`: find `start`, the index of the first non-system message; advance `end` from `start + 1` until `messages[end].role === "user"` or the list ends. If `end` reached the end, the only turn left is the current one, so throw. Otherwise `splice(start, end - start)`, add `end - start` to `dropped`, re-render and re-encode.',
        '```js\nwhile (ids.length > budget) {\n  const start = messages.findIndex((m) => m.role !== \'system\');\n  let end = start + 1;\n  while (/* … not past the end, and not at the next user message … */) end++;\n  if (/* … no non-system message, or the turn runs to the end of the list … */) throw new Error(`…`);\n  messages.splice(start, end - start);\n  dropped += end - start;\n  // re-render with renderChat(messages, prefill) and re-encode\n}\n```',
      ],
    },
    {
      id: 'prefix',
      title: 'Prefill with prefix reuse across turns',
      instructions: `
Two functions.

\`truncateCache(cache, n)\`: keep positions \`0 … n−1\` of every layer's keys and values (\`cache.k[layer]\` and \`cache.v[layer]\` are raw \`[H, T, dh]\` tensors; \`ops.slice(t, 1, 0, n)\` cuts axis 1), set \`cache.length = n\`, and return the same cache object.

\`syncCache(model, session, ids)\` → \`{ logits, reused, prefilled }\`: make the session hold exactly \`ids\`.

1. \`reused = sharedPrefixLength(session.ids, ids)\` (worked for you).
2. If \`reused === ids.length\`, the whole prompt is cached. A KV cache holds keys and values, not logits, so back off by one: \`reused = ids.length − 1\`.
3. \`truncateCache(session.cache, reused)\`. Everything after the first differing token is stale, even when later tokens happen to match.
4. \`logits = prefill(model, session.cache, ids.slice(reused))\` from \`lib/infer.js\`.
5. \`session.ids = ids.slice()\` (a copy), and \`prefilled = ids.length − reused\`.

Throw if \`ids\` is empty or longer than \`model.config.blockSize\`. The tests compare your logits and cache with a fresh prefill of the same ids. One test also tampers with the stored values, which a pipeline that quietly recomputes everything would never notice.
`,
      predict: { question: 'The cache holds A+B. The new prompt is A+C, where C happens to end with the same three tokens as B. How many tokens are reused?', answer: 'Exactly |A|. The key and value at each position depend on every earlier token, so after the first difference nothing matches, even tokens that look identical. The cache is cut back to |A| and all of C is prefilled.' },
      hints: [
        'The cache is valid for exactly the tokens on which `session.ids` and `ids` agree from the start. Everything after the first disagreement has to go before you prefill anything.',
        'truncateCache: loop over layers, replacing `cache.k[l]` and `cache.v[l]` by their slice on axis 1, then set `cache.length`. syncCache: shared prefix → back off by one if it covers everything → truncate → prefill the suffix → store a copy of ids.',
        '```js\nlet reused = sharedPrefixLength(session.ids, ids);\n// … the whole prompt is cached: keep one token to run ...\ntruncateCache(session.cache, reused);\nconst logits = /* prefill only what the cache does not hold */;\nsession.ids = ids.slice();\nreturn { logits, reused, prefilled: ids.length - reused };\n```',
      ],
    },
    {
      id: 'decode',
      title: 'Decoding until a stop token',
      instructions: `
Implement \`decodeReply(model, tokenizer, session, logits, { maxNewTokens = 16, stopIds = [], next, temperature = 1, topK = 0, topP = 1 })\` → \`{ text, ids, stopReason }\`.

This is module 14's generation loop, attached to the session instead of to a fresh cache. Loop:

1. If \`ids.length >= maxNewTokens\`: stop with \`'length'\`.
2. If \`session.cache.length >= model.config.blockSize\`: stop with \`'context'\`. There is no position embedding left, and \`forwardStep\` would throw.
3. \`id = sample(logits, { temperature, topK, topP, next })\` from \`lib/sampling.js\`, called exactly once per token so a seed reproduces the reply.
4. If \`stopIds\` includes \`id\`: stop with \`'stop'\`. The stop token is **not** kept: it goes neither in \`ids\` nor in the cache.
5. Otherwise push \`id\` onto \`ids\` and onto \`session.ids\`, and \`logits = forwardStep(model, session.cache, id)\`.

\`text = tokenizer.decode(ids)\`. At the end \`session.cache.length === session.ids.length\` still holds. That is what lets the next turn reuse the reply you just generated.
`,
      predict: { question: 'The chat state stops on every role marker (`<|user|>`, `<|system|>`, …), not only on `<|end|>`. What would go wrong without the extra stop ids?', answer: 'An untrained or confused model could write `<|user|>` in the middle of its reply and then make up the user\'s next message. The next turn\'s template would render that as a real turn boundary, so the model would be putting words in the user\'s mouth. Llama 3.1\'s template also lists several stop tokens (`<|eot_id|>`, `<|eom_id|>`, `<|end_of_text|>`) for this reason.' },
      hints: [
        'Three things can end a reply, and a sampled token can go to three places. Decide the order: which checks come before sampling, which after, and when does a token enter the cache?',
        'Before sampling, check the budget (`length`), then the window (`context`). Sample once. A stop id breaks out without being stored. Any other id is appended to both `ids` and `session.ids` and fed through `forwardStep`, which returns the logits for the next iteration.',
        '```js\nwhile (true) {\n  if (ids.length >= maxNewTokens) { stopReason = \'length\'; break; }\n  if (/* … the cache is full … */) { stopReason = \'context\'; break; }\n  const id = sample(logits, { temperature, topK, topP, next });\n  if (stopIds.includes(id)) { stopReason = \'stop\'; break; }\n  ids.push(id);\n  session.ids.push(id);\n  // … run id through the cache to get the next logits\n}\n```',
      ],
    },
    {
      id: 'tools',
      title: 'Tool calls through the registry',
      instructions: `
Two functions for the tool format \`<tool_call>name(argument)</tool_call>\`. Module 20 used JSON. With a 64-token window and a vocabulary without \`{\` or \`}\`, the capstone uses a Python-style call instead, close to Llama 3.2's format for lightweight models.

\`findToolCalls(text)\` → \`[{ name, input }]\`, in order. Scan for \`TOOL_OPEN\`, then the next \`TOOL_CLOSE\` after it. If there is no closing tag, stop scanning. Trim the body between them and match \`name(argument)\`: an identifier, optional spaces, \`(\`, anything, \`)\` at the very end. \`input\` is everything between the **first** \`(\` and the **last** \`)\`, trimmed, so \`calc((2+3)*4)\` gives \`(2+3)*4\`. Skip a body that does not match.

\`runToolCalls(text, tools)\` (async) → one \`{ role: 'tool', name, content }\` per call, where \`content = await tools.call(name, { input })\`. \`makeTools()\` (worked) registers the calculator \`calc\` in module 20's \`ToolRegistry\`. Unknown tools, bad input and division by zero all come back as \`'Error: …'\` strings, so the model can read what went wrong and a bad call cannot crash the chat.
`,
      hints: [
        'This has two layers. First find the text between the tags; `indexOf` with a moving start position is enough. Then split the name from the argument; one regular expression anchored at both ends is enough.',
        'Keep `at = 0`. Loop: `open = text.indexOf(TOOL_OPEN, at)`, `close = text.indexOf(TOOL_CLOSE, open + TOOL_OPEN.length)`; break if either is -1. Take the trimmed body and move `at` past the close tag. Then one regular expression anchored with `^` and `$`: an identifier, optional spaces, `(`, a greedy group for the argument, `)`. Because the group is greedy and `)` must be the last character, the argument runs to the LAST parenthesis.',
        '```js\nconst match = /^([A-Za-z_][A-Za-z0-9_]*)\\s*\\(([\\s\\S]*)\\)$/.exec(body);\nif (match) calls.push({ name: match[1], input: match[2].trim() });\n// runToolCalls:\nfor (const call of findToolCalls(text)) {\n  const content = /* … ask the registry … */;\n  messages.push({ role: \'tool\', name: call.name, content });\n}\n```',
      ],
    },
    {
      id: 'chat',
      title: 'The chat function',
      instructions: `
Implement \`chat(state, userMessage, { maxNewTokens = 12, prefill = '', maxToolRounds = 2, next, temperature, topK, topP })\` → \`{ reply, stats }\`. \`state\` comes from the worked \`createChat({ model, tokenizer, system })\`: \`{ model, tokenizer, tools, history, session, stopIds }\`.

Push \`{ role: 'user', content: userMessage }\` onto \`state.history\`, then run **rounds**:

1. \`buildPrompt(tokenizer, state.history, { budget: blockSize − maxNewTokens, prefill })\`.
2. \`syncCache(model, state.session, prompt.ids)\`.
3. If \`findToolCalls(prefill)\` is non-empty, the prefill already holds a complete call (a forced tool call), so generate nothing: \`{ text: '', ids: [], stopReason: 'forced' }\`. Otherwise \`decodeReply(…, { maxNewTokens, stopIds: state.stopIds, next, temperature, topK, topP })\`.
4. \`reply = prefill + text\`. Push it as an \`assistant\` message and record the round's stats.
5. Rounds are numbered from 0. Unless this was round \`maxToolRounds\` (so \`maxToolRounds: 0\` means one round and no tool is run), \`runToolCalls(reply, state.tools)\`. If that returned messages, push them, add their count to \`toolCalls\`, set \`prefill = ''\` and run another round. Otherwise stop.

Stats: \`stats.rounds[i] = { promptTokens, cachedTokens (= reused), prefilledTokens, generatedTokens, dropped, stopReason }\`. The totals \`promptTokens\`, \`cachedTokens\`, \`prefilledTokens\` and \`generatedTokens\` are sums over rounds; \`dropped\` is the largest per-round value; \`toolCalls\` is the number of calls executed; \`stopReason\` is the last round's; \`ms\` is the wall-clock time of the turn (\`now()\` from \`lib/util.js\`). \`reply\` is the last assistant message.

The worked \`renderTranscript(history)\` turns the transcript into markdown for the demo.
`,
      predict: { question: 'In the calculator turn, round 1 has a 36-token prompt ending in the forced call. How many tokens of round 2\'s prompt are cache hits?', answer: 'All 36. Round 2\'s prompt is round 1\'s prompt, then `<|end|>`, the tool message and `<|assistant|>`. Only those 7 new tokens are prefilled. A tool round trip costs one short prefill, not a second pass over the conversation.' },
      hints: [
        '`chat` adds no new mechanism. Before coding, write down the order of calls: which function produces what the next one consumes, and which values go into the stats?',
        'Push the user message. `for (round = 0; round <= maxToolRounds; round++)`: build the prompt, sync the cache, decode (unless the prefill is a complete call), push the assistant message, push a stats row, then break if this is the last round or there are no tool results. Otherwise push the tool messages, count them, clear the prefill and loop. Sum the rounds at the end.',
        '```js\nconst prompt = buildPrompt(tokenizer, state.history, { budget: model.config.blockSize - maxNewTokens, prefill: forced });\nconst { logits, reused, prefilled } = syncCache(model, state.session, prompt.ids);\nconst gen = findToolCalls(forced).length\n  ? { text: \'\', ids: [], stopReason: \'forced\' }\n  : /* … decodeReply with state.stopIds and the sampling options … */;\nreply = forced + gen.text;\nstate.history.push({ role: \'assistant\', content: reply });\nrounds.push({ promptTokens: prompt.ids.length, cachedTokens: reused, /* … */ });\n// … tool results → another round, or break\n```',
      ],
    },
  ],
  reflection: [
    'Name every stage of the request path that ran in your demo and the module where you built it. Which stage has the fewest lines of code in `chat`, and which invariant does that stage depend on?',
    'Turn 2 reused most of its prompt and turn 4 reused only the system prompt. Explain the difference in terms of token positions. Then propose a context policy that keeps the reuse high, and say what it costs.',
    'Your model answers with corpus-like fragments even though every join works. Which layer would you change to get helpful answers, and what evidence from this module and module 10 supports that choice?',
  ],
  stretch: [
    'Wire `chat` into the page: the lab\'s Chat playground (`app/chat-worker.js`) does its own prefix reuse. Replace its `generate` with your `createChat`/`chat`, stream each token back with `postMessage` as `decodeReply` samples it, and show the stats under every reply. This is the streaming stage that production servers such as vLLM\'s OpenAI-compatible server run over server-sent events.',
    'Fine-tune the checkpoint on the chat template with module 10 (add a few calculator examples in the `<tool_call>calc(…)</tool_call>` format), save it, load it in the demo, and count how many replies now end on `<|end|>` and how many turns call the tool without a prefill.',
    'Replace turn dropping with module 21\'s compaction: summarise old turns into one pinned message only when the budget is exceeded by a margin, and measure the cached-token fraction over a 12-turn conversation. This is the trade-off long-running agents such as Claude Code make.',
    'Keep the conversation as token ids instead of re-encoding text every turn, as SGLang\'s RadixAttention does, and find a reply for which re-tokenizing the text would have broken the prefix match.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
