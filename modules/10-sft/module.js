export default {
  id: '10-sft',
  title: 'Supervised fine-tuning',
  track: 'posttraining',
  minutes: 90,
  threshold: 'SFT is pre-training on a different distribution with a mask: the model keeps predicting the next token, but only the assistant\'s tokens count, so what you mask out decides what the model learns to imitate.',
  goal: 'A chat-formatted SFT trainer with assistant-only loss masking that turns the pre-trained model into one that follows the chat template: it answers after `<|assistant|>` and stops with `<|end|>`.',
  prereqs: ['03-tokenizer', '06-transformer', '07-pretraining'],
  recall: [
    { q: 'In module 03, how does `BPETokenizer.encode` treat a special token such as `<|endoftext|>` that appears inside the text?', options: ['It is split into characters and merged like any word', 'It is matched before pre-tokenisation and emitted as one id', 'It is dropped'], answer: 1,
      why: 'Specials are matched first (longest first) and become a single id. That is why the chat markers must be registered as specials: `<|user|>` spelled out as ordinary text would be a dozen sub-word ids the model could never learn to emit in one step.' },
    { q: 'In module 06, the language-model head is tied to `wte`, so the logits are `x · wteᵀ`. If `wte` gains four rows, the logits…', options: ['Keep their shape; the new rows are ignored', 'Gain four entries per position, scored against the new rows', 'Become invalid until the head is retrained'], answer: 1,
      why: 'Tying means the same table reads tokens in and scores them out. Growing it by four rows gives the model four new tokens to read and four new logits to emit, with no other parameter changed.' },
    { q: 'In module 07, `getBatch` returns `y` equal to `x` shifted left by one token because…', options: ['It halves the memory of the batch', 'The causal transformer predicts every position at once, so the target at position t is the token at t + 1', 'The model needs two copies to train'], answer: 1,
      why: 'One forward pass over a window gives one next-token prediction per position. The mask in this module says which of those predictions count, so it is sliced exactly as `y` is.' },
    { q: 'In module 02, if a scalar loss is `(a * c).sum()` and `c` is a constant tensor, the gradient of the loss with respect to `a` is…', options: ['`c`, entry by entry', 'All ones', 'Zero everywhere'], answer: 0,
      why: 'd(a·c)/da = c. Where `c` is 0 the gradient is exactly 0, which is how a mask silences positions: no clever indexing, just multiplication by a constant.' },
    { q: 'Module 07 starts every training step with `optimizer.zeroGrad()` (or ends it with one). What happens if you forget?', options: ['Nothing; `backward` overwrites the gradients', 'Gradients from the previous batch are added to the new ones', 'The optimizer throws'], answer: 1,
      why: '`backward` accumulates with `+=`. Stale gradients make the update a blend of two batches and the loss curve turns noisy without any error message.' },
  ],
  review: [
    { q: 'In `tokenizeExample`, which tokens get mask 1?', options: ['Every token after `<|user|>`', 'The response tokens and the `<|end|>` that follows them', 'Only the response tokens, not the end marker'], answer: 1,
      why: 'The response is what the assistant should say, and `<|end|>` is how it stops. A model trained without the end marker in the mask never learns to finish a turn and rambles until `maxNewTokens`.' },
    { q: 'The masked cross-entropy divides the sum of per-position losses by…', options: ['B · T, the number of positions', 'The number of masked-in positions, `sum(mask)`', 'The batch size B'], answer: 1,
      why: 'Dividing by B · T would make a batch with few assistant tokens look easy and shrink its gradient for no reason. Dividing by `sum(mask)` gives a mean over the tokens that actually count, in nats per assistant token, comparable across batches.' },
    { q: 'Why does `resizeEmbeddings` fill the new rows with the mean of the old rows rather than fresh random values?', options: ['It is faster', 'A mean row is a typical trained embedding, so the new tokens start where the model already knows how to read and score them', 'Random rows would make the tokenizer fail'], answer: 1,
      why: 'Hewitt (2021) showed that random new rows sit far from the trained distribution and get anomalous logits, and recommended initialising them at the average of the existing embeddings. Hugging Face\'s `resize_token_embeddings` does a close variant by default (`mean_resizing=True`: new rows are sampled from a normal with the old rows\' mean and covariance).' },
    { q: 'After `shift`, the mask is `mask.slice(1)`. Why not `mask.slice(0, -1)`?', options: ['Either works; they have the same length', 'The mask says which targets count, and the targets are `y = ids.slice(1)`', 'Because the first token is always a marker'], answer: 1,
      why: 'Both slices have the same length, which is exactly why this off-by-one is easy to miss. Aligned with `x`, the mask would be 1 where the INPUT is a response token, so the first counted prediction is the second response token (made from the first), and the most important one, the first response token predicted from `<|assistant|>`, is silently dropped.' },
    { q: 'SFT runs use a learning rate well below the pre-training peak (Llama 2 7B: approximately 3e-4 for pre-training, 2e-5 for SFT) because…', options: ['The SFT dataset is small, so the loss is cheap to compute', 'The weights already encode the language; large steps on a narrow dataset overwrite that knowledge (catastrophic forgetting)', 'AdamW is unstable at high learning rates'], answer: 1,
      why: 'Fine-tuning moves an already good model a short distance. A small learning rate and few epochs keep the base capabilities while the format is learned.' },
  ],
  concept: `
## The same loop, a different distribution

The checkpoint you load in this module has never seen a chat. It has read the lab corpus and learned to continue it. Supervised fine-tuning (SFT) does not add a new objective or a new architecture: it runs the module 07 loop again, on a different token stream, with one change to the loss. Everything you need is already built; this module is about **which tokens you train on**.

Three facts make SFT work with far less data than pre-training.

1. **The base model already knows the language.** InstructGPT (Ouyang et al. 2022) used roughly 13,000 demonstrations on top of GPT-3; LIMA (Zhou et al. 2023) fine-tuned a 65B model on 1,000. The format and the role are cheap to learn; the knowledge underneath came from pre-training and is what you must not destroy.
2. **A chat template turns a conversation into one string** the model can continue. This lab uses \`<|user|>…<|end|><|assistant|>…<|end|>\`. Llama 3 uses \`<|start_header_id|>user<|end_header_id|>\` … \`<|eot_id|>\`; ChatML (OpenAI, then Qwen and many others) uses \`<|im_start|>user\\n…<|im_end|>\`. Hugging Face ships each model's template as a Jinja string in \`tokenizer_config.json\` and \`apply_chat_template\` renders it. The markers are **special tokens**: single ids that ordinary text can never produce, appended to the vocabulary so no existing id moves.
3. **The loss is masked to the assistant's tokens.** For every position \`t\` the model predicts \`y[t]\` from \`x[0..t]\`, exactly as in pre-training; the mask \`m[t] ∈ {0, 1}\` says whether that prediction counts:

\`\`\`
loss = sum_t( m[t] · nll[t] ) / sum_t( m[t] )      nll[t] = -log softmax(logits[t])[y[t]]
\`\`\`

The division is by the **number of masked-in positions**, not by \`B · T\`, so the loss is nats per assistant token and a batch with short answers is not artificially easy.

:::predict
You load the checkpoint, add the four markers with mean-initialised embedding rows, and greedily decode from \`<|user|>Say hello.<|end|><|assistant|>\` **before** any fine-tuning. What comes out?
---
Corpus-like text with no relation to the prompt, and it never emits \`<|end|>\`. The four new rows are averages of trained rows, so the model reads the markers as a vague "typical token" and continues as it would continue anything. The demo shows this before/after.
:::

## Why mask at all?

Without the mask the model is trained to predict the user's words too. That wastes capacity on imitating prompts and, worse, teaches the model to *write questions*: greedy decoding after \`<|assistant|>\` often produces a new \`<|user|>\` turn. In PyTorch and Hugging Face code the mask is usually implemented by setting prompt labels to \`-100\`, the default \`ignore_index\` of \`CrossEntropyLoss\`; in TRL it is \`assistant_only_loss\`. Your version multiplies a one-hot pick tensor by the mask, which makes the gradient at a masked-out position exactly zero (module 02: \`d(a·c)/da = c\`); the tests check that changing prompt logits changes nothing.

The end marker is masked **in**. It is the one token that teaches the model to stop.

## Packing

Examples are 16–43 tokens; the window is 64. Padding every example to 64 would spend most of the compute on \`<|endoftext|>\`. Packing concatenates examples in order, separated by \`eos\`, until the next one would not fit, then pads only the tail of each window. The mask travels with each example, so nothing in a separator, in padding, or in the next example's prompt ever counts.

One honest caveat: your packed window has a single causal mask, so the second example can attend to the first. The correct fix is a block-diagonal attention mask with position ids restarting at each example (FlashAttention's \`varlen\` kernels; Hugging Face \`padding_free\` batching). Many training libraries simply let the contamination happen, as GPT-3-style pre-training does with documents separated only by \`eos\`, and the models cope because \`eos\` is a strong "start over" cue.

:::predict
The 66 \`INSTRUCTIONS\` pairs contain about 1,700 tokens, about 630 of them assistant tokens, packed into 38 windows of 64. Roughly what fraction of the 2,432 window positions produce gradient?
---
About a quarter (626 / 2,432 ≈ 26%). The rest are prompt tokens, markers, separators and padding; they are computed, attended to, and ignored by the loss. That is normal: in production SFT runs with long system prompts the fraction is often lower.
:::

## Learning rate, forgetting, and LoRA

SFT uses a learning rate about an order of magnitude below the pre-training peak (Llama 2 7B, Touvron et al. 2023: approximately 3e-4 for pre-training, 2e-5 for SFT) and one to three epochs. Larger steps on a narrow dataset overwrite what pre-training learned, which is called **catastrophic forgetting**; you can measure it as the base corpus's perplexity rising during SFT. Mixing a fraction of pre-training data into the fine-tuning batches is a common antidote.

**LoRA** (Hu et al. 2021) freezes every weight \`W\` of shape \`[d, k]\` and trains a low-rank update, using \`W + B·A\` with \`B\` of shape \`[d, r]\` (initialised to zero, so training starts from the base model) and \`A\` of shape \`[r, k]\`, \`r\` typically 8–64. Only \`A\` and \`B\` receive gradients and AdamW moments: full fine-tuning of a 7B model keeps two fp32 moments per parameter, approximately 7e9 × 8 bytes = 56 GB of optimizer state, while LoRA's moments take from tens of megabytes to about a gigabyte depending on the rank and which matrices are adapted. Combined with a frozen base stored in 4 bits, that is how QLoRA (Dettmers et al. 2023) fits a 65B fine-tune on one 48 GB GPU.

## Where the toy differs from production

Your window is 64 tokens and the dataset 66 pairs; the demo runs 150 steps at batch 2 with \`lr = 1e-3\`, only 3× below the checkpoint's 3e-3 peak, because the budget is 60 seconds and the loss must visibly fall. No validation split, no epochs, no block-diagonal attention, no LoRA, no mixed pre-training data. The completions after SFT follow the template and stop; on prompts outside the training set they are fluent nonsense, because 66 pairs teach a format, not a world.
`,
  steps: [
    {
      id: 'template',
      title: 'The chat template and its tokens',
      instructions: `
Two functions.

\`formatChat(messages)\`: render \`[{ role, content }]\` as one string. For each message append \`CHAT[role]\` (unknown roles use \`CHAT.user\`), the content, then \`CHAT.end\`. If the **last** message is from the user, append \`CHAT.assistant\` so the model continues as the assistant; after a finished assistant turn append nothing.

\`\`\`js
formatChat([{ role: 'user', content: 'Hi' }])   // '<|user|>Hi<|end|><|assistant|>'
\`\`\`

\`addChatTokens(tokenizer)\`: return a **new** \`BPETokenizer\` that knows the four \`MARKERS\` as special tokens. The constructor takes \`{ vocab, merges, specials }\`; append the markers that are not already in \`vocab\` to the end of both \`vocab\` and \`specials\`, so every existing id (including \`eos\`) keeps its meaning and the checkpoint's embedding rows still point at the right tokens. Do not mutate the input; calling it twice must not add the markers twice.

\`lib/data.js\` exports a reference \`formatChat\`. Write yours first; the tests hold you to the same convention.
`,
      predict: { question: 'After `addChatTokens`, `tokenizer.encode("<|user|>hi")` returns how many ids compared with `base.encode("hi")`?', answer: 'Exactly one more: the marker is a special token and becomes a single id, and the text after it tokenizes exactly as before. If the marker were not registered as a special it would be split into a dozen sub-word ids.' },
      hints: [
        'Each message contributes three pieces: a marker looked up by role, the content, and the end marker. The only decision after the loop is whether to open the assistant\'s turn. For the tokenizer, the constructor already accepts arrays; you only have to build the right ones.',
        'formatChat: inside the loop, look the role up in `CHAT` and fall back to the user marker when the lookup gives `undefined` (the `??` operator does exactly this). After the loop, open the assistant turn only if there is a last message and its role is literally `user`. addChatTokens: work out which of the `MARKERS` are missing from `tokenizer.vocab`, then build a new tokenizer whose vocab and specials are copies of the old arrays with those missing markers appended, and whose merges are the old merges.',
        'addChatTokens: `const missing = MARKERS.filter((mk) => !tokenizer.vocab.includes(mk)); return new BPETokenizer({ vocab: [...tokenizer.vocab, ...missing], merges: tokenizer.merges, specials: /* … the same idea for specials … */ });`. formatChat ends with `const last = messages[messages.length - 1]; if (last && /* … */) out += CHAT.assistant;`.',
      ],
    },
    {
      id: 'example',
      title: 'One example and its loss mask',
      instructions: `
\`tokenizeExample(tokenizer, prompt, response)\` returns \`{ ids, mask }\`. The token stream is the rendered chat for a single user message (which ends with \`<|assistant|>\`) followed by the response and \`<|end|>\`. The mask has one entry per id: \`0\` for every prompt token (markers included), \`1\` for every response token **and** for the end marker that follows it.

\`buildExample(tokenizer, prompt, response)\` returns \`shift(tokenizeExample(...))\`: \`{ x, y, mask }\` where \`x\` is every token but the last, \`y\` every token but the first, and the mask is sliced like \`y\`, because it says which **targets** count. The worked \`shift\` above the TODO line does that slicing; read it and notice which slice the mask takes.

Encode the prompt part and the response part separately: the boundary between them is where the mask flips, and you cannot recover it from one combined encode.
`,
      predict: { question: 'In `buildExample`, at the first position where `mask` is 1, what is `x[t]`?', answer: 'The `<|assistant|>` marker. The first response token is predicted from the assistant marker; that prediction is the first one that counts. If your mask were aligned with `x` instead of `y`, the first counted position would be one step LATER (x is the first response token, y the second), so the prediction that starts the answer would never be trained.' },
      hints: [
        'Two encodes, not one. The prompt part is `formatChat` of a single user message; the response part is the response with the end marker appended. The mask is zeros for the length of the first and ones for the length of the second.',
        'Encode what your `formatChat` renders for one user message (it already ends with `<|assistant|>`). Separately encode the response with `CHAT.end` appended as text: the end marker is a special, so it becomes one id at the end. The ids are the two arrays concatenated; the mask is as many zeros as prompt ids followed by as many ones as response ids. `buildExample` is one line that hands that result to `shift`.',
        '`const ids = promptIds.concat(responseIds); const mask = promptIds.map(() => 0).concat(responseIds.map(() => /* … */)); return { ids, mask };` and `export function buildExample(tokenizer, prompt, response) { return shift(/* … */); }`.',
      ],
    },
    {
      id: 'masked-loss',
      title: 'Masked cross-entropy',
      instructions: `
\`maskedCrossEntropy(logits, y, mask)\`: \`logits\` is a \`Tensor\` of shape \`[B, T, V]\` (or \`[N, V]\`), \`y\` and \`mask\` are nested integer arrays over the leading dimensions. Return a scalar \`Tensor\`:

\`\`\`
loss = sum_t( mask[t] · nll[t] ) / sum_t( mask[t] )     nll[t] = -logSoftmax(logits)[t, y[t]]
\`\`\`

Build it from \`Tensor\` ops so that \`backward()\` flows through it. One way: a \`Float32Array\` \`pick\` of the same size as \`logits\`, with \`pick[t * V + y[t]] = mask[t]\` and zeros elsewhere; then \`-(logits.logSoftmax() * pick).sum() / count\`. Flatten \`y\` and \`mask\` with \`flat(Infinity)\`. Throw an \`Error\` if \`count\` is zero (the mean would be \`0 / 0\`) and if a target is outside \`0..V-1\`.

\`lib/tensor.js\` has \`crossEntropy\`, which averages over **all** positions. The tests check that yours equals it for an all-ones mask, that prompt-position logits do not affect it, and that the gradient at masked-out positions is exactly zero.
`,
      hints: [
        '`crossEntropy` in the lib is a mean over every position; you need a weighted mean. `logSoftmax()` gives log-probabilities for every vocabulary entry; at each position you want one of them, the target\'s, multiplied by that position\'s mask.',
        'Let `V = logits.shape.at(-1)`, `targets = y.flat(Infinity)`, `weights = mask.flat(Infinity)`, `N = targets.length`. Fill `pick` (length `N * V`) with `weights[i]` at `i * V + targets[i]`. Wrap it in `new Tensor({ shape: logits.shape.slice(), data: pick })`, multiply by `logits.logSoftmax()`, sum, and scale by `-1 / count` where `count` is the sum of the weights.',
        '`const pick = new Float32Array(N * V); for (let i = 0; i < N; i++) pick[/* … */] = weights[i]; const picked = logits.logSoftmax().mul(new Tensor({ shape: logits.shape.slice(), data: pick })).sum(); return picked.scale(/* … */);` with `count` computed and checked before use.',
      ],
    },
    {
      id: 'packing',
      title: 'Packing examples into windows',
      instructions: `
\`packExamples(examples, { blockSize, eos })\`: \`examples\` are \`{ ids, mask }\` objects from \`tokenizeExample\`. Concatenate them **in order**, each followed by one \`eos\` id with mask \`0\`, into windows of \`blockSize + 1\` tokens (a window of \`blockSize\` positions needs one extra token because \`x\` and \`y\` overlap). When the next example plus its separator would not fit, pad the current window to \`blockSize + 1\` with \`eos\` (mask \`0\`), \`shift\` it, and start a new one. Pad and shift the last window too. Return the array of \`{ x, y, mask }\` packs, each with exactly \`blockSize\` entries.

Throw if a single example is longer than \`blockSize\`. No examples gives no packs.

The mask must never cross an example boundary: predicting the separator, the padding, or the next example's \`<|user|>\` is never an assistant token. If you keep one \`ids\` array and one \`mask\` array per window and push to both in lockstep, this is automatic.
`,
      hints: [
        'Keep two growing arrays for the current window, `ids` and `mask`, plus a `flush` helper that pads them, shifts them into the output, and resets them. The only arithmetic is the fit test, and it must count the separator.',
        'For each example: if the tokens already in the window, plus the example, plus its one separator would exceed the `blockSize + 1` tokens a window holds, flush first. Then push the example\'s ids and mask entries, then `eos` with mask `0`. After the loop, flush once more. `flush` does nothing on an empty window; otherwise it pads with `eos`/`0` up to `blockSize + 1` tokens, pushes `shift({ ids, mask })` and starts fresh arrays.',
        '`const flush = () => { if (ids.length === 0) return; while (ids.length < blockSize + 1) { ids.push(eos); mask.push(0); } packs.push(shift({ ids, mask })); ids = []; mask = []; }; for (const example of examples) { if (/* … the fit test … */) flush(); … }`.',
      ],
    },
    {
      id: 'resize',
      title: 'Embedding rows for the new tokens',
      instructions: `
\`resizeEmbeddings(model, newVocabSize)\`: return a **new** \`GPT\` built with \`{ ...model.config, vocabSize: newVocabSize }\` whose parameters are copied from \`model\`, with one exception: \`wte.weight\` has shape \`[newVocabSize, nEmbd]\`; rows \`0..oldV-1\` are copied from the old table and every new row is the **column-wise mean of the old rows**. Throw if \`newVocabSize\` is smaller than the old vocabulary. Do not modify \`model\`.

\`model.parameters()\` and \`paramNames(model)\` line up index by index, so you can loop over both models' parameter lists together and treat the entry named \`'wte.weight'\` specially. The table is row-major: row \`i\` occupies \`data[i * d .. i * d + d - 1]\`, and \`Float32Array.prototype.set(src, offset)\` copies a row in one call.

Why the mean? A fresh random row sits far from every trained embedding, so the tied head gives the new token strange logits and the first gradient steps are spent undoing that. The mean is a typical trained row (Hewitt 2021; Hugging Face \`resize_token_embeddings\` by default samples new rows around that mean, using the old rows' covariance). Because the head is tied, the old tokens' logits are unchanged after the resize; the tests check that too.
`,
      hints: [
        'Two models with the same list of parameters in the same order: copy each one across with `dst.data.set(src.data)`. Only `wte.weight` has a different size, and there the old data is a prefix of the new.',
        'For `wte.weight`: `table.set(old)` copies rows `0..oldV-1` in one call. Then accumulate `mean[j] += old[i * d + j] / oldV` over `i < oldV`, and for each `i` from `oldV` to `newVocabSize - 1` do `table.set(mean, i * d)`.',
        '`const grown = new GPT({ ...model.config, vocabSize: newVocabSize }); const names = paramNames(model); const src = model.parameters(); const dst = grown.parameters(); for (let k = 0; k < names.length; k++) { if (names[k] !== "wte.weight") { dst[k].data.set(src[k].data); continue; } /* … copy the old rows, compute the mean, set the new rows … */ } return grown;`',
      ],
    },
    {
      id: 'finetune',
      title: 'The fine-tuning loop',
      instructions: `
\`sftStep(model, optimizer, batch, { maxGradNorm })\`: the module 07 step with the masked loss. \`batch\` is \`{ x, y, mask }\` with \`x\` and \`y\` of shape \`B × T\`. Forward, \`maskedCrossEntropy\`, \`backward\`, \`clipGradNorm(model.parameters(), maxGradNorm)\`, \`optimizer.step()\`, \`optimizer.zeroGrad()\`. Return \`{ loss, gradNorm }\` as plain numbers.

\`finetune(model, packs, { steps, lr, batchSize, weightDecay, maxGradNorm, next, onStep })\`: create \`new AdamW(model.parameters(), { lr, betas: [0.9, 0.95], weightDecay })\`, then for each step draw \`sampleBatch(packs, batchSize, next)\` (the worked helper), call \`sftStep\`, push the loss, and \`await onStep(step, loss)\` if \`onStep\` is given. Return the array of losses. All randomness must come from \`next\`, so two runs with the same seed are identical.

No warmup or cosine schedule here, to keep the loop minimal: the run is 150 steps from an already-trained model. Production SFT recipes usually do keep a schedule (Llama 2's SFT decays a 2e-5 learning rate with a cosine; Stanford Alpaca warms up for 3% of steps, then cosine), which module 07's \`cosineWithWarmup\` would give you.
`,
      hints: [
        'This is `trainStep` from module 07 with one substitution (the loss) and one extra argument threaded through (the mask). The order of the six calls matters: gradients must exist before clipping and must be cleared after the step.',
        'sftStep: forward `batch.x` to logits, score them with your `maskedCrossEntropy` against `batch.y` and `batch.mask`, backward, then clip (its return value is the pre-clip norm you report), step, zeroGrad; return the loss as a plain number via `.item()`. finetune: build ONE optimizer before the loop, because AdamW\'s moments must carry from step to step; each step draws a batch, calls `sftStep` with `maxGradNorm`, records the loss and awaits `onStep`.',
        'sftStep: `const logits = model.forward(batch.x); const loss = /* … */; loss.backward(); const gradNorm = clipGradNorm(/* … */); /* … step, then zeroGrad … */ return { loss: loss.item(), gradNorm };`. finetune: `const optimizer = new AdamW(model.parameters(), { lr, betas: [0.9, 0.95], weightDecay }); const losses = []; for (let step = 0; step < steps; step++) { const batch = sampleBatch(/* … */); /* … sftStep, push, onStep … */ } return losses;`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why SFT is "pre-training with a mask": what stays the same in the loop, what changes, and why the mask, not the data, decides what the model learns to imitate.',
    'In the demo, the model learned to answer inside the template and stop after 150 steps on 66 examples, but its answers to unseen prompts were nonsense. Which of those two facts is about the format and which is about knowledge, and what would you change to improve each?',
    'Suppose you dropped the mask and trained on every token. Predict what greedy decoding from `<|user|>Say hello.<|end|><|assistant|>` would look like after training, and explain the mechanism.',
  ],
  stretch: [
    'Build a block-diagonal attention mask so that packed examples cannot attend across their `eos` boundaries (what FlashAttention\'s `varlen` kernels and Hugging Face `padding_free` batching do), and measure whether the masked loss after 150 steps changes.',
    'Module 30 later builds LoRA in full. As a preview, freeze everything except the final LayerNorm (`lnF`) and the head (tied to `wte`, so this also trains the token embeddings), and compare the number of trained parameters and the loss with full fine-tuning.',
    'Measure catastrophic forgetting: compute the checkpoint\'s loss on a slice of `CORPUS` before and after SFT, then mix one pre-training window into every SFT batch (InstructGPT\'s PPO-ptx mixed pre-training gradients into RL for the same reason; Ouyang et al. 2022) and measure again.',
    'Extend `tokenizeExample` to multi-turn conversations where every assistant turn is masked in and every user turn masked out, matching TRL\'s `assistant_only_loss` on the Llama 3 template.',
  ],
  timeouts: { tests: 20000, demo: 120000 },
};
