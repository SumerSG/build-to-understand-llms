export default {
  id: '32-multimodal',
  title: 'Vision tokens: a tiny multimodal model',
  track: 'transformer',
  minutes: 105,
  threshold: 'A language model does not care where its input vectors come from: any modality becomes a sequence of vectors in the same embedding space, so "seeing" is a projection problem plus training data that pairs images with text.',
  goal: 'A patch embedder and projector that feed image tokens into your GPT, a contrastive alignment head, and a caption trainer that reads synthetic shapes: after about 120 steps a fresh two-layer GPT writes the exact caption for roughly 90% of held-out 16×16 images.',
  prereqs: ['05-attention', '06-transformer', '07-pretraining'],
  recall: [
    { q: 'In module 06, what does `GPT.forward(ids)` add to the token embeddings before the first block?', options: ['Nothing; attention works out the order', 'A learned position embedding, one row of `wpe` per slot', 'A LayerNorm of the ids'], answer: 1,
      why: '`wte.forward(ids) + wpe.forward(positions)`. Everything after that line is indifferent to where the vectors came from, which is why this module can start the forward pass from image vectors instead of ids.' },
    { q: 'The module-06 GPT ties its head to the embedding table: `logits = x · wteᵀ`. What does that say about the space the model computes in?', options: ['Inputs and outputs live in the same C-dimensional space as the token embeddings', 'The head has its own vocabulary', 'The logits are probabilities'], answer: 0,
      why: 'Every input vector and every prediction is compared with the rows of `wte`. Image tokens have to land in that same space to be of any use, which is exactly the projector\'s job.' },
    { q: 'With the causal mask of module 05, which positions can the token at position 20 attend to?', options: ['Only position 20', 'Positions 0 through 20', 'All positions, including later ones'], answer: 1,
      why: 'Put the image first and every caption token can look at every image token, while the image never sees the caption. Order in the sequence decides who conditions on whom.' },
    { q: 'In module 07, `crossEntropy(logits, y)` with `y[t] = x[t + 1]` averages the loss over…', options: ['Only the last position', 'Every one of the B·T positions', 'A random subset of positions'], answer: 1,
      why: 'Every position predicts its successor and every position counts. Here most positions are image tokens that have no "next word" to predict, so you will need a mask that drops them from the average.' },
    { q: 'A bias vector of shape `[d]` is added to every row of a `[B, T, d]` tensor. Its gradient is…', options: ['The upstream gradient at row 0', 'The upstream gradient summed over all B·T rows', 'The upstream gradient averaged over rows'], answer: 1,
      why: 'A tensor used many times receives the sum of its gradients (module 02). The patch position table `[P, dim]` is shared across the batch, so its gradient is the upstream gradient summed over B.' },
  ],
  review: [
    { q: 'How many patch tokens does a 224×224 image produce with 16×16 patches (ViT-B/16)?', options: ['16', '196', '50,176'], answer: 1,
      why: '(224 / 16)² = 14² = 196. LLaVA-1.5\'s CLIP ViT-L/14 at 336 pixels gives (336 / 14)² = 576. Token count grows with the square of the side length.' },
    { q: 'In the sequence [16 image tokens][caption], which position\'s output predicts the first caption word?', options: ['Position 16, the first caption slot', 'Position 15, the last image token', 'Position 0'], answer: 1,
      why: 'Position i predicts element i + 1. Starting the mask at 16 is an off-by-one that never trains the first word.' },
    { q: 'Why does CLIP divide cosine similarities by a temperature τ (≈ 0.07 at initialisation)?', options: ['To keep the logits positive', 'Cosines lie in [-1, 1]; without scaling, the softmax can never become confident, so the loss has a high floor', 'To normalise the embeddings'], answer: 1,
      why: 'With τ = 1 and 16 captions the loss cannot fall below about 1.1 nats. Dividing by 0.07 stretches the range to about ±14. CLIP learns τ, capping 1/τ at 100.' },
    { q: 'CLIP\'s symmetric loss averages two cross-entropies. The second one (over columns) asks…', options: ['Each image to pick its caption', 'Each caption to pick its image out of the batch', 'Each image to predict its pixels'], answer: 1,
      why: 'Rows train the image tower to find captions; columns train the text tower to find images. With only the row term, a caption is never pushed away from the wrong images.' },
    { q: 'With GPT-2\'s std 0.02 in the patch embedding and projector, fresh image tokens have an RMS of about 2e-4. Why does that slow captioning down so much?', options: ['The GPT cannot process small numbers', 'They are added to position embeddings about 100× larger, so after LayerNorm the image content is a tiny perturbation of the position signal', 'Small tokens overflow the softmax'], answer: 1,
      why: 'The first LayerNorm sees mostly `wpe`. Fan-in initialisation (std 1/sqrt(nIn)) keeps each layer near unit scale and the image becomes visible from step one.' },
  ],
  concept: `
## An image is just more tokens

Your module-06 GPT never sees token ids after its first line: \`wte.forward(ids)\` turns them into vectors of width \`C\`, and everything after that (position embeddings, blocks, LayerNorm, the tied head) works on vectors. Turn a picture into a sequence of \`C\`-dimensional vectors and the same GPT can read it. That is the whole trick behind LLaVA, Qwen-VL and most open vision-language models.

**Patches.** The Vision Transformer (ViT, Dosovitskiy et al. 2020) cuts an image into non-overlapping \`p × p\` squares, flattens each into \`p²\` pixels (times 3 for colour), and multiplies by one shared matrix. A learned **position embedding** is added per patch slot; otherwise a transformer sees a bag of patches. A 224×224 image with 16×16 patches becomes \`(224/16)² = 196\` tokens.

**Projector.** LLaVA (Liu et al. 2023) runs a pretrained CLIP ViT and maps its patch outputs into the language model's embedding space. LLaVA-1.5 uses a two-layer MLP (\`Linear → GELU → Linear\`) and splices the 576 image tokens into the prompt where an \`<image>\` placeholder sits, in front of the user's question. Training goes in two stages: first only the projector, on approximately 558K image-caption pairs, then the projector and the LLM together on approximately 665K instruction examples. Vision encoder + MLP projector + LLM is now the dominant recipe.

:::predict
Your \`PatchEmbed\` adds a learned vector per patch slot. Suppose you delete it. Can the captioner still tell "top left" from "bottom right"?
---
Yes. Your GPT adds its own \`wpe\` to every position, and slot 0 always holds the top-left patch, so the position signal arrives anyway. ViT needs its own because its encoder mixes patches with attention *before* any language model sees them. The CLIP head in step 4 averages over patches, and there position must be baked into each patch vector before the average or it is lost.
:::

## Which positions learn

The training sequence is \`[16 image tokens][caption words][padding]\`. Position \`i\` predicts element \`i + 1\`, as in module 07, but predicting the next *image token* is meaningless (there is no vocabulary entry for it), so a **loss mask** keeps only the positions whose target is a caption word or the final \`eos\`. The last image token predicts the first word. The same masking trains only the assistant's replies in supervised fine-tuning (module 10, later in the path).

:::predict
With GPT-2's init (std 0.02 in every layer), a fresh patch embedding followed by the projector gives image tokens with an RMS of about 2e-4. The GPT's position embeddings have std 0.02. What happens in training?
---
The first LayerNorm sees almost nothing but the position signal, and the image is a hundredfold-smaller perturbation. Measured on this module's demo setup, the run is at about 19% held-out exact-match accuracy after 80 steps and 23% after 120, with the shape word right only about a third of the time. With fan-in initialisation (std \`1/sqrt(nIn)\`, so each layer's output keeps roughly the scale of its input; PyTorch's \`nn.Linear\` scales by fan-in too, drawing uniformly from \`±1/sqrt(nIn)\`) the tokens start near RMS 0.14, and the same run reaches 88% after 80 steps and 90% after 120. The tests check the scale.
:::

## Alignment without generation: CLIP

CLIP (Radford et al. 2021) trains an image tower and a text tower on approximately 400 million image-text pairs. For a batch of \`B\` pairs it builds the \`[B, B]\` matrix of cosine similarities divided by a learned temperature \`τ\` (initially 0.07), and asks each image to pick its caption (cross-entropy over rows) and each caption its image (over columns). This symmetric **InfoNCE** loss uses every other pair as a negative, so CLIP used batches of 32,768. Its image tower already puts images near their descriptions, which is why LLaVA starts from one.

## Early fusion, late fusion

LLaVA **fuses early**: image vectors enter the sequence and the LLM's own attention does the mixing. Fuyu-8B (Adept) even drops the vision encoder and feeds projected patches straight in; Chameleon (Meta, 2024) turns images into discrete codebook tokens and trains one model on interleaved sequences. **Late fusion** keeps the modalities apart longer: Flamingo (DeepMind, 2022) freezes the LLM, compresses each image to 64 vectors and inserts cross-attention layers that read them.

## Image tokens cost context

Tokens grow with the square of resolution: a 1024×1024 image at patch 16 is 4,096 patches. Production systems merge neighbours (Qwen2-VL merges 2×2 patches into one token) or pair a low-resolution overview with high-resolution crops (LLaVA-NeXT), landing between a few hundred and a few thousand tokens per image. Each costs KV cache like any other token (module 15, later in the path) and a slice of the context budget (module 21). A 7B Llama-2-shaped model stores \`2 · 32 layers · 4096 · 2 bytes\` ≈ 0.5 MiB per token in fp16, so LLaVA-1.5's 576 image tokens take about 288 MiB before any text. That is why image-heavy prompts fill context windows and serving memory quickly.

## What is toy here

The images are 16×16 synthetic shapes: four shapes in four quadrants, two sizes, one pixel of jitter, random brightness and noise, giving 16 possible captions over 11 words. The "vision encoder" is one linear map per patch, and nothing is pretrained: a two-layer, 27K-parameter GPT learns vision and language together from 200 images. Held-out images are new draws from the same generator, so their accuracy measures robustness to noise, not understanding of new scenes. LLaVA-1.5 pairs a pretrained ViT-L/14 (approximately 300M parameters) with a 7B or 13B Vicuna. Your CLIP head trains on batches of 16 where CLIP used 32,768. \`concatTokens\` concatenates by multiplying with a 0/1 matrix because \`lib/tensor.js\` has no concat op; PyTorch's \`torch.cat\` copies memory.
`,
  steps: [
    {
      id: 'patchify',
      title: 'Cut the image into patch tokens',
      instructions: `
Implement \`patchify(image, patch)\`. \`image\` is a raw tensor \`[H, W]\`. Return a raw tensor \`[(H/patch)·(W/patch), patch·patch]\`:

- one row per patch, in **reading order**: left to right along the top row of patches, then the next row down;
- inside a row, the patch's pixels **row-major**: its top line of \`patch\` pixels, then the next line.

Throw an \`Error\` if \`H\` or \`W\` is not divisible by \`patch\`. Do not modify the input.

\`\`\`
4×4 image, patch 2            patchify → [4, 4]
 0  1 |  2  3                 [ 0, 1, 4, 5]    patch 0 (top left)
 4  5 |  6  7                 [ 2, 3, 6, 7]    patch 1 (top right)
------+------                 [ 8, 9,12,13]    patch 2
 8  9 | 10 11                 [10,11,14,15]    patch 3
12 13 | 14 15
\`\`\`

A plain \`reshape\` to \`[4, 4]\` would give \`[0, 1, 2, 3]\` as the first row: a strip of the image, not a square. This is the index arithmetic of module 01 with one more level. The worked \`stackPatches\` above calls your function once per image to build a batch \`[B, P, patch²]\`.
`,
      predict: { question: 'A 16×16 image gives 16 tokens with patch 4. How many with patch 2, and how much more attention work (which grows with the square of sequence length) does that cost?', answer: '(16/2)² = 64 tokens, 4× as many. The attention score matrix grows 16×. Halving the patch side quadruples the tokens, and this is why real systems pick patches of 14 to 16 pixels and still merge neighbours.' },
      hints: [
        'Give every pixel two addresses: which patch it is in, and where it sits inside that patch. Which of those decides the output row, and which decides the column? Try it by hand on pixel 6 of the 4×4 example.',
        'Loop over patch rows py and patch columns px (there are `gw = W / patch` patches per row), then over the line y and column x inside the patch. Pixel (y, x) of patch (py, px) is image pixel (py·patch + y, px·patch + x); turn that into a flat offset with the image width W. It goes to column y·patch + x of output row py·gw + px.',
        '`for (py) for (px) { const base = (py * gw + px) * patch * patch; for (y) for (x) out[base + y * patch + x] = image.data[ … ]; }` then `return { shape: [gh * gw, patch * patch], data: out };`',
      ],
    },
    {
      id: 'embed',
      title: 'Patch embedding and projector',
      instructions: `
Two small classes. Together they turn pixels into vectors the GPT can read.

**\`PatchEmbed({ patchDim, nPatches, dim, next })\`**, the ViT stem:

- \`this.proj\`: a \`Linear\` from \`patchDim\` to \`dim\` whose weights have fan-in std \`1/sqrt(patchDim)\` (\`Linear\` in \`lib/layers.js\` takes \`{ next, std }\`; its default std is 0.02).
- \`this.pos\`: a trainable \`[nPatches, dim]\` table, Gaussian with std 0.02, one learned vector per patch slot.
- Draw both from the one \`next\` you are given, \`proj\` first and \`pos\` second. The tests rebuild the same draws from the same seed, so a different order or a fresh rng fails them.
- \`forward(patches)\`: \`[B, P, patchDim] → [B, P, dim]\` is \`proj(patches) + pos\`. Use Tensor ops so the gradient reaches \`pos\`. The \`[P, dim]\` table broadcasts over the batch.
- \`parameters()\`: proj's weight and bias, then \`pos\`.

**\`Projector(dIn, dOut, { hidden = dOut, next })\`**, LLaVA-1.5's MLP:

- \`this.fc1\`: a \`Linear\` from \`dIn\` to \`hidden\` with std \`1/sqrt(dIn)\`; then \`this.fc2\`: a \`Linear\` from \`hidden\` to \`dOut\` with std \`1/sqrt(hidden)\`. Both from \`next\`, in that order.
- \`forward(x)\` is \`fc2(gelu(fc1(x)))\`. \`parameters()\` returns fc1's then fc2's.

**Why these standard deviations?** \`Linear\` defaults to GPT-2's 0.02, which is right for a width-768 residual stream but makes fresh image tokens about 100× quieter than the GPT's position embeddings (see the second predict card in the concept). The \`1/sqrt(nIn)\` fan-in scale keeps each layer's output near the scale of its input. One test checks that fresh image tokens are at least as loud as the position embeddings.
`,
      hints: [
        '`Linear` and `Tensor.param` do the work, and `forward` is one line in each class. Before you write it, predict the RMS of a fresh image token if every layer used the default std 0.02 instead of `1/sqrt(nIn)`: each layer multiplies the scale by roughly `std · sqrt(nIn)`.',
        'PatchEmbed: build `proj`, then `pos`, in that order, both from the same `next`. Its forward projects the patches and then adds the position table with a Tensor op; broadcasting spreads the `[P, dim]` table over the batch. Projector: `fc1`, a GELU (a Tensor method), then `fc2`. `parameters()` concatenates the layers\' own `parameters()` lists.',
        '`this.proj = new Linear(patchDim, dim, { next, std: 1 / Math.sqrt(patchDim) }); this.pos = Tensor.param(ops.randn(/* shape, rng, std */));` and in Projector `this.fc1 = new Linear(dIn, hidden, { next, std: 1 / Math.sqrt(dIn) }); this.fc2 = …` with `forward(x) { return this.fc2.forward(/* fc1, then GELU */); }`',
      ],
    },
    {
      id: 'sequence',
      title: 'One sequence, loss on the caption only',
      instructions: `
Three functions build the training example \`[image tokens][caption, padded]\` and score only the caption.

**\`embedSequence(gpt, imageTokens, textIds)\`**: \`imageTokens\` is a Tensor \`[B, P, C]\` and \`textIds\` is \`number[][]\` (B×L). Embed the text with \`gpt.wte.forward(textIds)\` and join it after the image with the worked \`concatTokens\`. The result is \`[B, P + L, C]\`, ready for the worked \`forwardEmbeds(gpt, x)\`.

**\`captionTargets(nImage, captionIds, eos, width = captionIds.length)\`** returns \`{ textIds, targets, mask }\`:

- \`textIds\`: the caption padded with \`eos\` to \`width\` (batches need equal lengths);
- \`targets\` and \`mask\`: flat arrays of length \`N = nImage + width\` for this one example. Position \`i\` predicts element \`i + 1\`, so positions \`nImage - 1 … nImage - 1 + L\` predict \`caption[0] … caption[L-1], eos\` and get mask 1. Every other position gets target 0 and mask 0.

\`width\` must be at least the caption length \`L\`; throw an \`Error\` otherwise rather than cutting the caption.

\`\`\`
captionTargets(3, [7, 8], 9, 4)
textIds [7, 8, 9, 9]
targets [0, 0, 7, 8, 9, 0, 0]
mask    [0, 0, 1, 1, 1, 0, 0]
\`\`\`

**\`maskedCrossEntropy(logits, targets, mask)\`**: \`logits\` is a Tensor \`[B, N, V]\`; \`targets\` and \`mask\` are \`number[][]\`, B rows of length N (one \`captionTargets\` result per row). Flattened in order, row \`b\` position \`t\` lines up with row \`b·N + t\` of the logits viewed as \`[B·N, V]\`. Return the mean of \`-log softmax(logits)[target]\` over the positions where \`mask\` is 1, as a scalar Tensor with gradients. Divide by the number of 1s, not by \`B·N\`; if the mask has no 1s, throw an \`Error\` instead of dividing by zero. \`lib/tensor.js\` has no gather op. One way is to multiply \`logits.logSoftmax()\` by a constant \`[B, N, V]\` tensor that holds \`1/count\` at each masked-in target and 0 elsewhere, then \`.sum().neg()\`.
`,
      predict: { question: 'If your mask started at position nImage instead of nImage − 1, what would the trained captioner write first?', answer: 'Nothing trained: the first caption word is never a target, so its logits at the last image slot stay at their random initial values. Greedy decoding then starts with an arbitrary word (or eos, and an empty caption), and the rest of the caption is conditioned on that wrong start.' },
      hints: [
        'The only index that matters in captionTargets is where the supervised run starts: the last image token, `nImage - 1`. Write out the example above by hand before coding.',
        'captionTargets: let `want` be the caption followed by eos, and write want[j] (with mask 1) at the slot one before caption slot j, so want[0] lands on the last image slot. maskedCrossEntropy: count the 1s, then build a constant tensor shaped like the logits that is 1/count at each masked-in (position, target) pair and 0 elsewhere; its elementwise product with the log-probabilities, summed and negated, is the masked mean.',
        '`const pick = ops.zeros(logits.shape); for (let i = 0; i < t.length; i++) if (mk[i]) pick.data[i * V + t[i]] = 1 / count; return /* logSoftmax, times pick, summed, negated */;` with `t = targets.flat()` and `mk = mask.flat()`.',
      ],
    },
    {
      id: 'clip',
      title: 'Contrastive alignment (CLIP)',
      instructions: `
Implement CLIP's loss. You do not need a GPT for this step.

**\`l2normalize(x, eps = 1e-8)\`**: each row of a Tensor \`[B, D]\` divided by its length \`sqrt(Σ x² + eps)\`. Use Tensor ops (\`mul\`, \`sum(-1, true)\`, \`add\`, \`sqrt\`, \`div\`) so it is differentiable.

**\`clipLoss(img, txt, temperature = 0.07)\`** takes two Tensors \`[B, D]\`, where row \`i\` of each describes the same example. Return \`{ loss, logits }\` with

\`\`\`
logits = l2normalize(img) · l2normalize(txt)ᵀ / temperature      // [B, B]
loss   = ( CE(logits, [0..B-1]) + CE(logitsᵀ, [0..B-1]) ) / 2
\`\`\`

where \`CE\` is \`crossEntropy\` from \`lib/tensor.js\`. The first term makes each image pick its caption out of the batch. The second makes each caption pick its image. Normalising turns dot products into cosines, so no tower can win by making its vectors longer. The temperature turns cosines in \`[-1, 1]\` into confident logits.

The worked \`ClipHead\` provides the two towers. The image tower is your \`PatchEmbed\`, a GELU, a mean over patches and a Linear. The text tower is the mean of the caption's word embeddings and a Linear. The goal demo trains them with your loss and then labels held-out images by picking the most similar of the 16 captions ("zero-shot", as CLIP classifies ImageNet).
`,
      predict: { question: 'With τ = 1 and a batch of 16, what is the lowest loss clipLoss can possibly reach?', answer: 'About 1.1 nats. Cosines are at most 1 on the diagonal and, at best, −1 elsewhere, so each row\'s loss is at least `log(1 + 15·e⁻²) ≈ 1.11`. In practice it stays higher, because 16 vectors cannot all be at cosine −1 from each other. Dividing by τ = 0.07 multiplies the logits by about 14 and removes the floor. That is why CLIP learns τ.' },
      hints: [
        'Build the logits first and check their shape, [B, B]. The labels for both cross-entropies are simply 0, 1, …, B−1, because the match for row i is column i.',
        'Normalise both inputs, multiply the image rows by the transposed text rows, and scale by 1 / temperature. The column direction is the same crossEntropy call on the transposed logits. Average the two losses with `.add(…).scale(0.5)`.',
        '`const labels = Array.from({ length: img.shape[0] }, (_, i) => i); const loss = crossEntropy(logits, labels) /* plus the transposed direction, halved */; return { loss, logits };`',
      ],
    },
    {
      id: 'captioner',
      title: 'The caption trainer and caption()',
      instructions: `
The worked \`Captioner\` wires your pieces together: \`model.imageTokens(images)\` runs \`stackPatches → PatchEmbed → Projector\` and returns \`[B, P, C]\` in the GPT's embedding space. It also holds \`model.gpt\`, \`model.tokenizer\` (\`encode\`, \`decode\`), \`model.eos\` and \`model.nPatches\`. Always use \`model.eos\`: this module's tokenizer happens to give eos id 0, but the tests use one where it is not.

**\`captionLoss(model, images, captions)\`**: \`images\` is an array of raw \`[16, 16]\` images, \`captions\` an array of strings. Encode each caption, pad all of them to the longest with \`captionTargets(model.nPatches, ids, model.eos, width)\`, build the sequence with \`embedSequence\`, run \`forwardEmbeds\`, and return \`maskedCrossEntropy\`. Use one batched forward pass. \`captionTargets\` describes one example, so collect the per-example \`textIds\`, \`targets\` and \`mask\` into B-row arrays (for example \`rows.map((r) => r.targets)\`) before you pass them on.

**\`caption(model, image, { maxNewTokens = 12 })\`**: greedy decoding inside \`noGrad\`. Start from the image tokens alone. At each step run \`forwardEmbeds\` on image + words so far, take the **argmax** of the last position's logits, stop at \`eos\` (do not include it), and otherwise append the word. Stop after \`min(maxNewTokens, blockSize − nPatches)\` words so the sequence never exceeds \`blockSize\`. Return \`model.tokenizer.decode(words)\`.

The goal demo trains a fresh two-layer GPT (\`nEmbd\` 32) with AdamW for 120 steps of 16 images and reports exact-match accuracy on 48 held-out images.
`,
      predict: { question: 'By the end of training the demo\'s loss is around 0.06 nats per caption token, yet about 1 caption in 10 is still wrong. How can both be true?', answer: 'Four of the seven supervised targets ("a", "at", "the" and eos) are fixed and cost almost nothing, and the position words are easy. Almost all of the remaining loss sits on one token, the shape word, for the few images whose shape is ambiguous at 5 pixels wide. An average over seven tokens hides an error that decides the whole caption.' },
      hints: [
        'captionLoss is plumbing: every piece already exists (captionTargets, embedSequence, forwardEmbeds, maskedCrossEntropy). The only new decision is the padding width, which is the longest encoded caption.',
        'caption: `img = model.imageTokens([image])`. Loop: `x = out.length ? embedSequence(model.gpt, img, [out]) : img`, `logits = forwardEmbeds(model.gpt, x)`, then read the last row `logits.data.subarray((T - 1) * V, T * V)` and take its argmax.',
        '`for (let s = 0; s < limit; s++) { … const best = argmax(lastRow); if (best === model.eos) break; out.push(best); } return model.tokenizer.decode(out);` all wrapped in `noGrad(() => { … })`.',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why the GPT did not need a single change to read images. What exactly does the projector have to learn, and what does the loss mask have to do with it?',
    'Your captioner learns "where" (the quadrant) before "what" (the shape), and in the demo the position word is usually right more often than the shape word. Which parts of the model carry each kind of information, and why is one easier?',
    'CLIP never generates a word, and the captioner never compares two images. What does each objective teach, and why does LLaVA start from a CLIP-trained vision tower rather than training one through the caption loss alone?',
  ],
  stretch: [
    'Replace the per-patch vision tower with a tiny ViT: two non-causal transformer blocks over the 16 patch tokens (a copy of `Block` from lib/gpt.js whose attention calls `attention(q, k, v, { causal: false })` from lib/attention.js) before the projector. This is the shape of CLIP ViT-L/14 inside LLaVA-1.5. Does it learn the shape word faster?',
    'Reproduce LLaVA\'s two-stage recipe: train the ClipHead first, freeze its image tower as the captioner\'s vision encoder, train only the Projector (stage 1), then unfreeze the GPT (stage 2). Compare accuracy at equal steps with the from-scratch run.',
    'Compress the 16 image tokens to 4 with learned queries that cross-attend to the patches, as in Flamingo\'s Perceiver Resampler or BLIP-2\'s Q-Former. Measure caption accuracy against sequence length: this is the trade-off that sets image-token budgets and KV-cache size in serving.',
    'Try early fusion the Chameleon way: quantise each patch to the nearest of K learned codebook vectors (a tiny VQ), add those K ids to the GPT\'s vocabulary, and train on sequences of pure token ids. What does discretising cost in accuracy?',
  ],
  timeouts: { tests: 20000, demo: 150000 },
};
