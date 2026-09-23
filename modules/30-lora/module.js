export default {
  id: '30-lora',
  title: 'Parameter-efficient fine-tuning (LoRA)',
  track: 'posttraining',
  minutes: 90,
  threshold: 'A fine-tuning update lives in a low-rank subspace, so training a small fraction of the parameters (with no gradients or optimizer state for the rest) recovers most of full fine-tuning, and the adapter folds back into the weights for free at inference.',
  goal: 'Low-rank adapters wrapped around your GPT\'s projections, trained on a small fraction of the parameters (12% at the toy\'s d = 64, under 1% at production widths), merged back into plain weights, and compared with full fine-tuning.',
  prereqs: ['02-autograd', '06-transformer', '08-scaling', '10-sft'],
  recall: [
    { q: 'In lib/tensor.js, what happens during `backward()` to a leaf tensor whose `requiresGrad` is false?',
      options: ['Its gradient is computed and then thrown away', 'Nothing: no gradient is computed or stored for it, and its `.grad` stays null', 'backward() throws an error'], answer: 1,
      why: 'The backward closures skip any input that does not require grad. Setting `requiresGrad = false` is therefore the whole mechanism of freezing: no gradient memory, no weight-gradient matmul.' },
    { q: 'In module 10\'s SFT loss, which positions contribute to the cross-entropy?',
      options: ['Every token in the window', 'Only the assistant\'s response tokens (mask 1)', 'Only the prompt tokens'], answer: 1,
      why: 'Prompt tokens are context, not targets. This module reuses the same masked loss; LoRA changes which parameters learn, not what they learn from.' },
    { q: 'Module 08 counted a bf16 training step with fp32 AdamW at how many bytes per parameter of persistent state?',
      options: ['4', '8', '16'], answer: 2,
      why: '2 (bf16 weight) + 2 (bf16 gradient) + 12 (fp32 master copy, first moment, second moment). LoRA\'s memory win comes from paying the last 14 bytes only for the adapter.' },
    { q: 'In module 06\'s GPT, which part of a block holds most of its parameters?',
      options: ['The two LayerNorms', 'The attention projections (4C²)', 'The MLP (8C²)'], answer: 2,
      why: 'fc is C→4C and proj is 4C→C. That is why QLoRA\'s "adapt all linear layers" adds adapters to the MLP, not just to attention.' },
  ],
  review: [
    { q: 'Why is B initialised to zeros rather than randomly?',
      options: ['To save memory', 'So that ΔW = (alpha/r)·A·B is exactly zero and the adapted model starts identical to the base model', 'Because Adam cannot handle random initial values'], answer: 1,
      why: 'Fine-tuning should start from the pre-trained function, not from a randomly perturbed one. A must still be random, otherwise both gradients stay zero forever.' },
    { q: 'For a 4096 × 4096 projection with rank r = 8, what fraction of that matrix\'s parameters does the adapter train?',
      options: ['About 0.4%', 'About 4%', 'About 12%'], answer: 0,
      why: 'r·(nIn + nOut) / (nIn·nOut) = 8·8192 / 16.8M ≈ 0.39%, which is 2r/d for a square matrix. For the lab\'s 64 × 64 attention output projection the same formula gives 25%, and across the whole d = 64 model (embeddings included) rank 8 comes to about 12%, which is why the lab\'s percentage looks large.' },
    { q: 'After `merge()`, what does the adapter cost at inference?',
      options: ['One extra matmul per layer', 'Nothing: W + (alpha/r)·A·B is a plain weight of the original shape', 'r extra tokens of context'], answer: 1,
      why: 'Matrix multiplication is linear, so x·W + s·(x·A)·B = x·(W + s·A·B). The merged model has exactly the base parameter count and speed.' },
    { q: 'Under mixed-precision AdamW, a frozen parameter costs how many bytes of persistent training state?',
      options: ['2 (its bf16 weight only)', '8', '16'], answer: 0,
      why: 'No gradient, no master copy, no moments. Trainable parameters cost 16 bytes, so LoRA on a 7B model needs approximately 14 GB of state against approximately 108 GB for full fine-tuning.' },
    { q: 'On the very first optimizer step, which adapter matrix receives a non-zero gradient?',
      options: ['A only', 'B only', 'Both'], answer: 1,
      why: 'dL/dA = s·xᵀ·(g·Bᵀ) is zero while B = 0, and dL/dB = s·(x·A)ᵀ·g is not, because A is random. B moves first, and A starts to learn one step later.' },
  ],
  concept: `
## What full fine-tuning costs, and why you can skip most of it

Module 10 fine-tuned every parameter of the checkpoint. That costs what training costs: a gradient, an fp32 master copy and two AdamW moments per weight, 16 bytes per parameter (module 08). For a 7-billion-parameter model that is approximately 108 GB before activations, and every fine-tune produces a complete new copy of the model to store and serve.

Two results say you do not need most of that. Aghajanyan et al. (2020) showed that fine-tuning has a low **intrinsic dimension**: optimising only about 200 numbers, projected into RoBERTa's full weight space through a fixed random projection, reaches 90% of full fine-tuning's accuracy on the MRPC paraphrase task, and the larger the pre-trained model, the fewer such numbers it needs. Hu et al. (2021) turned this into **LoRA** (low-rank adaptation): freeze every pre-trained weight and learn each update as a product of two thin matrices. On GPT-3 175B they reported roughly 10,000× fewer trainable parameters and 3× less GPU memory, with quality on par with full fine-tuning.

## The adapter

In the lab's layout a Linear stores \`W\` with shape \`[nIn, nOut]\` and computes \`y = x·W + b\`. A LoRA adapter of rank \`r\` adds two matrices:

- \`A\` with shape \`[nIn, r]\`, Gaussian at initialisation (it projects the input down to \`r\` numbers),
- \`B\` with shape \`[r, nOut]\`, zero at initialisation (it projects those back up),

and computes \`y = x·W + b + (alpha/r)·(x·A)·B\`. The update it represents, \`ΔW = (alpha/r)·A·B\`, has rank at most \`r\` and costs \`r·(nIn + nOut)\` parameters instead of \`nIn·nOut\`. (The paper stores \`W\` as \`[out, in]\` and writes \`W + B·A\`; it is the same object transposed.)

:::predict
Both \`A\` and \`B\` could start at zero, which would also make the adapted model equal to the base model. What would go wrong?
---
Nothing would ever train. \`dL/dB = s·(x·A)ᵀ·g\` is zero when \`A = 0\`, and \`dL/dA = s·xᵀ·(g·Bᵀ)\` is zero when \`B = 0\`. With \`A\` random and \`B = 0\`, \`B\` receives a gradient on the first step and \`A\` from the second step on. You will check exactly this in step 2.
:::

\`W\` is frozen: autograd never computes its gradient and the optimizer holds no state for it. The constant \`alpha/r\` (\`alpha\` is a hyperparameter, commonly \`r\` or \`2r\`) keeps the size of the update roughly stable when you change \`r\`, so a learning rate tuned at one rank transfers to another. **rsLoRA** (Kalajdzievski 2023) argues \`alpha/sqrt(r)\` is the stable choice at large rank, and **DoRA** (Liu et al. 2024) splits each weight column into a magnitude and a direction and adapts the direction with LoRA.

## Which matrices, and how much it saves

Hu et al. adapted only the attention query and value projections. **QLoRA** (Dettmers et al. 2023) found that adapting *all* linear layers is what matches full fine-tuning, and that the rank matters less than coverage. In your GPT \`attn.qkv\` is one fused Linear, so targeting it adapts query, key and value together.

QLoRA also stores the frozen base in 4-bit **NF4** (a 16-level grid placed at normal-distribution quantiles, a block-wise quantisation of the kind module 19 builds later), dequantises each block on the fly, and pages optimizer state to CPU RAM on memory spikes. The adapters stay in bf16. That combination fine-tuned a 65B model on a single 48 GB GPU.

:::predict
Llama 2 7B has approximately 6.74 billion parameters. With \`r = 8\` on all seven linear projections of its 32 layers, the adapters hold approximately 20 million. Using 2 bytes per frozen parameter and 16 per trainable one, how much persistent training state do you need, against full fine-tuning?
---
\`2 × 6.74e9 + 16 × 2.0e7 ≈ 13.8 GB\` against \`16 × 6.74e9 ≈ 108 GB\`: about 8× less, dominated now by the frozen bf16 weights. That is why QLoRA's next move is to shrink those to 4 bits (approximately 3.4 GB).
:::

## Merge, or keep them apart

Because \`x·W + s·(x·A)·B = x·(W + s·A·B)\`, you can fold the adapter back into the weight after training. The merged model has the base's shape, parameter count and speed; the adapter was free at inference.

Keeping adapters **unmerged** has its own payoff. One base model in GPU memory can serve hundreds of fine-tunes, each only megabytes to tens of megabytes (the rank-8, all-linear Llama 2 7B adapter above is 20 million parameters, about 40 MB in bf16), by applying the right \`A\` and \`B\` per request inside a batch. S-LoRA (Sheng et al. 2023) and Punica (Chen et al. 2023, with its batched SGMV kernel) do exactly that, and the cluster router of module 26, later in the path, has to become adapter-aware to exploit it.

## Where the toy differs from production

Your model has \`d = 64\`, so a rank-8 adapter is about 12% of the model rather than the 0.1–1% typical at \`d = 4096\` (for a square \`d × d\` matrix the adapter's share is \`2r/d\`: 25% at \`d = 64\`, 0.4% at \`d = 4096\`). Everything here is fp32 on one CPU thread: there is no quantised base, no bf16, no paged optimizer, and the memory figures are computed with the production accounting, not measured. The dataset is 66 short instruction pairs and each run is 100 steps, so neither method converges: both learn the reply format (a short answer, then a newline) well before they learn which answer belongs to which prompt. The demo gives LoRA a higher learning rate than full fine-tuning (3e-3 against 1e-3), as is usual in practice, since the adapter starts at zero and moves few numbers. On this tiny model the loss gap between LoRA and full fine-tuning is still visible, which is a result about \`d = 64\` and 100 steps as much as about LoRA.

Do not expect a LoRA step to be much faster. Backward still carries activation gradients through every frozen layer; freezing skips only the weight-gradient matmuls (about a third of a step's matmul work), and the adapter adds its own small matmuls. The demo's LoRA run is only somewhat quicker than the full run. LoRA's win is memory, not step time.
`,
  steps: [
    {
      id: 'forward',
      title: 'The adapter\'s forward pass',
      instructions: `
Complete the first half of \`LoRALinear\`, a wrapper around a \`Linear\` called \`base\`.

In the constructor create:

- \`this.A\`: shape \`[nIn, rank]\`, Gaussian with standard deviation \`1 / sqrt(nIn)\` drawn from \`next\` (\`ops.randn(shape, next, std)\`),
- \`this.B\`: shape \`[rank, nOut]\`, all zeros,
- both wrapped with \`Tensor.param\` so they are trainable leaves,
- \`this.scaling = alpha / rank\`.

Then \`forward(x)\` returns \`x·W + b + scaling·(x·A)·B\`. Use \`this.base.forward(x)\` for the first two terms. Compute the adapter path as \`(x·A)·B\`, never as \`x·(A·B)\`: the first costs \`r·(nIn + nOut)\` multiply-adds per token, the second builds a full \`[nIn, nOut]\` matrix every call. \`x\` may be \`[N, nIn]\` or \`[B, T, nIn]\`; \`Tensor.matmul\` with a 2-D right operand handles both. The tests check the statistics of \`A\` (a constant \`A\` would have identical columns, so the adapter could never exceed rank 1), the scaling, and that \`forward\` never multiplies \`A\` by \`B\`.

The getters \`weight\` and \`bias\` are already written: they let \`lib/gpt.js\` keep finding \`blocks.0.attn.qkv.weight\` after you swap the Linear for a \`LoRALinear\` in step 3.
`,
      predict: { question: 'Right after construction, how different is `lora.forward(x)` from `base.forward(x)`?', answer: 'Identical: `(x·A)·B` is `(x·A)·0 = 0`. Fine-tuning therefore starts exactly at the pre-trained model, and the first test checks this.' },
      hints: [
        'Two new trainable tensors and one number in the constructor; one extra term in forward. The Linear in lib/layers.js shows how a trainable weight is created.',
        'A is Gaussian noise made with ops.randn (shape and std from the instructions), B is ops.zeros of its shape; wrap both in Tensor.param. In forward, multiply x by A first, multiply that result by B, scale it by this.scaling, and add it to the base layer\'s output.',
        '`this.A = Tensor.param(ops.randn([this.nIn, rank], next, /* std */));` and the same pattern with `ops.zeros` for B. In forward: `const delta = x.matmul(this.A).matmul(this.B);` then add `delta`, scaled with `.scale(number)`, to `this.base.forward(x)`.',
      ],
    },
    {
      id: 'freeze',
      title: 'Freeze the base, train only A and B',
      instructions: `
Two changes to \`LoRALinear\`:

1. At the end of the constructor, set \`requiresGrad = false\` on \`base.weight\` and, if it exists, on \`base.bias\`. LoRA's usual default (\`bias="none"\` in Hugging Face \`peft\`) leaves biases frozen; if they trained, the adapter would no longer be just \`A\` and \`B\`.
2. \`parameters()\` returns \`[this.A, this.B]\`, the very same Tensor objects, so an optimizer built from it updates the adapter and nothing else.

Freezing is not a flag the optimizer reads. It is autograd not computing the gradient at all: \`lib/tensor.js\` skips every input with \`requiresGrad === false\`, so the frozen weight's \`.grad\` stays \`null\`, the weight-gradient matmul is never run, and no gradient memory is allocated. The tests run a backward pass and three AdamW steps and check that \`W\` and \`b\` are bit-for-bit unchanged.
`,
      predict: { question: 'After one backward pass on a fresh adapter, which of W, A and B hold a non-zero gradient?', answer: 'Only B. W is frozen (its grad stays null). dL/dA = s·xᵀ·(g·Bᵀ) is exactly zero because B = 0. After the first optimizer step B is non-zero and A starts receiving gradient.' },
      hints: [
        'Freezing is one property on each base tensor. parameters() decides what the optimizer can see.',
        'In the constructor, switch off requiresGrad on the base weight and, when the layer has one, on the base bias. parameters() lists the two adapter matrices and nothing from the base.',
        '`base.weight.requiresGrad = false; if (base.bias) …;` and `parameters() { return [/* the two adapter tensors */]; }`',
      ],
    },
    {
      id: 'apply',
      title: 'Adapters into the GPT, and what they cost',
      instructions: `
Three functions.

\`applyLora(model, { rank, alpha, targets, next })\`:

1. Freeze **every** tensor in \`model.parameters()\` (embeddings, LayerNorms, and Linears you are not targeting).
2. Walk \`paramNames(model)\`. For each name ending in \`.weight\`, strip the suffix to get a module path such as \`blocks.1.mlp.fc\`. If the path ends with \`'.' + t\` for some \`t\` in \`targets\` (default \`ALL_LINEAR = ['attn.qkv', 'attn.proj', 'mlp.fc', 'mlp.proj']\`), wrap it: \`new LoRALinear(getModule(model, path), { rank, alpha, next })\`, and put the wrapper back with \`setModule\`.
3. Return \`[{ name: path, layer }]\` for the wrapped layers, in \`paramNames\` order.

\`trainableParameters(model)\`: every tensor in \`model.parameters()\` whose \`requiresGrad\` is still true, followed by \`layer.parameters()\` for each adapter found by \`loraLayers(model)\` (written for you). On an un-adapted model this returns everything, which is full fine-tuning; one function serves both runs in the demo.

\`countParams(model)\` returns \`{ trainable, total, percent }\`. \`total\` is every number the adapted model holds: \`model.numParams()\` (the frozen base, found through the \`weight\`/\`bias\` getters) plus the adapters. \`percent = 100·trainable/total\`. On the lab config (\`nEmbd = 64\`, 2 layers) with \`r = 4\` on all four linear layers that is 8,192 of 128,768.
`,
      hints: [
        'The worked `loraLayers` already walks paramNames and strips `.weight`; applyLora is the same walk with a target test and a replacement.',
        'Freeze first (loop over model.parameters()), then walk the names. For counting, sum `p.size` over trainableParameters for `trainable`, and add the adapter sizes to `model.numParams()` for `total`.',
        '`const path = name.slice(0, -".weight".length); if (!targets.some((t) => path.endsWith("." + t))) continue; const layer = new LoRALinear(…); setModule(model, path, layer); wrapped.push({ name: path, layer });`',
      ],
    },
    {
      id: 'merge',
      title: 'Merge the adapter away',
      instructions: `
\`LoRALinear.merge()\` returns a **new** plain \`Linear\` whose weight is \`W + scaling·A·B\` and whose bias is a copy of the base bias. \`A·B\` is \`[nIn, r]·[r, nOut] = [nIn, nOut]\`, the same shape as \`W\`, so no transpose is needed. Do not modify \`base.weight\` in place: keeping the base intact is what lets a server swap one adapter for another.

Creating a \`Linear\` requires a seeded rng for its (immediately overwritten) initialisation; any function returning numbers in (0, 1) works, e.g. \`next: () => 0.5\`. Pass \`bias: this.base.bias !== null\`.

\`mergeLora(model)\` replaces every adapter found by \`loraLayers(model)\` with its merged Linear (use \`setModule\`) and returns how many it merged. The tests compare the model's logits before and after within \`1e-5\` and check that no adapter and no extra parameter remains.

The merged Linear is a fresh layer, so its tensors are \`Tensor.param\` (trainable), while the embeddings and LayerNorms stay frozen from \`applyLora\`. The merged model is meant for inference; \`countParams\` on it reports the Linears as trainable. To fine-tune it again, set \`requiresGrad = true\` on every tensor in \`model.parameters()\` first.
`,
      hints: [
        'Matrix multiplication is linear: x·W + s·(x·A)·B = x·(W + s·A·B). Compute A·B once with `ops.matmul` (it accepts Tensors, which have shape and data).',
        'Build `new Linear(nIn, nOut, { bias: …, next: () => 0.5 })`, then fill its weight.data element by element from W and the product, and copy the bias with `.data.set(...)`.',
        '`const delta = ops.matmul(this.A, this.B); for (let i = 0; i < W.length; i++) merged.weight.data[i] = /* W[i] plus the scaled delta */;`',
      ],
    },
    {
      id: 'finetune',
      title: 'What training costs, and the training loop',
      instructions: `
\`trainingMemory({ total, trainable })\` returns persistent training state in bytes under mixed-precision AdamW (module 08's accounting): \`weights = 2·total\` (bf16 copy of everything), \`grads = 2·trainable\`, \`optimizer = 12·trainable\` (fp32 master copy plus the two moments), and \`total\` their sum. So a frozen parameter costs 2 bytes and a trainable one 16.

\`finetune(model, examples, { steps, lr, batchSize, maxGradNorm = 1, next, onStep })\` is module 10's loop, with one decisive change: the optimizer owns \`trainableParameters(model)\` and nothing else.

\`\`\`
params = trainableParameters(model)
optimizer = AdamW(params, lr, betas [0.9, 0.95], no weight decay)
repeat steps times:
  batch = makeBatch(examples, { batchSize, next })
  loss  = maskedLoss(model.forward(batch.x), batch.y, batch.mask)
  backward, clipGradNorm(params, maxGradNorm), step, zeroGrad
  value = loss.item(); losses.push(value)
  if onStep: await onStep(step, value)   // step counts from 0; value is a plain number
return { losses, optimizer }
\`\`\`

\`onStep(step, loss)\` receives the loss as a JavaScript number (\`loss.item()\`), not the Tensor: the goal demo formats it with \`toFixed\`, and a test calls \`finetune\` with a recorder and checks the steps \`0 … steps-1\` and the type of each loss.

Handing AdamW \`model.parameters()\` instead would still leave the frozen weights unchanged (they have no gradient), but it would allocate \`m\` and \`v\` buffers for every one of them, which is exactly the memory LoRA exists to save. The tests check the size of \`optimizer.m\`, and compare your losses step by step with this exact loop (so a missing clip, a different beta or a forgotten \`zeroGrad\` shows up).
`,
      hints: [
        'trainingMemory is three multiplications and a sum. finetune is the loop you wrote in module 10 with the parameter list swapped.',
        'Build the optimizer once, outside the loop, from trainableParameters(model). Clip the same list you gave the optimizer. Return the optimizer so its state can be inspected.',
        '`const params = trainableParameters(model); const optimizer = new AdamW(params, { lr, betas: [0.9, 0.95], weightDecay: 0 }); for (…) { const batch = makeBatch(…); const loss = maskedLoss(…); loss.backward(); /* clip, step, zero */ losses.push(loss.item()); … }`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why training `r·(nIn + nOut)` numbers per matrix can recover most of what training all `nIn·nOut` achieves. What would have to be true about fine-tuning updates for LoRA to fail?',
    'Your demo compared LoRA and full fine-tuning on the same data and steps. Account for the difference in trainable parameters and in gradient-plus-optimizer bytes (the `grads` and `optimizer` fields of `trainingMemory`), and say which of the two memory savings (no gradients, no optimizer state) matters more.',
    'When would you merge an adapter into the weights, and when would you keep it separate? Think about a company serving one fine-tune versus a platform serving five hundred customers\' fine-tunes.',
  ],
  stretch: [
    'Implement rsLoRA (scaling `alpha / sqrt(r)`, Kalajdzievski 2023) and sweep r = 2, 4, 8, 16 with a fixed learning rate; plot final loss against r for both scalings. Hugging Face `peft` exposes this as `use_rslora=True`.',
    'Implement DoRA (Liu et al. 2024): store each column\'s magnitude `m = ||W[:, j]||` as a trainable vector and apply LoRA to the normalised direction. Compare it with plain LoRA at the same rank.',
    'Build QLoRA in miniature: after module 19, quantise the frozen base weights with its group-wise int4 (or an NF4 grid, as `bitsandbytes` does), dequantise in the forward pass, and train bf16-style adapters on top. Measure the loss penalty against an fp32 base.',
    'Serve several adapters unmerged at once, S-LoRA/Punica style: keep one base model, give each request in a batch its own (A, B), and compute `x·W` once for the batch plus a per-request `(x·A)·B`. Count the extra FLOPs against merging each adapter into its own model copy.',
  ],
  timeouts: { tests: 20000, demo: 120000 },
};
