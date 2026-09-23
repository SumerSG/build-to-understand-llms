export default {
  id: '02-autograd',
  title: 'Autograd from scratch',
  track: 'foundations',
  minutes: 120,
  threshold: 'A computation graph records how each value was made; running the chain rule backwards over that record gives every gradient in one backward pass that costs a small constant multiple of the forward pass (about 2×), however many parameters there are.',
  goal: 'A reverse-mode automatic differentiation engine (a Tensor class whose ops record a graph and whose backward() fills every parameter\'s .grad) that passes numerical gradient checks and trains a linear model by gradient descent, recovering w ≈ 3 and b ≈ 2 from noisy data.',
  prereqs: ['01-tensors'],
  recall: [
    { q: 'In module 01, adding a length-3 bias to a [4, 3] tensor read the bias at index…', options: ['`i`', '`i % 3`', '`i / 3`'], answer: 1,
      why: 'Broadcasting re-reads the same bias entry for every row. In this module the gradient runs the other way: every row that read bias[j] sends a gradient back to it, so bias.grad[j] is a sum over rows.' },
    { q: 'For `matmul(a, b)` with `a` of shape [n, k] and `b` of shape [k, m], the result has shape…', options: ['[k, k]', '[n, m]', '[m, n]'], answer: 1,
      why: 'You will need this to check the two backward products: dA = dC·Bᵀ is [n, m]·[m, k] = [n, k], the shape of A, and dB = Aᵀ·dC is [k, n]·[n, m] = [k, m], the shape of B.' },
    { q: 'Why does softmax (module 01) subtract the row maximum before exponentiating?', options: ['To make the probabilities sum to 1', 'To avoid overflow; the result is mathematically identical', 'To make the loop faster'], answer: 1,
      why: 'The fused cross-entropy in this module relies on the same trick inside log-softmax, which is why it stays finite for logits of 1000.' },
    { q: 'What does `new Float32Array(6)` hold right after creation?', options: ['Six `undefined` values', 'Six zeros', 'Whatever was in memory'], answer: 1,
      why: 'Gradients are accumulated with `+=` into a freshly allocated Float32Array; that only works because typed arrays start at zero.' },
    { q: 'A float32 number carries roughly how many significant decimal digits?', options: ['About 3', 'About 7', 'About 16'], answer: 1,
      why: '24 bits of mantissa is about 7 decimal digits. This sets the perturbation size and tolerance your numerical gradient check can use: `x + 1e-8` rounds back to `x` in float32.' },
  ],
  review: [
    { q: 'Reverse-mode autodiff computes the gradient of one scalar loss with respect to N parameters in about…', options: ['N extra forward passes', 'One extra pass over the recorded graph, whatever N is', 'log N passes'], answer: 1,
      why: 'Each node\'s _backward runs once and does a small constant multiple of its forward op\'s work (a matmul\'s backward is two matmuls of the same size), so the backward pass costs about 2× the forward regardless of the parameter count. Forward mode or finite differences would cost one pass per parameter.' },
    { q: 'A tensor `x` is used twice in a graph. After `backward()`, `x.grad` holds…', options: ['The gradient from the last use only', 'The sum of the gradients from both uses', 'The larger of the two'], answer: 1,
      why: 'The chain rule sums over every path from the loss to x, which is why every backward closure accumulates with `+=` instead of assigning.' },
    { q: 'A bias of shape [3] was broadcast over a [64, 3] output. Its gradient is…', options: ['The [64, 3] output gradient', 'The output gradient summed over the 64 rows, shape [3]', 'The first row of the output gradient'], answer: 1,
      why: 'Broadcasting copies forward, so the backward pass sums the copies back: every row that used bias[j] contributes to bias.grad[j]. This is exactly what `unbroadcast` does.' },
    { q: 'The gradient of the mean cross-entropy over N rows with respect to the logits is…', options: ['`softmax − onehot`', '`(softmax − onehot) / N`', '`onehot − softmax`'], answer: 1,
      why: 'Each row contributes softmax − onehot, and the mean divides by N. Without the 1/N the gradient is N times too large, so a learning rate tuned for the mean loss acts N times bigger, and changing the batch size silently changes the step size.' },
    { q: 'Why does the gradient check use eps ≈ 1e-3 and tol ≈ 1e-2 rather than eps 1e-8 and tol 1e-6?', options: ['To make the check faster', 'Data is float32, so smaller perturbations vanish in rounding and the difference quotient becomes noise', 'Because the chain rule is only approximate'], answer: 1,
      why: 'With about 7 significant digits, a perturbation of 1e-8 on a value near 1 is lost entirely, and even 1e-3 leaves the difference `f(x+eps) − f(x−eps)` with only three or four good digits. The tolerance has to allow for that.' },
  ],
  concept: `
## One loss, a million knobs

Training adjusts every parameter to reduce one scalar loss, so you need \`dL/dθ\` for every parameter \`θ\`. GPT-2 small has about 124 million of them. Nudging each one and re-running the model would take 124 million forward passes per update. Automatic differentiation gets every derivative from a single backward pass, which costs about twice the forward pass: a matmul's backward is two matmuls. (The usual training estimate of \`6·N·D\` FLOPs for \`N\` parameters and \`D\` tokens is \`2·N·D\` forward plus \`4·N·D\` backward.)

## Forward mode, reverse mode

The chain rule for \`L = f(g(h(θ)))\` multiplies the local derivatives \`f'·g'·h'\`, and that product can be evaluated in two orders. **Forward mode** starts at one input and carries "how much does this value change when θ₁ changes" forward through every op: one pass per input. **Reverse mode** starts at one output and carries "how much does the loss change when this value changes" backward: one pass per output. Training has millions of inputs and exactly one output, so reverse mode wins by a factor of the parameter count. PyTorch's autograd and JAX's \`grad\` are both reverse mode.

The price is memory: local derivatives such as \`d(a·b)/da = b\` need the forward values, so every intermediate is kept until backward. That activation memory grows with batch size and sequence length; for long sequences it outgrows the weights.

## The record

Every op in this module returns a new \`Tensor\` that remembers three things: the Tensors it was computed from (\`_children\`), a name (\`_op\`) and a closure (\`_backward\`). The closure is the micrograd pattern (Karpathy's 100-line autograd): it reads the gradient of the loss with respect to this op's *output*, sitting in \`this.grad\`, multiplies by the op's local derivative, and **adds** the result into each input's \`.grad\`. For \`c = a.mul(b)\`:

\`\`\`
a.grad += c.grad * b      // d(a*b)/da = b
b.grad += c.grad * a      // d(a*b)/db = a
\`\`\`

Each op knows only its own local rule; the chain rule emerges from running the closures in the right order.

:::predict
\`x = Tensor.from([3], { requiresGrad: true })\`, then \`x.mul(x).sum().backward()\`. What number is in \`x.grad\`?
---
6. The closure for \`mul\` sends \`c.grad · b = 1 · 3\` to the left operand and \`c.grad · a = 1 · 3\` to the right operand. Both operands are the same tensor, so the two contributions add: \`3 + 3 = 6 = 2x\`. Assigning with \`=\` instead of \`+=\` would give 3, and every model with weight sharing (tied embeddings, residual streams) would train wrong.
:::

## Walking the record

A node's closure must not run until its own gradient is complete, which means after every node that consumed it. A post-order depth-first search from the loss lists every node after all of its inputs; running the closures in **reverse** of that order visits every consumer before its inputs. \`backward()\` therefore does four things: check the root is a scalar, seed \`root.grad = 1\` (\`dL/dL\`), reset the gradients of intermediate nodes (they are scratch space for one pass), and run the closures from the root backwards. Leaves are never reset by \`backward\`: their \`.grad\` accumulates across calls until you ask otherwise.

## Why \`zeroGrad\` exists

Accumulation is the right rule inside one pass and a trap across passes. If a training loop forgets to zero the parameter gradients, each step's update is the sum of *every* gradient so far. That is momentum with no friction: the parameters overshoot the minimum and swing back and forth without ever settling (the train step's predict card has the numbers). Frameworks make the reset explicit (\`optimizer.zero_grad()\` in PyTorch) because sometimes you *want* accumulation: summing gradients over several micro-batches is how large models train when a full batch does not fit in memory.

## Broadcasting, backwards

A bias of shape \`[3]\` added to a \`[64, 3]\` matrix is copied to 64 rows in the forward pass. Copying is the transpose of summing, so the backward pass sums the 64 rows of the output gradient into 3 bias gradients. \`unbroadcast(grad, gradShape, targetShape)\` does that in general: sum away leading dimensions the input never had, then sum dimensions the input had as size 1 back down to 1.

## The fused cross-entropy

Computing \`log(softmax(x))\` as two ops is a mistake in float32: \`exp(1000)\` overflows and \`log(0)\` for a confident wrong prediction is \`−Infinity\`. Fusing them into one op with the row maximum subtracted inside keeps every number finite, and the gradient collapses to \`(softmax − onehot) / N\`. PyTorch's \`F.cross_entropy\` is a stable \`log_softmax\` followed by \`nll_loss\` for exactly this reason; Liger Kernel's fused kernels go further and write the gradient over the logits in the same pass as the loss.

:::predict
You run your gradient check on a correct implementation with \`eps = 1e-8\` and \`tol = 1e-6\`, as you would in float64. Does it pass?
---
No. Float32 carries about 7 significant digits, so \`x + 1e-8\` rounds back to \`x\` for any \`x\` near 1: both perturbed losses are identical and the numeric gradient is 0 (or 0/0). Use \`eps ≈ 1e-3\` and \`tol ≈ 1e-2\`, and divide by the perturbation that actually landed in float32 rather than by \`2·eps\`.
:::

## Where this toy differs from production

This engine keeps the whole graph alive after \`backward()\`; PyTorch frees it by default and makes you ask for \`retain_graph\`. It has no in-place operations, so it needs none of the version counters PyTorch uses to catch a tensor modified after being recorded. It differentiates only a scalar root; production engines accept a vector "seed" gradient so a non-scalar output can be back-propagated. It records everything, always; the reference in \`lib/tensor.js\` adds \`noGrad\` so inference leaves no record, and real frameworks add fused kernels, higher-order derivatives and activation checkpointing (recompute intermediates during backward instead of storing them) to fit long sequences into memory.
`,
  steps: [
    {
      id: 'graph',
      title: 'Walk the graph: topoSort and backward',
      instructions: `
Two functions. Everything else in the file (forward ops, \`accumulate\`, \`fromOp\`, the fully worked \`add\`) is done; read \`fromOp\` and \`add\` first, because every later step follows their pattern.

\`topoSort(root)\`: return an array of every Tensor reachable from \`root\` through \`_children\`, each exactly once, with every tensor listed **after** all of its children and \`root\` last. A post-order depth-first search with a \`Set\` of visited nodes does it.

\`Tensor.prototype.backward()\`: the reverse-mode pass. The two error checks are written. Then: get the topological order; set \`.grad = null\` on every node that has children (intermediate gradients are scratch space for one pass, so two calls to \`backward()\` give exactly twice the leaf gradients); seed this tensor's gradient with 1 using \`accumulate(this, Float32Array.of(1))\`; walk the order from the end to the start and call each node's \`_backward()\` where it is not \`null\`.

Why this order: a node's \`_backward\` reads \`this.grad\`, which is only complete once every node that used it has run. Reverse post-order guarantees that.
`,
      predict: { question: 'With `x = Tensor.from([2], { requiresGrad: true })`, what does `x.grad` hold after `x.add(x).add(x).backward()`?', answer: '[3]. Three paths lead from the loss to x (two through the inner add, one direct), and each contributes 1. The contributions add because accumulate uses +=.' },
      hints: [
        'Before a node can go into the list, what has to be in the list already? Now picture a diamond, where one node is reachable from the root along two paths: what stops it from being listed twice? For backward: the finished list has the root at one end, and the gradient of the loss starts at the root.',
        'topoSort: keep an output list and a set of nodes already visited. Write a recursive helper that returns at once for a visited node; otherwise it marks the node, recurses into each of its children, and only then appends the node. Call it on the root. backward: after the two checks, get the order, clear the gradient of every node that has children, seed the root with a gradient of 1 through accumulate, then walk the order from the root end towards the leaves, calling each non-null _backward.',
        '```js\n// topoSort\nconst order = [], seen = new Set();\nfunction visit(t) {\n  if (seen.has(t)) return;\n  seen.add(t);\n  /* … the children first, then t itself … */\n}\nvisit(root);\nreturn order;\n\n// backward, after the two checks\nconst order = topoSort(this);\nfor (const t of order) if (t._children.length > 0) t.grad = null;\naccumulate(this, Float32Array.of(1));\nfor (let i = /* … which end, and which direction? … */) {\n  if (order[i]._backward !== null) order[i]._backward();\n}\n```',
      ],
    },
    {
      id: 'matmul',
      title: 'Backward for sum, mean and matmul',
      instructions: `
Fill in the three closures. Each receives \`g\`, a raw tensor \`{ shape, data }\` holding the gradient of the loss with respect to the op's **output**, and must call \`accumulate(input, dInput)\` with a \`Float32Array\` the size of the input.

\`sum(axis)\`: every element that was summed receives the gradient of the sum it went into. The helper \`expandAlongAxis(g.data, this.shape, axis, factor)\` spreads a reduced gradient back over the input shape; use factor 1.

\`mean(axis)\`: the same, divided by the number of elements averaged (\`this.data.length\` when \`axis\` is \`null\`, otherwise \`this.shape[axis]\`). Compute that count before the closure, from the input shape.

\`matmul(o)\`: for \`C = A·B\` with output gradient \`dC\`: \`dA = dC·Bᵀ\` and \`dB = Aᵀ·dC\`. Use \`ops.matmul\` and \`ops.transpose\` (raw kernels from module 01); \`this\` and \`o\` are valid raw tensors because they have \`shape\` and \`data\`. Skip the product for an operand whose \`requiresGrad\` is false: \`accumulate\` would ignore it anyway, and a matmul is the most expensive thing in the file. Write the shapes down before you write the code: \`dA\` must have the shape of \`A\`.
`,
      predict: { question: 'x has shape [2, 3]. After `x.mean().backward()`, what is in `x.grad`?', answer: 'Six copies of 1/6. The mean is (1/6)·Σx, so each element has derivative 1/6. Forgetting the division gives six 1s, and a training loop on the mean loss would then take steps 6× too large.' },
      hints: [
        'A reduction throws information away in the forward pass, so its backward pass copies: one incoming number spreads to every element that was reduced. For mean, what constant multiplies each derivative? For matmul, write dC as [n, m] and ask which product of dC and B has the shape [n, k] of A.',
        'sum: every element of the input receives the output gradient of the sum it fed into, unchanged; expandAlongAxis does exactly that spreading when the factor is 1. mean: the same spreading, with every value multiplied by one over the number of elements averaged; count them from the input shape before building the closure (all elements when axis is null, the length of that axis otherwise). matmul: dA is the output gradient times the transpose of B, and dB is the transpose of A times the output gradient, each computed only when that operand requires a gradient. The raw kernels return raw tensors, and accumulate wants their data arrays.',
        '```js\nmatmul(o) {\n  return fromOp(ops.matmul(this, o), \'matmul\', tensorInputs(this, o), (g) => {\n    if (this.requiresGrad) accumulate(this, ops.matmul(g, ops.transpose(o)).data);\n    if (o.requiresGrad) accumulate(o, /* … the other product: A transposed, on the other side of g … */.data);\n  });\n}\n```',
      ],
    },
    {
      id: 'broadcast',
      title: 'Broadcasting backwards: unbroadcast and the rest of mul',
      instructions: `
\`unbroadcast(grad, gradShape, targetShape)\`: \`grad\` is a \`Float32Array\` laid out as \`gradShape\`, the shape of an op's output. Return a \`Float32Array\` laid out as \`targetShape\`, the shape of the input that was broadcast to produce that output, by summing over every dimension broadcasting created. Two cases, in this order:

1. leading dimensions the input never had (\`gradShape\` is longer than \`targetShape\`): sum over axis 0 until the ranks match;
2. dimensions the input had as size 1 that were stretched: sum over that axis with \`keepDims = true\` so the size-1 dimension stays.

\`ops.sum(t, axis, keepDims)\` from lib/ops.js (the axis-aware version of module 01's \`sum\`) does the summing; wrap \`grad\` as \`{ shape: gradShape, data: grad }\` first. \`add\` and \`sub\` already call \`unbroadcast\`. Same-shape adds have worked since step 1, because the placeholder returns the gradient unchanged when no axis was broadcast; their broadcasting tests (a bias added to a matrix) only start passing now.

Then finish \`mul\`: the gradient for the second operand is \`g · this\` (because \`d(a·b)/db = a\`), unbroadcast to \`o.shape\`, and only when \`o\` is a Tensor. Now \`x.mul(x)\` sends two contributions into the same \`x.grad\`.
`,
      predict: { question: 'A bias of shape [3] is added to a [4, 3] matrix and the result is summed. What shape is `bias.grad`, and what is in it?', answer: 'Shape [3], every entry 4. Each bias entry was copied to 4 rows; each copy receives gradient 1 from the sum; the copies are summed back, giving 4. Returning the [4, 3] gradient unchanged would make accumulate throw a length mismatch.' },
      hints: [
        'Broadcasting copies one value to many output positions in the forward pass. Every one of those positions sends a gradient back to the same value. Summing is the reverse of copying; the only question is which axes to sum over.',
        'Start from the output gradient as a raw tensor. First, while it has more dimensions than the target, sum over its leading axis: those are the dimensions the input never had. Then walk the target axes: wherever the target has size 1 but the gradient does not, that axis was stretched, so sum over it while keeping the dimension. Return the data. For mul, the missing line mirrors the one already written for the first operand: swap which operand multiplies g and which shape you unbroadcast to.',
        '```js\nlet t = { shape: gradShape, data: grad };\nwhile (t.shape.length > targetShape.length) t = ops.sum(t, 0);\nfor (let i = 0; i < targetShape.length; i++) {\n  if (/* … this axis was stretched from size 1 … */) t = ops.sum(t, i, true);\n}\nreturn t.data;\n```',
      ],
    },
    {
      id: 'nonlinear',
      title: 'Nonlinearities and the fused cross-entropy',
      instructions: `
Four closures, all elementwise: \`dx[i] = g.data[i] × (local derivative at element i)\`, then \`accumulate(this, dx)\`.

- \`exp\`: the derivative of \`eˣ\` is \`eˣ\`, which is the forward output already in scope as the raw tensor \`y\` (a \`{ shape, data }\` object, so read \`y.data[i]\`). Use it rather than recomputing.
- \`log\`: the derivative of \`ln x\` is \`1 / x\`; \`x\` is \`this.data[i]\`.
- \`relu\`: the derivative is 1 where \`x > 0\` and 0 elsewhere, including at exactly 0.
- \`crossEntropy\`: the forward pass (fused log-softmax, then pick the target's log-probability, then mean over the \`N\` rows) is written. \`logProbs\` is in scope, and \`softmax\` of row \`i\` is \`exp(logProbs[i·V + j])\`. The gradient with respect to logit \`j\` of row \`i\` is \`(softmax_ij − [j === target_i]) / N\`, times the incoming scalar gradient \`g.data[0]\` (the loss is not always the root: someone may scale it).

Each row of the cross-entropy gradient sums to zero. That is worth checking by hand once: raising every logit of a row by the same amount changes no probability, so it must not change the loss.
`,
      predict: { question: 'What gradient does `relu` pass at an input of exactly 0, and does the choice matter in practice?', answer: '0 here (the test requires it), matching PyTorch. Mathematically the derivative is undefined at 0; any value in [0, 1] is a valid subgradient. In float32 training the input is exactly 0 so rarely that the choice is irrelevant, except for inputs that are 0 by construction, such as padded positions.' },
      hints: [
        'Every one of these is a single loop that fills a new Float32Array the size of the input. The only decisions are the local derivative (write it in the margin for each op) and, for cross-entropy, where the onehot subtraction lands.',
        'exp: incoming gradient times the forward output. log: incoming gradient divided by the input. relu: incoming gradient where the input is strictly positive, zero everywhere else. crossEntropy: work out one scale factor, the incoming scalar gradient divided by the number of rows. Fill every entry with that row’s probability (the exponential of the stored log-probability) times the scale, then subtract the scale once per row, at the target column. Accumulate into the logits, not into the loss.',
        '```js\nconst s = g.data[0] / N;\nconst dLogits = new Float32Array(logProbs.data.length);\nfor (let i = 0; i < N; i++) {\n  for (let j = 0; j < V; j++) dLogits[i * V + j] = /* … the probability, scaled by s … */;\n  dLogits[i * V + ids[i]] -= s;\n}\naccumulate(logits, dLogits);\n```',
      ],
    },
    {
      id: 'gradcheck',
      title: 'The numerical gradient check',
      instructions: `
\`gradCheck(fn, inputs, { eps = 1e-3, tol = 1e-2 })\`: \`fn(...inputs)\` returns a scalar Tensor and every input is a leaf with \`requiresGrad\`. Compare the analytic gradient your engine computes with a central-difference estimate, element by element:

\`\`\`
numeric = (fn(x + eps) − fn(x − eps)) / (2·eps)                 // one element perturbed at a time
relErr  = |analytic − numeric| / max(1, |analytic|, |numeric|)
\`\`\`

Return \`{ ok: maxRelErr <= tol, maxRelErr, details }\` where \`details\` has one \`{ input, index, analytic, numeric, relErr }\` per element of every input, where \`input\` is the 0-based position of that tensor in \`inputs\` and \`index\` is the flat element index within its \`data\`. Throw if an input lacks \`requiresGrad\` or \`fn\` returns a non-scalar.

Clear every input's gradient (\`zeroGrad\`) before your one backward pass, or a gradient left over from an earlier step leaks into the analytic side. If any \`relErr\` is NaN (a closure that divides by zero, say), \`maxRelErr\` must end up NaN and \`ok\` false: \`NaN > x\` is false for every \`x\`, so a plain running maximum skips it silently.

Two float32 details matter. Copy the analytic gradients (\`Float32Array.from(t.grad)\`) before you start perturbing, because each perturbed forward call builds a new graph. And perturb the float32 storage in place, then read the value back: \`x + eps\` rounds, so divide by the difference that actually landed (\`hi − lo\`) rather than by \`2·eps\`. Restore every value you touch.

This function is the tool that finds bugs in every closure you have written so far and every one you will write later (layer norm and attention have subtle ones). Central differences are accurate to order \`eps²\`; a one-sided difference is only order \`eps\`, and one of the tests can tell them apart.
`,
      predict: { question: 'A backward closure claims the derivative of `sum` is 2 instead of 1. What `maxRelErr` does gradCheck report?', answer: '0.5: |2 − 1| / max(1, 2, 1). The error is relative to the larger magnitude once values exceed 1, and absolute below 1, where float32 rounding noise would make a pure relative error meaningless.' },
      hints: [
        'Two phases. Phase one: one forward pass and one backward pass give every analytic gradient at once. Phase two: for every single element of every input, two more forward passes (plus and minus) give one numeric derivative. Which values do you need to have saved before phase two starts modifying the inputs?',
        'Phase one: check every input requires a gradient and clear its old gradient, run fn once, check the result is a scalar, run backward, and copy each input’s gradient (zeros if it is still null). Phase two, for every element of every input: save the value, write value plus eps and read back what the float32 array actually stored, evaluate fn; do the same with minus eps; restore the saved value. The numeric derivative is the change in fn divided by the change in the stored value. Compute the relative error, record a detail entry, and keep a running maximum that a NaN cannot slip past (a NaN compares false with everything).',
        '```js\nconst saved = t.data[index];\nt.data[index] = saved + eps; const hi = t.data[index];\nconst fPlus = fn(...inputs).item();\nt.data[index] = saved - eps; const lo = t.data[index];\nconst fMinus = fn(...inputs).item();\nt.data[index] = saved;\nconst numeric = /* … central difference over the perturbation that actually landed … */;\nconst a = analytic[k][index];\nconst relErr = Math.abs(a - numeric) / Math.max(1, Math.abs(a), Math.abs(numeric));\n```',
      ],
    },
    {
      id: 'train',
      title: 'SGD and linear regression',
      instructions: `
\`sgdStep(params, lr)\`: for every parameter that has a gradient, \`p.data[i] -= lr * p.grad[i]\`. Skip parameters whose \`grad\` is \`null\`. Do **not** clear the gradient here: \`zeroGrad\` is a separate, explicit call, exactly as in PyTorch, and the reference optimizer you will use from module 04 onwards (\`lib/optim.js\`) keeps the same split.

\`trainLinear(xs, ys, { steps = 200, lr = 0.1 })\`: fit \`y ≈ w·x + b\` to \`N\` points by full-batch gradient descent on the mean squared error, starting from \`w = 0, b = 0\`. Build \`X\` and \`Y\` as \`[N, 1]\` Tensors, \`w\` as a \`[1, 1]\` parameter and \`b\` as a \`[1]\` parameter (it is broadcast over the rows by \`add\`). Each step: \`pred = X.matmul(w).add(b)\`, \`loss = mean((pred − Y)²)\` built from \`sub\`, \`mul\` and \`mean\`, record \`loss.item()\`, zero both gradients, \`loss.backward()\`, \`sgdStep\`, record \`w.data[0]\` and \`b.data[0]\`. Return \`{ w, b, losses, ws, bs }\`.

This is the entire training loop of every later module, with a GPT in place of \`w·x + b\` and AdamW in place of \`sgdStep\`.
`,
      predict: { question: 'You forget the two zeroGrad calls in the loop. What happens to the loss over 200 steps?', answer: 'It never converges, but it does not blow up either. The update at step t is the sum of all t gradients so far, which is momentum with no friction: w and b accelerate toward the minimum, overshoot, and swing back and forth forever. With lr = 0.1 on 32 points in [−1, 1] the loss keeps oscillating between about 0.3 and 7 for all 200 steps (with zeroGrad it falls below 0.001). Only when lr × curvature exceeds 4 does each swing grow and end in Infinity.' },
      hints: [
        'sgdStep is two nested loops and a minus sign: the gradient points uphill, so subtract it. trainLinear is a five-line loop: forward, record, zero, backward, step.',
        'sgdStep: for each parameter whose grad is not null, subtract learning rate times gradient from every element of its data. trainLinear: wrap xs and ys as [N, 1] column tensors, create w as a [1, 1] zero parameter and b as a [1] zero parameter. Each step, build the prediction (X times w, plus b), the difference from Y, and the mean of its square; record the loss number; clear both gradients; run backward; take the SGD step; record w and b. Return the final numbers with the three histories.',
        '```js\nfor (let step = 0; step < steps; step++) {\n  const pred = X.matmul(w).add(b);      // [N, 1]; b is broadcast over the rows\n  const diff = pred.sub(Y);\n  const loss = /* … mean squared error, built from diff … */;\n  losses.push(loss.item());\n  w.zeroGrad(); b.zeroGrad();\n  loss.backward();\n  sgdStep([w, b], lr);\n  ws.push(w.data[0]); bs.push(b.data[0]);\n}\n```',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague, without code, why reverse-mode autodiff gets the gradient of a scalar loss with respect to 124 million parameters for about twice the cost of one forward pass, rather than 124 million passes, and what it has to store to do so.',
    'Your `_backward` closures accumulate with `+=`. Give one example inside a single backward pass where assignment would be wrong, and one example across training steps where accumulation is wrong unless you reset. What does each one look like when it goes wrong?',
    'The fused cross-entropy has the gradient `(softmax − onehot) / N`. Derive it in your own words from the log-softmax, and say why computing `log(softmax(x))` as two ops is both less stable and more expensive.',
  ],
  stretch: [
    'Add `noGrad(fn)` and `detach()` as in `lib/tensor.js`: a flag that stops `fromOp` from recording. PyTorch\'s `torch.no_grad()` does the same, and module 04\'s sampling and module 07\'s `estimateLoss` wrap their forward passes in `noGrad` so evaluation builds no graph.',
    'Free the graph after `backward()` (set `_children = []` and `_backward = null` on every visited node), as PyTorch does unless `retain_graph=True`. Then measure the memory a 1000-step loop retains with and without freeing; in a browser, `performance.memory` in Chrome shows it.',
    'Implement `softmax()` and `logSoftmax()` as separate ops with their own closures, then compare `crossEntropy` against `logits.logSoftmax()` picked and averaged by hand, on logits near 1000. This is the numerical reason PyTorch\'s `F.cross_entropy` and the Liger Kernel fused-loss kernels never compute `log(softmax(x))` as two separate steps.',
    'Implement activation checkpointing: an op that stores no intermediates and instead re-runs its forward inside `_backward`. Megatron-LM and DeepSpeed use this to trade about a third more compute for the activation memory of long sequences.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
