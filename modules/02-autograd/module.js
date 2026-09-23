export default {
  id: '02-autograd',
  title: 'Autograd from scratch',
  track: 'foundations',
  minutes: 180,
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
:::plain
Training a model means adjusting its parameters, the adjustable numbers inside it, until its predictions are good. One number, the loss, measures how bad the predictions currently are. For each parameter, the gradient says which way to nudge it, and how strongly, to make the loss smaller. Nudging every parameter one at a time to find out would take forever for a model with millions of them, so frameworks record every calculation as it happens and then walk that record backwards, getting every gradient in one sweep. That backward sweep is called automatic differentiation. In this module you build it, then use it to fit a straight line through noisy points and to train a small classifier.
:::

## Start with a line

Here is the smallest model worth training. You have some points that lie roughly on a straight line, and you want the line \`y = w·x + b\` that fits them best: \`w\` is its slope and \`b\` is where it crosses the y axis. A spreadsheet's trendline does exactly this. \`w\` and \`b\` are the model's two **parameters**.

To say how wrong a guess is, compute the **loss**: for every point, take the prediction \`w·x + b\` minus the true \`y\`, square it (so misses in both directions count, and big misses count more), and average over the points. This is the *mean squared error*; a perfect line has loss 0.

Take two points on the line \`y = 3x + 2\`: \`(1, 5)\` and \`(2, 8)\`. Guess \`w = 0, b = 0\`. The predictions are 0 and 0, the errors are −5 and −8, and the loss is \`(25 + 64) / 2 = 44.5\`. Now nudge \`w\` up to 0.001 and recompute: the loss becomes 44.479, down by 0.021. The ratio \`−0.021 / 0.001 = −21\` is the **derivative** of the loss with respect to \`w\`, written \`dL/dw\`: the slope of the loss when you plot it against \`w\`. It is negative, so a larger \`w\` lowers the loss. Do the same for \`b\` and you have the **gradient**, the list of the loss's slopes with respect to every parameter. **Gradient descent** then takes a small step downhill, \`w ← w − lr · dL/dw\` (and the same for \`b\`), where the *learning rate* \`lr\` is a small number such as 0.1, and repeats until the loss stops falling. The last build step does exactly this and recovers \`w ≈ 3, b ≈ 2\`.

## Five lines of calculus

Nudging works, but it costs a full recomputation of the loss per parameter. Calculus gives the slope directly. Everything this module needs fits in five lines:

\`\`\`
slope        f'(x) is the slope of f at x: rise over run, for a tiny run
powers       x² → 2x     x³ → 3x²     a·x → a     a constant → 0
exp, log     eˣ → eˣ     ln x → 1/x
sums         (f + g)' = f' + g'
chain rule   y = f(u) and u = g(x)  give  dy/dx = f'(u) · g'(x)
\`\`\`

The chain rule in words: to get the slope through a chain of steps, multiply the slopes of the steps. Check it on the example. For one point the loss is \`u²\` with \`u = w·x + b − y\`. The slope of \`u²\` with respect to \`u\` is \`2u\`, and the slope of \`u\` with respect to \`w\` is \`x\`, so the chain rule gives \`2u · x\`. The first point has \`u = −5, x = 1\`, giving −10; the second has \`u = −8, x = 2\`, giving −32; their average is −21, the number the nudge found.

## One loss, a million knobs

GPT-2 small has about 124 million parameters. Nudging each one and re-running the model would take 124 million runs per update. Automatic differentiation gets every derivative from a single backward pass, which costs about twice the forward pass (the forward pass is the ordinary calculation of the loss from the inputs). It works by applying the chain rule in reverse: start at the loss, whose slope with respect to itself is 1, and walk back through the calculation, multiplying by one local slope at each step, until every parameter has its number. Because the walk starts from the one output and fans out to all the inputs, one pass serves every parameter at once. This is called **reverse mode**, and it is what PyTorch's autograd and JAX's \`grad\` do.

:::deeper Going deeper: forward mode, the 6·N·D rule and activation memory
The chain rule for \`L = f(g(h(θ)))\` multiplies the local derivatives \`f'·g'·h'\`, and that product can be evaluated in two orders. **Forward mode** starts at one input and carries "how much does this value change when θ₁ changes" forward through every op: one pass per input. **Reverse mode** starts at one output and carries "how much does the loss change when this value changes" backward: one pass per output. Training has millions of inputs and exactly one output, so reverse mode wins by a factor of the parameter count. The usual training estimate of \`6·N·D\` FLOPs for \`N\` parameters and \`D\` tokens is \`2·N·D\` forward plus \`4·N·D\` backward: a matmul's backward is two matmuls. The price is memory: local derivatives such as \`d(a·b)/da = b\` need the forward values, so every intermediate is kept until backward. That activation memory grows with batch size and sequence length; for long sequences it outgrows the weights.
:::

## The record

Every op in this module returns a new \`Tensor\` that remembers three things: the Tensors it was computed from (\`_children\`), a name (\`_op\`) and a closure (\`_backward\`), which is a function created inside the op that still remembers the op's inputs when it runs later. The closure is the micrograd pattern (Karpathy's 100-line autograd): it reads the gradient of the loss with respect to this op's *output*, sitting in \`this.grad\`, multiplies by the op's local derivative, and **adds** the result into each input's \`.grad\`. For \`c = a.mul(b)\`:

\`\`\`
a.grad += c.grad * b      // d(a*b)/da = b
b.grad += c.grad * a      // d(a*b)/db = a
\`\`\`

Here is the smallest interesting record, for \`L = a·b + a\`. The arrows point from inputs to the values made from them:

\`\`\`
 a ──┐
     ├──► c = a · b ──┐
 b ──┘                ├──► L = c + a
 a ───────────────────┘      (a is used twice)
\`\`\`

Listed so that every node comes after its inputs, the order is \`a, b, c, L\`. The backward pass runs it the other way. \`L\` starts with gradient 1. \`L\`'s closure (an add) sends 1 to \`c\` and 1 to \`a\`. \`c\`'s closure (a mul) sends \`1 · b\` to \`a\` and \`1 · a\` to \`b\`. So \`a.grad = 1 + b\`: two paths lead from \`L\` to \`a\`, and their contributions add. Each op knows only its own local rule; the chain rule emerges from running the closures in the right order.

:::predict
\`x = Tensor.from([3], { requiresGrad: true })\`, then \`x.mul(x).sum().backward()\`. What number is in \`x.grad\`?
---
6. The closure for \`mul\` sends \`c.grad · b = 1 · 3\` to the left operand and \`c.grad · a = 1 · 3\` to the right operand. Both operands are the same tensor, so the two contributions add: \`3 + 3 = 6 = 2x\`. Assigning with \`=\` instead of \`+=\` would give 3, and every model that uses one weight in two places (tied embeddings, residual streams) would train wrong.
:::

## Walking the record

A node's closure must not run until its own gradient is complete, which means after every node that used it has run. A *post-order depth-first search* from the loss produces the right list: to add a node, first add each of its inputs (by calling the same function on them, so it works to any depth), then add the node itself, skipping nodes already listed. Running the closures in **reverse** of that list visits every consumer before its inputs. \`backward()\` therefore does four things: check the root is a single number, seed \`root.grad = 1\` (\`dL/dL\`), reset the gradients of intermediate nodes (they are scratch space for one pass), and run the closures from the root backwards. Leaves (the parameters and inputs at the start of the record) are never reset by \`backward\`: their \`.grad\` accumulates across calls until you ask otherwise.

## Why \`zeroGrad\` exists

Accumulation is the right rule inside one pass and a trap across passes. If a training loop forgets to zero the parameter gradients, each step's update is the sum of *every* gradient so far, like a ball rolling downhill with no friction: the parameters overshoot the minimum and swing back and forth without ever settling (the train step's predict card has the numbers). Accumulating on purpose has its uses, so the reset is a separate, explicit call.

:::deeper Going deeper: accumulation on purpose
Frameworks make the reset explicit (\`optimizer.zero_grad()\` in PyTorch) because sometimes you *want* accumulation: summing gradients over several small batches (micro-batches) is how large models train when a full batch does not fit in memory.
:::

## Broadcasting, backwards

A bias of shape \`[3]\` added to a \`[64, 3]\` matrix is copied to 64 rows in the forward pass (module 01's broadcasting). Every copy influences the loss, so the backward pass sums the 64 rows of the output gradient into 3 bias gradients. \`unbroadcast(grad, gradShape, targetShape)\` does that in general: sum away leading dimensions the input never had, then sum dimensions the input had as size 1 back down to 1.

## The fused cross-entropy

A classifier, and a language model choosing its next token, is trained with the **cross-entropy** loss: minus the log of the probability the model gave to the right answer (the *target*), averaged over the rows. Probability 1 for the right answer costs 0; probability 0.01 costs \`−ln 0.01 ≈ 4.6\`. The probabilities come from module 01's softmax of the scores (logits). Computing \`log(softmax(x))\` as two ops is a mistake in float32: \`exp(1000)\` overflows and \`log(0)\` for a confident wrong prediction is \`−Infinity\`. Fusing them into one op with the row maximum subtracted inside keeps every number finite, and the gradient collapses to \`(softmax − onehot) / N\`, where *onehot* is a row of zeros with a single 1 at the target's position and \`N\` is the number of rows.

:::predict
You run your gradient check on a correct implementation with \`eps = 1e-8\` and \`tol = 1e-6\`, as you would in float64. Does it pass?
---
No. Float32 carries about 7 significant digits, so \`x + 1e-8\` rounds back to \`x\` for any \`x\` near 1: both perturbed losses are identical and the numeric gradient is 0 (or 0/0). Use \`eps ≈ 1e-3\` and \`tol ≈ 1e-2\`, and divide by the perturbation that actually landed in float32 rather than by \`2·eps\`.
:::

:::deeper Going deeper: where this toy differs from production
PyTorch's \`F.cross_entropy\` is a stable \`log_softmax\` followed by \`nll_loss\` for exactly the reason above; Liger Kernel's fused kernels go further and write the gradient over the logits in the same pass as the loss. This engine keeps the whole graph alive after \`backward()\`; PyTorch frees it by default and makes you ask for \`retain_graph\`. It has no in-place operations, so it needs none of the version counters PyTorch uses to catch a tensor modified after being recorded. It differentiates only a scalar root; production engines accept a vector "seed" gradient so a non-scalar output can be back-propagated. It records everything, always; the reference in \`lib/tensor.js\` adds \`noGrad\` so inference leaves no record, and real frameworks add fused kernels, higher-order derivatives and activation checkpointing (recompute intermediates during backward instead of storing them) to fit long sequences into memory.
:::
`,
  steps: [
    {
      id: 'graph',
      title: 'Walk the graph: topoSort and backward',
      instructions: `
Two functions. Everything else in the file (forward ops, \`accumulate\`, \`fromOp\`, the fully worked \`add\`) is done; read \`fromOp\` and \`add\` first, because every later step follows their pattern.

\`topoSort(root)\`: return an array of every Tensor reachable from \`root\` through \`_children\`, each exactly once, with every tensor listed **after** all of its children and \`root\` last. A post-order depth-first search with a \`Set\` of visited nodes does it: a helper function \`visit(t)\` that returns at once if \`t\` is already in the set, otherwise adds it to the set, calls \`visit\` on each of its children (a function may call itself; this is *recursion*, and it stops at leaves, which have no children), and only then pushes \`t\` onto the list. A \`Set\` is a collection that remembers which values it holds: \`seen.add(t)\` and \`seen.has(t)\`. The concept's drawing of \`L = a·b + a\` gives \`a, b, c, L\`.

\`backward()\`, a method of the \`Tensor\` class: the reverse-mode pass. Inside a method, \`this\` is the tensor it was called on, so in \`loss.backward()\` \`this\` is \`loss\`. The two error checks are written. Then: get the topological order; set \`.grad = null\` on every node that has children (intermediate gradients are scratch space for one pass, so two calls to \`backward()\` give exactly twice the leaf gradients); seed this tensor's gradient with 1 using \`accumulate(this, Float32Array.of(1))\`; walk the order from the end to the start and call each node's \`_backward()\` where it is not \`null\`.

Why this order: a node's \`_backward\` reads \`this.grad\`, which is only complete once every node that used it has run. Reverse post-order guarantees that.

Where the build is going: the trendline fit from the concept is the last step (step 7, \`trainLinear\`), because it needs the matmul, mean and broadcasting backward passes of steps 2 and 3 first; steps 4 to 6 add what the goal's classifier and the gradient check need.
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

\`matmul(o)\`: for \`C = A·B\` with output gradient \`dC\`: \`dA = dC·Bᵀ\` and \`dB = Aᵀ·dC\`. The small raised \`ᵀ\` means **transpose** (rows and columns swapped, the \`transpose\` you wrote in module 01), so \`Bᵀ\` is B turned on its side.

**Why those formulas: the smallest case.** Take \`A = [2, 3]\` (shape \`[1, 2]\`) and \`B\` the column \`[4, 5]\` (shape \`[2, 1]\`). Then \`C = 2·4 + 3·5 = 23\`, a single number. Nudge A's first entry from 2 to 2.01 and C becomes 23.04: C moves 4 times as much as that entry, and 4 is B's first entry. By the chain rule, that entry's gradient is then 4 times \`dC\`, the gradient arriving at C. Entry by entry:

\`\`\`
C     = A[0]·B[0] + A[1]·B[1] = 2·4 + 3·5 = 23
dA[0] = B[0]·dC = 4·dC
dA[1] = B[1]·dC = 5·dC   so dA = dC · [4, 5] = dC · Bᵀ
dB[0] = A[0]·dC = 2·dC
dB[1] = A[1]·dC = 3·dC   so dB = [2, 3] stood up
                         as a column, times dC = Aᵀ · dC
\`\`\`

Each entry of A was multiplied by an entry of B, so its gradient is that entry of B times \`dC\`, and the other way round. With more rows and columns every entry takes part in several products, and the matmuls \`dC·Bᵀ\` and \`Aᵀ·dC\` add those contributions up. The shapes confirm it: \`dA\` must have A's shape \`[n, k]\`; \`dC\` is \`[n, m]\` and \`B\` is \`[k, m]\`, so \`dC·Bᵀ\` (\`[n, m] × [m, k]\`) is the only product that fits.

Use \`ops.matmul\` and \`ops.transpose\` (raw kernels from module 01); \`this\` and \`o\` are valid raw tensors because they have \`shape\` and \`data\`. Skip the product for an operand whose \`requiresGrad\` is false: \`accumulate\` would ignore it anyway, and a matmul is the most expensive thing in the file. Write the shapes down before you write the code: \`dA\` must have the shape of \`A\`.
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
- \`relu\` (\`relu(x) = max(x, 0)\`, the simplest way to bend a straight-line model so it can learn curved boundaries): the derivative is 1 where \`x > 0\` and 0 elsewhere, including at exactly 0.
- \`crossEntropy\`: the forward pass (fused log-softmax, then pick the target's log-probability, then mean over the \`N\` rows) is written. \`logProbs\` is in scope as a raw tensor \`{ shape: [N, V], data }\` holding the log of each probability, so the softmax probability of class \`j\` in row \`i\` is \`Math.exp(logProbs.data[i * V + j])\` (index its \`.data\`: the object itself has no numbered entries). The target class of row \`i\` is \`ids[i]\`. The gradient with respect to logit \`j\` of row \`i\` is \`(softmax_ij − onehot_ij) / N\`, where \`onehot_ij\` is 1 when \`j === ids[i]\` and 0 otherwise, times the incoming scalar gradient \`g.data[0]\` (the loss is not always the root: someone may scale it).

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
      id: 'numgrad',
      title: 'A numeric derivative of one element',
      instructions: `
Before trusting the closures you have written, you need an independent way to measure a derivative: the nudge from the concept's line fit, done carefully. Implement \`numericDerivative(fn, inputs, k, index, eps = 1e-3)\`.

\`fn\` is a function that takes the input Tensors and returns a scalar Tensor (a loss). \`fn(...inputs)\` calls it with every element of the array \`inputs\` as a separate argument: with two inputs it means \`fn(inputs[0], inputs[1])\`. \`.item()\` turns the size-1 result into a plain number.

The function measures how \`fn\` changes when one number, element \`index\` of input \`k\` (that is, \`inputs[k].data[index]\`), moves a little each way. It uses a **central difference**, a nudge up and a nudge down:

\`\`\`
numeric = (fn(x + eps) − fn(x − eps)) / (2·eps)         only element index of input k is moved
\`\`\`

Two float32 details matter, and the tests check both:

1. **Perturb the storage in place and read back what landed.** Write \`saved + eps\` into \`inputs[k].data[index]\`, then read the element back as \`hi\`: a \`Float32Array\` rounds to about 7 significant digits, so near 1000.3 the stored value moves by 0.000977, not 0.001. Do the same with \`saved − eps\` to get \`lo\`, and divide by \`hi − lo\`, the perturbation that actually happened, instead of by \`2·eps\`.
2. **Restore the value.** Put \`saved\` back before you return, so the caller's tensor is unchanged.

Central differences are accurate to order \`eps²\`; a one-sided difference \`(fn(x + eps) − fn(x)) / eps\` is only accurate to order \`eps\`, and one of the tests can tell them apart.
`,
      predict: { question: 'With eps = 0.1, what does the central difference give for f(x) = x³ at x = 2 (the true derivative is 3·2² = 12)? What would the one-sided difference (f(2.1) − f(2)) / 0.1 give?', answer: 'Central: (2.1³ − 1.9³) / 0.2 = (9.261 − 6.859) / 0.2 = 12.01, an error of 0.01 (= eps²). One-sided: (9.261 − 8) / 0.1 = 12.61, an error of 0.61, sixty times worse. The errors of the two one-sided halves cancel in the central form.' },
      hints: [
        'The derivative is a slope: a change in the output divided by the change in the input that caused it. Which change in the input can you trust in float32, the one you asked for or the one you can read back?',
        'Take the tensor `inputs[k]` and remember its value at `index`. Write the value plus eps, read the element back, and evaluate fn on all the inputs; write the value minus eps, read it back, and evaluate fn again. Put the saved value back. Return the change in fn divided by the change in the stored value.',
        '```js\nconst t = inputs[k];\nconst saved = t.data[index];\nt.data[index] = saved + eps;\nconst hi = t.data[index];                   // what float32 actually stored\nconst fPlus = fn(...inputs).item();\n/* … the same with saved - eps, giving lo and fMinus … */\nt.data[index] = saved;\nreturn (fPlus - fMinus) / (hi - lo);\n```',
      ],
    },
    {
      id: 'gradcheck',
      title: 'The gradient check: loops and report',
      instructions: `
\`gradCheck(fn, inputs, { eps = 1e-3, tol = 1e-2 })\` compares, for every element of every input, the gradient your engine computes (the *analytic* gradient) with \`numericDerivative\` from the previous step. This is the tool that finds bugs in every closure you have written so far and every one you will write later (layer norm and attention have subtle ones).

This is a completion problem, like \`offset\` in module 01: the starter already has the two loops (over the inputs \`k\`, and over the elements \`index\` of each input), the \`details\` array, the running \`maxRelErr\` and the final \`return\`. You fill in two places.

**Part 1, before the loops: one backward pass.** Throw an \`Error\` if any input has \`requiresGrad\` false (it has no analytic gradient to compare). Clear every input's old gradient with \`zeroGrad()\`, or a gradient left over from an earlier training step leaks into the comparison. Run \`fn(...inputs)\` once, throw if the result's \`size\` is not 1, and call \`backward()\` on it. Then push a *copy* of each input's gradient into \`analytic\`: \`Float32Array.from(t.grad)\`, or \`new Float32Array(t.size)\` (zeros) when \`t.grad\` is still \`null\` because the loss does not depend on that input. Copy rather than keep \`t.grad\` itself, so nothing later can change what you compare against.

**Part 2, inside the loops: one comparison per element.**

\`\`\`
analytic = analytic[k][index]
numeric  = numericDerivative(fn, inputs, k, index, eps)
relErr   = |analytic − numeric| / max(1, |analytic|, |numeric|)
\`\`\`

(\`Math.abs\` and \`Math.max\` do the \`| |\` and the \`max\`.) Push \`{ input: k, index, analytic, numeric, relErr }\` into \`details\`, then update \`maxRelErr\`. If any \`relErr\` is \`NaN\` (a closure that divides by zero, say), \`maxRelErr\` must end up \`NaN\` so \`ok\` is false. That needs care: every comparison with \`NaN\` is false, so \`relErr > maxRelErr\` never lets a NaN in, and once \`maxRelErr\` is NaN, \`relErr > maxRelErr\` is also false for every later element, which is what keeps it there. Let the NaN in with \`Number.isNaN(relErr)\`.

The error is relative to the larger magnitude once values exceed 1, and absolute below 1, where float32 rounding noise would make a pure relative error meaningless.
`,
      predict: { question: 'A backward closure claims the derivative of `sum` is 2 instead of 1. What `maxRelErr` does gradCheck report?', answer: '0.5: |2 − 1| / max(1, 2, 1). The error is relative to the larger magnitude once values exceed 1, and absolute below 1, where float32 rounding noise would make a pure relative error meaningless.' },
      hints: [
        'Two phases. Phase one, before the loops: one forward and one backward pass give every analytic gradient at once. Phase two, in the loops: for each element, the numeric derivative from the previous step, and one comparison. Why must phase one save copies before phase two starts?',
        'Part 1: loop over the inputs to check requiresGrad and call zeroGrad; call fn with all the inputs; check the size of the result and throw if it is not 1; call backward; loop over the inputs again, pushing a copy of each gradient (zeros if it is null) into analytic. Part 2: read the analytic value, compute the numeric one, the relative error, push the detail object, and raise maxRelErr when relErr is NaN or larger than it.',
        '```js\n// part 1\nfor (const t of inputs) {\n  if (!t.requiresGrad) throw new Error(\'gradCheck: every input needs requiresGrad\');\n  t.zeroGrad();\n}\nconst out = fn(...inputs);\n/* … throw unless out.size is 1, then out.backward() … */\nfor (const t of inputs) analytic.push(t.grad ? Float32Array.from(t.grad) : new Float32Array(t.size));\n\n// part 2, inside the loops\nconst a = analytic[k][index];\nconst numeric = numericDerivative(fn, inputs, k, index, eps);\nconst relErr = /* … the formula from the instructions … */;\ndetails.push({ input: k, index, analytic: a, numeric, relErr });\nif (Number.isNaN(relErr) || relErr > maxRelErr) maxRelErr = relErr;\n```',
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
