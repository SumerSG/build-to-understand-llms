# Shared library API (`lib/`)

`lib/` holds the vetted reference implementation that modules build on. Module N's learner code imports
the reference for modules < N (see docs/PEDAGOGY.md, "Isolation of failure"). Everything is plain ES
modules with **no dependencies**, runs in Node 22 and in a browser Web Worker, and is deterministic given
a seed. Signatures below are the contract; implementations must match them exactly.

Conventions: a **raw tensor** is `{ shape: number[], data: Float32Array }`, row-major. Batched sequences
are `[B, T, C]` (batch, time, channels). Attention heads are `[B, H, T, dh]`. Token ids are plain JS
arrays of integers. Random draws take a `next` function from `rng(seed)`.

## `lib/util.js` (done)
`rng(seed) → next`, `randn(next)`, `randInt(next, n)`, `choice(next, arr)`, `shuffle(next, arr)`,
`assert(cond, msg)`, `approx(a, b, tol)`, `sumArray`, `meanArray`, `argmaxArray`, `softmaxArray(logits, temperature)`,
`sampleIndex(probs, u)`, `fmt(n, digits)`, `now()`, `hash32(str)`.

## `lib/ops.js` (done) — raw kernels, no autograd
Creation: `size(shape)`, `raw(shape, data?)`, `zeros`, `ones`, `full(shape, v)`, `randn(shape, next, std=1)`,
`fromArray(nested)`, `toArray(t)`, `clone(t)`, `reshape(t, shape)` (supports one `-1`).
Elementwise with numpy broadcasting: `binary(a, b, fn)`, `unary(a, fn)`, `add, sub, mul, div` (b may be a number),
`scale(a, s)`, `neg, exp, log, sqrt, tanh, relu, sigmoid, gelu, pow(a, p)`, `geluScalar(x)`.
Linear algebra: `matmul(a, b)` (2D or batched; b may be 2D and shared), `transpose(a)` (last two dims),
`permute(a, order)`.
Reductions: `reduce(a, axis, init, fn, finish, keepDims)`, `sum(a, axis=null, keepDims=false)`, `mean`, `max`,
`argmax(a)` (last dim → plain int array).
Row ops (last dim): `softmax`, `logSoftmax`, `layerNorm(a, gamma=null, beta=null, eps=1e-5)`.
Indexing: `embed(table, ids)`, `causalMask(n)` (1 = allowed), `maskedFill(a, mask, value)` (fills where mask==0),
`slice(a, axis, start, end)`, `concat(a, b, axis)`, `broadcastShapes(a, b)`, `allClose(a, b, tol=1e-4)`.

## `lib/tensor.js` — reverse-mode autograd on top of ops
```js
export class Tensor {
  constructor(raw, { requiresGrad = false } = {})   // wraps raw (no copy)
  shape; data; grad /* Float32Array|null */; requiresGrad; _children; _backward; _op
  static from(nested, opts); static zeros(shape, opts); static ones(shape, opts); static full(shape, v, opts)
  static randn(shape, next, std = 1, opts); static param(raw)  // requiresGrad: true
  get size(); item(); toArray(); detach(); zeroGrad()
  add(o) sub(o) mul(o) div(o)          // o: Tensor | number; broadcasting; grads unbroadcast (summed) to input shapes
  scale(s) neg() matmul(o) transpose() permute(order) reshape(shape)
  exp() log() tanh() relu() gelu() sqrt() pow(p)
  sum(axis = null, keepDims = false) mean(axis = null, keepDims = false)
  softmax() logSoftmax()               // last dim; softmax backward: g_x = p * (g - sum(g*p))
  layerNorm(gamma /* Tensor|null */, beta, eps = 1e-5)
  maskedFill(mask /* raw|Tensor */, value)
  embed(ids)                           // this is the [V,d] table; backward scatter-adds into table.grad
  slice(axis, start, end)
  backward()                           // requires size 1; seeds grad=1; topo order; accumulates into leaf .grad
}
export function crossEntropy(logits /* Tensor [N,V] or [B,T,V] */, targets /* int array, any nesting */) → Tensor scalar (mean NLL)
export function unbroadcast(grad /* Float32Array */, gradShape, targetShape) → Float32Array
export function gradCheck(fn /* (...tensors) => scalar Tensor */, inputs /* Tensor[] */, { eps = 1e-3, tol = 1e-2 } = {})
  → { ok: boolean, maxRelErr: number, details: [{ index, analytic, numeric, relErr }] }  // central differences
export function noGrad(fn)             // run fn with graph recording disabled; returns fn()
export function isGradEnabled()
```
Tensor data is Float32Array; numeric gradient checks therefore use eps ≈ 1e-3 and tolerance ≈ 1e-2.

## `lib/optim.js`
```js
export class SGD   { constructor(params, { lr = 0.01, momentum = 0 } = {}); step(); zeroGrad(); lr }
export class AdamW { constructor(params, { lr = 1e-3, betas = [0.9, 0.95], eps = 1e-8, weightDecay = 0 } = {}); step(); zeroGrad(); lr; t }
export function clipGradNorm(params, maxNorm) → totalNorm   // scales grads in place if norm > maxNorm
export function cosineWithWarmup(step, { warmup, total, peak, min = peak / 10 }) → lr
```
AdamW applies decoupled weight decay (`p -= lr * wd * p`) and bias-corrected moments, as in Loshchilov & Hutter.

## `lib/tokenizer.js`
```js
export class CharTokenizer {
  constructor(text)                    // vocab = sorted unique chars of text
  vocabSize; itos: string[]; stoi: Map
  encode(str) → number[]; decode(ids) → string; toJSON(); static fromJSON(o)
}
export const PRETOKEN_RE = / ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+/g   // GPT-2-style, simplified
export class BPETokenizer {
  static train(text, { vocabSize, specials = ['<|endoftext|>'] } = {}) → BPETokenizer
     // base vocab = unique characters of text (sorted); then greedy pair merges within pre-tokens
     // until vocab.length === vocabSize or no pair occurs twice; specials appended last
  vocab: string[]; merges: Array<[string, string]>; specials: string[]; vocabSize; eos /* id of specials[0] */
  encode(str) → number[]              // specials in the text are matched first and emitted as single ids
  decode(ids) → string
  toJSON(); static fromJSON(o)
}
```

## `lib/attention.js`
```js
export function attention(q, k, v, { causal = true, scale = null } = {})
  // q,k,v: Tensor [B,H,T,dh] (or [T,dh]); scale defaults to 1/sqrt(dh)
  → { out: Tensor /* same shape as q */, weights: Tensor /* [B,H,T,T] softmaxed */ }
export class MultiHeadAttention {
  constructor({ nEmbd, nHead, next })  // qkv: Linear(nEmbd, 3*nEmbd), proj: Linear(nEmbd, nEmbd)
  forward(x /* Tensor [B,T,C] */) → Tensor [B,T,C]   // causal
  parameters() → Tensor[]
  lastWeights                          // raw [B,H,T,T] from the most recent forward (for visualisation)
}
```

## `lib/gpt.js`
```js
export class Linear    { constructor(nIn, nOut, { bias = true, next, std = 0.02 } = {}); weight /* [nIn,nOut] */; bias; forward(x); parameters() }
export class Embedding { constructor(n, d, { next, std = 0.02 } = {}); weight /* [n,d] */; forward(ids); parameters() }
export class LayerNorm { constructor(d); gamma; beta; forward(x); parameters() }
export class MLP       { constructor(nEmbd, { next } = {}); fc; proj; forward(x); parameters() }        // fc → gelu → proj, 4× hidden
export class Block     { constructor(cfg, { next } = {}); ln1; attn; ln2; mlp; forward(x); parameters() }  // pre-LN residual
export class GPT {
  constructor({ vocabSize, blockSize, nLayer, nHead, nEmbd, seed = 0 })
  config; wte; wpe; blocks; lnF        // lm head tied to wte: logits = x · wteᵀ
  forward(ids /* number[][] (B×T) */) → Tensor logits [B,T,V]
  parameters() → Tensor[]; numParams() → number
  generate(ids /* number[] */, { maxNewTokens, temperature = 1, topK = null, next }) → number[]  // recompute each step, inside noGrad
  toJSON() → { config, params: { [name]: { shape, data: number[] } } }
  static fromJSON(o) → GPT
}
export function paramNames(model) → string[]   // stable names like 'wte.weight', 'blocks.0.attn.qkv.weight'
```

## `lib/infer.js` — forward-only inference on raw tensors (the base for the inference track)
```js
export function loadModel(json /* GPT.toJSON() */) → InferModel { config, w: { [name]: raw } }
export function forward(model, ids /* number[] */) → raw [T, V]      // full recompute, no cache
export function newCache(model) → { k: raw[nLayer] [H,0,dh]…, v: …, length: 0 }   // reference KV cache
export function forwardStep(model, cache, id) → Float32Array [V]      // appends one token; returns its logits
export function prefill(model, cache, ids) → Float32Array [V]         // runs all ids through the cache
export function flopsPerToken(config, contextLen, { cached }) → number  // 2·params + attention terms
```

## `lib/sampling.js`
```js
export function applyTemperature(logits, t) → Float32Array
export function topKFilter(logits, k) → Float32Array          // others set to -Infinity
export function topPFilter(logits, p) → Float32Array          // nucleus, on probabilities computed inside
export function minPFilter(logits, p) → Float32Array
export function repetitionPenalty(logits, prevIds, penalty) → Float32Array   // CTRL-style: divide positive, multiply negative
export function softmaxLogits(logits) → Float32Array
export function sample(logits, { temperature = 1, topK = 0, topP = 1, minP = 0, next }) → id
export function greedy(logits) → id
export function generate(model /* InferModel */, tokenizer, prompt, { maxNewTokens = 50, next, stopAtEos = true, ...samplingOpts }) → string
```

## `lib/data.js`
```js
export const CORPUS: string                       // ≈43 KB: toyCorpus(800, 1) + '\n\n' + PROSE
export const PROSE: string                        // ≈12 KB public-domain English prose/verse
export function toyCorpus(nSentences = 4000, seed = 1) → string   // procedural grammar with clear structure
export const INSTRUCTIONS: Array<{ prompt: string, response: string }>      // ≥ 60 short pairs
export const PREFERENCES: Array<{ prompt: string, chosen: string, rejected: string }>  // ≥ 40
export const MATH_TASKS: Array<{ question: string, answer: string }>       // ≥ 100 one-step arithmetic tasks (verifiable rewards)
export function getBatch(ids /* number[] */, { blockSize, batchSize, next }) → { x: number[][], y: number[][] }
export function trainValSplit(ids, frac = 0.9) → { train, val }   // contiguous: the last (1-frac) of the tokens
export function interleavedSplit(ids, { chunk = 256, holdOut = 10 } = {}) → { train, val }  // every holdOut-th chunk held out (representative of all sources)
export const CHAT = { system: '<|system|>', user: '<|user|>', assistant: '<|assistant|>', end: '<|end|>' }
export function formatChat(messages /* [{role, content}] */) → string
```

## `lib/harness.js` — reference agent loop
```js
export class ToolRegistry { register(name, { description, parameters, handler }); schemas(); has(name); async call(name, args) }
export function parseToolCalls(text) → Array<{ name, args }>   // <tool_call>{"name":"x","args":{...}}</tool_call>
export async function runAgentLoop({ model /* async (messages, tools) => string */, tools, messages, maxTurns = 8, onEvent })
  → { messages, turns, stopReason: 'final' | 'max_turns' | 'error' }
export function mockModel(script /* string[] | (messages) => string */) → model
```

## Checkpoints
`lib/checkpoints/tiny-gpt.json` — a GPT trained by `tools/pretrain.mjs` on `CORPUS` with the
`lib/checkpoints/tokenizer.json` BPE tokenizer (vocabSize 256, blockSize 64, nLayer 2, nHead 4, nEmbd 64).
Modules that need a trained model load these so the learner can skip ahead without having trained one.
