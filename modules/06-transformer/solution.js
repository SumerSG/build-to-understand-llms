// Module 06 — reference solution. The same architecture lives in lib/gpt.js; modules 07+ import that one.
//
// A GPT is a stack of identical residual blocks over a "residual stream" x of shape [B,T,C]. Each block
// writes two updates into that stream and touches nothing else:
//
//   x = x + attn(ln1(x))   mixes information ACROSS positions (the only place positions talk)
//   x = x + mlp(ln2(x))    transforms EACH position on its own (where most of the parameters live)
//
// Everything else is plumbing: token + position embeddings at the bottom, a final LayerNorm at the top,
// and a language-model head tied to the token embedding (logits = x · wteᵀ).
//
// Parameter budget: qkv 3C² + proj C² = 4C² for attention, 4C² up + 4C² down = 8C² for the MLP, so 12C²
// of weights per block and about 12·L·C² for the whole stack, plus V·C for the token table.

import { Tensor } from 'lib/tensor.js';
import { MultiHeadAttention } from 'lib/attention.js';
import { rng } from 'lib/util.js';

// ---------- worked example: LayerNorm (done for you; it sets the conventions every layer follows) ----------

/**
 * Normalise every position to zero mean and unit variance over its C channels, then rescale by gamma and
 * shift by beta. Conventions used by every layer in this file:
 *   - parameters are Tensor leaves made with Tensor.param(...) so the optimizer in module 07 can find them;
 *   - forward(x) builds the autograd graph with Tensor methods (never raw ops, or gradients stop here);
 *   - parameters() lists every trainable tensor in a fixed, documented order.
 */
export class LayerNorm {
  /** gamma starts at 1 and beta at 0, so a fresh LayerNorm is exactly the normalisation itself. */
  constructor(d, { eps = 1e-5 } = {}) {
    this.d = d;
    this.eps = eps;
    this.gamma = Tensor.param(Tensor.ones([d]));
    this.beta = Tensor.param(Tensor.zeros([d]));
  }

  /** x [..., d] -> [..., d], normalised along the last dimension (module 01's layerNorm, with autograd). */
  forward(x) {
    return x.layerNorm(this.gamma, this.beta, this.eps);
  }

  /** gamma first, then beta. */
  parameters() {
    return [this.gamma, this.beta];
  }
}

// ---------- step 1: Linear and Embedding ----------

/** Fail loudly when a layer is built without a seeded rng: initialisation must be reproducible. */
function requireRng(next, where) {
  if (typeof next !== 'function') {
    throw new Error(`${where}: pass a seeded rng function as \`next\` (rng(seed) from lib/util.js)`);
  }
}

/** A fully connected layer: y = x · W + b, with W of shape [nIn, nOut] and b of shape [nOut]. */
export class Linear {
  /** W is Gaussian with std 0.02 (GPT-2's choice); b starts at zero; `bias: false` drops it entirely. */
  constructor(nIn, nOut, { bias = true, next, std = 0.02 } = {}) {
    requireRng(next, 'Linear');
    this.nIn = nIn;
    this.nOut = nOut;
    this.weight = Tensor.param(Tensor.randn([nIn, nOut], next, std));
    this.bias = bias ? Tensor.param(Tensor.zeros([nOut])) : null;
  }

  /** x [..., nIn] -> [..., nOut]; a 2-D W is shared across every leading dimension of x. */
  forward(x) {
    const y = x.matmul(this.weight);
    return this.bias === null ? y : y.add(this.bias);
  }

  /** weight, then bias (if any). */
  parameters() {
    return this.bias === null ? [this.weight] : [this.weight, this.bias];
  }
}

/** A table of n rows of width d: turning an integer id into a vector is picking a row, no matmul. */
export class Embedding {
  /** The table is Gaussian with std 0.02; there is no bias. */
  constructor(n, d, { next, std = 0.02 } = {}) {
    requireRng(next, 'Embedding');
    this.n = n;
    this.d = d;
    this.weight = Tensor.param(Tensor.randn([n, d], next, std));
  }

  /** ids (any nesting of ints) -> [...idsShape, d]; gradients scatter-add back into the rows that were used. */
  forward(ids) {
    return this.weight.embed(ids);
  }

  /** The single table. */
  parameters() {
    return [this.weight];
  }
}

// ---------- step 2: token + position embeddings ----------

/**
 * The input to the first block: wte[ids] + wpe[0..T-1]. ids is number[][] of shape B×T; the position
 * embedding [T,C] broadcasts over the batch. T may not exceed wpe.n (the block size): there is no row
 * in the table for position blockSize, which is the only reason a GPT has a context limit.
 */
export function embedInputs(wte, wpe, ids) {
  if (!Array.isArray(ids) || !Array.isArray(ids[0])) throw new Error('embedInputs: ids must be number[][] (B×T)');
  const time = ids[0].length;
  if (time > wpe.n) throw new Error(`embedInputs: sequence of ${time} exceeds blockSize ${wpe.n}`);
  const positions = [];
  for (let t = 0; t < time; t++) positions.push(t);
  return wte.forward(ids).add(wpe.forward(positions)); // [B,T,C] + [T,C]
}

// ---------- step 3: the MLP and the pre-LN block ----------

/** The per-position feed-forward network: widen C -> 4C, GELU, narrow 4C -> C. */
export class MLP {
  /** fc holds 4C² weights and proj another 4C²: two thirds of every block's parameters are here. */
  constructor(nEmbd, { next } = {}) {
    this.fc = new Linear(nEmbd, 4 * nEmbd, { next });
    this.proj = new Linear(4 * nEmbd, nEmbd, { next });
  }

  /** x [..., C] -> [..., C]. Every position is transformed on its own; no position sees another. */
  forward(x) {
    return this.proj.forward(this.fc.forward(x).gelu());
  }

  /** fc (weight, bias) then proj (weight, bias). */
  parameters() {
    return [...this.fc.parameters(), ...this.proj.parameters()];
  }
}

/** One pre-LN transformer block: normalise, mix across positions, add; normalise, transform, add. */
export class Block {
  /** cfg supplies nEmbd and nHead; construct attn before mlp so the seeded init order is reproducible. */
  constructor(cfg, { next } = {}) {
    this.ln1 = new LayerNorm(cfg.nEmbd);
    this.attn = new MultiHeadAttention({ nEmbd: cfg.nEmbd, nHead: cfg.nHead, next });
    this.ln2 = new LayerNorm(cfg.nEmbd);
    this.mlp = new MLP(cfg.nEmbd, { next });
  }

  /** x [B,T,C] -> [B,T,C]. The two `add`s ARE the residual stream; the LayerNorms sit inside the branches. */
  forward(x) {
    const afterAttention = x.add(this.attn.forward(this.ln1.forward(x)));
    return afterAttention.add(this.mlp.forward(this.ln2.forward(afterAttention)));
  }

  /** ln1, attn, ln2, mlp — the order lib/gpt.js paramNames() and the checkpoints use. */
  parameters() {
    return [
      ...this.ln1.parameters(),
      ...this.attn.parameters(),
      ...this.ln2.parameters(),
      ...this.mlp.parameters(),
    ];
  }
}

// ---------- step 4: the GPT ----------

/** A decoder-only transformer: embeddings, a stack of blocks, a final LayerNorm and a tied LM head. */
export class GPT {
  /** The whole shape lives in config; `seed` makes every initial weight reproducible. */
  constructor({ vocabSize, blockSize, nLayer, nHead, nEmbd, seed = 0 }) {
    this.config = { vocabSize, blockSize, nLayer, nHead, nEmbd, seed };
    const next = rng(seed);
    this.wte = new Embedding(vocabSize, nEmbd, { next }); // token table, also the LM head (tied)
    this.wpe = new Embedding(blockSize, nEmbd, { next }); // one learned vector per position slot
    this.blocks = [];
    for (let i = 0; i < nLayer; i++) this.blocks.push(new Block({ nEmbd, nHead }, { next }));
    this.lnF = new LayerNorm(nEmbd);
  }

  /** ids number[][] (B×T, T <= blockSize) -> logits Tensor [B,T,V]: one next-token prediction per position. */
  forward(ids) {
    let x = embedInputs(this.wte, this.wpe, ids);
    for (const block of this.blocks) x = block.forward(x);
    x = this.lnF.forward(x);
    // Tied head: score each position's vector against every token's embedding. No new parameters.
    return x.matmul(this.wte.weight.transpose());
  }

  /** Every trainable tensor, in the fixed order wte, wpe, blocks[0..L-1], lnF. The tied head is not repeated. */
  parameters() {
    const params = [...this.wte.parameters(), ...this.wpe.parameters()];
    for (const block of this.blocks) params.push(...block.parameters());
    params.push(...this.lnF.parameters());
    return params;
  }

  /** Total number of trainable scalars. */
  numParams() {
    let total = 0;
    for (const p of this.parameters()) total += p.size;
    return total;
  }
}

// ---------- step 5: the parameter and compute budget in closed form ----------

/**
 * Where the parameters of a GPT(config) live, component by component. Every entry is exact for the
 * model built above: V·C token table, T·C position table, per block 4C²+4C attention and 8C²+5C MLP,
 * and 2C per LayerNorm (two per block plus the final one).
 */
export function paramBreakdown({ vocabSize, blockSize, nLayer, nEmbd }) {
  const C = nEmbd;
  const tokenEmbedding = vocabSize * C;
  const positionEmbedding = blockSize * C;
  const attention = nLayer * (4 * C * C + 4 * C);
  const mlp = nLayer * (8 * C * C + 5 * C);
  const layerNorm = nLayer * 4 * C + 2 * C;
  const total = tokenEmbedding + positionEmbedding + attention + mlp + layerNorm;
  return { tokenEmbedding, positionEmbedding, attention, mlp, layerNorm, total };
}

/** The closed form: V·C + T·C + L·(12C² + 13C) + 2C. Equals new GPT(config).numParams() exactly. */
export function countParams(config) {
  return paramBreakdown(config).total;
}

/**
 * FLOPs to process one token with contextLen tokens in context: every weight is used once in a
 * multiply-add (2 FLOPs per parameter), and attention adds 4·L·T·C for the score and value products
 * (each 2·T·C per layer), which grow with the context and with nothing else.
 */
export function flopsPerToken(config, contextLen) {
  return 2 * countParams(config) + 4 * config.nLayer * contextLen * config.nEmbd;
}
