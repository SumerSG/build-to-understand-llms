// Module 06 — The GPT architecture.
//
// A GPT is a stack of identical residual blocks over a "residual stream" x of shape [B,T,C]
// (batch, time, channels). Each block writes two updates into that stream and touches nothing else:
//
//   x = x + attn(ln1(x))   mixes information ACROSS positions (the only place positions talk)
//   x = x + mlp(ln2(x))    transforms EACH position on its own (where most of the parameters live)
//
// Everything else is plumbing: token + position embeddings at the bottom, a final LayerNorm at the top,
// and a language-model head tied to the token embedding (logits = x · wteᵀ).
//
// You build it on lib/tensor.js (module 02's autograd) with the multi-head attention from lib/attention.js
// (module 05). Everything below the worked example is yours to implement.

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

/** Fail loudly when a layer is built without a seeded rng: initialisation must be reproducible. */
function requireRng(next, where) {
  if (typeof next !== 'function') {
    throw new Error(`${where}: pass a seeded rng function as \`next\` (rng(seed) from lib/util.js)`);
  }
}

/** Placeholder for code you have not written yet; the tests report which step it belongs to. */
function todo(step) {
  throw new Error(`TODO: ${step} is not implemented yet`);
}

// ---------- step 1: Linear and Embedding ----------

/** A fully connected layer: y = x · W + b, with W of shape [nIn, nOut] and b of shape [nOut]. */
export class Linear {
  /** W is Gaussian with std 0.02 (GPT-2's choice); b starts at zero; `bias: false` drops it entirely. */
  constructor(nIn, nOut, { bias = true, next, std = 0.02 } = {}) {
    requireRng(next, 'Linear');
    this.nIn = nIn;
    this.nOut = nOut;
    this.weight = null; // TODO: step 1 — Tensor.param(Tensor.randn(...))
    this.bias = null;   // TODO: step 1 — zeros of shape [nOut], or null when bias is false
  }

  /** x [..., nIn] -> [..., nOut]; a 2-D W is shared across every leading dimension of x. */
  forward(x) {
    todo('step 1 (Linear.forward)');
  }

  /** weight, then bias (if any). */
  parameters() {
    todo('step 1 (Linear.parameters)');
  }
}

/** A table of n rows of width d: turning an integer id into a vector is picking a row, no matmul. */
export class Embedding {
  /** The table is Gaussian with std 0.02; there is no bias. */
  constructor(n, d, { next, std = 0.02 } = {}) {
    requireRng(next, 'Embedding');
    this.n = n;
    this.d = d;
    this.weight = null; // TODO: step 1
  }

  /** ids (any nesting of ints) -> [...idsShape, d]; gradients scatter-add back into the rows that were used. */
  forward(ids) {
    todo('step 1 (Embedding.forward)');
  }

  /** The single table. */
  parameters() {
    todo('step 1 (Embedding.parameters)');
  }
}

// ---------- step 2: token + position embeddings ----------

/**
 * The input to the first block: wte[ids] + wpe[0..T-1]. ids is number[][] of shape B×T; the position
 * embedding [T,C] broadcasts over the batch. T may not exceed wpe.n (the block size): there is no row
 * in the table for position blockSize, which is the only reason a GPT has a context limit.
 */
export function embedInputs(wte, wpe, ids) {
  todo('step 2 (embedInputs)');
}

// ---------- step 3: the MLP and the pre-LN block ----------

/** The per-position feed-forward network: widen C -> 4C, GELU, narrow 4C -> C. */
export class MLP {
  /** fc holds 4C² weights and proj another 4C²: two thirds of every block's parameters are here. */
  constructor(nEmbd, { next } = {}) {
    this.fc = null;   // TODO: step 3
    this.proj = null; // TODO: step 3
  }

  /** x [..., C] -> [..., C]. Every position is transformed on its own; no position sees another. */
  forward(x) {
    todo('step 3 (MLP.forward)');
  }

  /** fc (weight, bias) then proj (weight, bias). */
  parameters() {
    todo('step 3 (MLP.parameters)');
  }
}

/** One pre-LN transformer block: normalise, mix across positions, add; normalise, transform, add. */
export class Block {
  /** cfg supplies nEmbd and nHead; construct attn before mlp so the seeded init order is reproducible. */
  constructor(cfg, { next } = {}) {
    this.ln1 = new LayerNorm(cfg.nEmbd);
    this.attn = null; // TODO: step 3 — new MultiHeadAttention({ nEmbd, nHead, next })
    this.ln2 = new LayerNorm(cfg.nEmbd);
    this.mlp = null;  // TODO: step 3
  }

  /** x [B,T,C] -> [B,T,C]. The two residual adds ARE the residual stream; the LayerNorms sit inside the branches. */
  forward(x) {
    todo('step 3 (Block.forward)');
  }

  /** ln1, attn, ln2, mlp — the order lib/gpt.js paramNames() and the checkpoints use. */
  parameters() {
    todo('step 3 (Block.parameters)');
  }
}

// ---------- step 4: the GPT ----------

/** A decoder-only transformer: embeddings, a stack of blocks, a final LayerNorm and a tied LM head. */
export class GPT {
  /** The whole shape lives in config; `seed` makes every initial weight reproducible. */
  constructor({ vocabSize, blockSize, nLayer, nHead, nEmbd, seed = 0 }) {
    this.config = { vocabSize, blockSize, nLayer, nHead, nEmbd, seed };
    const next = rng(seed);
    this.wte = null;    // TODO: step 4 — token table [vocabSize, nEmbd]
    this.wpe = null;    // TODO: step 4 — position table [blockSize, nEmbd]
    this.blocks = [];   // TODO: step 4 — nLayer Blocks, all built from the same `next`
    this.lnF = new LayerNorm(nEmbd);
  }

  /** ids number[][] (B×T, T <= blockSize) -> logits Tensor [B,T,V]: one next-token prediction per position. */
  forward(ids) {
    todo('step 4 (GPT.forward)');
  }

  /** Every trainable tensor, in the fixed order wte, wpe, blocks[0..L-1], lnF. The tied head is not repeated. */
  parameters() {
    todo('step 4 (GPT.parameters)');
  }

  /** Total number of trainable scalars. */
  numParams() {
    todo('step 4 (GPT.numParams)');
  }
}

// ---------- step 5: the parameter and compute budget in closed form ----------

/**
 * Where the parameters of a GPT(config) live, component by component:
 * { tokenEmbedding, positionEmbedding, attention, mlp, layerNorm, total }. Every entry must be exact
 * for the model you built above (weights AND biases AND LayerNorm gains/shifts).
 */
export function paramBreakdown({ vocabSize, blockSize, nLayer, nEmbd }) {
  todo('step 5 (paramBreakdown)');
}

/** The closed form: equals new GPT(config).numParams() exactly. */
export function countParams(config) {
  todo('step 5 (countParams)');
}

/**
 * FLOPs to process one token with contextLen tokens in context: 2 per parameter (multiply + add), plus
 * the attention products q·kᵀ and weights·v, which cost 4·L·T·C and are the only term that grows with T.
 */
export function flopsPerToken(config, contextLen) {
  todo('step 5 (flopsPerToken)');
}
