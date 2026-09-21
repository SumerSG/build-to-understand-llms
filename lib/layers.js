// lib/layers.js — the three parameterised building blocks a transformer is made of.
//
// They live in their own file because both lib/attention.js (module 05, which projects q/k/v with a
// Linear) and lib/gpt.js (module 06, which stacks all three) need them, and a shared file is clearer than
// two files importing each other in a cycle. lib/gpt.js re-exports all three, so
// `import { Linear, Embedding, LayerNorm } from './gpt.js'` works exactly as docs/LIB_API.md describes.
//
// Every layer owns its parameters as Tensor leaves with requiresGrad, hands them out through
// parameters() so an optimizer can collect them, and does its work in forward().

import * as ops from './ops.js';
import { Tensor } from './tensor.js';

/** Fail loudly when a layer is built without a seeded rng: initialisation must be reproducible. */
function requireRng(next, where) {
  if (typeof next !== 'function') {
    throw new Error(`${where}: pass a seeded rng function as \`next\` (rng(seed) from lib/util.js)`);
  }
}

/** A fully connected layer: y = x · W + b, with W of shape [nIn, nOut]. */
export class Linear {
  /** Weights are Gaussian with std (0.02, as in GPT-2); the bias starts at zero. */
  constructor(nIn, nOut, { bias = true, next, std = 0.02 } = {}) {
    requireRng(next, 'Linear');
    this.nIn = nIn;
    this.nOut = nOut;
    this.weight = Tensor.param(ops.randn([nIn, nOut], next, std));
    this.bias = bias ? Tensor.param(ops.zeros([nOut])) : null;
  }

  /** x [..., nIn] -> [..., nOut]; the bias broadcasts over every leading dimension. */
  forward(x) {
    const y = x.matmul(this.weight);
    return this.bias === null ? y : y.add(this.bias);
  }

  /** The trainable tensors of this layer, weight first. */
  parameters() {
    return this.bias === null ? [this.weight] : [this.weight, this.bias];
  }
}

/** A lookup table of n rows of width d: turning an integer id into a vector is just picking a row. */
export class Embedding {
  /** The table is Gaussian with std (0.02); there is no bias and no matmul involved. */
  constructor(n, d, { next, std = 0.02 } = {}) {
    requireRng(next, 'Embedding');
    this.n = n;
    this.d = d;
    this.weight = Tensor.param(ops.randn([n, d], next, std));
  }

  /** ids (any nesting of ints) -> [...idsShape, d]. Gradients scatter-add back into the used rows. */
  forward(ids) {
    return this.weight.embed(ids);
  }

  /** The trainable tensors of this layer. */
  parameters() {
    return [this.weight];
  }
}

/** Normalise every position to zero mean and unit variance over its channels, then rescale and shift. */
export class LayerNorm {
  /** gamma starts at one and beta at zero, so a fresh LayerNorm is exactly the normalisation itself. */
  constructor(d, { eps = 1e-5 } = {}) {
    this.d = d;
    this.eps = eps;
    this.gamma = Tensor.param(ops.ones([d]));
    this.beta = Tensor.param(ops.zeros([d]));
  }

  /** x [..., d] -> [..., d], normalised along the last dimension. */
  forward(x) {
    return x.layerNorm(this.gamma, this.beta, this.eps);
  }

  /** The trainable tensors of this layer, gamma first. */
  parameters() {
    return [this.gamma, this.beta];
  }
}
