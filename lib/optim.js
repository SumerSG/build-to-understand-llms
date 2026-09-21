// lib/optim.js — optimizers and learning-rate schedules. This file is the REFERENCE SOLUTION for the
// training-loop half of module 04.
//
// An optimizer owns a list of parameter Tensors (leaves with requiresGrad) and nothing else. backward()
// has already filled each parameter's .grad; step() turns those gradients into an in-place update of
// .data, and zeroGrad() clears them again before the next pass. Nothing here builds a graph, so the
// updates never show up in an autograd record.

/** Stochastic gradient descent, optionally with classic (non-Nesterov) momentum. */
export class SGD {
  /** params: Tensor[] with requiresGrad. momentum 0 means plain SGD (no velocity buffers allocated). */
  constructor(params, { lr = 0.01, momentum = 0 } = {}) {
    this.params = params;
    this.lr = lr;
    this.momentum = momentum;
    this.velocity = momentum > 0 ? params.map((p) => new Float32Array(p.data.length)) : null;
  }

  /** One update: p -= lr * g, or with momentum v = m*v + g and p -= lr * v. */
  step() {
    for (let k = 0; k < this.params.length; k++) {
      const p = this.params[k];
      if (!p.grad) continue;
      const data = p.data, g = p.grad;
      if (this.velocity === null) {
        for (let i = 0; i < data.length; i++) data[i] -= this.lr * g[i];
      } else {
        const v = this.velocity[k];
        for (let i = 0; i < data.length; i++) {
          v[i] = this.momentum * v[i] + g[i];
          data[i] -= this.lr * v[i];
        }
      }
    }
  }

  /** Drop every parameter's accumulated gradient, ready for the next backward pass. */
  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }
}

/** Adam with decoupled weight decay (Loshchilov & Hutter): bias-corrected moments, decay applied to p directly. */
export class AdamW {
  /** betas are [beta1, beta2] for the first and second moment; weightDecay 0 turns the decay off. */
  constructor(params, { lr = 1e-3, betas = [0.9, 0.95], eps = 1e-8, weightDecay = 0 } = {}) {
    this.params = params;
    this.lr = lr;
    this.beta1 = betas[0];
    this.beta2 = betas[1];
    this.eps = eps;
    this.weightDecay = weightDecay;
    this.t = 0;
    this.m = params.map((p) => new Float32Array(p.data.length));
    this.v = params.map((p) => new Float32Array(p.data.length));
  }

  /** One update: decay p, update both moments, bias-correct them, then p -= lr * mHat / (sqrt(vHat) + eps). */
  step() {
    this.t += 1;
    // Both moments start at zero, so early estimates are biased towards zero; these divisors undo that.
    const correction1 = 1 - Math.pow(this.beta1, this.t);
    const correction2 = 1 - Math.pow(this.beta2, this.t);
    const decay = this.lr * this.weightDecay;
    for (let k = 0; k < this.params.length; k++) {
      const p = this.params[k];
      if (!p.grad) continue;
      const data = p.data, g = p.grad, m = this.m[k], v = this.v[k];
      for (let i = 0; i < data.length; i++) {
        // Decoupled: the decay pulls p towards zero on its own, and never enters the moment estimates.
        if (decay !== 0) data[i] -= decay * data[i];
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g[i];
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g[i] * g[i];
        const mHat = m[i] / correction1;
        const vHat = v[i] / correction2;
        data[i] -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }

  /** Drop every parameter's accumulated gradient, ready for the next backward pass. */
  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }
}

/** Global L2 norm over all parameter gradients; if it exceeds maxNorm, scale every gradient down in place. */
export function clipGradNorm(params, maxNorm) {
  let sumSquares = 0;
  for (const p of params) {
    if (!p.grad) continue;
    const g = p.grad;
    for (let i = 0; i < g.length; i++) sumSquares += g[i] * g[i];
  }
  const totalNorm = Math.sqrt(sumSquares);
  if (totalNorm > maxNorm) {
    const scale = maxNorm / totalNorm;
    for (const p of params) {
      if (!p.grad) continue;
      const g = p.grad;
      for (let i = 0; i < g.length; i++) g[i] *= scale;
    }
  }
  return totalNorm;
}

/** Learning rate at `step`: linear 0 -> peak over `warmup` steps, then a cosine down to `min` at `total`. */
export function cosineWithWarmup(step, { warmup, total, peak, min = peak / 10 }) {
  if (step < warmup) return (peak * step) / warmup;
  if (step >= total) return min;
  const progress = (step - warmup) / (total - warmup);
  return min + 0.5 * (peak - min) * (1 + Math.cos(Math.PI * progress));
}
