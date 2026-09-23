// Mixture of experts.
// Tensors are lib/tensor.js Tensors (autograd). A batch of tokens is flattened to x: [N, C]
// (N tokens, C channels). E is the number of experts and k the number each token is routed to.
// The worked examples at the top and the plumbing at the bottom are done; every function marked
// TODO (with its step number) is yours. Run the step's tests after each one.

import { Tensor, noGrad } from 'lib/tensor.js';
import { Linear, LayerNorm, GPT } from 'lib/gpt.js';
import { MultiHeadAttention } from 'lib/attention.js';
import * as ops from 'lib/ops.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/** One expert: a GPT MLP with a configurable hidden width. x [n, C] -> [n, C]. */
export class Expert {
  constructor(nEmbd, hidden, { next } = {}) {
    this.fc = new Linear(nEmbd, hidden, { next });
    this.proj = new Linear(hidden, nEmbd, { next });
  }

  forward(x) {
    return this.proj.forward(this.fc.forward(x).gelu());
  }

  parameters() {
    return [...this.fc.parameters(), ...this.proj.parameters()];
  }
}

/** Gather: rows `rows` of x [N, C] -> [rows.length, C]. Backward scatter-adds into x's gradient. */
export function gatherRows(x, rows) {
  return x.embed(rows);
}

/**
 * Scatter: y [rows.length, C] placed into a zero [n, C] at positions `rows` (y's row j lands at rows[j]).
 * Done as a matmul with a constant one-hot matrix so that autograd handles the backward pass.
 */
export function scatterRows(y, rows, n) {
  const oneHot = ops.zeros([n, rows.length]);
  for (let j = 0; j < rows.length; j++) oneHot.data[rows[j] * rows.length + j] = 1;
  return new Tensor(oneHot).matmul(y);
}

/**
 * The per-token reference: for every token t and every expert e in topIdx[t], add
 * gates[t, e] · expert_e(x[t]). Slow (one expert call per token per choice) but obviously right.
 * Runs without building a graph and returns a raw tensor { shape: [N, C], data }.
 */
export function naiveMoE(x, gates, topIdx, experts) {
  return noGrad(() => {
    const [N, C] = x.shape;
    const E = gates.shape[1];
    const out = new Float32Array(N * C);
    for (let t = 0; t < N; t++) {
      const xt = new Tensor(ops.raw([1, C], x.data.slice(t * C, (t + 1) * C)));
      for (const e of topIdx[t]) {
        const y = experts[e].forward(xt).data;
        const g = gates.data[t * E + e];
        for (let c = 0; c < C; c++) out[t * C + c] += g * y[c];
      }
    }
    return ops.raw([N, C], out);
  });
}

// ---------- step 1: the router ----------

/** Indices of the k largest entries of `row`, largest first; ties go to the lower index. */
export function topKIndices(row, k) {
  // TODO: step 1
  return [];
}

/** A linear gate over the experts: probs = softmax(x · W_g), then top-k with renormalised weights. */
export class Router {
  constructor(nEmbd, nExperts, k, { next } = {}) {
    this.nExperts = nExperts;
    this.k = k;
    this.gate = new Linear(nEmbd, nExperts, { bias: false, next });
  }

  /**
   * x [N, C] -> { probs: Tensor [N, E], topIdx: number[][] (N × k), gates: Tensor [N, E] }.
   * gates is zero outside each token's top-k and sums to 1 over the chosen experts.
   */
  forward(x) {
    // TODO: step 1 — probs = softmax(x · W_g); per token, the top-k experts; gates = probs masked to
    // the chosen experts and renormalised to sum to 1. Keep gates on the autograd graph.
    return { probs: null, topIdx: [], gates: null };
  }

  parameters() {
    return this.gate.parameters();
  }
}

// ---------- step 2: dispatch and combine ----------

/** For each expert, the tokens routed to it, in increasing token order: number[E][]. */
export function groupByExpert(topIdx, nExperts) {
  // TODO: step 2
  return [];
}

/**
 * Run each expert once on the tokens in lists[e], weight its outputs by gates[t, e], and scatter the
 * results back to their token positions. x [N, C], gates [N, E] -> Tensor [N, C].
 */
export function dispatchCombine(x, gates, lists, experts) {
  // TODO: step 2 — for each expert with tokens: gather its rows, run it once, weight by its gate
  // column, scatter back to [N, C], and sum over experts.
  return x;
}

// ---------- step 3: the load-balancing loss ----------

/** f_i: the fraction of all N·k routing assignments that went to expert i (sums to 1). */
export function routingFractions(topIdx, nExperts) {
  // TODO: step 3
  return new Array(nExperts).fill(0);
}

/**
 * Switch Transformer's auxiliary loss: E · Σ_i f_i · P_i, where P_i is the mean router probability of
 * expert i over the tokens. f is a constant (counts have no gradient); the gradient flows through P.
 * Equals 1 when routing is perfectly balanced, E when everything goes to one expert.
 */
export function loadBalanceLoss(probs, topIdx) {
  // TODO: step 3
  return new Tensor(ops.zeros([1]));
}

// ---------- step 4: capacity and dropped tokens ----------

/** Slots per expert: ceil(factor · N · k / E). factor 1 means "exactly the balanced share". */
export function expertCapacity(nTokens, nExperts, k, factor) {
  // TODO: step 4
  return nTokens;
}

/**
 * Keep the first `capacity` tokens of every expert's list (token order = priority) and drop the rest.
 * Returns { kept: number[][], dropped: number } without mutating `lists`.
 */
export function applyCapacity(lists, capacity) {
  // TODO: step 4
  return { kept: lists, dropped: 0 };
}

// ---------- step 5: the MoE layer and the accounting ----------

/** A sparse MoE feed-forward layer: router -> (capacity) -> dispatch -> experts -> combine. */
export class MoE {
  constructor({ nEmbd, nExperts = 4, k = 2, hidden = 2 * nEmbd, capacityFactor = Infinity, next }) {
    this.nEmbd = nEmbd;
    this.nExperts = nExperts;
    this.k = k;
    this.hidden = hidden;
    this.capacityFactor = capacityFactor;
    this.router = new Router(nEmbd, nExperts, k, { next });
    this.experts = [];
    for (let e = 0; e < nExperts; e++) this.experts.push(new Expert(nEmbd, hidden, { next }));
    this.lastAux = null;
    this.lastStats = null;
  }

  /**
   * x [..., C] -> [..., C]. Also sets this.lastAux (the load-balancing loss, a scalar Tensor) and
   * this.lastStats = { counts: number[E] (routed, before capacity), dropped, assignments, capacity }.
   */
  forward(x) {
    // TODO: step 5 — flatten to [N, C]; route; group; apply capacity (only if capacityFactor is finite,
    // otherwise capacity = N); set this.lastAux and this.lastStats; dispatch/combine; reshape back.
    return x;
  }

  parameters() {
    return [...this.router.parameters(), ...this.experts.flatMap((ex) => ex.parameters())];
  }
}

/**
 * Parameters and FLOPs of ONE MoE feed-forward layer with GELU experts of width `hidden`.
 * expert = 2·C·H + H + C (two weight matrices and two biases); router = C·E (no bias).
 * total = E·expert + router; active = k·expert + router; flopsPerToken = 2·active.
 */
export function countParams({ nEmbd, hidden, nExperts, k }) {
  // TODO: step 5
  return { total: 0, active: 0, flopsPerToken: 0 };
}

// ---------- worked plumbing (done for you): the block and the GPT swap ----------

/** The pre-LN block from the GPT architecture module, with the MLP swapped for an MoE layer. */
export class MoEBlock {
  constructor(cfg, moeOpts, { next } = {}) {
    this.ln1 = new LayerNorm(cfg.nEmbd);
    this.attn = new MultiHeadAttention({ nEmbd: cfg.nEmbd, nHead: cfg.nHead, next });
    this.ln2 = new LayerNorm(cfg.nEmbd);
    this.moe = new MoE({ nEmbd: cfg.nEmbd, ...moeOpts, next });
  }

  forward(x) {
    const h = x.add(this.attn.forward(this.ln1.forward(x)));
    return h.add(this.moe.forward(this.ln2.forward(h)));
  }

  parameters() {
    return [...this.ln1.parameters(), ...this.attn.parameters(), ...this.ln2.parameters(), ...this.moe.parameters()];
  }
}

/**
 * A lib/gpt.js GPT whose blocks are MoEBlocks. parameters() is overridden to collect the MoE weights,
 * and auxLoss() sums every layer's load-balancing loss from the most recent forward pass.
 */
export function moeGPT(config, moeOpts, { next } = {}) {
  const model = new GPT(config);
  model.blocks = model.blocks.map(() => new MoEBlock(config, moeOpts, { next }));
  model.parameters = () => [
    model.wte.weight, model.wpe.weight,
    ...model.blocks.flatMap((b) => b.parameters()),
    model.lnF.gamma, model.lnF.beta,
  ];
  model.numParams = () => model.parameters().reduce((s, p) => s + p.size, 0);
  model.auxLoss = () => model.blocks.map((b) => b.moe.lastAux).reduce((a, b) => a.add(b));
  return model;
}
