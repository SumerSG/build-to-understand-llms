// lib/gpt.js — the GPT architecture on top of lib/tensor.js.
// This file is the REFERENCE SOLUTION for module 06 (The GPT architecture).
//
// A GPT is a stack of identical residual blocks over a "residual stream" of shape [B,T,C]. Each block
// writes two updates into that stream:
//
//   x = x + attn(ln1(x))   mixes information ACROSS positions (the only place positions talk)
//   x = x + mlp(ln2(x))    transforms EACH position on its own (the only place per-position compute lives)
//
// Everything else is plumbing: token and position embeddings at the bottom, a final LayerNorm at the top,
// and a language-model head that is tied to the token embedding (logits = x · wteᵀ), so the vector that
// represents a token when reading it is also the vector it is scored against when predicting it.
//
// This is pre-LN (the LayerNorm sits inside the residual branch, not around it), which is what makes deep
// stacks train without a warmup-dependent explosion — the identity path from input to output is clean.
//
// Parameter budget per block: attention 4C² (qkv 3C², proj C²), MLP 8C² (4C² up, 4C² down) = 12C²,
// so a model is roughly 12·L·C² plus V·C for the embedding table.

import { noGrad } from './tensor.js';
import { rng } from './util.js';
import { Linear, Embedding, LayerNorm } from './layers.js';
import { MultiHeadAttention } from './attention.js';
import { sample } from './sampling.js';

// The building blocks live in lib/layers.js (lib/attention.js needs them too); re-exported so that
// docs/LIB_API.md's `lib/gpt.js` contract holds: import { Linear, Embedding, LayerNorm } from './gpt.js'.
export { Linear, Embedding, LayerNorm } from './layers.js';

/** The per-position feed-forward network: expand to 4×, apply GELU, project back. */
export class MLP {
  /** fc widens C -> 4C and proj narrows 4C -> C; the widening is where most parameters sit. */
  constructor(nEmbd, { next } = {}) {
    this.fc = new Linear(nEmbd, 4 * nEmbd, { next });
    this.proj = new Linear(4 * nEmbd, nEmbd, { next });
  }

  /** x [..., C] -> [..., C]. */
  forward(x) {
    return this.proj.forward(this.fc.forward(x).gelu());
  }

  /** fc then proj, each weight before its bias. */
  parameters() {
    return [...this.fc.parameters(), ...this.proj.parameters()];
  }
}

/** One pre-LN transformer block: normalise, mix across positions, add; normalise, transform, add. */
export class Block {
  /** cfg supplies nEmbd and nHead; `next` seeds both sub-layers' initialisation. */
  constructor(cfg, { next } = {}) {
    this.ln1 = new LayerNorm(cfg.nEmbd);
    this.attn = new MultiHeadAttention({ nEmbd: cfg.nEmbd, nHead: cfg.nHead, next });
    this.ln2 = new LayerNorm(cfg.nEmbd);
    this.mlp = new MLP(cfg.nEmbd, { next });
  }

  /** x [B,T,C] -> [B,T,C]. The two `add`s are the residual stream; nothing else touches it. */
  forward(x) {
    const afterAttention = x.add(this.attn.forward(this.ln1.forward(x)));
    return afterAttention.add(this.mlp.forward(this.ln2.forward(afterAttention)));
  }

  /** ln1, attn, ln2, mlp — the order lib/gpt.js paramNames() mirrors. */
  parameters() {
    return [
      ...this.ln1.parameters(),
      ...this.attn.parameters(),
      ...this.ln2.parameters(),
      ...this.mlp.parameters(),
    ];
  }
}

/** A decoder-only transformer: embeddings, a stack of blocks, a final LayerNorm and a tied LM head. */
export class GPT {
  /** All of the model's shape lives in config; `seed` makes the whole initialisation reproducible. */
  constructor({ vocabSize, blockSize, nLayer, nHead, nEmbd, seed = 0 }) {
    this.config = { vocabSize, blockSize, nLayer, nHead, nEmbd, seed };
    const next = rng(seed);
    this.wte = new Embedding(vocabSize, nEmbd, { next }); // token embeddings, also the LM head (tied)
    this.wpe = new Embedding(blockSize, nEmbd, { next }); // learned position embeddings, one per slot
    this.blocks = [];
    for (let i = 0; i < nLayer; i++) this.blocks.push(new Block({ nEmbd, nHead }, { next }));
    this.lnF = new LayerNorm(nEmbd);
  }

  /** ids: number[][] of shape B×T (T <= blockSize) -> logits Tensor [B,T,V], one prediction per position. */
  forward(ids) {
    if (!Array.isArray(ids) || !Array.isArray(ids[0])) throw new Error('GPT.forward: ids must be number[][] (B×T)');
    const time = ids[0].length;
    if (time > this.config.blockSize) {
      throw new Error(`GPT.forward: sequence of ${time} exceeds blockSize ${this.config.blockSize}`);
    }
    const positions = [];
    for (let t = 0; t < time; t++) positions.push(t);

    // Token identity plus position: the only thing that tells the model what order the tokens came in.
    let x = this.wte.forward(ids).add(this.wpe.forward(positions)); // [B,T,C] + [T,C] broadcasts
    for (const block of this.blocks) x = block.forward(x);
    x = this.lnF.forward(x);
    // Tied head: scoring against the embedding table costs no extra parameters.
    return x.matmul(this.wte.weight.transpose());
  }

  /** [name, Tensor] for every trainable tensor, in a fixed order that toJSON() and paramNames() share. */
  namedParameters() {
    const named = [
      ['wte.weight', this.wte.weight],
      ['wpe.weight', this.wpe.weight],
    ];
    this.blocks.forEach((block, i) => {
      named.push([`blocks.${i}.ln1.gamma`, block.ln1.gamma]);
      named.push([`blocks.${i}.ln1.beta`, block.ln1.beta]);
      named.push([`blocks.${i}.attn.qkv.weight`, block.attn.qkv.weight]);
      named.push([`blocks.${i}.attn.qkv.bias`, block.attn.qkv.bias]);
      named.push([`blocks.${i}.attn.proj.weight`, block.attn.proj.weight]);
      named.push([`blocks.${i}.attn.proj.bias`, block.attn.proj.bias]);
      named.push([`blocks.${i}.ln2.gamma`, block.ln2.gamma]);
      named.push([`blocks.${i}.ln2.beta`, block.ln2.beta]);
      named.push([`blocks.${i}.mlp.fc.weight`, block.mlp.fc.weight]);
      named.push([`blocks.${i}.mlp.fc.bias`, block.mlp.fc.bias]);
      named.push([`blocks.${i}.mlp.proj.weight`, block.mlp.proj.weight]);
      named.push([`blocks.${i}.mlp.proj.bias`, block.mlp.proj.bias]);
    });
    named.push(['lnF.gamma', this.lnF.gamma]);
    named.push(['lnF.beta', this.lnF.beta]);
    return named;
  }

  /** Every trainable tensor, ready to hand to an optimizer. The LM head is tied, so it is not listed twice. */
  parameters() {
    return this.namedParameters().map(([, tensor]) => tensor);
  }

  /** Total number of trainable scalars. */
  numParams() {
    let total = 0;
    for (const p of this.parameters()) total += p.size;
    return total;
  }

  /**
   * Continue `ids` by maxNewTokens, sampling one token at a time. Everything runs inside noGrad (no graph
   * is built) and the context is cropped to blockSize. This recomputes the whole prefix every step — the
   * KV cache in lib/infer.js is exactly the fix for that.
   */
  generate(ids, { maxNewTokens = 20, temperature = 1, topK = null, next } = {}) {
    return noGrad(() => {
      const out = ids.slice();
      const vocabSize = this.config.vocabSize;
      for (let step = 0; step < maxNewTokens; step++) {
        const context = out.slice(Math.max(0, out.length - this.config.blockSize));
        const logits = this.forward([context]); // [1,T,V]
        const lastRow = logits.data.subarray((context.length - 1) * vocabSize, context.length * vocabSize);
        out.push(sample(lastRow, { temperature, topK: topK === null ? 0 : topK, next }));
      }
      return out;
    });
  }

  /** Plain JSON: the config plus every parameter by name. Floats keep 6 significant digits to stay small. */
  toJSON() {
    const params = {};
    for (const [name, tensor] of this.namedParameters()) {
      const data = new Array(tensor.data.length);
      for (let i = 0; i < data.length; i++) data[i] = Number(tensor.data[i].toPrecision(6));
      params[name] = { shape: tensor.shape.slice(), data };
    }
    return { config: { ...this.config }, params };
  }

  /** Rebuild a GPT from toJSON() output: same config, then every parameter copied back in. */
  static fromJSON(o) {
    const model = new GPT(o.config);
    for (const [name, tensor] of model.namedParameters()) {
      const saved = o.params[name];
      if (saved === undefined) throw new Error(`GPT.fromJSON: checkpoint is missing parameter '${name}'`);
      if (saved.data.length !== tensor.data.length) {
        throw new Error(`GPT.fromJSON: '${name}' has ${saved.data.length} values, expected ${tensor.data.length}`);
      }
      tensor.data.set(saved.data);
    }
    return model;
  }
}

/** The names of model.parameters(), in the same order: 'wte.weight', 'blocks.0.ln1.gamma', … 'lnF.beta'. */
export function paramNames(model) {
  return model.namedParameters().map(([name]) => name);
}
