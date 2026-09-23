// Vision tokens: a tiny multimodal model. Reference solution.
// An image becomes a sequence of vectors in the GPT's embedding space; from then on the GPT treats them
// exactly like token embeddings. Raw tensors are { shape, data: Float32Array }; Tensor is lib/tensor.js.

import * as ops from 'lib/ops.js';
import { Tensor, crossEntropy, noGrad } from 'lib/tensor.js';
import { Linear, Embedding } from 'lib/gpt.js';
import { rng, randn, randInt } from 'lib/util.js';

// ---------- worked examples: synthetic images, sequence plumbing ----------

export const IMAGE_SIZE = 16;
export const SHAPES = ['square', 'circle', 'cross', 'triangle'];
export const PLACES = ['top left', 'top right', 'bottom left', 'bottom right'];

/** The caption the model must learn to write: captionOf('circle', 'top left') === 'a circle at the top left'. */
export function captionOf(shape, place) {
  return `a ${shape} at the ${place}`;
}

/**
 * Draw one shape into a 16×16 grayscale image (raw [16, 16], values roughly in [0, 1]).
 * The shape sits inside one 8×8 quadrant: a random half-size h (2 or 3 pixels, so 5 or 7 wide), a
 * centre jittered by one pixel in each direction, a random brightness and a little Gaussian noise.
 */
export function drawShape(shape, place, next) {
  const S = IMAGE_SIZE;
  const img = new Float32Array(S * S);
  const h = 2 + randInt(next, 2);
  const cy = (place.startsWith('top') ? 3 : 11) + randInt(next, 2);
  const cx = (place.endsWith('left') ? 3 : 11) + randInt(next, 2);
  const ink = 0.7 + 0.3 * next();
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const dy = y - cy, dx = x - cx;
      let on = false;
      if (shape === 'square') on = Math.abs(dx) <= h && Math.abs(dy) <= h;
      else if (shape === 'circle') { const r = Math.sqrt(dx * dx + dy * dy); on = r <= h + 0.5 && r >= h - 0.7; }
      else if (shape === 'cross') on = (dx === 0 && Math.abs(dy) <= h) || (dy === 0 && Math.abs(dx) <= h);
      else if (shape === 'triangle') on = dy >= -h && dy <= h && Math.abs(dx) <= (dy + h) / 2;
      img[y * S + x] = (on ? ink : 0) + 0.05 * randn(next);
    }
  }
  return { shape: [S, S], data: img };
}

/** n labelled examples { image, shape, place, caption }, reproducible from the seed. */
export function makeDataset(n, seed = 0) {
  const next = rng(seed);
  const out = [];
  for (let i = 0; i < n; i++) {
    const shape = SHAPES[randInt(next, SHAPES.length)];
    const place = PLACES[randInt(next, PLACES.length)];
    out.push({ image: drawShape(shape, place, next), shape, place, caption: captionOf(shape, place) });
  }
  return out;
}

/**
 * A word-level tokenizer over a fixed list of texts: id 0 is '<eos>', the rest are the distinct words in
 * sorted order. The captions use 11 distinct words, so a caption is 6 tokens and the LM head has 12 rows.
 */
export class WordTokenizer {
  constructor(texts) {
    const words = new Set();
    for (const t of texts) for (const w of t.split(' ')) if (w) words.add(w);
    this.itos = ['<eos>', ...[...words].sort()];
    this.stoi = new Map(this.itos.map((w, i) => [w, i]));
    this.eos = 0;
    this.vocabSize = this.itos.length;
  }

  encode(text) {
    return text.split(' ').filter(Boolean).map((w) => {
      if (!this.stoi.has(w)) throw new Error(`WordTokenizer: unknown word "${w}"`);
      return this.stoi.get(w);
    });
  }

  decode(ids) {
    return ids.map((i) => this.itos[i]).join(' ');
  }
}

/** The tokenizer for every caption this module can draw. */
export function captionTokenizer() {
  const all = [];
  for (const s of SHAPES) for (const p of PLACES) all.push(captionOf(s, p));
  return new WordTokenizer(all);
}

/**
 * Join two token sequences along time: a [B, P, C] and b [B, L, C] -> [B, P + L, C], with gradients.
 * lib/tensor.js has no concat op, so this places each input with a constant 0/1 "placement" matrix:
 * aᵀ [B, C, P] · E_a [P, P + L] puts a's P columns first, b's go after, and the sum is the concatenation.
 * The backward pass of a matmul by a 0/1 matrix is exactly "slice the gradient back apart".
 * (PyTorch's torch.cat just copies memory; the result is the same.)
 */
export function concatTokens(a, b) {
  const P = a.shape[1], L = b.shape[1], N = P + L;
  const placeA = ops.zeros([P, N]);
  for (let i = 0; i < P; i++) placeA.data[i * N + i] = 1;
  const placeB = ops.zeros([L, N]);
  for (let i = 0; i < L; i++) placeB.data[i * N + P + i] = 1;
  const joined = a.transpose().matmul(new Tensor(placeA)).add(b.transpose().matmul(new Tensor(placeB)));
  return joined.transpose();
}

/**
 * The GPT forward pass from the GPT architecture module, started from embeddings instead of token ids.
 * lib/gpt.js GPT.forward(ids) begins with wte.forward(ids); everything after that line is here unchanged:
 * add the learned position embedding, run the blocks, final LayerNorm, tied LM head.
 * x: Tensor [B, N, C] -> logits Tensor [B, N, V].
 */
export function forwardEmbeds(gpt, x) {
  const N = x.shape[1];
  if (N > gpt.config.blockSize) throw new Error(`forwardEmbeds: ${N} positions exceed blockSize ${gpt.config.blockSize}`);
  const positions = Array.from({ length: N }, (_, t) => t);
  let h = x.add(gpt.wpe.forward(positions));
  for (const block of gpt.blocks) h = block.forward(h);
  h = gpt.lnF.forward(h);
  return h.matmul(gpt.wte.weight.transpose());
}

/**
 * A batch of images as one Tensor of patches: raw [H, W][] -> Tensor [B, P, patch²]. It calls your
 * patchify (step 1) once per image. No gradient: pixels are data, not parameters.
 */
export function stackPatches(images, patch) {
  const first = patchify(images[0], patch);
  const [P, d] = first.shape;
  const data = new Float32Array(images.length * P * d);
  images.forEach((img, b) => data.set(b === 0 ? first.data : patchify(img, patch).data, b * P * d));
  return new Tensor({ shape: [images.length, P, d], data });
}

// ---------- step 1: patchify ----------

/**
 * Cut an image raw [H, W] into non-overlapping patch×patch squares, in reading order (left to right,
 * then top to bottom), each flattened row-major: -> raw [(H/patch)·(W/patch), patch·patch].
 */
export function patchify(image, patch) {
  const [H, W] = image.shape;
  if (H % patch !== 0 || W % patch !== 0) throw new Error(`patchify: ${H}×${W} is not divisible by patch ${patch}`);
  const gh = H / patch, gw = W / patch, d = patch * patch;
  const out = new Float32Array(gh * gw * d);
  for (let py = 0; py < gh; py++) {
    for (let px = 0; px < gw; px++) {
      const base = (py * gw + px) * d;
      for (let y = 0; y < patch; y++) {
        for (let x = 0; x < patch; x++) out[base + y * patch + x] = image.data[(py * patch + y) * W + px * patch + x];
      }
    }
  }
  return { shape: [gh * gw, d], data: out };
}

// ---------- step 2: patch embedding and projector ----------

/** ViT's patch embedding: a linear map of each flattened patch plus a learned vector for each patch slot. */
export class PatchEmbed {
  constructor({ patchDim, nPatches, dim, next }) {
    this.proj = new Linear(patchDim, dim, { next, std: 1 / Math.sqrt(patchDim) });
    this.pos = Tensor.param(ops.randn([nPatches, dim], next, 0.02));
  }

  /** patches Tensor [B, P, patchDim] -> [B, P, dim]. */
  forward(patches) {
    return this.proj.forward(patches).add(this.pos);
  }

  parameters() {
    return [...this.proj.parameters(), this.pos];
  }
}

/** LLaVA-1.5's projector: Linear -> GELU -> Linear, from the vision width into the GPT's width. */
export class Projector {
  constructor(dIn, dOut, { hidden = dOut, next } = {}) {
    this.fc1 = new Linear(dIn, hidden, { next, std: 1 / Math.sqrt(dIn) });
    this.fc2 = new Linear(hidden, dOut, { next, std: 1 / Math.sqrt(hidden) });
  }

  /** x [..., dIn] -> [..., dOut]. */
  forward(x) {
    return this.fc2.forward(this.fc1.forward(x).gelu());
  }

  parameters() {
    return [...this.fc1.parameters(), ...this.fc2.parameters()];
  }
}

// ---------- step 3: one sequence, loss on the caption only ----------

/** Image tokens [B, P, C] followed by the text tokens' embeddings: -> [B, P + L, C]. */
export function embedSequence(gpt, imageTokens, textIds /* number[][] B×L */) {
  return concatTokens(imageTokens, gpt.wte.forward(textIds));
}

/**
 * Inputs, targets and loss mask for one example laid out as [nImage image tokens][caption, padded].
 * Position i predicts the element at i + 1, so the LAST image token predicts the first caption token and
 * the last caption token predicts eos. Everything else (predicting image tokens, padding) has mask 0.
 */
export function captionTargets(nImage, captionIds, eos, width = captionIds.length) {
  if (width < captionIds.length) throw new Error(`captionTargets: width ${width} < caption length ${captionIds.length}`);
  const textIds = captionIds.concat(new Array(width - captionIds.length).fill(eos));
  const N = nImage + width;
  const targets = new Array(N).fill(0);
  const mask = new Array(N).fill(0);
  const want = captionIds.concat([eos]);
  for (let j = 0; j < want.length; j++) {
    targets[nImage - 1 + j] = want[j];
    mask[nImage - 1 + j] = 1;
  }
  return { textIds, targets, mask };
}

/** Mean cross-entropy over the positions where mask is 1. logits Tensor [B, N, V]; targets, mask number[][]. */
export function maskedCrossEntropy(logits, targets, mask) {
  const V = logits.shape[logits.shape.length - 1];
  const t = targets.flat(), mk = mask.flat();
  let count = 0;
  for (const v of mk) count += v ? 1 : 0;
  if (count === 0) throw new Error('maskedCrossEntropy: mask selects no positions');
  const pick = ops.zeros(logits.shape);
  for (let i = 0; i < t.length; i++) if (mk[i]) pick.data[i * V + t[i]] = 1 / count;
  return logits.logSoftmax().mul(new Tensor(pick)).sum().neg();
}

// ---------- step 4: contrastive alignment (CLIP) ----------

/** Scale every row of x [B, D] to unit length (eps keeps a zero row finite). */
export function l2normalize(x, eps = 1e-8) {
  return x.div(x.mul(x).sum(-1, true).add(eps).sqrt());
}

/**
 * CLIP's symmetric InfoNCE loss. img, txt: Tensor [B, D], row i of each describes the same example.
 * logits = normalize(img) · normalize(txt)ᵀ / temperature; each image must pick its caption out of the
 * batch (rows) and each caption its image (columns); the loss is the mean of the two cross-entropies.
 */
export function clipLoss(img, txt, temperature = 0.07) {
  const logits = l2normalize(img).matmul(l2normalize(txt).transpose()).scale(1 / temperature);
  const labels = Array.from({ length: img.shape[0] }, (_, i) => i);
  const loss = crossEntropy(logits, labels).add(crossEntropy(logits.transpose(), labels)).scale(0.5);
  return { loss, logits };
}

/**
 * Two tiny towers for the contrastive demo, both landing in [B, dim].
 * Image: patch embedding, GELU, mean over patches, linear. The GELU matters: the mean of a LINEAR
 * function of the patches equals that function of the mean patch, and the mean patch has forgotten
 * where the ink was. CLIP's real image tower runs a whole ViT before pooling.
 * Text: bag of words, i.e. the mean of the caption's token embeddings, then linear.
 */
export class ClipHead {
  constructor({ vocabSize, patch = 4, imageSize = IMAGE_SIZE, dim = 32, next }) {
    this.patch = patch;
    this.vocabSize = vocabSize;
    const nPatches = (imageSize / patch) ** 2;
    this.patchEmbed = new PatchEmbed({ patchDim: patch * patch, nPatches, dim, next });
    this.imageOut = new Linear(dim, dim, { next, std: 1 / Math.sqrt(dim) });
    this.tokens = new Embedding(vocabSize, dim, { next });
    this.textOut = new Linear(dim, dim, { next, std: 1 / Math.sqrt(dim) });
  }

  /** images: raw [H, W][] -> Tensor [B, dim]. */
  encodeImages(images) {
    const x = stackPatches(images, this.patch);
    return this.imageOut.forward(this.patchEmbed.forward(x).gelu().mean(1));
  }

  /** captions: number[][] (any lengths) -> Tensor [B, dim]. Mean pooling is a [B, V] averaging matrix times the table. */
  encodeTexts(captions) {
    const avg = ops.zeros([captions.length, this.vocabSize]);
    captions.forEach((ids, b) => { for (const id of ids) avg.data[b * this.vocabSize + id] += 1 / ids.length; });
    return this.textOut.forward(new Tensor(avg).matmul(this.tokens.weight));
  }

  parameters() {
    return [...this.patchEmbed.parameters(), ...this.imageOut.parameters(), ...this.tokens.parameters(), ...this.textOut.parameters()];
  }
}

// ---------- step 5: the captioner ----------

/**
 * A LLaVA-shaped captioner: patchify -> PatchEmbed (the "vision encoder") -> Projector -> the GPT.
 * tokenizer needs encode(str) -> ids and decode(ids) -> str; eos is the id that ends a caption.
 */
export class Captioner {
  constructor({ gpt, tokenizer, eos, patch = 4, imageSize = IMAGE_SIZE, dVision = 32, seed = 0 }) {
    const next = rng(seed);
    this.gpt = gpt;
    this.tokenizer = tokenizer;
    this.eos = eos;
    this.patch = patch;
    this.nPatches = (imageSize / patch) ** 2;
    this.vision = new PatchEmbed({ patchDim: patch * patch, nPatches: this.nPatches, dim: dVision, next });
    this.projector = new Projector(dVision, gpt.config.nEmbd, { next });
  }

  /** images: raw [H, W][] -> Tensor [B, P, nEmbd], vectors living in the GPT's token-embedding space. */
  imageTokens(images) {
    return this.projector.forward(this.vision.forward(stackPatches(images, this.patch)));
  }

  parameters() {
    return [...this.vision.parameters(), ...this.projector.parameters(), ...this.gpt.parameters()];
  }
}

/** Mean caption-token cross-entropy of a batch: images raw[], captions string[]. */
export function captionLoss(model, images, captions) {
  const ids = captions.map((c) => model.tokenizer.encode(c));
  const width = Math.max(...ids.map((c) => c.length));
  const ex = ids.map((c) => captionTargets(model.nPatches, c, model.eos, width));
  const x = embedSequence(model.gpt, model.imageTokens(images), ex.map((e) => e.textIds));
  const logits = forwardEmbeds(model.gpt, x);
  return maskedCrossEntropy(logits, ex.map((e) => e.targets), ex.map((e) => e.mask));
}

/** Greedy caption for one image: feed the image tokens, then append the argmax token until eos. */
export function caption(model, image, { maxNewTokens = 12 } = {}) {
  return noGrad(() => {
    const img = model.imageTokens([image]);
    const V = model.gpt.config.vocabSize;
    const limit = Math.min(maxNewTokens, model.gpt.config.blockSize - model.nPatches);
    const out = [];
    for (let s = 0; s < limit; s++) {
      const x = out.length ? embedSequence(model.gpt, img, [out]) : img;
      const logits = forwardEmbeds(model.gpt, x);
      const last = logits.data.subarray((x.shape[1] - 1) * V, x.shape[1] * V);
      let best = 0;
      for (let v = 1; v < V; v++) if (last[v] > last[best]) best = v;
      if (best === model.eos) break;
      out.push(best);
    }
    return model.tokenizer.decode(out);
  });
}
