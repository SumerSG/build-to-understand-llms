import * as ops from 'lib/ops.js';
import { Tensor } from 'lib/tensor.js';
import { GPT, Linear, Embedding } from 'lib/gpt.js';
import { AdamW } from 'lib/optim.js';
import { rng } from 'lib/util.js';

/** A raw [H, W] image whose pixel values are 0, 1, 2, … in reading order, so every pixel is identifiable. */
function arangeImage(H, W) {
  const data = new Float32Array(H * W);
  for (let i = 0; i < data.length; i++) data[i] = i;
  return { shape: [H, W], data };
}

function randomTensor(shape, next, scale = 1, requiresGrad = false) {
  const data = new Float32Array(shape.reduce((a, b) => a * b, 1));
  for (let i = 0; i < data.length; i++) data[i] = (next() * 2 - 1) * scale;
  return new Tensor({ shape, data }, { requiresGrad });
}

/** Plain-JS log-softmax of one row. */
function logSoftmaxRow(row) {
  const m = Math.max(...row);
  const z = row.reduce((s, v) => s + Math.exp(v - m), 0);
  return row.map((v) => v - m - Math.log(z));
}

/** Plain-JS symmetric InfoNCE on nested arrays: the reference for step 4. */
function refClip(img, txt, tau) {
  const norm = (r) => { const n = Math.sqrt(r.reduce((s, v) => s + v * v, 0)); return r.map((v) => v / n); };
  const a = img.map(norm), b = txt.map(norm);
  const B = a.length;
  const L = a.map((ra) => b.map((rb) => ra.reduce((s, v, k) => s + v * rb[k], 0) / tau));
  let rows = 0, cols = 0;
  for (let i = 0; i < B; i++) {
    rows -= logSoftmaxRow(L[i])[i];
    cols -= logSoftmaxRow(L.map((r) => r[i]))[i];
  }
  return { loss: (rows / B + cols / B) / 2, logits: L };
}

/**
 * A word tokenizer with captions of different lengths (the module's own captions all have 6 words).
 * Its eos is the LAST id (9), not 0 as in captionTokenizer, so code that hard-codes 0 for eos fails here.
 */
function testTokenizer() {
  const itos = ['a', 'at', 'bottom', 'circle', 'cross', 'left', 'right', 'the', 'top', '<eos>'];
  return {
    itos, eos: 9, vocabSize: itos.length,
    encode: (s) => s.split(' ').map((w) => itos.indexOf(w)),
    decode: (ids) => ids.map((i) => itos[i]).join(' '),
  };
}

function tinyCaptioner(m, tokenizer, seed = 1) {
  const gpt = new GPT({ vocabSize: tokenizer.vocabSize, blockSize: 16, nLayer: 1, nHead: 2, nEmbd: 16, seed });
  return new m.Captioner({ gpt, tokenizer, eos: tokenizer.eos, patch: 8, dVision: 16, seed: seed + 10 });
}

export const tests = [
  // ---------- step 1: patchify ----------
  { step: 'patchify', name: 'a 4×4 image with 2×2 patches gives four squares in reading order', run(m, T) {
    const p = m.patchify(arangeImage(4, 4), 2);
    T.shape(p, [4, 4], 'a 4×4 image cut into 2×2 patches is 4 patches of 4 pixels each: [nPatches, patch²]');
    T.eq(T.arr(p)[0], [0, 1, 4, 5], 'patch 0 is the top-left 2×2 square: pixels (0,0) (0,1) (1,0) (1,1)');
    T.eq(T.arr(p)[1], [2, 3, 6, 7], 'patch 1 is the top-RIGHT square [2, 3, 6, 7]; a plain reshape of the image would give [4, 5, 6, 7], which is half of two rows, not a square');
    T.eq(T.arr(p), [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]], 'patches go left to right, then top to bottom; pixels inside a patch are row-major');
  } },
  { step: 'patchify', name: 'patches tile the image exactly (16×16 and non-square), and bad sizes throw', run(m, T) {
    const img = arangeImage(16, 16);
    const before = Array.from(img.data);
    const p = m.patchify(img, 4);
    T.shape(p, [16, 16], 'a 16×16 image with 4×4 patches is a 4×4 grid: 16 tokens of 16 pixels');
    const sorted = Array.from(p.data).sort((a, b) => a - b);
    T.eq(sorted, before, 'every pixel must appear exactly once across all patches (no overlap, no gaps)');
    // pixel (y, x) must land in patch (y/4, x/4) at offset (y%4, x%4)
    for (const [y, x] of [[0, 5], [7, 3], [13, 14], [9, 8]]) {
      const pi = Math.floor(y / 4) * 4 + Math.floor(x / 4), off = (y % 4) * 4 + (x % 4);
      T.eq(p.data[pi * 16 + off], y * 16 + x, `pixel (${y}, ${x}) belongs in patch ${pi} at offset ${off}`);
    }
    T.eq(Array.from(img.data), before, 'patchify must not modify the image');
    const wide = m.patchify(arangeImage(8, 12), 4);
    T.shape(wide, [6, 16], 'an 8×12 image has a 2×3 grid of 4×4 patches; use H and W separately');
    T.eq(T.arr(wide)[2].slice(0, 4), [8, 9, 10, 11], 'patch 2 of the 8×12 image starts at column 8 of row 0');
    T.throws(() => m.patchify(arangeImage(16, 16), 5), '16 is not divisible by 5: patchify must throw rather than silently drop pixels');
  } },

  // ---------- step 2: patch embedding and projector ----------
  { step: 'embed', name: 'PatchEmbed is x·W + b plus one learned vector per patch slot', run(m, T) {
    const pe = new m.PatchEmbed({ patchDim: 4, nPatches: 3, dim: 5, next: rng(1) });
    T.ok(pe.proj && pe.proj.weight && pe.pos, 'PatchEmbed needs `proj` (a Linear) and `pos` (a Tensor parameter)');
    T.shape(pe.proj.weight, [4, 5], 'proj maps patchDim=4 to dim=5');
    T.shape(pe.pos, [3, 5], 'pos has one learned vector per patch slot: [nPatches, dim]');
    T.ok(pe.pos.requiresGrad, 'pos must be a trainable parameter (Tensor.param)');
    const x = randomTensor([2, 3, 4], rng(2));
    for (let i = 0; i < 4; i++) x.data[4 + i] = x.data[i];           // patch 1 = patch 0 in batch 0
    const y = pe.forward(x);
    T.shape(y, [2, 3, 5], '[B, P, patchDim] -> [B, P, dim]');
    const ref = ops.add(ops.add(ops.matmul(x, pe.proj.weight), pe.proj.bias), pe.pos);
    T.close(y, ref, 1e-5, 'output must equal x·W + b + pos');
    const Y = T.arr(y)[0];
    const P = T.arr(pe.pos);
    T.close(Y[0].map((v, j) => v - Y[1][j]), P[0].map((v, j) => v - P[1][j]), 1e-5,
      'two identical patches at slots 0 and 1 must differ by exactly pos[0] - pos[1]: position is the only thing that tells the model WHERE a patch was');
    T.ok(pe.parameters().includes(pe.pos) && pe.parameters().includes(pe.proj.weight), 'parameters() must list proj\'s weight and bias and pos, or the optimizer never updates them');
  } },
  { step: 'embed', name: 'gradients reach every PatchEmbed parameter with the right values', run(m, T) {
    const pe = new m.PatchEmbed({ patchDim: 4, nPatches: 3, dim: 2, next: rng(3) });
    const x = randomTensor([2, 3, 4], rng(4));
    const R = randomTensor([2, 3, 2], rng(5));
    pe.forward(x).mul(R).sum().backward();
    T.ok(pe.pos.grad, 'pos received no gradient: add it with a Tensor op (.add) so autograd records it');
    const expectPos = ops.sum(R, 0);                     // pos is shared across the batch: its gradient sums over B
    T.close(Array.from(pe.pos.grad), Array.from(expectPos.data), 1e-5, 'dLoss/dpos = the upstream gradient summed over the batch');
    T.ok(pe.proj.weight.grad && pe.proj.weight.grad.some((v) => v !== 0), 'proj.weight must receive a non-zero gradient');
    const expectW = ops.sum(ops.matmul(ops.transpose(x), R), 0);
    T.close(Array.from(pe.proj.weight.grad), Array.from(expectW.data), 1e-4, 'dLoss/dW = Σ_b xᵀ·R');
  } },
  { step: 'embed', name: 'initialisation: fan-in std, created in the stated order from one rng', run(m, T) {
    const pe = new m.PatchEmbed({ patchDim: 16, nPatches: 4, dim: 8, next: rng(21) });
    T.ok(pe.proj && pe.proj.weight && pe.pos, 'PatchEmbed needs `proj` (a Linear) and `pos` (a Tensor parameter)');
    const r = rng(21);
    const W = ops.randn([16, 8], r, 1 / 4), pos = ops.randn([4, 8], r, 0.02);
    T.close(pe.proj.weight, W, 1e-6, 'proj.weight must be drawn FIRST from `next` with std 1/sqrt(patchDim) = 0.25 (new Linear(patchDim, dim, { next, std })); the same seed must give the same model so the goal demo is reproducible');
    T.close(pe.pos, pos, 1e-6, 'pos must be drawn AFTER proj from the same `next`, std 0.02');
    T.ok(T.arr(pe.proj.bias).every((v) => v === 0), 'Linear starts its bias at zero');
    const pr = new m.Projector(8, 6, { hidden: 12, next: rng(22) });
    T.ok(pr.fc1 && pr.fc2, 'Projector needs fc1 and fc2 (Linear layers)');
    const r2 = rng(22);
    const W1 = ops.randn([8, 12], r2, 1 / Math.sqrt(8)), W2 = ops.randn([12, 6], r2, 1 / Math.sqrt(12));
    T.close(pr.fc1.weight, W1, 1e-6, 'fc1 is drawn first, with std 1/sqrt(dIn)');
    T.close(pr.fc2.weight, W2, 1e-6, 'fc2 is drawn second, with std 1/sqrt(hidden)');
    const d = new m.Projector(8, 6, { next: rng(23) });
    T.shape(d.fc1.weight, [8, 6], 'hidden defaults to dOut');
  } },
  { step: 'embed', name: 'Projector is Linear → GELU → Linear into the GPT width', run(m, T) {
    const pr = new m.Projector(6, 4, { hidden: 8, next: rng(6) });
    T.ok(pr.fc1 && pr.fc2, 'Projector needs fc1 and fc2 (Linear layers)');
    T.shape(pr.fc1.weight, [6, 8], 'fc1: dIn=6 -> hidden=8');
    T.shape(pr.fc2.weight, [8, 4], 'fc2: hidden=8 -> dOut=4');
    T.eq(pr.parameters().length, 4, 'parameters(): fc1 weight and bias, fc2 weight and bias');
    const x = randomTensor([2, 3, 6], rng(7), 2);
    const y = pr.forward(x);
    T.shape(y, [2, 3, 4]);
    const h = ops.gelu(ops.add(ops.matmul(x, pr.fc1.weight), pr.fc1.bias));
    const ref = ops.add(ops.matmul(h, pr.fc2.weight), pr.fc2.bias);
    T.close(y, ref, 1e-4, 'output must be fc2(gelu(fc1(x))); without the GELU two Linears collapse into one matrix');
  } },
  { step: 'embed', name: 'fresh image tokens are loud enough for the GPT to hear', run(m, T) {
    const next = rng(8);
    const pe = new m.PatchEmbed({ patchDim: 16, nPatches: 16, dim: 32, next });
    const pr = new m.Projector(32, 32, { next });
    const img = m.drawShape('square', 'top left', rng(9));
    const pix = new Tensor(ops.reshape(m.patchify(img, 4), [1, 16, 16]));
    const out = pr.forward(pe.forward(pix));
    T.shape(out, [1, 16, 32], 'PatchEmbed then Projector: 16 patch tokens of GPT width 32');
    let ss = 0;
    for (const v of out.data) ss += v * v;
    const rms = Math.sqrt(ss / out.data.length);
    T.ok(rms > 0.03, `image tokens have RMS ${rms.toExponential(2)} at initialisation; expected > 0.03, at least as loud as the GPT's position embeddings (std 0.02). With GPT-2's std 0.02 in every layer they come out around 2e-4, a hundred times quieter than the position signal they are added to, and captioning barely trains. Initialise proj with std 1/sqrt(patchDim), fc1 with 1/sqrt(dIn) and fc2 with 1/sqrt(hidden) (fan-in scaling: each layer's output keeps roughly the scale of its input).`);
    T.ok(rms < 10, `image tokens have RMS ${rms.toFixed(2)}; expected well below 10, the fan-in scale keeps each layer near unit variance`);
  } },

  // ---------- step 3: sequence, targets, masked loss ----------
  { step: 'sequence', name: 'captionTargets: the last image token predicts the first word; padding and image slots are masked', run(m, T) {
    const r = m.captionTargets(3, [7, 8], 9, 4);
    T.eq(r.textIds, [7, 8, 9, 9], 'text inputs are the caption padded with eos up to width');
    T.eq(r.mask, [0, 0, 1, 1, 1, 0, 0], 'mask is 1 at positions nImage-1 … nImage-1+L (L caption tokens plus the eos after them); starting at nImage is the off-by-one that never trains the first word');
    T.eq(r.targets, [0, 0, 7, 8, 9, 0, 0], 'position i predicts element i+1: image slot 2 predicts 7, then 8, then eos=9; masked slots hold 0');
    const d = m.captionTargets(2, [5, 6, 4], 1);
    T.eq(d.textIds, [5, 6, 4], 'width defaults to the caption length (no padding)');
    T.eq(d.mask, [0, 1, 1, 1, 1], 'nImage=2, L=3: four supervised positions (three words and eos)');
    T.eq(d.targets, [0, 5, 6, 4, 1]);
    T.throws(() => m.captionTargets(2, [5, 6, 4], 1, 2), 'width 2 is shorter than the 3-token caption: captionTargets must throw rather than cut the caption');
  } },
  { step: 'sequence', name: 'maskedCrossEntropy averages only over masked-in positions', run(m, T) {
    const next = rng(10);
    const logits = randomTensor([2, 4, 5], next, 3, true);
    const targets = [[1, 2, 3, 4], [0, 0, 2, 1]];
    const mask = [[0, 1, 1, 0], [0, 0, 1, 1]];
    const loss = m.maskedCrossEntropy(logits, targets, mask);
    T.ok(loss instanceof Tensor && loss.size === 1, 'return a scalar Tensor so you can call backward()');
    const L = T.arr(logits);
    let ref = 0;
    for (let b = 0; b < 2; b++) for (let t = 0; t < 4; t++) if (mask[b][t]) ref -= logSoftmaxRow(L[b][t])[targets[b][t]];
    T.close(loss.item(), ref / 4, 1e-4, 'mean of -log p(target) over the 4 masked-in positions (divide by the mask count, not by B·T = 8)');
    loss.backward();
    const g = logits.grad;
    for (const [b, t] of [[0, 0], [0, 3], [1, 0], [1, 1]]) {
      const row = Array.from(g.subarray((b * 4 + t) * 5, (b * 4 + t + 1) * 5));
      T.ok(row.every((v) => v === 0), `position [${b}, ${t}] is masked out, so its logits must get zero gradient; got ${row.map((v) => v.toFixed(3))}`);
    }
    const row = Array.from(g.subarray(5, 10));
    T.ok(row.some((v) => v !== 0), 'masked-in positions must receive gradient');
    T.throws(() => m.maskedCrossEntropy(logits, targets, [[0, 0, 0, 0], [0, 0, 0, 0]]), 'a mask with no 1s has nothing to average: throw an Error instead of dividing by zero (which gives NaN)');
  } },
  { step: 'sequence', name: 'embedSequence puts image tokens first, then the text embeddings, with gradients to both', run(m, T) {
    const gpt = new GPT({ vocabSize: 7, blockSize: 8, nLayer: 1, nHead: 1, nEmbd: 4, seed: 1 });
    const img = randomTensor([2, 3, 4], rng(11), 1, true);
    const x = m.embedSequence(gpt, img, [[1, 2], [3, 4]]);
    T.shape(x, [2, 5, 4], '3 image tokens + 2 text tokens = 5 positions of width nEmbd');
    const X = T.arr(x), I = T.arr(img), W = T.arr(gpt.wte.weight);
    T.close(X[1].slice(0, 3), I[1], 1e-6, 'positions 0..2 are the image tokens unchanged');
    T.close(X[1][3], W[3], 1e-6, 'position 3 of example 1 is the embedding of token 3 (wte row 3)');
    T.close(X[0][4], W[2], 1e-6, 'position 4 of example 0 is the embedding of token 2');
    const R = randomTensor([2, 5, 4], rng(12));
    x.mul(R).sum().backward();
    T.ok(img.grad, 'the image tokens must receive gradient through the sequence (that is how the projector learns)');
    T.close(Array.from(img.grad.subarray(0, 12)), Array.from(R.data.subarray(0, 12)), 1e-6, 'image-token gradient = upstream gradient at positions 0..2');
    T.ok(gpt.wte.weight.grad && gpt.wte.weight.grad.some((v) => v !== 0), 'the text embeddings must receive gradient too');
    T.shape(m.forwardEmbeds(gpt, x), [2, 5, 7], 'the sequence runs through the GPT and gives one prediction per position');
  } },

  // ---------- step 4: contrastive alignment ----------
  { step: 'clip', name: 'clipLoss matches hand-computed values, normalises, and uses the temperature', run(m, T) {
    const I = Tensor.from([[1, 0], [0, 1]]), X = Tensor.from([[1, 0], [0, 1]]);
    const a = m.clipLoss(I, X, 1);
    T.ok(a && a.loss && a.logits, 'return { loss, logits }');
    T.shape(a.logits, [2, 2], 'logits is the [B, B] similarity matrix');
    T.close(a.loss.item(), 0.31326, 1e-4, 'τ=1, orthogonal pairs: logits [[1,0],[0,1]], loss = -log(e/(e+1)) = 0.3133');
    T.close(m.clipLoss(I, X, 0.5).loss.item(), 0.12693, 1e-4, 'τ=0.5 doubles the logits: loss = -log(e²/(e²+1)) = 0.1269; an unscaled score ignores τ');
    const big = m.clipLoss(Tensor.from([[5, 0], [0, 5]]), Tensor.from([[1, 0], [0, 3]]), 0.5);
    T.close(big.loss.item(), 0.12693, 1e-4, 'scaling a row must not change the loss: normalise rows to unit length before the dot product (cosine similarity)');
    T.close(big.logits, [[2, 0], [0, 2]], 1e-4, 'logits = cosine similarity / τ');
    const n = m.l2normalize(Tensor.from([[3, 4], [0, -2]]));
    T.close(n, [[0.6, 0.8], [0, -1]], 1e-5, 'l2normalize scales each row to length 1');
  } },
  { step: 'clip', name: 'clipLoss is symmetric: images pick captions AND captions pick images', run(m, T) {
    const next = rng(13);
    const img = randomTensor([3, 4], next), txt = randomTensor([3, 4], next);
    const ref = refClip(T.arr(img), T.arr(txt), 0.2);
    const got = m.clipLoss(img, txt, 0.2);
    T.close(T.arr(got.logits), ref.logits, 1e-4, 'logits[i][j] = cos(img_i, txt_j) / τ');
    T.close(got.loss.item(), ref.loss, 1e-4, 'loss = (CE over rows + CE over columns) / 2; only the row direction trains images to find captions, not captions to find images');
    T.close(m.clipLoss(txt, img, 0.2).loss.item(), ref.loss, 1e-4, 'swapping the two towers must give the same loss');
  } },
  { step: 'clip', name: 'training two towers with your loss makes the diagonal win', run(m, T) {
    const next = rng(14);
    const shapes = ['square', 'circle', 'cross', 'triangle'];
    const places = ['top left', 'bottom right', 'top right', 'bottom left'];
    const images = shapes.map((s, i) => m.drawShape(s, places[i], next));
    const caps = shapes.map((s, i) => [i, 4 + i]);                  // two word ids per caption
    const pix = new Tensor({ shape: [4, 256], data: Float32Array.from(images.flatMap((im) => Array.from(im.data))) });
    const imgTower = new Linear(256, 16, { next, std: 1 / 16 });
    const words = new Embedding(8, 16, { next });
    const bag = ops.zeros([4, 8]);
    caps.forEach((ids, b) => ids.forEach((id) => { bag.data[b * 8 + id] = 0.5; }));
    const params = [...imgTower.parameters(), ...words.parameters()];
    const opt = new AdamW(params, { lr: 0.02 });
    let first = null, last = null;
    for (let s = 0; s < 40; s++) {
      const { loss } = m.clipLoss(imgTower.forward(pix), new Tensor(bag).matmul(words.weight), 0.1);
      if (s === 0) first = loss.item();
      last = loss.item();
      opt.zeroGrad(); loss.backward(); opt.step();
    }
    T.ok(last < first * 0.5, `loss should fall by more than half in 40 steps (${first.toFixed(3)} -> ${last.toFixed(3)}); check the gradient path through your loss`);
    const L = T.arr(m.clipLoss(imgTower.forward(pix), new Tensor(bag).matmul(words.weight), 0.1).logits);
    for (let i = 0; i < 4; i++) {
      const rowBest = L[i].indexOf(Math.max(...L[i]));
      const col = L.map((r) => r[i]);
      const colBest = col.indexOf(Math.max(...col));
      T.eq(rowBest, i, `image ${i} (${shapes[i]}) should be most similar to its own caption`);
      T.eq(colBest, i, `caption ${i} should be most similar to its own image`);
    }
  } },

  // ---------- step 5: the captioner ----------
  { step: 'captioner', name: 'captionLoss is the masked loss over [image tokens][caption], padding included', run(m, T) {
    const tok = testTokenizer();
    const model = tinyCaptioner(m, tok);
    const images = [m.drawShape('circle', 'top left', rng(15)), m.drawShape('cross', 'bottom right', rng(16))];
    const captions = ['a circle', 'a cross at the bottom right'];
    const loss = m.captionLoss(model, images, captions);
    T.ok(loss instanceof Tensor && loss.size === 1, 'captionLoss returns a scalar Tensor');
    // reference: build the padded sequence and the masked mean by hand
    const ids = captions.map((c) => tok.encode(c));
    const width = 6, P = 4;
    const padded = ids.map((c) => c.concat(new Array(width - c.length).fill(tok.eos)));
    const logits = T.arr(m.forwardEmbeds(model.gpt, m.concatTokens(model.imageTokens(images), model.gpt.wte.forward(padded))));
    let ref = 0, count = 0;
    ids.forEach((c, b) => {
      const want = c.concat([tok.eos]);
      want.forEach((id, j) => { ref -= logSoftmaxRow(logits[b][P - 1 + j])[id]; count++; });
    });
    T.eq(count, 10, 'sanity: 3 + 7 supervised positions');
    T.close(loss.item(), ref / count, 1e-4, 'captionLoss = mean over the 10 caption positions (each caption plus its eos) after padding the short caption with eos; padding and image slots must not count');
    loss.backward();
    T.ok(model.projector.fc1.weight.grad && model.projector.fc1.weight.grad.some((v) => v !== 0), 'the caption loss must train the projector: this is the only way pixels learn to speak the GPT\'s language');
  } },
  { step: 'captioner', name: 'caption() decodes greedily, stops at eos and respects maxNewTokens', run(m, T) {
    const tok = testTokenizer();
    const model = tinyCaptioner(m, tok);
    const img = m.drawShape('square', 'top right', rng(17));
    const C = model.gpt.config.nEmbd;
    // Rig the GPT: final LayerNorm outputs beta everywhere, and beta lines up with one word's embedding.
    const rig = (id) => {
      model.gpt.lnF.gamma.data.fill(0);
      for (let j = 0; j < C; j++) { model.gpt.wte.weight.data[id * C + j] = 1; model.gpt.lnF.beta.data[j] = 1; }
    };
    const circle = tok.itos.indexOf('circle');
    rig(circle);                                              // 'circle' always wins
    const c = m.caption(model, img, { maxNewTokens: 3 });
    T.eq(typeof c, 'string', 'caption() returns the decoded string');
    T.eq(c, 'circle circle circle', 'with "circle" always the argmax, greedy decoding writes it maxNewTokens=3 times and stops');
    T.eq(m.caption(model, img, { maxNewTokens: 3 }), c, 'greedy decoding is deterministic');
    let long;
    try { long = m.caption(model, img, { maxNewTokens: 50 }); }
    catch (e) { T.fail(`caption() with maxNewTokens 50 threw "${e.message}": stop after min(maxNewTokens, blockSize - nPatches) words so the sequence never exceeds blockSize`); }
    T.eq(long.split(' ').length, model.gpt.config.blockSize - model.nPatches,
      `blockSize 16 with 4 image tokens leaves room for 12 words: stop after min(maxNewTokens, blockSize - nPatches) = 12, got ${long.split(' ').length}`);
    rig(tok.eos);                                             // now eos (id 9, not 0) wins
    for (let j = 0; j < C; j++) model.gpt.wte.weight.data[circle * C + j] = 0;
    T.eq(m.caption(model, img, { maxNewTokens: 5 }), '', 'when eos is the argmax at the first step the caption is empty: stop at model.eos (here id 9, not 0) and do not include it');
  } },
  { step: 'captioner', name: 'a tiny captioner overfits two images and captions each correctly', run(m, T) {
    const tok = testTokenizer();
    const model = tinyCaptioner(m, tok, 2);
    const images = [m.drawShape('circle', 'top left', rng(18)), m.drawShape('cross', 'bottom right', rng(19))];
    const captions = ['a circle at the top left', 'a cross at the bottom right'];
    const opt = new AdamW(model.parameters(), { lr: 0.01 });
    let loss = null;
    for (let s = 0; s < 40; s++) {
      loss = m.captionLoss(model, images, captions);
      opt.zeroGrad(); loss.backward(); opt.step();
    }
    T.ok(loss.item() < 0.2, `loss after 40 steps is ${loss.item().toFixed(3)}; expected < 0.2 on two memorisable examples`);
    T.eq(m.caption(model, images[0]), captions[0], 'the trained captioner must read the circle image');
    T.eq(m.caption(model, images[1]), captions[1], 'the trained captioner must read the cross image');
  } },
];
