import { GPT } from 'lib/gpt.js';
import { noGrad } from 'lib/tensor.js';
import { AdamW, clipGradNorm } from 'lib/optim.js';
import { rng, randInt, shuffle } from 'lib/util.js';

// The goal: pixels in, words out. Your patchify → PatchEmbed → Projector turns each 16×16 image into 16
// vectors in a fresh two-layer GPT's embedding space, your masked loss teaches the GPT to caption them,
// and your clipLoss separately aligns an image tower with a text tower (CLIP-style zero-shot labels).
export default async function demo(m, lab) {
  const t0 = performance.now();
  const tokenizer = m.captionTokenizer();
  const train = m.makeDataset(200, 1);
  const heldOut = m.makeDataset(48, 99);
  const patch = 4;

  // 1. What an image looks like to the model: a grid of pixels becomes a sequence of patch tokens.
  const example = heldOut[0];
  const grid = m.patchify(example.image, patch);
  const [nPatches, patchDim] = grid.shape;
  lab.check(nPatches === 16 && patchDim === 16, `patchify(16×16, 4) should give [16, 16], got [${grid.shape}]`);
  const rows = (t) => Array.from({ length: t.shape[0] }, (_, i) => Array.from(t.data.subarray(i * t.shape[1], (i + 1) * t.shape[1])));
  lab.heatmap({ title: `The image: "${example.caption}" (16×16 pixels)`, rows: rows(example.image), min: 0, max: 1 });
  lab.heatmap({
    title: 'The same image as 16 patch tokens (one row per 4×4 patch, reading order; 16 pixels each)',
    rows: rows(grid), rowLabels: Array.from({ length: nPatches }, (_, i) => `patch ${i} (r${Math.floor(i / 4)} c${i % 4})`), min: 0, max: 1,
  });

  // 2. The captioner: a fresh module-06 GPT that has never seen an image, plus your vision tower and projector.
  const gpt = new GPT({ vocabSize: tokenizer.vocabSize, blockSize: 32, nLayer: 2, nHead: 2, nEmbd: 32, seed: 1 });
  const model = new m.Captioner({ gpt, tokenizer, eos: tokenizer.eos, patch, dVision: 32, seed: 2 });
  const visionParams = model.vision.parameters().concat(model.projector.parameters()).reduce((s, p) => s + p.size, 0);
  lab.log(`vocab ${tokenizer.vocabSize} words; each example is ${nPatches} image tokens + 6 caption tokens; GPT ${gpt.numParams()} params, vision tower + projector ${visionParams}`);

  // Exact-match accuracy in ONE batched forward pass. Greedy decoding writes the true caption exactly when,
  // fed the true prefix, the argmax at every caption position (and at the final eos) is the true token.
  // So this equals caption()'s exact-match rate, at the cost of one forward instead of seven per image.
  const accuracy = (set) => noGrad(() => {
    const ex = set.map((e) => m.captionTargets(nPatches, tokenizer.encode(e.caption), tokenizer.eos));
    const x = m.embedSequence(gpt, model.imageTokens(set.map((e) => e.image)), ex.map((e) => e.textIds));
    const logits = m.forwardEmbeds(gpt, x);
    const [, N, V] = logits.shape;
    let ok = 0;
    ex.forEach((e, b) => {
      let all = true;
      for (let t = 0; t < N && all; t++) {
        if (!e.mask[t]) continue;
        const row = logits.data.subarray((b * N + t) * V, (b * N + t + 1) * V);
        let best = 0;
        for (let v = 1; v < V; v++) if (row[v] > row[best]) best = v;
        all = best === e.targets[t];
      }
      if (all) ok++;
    });
    return ok / set.length;
  });
  const steps = 120, batchSize = 16, evalEvery = 20, clipSteps = 100, temperature = 0.1;
  const opt = new AdamW(model.parameters(), { lr: 3e-3, weightDecay: 0 });
  const next = rng(3);
  const losses = [], accSteps = [0], accs = [accuracy(heldOut)];
  for (let s = 1; s <= steps; s++) {
    const batch = Array.from({ length: batchSize }, () => train[randInt(next, train.length)]);
    const loss = m.captionLoss(model, batch.map((b) => b.image), batch.map((b) => b.caption));
    opt.zeroGrad();
    loss.backward();
    clipGradNorm(model.parameters(), 1.0);
    opt.step();
    losses.push(loss.item());
    if (s % evalEvery === 0) { accSteps.push(s); accs.push(accuracy(heldOut)); }
    if (s % 5 === 0) { lab.progress(s / (steps + clipSteps), `captioner step ${s}: loss ${losses[s - 1].toFixed(3)}`); await lab.tick(); }
  }
  const trainSeconds = (performance.now() - t0) / 1000;
  lab.check(Number.isFinite(losses[losses.length - 1]), 'caption loss became NaN');
  lab.plot({ title: 'Caption loss (caption tokens only; image slots and padding are masked out)', series: [{ name: 'train loss', values: losses }], xlabel: 'step', ylabel: 'cross-entropy (nats/token)', yscale: 'log' });
  lab.plot({ title: `Exact-match caption accuracy on ${heldOut.length} held-out images (teacher-forced: each word predicted from the true previous words)`, x: accSteps, series: [{ name: 'held-out accuracy', values: accs }], xlabel: 'step', ylabel: 'fraction correct' });

  const tokens = noGrad(() => model.imageTokens([example.image]));
  lab.heatmap({
    title: `After training: the same image as the GPT receives it, ${nPatches} projected tokens × ${gpt.config.nEmbd} dims (the words' embedding space)`,
    rows: rows({ shape: [nPatches, gpt.config.nEmbd], data: tokens.data }),
    rowLabels: Array.from({ length: nPatches }, (_, i) => `token ${i}`),
  });

  const predicted = heldOut.map((ex) => m.caption(model, ex.image));
  let exact = 0, shapeOk = 0, placeOk = 0;
  predicted.forEach((p, i) => {
    if (p === heldOut[i].caption) exact++;
    if (p.split(' ')[1] === heldOut[i].shape) shapeOk++;
    if (p.endsWith(heldOut[i].place)) placeOk++;
  });
  const finalAcc = exact / heldOut.length;
  const tfAcc = accs[accs.length - 1];

  // 3. The contrastive head: two towers that never generate a word, trained only to agree.
  const clip = new m.ClipHead({ vocabSize: tokenizer.vocabSize, patch, dim: 32, next: rng(4) });
  const clipOpt = new AdamW(clip.parameters(), { lr: 1e-2, weightDecay: 0 });
  const byCaption = new Map();
  for (const ex of train) { if (!byCaption.has(ex.caption)) byCaption.set(ex.caption, []); byCaption.get(ex.caption).push(ex); }
  const captions = [...byCaption.keys()].sort();
  const clipNext = rng(5);
  const clipLosses = [];
  for (let s = 0; s < clipSteps; s++) {
    // One image for each of the 16 captions per batch. Two images with the SAME caption in one batch
    // would be scored as a wrong match for each other (a false negative), so the batch avoids them.
    const chosen = shuffle(clipNext, captions.slice());
    const exs = chosen.map((c) => { const pool = byCaption.get(c); return pool[randInt(clipNext, pool.length)]; });
    const { loss } = m.clipLoss(clip.encodeImages(exs.map((e) => e.image)), clip.encodeTexts(exs.map((e) => tokenizer.encode(e.caption))), temperature);
    clipOpt.zeroGrad();
    loss.backward();
    clipOpt.step();
    clipLosses.push(loss.item());
    if (s % 5 === 0) { lab.progress((steps + s) / (steps + clipSteps), `CLIP step ${s}: loss ${loss.item().toFixed(3)}`); await lab.tick(); }
  }
  // Zero-shot labelling: score each held-out image against all 16 captions and take the best.
  const textEmb = clip.encodeTexts(captions.map((c) => tokenizer.encode(c)));
  const imgEmb = clip.encodeImages(heldOut.map((e) => e.image));
  const scores = m.l2normalize(imgEmb).matmul(m.l2normalize(textEmb).transpose()).scale(1 / temperature);   // [48, 16]
  const C = captions.length;
  let zeroShot = 0;
  const bestCaption = heldOut.map((ex, i) => {
    let best = 0;
    for (let j = 1; j < C; j++) if (scores.data[i * C + j] > scores.data[i * C + best]) best = j;
    if (captions[best] === ex.caption) zeroShot++;
    return captions[best];
  });
  const zeroShotAcc = zeroShot / heldOut.length;
  const firstPerCaption = captions.map((c) => heldOut.findIndex((ex) => ex.caption === c));
  const shown = firstPerCaption.filter((i) => i >= 0).slice(0, 10);
  lab.heatmap({
    title: `CLIP similarity / τ: held-out images (rows) against all ${C} captions (columns); the true caption should be brightest`,
    rows: shown.map((i) => Array.from(scores.data.subarray(i * C, (i + 1) * C))),
    rowLabels: shown.map((i) => heldOut[i].caption.replace('a ', '').replace(' at the', ',')),
    colLabels: captions.map((c) => c.replace('a ', '').replace(' at the', ',')),
  });

  lab.table({
    title: 'Six held-out images: what the captioner wrote, and which caption CLIP scored highest',
    columns: ['image', 'true caption', 'captioner (greedy)', 'correct', 'CLIP best match'],
    rows: heldOut.slice(0, 6).map((ex, i) => [i, ex.caption, predicted[i], predicted[i] === ex.caption ? 'yes' : 'no', bestCaption[i]]),
  });

  const tokensFor = (side, p) => (side / p) ** 2;
  lab.bar({
    title: 'Image tokens per image (patch tokens = (side / patch)²) vs the 6-token caption',
    labels: ['ours: 16 px, patch 4', 'ViT-B/16 at 224 px', 'LLaVA-1.5: 336 px, patch 14', '1024 px, patch 16', 'caption'],
    values: [tokensFor(16, 4), tokensFor(224, 16), tokensFor(336, 14), tokensFor(1024, 16), 6],
  });

  const seconds = (performance.now() - t0) / 1000;
  lab.done(`Your multimodal GPT learned to read pixels. After **${steps} steps** (${trainSeconds.toFixed(1)} s, batch ${batchSize}) the captioner writes the exact caption for **${(100 * finalAcc).toFixed(0)}%** of ${heldOut.length} held-out images (shape word right ${(100 * shapeOk / heldOut.length).toFixed(0)}%, position right ${(100 * placeOk / heldOut.length).toFixed(0)}%; teacher-forced, where every word is predicted from the true previous words in one batched pass, ${(100 * tfAcc).toFixed(0)}%, never lower than greedy, because greedy decoding also has to recover from its own earlier words; chance is ${(100 / C).toFixed(1)}%). Caption loss fell from ${losses[0].toFixed(2)} to ${losses[losses.length - 1].toFixed(3)} nats per token. The GPT was never told these were pictures: it saw ${nPatches} vectors per image from your projector, in the same space as its word embeddings. Separately, the CLIP head trained for ${clipSteps} steps (loss ${clipLosses[0].toFixed(2)} → ${clipLosses[clipLosses.length - 1].toFixed(2)}) labels **${(100 * zeroShotAcc).toFixed(0)}%** of held-out images correctly by similarity alone, without generating a word. Total ${seconds.toFixed(1)} s. At LLaVA-1.5's resolution one image costs ${tokensFor(336, 14)} tokens, roughly 100× the caption.`);
}
