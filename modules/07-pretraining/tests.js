import { Tensor, crossEntropy, noGrad, isGradEnabled } from 'lib/tensor.js';
import { GPT } from 'lib/gpt.js';
import { AdamW as RefAdamW, clipGradNorm as refClip } from 'lib/optim.js';
import { CharTokenizer } from 'lib/tokenizer.js';
import { toyCorpus } from 'lib/data.js';
import { rng, randInt } from 'lib/util.js';

// A tiny corpus and model so every test stays well under a second.
const TEXT = toyCorpus(60, 3);
const TOK = new CharTokenizer(TEXT);
const IDS = TOK.encode(TEXT);
const TINY = { vocabSize: TOK.vocabSize, blockSize: 8, nLayer: 1, nHead: 2, nEmbd: 16, seed: 5 };
const tinyModel = () => new GPT(TINY);

function param(values) {
  return new Tensor({ shape: [values.length], data: Float32Array.from(values) }, { requiresGrad: true });
}
function setGrad(p, values) {
  p.grad = Float32Array.from(values);
}
// A fixed batch drawn straight from the corpus, independent of the learner's getBatch.
function fixedBatch(blockSize, batchSize, starts) {
  const x = [], y = [];
  for (let b = 0; b < batchSize; b++) {
    const s = starts[b % starts.length];
    x.push(IDS.slice(s, s + blockSize));
    y.push(IDS.slice(s + 1, s + blockSize + 1));
  }
  return { x, y };
}
function findStart(row) {
  outer: for (let s = 0; s + row.length <= IDS.length; s++) {
    for (let t = 0; t < row.length; t++) if (IDS[s + t] !== row[t]) continue outer;
    return s;
  }
  return -1;
}

// The loop the instructions describe, built from the reference optimizer and clipping and the learner's
// (already tested) getBatch: one AdamW for the whole run, lr from the schedule written before each step,
// one rng shared by training and validation batches.
function referenceRun(m, cfg) {
  const model = new GPT({ vocabSize: cfg.vocabSize, blockSize: cfg.blockSize, nLayer: cfg.nLayer, nHead: cfg.nHead, nEmbd: cfg.nEmbd, seed: cfg.seed });
  const opt = new RefAdamW(model.parameters(), { lr: cfg.lr, betas: [0.9, 0.95], weightDecay: cfg.weightDecay });
  const next = rng(cfg.seed);
  const out = [];
  for (let step = 0; step < cfg.steps; step++) {
    opt.lr = m.cosineWithWarmup(step, { warmup: cfg.warmup, total: cfg.steps, peak: cfg.lr });
    const { x, y } = m.getBatch(cfg.trainIds, cfg.blockSize, cfg.batchSize, next);
    opt.zeroGrad();
    const loss = crossEntropy(model.forward(x), y);
    loss.backward();
    const gradNorm = refClip(model.parameters(), cfg.maxGradNorm);
    opt.step();
    const r = { step, loss: loss.item(), gradNorm };
    if ((step + 1) % cfg.evalInterval === 0 || step === cfg.steps - 1) {
      r.valLoss = noGrad(() => {
        let total = 0;
        for (let k = 0; k < cfg.evalBatches; k++) { const b = m.getBatch(cfg.valIds, cfg.blockSize, cfg.batchSize, next); total += crossEntropy(model.forward(b.x), b.y).item(); }
        return total / cfg.evalBatches;
      });
    }
    out.push(r);
  }
  return { model, history: out };
}

export const tests = [
  // ---------- step 1: getBatch ----------
  { step: 'batch', name: 'getBatch returns batchSize windows of blockSize ids, each a contiguous slice of the corpus', run(m, T) {
    const { x, y } = m.getBatch(IDS, 8, 4, T.rng(1));
    T.shape(x, [4, 8], 'x must be batchSize rows of blockSize ids');
    T.shape(y, [4, 8], 'y must have the same shape as x');
    for (let b = 0; b < 4; b++) {
      T.ok(findStart(x[b]) >= 0, `x[${b}] = [${x[b]}] is not a contiguous window of the corpus`);
      for (let t = 0; t < 8; t++) T.ok(Number.isInteger(y[b][t]), `y[${b}][${t}] is ${y[b][t]}; every target must be a token id (did a window run past the end?)`);
    }
  } },
  { step: 'batch', name: 'y is x shifted right by exactly one token (the target for position t is the token at t+1)', run(m, T) {
    const { x, y } = m.getBatch(IDS, 8, 6, T.rng(2));
    for (let b = 0; b < 6; b++) {
      const s = findStart(x[b]);
      T.ok(s >= 0, 'x rows must come from the corpus');
      for (let t = 0; t < 7; t++) T.eq(y[b][t], x[b][t + 1], `y[${b}][${t}] must equal x[${b}][${t + 1}]: each position predicts the NEXT token, so y is x shifted by one`);
      T.eq(y[b][7], IDS[s + 8], 'the last target is the token just after the window (one past the end of x)');
    }
  } },
  { step: 'batch', name: 'windows are seeded by `next`, can start at the last valid offset, and a too-short corpus throws', run(m, T) {
    const a = m.getBatch(IDS, 8, 4, T.rng(9));
    const b = m.getBatch(IDS, 8, 4, T.rng(9));
    const c = m.getBatch(IDS, 8, 4, T.rng(10));
    T.eq(a.x, b.x, 'the same seed must give the same batch (reproducibility)');
    const draw = T.rng(9), lastStart = IDS.length - 8 - 1;
    const want = Array.from({ length: 4 }, () => { const s = randInt(draw, lastStart + 1); return IDS.slice(s, s + 8); });
    T.eq(a.x, want, 'each row must draw its OWN start with randInt(next, lastStart + 1), in row order, where lastStart = ids.length − blockSize − 1');
    T.ok(new Set(a.x.map((r) => r.join(','))).size > 1, 'the rows of a batch must come from different random offsets: drawing one start for the whole batch gives batchSize copies of the same window');
    T.ok(JSON.stringify(a.x) !== JSON.stringify(c.x), 'different seeds must give different windows: draw the start offset from `next`');
    const short = IDS.slice(0, 12); // valid starts are 0..3 for blockSize 8
    const seen = new Set();
    for (let k = 0; k < 40; k++) {
      for (const row of m.getBatch(short, 8, 4, T.rng(100 + k)).x) {
        let s = -1;
        for (let q = 0; q + 8 <= short.length; q++) if (short.slice(q, q + 8).every((v, i) => v === row[i])) { s = q; break; }
        seen.add(s);
      }
    }
    T.ok(seen.has(3), 'with 12 ids and blockSize 8 the start offset 3 (= length − blockSize − 1) is valid and must be reachable');
    T.ok(!seen.has(4) && !seen.has(-1), 'a start of 4 would need id 12, which does not exist: the last valid start is length − blockSize − 1');
    T.throws(() => m.getBatch(IDS.slice(0, 8), 8, 1, T.rng(1)), 'a corpus of exactly blockSize ids has no room for the shifted target and must throw');
  } },

  // ---------- step 2: AdamW ----------
  { step: 'adamw', name: 'the first step is bias-corrected: every parameter moves by about lr, whatever its gradient scale', run(m, T) {
    const p = param([1, 1, 1, 1, 1]);
    setGrad(p, [1e-3, 1, 1e3, -50, 1e-5]);
    const opt = new m.AdamW([p], { lr: 0.1 });
    opt.step();
    T.close(T.arr(p.data), [0.9, 0.9, 0.9, 1.1, 0.9], 1e-3, 'after one step p = 1 − lr·sign(g): with bias correction mHat/sqrt(vHat) = g/|g| (without it you would get 0.447·lr). eps goes OUTSIDE the square root: sqrt(vHat + eps) would shrink the step of the 1e-5 gradient tenfold');
    T.eq(opt.t, 1, 'step must count steps (t) so the bias correction can use beta^t');
  } },
  { step: 'adamw', name: 'weight decay is decoupled: it shrinks p directly and never enters the moments', run(m, T) {
    const p = param([2, -4]);
    setGrad(p, [0, 0]);
    const opt = new m.AdamW([p], { lr: 0.1, weightDecay: 0.5 });
    opt.step();
    T.close(T.arr(p.data), [2 * (1 - 0.05), -4 * (1 - 0.05)], 1e-5, 'with zero gradient the only change is p −= lr·wd·p; coupled L2 (adding wd·p to g) would move p by ±lr instead');
    const q = param([3]);
    setGrad(q, [0]);
    const opt2 = new m.AdamW([q], { lr: 0.1, weightDecay: 0 });
    opt2.step();
    T.close(q.data[0], 3, 1e-7, 'weightDecay 0 and zero gradient must leave p unchanged');
  } },
  { step: 'adamw', name: 'matches the reference AdamW over 6 steps with running moments, and zeroGrad clears grads', run(m, T) {
    const next = T.rng(4);
    const init = Array.from({ length: 12 }, () => next() - 0.5);
    const p = param(init), r = param(init);
    const opt = new m.AdamW([p], { lr: 0.05, betas: [0.9, 0.95], weightDecay: 0.1 });
    const ref = new RefAdamW([r], { lr: 0.05, betas: [0.9, 0.95], weightDecay: 0.1 });
    for (let s = 0; s < 6; s++) {
      const g = Array.from({ length: 12 }, () => (next() - 0.5) * (s + 1));
      setGrad(p, g); setGrad(r, g);
      opt.step(); ref.step();
      T.close(T.arr(p.data), T.arr(r.data), 1e-5, `after step ${s + 1} the parameters differ from the reference: m and v must persist across steps and be bias-corrected with beta^t`);
    }
    opt.zeroGrad();
    T.ok(p.grad === null || Array.from(p.grad).every((v) => v === 0), 'zeroGrad must clear the gradient so the next backward starts from zero');
    const noGradParam = param([1]);
    T.ok((() => { new m.AdamW([noGradParam], { lr: 1 }).step(); return true; })(), 'a parameter without a gradient must be skipped, not crash');
    T.close(noGradParam.data[0], 1, 1e-7, 'a parameter without a gradient must not move');
  } },

  // ---------- step 3: clipping and one training step ----------
  { step: 'trainstep', name: 'clipGradNorm uses the GLOBAL norm across all parameters and rescales only when it exceeds maxNorm', run(m, T) {
    const a = param([0, 0, 0]), b = param([0, 0, 0]);
    setGrad(a, [3, 0, 0]); setGrad(b, [0, 4, 0]); // per-tensor norms 3 and 4; global norm 5
    const norm = m.clipGradNorm([a, b], 4.5);
    T.close(norm, 5, 1e-5, 'the returned value is the norm BEFORE clipping, over all parameters together (sqrt(3² + 4²) = 5), not per tensor');
    T.close(T.arr(a.grad), [2.7, 0, 0], 1e-5, 'every gradient is scaled by maxNorm / norm = 0.9 (per-tensor clipping would leave a alone since 3 < 4.5)');
    T.close(T.arr(b.grad), [0, 3.6, 0], 1e-5, 'b is scaled by the same factor');
    const c = param([0, 0]);
    setGrad(c, [0.3, -0.4]);
    const small = m.clipGradNorm([c], 1);
    T.close(small, 0.5, 1e-6);
    T.close(T.arr(c.grad), [0.3, -0.4], 1e-7, 'a gradient below maxNorm must be left exactly as it is');
    const d = param([0]);
    T.ok(Number.isFinite(m.clipGradNorm([d], 1)), 'parameters without a gradient are skipped');
  } },
  { step: 'trainstep', name: 'trainStep returns the loss of the batch before the update plus the pre-clip grad norm, and updates the model', run(m, T) {
    const model = tinyModel();
    const { x, y } = fixedBatch(8, 4, [3, 40, 77, 120]);
    const before = noGrad(() => crossEntropy(model.forward(x), y).item());
    const snapshot = Float32Array.from(model.wte.weight.data);
    const opt = new RefAdamW(model.parameters(), { lr: 1e-2 });
    const out = m.trainStep(model, opt, x, y, { maxGradNorm: 1 });
    T.ok(out && typeof out.loss === 'number' && typeof out.gradNorm === 'number', 'return { loss, gradNorm } as plain numbers (use .item())');
    T.close(out.loss, before, 1e-5, 'loss must be the mean cross-entropy of the forward pass that produced the gradients, i.e. before optimizer.step()');
    T.close(out.loss, Math.log(TINY.vocabSize), 0.25, `a freshly initialised model predicts nearly uniformly, so the loss is about ln(V) = ${Math.log(TINY.vocabSize).toFixed(3)} nats`);
    T.ok(out.gradNorm > 0, 'gradNorm must be the global gradient norm computed after backward()');
    let changed = 0;
    for (let i = 0; i < snapshot.length; i++) if (snapshot[i] !== model.wte.weight.data[i]) changed++;
    T.ok(changed > 0, 'optimizer.step() must run: the embedding weights did not change');
    T.ok(refClip(model.parameters(), Infinity) <= 1 + 1e-4, 'after trainStep the stored gradients must have been clipped to maxGradNorm = 1');
  } },
  { step: 'trainstep', name: 'two consecutive steps match a reference loop (grads are zeroed each step, clipped, then applied)', run(m, T) {
    const a = tinyModel(), b = tinyModel();
    const optA = new RefAdamW(a.parameters(), { lr: 5e-3 });
    const optB = new RefAdamW(b.parameters(), { lr: 5e-3 });
    const batch = fixedBatch(8, 4, [10, 55, 90, 130]);
    const ref = [];
    for (let s = 0; s < 2; s++) {
      optB.zeroGrad();
      const loss = crossEntropy(b.forward(batch.x), batch.y);
      loss.backward();
      const gn = refClip(b.parameters(), 0.5);
      optB.step();
      ref.push({ loss: loss.item(), gradNorm: gn });
    }
    for (let s = 0; s < 2; s++) {
      const out = m.trainStep(a, optA, batch.x, batch.y, { maxGradNorm: 0.5 });
      T.close(out.loss, ref[s].loss, 1e-4, `step ${s + 1}: loss differs from the reference loop`);
      T.close(out.gradNorm, ref[s].gradNorm, 1e-3, `step ${s + 1}: gradNorm differs (forgetting zeroGrad makes gradients accumulate; forgetting to clip changes the update)`);
    }
    T.ok(ref[1].loss < ref[0].loss, 'sanity: the reference loss falls on a repeated batch');
    T.close(T.arr(a.wte.weight.data), T.arr(b.wte.weight.data), 1e-4, 'the weights after two steps must match the reference loop');
  } },

  // ---------- step 4: the schedule ----------
  { step: 'schedule', name: 'linear warmup from 0 to peak, then peak/10 at the end and beyond', run(m, T) {
    const o = { warmup: 100, total: 1000, peak: 1e-3 };
    T.close(m.cosineWithWarmup(0, o), 0, 1e-12, 'step 0 is lr 0: Adam has no moment estimates yet');
    T.close(m.cosineWithWarmup(25, o), 0.25e-3, 1e-9, 'a quarter of the way through warmup the lr is a quarter of the peak');
    T.close(m.cosineWithWarmup(100, o), 1e-3, 1e-9, 'at step = warmup the lr reaches the peak');
    T.close(m.cosineWithWarmup(1000, o), 1e-4, 1e-9, 'at step = total the lr is min, which defaults to peak/10');
    T.close(m.cosineWithWarmup(5000, o), 1e-4, 1e-9, 'after total the lr stays at min');
    T.close(m.cosineWithWarmup(1000, { ...o, min: 3e-4 }), 3e-4, 1e-9, 'an explicit min is respected');
  } },
  { step: 'schedule', name: 'the decay is a cosine (not linear) and never increases after warmup', run(m, T) {
    const o = { warmup: 100, total: 1000, peak: 1e-3 };
    const q = m.cosineWithWarmup(325, o); // progress 0.25 through the decay
    T.close(q, 1e-4 + 0.5 * 9e-4 * (1 + Math.cos(Math.PI * 0.25)), 1e-9, 'a quarter into the decay a cosine gives min + 0.5·(peak−min)·(1+cos(π/4)) ≈ 0.868e-3; linear decay would give 0.775e-3');
    T.close(m.cosineWithWarmup(550, o), 0.55e-3, 1e-9, 'halfway through the decay the lr is exactly (peak + min)/2');
    let prev = Infinity;
    for (let s = 100; s <= 1000; s += 5) { const v = m.cosineWithWarmup(s, o); T.ok(v <= prev + 1e-15, `lr rose between steps ${s - 5} and ${s}`); prev = v; }
    T.close(m.cosineWithWarmup(3, { warmup: 0, total: 10, peak: 1 }), 0.1 + 0.45 * (1 + Math.cos(Math.PI * 0.3)), 1e-9, 'warmup 0 must not divide by zero; the decay starts at step 0');
  } },

  // ---------- step 5: estimateLoss ----------
  { step: 'eval', name: 'estimateLoss averages the cross-entropy of evalBatches batches drawn with `next`', run(m, T) {
    const model = tinyModel();
    const got = m.estimateLoss(model, IDS, { blockSize: 8, batchSize: 4, evalBatches: 3, next: T.rng(21) });
    const next = T.rng(21);
    let total = 0;
    for (let k = 0; k < 3; k++) { const { x, y } = m.getBatch(IDS, 8, 4, next); total += noGrad(() => crossEntropy(model.forward(x), y).item()); }
    T.ok(typeof got === 'number', 'return a plain number (the mean), not a Tensor');
    T.close(got, total / 3, 1e-5, 'must equal the mean over exactly evalBatches batches drawn with getBatch from the same rng');
    const one = m.estimateLoss(model, IDS, { blockSize: 8, batchSize: 4, evalBatches: 1, next: T.rng(21) });
    T.ok(Math.abs(one - got) > 1e-6, 'with evalBatches 1 the estimate must differ: average several batches, do not stop after the first');
  } },
  { step: 'eval', name: 'runs under noGrad and leaves the model untouched (no graph, no grads, no updates)', run(m, T) {
    const model = tinyModel();
    const snapshot = model.parameters().map((p) => Float32Array.from(p.data));
    const flags = [];
    const original = model.forward.bind(model);
    model.forward = (ids) => { flags.push(isGradEnabled()); return original(ids); };
    m.estimateLoss(model, IDS, { blockSize: 8, batchSize: 2, evalBatches: 2, next: T.rng(1) });
    T.ok(flags.length === 2, `forward must be called exactly evalBatches (2) times, got ${flags.length}`);
    T.ok(flags.every((f) => f === false), 'forward must run inside noGrad(...): recording a graph for evaluation wastes memory and is how real runs OOM during validation');
    T.ok(isGradEnabled(), 'grad recording must be switched back on after estimateLoss returns');
    T.ok(model.parameters().every((p) => p.grad === null), 'evaluation must not call backward(): parameters must have no gradient afterwards');
    model.parameters().forEach((p, i) => T.eq(T.arr(p.data), T.arr(snapshot[i]), 'evaluation must not change any parameter'));
  } },

  // ---------- step 6: train and sample ----------
  { step: 'train', name: 'train runs `steps` steps, records lr/loss/gradNorm per step, follows the schedule and calls onStep', async run(m, T) {
    const cfg = { ...TINY, trainIds: IDS.slice(0, 600), valIds: IDS.slice(600), steps: 12, batchSize: 4, lr: 1e-2, warmup: 4, weightDecay: 0.1, maxGradNorm: 1, evalInterval: 5, evalBatches: 2 };
    const seen = [];
    const out = await m.train(cfg, (r) => seen.push(r.step));
    T.ok(out && out.model instanceof GPT, 'resolve to { model, history, tokensSeen } with the trained GPT');
    T.eq(out.history.length, 12, 'history must have one record per step');
    T.eq(seen, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'onStep must be called once per step, in order, with the record');
    T.eq(out.tokensSeen, 12 * 4 * 8, 'tokensSeen = steps × batchSize × blockSize');
    for (const r of out.history) {
      T.ok(Number.isFinite(r.loss) && Number.isFinite(r.gradNorm), `step ${r.step}: loss and gradNorm must be finite numbers`);
      const want = m.cosineWithWarmup(r.step, { warmup: 4, total: 12, peak: 1e-2 });
      T.close(r.lr, want, 1e-9, `step ${r.step}: lr must follow cosineWithWarmup(step, { warmup, total: steps, peak: lr }) and be written to optimizer.lr before the step`);
    }
    T.close(out.history[0].lr, 0, 1e-12, 'step 0 must use lr 0 (warmup starts at zero)');
  } },
  { step: 'train', name: 'validation loss appears every evalInterval steps and on the last step, and training lowers the loss', async run(m, T) {
    const cfg = { ...TINY, trainIds: IDS.slice(0, 600), valIds: IDS.slice(600), steps: 24, batchSize: 4, lr: 2e-2, warmup: 4, weightDecay: 0.1, maxGradNorm: 1, evalInterval: 10, evalBatches: 2 };
    const { history } = await m.train(cfg);
    const withVal = history.filter((r) => typeof r.valLoss === 'number').map((r) => r.step);
    T.eq(withVal, [9, 19, 23], 'valLoss must be recorded on steps 9 and 19 (every 10th step) and on the final step 23, and nowhere else');
    const first = history.slice(0, 4).reduce((s, r) => s + r.loss, 0) / 4;
    const last = history.slice(-4).reduce((s, r) => s + r.loss, 0) / 4;
    T.ok(last < first - 0.1, `the mean loss of the last 4 steps (${last.toFixed(3)}) must be clearly below the first 4 (${first.toFixed(3)}): the loop is not learning`);
    const again = await m.train(cfg);
    T.close(again.history.map((r) => r.loss), history.map((r) => r.loss), 1e-6, 'the same config and seed must reproduce the same run exactly');
  } },
  { step: 'train', name: 'the run matches a reference loop step for step: one persistent AdamW, the scheduled lr applied, clipping and weight decay from the config, one shared rng', async run(m, T) {
    const cfg = { ...TINY, trainIds: IDS.slice(0, 600), valIds: IDS.slice(600), steps: 10, batchSize: 4, lr: 2e-2, warmup: 3, weightDecay: 0.5, maxGradNorm: 0.3, evalInterval: 4, evalBatches: 2 };
    const fresh = m.makeModel(cfg).parameters().map((p) => T.arr(p.data));
    let atStep0 = null;
    const out = await m.train(cfg, (r, model) => { if (r.step === 0) atStep0 = model.parameters().map((p) => T.arr(p.data)); });
    T.ok(atStep0 !== null, 'onStep(record, model) must be called with the model');
    T.close(atStep0, fresh, 1e-7, 'the scheduled lr at step 0 is 0, so after the first step the weights must be exactly the initial ones: write cosineWithWarmup(step, …) into optimizer.lr BEFORE trainStep, not only into the record');
    const ref = referenceRun(m, cfg);
    for (let s = 0; s < cfg.steps; s++) {
      const a = out.history[s], b = ref.history[s];
      T.close(a.loss, b.loss, 2e-3, `step ${s}: train loss ${a.loss.toFixed(5)} differs from the reference loop's ${b.loss.toFixed(5)}. Build ONE AdamW before the loop (a new optimizer each step resets m, v and t), pass it { lr, betas: [0.9, 0.95], weightDecay }, set optimizer.lr from the schedule each step, and draw batches from the one rng`);
      T.close(a.gradNorm, b.gradNorm, 2e-3, `step ${s}: gradNorm differs from the reference loop`);
      if (b.valLoss !== undefined) T.close(a.valLoss, b.valLoss, 2e-3, `step ${s}: valLoss ${a.valLoss} differs from the reference ${b.valLoss}: estimate it on valIds (not trainIds) with the same \`next\` the training batches use`);
    }
    const w = out.model.parameters(), rw = ref.model.parameters();
    w.forEach((p, i) => T.close(T.arr(p.data), T.arr(rw[i].data), 2e-3, 'the final weights differ from the reference loop: check that maxGradNorm and weightDecay from the config reach trainStep and AdamW'));
  } },
  { step: 'train', name: 'sample continues the prompt with maxNewTokens tokens decoded through the tokenizer', run(m, T) {
    const model = tinyModel();
    const text = m.sample(model, TOK, 'The ', { maxNewTokens: 12, temperature: 0.9, next: T.rng(8) });
    T.ok(typeof text === 'string', 'sample must return a string');
    T.ok(text.startsWith('The '), 'the prompt must be included at the start of the returned text');
    T.eq(text.length, 16, 'a character tokenizer decodes one character per token: 4 prompt chars + 12 new tokens');
    const expect = TOK.decode(model.generate(TOK.encode('The '), { maxNewTokens: 12, temperature: 0.9, next: T.rng(8) }));
    T.eq(text, expect, 'must match encode → model.generate → decode with the same rng and temperature');
    const other = m.sample(model, TOK, 'The ', { maxNewTokens: 12, temperature: 0.9, next: T.rng(9) });
    T.ok(other !== text, 'a different seed must give a different continuation: pass `next` through to generate');
  } },
];
