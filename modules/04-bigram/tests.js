import { Tensor, crossEntropy } from 'lib/tensor.js';
import { SGD, AdamW } from 'lib/optim.js';
import { randInt } from 'lib/util.js';

/** Ids from a small seeded Markov chain over V symbols, so tests have realistic bigram statistics. */
function chainIds(T, V, n, seed) {
  const next = T.rng(seed);
  const ids = [0];
  for (let t = 1; t < n; t++) {
    const prev = ids[t - 1];
    // Mostly move to (prev + 1) mod V, sometimes stay, rarely jump anywhere.
    const u = next();
    ids.push(u < 0.6 ? (prev + 1) % V : u < 0.85 ? prev : Math.floor(next() * V));
  }
  return ids;
}

function table(rows) {
  const V = rows.length;
  return { shape: [V, V], data: Float32Array.from(rows.flat()) };
}

const rowSums = (t) => {
  const V = t.shape[1];
  const out = [];
  for (let i = 0; i < V; i++) { let s = 0; for (let j = 0; j < V; j++) s += t.data[i * V + j]; out.push(s); }
  return out;
};

export const tests = [
  // ---------- step 1: counting ----------
  { step: 'count', name: 'countBigrams tallies every adjacent pair into a [V, V] Float32Array table', run(m, T) {
    const c = m.countBigrams([0, 1, 1, 2, 0, 1], 3);
    T.shape(c, [3, 3]);
    T.ok(c.data instanceof Float32Array, 'the table must be a raw tensor { shape, data: Float32Array } like every table in this module');
    T.eq(T.arr(c), [[0, 2, 0], [0, 1, 1], [1, 0, 0]], 'pair (0,1) occurs twice, (1,1), (1,2) and (2,0) once; nothing else');
    let total = 0;
    for (const v of c.data) total += v;
    T.eq(total, 5, 'six ids contain exactly five transitions: do not count a pair after the last id, and do not wrap around to the first');
  } },
  { step: 'count', name: 'bigramProbs with alpha = 0 normalises each row of counts to a distribution', run(m, T) {
    const p = m.bigramProbs(table([[3, 1, 0], [0, 0, 0], [2, 2, 2]]), 0);
    T.shape(p, [3, 3]);
    T.close(T.arr(p)[0], [0.75, 0.25, 0], 1e-6, 'row 0 must be counts / row total (3+1+0 = 4)');
    T.close(T.arr(p)[2], [1 / 3, 1 / 3, 1 / 3], 1e-6, 'row 2 has three equal counts');
    T.close(T.arr(p)[1], [1 / 3, 1 / 3, 1 / 3], 1e-6, 'a row with no counts and alpha = 0 must fall back to uniform (1/V) rather than 0/0 = NaN');
    T.close(rowSums(p), [1, 1, 1], 1e-6, 'every row is a probability distribution over the next token');
  } },
  { step: 'count', name: 'add-alpha smoothing gives every pair (count + alpha) / (row total + alpha·V) and leaves the counts untouched', run(m, T) {
    const counts = table([[3, 1, 0], [0, 0, 0], [2, 2, 2]]);
    const before = Array.from(counts.data);
    const p = m.bigramProbs(counts, 1);
    T.close(T.arr(p)[0], [4 / 7, 2 / 7, 1 / 7], 1e-6, 'row 0 with alpha = 1: (3+1)/(4+3), (1+1)/7, (0+1)/7 — the unseen pair gets mass');
    T.close(T.arr(p)[1], [1 / 3, 1 / 3, 1 / 3], 1e-6, 'an empty row becomes uniform: the prior alone');
    T.close(rowSums(p), [1, 1, 1], 1e-6, 'rows must still sum to 1: the denominator gains alpha·V, one alpha per column');
    const q = m.bigramProbs(counts, 0.5);
    T.close(T.arr(q)[0], [3.5 / 5.5, 1.5 / 5.5, 0.5 / 5.5], 1e-6, 'alpha = 0.5 must give a different, less smoothed row than alpha = 1');
    T.eq(Array.from(counts.data), before, 'bigramProbs must not modify the count table it was given');
  } },

  // ---------- step 2: evaluation ----------
  { step: 'evaluate', name: 'negLogLikelihood averages −log P(next | prev) over the n − 1 transitions', run(m, T) {
    const p = table([[0.5, 0.5, 0], [0.25, 0.25, 0.5], [0.1, 0.1, 0.8]]);
    const ids = [0, 1, 2, 2, 0];
    const expected = -(Math.log(0.5) + Math.log(0.5) + Math.log(0.8) + Math.log(0.1)) / 4;
    T.close(m.negLogLikelihood(p, ids), expected, 1e-6, 'sum −log p over the 4 transitions (0→1, 1→2, 2→2, 2→0) and divide by 4, not by 5 and not left as a sum');
    T.close(m.negLogLikelihood(p, [2, 2]), -Math.log(0.8), 1e-6, 'a single transition: the mean equals that one −log p');
  } },
  { step: 'evaluate', name: 'a uniform table scores NLL log V, so its perplexity is V (as if guessing among V equally likely tokens)', run(m, T) {
    const V = 7;
    const uniform = { shape: [V, V], data: new Float32Array(V * V).fill(1 / V) };
    const ids = chainIds(T, V, 200, 11);
    const nll = m.negLogLikelihood(uniform, ids);
    T.close(nll, Math.log(V), 1e-5, 'every transition has probability 1/V, so mean −log p = log V');
    T.close(m.perplexity(nll), V, 1e-4, 'perplexity = exp(NLL) turns log V back into V');
  } },
  { step: 'evaluate', name: 'a transition never seen in training gives infinite perplexity at alpha = 0 and a finite one for any alpha > 0', run(m, T) {
    const V = 5;
    const train = chainIds(T, V, 400, 3).filter((x, i, a) => !(i > 0 && a[i - 1] === 3 && x === 0)); // remove 3→0
    const counts = m.countBigrams(train, V);
    T.eq(counts.data[3 * V + 0], 0, 'setup: the pair 3→0 must be absent from the training ids');
    const heldOut = [1, 2, 3, 0, 1];
    const ppl0 = m.perplexity(m.negLogLikelihood(m.bigramProbs(counts, 0), heldOut));
    T.eq(ppl0, Infinity, 'P(0 | 3) = 0 under pure counting, so −log 0 = Infinity and the whole held-out score is Infinity: one unseen pair ruins the model');
    const ppl1 = m.perplexity(m.negLogLikelihood(m.bigramProbs(counts, 0.1), heldOut));
    T.ok(Number.isFinite(ppl1) && ppl1 > 1, `with alpha = 0.1 every pair has probability > 0, so perplexity must be finite (got ${ppl1})`);
  } },
  { step: 'evaluate', name: 'perplexity is exp of the NLL', run(m, T) {
    T.close(m.perplexity(0), 1, 1e-9, 'NLL 0 means every transition had probability 1: perplexity 1');
    T.close(m.perplexity(Math.log(20)), 20, 1e-6, 'NLL log 20 means the model is as uncertain as a fair 20-sided die');
    T.close(m.perplexity(2), Math.exp(2), 1e-6);
  } },

  // ---------- step 3: sampling ----------
  { step: 'sample', name: 'sampleNext inverts the cumulative distribution of the row for `prev`', run(m, T) {
    const p = table([[0.25, 0.5, 0.25], [1, 0, 0], [0, 0, 1]]);
    const fixed = (u) => () => u;
    T.eq(m.sampleNext(p, 0, fixed(0.1)), 0, 'u = 0.1 falls in the first 0.25 of the row');
    T.eq(m.sampleNext(p, 0, fixed(0.25)), 1, 'u = 0.25 is not < 0.25, so it belongs to the next token (cumulative boundaries are half-open)');
    T.eq(m.sampleNext(p, 0, fixed(0.7)), 1, 'u = 0.7 is inside [0.25, 0.75)');
    T.eq(m.sampleNext(p, 0, fixed(0.75)), 2, 'u = 0.75 is past the second boundary');
    T.eq(m.sampleNext(p, 0, fixed(0.999)), 2);
    T.eq(m.sampleNext(p, 1, fixed(0.9)), 0, 'row 1 puts all its mass on token 0: the sampler must read the row of prev, not row 0');
    T.eq(m.sampleNext(p, 2, fixed(0.001)), 2, 'row 2 puts all its mass on token 2');
  } },
  { step: 'sample', name: 'temperature reshapes the row (p^(1/T), renormalised) and sample frequencies match the row', run(m, T) {
    const p = table([[0.1, 0.9], [0.5, 0.5]]);
    const fixed = (u) => () => u;
    T.eq(m.sampleNext(p, 0, fixed(0.2), 2), 0, 'at T = 2 the row becomes [0.25, 0.75] (square roots, renormalised), so u = 0.2 picks token 0; at T = 1 it would pick token 1');
    T.eq(m.sampleNext(p, 0, fixed(0.05), 0.2), 1, 'at T = 0.2 the row becomes about [0.00002, 0.99998] (fifth powers), so u = 0.05 picks token 1; at T = 1 it would pick token 0');
    const r = table([[0.25, 0.25, 0.5], [1, 0, 0], [0, 0, 1]]);
    const before = Array.from(r.data);
    T.eq(m.sampleNext(r, 0, fixed(0.1), 0.5), 0, 'at T = 0.5 the row [0.25, 0.25, 0.5] becomes [1/6, 1/6, 2/3] (squares, renormalised), so u = 0.1 picks token 0; without renormalising, the squares sum to 0.375 and the walk would run off the end');
    T.eq(m.sampleNext(r, 0, fixed(0.2), 0.5), 1, 'at T = 0.5, u = 0.2 lies in [1/6, 1/3), so it picks token 1: the reshaped row must be divided by its sum');
    T.eq(Array.from(r.data), before, 'temperature must reshape a copy of the row: sampling must never modify the model\'s table');
    const q = table([[0.2, 0.5, 0.3], [1, 0, 0], [0, 0, 1]]);
    const next = T.rng(5);
    const freq = [0, 0, 0];
    for (let i = 0; i < 3000; i++) freq[m.sampleNext(q, 0, next)] += 1;
    T.close(freq.map((f) => f / 3000), [0.2, 0.5, 0.3], 0.04, 'over 3000 seeded draws the empirical frequencies must track the row within a few percent');
  } },
  { step: 'sample', name: 'generate chains n samples, each conditioned on the previous one, reproducibly from the seed', run(m, T) {
    const cycle = table([[0, 1, 0], [0, 0, 1], [1, 0, 0]]);
    T.eq(m.generate(cycle, 0, 5, T.rng(1)), [1, 2, 0, 1, 2], 'a deterministic cycle table must produce the cycle: the previous sample becomes the next context');
    T.eq(m.generate(cycle, 2, 3, T.rng(1)), [0, 1, 2], 'the start token is the context for the first sample and is not included in the output');
    const p = table([[0.2, 0.5, 0.3], [0.6, 0.2, 0.2], [0.3, 0.3, 0.4]]);
    const a = m.generate(p, 0, 40, T.rng(9)), b = m.generate(p, 0, 40, T.rng(9));
    T.eq(a.length, 40, 'generate(probs, start, n, next) returns exactly n ids');
    T.eq(a, b, 'the same seed must give the same text: all randomness comes from `next`');
    T.ok(a.every((x) => x === 0 || x === 1 || x === 2), 'every id must be a valid token index');
  } },

  // ---------- step 4: the neural bigram ----------
  { step: 'neural', name: 'initNeural builds a trainable [V, V] table of small random logits', run(m, T) {
    const model = m.initNeural(6, T.rng(1));
    T.ok(model && model.W instanceof Tensor, 'model.W must be a Tensor from lib/tensor.js so autograd can reach it');
    T.shape(model.W, [6, 6]);
    T.eq(model.V, 6);
    T.ok(model.W.requiresGrad === true, 'W is the parameter being trained: it must have requiresGrad = true (Tensor.param or { requiresGrad: true })');
    const vals = Array.from(model.W.data);
    T.ok(vals.some((v) => v !== 0), 'W must be randomly initialised, not all zeros (zeros are fine mathematically but hide bugs in the lookup)');
    T.ok(vals.every((v) => Math.abs(v) < 0.2), 'initial logits must be small (std about 0.01) so the model starts near uniform');
    T.ok(new Set(vals).size > 6, 'entries must differ from one another');
    const sd = (xs) => { const mu = xs.reduce((a, b) => a + b, 0) / xs.length; return Math.sqrt(xs.reduce((a, b) => a + (b - mu) ** 2, 0) / xs.length); };
    const sd0 = sd(Array.from(m.initNeural(30, T.rng(4)).W.data));
    T.ok(sd0 > 0.008 && sd0 < 0.012, `the default initialisation must be Gaussian with std 0.01 (measured ${sd0.toFixed(4)} over 900 entries)`);
    const sd1 = sd(Array.from(m.initNeural(30, T.rng(4), 0.5).W.data));
    T.ok(sd1 > 0.4 && sd1 < 0.6, `initNeural(V, next, std) must honour its std argument (asked for 0.5, measured ${sd1.toFixed(3)})`);
    T.eq(Array.from(m.initNeural(6, T.rng(1)).W.data), vals, 'the same seed must give the same initialisation');
    T.close(vals, Array.from(Tensor.randn([6, 6], T.rng(1), 0.01).data), 1e-7, 'W must be drawn with Tensor.randn([V, V], next, std, { requiresGrad: true }): the training tests seed `next` and compare against a reference W drawn that way, so any other draw order gives different numbers');
  } },
  { step: 'neural', name: 'neuralLogits(model, xs) is an embedding lookup: row xs[i] of W, still attached to the graph', run(m, T) {
    const model = m.initNeural(4, T.rng(2));
    const W = model.W;
    for (let i = 0; i < 16; i++) W.data[i] = i / 10;
    const logits = m.neuralLogits(model, [2, 0, 2]);
    T.ok(logits instanceof Tensor, 'return a Tensor (use W.embed), not a raw array: the loss must be able to backpropagate into W');
    T.shape(logits, [3, 4]);
    T.close(T.arr(logits), [[0.8, 0.9, 1.0, 1.1], [0, 0.1, 0.2, 0.3], [0.8, 0.9, 1.0, 1.1]], 1e-6, 'row i of the output is W[xs[i], :]');
    logits.sum().backward();
    T.ok(W.grad !== null, 'the lookup must record W as its input so backward() reaches W.grad');
    T.close(Array.from(W.grad), [1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0], 1e-6, 'row 2 was looked up twice and row 0 once: gradients scatter-add into the rows that were used');
  } },
  { step: 'neural', name: 'neuralLoss is the mean cross-entropy: −log softmax(W[x])[y] averaged over the batch', run(m, T) {
    const V = 5;
    const model = m.initNeural(V, T.rng(3));
    model.W.data.fill(0);
    const loss0 = m.neuralLoss(model, [0, 1, 2, 3], [1, 2, 3, 4]);
    T.ok(loss0 instanceof Tensor && loss0.size === 1, 'the loss is a scalar Tensor (crossEntropy from lib/tensor.js returns one)');
    T.close(loss0.item(), Math.log(V), 1e-5, 'with all-zero logits every row is uniform, so the loss is log V: the softmax must be normalised over the row');
    model.W.data.set([1, 2, 3, 0, 0], 0);  // row 0
    const z = Math.exp(1) + Math.exp(2) + Math.exp(3) + 2;
    const expected = (-(2 - Math.log(z)) - (3 - Math.log(z))) / 2;
    T.close(m.neuralLoss(model, [0, 0], [1, 2]).item(), expected, 1e-5, 'mean over the two targets of −log softmax(row 0)[y]');
    model.W.zeroGrad();
    const loss = m.neuralLoss(model, [0, 0], [1, 2]);
    loss.backward();
    const g = Array.from(model.W.grad.slice(0, 5));
    const p = [1, 2, 3, 0, 0].map((l) => Math.exp(l) / z);
    T.close(g, [p[0], p[1] - 0.5, p[2] - 0.5, p[3], p[4]], 1e-5, 'd loss / d W[0] = softmax − onehot, averaged over the batch: this is what training pushes on');
  } },

  // ---------- step 5: training and the comparison ----------
  { step: 'train', name: 'trainStep does zeroGrad → loss → backward → step and returns the loss it stepped on', run(m, T) {
    const ids = chainIds(T, 4, 200, 21);
    const xs = ids.slice(0, -1), ys = ids.slice(1);
    const model = m.initNeural(4, T.rng(7));
    const before = Array.from(model.W.data);
    const lossBefore = m.neuralLoss(model, xs, ys).item();
    const opt = new SGD([model.W], { lr: 1 });
    const returned = m.trainStep(model, opt, xs, ys);
    T.close(returned, lossBefore, 1e-6, 'return the scalar loss (a plain number) measured before the parameters moved');
    T.ok(Array.from(model.W.data).some((v, i) => v !== before[i]), 'opt.step() must move W');
    T.ok(m.neuralLoss(model, xs, ys).item() < lossBefore, 'one gradient step on the same batch must lower its loss');
    // Reference: a second step must use only the fresh gradient, so the grads must have been zeroed first.
    const ref = m.initNeural(4, T.rng(7));
    const refOpt = new SGD([ref.W], { lr: 1 });
    for (let k = 0; k < 2; k++) { refOpt.zeroGrad(); const l = m.neuralLoss(ref, xs, ys); l.backward(); refOpt.step(); }
    m.trainStep(model, opt, xs, ys);
    T.close(Array.from(model.W.data), Array.from(ref.W.data), 1e-5, 'after two steps W must match a reference that zeroes gradients before each backward: forgetting zeroGrad accumulates stale gradients');
  } },
  { step: 'train', name: 'trainNeural returns one loss per step and the loss falls from about log V', run(m, T) {
    const V = 6;
    const ids = chainIds(T, V, 3000, 4);
    const model = m.initNeural(V, T.rng(1));
    const losses = m.trainNeural(model, ids, { steps: 120, batchSize: 256, lr: 0.1, next: T.rng(2) });
    T.eq(losses.length, 120, 'one loss value per training step');
    T.ok(losses.every((l) => Number.isFinite(l)), 'every loss must be a finite number');
    T.close(losses[0], Math.log(V), 0.05, 'the first loss is about log V: the fresh model is nearly uniform');
    const tail = losses.slice(-10).reduce((s, l) => s + l, 0) / 10;
    T.ok(tail < losses[0] - 0.3, `the loss must fall substantially (first ${losses[0].toFixed(3)}, last-10 mean ${tail.toFixed(3)}): the optimiser must update W`);
    // Reference loop, written independently: ONE AdamW with the given lr, a fresh makeBatch per step.
    const ref = Tensor.randn([V, V], T.rng(1), 0.01, { requiresGrad: true });
    const refOpt = new AdamW([ref], { lr: 0.03 });
    const refNext = T.rng(12);
    const refLosses = [];
    for (let s = 0; s < 25; s++) {
      const xs = [], ys = [];
      for (let b = 0; b < 64; b++) { const t = randInt(refNext, ids.length - 1); xs.push(ids[t]); ys.push(ids[t + 1]); }
      refOpt.zeroGrad(); const l = crossEntropy(ref.embed(xs), ys); l.backward(); refOpt.step(); refLosses.push(l.item());
    }
    const mine = m.initNeural(V, T.rng(1));
    const mineLosses = m.trainNeural(mine, ids, { steps: 25, batchSize: 64, lr: 0.03, next: T.rng(12) });
    T.close(mineLosses, refLosses, 1e-5, 'with steps 25, batchSize 64, lr 0.03 the losses must match a reference that builds ONE AdamW([W], { lr }) before the loop and draws a fresh makeBatch(ids, batchSize, next) every step: a new optimiser per step, a hard-coded lr, a reused batch or a different optimiser all change these numbers');
    T.close(Array.from(mine.W.data), Array.from(ref.data), 1e-5, 'and W must end where the reference W ends');
    const again = m.initNeural(V, T.rng(1));
    T.close(m.trainNeural(again, ids, { steps: 120, batchSize: 256, lr: 0.1, next: T.rng(2) }), losses, 1e-6, 'the same seeds must reproduce the same losses: draw batches from `next`');
  } },
  { step: 'train', name: 'trained on all pairs, the neural bigram converges to the count table (maximum likelihood = counting)', run(m, T) {
    const V = 4;
    const ids = chainIds(T, V, 300, 8);
    const counts = m.countBigrams(ids, V);
    const countProbs = m.bigramProbs(counts, 0);
    const model = m.initNeural(V, T.rng(3));
    const opt = new AdamW([model.W], { lr: 0.1 });
    const xs = ids.slice(0, -1), ys = ids.slice(1);
    for (let s = 0; s < 300; s++) m.trainStep(model, opt, xs, ys);
    const np = m.neuralProbs(model);
    T.shape(np, [V, V]);
    T.close(rowSums(np), [1, 1, 1, 1], 1e-4, 'neuralProbs must be softmax(W) row by row: a table of probabilities, not logits');
    let maxDiff = 0;
    for (let i = 0; i < V * V; i++) maxDiff = Math.max(maxDiff, Math.abs(np.data[i] - countProbs.data[i]));
    T.ok(maxDiff < 0.02, `after 300 full-batch AdamW steps every probability should be within 0.02 of the count model (max difference ${maxDiff.toFixed(4)}): gradient descent on cross-entropy recovers the counts`);
    T.close(m.negLogLikelihood(np, ids), m.negLogLikelihood(countProbs, ids), 0.01, 'and its NLL on the training ids matches the count model, the best any bigram can do');
  } },
];
