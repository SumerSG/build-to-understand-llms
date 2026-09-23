// Attention from scratch demo: load the first block's attention weights from the pre-trained checkpoint into YOUR
// MultiHeadAttention, run it on a real token sequence, draw one heatmap per head, check it against the
// reference implementation, prove causality numerically, gradient-check it, and show why long context is
// expensive. If the checkpoint cannot be loaded (mid-retrain, or a browser without JSON imports), the
// demo falls back to random weights and character labels: every check still runs.
import * as ops from 'lib/ops.js';
import { Tensor } from 'lib/tensor.js';
import { MultiHeadAttention as RefMHA } from 'lib/attention.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { rng } from 'lib/util.js';

const SENTENCE = 'The cat sees the dog and the dog sees the cat.';

/** Try the checkpoint; return null when it is unavailable so the demo can fall back. */
async function loadCheckpoint(lab) {
  try {
    const model = (await import('lib/checkpoints/tiny-gpt.json', { with: { type: 'json' } })).default;
    const tokenizer = (await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default;
    if (!model || !model.params || !model.params['blocks.0.attn.qkv.weight']) throw new Error('checkpoint has no block 0 attention weights');
    return { model, tokenizer: BPETokenizer.fromJSON(tokenizer) };
  } catch (e) {
    lab.log(`Checkpoint not available (${String(e.message).split(' imported from')[0].slice(0, 120)}); using random weights instead.`);
    return null;
  }
}

const rawOf = (p) => ops.raw(p.shape, Float32Array.from(p.data));

export default async function demo(m, lab) {
  const ckpt = await loadCheckpoint(lab);
  const nEmbd = ckpt ? ckpt.model.config.nEmbd : 64;
  const nHead = ckpt ? ckpt.model.config.nHead : 4;
  const next = rng(5);

  // ---------- 1. tokens and the layer input ----------
  let ids, labels, x;
  if (ckpt) {
    ids = ckpt.tokenizer.encode(SENTENCE);
    labels = ids.map((id) => ckpt.tokenizer.decode([id]).replace(/ /g, '␣'));
    const w = ckpt.model.params;
    // Same input block 0 sees: token embedding + position embedding, then the block's first LayerNorm.
    const emb = ops.add(ops.embed(rawOf(w['wte.weight']), ids), ops.embed(rawOf(w['wpe.weight']), ids.map((_, i) => i)));
    const normed = ops.layerNorm(emb, rawOf(w['blocks.0.ln1.gamma']), rawOf(w['blocks.0.ln1.beta']));
    x = new Tensor(ops.reshape(normed, [1, ids.length, nEmbd]), { requiresGrad: true });
  } else {
    labels = SENTENCE.slice(0, 12).split('').map((c) => (c === ' ' ? '␣' : c));
    ids = labels.map((_, i) => i);
    x = Tensor.randn([1, ids.length, nEmbd], next, 1, { requiresGrad: true });
  }
  const T = ids.length;
  lab.log(`Sequence of ${T} tokens: ${labels.join(' | ')}`);

  // ---------- 2. your layer, with the checkpoint's block-0 attention weights ----------
  const layer = new m.MultiHeadAttention({ nEmbd, nHead, next: rng(1) });
  const ref = new RefMHA({ nEmbd, nHead, next: rng(1) });
  if (ckpt) {
    const w = ckpt.model.params;
    for (const [name, dst] of [['blocks.0.attn.qkv.weight', 'qkv.weight'], ['blocks.0.attn.qkv.bias', 'qkv.bias'], ['blocks.0.attn.proj.weight', 'proj.weight'], ['blocks.0.attn.proj.bias', 'proj.bias']]) {
      const [module, field] = dst.split('.');
      layer[module][field].data.set(w[name].data);
      ref[module][field].data.set(w[name].data);
    }
  } else {
    // Random weights: large enough that the heads differ from uniform averaging, small enough that the
    // float32 gradient check below stays meaningful (std 0.3 makes the sum-of-squares loss too large).
    for (const [a, b] of layer.parameters().map((p, i) => [p, ref.parameters()[i]])) {
      const r = ops.randn(a.shape, next, 0.1);
      a.data.set(r.data);
      b.data.set(r.data);
    }
  }
  const t0 = performance.now();
  const out = layer.forward(x);
  const forwardMs = performance.now() - t0;
  const refOut = ref.forward(x);
  lab.check(out.shape.length === 3 && out.shape[1] === T && out.shape[2] === nEmbd, `forward must return [1, ${T}, ${nEmbd}], got [${out.shape}]`);
  let maxDiff = 0;
  for (let i = 0; i < out.data.length; i++) maxDiff = Math.max(maxDiff, Math.abs(out.data[i] - refOut.data[i]));
  lab.check(maxDiff < 1e-4, `your output differs from lib/attention.js by ${maxDiff}`);
  lab.check(layer.lastWeights && layer.lastWeights.shape.length === 4, 'forward must store lastWeights [B, H, T, T]');
  await lab.tick();

  // ---------- 3. one heatmap per head ----------
  const W = layer.lastWeights;
  const rowsOf = (h) => Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) => +W.data[(h * T + i) * T + j].toFixed(4)));
  const prevTokenShare = [];
  for (let h = 0; h < nHead; h++) {
    const rows = rowsOf(h);
    // How much of each row (after the first) lands on the immediately previous token?
    let share = 0;
    for (let i = 1; i < T; i++) share += rows[i][i - 1];
    prevTokenShare.push(share / (T - 1));
    lab.heatmap({ title: `Head ${h}: attention weights (rows = query token, columns = key token)`, rows, rowLabels: labels, colLabels: labels, min: 0, max: 1 });
  }
  await lab.tick();

  // ---------- 4. entropy per head, against the uniform-over-visible-keys ceiling ----------
  const entropy = m.entropyPerHead(W);
  let ceiling = 0;
  for (let i = 0; i < T; i++) ceiling += Math.log(i + 1);
  ceiling /= T;
  lab.bar({ title: `Mean attention entropy per head (nats; uniform over the visible keys would be ${ceiling.toFixed(2)})`, labels: entropy.map((_, h) => `head ${h}`), values: entropy.map((e) => +e.toFixed(3)) });
  lab.table({
    title: 'Per-head summary',
    columns: ['head', 'mean entropy (nats)', 'share on previous token', 'share on itself'],
    rows: entropy.map((e, h) => {
      const rows = rowsOf(h);
      let self = 0;
      for (let i = 0; i < T; i++) self += rows[i][i];
      return [h, +e.toFixed(3), +prevTokenShare[h].toFixed(3), +(self / T).toFixed(3)];
    }),
  });

  // ---------- 5. causality, numerically ----------
  const t = Math.floor(T / 2) - 1;
  const probe = m.causalityProbe(layer, x, t, rng(2));
  lab.check(probe.maxBefore === 0, `positions 0..${t} changed by ${probe.maxBefore} when later tokens were perturbed`);
  lab.check(probe.maxAfter > 0, 'perturbing later tokens must change later outputs');
  lab.md(`**Causality probe.** Gaussian noise was added to tokens ${t + 1}…${T - 1}. Outputs at positions 0…${t} moved by **${probe.maxBefore}** (exactly zero); outputs at positions ${t + 1}…${T - 1} moved by up to **${probe.maxAfter.toFixed(4)}**.`);
  await lab.tick();

  // ---------- 6. gradient check through the whole layer ----------
  const gc = m.gradCheckAttention(layer, x);
  const nInputs = 1 + layer.parameters().length;
  lab.check(gc.ok, `gradient check failed with max relative error ${gc.maxRelErr}`);
  lab.md(`**Gradient check.** ${gc.details.length} analytic-vs-numerical comparisons across ${nInputs} tensors (the input and ${nInputs - 1} parameters): max relative error **${gc.maxRelErr.toExponential(2)}**, tolerance 1e-2.`);
  await lab.tick();

  // ---------- 7. why long context is expensive ----------
  // Per layer: projections 8·T·C² FLOPs, score matmul + weighted sum 4·T²·C. Share of the quadratic term:
  const lengths = [];
  for (let L = 64; L <= 131072; L *= 2) lengths.push(L);
  const share = (C) => lengths.map((L) => +(100 * (4 * L * L * C) / (8 * L * C * C + 4 * L * L * C)).toFixed(1));
  lab.plot({
    title: 'Share of attention-layer FLOPs spent on the T×T score and value matmuls',
    x: lengths.map((L) => Math.log2(L)),
    series: [{ name: `this model, C = ${nEmbd}`, values: share(nEmbd) }, { name: 'GPT-2 small, C = 768', values: share(768) }, { name: 'C = 4096', values: share(4096) }],
    xlabel: 'log2(T)  (6 = 64 tokens, 17 = 131,072 tokens)', ylabel: '% of layer FLOPs',
  });
  const gpt2At1k = share(768)[lengths.indexOf(1024)];
  const gpt2At32k = share(768)[lengths.indexOf(32768)];
  const bytesAt32k = 32768 * 32768 * 4;

  const peak = prevTokenShare.indexOf(Math.max(...prevTokenShare));
  lab.done(`Your multi-head attention (C = ${nEmbd}, H = ${nHead}, ${layer.parameters().reduce((s, p) => s + p.data.length, 0)} parameters) ran on ${T} tokens in ${forwardMs.toFixed(1)} ms and matched lib/attention.js to within **${maxDiff.toExponential(1)}** ${ckpt ? 'using the pre-trained checkpoint\'s block-0 weights' : 'on random weights'}. Head entropies were ${entropy.map((e) => e.toFixed(2)).join(', ')} nats against a uniform ceiling of ${ceiling.toFixed(2)}; head ${peak} put the most weight on the previous token (**${(100 * prevTokenShare[peak]).toFixed(0)}%** of each row on average). Perturbing tokens after position ${t} moved earlier outputs by exactly **${probe.maxBefore}** and later ones by up to ${probe.maxAfter.toFixed(3)}, and the gradient check passed with max relative error **${gc.maxRelErr.toExponential(1)}**. For GPT-2 small the T×T matmuls are ${gpt2At1k}% of attention FLOPs at T = 1,024 and ${gpt2At32k}% at T = 32,768, where one head's float32 weight matrix alone would be ${(bytesAt32k / 2 ** 30).toFixed(0)} GiB: the reason the KV cache and FlashAttention exist (see the KV cache module).`);
}
