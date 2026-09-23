// Module 19 demo — quantise the pre-trained checkpoint with YOUR quantisers, run the model through YOUR
// weight-only kernel, and measure what the integers cost: reconstruction error against group size,
// next-token agreement with fp32 over 100 positions, the memory arithmetic for Llama 3, and what a single
// outlier weight does to each scheme. If the checkpoint cannot be loaded (mid-retrain, or a browser
// without JSON imports) a random-weight model stands in; the mechanics are identical, the text is noise.
import * as ops from 'lib/ops.js';
import { loadModel, forward } from 'lib/infer.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { GPT } from 'lib/gpt.js';
import { CORPUS, toyCorpus } from 'lib/data.js';

// The linear layers whose weights are quantised (W is [K, N] for y = x · W). Embedding lookups stay fp32,
// as most production recipes keep them; the tied LM head is quantised because it is a matmul.
const LINEARS = ['attn.qkv', 'attn.proj', 'mlp.fc', 'mlp.proj'];

// Error vs group size and bit width. "pc" is per-channel: one scale per output channel over all K inputs.
const SCHEMES = [
  { name: 'int8 pc', bits: 8, group: 'pc' },
  { name: 'int4 pc', bits: 4, group: 'pc' },
  { name: 'int4 g32', bits: 4, group: 32 },
  { name: 'int4 g16', bits: 4, group: 16 },
  { name: 'int4 g8', bits: 4, group: 8 },
  { name: 'int3 g16', bits: 3, group: 16 },
  { name: 'int2 g16', bits: 2, group: 16 },
];

async function loadCheckpoint(lab) {
  try {
    const json = (await import('lib/checkpoints/tiny-gpt.json', { with: { type: 'json' } })).default;
    const tok = (await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default;
    if (!json || !json.params || !json.config) throw new Error('checkpoint has no params');
    return { model: loadModel(json), tokenizer: BPETokenizer.fromJSON(tok), trained: true };
  } catch (e) {
    lab.log(`Checkpoint not available (${String(e.message).split(' imported from')[0].slice(0, 120)}); using a random-weight model instead.`);
    const tokenizer = BPETokenizer.train(toyCorpus(200, 1), { vocabSize: 256 });
    const gpt = new GPT({ vocabSize: tokenizer.vocabSize, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 64, seed: 3 });
    return { model: loadModel(gpt.toJSON()), tokenizer, trained: false };
  }
}

/** Quantise every linear weight and the LM head with the learner's quantizeWeight; returns { name: qw }. */
function quantizeModel(m, model, scheme) {
  const q = {};
  const opts = (K) => ({ bits: scheme.bits, groupSize: scheme.group === 'pc' ? K : scheme.group });
  for (let layer = 0; layer < model.config.nLayer; layer++) {
    for (const lin of LINEARS) {
      const name = `blocks.${layer}.${lin}.weight`;
      const W = model.w[name];
      q[name] = m.quantizeWeight(W, opts(W.shape[0]));
    }
  }
  // The head is logits = x · wteᵀ, so W = wteᵀ is [C, V]; stored [V, C], i.e. the layout of wte itself.
  const headW = ops.transpose(model.w['wte.weight']);
  q.head = m.quantizeWeight(headW, opts(headW.shape[0]));
  return q;
}

/** The same forward pass as lib/infer.js, with every linear layer routed through the learner's kernel. */
function forwardQuantized(m, model, q, ids) {
  const { nLayer, nHead, nEmbd } = model.config;
  const w = model.w;
  const time = ids.length;
  const headDim = nEmbd / nHead;
  const mask = ops.causalMask(time);
  const linear = (x, name) => ops.add(m.quantizedMatmul(x, q[name]), w[name.replace('.weight', '.bias')]);
  const split = (x) => ops.permute(ops.reshape(x, [time, nHead, headDim]), [1, 0, 2]);
  let x = ops.add(ops.embed(w['wte.weight'], ids), ops.embed(w['wpe.weight'], ids.map((_, i) => i)));
  for (let layer = 0; layer < nLayer; layer++) {
    const p = `blocks.${layer}`;
    const normed = ops.layerNorm(x, w[`${p}.ln1.gamma`], w[`${p}.ln1.beta`]);
    const qkv = linear(normed, `${p}.attn.qkv.weight`);
    const qh = split(ops.slice(qkv, 1, 0, nEmbd));
    const kh = split(ops.slice(qkv, 1, nEmbd, 2 * nEmbd));
    const vh = split(ops.slice(qkv, 1, 2 * nEmbd, 3 * nEmbd));
    const scores = ops.scale(ops.matmul(qh, ops.transpose(kh)), 1 / Math.sqrt(headDim));
    const weights = ops.softmax(ops.maskedFill(scores, mask, -Infinity));
    const attended = ops.reshape(ops.permute(ops.matmul(weights, vh), [1, 0, 2]), [time, nEmbd]);
    x = ops.add(x, linear(attended, `${p}.attn.proj.weight`));
    const normed2 = ops.layerNorm(x, w[`${p}.ln2.gamma`], w[`${p}.ln2.beta`]);
    const hidden = ops.gelu(linear(normed2, `${p}.mlp.fc.weight`));
    x = ops.add(x, linear(hidden, `${p}.mlp.proj.weight`));
  }
  const normed = ops.layerNorm(x, w['lnF.gamma'], w['lnF.beta']);
  return m.quantizedMatmul(normed, q.head); // [T, V]
}

const GB = 1e9;
const GiB = 1024 ** 3;

export default async function demo(m, lab) {
  const { model, tokenizer, trained } = await loadCheckpoint(lab);
  const cfg = model.config;
  const linearParams = Object.keys(model.w).filter((n) => n.endsWith('.weight') && LINEARS.some((l) => n.includes(l))).reduce((s, n) => s + model.w[n].data.length, 0) + model.w['wte.weight'].data.length;
  lab.log(`Model: ${cfg.nLayer} layers, ${cfg.nEmbd} wide, vocab ${cfg.vocabSize}${trained ? ' (pre-trained checkpoint)' : ' (random weights)'}; ${linearParams.toLocaleString()} weights in linear layers and the head will be quantised.`);

  // ---------- 1. reconstruction error vs scheme, and next-token agreement with fp32 ----------
  // Two 50-token windows of the corpus give 100 next-token predictions to compare (blockSize is 64).
  const text = trained ? CORPUS : toyCorpus(200, 1);
  const allIds = tokenizer.encode(text.slice(0, 4000));
  const windows = [allIds.slice(100, 150), allIds.slice(700, 750)];
  const refLogits = windows.map((ids) => forward(model, ids));
  const refTop = refLogits.map((l) => ops.argmax(l));
  const positions = refTop.reduce((s, t) => s + t.length, 0);

  const rows = [];
  const agree = [], mse = [];
  for (let i = 0; i < SCHEMES.length; i++) {
    const scheme = SCHEMES[i];
    const q = quantizeModel(m, model, scheme);
    // Weight error: relative L2 error of the dequantised weights, averaged over the quantised matrices.
    let wRel = 0, wMse = 0, count = 0;
    for (const name of Object.keys(q)) {
      const original = name === 'head' ? model.w['wte.weight'] : ops.transpose(model.w[name]);
      const e = m.errorStats(original, m.dequantize(q[name]));
      wRel += e.rel; wMse += e.mse; count++;
    }
    wRel /= count; wMse /= count;
    // Output error: run the model through the quantised kernel and compare with fp32.
    let same = 0, logitRel = 0;
    for (let wi = 0; wi < windows.length; wi++) {
      const logits = forwardQuantized(m, model, q, windows[wi]);
      const top = ops.argmax(logits);
      for (let t = 0; t < top.length; t++) if (top[t] === refTop[wi][t]) same++;
      logitRel += m.errorStats(refLogits[wi], logits).rel / windows.length;
    }
    const bpp = m.bitsPerParam({ bits: scheme.bits, groupSize: scheme.group === 'pc' ? 64 : scheme.group, scaleBits: 16 });
    mse.push(wMse); agree.push((100 * same) / positions);
    rows.push([scheme.name, +bpp.toFixed(3), wMse.toExponential(2), +(100 * wRel).toFixed(2), +(100 * logitRel).toFixed(2), `${same}/${positions}`]);
    lab.progress((i + 1) / SCHEMES.length, scheme.name);
    await lab.tick();
  }
  lab.table({
    title: 'Each scheme applied to every linear weight and the LM head (bits/param assumes fp16 scales; pc = per-channel, one scale per output row of K inputs: 64, or 256 for mlp.proj; bits/param shown for K = 64)',
    columns: ['scheme', 'bits/param', 'weight MSE', 'weight rel err %', 'logit rel err %', 'top-1 agreement with fp32'],
    rows,
  });
  const extraBits = m.bitsPerParam({ bits: 4, groupSize: 8 }) - m.bitsPerParam({ bits: 4, groupSize: 64 });
  const int4 = SCHEMES.map((s, i) => ({ s, i })).filter(({ s }) => s.bits === 4);
  lab.plot({
    title: 'int4: reconstruction error vs group size (per-channel plotted at 64, the K of most layers; mlp.proj has K = 256)',
    x: int4.map(({ s }) => (s.group === 'pc' ? 64 : s.group)),
    series: [{ name: 'weight MSE', values: int4.map(({ i }) => mse[i]) }],
    xlabel: 'values per scale (group size)', ylabel: 'MSE of dequantised weights', yscale: 'log',
  });
  lab.bar({ title: `Next-token top-1 agreement with fp32 over ${positions} positions (%)`, labels: SCHEMES.map((s) => s.name), values: agree.map((a) => +a.toFixed(1)) });
  lab.md(`The 8-bit and per-channel 4-bit models agree with fp32 on **${agree[0].toFixed(0)}%** and **${agree[1].toFixed(0)}%** of positions; shrinking the int4 groups from 64 to 8 cuts the weight MSE by **${(mse[1] / mse[4]).toFixed(1)}x** at a cost of ${extraBits.toFixed(2)} extra bits per weight. At 2 bits the grid has only four codes and agreement collapses to **${agree[6].toFixed(0)}%**. With 100 positions one flipped prediction moves agreement by a whole point, so differences of 1–2 points between the int4 rows are noise.`);

  // ---------- 2. memory: Llama 3 weights and KV cache at each precision ----------
  const models = [m.LLAMA3_8B, m.LLAMA3_70B];
  const precisions = [
    { name: 'bf16', opts: { bits: 16 } },
    { name: 'int8', opts: { bits: 8, groupSize: Infinity } },
    { name: 'int4 g128', opts: { bits: 4, groupSize: 128, scaleBits: 16 } },
    { name: 'int4 g32 + zero', opts: { bits: 4, groupSize: 32, scaleBits: 16, zeroBits: 4 } },
  ];
  const memLabels = [], memValues = [];
  for (const md of models) for (const p of precisions) { memLabels.push(`${md.name.replace('Llama-3-', '')} ${p.name}`); memValues.push(+(m.weightBytes(md.params, p.opts) / GB).toFixed(2)); }
  lab.bar({ title: 'Weight memory in GB (one H100 has 80 GB of HBM)', labels: memLabels, values: memValues });
  const kvRows = [];
  for (const md of models) for (const [name, bits] of [['fp16', 16], ['int8 / fp8', 8], ['int4', 4]]) {
    kvRows.push([md.name, name, +(m.kvCacheBytes(md, { contextLen: 1, bits }) / 1024).toFixed(0), +(m.kvCacheBytes(md, { contextLen: 8192, bits }) / GiB).toFixed(2), +(m.kvCacheBytes(md, { contextLen: 8192, batch: 32, bits }) / GiB).toFixed(1)]);
  }
  lab.table({ title: 'KV cache (GQA, 8 KV heads, head dim 128): per token, per 8k-token sequence, and for a batch of 32 such sequences', columns: ['model', 'precision', 'KB / token', 'GiB @ 8k', 'GiB @ 8k x 32'], rows: kvRows });
  const w70int4 = m.weightBytes(m.LLAMA3_70B.params, precisions[2].opts) / GB;
  const w70bf16 = m.weightBytes(m.LLAMA3_70B.params, precisions[0].opts) / GB;
  const kv8b32 = m.kvCacheBytes(m.LLAMA3_8B, { contextLen: 8192, batch: 32, bits: 16 }) / GiB;

  // ---------- 3. the outlier experiment ----------
  // Take a real weight matrix, plant one weight 50x the largest existing one, and measure the damage
  // to all the OTHER weights under each scheme. The threshold concept in one chart.
  const base = ops.transpose(model.w['blocks.0.mlp.fc.weight']); // [256, 64]: 256 output channels, 64 inputs each
  let amax = 0;
  for (let i = 0; i < base.data.length; i++) amax = Math.max(amax, Math.abs(base.data[i]));
  const spiked = ops.clone(base);
  const spikeAt = 17 * 64 + 5; // row 17, column 5
  spiked.data[spikeAt] = 50 * amax;
  const others = (t) => { const out = new Float32Array(t.data.length - 1); let j = 0; for (let i = 0; i < t.data.length; i++) if (i !== spikeAt) out[j++] = t.data[i]; return out; };
  const outlierSchemes = [
    { name: 'int4 per-tensor', run: (t) => ({ shape: t.shape, data: m.dequantizeAbsmax(m.quantizeAbsmax(t, 4)) }) },
    { name: 'int4 per-channel', run: (t) => m.dequantize(m.quantizePerChannel(t, 4)) },
    { name: 'int4 g16', run: (t) => m.dequantize(m.quantizeGroups(t, { bits: 4, groupSize: 16 })) },
    { name: 'int4 g16 + zero', run: (t) => m.dequantize(m.quantizeGroups(t, { bits: 4, groupSize: 16, symmetric: false })) },
    { name: 'int8 per-tensor', run: (t) => ({ shape: t.shape, data: m.dequantizeAbsmax(m.quantizeAbsmax(t, 8)) }) },
  ];
  const outRows = [], ratios = [];
  for (const s of outlierSchemes) {
    const clean = m.errorStats(others(base), others(s.run(base))).mse;
    const hit = m.errorStats(others(spiked), others(s.run(spiked))).mse;
    const zeroFrac = (t) => { const d = s.run(t).data; let z = 0; for (let i = 0; i < d.length; i++) if (d[i] === 0) z++; return (100 * z) / d.length; };
    ratios.push(hit / clean);
    outRows.push([s.name, clean.toExponential(2), hit.toExponential(2), +(hit / clean).toFixed(1), +zeroFrac(spiked).toFixed(1)]);
    await lab.tick();
  }
  lab.table({ title: 'One planted outlier (50x the largest weight) in a [256, 64] matrix: MSE of the other 16,383 weights', columns: ['scheme', 'MSE without outlier', 'MSE with outlier', 'damage (x)', '% of weights rounded to 0'], rows: outRows });
  lab.bar({ title: 'How much one outlier multiplies the error of everyone else (log10 of the MSE ratio)', labels: outlierSchemes.map((s) => s.name), values: ratios.map((r) => +Math.log10(Math.max(r, 1)).toFixed(2)) });

  lab.done(
    `Your quantisers ran the checkpoint through your weight-only kernel: **int8 per-channel** agreed with fp32 on **${agree[0].toFixed(0)}%** of ${positions} next-token predictions, **int4 g32** on **${agree[2].toFixed(0)}%** and **int2 g16** on **${agree[6].toFixed(0)}%**. ` +
    `Shrinking int4 groups from 64 to 8 lowered weight MSE ${(mse[1] / mse[4]).toFixed(1)}x (from ${mse[1].toExponential(2)} to ${mse[4].toExponential(2)}) for ${extraBits.toFixed(2)} more bits per weight. ` +
    `Llama-3-70B needs **${w70bf16.toFixed(1)} GB** in bf16 and **${w70int4.toFixed(1)} GB** in int4 g128 (4.125 bits/param), so it fits one 80 GB H100 only quantised; Llama-3-8B's KV cache for 32 sequences of 8k tokens is **${kv8b32.toFixed(0)} GiB** in fp16, more than its ${(m.weightBytes(m.LLAMA3_8B.params, { bits: 16 }) / GB).toFixed(1)} GB of weights. ` +
    `A single outlier weight multiplied the int4 error of all other weights by **${ratios[0].toFixed(0)}x** with one tensor-wide scale, **${ratios[1].toFixed(1)}x** per-channel and **${ratios[2].toFixed(1)}x** with groups of 16.`,
  );
}
