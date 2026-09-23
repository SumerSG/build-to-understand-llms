// Module 29 demo — YOUR RoPE, GQA, MLA, sliding-window cache and budget, exercised three ways:
//   1. RoPE scores vs absolute position and offset, on a real query and key from the checkpoint's layer 0;
//   2. equivalence checks at the checkpoint's dims (random weights: nothing here is trained);
//   3. the KV-cache budget at Llama-3-8B dims, and the context-extension trade at Llama-like dims.
// If the checkpoint cannot be loaded the demo falls back to a random-weight GPT of the same shape.
import { loadModel } from 'lib/infer.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { GPT } from 'lib/gpt.js';
import { toyCorpus } from 'lib/data.js';
import * as ops from 'lib/ops.js';
import { rng, randn } from 'lib/util.js';

const PROMPT = 'The cat sat on the mat.';
const LLAMA3_8B = { nLayer: 32, nHead: 32, nKVHead: 8, headDim: 128, dLatent: 512, dRope: 64, window: 4096, bytesPerElement: 2 };

async function loadCheckpoint(lab) {
  try {
    const json = (await import('lib/checkpoints/tiny-gpt.json', { with: { type: 'json' } })).default;
    const tok = (await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default;
    if (!json || !json.params || !json.config) throw new Error('checkpoint has no params');
    return { model: loadModel(json), tokenizer: BPETokenizer.fromJSON(tok), trained: true };
  } catch (e) {
    lab.log(`Checkpoint not available (${String(e.message).slice(0, 100)}); using a random-weight model of the same shape.`);
    const tokenizer = BPETokenizer.train(toyCorpus(200, 1), { vocabSize: 256 });
    const gpt = new GPT({ vocabSize: tokenizer.vocabSize, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 64, seed: 3 });
    return { model: loadModel(gpt.toJSON()), tokenizer, trained: false };
  }
}

/** Layer-0 query and key of head 0 for two tokens of the prompt, as the checkpoint computes them. */
function layer0QK(model, ids, qi, ki) {
  const { nEmbd, nHead } = model.config;
  const dh = nEmbd / nHead;
  const w = model.w;
  const x = ops.add(ops.embed(w['wte.weight'], ids), ops.embed(w['wpe.weight'], ids.map((_, i) => i)));
  const normed = ops.layerNorm(x, w['blocks.0.ln1.gamma'], w['blocks.0.ln1.beta']);
  const qkv = ops.add(ops.matmul(normed, w['blocks.0.attn.qkv.weight']), w['blocks.0.attn.qkv.bias']);
  const width = 3 * nEmbd;
  const q = qkv.data.slice(qi * width, qi * width + dh);
  const k = qkv.data.slice(ki * width + nEmbd, ki * width + nEmbd + dh);
  return { q, k, dh };
}

const maxAbsDiff = (a, b) => { let d = 0; for (let i = 0; i < a.data.length; i++) d = Math.max(d, Math.abs(a.data[i] - b.data[i])); return d; };
const fmtBytes = (b) => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(2)} GiB` : b >= 2 ** 20 ? `${(b / 2 ** 20).toFixed(1)} MiB` : `${(b / 1024).toFixed(1)} KiB`);

export default async function demo(m, lab) {
  const { model, tokenizer, trained } = await loadCheckpoint(lab);
  const cfg = model.config;
  const H = cfg.nHead, C = cfg.nEmbd, dh = C / H, Tmax = cfg.blockSize;
  lab.log(`Checkpoint dims: ${cfg.nLayer} layers, ${H} heads of ${dh} channels, width ${C}, context ${Tmax}${trained ? '' : ' (random weights)'}.`);

  // ---------- 1. RoPE: the score depends only on the offset ----------
  const ids = tokenizer.encode(PROMPT).slice(0, Tmax);
  const { q, k } = layer0QK(model, ids, ids.length - 1, Math.max(0, ids.length - 3));
  const qT = { shape: [1, dh], data: q }, kT = { shape: [1, dh], data: k };
  const offsets = Array.from({ length: 16 }, (_, d) => d);
  const queryPositions = [15, 23, 31, 39, 47, 55, 63];
  const rows = [];
  for (const mp of queryPositions) {
    const rq = m.applyRope(qT, [mp]).data;
    rows.push(offsets.map((d) => { const rk = m.applyRope(kT, [mp - d]).data; let s = 0; for (let c = 0; c < dh; c++) s += rq[c] * rk[c]; return s / Math.sqrt(dh); }));
  }
  let rowSpread = 0;
  for (let j = 0; j < offsets.length; j++) for (const r of rows) rowSpread = Math.max(rowSpread, Math.abs(r[j] - rows[0][j]));
  const colRange = Math.max(...rows[0]) - Math.min(...rows[0]);
  lab.heatmap({ title: 'Your RoPE: layer-0 q·k / sqrt(dh) by query position (rows) and offset m − n (columns)', rows, rowLabels: queryPositions.map((p) => `m=${p}`), colLabels: offsets.map(String) });
  lab.log(`Across ${queryPositions.length} absolute positions, the score at each offset varies by at most ${rowSpread.toExponential(1)}; across offsets it spans ${colRange.toFixed(3)}. Every row is the same row.`);
  await lab.tick();

  // ---------- 2. equivalence at the checkpoint's dims, random weights ----------
  const next = rng(29);
  const T = Math.min(24, Tmax);
  const rq = ops.randn([H, T, dh], next), rk = ops.randn([H, T, dh], next), rv = ops.randn([H, T, dh], next);
  const gqaVsMha = maxAbsDiff(m.gqaAttention(rq, rk, rv), m.mha(rq, rk, rv));
  const k1 = ops.randn([1, T, dh], next), v1 = ops.randn([1, T, dh], next);
  const rep = (t) => ({ shape: [H, T, dh], data: Float32Array.from({ length: H * T * dh }, (_, i) => t.data[i % (T * dh)]) });
  const mqaVsMha = maxAbsDiff(m.gqaAttention(rq, k1, v1), m.mha(rq, rep(k1), rep(v1)));
  await lab.tick();

  const dLatent = C / 4;
  const w = m.initMLA({ nEmbd: C, nHead: H, headDim: dh, dLatent, next });
  const x = ops.randn([T, C], next);
  const full = m.mlaAttention(x, w, { nHead: H });
  const cache = m.newLatentCache(dLatent);
  let mlaDiff = 0;
  for (let t = 0; t < T; t++) {
    const out = m.mlaDecodeStep(ops.slice(x, 0, t, t + 1), cache, w, { nHead: H });
    for (let h = 0; h < H; h++) for (let c = 0; c < dh; c++) mlaDiff = Math.max(mlaDiff, Math.abs(out.data[h * dh + c] - full.data[(h * T + t) * dh + c]));
    if (t % 8 === 0) await lab.tick();
  }

  const W = 8;
  const swa = m.slidingWindowAttention(rq, rk, rv, W);
  const ring = new m.RingKVCache({ nHead: H, headDim: dh, window: W });
  let ringDiff = 0;
  const ringBytes = [];
  for (let t = 0; t < T; t++) {
    ring.append(ops.slice(rk, 1, t, t + 1), ops.slice(rv, 1, t, t + 1));
    const out = m.mha(ops.slice(rq, 1, t, t + 1), ring.keys(), ring.values());
    for (let h = 0; h < H; h++) for (let c = 0; c < dh; c++) ringDiff = Math.max(ringDiff, Math.abs(out.data[h * dh + c] - swa.data[(h * T + t) * dh + c]));
    ringBytes.push(ring.bytes(4));
  }
  const growingBytes = 2 * H * T * dh * 4;
  lab.table({
    title: `Equivalence checks at the checkpoint's dims (H=${H}, dh=${dh}, T=${T}; random weights)`,
    columns: ['check', 'max |difference|', 'cache per token (floats, one layer)'],
    rows: [
      ['GQA with H_kv = H  vs  MHA', gqaVsMha.toExponential(1), 2 * H * dh],
      ['GQA with H_kv = 1  vs  MHA on copied heads (MQA)', mqaVsMha.toExponential(1), 2 * dh],
      [`MLA decode from latent cache (d_c = ${dLatent})  vs  full pass`, mlaDiff.toExponential(1), dLatent],
      [`ring cache (W = ${W})  vs  sliding-window attention`, ringDiff.toExponential(1), `${2 * H * dh}, at most ${W} tokens`],
    ],
  });
  lab.check(gqaVsMha < 1e-4 && mqaVsMha < 1e-4 && mlaDiff < 1e-3 && ringDiff < 1e-4, 'every variant must match its reference');
  lab.log(`Ring buffer after ${T} tokens: ${ringBytes.at(-1)} bytes (it was ${ringBytes[0]} after the first); a growing cache would hold ${growingBytes}.`);
  await lab.tick();

  // ---------- 3a. the budget at Llama-3-8B dims ----------
  const CTX = 32768;
  const budget = m.cacheBudget(LLAMA3_8B, CTX);
  lab.bar({ title: `Your cacheBudget: KV cache for one ${CTX.toLocaleString('en-US')}-token sequence at Llama-3-8B dims (GiB, bf16)`, labels: budget.map((r) => r.variant), values: budget.map((r) => +(r.totalBytes / 2 ** 30).toFixed(3)) });
  lab.table({
    title: 'Per-token and per-sequence cache (32 layers, 32 query heads, dh 128; GQA/SWA 8 KV heads; MLA d_c 512 + 64; HYBRID 1 GQA layer in 4)',
    columns: ['variant', 'bytes per token', 'tokens held', 'constant state', 'total'],
    rows: budget.map((r) => [r.variant, r.bytesPerToken.toLocaleString('en-US'), r.tokensHeld.toLocaleString('en-US'), (r.fixedBytes ? fmtBytes(r.fixedBytes) : '—'), fmtBytes(r.totalBytes)]),
  });
  const by = Object.fromEntries(budget.map((r) => [r.variant, r.totalBytes]));
  await lab.tick();

  // ---------- 3b. stretching RoPE 4x at Llama-2-like dims ----------
  const DH = 128, TRAIN = 4096, FACTOR = 4;
  const base = m.ropeFrequencies(DH);
  const methods = ['none', 'pi', 'ntk', 'yarn'];
  const freqs = Object.fromEntries(methods.map((meth) => [meth, m.scaledRopeFrequencies(DH, { method: meth, factor: FACTOR, trainLen: TRAIN })]));
  const lengths = [];
  for (let L = 1024; L <= TRAIN * FACTOR; L += 1024) lengths.push(L);
  lab.plot({
    title: `Your unseenRotation: RoPE angles never seen in ${TRAIN}-token training, stretched to ${FACTOR * TRAIN}`,
    x: lengths,
    series: methods.map((meth) => ({ name: meth, values: lengths.map((L) => m.unseenRotation(freqs[meth], base, TRAIN, L)) })),
    xlabel: 'context length (tokens)', ylabel: 'mean fraction of a turn out of range',
  });
  const dnext = rng(4);
  const drift = Object.fromEntries(methods.map((meth) => [meth, 0]));
  const SAMPLES = 20, NEAR = 32;
  for (let s = 0; s < SAMPLES; s++) {
    const qv = Float32Array.from({ length: DH }, () => randn(dnext));
    const kv = Float32Array.from({ length: DH }, () => randn(dnext));
    for (const meth of methods) drift[meth] += m.scoreDrift(qv, kv, freqs[meth], base, NEAR) / SAMPLES;
    if (s % 5 === 0) { lab.progress(s / SAMPLES, 'score drift'); await lab.tick(); }
  }
  const unseen = Object.fromEntries(methods.map((meth) => [meth, m.unseenRotation(freqs[meth], base, TRAIN, TRAIN * FACTOR)]));
  // Drift against the unscaled scores as the offset range widens: the price each method pays in-distribution.
  const ranges = [4, 8, 16, 32, 64, 128, 256];
  const dq = Float32Array.from({ length: DH }, () => randn(dnext)), dk = Float32Array.from({ length: DH }, () => randn(dnext));
  lab.plot({
    title: 'Your scoreDrift: change in q·k (× |q||k|) versus the unscaled model, over offsets 0 … N − 1',
    x: ranges,
    series: methods.map((meth) => ({ name: meth, values: ranges.map((n) => m.scoreDrift(dq, dk, freqs[meth], base, n)) })),
    xlabel: 'N (offsets measured)', ylabel: 'mean |score drift|',
  });
  await lab.tick();
  lab.table({
    title: `The context-extension trade at dh ${DH}, ${TRAIN} → ${TRAIN * FACTOR} tokens`,
    columns: ['method', `unseen rotation at ${TRAIN * FACTOR}`, `score drift at offsets 0–${NEAR - 1} (× |q||k|)`],
    rows: methods.map((meth) => [meth, unseen[meth].toFixed(4), drift[meth].toFixed(4)]),
  });

  lab.done(
    `Your RoPE scores depend only on offset (spread across absolute positions **${rowSpread.toExponential(1)}**), and GQA, MQA, MLA and the ring cache each matched their reference (worst difference **${Math.max(gqaVsMha, mqaVsMha, mlaDiff, ringDiff).toExponential(1)}**). ` +
    `At Llama-3-8B dims and ${CTX.toLocaleString('en-US')} tokens: MHA **${fmtBytes(by.MHA)}**, GQA **${fmtBytes(by.GQA)}** (${(by.MHA / by.GQA).toFixed(1)}× smaller), MLA **${fmtBytes(by.MLA)}** (${(by.GQA / by.MLA).toFixed(2)}× smaller than GQA), MQA **${fmtBytes(by.MQA)}**, the 4096-token sliding window **${fmtBytes(by.SWA)}** (${(by.GQA / by.SWA).toFixed(0)}× smaller than GQA, and constant from here on), and a 1-in-4 hybrid **${fmtBytes(by.HYBRID)}** (${(by.GQA / by.HYBRID).toFixed(1)}× smaller than GQA, growing at a quarter of its rate). ` +
    `Stretching RoPE 4×: unscaled leaves **${unseen.none.toFixed(3)}** of a turn out of range with no near-offset drift; PI brings it to **${unseen.pi.toFixed(3)}** at a short-range score drift of **${drift.pi.toFixed(4)}**, NTK **${unseen.ntk.toFixed(3)}** / **${drift.ntk.toFixed(4)}**, YaRN **${unseen.yarn.toFixed(3)}** / **${drift.yarn.toFixed(4)}**.`,
  );
}
