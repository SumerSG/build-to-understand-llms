import { rng, softmaxArray } from 'lib/util.js';

const LAB = { vocabSize: 256, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 64 };

/** Mean L2 norm of the C-vectors in a [B,T,C] tensor: how "loud" the residual stream is. */
function meanNorm(x) {
  const C = x.shape[x.shape.length - 1];
  const rows = x.data.length / C;
  let total = 0;
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let j = 0; j < C; j++) s += x.data[r * C + j] ** 2;
    total += Math.sqrt(s);
  }
  return total / rows;
}

const pct = (part, whole) => `${((100 * part) / whole).toFixed(1)}%`;
const fmtInt = (n) => n.toLocaleString('en-US');

export default async function demo(m, lab) {
  // 1. Build the lab model and check the closed form against the built object.
  lab.log('Building GPT(vocab 256, block 64, 2 layers, 4 heads, 64 dims) with your layers…');
  const model = new m.GPT({ ...LAB, seed: 42 });
  const built = model.numParams();
  const breakdown = m.paramBreakdown(LAB);
  lab.check(m.countParams(LAB) === built, `countParams(LAB) = ${m.countParams(LAB)} but the built model has ${built} parameters`);
  const components = [
    ['token embedding (wte)', breakdown.tokenEmbedding],
    ['position embedding (wpe)', breakdown.positionEmbedding],
    ['attention (qkv + proj)', breakdown.attention],
    ['MLP (fc + proj)', breakdown.mlp],
    ['LayerNorm gains/shifts', breakdown.layerNorm],
  ];
  lab.table({
    title: `Where the ${fmtInt(built)} parameters of the lab model live`,
    columns: ['component', 'parameters', 'share'],
    rows: [...components.map(([name, n]) => [name, fmtInt(n), pct(n, built)]), ['total (= numParams())', fmtInt(built), '100%']],
  });
  lab.bar({ title: 'Parameters by component (lab model)', labels: components.map((c) => c[0]), values: components.map((c) => c[1]) });
  await lab.tick();

  // 2. The same formula on the real GPT-2 family: the published counts fall out exactly.
  const family = [
    ['lab model', LAB],
    ['GPT-2 small', { vocabSize: 50257, blockSize: 1024, nLayer: 12, nHead: 12, nEmbd: 768 }],
    ['GPT-2 medium', { vocabSize: 50257, blockSize: 1024, nLayer: 24, nHead: 16, nEmbd: 1024 }],
    ['GPT-2 large', { vocabSize: 50257, blockSize: 1024, nLayer: 36, nHead: 20, nEmbd: 1280 }],
    ['GPT-2 XL', { vocabSize: 50257, blockSize: 1024, nLayer: 48, nHead: 25, nEmbd: 1600 }],
  ];
  const familyRows = family.map(([name, cfg]) => {
    const b = m.paramBreakdown(cfg);
    return [name, `L=${cfg.nLayer}, C=${cfg.nEmbd}`, fmtInt(b.total), pct(b.tokenEmbedding, b.total), pct(b.attention, b.total), pct(b.mlp, b.total), fmtInt(12 * cfg.nLayer * cfg.nEmbd ** 2)];
  });
  lab.table({ title: 'Your countParams on real configs (GPT-2 small is published as 124M, XL as 1.5B)', columns: ['model', 'shape', 'countParams', 'token table', 'attention', 'MLP', '12·L·C² alone'], rows: familyRows });
  const gpt2 = m.countParams(family[1][1]);

  // 3. Forward pass on random tokens, watching the residual stream grow block by block.
  const next = rng(7);
  const T = LAB.blockSize;
  const ids = [Array.from({ length: T }, () => Math.floor(next() * LAB.vocabSize))];
  lab.log(`Forward pass on ${T} random tokens: ${ids[0].slice(0, 12).join(' ')} …`);
  const norms = []; // norms[i] = mean ‖x‖ after i blocks; norms[0] is the embedding itself
  let x = m.embedInputs(model.wte, model.wpe, ids);
  norms.push(meanNorm(x));
  for (let i = 0; i < model.blocks.length; i++) {
    x = model.blocks[i].forward(x);
    norms.push(meanNorm(x));
    lab.progress((i + 1) / (model.blocks.length + 1), `after block ${i + 1}`);
    await lab.tick();
  }
  const normAfterLnF = meanNorm(model.lnF.forward(x));
  lab.plot({
    title: 'Mean L2 norm of the residual stream after each block (0 = the embeddings; every block only adds to it)',
    x: norms.map((_, i) => i),
    series: [{ name: 'mean ‖x‖ per position', values: norms.map((v) => +v.toFixed(4)) }],
    xlabel: 'blocks applied', ylabel: 'mean ‖x‖',
  });

  // 4. The logits at the last position: what an untrained model "predicts".
  const logits = model.forward(ids);
  lab.check(logits.shape.length === 3 && logits.shape[0] === 1 && logits.shape[1] === T && logits.shape[2] === LAB.vocabSize, `logits must be [1,${T},${LAB.vocabSize}], got [${logits.shape}]`);
  const V = LAB.vocabSize;
  const lastRow = Array.from(logits.data.subarray((T - 1) * V, T * V));
  const probs = softmaxArray(lastRow);
  const top = probs.map((p, id) => [id, p]).sort((a, b) => b[1] - a[1]).slice(0, 10);
  lab.bar({ title: `Top-10 next-token probabilities at position ${T - 1} (untrained: nearly uniform, 1/V = ${(1 / V).toFixed(4)})`, labels: top.map(([id]) => `tok ${id}`), values: top.map(([, p]) => +p.toFixed(5)) });
  let entropy = 0;
  for (const p of probs) if (p > 0) entropy -= p * Math.log(p);

  // 5. FLOPs: the formula versus the wall clock of your forward pass.
  const flops = m.flopsPerToken(LAB, T);
  let forwardFlops = 0;
  for (let t = 1; t <= T; t++) forwardFlops += m.flopsPerToken(LAB, t); // position t sees t tokens of context
  let best = Infinity;
  for (let rep = 0; rep < 3; rep++) {
    const t0 = performance.now();
    model.forward(ids);
    best = Math.min(best, performance.now() - t0);
    lab.progress(0.8 + 0.2 * ((rep + 1) / 3), `timing forward ${rep + 1}/3`);
    await lab.tick();
  }
  const gflops = forwardFlops / (best / 1000) / 1e9;
  lab.table({
    title: 'FLOPs per token = 2·params + 4·L·T·C',
    columns: ['context T', '2·params', 'attention term', 'FLOPs/token'],
    rows: [1, 16, 64].map((t) => [t, fmtInt(2 * built), fmtInt(4 * LAB.nLayer * t * LAB.nEmbd), fmtInt(m.flopsPerToken(LAB, t))]),
  });

  lab.done(
    `Your GPT has **${fmtInt(built)}** parameters and \`countParams\` agrees exactly; ` +
    `**${pct(breakdown.mlp, built)}** of them are MLP parameters and **${pct(breakdown.attention, built)}** attention (46% and 23% in GPT-2 small, where the 50k-token table takes a bigger share), ` +
    `and the same formula gives **${fmtInt(gpt2)}** for GPT-2 small (published: 124M). ` +
    `On ${T} random tokens the residual stream norm grew from ${norms[0].toFixed(3)} after the embeddings to ${norms[norms.length - 1].toFixed(3)} after block ${LAB.nLayer} ` +
    `(lnF then rescales every position to ${normAfterLnF.toFixed(3)} ≈ sqrt(C) = ${Math.sqrt(LAB.nEmbd).toFixed(1)}), and the untrained head puts ${(100 * top[0][1]).toFixed(2)}% on token ${top[0][0]} with an entropy of ${entropy.toFixed(3)} nats (uniform would be ${Math.log(V).toFixed(3)}). ` +
    `A token with ${T} tokens of context costs **${fmtInt(flops)}** FLOPs (${fmtInt(2 * built)} from the weights + ${fmtInt(flops - 2 * built)} from attention); ` +
    `the whole ${T}-token forward is ${fmtInt(forwardFlops)} FLOPs and ran in ${best.toFixed(1)} ms, about ${gflops.toFixed(2)} GFLOP/s.`,
  );
}
