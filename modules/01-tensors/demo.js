import { rng } from 'lib/util.js';

export default async function demo(m, lab) {
  const next = rng(7);
  const rand = (shape) => { const t = m.raw(shape); for (let i = 0; i < t.data.length; i++) t.data[i] = next() * 2 - 1; return t; };
  // 1. matmul throughput
  const sizes = [16, 32, 64, 128, 192];
  const gflops = [];
  for (const n of sizes) {
    const a = rand([n, n]), b = rand([n, n]);
    const reps = n <= 64 ? 20 : 3;
    const t0 = performance.now();
    for (let r = 0; r < reps; r++) m.matmul(a, b);
    const dt = (performance.now() - t0) / reps / 1000;
    gflops.push((2 * n * n * n) / dt / 1e9);
    lab.progress((sizes.indexOf(n) + 1) / sizes.length, `matmul ${n}x${n}`);
    await lab.tick();
  }
  lab.plot({ title: 'Your matmul: GFLOP/s vs matrix size', x: sizes, series: [{ name: 'GFLOP/s', values: gflops }], xlabel: 'n (n x n matrices)', ylabel: 'GFLOP/s' });
  // 2. softmax temperature
  const logits = [2.0, 1.0, 0.5, 0.1, -1.0, -2.0];
  const temps = [0.25, 0.5, 1, 2, 4];
  const rows = temps.map((T) => Array.from(m.softmax(m.fromArray([logits.map((l) => l / T)])).data));
  lab.heatmap({ title: 'softmax(logits / T): temperature sharpens or flattens', rows, rowLabels: temps.map((t) => `T=${t}`), colLabels: logits.map(String), min: 0, max: 1 });
  // 3. layernorm
  const x = rand([4, 8]);
  for (let i = 0; i < x.data.length; i++) x.data[i] = x.data[i] * 10 + 3;
  const y = m.layerNorm(x);
  const stats = (t) => m.toArray(t).map((r) => { const mu = r.reduce((s, v) => s + v, 0) / r.length; const v = r.reduce((s, q) => s + (q - mu) ** 2, 0) / r.length; return [mu, Math.sqrt(v)]; });
  const sx = stats(x), sy = stats(y);
  lab.table({ title: 'LayerNorm: per-row mean and std before / after', columns: ['row', 'mean before', 'std before', 'mean after', 'std after'], rows: sx.map((s, i) => [i, +s[0].toFixed(3), +s[1].toFixed(3), +sy[i][0].toFixed(4), +sy[i][1].toFixed(4)]) });
  const peak = Math.max(...gflops);
  lab.done(`Your kernels run: matmul peaks at **${peak.toFixed(2)} GFLOP/s** at n=${sizes[gflops.indexOf(peak)]} (an H100 does roughly 1,000,000 GFLOP/s in bf16, so about **${(1e6 / peak).toExponential(1)}×** faster). Softmax at T=0.25 puts ${(100 * rows[0][0]).toFixed(0)}% of the mass on the top logit versus ${(100 * rows[4][0]).toFixed(0)}% at T=4, and LayerNorm brought every row to mean ≈ 0 and std ≈ 1.`);
}
