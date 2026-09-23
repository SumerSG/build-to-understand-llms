import { rng } from 'lib/util.js';
import * as ops from 'lib/ops.js';

// The textbook i-j-p loop order, used only as a comparison for the learner's matmul.
function matmulIJP(a, b) {
  const [n, k] = a.shape, m = b.shape[1];
  const out = new Float32Array(n * m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    let s = 0;
    for (let p = 0; p < k; p++) s += a.data[i * k + p] * b.data[p * m + j];
    out[i * m + j] = s;
  }
  return { shape: [n, m], data: out };
}

function maxAbsDiff(x, y) {
  const a = x.data ?? x, b = y.data ?? y;
  if (a.length !== b.length) return Infinity;
  let d = 0;
  for (let i = 0; i < a.length; i++) { const e = Math.abs(a[i] - b[i]); d = Number.isNaN(e) ? Infinity : Math.max(d, e); }
  return d;
}

// Best-of-`rounds` timing, each round repeating the multiply until at least ~60 ms have elapsed, so a
// single garbage-collection pause or JIT recompile does not decide the comparison.
function gflopsOf(fn, a, b, rounds = 3) {
  const n = a.shape[0];
  let best = Infinity;
  for (let round = 0; round < rounds; round++) {
    let reps = 0;
    const t0 = performance.now();
    do { fn(a, b); reps++; } while (performance.now() - t0 < 60);
    best = Math.min(best, (performance.now() - t0) / reps / 1000);
  }
  return (2 * n * n * n) / Math.max(best, 1e-9) / 1e9;
}

export default async function demo(m, lab) {
  const next = rng(7);
  const rand = (shape) => { const t = m.raw(shape); for (let i = 0; i < t.data.length; i++) t.data[i] = next() * 2 - 1; return t; };

  // 1. correctness: every kernel against the vetted reference in lib/ops.js
  const A = rand([6, 5]), B = rand([5, 7]), X = rand([3, 4, 5]), bias = rand([5]);
  const gamma = rand([5]), beta = rand([5]);
  const logits = rand([4, 9]);
  for (let i = 0; i < logits.data.length; i++) logits.data[i] *= 800;          // large logits test stability
  const checks = [
    ['offset([3,4,5], [2,1,3])', Math.abs(m.offset([3, 4, 5], [2, 1, 3]) - (2 * 20 + 1 * 5 + 3))],
    ['transpose [6,5]', maxAbsDiff(m.transpose(A), ops.transpose(A))],
    ['matmul [6,5]x[5,7]', maxAbsDiff(m.matmul(A, B), ops.matmul(A, B))],
    ['add bias [3,4,5]+[5]', maxAbsDiff(m.add(X, bias), ops.add(X, bias))],
    ['mul scalar', maxAbsDiff(m.mul(X, -2.5), ops.mul(X, -2.5))],
    ['sum last axis', maxAbsDiff(m.sum(X), ops.sum(X, -1))],
    ['argmax last axis', m.argmax(X).every((v, i) => v === ops.argmax(X)[i]) ? 0 : 1],
    ['softmax, logits up to ±800', maxAbsDiff(m.softmax(logits), ops.softmax(logits))],
    ['layerNorm with gamma, beta', maxAbsDiff(m.layerNorm(X, gamma, beta), ops.layerNorm(X, gamma, beta))],
  ];
  const tol = 1e-4;
  const passed = checks.filter(([, e]) => e <= tol).length;
  lab.table({ title: `Your kernels vs lib/ops.js (tolerance ${tol})`, columns: ['op', 'max |yours − reference|', 'match'],
    rows: checks.map(([name, e]) => [name, Number.isFinite(e) ? +e.toExponential(2) : String(e), e <= tol ? 'yes' : 'NO']) });
  await lab.tick();

  // 2. matmul throughput: your loop vs the textbook i-j-p order. At n = 512 each matrix is 1 MB, far
  // beyond the L1 cache (tens of KB), which is where loop order starts to matter in JavaScript.
  const sizes = [];
  const yours = [], naive = [];
  const warm = rand([32, 32]);
  m.matmul(warm, warm); matmulIJP(warm, warm);                  // let the JIT compile both loops
  for (const n of [64, 128, 256, 512]) {
    const a = rand([n, n]), b = rand([n, n]);
    const t0 = performance.now();
    m.matmul(a, b);
    const once = performance.now() - t0;
    if (n > 128 && once > 2500) { lab.log(`skipping n=${n}: one multiply took ${Math.round(once)} ms`); break; }
    const rounds = n >= 512 ? 2 : 3;
    sizes.push(n);
    yours.push(gflopsOf(m.matmul, a, b, rounds));
    naive.push(gflopsOf(matmulIJP, a, b, rounds));
    lab.progress(sizes.length / 4, `matmul ${n}x${n}`);
    await lab.tick();
  }
  lab.plot({ title: 'Matmul throughput: your loop vs the i-j-p order', x: sizes,
    series: [{ name: 'your matmul', values: yours }, { name: 'i-j-p reference loop', values: naive }],
    xlabel: 'n (n x n matrices)', ylabel: 'GFLOP/s' });

  // 3. softmax temperature
  const row = [2.0, 1.0, 0.5, 0.1, -1.0, -2.0];
  const temps = [0.25, 0.5, 1, 2, 4];
  const rows = temps.map((T) => Array.from(m.softmax(m.fromArray([row.map((l) => l / T)])).data));
  lab.heatmap({ title: 'softmax(logits / T): low T sharpens, high T flattens', rows, rowLabels: temps.map((t) => `T=${t}`), colLabels: row.map(String), min: 0, max: 1 });

  // 4. layernorm
  const x = rand([4, 8]);
  for (let i = 0; i < x.data.length; i++) x.data[i] = x.data[i] * 10 + 3;
  const y = m.layerNorm(x);
  const stats = (t) => m.toArray(t).map((r) => { const mu = r.reduce((s, v) => s + v, 0) / r.length; const v = r.reduce((s, q) => s + (q - mu) ** 2, 0) / r.length; return [mu, Math.sqrt(v)]; });
  const sx = stats(x), sy = stats(y);
  lab.table({ title: 'LayerNorm: per-row mean and std before / after', columns: ['row', 'mean before', 'std before', 'mean after', 'std after'], rows: sx.map((s, i) => [i, +s[0].toFixed(3), +s[1].toFixed(3), +sy[i][0].toFixed(4), +sy[i][1].toFixed(4)]) });
  const worstMean = Math.max(...sy.map((s) => Math.abs(s[0])));
  const worstStd = Math.max(...sy.map((s) => Math.abs(s[1] - 1)));

  const peak = Math.max(...yours);
  const nPeak = sizes[yours.indexOf(peak)];
  const r0 = yours[0] / naive[0], r1 = yours[yours.length - 1] / naive[naive.length - 1];
  const nLast = sizes[sizes.length - 1];
  lab.done(`**${passed} of ${checks.length}** kernels match \`lib/ops.js\` within ${tol}. Your matmul peaks at **${peak.toFixed(2)} GFLOP/s** at n=${nPeak}. Against the i-j-p loop it runs at **${r0.toFixed(2)}×** the speed at n=${sizes[0]} and **${r1.toFixed(2)}×** at n=${nLast}. An H100 (approximately 989 TFLOP/s dense bf16, NVIDIA datasheet) is about **${(989e3 / peak).toExponential(1)}×** faster. Softmax at T=0.25 puts ${(100 * rows[0][0]).toFixed(0)}% of the mass on the top logit versus ${(100 * rows[4][0]).toFixed(0)}% at T=4. After LayerNorm the worst row mean is ${worstMean.toExponential(1)} and the worst |std − 1| is ${worstStd.toExponential(1)} (eps = 1e-5 makes the std slightly below 1).`);
}
