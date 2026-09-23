// Goal demo for "Autograd from scratch": your autograd engine fits y = 3x + 2 by gradient descent, then trains a small
// MLP with the fused cross-entropy after gradCheck has vouched for every closure it uses.
import { rng, randn } from 'lib/util.js';

export default async function demo(m, lab) {
  const next = rng(2);

  // ---------- 1. noisy data from y = 3x + 2 ----------
  const N = 64;
  const xs = Array.from({ length: N }, () => next() * 4 - 2);
  const ys = xs.map((x) => 3 * x + 2 + 0.3 * randn(next));
  lab.log(`${N} points from y = 3x + 2 with Gaussian noise (std 0.3), x in [-2, 2)`);

  // ---------- 2. the record: build one loss and show the graph backward() will walk ----------
  const X = m.Tensor.from(xs.map((x) => [x]));
  const Y = m.Tensor.from(ys.map((y) => [y]));
  const w0 = m.Tensor.zeros([1, 1], { requiresGrad: true });
  const b0 = m.Tensor.zeros([1], { requiresGrad: true });
  const diff0 = X.matmul(w0).add(b0).sub(Y);
  const loss0 = diff0.mul(diff0).mean();
  const order = m.topoSort(loss0);
  const label = (t) => (t._op ? t._op : t.requiresGrad ? 'leaf (parameter)' : 'leaf (data)');
  lab.table({
    title: 'The recorded graph of one MSE loss, in topological order (backward() runs it from the bottom up)',
    columns: ['#', 'node', 'shape', 'inputs (#)'],
    rows: order.map((t, i) => [i, label(t), `[${t.shape.join(', ')}]`, t._children.map((c) => order.indexOf(c)).join(', ') || '—']),
  });
  loss0.backward();
  lab.log(`At w = 0, b = 0: loss ${loss0.item().toFixed(3)}, dL/dw = ${w0.grad[0].toFixed(3)}, dL/db = ${b0.grad[0].toFixed(3)} (both negative: increase w and b)`);
  await lab.tick();

  // ---------- 3. linear regression by SGD ----------
  const steps = 200, lr = 0.1;
  const r = m.trainLinear(xs, ys, { steps, lr });
  lab.plot({ title: 'Linear regression: mean squared error per step', series: [{ name: 'MSE', values: r.losses }], xlabel: 'step', ylabel: 'loss', yscale: 'log' });
  lab.plot({
    title: 'w and b during training (true values 3 and 2)',
    series: [
      { name: 'w', values: r.ws },
      { name: 'b', values: r.bs },
      { name: 'w = 3', values: new Array(steps).fill(3) },
      { name: 'b = 2', values: new Array(steps).fill(2) },
    ],
    xlabel: 'step', ylabel: 'value',
  });
  lab.log(`after ${steps} steps: w = ${r.w.toFixed(4)}, b = ${r.b.toFixed(4)}, loss ${r.losses[0].toFixed(3)} → ${r.losses[steps - 1].toFixed(4)}`);
  await lab.tick();

  // ---------- 4. a two-layer MLP on a ring-vs-disc problem, checked first, then trained ----------
  const M = 96, H = 16;
  const pts = [], cls = [];
  for (let i = 0; i < M; i++) {
    const c = i % 2;
    const radius = c === 0 ? 0.3 + 0.5 * next() : 1.3 + 0.6 * next();
    const angle = 2 * Math.PI * next();
    pts.push([radius * Math.cos(angle), radius * Math.sin(angle)]);
    cls.push(c);
  }
  const P = m.Tensor.from(pts);                                     // [M, 2]
  const W1 = m.Tensor.randn([2, H], next, 0.8, { requiresGrad: true });
  const b1 = m.Tensor.zeros([H], { requiresGrad: true });
  const W2 = m.Tensor.randn([H, 2], next, 0.5, { requiresGrad: true });
  const b2 = m.Tensor.zeros([2], { requiresGrad: true });
  const params = [W1, b1, W2, b2];
  const names = ['W1 [2,16]', 'b1 [16]', 'W2 [16,2]', 'b2 [2]'];
  const forward = (inp, W1, b1, W2, b2) => inp.matmul(W1).add(b1).relu().matmul(W2).add(b2);
  const mlpLoss = (W1, b1, W2, b2) => m.crossEntropy(forward(P, W1, b1, W2, b2), cls);

  const check = m.gradCheck(mlpLoss, params, { eps: 1e-3, tol: 1e-2 });
  const perParam = names.map((name, k) => {
    const d = check.details.filter((x) => x.input === k);
    const worst = d.reduce((a, x) => (x.relErr > a.relErr ? x : a), d[0]);
    return [name, d.length, +worst.analytic.toFixed(5), +worst.numeric.toFixed(5), worst.relErr.toExponential(2)];
  });
  lab.table({
    title: `gradCheck on the MLP (matmul, add, relu, crossEntropy): ${check.ok ? 'PASS' : 'FAIL'}, max relative error ${check.maxRelErr.toExponential(2)}`,
    columns: ['parameter', 'elements checked', 'worst analytic', 'worst numeric', 'relErr'],
    rows: perParam,
  });
  lab.check(check.ok, `gradCheck failed with maxRelErr ${check.maxRelErr}: one of your backward closures disagrees with central differences`);
  await lab.tick();

  const accuracy = () => {
    const logits = forward(P, W1, b1, W2, b2).data;
    let correct = 0;
    for (let i = 0; i < M; i++) correct += (logits[2 * i + 1] > logits[2 * i] ? 1 : 0) === cls[i] ? 1 : 0;
    return correct / M;
  };
  const accBefore = accuracy();
  const mlpSteps = 300, mlpLr = 0.5;
  const mlpLosses = [];
  for (let step = 0; step < mlpSteps; step++) {
    const loss = mlpLoss(W1, b1, W2, b2);
    mlpLosses.push(loss.item());
    for (const p of params) p.zeroGrad();
    loss.backward();
    m.sgdStep(params, mlpLr);
    if (step % 25 === 0) { lab.progress(step / mlpSteps, `MLP step ${step}, loss ${loss.item().toFixed(3)}`); await lab.tick(); }
  }
  const accAfter = accuracy();
  lab.plot({ title: 'MLP: cross-entropy per step', series: [{ name: 'cross-entropy', values: mlpLosses }], xlabel: 'step', ylabel: 'loss' });

  // Decision map: P(class 1) on a grid, so the learned boundary is visible.
  const G = 21, lim = 2;
  const axis = Array.from({ length: G }, (_, i) => -lim + (2 * lim * i) / (G - 1));
  const grid = [];
  for (const y of axis) for (const x of axis) grid.push([x, y]);
  const gl = forward(m.Tensor.from(grid), W1, b1, W2, b2).data;
  const rows = [];
  for (let gy = G - 1; gy >= 0; gy--) {                            // top row first, so y increases upwards
    const row = [];
    for (let gx = 0; gx < G; gx++) {
      const i = gy * G + gx;
      const z = gl[2 * i + 1] - gl[2 * i];
      row.push(1 / (1 + Math.exp(-z)));
    }
    rows.push(row);
  }
  // Label every 5th column only (x = -2, -1, 0, 1, 2): 21 labels side by side would overlap.
  const colLabels = axis.map((v, i) => (i % 5 === 0 ? String(Math.round(v)) : ''));
  const rowLabels = axis.map((v) => `y = ${v.toFixed(1)}`).reverse();
  lab.log('The MLP (a multilayer perceptron: two layers of weights with a ReLU, max(x, 0), between them) sees only the two coordinates (x, y) of a point and must say whether it belongs to the disc (distance from the centre below 0.8) or to the ring around it (distance 1.3 to 1.9). The map below shows its answer at every point of the plane.');
  lab.heatmap({ title: 'MLP decision map: probability the point is "ring" (dark = ring, light = disc; x from −2 to 2 across, y from −2 to 2 upwards)', rows, rowLabels, colLabels, min: 0, max: 1 });
  lab.progress(1, 'done');

  lab.done(`Your engine fitted **w = ${r.w.toFixed(3)}, b = ${r.b.toFixed(3)}** (true 3 and 2) in ${steps} SGD steps, taking the MSE from ${r.losses[0].toFixed(3)} to **${r.losses[steps - 1].toFixed(4)}**. ` +
    `gradCheck compared all ${check.details.length} analytic gradients of the ${2 * H + H + H * 2 + 2}-parameter MLP against central differences: max relative error **${check.maxRelErr.toExponential(2)}** (tolerance 1e-2). ` +
    `Trained with the fused cross-entropy for ${mlpSteps} steps, that MLP went from ${(100 * accBefore).toFixed(0)}% to **${(100 * accAfter).toFixed(0)}%** accuracy on the ring-vs-disc data, loss ${mlpLosses[0].toFixed(3)} → ${mlpLosses[mlpSteps - 1].toFixed(3)}: from nothing but gradients, it learned to tell a point in the central disc from a point on the ring around it, which a single straight line could never separate.`);
}
