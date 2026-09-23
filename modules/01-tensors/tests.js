import { rng } from 'lib/util.js';

function randomTensor(m, shape, seed) {
  const next = rng(seed);
  const t = m.raw(shape);
  for (let i = 0; i < t.data.length; i++) t.data[i] = next() * 2 - 1;
  return t;
}

export const tests = [
  { step: 'indexing', name: 'offset follows row-major order', run(m, T) {
    T.eq(m.offset([2, 3], [0, 0]), 0);
    T.eq(m.offset([2, 3], [1, 2]), 5, 'row 1 starts at 3, column 2 adds 2');
    T.eq(m.offset([2, 3, 4], [1, 2, 3]), 23, 'last axis varies fastest');
    T.eq(m.offset([5], [4]), 4);
    T.eq(m.offset([4, 5, 6], [2, 3, 4]), 82, 'strides of [4,5,6] are [30,6,1]: 2*30 + 3*6 + 4 = 82 (column-major order would give 94)');
    T.eq(m.offset([3, 4], [1, 2]), 6, 'row 1 of a [3,4] matrix starts at 4 (the row length), plus column 2');
  } },
  { step: 'indexing', name: 'offset enumerates every element of a [2,3,4] tensor in row-major order', run(m, T) {
    // Walking the indices with the LAST one fastest must visit offsets 0, 1, 2, … with no gaps or repeats.
    let expect = 0;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 4; k++) {
      T.eq(m.offset([2, 3, 4], [i, j, k]), expect, `offset([2,3,4], [${i},${j},${k}]) should be ${expect}: the last index varies fastest, so strides are [12, 4, 1]`);
      expect++;
    }
  } },
  { step: 'indexing', name: 'transpose swaps rows and columns', run(m, T) {
    const t = m.transpose(m.fromArray([[1, 2, 3], [4, 5, 6]]));
    T.shape(t, [3, 2]);
    T.eq(m.toArray(t), [[1, 4], [2, 5], [3, 6]]);
    const r = randomTensor(m, [3, 5], 4);
    const rt = m.transpose(r);
    T.shape(rt, [5, 3]);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 5; j++) T.eq(rt.data[j * 3 + i], r.data[i * 5 + j], `out[${j}, ${i}] must equal a[${i}, ${j}]`);
  } },
  { step: 'indexing', name: 'transposing twice returns the original and does not mutate the input', run(m, T) {
    const a = m.fromArray([[1, 2], [3, 4], [5, 6]]);
    const before = Array.from(a.data);
    const once = m.transpose(a);
    T.shape(once, [2, 3], 'a single transpose of [3,2] must have shape [2,3]');
    const tt = m.transpose(once);
    T.eq(m.toArray(tt), [[1, 2], [3, 4], [5, 6]]);
    T.eq(Array.from(a.data), before, 'input must not be modified');
  } },
  { step: 'matmul', name: 'multiplies a 2x3 by a 3x2', run(m, T) {
    const c = m.matmul(m.fromArray([[1, 2, 3], [4, 5, 6]]), m.fromArray([[1, 0], [0, 1], [1, 1]]));
    T.shape(c, [2, 2]);
    T.eq(m.toArray(c), [[4, 5], [10, 11]]);
  } },
  { step: 'matmul', name: 'identity leaves a matrix unchanged; inner-dimension mismatch throws', run(m, T) {
    const a = m.fromArray([[2, 3], [4, 5], [6, 7]]);
    T.eq(m.toArray(m.matmul(a, m.fromArray([[1, 0], [0, 1]]))), [[2, 3], [4, 5], [6, 7]]);
    T.throws(() => m.matmul(a, a), '[3,2] x [3,2] has mismatched inner dimensions and must throw');
  } },
  { step: 'matmul', name: 'matches a reference on random 7x5 x 5x4 and is fast enough at 128x128', run(m, T) {
    const a = randomTensor(m, [7, 5], 1), b = randomTensor(m, [5, 4], 2);
    const expect = new Array(28).fill(0);
    for (let i = 0; i < 7; i++) for (let j = 0; j < 4; j++) for (let p = 0; p < 5; p++) expect[i * 4 + j] += a.data[i * 5 + p] * b.data[p * 4 + j];
    const c = m.matmul(a, b);
    T.shape(c, [7, 4], '[7,5] x [5,4] must have shape [7,4]');
    T.close(Array.from(c.data), expect, 1e-4);
    T.eq(Array.from(a.data), Array.from(randomTensor(m, [7, 5], 1).data), 'matmul must not modify its inputs');
    const big = randomTensor(m, [128, 128], 3);
    const t0 = Date.now();
    m.matmul(big, big);
    T.ok(Date.now() - t0 < 2000, `128x128 matmul took ${Date.now() - t0} ms; use typed arrays and a plain triple loop`);
  } },
  { step: 'broadcast', name: 'adds same-shape tensors and scalars', run(m, T) {
    const a = m.fromArray([[1, 2], [3, 4]]);
    T.eq(m.toArray(m.add(a, m.fromArray([[10, 20], [30, 40]]))), [[11, 22], [33, 44]]);
    T.eq(m.toArray(m.add(a, 1)), [[2, 3], [4, 5]]);
    T.eq(m.toArray(m.mul(a, 2)), [[2, 4], [6, 8]]);
  } },
  { step: 'broadcast', name: 'broadcasts a row vector across every row (a bias add)', run(m, T) {
    const a = m.fromArray([[1, 2, 3], [4, 5, 6]]);
    T.eq(m.toArray(m.add(a, m.fromArray([10, 20, 30]))), [[11, 22, 33], [14, 25, 36]]);
    T.eq(m.toArray(m.mul(a, m.fromArray([1, 0, -1]))), [[1, 0, -3], [4, 0, -6]]);
    const x3 = m.fromArray([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]);
    const y3 = m.add(x3, m.fromArray([100, 200, 300]));
    T.shape(y3, [2, 2, 3], 'the output keeps the shape of a');
    T.eq(m.toArray(y3), [[[101, 202, 303], [104, 205, 306]], [[107, 208, 309], [110, 211, 312]]], 'a bias of length d is added to every row, whatever the number of leading axes');
  } },
  { step: 'broadcast', name: 'rejects incompatible shapes and does not mutate inputs', run(m, T) {
    const a = m.fromArray([[1, 2, 3], [4, 5, 6]]);
    T.throws(() => m.add(a, m.fromArray([1, 2])), 'a length-2 vector cannot broadcast onto rows of length 3');
    T.throws(() => m.add(a, m.fromArray([[1, 2], [3, 4], [5, 6]])), 'a [3,2] tensor has the same number of elements as [2,3] but not the same shape; compare shapes, not lengths');
    T.throws(() => m.mul(a, m.fromArray([1, 2, 3, 4, 5, 6])), 'a length-6 vector matches neither the shape [2,3] nor the row length 3 and must throw');
    m.add(a, 5);
    T.eq(m.toArray(a), [[1, 2, 3], [4, 5, 6]], 'input must not be modified');
  } },
  { step: 'rowops', name: 'sum reduces the last axis', run(m, T) {
    const s = m.sum(m.fromArray([[1, 2, 3], [4, 5, 6]]));
    T.shape(s, [2]);
    T.eq(m.toArray(s), [6, 15]);
    T.eq(m.toArray(m.sum(m.fromArray([[[1, 1], [2, 2]], [[3, 3], [4, 4]]]))), [[2, 4], [6, 8]]);
  } },
  { step: 'rowops', name: 'argmax picks the largest entry per row (first on ties)', run(m, T) {
    T.eq(m.argmax(m.fromArray([[1, 5, 2], [7, 7, 0], [-1, -2, -3]])), [1, 0, 0]);
  } },
  { step: 'rowops', name: 'softmax rows sum to one and match exp(x)/sum', run(m, T) {
    const p = m.softmax(m.fromArray([[1, 2, 3]]));
    T.close(m.toArray(p), [[0.09003, 0.24473, 0.66524]], 1e-4);
    T.close(m.sum(p).data[0], 1, 1e-5);
  } },
  { step: 'rowops', name: 'softmax normalises each row on its own', run(m, T) {
    const p = m.softmax(m.fromArray([[1, 2, 3], [1, 1, 1], [0, 0, 5]]));
    T.shape(p, [3, 3]);
    T.close(m.toArray(p), [[0.09003, 0.24473, 0.66524], [1 / 3, 1 / 3, 1 / 3], [0.00664, 0.00664, 0.98672]], 1e-4, 'each row is a separate distribution: use that row\'s own max and sum, not the whole tensor\'s');
    T.close(Array.from(m.sum(p).data), [1, 1, 1], 1e-5, 'every row must sum to 1');
  } },
  { step: 'rowops', name: 'softmax is stable for huge logits (no NaN)', run(m, T) {
    const p = m.softmax(m.fromArray([[1000, 1000, 999]]));
    T.ok(!Number.isNaN(p.data[0]), 'exp(1000) overflows; subtract the row max first');
    T.close(m.toArray(p), [[0.42232, 0.42232, 0.15536]], 1e-4);
    const q = m.softmax(m.fromArray([[-1000, 0, 1000]]));
    T.close(m.toArray(q), [[0, 0, 1]], 1e-6, 'with a spread of 2000 only the max is safe to subtract: subtracting the min (or the mean) still overflows exp');
  } },
  { step: 'layernorm', name: 'normalises each row to mean 0 and variance 1', run(m, T) {
    const y = m.layerNorm(m.fromArray([[1, 2, 3], [10, 20, 60]]));
    const rows = m.toArray(y);
    for (const r of rows) {
      const mu = r.reduce((s, v) => s + v, 0) / r.length;
      const v = r.reduce((s, x) => s + (x - mu) ** 2, 0) / r.length;
      T.close(mu, 0, 1e-5, 'row mean'); T.close(v, 1, 1e-3, 'row variance (biased, divide by d)');
    }
    T.close(rows[0], [-1.22474, 0, 1.22474], 1e-3);
  } },
  { step: 'layernorm', name: 'applies gamma and beta per feature, on every row', run(m, T) {
    const y = m.layerNorm(m.fromArray([[1, 2, 3]]), m.fromArray([2, 2, 2]), m.fromArray([1, 1, 1]));
    T.close(m.toArray(y), [[-1.44949, 1, 3.44949]], 1e-3);
    const w = m.layerNorm(m.fromArray([[1, 2, 3], [10, 20, 60]]), m.fromArray([1, 2, -1]), m.fromArray([0, 0.5, 1]));
    T.close(m.toArray(w), [[-1.22474, 0.5, -0.22474], [-0.92582, -0.42582, -0.38873]], 1e-3, 'feature j of every row is scaled by gamma[j] and shifted by beta[j] (index gamma by column j, not by flat offset)');
  } },
  { step: 'layernorm', name: 'eps goes inside the square root and the eps argument is honoured', run(m, T) {
    const z = m.layerNorm(m.fromArray([[5, 5, 5]]), null, null, 1e-5);
    T.ok(!Number.isNaN(z.data[0]) && Math.abs(z.data[0]) < 1e-3, 'a constant row has zero variance: eps must prevent division by zero');
    T.close(m.toArray(m.layerNorm(m.fromArray([[1, 2, 3]]), null, null, 1)), [[-0.77460, 0, 0.77460]], 1e-4, 'with eps = 1 the divisor is sqrt(2/3 + 1); a hard-coded 1e-5 ignores the argument');
    T.close(m.toArray(m.layerNorm(m.fromArray([[0, 0.001, 0.002]]))), [[-0.30619, 0, 0.30619]], 1e-3, 'for a row with variance 6.7e-7 the result is (x - mu) / sqrt(var + eps); sqrt(var) + eps gives about ±1.21 instead');
  } },
];
