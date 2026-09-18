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
  } },
  { step: 'indexing', name: 'transpose swaps rows and columns', run(m, T) {
    const t = m.transpose(m.fromArray([[1, 2, 3], [4, 5, 6]]));
    T.shape(t, [3, 2]);
    T.eq(m.toArray(t), [[1, 4], [2, 5], [3, 6]]);
  } },
  { step: 'indexing', name: 'transposing twice returns the original and does not mutate the input', run(m, T) {
    const a = m.fromArray([[1, 2], [3, 4], [5, 6]]);
    const before = Array.from(a.data);
    const tt = m.transpose(m.transpose(a));
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
    T.close(Array.from(m.matmul(a, b).data), expect, 1e-4);
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
  } },
  { step: 'broadcast', name: 'rejects incompatible shapes and does not mutate inputs', run(m, T) {
    const a = m.fromArray([[1, 2, 3], [4, 5, 6]]);
    T.throws(() => m.add(a, m.fromArray([1, 2])), 'a length-2 vector cannot broadcast onto rows of length 3');
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
  { step: 'rowops', name: 'softmax is stable for huge logits (no NaN)', run(m, T) {
    const p = m.softmax(m.fromArray([[1000, 1000, 999]]));
    T.ok(!Number.isNaN(p.data[0]), 'exp(1000) overflows; subtract the row max first');
    T.close(m.toArray(p), [[0.42232, 0.42232, 0.15536]], 1e-4);
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
  { step: 'layernorm', name: 'applies gamma and beta and respects eps', run(m, T) {
    const y = m.layerNorm(m.fromArray([[1, 2, 3]]), m.fromArray([2, 2, 2]), m.fromArray([1, 1, 1]));
    T.close(m.toArray(y), [[-1.44949, 1, 3.44949]], 1e-3);
    const z = m.layerNorm(m.fromArray([[5, 5, 5]]), null, null, 1e-5);
    T.ok(!Number.isNaN(z.data[0]) && Math.abs(z.data[0]) < 1e-3, 'a constant row has zero variance: eps must prevent division by zero');
  } },
];
