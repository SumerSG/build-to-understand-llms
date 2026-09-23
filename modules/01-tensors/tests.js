import { rng } from 'lib/util.js';

function randomTensor(m, shape, seed) {
  const next = rng(seed);
  const t = m.raw(shape);
  for (let i = 0; i < t.data.length; i++) t.data[i] = next() * 2 - 1;
  return t;
}

// Row sums computed here rather than with the learner's sum(), so a softmax test never fails because of sum.
function rowSums(t) {
  const d = t.shape[t.shape.length - 1], out = [];
  for (let r = 0; r < t.data.length; r += d) { let z = 0; for (let j = 0; j < d; j++) z += t.data[r + j]; out.push(z); }
  return out;
}

const sameNumbers = (x, y) => x.length === y.length && x.every((v, i) => (Number.isNaN(v) && Number.isNaN(y[i])) || Math.abs(v - y[i]) < 1e-3);

// LayerNorm as a learner writes it when the formula's (row - mu)^2 is typed literally: in JavaScript ^ is a bit
// operation on whole numbers, not a power, so the "variance" is not a real square (it can even be negative).
function layerNormCaret(rows, gamma, beta, eps) {
  return rows.flatMap((r) => {
    const d = r.length, mu = r.reduce((s, v) => s + v, 0) / d;
    const v = r.reduce((s, x) => s + ((x - mu) ^ 2), 0) / d;
    return r.map((x, j) => (x - mu) / Math.sqrt(v + eps) * (gamma ? gamma[j] : 1) + (beta ? beta[j] : 0));
  });
}

/** Name the ^ slip, or a negative variance in general, before the numeric comparison reports a bare NaN. */
function layerNormSlip(T, y, rows, gamma = null, beta = null, eps = 1e-5) {
  const got = y && y.data ? Array.from(y.data) : null;
  if (!got) return;
  const caret = layerNormCaret(rows, gamma, beta, eps);
  if (sameNumbers(got, caret)) {
    T.fail(`layerNorm gave [${got.map((v) => (Number.isNaN(v) ? 'NaN' : +v.toFixed(4))).join(', ')}], which is exactly what (x - mu) ^ 2 gives. In JavaScript ^ is not "to the power of" (3 ^ 2 is 1), so the variance is not a real square${got.some(Number.isNaN) ? ' (here it even came out negative, and Math.sqrt of a negative number is NaN)' : ''}. Square with c * c or c ** 2, where c = a.data[base + j] - mu`);
  }
  if (got.some(Number.isNaN)) {
    T.fail('the result contains NaN although the input has none. Two usual causes: the variance came out negative (a real sum of squares never is: square with c * c or c ** 2, never ^), so Math.sqrt gave NaN; or the code read an element that does not exist (undefined), for example a.data at an offset past the end of the row or gamma[j] instead of gamma.data[j]');
  }
}

/** NaN in a matmul result means a number that does not exist was read: say which reads usually do it. */
function matmulNaN(T, c) {
  if (c && c.data && Array.from(c.data).some(Number.isNaN)) {
    T.fail(`the result contains NaN ([${Array.from(c.data).slice(0, 8).join(', ')}${c.data.length > 8 ? ', …' : ''}]): NaN appears when a number that does not exist is read (it comes back as undefined). Two usual causes. (1) Reading the tensor object instead of its numbers: a[3] is undefined; the numbers are in a.data and b.data. (2) An offset past the end of the data: a row of A has k numbers, so A[i, p] is a.data[i * k + p]; a row of B has m numbers, so B[p, j] is b.data[p * m + j] (b.data[p * k + j] runs past the end when k and m differ)`);
  }
}

/** transpose must move every number exactly once: zeros and repeats mean wrong new positions. */
function lostNumbers(T, t, nested) {
  const n = nested.length, mm = nested[0].length, input = nested.flat();
  const vals = t && t.data ? Array.from(t.data) : [];
  const sorted = (xs) => [...xs].sort((x, y) => x - y).join();
  if (vals.length === input.length && sorted(vals) !== sorted(input)) {
    T.fail(`transpose of ${JSON.stringify(nested)} (shape [${n}, ${mm}]) gave the numbers [${vals.join(', ')}]: ${vals.includes(0) ? 'some positions of out were never written (they are still 0) and ' : ''}some of the ${input.length} numbers are missing or appear twice, so the new positions are wrong. The result has shape [m, n] = [${mm}, ${n}]: it has n = ${n} columns, not m, so element [j, i] sits at j * n + i. (A Float32Array silently ignores a write past its end, which is where the lost numbers went.)`);
  }
}

/** add or mul gave back nothing: usually the top-level binary stub is still empty, or a return is missing. */
function noResult(T, got, call) {
  if (got === undefined) {
    T.fail(`${call} returned undefined (nothing). Either the binary stub above export function add still has an empty body (a function without return gives back undefined; write the body there, in that stub, not in a second binary inside add), or add / mul forgot to return: their body is return binary(a, b, (x, y) => ...);`);
  }
}

/** A function that is "not defined" in step 3 is almost always the binary helper written in the wrong place. */
function notDefined(T, e, call) {
  const msg = String(e && e.message);
  const name = (msg.match(/^(\w+) is not defined/) || [])[1];
  if (!name) return;
  T.fail(`${call} stopped with "${msg}". ${name === 'binary' ? 'The helper binary exists only where it is written: if it sits inside add (or inside any other function), mul and the rest of the file cannot see it. Write its body in the empty function binary(a, b, fn) stub above export function add, on its own at the top level of the file' : `You used the name ${name} before creating it, or you created it inside another function, where the rest of the file cannot see it: move it out to the top level`}`);
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
    lostNumbers(T, t, [[1, 2, 3], [4, 5, 6]]);
    T.eq(m.toArray(t), [[1, 4], [2, 5], [3, 6]]);
    const r = randomTensor(m, [3, 5], 4);
    // Expected values come from a fresh copy of the input, so a transpose that overwrites its argument
    // is reported by the next test rather than showing up here as wrong index arithmetic.
    const src = Array.from(randomTensor(m, [3, 5], 4).data);
    const rt = m.transpose(r);
    T.shape(rt, [5, 3]);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 5; j++) T.eq(rt.data[j * 3 + i], src[i * 5 + j], `out[${j}, ${i}] must equal a[${i}, ${j}], which sits at flat offset ${i * 5 + j} of the [3,5] input and must land at offset ${j * 3 + i} of the [5,3] output`);
  } },
  { step: 'indexing', name: 'transposing twice returns the original and does not mutate the input', run(m, T) {
    const a = m.fromArray([[1, 2], [3, 4], [5, 6]]);
    const before = Array.from(a.data), dataBefore = a.data;
    const once = m.transpose(a);
    T.ok(once !== a, 'transpose returned its input object: build and return a new { shape, data } object instead of assigning to a.shape or a.data');
    T.eq(a.shape, [3, 2], 'transpose changed a.shape: the input must keep its shape [3,2]; put [m, n] on the new object you return');
    T.ok(a.data === dataBefore && once.data !== a.data, 'transpose must write into a freshly allocated Float32Array, not replace or reuse a.data');
    T.eq(Array.from(a.data), before, 'transpose changed the values in a.data: read from the input, write only into the new array');
    T.shape(once, [2, 3], 'a single transpose of [3,2] must have shape [2,3]');
    lostNumbers(T, once, [[1, 2], [3, 4], [5, 6]]);
    const tt = m.transpose(once);
    T.eq(m.toArray(tt), [[1, 2], [3, 4], [5, 6]]);
  } },
  { step: 'matmul', name: 'multiplies a 2x3 by a 3x2', run(m, T) {
    const c = m.matmul(m.fromArray([[1, 2, 3], [4, 5, 6]]), m.fromArray([[1, 0], [0, 1], [1, 1]]));
    matmulNaN(T, c);
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
    matmulNaN(T, c);
    T.close(Array.from(c.data), expect, 1e-4);
    T.eq(Array.from(a.data), Array.from(randomTensor(m, [7, 5], 1).data), 'matmul must not modify its inputs');
    // A 32x32 multiply first (a typed-array loop takes well under a millisecond), so a very slow loop fails
    // in about a second instead of running synchronously for minutes at 128x128.
    const small = randomTensor(m, [32, 32], 5);
    const s0 = performance.now();
    m.matmul(small, small);
    const smallMs = performance.now() - s0;
    T.ok(smallMs < 250, `matmul took ${Math.round(smallMs)} ms at 32x32 (budget 250 ms, so 128x128 would take about ${Math.round(smallMs * 64 / 1000)} s); are you calling toArray, offset or allocating inside the loop? Index a.data and b.data directly`);
    const big = randomTensor(m, [128, 128], 3);
    const t0 = performance.now();
    m.matmul(big, big);
    const bigMs = performance.now() - t0;
    T.ok(bigMs < 2000, `128x128 matmul took ${Math.round(bigMs)} ms (budget 2000 ms); use typed arrays and a plain triple loop`);
  } },
  { step: 'broadcast', name: 'adds same-shape tensors and scalars', run(m, T) {
    const a = m.fromArray([[1, 2], [3, 4]]);
    let same;
    try { same = m.add(a, m.fromArray([[10, 20], [30, 40]])); } catch (e) {
      notDefined(T, e, 'add(a, b) with two [2,2] tensors');
      T.fail(`add threw "${e.message}" for two tensors that both have shape [2,2]. If your code compares the shapes with == or ===, that is the cause: in JavaScript [2, 2] === [2, 2] is false, because arrays compare by identity (the same array object), not by contents. Compare the lengths, then each entry`);
    }
    noResult(T, same, 'add(a, b) with two [2,2] tensors');
    T.eq(m.toArray(same), [[11, 22], [33, 44]]);
    const plusOne = m.add(a, 1);
    noResult(T, plusOne, 'add(a, 1)');
    T.eq(m.toArray(plusOne), [[2, 3], [4, 5]]);
    let doubled;
    try { doubled = m.mul(a, 2); } catch (e) { notDefined(T, e, 'mul(a, 2)'); throw e; }
    noResult(T, doubled, 'mul(a, 2)');
    T.eq(m.toArray(doubled), [[2, 4], [6, 8]]);
  } },
  { step: 'broadcast', name: 'broadcasts a row vector across every row (a bias add)', run(m, T) {
    const a = m.fromArray([[1, 2, 3], [4, 5, 6]]);
    try {
      noResult(T, m.add(a, m.fromArray([10, 20, 30])), 'add(a, bias)');
      noResult(T, m.mul(a, m.fromArray([1, 0, -1])), 'mul(a, row)');
    } catch (e) { notDefined(T, e, 'add or mul'); throw e; }
    T.eq(m.toArray(m.add(a, m.fromArray([10, 20, 30]))), [[11, 22, 33], [14, 25, 36]]);
    T.eq(m.toArray(m.mul(a, m.fromArray([1, 0, -1]))), [[1, 0, -3], [4, 0, -6]]);
    const x3 = m.fromArray([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]);
    const y3 = m.add(x3, m.fromArray([100, 200, 300]));
    T.shape(y3, [2, 2, 3], 'the output keeps the shape of a');
    T.eq(m.toArray(y3), [[[101, 202, 303], [104, 205, 306]], [[107, 208, 309], [110, 211, 312]]], 'a bias of length d is added to every row, whatever the number of leading axes');
  } },
  { step: 'broadcast', name: 'rejects incompatible shapes and does not mutate inputs', run(m, T) {
    const a = m.fromArray([[1, 2, 3], [4, 5, 6]]);
    try { noResult(T, m.mul(a, 2), 'mul(a, 2)'); } catch (e) { notDefined(T, e, 'mul(a, 2)'); throw e; }
    T.throws(() => m.add(a, m.fromArray([1, 2])), 'a length-2 vector cannot broadcast onto rows of length 3');
    T.throws(() => m.add(a, m.fromArray([[1, 2], [3, 4], [5, 6]])), 'a [3,2] tensor has the same number of elements as [2,3] but not the same shape; compare shapes, not lengths');
    T.throws(() => m.mul(a, m.fromArray([1, 2, 3, 4, 5, 6])), 'a length-6 vector matches neither the shape [2,3] nor the row length 3 and must throw');
    m.add(a, 5);
    T.eq(m.toArray(a), [[1, 2, 3], [4, 5, 6]], 'input must not be modified');
  } },
  { step: 'rowops', name: 'sum reduces the last axis', run(m, T) {
    const s = m.sum(m.fromArray([[1, 2, 3], [4, 5, 6]]));
    const kind = s === undefined ? 'undefined' : ArrayBuffer.isView(s) ? s.constructor.name : Array.isArray(s) ? 'a plain array' : typeof s;
    T.ok(s && typeof s === 'object' && Array.isArray(s.shape) && s.data, `sum must return a tensor { shape: a.shape.slice(0, -1), data } (unlike argmax, which returns a plain array); got ${kind}`);
    T.shape(s, [2], 'summing a [2,3] tensor along the last axis leaves shape [2]');
    T.eq(m.toArray(s), [6, 15]);
    T.eq(m.toArray(m.sum(m.fromArray([[[1, 1], [2, 2]], [[3, 3], [4, 4]]]))), [[2, 4], [6, 8]], 'a [2,2,2] tensor sums to shape [2,2]: every leading axis is kept, only the last is reduced');
  } },
  { step: 'rowops', name: 'argmax picks the largest entry per row (first on ties)', run(m, T) {
    const got = m.argmax(m.fromArray([[1, 5, 2], [7, 7, 0], [-1, -2, -3]]));
    const list = got && typeof got.length === 'number' ? Array.from(got) : null;
    if (list && list.join() === '5,7,-1') {
      T.fail('argmax gave [5, 7, -1]: these are the largest values themselves. argmax wants their position j in the row (5 is at position 1 of [1, 5, 2]), so keep track of the best j, not the best value');
    }
    if (list && list.join() === '1,3,6') {
      T.fail('argmax gave [1, 3, 6]: these are flat offsets in the whole data array (r * d + j). argmax wants the position j inside each row, a number from 0 to d - 1');
    }
    if (list && list.join() === '1,1,0') {
      T.fail('argmax gave [1, 1, 0]: on the tie in row [7, 7, 0] it picked the later 7. Keep the first position on ties: replace the best only when a value is strictly larger (>), not larger or equal (>=)');
    }
    T.eq(list || got, [1, 0, 0], 'for each row, the position of its largest value: 5 is at position 1 of [1, 5, 2]; 7 first appears at position 0 of [7, 7, 0]; -1 is at position 0 of [-1, -2, -3]');
  } },
  { step: 'rowops', name: 'softmax rows sum to one and match exp(x)/sum', run(m, T) {
    const p = m.softmax(m.fromArray([[1, 2, 3]]));
    T.close(m.toArray(p), [[0.09003, 0.24473, 0.66524]], 1e-4);
    T.close(rowSums(p), [1], 1e-5, 'the row must sum to 1');
  } },
  { step: 'rowops', name: 'softmax normalises each row on its own', run(m, T) {
    const p = m.softmax(m.fromArray([[1, 2, 3], [1, 1, 1], [0, 0, 5]]));
    T.shape(p, [3, 3]);
    T.close(m.toArray(p), [[0.09003, 0.24473, 0.66524], [1 / 3, 1 / 3, 1 / 3], [0.00664, 0.00664, 0.98672]], 1e-4, 'each row is a separate distribution: use that row\'s own max and sum, not the whole tensor\'s');
    T.close(rowSums(p), [1, 1, 1], 1e-5, 'every row must sum to 1');
    const q = m.softmax(m.fromArray([[1000, 999], [-1000, -999]]));
    T.close(m.toArray(q), [[0.73106, 0.26894], [0.26894, 0.73106]], 1e-4, 'shift each row by its own max: shifting the second row by the whole tensor\'s max (1000) gives exp(-2000) = 0 for both entries, and 0 / 0 = NaN');
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
    layerNormSlip(T, y, [[1, 2, 3], [10, 20, 60]]);
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
    layerNormSlip(T, w, [[1, 2, 3], [10, 20, 60]], [1, 2, -1], [0, 0.5, 1]);
    T.close(m.toArray(w), [[-1.22474, 0.5, -0.22474], [-0.92582, -0.42582, -0.38873]], 1e-3, 'feature j of every row is scaled by gamma[j] and shifted by beta[j] (index gamma by column j, not by flat offset)');
  } },
  { step: 'layernorm', name: 'eps goes inside the square root and the eps argument is honoured', run(m, T) {
    const z = m.layerNorm(m.fromArray([[5, 5, 5]]), null, null, 1e-5);
    T.ok(!Number.isNaN(z.data[0]) && Math.abs(z.data[0]) < 1e-3, 'a constant row has zero variance: eps must prevent division by zero');
    const small = m.layerNorm(m.fromArray([[0, 0.001, 0.002]]));
    const smallRow = Array.from(new Float32Array([0, 0.001, 0.002]));
    if (small && small.data && sameNumbers(Array.from(small.data), layerNormCaret([smallRow], null, null, 1e-5))) {
      T.fail(`layerNorm([[0, 0.001, 0.002]]) gave [${Array.from(small.data).map((v) => +v.toFixed(5)).join(', ')}], which is what (x - mu) ^ 2 gives: ^ is not a power in JavaScript (it works on whole numbers, so 0.001 ^ 2 is 2), and the "variance" comes out as 2 instead of 0.00000067. Square with c * c or c ** 2`);
    }
    T.close(m.toArray(small), [[-0.30619, 0, 0.30619]], 1e-3, 'for a row with variance 6.7e-7 the result is (x - mu) / sqrt(var + eps); sqrt(var) + eps gives about ±1.21 instead');
    const y1 = Array.from(m.layerNorm(m.fromArray([[1, 2, 3]]), null, null, 1).data);
    const near = (ys, v) => ys.length === 3 && Math.abs(ys[0] + v) < 1e-3 && Math.abs(ys[2] - v) < 1e-3;
    const why = near(y1, 1 / (Math.sqrt(2 / 3) + 1)) ? 'eps must be added to the variance inside the square root: sqrt(var + eps), not sqrt(var) + eps'
      : near(y1, 1 / Math.sqrt(2 / 3 + 1e-5)) ? 'the eps argument is ignored: this output is what eps = 1e-5 gives, so use the eps parameter rather than a hard-coded constant'
      : 'with eps = 1 the divisor is sqrt(2/3 + 1)';
    T.close(y1, [-0.77460, 0, 0.77460], 1e-4, why);
  } },
];
