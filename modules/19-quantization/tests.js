import * as ops from 'lib/ops.js';

// Independent references so a test never trusts another step's learner code.
function refAbsmax(data, bits) {
  const qmax = 2 ** (bits - 1) - 1;
  let amax = 0;
  for (const v of data) amax = Math.max(amax, Math.abs(v));
  const scale = amax > 0 ? amax / qmax : 1;
  const q = Array.from(data, (v) => Math.max(-qmax - 1, Math.min(qmax, Math.round(v / scale))));
  return { q, scale };
}
function mse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s / a.length;
}
function gaussian(shape, seed, std = 1) {
  const next = (function (s) { let a = s >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })(seed);
  return ops.randn(shape, next, std);
}
const isIntArray = (q) => (ArrayBuffer.isView(q) || Array.isArray(q)) && Array.from(q).every((v) => Number.isInteger(v));

export const tests = [
  // ---------- step 1: absmax ----------
  { step: 'absmax', name: 'int8 absmax maps the largest |x| to 127 and rounds the rest', run(m, T) {
    const r = m.quantizeAbsmax([1.27, -0.5, 0.31, 0.02], 8);
    T.ok(r && isIntArray(r.q) && r.q.length === 4, 'quantizeAbsmax must return { q, scale } with one integer code per input value');
    T.close(r.scale, 0.01, 1e-6, 'scale must be max|x| / 127 = 1.27 / 127 so that the largest value uses the full int8 range');
    T.eq(Array.from(r.q), [127, -50, 31, 2], 'q = round(x / scale): the largest |x| must land exactly on 127, the others in proportion');
    const back = m.dequantizeAbsmax(r);
    T.close(Array.from(back), [1.27, -0.5, 0.31, 0.02], 1e-6, 'dequantizeAbsmax must return q * scale');
  } },
  { step: 'absmax', name: 'int4 uses the range [-8, 7] with scale max|x| / 7; zeros stay finite', run(m, T) {
    const x = [0.7, -0.32, 0.1, -0.7, 0.06];
    const r = m.quantizeAbsmax(x, 4);
    T.close(r.scale, 0.1, 1e-6, 'for 4 bits the symmetric range is [-8, 7], so scale = max|x| / 7');
    T.eq(Array.from(r.q), [7, -3, 1, -7, 1], 'int4 codes are round(x / 0.1): 0.7 -> 7, -0.32 -> -3, 0.1 -> 1, -0.7 -> -7, 0.06 -> 1');
    const maxCode = Math.max(...Array.from(r.q).map(Math.abs));
    T.ok(maxCode === 7, `the largest |q| must be 7 for int4, got ${maxCode}: an int8 range would waste 4-bit codes`);
    const z = m.quantizeAbsmax(new Float32Array(6), 8);
    T.ok(Number.isFinite(z.scale) && Array.from(z.q).every((v) => v === 0), 'an all-zero tensor must give a finite scale and all-zero codes (no NaN from 0/0)');
  } },
  { step: 'absmax', name: 'round-trip error is at most scale/2 per value and int4 is worse than int8', run(m, T) {
    const x = gaussian([1000], 11, 0.5).data;
    const r8 = m.quantizeAbsmax(x, 8), r4 = m.quantizeAbsmax(x, 4);
    const y8 = m.dequantizeAbsmax(r8), y4 = m.dequantizeAbsmax(r4);
    const ref8 = refAbsmax(x, 8);
    T.close(r8.scale, ref8.scale, 1e-6, 'scale must be max|x| / 127');
    T.eq(Array.from(r8.q), ref8.q, 'codes must be round(x / scale) clamped to [-128, 127]');
    const e8 = m.errorStats(x, y8), e4 = m.errorStats(x, y4);
    T.ok(e8.maxAbs <= r8.scale / 2 + 1e-6, `rounding to the nearest code bounds the error by scale/2 = ${(r8.scale / 2).toFixed(5)}, got max error ${e8.maxAbs.toFixed(5)}`);
    T.ok(e4.maxAbs <= r4.scale / 2 + 1e-6, `int4 max error must be at most scale/2 = ${(r4.scale / 2).toFixed(4)}, got ${e4.maxAbs.toFixed(4)}`);
    T.ok(e4.mse > 50 * e8.mse, `int4's step is max|x|/7 against int8's max|x|/127, so its MSE should be about (127/7)^2 ≈ 330x larger than int8 (got ${e4.mse.toExponential(2)} vs ${e8.mse.toExponential(2)})`);
    T.ok(e8.cosine > 0.9999, `int8 should preserve direction almost perfectly (cosine ${e8.cosine.toFixed(5)})`);
  } },

  // ---------- step 2: per-channel ----------
  { step: 'perchannel', name: 'one scale per row: a small row and a large row each fill the int8 range', run(m, T) {
    const w = ops.fromArray([[0.1, -0.2, 0.3], [10, -20, 30]]);
    const r = m.quantizePerChannel(w, 8);
    T.eq(r.shape, [2, 3], 'the quantised matrix must remember its shape');
    T.close(Array.from(r.scales), [0.3 / 127, 30 / 127], 1e-6, 'scales[r] = max|row r| / 127: the small row must NOT be crushed by the large row');
    T.eq(Array.from(r.q), [42, -85, 127, 42, -85, 127], 'both rows quantise to the same codes because each has its own scale');
    T.eq(r.groupSize, 3, 'per-channel is the special case groupSize = cols; record it so dequantize can index the scales');
    T.ok(r.zeros === null || r.zeros === undefined, 'symmetric quantisation has no zero point (zeros: null)');
    const r4 = m.quantizePerChannel(ops.fromArray([[0.7, -0.32, 0.1], [7, -3.2, 1]]), 4);
    T.close(Array.from(r4.scales), [0.1, 1], 1e-6, 'quantizePerChannel(w, 4) must use the int4 range: scales[r] = max|row r| / 7, not / 127 (read qmax from qrange(bits))');
    T.eq(Array.from(r4.q), [7, -3, 1, 7, -3, 1], 'int4 per-channel codes are round(x / scale) in [-8, 7]: each row maps 0.7, -0.32, 0.1 (times 1 or 10) to 7, -3, 1');
  } },
  { step: 'perchannel', name: 'dequantize handles groupSize < cols with scales indexed row-major by [row, group]', run(m, T) {
    const qw = { shape: [2, 4], q: Int8Array.from([1, 2, 3, 4, -1, -2, -3, -4]), scales: Float32Array.from([1, 10, 100, 1000]), zeros: null, bits: 8, groupSize: 2 };
    const back = m.dequantize(qw);
    T.shape(back, [2, 4]);
    T.close(back, ops.fromArray([[1, 2, 30, 40], [-100, -200, -3000, -4000]]), 1e-6, 'value [r, c] must use scales[r * nGroups + floor(c / groupSize)]; step 3 relies on this');
    const z = { ...qw, zeros: Float32Array.from([0, 0, 1, 1]) };
    T.close(m.dequantize(z), ops.fromArray([[1, 2, 30, 40], [-200, -300, -4000, -5000]]), 1e-6, 'when zeros is present the code is (q - zero) * scale');
  } },
  { step: 'perchannel', name: 'per-channel error on a matrix with one huge row is far below per-tensor absmax', run(m, T) {
    const w = gaussian([8, 32], 5, 0.02);
    for (let c = 0; c < 32; c++) w.data[3 * 32 + c] *= 100; // row 3 is 100x larger than the rest
    const pc = m.quantizePerChannel(w, 8);
    const back = m.dequantize(pc);
    const ref = refAbsmax(w.data, 8);
    const tensorBack = ref.q.map((q) => q * ref.scale);
    const small = (arr) => Array.from(arr).filter((_, i) => Math.floor(i / 32) !== 3);
    const ePer = mse(small(w.data), small(back.data)), eTensor = mse(small(w.data), small(tensorBack));
    T.ok(ePer * 100 < eTensor, `with one scale per row, the 7 small rows should have >100x lower MSE than with a single tensor scale set by the big row (got ${ePer.toExponential(2)} vs ${eTensor.toExponential(2)})`);
    T.ok(Array.from(pc.q).every((v) => Number.isInteger(v) && v >= -128 && v <= 127), 'codes must be integers in [-128, 127]');
    T.ok(pc.scales.length === 8, 'need exactly one scale per row');
  } },

  // ---------- step 3: groups ----------
  { step: 'groups', name: 'symmetric int4 groups: each group of groupSize values gets its own scale', run(m, T) {
    const w = ops.fromArray([[0.7, -0.32, 0.1, 0.2, 7, -3.2, 1, 2]]);
    const r = m.quantizeGroups(w, { bits: 4, groupSize: 4 });
    T.eq(r.groupSize, 4);
    T.close(Array.from(r.scales), [0.1, 1], 1e-6, 'two groups in the row: scales = max|group| / 7');
    T.eq(Array.from(r.q), [7, -3, 1, 2, 7, -3, 1, 2], 'codes are round(x / scale) within each group, each group using its own scale');
    T.ok(r.zeros === null || r.zeros === undefined, 'symmetric groups carry no zero points');
    T.close(m.dequantize(r), ops.fromArray([[0.7, -0.3, 0.1, 0.2, 7, -3, 1, 2]]), 1e-6, 'dequantize must apply the right scale to each group');
    T.throws(() => m.quantizeGroups(w, { bits: 4, groupSize: 3 }), 'cols = 8 is not divisible by groupSize = 3; throw rather than silently mis-index');
    const w2 = ops.fromArray([[0.7, 0.1, 7, 1], [0.07, 0.01, 70, 10]]);
    const r2 = m.quantizeGroups(w2, { bits: 4, groupSize: 2 });
    T.close(Array.from(r2.scales), [0.1, 1, 0.01, 10], 1e-6, 'with 2 rows of 2 groups the scales are laid out row-major: scales[r * nGroups + g] = [row0 g0, row0 g1, row1 g0, row1 g1]');
    T.eq(Array.from(r2.q), [7, 1, 7, 1, 7, 1, 7, 1], 'every group here quantises to codes 7, 1 under its own scale');
  } },
  { step: 'groups', name: 'smaller groups give strictly lower error on a matrix with outliers', run(m, T) {
    const w = gaussian([16, 256], 21, 0.02);
    for (let r = 0; r < 16; r++) for (let j = 0; j < 4; j++) w.data[r * 256 + (r * 37 + j * 61) % 256] *= 25; // a few outliers per row
    const errs = [16, 64, 256].map((G) => m.errorStats(w, m.dequantize(m.quantizeGroups(w, { bits: 4, groupSize: G }))).mse);
    T.ok(errs[0] < errs[1] && errs[1] < errs[2], `MSE must fall as the group shrinks because fewer values share an outlier's scale: g16 ${errs[0].toExponential(2)}, g64 ${errs[1].toExponential(2)}, g256 ${errs[2].toExponential(2)}`);
    T.ok(errs[0] * 3 < errs[2], `g16 should be at least 3x better than one scale per row here (got ${(errs[2] / errs[0]).toFixed(1)}x)`);
    const r = m.quantizeGroups(w, { bits: 4, groupSize: 64 });
    T.eq(r.scales.length, 16 * 4, 'a [16, 256] matrix with groupSize 64 has 16 * 4 = 64 scales');
    T.ok(Array.from(r.q).every((v) => v >= -8 && v <= 7), 'int4 symmetric codes must lie in [-8, 7]');
  } },
  { step: 'groups', name: 'asymmetric int4 spends all 16 codes on the actual [min, max] and stores a zero point', run(m, T) {
    const vals = [2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 2.9, 3.0];
    const w = ops.fromArray([vals]);
    const asym = m.quantizeGroups(w, { bits: 4, groupSize: 8, symmetric: false });
    T.ok(asym.zeros && asym.zeros.length === 1, 'asymmetric quantisation must return one zero point per group');
    T.close(asym.scales[0], 1 / 15, 1e-6, 'scale = (max - min) / 15 = 1 / 15: the whole 4-bit range covers 2.0..3.0');
    const q = Array.from(asym.q);
    T.ok(q.every((v) => v >= 0 && v <= 15), `asymmetric codes are unsigned, in [0, 15]; got ${q}`);
    T.ok(Math.min(...q) === 0 && Math.max(...q) === 15, `min must map to code 0 and max to code 15 (got ${q})`);
    const back = m.dequantize(asym);
    const eA = m.errorStats(w, back), eS = m.errorStats(w, m.dequantize(m.quantizeGroups(w, { bits: 4, groupSize: 8, symmetric: true })));
    T.ok(eA.maxAbs <= 1 / 30 + 1e-6, `asymmetric max error must be at most scale/2 = ${(1 / 30).toFixed(4)}; got ${eA.maxAbs.toFixed(4)} — is dequantize subtracting the zero point?`);
    T.ok(Array.from(asym.zeros).every((z) => Number.isInteger(z)), `the zero point is an integer code, zero = round(-min / scale); got ${Array.from(asym.zeros)}`);
    const st = m.quantizeGroups(ops.fromArray([[-0.46, 0.2, 1, 2.54]]), { bits: 4, groupSize: 4, symmetric: false });
    T.close(st.scales[0], 0.2, 1e-6, 'for a group spanning [-0.46, 2.54], scale = 3 / 15 = 0.2');
    T.eq(Array.from(st.zeros), [2], 'zero = round(-min / scale) = round(2.3) = 2: an integer code, the one that stands for 0.0');
    T.eq(Array.from(st.q), [0, 3, 7, 15], 'codes are round(x / scale) + zero: -0.46 -> -2 + 2 = 0, 0.2 -> 1 + 2 = 3, 1 -> 5 + 2 = 7, 2.54 -> 13 + 2 = 15');
    T.ok(eA.mse * 4 < eS.mse, `symmetric wastes the negative half of the range on all-positive values: expected asymmetric MSE < 1/4 of symmetric (got ${eA.mse.toExponential(2)} vs ${eS.mse.toExponential(2)})`);
  } },

  // ---------- step 4: quantised matmul ----------
  { step: 'qmatmul', name: 'quantizeWeight stores W [K, N] as [N, K] with groups along K', run(m, T) {
    const W = gaussian([32, 6], 8, 0.05);
    const qw = m.quantizeWeight(W, { bits: 8, groupSize: 16 });
    T.eq(qw.shape, [6, 32], 'rows of the stored matrix are output channels: shape must be [N, K]');
    T.eq(qw.scales.length, 6 * 2, '6 output channels x 2 groups of 16 along the input dimension');
    const back = ops.transpose(m.dequantize(qw));
    T.shape(back, [32, 6]);
    const e = m.errorStats(W, back);
    T.ok(e.maxAbs <= Math.max(...Array.from(qw.scales)) / 2 + 1e-6, `dequantised weight must be within half a scale of W everywhere (max error ${e.maxAbs.toExponential(2)})`);
  } },
  { step: 'qmatmul', name: 'quantizedMatmul equals x times the dequantised weight, for symmetric and zero-point weights', run(m, T) {
    const x = gaussian([5, 32], 9);
    const W = gaussian([32, 6], 10, 0.05);
    for (const opts of [{ bits: 8, groupSize: 16 }, { bits: 4, groupSize: 8, symmetric: false }]) {
      const qw = m.quantizeWeight(W, opts);
      const y = m.quantizedMatmul(x, qw);
      T.shape(y, [5, 6], 'x [T, K] times a [N, K] quantised weight gives [T, N]');
      const ref = ops.matmul(x, ops.transpose(m.dequantize(qw)));
      T.close(y, ref, 1e-3, `with ${JSON.stringify(opts)} the on-the-fly kernel must reproduce x · dequantize(W): check the per-group scale and the zero point`);
    }
    T.throws(() => m.quantizedMatmul(gaussian([2, 16], 1), m.quantizeWeight(W, { bits: 8, groupSize: 16 })), 'a [2, 16] input cannot multiply a weight expecting 32 channels; throw');
  } },
  { step: 'qmatmul', name: 'relative error against fp32: int8 per-channel well under 1%, int4 g32 under 10%', run(m, T) {
    const x = gaussian([8, 64], 12);
    const W = gaussian([64, 16], 13, 0.05);
    const ref = ops.matmul(x, W);
    const rel8 = m.errorStats(ref, m.quantizedMatmul(x, m.quantizeWeight(W, { bits: 8, groupSize: 64 }))).rel;
    const rel4 = m.errorStats(ref, m.quantizedMatmul(x, m.quantizeWeight(W, { bits: 4, groupSize: 32 }))).rel;
    T.ok(rel8 < 0.01, `int8 with one scale per output channel should give < 1% relative L2 error on the layer output, got ${(100 * rel8).toFixed(3)}%`);
    T.ok(rel4 < 0.10, `int4 with groups of 32 should give < 10% relative error, got ${(100 * rel4).toFixed(2)}%`);
    T.ok(rel4 > rel8, 'int4 must be less accurate than int8; if not, one of the paths is not using its codes');
  } },

  // ---------- step 5: memory ----------
  { step: 'memory', name: 'weightBytes counts codes plus one scale (and zero) per group', run(m, T) {
    T.close(m.weightBytes(8.03e9, { bits: 16 }), 16.06e9, 1e-9, 'fp16/bf16: 2 bytes per parameter, 16.06 GB for Llama-3-8B');
    T.close(m.weightBytes(8.03e9, { bits: 8 }), 8.03e9, 1e-9, 'int8 with a per-tensor scale: 1 byte per parameter');
    T.close(m.weightBytes(8.03e9, { bits: 4, groupSize: 128, scaleBits: 16 }), 4.14046875e9, 1e-9, 'int4 g128: 4.015 GB of codes + 8.03e9/128 fp16 scales = 0.1255 GB');
    T.close(m.bitsPerParam({ bits: 4, groupSize: 128, scaleBits: 16 }), 4.125, 1e-9, '4 + 16/128 = 4.125 bits per parameter');
    T.close(m.bitsPerParam({ bits: 4, groupSize: 32, scaleBits: 16, zeroBits: 4 }), 4.625, 1e-9, '4 + (16 + 4)/32 = 4.625: small groups cost real bits');
    T.close(m.bitsPerParam({ bits: 4 }), 4, 1e-9, 'a single per-tensor scale adds nothing measurable (groupSize defaults to Infinity)');
  } },
  { step: 'memory', name: 'kvCacheBytes = 2 * layers * kvHeads * headDim * context * batch * bytes', run(m, T) {
    T.ok(m.LLAMA3_8B && m.LLAMA3_8B.nLayer === 32 && m.LLAMA3_8B.nKvHeads === 8 && m.LLAMA3_8B.headDim === 128, 'LLAMA3_8B must record 32 layers, 8 KV heads (GQA), head dim 128');
    T.ok(m.LLAMA3_70B && m.LLAMA3_70B.nLayer === 80 && m.LLAMA3_70B.nKvHeads === 8, 'LLAMA3_70B must record 80 layers and 8 KV heads');
    T.close(m.kvCacheBytes(m.LLAMA3_8B, { contextLen: 8192, bits: 16 }), 1073741824, 1e-9, '8B at 8k context in fp16: 2 * 32 * 8 * 128 * 8192 * 2 = 1 GiB (128 KB per token)');
    T.close(m.kvCacheBytes(m.LLAMA3_8B, { contextLen: 8192, bits: 8 }), 536870912, 1e-9, 'int8 or fp8 KV halves it');
    T.close(m.kvCacheBytes(m.LLAMA3_8B, { contextLen: 1, batch: 16, bits: 16 }), 16 * 131072, 1e-9, 'batch multiplies it: 16 sequences x 128 KB per token');
    T.close(m.kvCacheBytes(m.LLAMA3_70B, { contextLen: 8192, bits: 16 }), 2.5 * 1073741824, 1e-9, '70B has 80 layers vs 32 and the same 8 KV heads of dimension 128: exactly 2.5x the 8B cache (without GQA its 64 heads would make it 20x)');
  } },
];
