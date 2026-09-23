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
// int4 symmetric absmax with groups of G consecutive values, independent of the learner's quantizeGroups.
function refInt4Groups(data, G) {
  const out = new Float32Array(data.length);
  for (let b = 0; b < data.length; b += G) {
    let amax = 0;
    for (let i = b; i < b + G; i++) amax = Math.max(amax, Math.abs(data[i]));
    const s = amax > 0 ? amax / 7 : 1;
    for (let i = b; i < b + G; i++) out[i] = Math.max(-8, Math.min(7, Math.round(data[i] / s))) * s;
  }
  return out;
}
const E2M1_GRID = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6];
const onGrid = (v) => E2M1_GRID.includes(v + 0);
const isPow2 = (s) => s > 0 && Number.isInteger(Math.log2(s)) && 2 ** Math.log2(s) === s;
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
  { step: 'groups', name: 'asymmetric 8-bit throws: unsigned codes up to 255 do not fit the Int8Array store', run(m, T) {
    const w = ops.fromArray([[0, 0.1, 0.2, 1]]);
    T.throws(() => m.quantizeGroups(w, { bits: 8, groupSize: 4, symmetric: false }), 'asymmetric 8-bit codes run 0..255, but an Int8Array wraps 200 to -56 and dequantize returns garbage; throw when symmetric is false and bits > 7');
    const ok = m.quantizeGroups(w, { bits: 7, groupSize: 4, symmetric: false });
    T.ok(Array.from(ok.q).every((v) => v >= 0 && v <= 127), `asymmetric 7-bit codes lie in [0, 127] and fit; got ${Array.from(ok.q)}`);
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
    T.close(m.bitsPerParam({ bits: 4, groupSize: 32, scaleBits: 8 }), 4.25, 1e-9, 'MXFP4: 4-bit elements plus one 8-bit power-of-two scale per 32 = 4.25 bits per weight');
    T.close(m.bitsPerParam({ bits: 4, groupSize: 16, scaleBits: 8 }), 4.5, 1e-9, 'NVFP4: 4-bit elements plus one 8-bit E4M3 scale per 16 = 4.5 bits per weight');
  } },
  { step: 'memory', name: 'kvCacheBytes = 2 * layers * kvHeads * headDim * context * batch * bytes', run(m, T) {
    T.ok(m.LLAMA3_8B && m.LLAMA3_8B.nLayer === 32 && m.LLAMA3_8B.nKvHeads === 8 && m.LLAMA3_8B.headDim === 128, 'LLAMA3_8B must record 32 layers, 8 KV heads (GQA), head dim 128');
    T.ok(m.LLAMA3_70B && m.LLAMA3_70B.nLayer === 80 && m.LLAMA3_70B.nKvHeads === 8, 'LLAMA3_70B must record 80 layers and 8 KV heads');
    T.close(m.kvCacheBytes(m.LLAMA3_8B, { contextLen: 8192, bits: 16 }), 1073741824, 1e-9, '8B at 8k context in fp16: 2 * 32 * 8 * 128 * 8192 * 2 = 1 GiB (128 KB per token)');
    T.close(m.kvCacheBytes(m.LLAMA3_8B, { contextLen: 8192, bits: 8 }), 536870912, 1e-9, 'int8 or fp8 KV halves it');
    T.close(m.kvCacheBytes(m.LLAMA3_8B, { contextLen: 1, batch: 16, bits: 16 }), 16 * 131072, 1e-9, 'batch multiplies it: 16 sequences x 128 KB per token');
    T.close(m.kvCacheBytes(m.LLAMA3_70B, { contextLen: 8192, bits: 16 }), 2.5 * 1073741824, 1e-9, '70B has 80 layers vs 32 and the same 8 KV heads of dimension 128: exactly 2.5x the 8B cache (without GQA its 64 heads would make it 20x)');
  } },
  // ---------- step 6: floating-point formats and microscaling ----------
  { step: 'floats', name: 'fpRound rounds to nearest, ties to even, and fp16 overflows at 65520', run(m, T) {
    const { bf16, fp16 } = m.FORMATS;
    T.eq(m.fpRound(1 + 2 ** -8, bf16), 1, 'bf16 has 7 mantissa bits, so 1 + 2^-8 is exactly halfway between 1 and 1 + 2^-7; ties go to the even neighbour, 1 (Math.round would send it up)');
    T.eq(m.fpRound(1 + 3 * 2 ** -8, bf16), 1 + 2 ** -6, '1 + 3·2^-8 is halfway between 1 + 2^-7 (odd last bit) and 1 + 2^-6 (even): it must round to 1 + 2^-6');
    T.eq(m.fpRound(1 + 2 ** -8 + 2 ** -20, bf16), 1 + 2 ** -7, 'just above the halfway point rounds up to 1 + 2^-7');
    T.eq(m.fpRound(-1 - 2 ** -8, bf16), -1, 'rounding is symmetric: work on |x| and restore the sign');
    T.eq(m.fpRound(0.1, fp16), 0.0999755859375, 'fp16 keeps 10 mantissa bits: 0.1 becomes 1638 · 2^-14 = 0.0999755859375');
    T.eq(m.fpRound(65519, fp16), 65504, '65519 is below the halfway point 65520, so it rounds to fp16\'s largest value 65504');
    T.eq(m.fpRound(65520, fp16), Infinity, '65520 is halfway between 65504 and 65536; the tie goes to the even 65536, which does not exist in fp16, so the result overflows to Infinity');
    T.eq(m.fpRound(-1e5, fp16), -Infinity, 'fp16 does not saturate: large negative values overflow to -Infinity');
    const big = m.fpRound(3e38, bf16);
    T.ok(Number.isFinite(big) && Math.abs(big - 3e38) / 3e38 < 2 ** -8, `bf16 keeps fp32's 8-bit exponent, so 3e38 must stay finite and within half a bf16 step (got ${big})`);
    T.eq(m.fpRound(1e5, { exp: 5, man: 10 }), Infinity, 'with bias and max omitted, { exp: 5, man: 10 } must default to fp16 (bias 15, max 65504)');
    T.eq(m.fpRound(0, bf16), 0, 'zero stays zero');
  } },
  { step: 'floats', name: 'fp8 E4M3 saturates at 448 and has subnormals; the FP4 E2M1 grid is exact', run(m, T) {
    const { e4m3, e5m2, e2m1 } = m.FORMATS;
    T.eq(m.fpRound(1000, e4m3), 448, 'E4M3 has no infinities: anything beyond 448 saturates to 448');
    T.eq(m.fpRound(-1e6, e4m3), -448, 'saturation is symmetric: -1e6 becomes -448');
    T.eq(m.fpRound(464, e4m3), 448, '464 is halfway between 448 (mantissa 110, even) and 480 (not representable); ties to even gives 448');
    T.eq(m.fpRound(0.3, e4m3), 0.3125, 'near 0.3 the E4M3 spacing is 2^(-2-3) = 1/32, and the nearest multiple is 10/32 = 0.3125');
    T.eq(m.fpRound(0.001, e4m3), 2 ** -9, 'below 2^-6 E4M3 is subnormal with spacing 2^-9; 0.001 rounds to 2^-9 (keep the exponent at 1 - bias or you will invent finer values)');
    T.eq(m.fpRound(60000, e5m2), 57344, 'E5M2 has 2 mantissa bits: its largest value is 1.75 · 2^15 = 57344');
    T.eq(m.fpRound(61440, e5m2), Infinity, 'E5M2 is IEEE-like: 61440 is the tie between 57344 and 65536 and overflows to Infinity');
    const seen = new Set();
    for (let v = -8; v <= 8; v += 1 / 64) seen.add(m.fpRound(v, e2m1) + 0);
    const got = [...seen].sort((a, b) => a - b);
    T.eq(got, E2M1_GRID, 'sweeping [-8, 8] through E2M1 must produce exactly its 15 values: 0 and ±{0.5, 1, 1.5, 2, 3, 4, 6}');
    T.eq([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7].map((v) => m.fpRound(v, e2m1) + 0), [0, 1, 1, 2, 2, 4, 4, 6], 'E2M1 midpoints: every tie goes to the neighbour with an even mantissa, and 7 saturates to 6');
  } },
  { step: 'floats', name: 'mxQuantize: one power-of-two scale per 32 values, elements on the E2M1 grid', run(m, T) {
    const x = new Float32Array(70);
    for (let i = 0; i < 32; i++) x[i] = (i % 2 ? -1 : 1) * 5 * (i + 1) / 32; // block 0: largest |x| = 5
    for (let i = 32; i < 64; i++) x[i] = 0.3 * (i - 31) / 32; // block 1: largest |x| = 0.3
    x[64] = 7; x[65] = -0.5; // block 2: 6 values, largest 7
    const r = m.mxQuantize(x);
    T.eq(r.block, 32, 'the MX block size defaults to 32');
    T.eq(r.scales.length, 3, '70 values in blocks of 32 make 3 blocks (the last one short)');
    T.ok(Array.from(r.scales).every(isPow2), `every MX scale is a power of two (an E8M0 exponent); got ${Array.from(r.scales)}`);
    T.eq(Array.from(r.scales), [1, 2 ** -4, 1], 'scale = 2^(floor(log2 amax) - 2) as in the OCP MX spec: amax 5 -> 2^(2-2) = 1, amax 0.3 -> 2^(-2-2), amax 7 -> 2^(2-2) = 1 (not rounded up to 2)');
    T.ok(Array.from(r.elems).every(onGrid), 'every element must be an E2M1 value, i.e. fpRound(x / scale, FORMATS.e2m1)');
    T.eq(r.elems[31], -4, 'the block maximum |-5| at scale 1 is a tie between 4 and 6 and rounds to 4 (even mantissa)');
    T.eq(r.elems[63], 4, '0.3 at scale 2^-4 is 4.8, nearest E2M1 value 4');
    T.eq(r.elems[64], 6, 'the floor rule leaves amax / scale anywhere in [4, 8): 7 at scale 1 saturates to E2M1\'s largest value, 6');
    T.eq(m.mxQuantize([100, 1], { block: 2, elem: 'e4m3' }).scales[0], 2 ** -2, 'with E4M3 elements emax = floor(log2 448) = 8, so amax 100 gives scale 2^(6-8)');
  } },
  { step: 'floats', name: 'nvfp4Quantize: E4M3 block scales of 16 under a per-tensor scale; both FP4 formats beat int4 g128 on outliers', run(m, T) {
    const w = gaussian([4096], 31, 0.02).data;
    for (let i = 0; i < w.length; i += 97) w[i] *= 20; // scattered outliers, about one per 97 weights
    let amax = 0;
    for (const v of w) amax = Math.max(amax, Math.abs(v));
    const nv = m.nvfp4Quantize(w);
    T.eq(nv.block, 16, 'NVFP4 blocks hold 16 values');
    T.eq(nv.scales.length, 256, '4096 values in blocks of 16 make 256 block scales');
    T.close(nv.tensorScale, amax / (6 * 448), 1e-6, 'tensorScale = max|x| / (6 · 448), so the block holding the largest value gets scale 448');
    T.ok(Array.from(nv.scales).every((s) => s >= 0 && s <= 448 && m.fpRound(s, m.FORMATS.e4m3) === s), 'every block scale must itself be an E4M3 value in [0, 448]');
    T.eq(Math.max(...nv.scales), 448, 'the block with the tensor maximum must use the top of E4M3\'s range, 448');
    T.ok(Array.from(nv.elems).every(onGrid), 'every NVFP4 element must be an E2M1 value');
    const y = new Float32Array(64);
    y[0] = 2688; y[1] = -1000; // block 0 holds the tensor maximum 6 · 448, so tensorScale = 1
    y[16] = 6; y[17] = -3; y[18] = 0.5; // block 1: amax 6
    y[32] = 7.5; y[33] = -2.5; // block 2: amax 7.5; block 3 stays all zero
    const q = m.nvfp4Quantize(y);
    T.close(q.tensorScale, 1, 1e-9, 'max|x| = 2688 = 6 · 448 gives tensorScale 1');
    T.eq(Array.from(q.scales), [448, 1, 1.25, 0], 'each block scale is amax / 6 / tensorScale rounded to E4M3: 448, 1, 1.25 (exact in E4M3), and 0 for the all-zero block');
    T.eq([q.elems[16], q.elems[17], q.elems[32], q.elems[33]].map((v) => v + 0), [6, -3, 6, -2], 'each block maximum lands on 6, the top of E2M1: 7.5 / 1.25 = 6 exactly, where MXFP4\'s power-of-two scale would saturate it');
    const eNv = mse(w, m.dequantizeBlocks(nv));
    const eMx = mse(w, m.dequantizeBlocks(m.mxQuantize(w)));
    const eG128 = mse(w, refInt4Groups(w, 128));
    T.ok(eMx * 1.5 < eG128, `MXFP4 (4.25 bits) should beat int4 g128 (4.125 bits) by >1.5x when outliers are scattered, since each outlier coarsens only its block of 32 (MXFP4 ${eMx.toExponential(2)}, int4 g128 ${eG128.toExponential(2)})`);
    T.ok(eNv * 2 < eMx, `NVFP4's blocks of 16 with E4M3 scales should give less than half MXFP4's error here (NVFP4 ${eNv.toExponential(2)}, MXFP4 ${eMx.toExponential(2)})`);
  } },
];
