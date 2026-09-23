import * as ops from 'lib/ops.js';
import { rng } from 'lib/util.js';
import { Tensor } from 'lib/tensor.js';
import { attention } from 'lib/attention.js';

// Hardware figures used by the tests, so that a learner who edits the constants still gets
// meaningful failures. These are the same approximate datasheet numbers as in the starter.
const H = { flops: 989e12, bandwidth: 3.35e12 };   // H100 SXM, ridge = 295.2 FLOP/byte
const A = { flops: 312e12, bandwidth: 2.04e12 };   // A100 SXM, ridge = 152.9 FLOP/byte
const SLOW = { flops: 2e12, bandwidth: 0.2e12 };   // server CPU, ridge = 10 FLOP/byte
const B = { flops: 2.25e15, bandwidth: 8e12 };     // B200 (NVIDIA DGX B200, per GPU), ridge = 281.25 FLOP/byte

// The reference: lib/attention.js (the attention module) on the same raw tensors. Returns { shape, data }.
function reference(q, k, v, causal) {
  return attention(new Tensor(q), new Tensor(k), new Tensor(v), { causal }).out;
}

function matrix(n, seed) {
  const next = rng(seed);
  const a = new Float32Array(n * n);
  for (let i = 0; i < a.length; i++) a[i] = next() * 2 - 1;
  return a;
}

export const tests = [
  // ---------- step 1: the roofline ----------
  { step: 'roofline', name: 'arithmeticIntensity is FLOPs per byte and ridgePoint is peak over bandwidth', run(m, T) {
    T.close(m.arithmeticIntensity(2e9, 1e9), 2, 1e-9, 'a kernel doing 2 GFLOP while moving 1 GB has intensity 2 FLOP/byte');
    T.close(m.arithmeticIntensity(1, 4), 0.25, 1e-9, 'intensity is flops / bytes, not bytes / flops');
    T.close(m.ridgePoint(H.flops, H.bandwidth), 295.2239, 1e-4, 'H100: 989e12 FLOP/s / 3.35e12 B/s = 295.2 FLOP/byte');
    T.close(m.ridgePoint(A.flops, A.bandwidth), 152.9412, 1e-4, 'A100: 312e12 / 2.04e12 = 152.9 FLOP/byte');
    T.throws(() => m.arithmeticIntensity(1e9, 0), 'intensity is undefined when bytes is 0; throw instead of returning Infinity');
    T.throws(() => m.arithmeticIntensity(1, -8), 'negative traffic is a bug upstream: `!(bytes > 0)` rejects it where `bytes === 0` does not');
  } },
  { step: 'roofline', name: 'attainable is the LOWER of the two roofs, not one of them', run(m, T) {
    const ridge = 989e12 / 3.35e12;
    T.close(m.attainable(1, H.flops, H.bandwidth), 3.35e12, 1e-6, 'at 1 FLOP/byte you are on the memory roof: bandwidth x intensity = 3.35 TFLOP/s');
    T.close(m.attainable(1e6, H.flops, H.bandwidth), 989e12, 1e-6, 'far above the ridge the compute roof clamps you at peak; intensity x bandwidth would give 3.35e18');
    T.close(m.attainable(ridge, H.flops, H.bandwidth), 989e12, 1e-6, 'at exactly the ridge point the two roofs are equal');
    T.close(m.attainable(ridge / 2, H.flops, H.bandwidth), 989e12 / 2, 1e-6, 'half the ridge intensity buys half of peak FLOP/s');
    T.close(m.attainable(4, SLOW.flops, SLOW.bandwidth), 0.8e12, 1e-6, 'the same intensity gives a different answer on a different device');
  } },
  { step: 'roofline', name: 'the B200 row: more of everything, but almost the same ridge', run(m, T) {
    T.ok(m.HARDWARE.includes(m.B200), 'HARDWARE must list the B200 so the demo plots its roofline');
    T.close(m.ridgePoint(B.flops, B.bandwidth), 281.25, 1e-9, 'B200: approximately 2.25e15 FLOP/s bf16 / 8e12 B/s = 281.25 FLOP/byte, close to the H100 ridge of 295');
    T.close(m.ridgePoint(9e15, B.bandwidth) / m.ridgePoint(B.flops, B.bandwidth), 4, 1e-9,
      'at FP4 (approximately 9 PFLOP/s dense) on the same 8 TB/s the ridge is 4x higher, so even more of inference is memory-bound');
    T.close(m.attainable(1, B.flops, B.bandwidth) / m.attainable(1, H.flops, H.bandwidth), 8 / 3.35, 1e-9,
      'a batch-1 decode matmul on a B200 speeds up only by the bandwidth ratio (about 2.4x), whatever the FLOP/s');
  } },
  { step: 'roofline', name: 'kernelTime takes the max of the two times, never the sum', run(m, T) {
    T.close(m.kernelTime(1e12, 1e9, H.flops, H.bandwidth), 1e12 / 989e12, 1e-9, 'compute-heavy kernel: 1.011 ms of arithmetic hides 0.299 ms of traffic');
    T.close(m.kernelTime(1e9, 1e10, H.flops, H.bandwidth), 1e10 / 3.35e12, 1e-9, 'memory-heavy kernel: the 2.985 ms of traffic sets the time');
    const flops = 7e10, bytes = 5e8;
    const t = m.kernelTime(flops, bytes, H.flops, H.bandwidth);
    T.close(flops / t, m.attainable(m.arithmeticIntensity(flops, bytes), H.flops, H.bandwidth), 1e-6,
      'flops / kernelTime must equal the attainable FLOP/s at that intensity; adding the two times instead of taking the max breaks this');
  } },

  // ---------- step 2: what one matmul costs ----------
  { step: 'opcost', name: 'matmulCost counts 2MKN FLOPs and all three matrices', run(m, T) {
    const tiny = m.matmulCost(2, 3, 4, 1);
    T.eq(tiny.flops, 48, '[2,3]x[3,4] does 2*2*3*4 = 48 FLOPs (one multiply and one add per term)');
    T.eq(tiny.bytes, 26, 'bytes = (M*K + K*N + M*N) * 1 = 6 + 12 + 8 = 26; forgetting the output matrix gives 18');
    const decode = m.matmulCost(1, 4096, 4096, 2);
    T.eq(decode.flops, 33554432, 'decode matmul [1,4096]x[4096,4096] does 2*1*4096*4096 FLOPs');
    T.eq(decode.bytes, 33570816, 'the [4096,4096] weight matrix alone is 33.55 MB in bf16');
    T.close(decode.intensity, 0.99951, 1e-4, 'a batch-1 matmul does about one FLOP per byte read');
    const prefill = m.matmulCost(2048, 4096, 4096, 2);
    T.close(prefill.intensity, 1024, 1e-6, 'with M = 2048 the same weights are reused 2048 times, so intensity is about 1000x higher');
    T.close(m.matmulCost(1, 4096, 4096, 4).intensity, decode.intensity / 2, 1e-6, 'fp32 moves twice the bytes for the same FLOPs, so it halves the intensity');
    T.throws(() => m.matmulCost(0, 4096, 4096), 'a zero dimension is a bug, not a free matmul: throw');
  } },
  { step: 'opcost', name: 'analyseOp puts decode on the memory roof and prefill on the compute roof', run(m, T) {
    const decode = m.analyseOp({ name: 'decode', ...m.matmulCost(1, 4096, 4096, 2) }, H);
    T.eq(decode.bound, 'memory', 'intensity 1.0 is far below the H100 ridge of 295 FLOP/byte');
    T.close(decode.fractionOfPeak, 0.0033856, 1e-4, 'a batch-1 matmul reaches about 0.34% of the H100 bf16 peak');
    T.close(decode.seconds, 33570816 / 3.35e12, 1e-6, 'a memory-bound op takes bytes / bandwidth');
    const prefill = m.analyseOp({ name: 'prefill', ...m.matmulCost(2048, 4096, 4096, 2) }, H);
    T.eq(prefill.bound, 'compute', 'intensity 1024 is above the ridge, so the tensor cores are the limit');
    T.close(prefill.fractionOfPeak, 1, 1e-6, 'above the ridge you reach 100% of the roofline peak in this model');
    T.close(prefill.seconds, 68719476736 / 989e12, 1e-6, 'a compute-bound op takes flops / peak');
  } },
  { step: 'opcost', name: 'the same op changes sides on a different device', run(m, T) {
    const op = { name: 'mid', flops: 50e9, bytes: 1e9 };   // intensity 50 FLOP/byte
    T.eq(m.analyseOp(op, H).bound, 'memory', 'intensity 50 is below the H100 ridge of 295');
    T.eq(m.analyseOp(op, SLOW).bound, 'compute', 'the same intensity 50 is above the CPU ridge of 10: "memory-bound" is a property of the pair, not of the kernel');
    T.close(m.analyseOp(op, H).intensity, 50, 1e-9, 'intensity itself does not depend on the device');
    T.ok(m.analyseOp(op, H).attainedFlops < H.flops, 'a memory-bound op must report less than peak FLOP/s');
    // SLOW's ridge is exactly 10 FLOP/byte (2e12 / 0.2e12), so this op sits exactly on it.
    T.eq(m.analyseOp({ name: 'ridge', flops: 10e9, bytes: 1e9 }, SLOW).bound, 'compute',
      'at exactly the ridge the memory roof has caught up with the compute roof, so the op is (just) compute-bound: use intensity < ridge for "memory"');
  } },

  // ---------- step 3: how much batch buys you ----------
  { step: 'batching', name: 'minBatchForCompute lands exactly on the ridge point', run(m, T) {
    const M = m.minBatchForCompute(H, { K: 4096, N: 4096 });
    T.close(M, 344.949, 1e-3, 'on an H100 a [M,4096]x[4096,4096] bf16 matmul reaches the ridge at M around 345');
    T.close(m.matmulCost(M, 4096, 4096, 2).intensity, m.ridgePoint(H.flops, H.bandwidth), 1e-4,
      'feeding your own answer back into matmulCost must give exactly the ridge intensity; 295 (the weight-stream-only rule of thumb) ignores the activation bytes that matmulCost counts');
    T.close(m.minBatchForCompute(H, { K: 4096, N: 4096, bytesPerElement: 1 }), 159.078, 1e-3, 'int8 weights halve the bytes, so the ridge arrives at less than half the batch');
    T.close(m.minBatchForCompute(A, { K: 4096, N: 4096 }), 165.284, 1e-3, 'the A100 has a lower ridge (152.9), so it needs a smaller batch than the H100');
    T.eq(m.minBatchForCompute({ flops: 1e18, bandwidth: 1e12 }, { K: 4096, N: 4096 }), Infinity,
      'a device with a ridge of 1e6 FLOP/byte can never reach it with 4096-wide weights: return Infinity, not a negative batch');
  } },
  { step: 'batching', name: 'decodeThroughput reports both ceilings and the crossover batch', run(m, T) {
    const one = m.decodeThroughput(H, { params: 8.03e9, batch: 1 });
    T.close(one.memoryBound, 208.593, 1e-3, 'one H100 streams 8.03e9 bf16 weights (16.06 GB) at 3.35 TB/s: about 209 tokens/s for a single stream');
    T.close(one.computeBound, 61581.6, 1e-3, 'the compute ceiling is peak / (2 * params) and does not depend on the batch');
    T.eq(one.bound, 'memory', 'single-stream decoding is memory-bound on every GPU in the table');
    T.close(m.decodeThroughput(H, { params: 8.03e9, batch: 64 }).tokensPerSecond, 64 * 208.593, 1e-3, 'below the crossover, throughput is linear in the batch: 64 streams cost the same weight reads as 1');
    T.eq(m.decodeThroughput(H, { params: 8.03e9, batch: 295 }).bound, 'memory', 'batch 295 is still (just) memory-bound');
    T.eq(m.decodeThroughput(H, { params: 8.03e9, batch: 296 }).bound, 'compute', 'the crossover sits at batch = ridge * bytesPerParam / 2 = 295.2, which is where batching stops being free');
    T.close(m.decodeThroughput(H, { params: 8.03e9, bytesPerParam: 0.5, batch: 1 }).tokensPerSecond, 834.371, 1e-3, 'int4 weights move a quarter of the bytes, so a memory-bound decode runs about 4x faster');
    // On SLOW (ridge 10) with 1e9 bf16 params, batch 10 makes both ceilings exactly 1000 tokens/s.
    const tie = m.decodeThroughput(SLOW, { params: 1e9, batch: 10 });
    T.close(tie.memoryBound, tie.computeBound, 1e-12, 'at batch = ridge * bytesPerParam / 2 the two ceilings are equal (1000 tokens/s each here)');
    T.eq(tie.bound, 'memory', 'when the two ceilings are exactly equal, report the memory roof, as the instructions say: it is the one batching can still move');
    T.throws(() => m.decodeThroughput(H, { params: 8.03e9, batch: 0 }), 'a batch of 0 generates no tokens; throw rather than return 0 or Infinity');
  } },

  // ---------- step 4: raising intensity by tiling ----------
  { step: 'tiling', name: 'tiledMatmul matches the reference for tile sizes that do not divide n', run(m, T) {
    for (const [n, bs] of [[13, 4], [32, 8], [64, 16], [17, 5]]) {
      const A2 = matrix(n, n + 1), B2 = matrix(n, n + 2);
      const got = m.tiledMatmul(A2, B2, n, bs);
      const want = ops.matmul({ shape: [n, n], data: A2 }, { shape: [n, n], data: B2 }).data;
      T.eq(got.length, n * n, `tiledMatmul must return n*n values for n=${n}`);
      T.close(Array.from(got), Array.from(want), 1e-3, `n=${n}, blockSize=${bs}: edge tiles are partial when blockSize does not divide n (Math.min against n)`);
    }
  } },
  { step: 'tiling', name: 'tiledMatmul equals naiveMatmul for any block size and rejects blockSize 0', run(m, T) {
    const n = 48, A2 = matrix(n, 3), B2 = matrix(n, 4);
    const want = Array.from(m.naiveMatmul(A2, B2, n));
    for (const bs of [1, 7, 16, 48, 100]) {
      T.close(Array.from(m.tiledMatmul(A2, B2, n, bs)), want, 1e-3, `blockSize ${bs} must give the same answer as the naive loop; a block larger than n is just one tile`);
    }
    T.throws(() => m.tiledMatmul(A2, B2, n, 0), 'blockSize 0 would loop forever: throw instead');
  } },
  { step: 'tiling', name: 'tiledMatmul actually walks the matrices tile by tile', run(m, T) {
    // Record every element index tiledMatmul reads from A and B. A blocked kernel revisits a few
    // small tiles; the naive i,j,k loop sweeps a whole column of B for every output element.
    const n = 16, bs = 4, seen = [];
    const P = new Proxy(new Array(n * n).fill(1), {
      get(t, k) { if (typeof k === 'string' && /^\d+$/.test(k)) seen.push(+k); return t[k]; },
    });
    const C = m.tiledMatmul(P, P, n, bs);
    T.eq(C[0], n, 'an all-ones 16x16 product has 16 in every entry');
    T.ok(seen.length >= 2 * n ** 3, `a matmul must read A and B once per multiply-add (expected ${2 * n ** 3} reads, got ${seen.length})`);
    let worst = 0;
    for (let i = 0; i + 512 <= seen.length; i += 512) worst = Math.max(worst, new Set(seen.slice(i, i + 512)).size);
    T.ok(worst <= 200, `in any 512 consecutive reads a 4x4-blocked kernel touches only a few tiles' worth of elements (about 112); got ${worst}. A value near 256 means the loops are not tiled at all, so nothing is reused from fast memory`);
  } },
  { step: 'tiling', name: 'blockTraffic falls like 1/blockSize and reproduces the naive count at blockSize 1', run(m, T) {
    T.close(m.blockTraffic(256, 1, 4), (2 * 256 ** 3 + 256 * 256) * 4, 1e-9, 'blockSize 1 is the naive matmul: 2n^3 element reads plus one write of C');
    T.close(m.blockTraffic(256, 32, 4), 4456448, 1e-9, 'with 32x32 tiles: (2*256^3/32 + 256^2) * 4 bytes');
    T.close(m.blockTraffic(256, 1, 4) / m.blockTraffic(256, 32, 4), 30.1765, 1e-4, '32x32 tiles cut slow-memory traffic by about 30x');
    const intensity = m.arithmeticIntensity(2 * 256 ** 3, m.blockTraffic(256, 32, 4));
    T.close(intensity, 7.5294, 1e-4, 'arithmetic intensity of a tiled fp32 matmul is about blockSize / bytesPerElement = 32/4 = 8');
    T.ok(m.blockTraffic(512, 64, 2) < m.blockTraffic(512, 32, 2), 'doubling the tile size must roughly halve the traffic');
    T.throws(() => m.blockTraffic(256, 0, 4), 'blockTraffic(256, 0, 4) must throw: the formula divides by blockSize, and a block of 0 describes no kernel (check blockSize >= 1 here as well as in tiledMatmul)');
  } },

  // ---------- step 5: the memory hierarchy and FlashAttention ----------
  { step: 'hierarchy', name: 'attentionCost separates the quadratic score traffic from the linear stream', run(m, T) {
    const naive = m.attentionCost(1024, 64);
    T.eq(naive.flops, 268435456, 'one head over 1024 tokens with headDim 64 does 4*1024^2*64 FLOPs in the two matmuls');
    T.eq(naive.bytes, 8912896, 'Q, K, V and O move once (4*1024*64) and the [1024,1024] score matrix moves four times, all in bf16');
    T.close(naive.intensity, 30.1176, 1e-4, 'naive attention sits at about 30 FLOP/byte, well below the H100 ridge of 295');
    const flash = m.attentionCost(1024, 64, { flash: true });
    T.eq(flash.bytes, 524288, 'streaming attention never writes the score matrix to HBM: only Q, K, V and O move');
    T.close(flash.intensity, 512, 1e-6, 'the same FLOPs over 17x fewer bytes puts attention above the ridge');
    T.eq(flash.flops, naive.flops, 'FlashAttention does not remove a single FLOP; it removes memory traffic');
  } },
  { step: 'hierarchy', name: 'naive attention traffic grows quadratically, streaming traffic linearly', run(m, T) {
    const n1 = m.attentionCost(1024, 128).bytes, n2 = m.attentionCost(2048, 128).bytes;
    const f1 = m.attentionCost(1024, 128, { flash: true }).bytes, f2 = m.attentionCost(2048, 128, { flash: true }).bytes;
    T.ok(n2 / n1 > 3.4, `doubling the sequence must roughly quadruple naive traffic, got ${(n2 / n1).toFixed(2)}x`);
    T.close(f2 / f1, 2, 1e-6, 'streamed traffic is linear in the sequence length: exactly 2x');
    T.close(m.attentionCost(4096, 128).bytes / m.attentionCost(4096, 128, { flash: true }).bytes, 33, 1e-6,
      'at 4096 tokens and headDim 128 the score matrix is 33x the streamed tensors');
  } },
  { step: 'hierarchy', name: 'flashBlockSize fits four tiles in SRAM', run(m, T) {
    T.eq(m.flashBlockSize(228 * 1024, 128, 2), 228, 'four tiles of 128 bf16 values per row need 1024 bytes per row; 228 KiB of shared memory (228 * 1024 = 233,472 bytes) holds 228 rows');
    T.eq(m.flashBlockSize(228 * 1024, 64, 2), 456, 'halving headDim doubles the block that fits');
    T.eq(m.flashBlockSize(228 * 1024, 128, 1), 456, 'an int8 KV cache doubles the block that fits, which is a second reason to quantise it');
    T.eq(m.flashBlockSize(100 * 1024, 96, 2), 133, 'a block is a whole number of rows: 102400 bytes / (4 * 96 * 2) = 133.3 must floor to 133, not round up and overflow SRAM');
    T.eq(m.flashBlockSize(1000, 128, 2), 1, 'when nothing fits, return 1 rather than 0: a zero block size would divide by zero downstream');
  } },
  { step: 'hierarchy', name: 'tiledAttention equals lib/attention.js at T = 64 for every block size', run(m, T) {
    const n = 64, dh = 16;
    const q = ops.randn([n, dh], T.rng(11), 1.5), k = ops.randn([n, dh], T.rng(12), 1.5), v = ops.randn([n, dh], T.rng(13));
    for (const causal of [true, false]) {
      const want = reference(q, k, v, causal);
      for (const bs of [1, 5, 16, 64, 100]) {
        const got = m.tiledAttention(q, k, v, bs, { causal });
        T.shape(got, [n, dh], 'tiledAttention returns one output row of dh values per query');
        T.close(got, want, 1e-5, `causal=${causal}, blockSize=${bs}: the online softmax must give exactly softmax(q·kᵀ/sqrt(dh))·v; a block of 1, a block that does not divide T and a block larger than T are all the same maths`);
      }
    }
    // An empty sequence, so that a kernel that forgets the check fails here instead of looping forever.
    const empty = { shape: [0, dh], data: new Float32Array(0) };
    T.throws(() => m.tiledAttention(empty, empty, empty, 0), 'blockSize 0 would never advance through the keys: check it before any loop and throw');
  } },
  { step: 'hierarchy', name: 'tiledAttention rescales by exp(mOld - mNew) when a later block raises the max', run(m, T) {
    // Scores rise by 50 per key: key 15 scores 750, where exp(750) overflows to Infinity. The early
    // blocks are summed against a max that later blocks beat, so their l and acc must be rescaled.
    const n = 16, dh = 4;
    const q = { shape: [n, dh], data: new Float32Array(n * dh) };
    const k = { shape: [n, dh], data: new Float32Array(n * dh) };
    const v = ops.randn([n, dh], T.rng(21));
    for (let i = 0; i < n; i++) q.data[i * dh] = 10;
    for (let j = 0; j < n; j++) { k.data[j * dh] = 10 * j; k.data[j * dh + 1] = Math.sin(j); }
    for (const causal of [true, false]) {
      const got = m.tiledAttention(q, k, v, 4, { causal });
      T.ok(Array.from(got.data).every(Number.isFinite), `causal=${causal}: scores of 750 overflow exp(); subtract the running max m before exponentiating, never the raw score`);
      T.close(got, reference(q, k, v, causal), 1e-5, `causal=${causal}, blockSize 4: when a new block raises the max, multiply the old l and acc by exp(mOld - mNew) before adding the block, or the early blocks are weighted against the wrong max`);
    }
    // Softer scores, so a missing rescale shows up as a wrong answer rather than an overflow.
    for (let j = 0; j < n; j++) k.data[j * dh] = 0.3 * j;
    T.close(m.tiledAttention(q, k, v, 3, { causal: false }), reference(q, k, v, false), 1e-5,
      'scores rising from 0 to 22.5 across blocks of 3: every block raises the max, so every block needs the exp(mOld - mNew) correction');
  } },
  { step: 'hierarchy', name: 'tiledAttention streams K and V block by block instead of scoring every key first', run(m, T) {
    // One query row, 32 keys, blocks of 8. Record which rows of K and V are read, in order.
    const n = 32, dh = 4, bs = 8, reads = [];
    const watch = (raw, name) => ({ shape: raw.shape, data: new Proxy(Array.from(raw.data), {
      get(t, key) { if (typeof key === 'string' && /^\d+$/.test(key)) reads.push([name, Math.floor(+key / dh)]); return t[key]; },
    }) });
    const q = ops.randn([1, dh], T.rng(31)), k = ops.randn([n, dh], T.rng(32)), v = ops.randn([n, dh], T.rng(33));
    const got = m.tiledAttention(q, watch(k, 'k'), watch(v, 'v'), bs, { causal: false });
    T.close(got, reference(q, k, v, false), 1e-5, 'reading the data by index (data[j * dh + c]) must still give the reference answer');
    const firstV = reads.findIndex(([name]) => name === 'v');
    T.ok(firstV >= 0, 'tiledAttention must read V');
    const kBefore = reads.slice(0, firstV).filter(([name]) => name === 'k').map(([, row]) => row);
    T.ok(kBefore.length > 0 && Math.max(...kBefore) < bs,
      `before touching V a streaming kernel has scored only the first block of ${bs} keys; got K rows up to ${Math.max(-1, ...kBefore)}. Scoring all ${n} keys first is the naive kernel, whose [T, T] scores are what attentionCost({ flash: false }) charges to HBM`);
    const vRows = new Set(reads.filter(([name]) => name === 'v').map(([, row]) => row));
    T.eq(vRows.size, n, 'every value row contributes once the non-causal query has seen all its keys');
  } },
];
