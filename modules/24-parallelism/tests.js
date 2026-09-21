// Module 24 — tests. Everything here is arithmetic on plain numbers, so every expectation is a
// value you can check by hand. Where a number is not obvious, the comment shows the calculation.

// A link with round numbers so the expected values are exact: 1 ms latency, 1 MB/s bandwidth.
const ROUND = { name: 'round-numbers link', latency: 1e-3, bandwidth: 1e6 };

// A small model used for the degenerate single-GPU plan.
const TINY = { name: 'tiny', params: 1e9, layers: 8, dModel: 512, dFF: 2048, seqLen: 128, batchSeqs: 8 };

export const tests = [
  // ---------- step 1: the ring all-reduce ----------
  {
    step: 'ring',
    name: 'ring all-reduce moves 2(n-1)/n bytes per GPU, never 2x and never 0',
    run(m, T) {
      T.close(m.ringAllReduceBytes(1e9, 1), 0, 1e-9, 'one GPU has nobody to reduce with, so it sends nothing');
      T.close(m.ringAllReduceBytes(1e9, 2), 1e9, 1e-6, 'n=2: 2(2-1)/2 = 1, so each GPU sends the whole buffer once');
      T.close(m.ringAllReduceBytes(1e9, 4), 1.5e9, 1e-6, 'n=4: 2(4-1)/4 = 1.5');
      T.close(m.ringAllReduceBytes(1e9, 8), 1.75e9, 1e-6, 'n=8: 2(8-1)/8 = 1.75');
      T.ok(m.ringAllReduceBytes(1e9, 1024) < 2e9, 'the factor 2(n-1)/n approaches 2 from below; it never reaches or exceeds 2');
      T.throws(() => m.ringAllReduceBytes(1e9, 0), 'n = 0 is not a ring; the function must throw rather than return Infinity');
    },
  },
  {
    step: 'ring',
    name: 'ring all-reduce time is 2(n-1) steps of latency + (bytes/n)/bandwidth',
    run(m, T) {
      T.close(m.ringAllReduceTime(4e6, 1, ROUND), 0, 1e-12, 'a single GPU does no communication at all');
      // n = 4, 4 MB buffer: chunk = 1 MB, one step = 1e-3 + 1 s, and there are 2(4-1) = 6 steps.
      T.close(m.ringAllReduceTime(4e6, 4, ROUND), 6.006, 1e-6,
        'expected 6 steps x (1 ms latency + 1 MB / 1 MB/s) = 6.006 s; sending the WHOLE buffer each step (24.006) or doing only n-1 steps (3.003) is wrong');
      T.close(m.ringAllReduceTime(2e6, 2, ROUND), 2.002, 1e-6, 'n=2: 2 steps x (1 ms + 1 MB / 1 MB/s)');
    },
  },
  {
    step: 'ring',
    name: 'small messages are latency-bound and large ones bandwidth-bound',
    run(m, T) {
      const small = m.ringAllReduceTime(1024, 64, m.INFINIBAND);
      T.close(small, 2 * 63 * m.INFINIBAND.latency, 0.01,
        '1 KB across 64 GPUs is pure latency: 2(n-1) x 15 us = 1.89 ms. If you dropped the latency term you get ~4e-8 s instead');
      const big = m.ringAllReduceTime(16e9, 64, m.INFINIBAND);
      T.close(big, m.ringAllReduceBytes(16e9, 64) / m.INFINIBAND.bandwidth, 0.01,
        '16 GB across 64 GPUs is bandwidth-bound: the time must match bytes-moved / bandwidth to within 1%');
      T.ok(small < big / 100, 'the same collective on 1 KB must be orders of magnitude cheaper than on 16 GB');
    },
  },
  {
    step: 'ring',
    name: 'time matches the closed form on random buffer sizes and grows with n',
    run(m, T) {
      const next = T.rng(24);
      for (let i = 0; i < 40; i++) {
        const bytes = Math.floor(next() * 4e9) + 1;
        const n = 2 + Math.floor(next() * 30);
        const expect = 2 * (n - 1) * (m.NVLINK.latency + bytes / n / m.NVLINK.bandwidth);
        T.close(m.ringAllReduceTime(bytes, n, m.NVLINK), expect, 1e-6,
          `ringAllReduceTime(${bytes}, ${n}, NVLINK) must equal 2(n-1)(latency + bytes/n/bandwidth)`);
      }
      T.ok(m.ringAllReduceTime(1e9, 16, m.NVLINK) > m.ringAllReduceTime(1e9, 4, m.NVLINK),
        'more ranks means more sequential steps, so the same buffer takes longer');
    },
  },

  // ---------- step 2: data parallelism ----------
  {
    step: 'dp',
    name: 'without overlap the step is compute plus the whole gradient all-reduce',
    run(m, T) {
      const cfg = { params: 8e9, tokensPerGpu: 16384, dp: 8, gpu: m.H100, link: m.INFINIBAND, overlap: 0 };
      const r = m.dataParallelStep(cfg);
      T.close(r.compute, m.computeTime(8e9, 16384, m.H100), 1e-6, 'compute must be the module-08 6ND time on one GPU');
      T.close(r.comm, m.ringAllReduceTime(2 * 8e9, 8, m.INFINIBAND), 1e-6,
        'gradients are bf16: 2 bytes per parameter, all-reduced over the dp replicas');
      T.close(r.exposed, r.comm, 1e-6, 'with overlap = 0 nothing is hidden');
      T.close(r.step, r.compute + r.comm, 1e-6, 'step = compute + exposed comm');
    },
  },
  {
    step: 'dp',
    name: 'overlap hides communication but never makes the step shorter than compute',
    run(m, T) {
      const base = { params: 8e9, tokensPerGpu: 16384, dp: 8, gpu: m.H100, link: m.INFINIBAND };
      const full = m.dataParallelStep({ ...base, overlap: 1 });
      T.close(full.exposed, 0, 1e-9, 'with overlap = 1 and comm < compute the all-reduce is completely hidden');
      T.close(full.step, full.compute, 1e-9, 'a fully hidden all-reduce costs nothing: the step is pure compute');
      const half = m.dataParallelStep({ ...base, overlap: 0.5 });
      const expectExposed = Math.max(0, half.comm - 0.5 * half.compute);
      T.close(half.exposed, expectExposed, 1e-9, 'exposed comm = comm - min(comm, overlap x compute); it is clamped at zero, never negative');
      T.ok(half.step >= half.compute - 1e-9, 'overlap can hide comm, it cannot make arithmetic faster');
      const tiny = m.dataParallelStep({ params: 8e9, tokensPerGpu: 128, dp: 1024, gpu: m.H100, link: m.INFINIBAND, overlap: 0.6 });
      T.ok(tiny.exposed > 0, 'with only 128 tokens per GPU there is not enough compute to hide a 16 GB all-reduce behind');
    },
  },
  {
    step: 'dp',
    name: 'efficiency is 1 when comm is hidden and falls as replicas are added',
    run(m, T) {
      const cfg = { params: 8e9, tokensPerGpu: 8 * 8192, gpu: m.H100, link: m.INFINIBAND, overlap: 0.6 };
      T.close(m.dpEfficiency(cfg, 1), 1, 1e-9, 'one replica does no all-reduce, so efficiency is exactly 1');
      T.close(m.dpEfficiency(cfg, 8), 1, 1e-9, 'with 8 sequences per GPU the all-reduce fits inside the compute');
      const small = { ...cfg, tokensPerGpu: 8192, overlap: 0 };
      const e8 = m.dpEfficiency(small, 8), e1024 = m.dpEfficiency(small, 1024);
      T.ok(e1024 < e8, `efficiency must fall as dp grows (got ${e8.toFixed(3)} at dp=8 and ${e1024.toFixed(3)} at dp=1024); returning a constant 1 hides the whole point`);
      T.ok(e1024 > 0 && e8 <= 1 + 1e-9, 'efficiency is a fraction in (0, 1]');
    },
  },

  // ---------- step 3: tensor parallelism ----------
  {
    step: 'tp',
    name: 'Megatron splits the first matrix of each pair by column and the second by row',
    run(m, T) {
      const cfg = { dModel: 512, dFF: 2048 };
      const whole = m.tpShards(cfg, 1);
      T.eq(whole.map((s) => s.name), ['attn.qkv', 'attn.proj', 'mlp.fc', 'mlp.proj'], 'return the four big matrices of one layer, in this order');
      T.eq(whole.map((s) => s.split), ['column', 'row', 'column', 'row'], 'qkv and mlp.fc are column-parallel; the projections that follow them are row-parallel');
      T.eq(whole.map((s) => s.shape), [[512, 1536], [512, 512], [512, 2048], [2048, 512]], 'tp = 1 must give the unsharded shapes');
      const four = m.tpShards(cfg, 4);
      T.eq(four.map((s) => s.shape), [[512, 384], [128, 512], [512, 512], [512, 512]],
        'column-parallel shards divide the OUTPUT dimension (columns); row-parallel shards divide the INPUT dimension (rows)');
    },
  },
  {
    step: 'tp',
    name: 'the shards partition the layer exactly and reject indivisible shapes',
    run(m, T) {
      const cfg = { dModel: 512, dFF: 2048 };
      const params = (shards) => shards.reduce((s, x) => s + x.shape[0] * x.shape[1], 0);
      const whole = params(m.tpShards(cfg, 1));
      for (const tp of [2, 4, 8]) {
        T.close(params(m.tpShards(cfg, tp)), whole / tp, 1e-9,
          `with tp=${tp} each GPU must hold exactly 1/${tp} of the layer's ${whole} weights — no replication, no loss`);
      }
      T.throws(() => m.tpShards({ dModel: 512, dFF: 2000 }, 3), 'dFF = 2000 is not divisible by tp = 3, so the split is impossible and must throw');
    },
  },
  {
    step: 'tp',
    name: 'a tensor-parallel layer costs four all-reduces of the full activation',
    run(m, T) {
      const none = m.tpLayerComm({ dModel: 4096, tokens: 2048 }, 1, m.NVLINK);
      T.eq(none.allReduces, 0, 'tp = 1 keeps the whole layer on one GPU: zero all-reduces');
      T.close(none.time, 0, 1e-12, 'zero all-reduces cost zero seconds');
      const c = m.tpLayerComm({ dModel: 4096, tokens: 2048 }, 8, m.NVLINK);
      T.eq(c.allReduces, 4, 'two in the forward pass (after attn.proj and after mlp.proj) and two in the backward pass');
      T.close(c.bytesPerAllReduce, 2 * 2048 * 4096, 1e-6,
        'each all-reduce carries the whole bf16 [tokens, dModel] activation: 2 x 2048 x 4096 bytes. It does NOT shrink with tp');
      T.close(c.time, 4 * m.ringAllReduceTime(2 * 2048 * 4096, 8, m.NVLINK), 1e-9, 'time is the four all-reduces, each priced by your ring model');
      const c2 = m.tpLayerComm({ dModel: 4096, tokens: 2048 }, 2, m.NVLINK);
      T.close(c2.bytesPerAllReduce, c.bytesPerAllReduce, 1e-6, 'the bytes per all-reduce are the same at tp=2 and tp=8; only the number of ring steps changes');
    },
  },

  // ---------- step 4: pipeline parallelism ----------
  {
    step: 'pp',
    name: 'the bubble is (p-1)/m, not (p-1)/p',
    run(m, T) {
      T.close(m.pipelineBubble(1, 16), 0, 1e-12, 'one stage is never idle: no bubble');
      T.close(m.pipelineBubble(8, 8), 0.875, 1e-9, '(8-1)/8 = 0.875 — with as many micro-batches as stages, most of the step is idle');
      T.close(m.pipelineBubble(16, 64), 0.234375, 1e-9, '(16-1)/64 = 0.234375: more micro-batches amortise the fill and drain');
      T.ok(m.pipelineBubble(16, 1024) < m.pipelineBubble(16, 64), 'the bubble must shrink as micro-batches are added; a value that ignores m is wrong');
      T.throws(() => m.pipelineBubble(4, 0), 'zero micro-batches is not a pipeline step and must throw, not divide by zero');
    },
  },
  {
    step: 'pp',
    name: '1F1B keeps min(p, m) micro-batches of activations, GPipe keeps all m',
    run(m, T) {
      const r = m.pipelineStep({ modelTime: 80, p: 8, m: 64, activationBytesPerMicroBatch: 1e9 });
      T.close(r.stageTime, 10, 1e-9, '8 stages share 80 s of model time, so one stage does 10 s of work');
      T.close(r.bubble, 7 / 64, 1e-9, '(p-1)/m with p=8, m=64');
      T.close(r.step, 10 * (1 + 7 / 64), 1e-9, 'the step is one stage\'s work stretched by the bubble: stageTime x (1 + bubble)');
      T.close(r.activationBytes1F1B, 8e9, 1e-6, '1F1B has at most min(8, 64) = 8 micro-batches in flight per stage');
      T.close(r.activationBytesGPipe, 64e9, 1e-6, 'GPipe runs all 64 forwards before any backward, so it holds 64 micro-batches');
      T.ok(r.activationBytes1F1B < r.activationBytesGPipe, '1F1B exists precisely because it holds less activation memory than GPipe');
    },
  },
  {
    step: 'pp',
    name: 'a single stage has no bubble and the two schedules agree when m <= p',
    run(m, T) {
      const one = m.pipelineStep({ modelTime: 42, p: 1, m: 10, activationBytesPerMicroBatch: 1e6 });
      T.close(one.step, 42, 1e-9, 'with one stage the step is the whole model time, unstretched');
      T.close(one.activationBytes1F1B, 1e6, 1e-6, 'one stage runs one micro-batch at a time: min(1, 10) = 1');
      const few = m.pipelineStep({ modelTime: 40, p: 8, m: 4, activationBytesPerMicroBatch: 1e6 });
      T.close(few.activationBytes1F1B, few.activationBytesGPipe, 1e-6, 'with fewer micro-batches than stages there is nothing for 1F1B to save');
      T.close(few.step, 5 * (1 + 7 / 4), 1e-9, 'p=8, m=4: the bubble is 1.75, so the step is nearly three times a stage\'s work');
    },
  },

  // ---------- step 5: ZeRO / FSDP memory ----------
  {
    step: 'zero',
    name: 'each ZeRO stage shards one more slice of the 16 bytes per parameter',
    run(m, T) {
      const N = 1e9, dp = 8;
      const s0 = m.zeroMemoryPerGpu(N, dp, 0);
      T.close([s0.params, s0.grads, s0.optimizer, s0.total], [2e9, 2e9, 12e9, 16e9], 1e-6,
        'stage 0 replicates everything: 2 bytes of bf16 weights + 2 of bf16 grads + 12 of fp32 AdamW state');
      const s1 = m.zeroMemoryPerGpu(N, dp, 1);
      T.close([s1.params, s1.grads, s1.optimizer], [2e9, 2e9, 1.5e9], 1e-6, 'stage 1 shards only the 12-byte optimizer state across dp=8');
      const s2 = m.zeroMemoryPerGpu(N, dp, 2);
      T.close([s2.params, s2.grads, s2.optimizer], [2e9, 0.25e9, 1.5e9], 1e-6, 'stage 2 also shards the gradients');
      const s3 = m.zeroMemoryPerGpu(N, dp, 3);
      T.close(s3.total, 16e9 / dp, 1e-6, 'stage 3 (FSDP) shards the parameters too, so the whole 16 bytes/param is divided by dp');
    },
  },
  {
    step: 'zero',
    name: 'stages are monotone, collapse at dp = 1, and reject unknown stages',
    run(m, T) {
      const totals = [0, 1, 2, 3].map((s) => m.zeroMemoryPerGpu(70e9, 64, s).total);
      for (let i = 1; i < totals.length; i++) {
        T.ok(totals[i] < totals[i - 1], `stage ${i} must use strictly less memory per GPU than stage ${i - 1} (got ${totals.map((t) => (t / 1e9).toFixed(1)).join(', ')} GB)`);
      }
      T.close(totals, [1120e9, 293.125e9, 155.3125e9, 17.5e9], 1e-6, 'Llama-3-70B over 64 ranks: 1120, 293, 155 and 17.5 GB per GPU for stages 0-3');
      for (const s of [0, 1, 2, 3]) {
        T.close(m.zeroMemoryPerGpu(1e9, 1, s).total, 16e9, 1e-6, 'with a single data-parallel rank there is nothing to shard: every stage costs 16 bytes/param');
      }
      T.throws(() => m.zeroMemoryPerGpu(1e9, 8, 4), 'there is no ZeRO stage 4; an unknown stage must throw rather than silently return stage 3');
    },
  },

  // ---------- step 6: the planner ----------
  {
    step: 'planner',
    name: 'a one-GPU plan is pure compute with no communication anywhere',
    run(m, T) {
      const p = m.plan({ model: TINY, gpus: 1, strategy: { dp: 1, tp: 1, pp: 1, zero: 0 } });
      T.eq(p.m, 8, 'with dp=1 and one sequence per micro-batch, all 8 sequences of the batch are 8 micro-batches');
      T.close(p.stepTime, m.computeTime(TINY.params, TINY.batchSeqs * TINY.seqLen, m.H100), 1e-9,
        'on one GPU the step is exactly the 6ND compute time of the whole batch');
      T.close([p.breakdown.tpComm, p.breakdown.ppComm, p.breakdown.dpComm, p.breakdown.bubble], [0, 0, 0, 0], 1e-9,
        'one GPU means no tensor, pipeline or data-parallel traffic and no pipeline bubble');
      T.close(p.memory.optimizer, 12 * TINY.params, 1e-6, 'zero stage 0 on one rank keeps the full 12 bytes/param of optimizer state');
      T.close(p.tokensPerSec, (TINY.batchSeqs * TINY.seqLen) / p.stepTime, 1e-6, 'tokensPerSec is the global batch divided by the step time');
    },
  },
  {
    step: 'planner',
    name: 'illegal layouts are rejected with an explanation',
    run(m, T) {
      T.throws(() => m.plan({ model: m.LLAMA3_70B, gpus: 64, strategy: { dp: 8, tp: 4, pp: 1 } }), 'dp x tp x pp = 32 does not use all 64 GPUs and must throw');
      T.throws(() => m.plan({ model: m.LLAMA3_70B, gpus: 64, strategy: { dp: 4, tp: 16, pp: 1 } }), 'tp = 16 spans two nodes; tensor parallelism must stay on the intra-node link');
      T.throws(() => m.plan({ model: m.LLAMA3_70B, gpus: 64, strategy: { dp: 2, tp: 1, pp: 32 } }), '80 layers do not divide into 32 equal stages and must throw');
      T.throws(() => m.plan({ model: m.LLAMA3_8B, gpus: 256, strategy: { dp: 256, tp: 1, pp: 1 } }), 'a 128-sequence batch cannot be split across 256 data-parallel replicas and must throw');
      const ok = m.plan({ model: m.LLAMA3_70B, gpus: 64, strategy: { dp: 4, tp: 4, pp: 4 } });
      T.ok(ok.stepTime > 0 && ok.m === 128, 'dp=4, tp=4, pp=4 is legal on 64 GPUs and gives 512/4 = 128 micro-batches per pipeline');
    },
  },
  {
    step: 'planner',
    name: 'tensor parallelism buys the memory that pure data parallelism cannot fit',
    run(m, T) {
      const pure = m.plan({ model: m.LLAMA3_70B, gpus: 64, strategy: { dp: 64, tp: 1, pp: 1, zero: 3 } });
      const shard = m.plan({ model: m.LLAMA3_70B, gpus: 64, strategy: { dp: 16, tp: 4, pp: 1, zero: 3 } });
      T.ok(!pure.fits, `64-way data parallelism needs ${(pure.memoryPerGpu / 1e9).toFixed(0)} GB per GPU, far past the H100's 80 GB — activations are what blow up`);
      T.ok(shard.fits, `tp=4 quarters the activation memory and must fit in 80 GB (got ${(shard.memoryPerGpu / 1e9).toFixed(1)} GB)`);
      T.ok(pure.stepTime < shard.stepTime, 'pure data parallelism is the FASTEST layout; tensor parallelism is the tax you pay to fit at all');
      T.close(shard.breakdown.compute, pure.breakdown.compute, 1e-9, 'both layouts do the same arithmetic on the same 64 GPUs, so compute is identical');
      T.ok(shard.breakdown.tpComm > 0.05 * shard.breakdown.compute,
        'tensor-parallel all-reduces are not free: they should add several percent to a 70B step, not zero');
    },
  },
  {
    step: 'planner',
    name: 'bestPlan returns the fastest layout that fits, and only legal layouts are considered',
    run(m, T) {
      const all = m.enumerateStrategies(m.LLAMA3_70B, 64);
      T.ok(all.length > 10, `expected many candidate layouts for 64 GPUs, got ${all.length}`);
      for (const s of all) {
        T.eq(s.dp * s.tp * s.pp, 64, `layout dp=${s.dp}, tp=${s.tp}, pp=${s.pp} does not use exactly 64 GPUs`);
        T.ok(s.tp <= 8, `layout with tp=${s.tp} would span nodes; enumerateStrategies must not propose it`);
        T.ok(m.LLAMA3_70B.layers % s.pp === 0, `pp=${s.pp} does not divide 80 layers evenly`);
      }
      const best = m.bestPlan({ model: m.LLAMA3_70B, gpus: 64 });
      T.ok(best && best.fits, 'a 70B model does fit on 64 H100s under some layout; bestPlan must find one');
      for (const s of all) {
        const p = m.plan({ model: m.LLAMA3_70B, gpus: 64, strategy: s });
        if (p.memoryPerGpu <= m.H100.memory) {
          T.ok(best.stepTime <= p.stepTime + 1e-9,
            `bestPlan chose ${best.stepTime.toFixed(2)} s but dp=${s.dp}, tp=${s.tp}, pp=${s.pp}, zero=${s.zero} fits and takes ${p.stepTime.toFixed(2)} s`);
        }
      }
      const capped = m.bestPlan({ model: m.LLAMA3_70B, gpus: 64, memoryCap: 1e9 });
      T.eq(capped, null, 'with a 1 GB budget nothing fits and bestPlan must return null rather than an over-budget layout');
    },
  },
];
