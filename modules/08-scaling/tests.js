// Scaling laws & the arithmetic of compute — tests. All numbers are computed independently here so a wrong constant (2ND, 4ND,
// forgetting MFU, 8 bytes/param) is caught with a message that says which multiplication went wrong.

const PAPER = { E: 1.69, A: 406.4, B: 410.7, alpha: 0.34, beta: 0.28 };
const REFIT = { E: 1.8172, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658 };
const lossOf = (N, D, f) => f.E + f.A / Math.pow(N, f.alpha) + f.B / Math.pow(D, f.beta);
// Closed-form optimum of E + A/N^α + B/D^β subject to 6ND = C (Hoffmann et al. 2022, Appendix D).
function analyticOptimum(C, f) {
  const G = Math.pow((f.alpha * f.A) / (f.beta * f.B), 1 / (f.alpha + f.beta));
  const N = G * Math.pow(C / 6, f.beta / (f.alpha + f.beta));
  return { N, D: C / (6 * N) };
}

export const tests = [
  // ---------- step 1 ----------
  { step: 'flops', name: 'trainingFlops is exactly 6·N·D (2 forward + 4 backward per parameter per token)', run(m, T) {
    T.close(m.trainingFlops(1e9, 1e10), 6e19, 1e-9, '1B params × 10B tokens must be 6e19 FLOPs; 2ND is the forward pass only, 4ND is the backward pass only');
    T.close(m.trainingFlops(175e9, 300e9), 3.15e23, 1e-6, 'GPT-3 (175B params, 300B tokens) is approximately 3.1e23 FLOPs; Brown et al. 2020 report 3.14e23');
    const next = T.rng(8);
    for (let i = 0; i < 5; i++) {
      const N = 1e6 + next() * 1e12, D = 1e8 + next() * 1e13;
      T.close(m.trainingFlops(2 * N, D), 2 * m.trainingFlops(N, D), 1e-9, 'doubling N must double the FLOPs');
      T.close(m.trainingFlops(N, 3 * D), 3 * m.trainingFlops(N, D), 1e-9, 'tripling D must triple the FLOPs');
    }
  } },
  { step: 'flops', name: 'inferenceFlops is 2·N per generated token and scales with the token count', run(m, T) {
    T.close(m.inferenceFlops(8e9), 1.6e10, 1e-9, 'one token through an 8B model is one multiply-add per parameter: 2·N = 1.6e10 FLOPs');
    T.close(m.inferenceFlops(8e9, 1000), 1.6e13, 1e-9, 'tokens defaults to 1; with tokens=1000 the cost is 2·N·1000');
    T.close(m.trainingFlops(3e9, 5e9) / m.inferenceFlops(3e9, 5e9), 3, 1e-9, 'training on D tokens costs exactly 3× a forward pass over those tokens (forward 2N + backward 4N)');
  } },
  { step: 'flops', name: 'rejects non-positive sizes', run(m, T) {
    T.throws(() => m.trainingFlops(0, 1e9), 'N = 0 is not a model; throw rather than return 0');
    T.throws(() => m.trainingFlops(1e9, -5), 'negative token counts must throw');
  } },

  // ---------- step 2 ----------
  { step: 'wallclock', name: 'wallClockSeconds divides FLOPs by gpus × peak × MFU', run(m, T) {
    const flops = 3.15e23;
    T.close(m.wallClockSeconds(flops, { gpus: 1, peakFlops: 989e12, mfu: 1 }), 3.15e23 / 989e12, 1e-9, 'one H100 at 100% MFU: FLOPs / peak (about 10 years for GPT-3)');
    T.close(m.wallClockSeconds(flops, { gpus: 1, peakFlops: 989e12, mfu: 0.4 }), 3.15e23 / 989e12 / 0.4, 1e-9, 'at 40% MFU the run takes 2.5× longer; the MFU factor must divide, not multiply');
    T.close(m.wallClockSeconds(flops, { gpus: 1024, peakFlops: 989e12, mfu: 0.4 }), 777596.53, 1e-6, '1024 H100s at 40% MFU on GPT-3: about 9 days');
    const next = T.rng(2);
    for (let i = 0; i < 4; i++) {
      const g = 1 + Math.floor(next() * 4000), p = 1e14 + next() * 1e15, u = 0.2 + next() * 0.6;
      T.close(m.wallClockSeconds(flops, { gpus: 2 * g, peakFlops: p, mfu: u }), m.wallClockSeconds(flops, { gpus: g, peakFlops: p, mfu: u }) / 2, 1e-9, 'doubling the GPU count must halve the time (perfect scaling is the assumption here)');
    }
  } },
  { step: 'wallclock', name: 'gpuHours is the single-GPU time in hours, independent of how many GPUs share the work', run(m, T) {
    T.close(m.gpuHours(3.15e23, { peakFlops: 989e12, mfu: 0.4 }), 3.15e23 / (989e12 * 0.4) / 3600, 1e-9, 'GPU-hours = FLOPs / (peak × MFU) / 3600; forgetting the /3600 gives GPU-seconds');
    T.close(m.gpuHours(1e21, { peakFlops: 1e15, mfu: 0.5 }), 1e21 / 5e14 / 3600, 1e-9);
    T.close(m.gpuHours(3.15e23, { peakFlops: 989e12, mfu: 0.4 }), m.wallClockSeconds(3.15e23, { gpus: 128, peakFlops: 989e12, mfu: 0.4 }) * 128 / 3600, 1e-9, 'GPU-hours must equal wall-clock × GPU count');
  } },
  { step: 'wallclock', name: 'MFU must be a fraction in (0, 1]: 40 is a percentage, not an MFU', run(m, T) {
    T.throws(() => m.wallClockSeconds(1e20, { gpus: 8, peakFlops: 989e12, mfu: 40 }), 'mfu: 40 must throw; the correct value is 0.4');
    T.throws(() => m.wallClockSeconds(1e20, { gpus: 8, peakFlops: 989e12, mfu: 0 }), 'mfu: 0 would divide by zero and must throw');
    T.throws(() => m.gpuHours(1e20, { peakFlops: 989e12, mfu: 1.5 }), 'nothing runs above peak; mfu > 1 must throw');
  } },

  // ---------- step 3 ----------
  { step: 'memory', name: 'bf16 mixed precision with AdamW costs 16 bytes per parameter (2 + 2 + 8 + 4)', run(m, T) {
    const r = m.trainingMemory(1e9, { precision: 'bf16', optimizer: 'adamw' });
    T.close(r.weights, 2e9, 1e-9, 'bf16 weights are 2 bytes each');
    T.close(r.grads, 2e9, 1e-9, 'gradients are kept in the same precision as the weights');
    T.close(r.optimizerStates, 8e9, 1e-9, 'AdamW keeps m and v in fp32: 4 + 4 bytes per parameter');
    T.close(r.masterWeights, 4e9, 1e-9, 'mixed precision needs an fp32 master copy of the weights (4 bytes) for the update');
    T.close(r.total, 16e9, 1e-9, 'the total must be the sum of the four: 16 bytes/param, as in the ZeRO paper');
    T.close(r.bytesPerParam, 16, 1e-9);
    T.close(m.trainingMemory(175e9).total, 2.8e12, 1e-9, 'defaults are bf16 + AdamW; GPT-3 training state is 2.8 TB before activations');
  } },
  { step: 'memory', name: 'precision and optimizer change the bill correctly; unknown names throw', run(m, T) {
    T.close(m.trainingMemory(1e9, { precision: 'fp32', optimizer: 'adamw' }).total, 16e9, 1e-9, 'fp32 + AdamW: 4 + 4 + 8 and no master copy = 16 bytes/param');
    T.close(m.trainingMemory(1e9, { precision: 'fp32', optimizer: 'adamw' }).masterWeights, 0, 1e-9, 'fp32 weights need no separate master copy');
    T.close(m.trainingMemory(1e9, { precision: 'bf16', optimizer: 'sgd' }).total, 8e9, 1e-9, 'bf16 + plain SGD: 2 + 2 + 0 + 4 = 8 bytes/param');
    T.close(m.trainingMemory(1e9, { precision: 'bf16', optimizer: 'sgd-momentum' }).total, 12e9, 1e-9, 'momentum is one fp32 buffer: 4 bytes/param');
    T.close(m.trainingMemory(1e9, { precision: 'fp8', optimizer: 'adamw' }).total, 14e9, 1e-9, 'fp8 weights and grads: 1 + 1 + 8 + 4 = 14 bytes/param');
    T.close(m.trainingMemory(1e9, { precision: 'bf16', optimizer: 'sgd' }).bytesPerParam, 8, 1e-9, 'bytesPerParam must be total / N, not a constant 16: bf16 + SGD is 8 bytes/param');
    T.close(m.trainingMemory(3e9, { precision: 'fp8' }).bytesPerParam, 14, 1e-9, 'bytesPerParam must be total / N: fp8 + AdamW is 14 bytes/param whatever N is');
    T.throws(() => m.trainingMemory(1e9, { precision: 'int4' }), 'unknown precision must throw, not silently give NaN');
    T.throws(() => m.trainingMemory(1e9, { optimizer: 'lion' }), 'unknown optimizer must throw');
  } },
  { step: 'memory', name: 'activationBytes follows the Megatron formula: layers · s·b·h·(34 + 5·a·s/h) bytes, minus the 5·a·s²·b term with FlashAttention', run(m, T) {
    const gpt3 = { batch: 1, seq: 2048, hidden: 12288, layers: 96, heads: 96 };
    T.close(m.activationBytes(gpt3), 96 * 2048 * 12288 * (34 + (5 * 96 * 2048) / 12288), 1e-6, 'GPT-3, one sequence of 2048, no recomputation: about 275 GB');
    T.close(m.activationBytes({ ...gpt3, flashAttention: true }), 96 * 2048 * 12288 * 34, 1e-6, 'FlashAttention never materialises the s×s score matrices: 34·s·b·h per layer, about 82 GB');
    T.close(m.activationBytes({ ...gpt3, batch: 4 }), 4 * m.activationBytes(gpt3), 1e-9, 'activations are linear in batch size');
    const small = { batch: 2, seq: 512, hidden: 1024, layers: 4, heads: 16 };
    const ratio = m.activationBytes({ ...small, seq: 1024 }) / m.activationBytes(small);
    T.ok(ratio > 2.5 && ratio < 4, `doubling the sequence length must more than double activations without FlashAttention (got ${ratio.toFixed(2)}×); the score matrices grow as s²`);
    T.close(m.activationBytes({ ...small, bytesPerValue: 4 }), 2 * m.activationBytes(small), 1e-9, 'bytesPerValue must scale both terms: fp32 activations cost twice bf16');
    T.close(m.activationBytes({ ...small, bytesPerValue: 1 }), 0.5 * m.activationBytes(small), 1e-9, 'bytesPerValue: 1 (fp8 activations) must halve the bf16 bill');
    T.close(m.activationBytes({ ...small, flashAttention: true, seq: 1024 }) / m.activationBytes({ ...small, flashAttention: true }), 2, 1e-9, 'with FlashAttention activations are linear in sequence length');
  } },
  { step: 'memory', name: 'planRun assembles FLOPs, time, GPU-hours, dollars and memory for GPT-3 on 1024 H100s', run(m, T) {
    const p = m.planRun({ params: 175e9, tokens: 300e9, gpus: 1024, gpuFlops: 989e12, gpuMemoryBytes: 80e9, mfu: 0.4, precision: 'bf16', optimizer: 'adamw', pricePerGpuHour: 2 });
    T.close(p.flops, 3.15e23, 1e-6, 'flops must come from trainingFlops');
    T.close(p.seconds, 777596.53, 1e-6, 'seconds must come from wallClockSeconds');
    T.close(p.days, 777596.53 / 86400, 1e-6, 'days = seconds / 86400 (about 9 days)');
    T.close(p.gpuHours, 221183.01, 1e-6, 'GPU-hours must come from gpuHours');
    T.close(p.dollars, 442366.03, 1e-6, 'dollars = GPU-hours × price per GPU-hour');
    T.close(p.memory.total, 2.8e12, 1e-9, 'memory must come from trainingMemory');
    T.eq(p.minGpusForState, 35, 'ceil(2.8 TB / 80 GB) = 35 GPUs just to hold the training state, before activations');
    T.close(p.tokensPerParam, 300 / 175, 1e-9);
    const d = m.planRun({ params: 8e9, tokens: 15e12, gpus: 8 });
    T.close(d.flops, 7.2e23, 1e-6, 'defaults: H100 peak, 40% MFU, bf16 + AdamW must give 7.2e23 FLOPs for Llama-3-8B');
    T.close(d.seconds, 7.2e23 / (8 * 989e12 * 0.4), 1e-6, 'default gpuFlops must be the H100 dense bf16 peak (989e12) and default mfu 0.4');
    T.eq(d.minGpusForState, 2, '8B × 16 bytes = 128 GB does not fit one 80 GB H100');
    const f = m.planRun({ params: 1e9, tokens: 1e11, gpus: 8, gpuMemoryBytes: 80e9, precision: 'fp8', optimizer: 'sgd' });
    T.close(f.memory.total, 6e9, 1e-9, 'planRun must pass precision and optimizer through to trainingMemory: fp8 + SGD is 1 + 1 + 0 + 4 = 6 bytes/param');
    T.eq(f.minGpusForState, 1, '1B × 6 bytes = 6 GB fits on one 80 GB GPU');
  } },

  // ---------- step 4 ----------
  { step: 'chinchilla', name: 'scalingLoss is E + A/N^α + B/D^β with the paper\'s constants by default', run(m, T) {
    T.close(m.scalingLoss(7e10, 1.4e12), lossOf(7e10, 1.4e12, PAPER), 1e-9, 'Chinchilla itself (70B, 1.4T tokens) under the printed fit');
    T.close(m.scalingLoss(1e30, 1e30), 1.69, 1e-3, 'as N and D go to infinity the loss approaches the irreducible term E = 1.69');
    T.close(m.scalingLoss(1e3, 1e4, { E: 0, A: 1, B: 1, alpha: 1, beta: 1 }), 1e-3 + 1e-4, 1e-9, 'a custom fit must be honoured: 1/N + 1/D');
    T.close(m.scalingLoss(7e10, 1.4e12, REFIT), lossOf(7e10, 1.4e12, REFIT), 1e-9, 'the fit argument selects the constants');
    const next = T.rng(4);
    for (let i = 0; i < 6; i++) {
      const N = Math.pow(10, 7 + next() * 5), D = Math.pow(10, 9 + next() * 5);
      T.close(m.scalingLoss(N, D), lossOf(N, D, PAPER), 1e-9);
      T.ok(m.scalingLoss(2 * N, D) < m.scalingLoss(N, D) && m.scalingLoss(N, 2 * D) < m.scalingLoss(N, D), 'loss must fall when either N or D grows');
    }
  } },
  { step: 'chinchilla', name: 'chinchillaOptimal minimises the loss on the 6ND = C constraint', run(m, T) {
    for (const [name, fit] of [['paper', PAPER], ['refit', REFIT]]) {
      for (const C of [1e18, 1e21, 5.76e23, 3.8e25]) {
        const o = m.chinchillaOptimal(C, fit);
        T.ok(o.N > 0 && o.D > 0, `${name}, C=${C}: N and D must be positive`);
        T.close(6 * o.N * o.D, C, 1e-6, `${name}, C=${C}: the returned N and D must spend exactly the budget (6·N·D = C)`);
        const ref = analyticOptimum(C, fit);
        T.close(o.N, ref.N, 0.03, `${name}, C=${C}: N is off from the true minimiser ${ref.N.toExponential(3)} by more than 3%`);
        T.close(o.loss, lossOf(o.N, o.D, fit), 1e-9, 'loss must be scalingLoss(N, D, fit) at the optimum');
        T.close(o.tokensPerParam, o.D / o.N, 1e-9, 'tokensPerParam = D / N');
        for (const k of [1.2, 1 / 1.2]) {
          T.ok(lossOf(o.N * k, C / (6 * o.N * k), fit) >= o.loss - 1e-12, `${name}, C=${C}: moving N by ${k > 1 ? '+' : '-'}20% along the constraint must not lower the loss`);
        }
      }
    }
    const sym = m.chinchillaOptimal(6e20, { E: 1, A: 100, B: 100, alpha: 0.5, beta: 0.5 });
    T.close(sym.N, 1e10, 1e-3, 'a symmetric fit (A = B, α = β) must split the budget evenly: N = D = sqrt(C/6)');
    T.close(sym.D, 1e10, 1e-3);
  } },
  { step: 'chinchilla', name: 'the two fits disagree about tokens per parameter at Gopher\'s budget', run(m, T) {
    const refit = m.chinchillaOptimal(5.76e23, REFIT);
    T.ok(refit.N > 5.5e10 && refit.N < 9e10, `the replication fit gives roughly 70B parameters for 5.76e23 FLOPs (got ${refit.N.toExponential(2)})`);
    T.ok(refit.tokensPerParam > 15 && refit.tokensPerParam < 25, `the replication fit gives roughly 20 tokens per parameter (got ${refit.tokensPerParam.toFixed(1)})`);
    const paper = m.chinchillaOptimal(5.76e23, PAPER);
    T.ok(paper.tokensPerParam > 60, `the printed constants imply far more than 20 tokens per parameter (got ${paper.tokensPerParam.toFixed(1)}); this is the discrepancy Besiroglu et al. 2024 report`);
    T.ok(m.chinchillaOptimal(3.8e25, PAPER).tokensPerParam > paper.tokensPerParam, 'with α ≠ β the optimal ratio drifts with the budget instead of staying constant');
  } },

  // ---------- step 5 ----------
  { step: 'overtrain', name: 'tokensForLoss inverts scalingLoss in D and returns Infinity for unreachable targets', run(m, T) {
    for (const fit of [PAPER, REFIT]) {
      for (const N of [1e9, 8e9, 7e10]) {
        const target = lossOf(N, 3e12, fit);
        const D = m.tokensForLoss(N, target, fit);
        T.close(D, 3e12, 1e-4, 'tokensForLoss(N, scalingLoss(N, D)) must give back D');
        T.close(lossOf(N, D, fit), target, 1e-9);
      }
      T.eq(m.tokensForLoss(1e9, fit.E, fit), Infinity, 'no amount of data reaches the irreducible loss');
      const floor = fit.E + fit.A / Math.pow(1e9, fit.alpha);
      T.eq(m.tokensForLoss(1e9, floor - 1e-6, fit), Infinity, 'a 1B-parameter model cannot go below the loss its size term alone leaves, however much data it sees');
      T.ok(m.tokensForLoss(1e9, floor + 1e-6, fit) > 1e20, `a target a hair above the size floor must demand an absurd number of tokens (got ${m.tokensForLoss(1e9, floor + 1e-6, fit).toExponential(2)}); the data term B/D^beta shrinks only polynomially`);
    }
    T.close(m.tokensForLoss(8e9, lossOf(8e9, 15e12, PAPER)), 15e12, 1e-4, 'the fit defaults to the paper\'s constants');
  } },
  { step: 'overtrain', name: 'optimalForLoss finds the cheapest (N, D) that reaches a target loss', run(m, T) {
    for (const fit of [PAPER, REFIT]) {
      for (const target of [2.4, 2.1, 1.95]) {
        const o = m.optimalForLoss(target, fit);
        T.close(lossOf(o.N, o.D, fit), target, 1e-4, `the returned model must reach the target loss ${target}`);
        T.close(o.flops, 6 * o.N * o.D, 1e-6, 'flops must be 6·N·D of the returned model');
        const ref = analyticOptimum(o.flops, fit);
        T.close(o.N, ref.N, 0.03, 'the returned model must sit on the compute-optimal frontier for its budget');
        const alt = 1.5 * o.N;
        T.ok(6 * alt * m.tokensForLoss(alt, target, fit) > o.flops * 0.999, 'a 1.5× larger model reaching the same loss must not be cheaper');
      }
    }
    T.throws(() => m.optimalForLoss(1.0, PAPER), 'a target below E = 1.69 is unreachable and must throw');
  } },
  { step: 'overtrain', name: 'overtrainingAnalysis of Llama-3-8B: more training now, cheaper tokens forever, break-even near 1e13 inference tokens', run(m, T) {
    const r = m.overtrainingAnalysis({ N: 8e9, D: 15e12, inferenceTokens: 2e13, fit: REFIT });
    T.close(r.loss, lossOf(8e9, 15e12, REFIT), 1e-9);
    T.close(r.trainingFlops, 7.2e23, 1e-6);
    T.ok(r.optimal.N > 2.5e10 && r.optimal.N < 4.5e10, `the compute-optimal model for the same loss is roughly 34B parameters (got ${r.optimal.N.toExponential(2)})`);
    T.close(r.extraTrainingFlops, 7.2e23 - r.optimal.flops, 1e-6, 'extra training = 6ND of the actual run minus the optimal budget');
    T.close(r.savingPerInferenceToken, 2 * (r.optimal.N - 8e9), 1e-9, 'each served token saves 2·(N_opt − N) FLOPs');
    T.close(r.breakEvenInferenceTokens, r.extraTrainingFlops / r.savingPerInferenceToken, 1e-9, 'break-even = extra training FLOPs / saving per token');
    T.ok(r.breakEvenInferenceTokens > 5e12 && r.breakEvenInferenceTokens < 2.5e13, `break-even should land near 1.1e13 tokens (got ${r.breakEvenInferenceTokens.toExponential(2)})`);
    T.close(r.lifetimeFlops, 7.2e23 + 2 * 8e9 * 2e13, 1e-9, 'lifetime = training + 2·N·inference tokens');
    T.close(r.lifetimeFlopsOptimal, r.optimal.flops + 2 * r.optimal.N * 2e13, 1e-9);
    T.ok(r.lifetimeFlops < r.lifetimeFlopsOptimal, 'past break-even the over-trained small model is cheaper over its lifetime');
    const z = m.overtrainingAnalysis({ N: 8e9, D: 15e12, fit: REFIT });
    T.close(z.lifetimeFlops, 7.2e23, 1e-9, 'with no inference tokens the lifetime cost is the training cost');
    T.ok(z.lifetimeFlops > z.lifetimeFlopsOptimal, 'with no inference the over-trained run is strictly the worse deal');
  } },
  { step: 'overtrain', name: 'an under-trained run (too few tokens for its size) has no inference break-even', run(m, T) {
    const r = m.overtrainingAnalysis({ N: 175e9, D: 300e9, inferenceTokens: 1e12, fit: REFIT });
    T.ok(r.optimal.N < 175e9, 'GPT-3 was larger than compute-optimal for its loss, so the optimal model is smaller');
    T.ok(r.savingPerInferenceToken < 0, 'a smaller optimal model would have been cheaper per inference token, so the saving is negative');
    T.eq(r.breakEvenInferenceTokens, Infinity, 'when inference does not get cheaper, break-even is never reached: Infinity');
    T.ok(r.extraTrainingFlops > 0, 'under-training still wastes compute relative to the optimal run for that loss');
  } },
];
