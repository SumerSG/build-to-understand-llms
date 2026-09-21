// Module 08 demo — price four famous training runs with the learner's planner, draw the
// fixed-compute U-shape, and settle the over-training question for Llama-3-8B.
// Reported sizes and token counts are approximate and taken from each system's own report:
// Brown et al. 2020 (GPT-3), Hoffmann et al. 2022 (Chinchilla), Grattafiori et al. 2024 (Llama 3).

const RUNS = [
  { label: 'GPT-3 175B', params: 175e9, tokens: 300e9, gpus: 1024 },
  { label: 'Chinchilla 70B', params: 70e9, tokens: 1.4e12, gpus: 1024 },
  { label: 'Llama-3 8B', params: 8e9, tokens: 15e12, gpus: 16384 },
  { label: 'Llama-3 405B', params: 405e9, tokens: 15.6e12, gpus: 16384 },
];

const PRICE = 2; // assumed US$ per H100-hour on a rented cluster in 2024–2025; buy in bulk and it is lower.

export default async function demo(m, lab) {
  const H100 = m.GPUS.H100;
  lab.log(`Pricing every run on ${H100.name} at approximately ${(H100.peakFlops / 1e12).toFixed(0)} TFLOP/s dense bf16, 40% MFU, bf16 + AdamW, $${PRICE}/GPU-hour.`);

  // ---------- 1. the bill for four real runs ----------
  const plans = [];
  for (const r of RUNS) {
    const p = m.planRun({ params: r.params, tokens: r.tokens, gpus: r.gpus, gpuFlops: H100.peakFlops,
      gpuMemoryBytes: H100.memoryBytes, mfu: 0.4, precision: 'bf16', optimizer: 'adamw', pricePerGpuHour: PRICE });
    lab.check(p.flops > 0 && p.gpuHours > 0, `planRun returned no FLOPs for ${r.label}`);
    plans.push({ ...r, ...p });
    lab.progress(plans.length / RUNS.length, r.label);
    await lab.tick();
  }
  lab.table({
    title: 'What each run costs, computed by your planner',
    columns: ['run', 'N', 'D', 'tok/param', 'FLOPs', 'GPUs', 'wall-clock', 'H100-hours', 'US$ (assumed)', 'state in HBM', 'min GPUs'],
    rows: plans.map((p) => [
      p.label, `${(p.params / 1e9).toFixed(0)}B`, `${(p.tokens / 1e12).toFixed(2)}T`,
      p.tokensPerParam.toFixed(1), p.flops.toExponential(2), p.gpus, m.formatDuration(p.seconds),
      Math.round(p.gpuHours).toLocaleString('en-US'), `$${(p.dollars / 1e6).toFixed(1)}M`,
      m.formatBytes(p.memory.total), p.minGpusForState,
    ]),
  });

  // ---------- 2. the U-shape: loss against model size at a fixed compute budget ----------
  const fit = m.CHINCHILLA_REFIT;
  const budgets = [1e21, 1e23, 1e25];
  const sizes = [];
  for (let e = 7; e <= 12.5; e += 0.1) sizes.push(Math.pow(10, e));
  const series = [];
  const optima = [];
  for (const C of budgets) {
    const values = sizes.map((N) => m.scalingLoss(N, C / (6 * N), fit));
    series.push({ name: `C = ${C.toExponential(0)} FLOPs`, values });
    const o = m.chinchillaOptimal(C, fit);
    optima.push({ C, ...o, argminN: sizes[values.indexOf(Math.min(...values))] });
    await lab.tick();
  }
  lab.plot({
    title: 'Fixed compute, free choice of N: the loss is U-shaped (Chinchilla replication fit)',
    x: sizes, series, xlabel: 'N (parameters); D = C / 6N follows', ylabel: 'predicted loss',
  });
  lab.table({
    title: 'The bottom of each U, from your chinchillaOptimal',
    columns: ['budget C (FLOPs)', 'optimal N', 'optimal D', 'tokens/param', 'loss', 'N at the grid minimum'],
    rows: optima.map((o) => [o.C.toExponential(0), `${(o.N / 1e9).toFixed(2)}B`, `${(o.D / 1e12).toFixed(2)}T`,
      o.tokensPerParam.toFixed(1), o.loss.toFixed(4), `${(o.argminN / 1e9).toFixed(2)}B`]),
  });

  // ---------- 3. what a parameter costs to keep in memory ----------
  const combos = [
    ['fp32 + AdamW', { precision: 'fp32', optimizer: 'adamw' }],
    ['bf16 + AdamW', { precision: 'bf16', optimizer: 'adamw' }],
    ['fp8 + AdamW', { precision: 'fp8', optimizer: 'adamw' }],
    ['bf16 + SGD-momentum', { precision: 'bf16', optimizer: 'sgd-momentum' }],
    ['bf16 + SGD', { precision: 'bf16', optimizer: 'sgd' }],
  ];
  const perParam = combos.map(([, opt]) => m.trainingMemory(1e9, opt).bytesPerParam);
  lab.bar({ title: 'Bytes of persistent state per parameter (weights + grads + optimizer + fp32 master)',
    labels: combos.map(([name]) => name), values: perParam });

  const act = m.activationBytes({ batch: 1, seq: 2048, hidden: 12288, layers: 96, heads: 96 });
  const actFlash = m.activationBytes({ batch: 1, seq: 2048, hidden: 12288, layers: 96, heads: 96, flashAttention: true });
  lab.log(`GPT-3 shaped model, one sequence of 2048 tokens: activations ${m.formatBytes(act)} without FlashAttention, ${m.formatBytes(actFlash)} with it.`);

  // ---------- 4. was Llama-3-8B over-trained? ----------
  const llama = m.overtrainingAnalysis({ N: 8e9, D: 15e12, inferenceTokens: 2e13, fit });
  const equivalent = m.planRun({ params: llama.optimal.N, tokens: llama.optimal.D, gpus: 16384, mfu: 0.4, pricePerGpuHour: PRICE });
  lab.table({
    title: 'Llama-3 8B on 15T tokens versus the compute-optimal model at the same predicted loss',
    columns: ['', 'N', 'D', 'training FLOPs', 'FLOPs per served token', 'H100-days on 16384 GPUs'],
    rows: [
      ['as trained', '8.00B', '15.00T', llama.trainingFlops.toExponential(2), m.inferenceFlops(8e9).toExponential(2),
        (plans[2].seconds / 86400).toFixed(2)],
      ['compute-optimal', `${(llama.optimal.N / 1e9).toFixed(2)}B`, `${(llama.optimal.D / 1e12).toFixed(2)}T`,
        llama.optimal.flops.toExponential(2), m.inferenceFlops(llama.optimal.N).toExponential(2),
        (equivalent.seconds / 86400).toFixed(2)],
    ],
  });
  await lab.tick();

  const h100Days = plans[3].seconds / 86400;
  const overtrainFactor = llama.trainingFlops / llama.optimal.flops;
  lab.done(`Your planner priced four real runs. **Llama-3 405B** (405B parameters, 15.6T tokens) is **${plans[3].flops.toExponential(2)} FLOPs**: about **${h100Days.toFixed(1)} H100-days** of wall-clock on 16,384 H100s at 40% MFU, **${Math.round(plans[3].gpuHours).toLocaleString('en-US')} H100-hours**, roughly **$${(plans[3].dollars / 1e6).toFixed(0)}M** at the assumed $${PRICE}/GPU-hour — and its ${m.formatBytes(plans[3].memory.total)} of weights, gradients and AdamW state needs at least **${plans[3].minGpusForState} GPUs** just to be held, which is why module 24 exists.

At a fixed budget the loss is U-shaped in N: for C = 1e23 FLOPs the fit puts the minimum at **${(optima[1].N / 1e9).toFixed(2)}B parameters** and **${(optima[1].D / 1e12).toFixed(2)}T tokens**, about **${optima[1].tokensPerParam.toFixed(0)} tokens per parameter**. Llama-3 8B used **${plans[2].tokensPerParam.toFixed(0)}** — over-trained, spending **${overtrainFactor.toFixed(1)}x** the training compute of the ${(llama.optimal.N / 1e9).toFixed(0)}B-parameter model that the same fit says reaches the same loss. That buys **${llama.savingPerInferenceToken.toExponential(2)} FLOPs saved per served token**, so the trade pays for itself after **${llama.breakEvenInferenceTokens.toExponential(2)} inference tokens** — a number a production endpoint passes in weeks.`);
}
