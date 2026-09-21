export default async function demo(m, lab) {
  const SLO = { ttft: 1.0, tpot: 0.01 };          // 1 s to first token, 10 ms per output token (100 tok/s)
  const REPLICAS = 4;
  const SECONDS = 120;
  const N = 2600;

  const work = m.makeWorkload({ n: N, seconds: SECONDS, seed: 11, nPrefixes: 16, alpha: 1.0, peak: 3 });
  const meanPrompt = work.reduce((s, r) => s + r.promptLen, 0) / work.length;
  const meanOut = work.reduce((s, r) => s + r.outputLen, 0) / work.length;
  lab.log(`${work.length} requests over ${SECONDS} simulated seconds (${(work.length / SECONDS).toFixed(1)}/s average),`,
    `mean prompt ${meanPrompt.toFixed(0)} tokens, mean answer ${meanOut.toFixed(0)} tokens,`,
    `16 shared prefixes with Zipf popularity, SLO = TTFT <= ${SLO.ttft} s and TPOT <= ${(SLO.tpot * 1000).toFixed(0)} ms.`);

  const pools = m.planPools(work, REPLICAS);
  lab.log(`planPools split ${REPLICAS} replicas into ${pools.prefill} prefill + ${pools.decode} decode`,
    `(${(100 * pools.prefillWork / (pools.prefillWork + pools.decodeWork)).toFixed(0)}% of the work in this trace is prefill).`);

  const configs = [
    { name: 'round-robin, colocated', opts: { policy: 'round-robin', replicas: REPLICAS } },
    { name: 'cache-aware, colocated', opts: { policy: 'cache-aware', replicas: REPLICAS } },
    { name: 'cache-aware, disaggregated', opts: { policy: 'cache-aware', disaggregate: true, pools } },
    { name: `cache-aware, autoscaled ${m.DEFAULT_CONFIG.minReplicas}-${m.DEFAULT_CONFIG.maxReplicas}`,
      opts: { policy: 'cache-aware', replicas: REPLICAS, autoscale: true } },
  ];

  const runs = [];
  for (let i = 0; i < configs.length; i++) {
    lab.progress(i / configs.length, configs[i].name);
    await lab.tick();
    const res = m.runCluster(work, configs[i].opts);
    const rep = m.sloReport(res.records, SLO);
    runs.push({ ...configs[i], res, rep });
    lab.log(`${configs[i].name}: attainment ${(100 * rep.attainment).toFixed(0)}%,`,
      `p95 TTFT ${rep.ttftP95.toFixed(2)} s, p95 TPOT ${(1000 * rep.tpotP95).toFixed(1)} ms,`,
      `cache hit rate ${(100 * res.cacheHitRate).toFixed(0)}%, ${res.gpuSeconds.toFixed(0)} GPU-seconds.`);
  }
  lab.progress(1, 'done');

  // --- p95 TTFT over time, per 10-second window of arrival ---
  const W = 10;
  const nWindows = Math.ceil(SECONDS / W);
  const xs = [];
  for (let w = 0; w < nWindows; w++) xs.push(w * W);
  const series = [];
  for (const run of runs.slice(0, 3)) {
    const values = [];
    for (let w = 0; w < nWindows; w++) {
      const slice = run.res.records.filter((r) => r.arrival >= w * W && r.arrival < (w + 1) * W);
      values.push(slice.length ? m.sloReport(slice, SLO).ttftP95 : 0);
    }
    series.push({ name: run.name, values });
    await lab.tick();
  }
  lab.plot({ title: 'p95 time-to-first-token per 10-second window', x: xs, series,
    xlabel: 'simulated time (s)', ylabel: 'p95 TTFT (s)' });

  // --- goodput ---
  lab.bar({ title: `Goodput: requests per second meeting BOTH SLOs (TTFT <= ${SLO.ttft} s, TPOT <= ${(1000 * SLO.tpot).toFixed(0)} ms)`,
    labels: runs.map((r) => r.name), values: runs.map((r) => +r.rep.goodput.toFixed(2)) });

  // --- the operator's table ---
  lab.table({
    title: 'What each configuration costs and delivers',
    columns: ['configuration', 'attainment', 'goodput (req/s)', 'p95 TTFT (s)', 'p95 TPOT (ms)',
      'cache hits', 'decode stall (s)', 'GPU-seconds', '$ / M tokens'],
    rows: runs.map((r) => [
      r.name,
      (100 * r.rep.attainment).toFixed(0) + '%',
      +r.rep.goodput.toFixed(2),
      +r.rep.ttftP95.toFixed(3),
      +(1000 * r.rep.tpotP95).toFixed(1),
      (100 * r.res.cacheHitRate).toFixed(0) + '%',
      +r.res.stallSeconds.toFixed(1),
      Math.round(r.res.gpuSeconds),
      +m.costPerMillionTokens(r.res.gpuSeconds, r.rep.outputTokens, m.DEFAULT_CONFIG.dollarsPerGpuHour).toFixed(2),
    ]),
  });

  // --- what the autoscaler did ---
  const auto = runs[3];
  const trace = auto.res.scaleTrace;
  lab.plot({
    title: 'The autoscaler: what it asked for and what it had',
    x: trace.map((p) => +p.t.toFixed(1)),
    series: [
      { name: 'replicas alive', values: trace.map((p) => p.alive) },
      { name: 'raw target from the queue', values: trace.map((p) => p.raw) },
      { name: 'decision after stabilisation', values: trace.map((p) => p.desired) },
    ],
    xlabel: 'simulated time (s)', ylabel: 'replicas',
  });

  const [rr, ca, di, au] = runs;
  const peakCost = m.gpuSeconds([{ start: 0, end: au.res.makespan }], au.res.makespan) * m.DEFAULT_CONFIG.maxReplicas;
  lab.done(`Same ${work.length} requests, same ${REPLICAS} GPUs, four ways of arranging them.

**Routing.** Cache-aware routing lifted the prefix-cache hit rate from **${(100 * rr.res.cacheHitRate).toFixed(0)}%** to **${(100 * ca.res.cacheHitRate).toFixed(0)}%**, which cut median TTFT from ${rr.rep.ttftP50.toFixed(3)} s to **${ca.rep.ttftP50.toFixed(3)} s** and raised SLO attainment from ${(100 * rr.rep.attainment).toFixed(0)}% to **${(100 * ca.rep.attainment).toFixed(0)}%**. No extra hardware; the requests just went to the replica that already held their prompt.

**Pooling.** Splitting the same 4 GPUs into ${pools.prefill} prefill + ${pools.decode} decode removed every prefill stall from the decode path: **${ca.res.stallSeconds.toFixed(0)} s** of decode stall colocated against **${di.res.stallSeconds.toFixed(0)} s** disaggregated, so p95 TPOT fell from ${(1000 * ca.rep.tpotP95).toFixed(1)} ms to **${(1000 * di.rep.tpotP95).toFixed(1)} ms** and attainment reached **${(100 * di.rep.attainment).toFixed(0)}%** at **$${m.costPerMillionTokens(di.res.gpuSeconds, di.rep.outputTokens, m.DEFAULT_CONFIG.dollarsPerGpuHour).toFixed(2)} per million output tokens**. It cost ${di.res.handoffs} KV handoffs totalling ${di.res.transferSecondsTotal.toFixed(1)} s of transfer.

**Scaling.** The autoscaler ran from ${REPLICAS} up to ${au.res.peakReplicas} replicas against the diurnal hump and reached ${(100 * au.rep.attainment).toFixed(0)}% attainment for ${Math.round(au.res.gpuSeconds)} GPU-seconds — ${(au.res.gpuSeconds / di.res.gpuSeconds).toFixed(1)}x the bill of the disaggregated cluster for ${((di.rep.attainment - au.rep.attainment) * 100).toFixed(0)} points LESS attainment, and still ${(100 * (1 - au.res.gpuSeconds / peakCost)).toFixed(0)}% cheaper than standing all ${m.DEFAULT_CONFIG.maxReplicas} replicas up for the whole window. A ${m.DEFAULT_CONFIG.coldStartSeconds} s cold start against a ${SECONDS} s ramp is why arranging the GPUs you already have beat buying more of them.`);
}
