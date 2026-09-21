// Module 16 demo — run one trace through three scheduling policies with the learner's engine,
// then measure what paging buys: fragmentation, preemption and prefix sharing.

export default async function demo(m, lab) {
  const cfg = m.DEFAULT_CONFIG;
  const requests = m.makeWorkload({ n: 200, seed: 7, rate: 80 });
  const span = requests[requests.length - 1].arrival;
  const outs = requests.map((r) => r.outputLen).sort((a, b) => a - b);
  const promptTokens = requests.reduce((s, r) => s + r.promptLen, 0);
  lab.log(`200 requests arriving over ${span.toFixed(2)} s (Poisson, 80/s). Prompts total ${promptTokens} tokens; output lengths: p50 ${outs[100]}, p95 ${outs[190]}, max ${outs[199]}.`);
  lab.log(`Machine model: ${(cfg.tFixed * 1000).toFixed(1)} ms fixed per iteration + ${(cfg.tPerToken * 1e6).toFixed(0)} us per token in the batch, up to ${cfg.maxBatch} concurrent requests.`);

  // ---------- 1. three policies on the same trace ----------
  const runs = [];
  runs.push(m.runStatic(requests, cfg));
  await lab.tick();
  runs.push(m.runContinuous(requests, cfg));
  await lab.tick();
  const roomy = { ...cfg, numBlocks: 4096 };
  const tight = { ...cfg, numBlocks: 256 };
  runs.push(m.runPaged(requests, roomy));
  await lab.tick();
  runs.push(m.runPaged(requests, tight));
  for (const r of runs) lab.check(r.completed === 200, `${r.policy} dropped requests: ${r.completed}/200 completed`);
  const names = ['static (batch of 32)', 'continuous', `continuous + paging, ${roomy.numBlocks} blocks`, `continuous + paging, ${tight.numBlocks} blocks`];
  lab.table({
    title: 'One trace, four schedulers',
    columns: ['policy', 'makespan (s)', 'output tok/s', 'all tok/s', 'p50 TTFT (s)', 'p95 TTFT (s)', 'TPOT (ms)', 'tokens processed', 'preemptions'],
    rows: runs.map((r, i) => [
      names[i], r.makespan.toFixed(2), r.throughput.toFixed(0), (r.batchTokens / r.makespan).toFixed(0),
      r.ttftP50.toFixed(3), r.ttftP95.toFixed(3), (r.tpotMean * 1000).toFixed(1),
      r.batchTokens, r.preemptions === undefined ? '—' : r.preemptions,
    ]),
  });
  lab.bar({ title: 'Output tokens per second, same hardware, same trace', labels: names, values: runs.map((r) => r.throughput) });
  await lab.tick();

  // ---------- 2. why a bigger batch is faster, and what it costs ----------
  const batches = [1, 2, 4, 8, 16, 32, 64, 128];
  const measured = [], ceiling = [], p95 = [], tpot = [];
  for (const b of batches) {
    const r = m.runContinuous(requests, { ...cfg, maxBatch: b });
    measured.push(r.batchTokens / r.makespan);
    ceiling.push(b / (cfg.tFixed + cfg.tPerToken * b));
    p95.push(r.ttftP95);
    tpot.push(r.tpotMean * 1000);
    lab.progress(batches.indexOf(b) / batches.length, `maxBatch ${b}`);
    await lab.tick();
  }
  lab.plot({
    title: 'Throughput against batch size: amortising the fixed cost, until the requests run out',
    x: batches,
    series: [
      { name: 'measured (all tokens/s)', values: measured },
      { name: 'cost-model ceiling, decode only', values: ceiling },
    ],
    xlabel: 'maxBatch (concurrent requests)', ylabel: 'tokens / s',
  });
  lab.table({
    title: 'The dial: batch size trades tail latency against per-token latency',
    columns: ['maxBatch', 'all tok/s', 'p95 TTFT (s)', 'TPOT (ms)'],
    rows: batches.map((b, i) => [b, measured[i].toFixed(0), p95[i].toFixed(2), tpot[i].toFixed(1)]),
  });

  // ---------- 3. what block size costs in wasted memory ----------
  const resident = requests.map((r) => r.promptLen + r.outputLen - 1);
  const blockSizes = [1, 8, 16, 32, 64, 256, 1024];
  const waste = blockSizes.map((b) => 100 * m.internalFragmentation(resident, b).fraction);
  lab.bar({
    title: 'Internal fragmentation: % of allocated KV slots holding no token (1024 = one contiguous slab per request)',
    labels: blockSizes.map((b) => `${b}-token blocks`), values: waste,
  });
  lab.log(`With ${cfg.blockSize}-token blocks the running engine wasted ${(100 * runs[2].wastedFraction).toFixed(1)}% of the slots it had allocated, averaged over ${runs[2].iterations} iterations; peak use was ${runs[2].peakBlocks} of ${roomy.numBlocks} blocks.`);
  await lab.tick();

  // ---------- 4. prefix sharing ----------
  const shared = m.makeWorkload({ n: 200, seed: 11, rate: 80, sharedFraction: 0.6 });
  const alloc = new m.BlockAllocator({ numBlocks: 200000, blockSize: cfg.blockSize });
  const cache = new Map();
  let sharedBlocks = 0, newBlocks = 0;
  for (const r of shared) {
    const res = m.allocateShared(alloc, cache, r.key, r.promptLen);
    lab.check(res !== null, `allocateShared ran out of blocks on request ${r.id}`);
    sharedBlocks += res.sharedBlocks;
    newBlocks += res.newBlocks;
  }
  const withoutSharing = shared.reduce((s, r) => s + Math.ceil(r.promptLen / cfg.blockSize), 0);
  lab.bar({
    title: 'KV blocks for 200 prompts, 60% of them drawn from four shared system prompts',
    labels: ['one copy per request', 'with prefix sharing'], values: [withoutSharing, newBlocks],
  });
  await lab.tick();

  const speedup = runs[1].throughput / runs[0].throughput;
  const ttftDrop = runs[0].ttftP95 / runs[1].ttftP95;
  const savedPct = 100 * (1 - newBlocks / withoutSharing);
  lab.done(`Your engine served the same 200 requests four ways.

**Static batching** finished in ${runs[0].makespan.toFixed(1)} s at **${runs[0].throughput.toFixed(0)} output tokens/s**, processing ${runs[0].batchTokens} tokens — ${(100 * (1 - runs[1].batchTokens / runs[0].batchTokens)).toFixed(0)}% of them padding for slots whose request had already finished. **Continuous batching** processed only the ${runs[1].batchTokens} tokens somebody asked for, finished in ${runs[1].makespan.toFixed(1)} s at **${runs[1].throughput.toFixed(0)} output tokens/s** (**${speedup.toFixed(1)}x**), and cut p95 time-to-first-token from ${runs[0].ttftP95.toFixed(2)} s to **${runs[1].ttftP95.toFixed(3)} s** (${ttftDrop.toFixed(0)}x) — no request waits for a batch boundary any more.

Paging cost nothing when memory was ample (${roomy.numBlocks} blocks: ${runs[2].throughput.toFixed(0)} tok/s, 0 preemptions, ${(100 * runs[2].wastedFraction).toFixed(1)}% of allocated slots wasted). Squeezed to ${tight.numBlocks} blocks (${tight.numBlocks * cfg.blockSize} KV slots) the scheduler preempted **${runs[3].preemptions}** times and recomputed **${runs[3].recomputedTokens}** tokens, which is where the ${(100 * (1 - runs[3].throughput / runs[2].throughput)).toFixed(0)}% throughput it lost went. Fragmentation is what makes that budget tight: ${cfg.blockSize}-token blocks waste **${waste[2].toFixed(1)}%** of the slots they allocate, while one contiguous 1024-token slab per request wastes **${waste[6].toFixed(1)}%** — the 60-80% that PagedAttention was built to recover.

Sharing the four repeated system prompts took the prompt cache from ${withoutSharing} blocks to **${newBlocks}** (**${savedPct.toFixed(0)}% saved**, ${sharedBlocks} block references reused) — and nothing in it required touching the model.`);
}
