import { rng, randInt } from 'lib/util.js';

/**
 * One request stream in which a 16-token timestamp is injected into every prompt, either before
 * the stable system prompt (`where === 'first'`) or after the user's turn (`where === 'last'`).
 * The token content is identical in both streams; only the order differs.
 */
function timestampedStream(where, { conversations = 30, turns = 6, seed = 101 } = {}) {
  const next = rng(seed);
  const tok = (n) => Array.from({ length: n }, () => randInt(next, 4096));
  const system = tok(64);          // identical for every request
  const fewShot = tok(96);         // identical for every request
  const stable = system.concat(fewShot);
  const requests = [];
  for (let c = 0; c < conversations; c++) {
    let history = [];
    for (let t = 0; t < turns; t++) {
      const user = tok(32);
      const stamp = tok(16);       // a fresh timestamp on every single request
      const tokens = where === 'first'
        ? stamp.concat(stable, history, user)
        : stable.concat(history, user, stamp);
      requests.push({ id: requests.length, conv: c, turn: t, tokens, outputTokens: 48 });
      history = history.concat(user, tok(48));
    }
  }
  return requests;
}

export default async function demo(m, lab) {
  // ---------- 1. what a hash chain does to an edited prompt ----------
  const base = Array.from({ length: 96 }, (_, i) => i); // 6 blocks of 16
  const cache = new m.HashCache();
  cache.insert(base);
  const edits = [
    ['unchanged', base],
    ['token 3 changed (block 0)', base.map((t, i) => (i === 3 ? 9999 : t))],
    ['token 20 changed (block 1)', base.map((t, i) => (i === 20 ? 9999 : t))],
    ['token 90 changed (block 5)', base.map((t, i) => (i === 90 ? 9999 : t))],
    ['16 tokens prepended', Array.from({ length: 16 }, () => 9999).concat(base)],
    ['16 tokens appended', base.concat(Array.from({ length: 16 }, () => 9999))],
  ];
  lab.table({
    title: 'One 96-token prompt in the cache: what each edit costs you',
    columns: ['prompt', 'cached blocks (of 6)', 'tokens reused', 'tokens to prefill'],
    rows: edits.map(([name, toks]) => {
      const r = cache.lookup(toks);
      return [name, r.blocks, r.tokens, toks.length - r.tokens];
    }),
  });

  // ---------- 2. hit rate vs cache capacity on 500 requests ----------
  const requests = m.makeWorkload({ conversations: 50, turns: 10, seed: 17 });
  const capacities = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
  const runs = [];
  for (let i = 0; i < capacities.length; i++) {
    runs.push(m.simulate(requests, { capacityBlocks: capacities[i], concurrency: 4 }));
    lab.progress((i + 1) / capacities.length, `capacity ${capacities[i]} blocks`);
    await lab.tick();
  }
  const hitRates = runs.map((r) => r.hitRate);
  const best = runs[runs.length - 1];
  const worst = runs[0];
  lab.plot({
    title: 'Prefix hit rate vs cache capacity (500 requests, 50 chats, 4 in flight)',
    x: capacities,
    series: [
      { name: 'hit rate', values: hitRates },
      { name: 'break-even', values: capacities.map(() => m.breakEvenHitRate()) },
    ],
    xlabel: 'cache capacity (16-token blocks)',
    ylabel: 'fraction of prompt blocks served from cache',
  });

  // ---------- 3. what the hit rate is worth ----------
  const noCache = m.baselineCost(best.promptTokens, best.outputTokens);
  const costs = runs.map((r) => m.requestCost(r.promptTokens, r.cachedTokens, r.outputTokens));
  lab.bar({
    title: 'Total bill for the same 500 requests (units of one base input token)',
    labels: ['no caching', `${capacities[0]} blocks`, `${capacities[3]} blocks`, `${capacities[capacities.length - 1]} blocks`],
    values: [noCache, costs[0], costs[3], costs[costs.length - 1]],
  });

  // ---------- 4. the tree, and the latency the cache buys ----------
  const perReq = (r) => ({ p: r.promptTokens / r.requests, c: r.cachedTokens / r.requests });
  lab.table({
    title: 'Cache state and mean time to first token at each capacity',
    columns: ['capacity (blocks)', 'hit rate', 'nodes', 'blocks held', 'blocks evicted', 'bill', 'mean TTFT (ms)'],
    rows: runs.map((r, i) => {
      const { p, c } = perReq(r);
      return [capacities[i], +r.hitRate.toFixed(3), r.nodes, r.sizeBlocks, r.evictedBlocks,
        Math.round(costs[i]), +m.ttftMs(p, c).toFixed(1)];
    }),
  });

  // ---------- 5. segment order: stable content first ----------
  const stampFirst = m.simulate(timestampedStream('first'), { capacityBlocks: 8192 });
  const stampLast = m.simulate(timestampedStream('last'), { capacityBlocks: 8192 });
  await lab.tick();
  lab.bar({
    title: 'Identical tokens, different order: a 16-token timestamp in every prompt',
    labels: ['timestamp before the system prompt', 'timestamp after the user turn'],
    values: [+stampFirst.hitRate.toFixed(4), +stampLast.hitRate.toFixed(4)],
  });

  const ttftCold = m.ttftMs(perReq(best).p, 0);
  const ttftWarm = m.ttftMs(perReq(best).p, perReq(best).c);
  const saved = (1 - costs[costs.length - 1] / noCache) * 100;
  lab.done(`Your radix cache served **${(100 * best.hitRate).toFixed(1)}%** of prompt blocks from cache at ${capacities[capacities.length - 1]} blocks of capacity, against **${(100 * worst.hitRate).toFixed(1)}%** at ${capacities[0]} blocks — the tree ended with ${best.nodes} nodes holding ${best.sizeBlocks} blocks and evicted ${worst.evictedBlocks.toLocaleString()} blocks in the tight run.

At the example multipliers (write ${m.PRICING.cacheWrite}x, read ${m.PRICING.cacheRead}x) the bill for these 500 requests fell from ${Math.round(noCache).toLocaleString()} to **${Math.round(costs[costs.length - 1]).toLocaleString()}** units, a **${saved.toFixed(1)}%** saving, and mean time to first token fell from ${ttftCold.toFixed(0)} ms to **${ttftWarm.toFixed(0)} ms**. Break-even was at a hit rate of ${(100 * m.breakEvenHitRate()).toFixed(1)}%; the ${capacities[0]}-block cache reached ${(100 * worst.hitRate).toFixed(1)}%, so it ${worst.hitRate > m.breakEvenHitRate() ? 'still paid for itself' : 'actually cost more than no caching at all'}.

Injecting a 16-token timestamp into every prompt costs nothing when it goes last (**${(100 * stampLast.hitRate).toFixed(1)}%** hit rate) and destroys the cache when it goes first (**${(100 * stampFirst.hitRate).toFixed(1)}%**), on exactly the same tokens.`);
}
