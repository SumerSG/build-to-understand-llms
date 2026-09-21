// Module 16 — tests. Every workload here is built explicitly so the numbers are checkable by hand.

const CFG = { tFixed: 0.01, tPerToken: 0.001, maxBatch: 4, blockSize: 16, numBlocks: 64 };

function state(m, over = {}) {
  return { ...m.newRequestState({ id: 0, arrival: 0, promptLen: 100, outputLen: 5, key: 'k' }), ...over };
}

/** 80% of requests want 4 tokens, 20% want 120: the skew that static batching cannot handle. */
function skewed(T, n = 24, maxOut = 120) {
  const next = T.rng(5);
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push({ id: i, arrival: i * 0.002, promptLen: 32 + Math.floor(next() * 64),
      outputLen: next() < 0.8 ? 4 : maxOut, key: `k${i}` });
  }
  return out;
}

export const tests = [
  // ---------- step 1: the iteration cost model ----------
  { step: 'cost', name: 'an iteration costs tFixed once plus tPerToken per token in the batch', run(m, T) {
    const dec = (n) => Array.from({ length: n }, () => state(m, { prefilled: true, generated: 1 }));
    T.close(m.iterationSeconds([], CFG), 0.01, 1e-9, 'an empty batch still pays the fixed launch overhead');
    T.close(m.iterationSeconds(dec(1), CFG), 0.011, 1e-9, 'one decode = tFixed + 1 token');
    T.close(m.iterationSeconds(dec(32), CFG), 0.042, 1e-9, 'tFixed is paid ONCE per iteration, not once per request');
    const perToken1 = m.iterationSeconds(dec(1), CFG) / 1;
    const perToken32 = m.iterationSeconds(dec(32), CFG) / 32;
    T.ok(perToken32 < perToken1 / 4, `batching must amortise the fixed cost: got ${perToken32.toFixed(6)} s/token at batch 32 versus ${perToken1.toFixed(6)} at batch 1`);
    const mixed = [state(m, { promptLen: 128 }), ...dec(3)];
    T.close(m.iterationSeconds(mixed, CFG), 0.01 + 0.001 * 131, 1e-9, 'a prefill of 128 tokens contributes 128 tokens to the same iteration, not 1');
  } },
  { step: 'cost', name: 'prefill processes the whole prompt, decode one token, recompute prompt + generated', run(m, T) {
    T.eq(m.tokensThisIteration(state(m, { promptLen: 200 })), 200, 'a fresh request prefills its whole prompt in one iteration');
    T.eq(m.tokensThisIteration(state(m, { promptLen: 200, prefilled: true, generated: 7 })), 1, 'a decode step processes exactly one token per request');
    T.eq(m.tokensThisIteration(state(m, { promptLen: 200, prefilled: false, generated: 7 })), 207,
      'a preempted request has lost its KV: to emit token 8 it must re-process the prompt plus the 7 tokens it already produced');
  } },
  { step: 'cost', name: 'kvTokens is promptLen + generated − 1 (the off-by-one that sizes the cache)', run(m, T) {
    T.eq(m.kvTokens(state(m, { promptLen: 50 })), 0, 'nothing is cached before the prefill iteration runs');
    T.eq(m.kvTokens(state(m, { promptLen: 50, prefilled: true, generated: 1 })), 50, 'after prefill the cache holds exactly the prompt');
    T.eq(m.kvTokens(state(m, { promptLen: 50, prefilled: true, generated: 4 })), 53, 'each decode adds one entry: prompt + generated − 1');
    T.eq(m.kvTokens(state(m, { promptLen: 50, outputLen: 20, prefilled: true, generated: 20 })), 69,
      'a finished request peaked at promptLen + outputLen − 1 entries; that is what you must have budgeted for');
  } },

  // ---------- step 2: static batching ----------
  { step: 'static', name: 'one request: prefill emits the first token, then outputLen − 1 decodes', run(m, T) {
    const r = m.runStatic([{ id: 0, arrival: 0.5, promptLen: 100, outputLen: 3, key: 'a' }], CFG);
    T.eq(r.completed, 1);
    T.close(r.records[0].firstToken, 0.61, 1e-9, 'the engine idles until 0.5, then one prefill iteration of 0.01 + 100·0.001');
    T.close(r.records[0].end, 0.632, 1e-9, 'two more decode iterations of 0.011 each');
    T.close(r.ttftP50, 0.11, 1e-9, 'TTFT is measured from arrival, not from the start of the run');
    T.close(r.throughput, 3 / 0.632, 1e-6, 'throughput is output tokens divided by makespan');
  } },
  { step: 'static', name: 'head-of-line blocking: a 1-token request leaves only when the batch does', run(m, T) {
    const reqs = [
      { id: 0, arrival: 0, promptLen: 16, outputLen: 1, key: 'a' },
      { id: 1, arrival: 0, promptLen: 16, outputLen: 20, key: 'b' },
    ];
    const r = m.runStatic(reqs, CFG);
    T.eq(r.completed, 2);
    T.close(r.records[0].end, r.records[1].end, 1e-9,
      'in static batching the whole batch returns together: the 1-token request waits for the 20-token one');
    T.ok(r.records[0].end > r.records[0].firstToken + 0.1,
      `the short request finished generating at ${r.records[0].firstToken.toFixed(3)} s but must leave at ${r.records[0].end.toFixed(3)} s`);
    T.eq(r.iterations, 20, 'one prefill iteration plus 19 decode iterations, sized by the LONGEST request');
    T.ok(r.batchTokens >= 32 + 19 * 2, 'finished slots are padded and still cost a token every iteration');
  } },
  { step: 'static', name: 'the next batch cannot start until the current one is completely done', run(m, T) {
    const reqs = [
      { id: 0, arrival: 0, promptLen: 16, outputLen: 30, key: 'a' },
      { id: 1, arrival: 0, promptLen: 16, outputLen: 1, key: 'b' },
      { id: 2, arrival: 0, promptLen: 16, outputLen: 1, key: 'c' },
    ];
    const r = m.runStatic(reqs, { ...CFG, maxBatch: 2 });
    T.ok(r.records[2].firstToken >= r.records[0].end,
      `request 2 arrived at t=0 but cannot be admitted until the first batch returns (${r.records[0].end.toFixed(3)} s); got a first token at ${r.records[2].firstToken.toFixed(3)} s`);
    T.eq(r.completed, 3);
    T.eq(r.outputTokens, 32);
  } },

  // ---------- step 3: continuous batching ----------
  { step: 'continuous', name: 'a finished request leaves the batch immediately (no head-of-line blocking)', run(m, T) {
    const reqs = [
      { id: 0, arrival: 0, promptLen: 16, outputLen: 1, key: 'a' },
      { id: 1, arrival: 0, promptLen: 16, outputLen: 20, key: 'b' },
    ];
    const r = m.runContinuous(reqs, CFG);
    T.close(r.records[0].end, r.records[0].firstToken, 1e-9, 'the 1-token request is retired in the iteration that produced its token');
    T.ok(r.records[1].end > r.records[0].end + 0.15,
      'the long request keeps running afterwards; the short one must not have waited for it');
    T.eq(r.iterations, 20, 'still 20 iterations, but the later ones run a batch of 1 instead of a padded batch of 2');
  } },
  { step: 'continuous', name: 'no padding: every token processed is either a prompt token or a real decode', run(m, T) {
    const reqs = skewed(T, 12);
    const r = m.runContinuous(reqs, CFG);
    const expected = reqs.reduce((s, q) => s + q.promptLen + q.outputLen - 1, 0);
    T.eq(r.completed, 12, 'every request must complete');
    T.eq(r.outputTokens, reqs.reduce((s, q) => s + q.outputLen, 0), 'every request must emit exactly outputLen tokens');
    T.eq(r.batchTokens, expected,
      `a continuous batch processes sum(promptLen) + sum(outputLen − 1) = ${expected} tokens; anything more means finished slots are still being padded`);
  } },
  { step: 'continuous', name: 'beats static on a skewed length distribution', run(m, T) {
    const reqs = skewed(T, 24);
    const cfg = { ...CFG, maxBatch: 8 };
    const s = m.runStatic(reqs, cfg);
    const c = m.runContinuous(reqs, cfg);
    T.eq(c.completed, s.completed, 'both policies must serve every request');
    T.ok(c.throughput > 1.5 * s.throughput,
      `continuous batching should be well over 1.5x static here: got ${c.throughput.toFixed(0)} vs ${s.throughput.toFixed(0)} output tokens/s`);
    T.ok(c.ttftP95 < s.ttftP95 / 2,
      `p95 TTFT should collapse when requests no longer queue behind a whole batch: got ${c.ttftP95.toFixed(3)} s vs ${s.ttftP95.toFixed(3)} s`);
  } },
  { step: 'continuous', name: 'with maxBatch = 1 the two policies degenerate to the same schedule', run(m, T) {
    const reqs = skewed(T, 8, 12);
    const cfg = { ...CFG, maxBatch: 1 };
    const s = m.runStatic(reqs, cfg), c = m.runContinuous(reqs, cfg);
    T.ok(c.makespan > 0 && c.completed === 8, 'both policies must actually run the 8 requests');
    T.close(c.makespan, s.makespan, 1e-9, 'with one slot there is nothing to batch, so the schedules must be identical');
    T.eq(c.iterations, s.iterations);
  } },

  // ---------- step 4: the paged block allocator ----------
  { step: 'allocator', name: 'blocks are whole: 17 tokens need 2 blocks of 16', run(m, T) {
    const a = new m.BlockAllocator({ numBlocks: 8, blockSize: 16 });
    T.eq(a.blocksNeeded(0), 0);
    T.eq(a.blocksNeeded(1), 1, 'one token still occupies a whole block');
    T.eq(a.blocksNeeded(16), 1);
    T.eq(a.blocksNeeded(17), 2, 'ceil, not floor: the 17th token needs a second block');
    const table = a.allocate(33);
    T.eq(table.length, 3, '33 tokens need 3 blocks');
    T.eq(new Set(table).size, 3, 'the same physical block must never be handed out twice');
    T.eq(a.freeCount, 5);
    T.eq(a.usedCount, 3);
  } },
  { step: 'allocator', name: 'allocation is all-or-nothing and reports failure instead of throwing', run(m, T) {
    const a = new m.BlockAllocator({ numBlocks: 4, blockSize: 16 });
    T.eq(a.allocate(200), null, 'a request needing 13 blocks cannot be served from 4: return null');
    T.eq(a.freeCount, 4, 'a failed allocation must not consume blocks — a half-allocated request leaks memory forever');
    const t = a.allocate(64);
    T.eq(t.length, 4);
    T.eq(a.allocate(1), null, 'nothing left');
    a.free(t);
    T.eq(a.freeCount, 4, 'freeing returns every block to the pool');
  } },
  { step: 'allocator', name: 'appendToken allocates only when the last block is full', run(m, T) {
    const a = new m.BlockAllocator({ numBlocks: 3, blockSize: 16 });
    const t = a.allocate(16);
    T.eq(t.length, 1);
    T.eq(a.appendToken(t, 15), true);
    T.eq(t.length, 1, 'token 16 fits in the block you already hold; allocating here wastes a whole block');
    T.eq(a.appendToken(t, 16), true);
    T.eq(t.length, 2, 'token 17 does not fit: the table grows by exactly one block');
    T.eq(a.appendToken(t, 32), true);
    T.eq(t.length, 3);
    T.eq(a.appendToken(t, 48), false, 'the pool is empty: report the failure so the scheduler can preempt someone');
    T.eq(t.length, 3, 'a failed append must leave the block table untouched');
  } },
  { step: 'allocator', name: 'double free is an error, not a silently corrupted free list', run(m, T) {
    const a = new m.BlockAllocator({ numBlocks: 4, blockSize: 16 });
    const t = a.allocate(32);
    a.free(t);
    T.eq(a.freeCount, 4);
    T.throws(() => a.free(t), 'freeing the same block twice would hand one block to two requests');
  } },
  { step: 'allocator', name: 'internal fragmentation is the wasted tail of the last block', run(m, T) {
    const f = m.internalFragmentation([17, 1], 16);
    T.eq(f.tokens, 18);
    T.eq(f.slots, 48, '17 tokens take 2 blocks and 1 token takes 1 block: 3 blocks of 16 slots');
    T.eq(f.wasted, 30);
    T.close(f.fraction, 30 / 48, 1e-9);
    T.close(m.internalFragmentation([32, 16], 16).fraction, 0, 1e-9, 'exact multiples waste nothing');
    const contiguous = m.internalFragmentation([17, 1], 1024);
    T.ok(contiguous.fraction > 0.98,
      `reserving a full 1024-token slab per request wastes almost everything: got ${(100 * contiguous.fraction).toFixed(1)}%`);
  } },

  // ---------- step 5: admission control and preemption ----------
  { step: 'paged', name: 'with memory to spare, paging changes nothing about the schedule', run(m, T) {
    const reqs = skewed(T, 16);
    const cfg = { ...CFG, maxBatch: 8, numBlocks: 4096 };
    const c = m.runContinuous(reqs, cfg);
    const p = m.runPaged(reqs, cfg);
    T.ok(c.completed === 16 && c.makespan > 0, 'runContinuous must serve the whole workload first');
    T.eq(p.completed, 16, 'the paged engine must serve every request too');
    T.eq(p.preemptions, 0, 'nothing should be preempted when the cache is large');
    T.close(p.makespan, c.makespan, 1e-6, 'admission control must not slow down a run that fits');
    T.eq(p.iterations, c.iterations);
    T.eq(p.outputTokens, c.outputTokens);
    T.ok(p.peakBlocks <= cfg.numBlocks, 'the allocator must never hand out more blocks than exist');
  } },
  { step: 'paged', name: 'a tight KV budget preempts, recomputes, and still finishes every request', run(m, T) {
    const reqs = skewed(T, 16);
    const roomy = { ...CFG, maxBatch: 8, numBlocks: 4096 };
    const tight = { ...CFG, maxBatch: 8, numBlocks: 14 };
    const p = m.runPaged(reqs, tight);
    T.eq(p.completed, 16, 'preemption must never drop a request');
    T.eq(p.outputTokens, reqs.reduce((s, q) => s + q.outputLen, 0), 'a preempted request still owes every one of its tokens');
    T.ok(p.preemptions > 0, `14 blocks (224 slots) cannot hold 8 concurrent requests, so somebody must be preempted; got ${p.preemptions}`);
    T.ok(p.recomputedTokens > 0, 'a request preempted by recomputation has to re-process its prompt and its own generated tokens');
    T.ok(p.peakBlocks <= 14, `the allocator handed out ${p.peakBlocks} of 14 blocks`);
    T.ok(p.makespan > m.runPaged(reqs, roomy).makespan, 'the recomputed tokens are real work: the tight run must take longer');
  } },
  { step: 'paged', name: 'a request larger than the whole cache is rejected, not livelocked', run(m, T) {
    const reqs = [{ id: 0, arrival: 0, promptLen: 200, outputLen: 200, key: 'a' }];
    T.throws(() => m.runPaged(reqs, { ...CFG, numBlocks: 4 }),
      '399 KV slots cannot fit in 4 blocks of 16, and preempting the only running request would loop forever');
    const ok = m.runPaged(reqs, { ...CFG, numBlocks: 64 });
    T.eq(ok.completed, 1, 'the same request fits in 64 blocks and must run');
  } },
  { step: 'paged', name: 'fragmentation is measured, and it is small with 16-token blocks', run(m, T) {
    const reqs = skewed(T, 16);
    const p = m.runPaged(reqs, { ...CFG, maxBatch: 8, numBlocks: 4096 });
    T.ok(p.wastedFraction > 0, 'block-granular allocation always wastes part of the last block');
    T.ok(p.wastedFraction < 0.25, `16-token blocks should waste well under a quarter of the cache; got ${(100 * p.wastedFraction).toFixed(1)}%`);
  } },

  // ---------- step 6: prefix sharing ----------
  { step: 'prefix', name: 'a repeated prompt shares its full blocks and copies only the tail', run(m, T) {
    const a = new m.BlockAllocator({ numBlocks: 100, blockSize: 16 });
    const cache = new Map();
    const first = m.allocateShared(a, cache, 'sys', 40);
    T.eq(first.sharedBlocks, 0, 'nothing to share the first time');
    T.eq(first.newBlocks, 3, '40 tokens need 3 blocks');
    T.eq(a.freeCount, 97);
    const second = m.allocateShared(a, cache, 'sys', 40);
    T.eq(second.sharedBlocks, 2, 'the two FULL blocks are identical for both requests and can be shared');
    T.eq(second.newBlocks, 1, 'the partially filled last block must be copied: both requests will write their own tokens into it');
    T.eq(second.blocks.length, 3, 'the block table still addresses all 40 tokens');
    T.eq(second.blocks[0], first.blocks[0], 'shared blocks are the same physical blocks');
    T.eq(a.freeCount, 96, 'the second request cost 1 block instead of 3');
  } },
  { step: 'prefix', name: 'a prompt that ends on a block boundary shares everything', run(m, T) {
    const a = new m.BlockAllocator({ numBlocks: 100, blockSize: 16 });
    const cache = new Map();
    m.allocateShared(a, cache, 'sys', 32);
    const before = a.freeCount;
    const second = m.allocateShared(a, cache, 'sys', 32);
    T.eq(second.sharedBlocks, 2);
    T.eq(second.newBlocks, 0, '32 tokens are two full blocks; the next token will allocate a fresh block through appendToken');
    T.eq(a.freeCount, before, 'a fully shared prompt costs no new memory at all');
    const other = m.allocateShared(a, cache, 'different-prompt', 32);
    T.eq(other.sharedBlocks, 0, 'different prompts must never share blocks');
    T.eq(other.newBlocks, 2);
  } },
  { step: 'prefix', name: 'reference counts keep a shared block alive until the last user frees it', run(m, T) {
    const a = new m.BlockAllocator({ numBlocks: 100, blockSize: 16 });
    const cache = new Map();
    const r1 = m.allocateShared(a, cache, 'sys', 40);
    const r2 = m.allocateShared(a, cache, 'sys', 40);
    a.free(r2.blocks);
    T.eq(a.freeCount, 97, 'only r2\'s private tail block comes back; the two shared blocks are still in use by r1');
    a.free(r1.blocks);
    T.eq(a.freeCount, 98, 'r1\'s tail comes back too, but the cache still holds a reference to the 2 shared blocks');
    T.eq(cache.get('sys').length, 2, 'the cached prefix survives both requests — that is what makes the next one free');
  } },
  { step: 'prefix', name: 'out of memory returns null without consuming blocks', run(m, T) {
    const a = new m.BlockAllocator({ numBlocks: 3, blockSize: 16 });
    const cache = new Map();
    T.eq(m.allocateShared(a, cache, 'sys', 100), null, '100 tokens need 7 blocks and only 3 exist');
    T.eq(a.freeCount, 3, 'a failed shared allocation must leave the pool untouched');
    T.eq(cache.size, 0, 'and must not cache a prefix it never allocated');
    const fits = m.allocateShared(a, cache, 'sys', 33);
    T.eq(fits.newBlocks, 3, '33 tokens do fit in 3 blocks and must be allocated');
    T.eq(a.freeCount, 0);
  } },
];
