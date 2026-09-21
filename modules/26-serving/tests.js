import { hash32 } from 'lib/util.js';

// --- fixtures ---------------------------------------------------------------

function state(m, over = {}) {
  const base = m.newRequestState({ id: 0, arrival: 0, prefix: 'p0', prefixLen: 64, promptLen: 100, outputLen: 10 });
  return Object.assign(base, over);
}

function replicaWith(m, id, states, opts = {}) {
  const r = m.makeReplica(id, opts);
  for (const s of states) r.running.push(s);
  return r;
}

/** A replica whose load is exactly `load` tokens, holding `prefixes` in its cache. */
function loaded(m, id, load, prefixes = []) {
  const r = m.makeReplica(id);
  if (load > 0) r.queue.push(state(m, { promptLen: load, outputLen: 0, prefixLen: 0 }));
  for (const p of prefixes) r.cache.set(p, 0);
  return r;
}

const CFG = (m) => m.DEFAULT_CONFIG;

export const tests = [
  // ---------------- step 1: the replica cost model ----------------
  { step: 'cost', name: 'a request contributes its uncached prompt during prefill and 1 token while decoding', run(m, T) {
    T.eq(m.tokensThisIteration(state(m, { promptLen: 100, cached: 0 })), 100, 'an unprefilled request costs its whole prompt in the iteration that prefills it');
    T.eq(m.tokensThisIteration(state(m, { promptLen: 100, cached: 64 })), 36, 'a prefix-cache hit of 64 tokens leaves 36 tokens of prefill to compute');
    T.eq(m.tokensThisIteration(state(m, { promptLen: 100, prefilled: true, generated: 3 })), 1, 'decoding is one token per iteration however long the context is');
  } },
  { step: 'cost', name: 'iteration time is tFixed + tPerToken x tokens in the batch', run(m, T) {
    const cfg = CFG(m);
    const decode = [state(m, { prefilled: true }), state(m, { prefilled: true }), state(m, { prefilled: true }), state(m, { prefilled: true })];
    T.close(m.iterationSeconds(decode, cfg), cfg.tFixed + 4 * cfg.tPerToken, 1e-12, 'four decoding requests move 4 tokens, so the fixed weight-reading cost dominates');
    T.close(m.iterationSeconds([state(m, { promptLen: 2000 })], cfg), cfg.tFixed + 2000 * cfg.tPerToken, 1e-12, 'one 2000-token prefill costs 2000 x tPerToken on top of tFixed');
    T.close(m.iterationSeconds([], cfg), cfg.tFixed, 1e-12, 'an empty batch still pays the fixed cost; the term must not be dropped');
    T.ok(m.iterationSeconds(decode, cfg) < m.iterationSeconds([state(m, { promptLen: 2000 })], cfg), 'a 2000-token prefill must cost far more than a 4-request decode step');
  } },
  { step: 'cost', name: 'replicaLoad counts work still owed, not work already done', run(m, T) {
    T.eq(m.remainingWork(state(m, { promptLen: 100, outputLen: 10 })), 110, 'nothing done yet: 100 prompt tokens + 10 output tokens');
    T.eq(m.remainingWork(state(m, { promptLen: 100, outputLen: 10, prefilled: true, generated: 4 })), 6, 'prefill is done and 4 of 10 tokens are out, so 6 tokens are owed');
    T.eq(m.remainingWork(state(m, { promptLen: 100, cached: 64, outputLen: 10 })), 46, 'a cache hit removes 64 tokens of prompt from the work owed');
    const r = replicaWith(m, 0, [state(m, { promptLen: 100, outputLen: 10, prefilled: true, generated: 4 })]);
    r.queue.push(state(m, { promptLen: 100, outputLen: 10 }));
    T.eq(m.replicaLoad(r), 116, 'load is the sum over the queue AND the running batch (6 + 110)');
    T.eq(m.replicaLoad(m.makeReplica(1)), 0, 'an idle replica has zero load');
  } },

  // ---------------- step 2: routing ----------------
  { step: 'routing', name: 'round-robin cycles through the candidates and ignores load', run(m, T) {
    const cands = [loaded(m, 0, 9000), loaded(m, 1, 0), loaded(m, 2, 0)];
    const seen = [];
    for (let d = 0; d < 6; d++) seen.push(m.chooseReplica('round-robin', cands, { prefix: 'p' }, { dispatched: d, cfg: CFG(m) }));
    T.eq(seen, [0, 1, 2, 0, 1, 2], 'round-robin must walk the candidate list in order, even though replica 0 is swamped');
  } },
  { step: 'routing', name: 'least-loaded picks the smallest load, breaking ties by position', run(m, T) {
    const cands = [loaded(m, 0, 500), loaded(m, 1, 120), loaded(m, 2, 120)];
    T.eq(m.chooseReplica('least-loaded', cands, { prefix: 'p' }, { dispatched: 3, cfg: CFG(m) }), 1, 'replica 1 owes 120 tokens against replica 0\'s 500');
    T.eq(m.chooseReplica('least-loaded', [loaded(m, 0, 7), loaded(m, 1, 7)], { prefix: 'p' }, { cfg: CFG(m) }), 0, 'on a tie take the first candidate so the choice is deterministic');
    T.throws(() => m.chooseReplica('least-loaded', [], { prefix: 'p' }, { cfg: CFG(m) }), 'with no eligible replica the balancer must throw rather than route into the void');
    T.throws(() => m.chooseReplica('random-guess', [loaded(m, 0, 0)], { prefix: 'p' }, { cfg: CFG(m) }), 'an unknown policy name must throw, not silently fall back');
  } },
  { step: 'routing', name: 'the hash ring is deterministic, balanced, and only remaps the keys of a removed replica', run(m, T) {
    const keys = [];
    for (let i = 0; i < 600; i++) keys.push(`prefix-${i}`);
    const ring8 = m.buildRing([0, 1, 2, 3, 4, 5, 6, 7], { vnodes: 64 });
    const home = keys.map((k) => m.pickOnRing(ring8, k));
    T.eq(home, keys.map((k) => m.pickOnRing(ring8, k)), 'the same key must always land on the same replica');
    T.ok(home.every((id) => id >= 0 && id <= 7), 'the ring may only return ids it was built from');
    const counts = new Array(8).fill(0);
    for (const id of home) counts[id]++;
    T.ok(new Set(home).size >= 6, `only ${new Set(home).size} of 8 replicas received any key; 64 virtual nodes each must spread the ring`);
    T.ok(Math.max(...counts) <= 0.45 * keys.length, `one replica took ${Math.max(...counts)} of ${keys.length} keys; the ring must not collapse onto one node`);
    const ring7 = m.buildRing([0, 1, 2, 3, 4, 5, 6], { vnodes: 64 });
    let moved = 0, movedOffSurvivor = 0;
    keys.forEach((k, i) => {
      const after = m.pickOnRing(ring7, k);
      if (after !== home[i]) { moved++; if (home[i] !== 7) movedOffSurvivor++; }
    });
    T.eq(movedOffSurvivor, 0, 'removing replica 7 must not move keys that were living on replicas 0-6 — that is the whole point of a ring');
    const modMoved = keys.filter((k) => hash32(k) % 8 !== hash32(k) % 7).length;
    T.ok(moved < 0.5 * modMoved, `the ring moved ${moved} keys where hash % n moves ${modMoved}; a ring must beat modulo by a wide margin`);
  } },
  { step: 'routing', name: 'cache-aware routing follows the prefix until the holder is overloaded', run(m, T) {
    const ctx = { dispatched: 0, ring: m.buildRing([0, 1, 2], { vnodes: 64 }), cfg: CFG(m) };
    const warm = [loaded(m, 0, 100), loaded(m, 1, 180, ['sys-a']), loaded(m, 2, 100)];
    T.eq(m.chooseReplica('cache-aware', warm, { prefix: 'sys-a' }, ctx), 1,
      'replica 1 already holds the prefix and owes 180 tokens against a mean of 127, inside the 1.5x band: keep the affinity even though replica 0 is emptier');
    const hot = [loaded(m, 0, 10), loaded(m, 1, 5000, ['sys-a']), loaded(m, 2, 10)];
    T.eq(m.chooseReplica('cache-aware', hot, { prefix: 'sys-a' }, ctx), 0, 'the holder owes 5000 tokens against a mean of 1673; affinity must yield to load or one replica becomes the queue for everybody');
    const cold = [loaded(m, 0, 10), loaded(m, 1, 10), loaded(m, 2, 10)];
    const first = m.chooseReplica('cache-aware', cold, { prefix: 'never-seen' }, ctx);
    T.eq(m.chooseReplica('cache-aware', cold, { prefix: 'never-seen' }, { ...ctx, dispatched: 17 }), first,
      'an uncached prefix must go to its ring home every time, otherwise the second request cannot hit the first one\'s cache');
  } },

  // ---------------- step 3: disaggregation ----------------
  { step: 'disagg', name: 'KV transfer time is a setup cost plus bytes / bandwidth', run(m, T) {
    const cfg = CFG(m);
    T.close(m.kvTransferSeconds(0, cfg), cfg.kvSetupSeconds, 1e-12, 'a zero-token transfer still pays the setup cost');
    T.close(m.kvTransferSeconds(1024, cfg), cfg.kvSetupSeconds + (1024 * cfg.kvBytesPerToken) / cfg.kvBandwidth, 1e-12,
      '1024 tokens x 128 KB = 134 MB; at 25 GB/s that is 5.4 ms');
    const a = m.kvTransferSeconds(1000, cfg) - cfg.kvSetupSeconds;
    const b = m.kvTransferSeconds(2000, cfg) - cfg.kvSetupSeconds;
    T.close(b / a, 2, 1e-6, 'the variable part must be linear in tokens; a constant transfer time hides the whole cost of disaggregation');
    T.throws(() => m.kvTransferSeconds(-1, cfg), 'a negative token count is a bug and must throw');
  } },
  { step: 'disagg', name: 'planPools follows the work, not a fixed 50/50 split', run(m, T) {
    const promptHeavy = [], outputHeavy = [];
    for (let i = 0; i < 50; i++) {
      promptHeavy.push({ id: i, arrival: i, prefix: 'p', prefixLen: 0, promptLen: 4000, outputLen: 8 });
      outputHeavy.push({ id: i, arrival: i, prefix: 'p', prefixLen: 0, promptLen: 40, outputLen: 600 });
    }
    const a = m.planPools(promptHeavy, 8, CFG(m));
    T.ok(a.prefill >= 6, `long prompts and 8-token answers need prefill machines; got ${a.prefill} prefill / ${a.decode} decode`);
    T.eq(a.prefill + a.decode, 8, 'every replica must be in exactly one pool');
    const b = m.planPools(outputHeavy, 8, CFG(m));
    T.ok(b.decode >= 6, `40-token prompts with 600-token answers need decode machines; got ${b.prefill} prefill / ${b.decode} decode`);
    T.ok(b.prefill >= 1 && b.decode >= 1, 'neither pool may be empty: an empty pool serves nobody');
    T.throws(() => m.planPools(promptHeavy, 1, CFG(m)), 'you cannot disaggregate onto a single replica');
  } },
  { step: 'disagg', name: 'disaggregation removes the prefill stalls that colocation forces on decoding', run(m, T) {
    const work = m.makeWorkload({ n: 150, seconds: 20, seed: 4, nPrefixes: 6 });
    const co = m.runCluster(work, { policy: 'least-loaded', replicas: 4 });
    const pools = m.planPools(work, 4, CFG(m));
    const di = m.runCluster(work, { policy: 'least-loaded', disaggregate: true, pools });
    T.eq(di.records.length, work.length, 'every request must still finish when the pools are split');
    T.ok(co.stallSeconds > 0, 'on a colocated replica each prefill step stalls every decode in flight; that time must be counted');
    T.eq(di.stallSeconds, 0, 'a decode pool never runs a prefill, so nothing stalls there');
    const coT = m.sloReport(co.records, { ttft: 1, tpot: 0.01 });
    const diT = m.sloReport(di.records, { ttft: 1, tpot: 0.01 });
    T.ok(diT.tpotP95 < coT.tpotP95, `p95 TPOT was ${coT.tpotP95.toFixed(4)} s colocated and ${diT.tpotP95.toFixed(4)} s disaggregated; separating the pools must improve it`);
    T.ok(di.handoffs > 0 && di.transferSecondsTotal > 0, 'every request that decodes elsewhere must pay a KV transfer');
  } },

  // ---------------- step 4: autoscaling ----------------
  { step: 'autoscale', name: 'queueTarget converts outstanding work into a replica count', run(m, T) {
    const cfg = { ...CFG(m), targetQueueTokens: 1000, minReplicas: 2, maxReplicas: 8 };
    T.eq(m.queueTarget({ pendingTokens: 0, ready: 5, total: 5 }, cfg), 2, 'with nothing outstanding the floor is minReplicas, not zero');
    T.eq(m.queueTarget({ pendingTokens: 3200, ready: 2, total: 2 }, cfg), 4, '3200 tokens at 1000 per replica needs 4 replicas: round up, a partial replica cannot serve anyone');
    T.eq(m.queueTarget({ pendingTokens: 3000, ready: 2, total: 2 }, cfg), 3, 'exactly 3 replicas of work needs exactly 3');
    T.eq(m.queueTarget({ pendingTokens: 900000, ready: 2, total: 2 }, cfg), 8, 'the ceiling is maxReplicas; past that you shed load instead');
  } },
  { step: 'autoscale', name: 'the controller scales up at once and down only when the window agrees', run(m, T) {
    const cfg = { ...CFG(m), minReplicas: 2, maxReplicas: 8, scaleDownStep: 1, stabilizationTicks: 6 };
    T.eq(m.autoscaleTarget({ total: 2, raw: 6, window: [2, 2, 2, 6] }, cfg), 6, 'a backlog is an emergency: go straight to the raw target');
    T.eq(m.autoscaleTarget({ total: 8, raw: 2, window: [8, 7, 2, 2] }, cfg), 8, 'the window still contains a demand for 8, so hold at 8 — the load was there 20 seconds ago');
    T.eq(m.autoscaleTarget({ total: 8, raw: 2, window: [2, 2, 2, 2, 2, 2] }, cfg), 7, 'once the whole window agrees, shrink by scaleDownStep (1), not all the way to 2');
    T.eq(m.autoscaleTarget({ total: 4, raw: 4, window: [4, 4] }, cfg), 4, 'a matched target must not move the cluster at all');
  } },
  { step: 'autoscale', name: 'gpuSeconds bills every replica from the moment it is created', run(m, T) {
    T.close(m.gpuSeconds([{ start: 0, end: 10 }, { start: 4, end: 10 }], 10), 16, 1e-9, 'two spans of 10 s and 6 s');
    T.close(m.gpuSeconds([{ start: 0, end: null }, { start: 90, end: null }], 100), 110, 1e-9, 'a replica still running at the end is billed up to endTime, including the seconds it spent loading weights');
    T.close(m.gpuSeconds([], 100), 0, 1e-9, 'no replicas, no bill');
    T.close(m.gpuSeconds([{ start: 5, end: 5 }], 100), 0, 1e-9, 'a zero-length span costs nothing');
  } },
  { step: 'autoscale', name: 'the autoscaler reacts to a burst within the cold start plus one tick', run(m, T) {
    const burst = [];
    for (let i = 0; i < 400; i++) burst.push({ id: i, arrival: 0.005 * i, prefix: `burst-${i}`, prefixLen: 0, promptLen: 2000, outputLen: 48 });
    const cfg = { ...CFG(m), minReplicas: 1, maxReplicas: 6, targetQueueTokens: 2000, coldStartSeconds: 5, scaleIntervalSeconds: 5 };
    const res = m.runCluster(burst, { policy: 'least-loaded', autoscale: true, replicas: 1, cfg });
    T.ok(res.peakReplicas > 1, 'a 400-request burst on one replica must make the autoscaler add capacity');
    const firstGrowth = res.scaleTrace.find((x) => x.desired > 1);
    T.ok(firstGrowth && firstGrowth.t <= cfg.scaleIntervalSeconds + 1e-9, `the controller asked for more replicas at t=${firstGrowth ? firstGrowth.t.toFixed(1) : 'never'} s; it must react on its first tick`);
    const fixed = m.runCluster(burst, { policy: 'least-loaded', replicas: 1, cfg });
    T.ok(m.sloReport(res.records, { ttft: 2, tpot: 0.02 }).ttftP95 < m.sloReport(fixed.records, { ttft: 2, tpot: 0.02 }).ttftP95,
      'adding replicas must actually lower p95 TTFT against the same burst on a fixed single replica');
    T.ok(res.gpuSeconds > fixed.gpuSeconds * 0.5, 'the extra replicas must be billed: GPU-seconds cannot fall when you add machines');
  } },

  // ---------------- step 5: the SLO report ----------------
  { step: 'slo', name: 'the report computes TTFT and TPOT percentiles over finished requests', run(m, T) {
    const recs = [
      { id: 0, arrival: 0, firstToken: 0.4, end: 1.4, outputLen: 21 },
      { id: 1, arrival: 1, firstToken: 1.2, end: 3.2, outputLen: 11 },
      { id: 2, arrival: 2, firstToken: 4.0, end: 5.0, outputLen: 11 },
      { id: 3, arrival: 3, firstToken: 3.1, end: 4.1, outputLen: 101 },
    ];
    const r = m.sloReport(recs, { ttft: 1, tpot: 0.05 });
    T.eq(r.n, 4);
    T.close(r.ttftP50, 0.3, 1e-6, 'TTFTs are 0.4, 0.2, 2.0, 0.1 -> median 0.3');
    T.close(r.ttftP95, 1.76, 1e-6, 'p95 interpolates between the top two values (0.4 and 2.0), it is not just the maximum');
    T.close(r.tpotP50, 0.075, 1e-6, 'TPOTs are 0.05, 0.2, 0.1, 0.01 -> median 0.075');
    T.close(r.makespan, 5, 1e-9, 'makespan runs from the first arrival to the last completion');
    T.eq(r.outputTokens, 144, 'every generated token counts towards throughput');
  } },
  { step: 'slo', name: 'goodput counts only requests that meet BOTH SLOs', run(m, T) {
    const recs = [
      { id: 0, arrival: 0, firstToken: 0.4, end: 1.4, outputLen: 21 },
      { id: 1, arrival: 1, firstToken: 1.2, end: 3.2, outputLen: 11 },
      { id: 2, arrival: 2, firstToken: 4.0, end: 5.0, outputLen: 11 },
      { id: 3, arrival: 3, firstToken: 3.1, end: 4.1, outputLen: 101 },
    ];
    const r = m.sloReport(recs, { ttft: 1, tpot: 0.05 });
    T.eq(r.met, 2, 'request 1 streams at 0.2 s/token and request 2 waited 2 s for its first token; both are failures a user notices');
    T.close(r.attainment, 0.5, 1e-9, 'attainment is met / n');
    T.close(r.goodput, 0.4, 1e-9, 'goodput is 2 good requests over a 5 s makespan, NOT 4/5');
    const loose = m.sloReport(recs, { ttft: 10, tpot: 10 });
    T.eq(loose.met, 4, 'with generous SLOs every request counts');
    const empty = m.sloReport([], { ttft: 1, tpot: 0.05 });
    T.eq(empty.n, 0);
    T.ok(!Number.isNaN(empty.goodput) && empty.goodput === 0, 'an empty window must report 0, not NaN — the demo charts a report per 10-second window');
  } },
  { step: 'slo', name: 'cost per million tokens turns GPU-seconds into the number the business sees', run(m, T) {
    T.close(m.costPerMillionTokens(3600, 1e6, 3), 3, 1e-9, 'one GPU-hour at $3 producing exactly a million tokens costs $3 per million');
    T.close(m.costPerMillionTokens(1800, 1e6, 3), 1.5, 1e-9, 'halve the GPU-seconds, halve the price');
    T.close(m.costPerMillionTokens(3600, 2e6, 3), 1.5, 1e-9, 'double the tokens on the same hardware, halve the price');
    T.throws(() => m.costPerMillionTokens(3600, 0, 3), 'dividing by zero tokens must throw rather than return Infinity');
  } },
];
