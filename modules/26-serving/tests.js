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

/** p95 of TTFT / TPOT computed from run records here, so steps 3 and 4 do not depend on your step-5 sloReport. */
function refP95(values) {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  const i = 0.95 * (s.length - 1), lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}
const ttftP95 = (records) => refP95(records.map((r) => r.firstToken - r.arrival));
const tpotP95 = (records) => refP95(records.filter((r) => r.outputLen > 1).map((r) => (r.end - r.firstToken) / (r.outputLen - 1)));

/** Reference ring lookup built from hash32 alone, so the expected value does not depend on the learner's ring. */
function refHome(ids, vnodes, key) {
  const pts = [];
  for (const id of ids) for (let v = 0; v < vnodes; v++) pts.push({ hash: hash32(`${id}#${v}`), id });
  pts.sort((a, b) => a.hash - b.hash);
  const h = hash32(String(key));
  const p = pts.find((x) => x.hash >= h);
  return (p || pts[0]).id;
}

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
  { step: 'routing', name: 'buildRing places vnodes points per id at hash32(`id#v`); pickOnRing takes the first point >= the key, wrapping', run(m, T) {
    const ring = m.buildRing(['a', 'b'], { vnodes: 3 });
    T.eq(ring.length, 6, '2 replicas x 3 virtual nodes = 6 points');
    const want = [];
    for (const id of ['a', 'b']) for (let v = 0; v < 3; v++) want.push({ hash: hash32(`${id}#${v}`), id });
    want.sort((x, y) => x.hash - y.hash);
    T.eq(ring.map((p) => [p.hash, p.id]), want.map((p) => [p.hash, p.id]), 'each point is { hash: hash32(`${id}#${v}`), id }, sorted by hash ascending; every router in the fleet must build the same ring');
    T.throws(() => m.buildRing([], { vnodes: 3 }), 'a ring with no replicas cannot route anything and must throw');
    const h = hash32('route-me');
    T.eq(m.pickOnRing([{ hash: h - 1, id: 'x' }, { hash: h, id: 'y' }, { hash: h + 1, id: 'z' }], 'route-me'), 'y',
      'a point whose hash EQUALS the key hash owns the key (>=, not >)');
    T.eq(m.pickOnRing([{ hash: h + 5, id: 'x' }, { hash: h + 9, id: 'y' }], 'route-me'), 'x', 'the first point at or after the key hash, not the nearest or the last');
    T.eq(m.pickOnRing([{ hash: h - 9, id: 'x' }, { hash: h - 5, id: 'y' }], 'route-me'), 'x',
      'every point is below the key hash, so the key wraps around the circle to ring[0], not to the last point');
    T.throws(() => m.pickOnRing([], 'route-me'), 'an empty ring must throw');
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
    const twoWarm = [loaded(m, 0, 100, ['sys-b']), loaded(m, 1, 60, ['sys-b']), loaded(m, 2, 50)];
    T.eq(m.chooseReplica('cache-aware', twoWarm, { prefix: 'sys-b' }, ctx), 1,
      'replicas 0 (100 tokens) and 1 (60) both hold the prefix and both are inside the 1.5x band (mean 70): take the less loaded holder, not the first one found, and not the emptier replica 2 that would have to recompute it');
  } },
  { step: 'routing', name: 'a cold prefix goes to its ring home, not to the least-loaded replica, unless home is overloaded', run(m, T) {
    const ids = [0, 1, 2, 3];
    const ctx = { dispatched: 0, ring: m.buildRing(ids, { vnodes: 64 }), cfg: CFG(m) };
    let checked = 0;
    for (let k = 0; k < 40 && checked < 3; k++) {
      const prefix = `cold-${k}`;
      const home = refHome(ids, 64, prefix);
      if (home === 0) continue;   // replica 0 is the least loaded below; we want the ring and least-loaded to disagree
      checked++;
      const within = ids.map((id) => loaded(m, id, id === 0 ? 20 : id === home ? 30 : 40));
      T.eq(m.chooseReplica('cache-aware', within, { prefix }, ctx), home,
        `nobody holds "${prefix}"; its ring home is replica ${home} with 30 tokens against a mean of 32.5, inside the band, so it must go home even though replica 0 is emptier — that is how the NEXT request with this prefix finds a warm cache`);
      const hot = ids.map((id) => loaded(m, id, id === home ? 900 : 10));
      T.eq(m.chooseReplica('cache-aware', hot, { prefix }, ctx), 0,
        `the ring home (replica ${home}) owes 900 tokens against a mean of 232: the same overload guard applies, so fall back to least-loaded (index 0)`);
    }
    T.ok(checked === 3, 'fixture: expected at least three cold prefixes whose home is not replica 0');
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
  { step: 'disagg', name: 'planPools prices each phase exactly as specified and rounds to the nearest machine', run(m, T) {
    const cfg = { ...CFG(m), tPerToken: 1, tFixed: 32, maxBatch: 32 };   // decode token = 1 + 32/32 = 2 units
    const reqs = [
      { id: 0, arrival: 0, prefix: 'p', prefixLen: 500, promptLen: 1000, outputLen: 11 },
      { id: 1, arrival: 0, prefix: 'p', prefixLen: 100, promptLen: 200, outputLen: 1 },
    ];
    const p = m.planPools(reqs, 4, cfg);
    T.close(p.prefillWork, 1200, 1e-9, 'prefill work is promptLen x tPerToken summed over requests (1000 + 200); the planner does not know which requests will hit the cache');
    T.close(p.decodeWork, 20, 1e-9, 'decode work is (outputLen - 1) x (tPerToken + tFixed / maxBatch): 10 tokens x 2 = 20, and a 1-token answer has no decode at all');
    const unit = { ...CFG(m), tPerToken: 1, tFixed: 0, maxBatch: 1 };
    const split = m.planPools([
      { id: 0, arrival: 0, prefix: 'p', prefixLen: 0, promptLen: 65, outputLen: 1 },
      { id: 1, arrival: 0, prefix: 'p', prefixLen: 0, promptLen: 0, outputLen: 36 },
    ], 4, unit);
    T.eq([split.prefill, split.decode], [3, 1], '65% of the work is prefill: 0.65 x 4 = 2.6 rounds to 3 prefill machines, not down to 2');
  } },
  { step: 'disagg', name: 'disaggregation removes the prefill stalls that colocation forces on decoding', run(m, T) {
    const work = m.makeWorkload({ n: 150, seconds: 20, seed: 4, nPrefixes: 6 });
    const co = m.runCluster(work, { policy: 'least-loaded', replicas: 4 });
    const pools = m.planPools(work, 4, CFG(m));
    const di = m.runCluster(work, { policy: 'least-loaded', disaggregate: true, pools });
    T.eq(di.records.length, work.length, 'every request must still finish when the pools are split');
    T.ok(co.stallSeconds > 0, 'on a colocated replica each prefill step stalls every decode in flight; that time must be counted');
    T.eq(di.stallSeconds, 0, 'a decode pool never runs a prefill, so nothing stalls there');
    // p95 TPOT is computed here from the records (TPOT = (end - firstToken) / (outputLen - 1)), not with your sloReport.
    const coP = tpotP95(co.records), diP = tpotP95(di.records);
    T.ok(diP < coP, `p95 TPOT was ${coP.toFixed(4)} s colocated and ${diP.toFixed(4)} s disaggregated; separating the pools must improve it. If both runs look right, check that planPools gives the prefill pool most of these prompt-heavy replicas and that kvTransferSeconds is milliseconds, not seconds`);
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
    T.eq(m.autoscaleTarget({ total: 8, raw: 2, window: [6, 2, 2] }, { ...cfg, scaleDownStep: 3 }), 6,
      'a big scaleDownStep may not undershoot what the window still asks for: max(6, 8 - 3) = 6, not 5');
  } },
  { step: 'autoscale', name: 'gpuSeconds bills every replica from the moment it is created', run(m, T) {
    T.close(m.gpuSeconds([{ start: 0, end: 10 }, { start: 4, end: 10 }], 10), 16, 1e-9, 'two spans of 10 s and 6 s');
    T.close(m.gpuSeconds([{ start: 0, end: null }, { start: 90, end: null }], 100), 110, 1e-9, 'a replica still running at the end is billed up to endTime, including the seconds it spent loading weights');
    T.close(m.gpuSeconds([], 100), 0, 1e-9, 'no replicas, no bill');
    T.close(m.gpuSeconds([{ start: 5, end: 5 }], 100), 0, 1e-9, 'a zero-length span costs nothing');
    T.close(m.gpuSeconds([{ start: 0, end: 10 }, { start: 20, end: 15 }], 100), 10, 1e-9, 'a span that ends before it starts is ignored, not billed as negative time');
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
    // p95 TTFT is computed here from the records (firstToken - arrival), not with your sloReport.
    const autoP = ttftP95(res.records), fixedP = ttftP95(fixed.records);
    T.ok(autoP < fixedP,
      `p95 TTFT was ${autoP.toFixed(2)} s autoscaled and ${fixedP.toFixed(2)} s on a fixed single replica; adding replicas must actually lower it against the same burst (peak ${res.peakReplicas} replicas)`);
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
    T.close(r.throughput, 144 / 5, 1e-9, 'throughput is output TOKENS per second of makespan (144 / 5), not requests per second');
    T.close(r.tpotP95, 0.185, 1e-6, 'p95 TPOT interpolates between 0.1 and 0.2');
  } },
  { step: 'slo', name: 'one-token answers count for TTFT only, and an SLO is met AT its threshold', run(m, T) {
    const recs = [
      { id: 0, arrival: 0, firstToken: 0.5, end: 0.5, outputLen: 1 },
      { id: 1, arrival: 0, firstToken: 0.2, end: 1.2, outputLen: 11 },
    ];
    const r = m.sloReport(recs, { ttft: 1, tpot: 0.05 });
    T.close(r.tpotP50, 0.1, 1e-9, 'the one-token answer has no inter-token interval, so it must not add a TPOT of 0 to the list');
    T.eq(r.met, 1, 'the one-token answer met its TTFT and has no TPOT to fail, so it counts; request 1 streams at 0.1 s/token and does not');
    const edge = m.sloReport([{ id: 0, arrival: 0, firstToken: 1, end: 1.5, outputLen: 11 }], { ttft: 1, tpot: 0.05 });
    T.eq(edge.met, 1, 'TTFT of exactly 1 s and TPOT of exactly 50 ms meet a 1 s / 50 ms SLO: the check is <=, not <');
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
