// Module 26 — Serving at scale: routing, disaggregation and autoscaling across a fleet of replicas.
// Everything above the "step 1" line is done for you, including the event loop `runCluster`, which calls
// the functions you write. Read it before you start: it is the contract for every shape you touch.
//
// Symbols used throughout:
//   promptLen  tokens in the prompt (prefix + user turn)
//   prefixLen  tokens of the prompt that belong to a SHARED prefix (system prompt, few-shot block, RAG template)
//   outputLen  tokens the request will generate, including the first one produced by prefill
//   load       tokens of work a replica still owes (unprefilled prompt tokens + ungenerated output tokens)
//   TTFT       time to first token = firstToken - arrival
//   TPOT       time per output token = (end - firstToken) / (outputLen - 1)

import { rng, hash32, sampleIndex, sumArray } from 'lib/util.js';

// ---------- conventions and worked code (read these; they set the shapes everything else uses) ----------

export const DEFAULT_CONFIG = {
  // One replica = one GPU running one copy of an 8B-class model. Same linear cost model as module 16:
  // an iteration costs tFixed (weight traffic, memory-bound) + tPerToken per token in the batch.
  tFixed: 0.005,          // s  — 16 GB of bf16 weights / approximately 3.35 TB/s HBM3 (NVIDIA H100 datasheet)
  tPerToken: 0.00005,     // s  — 50 us per token of prefill or decode work
  maxBatch: 32,           // requests per iteration
  maxBatchTokens: 2048,   // tokens per iteration (vLLM calls this max_num_batched_tokens)
  maxQueue: 64,           // requests one replica will hold; past that the balancer keeps them (backpressure)
  cacheSlots: 4,          // distinct prefixes a replica keeps hot, LRU
  overloadFactor: 1.5,    // cache affinity is dropped when a replica is this much busier than the mean
  kvBytesPerToken: 131072,// 128 KB/token: 32 layers x 8 KV heads x 128 dim x 2 (K and V) x 2 bytes (Llama-3-8B, GQA)
  kvBandwidth: 25e9,      // B/s — approximately 200 Gb/s of usable RDMA bandwidth per NIC
  kvSetupSeconds: 0.0005, // s  — fixed cost of one transfer (the alpha of module 24's alpha-beta model)
  vnodes: 64,             // virtual nodes per replica on the hash ring
  scaleIntervalSeconds: 5,
  coldStartSeconds: 30,   // approximately how long an 8B replica takes to pull weights and warm up
  targetQueueTokens: 1500,// tokens of outstanding work one replica is expected to absorb
  minReplicas: 2,
  maxReplicas: 8,
  scaleDownStep: 1,       // replicas removed per autoscaler tick (scale up fast, down slowly)
  stabilizationTicks: 6,  // ticks that must all agree before scaling down (Kubernetes HPA's downscale stabilization)
  dollarsPerGpuHour: 3,   // example rate; public H100 prices vary by roughly 2x between providers
};

/** Linear-interpolated percentile of a list of numbers. percentile([1,2,3,4], 50) === 2.5. */
export function percentile(values, p) {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  const i = (p / 100) * (s.length - 1);
  const lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}

/**
 * A synthetic hour of traffic. Prefix popularity is Zipf (module 00), output lengths are heavy-tailed,
 * and the arrival rate follows a single diurnal hump `1 + peak*sin(pi*x)` over `seconds`.
 * A request is { id, arrival, prefix, prefixLen, promptLen, outputLen }.
 */
export function makeWorkload({
  n = 400, seconds = 60, seed = 5, nPrefixes = 16, alpha = 1.0, peak = 3,
  prefixMin = 1024, prefixMax = 4096, userMin = 32, userMax = 512,
  outMin = 16, outMax = 1024, outAlpha = 1.2,
} = {}) {
  const next = rng(seed);
  const catalogue = [];
  for (let k = 0; k < nPrefixes; k++) {
    catalogue.push({
      key: `prefix-${k}`,
      weight: 1 / Math.pow(k + 1, alpha),
      len: prefixMin + Math.floor(next() * (prefixMax - prefixMin + 1)),
    });
  }
  const z = sumArray(catalogue.map((c) => c.weight));
  const probs = catalogue.map((c) => c.weight / z);
  // Inverse CDF of the arrival-rate profile, on a 512-point grid.
  const G = 512;
  const cdf = new Float64Array(G + 1);
  for (let i = 1; i <= G; i++) cdf[i] = cdf[i - 1] + (1 + peak * Math.sin((Math.PI * (i - 0.5)) / G));
  for (let i = 0; i <= G; i++) cdf[i] /= cdf[G];
  const invCdf = (u) => {
    let lo = 0, hi = G;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
    return (lo / G) * seconds;
  };
  const out = [];
  for (let i = 0; i < n; i++) {
    const arrival = invCdf((i + next()) / n);
    const c = catalogue[sampleIndex(probs, next())];
    const userLen = userMin + Math.floor(next() * (userMax - userMin + 1));
    const outputLen = Math.min(outMax, Math.max(1, Math.round(outMin * Math.pow(1 - next(), -1 / outAlpha))));
    out.push({ id: 0, arrival, prefix: c.key, prefixLen: c.len, promptLen: c.len + userLen, outputLen });
  }
  out.sort((a, b) => a.arrival - b.arrival);
  out.forEach((r, i) => { r.id = i; });
  return out;
}

/** Per-request simulation state. `cached` is how many prompt tokens a prefix-cache hit skipped. */
export function newRequestState(req) {
  return {
    id: req.id, arrival: req.arrival, prefix: req.prefix, prefixLen: req.prefixLen,
    promptLen: req.promptLen, outputLen: req.outputLen,
    cached: 0, prefilled: false, generated: 0, firstToken: null, lastToken: null, maxGap: 0,
    prefillReplica: null, decodeReplica: null, transferSeconds: 0,
  };
}

/** A replica: one GPU. role is 'both' (colocated), 'prefill' or 'decode'. */
export function makeReplica(id, { role = 'both', readyAt = 0 } = {}) {
  return {
    id, role, readyAt, ready: readyAt <= 0, draining: false,
    queue: [], running: [], batch: [], busyUntil: null,
    cache: new Map(), iterations: 0, tokens: 0,
  };
}

/** Whole-prefix LRU. Module 17's radix tree is block-granular; this is one slot per prefix. */
export function cacheTouch(replica, prefix, t, cfg = DEFAULT_CONFIG) {
  replica.cache.delete(prefix);
  replica.cache.set(prefix, t);
  while (replica.cache.size > cfg.cacheSlots) {
    const oldest = replica.cache.keys().next().value;
    replica.cache.delete(oldest);
  }
}

// ---------- step 1: the replica cost model ----------

/** Tokens this request contributes to one iteration: its uncached prompt during prefill, 1 while decoding. */
export function tokensThisIteration(s) {
  return s.prefilled ? 1 : Math.max(0, s.promptLen - s.cached);
}

/** Wall-clock seconds for one iteration over `batch` (an array of request states). */
export function iterationSeconds(batch, cfg = DEFAULT_CONFIG) {
  // TODO: step 1 — cfg.tFixed once, plus cfg.tPerToken for every token in the batch.
  return 0;
}

/** Tokens of work still owed by one request: unprefilled prompt tokens plus ungenerated output tokens. */
export function remainingWork(s) {
  // TODO: step 1 — uncached prompt tokens if it has not been prefilled, plus the tokens still to generate.
  return 0;
}

/** The queue-depth signal a router balances on: work owed by everything queued or running on a replica. */
export function replicaLoad(r) {
  // TODO: step 1 — sum remainingWork over r.queue and r.running.
  return 0;
}

// ---------- step 2: routing ----------

/** A consistent-hashing ring: `vnodes` points per replica id, sorted by hash. */
export function buildRing(ids, { vnodes = DEFAULT_CONFIG.vnodes } = {}) {
  // TODO: step 2 — vnodes points per id at hash32(`${id}#${v}`), sorted by hash.
  return [];
}

/** The replica a key belongs to: the first ring point at or after hash32(key), wrapping around. */
export function pickOnRing(ring, key) {
  // TODO: step 2 — the first ring point whose hash is >= hash32(key), wrapping past the end.
  return 0;
}

/**
 * Pick a replica for `req` among `candidates` (already filtered to ready, non-draining replicas of the
 * right role). Returns an INDEX into candidates.
 * ctx = { dispatched, ring, cfg }.
 */
export function chooseReplica(policy, candidates, req, ctx = {}) {
  // TODO: step 2 — 'round-robin', 'least-loaded' and 'cache-aware'; return an INDEX into candidates.
  return 0;
}

// ---------- step 3: disaggregation ----------

/** Seconds to move one request's KV cache from a prefill replica to a decode replica. */
export function kvTransferSeconds(tokens, cfg = DEFAULT_CONFIG) {
  // TODO: step 3 — a fixed setup cost plus bytes / bandwidth.
  return 0;
}

/** Split `nReplicas` between a prefill pool and a decode pool in proportion to the work each must do. */
export function planPools(requests, nReplicas, cfg = DEFAULT_CONFIG) {
  // TODO: step 3 — split the replicas in proportion to the prefill and decode work the trace implies.
  return { prefill: Math.floor(nReplicas / 2), decode: nReplicas - Math.floor(nReplicas / 2), prefillWork: 0, decodeWork: 0 };
}

// ---------- step 4: autoscaling and cost ----------

/**
 * The raw signal: how many replicas the outstanding work asks for, clamped to [minReplicas, maxReplicas].
 * snap = { pendingTokens, ready, total }.
 */
export function queueTarget(snap, cfg = DEFAULT_CONFIG) {
  // TODO: step 4 — how many replicas the outstanding work asks for, clamped to [minReplicas, maxReplicas].
  return cfg.minReplicas;
}

/**
 * What the controller actually does this tick.
 * snap = { total, raw, window } where `window` holds the raw targets of the last
 * cfg.stabilizationTicks ticks (most recent last). Scale up to `raw` at once; scale down only when the
 * whole window agrees, and then by at most cfg.scaleDownStep replicas.
 */
export function autoscaleTarget(snap, cfg = DEFAULT_CONFIG) {
  // TODO: step 4 — up immediately, down only when the whole window agrees, and then by scaleDownStep.
  return snap.total;
}

/** GPU-seconds billed by a list of spans { start, end } ; end === null means "still running at endTime". */
export function gpuSeconds(spans, endTime) {
  // TODO: step 4 — bill every span from start to end, treating end === null as endTime.
  return 0;
}

// ---------- step 5: the SLO report ----------

/** Dollars per million generated tokens. */
export function costPerMillionTokens(gpuSec, outputTokens, dollarsPerGpuHour = DEFAULT_CONFIG.dollarsPerGpuHour) {
  // TODO: step 5 — GPU-seconds -> GPU-hours -> dollars, divided by millions of tokens.
  return 0;
}

/** Latency percentiles, SLO attainment and goodput over finished requests. */
export function sloReport(records, slo = { ttft: 1, tpot: 0.05 }) {
  // TODO: step 5 — TTFT and TPOT per record, percentiles, attainment, goodput, throughput, makespan.
  return { n: 0, ttftP50: 0, ttftP95: 0, tpotP50: 0, tpotP95: 0, met: 0, attainment: 0,
    goodput: 0, throughput: 0, outputTokens: 0, makespan: 0 };
}

// ---------- the cluster driver (worked: it calls the functions above) ----------

/**
 * Discrete-event simulation of a serving cluster.
 * opts = { policy, replicas, disaggregate, pools: {prefill, decode}, autoscale, cfg }
 */
export function runCluster(requests, opts = {}) {
  const cfg = { ...DEFAULT_CONFIG, ...(opts.cfg || {}) };
  const policy = opts.policy || 'round-robin';
  const disagg = !!opts.disaggregate;
  const autoscale = !!opts.autoscale;
  const arrivals = [...requests].sort((a, b) => a.arrival - b.arrival || a.id - b.id);
  const EPS = 1e-9;

  const replicas = [];
  const spans = [];
  let nextId = 0;
  const addReplica = (role, createdAt, readyAt) => {
    const r = makeReplica(nextId++, { role, readyAt });
    replicas.push(r);
    spans.push({ id: r.id, role, start: createdAt, end: null });
    return r;
  };
  if (disagg) {
    const pools = opts.pools || { prefill: 1, decode: 1 };
    for (let i = 0; i < pools.prefill; i++) addReplica('prefill', 0, 0);
    for (let i = 0; i < pools.decode; i++) addReplica('decode', 0, 0);
  } else {
    const n = opts.replicas || (autoscale ? cfg.minReplicas : 8);
    for (let i = 0; i < n; i++) addReplica('both', 0, 0);
  }

  const eligible = (role) => replicas.filter((r) => r.ready && !r.draining && (r.role === role || r.role === 'both'));
  let ringKey = '', ringCache = null;
  const ringFor = (cands) => {
    const key = cands.map((r) => r.id).join(',');
    if (key !== ringKey) { ringKey = key; ringCache = buildRing(cands.map((r) => r.id), { vnodes: cfg.vnodes }); }
    return ringCache;
  };

  const records = [];
  const pending = [];
  const transfers = [];
  const scaleTrace = [];
  let t = arrivals.length ? arrivals[0].arrival : 0;
  let ai = 0, dispatched = 0, guard = 0;
  let iterations = 0, prefillIterations = 0, decodeIterations = 0, stallSeconds = 0;
  let promptTokens = 0, computedPromptTokens = 0, cachedTokens = 0, cacheHits = 0;
  let decodeTokens = 0, transferSecondsTotal = 0, handoffs = 0, peakReplicas = replicas.length;
  let nextScaleAt = t + cfg.scaleIntervalSeconds;
  // Seed the stabilisation window with the starting size, so a freshly started cluster cannot shrink
  // before a whole window of ticks has actually agreed that it should.
  const rawWindow = new Array(cfg.stabilizationTicks).fill(replicas.length);

  const drop = (r, s) => {
    const i = r.running.indexOf(s);
    if (i >= 0) r.running.splice(i, 1);
  };

  const finish = (s, tEnd, replicaId) => {
    records.push({
      id: s.id, arrival: s.arrival, firstToken: s.firstToken, end: tEnd, prefix: s.prefix,
      promptLen: s.promptLen, outputLen: s.outputLen, cached: s.cached, maxGap: s.maxGap,
      prefillReplica: s.prefillReplica, decodeReplica: replicaId, transferSeconds: s.transferSeconds,
    });
  };

  const send = (cands, s, pol, ring = null) => {
    const idx = chooseReplica(pol, cands, s, { dispatched, ring, cfg });
    const r = cands[idx];
    if (!r) throw new Error(`chooseReplica returned index ${idx}, which is not a replica in candidates (0..${cands.length - 1})`);
    dispatched++;
    r.queue.push(s);
    return r;
  };

  for (;;) {
    if (++guard > 5000000) throw new Error('runCluster: event budget exhausted — an iteration is probably taking zero time');

    for (const r of replicas) if (!r.ready && r.readyAt <= t + EPS) r.ready = true;

    // finish iterations that end at or before now
    for (const r of replicas) {
      if (r.busyUntil === null || r.busyUntil > t + EPS) continue;
      const tEnd = r.busyUntil;
      const batch = r.batch;
      r.batch = [];
      for (const s of batch) {
        if (!s.prefilled) {
          s.prefilled = true;
          s.generated = 1;
          s.firstToken = tEnd;
          s.lastToken = tEnd;
          if (r.role === 'prefill') {
            drop(r, s);
            if (s.generated >= s.outputLen) { finish(s, tEnd, r.id); continue; }
            const dt = kvTransferSeconds(s.promptLen, cfg);
            if (!(dt >= 0)) throw new Error(`kvTransferSeconds returned ${dt}; a transfer cannot take negative time`);
            s.transferSeconds = dt;
            transferSecondsTotal += dt;
            handoffs++;
            transfers.push({ readyAt: tEnd + dt, state: s });
            continue;
          }
        } else {
          s.generated++;
          decodeTokens++;
          const gap = tEnd - s.lastToken;
          if (gap > s.maxGap) s.maxGap = gap;
          s.lastToken = tEnd;
        }
        if (s.generated >= s.outputLen) { drop(r, s); finish(s, tEnd, r.id); }
      }
      r.busyUntil = null;
    }

    // completed KV transfers join the decode pool
    for (let i = transfers.length - 1; i >= 0; i--) {
      if (transfers[i].readyAt > t + EPS) continue;
      const s = transfers[i].state;
      const cands = eligible('decode');
      if (!cands.length) continue;
      transfers.splice(i, 1);
      s.decodeReplica = send(cands, s, 'least-loaded').id;
    }

    // arrivals. The balancer holds a request until some replica has room in its queue: routing a
    // request to a replica commits it there, so a router without backpressure cannot use new capacity.
    while (ai < arrivals.length && arrivals[ai].arrival <= t + EPS) pending.push(newRequestState(arrivals[ai++]));
    if (pending.length) {
      const all = eligible('prefill');
      const ring = all.length ? ringFor(all) : null;
      let cands = all.filter((r) => r.queue.length < cfg.maxQueue);
      while (pending.length && cands.length) {
        const s = pending.shift();
        s.prefillReplica = send(cands, s, policy, ring).id;
        if (!disagg) s.decodeReplica = s.prefillReplica;
        cands = cands.filter((r) => r.queue.length < cfg.maxQueue);
      }
    }

    // autoscaler tick
    const anyWork = () => ai < arrivals.length || pending.length > 0 || transfers.length > 0 ||
      replicas.some((r) => r.queue.length || r.running.length || r.busyUntil !== null);
    if (autoscale && t >= nextScaleAt - EPS) {
      let pendingTokens = 0;
      for (const s of pending) pendingTokens += remainingWork(s);
      for (const tr of transfers) pendingTokens += remainingWork(tr.state);
      for (const r of replicas) pendingTokens += replicaLoad(r);
      const alive = replicas.filter((x) => !x.draining);
      const snap = { pendingTokens, ready: alive.filter((x) => x.ready).length, total: alive.length };
      const raw = queueTarget(snap, cfg);
      if (!Number.isFinite(raw)) throw new Error(`queueTarget returned ${raw}; it must be a replica count`);
      rawWindow.push(raw);
      while (rawWindow.length > cfg.stabilizationTicks) rawWindow.shift();
      const desired = autoscaleTarget({ ...snap, raw, window: rawWindow }, cfg);
      if (!Number.isFinite(desired) || desired < 0) throw new Error(`autoscaleTarget returned ${desired}; it must be a non-negative replica count`);
      if (desired > alive.length) {
        for (let k = alive.length; k < desired; k++) addReplica('both', t, t + cfg.coldStartSeconds);
      } else if (desired < alive.length) {
        const victims = alive.filter((x) => x.ready).sort((a, b) => replicaLoad(a) - replicaLoad(b));
        for (let k = 0; k < victims.length && alive.length - k > Math.max(1, desired); k++) victims[k].draining = true;
      }
      peakReplicas = Math.max(peakReplicas, replicas.filter((x) => !x.draining).length);
      scaleTrace.push({ t, pendingTokens, alive: snap.total, ready: snap.ready, raw, desired });
      nextScaleAt += cfg.scaleIntervalSeconds;
    }

    // start an iteration on every idle replica that has work.
    // Prefill-priority scheduling (Orca / vLLM v0): if anything is waiting, this iteration is a
    // prefill step and every running decode stalls for its duration. Otherwise it is a decode step.
    for (const r of replicas) {
      if (r.busyUntil !== null) continue;
      while (r.queue.length && r.queue[0].prefilled && r.running.length < cfg.maxBatch) r.running.push(r.queue.shift());
      const batch = [];
      let batchTokens = 0;
      while (r.queue.length && r.running.length < cfg.maxBatch) {
        const s = r.queue[0];
        const hit = r.cache.has(s.prefix);
        s.cached = hit ? s.prefixLen : 0;
        const add = tokensThisIteration(s);
        if (batch.length > 0 && batchTokens + add > cfg.maxBatchTokens) { s.cached = 0; break; }
        batchTokens += add;
        promptTokens += s.promptLen;
        computedPromptTokens += s.promptLen - s.cached;
        cachedTokens += s.cached;
        if (hit) cacheHits++;
        cacheTouch(r, s.prefix, t, cfg);
        r.queue.shift();
        r.running.push(s);
        batch.push(s);
      }
      if (batch.length) prefillIterations++;
      else { for (const s of r.running) batch.push(s); if (batch.length) decodeIterations++; }
      if (!batch.length) continue;
      const dt = iterationSeconds(batch, cfg);
      if (!(dt > 0)) throw new Error(`iterationSeconds returned ${dt}; one iteration must take tFixed + tPerToken * (tokens in the batch) > 0`);
      if (batch.length < r.running.length) stallSeconds += dt * (r.running.length - batch.length);
      r.batch = batch;
      r.busyUntil = t + dt;
      r.iterations++;
      for (const s of batch) r.tokens += tokensThisIteration(s);
      iterations++;
    }

    // retire drained replicas
    for (let i = replicas.length - 1; i >= 0; i--) {
      const r = replicas[i];
      if (!r.draining || r.queue.length || r.running.length || r.busyUntil !== null) continue;
      const sp = spans.find((x) => x.id === r.id);
      if (sp) sp.end = t;
      replicas.splice(i, 1);
      ringKey = '';
    }

    // next event
    let tn = Infinity;
    if (ai < arrivals.length) tn = Math.min(tn, arrivals[ai].arrival);
    for (const r of replicas) {
      if (r.busyUntil !== null) tn = Math.min(tn, r.busyUntil);
      if (!r.ready) tn = Math.min(tn, r.readyAt);
    }
    for (const tr of transfers) tn = Math.min(tn, tr.readyAt);
    const work = anyWork();
    if (autoscale && work) tn = Math.min(tn, nextScaleAt);
    if (!work || !Number.isFinite(tn)) break;
    t = Math.max(tn, t + EPS);
  }

  for (const sp of spans) if (sp.end === null) sp.end = t;
  records.sort((a, b) => a.id - b.id);
  return {
    policy, disaggregated: disagg, autoscaled: autoscale, cfg,
    records, spans, makespan: t, iterations, prefillIterations, decodeIterations, stallSeconds,
    promptTokens, computedPromptTokens, cachedTokens, cacheHits,
    cacheHitRate: records.length ? cacheHits / records.length : 0,
    decodeTokens, handoffs, transferSecondsTotal, scaleTrace, peakReplicas,
    finalReplicas: replicas.length,
    gpuSeconds: gpuSeconds(spans, t),
  };
}
