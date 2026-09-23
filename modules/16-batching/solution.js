// Continuous batching & paged attention — reference solution: a serving engine as a discrete-event simulator.
// Symbols: P = prompt length in tokens, G = tokens generated so far, KV = key/value cache entries.

import { rng, meanArray } from 'lib/util.js';

// ---------- conventions ----------

export const DEFAULT_CONFIG = {
  tFixed: 0.005,
  tPerToken: 0.00005,
  maxBatch: 32,
  blockSize: 16,
  numBlocks: 512,
};

export const SHARED_PROMPTS = [
  { key: 'support-system', promptLen: 320 },
  { key: 'code-system', promptLen: 208 },
  { key: 'rag-template', promptLen: 464 },
  { key: 'json-system', promptLen: 96 },
];

export function makeWorkload({ n = 200, seed = 7, rate = 40, promptMin = 16, promptMax = 256,
  outMin = 8, outMax = 512, alpha = 1.1, sharedFraction = 0 } = {}) {
  const next = rng(seed);
  const out = [];
  let t = 0;
  for (let i = 0; i < n; i++) {
    t += -Math.log(1 - next()) / rate;
    let promptLen = promptMin + Math.floor(next() * (promptMax - promptMin + 1));
    let key = `unique-${i}`;
    if (next() < sharedFraction) {
      const s = SHARED_PROMPTS[Math.floor(next() * SHARED_PROMPTS.length)];
      key = s.key;
      promptLen = s.promptLen;
    }
    const outputLen = Math.min(outMax, Math.max(1, Math.round(outMin * Math.pow(1 - next(), -1 / alpha))));
    out.push({ id: i, arrival: t, promptLen, outputLen, key });
  }
  return out;
}

export function newRequestState(req) {
  return {
    id: req.id, arrival: req.arrival, promptLen: req.promptLen, outputLen: req.outputLen, key: req.key,
    prefilled: false, generated: 0, firstToken: null, restarts: 0, blocks: [],
  };
}

export function percentile(values, p) {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  const i = (p / 100) * (s.length - 1);
  const lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}

export function summarize(policy, records, extra = {}) {
  const sorted = [...records].sort((a, b) => a.id - b.id);
  const ttft = sorted.map((r) => r.firstToken - r.arrival);
  const tpot = sorted.filter((r) => r.outputLen > 1).map((r) => (r.end - r.firstToken) / (r.outputLen - 1));
  const outputTokens = sorted.reduce((s, r) => s + r.outputLen, 0);
  const makespan = extra.makespan || 0;
  return {
    policy,
    records: sorted,
    completed: sorted.length,
    outputTokens,
    throughput: makespan > 0 ? outputTokens / makespan : 0,
    ttftP50: percentile(ttft, 50),
    ttftP95: percentile(ttft, 95),
    ttftMean: meanArray(ttft),
    tpotMean: meanArray(tpot),
    ...extra,
  };
}

// ---------- step 1: the iteration cost model ----------

export function tokensThisIteration(s) {
  return s.prefilled ? 1 : s.promptLen + s.generated;
}

export function iterationSeconds(batch, cfg = DEFAULT_CONFIG) {
  let tokens = 0;
  for (const s of batch) tokens += tokensThisIteration(s);
  return cfg.tFixed + cfg.tPerToken * tokens;
}

export function kvTokens(s) {
  return s.prefilled ? s.promptLen + s.generated - 1 : 0;
}

// ---------- step 2: static batching ----------

export function runStatic(requests, cfg = DEFAULT_CONFIG) {
  const arrivals = [...requests].sort((a, b) => a.arrival - b.arrival || a.id - b.id);
  const records = [];
  let clock = 0, iterations = 0, batchTokens = 0, i = 0;
  while (i < arrivals.length) {
    if (clock < arrivals[i].arrival) clock = arrivals[i].arrival;
    const batch = [];
    while (batch.length < cfg.maxBatch && i < arrivals.length && arrivals[i].arrival <= clock) {
      batch.push(newRequestState(arrivals[i++]));
    }
    batchTokens += batch.reduce((s, x) => s + tokensThisIteration(x), 0);
    clock += iterationSeconds(batch, cfg);
    iterations++;
    for (const s of batch) { s.prefilled = true; s.generated = 1; s.firstToken = clock; }
    let longest = 0;
    for (const s of batch) if (s.outputLen > longest) longest = s.outputLen;
    for (let step = 1; step < longest; step++) {
      batchTokens += batch.length;
      clock += iterationSeconds(batch, cfg);
      iterations++;
      for (const s of batch) if (s.generated < s.outputLen) s.generated++;
    }
    for (const s of batch) {
      records.push({ id: s.id, arrival: s.arrival, firstToken: s.firstToken, end: clock,
        promptLen: s.promptLen, outputLen: s.outputLen, restarts: 0 });
    }
  }
  return summarize('static', records, { makespan: clock, iterations, batchTokens });
}

// ---------- step 3: continuous batching ----------

export function runContinuous(requests, cfg = DEFAULT_CONFIG) {
  const arrivals = [...requests].sort((a, b) => a.arrival - b.arrival || a.id - b.id);
  const records = [];
  const running = [];
  let clock = 0, iterations = 0, batchTokens = 0, i = 0;
  while (i < arrivals.length || running.length) {
    while (running.length < cfg.maxBatch && i < arrivals.length && arrivals[i].arrival <= clock) {
      running.push(newRequestState(arrivals[i++]));
    }
    if (!running.length) { clock = arrivals[i].arrival; continue; }
    batchTokens += running.reduce((s, x) => s + tokensThisIteration(x), 0);
    clock += iterationSeconds(running, cfg);
    iterations++;
    for (const s of running) {
      if (!s.prefilled) { s.prefilled = true; s.generated = 1; s.firstToken = clock; }
      else s.generated++;
    }
    for (let k = running.length - 1; k >= 0; k--) {
      const s = running[k];
      if (s.generated >= s.outputLen) {
        records.push({ id: s.id, arrival: s.arrival, firstToken: s.firstToken, end: clock,
          promptLen: s.promptLen, outputLen: s.outputLen, restarts: s.restarts });
        running.splice(k, 1);
      }
    }
  }
  return summarize('continuous', records, { makespan: clock, iterations, batchTokens });
}

// ---------- step 4: the paged block allocator ----------

export class BlockAllocator {
  constructor({ numBlocks, blockSize }) {
    this.numBlocks = numBlocks;
    this.blockSize = blockSize;
    this.freeList = [];
    for (let b = 0; b < numBlocks; b++) this.freeList.push(b);
    this.refs = new Array(numBlocks).fill(0);
  }

  get freeCount() { return this.freeList.length; }

  get usedCount() { return this.numBlocks - this.freeList.length; }

  blocksNeeded(nTokens) { return Math.ceil(nTokens / this.blockSize); }

  allocate(nTokens) {
    const need = this.blocksNeeded(nTokens);
    if (need > this.freeList.length) return null;
    const table = [];
    for (let i = 0; i < need; i++) {
      const b = this.freeList.shift();
      this.refs[b] = 1;
      table.push(b);
    }
    return table;
  }

  appendToken(table, lenBefore) {
    if (lenBefore < table.length * this.blockSize) return true;
    if (!this.freeList.length) return false;
    const b = this.freeList.shift();
    this.refs[b] = 1;
    table.push(b);
    return true;
  }

  retain(blocks) {
    for (const b of blocks) {
      if (this.refs[b] <= 0) throw new Error(`BlockAllocator.retain: block ${b} is not allocated`);
      this.refs[b]++;
    }
  }

  free(table) {
    for (const b of table) {
      if (this.refs[b] <= 0) throw new Error(`BlockAllocator.free: block ${b} is already free (double free)`);
      this.refs[b]--;
      if (this.refs[b] === 0) this.freeList.push(b);
    }
  }
}

export function internalFragmentation(lengths, blockSize) {
  let tokens = 0, slots = 0;
  for (const L of lengths) {
    tokens += L;
    slots += Math.ceil(L / blockSize) * blockSize;
  }
  return { tokens, slots, wasted: slots - tokens, fraction: slots > 0 ? (slots - tokens) / slots : 0 };
}

// ---------- step 5: admission control and preemption ----------

export function runPaged(requests, cfg = DEFAULT_CONFIG) {
  const capacity = cfg.numBlocks * cfg.blockSize;
  for (const r of requests) {
    const need = r.promptLen + r.outputLen - 1;
    if (need > capacity) {
      throw new Error(`runPaged: request ${r.id} needs ${need} KV slots but the whole cache holds ${capacity}; reject it at admission instead of livelocking`);
    }
  }
  const alloc = new BlockAllocator({ numBlocks: cfg.numBlocks, blockSize: cfg.blockSize });
  const arrivals = [...requests].sort((a, b) => a.arrival - b.arrival || a.id - b.id);
  const waiting = [];
  const running = [];
  const records = [];
  let clock = 0, iterations = 0, batchTokens = 0, i = 0;
  let preemptions = 0, recomputedTokens = 0, peakBlocks = 0, slotSum = 0, kvSum = 0;

  const preempt = (s) => {
    alloc.free(s.blocks);
    s.blocks = [];
    s.prefilled = false;
    s.restarts++;
    preemptions++;
    waiting.unshift(s);
  };

  while (i < arrivals.length || waiting.length || running.length) {
    while (i < arrivals.length && arrivals[i].arrival <= clock) waiting.push(newRequestState(arrivals[i++]));
    while (running.length < cfg.maxBatch && waiting.length) {
      const s = waiting[0];
      const table = alloc.allocate(s.promptLen + s.generated);
      if (table === null) break;
      s.blocks = table;
      running.push(waiting.shift());
    }
    if (!running.length) {
      if (i >= arrivals.length) throw new Error('runPaged: nothing can run and nothing is arriving — the KV budget is too small');
      clock = Math.max(clock, arrivals[i].arrival);
      continue;
    }
    let k = 0;
    while (k < running.length) {
      const s = running[k];
      if (!s.prefilled) { k++; continue; }
      if (alloc.appendToken(s.blocks, kvTokens(s))) { k++; continue; }
      const victim = running[running.length - 1];
      running.pop();
      preempt(victim);
    }
    for (const s of running) {
      const t = tokensThisIteration(s);
      batchTokens += t;
      if (!s.prefilled && s.generated > 0) recomputedTokens += t;
    }
    clock += iterationSeconds(running, cfg);
    iterations++;
    for (const s of running) {
      if (!s.prefilled) { s.prefilled = true; s.generated++; if (s.firstToken === null) s.firstToken = clock; }
      else s.generated++;
    }
    if (alloc.usedCount > peakBlocks) peakBlocks = alloc.usedCount;
    slotSum += alloc.usedCount * cfg.blockSize;
    for (const s of running) kvSum += kvTokens(s);
    for (let j = running.length - 1; j >= 0; j--) {
      const s = running[j];
      if (s.generated >= s.outputLen) {
        alloc.free(s.blocks);
        records.push({ id: s.id, arrival: s.arrival, firstToken: s.firstToken, end: clock,
          promptLen: s.promptLen, outputLen: s.outputLen, restarts: s.restarts });
        running.splice(j, 1);
      }
    }
  }
  return summarize('paged', records, {
    makespan: clock, iterations, batchTokens, preemptions, recomputedTokens, peakBlocks,
    wastedFraction: slotSum > 0 ? 1 - kvSum / slotSum : 0,
  });
}

// ---------- step 6: prefix sharing with copy-on-write reference counts ----------

export function allocateShared(alloc, cache, key, nTokens) {
  const total = alloc.blocksNeeded(nTokens);
  const full = Math.floor(nTokens / alloc.blockSize);
  const cached = cache.get(key);
  const shared = cached ? Math.min(cached.length, full) : 0;
  const fresh = total - shared;
  if (alloc.freeCount < fresh) return null;
  const table = [];
  if (shared > 0) {
    const reused = cached.slice(0, shared);
    alloc.retain(reused);
    for (const b of reused) table.push(b);
  }
  const tail = alloc.allocate(fresh * alloc.blockSize);
  for (const b of tail) table.push(b);
  if (!cached && full > 0) {
    const entry = table.slice(0, full);
    alloc.retain(entry);
    cache.set(key, entry);
  }
  return { blocks: table, sharedBlocks: shared, newBlocks: fresh };
}
