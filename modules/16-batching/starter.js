// Module 16 — Continuous batching & paged attention.
// You are building a serving engine as a discrete-event simulator: requests arrive, a scheduler
// decides every iteration who runs, and a block allocator decides who fits in the KV cache.
//
// Symbols used throughout:
//   P  = promptLen, the number of prompt tokens of a request
//   G  = generated, the number of tokens that request has emitted so far
//   KV = the key/value cache entries a request currently owns (module 15)
// A "request" is plain data: { id, arrival, promptLen, outputLen, key }. Times are seconds.
// A "request state" is the scheduler's mutable copy of it (see newRequestState below).

import { rng, meanArray } from 'lib/util.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/**
 * The machine this simulator models: roughly an 8B model in bf16 on one H100.
 * tFixed: every iteration pays a fixed cost no matter how small the batch. Reading approximately
 *   16 GB of weights at the approximately 3.35 TB/s of HBM3 bandwidth in NVIDIA's H100 datasheet is
 *   about 4.8 ms; kernel launches and sampling round it to 5 ms. This is the memory-bound term.
 * tPerToken: each token in the batch costs approximately 2N = 16 GFLOP (module 08); at an assumed
 *   350 TFLOP/s of achieved bf16 throughput that is about 46 us, rounded to 50 us for the KV reads.
 * blockSize 16 is vLLM's default. numBlocks is set deliberately small so memory pressure is visible.
 */
export const DEFAULT_CONFIG = {
  tFixed: 0.005,
  tPerToken: 0.00005,
  maxBatch: 32,
  blockSize: 16,
  numBlocks: 512,
};

/** A few system prompts that many requests share, for the prefix-sharing step. */
export const SHARED_PROMPTS = [
  { key: 'support-system', promptLen: 320 },
  { key: 'code-system', promptLen: 208 },
  { key: 'rag-template', promptLen: 464 },
  { key: 'json-system', promptLen: 96 },
];

/**
 * A synthetic trace. Arrivals are a Poisson process of `rate` requests per second (exponential
 * gaps). Output lengths are a Pareto draw — a few requests generate hundreds of tokens while most
 * generate a handful — because that skew is what breaks static batching.
 */
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

/** The scheduler's per-request state. `blocks` is the block table you fill in from step 4 on. */
export function newRequestState(req) {
  return {
    id: req.id, arrival: req.arrival, promptLen: req.promptLen, outputLen: req.outputLen, key: req.key,
    prefilled: false, generated: 0, firstToken: null, restarts: 0, blocks: [],
  };
}

/** Linear-interpolated percentile, the same convention numpy uses. percentile([1,2,3,4], 50) === 2.5. */
export function percentile(values, p) {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  const i = (p / 100) * (s.length - 1);
  const lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}

/**
 * Turn finished-request records into the metrics a serving team actually watches.
 * A record is { id, arrival, firstToken, end, promptLen, outputLen, restarts }.
 *   TTFT (time to first token) = firstToken − arrival, including queueing.
 *   TPOT (time per output token, also called inter-token latency) = (end − firstToken) / (outputLen − 1).
 *   throughput = output tokens / makespan.
 * Every run function you write below ends with a call to this.
 */
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

/**
 * How many tokens this request contributes to the current iteration.
 * A request that has not been prefilled processes its whole prompt (plus any tokens it had already
 * generated before it was preempted and lost its cache, step 5). A prefilled request decodes one token.
 */
export function tokensThisIteration(s) {
  // TODO: step 1
  return 0;
}

/** Seconds for one iteration running `batch` (an array of request states). */
export function iterationSeconds(batch, cfg = DEFAULT_CONFIG) {
  let tokens = 0;
  for (const s of batch) {
    tokens += 0; // TODO: step 1 — this request's contribution
  }
  return 0; // TODO: step 1 — one fixed cost per iteration, plus a cost per token in the batch
}

/** How many KV cache entries this request currently owns. */
export function kvTokens(s) {
  // TODO: step 1
  return 0;
}

// ---------- step 2: static batching ----------

/**
 * Static batching: fill a batch, run it to completion, return it, then start the next one.
 * Returns summarize('static', records, { makespan, iterations, batchTokens }).
 */
export function runStatic(requests, cfg = DEFAULT_CONFIG) {
  const records = [];
  // TODO: step 2
  return summarize('static', records, { makespan: 0, iterations: 0, batchTokens: 0 });
}

// ---------- step 3: continuous batching ----------

/**
 * Continuous (iteration-level) batching: every iteration, admit whoever fits and retire whoever finished.
 * Returns summarize('continuous', records, { makespan, iterations, batchTokens }).
 */
export function runContinuous(requests, cfg = DEFAULT_CONFIG) {
  const records = [];
  // TODO: step 3
  return summarize('continuous', records, { makespan: 0, iterations: 0, batchTokens: 0 });
}

// ---------- step 4: the paged block allocator ----------

/** A KV cache cut into fixed-size blocks, handed out through a free list. */
export class BlockAllocator {
  constructor({ numBlocks, blockSize }) {
    this.numBlocks = numBlocks;
    this.blockSize = blockSize;
    this.freeList = [];
    for (let b = 0; b < numBlocks; b++) this.freeList.push(b);
    this.refs = new Array(numBlocks).fill(0); // reference count per block; you use it in step 6
  }

  get freeCount() { return this.freeList.length; }

  get usedCount() { return this.numBlocks - this.freeList.length; }

  /** Blocks needed to hold nTokens KV entries. */
  blocksNeeded(nTokens) {
    // TODO: step 4
    return 0;
  }

  /** Reserve enough blocks for nTokens. Return the block table, or null if they do not all fit. */
  allocate(nTokens) {
    // TODO: step 4
    return null;
  }

  /** Make room for one more KV entry in `table`, which currently holds lenBefore entries. */
  appendToken(table, lenBefore) {
    // TODO: step 4
    return false;
  }

  /** Add one reference to every block in `blocks`. */
  retain(blocks) {
    // TODO: step 6
  }

  /** Drop one reference to every block in `table`, returning a block to the pool at zero. */
  free(table) {
    // TODO: step 4
  }
}

/** Wasted slots in the last block of each sequence: { tokens, slots, wasted, fraction }. */
export function internalFragmentation(lengths, blockSize) {
  // TODO: step 4
  return { tokens: 0, slots: 0, wasted: 0, fraction: 0 };
}

// ---------- step 5: admission control and preemption ----------

/**
 * Continuous batching with a real KV budget: admit only what the allocator can hold, and when a
 * running request cannot grow, preempt the most recently admitted one by recomputation.
 * Returns summarize('paged', records, { makespan, iterations, batchTokens, preemptions,
 * recomputedTokens, peakBlocks, wastedFraction }).
 */
export function runPaged(requests, cfg = DEFAULT_CONFIG) {
  const records = [];
  // TODO: step 5
  return summarize('paged', records, { makespan: 0, iterations: 0, batchTokens: 0, preemptions: 0,
    recomputedTokens: 0, peakBlocks: 0, wastedFraction: 0 });
}

// ---------- step 6: prefix sharing with copy-on-write reference counts ----------

/**
 * Allocate a block table for a prompt of nTokens, sharing the full blocks of an identical prompt
 * that is already in `cache` (a Map from prompt key to that prompt's full blocks).
 * Returns { blocks, sharedBlocks, newBlocks }, or null if the new blocks do not fit.
 */
export function allocateShared(alloc, cache, key, nTokens) {
  // TODO: step 6
  return null;
}
