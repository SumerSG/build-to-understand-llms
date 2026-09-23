// Prefix caching & prompt caching.
// A "KV block" here is represented by a single 32-bit hash of the token window it covers. That is
// exactly how vLLM's automatic prefix caching identifies a block: it never compares tokens at
// serving time, only hashes.
//
// Everything above the "step 1" line is done for you; read it, it sets the conventions.

import { hash32, rng, randInt } from 'lib/util.js';

// ---------- constants and worked examples (done for you) ----------

/**
 * Tokens per KV block. vLLM's default on GPUs is 16; TensorRT-LLM's tokens-per-block is configurable
 * (32 or 64 are common); SGLang's radix cache defaults to a page of 1 token and matches token by token. Caching is
 * block-granular: a 33-token prompt has two complete blocks and one leftover token that is
 * recomputed every time, because its block is not full yet and so has no stable identity.
 */
export const BLOCK_SIZE = 16;

/**
 * One link of the hash chain: the identity of a block is the identity of everything before it
 * plus its own tokens. `prevHash` is 0 for the first block.
 * Two prompts collide on block i only if all tokens up to the end of block i are identical,
 * which is what makes a hash-table lookup a correct prefix test.
 */
export function hashChain(prevHash, block) {
  return hash32(prevHash + '|' + block.join(','));
}

/** Split tokens into complete blocks of `blockSize`. Any leftover tail is dropped. */
export function toBlocks(tokens, blockSize = BLOCK_SIZE) {
  const out = [];
  for (let i = 0; i + blockSize <= tokens.length; i += blockSize) out.push(tokens.slice(i, i + blockSize));
  return out;
}

/** A fresh radix-tree node. `key` is the run of block hashes this node owns. */
export function makeNode(key, parent) {
  return { key, children: new Map(), parent, refs: 0, lastUsed: 0 };
}

/**
 * Example multipliers on the base input-token price, of the same shape several hosted APIs use.
 * These are illustrative round numbers, not any vendor's published prices.
 */
export const PRICING = { base: 1, cacheWrite: 1.25, cacheRead: 0.1, output: 4 };

/**
 * Cost of a request with no caching at all, in units of the base input-token price:
 * every prompt token is billed at 1x and every output token at `pricing.output`.
 */
export function baselineCost(promptTokens, outputTokens, pricing = PRICING) {
  return promptTokens * pricing.base + outputTokens * pricing.output;
}

/** `n` random token ids in [0, vocab). All randomness goes through a seeded rng. */
function randomTokens(next, n, vocab) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = randInt(next, vocab);
  return out;
}

// ---------- step 1: block hashes and an exact-prefix cache ----------

/**
 * The chained hash of every complete block of `tokens`, in order.
 * blockHashes(tokens)[i] must depend on tokens 0 .. (i+1)*blockSize - 1 and nothing else.
 */
export function blockHashes(tokens, blockSize = BLOCK_SIZE) {
  // TODO: step 1 — call hashChain once per block, feeding the previous hash forward.
  return [];
}

/** A flat prefix cache: one Map entry per cached block, keyed by its chained hash. */
export class HashCache {
  constructor(blockSize = BLOCK_SIZE) {
    this.blockSize = blockSize;
    this.blocks = new Map(); // chained block hash -> position of that block in its prefix
  }

  /** Store every block of `tokens`. Returns how many of them were not already cached. */
  insert(tokens) {
    // TODO: step 1
    return 0;
  }

  /** Longest cached prefix: { blocks, tokens }. Stop at the first block that is missing. */
  lookup(tokens) {
    // TODO: step 1
    return { blocks: 0, tokens: 0 };
  }

  size() {
    return this.blocks.size;
  }
}

// ---------- steps 2 and 3: the radix tree ----------

export class PrefixTree {
  constructor() {
    this.root = makeNode([], null);
    this.blocks = 0; // total block hashes stored across all nodes
    this.clock = 0;  // logical time for LRU, so eviction order is deterministic
  }

  /** Total blocks held by the tree (done for you). */
  size() { return this.blocks; }

  /** Number of nodes below the root (done for you; also shows how to walk the tree). */
  nodeCount() {
    let n = 0;
    const stack = [this.root];
    while (stack.length) {
      const node = stack.pop();
      for (const c of node.children.values()) { n++; stack.push(c); }
    }
    return n;
  }

  /** Mark `node` and all of its ancestors as used now (done for you). */
  touch(node) {
    const t = ++this.clock;
    for (let n = node; n; n = n.parent) n.lastUsed = t;
  }

  /**
   * Longest prefix of `hashes` already in the tree.
   * Returns { blocks, node } where `node` is the deepest node fully covered by the match,
   * and `blocks` may run part-way into one more node. Call `touch` on the matched path.
   */
  match(hashes) {
    // TODO: step 2
    return { blocks: 0, node: this.root };
  }

  /**
   * Insert the whole sequence, splitting a node when a path diverges inside it.
   * Returns the node at depth `hashes.length`. Keep `this.blocks` correct.
   */
  insert(hashes) {
    // TODO: step 2
    return this.root;
  }

  /** Insert and pin: every node on the root-to-end path becomes ineligible for eviction. */
  acquire(hashes) {
    // TODO: step 3
    return this.root;
  }

  /** Unpin a path previously returned by `acquire`. */
  release(node) {
    // TODO: step 3
    return node;
  }

  /**
   * Evict least-recently-used leaves until the tree holds at most `capacityBlocks` blocks.
   * Only childless, unpinned nodes may go. Returns the number of blocks freed.
   */
  evict(capacityBlocks) {
    // TODO: step 3
    return 0;
  }
}

// ---------- step 4: the traffic model ----------

/**
 * Build a deterministic stream of chat requests. Every conversation starts with the same system
 * prompt and one of `fewShotVariants` few-shot bundles; each turn's prompt is the previous turn's
 * prompt plus the assistant's reply plus a new user message, so turn t is a strict prefix of turn t+1.
 * Returns an array of { id, conv, turn, tokens, outputTokens }, in arrival order, with
 * conversations interleaved.
 */
export function makeWorkload({
  conversations = 50, turns = 10, seed = 17, vocab = 4096,
  systemTokens = 64, fewShotVariants = 3, fewShotTokens = 96,
  userTokens = 32, replyTokens = 48,
} = {}) {
  const next = rng(seed);
  // TODO: step 4 — build the shared system prompt and the few-shot variants with randomTokens(next, n, vocab),
  // then grow each conversation turn by turn and interleave the turns into one arrival order.
  return [];
}

/**
 * Replay the requests against a PrefixTree of bounded capacity, keeping up to `concurrency`
 * requests pinned at once (the continuous-batching window from the continuous batching module).
 */
export function simulate(requests, { capacityBlocks = 1 << 20, blockSize = BLOCK_SIZE, concurrency = 4 } = {}) {
  // TODO: step 4
  return {
    requests: 0, promptTokens: 0, cachedTokens: 0, outputTokens: 0,
    totalBlocks: 0, hitBlocks: 0, hitRate: 0, evictedBlocks: 0, sizeBlocks: 0, nodes: 0,
  };
}

// ---------- step 5: the cost and latency model ----------

/**
 * Prompt tokens served from cache are billed at `cacheRead`, the rest at `cacheWrite`
 * (they are written into the cache as a side effect of being processed), output at `output`.
 * Throw if `cachedTokens` exceeds `promptTokens`.
 */
export function requestCost(promptTokens, cachedTokens, outputTokens, pricing = PRICING) {
  // TODO: step 5
  return 0;
}

/** The prefix hit rate at which caching stops losing money on input tokens. Never negative. */
export function breakEvenHitRate(pricing = PRICING) {
  // TODO: step 5
  return 0;
}

/**
 * Time to first token: a fixed `overheadMs` plus the time to prefill the tokens that were NOT
 * served from cache, in milliseconds (the rate is in tokens per second).
 * `prefillTokensPerSecond` defaults to 10000, a deliberately conservative figure for an 8B model on
 * one H100: NVIDIA's H100 SXM datasheet lists approximately 989 TFLOPS of dense BF16, and prefill
 * costs about 2 x 8e9 = 16 GFLOP per prompt token, so 10,000 tokens/s is about 16% utilisation
 * (40% would be roughly 25,000 tokens/s). Treat it as an order of magnitude.
 */
export function ttftMs(promptTokens, cachedTokens, { prefillTokensPerSecond = 10000, overheadMs = 15 } = {}) {
  // TODO: step 5
  return 0;
}
