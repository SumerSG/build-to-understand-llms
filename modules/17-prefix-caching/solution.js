// Prefix caching & prompt caching (reference solution).
// Nothing here touches a GPU. A "KV block" is represented by a single 32-bit hash of the token
// window it covers, which is exactly how vLLM's automatic prefix caching identifies a block.

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
 * Cost of a request with no caching at all, in units of the base input-token price:
 * every prompt token is billed at 1x and every output token at `pricing.output`.
 */
export function baselineCost(promptTokens, outputTokens, pricing = PRICING) {
  return promptTokens * pricing.base + outputTokens * pricing.output;
}

// ---------- step 1: block hashes and an exact-prefix cache ----------

export function blockHashes(tokens, blockSize = BLOCK_SIZE) {
  const out = [];
  let h = 0;
  for (const block of toBlocks(tokens, blockSize)) {
    h = hashChain(h, block);
    out.push(h);
  }
  return out;
}

export class HashCache {
  constructor(blockSize = BLOCK_SIZE) {
    this.blockSize = blockSize;
    this.blocks = new Map(); // chained block hash -> position of that block in its prefix
  }

  insert(tokens) {
    const hashes = blockHashes(tokens, this.blockSize);
    let added = 0;
    for (let i = 0; i < hashes.length; i++) {
      if (!this.blocks.has(hashes[i])) { this.blocks.set(hashes[i], i); added++; }
    }
    return added;
  }

  lookup(tokens) {
    const hashes = blockHashes(tokens, this.blockSize);
    let n = 0;
    while (n < hashes.length && this.blocks.has(hashes[n])) n++;
    return { blocks: n, tokens: n * this.blockSize };
  }

  size() {
    return this.blocks.size;
  }
}

// ---------- step 2: the radix tree ----------

export class PrefixTree {
  constructor() {
    this.root = makeNode([], null);
    this.blocks = 0; // total block hashes stored across all nodes
    this.clock = 0;  // logical time for LRU, so eviction order is deterministic
  }

  size() { return this.blocks; }

  nodeCount() {
    let n = 0;
    const stack = [this.root];
    while (stack.length) {
      const node = stack.pop();
      for (const c of node.children.values()) { n++; stack.push(c); }
    }
    return n;
  }

  /** Mark `node` and all of its ancestors as used now. */
  touch(node) {
    const t = ++this.clock;
    for (let n = node; n; n = n.parent) n.lastUsed = t;
  }

  /**
   * Longest prefix of `hashes` already in the tree.
   * Returns { blocks, node } where `node` is the deepest node fully covered by the match.
   */
  match(hashes) {
    let node = this.root, i = 0;
    for (;;) {
      const child = i < hashes.length ? node.children.get(hashes[i]) : undefined;
      if (!child) break;
      let c = 0;
      while (c < child.key.length && i + c < hashes.length && child.key[c] === hashes[i + c]) c++;
      if (c === child.key.length) { node = child; i += c; continue; }
      this.touch(node);
      return { blocks: i + c, node };
    }
    this.touch(node);
    return { blocks: i, node };
  }

  /** Insert the whole sequence, splitting a node when a path diverges inside it. Returns the end node. */
  insert(hashes) {
    let node = this.root, i = 0;
    while (i < hashes.length) {
      const child = node.children.get(hashes[i]);
      if (!child) {
        const leaf = makeNode(hashes.slice(i), node);
        node.children.set(hashes[i], leaf);
        this.blocks += leaf.key.length;
        return leaf;
      }
      let c = 0;
      while (c < child.key.length && i + c < hashes.length && child.key[c] === hashes[i + c]) c++;
      if (c === child.key.length) { node = child; i += c; continue; }
      // Diverged inside `child`: split it into a head of length c and a tail holding the rest.
      // `child` itself becomes the tail, so any handle an earlier `acquire` returned still points
      // at the deepest node of its path and `release` walks the whole pinned path.
      const head = makeNode(child.key.slice(0, c), node);
      head.refs = child.refs;
      head.lastUsed = child.lastUsed;
      node.children.set(head.key[0], head);
      child.key = child.key.slice(c);
      child.parent = head;
      head.children.set(child.key[0], child);
      node = head;
      i += c;
    }
    return node;
  }

  // ---------- step 3: reference counting and LRU eviction ----------

  /** Insert and pin: every node on the root-to-end path becomes ineligible for eviction. */
  acquire(hashes) {
    const node = this.insert(hashes);
    for (let n = node; n; n = n.parent) n.refs++;
    this.touch(node);
    return node;
  }

  /** Unpin a path previously returned by `acquire`. */
  release(node) {
    for (let n = node; n; n = n.parent) n.refs--;
    return node;
  }

  /**
   * Evict least-recently-used leaves until the tree holds at most `capacityBlocks` blocks.
   * Only childless, unpinned nodes may go: a node's parent is a prefix of it, so evicting a
   * parent first would orphan a longer cached prefix. Returns the number of blocks freed.
   */
  evict(capacityBlocks) {
    let freed = 0;
    while (this.blocks > capacityBlocks) {
      let victim = null;
      const stack = [this.root];
      while (stack.length) {
        const node = stack.pop();
        for (const c of node.children.values()) stack.push(c);
        if (node !== this.root && node.children.size === 0 && node.refs === 0) {
          if (!victim || node.lastUsed < victim.lastUsed) victim = node;
        }
      }
      if (!victim) break; // everything left is pinned or is an interior node of a pinned path
      victim.parent.children.delete(victim.key[0]);
      this.blocks -= victim.key.length;
      freed += victim.key.length;
      victim.parent = null;
    }
    return freed;
  }
}

// ---------- step 4: the traffic model ----------

function randomTokens(next, n, vocab) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = randInt(next, vocab);
  return out;
}

/**
 * Build a deterministic stream of chat requests. Every conversation starts with the same system
 * prompt and one of `fewShotVariants` few-shot bundles; each turn's prompt is the previous turn's
 * prompt plus the assistant's reply plus a new user message, so turn t is a strict prefix of turn t+1.
 */
export function makeWorkload({
  conversations = 50, turns = 10, seed = 17, vocab = 4096,
  systemTokens = 64, fewShotVariants = 3, fewShotTokens = 96,
  userTokens = 32, replyTokens = 48,
} = {}) {
  const next = rng(seed);
  const system = randomTokens(next, systemTokens, vocab);
  const fewShot = [];
  for (let v = 0; v < fewShotVariants; v++) fewShot.push(randomTokens(next, fewShotTokens, vocab));

  const pending = [];
  for (let c = 0; c < conversations; c++) {
    pending.push({ conv: c, turn: 0, prompt: system.concat(fewShot[randInt(next, fewShotVariants)]) });
  }

  const requests = [];
  const live = pending.slice();
  while (live.length) {
    const idx = randInt(next, live.length);
    const state = live[idx];
    const prompt = state.prompt.concat(randomTokens(next, userTokens, vocab));
    requests.push({ id: requests.length, conv: state.conv, turn: state.turn, tokens: prompt, outputTokens: replyTokens });
    state.turn++;
    if (state.turn >= turns) live.splice(idx, 1);
    else state.prompt = prompt.concat(randomTokens(next, replyTokens, vocab));
  }
  return requests;
}

/**
 * Replay the requests against a PrefixTree of bounded capacity, keeping up to `concurrency`
 * requests pinned at once (the continuous-batching window from the continuous batching module).
 */
export function simulate(requests, { capacityBlocks = 1 << 20, blockSize = BLOCK_SIZE, concurrency = 4 } = {}) {
  const tree = new PrefixTree();
  const inFlight = [];
  let hitBlocks = 0, totalBlocks = 0, promptTokens = 0, cachedTokens = 0, evictedBlocks = 0;

  for (const r of requests) {
    const hashes = blockHashes(r.tokens, blockSize);
    const { blocks: hit } = tree.match(hashes);
    inFlight.push(tree.acquire(hashes));
    while (inFlight.length > concurrency) tree.release(inFlight.shift());
    evictedBlocks += tree.evict(capacityBlocks);
    hitBlocks += hit;
    totalBlocks += hashes.length;
    promptTokens += r.tokens.length;
    cachedTokens += hit * blockSize;
  }
  while (inFlight.length) tree.release(inFlight.shift());

  return {
    requests: requests.length,
    promptTokens,
    cachedTokens,
    outputTokens: requests.reduce((s, r) => s + r.outputTokens, 0),
    totalBlocks,
    hitBlocks,
    hitRate: totalBlocks ? hitBlocks / totalBlocks : 0,
    evictedBlocks,
    sizeBlocks: tree.size(),
    nodes: tree.nodeCount(),
  };
}

// ---------- step 5: the cost and latency model ----------

/**
 * Example multipliers on the base input-token price, of the same shape several hosted APIs use.
 * These are illustrative round numbers, not any vendor's published prices: writing a token into the
 * cache costs more than reading it fresh, reading it back costs far less, and output tokens cost
 * several times an input token.
 */
export const PRICING = { base: 1, cacheWrite: 1.25, cacheRead: 0.1, output: 4 };

/**
 * Prompt tokens served from cache are billed at `cacheRead`, the rest at `cacheWrite`
 * (they are written into the cache as a side effect of being processed), output at `output`.
 */
export function requestCost(promptTokens, cachedTokens, outputTokens, pricing = PRICING) {
  if (cachedTokens > promptTokens) throw new Error(`requestCost: cachedTokens ${cachedTokens} exceeds promptTokens ${promptTokens}`);
  return cachedTokens * pricing.cacheRead
    + (promptTokens - cachedTokens) * pricing.cacheWrite
    + outputTokens * pricing.output;
}

/**
 * The prefix hit rate `h` at which caching stops losing money on input tokens:
 * `h*cacheRead + (1-h)*cacheWrite = base`, so `h = (cacheWrite - base) / (cacheWrite - cacheRead)`.
 * Below this rate you pay the write premium more often than you collect the read discount.
 */
export function breakEvenHitRate(pricing = PRICING) {
  const denom = pricing.cacheWrite - pricing.cacheRead;
  if (denom <= 0) return 0;
  return Math.max(0, (pricing.cacheWrite - pricing.base) / denom);
}

/**
 * Time to first token. Prefill is compute-bound, so cached tokens are simply skipped;
 * `overheadMs` is everything that does not scale with prompt length (scheduling, sampling setup).
 * `prefillTokensPerSecond` defaults to 10000, a deliberately conservative figure for an 8B model on
 * one H100: NVIDIA's H100 SXM datasheet lists approximately 989 TFLOPS of dense BF16, and prefill
 * costs about 2 x 8e9 = 16 GFLOP per prompt token, so 10,000 tokens/s is about 16% utilisation
 * (40% would be roughly 25,000 tokens/s). Treat it as an order of magnitude.
 */
export function ttftMs(promptTokens, cachedTokens, { prefillTokensPerSecond = 10000, overheadMs = 15 } = {}) {
  const toPrefill = Math.max(0, promptTokens - cachedTokens);
  return overheadMs + (toPrefill / prefillTokensPerSecond) * 1000;
}
