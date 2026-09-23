// Module 17 — tests. Block hashes are opaque 32-bit numbers, so the tests never assert a specific
// hash value; they assert the *relationships* a hash chain must have. The radix tree is checked
// against a brute-force "longest common prefix over everything inserted so far" reference.

const B = 16; // BLOCK_SIZE; repeated here so a learner who changes the constant sees the tests move with it

/** Tokens 0,1,2,…,n-1 — distinct on purpose, so any off-by-one in blocking shows up. */
function ramp(n) {
  return Array.from({ length: n }, (_, i) => i);
}

/** A copy of `tokens` with position `i` replaced by a token that appears nowhere else. */
function mutate(tokens, i) {
  const out = tokens.slice();
  out[i] = 10000 + i;
  return out;
}

export const tests = [
  // ---------- step 1: block hashes and the exact-prefix cache ----------
  {
    step: 'blocks',
    name: 'hashes are block-granular: whole blocks only, and a prefix of a prompt hashes to a prefix of its hashes',
    run(m, T) {
      T.eq(m.blockHashes(ramp(33)).length, 2, `33 tokens contain two complete ${B}-token blocks; the 33rd token has no full block yet, so it is not cacheable`);
      T.eq(m.blockHashes(ramp(15)).length, 0, 'fewer tokens than one block means nothing can be cached');
      T.eq(m.blockHashes(ramp(32)).length, 2, '32 tokens is exactly two blocks');
      T.eq(m.blockHashes(ramp(48)).length, 3, '48 tokens is exactly three blocks');
      const full = m.blockHashes(ramp(64));
      T.eq(m.blockHashes(ramp(32)), full.slice(0, 2), 'the hash of block i must depend only on tokens 0..(i+1)*16-1, so hashing a prefix gives a prefix of the hashes');
      T.eq(m.blockHashes(ramp(33), 8).length, 4, 'blockSize must be honoured: 33 tokens is four complete blocks of 8');
    },
  },
  {
    step: 'blocks',
    name: 'the hash chains: identical blocks at different positions get different hashes, and an early edit invalidates everything after it',
    run(m, T) {
      const repeated = Array(B).fill(7).concat(Array(B).fill(7));
      const h = m.blockHashes(repeated);
      T.eq(h.length, 2);
      T.ok(h[0] !== h[1], 'two identical blocks must get different hashes: block 1 sits behind block 0, so its identity includes block 0. Hashing each block on its own would make these equal and would let a request match a prefix it does not share');
      const base = ramp(48);
      const edited = mutate(base, 3); // a token inside block 0
      const hb = m.blockHashes(base), he = m.blockHashes(edited);
      T.ok(hb[0] !== he[0], 'editing a token inside block 0 must change block 0\'s hash');
      T.ok(hb[1] !== he[1], 'editing a token inside block 0 must also change block 1\'s hash: every later block is invalidated');
      T.ok(hb[2] !== he[2], 'and block 2, and every block after it');
      const late = mutate(base, 40); // a token inside block 2
      const hl = m.blockHashes(late);
      T.eq(hl.slice(0, 2), hb.slice(0, 2), 'editing a token in block 2 must leave blocks 0 and 1 untouched, or nothing would ever hit');
      T.ok(hl[2] !== hb[2], 'but block 2 itself must change');
    },
  },
  {
    step: 'blocks',
    name: 'HashCache.lookup returns the longest cached prefix, counted in blocks and in tokens',
    run(m, T) {
      const cache = new m.HashCache();
      T.eq(cache.lookup(ramp(33)), { blocks: 0, tokens: 0 }, 'an empty cache can serve nothing');
      T.eq(cache.insert(ramp(33)), 2, 'inserting a 33-token prompt stores 2 blocks, not 3 and not 33');
      T.eq(cache.lookup(ramp(33)), { blocks: 2, tokens: 32 }, 'the same 33-token prompt hits both blocks: 32 of its 33 tokens are cached');
      T.eq(cache.insert(ramp(33)), 0, 'inserting the same prompt again stores nothing new');
      T.eq(cache.lookup(mutate(ramp(33), 20)), { blocks: 1, tokens: 16 },
        'token 20 lives in block 1, so block 0 still hits and block 1 does not: expected exactly one block');
      T.eq(cache.lookup(mutate(ramp(33), 0)), { blocks: 0, tokens: 0 },
        'token 0 lives in block 0, so nothing can be reused');
      T.eq(cache.lookup(ramp(64)), { blocks: 2, tokens: 32 },
        'a longer prompt that starts with the cached one hits its first two blocks and stops: lookup must stop at the first miss, not keep scanning');
      T.eq(cache.size(), 2, 'the cache holds 2 distinct blocks');

      // Simulate an eviction of block 0 by deleting its entry from the cache's Map directly.
      const gap = new m.HashCache();
      gap.insert(ramp(48));
      gap.blocks.delete(m.blockHashes(ramp(48))[0]);
      T.eq(gap.lookup(ramp(48)), { blocks: 0, tokens: 0 },
        'block 0 is gone, so blocks 1 and 2 are unusable even though their hashes are still in the Map: the KV of block 1 was computed on top of block 0. lookup must stop at the first miss, not count every hash it finds');
      const small = new m.HashCache(8);
      T.eq(small.insert(ramp(33)), 4, 'a HashCache built with blockSize 8 stores four blocks of a 33-token prompt');
      T.eq(small.lookup(ramp(40)), { blocks: 4, tokens: 32 }, 'and reports tokens in units of its own blockSize');
    },
  },

  // ---------- step 2: the radix tree ----------
  {
    step: 'radix',
    name: 'match returns the number of leading blocks found, and insert stores each distinct prefix once',
    run(m, T) {
      const tree = new m.PrefixTree();
      T.eq(tree.match([1, 2, 3]).blocks, 0, 'an empty tree matches nothing');
      T.eq(tree.size(), 0);
      tree.insert([1, 2, 3]);
      T.eq(tree.size(), 3, 'three block hashes were inserted, so the tree holds three blocks');
      T.eq(tree.nodeCount(), 1, 'a single un-branched path is one node, not three: that compression is what makes it a radix tree');
      T.eq(tree.match([1, 2, 3]).blocks, 3, 'the full sequence is cached');
      T.eq(tree.match([1, 2, 9]).blocks, 2, 'the first two blocks match and the third does not');
      T.eq(tree.match([1, 2]).blocks, 2, 'a shorter query matches as far as it goes');
      T.eq(tree.match([9, 2, 3]).blocks, 0, 'a different first block means no shared prefix at all, even though blocks 2 and 3 are in the tree');
      T.eq(tree.match([]).blocks, 0, 'an empty query matches zero blocks');
      T.ok(tree.match([9]).node === tree.root, 'when nothing matches, match must return the root as its node');
      tree.insert([1, 2, 4]);
      T.eq(tree.match([1, 2, 7]).node.key, [1, 2], 'match must return the deepest node FULLY covered by the match: here the shared head [1,2]');
    },
  },
  {
    step: 'radix',
    name: 'a diverging path splits the node it diverges inside, and shared blocks are still stored once',
    run(m, T) {
      const tree = new m.PrefixTree();
      tree.insert([1, 2, 3]);
      tree.insert([1, 2, 4]);
      T.eq(tree.size(), 4, 'blocks 1 and 2 are shared, so the tree holds 1,2,3,4 — four blocks, not six');
      T.eq(tree.nodeCount(), 3, 'expected three nodes: the shared head [1,2] and one leaf per branch. Appending without splitting gives 2, re-storing the shared head gives 4');
      T.eq(tree.match([1, 2]).blocks, 2, 'the shared head must still be matchable on its own');
      T.eq(tree.match([1, 2, 3]).blocks, 3);
      T.eq(tree.match([1, 2, 4]).blocks, 3, 'the other branch matches all three of its blocks too');
      tree.insert([1]);
      T.eq(tree.size(), 4, 'inserting a prefix that is already present adds no blocks');
      T.eq(tree.nodeCount(), 4, 'inserting [1] must split the head [1,2] into [1] and [2]');
      T.eq(tree.match([1, 5]).blocks, 1, 'after the split, a query that diverges after block 1 matches exactly one block');
    },
  },
  {
    step: 'radix',
    name: 'match finds the LONGEST prefix, and matches a brute-force reference on 40 random inserts',
    run(m, T) {
      const short = new m.PrefixTree();
      short.insert([1, 2]);
      short.insert([1, 2, 3, 4]);
      T.eq(short.match([1, 2, 3, 4, 5]).blocks, 4,
        'both [1,2] and [1,2,3,4] are cached; the match must walk past the first node and return 4, not 2');

      const next = T.rng(17);
      const tree = new m.PrefixTree();
      const inserted = [];
      for (let i = 0; i < 40; i++) {
        const len = 1 + Math.floor(next() * 6);
        const seq = Array.from({ length: len }, () => 1 + Math.floor(next() * 4));
        let best = 0;
        for (const s of inserted) {
          let c = 0;
          while (c < s.length && c < seq.length && s[c] === seq[c]) c++;
          if (c > best) best = c;
        }
        T.eq(tree.match(seq).blocks, best,
          `match([${seq}]) must equal the longest common prefix with anything inserted so far (${best})`);
        tree.insert(seq);
        inserted.push(seq);
      }
      const prefixes = new Set();
      for (const s of inserted) for (let k = 1; k <= s.length; k++) prefixes.add(s.slice(0, k).join(','));
      T.eq(tree.size(), prefixes.size,
        `the tree must hold each distinct prefix exactly once (${prefixes.size} of them); a larger number means shared blocks were duplicated`);
    },
  },

  // ---------- step 3: reference counting and LRU eviction ----------
  {
    step: 'evict',
    name: 'eviction removes the least recently used leaf first and stops at the capacity',
    run(m, T) {
      const tree = new m.PrefixTree();
      tree.insert([1, 2]);
      tree.insert([3, 4]);
      tree.insert([5, 6]);
      T.eq(tree.size(), 6);
      tree.match([5, 6]);
      tree.match([1, 2]); // [3,4] is now the least recently used leaf
      T.eq(tree.evict(4), 2, 'the tree holds 6 blocks and the capacity is 4, so exactly one 2-block leaf must go');
      T.eq(tree.size(), 4);
      T.eq(tree.match([3, 4]).blocks, 0, '[3,4] was the least recently used and must be the one evicted');
      T.eq(tree.match([1, 2]).blocks, 2, '[1,2] was used most recently and must survive');
      T.eq(tree.match([5, 6]).blocks, 2, '[5,6] was used after [3,4] and must survive');
      T.eq(tree.evict(4), 0, 'already at capacity: nothing more to free');
      T.eq(tree.evict(100), 0, 'a capacity above the current size frees nothing');
    },
  },
  {
    step: 'evict',
    name: 'a pinned path is never evicted, however old it is, and becomes evictable after release',
    run(m, T) {
      const tree = new m.PrefixTree();
      const pinned = tree.acquire([1, 2]); // oldest, but in use
      tree.insert([3, 4]);
      tree.match([3, 4]);
      T.eq(tree.size(), 4);
      T.eq(tree.evict(0), 2, 'capacity 0 must free everything that is free: the unpinned [3,4] leaf, 2 blocks');
      T.eq(tree.size(), 2, 'the pinned path must still be there');
      T.eq(tree.match([1, 2]).blocks, 2, 'a request is still reading these KV blocks; evicting them would corrupt it');
      T.eq(tree.evict(0), 0, 'nothing evictable remains, so evict must return 0 rather than loop forever');
      tree.release(pinned);
      T.eq(tree.evict(0), 2, 'once released, the same path is evictable');
      T.eq(tree.size(), 0);
    },
  },
  {
    step: 'evict',
    name: 'a node with children is never evicted before its children (a parent is a prefix of its child)',
    run(m, T) {
      const tree = new m.PrefixTree();
      tree.insert([1, 2, 3, 4]);
      tree.insert([1, 2, 9]);
      T.eq(tree.size(), 5, 'blocks 1,2 shared plus 3,4 plus 9');
      tree.match([1, 2, 3, 4]);
      tree.match([1, 2, 9]); // [3,4] is the LRU leaf
      T.eq(tree.evict(3), 2, 'the only evictable leaves are [3,4] and [9]; the older one, [3,4], goes first');
      T.eq(tree.match([1, 2, 9]).blocks, 3, 'the shared head [1,2] must survive because [9] still hangs off it');
      T.eq(tree.match([1, 2, 3, 4]).blocks, 2, 'only the [3,4] tail was lost');
      T.eq(tree.size(), 3);
      const pinnedTree = new m.PrefixTree();
      const deep = pinnedTree.acquire([1, 2, 3]);
      pinnedTree.insert([1, 2, 9]);
      T.eq(pinnedTree.evict(0), 1, 'the interior node [1,2] is on a pinned path, so only the unpinned [9] leaf, 1 block, can go');
      T.eq(pinnedTree.match([1, 2, 3]).blocks, 3);
      pinnedTree.release(deep);
      T.eq(pinnedTree.evict(0), 3,
        'the later insert split the pinned node in two; release must still unpin both halves, so the whole 3-block path is now evictable');
      T.eq(pinnedTree.size(), 0, 'a reference count that leaks on a split pins blocks forever and the cache slowly fills with garbage');

      const shared = new m.PrefixTree();
      const a = shared.acquire([1, 2]);   // request A is decoding over [1,2]
      const b = shared.acquire([1, 2, 3]); // request B extends it
      shared.release(b);                  // B finishes; A is still running
      T.eq(shared.evict(0), 1, 'only B\'s private block [3] may go: A still holds [1,2]. If acquire pins only the end node, B\'s release drops [1,2] to a count of zero and A\'s KV is evicted under it');
      T.eq(shared.match([1, 2]).blocks, 2, 'request A\'s prefix must survive while A is in flight');
      shared.release(a);
      T.eq(shared.evict(0), 2, 'after A is released its path is evictable too');
    },
  },

  // ---------- step 4: the traffic model ----------
  {
    step: 'traffic',
    name: 'the workload is deterministic and every turn extends the previous turn of its conversation',
    run(m, T) {
      const a = m.makeWorkload({ conversations: 6, turns: 4, seed: 3 });
      const b = m.makeWorkload({ conversations: 6, turns: 4, seed: 3 });
      T.eq(a.length, 24, '6 conversations x 4 turns = 24 requests');
      T.eq(a.map((r) => r.tokens.length), b.map((r) => r.tokens.length), 'the same seed must give byte-identical traffic, or no measurement is reproducible');
      T.eq(a[0].tokens, b[0].tokens);
      const c = m.makeWorkload({ conversations: 6, turns: 4, seed: 4 });
      T.ok(c.some((r, i) => r.tokens[r.tokens.length - 1] !== a[i].tokens[a[i].tokens.length - 1]), 'a different seed must give different traffic');

      const byConv = new Map();
      for (const r of a) {
        if (!byConv.has(r.conv)) byConv.set(r.conv, []);
        byConv.get(r.conv).push(r);
      }
      T.eq(byConv.size, 6, 'there must be one group of turns per conversation');
      for (const turnsOf of byConv.values()) {
        T.eq(turnsOf.map((r) => r.turn), [0, 1, 2, 3], 'turns of one conversation must arrive in order');
        for (let i = 1; i < turnsOf.length; i++) {
          const prev = turnsOf[i - 1].tokens, cur = turnsOf[i].tokens;
          T.ok(cur.length > prev.length, 'each turn must be longer than the last (it carries the whole history)');
          T.ok(prev.every((t, j) => cur[j] === t),
            'turn t must be a strict token prefix of turn t+1; if it is not, prefix caching has nothing to hit and the whole module measures noise');
        }
      }
      const sys = 64, fs = 96, user = 32, reply = 48; // the makeWorkload defaults
      T.eq(a.map((r) => r.id), a.map((_, i) => i), 'ids must be 0, 1, 2, … in arrival order');
      for (const r of a) {
        T.eq(r.tokens.length, sys + fs + (r.turn + 1) * user + r.turn * reply,
          `turn ${r.turn} must be system (${sys}) + few-shot (${fs}) + ${r.turn + 1} user messages (${user} each) + ${r.turn} replies (${reply} each); the assistant's replies are part of the history`);
        T.eq(r.outputTokens, reply, 'outputTokens is the assistant reply length, replyTokens');
      }
      T.ok(a.every((r) => r.tokens.slice(0, sys).every((t, j) => t === a[0].tokens[j])),
        'every request must open with the SAME system prompt; that shared prefix across conversations is what a cache exploits first');
      const big = m.makeWorkload({ conversations: 30, turns: 1, seed: 8 });
      const bundles = new Set(big.map((r) => r.tokens.slice(sys, sys + fs).join(',')));
      T.ok(bundles.size > 1 && bundles.size <= 3,
        `each conversation picks one of the 3 few-shot variants, so 30 conversations should show 2 or 3 distinct bundles; got ${bundles.size}`);
      const convs = new Set(a.slice(0, 12).map((r) => r.conv));
      T.ok(convs.size > 1, 'conversations must interleave rather than run one after another, or capacity would never matter');
      T.ok(a.every((r) => r.tokens.every((t) => Number.isInteger(t))), 'tokens must be integer ids');
    },
  },
  {
    step: 'traffic',
    name: 'the simulation counts hits consistently and reports zero hits for the first request',
    run(m, T) {
      const w = m.makeWorkload({ conversations: 4, turns: 3, seed: 5 });
      const s = m.simulate(w, { capacityBlocks: 1 << 20 });
      T.eq(s.requests, 12);
      T.eq(s.promptTokens, w.reduce((acc, r) => acc + r.tokens.length, 0), 'promptTokens must be the tokens actually sent');
      T.eq(s.totalBlocks, w.reduce((acc, r) => acc + m.blockHashes(r.tokens).length, 0), 'totalBlocks must count every block of every request, hit or miss');
      T.close(s.hitRate, s.hitBlocks / s.totalBlocks, 1e-9, 'hitRate is hit blocks divided by total blocks');
      T.eq(s.cachedTokens, s.hitBlocks * 16, 'each hit block saves exactly BLOCK_SIZE tokens of prefill');
      T.ok(s.hitBlocks < s.totalBlocks, 'the first request of the run cannot hit anything, so the hit rate must be below 1');
      T.ok(s.hitRate > 0.2, `with a shared system prompt and growing chats the hit rate must be well above zero; got ${s.hitRate.toFixed(3)}`);
      T.eq(s.evictedBlocks, 0, 'with a capacity of a million blocks nothing should ever be evicted');
      T.ok(s.nodes > 1 && s.sizeBlocks > 0, 'the tree must actually hold the traffic it saw');

      const eight = m.simulate(w, { capacityBlocks: 1 << 20, blockSize: 8 });
      T.eq(eight.totalBlocks, w.reduce((acc, r) => acc + m.blockHashes(r.tokens, 8).length, 0), 'simulate must hash with the blockSize it is given');
      T.eq(eight.cachedTokens, eight.hitBlocks * 8, 'with blockSize 8 each hit block saves 8 tokens');

      // Prompts that are not a whole number of blocks: the tail is never cached, so the block hit rate
      // and the token hit rate differ. hitRate is defined on blocks.
      const odd = m.makeWorkload({ conversations: 4, turns: 3, seed: 5, userTokens: 20, replyTokens: 30 });
      const so = m.simulate(odd, { capacityBlocks: 1 << 20 });
      T.ok(odd.some((r) => r.tokens.length % 16 !== 0), 'this workload has prompts with a partial last block');
      T.close(so.hitRate, so.hitBlocks / so.totalBlocks, 1e-9, 'hitRate is hit BLOCKS over total BLOCKS, not cachedTokens / promptTokens');
      T.eq(so.cachedTokens, so.hitBlocks * 16, 'a partial tail block is never cached, so cachedTokens is always a whole number of blocks');

      // In-flight requests are pinned. With zero capacity, a window of 4 still shares the system prompt;
      // a window of 0 releases every request before evicting, so nothing survives to be hit.
      const pinned = m.simulate(w, { capacityBlocks: 0, concurrency: 4 });
      const unpinned = m.simulate(w, { capacityBlocks: 0, concurrency: 0 });
      T.eq(unpinned.hitBlocks, 0, 'with capacity 0 and concurrency 0, each request is released before evict runs (release first, then evict, as the instructions order it), so every block is gone before the next request arrives');
      T.ok(pinned.hitBlocks > 0, 'with 4 requests pinned in flight, their blocks cannot be evicted and later requests hit them: simulate must keep `concurrency` requests acquired and release only the oldest');

      const single = m.simulate(w.slice(0, 1), { capacityBlocks: 1 << 20 });
      T.eq(single.hitBlocks, 0, 'a single request against a cold cache hits nothing');
      T.eq(single.hitRate, 0);
    },
  },
  {
    step: 'traffic',
    name: 'hit rate rises with cache capacity, and a tight capacity forces evictions',
    run(m, T) {
      const w = m.makeWorkload({ conversations: 20, turns: 6, seed: 9 });
      const tight = m.simulate(w, { capacityBlocks: 32 });
      const roomy = m.simulate(w, { capacityBlocks: 1 << 20 });
      T.ok(tight.evictedBlocks > 0, 'a 32-block cache cannot hold this traffic, so it must evict');
      T.ok(roomy.evictedBlocks === 0, 'an effectively unbounded cache must not evict');
      T.ok(tight.hitRate < roomy.hitRate - 0.05,
        `a tiny cache must lose hits: got ${tight.hitRate.toFixed(3)} at 32 blocks and ${roomy.hitRate.toFixed(3)} unbounded`);
      T.ok(tight.hitRate >= 0 && roomy.hitRate <= 1, 'a hit rate is a fraction between 0 and 1');
      T.eq(tight.totalBlocks, roomy.totalBlocks, 'capacity changes what is cached, never what is requested');
      const mid = m.simulate(w, { capacityBlocks: 512 });
      T.ok(mid.hitRate >= tight.hitRate, 'more capacity must never hurt: 512 blocks must do at least as well as 32');
    },
  },

  // ---------- step 5: the cost and latency model ----------
  {
    step: 'cost',
    name: 'the cost model charges cached tokens at the read multiplier and the rest at the write multiplier',
    run(m, T) {
      const p = m.PRICING;
      T.close(m.requestCost(1000, 0, 100, p), 1000 * p.cacheWrite + 100 * p.output, 1e-9,
        'with nothing cached every prompt token is written into the cache at 1.25x, so a cold request costs MORE than an uncached one');
      T.close(m.requestCost(1000, 1000, 100, p), 1000 * p.cacheRead + 100 * p.output, 1e-9,
        'with everything cached every prompt token is billed at 0.1x');
      T.close(m.requestCost(1000, 500, 100, p), 500 * p.cacheRead + 500 * p.cacheWrite + 100 * p.output, 1e-9,
        'half cached: the two halves are billed at different multipliers');
      T.close(m.requestCost(0, 0, 0, p), 0, 1e-9, 'an empty request costs nothing');
      T.ok(m.requestCost(1000, 1000, 100, p) < m.requestCost(1000, 500, 100, p),
        'cost must fall as more of the prompt is cached');
      T.ok(m.requestCost(1000, 0, 100, p) > m.baselineCost(1000, 100, p),
        'a cold cached request must cost more than the same request with caching switched off; that premium is why break-even exists');
      T.throws(() => m.requestCost(100, 200, 10, p), 'more cached tokens than prompt tokens is impossible and must throw rather than produce a negative bill');
      const cheapWrite = { base: 1, cacheWrite: 1, cacheRead: 0.5, output: 2 };
      T.close(m.requestCost(100, 40, 10, cheapWrite), 40 * 0.5 + 60 * 1 + 10 * 2, 1e-9, 'the pricing argument must actually be used, not the default');
    },
  },
  {
    step: 'cost',
    name: 'break-even hit rate follows (cacheWrite - base) / (cacheWrite - cacheRead)',
    run(m, T) {
      T.close(m.breakEvenHitRate(m.PRICING), 0.25 / 1.15, 1e-6,
        'with a 1.25x write and a 0.1x read the prefix hit rate must reach about 21.7% before caching saves money');
      T.close(m.breakEvenHitRate({ base: 1, cacheWrite: 1, cacheRead: 0.1, output: 4 }), 0, 1e-9,
        'if writing to the cache is free, any hit rate at all is a win');
      T.close(m.breakEvenHitRate({ base: 1, cacheWrite: 0.9, cacheRead: 0.1, output: 4 }), 0, 1e-9,
        'if writing is cheaper than the base price, caching wins at every hit rate: return 0, never a negative rate');
      T.close(m.breakEvenHitRate({ base: 1, cacheWrite: 2, cacheRead: 0, output: 4 }), 0.5, 1e-9,
        'a 2x write and a free read need half the prompt to hit before caching pays');
      T.ok(m.breakEvenHitRate({ base: 1, cacheWrite: 1.5, cacheRead: 0.1, output: 4 }) > m.breakEvenHitRate(m.PRICING),
        'a more expensive cache write must push break-even higher');
      // At exactly the break-even rate the input bill matches the uncached bill.
      const h = m.breakEvenHitRate(m.PRICING);
      const prompt = 10000, cached = Math.round(prompt * h);
      T.close(m.requestCost(prompt, cached, 0, m.PRICING), m.baselineCost(prompt, 0, m.PRICING), 1e-3,
        'at the break-even hit rate the cached bill must equal the uncached bill for the same prompt');
    },
  },
  {
    step: 'cost',
    name: 'time to first token drops linearly with the cached prefix',
    run(m, T) {
      const opts = { prefillTokensPerSecond: 10000, overheadMs: 15 };
      T.close(m.ttftMs(10000, 0, opts), 15 + 1000, 1e-6, '10,000 tokens at 10,000 tokens/s is 1 s of prefill plus 15 ms of fixed overhead');
      T.close(m.ttftMs(10000, 10000, opts), 15, 1e-6, 'a fully cached prompt skips prefill entirely and leaves only the fixed overhead');
      T.close(m.ttftMs(10000, 9000, opts), 15 + 100, 1e-6, 'only the uncached 1,000 tokens are prefilled');
      T.close(m.ttftMs(10000, 5000, opts) - m.ttftMs(10000, 6000, opts), 100, 1e-6,
        'each extra 1,000 cached tokens must remove the same 100 ms; the relationship is linear, not a flat discount');
      T.ok(m.ttftMs(10000, 0, { prefillTokensPerSecond: 20000, overheadMs: 15 }) < m.ttftMs(10000, 0, opts),
        'a faster prefill rate must lower TTFT; the rate argument must be used');
      T.ok(m.ttftMs(100, 100, opts) > 0, 'TTFT is never zero: scheduling and sampling still cost something');
    },
  },
];
