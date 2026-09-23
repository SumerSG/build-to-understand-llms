// Tests for module 33. Models are (context, question, next) → string; lengths are word counts.

const NEEDLE_RE = /The secret code for (\w+) is (\d+)\./g;

/** Five-word filler sentences with no digits of four characters and no city names. */
function uniformFiller(n = 60) {
  const words = ['amber', 'birch', 'cedar', 'delta', 'ember', 'fjord', 'grove', 'heath', 'inlet', 'juniper'];
  return Array.from({ length: n }, (_, k) => `The ${words[k % 10]} path was ${words[(k * 3 + 1) % 10]}.`);
}

/** Reads the context like a perfect retriever: answers every value stored under the asked key. */
function oracle(context, question) {
  const key = /for (\w+)\?/.exec(question)[1];
  const vals = [...context.matchAll(NEEDLE_RE)].filter((m) => m[1] === key).map((m) => m[2]);
  return vals.length ? `The secret code for ${key} is ${vals.join(', ')}.` : 'Not found.';
}

/** Where (as a fraction of the context, in words) the first needle for the asked key starts. */
function needleFraction(context, question) {
  const key = /for (\w+)\?/.exec(question)[1];
  const i = context.indexOf(`The secret code for ${key} is`);
  const before = context.slice(0, i).split(/\s+/).filter(Boolean).length;
  return before / context.split(/\s+/).filter(Boolean).length;
}

export const tests = [
  // ---------- step 1: haystack ----------
  { step: 'haystack', name: 'filler stops at the first sentence that reaches length − needle words; tokens and text agree', run(m, T) {
    const needle = { key: 'Oslo', value: '4321', sentence: 'The secret code for Oslo is 4321.' };
    const h = m.buildHaystack({ length: 100, needles: [needle], depths: [0.5], next: T.rng(1), filler: uniformFiller() });
    T.ok(Array.isArray(h.sentences) && typeof h.text === 'string', 'buildHaystack must return { sentences, text, positions, tokens }');
    T.eq(h.tokens, 102, 'needle is 7 words, so filler must reach 93 words: 19 five-word sentences = 95, plus 7 = 102. Counting the needle toward the length is what makes "length" mean the prompt the model actually reads');
    T.eq(m.countTokens(h.text), h.tokens, '`tokens` must equal countTokens(text)');
    T.eq(h.text, h.sentences.join(' '), '`text` is the sentences joined by single spaces');
    T.eq(h.sentences.length, 20, '19 filler sentences plus the needle');
    T.eq(h.sentences[h.positions[0]], needle.sentence, '`positions[k]` must be the index of needle k in `sentences`');
  } },
  { step: 'haystack', name: 'the needle lands at the boundary nearest the requested depth in filler words (0 = first, 1 = last)', run(m, T) {
    const needle = { key: 'Lima', value: '7788', sentence: 'The secret code for Lima is 7788.' };
    for (const depth of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
      const h = m.buildHaystack({ length: 1500, needles: [needle], depths: [depth], next: T.rng(11), filler: m.FILLER });
      const idx = h.positions[0];
      T.ok(idx >= 0 && h.sentences[idx] === needle.sentence, `depth ${depth}: positions[0] must point at the needle sentence`);
      const before = m.countTokens(h.sentences.slice(0, idx).join(' '));
      const fillerTokens = h.tokens - 7;
      T.ok(Math.abs(before - depth * fillerTokens) <= 10, `depth ${depth}: ${before} filler words precede the needle, expected about ${Math.round(depth * fillerTokens)} (within one sentence). A depth grid is only meaningful if the needle really sits where you asked`);
      if (depth === 0) T.eq(idx, 0, 'depth 0 puts the needle first');
      if (depth === 1) T.eq(idx, h.sentences.length - 1, 'depth 1 puts the needle last, right before the question');
    }
    const at = (depth) => m.buildHaystack({ length: 100, needles: [{ key: 'Oslo', value: '4321', sentence: 'The secret code for Oslo is 4321.' }], depths: [depth], next: T.rng(1), filler: uniformFiller() }).positions[0];
    T.eq(at(0.33), 6, 'depth 0.33 of 95 filler words is 31.35; the nearest boundary has 30 words (6 sentences) before it. Pick the nearest boundary, not the first one at or past the target, and measure depth against the filler words, not the whole text');
    T.eq(at(0.62), 12, 'depth 0.62 of 95 filler words is 58.9; the nearest boundary has 60 words (12 sentences) before it');
  } },
  { step: 'haystack', name: 'filler that mentions a needle key or value is skipped (no accidental needle)', run(m, T) {
    const needle = { key: 'Oslo', value: '4321', sentence: 'The secret code for Oslo is 4321.' };
    const poison = ['The Oslo ferry was late.', 'Room 4321 had no window.', 'The oslo map was torn.'];
    const filler = uniformFiller(30).flatMap((s, k) => (k % 3 === 0 ? [s, poison[(k / 3) % 3]] : [s]));
    filler.push('Ticket 43210 was never sold.');
    const h = m.buildHaystack({ length: 300, needles: [needle], depths: [0.4], next: T.rng(2), filler });
    const hits = h.sentences.filter((s) => /\boslo\b|\b4321\b/i.test(s));
    T.eq(hits.length, 1, `found ${hits.length} sentences mentioning Oslo or 4321; only the needle may. Otherwise a model can answer from filler and your eval measures the wrong thing`);
    T.ok(h.tokens >= 300, 'skipping sentences must not shorten the haystack');
    T.ok(h.sentences.includes('Ticket 43210 was never sold.'), '43210 is not the value 4321: match whole words only, or you throw away harmless filler');
    // All-poison filler must throw. The Proxy caps how many filler sentences may be read, so a missing
    // full-lap throw fails here with a message instead of hanging the page in an endless loop.
    let reads = 0;
    const LOOP_GUARD = 'test guard: buildHaystack read over 1000 filler sentences';
    const guarded = new Proxy(poison, { get(t, p, r) {
      if (typeof p === 'string' && /^\d+$/.test(p) && ++reads > 1000) throw new Error(LOOP_GUARD);
      return Reflect.get(t, p, r);
    } });
    let err = null;
    try { m.buildHaystack({ length: 50, needles: [needle], depths: [0.5], next: T.rng(3), filler: guarded }); } catch (e) { err = e; }
    T.ok(err && err.message !== LOOP_GUARD, err ? 'with every filler sentence an accidental needle, buildHaystack kept skipping (over 1000 filler reads for a 3-sentence filler) instead of throwing: after filler.length skips in a row, throw, because there is no valid haystack and the loop would otherwise never end' : 'when every filler sentence is an accidental needle there is no valid haystack: expected buildHaystack to throw, but it returned');
    const long = m.buildHaystack({ length: 1000, needles: [needle], depths: [0.5], next: T.rng(4), filler: [...uniformFiller(10), poison[0]] });
    T.ok(long.tokens >= 1000, 'one bad sentence in 11 must not make a long haystack throw: it wraps around the filler about 20 times and meets that sentence on every lap, so throw only after filler.length skips in a row');
  } },
  { step: 'haystack', name: 'several needles, each at its own depth; the same seed gives the same haystack', run(m, T) {
    const a = { key: 'Perth', value: '1111', sentence: 'The secret code for Perth is 1111.' };
    const b = { key: 'Riga', value: '2222', sentence: 'The secret code for Riga is 2222.' };
    const h1 = m.buildHaystack({ length: 800, needles: [a, b], depths: [0.8, 0.2], next: T.rng(5) });
    const h2 = m.buildHaystack({ length: 800, needles: [a, b], depths: [0.8, 0.2], next: T.rng(5) });
    const h3 = m.buildHaystack({ length: 800, needles: [a, b], depths: [0.8, 0.2], next: T.rng(6) });
    T.eq(h1.text, h2.text, 'buildHaystack must draw all randomness from `next` so a seed reproduces the eval item');
    T.ok(h1.text !== h3.text, 'a different seed should start the filler at a different place');
    T.eq(h1.sentences.filter((s) => s === a.sentence || s === b.sentence).length, 2, 'each needle appears exactly once');
    T.eq(h1.sentences[h1.positions[0]], a.sentence, 'positions follow the order of `needles`, not of depth');
    T.ok(h1.positions[1] < h1.positions[0], 'needle b (depth 0.2) must come before needle a (depth 0.8)');
    T.eq(m.makeTask('multikey', { length: 400, depth: 0.5, next: T.rng(9) }).positions.length, 4, 'makeTask builds on your buildHaystack: multikey plants 4 needles');
  } },

  // ---------- step 2: graders ----------
  { step: 'graders', name: 'exactMatch normalises case, whitespace, a trailing period and thousands separators', run(m, T) {
    T.eq(m.exactMatch('4,821.', '4821'), 1, '"4,821." is the same answer as "4821" once normalised');
    T.eq(m.exactMatch('  4821 ', '4821'), 1, 'surrounding whitespace does not change the answer');
    T.eq(m.exactMatch('LISBON', 'lisbon'), 1, 'case does not change the answer');
    T.eq(m.exactMatch('The code is 4821.', '4821'), 0, 'exact match is strict: a sentence around the value is not an exact match');
    T.eq(m.exactMatch('4822', '4821'), 0, 'different values must not match');
    T.eq(typeof m.exactMatch('1', '1'), 'number', 'graders return numbers (they are averaged), not booleans');
  } },
  { step: 'graders', name: 'fuzzyMatch finds the value as a whole word inside a chatty answer', run(m, T) {
    T.eq(m.fuzzyMatch('The secret code for Oslo is 4,821.', '4821'), 1, 'a sentence that states the value is a correct retrieval');
    T.eq(m.fuzzyMatch('It is 48210.', '4821'), 0, '48210 contains 4821 as a substring but is a different number: match whole words');
    T.eq(m.fuzzyMatch('It is 482.', '4821'), 0, 'a prefix of the value is not the value');
    T.eq(m.fuzzyMatch('I could not find it.', '4821'), 0, 'a refusal is not a retrieval');
  } },
  { step: 'graders', name: 'recallFraction scores multi-value answers by the share of values recovered', run(m, T) {
    T.close(m.recallFraction('The codes are 1111, 2222.', ['1111', '2222', '3333']), 2 / 3, 1e-9, 'two of three values recovered is 2/3, not 1 and not 0');
    T.eq(m.recallFraction('nothing here', ['1111', '2222']), 0);
    T.eq(m.recallFraction('1111 2222', []), 0, 'no references: define the score as 0 rather than NaN');
  } },
  { step: 'graders', name: 'gradeTask picks the grader by task kind and penalises answers that list a distractor', run(m, T) {
    const niah = { kind: 'niah', answers: ['4821'], distractors: [] };
    const mk = { kind: 'multikey', answers: ['4821'], distractors: ['1234', '5678'] };
    const mv = { kind: 'multivalue', answers: ['1111', '2222', '3333', '4444'], distractors: [] };
    T.eq(m.gradeTask(niah, 'The secret code for Oslo is 4821.'), 1, 'niah is graded with fuzzyMatch');
    T.eq(m.gradeTask(mk, 'The secret code for Oslo is 4821.'), 1, 'the right value and no distractor scores 1');
    T.eq(m.gradeTask(mk, 'It is 1234.'), 0, 'a distractor value is a wrong retrieval');
    T.eq(m.gradeTask(mk, 'It is 4821 or 1234 or 5678.'), 0, 'listing every value must not score: otherwise "dump all numbers" beats real retrieval');
    T.close(m.gradeTask(mv, 'The codes are 2222 and 4444.'), 0.5, 1e-9, 'multivalue is graded with recallFraction');
  } },

  // ---------- step 3: runner ----------
  { step: 'runner', name: 'returns acc, lo and hi as [lengths][depths] grids; a perfect retriever scores 1 everywhere', run(m, T) {
    const r = m.contextEval(oracle, { lengths: [60, 120], depths: [0, 0.5, 1], trials: 4, seed: 1 });
    for (const f of ['acc', 'lo', 'hi']) {
      T.ok(Array.isArray(r[f]) && r[f].length === 2 && r[f].every((row) => row.length === 3), `result.${f} must be 2 rows (lengths) × 3 columns (depths)`);
    }
    T.eq(r.acc, [[1, 1, 1], [1, 1, 1]], 'the oracle reads every needle, so every cell is 1');
    T.eq(r.lo, [[1, 1, 1], [1, 1, 1]], 'with all scores 1 every bootstrap mean is 1');
    T.eq(r.lengths, [60, 120]); T.eq(r.depths, [0, 0.5, 1]); T.eq(r.trials, 4); T.eq(r.kind, 'niah');
  } },
  { step: 'runner', name: 'runs `trials` items per cell at the right length and depth', run(m, T) {
    const seen = [];
    const recorder = (context, question) => { seen.push({ n: m.countTokens(context), f: needleFraction(context, question) }); return 'none'; };
    m.contextEval(recorder, { lengths: [60, 150], depths: [0, 1], trials: 3, seed: 2 });
    T.eq(seen.length, 12, `expected 2 lengths × 2 depths × 3 trials = 12 model calls, got ${seen.length}`);
    T.ok(seen.slice(0, 6).every((s) => s.n >= 60 && s.n < 72) && seen.slice(6).every((s) => s.n >= 150 && s.n < 162), 'lengths are the outer loop: the first 6 calls at 60 words, the next 6 at 150');
    T.ok(seen.slice(0, 3).every((s) => s.f === 0) && seen.slice(3, 6).every((s) => s.f > 0.8), 'within a length, depth 0 comes first (needle at the start) then depth 1 (needle at the end)');
    const lateOnly = (context, question) => (needleFraction(context, question) > 0.5 ? oracle(context, question) : 'Not found.');
    const r = m.contextEval(lateOnly, { lengths: [100], depths: [0, 0.25, 0.75, 1], trials: 3, seed: 3 });
    T.eq(r.acc[0], [0, 0, 1, 1], 'a model that only sees the second half must fail at depths 0 and 0.25 and pass at 0.75 and 1: pass each cell\'s depth to makeTask');
  } },
  { step: 'runner', name: 'deterministic for a seed; the bootstrap interval brackets the accuracy', run(m, T) {
    const model = m.makeFadingModel({ window: 1000, reach: 120, sinkSpan: 20 });
    const opts = { lengths: [100, 300], depths: [0, 0.5, 1], trials: 8 };
    const a = m.contextEval(model, { ...opts, seed: 4 });
    const b = m.contextEval(model, { ...opts, seed: 4 });
    const c = m.contextEval(model, { ...opts, seed: 5 });
    T.eq(a.acc, b.acc, 'same seed, same grid: every draw (haystack, model, bootstrap) must come from seeded rngs');
    T.eq(a.hi, b.hi, 'the bootstrap must be seeded too');
    T.ok(JSON.stringify(a.acc) !== JSON.stringify(c.acc), 'a different seed should give different items and a different grid');
    const fewB = m.contextEval(model, { ...opts, seed: 4, B: 50 });
    T.eq(fewB.acc, a.acc, 'changing B (bootstrap resamples) must not change which items were drawn or how the model answered: give the bootstrap its own generator, rng(seed + 7919), separate from the item generator');
    let wide = 0;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) {
      T.ok(a.lo[i][j] <= a.acc[i][j] + 1e-9 && a.acc[i][j] <= a.hi[i][j] + 1e-9, `cell [${i}][${j}]: expected lo ≤ acc ≤ hi, got ${a.lo[i][j]} ≤ ${a.acc[i][j]} ≤ ${a.hi[i][j]}`);
      if (a.hi[i][j] - a.lo[i][j] > 0.2) wide++;
    }
    T.ok(wide > 0, 'with 8 trials a cell near 50% accuracy has a wide interval; report it, it is the honest error bar');
  } },
  { step: 'runner', name: 'the kind option reaches makeTask and gradeTask', run(m, T) {
    const dumpAll = (context) => [...context.matchAll(NEEDLE_RE)].map((x) => x[2]).join(' ');
    const mk = m.contextEval(dumpAll, { lengths: [120], depths: [0.5], trials: 4, kind: 'multikey', seed: 6 });
    T.eq(mk.kind, 'multikey');
    T.ok(Array.isArray(mk.acc) && Array.isArray(mk.acc[0]), 'contextEval must return an acc grid for every kind');
    T.eq(mk.acc[0][0], 0, 'on multikey, listing every number in the context names distractors and must score 0');
    const mv = m.contextEval(oracle, { lengths: [120], depths: [0.5], trials: 4, kind: 'multivalue', seed: 6 });
    T.eq(mv.acc[0][0], 1, 'the oracle recovers all three values on multivalue');
  } },

  // ---------- step 4: effective context ----------
  { step: 'effective', name: 'effectiveContext is the longest length before the first failure, not the longest passing one', run(m, T) {
    const r = { lengths: [1000, 2000, 4000, 8000], depths: [0, 1], acc: [[1, 0.9], [0.9, 0.9], [0.4, 0.8], [0.9, 0.8]] };
    T.close(m.lengthCurve(r), [0.95, 0.9, 0.6, 0.85], 1e-9, 'lengthCurve averages each row over depths');
    T.eq(m.effectiveContext(r, 0.8), 2000, 'accuracy fails at 4000; that 8000 recovers to 0.85 is noise or luck, and a claim of 8000 would be wrong');
    T.eq(m.effectiveContext(r, 0.9), 2000, 'a length whose mean equals the threshold counts as passing (>=)');
    T.eq(m.effectiveContext(r, 0.95), 1000);
    T.eq(m.effectiveContext(r, 0.99), 0, 'if even the shortest length fails, the effective context is 0');
  } },
  { step: 'effective', name: 'depthCurve reads a row; middleDrop compares the edges with the middle', run(m, T) {
    const r = { lengths: [1000, 2000], depths: [0, 0.5, 1], acc: [[1, 1, 1], [0.9, 0.3, 1]] };
    T.eq(m.depthCurve(r, 2000), [0.9, 0.3, 1]);
    const c = m.depthCurve(r, 2000); c[0] = 0;
    T.eq(r.acc[1][0], 0.9, 'depthCurve must return a copy so a caller cannot corrupt the result');
    T.throws(() => m.depthCurve(r, 3000), 'asking for a length that was not evaluated must throw, not return undefined');
    T.close(m.middleDrop([1, 0.5, 0.4, 0.6, 1]), 0.5, 1e-9, 'mean of the two edges (1) minus the mean of the interior (0.5)');
    T.close(m.middleDrop([0.8, 0.8, 0.8]), 0, 1e-9, 'a flat curve has no middle drop');
    T.throws(() => m.middleDrop([1, 1]), 'two depths have no middle');
  } },
  { step: 'effective', name: 'on the fading model: effective context is well below the window, and the middle sags only when long', run(m, T) {
    const model = m.makeFadingModel({ window: 1000, reach: 250, sinkSpan: 30 });
    const r = m.contextEval(model, { lengths: [100, 200, 400, 800], depths: [0, 0.25, 0.5, 0.75, 1], trials: 12, seed: 1 });
    const eff = m.effectiveContext(r, 0.8);
    T.ok(eff >= 100 && eff <= 400, `effective context ${eff} should be 100–400 words for a 1000-word window with reach 250: the window is an upper bound, not a measurement`);
    const longDrop = m.middleDrop(m.depthCurve(r, 800)), shortDrop = m.middleDrop(m.depthCurve(r, 100));
    T.ok(longDrop > 0.3, `at 800 words the middle should sag well below the edges (got drop ${longDrop.toFixed(2)})`);
    T.ok(shortDrop < 0.15, `at 100 words every depth is easy (got drop ${shortDrop.toFixed(2)})`);
  } },

  // ---------- step 5: cost ----------
  { step: 'cost', name: 'prefillFlops sums module 15\'s per-token cost over positions 1…T', run(m, T) {
    const cfg = { params: 100, layers: 2, dModel: 4 };
    T.eq(m.prefillFlops(cfg, 3), 792, '2·N·T = 600 for the weights, plus attention 4·L·C·(1+2+3) = 192 for the causal triangle');
    let sum = 0;
    for (let t = 1; t <= 50; t++) sum += 2 * 100 + 4 * 2 * t * 4;
    T.eq(m.prefillFlops(cfg, 50), sum, 'prefill of T tokens is the per-token cost 2N + 4·L·t·C summed over t = 1…T');
    const L = m.LLAMA3_8B;
    const r = m.prefillFlops(L, 131072) / m.prefillFlops(L, 65536);
    T.ok(r > 2.5 && r < 4, `doubling a 64k prompt to 128k multiplies Llama-3-8B prefill by ${r.toFixed(2)}: between linear (2) and quadratic (4), because attention is now most of the work`);
  } },
  { step: 'cost', name: 'kvCacheBytes uses the KV heads (GQA), not the query heads', run(m, T) {
    const L = m.LLAMA3_8B;
    T.eq(m.kvCacheBytes(L, 1), 131072, 'Llama-3-8B: 2 · 32 layers · 8 KV heads · 128 · 2 bytes = 128 KiB per token');
    T.eq(m.kvCacheBytes(L, 131072), 2 ** 34, 'a 128k-token prompt holds 16 GiB of cache for one sequence');
    T.eq(m.kvCacheBytes(L, 1000, 1), 1000 * 65536, 'bytesPerElement = 1 (fp8 cache) halves it');
  } },
  { step: 'cost', name: 'longPromptCost reports flops, attention share, seconds and cache bytes', run(m, T) {
    const L = m.LLAMA3_8B;
    const c4 = m.longPromptCost(L, 4096), c128 = m.longPromptCost(L, 131072);
    T.eq(c4.flops, m.prefillFlops(L, 4096)); T.eq(c128.kvBytes, m.kvCacheBytes(L, 131072)); T.eq(c128.tokens, 131072);
    T.ok(c4.attnShare > 0.04 && c4.attnShare < 0.09, `at 4k tokens attention is ~6% of prefill (got ${(100 * c4.attnShare).toFixed(1)}%)`);
    T.ok(c128.attnShare > 0.6 && c128.attnShare < 0.75, `at 128k tokens attention is ~68% of prefill (got ${(100 * c128.attnShare).toFixed(1)}%)`);
    T.close(c128.seconds, c128.flops / (989e12 * 0.5), 1e-9, 'seconds = flops / (peakFlops · mfu), defaults 989e12 and 0.5');
    T.close(m.longPromptCost(L, 4096, { mfu: 0.25 }).seconds, 2 * c4.seconds, 1e-9, 'halving MFU doubles the time');
  } },
];
