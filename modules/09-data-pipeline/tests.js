import { hash32 } from 'lib/util.js';

// ---------- fixtures ----------

/** A clean 40-word paragraph that passes every default rule. */
const CLEAN = 'The river was wide and slow, and the boat drifted with the current for most of the afternoon. ' +
  'Nobody spoke. The oars lay across the seats, and the water carried them past the mill and the old stone bridge.';

/** The same paragraph with two words changed: a near-duplicate, not an exact one. */
const NEAR = CLEAN.replace('wide and slow', 'wide and calm').replace('old stone bridge', 'old iron bridge');

/** An unrelated 40-word paragraph. */
const OTHER = 'Every morning the baker opened the shutters before dawn and lit the ovens one by one. ' +
  'By six the first loaves were out, and the smell of bread reached the square long before the market stalls were built.';

/** A stub tokenizer: one id per character, eos = 0. Fast, deterministic, and easy to reason about. */
const STUB = { encode: (s) => Array.from(s, (c) => (c.charCodeAt(0) % 200) + 1), eos: 0 };

function doc(id, text, domain = 'web') { return { id, text, domain }; }

/** n words of the form word<i> (the last one ends the line with a full stop) so word counts and shingle sets are exact. */
function wordsOf(n, offset = 0) { return Array.from({ length: n }, (_, i) => `word${i + offset}`).join(' ') + '.'; }

export const tests = [
  // ---------- step 1: quality ----------
  { step: 'quality', name: 'a clean paragraph passes; too few or too many words are rejected with the right reason', run(m, T) {
    T.eq(m.qualityReason(CLEAN), null, 'a 40-word paragraph of ordinary prose must pass every default rule');
    T.eq(m.qualityReason('Mr. Bennet made no answer.'), 'too_short', 'five words is below minWords (20)');
    T.eq(m.qualityReason(wordsOf(19)), 'too_short', '19 words is below minWords (20): the bound is inclusive');
    T.eq(m.qualityReason(wordsOf(20)), null, 'exactly minWords must pass (off-by-one)');
    T.eq(m.qualityReason(CLEAN, { ...m.QUALITY_DEFAULTS, maxWords: 30 }), 'too_long', '40 words exceeds maxWords 30');
    T.eq(m.qualityReason(CLEAN, { ...m.QUALITY_DEFAULTS, minWords: 5 }), null, 'the config argument must be honoured');
  } },
  { step: 'quality', name: 'mean word length and symbol ratio rules (Gopher)', run(m, T) {
    const shortWords = Array.from({ length: 30 }, (_, i) => 'a b c d'.split(' ')[i % 4]).join(' ') + '.';
    T.eq(m.qualityReason(shortWords), 'word_length', 'mean word length 1 is below 3: a document of single letters is not prose');
    const longWords = Array.from({ length: 25 }, (_, i) => 'x'.repeat(14) + i).join(' ') + '.';
    T.eq(m.qualityReason(longWords), 'word_length', 'mean word length 15 is above 10: base64, URLs and hashes look like this');
    const hashy = CLEAN + ' #one #two #three #four #five #six #seven #eight #nine #ten #eleven';
    T.eq(m.qualityReason(hashy), 'symbol_ratio', '11 hashes over 51 words is a ratio of 0.22 > 0.1');
    const ellipses = CLEAN + ' this... and... that... or... maybe... never... ever...';
    T.eq(m.qualityReason(ellipses), 'symbol_ratio', '7 ellipses over 47 words is 0.15 > 0.1');
    T.eq(m.qualityReason('#summer ' + CLEAN), null, 'one hash in 41 words (0.024) is fine');
  } },
  { step: 'quality', name: 'terminal punctuation, repeated lines and boilerplate rules', run(m, T) {
    const menu = Array.from({ length: 24 }, (_, i) => `Menu item number ${i}`).join('\n');
    T.eq(m.qualityReason(menu), 'terminal_punct', 'no line ends in . ! or ? — a navigation menu, not prose');
    const mostlyList = 'Item one two three\nItem four five six\nItem seven eight nine\nItem ten eleven twelve\nItem thirteen fourteen fifteen\nItem sixteen seventeen eighteen\nThis one is a real sentence, nineteen twenty.';
    T.eq(m.qualityReason(mostlyList), 'terminal_punct', '1 of 7 lines (14%) ends in terminal punctuation, below 30%');
    const spam = Array.from({ length: 10 }, () => 'Buy now and save today.').join('\n') + '\nOffer ends soon, so hurry.';
    T.eq(m.qualityReason(spam), 'repeated_lines', '9 of 11 lines repeat an earlier line (82% > 30%)');
    const CLEAN2 = CLEAN.replace('river', 'lake');
    T.eq(m.qualityReason([CLEAN, OTHER, OTHER].join('\n')), 'repeated_lines', '1 of 3 lines (33%) repeats an earlier line, above the 30% limit');
    T.eq(m.qualityReason([CLEAN, OTHER, OTHER, CLEAN2].join('\n')), null, '1 duplicate line of 4 (25%) is under the limit: the rule is "> 0.3", and lines are compared exactly');
    T.eq(m.qualityReason(CLEAN + ' Lorem ipsum dolor sit amet.'), 'boilerplate', '"lorem ipsum" is placeholder text (C4 drops it)');
    T.eq(m.qualityReason(CLEAN + ' Please enable JavaScript to continue.'), 'boilerplate', 'boilerplate matching is case-insensitive');
  } },
  { step: 'quality', name: 'qualityFilter splits documents and records the reason per removed id', run(m, T) {
    const docs = [doc('a', CLEAN), doc('b', 'Too short.'), doc('c', OTHER), doc('d', CLEAN + ' Lorem ipsum.')];
    const r = m.qualityFilter(docs);
    T.eq(r.kept.map((d) => d.id), ['a', 'c'], 'kept documents, in input order');
    T.eq(r.removed, [{ id: 'b', reason: 'too_short' }, { id: 'd', reason: 'boilerplate' }], 'every removal names the rule that fired');
    T.eq(m.qualityFilter(docs, { ...m.QUALITY_DEFAULTS, minWords: 1, boilerplate: [] }).kept.length, 4, 'the config must be passed through to qualityReason');
  } },

  // ---------- step 2: exact dedup ----------
  { step: 'exact', name: 'normalizeText ignores case, punctuation and whitespace; docHash is hash32 of it', run(m, T) {
    T.eq(m.normalizeText('  Hello,   World! '), 'hello world');
    T.eq(m.normalizeText('A-B\tc\n\nD'), 'a b c d', 'newlines, tabs and hyphens all become single spaces');
    T.eq(m.normalizeText('HELLO WORLD'), m.normalizeText('hello world'));
    T.eq(m.docHash('Hello, World!'), hash32('hello world'), 'docHash must be hash32 of the normalised text, so equal texts collide on purpose');
    T.ok(m.docHash(CLEAN) !== m.docHash(OTHER), 'different texts must (almost always) get different fingerprints');
  } },
  { step: 'exact', name: 'exactDedup keeps the first copy and removes later ones that differ only in case, punctuation or spacing', run(m, T) {
    const docs = [doc('a', CLEAN), doc('b', OTHER), doc('c', CLEAN.toUpperCase()), doc('d', '  ' + CLEAN.replace(/,/g, '') + '\n'), doc('e', OTHER)];
    const r = m.exactDedup(docs);
    T.eq(r.kept.map((d) => d.id), ['a', 'b'], 'first occurrences are kept, in order');
    T.eq(r.removed, [{ id: 'c', dupOf: 'a' }, { id: 'd', dupOf: 'a' }, { id: 'e', dupOf: 'b' }], 'each removal names the surviving copy');
  } },
  { step: 'exact', name: 'exactDedup is exact: a single changed word is not a duplicate', run(m, T) {
    const r = m.exactDedup([doc('a', CLEAN), doc('b', NEAR)]);
    T.eq(r.kept.length, 2, 'NEAR differs from CLEAN by two words; exact dedup must keep both (that is the near-dedup step\'s job)');
    T.eq(r.removed, []);
    T.eq(m.exactDedup([]).kept, [], 'empty input');
  } },

  // ---------- step 3: near dedup ----------
  { step: 'minhash', name: 'shingles are k-word windows over the normalised text', run(m, T) {
    const s = m.shingles('The cat sat on the mat today.', 5);
    T.eq([...s].sort(), ['cat sat on the mat', 'sat on the mat today', 'the cat sat on the'].sort(), '7 words give 3 five-word shingles (off-by-one check)');
    T.eq(m.shingles('a b c d e f', 3).size, 4, '6 words, k=3 → 4 shingles');
    T.eq(m.shingles('Hello, World! Hello World', 2).size, 2, 'normalisation: "hello world" and "world hello" (case and punctuation ignored, duplicates collapse)');
    T.eq(m.shingles('one two', 5).size, 1, 'fewer than k words → one shingle');
    T.eq(m.shingles('', 5).size, 0, 'empty text → no shingles');
  } },
  { step: 'minhash', name: 'jaccard is exact and minhash signatures have one minimum per hash function', run(m, T) {
    const A = new Set(['a', 'b', 'c', 'd']), B = new Set(['c', 'd', 'e', 'f']);
    T.close(m.jaccard(A, B), 2 / 6, 1e-9, '|A∩B| = 2, |A∪B| = 6');
    T.close(m.jaccard(A, A), 1, 1e-9);
    T.close(m.jaccard(A, new Set(['z'])), 0, 1e-9);
    const sig = m.minhash(m.shingles(CLEAN), 32, 1);
    T.eq(sig.length, 32, 'one value per hash function');
    T.ok(new Set(sig).size > 16, 'the 32 hash functions must differ from each other (different a_i, b_i), otherwise the estimate has one sample instead of 32');
    T.ok(sig.every((v) => Number.isInteger(v) && v >= 0 && v < 4294967296), 'signature entries are unsigned 32-bit integers');
    T.eq(m.minhash(new Set(['x', 'y', 'z']), 16, 7), m.minhash(new Set(['z', 'x', 'y']), 16, 7), 'the signature depends on the set, not on insertion order');
    T.ok(m.minhash(new Set(['x']), 16, 7).some((v, i) => v !== m.minhash(new Set(['x']), 16, 8)[i]), 'a different seed gives different hash functions');
    const sx = m.minhash(new Set(['x']), 8, 3), sxy = m.minhash(new Set(['x', 'y']), 8, 3);
    T.ok(sxy.every((v, i) => v <= sx[i]), 'adding a shingle can only lower each minimum');
  } },
  { step: 'minhash', name: 'estimateJaccard is the fraction of agreeing positions and tracks the exact Jaccard', run(m, T) {
    const s1 = m.minhash(m.shingles(CLEAN), 32, 1);
    T.close(m.estimateJaccard(s1, s1), 1, 1e-9, 'identical sets agree everywhere: the estimate is a fraction in [0, 1], not a count');
    T.close(m.estimateJaccard([1, 2, 3, 4], [1, 9, 3, 9]), 0.5, 1e-9);
    T.throws(() => m.estimateJaccard([1, 2], [1, 2, 3]), 'signatures of different length cannot be compared');
    const a = m.shingles(CLEAN), b = m.shingles(NEAR), c = m.shingles(OTHER);
    const exactAB = m.jaccard(a, b), exactAC = m.jaccard(a, c);
    T.ok(exactAB > 0.6 && exactAB < 0.95, `fixture sanity: CLEAN vs NEAR exact Jaccard ${exactAB.toFixed(3)}`);
    const estAB = m.estimateJaccard(m.minhash(a, 128, 1), m.minhash(b, 128, 1));
    T.ok(Math.abs(estAB - exactAB) < 0.15, `with 128 hashes the estimate (${estAB.toFixed(3)}) must be within 0.15 of the exact Jaccard (${exactAB.toFixed(3)}); std of the estimate is sqrt(J(1-J)/128) ≈ 0.04`);
    const estAC = m.estimateJaccard(m.minhash(a, 128, 1), m.minhash(c, 128, 1));
    T.ok(estAC < 0.1, `unrelated paragraphs (exact ${exactAC.toFixed(3)}) must estimate near 0, got ${estAC.toFixed(3)}`);
  } },
  { step: 'minhash', name: 'nearDedup removes near-duplicates above the threshold, keeps unrelated documents, and reports pairs with both similarities', run(m, T) {
    const docs = [doc('a', CLEAN), doc('b', OTHER), doc('c', NEAR), doc('d', CLEAN + ' One extra sentence at the end.'), doc('e', wordsOf(40, 500))];
    const r = m.nearDedup(docs, { k: 5, numHashes: 64, threshold: 0.6, seed: 1 });
    T.eq(r.kept.map((d) => d.id), ['a', 'b', 'e'], 'c and d are near-copies of a; b and e are unrelated and must survive');
    T.eq(r.removed.map((x) => x.id), ['c', 'd']);
    T.eq(r.removed.map((x) => x.nearOf), ['a', 'a'], 'each removal names the earlier kept document it matched');
    T.ok(r.removed.every((x) => x.estimate >= 0.6 && x.estimate <= 1), 'the recorded estimate is the one that crossed the threshold');
    const ac = r.pairs.find((p) => p.a === 'a' && p.b === 'c');
    T.ok(ac && typeof ac.jaccard === 'number' && typeof ac.estimate === 'number', 'pairs carry the estimate and the exact Jaccard for comparison');
    T.close(ac.jaccard, m.jaccard(m.shingles(CLEAN), m.shingles(NEAR)), 1e-9);
    const strict = m.nearDedup(docs, { k: 5, numHashes: 64, threshold: 0.99, seed: 1 });
    T.eq(strict.kept.length, 5, 'with threshold 0.99 nothing is removed: the threshold must be applied, not hard-coded');
    T.eq(m.nearDedup([doc('x', CLEAN), doc('y', CLEAN)], { threshold: 0.8 }).removed.map((x) => x.id), ['y'], 'an exact copy is also a near-duplicate (estimate 1)');
  } },

  // ---------- step 4: mixing ----------
  { step: 'mix', name: 'a zero-weight or absent domain is never drawn; every drawn id exists once per epoch', run(m, T) {
    const docs = [];
    for (let i = 0; i < 6; i++) docs.push(doc(`a${i}`, wordsOf(10), 'a'));
    for (let i = 0; i < 4; i++) docs.push(doc(`b${i}`, wordsOf(10), 'b'));
    for (let i = 0; i < 3; i++) docs.push(doc(`c${i}`, wordsOf(10), 'c'));
    const r = m.mixDomains(docs, { weights: { a: 1, b: 0 }, maxEpochs: 1, seed: 1 });
    T.eq(r.order.slice().sort(), ['a0', 'a1', 'a2', 'a3', 'a4', 'a5'], 'weight 0 and unlisted domains contribute nothing; with maxEpochs 1 each doc of a appears exactly once');
    T.eq(r.report.a.draws, 6); T.close(r.report.a.epochs, 1, 1e-9);
    T.eq(r.total, 60, 'total is the sum of document sizes; the default size is the word count (6 docs × 10 words)');
    T.throws(() => m.mixDomains(docs, {}), 'weights are required');
  } },
  { step: 'mix', name: 'the running share of each domain tracks the weights (Megatron-style blending)', run(m, T) {
    const docs = [];
    for (let i = 0; i < 200; i++) docs.push(doc(`a${i}`, wordsOf(10 + (i % 7)), 'a'));
    for (let i = 0; i < 200; i++) docs.push(doc(`b${i}`, wordsOf(30), 'b'));
    for (let i = 0; i < 200; i++) docs.push(doc(`c${i}`, wordsOf(5), 'c'));
    const r = m.mixDomains(docs, { weights: { a: 0.5, b: 0.3, c: 0.2 }, budget: 3000, maxEpochs: 1, seed: 2 });
    T.ok(r.total >= 3000 && r.total < 3000 + 40, `stop as soon as the budget is met: total ${r.total}`);
    T.ok(Math.abs(r.report.a.share - 0.5) < 0.03, `domain a should hold ≈50% of the size, got ${(100 * r.report.a.share).toFixed(1)}%: weights are shares of size, not of document counts`);
    T.ok(Math.abs(r.report.b.share - 0.3) < 0.03, `domain b should hold ≈30%, got ${(100 * r.report.b.share).toFixed(1)}%`);
    T.ok(Math.abs(r.report.c.share - 0.2) < 0.03, `domain c should hold ≈20%, got ${(100 * r.report.c.share).toFixed(1)}%`);
    T.ok(r.report.b.draws < r.report.c.draws, 'b documents are 6× larger than c documents, so b must be drawn fewer times for a larger share');
    T.eq(r.order.length, r.report.a.draws + r.report.b.draws + r.report.c.draws);
    const early = r.order.slice(0, 60).filter((id) => id[0] === 'a').length;
    T.ok(early > 15 && early < 50, `domains must be interleaved, not concatenated: ${early} of the first 60 draws are from a`);
  } },
  { step: 'mix', name: 'epochs are capped per domain, and an exhausted domain\'s weight goes to the others', run(m, T) {
    const docs = [];
    for (let i = 0; i < 5; i++) docs.push(doc(`hq${i}`, wordsOf(10), 'hq'));
    for (let i = 0; i < 100; i++) docs.push(doc(`web${i}`, wordsOf(10), 'web'));
    const r = m.mixDomains(docs, { weights: { hq: 0.5, web: 0.5 }, budget: 400, maxEpochs: 2, seed: 3 });
    T.eq(r.report.hq.draws, 10, 'hq has 5 documents and a 2-epoch cap: at most 10 draws, however large its weight');
    T.close(r.report.hq.epochs, 2, 1e-9);
    T.eq(r.report.web.draws, 30, 'the remaining 300 words of the 400 budget come from web once hq is exhausted');
    T.ok(r.total >= 400, 'the budget must still be met from the domains that remain');
    const firstPass = r.order.filter((id) => id.startsWith('hq')).slice(0, 5), secondPass = r.order.filter((id) => id.startsWith('hq')).slice(5);
    T.eq(firstPass.slice().sort(), ['hq0', 'hq1', 'hq2', 'hq3', 'hq4'], 'the first epoch visits every hq document once');
    T.eq(secondPass.slice().sort(), ['hq0', 'hq1', 'hq2', 'hq3', 'hq4'], 'the second epoch visits every hq document once more');
    T.eq(m.mixDomains(docs, { weights: { hq: 1 }, budget: 400, maxEpochs: 2, seed: 3 }).total, 100, 'when every domain is exhausted, stop below the budget rather than looping forever');
    const s1 = m.mixDomains(docs, { weights: { hq: 0.5, web: 0.5 }, budget: 400, maxEpochs: 2, seed: 3 });
    T.eq(s1.order, r.order, 'the same seed gives the same order');
  } },

  // ---------- step 5: shards + pipeline ----------
  { step: 'shards', name: 'tokenizeDocs attaches ids and packShards cuts the eos-separated stream into fixed shards', run(m, T) {
    const docs = m.tokenizeDocs([doc('a', 'abcde'), doc('b', 'fgh'), doc('c', 'ijklmnop')], STUB);
    T.eq(docs.map((d) => d.ids.length), [5, 3, 8]);
    T.eq(docs[0].text, 'abcde', 'the original fields are preserved');
    const { shards, index } = m.packShards(docs, ['a', 'b', 'c'], { shardSize: 6, eos: 0 });
    T.eq(shards.map((s) => s.length), [6, 6, 6, 1], '5+1 + 3+1 + 8+1 = 19 tokens → three full shards of 6 and a tail of 1');
    T.eq(shards[0], [...docs[0].ids, 0], 'the first shard is doc a followed by one eos');
    T.eq(shards[1].slice(0, 4), [...docs[1].ids, 0], 'doc b and its eos start the second shard');
    T.eq(shards[3], [0], 'the final eos lands alone in the tail shard');
    T.eq(index, [
      { docId: 'a', shard: 0, offset: 0, length: 5 },
      { docId: 'b', shard: 1, offset: 0, length: 3 },
      { docId: 'c', shard: 1, offset: 4, length: 8 },
    ], 'the index records where each document starts and how many tokens it has (excluding eos)');
  } },
  { step: 'shards', name: 'the index reads every document back exactly, in the requested order, across shard boundaries', run(m, T) {
    const next = T.rng(5);
    const docs = m.tokenizeDocs(Array.from({ length: 12 }, (_, i) => doc(`d${i}`, 'x'.repeat(1 + Math.floor(next() * 40)) + i)), STUB);
    const order = ['d3', 'd7', 'd0', 'd11', 'd7', 'd5'];
    const { shards, index } = m.packShards(docs, order, { shardSize: 16, eos: 0 });
    T.eq(index.map((e) => e.docId), order, 'one index entry per position in `order` (a document drawn twice appears twice)');
    let total = 0;
    for (const id of order) total += docs.find((d) => d.id === id).ids.length + 1;
    T.eq(shards.reduce((s, x) => s + x.length, 0), total, 'no tokens lost or invented: sum of lengths = Σ(len + 1 eos)');
    T.ok(shards.slice(0, -1).every((s) => s.length === 16), 'every shard except the last has exactly shardSize tokens');
    for (const e of index) T.eq(m.readDoc(shards, e), docs.find((d) => d.id === e.docId).ids, `reading ${e.docId} back through the index must give its ids`);
    T.throws(() => m.packShards(docs, ['nope'], { shardSize: 16, eos: 0 }), 'an unknown id in `order` must throw');
  } },
  { step: 'shards', name: 'runPipeline chains the stages and reports what each removed', run(m, T) {
    const docs = [
      doc('k1', CLEAN, 'books'), doc('k2', OTHER, 'books'), doc('k3', wordsOf(40, 900), 'web'),
      doc('junk', 'Too short.', 'web'), doc('dup', CLEAN.toUpperCase(), 'web'), doc('near', NEAR, 'web'),
      doc('lorem', OTHER + ' Lorem ipsum dolor.', 'web'),
    ];
    const r = m.runPipeline(docs, { tokenizer: STUB, near: { threshold: 0.6, numHashes: 64 }, mix: { weights: { books: 0.5, web: 0.5 }, maxEpochs: 1 }, shardSize: 32 });
    T.eq(r.report.map((s) => [s.stage, s.in, s.out, s.removed]), [['quality', 7, 5, 2], ['exact-dedup', 5, 4, 1], ['near-dedup', 4, 3, 1], ['mix+shard', 3, 3, 0]],
      'stage report: quality drops junk and lorem, exact dedup drops dup, near dedup drops near');
    T.eq(r.removed.map((x) => [x.id, x.stage]), [['junk', 'quality'], ['lorem', 'quality'], ['dup', 'exact-dedup'], ['near', 'near-dedup']]);
    const survivors = new Set(r.index.map((e) => e.docId));
    T.eq([...survivors].sort(), ['k1', 'k2', 'k3']);
    const expectTokens = [CLEAN, OTHER, wordsOf(40, 900)].reduce((s, t) => s + STUB.encode(t).length + 1, 0);
    T.eq(r.tokens, expectTokens, 'total tokens = Σ(doc tokens + 1 eos) over the survivors');
    T.ok(r.shards.slice(0, -1).every((s) => s.length === 32), 'shardSize must be passed through');
    T.ok(r.mix && r.mix.report.books && r.mix.report.web, 'the mixing report is returned');
    T.close(r.mix.report.books.size + r.mix.report.web.size, r.tokens - r.index.length, 1e-9, 'mixing must measure size in tokens (ids), not words');
    T.ok(Array.isArray(r.pairs), 'the near-duplicate pairs are returned for the report');
    T.throws(() => m.runPipeline(docs, {}), 'a tokenizer is required');
  } },
];
