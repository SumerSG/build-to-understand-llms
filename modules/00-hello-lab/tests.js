// A comparator that subtracts the pairs themselves (y - x) gives NaN, so the counts never decide the order:
// the result keeps the Map's order (a NaN comparator changes nothing) or, after `|| localeCompare`, comes out
// alphabetical. Say so, instead of letting the generic "expected … but got …" message point elsewhere.
function ignoredCounts(T, got, original, alphabetical) {
  const words = Array.isArray(got) ? got.map((p) => (Array.isArray(p) ? p[0] : p)) : null;
  const same = (want) => words && words.length === want.length && words.every((w, i) => w === want[i]);
  if (same(original) || same(alphabetical)) {
    T.fail(`topK returned ${JSON.stringify(got)}: the ${same(original) ? 'original order' : 'alphabetical order'}, so the counts played no part in the sort. x and y in the comparator are whole [word, count] pairs, and subtracting two pairs (y - x) gives NaN. Subtract their counts instead: x[1] and y[1]`);
  }
}

export const tests = [
  { step: 'tokenize', name: 'splits a sentence into lowercase words', run(m, T) {
    T.eq(m.tokenizeWords('Alice was beginning to get very tired.'), ['alice', 'was', 'beginning', 'to', 'get', 'very', 'tired'], 'words should be lowercased and punctuation dropped');
  } },
  { step: 'tokenize', name: 'keeps apostrophes as part of words and splits on hyphens and digits', run(m, T) {
    T.eq(m.tokenizeWords("Alice's well-lit room, 3 chairs"), ["alice's", 'well', 'lit', 'room', 'chairs']);
  } },
  { step: 'tokenize', name: 'returns an empty array for text with no words', run(m, T) {
    T.eq(m.tokenizeWords('!!! 42 a'), ['a'], 'the one letter in "!!! 42 a" is a word (this check also makes sure the step is not still the starter stub)');
    T.eq(m.tokenizeWords('123 ... !!!'), [], 'no letters means no words (watch out: String.match returns null)');
  } },
  { step: 'count', name: 'counts repeated words', run(m, T) {
    const c = m.countFrequencies(['a', 'b', 'a', 'c', 'a', 'b']);
    T.ok(c instanceof Map, 'should return a Map');
    T.ok(!(c.has('0') || c.has(0)), `the Map's keys are positions ("0", "1", …) instead of words: for...in loops over an array's positions; use for (const w of words) to loop over the words themselves`);
    for (const [w, n] of [['a', 3], ['b', 2], ['c', 1]]) {
      T.ok(c.has(w), `the returned Map has no entry for "${w}" (it has ${c.size} entries): did you return the empty Map without adding each word to it?`);
      T.eq(c.get(w), n, `"${w}" appears ${n} time(s) in ['a', 'b', 'a', 'c', 'a', 'b']`);
    }
    T.eq(c.size, 3, 'one key per distinct word');
  } },
  { step: 'count', name: 'handles awkward keys like "constructor"', run(m, T) {
    const c = m.countFrequencies(['constructor', '__proto__', 'constructor']);
    T.ok(c instanceof Map, 'should return a Map');
    T.ok(!(c.has('0') || c.has(0)), `the Map's keys are positions ("0", "1", …) instead of words: for...in loops over an array's positions; use for (const w of words) to loop over the words themselves`);
    for (const [w, n] of [['constructor', 2], ['__proto__', 1]]) {
      T.ok(c.has(w), `the returned Map has no entry for "${w}" (it has ${c.size} entries): did you return the empty Map without adding each word to it?`);
      T.eq(c.get(w), n, `"${w}" appears ${n} time(s); a plain object would confuse this key with a built-in property`);
    }
  } },
  { step: 'topk', name: 'returns the k most frequent, most frequent first', run(m, T) {
    ignoredCounts(T, m.topK(new Map([['b', 2], ['a', 2], ['c', 5], ['d', 1]]), 2), ['b', 'a'], ['a', 'b']);
    T.eq(m.topK(new Map([['b', 2], ['a', 2], ['c', 5], ['d', 1]]), 2), [['c', 5], ['a', 2]]);
  } },
  { step: 'topk', name: 'compares counts as numbers, so 10 ranks above 9', run(m, T) {
    ignoredCounts(T, m.topK(new Map([['nine', 9], ['ten', 10], ['two', 2], ['eleven', 11]]), 3), ['nine', 'ten', 'two'], ['eleven', 'nine', 'ten']);
    T.eq(m.topK(new Map([['nine', 9], ['ten', 10], ['two', 2], ['eleven', 11]]), 3), [['eleven', 11], ['ten', 10], ['nine', 9]],
      'a string comparison puts "9" above "10" and "11"; subtract the counts instead');
    T.eq(m.topK(new Map([['a', 3]]), 0), [], 'k = 0 asks for no words');
  } },
  { step: 'topk', name: 'breaks ties alphabetically and copes with k larger than the vocabulary', run(m, T) {
    T.eq(m.topK(new Map([['pear', 1], ['apple', 1], ['fig', 1]]), 10), [['apple', 1], ['fig', 1], ['pear', 1]]);
  } },
  { step: 'zipf', name: 'predicts topCount / rank', run(m, T) {
    const p = m.zipfPredicted(100, 4);
    T.ok(Array.isArray(p) && p.length === 4, `zipfPredicted(100, 4) should return an array of 4 numbers, one per rank, but returned ${Array.isArray(p) ? `an array of ${p.length}` : String(p)}`);
    T.close(p, [100, 50, 33.3333, 25], 1e-3, 'element i is 100 / (i + 1)');
  } },
  { step: 'zipf', name: 'rank 1 equals topCount, n = 1 and n = 0 work', run(m, T) {
    const p = m.zipfPredicted(60, 6);
    T.ok(Array.isArray(p) && p.length === 6, `zipfPredicted(60, 6) should return an array of 6 numbers but returned ${Array.isArray(p) ? `an array of ${p.length}` : String(p)}`);
    T.close(p, [60, 30, 20, 15, 12, 10], 1e-9, 'element i is 60 / (i + 1), unrounded');
    T.close(m.zipfPredicted(7, 1), [7], 1e-9, 'with one rank the prediction is just topCount');
    T.eq(m.zipfPredicted(7, 0), [], 'n = 0 asks for no ranks, so return []');
  } },
  { step: 'logerr', name: 'log error is zero for a perfect Zipf sequence and positive otherwise', run(m, T) {
    // Literal arrays, so a bug in zipfPredicted cannot show up here.
    const ideal = [64, 32, 64 / 3, 16, 12.8, 64 / 6];
    const first = m.zipfLogError(ideal, ideal);
    if (Number.isNaN(first)) {
      T.fail('zipfLogError returned NaN for two identical arrays: is it still the starter stub (return NaN)? Otherwise NaN usually means an element that does not exist was read (undefined): loop over positions with for (let i = 0; i < actual.length; i++) and read actual[i] and predicted[i], rather than looping over values with for...of');
    }
    T.close(m.zipfLogError(ideal, ideal), 0, 1e-9, 'identical arrays are 0 apart');
    T.close(m.zipfLogError([64, 64, 64], [64, 32, 64 / 3]), (Math.log(2) + Math.log(3)) / 3, 1e-6, 'mean of |log a - log p|: (0 + log 2 + log 3) / 3');
  } },
  { step: 'logerr', name: 'log error takes the absolute value, so points above and below the line do not cancel', run(m, T) {
    // actual[1] is half the prediction (below the line), actual[2] is 3x the prediction (above it).
    // The prediction is written out so this test does not depend on zipfPredicted.
    const want = (Math.log(2) + Math.log(3)) / 3;
    const got = m.zipfLogError([60, 15, 60], [60, 30, 20]);
    if (typeof got === 'number' && !Number.isFinite(got)) {
      T.fail(`zipfLogError([60, 15, 60], [60, 30, 20]) returned ${got}, expected ≈ ${want.toFixed(3)}: is it still the starter stub, or did Math.log see 0, a negative number or undefined (an element that does not exist, for example from looping over values instead of positions)?`);
    }
    const signed = (Math.log(3) - Math.log(2)) / 3;
    const why = typeof got === 'number' && Math.abs(Math.abs(got) - signed) < 1e-6
      ? `; you got ≈ ${signed.toFixed(3)}, which is what happens without Math.abs: the −log 2 and +log 3 partly cancel`
      : '';
    T.close(got, want, 1e-6, `expected (log 2 + log 3) / 3 ≈ ${want.toFixed(3)}${why}`);
    T.close(m.zipfLogError([10], [5]), Math.log(2), 1e-9, 'one pair off by a factor of 2 gives natural log 2 ≈ 0.693 (use Math.log, not Math.log10)');
  } },
];
