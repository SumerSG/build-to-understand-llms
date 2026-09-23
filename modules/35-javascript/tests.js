// Tests for "JavaScript for this lab". Messages are written for someone who is new to programming.

// Fail with a plain message when a function is missing (usually a deleted `export` or a misspelt name).
function need(m, name) {
  if (typeof m[name] !== 'function') {
    throw new Error(`the tests cannot find a function called ${name}. Check that the line still starts with "export function ${name}(" and that the name is spelt exactly like that.`);
  }
  return m[name];
}

const show = (x) => (typeof x === 'string' ? `'${x}'` : Array.isArray(x) ? JSON.stringify(x) : String(x));
const plainPairs = (map) => (map instanceof Map ? [...map] : map);

export const tests = [
  // ---------- step 1: values ----------
  { step: 'values', name: 'label puts the letter and the count into one piece of text', run(m, T) {
    const label = need(m, 'label');
    const got = label('e', 3);
    if (typeof got === 'string' && got.includes('${')) {
      T.fail(`label('e', 3) gave ${show(got)}: the \${...} slots were not filled in. Only backticks \` \` fill them; ordinary quotes ' ' or " " keep the text exactly as typed.`);
    }
    T.eq(got, 'e: 3', `label('e', 3) should be the letter, a colon, one space, then the count`);
    T.eq(label('z', 0), 'z: 0', `label('z', 0): a count of 0 still appears`);
    T.eq(label('the', 12), 'the: 12', 'label works for any text and any number');
  } },
  { step: 'values', name: 'surprise is minus the natural log of the share', run(m, T) {
    const surprise = need(m, 'surprise');
    const got = surprise(1, 2);
    if (typeof got === 'number' && Math.abs(got + Math.log(2)) < 1e-9) {
      T.fail(`surprise(1, 2) gave ${got.toFixed(3)}, a negative number: the log of a share below 1 is negative, so put a minus sign in front: -Math.log(...)`);
    }
    if (typeof got === 'number' && Math.abs(Math.abs(got) - Math.log10(2)) < 1e-9) {
      T.fail(`surprise(1, 2) gave ${got.toFixed(3)}, which is the base-10 log. Use Math.log (the natural log), not Math.log10.`);
    }
    T.close(got, Math.log(2), 1e-9, 'surprise(1, 2): a share of 1/2 gives -Math.log(0.5) ≈ 0.693');
    T.close(surprise(3, 12), Math.log(4), 1e-9, 'surprise(3, 12): a share of 1/4 gives ≈ 1.386 (rarer means more surprising)');
    T.close(surprise(5, 5), 0, 1e-12, 'surprise(5, 5): something that always happens is no surprise at all, so 0');
  } },
  { step: 'values', name: 'a share of 0 is infinitely surprising, and Math.exp undoes Math.log', run(m, T) {
    const surprise = need(m, 'surprise');
    const s = surprise(3, 12);
    T.close(Math.exp(-s), 0.25, 1e-9, `Math.exp(-surprise(3, 12)) should give the share 3/12 = 0.25 back, but surprise(3, 12) returned ${s}`);
    T.eq(surprise(0, 10), Infinity, 'surprise(0, 10): the share is 0 and Math.log(0) is -Infinity, so the surprise is Infinity (JavaScript does not crash; it gives this special number)');
  } },

  // ---------- step 2: loops ----------
  { step: 'loops', name: 'range(n) lists the whole numbers from 0 up to n - 1', run(m, T) {
    const range = need(m, 'range');
    const got = range(4);
    if (Array.isArray(got) && got.length === 4 && got[0] === 1) T.fail(`range(4) gave ${show(got)}: counting starts at 0 here, so start the loop with let i = 0`);
    if (Array.isArray(got) && got.length === 5) T.fail(`range(4) gave ${show(got)}, five numbers: keep going while i < n, not i <= n`);
    T.eq(got, [0, 1, 2, 3], 'range(4) should be [0, 1, 2, 3]: four numbers, starting at 0');
    T.eq(range(1), [0], 'range(1) is just [0]');
    T.eq(range(0), [], 'range(0) is an empty array: the loop runs zero times');
  } },
  { step: 'loops', name: 'sum adds up the values of an array', run(m, T) {
    const sum = need(m, 'sum');
    const got = sum([2, 3, 5]);
    if (typeof got === 'string') {
      T.fail(`sum([2, 3, 5]) gave the text ${show(got)}. That is what for...in does: it walks the positions '0', '1', '2' as text, and adding text glues it together. Use for (const x of numbers) to get the values.`);
    }
    if (got === 3) T.fail('sum([2, 3, 5]) gave 3, which is 0 + 1 + 2: the loop added up the positions instead of the values. Use for (const x of numbers).');
    T.eq(got, 10, 'sum([2, 3, 5]) should be 10');
    T.eq(sum([]), 0, 'the sum of an empty array is 0');
    T.close(sum([0.5, -1.5, 4]), 3, 1e-12, 'sum works with fractions and negative numbers');
  } },
  { step: 'loops', name: 'sum also works on a Float32Array (the number arrays later modules use)', run(m, T) {
    const sum = need(m, 'sum');
    const got = sum(Float32Array.of(1, 2, 3, 4));
    if (typeof got === 'string') T.fail(`sum(Float32Array.of(1, 2, 3, 4)) gave the text ${show(got)}: for...in walks positions as text. Use for (const x of numbers).`);
    T.eq(got, 10, 'for...of walks a Float32Array exactly like an ordinary array');
  } },
  { step: 'loops', name: 'firstK copies the first k items and leaves the original alone', run(m, T) {
    const firstK = need(m, 'firstK');
    const items = [5, 6, 7, 8];
    const got = firstK(items, 2);
    T.eq(got, [5, 6], 'firstK([5, 6, 7, 8], 2) should be [5, 6]: items.slice(0, k) takes positions 0 up to (not including) k');
    T.ok(got !== items, 'firstK should return a new array, not the one it was given (slice makes a copy)');
    T.eq(items, [5, 6, 7, 8], 'the original array must be unchanged');
    T.eq(firstK([5], 3), [5], 'asking for more items than there are gives all of them');
    T.eq(firstK([5, 6], 0), [], 'k = 0 asks for no items');
  } },

  // ---------- step 3: compare ----------
  { step: 'compare', name: 'sameShape says true for equal shapes held in two different arrays', run(m, T) {
    const sameShape = need(m, 'sameShape');
    const got = sameShape([2, 3], [2, 3]);
    if (got === false) {
      T.fail('sameShape([2, 3], [2, 3]) said false. These are two separate arrays that hold the same numbers, and === on arrays asks "is this the very same array?", not "do they hold the same numbers?". Compare the lengths, then each position.');
    }
    T.eq(got, true, 'sameShape([2, 3], [2, 3]) should be true (return true or false, nothing else)');
    T.eq(sameShape([], []), true, 'two empty shapes are the same');
    T.eq(sameShape([4, 1, 7], [4, 1, 7]), true, 'three numbers, all equal');
  } },
  { step: 'compare', name: 'sameShape says false when the lengths or the numbers differ', run(m, T) {
    const sameShape = need(m, 'sameShape');
    T.eq(sameShape([2, 3], [2, 3]), true, 'first, a check that equal shapes give true (so a function that always says false cannot pass this test)');
    T.eq(sameShape([2, 3], [3, 2]), false, '[2, 3] and [3, 2] hold the same numbers in a different order: not the same shape');
    T.eq(sameShape([2, 3], [2, 3, 1]), false, '[2, 3] and [2, 3, 1] have different lengths; check a.length !== b.length first');
    T.eq(sameShape([2, 3, 1], [2, 3]), false, '[2, 3, 1] and [2, 3] have different lengths');
    T.eq(sameShape([6], [2, 3]), false, '[6] and [2, 3] both have 6 cells, but they are different shapes');
  } },

  // ---------- step 4: count ----------
  { step: 'count', name: 'counts each letter of a word', run(m, T) {
    const countLetters = need(m, 'countLetters');
    const c = countLetters('banana');
    T.ok(c instanceof Map, `countLetters should return a Map, but it returned ${show(c)}`);
    if (c.size === 0) T.fail("countLetters('banana') returned an empty Map: did you loop over the letters and call counts.set(letter, ...) for each one?");
    if (c.has('0') || c.has(0)) T.fail(`the Map's keys are positions (${[...c.keys()].map(show).join(', ')}), not letters: for...in walks positions. Loop with for (const letter of letters).`);
    for (const [k, v] of c) {
      if (typeof v === 'number' && Number.isNaN(v)) T.fail(`the count for '${k}' is NaN. The first time a letter appears, counts.get(letter) is undefined, and undefined + 1 is NaN. Write (counts.get(letter) || 0) + 1.`);
    }
    T.eq(plainPairs(c), [['b', 1], ['a', 3], ['n', 2]], "'banana' has b once, a three times, n twice (the Map keeps letters in the order they first appear)");
  } },
  { step: 'count', name: 'ignores upper/lower case, spaces, digits and punctuation', run(m, T) {
    const countLetters = need(m, 'countLetters');
    const c = countLetters("A a, B! It's 2024.");
    T.ok(c instanceof Map, 'countLetters should return a Map');
    if (c.has('A') || c.has('B')) T.fail('the Map has upper-case keys: lower-case the text first with text.toLowerCase()');
    T.eq(plainPairs(c), [['a', 2], ['b', 1], ['i', 1], ['t', 1], ['s', 1]], `"A a, B! It's 2024." has a twice, then b, i, t, s once each; spaces, digits and punctuation are not letters`);
  } },
  { step: 'count', name: 'text with no letters gives an empty Map instead of an error', run(m, T) {
    const countLetters = need(m, 'countLetters');
    T.eq(plainPairs(countLetters('Zz')), [['z', 2]], "'Zz' is two z's (this line also makes sure the step is not still the starter)");
    let c;
    try { c = countLetters('2024 ...!!!'); }
    catch (e) { T.fail(`countLetters('2024 ...!!!') crashed with "${e.message}". When there are no letters, match gives null, not an empty array; write match(...) || [] so the loop has an array to walk.`); }
    T.ok(c instanceof Map && c.size === 0, 'no letters means an empty Map');
    T.ok(countLetters('').size === 0, 'the empty text has no letters');
  } },

  // ---------- step 5: rank ----------
  { step: 'rank', name: 'puts the largest count first, comparing counts as numbers', run(m, T) {
    const rankCounts = need(m, 'rankCounts');
    const got = rankCounts(new Map([['c', 9], ['a', 2], ['b', 10]]));
    const s = JSON.stringify(got);
    if (s === JSON.stringify([['c', 9], ['a', 2], ['b', 10]])) {
      T.fail('the pairs came back in their original order, so nothing was sorted. Call pairs.sort((a, b) => ...). Remember a and b are whole pairs like [\'c\', 9]: the count is a[1] and b[1], so compare b[1] - a[1]. (b - a on two pairs is NaN, which sort treats as "keep the order".)');
    }
    if (s === JSON.stringify([['a', 2], ['b', 10], ['c', 9]])) {
      T.fail('the pairs came back in alphabetical order: sort() with no comparator compares everything as text. Pass a comparator that subtracts the counts: (a, b) => b[1] - a[1].');
    }
    if (s === JSON.stringify([['a', 2], ['c', 9], ['b', 10]])) {
      T.fail('the pairs are smallest count first. A comparator returning a[1] - b[1] puts small first; swap it to b[1] - a[1] for largest first.');
    }
    T.eq(got, [['b', 10], ['c', 9], ['a', 2]], '10 comes before 9 comes before 2');
  } },
  { step: 'rank', name: 'equal counts keep their original order, and the Map is left unchanged', run(m, T) {
    const rankCounts = need(m, 'rankCounts');
    const counts = new Map([['x', 1], ['y', 3], ['z', 1], ['w', 3]]);
    const got = rankCounts(counts);
    T.ok(Array.isArray(got), `rankCounts should return an array of [letter, count] pairs, but returned ${show(got)}`);
    T.eq(got, [['y', 3], ['w', 3], ['x', 1], ['z', 1]], 'y and w both have 3 and keep their Map order (y before w); x and z both have 1 (x before z). sort keeps equal items in order when the comparator returns 0');
    T.eq([...counts], [['x', 1], ['y', 3], ['z', 1], ['w', 3]], 'the Map you were given must not change');
  } },

  // ---------- step 6: table ----------
  { step: 'table', name: 'makeTable stores the values row by row in a Float32Array', run(m, T) {
    const makeTable = need(m, 'makeTable');
    const t = makeTable(2, 3, [1, 2, 3, 4, 5, 6]);
    T.ok(t && typeof t === 'object', `makeTable should return an object { shape, data }, but returned ${show(t)}`);
    T.eq(t.shape, [2, 3], 'shape should be [rows, cols], here [2, 3]');
    T.ok(t.data instanceof Float32Array, `data should be a Float32Array (Float32Array.from(values) makes one), but it is ${Array.isArray(t.data) ? 'an ordinary array' : show(t.data)}`);
    T.eq(Array.from(t.data), [1, 2, 3, 4, 5, 6], 'data holds the values in the order given: row 0 is 1, 2, 3 and row 1 is 4, 5, 6');
    const tiny = makeTable(1, 1, [0.1]);
    T.eq(tiny.data[0], Math.fround(0.1), 'a Float32Array keeps about 7 significant digits, so 0.1 is stored as 0.10000000149011612');
  } },
  { step: 'table', name: 'makeTable throws an Error when the number of values is wrong', run(m, T) {
    const makeTable = need(m, 'makeTable');
    T.throws(() => makeTable(2, 2, [1, 2, 3]), 'makeTable(2, 2, [1, 2, 3]) should throw: a 2 x 2 table needs 4 values. Check values.length !== rows * cols and throw new Error(...)');
    T.throws(() => makeTable(1, 2, [1, 2, 3]), 'makeTable(1, 2, [1, 2, 3]) should throw: too many values is also wrong');
    let t;
    try { t = makeTable(3, 1, [7, 8, 9]); }
    catch (e) { T.fail(`makeTable(3, 1, [7, 8, 9]) has exactly 3 values but threw "${e.message}"`); }
    T.eq(t.shape, [3, 1], 'a correct call still returns the table');
  } },
  { step: 'table', name: 'cell finds (row, col) at position row * cols + col', run(m, T) {
    const cell = need(m, 'cell');
    // Tables written out by hand, so a bug in makeTable cannot show up here.
    const t = { shape: [2, 3], data: Float32Array.of(1, 2, 3, 4, 5, 6) };
    if (cell(t, 1, 0) === 3 && cell(t, 1, 2) === 5) {
      T.fail('cell(t, 1, 0) gave 3 for a table of shape [2, 3]: that is position 1 * 2 + 0, so the row was multiplied by shape[0] (the number of rows). Each row holds shape[1] numbers (the number of columns), so multiply by shape[1].');
    }
    T.eq(cell(t, 0, 0), 1, 'cell(t, 0, 0) is the first number');
    T.eq(cell(t, 0, 2), 3, 'row 0, column 2 is position 0 * 3 + 2 = 2, which holds 3');
    T.eq(cell(t, 1, 0), 4, 'row 1 starts after the 3 numbers of row 0: position 1 * 3 + 0 = 3, which holds 4');
    T.eq(cell(t, 1, 2), 6, 'row 1, column 2 is position 1 * 3 + 2 = 5, the last number');
    const tall = { shape: [3, 2], data: Float32Array.of(10, 20, 30, 40, 50, 60) };
    T.eq(cell(tall, 2, 1), 60, 'in a 3 x 2 table each row has 2 numbers, so (2, 1) is position 2 * 2 + 1 = 5. Multiply the row by the number of columns, shape[1]');
  } },
];
