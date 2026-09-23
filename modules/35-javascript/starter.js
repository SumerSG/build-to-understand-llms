// JavaScript for this lab. Your job: fill in the functions marked TODO, one build step at a time.
//
// How to use this screen:
//   * Type your code between a function's curly braces { }, replacing its TODO comment and, when you
//     are ready, its placeholder `return` line. Leave everything else as it is.
//   * Press "Check this step" (or Ctrl+Enter). Green lines passed. Red lines say what the test expected
//     and what your code gave back; read them slowly, they are the most useful text on the page.
//   * console.log(x) prints x underneath the test results. Use it to look at a value while you work.
//   * `export` in front of `function` makes the function visible to the tests and to the Goal demo.
//     Keep it: without `export` the checker cannot find your function.
//   * Lines starting with // (like this one) are comments: notes for people, ignored by JavaScript.

// ---------- worked example (done for you: read it, it shows the style used in every module) ----------

/**
 * The fraction of `total` that `count` makes up: share(3, 12) gives 0.25.
 * `count` and `total` are the inputs. `return` hands one value back to whoever called the function.
 */
export function share(count, total) {
  const fraction = count / total;   // const gives a name to a value that will not change
  return fraction;
}

// ---------- step 1: values, strings and your first functions ----------

/** A readable label: label('e', 3) gives the text 'e: 3'. */
export function label(letter, count) {
  // TODO: step 1. Build the text with a template string: backticks ` ` with ${...} slots inside.
  return '';
}

/** How surprising an event with this share is, in "nats": minus the natural log of the share. */
export function surprise(count, total) {
  // TODO: step 1. Use share(count, total) and Math.log.
  return 0;
}

// ---------- step 2: arrays and loops ----------

/** The whole numbers 0, 1, ..., n - 1: range(4) gives [0, 1, 2, 3]. Like Python's range(n). */
export function range(n) {
  const out = [];
  // TODO: step 2. A counting loop that pushes each i onto `out`.
  return out;
}

/** Add up an array of numbers: sum([2, 3, 5]) gives 10, and sum([]) gives 0. */
export function sum(numbers) {
  let total = 0;   // let, not const: this value will change as you add to it
  // TODO: step 2. Loop over the values with for...of and add each one to `total`.
  return total;
}

/** The first k items of an array, as a new array: firstK([5, 6, 7], 2) gives [5, 6]. */
export function firstK(items, k) {
  // TODO: step 2. One line with slice.
  return items;
}

// ---------- step 3: decisions and equality ----------

/** True when two shapes list the same numbers in the same order: sameShape([2, 3], [2, 3]) is true. */
export function sameShape(a, b) {
  // TODO: step 3. This first try compares the two arrays with ===. Press Check to see why that is
  // not enough, then compare the lengths and each position instead.
  return a === b;
}

// ---------- step 4: counting with a Map ----------

/** Count each letter a–z in the text, ignoring upper/lower case. Returns a Map from letter to count. */
export function countLetters(text) {
  const counts = new Map();
  // TODO: step 4. (1) Get every letter with text.toLowerCase().match(...), using [] when there are none.
  //               (2) For each letter, store (its count so far, or 0) + 1 in `counts`.
  return counts;
}

// ---------- step 5: sorting with a function you pass in ----------

/** The [letter, count] pairs of a Map, largest count first; equal counts keep their original order. */
export function rankCounts(counts) {
  const pairs = [...counts];   // turns the Map into an array of pairs: [['a', 2], ['b', 10], ...]
  // TODO: step 5. Sort `pairs` with a comparator: an arrow function (a, b) => ... that compares counts.
  return pairs;
}

// ---------- step 6: objects, typed arrays and errors ----------

/**
 * A table of numbers stored row by row: { shape: [rows, cols], data: Float32Array }.
 * makeTable(2, 3, [1, 2, 3, 4, 5, 6]) holds rows [1, 2, 3] and [4, 5, 6].
 * Throws an Error if `values` does not have exactly rows * cols numbers.
 */
export function makeTable(rows, cols, values) {
  // TODO: step 6
  return { shape: [], data: new Float32Array(0) };
}

/** The number in row `row`, column `col` of a table made by makeTable (both counted from 0). */
export function cell(t, row, col) {
  // TODO: step 6. Unpack the table with const { shape, data } = t; then read the right position of data.
  return 0;
}
