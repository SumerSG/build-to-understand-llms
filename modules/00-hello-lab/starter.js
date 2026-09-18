// Module 00 — Zipf's law. Your job: fill in the four functions marked TODO.
// Press "Check this step" (Ctrl+Enter) to run the tests for the current step.

/**
 * Lowercase the text and collapse runs of whitespace into single spaces.
 * This one is done for you: it shows the house style (pure, documented, no globals).
 */
export function normalize(text) {
  return String(text).toLowerCase().replace(/\s+/g, ' ').trim();
}

/** Split text into lowercase words: maximal runs of letters and apostrophes. */
export function tokenizeWords(text) {
  // TODO: step 1
  return [];
}

/** Count how many times each word occurs. Returns a Map<string, number>. */
export function countFrequencies(words) {
  // TODO: step 2
  return new Map();
}

/** The k most frequent [word, count] pairs, most frequent first, ties alphabetical. */
export function topK(counts, k) {
  // TODO: step 3
  return [];
}

/** Ideal Zipf counts for ranks 1..n given the count of the top word. */
export function zipfPredicted(topCount, n) {
  // TODO: step 4
  return [];
}

/** Mean absolute difference between log(actual[i]) and log(predicted[i]). */
export function zipfLogError(actual, predicted) {
  // TODO: step 4
  return NaN;
}
