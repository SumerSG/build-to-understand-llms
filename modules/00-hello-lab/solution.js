// Module 00 — reference solution.

export function normalize(text) {
  return String(text).toLowerCase().replace(/\s+/g, ' ').trim();
}

export function tokenizeWords(text) {
  return normalize(text).match(/[a-z']+/g) || [];
}

export function countFrequencies(words) {
  const counts = new Map();
  for (const w of words) counts.set(w, (counts.get(w) || 0) + 1);
  return counts;
}

export function topK(counts, k) {
  return [...counts].sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0)).slice(0, k);
}

export function zipfPredicted(topCount, n) {
  return Array.from({ length: n }, (_, i) => topCount / (i + 1));
}

export function zipfLogError(actual, predicted) {
  let s = 0;
  for (let i = 0; i < actual.length; i++) s += Math.abs(Math.log(actual[i]) - Math.log(predicted[i]));
  return s / actual.length;
}
