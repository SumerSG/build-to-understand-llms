// lib/util.js — small shared helpers used by every module.
// Everything here is deterministic so that tests are reproducible.

/** Deterministic PRNG (mulberry32). Returns a function producing floats in [0, 1). */
export function rng(seed = 42) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample via Box–Muller, driven by a uniform rng function. */
export function randn(next) {
  let u = 0, v = 0;
  while (u === 0) u = next();
  while (v === 0) v = next();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** Integer in [0, n). */
export function randInt(next, n) {
  return Math.floor(next() * n);
}

/** Pick one element of an array. */
export function choice(next, arr) {
  return arr[randInt(next, arr.length)];
}

/** In-place Fisher–Yates shuffle. */
export function shuffle(next, arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = randInt(next, i + 1);
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function assert(cond, msg = 'assertion failed') {
  if (!cond) throw new Error(msg);
}

export function approx(a, b, tol = 1e-5) {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

export function sumArray(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s;
}

export function meanArray(arr) {
  return arr.length ? sumArray(arr) / arr.length : 0;
}

export function argmaxArray(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

/** Numerically stable softmax over a plain array of logits. */
export function softmaxArray(logits, temperature = 1) {
  const m = Math.max(...logits);
  const exps = logits.map((l) => Math.exp((l - m) / temperature));
  const z = sumArray(exps);
  return exps.map((e) => e / z);
}

/** Sample an index from a probability vector using a uniform sample u in [0,1). */
export function sampleIndex(probs, u) {
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (u < acc) return i;
  }
  return probs.length - 1;
}

/** Human-friendly number formatting: 1234567 -> "1.23M". */
export function fmt(n, digits = 2) {
  const abs = Math.abs(n);
  if (abs >= 1e12) return (n / 1e12).toFixed(digits) + 'T';
  if (abs >= 1e9) return (n / 1e9).toFixed(digits) + 'B';
  if (abs >= 1e6) return (n / 1e6).toFixed(digits) + 'M';
  if (abs >= 1e3) return (n / 1e3).toFixed(digits) + 'K';
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}

/** Simple wall-clock timer usable in browser and Node. */
export function now() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

/** FNV-1a 32-bit string hash. Handy for dedup and cache keys. */
export function hash32(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
