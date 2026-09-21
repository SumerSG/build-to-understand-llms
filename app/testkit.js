// app/testkit.js — the `T` assertion helper handed to tests (browser worker and Node).
import { rng } from '../lib/util.js';

function isTensor(x) {
  return !!(x && typeof x === 'object' && x.data instanceof Float32Array && Array.isArray(x.shape));
}

// Convert tensors, typed arrays, Maps and Sets to plain nested arrays so they can be compared and printed.
function toPlain(x) {
  if (isTensor(x)) {
    const { shape, data } = x;
    if (shape.length === 0) return data[0];
    let i = 0;
    const build = (d) => { const out = []; for (let j = 0; j < shape[d]; j++) out.push(d === shape.length - 1 ? data[i++] : build(d + 1)); return out; };
    return build(0);
  }
  if (ArrayBuffer.isView(x)) return Array.from(x);
  if (Array.isArray(x)) return x.map(toPlain);
  if (x instanceof Map) return [...x.entries()].sort((a, b) => String(a[0]).localeCompare(String(b[0]))).map(([k, v]) => [toPlain(k), toPlain(v)]);
  if (x instanceof Set) return [...x].map(toPlain);
  return x;
}

function fmtNumber(v, full = false) {
  if (Object.is(v, -0)) return '-0';
  if (!Number.isFinite(v)) return String(v);          // NaN, Infinity, -Infinity (JSON would print null)
  if (full || Number.isInteger(v)) return String(v);
  return String(+v.toFixed(5));
}

// Print a value for a failure message. Numbers are rounded to 5 decimals unless `full`; -0, NaN and
// ±Infinity print faithfully (JSON.stringify would print 0 / null); undefined and functions print as such.
const NUM_MARK = '';   // private-use char: JSON.stringify leaves it unescaped, and data never contains it
function fmt(x, full = false) {
  const p = toPlain(x);
  if (p === undefined) return 'undefined';
  if (typeof p === 'function') return '[function]';
  if (typeof p === 'symbol' || typeof p === 'bigint') return String(p);
  let s;
  try {
    s = JSON.stringify(p, (k, v) => (typeof v === 'number' ? `${NUM_MARK}${fmtNumber(v, full)}${NUM_MARK}` : v));
    if (s !== undefined) s = s.replace(new RegExp(`"${NUM_MARK}([^${NUM_MARK}]*)${NUM_MARK}"`, 'g'), '$1');
  } catch { s = undefined; }
  if (s === undefined) { try { s = String(p); } catch { s = '[unprintable]'; } }
  return s.length > 300 ? s.slice(0, 300) + '…' : s;
}

// -0 equals 0 (JS semantics); NaN equals NaN so a test can assert a NaN result deliberately.
function numEq(a, b) { return a === b || (Number.isNaN(a) && Number.isNaN(b)); }

function deepEq(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return numEq(a, b);
  if (Object.is(a, b)) return true;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a)) return a.length === b.length && a.every((v, i) => deepEq(v, b[i]));
  if (a && b && typeof a === 'object') {
    const pa = Object.getPrototypeOf(a), pb = Object.getPrototypeOf(b);
    if (pa !== pb) return false;
    if (a instanceof Date) return a.getTime() === b.getTime();
    const plain = pa === Object.prototype || pa === null;
    // Non-plain objects (RegExp, class instances with hidden state) must also print the same.
    if (!plain && String(a) !== String(b)) return false;
    const ka = Object.keys(a), kb = Object.keys(b);
    return ka.length === kb.length && ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && deepEq(a[k], b[k]));
  }
  return false;
}

function flatten(x, out = []) {
  if (Array.isArray(x)) { for (const v of x) flatten(v, out); return out; }
  out.push(x);
  return out;
}

function shapeOf(x) {
  if (x && x.shape && x.data) return x.shape.slice();
  const s = [];
  let c = x;
  while (Array.isArray(c) || ArrayBuffer.isView(c)) { s.push(c.length); c = c[0]; }
  return s;
}

const pre = (msg) => (msg ? msg + ' — ' : '');
const UNDEF_HINT = 'got undefined — did the function forget to return a value, or is the export missing?';

// "expected X but got Y" that stays informative when the rounded forms print identically
// (Float32 0.1 vs double 0.1, an array vs an object with the same keys).
function differs(msg, actual, expected) {
  let sa = fmt(actual), sb = fmt(expected);
  if (sa === sb) { sa = fmt(actual, true); sb = fmt(expected, true); }
  if (sa === sb) sa += ' (prints the same, but the type or structure differs)';
  return `${pre(msg)}expected ${sb} but got ${sa}`;
}

export function makeT() {
  const T = {
    ok(cond, msg = 'expected condition to hold') { if (!cond) throw new Error(msg); },
    fail(msg = 'test failed') { throw new Error(msg); },
    eq(actual, expected, msg = '') {
      if (actual === undefined && expected !== undefined) throw new Error(`${pre(msg)}expected ${fmt(expected)} but ${UNDEF_HINT}`);
      if (isTensor(actual) && isTensor(expected)) T.shape(actual, expected.shape, msg);
      if (!deepEq(toPlain(actual), toPlain(expected))) throw new Error(differs(msg, actual, expected));
    },
    close(actual, expected, tol = 1e-4, msg = '') {
      if (typeof tol === 'string') { msg = tol; tol = 1e-4; }
      if (actual === undefined) throw new Error(`${pre(msg)}expected ${fmt(expected)} but ${UNDEF_HINT}`);
      const sa = shapeOf(actual), sb = shapeOf(expected);
      if (sa.length && sb.length && (sa.length !== sb.length || sa.some((d, i) => d !== sb[i]))) {
        throw new Error(`${pre(msg)}shape mismatch: expected [${sb}] but got [${sa}]`);
      }
      const a = flatten(toPlain(actual)), b = flatten(toPlain(expected));
      if (a.length !== b.length) throw new Error(`${pre(msg)}length mismatch: expected ${b.length} values but got ${a.length}`);
      for (let i = 0; i < a.length; i++) {
        const x = a[i], y = b[i];
        const at = a.length > 1 ? `value ${i}` : 'value';
        if (typeof x !== 'number' || typeof y !== 'number') {
          throw new Error(`${pre(msg)}${at} is not a number: expected ${fmt(y)} but got ${fmt(x)} (${x === null ? 'null' : typeof x})`);
        }
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          // NaN only matches NaN and ±Infinity only the same-signed Infinity; never "within tolerance" of a finite value.
          if (!numEq(x, y)) {
            const why = Number.isFinite(y) ? ' (overflow, log(0) or division by zero?)' : '';
            throw new Error(`${pre(msg)}${at} differs: expected ${fmtNumber(y)} but got ${fmtNumber(x)}${why}. Expected ${fmt(expected)}, got ${fmt(actual)}`);
          }
          continue;
        }
        const diff = Math.abs(x - y);
        const lim = tol * Math.max(1, Math.abs(x), Math.abs(y));
        if (!(diff <= lim)) {
          throw new Error(`${pre(msg)}${at} differs: expected ${y} but got ${x} (tol ${tol}). Expected ${fmt(expected)}, got ${fmt(actual)}`);
        }
      }
    },
    shape(t, expected, msg = '') {
      if (t === undefined) throw new Error(`${pre(msg)}expected shape [${expected}] but ${UNDEF_HINT}`);
      const s = shapeOf(t);
      if (s.length !== expected.length || s.some((d, i) => d !== expected[i])) {
        throw new Error(`${pre(msg)}expected shape [${expected}] but got [${s}]`);
      }
    },
    throws(fn, msg = 'expected an error to be thrown') {
      let threw = false;
      try { fn(); } catch { threw = true; }
      if (!threw) throw new Error(msg);
    },
    rng(seed = 1) { return rng(seed); },
    arr: toPlain,
  };
  return T;
}
