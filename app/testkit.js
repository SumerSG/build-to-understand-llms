// app/testkit.js — the `T` assertion helper handed to tests (browser worker and Node).
import { rng } from '../lib/util.js';

function toPlain(x) {
  if (x && typeof x === 'object' && x.data instanceof Float32Array && Array.isArray(x.shape)) {
    const { shape, data } = x;
    if (shape.length === 0) return data[0];
    let i = 0;
    const build = (d) => { const out = []; for (let j = 0; j < shape[d]; j++) out.push(d === shape.length - 1 ? data[i++] : build(d + 1)); return out; };
    return build(0);
  }
  if (ArrayBuffer.isView(x)) return Array.from(x);
  if (Array.isArray(x)) return x.map(toPlain);
  return x;
}

function fmt(x) {
  const p = toPlain(x);
  let s;
  try { s = JSON.stringify(p, (k, v) => (typeof v === 'number' && !Number.isInteger(v) ? +v.toFixed(5) : v)); } catch { s = String(p); }
  return s.length > 300 ? s.slice(0, 300) + '…' : s;
}

function deepEq(a, b) {
  if (Object.is(a, b)) return true;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((v, i) => deepEq(v, b[i]));
  if (a && b && typeof a === 'object') {
    const ka = Object.keys(a), kb = Object.keys(b);
    return ka.length === kb.length && ka.every((k) => deepEq(a[k], b[k]));
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

export function makeT() {
  const T = {
    ok(cond, msg = 'expected condition to hold') { if (!cond) throw new Error(msg); },
    fail(msg = 'test failed') { throw new Error(msg); },
    eq(actual, expected, msg = '') {
      const a = toPlain(actual), b = toPlain(expected);
      if (!deepEq(a, b)) throw new Error(`${msg ? msg + ' — ' : ''}expected ${fmt(b)} but got ${fmt(a)}`);
    },
    close(actual, expected, tol = 1e-4, msg = '') {
      if (typeof tol === 'string') { msg = tol; tol = 1e-4; }
      const sa = shapeOf(actual), sb = shapeOf(expected);
      if (sa.length && sb.length && (sa.length !== sb.length || sa.some((d, i) => d !== sb[i]))) {
        throw new Error(`${msg ? msg + ' — ' : ''}shape mismatch: expected [${sb}] but got [${sa}]`);
      }
      const a = flatten(toPlain(actual)), b = flatten(toPlain(expected));
      if (a.length !== b.length) throw new Error(`${msg ? msg + ' — ' : ''}length mismatch: expected ${b.length} values but got ${a.length}`);
      for (let i = 0; i < a.length; i++) {
        const diff = Math.abs(a[i] - b[i]);
        const lim = tol * Math.max(1, Math.abs(a[i]), Math.abs(b[i]));
        if (!(diff <= lim)) {
          throw new Error(`${msg ? msg + ' — ' : ''}value ${i} differs: expected ${b[i]} but got ${a[i]} (tol ${tol}). Expected ${fmt(expected)}, got ${fmt(actual)}`);
        }
      }
    },
    shape(t, expected, msg = '') {
      const s = shapeOf(t);
      if (s.length !== expected.length || s.some((d, i) => d !== expected[i])) {
        throw new Error(`${msg ? msg + ' — ' : ''}expected shape [${expected}] but got [${s}]`);
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
