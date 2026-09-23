// app/testkit.js — the `T` assertion helper handed to tests (browser worker and Node), plus the plain-words
// explanations a beginner sees when a check fails or their code throws.
//
// Pass/fail is decided only by the comparison helpers below (numEq, deepEq, shapeOf, the tolerance rule);
// everything in the "explanations" half of the file runs after a check has already failed and only adds text.
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
const NUM_MARK = '\uE000';   // private-use char: JSON.stringify leaves it unescaped, and data never contains it
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

// "expected X but got Y" that stays informative when the rounded forms print identically
// (Float32 0.1 vs double 0.1, an array vs an object with the same keys).
function differs(msg, actual, expected) {
  let sa = fmt(actual), sb = fmt(expected);
  if (sa === sb) { sa = fmt(actual, true); sb = fmt(expected, true); }
  if (sa === sb) sa += ' (prints the same, but the type or structure differs)';
  return `${pre(msg)}expected ${sb} but got ${sa}`;
}

// ---------------------------------------------------------------------------------------------------
// Explanations. Everything below only builds text for a check that has already failed.
// ---------------------------------------------------------------------------------------------------

const EXPLAINED = Symbol('btu.explained');

/** Join a failure message and its plain-words notes: one note per line, each starting with an arrow. */
function withNotes(message, notes) {
  const list = [...new Set((notes || []).filter(Boolean))];
  return list.length ? `${message}\n${list.map((n) => `→ ${n}`).join('\n')}` : message;
}

/** Never let an explanation break the report of a failure. */
function safe(fn, fallback = []) { try { return fn(); } catch { return fallback; } }

/** A short plain description of a value: "an empty Map", "an array of 3 items", "undefined". */
function describe(v) {
  if (v === undefined) return 'undefined';
  if (v === null) return 'null';
  if (isTensor(v)) return `a tensor of shape [${v.shape}]`;
  if (v instanceof Map) return v.size ? `a Map with ${v.size} ${v.size === 1 ? 'entry' : 'entries'}` : 'an empty Map';
  if (v instanceof Set) return v.size ? `a Set of ${v.size} item${v.size === 1 ? '' : 's'}` : 'an empty Set';
  if (ArrayBuffer.isView(v)) return v.length ? `a ${v.constructor.name} of ${v.length} numbers` : `an empty ${v.constructor.name}`;
  if (Array.isArray(v)) return v.length ? `an array of ${v.length} item${v.length === 1 ? '' : 's'}` : 'an empty array';
  if (typeof v === 'number') return fmtNumber(v);
  if (typeof v === 'string') return v ? JSON.stringify(v.length > 40 ? v.slice(0, 40) + '…' : v) : "an empty string ''";
  if (typeof v === 'function') return 'a function';
  if (typeof v === 'object') {
    const keys = Object.keys(v);
    return keys.length ? `an object with the keys ${keys.slice(0, 6).join(', ')}${keys.length > 6 ? ', …' : ''}` : 'an empty object {}';
  }
  return String(v);
}

const isContainer = (v) => v instanceof Map || v instanceof Set || Array.isArray(v) || ArrayBuffer.isView(v) || (!!v && typeof v === 'object');
const isEmptyContainer = (v) => (Array.isArray(v) || ArrayBuffer.isView(v) ? v.length === 0
  : v instanceof Map || v instanceof Set ? v.size === 0 : v === '');

/** Numbers of a numeric array, typed array or tensor (null for anything else). */
function numbersOf(v) {
  if (isTensor(v)) return Array.from(v.data);
  if (ArrayBuffer.isView(v)) return Array.from(v);
  if (Array.isArray(v)) { const f = flatten(toPlain(v)); return f.length && f.every((x) => typeof x === 'number') ? f : null; }
  return null;
}

const INDEX_KEY = /^\d+$/;
/** Strings that are all array positions ('0', '1', …, including '0'): what a for...in loop over an array yields. */
function looksLikeIndexKeys(strings) {
  return strings.length > 0 && strings.every((s) => INDEX_KEY.test(s)) && strings.includes('0');
}
function stringsIn(v) { return safe(() => flatten(toPlain(v)).filter((s) => typeof s === 'string'), []); }
function keysOf(v) {
  if (v instanceof Map) return [...v.keys()].filter((k) => typeof k === 'string');
  if (v instanceof Set || Array.isArray(v)) return stringsIn(v);
  return [];
}

/** 1-based line of a character offset. */
const lineAt = (src, index) => src.slice(0, index).split('\n').length;

/** Source with comments and string contents blanked (same length, same line breaks), for pattern checks. */
function codeOnly(src) {
  return String(src || '').replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*|`(?:\\[\s\S]|[^`\\])*`|'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"/g,
    (m) => (m[0] === '/' ? m.replace(/[^\n]/g, ' ') : m[0] + m.slice(1, -1).replace(/[^\n]/g, ' ') + m[m.length - 1]));
}

const FOR_IN = /\bfor\s*\(\s*(?:const|let|var)?\s*([A-Za-z_$][\w$]*)\s+in\s+([A-Za-z_$][\w$.]*)/g;
const ARRAY_IDENTITY = [
  // a.shape == b.shape, a.shape !== [2, 3]: arrays compare by identity, so two different arrays are never ==
  /([A-Za-z_$][\w$]*(?:\.[\w$]+)*\.shape)\s*(===?|!==?)\s*((?:[A-Za-z_$][\w$]*(?:\.[\w$]+)*\.shape|\[[^\]\n]*\]))(?![\w$.[(])/g,
  /(?<![\w$.])([A-Za-z_$][\w$]*(?:\.[\w$]+)*)\s*(===?|!==?)\s*(\[[^\]\n]*\])/g,
];

// ---- which part of the learner's file a failing test ran ----------------------------------------------
// Line-level notes ("line 115 compares arrays with ==") are only true for a test that ran that line, so
// they are scoped: a note is shown only when its line sits in a top-level declaration the test used.

const TOP_DECL = /^(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function\s*\*?\s*([A-Za-z_$][\w$]*)|class\s+([A-Za-z_$][\w$]*)|(?:const|let|var)\s+([A-Za-z_$][\w$]*))/gm;
let parsedFor = null, parsed = null;
/** The source with comments and strings blanked, and its top-level declarations [{ name, start, end }]
 *  (a declaration runs to the next one; only declarations starting in column 0 count). Cached per source. */
function parse(source) {
  if (parsedFor !== source) {
    const code = codeOnly(source);
    const heads = [...code.matchAll(TOP_DECL)].filter((m) => m[1] || m[2] || m[3]);
    const chunks = heads.map((m, i) => ({ name: m[1] || m[2] || m[3], isClass: !!m[2], start: m.index, end: i + 1 < heads.length ? heads[i + 1].index : code.length }));
    parsedFor = source;
    parsed = { code, chunks };
  }
  return parsed;
}
const nameRe = (name) => new RegExp(`(?<![\\w$.])${name.replace(/\$/g, '\\$')}(?![\\w$])`);

/** Lines of the learner's file that appear in an error's stack (the code that was running when it threw). */
function learnerLines(err, watch) {
  if (!watch || !watch.file || !err || typeof err.stack !== 'string') return [];
  const esc = watch.file.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return [...err.stack.matchAll(new RegExp(`${esc}:(\\d+):\\d+`, 'g'))].map((m) => +m[1]);
}

/**
 * The top-level declarations of the learner's file that a failing test ran: the exports it called (the
 * recorder saw them), the classes it used, the declarations on the error's stack, and everything those
 * mention by name (a helper that add calls), transitively. null when there is no source to look at.
 * @returns {{code: string, chunks: {name: string, start: number, end: number}[], has: (index: number) => boolean} | null}
 */
function scopeOf(watch, err = null) {
  if (!watch || !watch.source) return null;
  const { code, chunks } = parse(watch.source);
  const reach = new Set(typeof watch.used === 'function' ? watch.used() : recentOf(watch).map(([n]) => n));
  const lineStarts = learnerLines(err, watch).map((ln) => (ln > 1 ? code.split('\n').slice(0, ln - 1).join('\n').length + 1 : 0));
  for (const at of lineStarts) { const c = chunks.find((x) => at >= x.start && at < x.end); if (c) reach.add(c.name); }
  const todo = chunks.filter((c) => reach.has(c.name));
  while (todo.length) {
    const c = todo.pop();
    const text = code.slice(c.start, c.end);
    for (const o of chunks) if (!reach.has(o.name) && nameRe(o.name).test(text)) { reach.add(o.name); todo.push(o); }
  }
  return scopeFrom(code, chunks.filter((c) => reach.has(c.name)));
}
const scopeFrom = (code, chunks) => ({ code, chunks, has: (i) => chunks.some((c) => i >= c.start && i < c.end) });

/** The first match of a global regex whose position is in scope. */
function firstInScope(re, scope) {
  if (!scope) return null;
  re.lastIndex = 0;
  for (const m of scope.code.matchAll(re)) if (scope.has(m.index)) return m;
  return null;
}

/** "line 12 compares arrays with ==" when code the test ran does that. */
function arrayIdentityNote(source, scope) {
  for (const re of ARRAY_IDENTITY) {
    const m = firstInScope(re, scope);
    if (m) {
      const text = String(source).slice(m.index, m.index + m[0].length);
      return `line ${lineAt(scope.code, m.index)} compares arrays with ${m[2]} (\`${text}\`). In JavaScript, == and === on two arrays ask "is this the very same array?", not "do they hold the same numbers?", so [2, 3] == [2, 3] is false. Compare the lengths, then each entry.`;
    }
  }
  return '';
}

/** Identifiers that one top-level function the test ran uses both as `x.data` / `x.shape` (so x is a tensor
 *  object) and as `x[...]` (as if x were its numbers), unless that function also makes x an array. */
function tensorIndexingNote(scope) {
  if (!scope) return '';
  const { code } = scope;
  for (const c of scope.chunks) {
    const chunk = code.slice(c.start, c.end);
    const tensors = new Set([...chunk.matchAll(/(?<![\w$.])([A-Za-z_$][\w$]*)\.(?:data|shape)\b/g)].map((m) => m[1]));
    for (const m of chunk.matchAll(/(?<![\w$.])([A-Za-z_$][\w$]*)\[/g)) {
      if (!tensors.has(m[1]) || new RegExp(`(?<![\\w$.])${m[1].replace(/\$/g, '\\$')}\\s*=\\s*(?:new\\s|\\[|Array)`).test(chunk)) continue;
      const at = c.start + m.index;
      return `line ${lineAt(code, at)} reads \`${m[1]}[…]\`, but ${m[1]} looks like a tensor object (your code also uses ${m[1]}.data or ${m[1]}.shape). Its numbers live in ${m[1]}.data, so write ${m[1]}.data[…]: indexing the object itself gives undefined, and arithmetic on undefined gives NaN.`;
    }
  }
  return '';
}

/** When the code the test ran reads its numbers through .data (and never indexes a tensor object), a NaN
 *  most likely comes from an index that runs past the end of a .data array. */
function pastEndNote(scope) {
  if (!scope || !scope.chunks.some((c) => /\.data\s*\[/.test(scope.code.slice(c.start, c.end)))) return '';
  const mm = scope.chunks.some((c) => /matmul/i.test(c.name)) ? ' In a matmul of [n, k] by [k, m], a row of B holds m numbers, so B[p, j] is b.data[p * m + j].' : '';
  return `your code reads the numbers through .data, so look for an index that runs past the end of a .data array (reading past the end gives undefined, and arithmetic on undefined gives NaN): an index formula error. Check each formula against the shape: a row of an [n, m] tensor holds m numbers, so element [i, j] is at i * m + j.${mm}`;
}

// ^ written as "to the power of": (x - mu)^2, x ^ 2, x ^ 0.5. In JavaScript ^ is bitwise XOR.
const XOR_POWER = /(?:\)|[\w$\]])\s*\^(?![=^])\s*(?:[234]|0?\.5)(?![\w$.])/g;
/** The operand in front of the ^ at `at` (a bracketed group or a name), for quoting the learner's code. */
function operandBefore(code, at) {
  let i = at - 1;
  while (i >= 0 && /\s/.test(code[i])) i--;
  if (code[i] === ')') {
    let d = 0;
    for (; i >= 0; i--) { if (code[i] === ')') d++; else if (code[i] === '(' && --d === 0) break; }
    while (i > 0 && /[\w$.]/.test(code[i - 1])) i--;          // Math.sqrt(…)^2 keeps its function name
  } else {
    while (i > 0 && /[\w$.[\]]/.test(code[i - 1])) i--;
  }
  return Math.max(0, i);
}
function xorPowerNote(source, scope, message = '') {
  const m = /\^/.test(message) ? null : firstInScope(XOR_POWER, scope);   // the test's own message already says it
  if (!m) return '';
  const caret = m.index + m[0].indexOf('^');
  const from = operandBefore(scope.code, caret);
  const text = String(source).slice(from, m.index + m[0].length).replace(/\s+/g, ' ');
  return `line ${lineAt(scope.code, m.index)} uses ^ as "to the power of" (\`${text}\`), but in JavaScript ^ is bitwise XOR: it works on the bits of whole numbers, so it silently gives wrong numbers (it can even make a variance negative, and the square root of a negative number is NaN). To square, write x * x or x ** 2; for a square root, Math.sqrt(x) or x ** 0.5.`;
}

/** A Map the test got back that has ordinary properties but few or no entries, and the line of the
 *  learner's code that wrote one with counts[w] = … (the Python dict habit). */
function mapAsDictNote(watch, scope, message = '') {
  if (!scope || /\.set\(/.test(message)) return '';
  const hit = recentOf(watch).find(([, v]) => v instanceof Map && Object.keys(v).length > v.size);
  if (!hit) return '';
  const props = Object.keys(hit[1]);
  for (const c of scope.chunks) {
    const chunk = scope.code.slice(c.start, c.end);
    const maps = new Set([...chunk.matchAll(/(?<![\w$.])([A-Za-z_$][\w$]*)\s*=\s*new\s+Map\b/g)].map((x) => x[1]));
    for (const m of chunk.matchAll(/(?<![\w$.])([A-Za-z_$][\w$]*)\s*\[([^\]\n]+)\]\s*(?:[-+*/%]?=(?![=>])|\+\+|--)/g)) {
      if (!maps.has(m[1])) continue;
      const name = m[1], at = c.start + m.index;
      const key = String(watch.source).slice(at + m[0].indexOf('[') + 1, at + m[0].indexOf(']')).trim() || 'key';
      const text = String(watch.source).slice(at, at + m[0].length).trim() + (/=$/.test(m[0].trim()) ? ' …' : '');
      const shown = props.slice(0, 5).map((p) => JSON.stringify(p)).join(', ') + (props.length > 5 ? ', …' : '');
      return `${hit[0]} returned a Map with ${hit[1].size === 0 ? 'no entries' : `only ${hit[1].size} ${hit[1].size === 1 ? 'entry' : 'entries'}`} but ${props.length} ordinary ${props.length === 1 ? 'property' : 'properties'} (${shown}). Line ${lineAt(scope.code, at)} writes \`${text}\`: on a Map, ${name}[${key}] = … sets a property on the Map object (the Python dict habit), not an entry of the Map. Use ${name}.set(${key}, …) to store a value and ${name}.get(${key}) to read one, e.g. \`${name}.set(${key}, (${name}.get(${key}) || 0) + 1)\`.`;
    }
  }
  return '';
}

const NAN_NOTE = 'NaN means "not a number". For a beginner it usually comes from reading an element that does not exist (undefined turns into NaN as soon as you do arithmetic with it): an index past the end of an array or of a tensor\'s .data (an index formula error), or indexing a tensor object itself instead of its .data. It can also be a placeholder that is not implemented yet, or come from 0 / 0, the log or square root of a negative number, or an overflow such as Infinity − Infinity.';
const INF_NOTE = 'Infinity usually comes from dividing by zero, log(0) (which is −Infinity) or an overflow such as Math.exp(1000). It can also be a placeholder that is not implemented yet.';

/**
 * Wrap a learner module so the tests' calls into it are recorded (which exported function ran, and what it
 * returned). The wrapper is a Proxy over the module namespace: every value, live binding and key is the real
 * one, and exported plain functions are forwarded through an `apply` trap that only records. Classes pass
 * through untouched (the wrapper only notes that a test used them). Calls the learner's code makes internally
 * are not seen (they use the real bindings).
 * @param {object} mod        the imported learner module
 * @param {object} [o]
 * @param {string} [o.source]  the learner's file (for line-level notes)
 * @param {string} [o.starter] the untouched starter file (to say "this function is not written yet")
 * @param {string} [o.file]    the URL or name the learner's module runs under (to find its frames in a stack)
 */
export function watchLearner(mod, { source = '', starter = '', file = '' } = {}) {
  const calls = new Map();            // name -> last return value, most recent call last
  const used = new Set();             // exports a test called (even if the call threw) or classes it used
  const view = new Map();             // real export -> what the tests see (recording Proxy, or itself for a class)
  const record = (name, value) => { calls.delete(name); calls.set(name, value); };
  const viewOf = (name, fn) => {
    let v = view.get(fn);
    if (!v) {
      const isClass = safe(() => /^class[\s{]/.test(Function.prototype.toString.call(fn)), true);
      v = isClass ? fn : new Proxy(fn, {
        apply(target, thisArg, args) {
          used.add(name);
          const out = Reflect.apply(target, thisArg, args);
          record(name, out);
          if (out instanceof Promise) out.then((r) => record(name, r), () => {});
          return out;
        },
      });
      view.set(fn, v);
    }
    return v;
  };
  const module = new Proxy(mod, {
    get(target, key, receiver) {
      const v = Reflect.get(target, key, receiver);
      if (typeof key !== 'string' || typeof v !== 'function') return v;
      const w = viewOf(key, v);
      if (w === v) used.add(key);     // a class: seen being used, since its methods are not recorded
      return w;
    },
  });
  const starterNorm = String(starter || '').replace(/\s+/g, ' ');
  const cache = new Map();
  /** True when the exported function is the starter's, character for character (spacing aside), TODO included. */
  const stillStarter = (name) => {
    if (!cache.has(name)) {
      cache.set(name, safe(() => {
        const fn = mod[name];
        if (typeof fn !== 'function' || !starterNorm) return false;
        const text = Function.prototype.toString.call(fn);
        return /\bTODO\b/.test(text) && starterNorm.includes(text.replace(/\s+/g, ' '));
      }, false));
    }
    return cache.get(name);
  };
  const names = safe(() => Object.keys(mod), []);
  let starterCode = null;             // built on the first failure that needs it
  /** Does the starter's version of `name` hand back a value (a `return <something>`)? null when unknown.
   *  A function that only changes its inputs (sgdStep, zeroGrad…) returns undefined by design. */
  const expectsReturn = (name) => safe(() => {
    if (starterCode === null) starterCode = codeOnly(starter);
    if (!starterCode) return null;
    const esc = name.replace(/\$/g, '\\$');
    const head = new RegExp(`(?:function\\s+${esc}\\s*\\(|(?<![\\w$.])${esc}\\s*=\\s*(?:async\\s*)?(?:function\\b|\\([^)]*\\)\\s*=>|[\\w$]+\\s*=>))`).exec(starterCode);
    if (!head) return null;
    const after = head.index + head[0].length;
    if (/=>\s*$/.test(head[0])) {
      if (!/^\s*\{/.test(starterCode.slice(after))) return true;   // `(a) => expr` returns expr
    }
    let i = after;
    if (/\($/.test(head[0])) {                                  // skip the parameter list, which may hold { … } defaults
      let d = 1;
      for (; i < starterCode.length && d > 0; i++) { if (starterCode[i] === '(') d++; else if (starterCode[i] === ')') d--; }
    }
    i = starterCode.indexOf('{', i);
    if (i < 0) return null;
    let depth = 0, j = i;
    for (; j < starterCode.length; j++) { if (starterCode[j] === '{') depth++; else if (starterCode[j] === '}' && --depth === 0) break; }
    return /\breturn\s*[^\s;}]/.test(starterCode.slice(i, j));
  }, null);
  return {
    module, source: String(source || ''), file: String(file || ''), exports: new Set(names), expectsReturn,
    reset() { calls.clear(); used.clear(); },
    /** [name, value] of the recorded calls, most recent first. */
    recent() { return [...calls].reverse(); },
    /** Names of the exports this test called (including calls that threw) and the classes it used. */
    used() { return new Set([...used, ...calls.keys()]); },
    stillStarter,
    /** Unwritten starter functions behind a call to `name`: itself, and exports its source calls by name. */
    unwrittenBehind(name) {
      const out = stillStarter(name) ? [name] : [];
      const fn = mod[name];
      const text = typeof fn === 'function' ? safe(() => codeOnly(Function.prototype.toString.call(fn)), '') : '';
      for (const other of names) {
        if (other !== name && stillStarter(other) && new RegExp(`(?<![\\w$.])${other.replace(/\$/g, '\\$')}\\s*\\(`).test(text)) out.push(other);
      }
      return out;
    },
  };
}

const recentOf = (watch) => (watch && typeof watch.recent === 'function' ? watch.recent() : []);

/** The most recent learner call that returned undefined although its starter version returns a value
 *  (unknown counts as yes only when there is no starter to ask). */
function missingReturn(watch) {
  return recentOf(watch).find(([name, v]) => v === undefined && (typeof watch.expectsReturn !== 'function' || watch.expectsReturn(name) !== false));
}
const missingReturnNote = (name) => `${name} returned undefined: did the function forget to \`return\` a value?`;

/** Notes that depend only on which learner functions this test called: a missing return, a starter function
 *  left untouched. */
function contextNotes(watch, { returns = true } = {}) {
  return safe(() => {
    const notes = [], seen = new Set();
    if (!watch || typeof watch.unwrittenBehind !== 'function') return notes;
    const undef = returns ? missingReturn(watch) : null;
    if (undef && !watch.stillStarter(undef[0])) notes.push(missingReturnNote(undef[0]));
    for (const [name] of recentOf(watch)) {
      for (const n of watch.unwrittenBehind(name)) {
        if (seen.has(n)) continue;
        seen.add(n);
        notes.push(n === name
          ? `${n} is still the starter code (its TODO is untouched): looks like this function is not written yet.`
          : `${name} calls ${n}, which is still the starter code (its TODO is untouched): looks like ${n} is not written yet.`);
      }
    }
    return notes;
  });
}

/** Source-level notes that are worth showing on any failure. */
/** Notes read from the learner's source, limited to the code this test ran (see scopeOf). `first` are notes
 *  that explain the failure better than anything else (shown before the others), `last` the rest. */
function sourceNotes(watch, err = null, message = '') {
  const none = { first: [], last: [] };
  if (!watch || !watch.source) return none;
  return safe(() => {
    const scope = scopeOf(watch, err);
    return { first: [xorPowerNote(watch.source, scope, message)], last: [arrayIdentityNote(watch.source, scope), mapAsDictNote(watch, scope, message)] };
  }, none);
}

/** A note about a for...in loop, when positions ('0', '1', …) show up where words were expected. */
function forInNote(watch, keys) {
  if (!looksLikeIndexKeys(keys)) return '';
  const scope = scopeOf(watch);
  const m = firstInScope(FOR_IN, scope);
  const where = m ? ` Line ${lineAt(scope.code, m.index)} has \`for (… ${m[1]} in ${m[2]})\`.` : '';
  return `the result holds the positions '0', '1', '2', … where words (the values) were expected. A \`for (const w in words)\` loop walks over the positions of an array; \`for (const w of words)\` walks over the values.${where}`;
}

const UNDEF_NOTE = 'undefined means "no value". If this came straight from your function, it probably forgot to `return` its result; if it was looked up in your result (a Map key, an array position, a property), that entry is missing.';

/** Notes explaining a wrong value (actual) given what the test wanted (expected). */
function valueNotes(actual, expected, watch) {
  return safe(() => {
    const notes = [];
    const recent = recentOf(watch);
    const starterNotes = contextNotes(watch, { returns: actual !== undefined });
    if (actual === undefined) {
      // the value most likely came straight from a call that returned undefined, or from a lookup in a result
      const undef = missingReturn(watch);
      const box = recent.find(([, v]) => isContainer(v));
      if (undef) notes.push(missingReturnNote(undef[0]));
      else if (box && isEmptyContainer(box[1])) notes.push(`${box[0]} returned ${describe(box[1])}, so looking anything up in it gives undefined: nothing was ever added to it.`);
      else if (box) {
        notes.push(`${box[0]} returned ${describe(box[1])}; undefined means the entry looked up here is not in it (a missing Map key, an array position past the end, or a misspelt property name).`);
        notes.push(forInNote(watch, keysOf(box[1])));
      } else if (!starterNotes.length) notes.push(UNDEF_NOTE);
    }
    notes.push(...starterNotes);
    if (actual !== undefined && !starterNotes.length) {
      const nums = numbersOf(actual), want = numbersOf(expected);
      if (isEmptyContainer(actual) && expected !== undefined && !isEmptyContainer(expected)) {
        notes.push(`you got ${describe(actual)}, the starter's usual placeholder: looks like this function is not written yet, or it never adds anything to its result.`);
      } else if (nums && nums.length > 1 && nums.every((x) => x === 0) && want && want.some((x) => x !== 0)) {
        notes.push("every value is 0, the starter's usual placeholder: looks like this function is not written yet, or its loop never writes into the output.");
      }
    }
    const got = stringsIn(actual);
    if (looksLikeIndexKeys(got) && stringsIn(expected).some((s) => !INDEX_KEY.test(s))) notes.push(forInNote(watch, got));
    return notes;
  });
}

/** Notes for a number that is NaN or ±Infinity where a finite number was expected. The general causes are
 *  skipped when an unwritten starter function already explains it. */
function nonFiniteNotes(x, watch, message = '') {
  return safe(() => {
    if (Number.isFinite(x)) return [];
    const scope = scopeOf(watch);
    // an unwritten starter function, or a ^ used as a power, already explains it
    const general = !contextNotes(watch).length && !(scope && xorPowerNote(watch.source, scope));
    if (!Number.isNaN(x)) return general ? [INF_NOTE] : [];
    const indexing = tensorIndexingNote(scope);
    if (indexing) return [indexing];
    if (/past the end|does not exist/.test(message)) return [];      // the test's own message already explains it
    const pastEnd = general ? pastEndNote(scope) : '';
    return [pastEnd || (general ? NAN_NOTE : '')];
  });
}

/** Position of flat index i in a nested shape, as "[1][2]" (or "[i]" for a flat list). */
function position(i, shape) {
  if (!shape || shape.length < 2 || shape.reduce((a, b) => a * b, 1) <= i) return `[${i}]`;
  const idx = [];
  for (let d = shape.length - 1; d >= 0; d--) { idx.unshift(i % shape[d]); i = Math.floor(i / shape[d]); }
  return idx.map((k) => `[${k}]`).join('');
}

function fail(message, notes) {
  const err = new Error(withNotes(message, notes));
  err[EXPLAINED] = true;
  throw err;
}

/**
 * @param {object} [watch]  from watchLearner(): lets failure messages name the learner function behind a
 *                          value. Optional; without it the messages are less specific but still correct.
 */
export function makeT(watch = null) {
  // Every failure goes through here; its notes are built only now, after the check has failed.
  const failWith = (message, notes) => {
    const src = sourceNotes(watch, null, message);
    fail(message, [...src.first, ...notes, ...src.last]);
  };
  // a test's own message that reports a NaN gets the same explanation a NaN value would
  const nanIn = (msg) => (/\bNaN\b/.test(msg) ? nonFiniteNotes(NaN, watch, msg) : []);
  const T = {
    ok(cond, msg = 'expected condition to hold') { if (!cond) failWith(msg, [...contextNotes(watch), ...nanIn(msg)]); },
    fail(msg = 'test failed') { failWith(msg, [...contextNotes(watch), ...nanIn(msg)]); },
    eq(actual, expected, msg = '') {
      if (actual === undefined && expected !== undefined) failWith(`${pre(msg)}expected ${fmt(expected)} but got undefined`, valueNotes(actual, expected, watch));
      if (isTensor(actual) && isTensor(expected)) T.shape(actual, expected.shape, msg);
      if (!deepEq(toPlain(actual), toPlain(expected))) {
        const notes = valueNotes(actual, expected, watch);
        safe(() => {
          const got = typeof actual === 'number' ? [actual] : numbersOf(actual) || [];
          const want = typeof expected === 'number' ? [expected] : numbersOf(expected) || [];
          const bad = got.find((x) => !Number.isFinite(x));
          if (bad !== undefined && want.length && want.every(Number.isFinite)) notes.push(...nonFiniteNotes(bad, watch));
        });
        failWith(differs(msg, actual, expected), notes);
      }
    },
    close(actual, expected, tol = 1e-4, msg = '') {
      if (typeof tol === 'string') { msg = tol; tol = 1e-4; }
      if (actual === undefined) failWith(`${pre(msg)}expected ${fmt(expected)} but got undefined`, valueNotes(actual, expected, watch));
      const sa = shapeOf(actual), sb = shapeOf(expected);
      if (sa.length && sb.length && (sa.length !== sb.length || sa.some((d, i) => d !== sb[i]))) {
        failWith(`${pre(msg)}shape mismatch: expected [${sb}] but got [${sa}]`,
          ['a shape lists the size along each dimension; for a plain list it is just [its length].', ...valueNotes(actual, expected, watch)]);
      }
      const a = flatten(toPlain(actual)), b = flatten(toPlain(expected));
      if (a.length !== b.length) failWith(`${pre(msg)}length mismatch: expected ${b.length} values but got ${a.length}`, valueNotes(actual, expected, watch));
      // message parts are built only on failure: this loop runs once per element of large tensors
      const at = (i) => (a.length > 1 ? `element ${position(i, sa)}` : 'value');
      const all = () => (a.length > 1 ? ` Expected ${fmt(expected)}, got ${fmt(actual)}` : '');
      for (let i = 0; i < a.length; i++) {
        const x = a[i], y = b[i];
        if (typeof x !== 'number' || typeof y !== 'number') {
          const why = x === undefined ? 'that position holds no value (undefined): the result is missing an entry there.' : '';
          failWith(`${pre(msg)}${at(i)} is not a number: expected ${fmt(y)} but got ${fmt(x)} (${x === null ? 'null' : typeof x})`, [why, ...valueNotes(actual, expected, watch)]);
        }
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          // NaN only matches NaN and ±Infinity only the same-signed Infinity; never "within tolerance" of a finite value.
          if (!numEq(x, y)) {
            failWith(`${pre(msg)}${at(i)} differs: expected ${fmtNumber(y)}, got ${fmtNumber(x)}.${all()}`,
              [...valueNotes(actual, expected, watch), ...(Number.isFinite(y) ? nonFiniteNotes(x, watch) : [])]);
          }
          continue;
        }
        const diff = Math.abs(x - y);
        const lim = tol * Math.max(1, Math.abs(x), Math.abs(y));
        if (!(diff <= lim)) {
          failWith(`${pre(msg)}${at(i)} differs: expected ${y}, got ${x} (allowed error ${fmtNumber(lim, lim < 1e-5)}).${all()}`, valueNotes(actual, expected, watch));
        }
      }
    },
    shape(t, expected, msg = '') {
      if (t === undefined) failWith(`${pre(msg)}expected shape [${expected}] but got undefined`, valueNotes(t, expected, watch));
      const s = shapeOf(t);
      if (s.length !== expected.length || s.some((d, i) => d !== expected[i])) {
        failWith(`${pre(msg)}expected shape [${expected}] but got [${s}]`, contextNotes(watch));
      }
    },
    throws(fn, msg = 'expected an error to be thrown') {
      let threw = false;
      try { fn(); } catch { threw = true; }
      if (!threw) failWith(msg, contextNotes(watch));
    },
    rng(seed = 1) { return rng(seed); },
    arr: toPlain,
  };
  return T;
}

// ---------------------------------------------------------------------------------------------------
// Raw JavaScript errors, in plain words. The original message is always kept; these only add notes.
// ---------------------------------------------------------------------------------------------------

const RESERVED = new Set(('break case catch class const continue debugger default delete do else enum export extends false finally for '
  + 'function if import in instanceof new null return super switch this throw true try typeof var void while with yield let static '
  + 'implements interface package private protected public await').split(' '));
const RENAME = { var: 'variance', new: 'next', class: 'label', function: 'fn', return: 'result', delete: 'removed', in: 'input',
  default: 'fallback', let: 'value', this: 'self', case: 'item', switch: 'toggle', do: 'step', for: 'each', if: 'cond', static: 'fixed',
  public: 'shared', private: 'secret', package: 'pkg', interface: 'api', import: 'incoming', export: 'outgoing', const: 'constant',
  super: 'parent', try: 'attempt', catch: 'caught', throw: 'thrown', while: 'during', with: 'using', yield: 'output', await: 'result',
  typeof: 'kind', void: 'empty', enum: 'choices', extends: 'base', finally: 'last', continue: 'more', break: 'stop', debugger: 'dbg',
  instanceof: 'isKind', true: 'yes', false: 'no', null: 'none', else: 'otherwise', implements: 'impl', protected: 'guarded' };

/** A reserved word the source uses as a name (a variable, function or parameter), with its line. */
function reservedAsName(source, fromLine = 0, toLine = 0) {
  const code = codeOnly(source);
  const lines = code.split('\n');
  const lo = fromLine ? Math.max(1, fromLine) : 1, hi = toLine ? Math.min(lines.length, toLine) : lines.length;
  const words = [...RESERVED].join('|');
  const decl = new RegExp(`\\b(?:const|let|var)\\s+(${words})(?![\\w$])(?!\\s*[(.])`);
  const destr = new RegExp(`\\b(?:const|let|var)\\s*[[{][^=\\]}]*?(?<![\\w$.])(${words})(?![\\w$])(?!\\s*:)[^=]*?[\\]}]\\s*=`);
  const fnName = new RegExp(`\\bfunction\\s+(${words})\\s*\\(`);
  const assign = new RegExp(`(?:^|[;{}(,]|\\s)(${words})\\s*(?:[-+*/%]?=(?![=>])|(?:\\+\\+|--)(?![\\w$(]))`);
  const params = /\bfunction\s*[\w$]*\s*\(([^)]*)\)|\(([^()]*)\)\s*=>/g;
  for (let i = lo; i <= hi; i++) {
    const text = lines[i - 1] || '';
    for (const re of [decl, destr, fnName, assign]) {
      const m = re.exec(text);
      if (m) return { word: m[1], line: i };
    }
    for (const m of text.matchAll(params)) {
      const bad = (m[1] ?? m[2] ?? '').split(',').map((p) => p.trim().replace(/\s*=.*$/, '')).find((p) => RESERVED.has(p));
      if (bad) return { word: bad, line: i };
    }
  }
  return null;
}

function reservedNote(hit) {
  const alt = RENAME[hit.word] || `my${hit.word[0].toUpperCase()}${hit.word.slice(1)}`;
  return `line ${hit.line} uses \`${hit.word}\` as a name, but ${hit.word} is a reserved word: JavaScript keeps it for its own grammar, so it cannot name a variable, function or parameter. Pick another name, e.g. \`${alt}\`.`;
}

const PYTHONISMS = {
  range: 'JavaScript has no range(). To count from 0 to n − 1, write `for (let i = 0; i < n; i++) { … }`.',
  len: 'JavaScript has no len(). Use `.length` on arrays and strings (`words.length`) and `.size` on a Map or Set.',
  print: 'JavaScript has no print(). Use `console.log(…)`; what it prints appears under the test results.',
  None: 'JavaScript writes Python\'s None as `null` (or `undefined`).',
  True: 'JavaScript writes true and false in lowercase.',
  False: 'JavaScript writes true and false in lowercase.',
  self: 'inside a class method JavaScript uses `this`, not self.',
  math: 'the maths functions live in `Math`, with a capital M: Math.exp, Math.log, Math.max.',
  elif: 'JavaScript writes elif as `else if`.',
  np: 'there is no NumPy here: plain loops over arrays and typed arrays do the work.',
};

/** Names the source creates (for "did you mean …?" on a wrongly capitalised name). */
function declaredNames(code) {
  const names = new Set();
  const add = (n) => { n = n.trim().replace(/\s*=.*$/, '').replace(/^\.\.\./, ''); if (/^[A-Za-z_$][\w$]*$/.test(n)) names.add(n); };
  for (const m of code.matchAll(/\b(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)/g)) names.add(m[1]);
  for (const m of code.matchAll(/\b(?:const|let|var)\s*[[{]([^=]*?)[\]}]\s*=/g)) m[1].split(/[\s,:]+/).forEach(add);
  for (const m of code.matchAll(/\bfunction\s*[\w$]*\s*\(([^)]*)\)|\(([^()]*)\)\s*=>|([A-Za-z_$][\w$]*)\s*=>/g)) (m[1] ?? m[2] ?? m[3] ?? '').split(',').forEach(add);
  return names;
}

/** Line and column of the first stack frame inside the learner's file (1-based), or null. */
function learnerFrame(err, watch) {
  if (!watch || !watch.file || !err || typeof err.stack !== 'string') return null;
  const esc = watch.file.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const m = new RegExp(`${esc}:(\\d+):(\\d+)`).exec(err.stack);
  return m ? { line: +m[1], col: +m[2] } : null;
}

const CHAIN = '[A-Za-z_$][\\w$]*(?:\\s*\\??\\.\\s*[A-Za-z_$][\\w$]*|\\[[^\\[\\]\\n]*\\]|\\([^()\\n]*\\))*?';

/** The expression in front of `.prop` nearest to the column, e.g. "pairs[i]" for pairs[i].count. */
function exprBefore(lineText, prop, col) {
  const code = codeOnly(lineText);
  const re = new RegExp(`(${CHAIN})\\s*\\??\\.\\s*${prop.replace(/\$/g, '\\$')}(?![\\w$])`, 'g');
  let best = null;
  for (const m of code.matchAll(re)) {
    const d = Math.abs(m.index + m[0].length - col);
    if (!best || d < best.d) best = { expr: lineText.slice(m.index, m.index + m[1].length).trim(), d };
  }
  return best ? best.expr : '';
}

/** Every indexing chain on a line, e.g. ["a[i][j]", "row[k]"] (for "reading '0' of undefined"). */
function indexingOn(lineText) {
  const code = codeOnly(lineText);
  return [...new Set([...code.matchAll(/(?<![\w$.])[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*(?:\[[^\[\]\n]*\])+/g)].map((m) => lineText.slice(m.index, m.index + m[0].length)))];
}

/** Where a function the code calls is declared inside another function (so no other function can see it):
 *  { line, outer, head } (head is how the outer declaration starts, e.g. "export function add"), or null. */
function nestedDeclaration(source, name) {
  const { code, chunks } = parse(source);
  const esc = name.replace(/\$/g, '\\$');
  const re = new RegExp(`\\bfunction\\s+${esc}\\s*\\(|(?<![\\w$.])(?:const|let|var)\\s+${esc}\\s*=\\s*(?:async\\s*)?(?:function\\b|\\([^)]*\\)\\s*=>|[A-Za-z_$][\\w$]*\\s*=>)`, 'g');
  for (const m of code.matchAll(re)) {
    let depth = 0;
    for (let i = 0; i < m.index; i++) { if (code[i] === '{') depth++; else if (code[i] === '}') depth--; }
    if (depth <= 0) continue;
    const outer = chunks.filter((c) => c.start <= m.index && c.name !== name).pop();
    const head = outer ? (/^[^(={\n]*/.exec(code.slice(outer.start)) || [''])[0].trim() : '';
    return { line: lineAt(code, m.index), outer: outer ? outer.name : '', head };
  }
  return null;
}

/** A Python f-string (f"…" or f'…') near the line of a syntax error: its line, and the same text written
 *  as a JavaScript template literal. */
function fStringHit(source, lineNo) {
  const code = codeOnly(source);
  for (const m of code.matchAll(/(?<![\w$.])f(["'])/g)) {
    const line = lineAt(code, m.index);
    if (lineNo && Math.abs(line - lineNo) > 1) continue;
    const close = code.indexOf(m[1], m.index + 2);
    const inner = close > 0 ? source.slice(m.index + 2, close) : '';
    const fixed = inner && !inner.includes('\n') ? '`' + inner.replace(/\{([^{}]*)\}/g, '${$1}') + '`' : '';
    return { line, text: close > 0 ? source.slice(m.index, close + 1) : `f${m[1]}…${m[1]}`, fixed };
  }
  return null;
}

/** The two things a matmul( … ) call on a line multiplies, as written, and whether it is ops.matmul. */
function matmulArgs(lineText) {
  const code = codeOnly(lineText);
  const call = /(?:(\bops)\s*\.\s*)?\bmatmul\s*\(/.exec(code);
  if (call) {
    const args = [];
    let depth = 0, from = call.index + call[0].length, i = from;
    for (; i < code.length; i++) {
      const ch = code[i];
      if ('([{'.includes(ch)) depth++;
      else if (')]}'.includes(ch)) { if (depth === 0) break; depth--; }
      else if (ch === ',' && depth === 0) { args.push(lineText.slice(from, i).trim()); from = i + 1; }
    }
    args.push(lineText.slice(from, i).trim());
    if (args.length === 2 && args.every((x) => x && x.length <= 30)) return { left: args[0], right: args[1], ops: !!call[1] };
  }
  return null;
}

const shapeList = (s) => s.split(',').map((x) => x.trim()).filter(Boolean).map(Number);
const swapLast = (s) => (s.length < 2 ? s : [...s.slice(0, -2), s[s.length - 1], s[s.length - 2]]);
const showShape = (s) => `[${s.join(', ')}]`;

/** Plain shape advice for "matmul: inner dims differ (k vs k2) for [A] x [B]" from lib/ops.js. */
function matmulInnerNotes(A, B, lineText) {
  const notes = [`matmul multiplies an [n, k] tensor by a [k, m] one and gives [n, m]: the last size of the left one must equal the first size of the right one (the two "inner" sizes). Here the left is ${showShape(A)} (last size ${A[A.length - 1]}) and the right is ${showShape(B)} (first size ${B[B.length - 2]}), so they do not fit.`];
  const args = matmulArgs(lineText || '');
  const tr = (x) => (args && args.ops ? `ops.transpose(${x})` : `transpose(${x})`);
  const mm = (x, y) => (args ? `${args.ops ? 'ops.' : ''}matmul(${x}, ${y})` : '');
  const L = args ? args.left : 'the left one', R = args ? args.right : 'the right one';
  const fits = [
    { x: A, y: swapLast(B), say: args ? mm(L, tr(R)) : 'the left one times the transpose of the right one' },
    { x: swapLast(A), y: B, say: args ? mm(tr(L), R) : 'the transpose of the left one times the right one' },
    { x: swapLast(A), y: swapLast(B), say: args ? mm(tr(L), tr(R)) : 'the transpose of both' },
  ].filter((c) => c.x[c.x.length - 1] === c.y[c.y.length - 2])
    .map((c) => `${c.say} (${showShape(c.x)} x ${showShape(c.y)} gives ${showShape([...c.x.slice(0, -1), c.y[c.y.length - 1]])})`);
  if (fits.length === 1) notes.push(`Transposing (swapping rows and columns) would fit, but only one way round: ${fits[0]}.`);
  else if (fits.length > 1) notes.push(`Transposing (swapping rows and columns) would fit: ${fits.join(', or ')}. Pick the one whose result has the shape you need.`);
  else notes.push('No transpose of either side fits either: check that these are the two tensors you meant to multiply.');
  if (/\baccumulate\s*\(|\bgrad\b|(?<![\w$])d[A-Z][\w$]*/.test(lineText || '')) {
    notes.push("In a backward pass every gradient has the shape of the value it belongs to: for C = A · B with A [n, k], B [k, m] and the incoming gradient dC [n, m], dA must be A's shape [n, k], and only dC · transpose(B) ([n, m] x [m, k]) gives that; dB must be B's shape [k, m], and only transpose(A) · dC ([k, n] x [n, m]) gives that.");
  }
  return notes;
}

/** Plain advice for "matmul: need at least 2D tensors" from lib/ops.js, with the learner's line that built a
 *  tensor straight from a list when the code the test ran has one. */
function matmul2dNotes(source, scope) {
  const notes = ['matmul works on tables of numbers (2D tensors, shape [rows, columns]), and one of its two inputs is a flat list (1D, shape [N]). A flat list of N numbers must become a column of shape [N, 1] first: wrap each number in its own array, e.g. `Tensor.from(xs.map((x) => [x]))` instead of `Tensor.from(xs)`.'];
  // the learner's own functions, not the class that implements Tensor.from
  const own = scope ? scopeFrom(scope.code, scope.chunks.filter((c) => !c.isClass)) : null;
  const m = firstInScope(/(?<![\w$.])(?:Tensor\s*\.\s*from|ops\s*\.\s*fromArray)\s*\(\s*([A-Za-z_$][\w$]*)\s*\)/g, own);
  if (m) notes.push(`line ${lineAt(scope.code, m.index)} builds a tensor straight from ${m[1]} (\`${String(source).slice(m.index, m.index + m[0].length)}\`): if ${m[1]} is a flat list of numbers, that tensor is 1D. Write \`${m[1]}.map((x) => [x])\` to make it a column.`);
  return notes;
}

/**
 * Plain-words notes for an error thrown by learner code (or by a test while reading the learner's result).
 * Returns sentences; the caller appends them to the original message, never replacing it.
 * @param {Error|string} err
 * @param {object} [watch]  from watchLearner(), or just { source } when the file did not load
 * @param {{line?: number}} [where]  the known line of a syntax error
 */
export function explainError(err, watch = null, where = {}) {
  return safe(() => {
    const message = String(err && err.message !== undefined ? err.message : err);
    const isSyntax = !!(err && err.name === 'SyntaxError');
    const source = watch && watch.source ? watch.source : '';
    const code = codeOnly(source);
    const lines = source.split('\n');
    const frame = learnerFrame(err, watch);
    const lineNo = where.line || (frame && frame.line) || 0;
    const lineText = lineNo ? lines[lineNo - 1] || '' : '';
    const recent = recentOf(watch);
    const notes = [];
    let m, nested;

    if ((m = /matmul: inner dims differ \((\d+) vs (\d+)\) for \[([\d,\s]*)\] x \[([\d,\s]*)\]/.exec(message))) {
      notes.push(...matmulInnerNotes(shapeList(m[3]), shapeList(m[4]), lineText));
    } else if (/matmul: need at least 2D tensors/.test(message)) {
      notes.push(...matmul2dNotes(source, source ? scopeOf(watch, err) : null));
    } else if ((m = /^([A-Za-z_$][\w$]*) is not defined$/.exec(message) || /^Can't find variable: ([A-Za-z_$][\w$]*)$/.exec(message))) {
      const name = m[1];
      const loop = new RegExp(`\\bfor\\s*\\(\\s*${name.replace(/\$/g, '\\$')}\\s+(in|of)\\b`).exec(code);
      const similar = [...declaredNames(code)].find((n) => n !== name && n.toLowerCase() === name.toLowerCase());
      if (loop) {
        const over = (/\bfor\s*\(\s*[\w$]+\s+(?:in|of)\s+([A-Za-z_$][\w$.]*)/.exec(code.slice(loop.index)) || [])[1] || '…';
        notes.push(`the loop on line ${lineAt(code, loop.index)} has to create its variable: write \`for (const ${name} of ${over})\` to walk over the values${loop[1] === 'in' ? `. (\`in\` would give the positions '0', '1', … instead of the values.)` : '.'}`);
      }
      else if ((nested = source ? nestedDeclaration(source, name) : null)) {
        notes.push(`${name} exists, but it is defined inside ${nested.outer || 'another function'} (line ${nested.line}), between that function's { and }, so only code inside ${nested.outer || 'that function'} can use it. Move the whole ${name} function out to the top level of the file, on its own${nested.head ? ` (for example just above \`${nested.head}\`)` : ''}, not inside another function.`);
      }
      else if (PYTHONISMS[name]) notes.push(PYTHONISMS[name]);
      else if (similar) notes.push(`you used the name ${name}, but your code creates ${similar}: names are case-sensitive, so ${name} and ${similar} are different names.`);
      else notes.push(`you used the name ${name} before creating it; declare it first, e.g. \`const ${name} = …\` (or \`let ${name} = …\` if it changes later). Also check the spelling: names are case-sensitive.`);
    } else if ((m = /Cannot access '([^']+)' before initialization/.exec(message) || /can't access lexical declaration '([^']+)' before initialization/.exec(message))) {
      notes.push(`${m[1]} is created (with const or let) further down than the line that first uses it. Move the line that creates ${m[1]} above its first use.`);
    } else if ((m = /Identifier '([^']+)' has already been declared/.exec(message) || /redeclaration of (?:let|const|var) (\S+)/.exec(message))) {
      notes.push(`you created ${m[1]} twice (two const or let lines with the same name in the same block). Rename one, or drop the second const or let to change the existing ${m[1]}.`);
    } else if (/Assignment to constant variable|invalid assignment to const|Attempted to assign to readonly property/.test(message)) {
      const consts = new Set([...code.matchAll(/\bconst\s+([A-Za-z_$][\w$]*)/g)].map((x) => x[1]));
      const changed = lineText ? [...codeOnly(lineText).matchAll(/(?<![\w$.])([A-Za-z_$][\w$]*)\s*(?:[-+*/%]?=(?![=>])|\+\+|--)|(?:\+\+|--)\s*([A-Za-z_$][\w$]*)/g)]
        .map((x) => x.slice(1).find(Boolean)).map((n) => [n]).find(([n]) => consts.has(n)) : null;
      notes.push(changed
        ? `line ${lineNo} changes ${changed[0]}, which was declared with const. A const cannot change after it is created; declare it with let instead (\`let ${changed[0]} = …\`) when it has to change, e.g. a running total or a counter.`
        : 'your code changes a variable declared with const. A const cannot change after it is created; declare it with let instead when it has to change, e.g. a running total or a counter.');
    } else if ((m = /Cannot (?:read|set) properties of (undefined|null) \((?:reading|setting) '([^']*)'\)/.exec(message))
      || (m = /can't access property "([^"]*)", (.+) is (undefined|null)/.exec(message))
      || (m = /(undefined|null) is not an object \(evaluating '(.+)'\)/.exec(message))) {
      let kind, prop, expr = '';
      if (m[0].startsWith('Cannot')) { kind = m[1]; prop = m[2]; }
      else if (m[0].startsWith("can't")) { prop = m[1]; expr = m[2]; kind = m[3]; }
      else { kind = m[1]; const p = /(?:\.([\w$]+)|\[([^\]]*)\])$/.exec(m[2]); prop = p ? (p[1] ?? p[2]) : ''; expr = p ? m[2].slice(0, p.index) : m[2]; }
      const numeric = /^\d+$/.test(prop);
      const access = numeric ? `[${prop}]` : `.${prop}`;
      if (!expr && frame && lineText && !numeric) expr = exprBefore(lineText, prop, frame.col);
      const last = recent[0];
      if (!frame && last && (last[1] === undefined || last[1] === null) && !(last[1] === undefined && watch.expectsReturn && watch.expectsReturn(last[0]) === false)) {
        notes.push(`${last[0]} returned ${last[1]}, so the test could not read ${access} from its result${last[1] === undefined ? ': did the function forget to `return` a value?' : '.'}`);
      } else if (expr) {
        const indexed = /\]\s*$/.test(expr);
        notes.push(`\`${expr}\` was ${kind}${lineNo ? ` on line ${lineNo}` : ''}, so it has no ${access}. ${indexed
          ? 'An index past the end of an array (or a Map key that is not there) gives undefined: check the name and the index, e.g. that a loop stops at i < arr.length.'
          : `Check the spelling of ${expr.split(/[.[(]/)[0]} and that it was given a value first.`}`);
      } else if (numeric && frame && lineText && indexingOn(lineText).length) {
        notes.push(`on line ${lineNo} something being indexed with [ ] was ${kind} (${indexingOn(lineText).slice(0, 3).map((s) => `\`${s}\``).join(', ')}). An index past the end of an array gives undefined, and undefined cannot be indexed again: check each name and index.`);
      } else {
        notes.push(`the code asked for ${access} on a value that is ${kind} (${kind === 'null' ? 'null is an empty value: many unwritten starter functions return null, and String.match returns null when it finds nothing' : 'it does not exist'}). Check the name, and the index, of the value in front of ${access}.`);
      }
      if (prop === 'data' || prop === 'shape') notes.push(`a tensor was expected there: tensors are objects { shape, data }, so ${access} only works on a tensor.`);
    } else if ((m = /^(.+?) is not a function(?: or its return value is not iterable)?$/.exec(message) || /^(.+?) is not a function\. \(In '/.exec(message))) {
      const what = m[1];
      const member = /\.([A-Za-z_$][\w$]*)$/.exec(what);
      const prop = member ? member[1] : what;
      if (/is not iterable/.test(message)) notes.push('for…of and [a, b] = … unpacking need something list-like (an array, Map, Set or string); here they got something else.');
      else if (watch && watch.exports && member && /^(m|mod|module|learner)$/.test(what.split('.')[0]) && !watch.exports.has(prop)) {
        notes.push(`the tests look for a function called ${prop}, and your file does not export one. Check that the line still starts with \`export function ${prop}(\` and that the name is spelt exactly like that.`);
      } else if (prop === 'length' || prop === 'size') {
        notes.push(`${prop} is a number, not a function: write \`.${prop}\` without the brackets ().`);
      } else {
        notes.push(`you called ${what}(…), but ${what} is not a function. Check the spelling (names are case-sensitive), and that this kind of value has that method: arrays have .map and .push, a Map has .get and .set, a Float32Array has no .push.`);
      }
    } else if (/is not iterable/.test(message)) {
      notes.push('for…of and [a, b] = … unpacking need something list-like: an array, Map, Set or string. Here they got a value that is not (often undefined, or a plain object: for an object use Object.entries(obj)).');
    } else if ((m = /^(.+?) is not a constructor$/.exec(message))) {
      notes.push(`\`new ${m[1]}(…)\` needs a class (like Map, Float32Array or Tensor), and ${m[1]} is not one. Check the spelling and the capital letters.`);
    } else if (/Maximum call stack size exceeded|too much recursion/.test(message)) {
      notes.push('a function kept calling itself (or two functions kept calling each other) without ever stopping. Check that the recursion has a case that returns without calling again.');
    } else if (/Invalid (typed )?array length/i.test(message)) {
      notes.push('an array was created with a length that is negative, fractional or NaN (new Array(n), new Float32Array(n) or Array.from({ length: n })). Check how n is computed.');
    } else if ((isSyntax || /^Unexpected|reserved word|Cannot use the keyword/.test(message)) && !/JSON/.test(message)) {
      const hit = source ? reservedAsName(source, lineNo ? lineNo - 1 : 0, lineNo || 0) : null;
      m = /Unexpected token '([^']+)'|unexpected token: keyword '([^']+)'|Cannot use the keyword '([^']+)'|Unexpected keyword '([^']+)'/.exec(message);
      const word = m && (m[1] || m[2] || m[3] || m[4]);
      const fs = source && !hit ? fStringHit(source, lineNo) : null;
      if (hit) notes.push(reservedNote(hit));
      else if (fs) {
        notes.push(`line ${fs.line} has a Python f-string (\`${fs.text}\`), and JavaScript has no f-strings. Write the text between backticks and put each value in a \${…} slot${fs.fixed ? `: ${fs.fixed}` : ', e.g. `${letter}: ${count}`'}.`);
      }
      else if (/reserved word|let is disallowed as a lexically bound name|Cannot use the keyword/.test(message)) {
        notes.push('a reserved word (one of the words JavaScript keeps for itself, such as var, class, function, return, new, delete, in, default, let, static) is used as a name. Pick another name.');
      } else if (/Unexpected end of input|end of script|expected expression, got end/.test(message)) {
        notes.push('the file ended while a bracket was still open: every { ( [ needs its matching } ) ]. Look for a missing } at the end of a function or loop.');
      } else if (word && /^[)\]}]$/.test(word)) {
        notes.push(`there is an extra or misplaced ${word}: every { ( [ needs exactly one matching } ) ]. Check the brackets just before this point.`);
      } else if (/Invalid or unexpected token|illegal character/.test(message)) {
        notes.push('JavaScript met a character it cannot read here: often a curly quote (“ ” ‘ ’) pasted from a document instead of a plain quote (\' or "), or a string that is never closed.');
      } else if (/missing \) after argument list/.test(message)) {
        notes.push('a ) is missing, or two arguments have no comma between them, e.g. f(a b) instead of f(a, b).');
      } else if (/Missing initializer in const declaration/.test(message)) {
        notes.push('a const needs its value on the same line: `const total = 0;`. Use let if it only gets a value later.');
      } else if (/Illegal return statement/.test(message)) {
        notes.push('a return sits outside any function: usually a } just above it closes the function too early.');
      } else if (/Unexpected (identifier|number|string)/.test(message)) {
        notes.push('JavaScript did not expect a name or value there: often a missing comma, operator or bracket just before it.');
      } else if (/await is only valid/.test(message)) {
        notes.push('await only works inside a function declared with async: `async function name(…) { … }`.');
      }
    }

    if (!isSyntax) {
      notes.push(...contextNotes(watch));
      const src = sourceNotes(watch, err, message);
      notes.push(...src.first, ...src.last);
    }
    return notes.filter(Boolean);
  });
}

/**
 * The message a learner sees for an error from a test run: an assertion from T already carries its notes;
 * anything else (a TypeError in their code, a thrown Error) keeps its message and gains plain-words notes.
 */
export function failureMessage(err, watch = null) {
  const message = err && err.message !== undefined ? String(err.message) : String(err);
  if (err && err[EXPLAINED]) return message;
  return withNotes(message, explainError(err, watch));
}

/** Append plain-words notes to a message: one per line, each starting with an arrow. */
export function explainMessage(message, notes) { return withNotes(message, notes); }
