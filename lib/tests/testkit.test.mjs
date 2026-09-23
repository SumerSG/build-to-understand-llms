// Tests for app/testkit.js (the T assertions every module's tests use, and the plain-words explanations)
// and the number formatting in app/charts.js — run with: node --test lib/tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeT, watchLearner, explainError, failureMessage } from '../../app/testkit.js';
import { format } from '../../app/charts.js';

/** The message T throws, or null when the assertion passes. */
function msgOf(fn) {
  try { fn(); return null; } catch (e) { return e.message; }
}

test('T.eq and T.close keep their pass/fail semantics: -0 equals 0, NaN only NaN, Infinity only itself', () => {
  const T = makeT();
  assert.equal(msgOf(() => T.eq(-0, 0)), null);
  assert.equal(msgOf(() => T.eq([NaN], [NaN])), null);
  assert.equal(msgOf(() => T.close(-0, 0, 1e-9)), null);
  assert.equal(msgOf(() => T.close([NaN, Infinity], [NaN, Infinity])), null);
  assert.notEqual(msgOf(() => T.close(Infinity, -Infinity)), null);
  assert.notEqual(msgOf(() => T.close(1e300, Infinity)), null);
  assert.notEqual(msgOf(() => T.close(NaN, 1)), null);
  assert.equal(msgOf(() => T.close(1000.05, 1000, 1e-4)), null, 'tolerance is relative above 1');
  assert.notEqual(msgOf(() => T.close(1.0002, 1, 1e-4)), null);
  assert.notEqual(msgOf(() => T.eq(new Map([['a', 1]]), new Map([['a', 2]]))), null);
  assert.equal(msgOf(() => T.eq(new Map([['b', 2], ['a', 1]]), new Map([['a', 1], ['b', 2]]))), null);
  assert.notEqual(msgOf(() => T.shape({ shape: [2, 3], data: new Float32Array(6) }, [3, 2])), null);
  assert.equal(msgOf(() => T.throws(() => { throw new Error('x'); })), null);
  assert.notEqual(msgOf(() => T.throws(() => {})), null);
});

test('element messages say which element and the allowed error, in plain words', () => {
  const T = makeT();
  const m = msgOf(() => T.close([0.5, -0.2], [0.25, -0.2], 1e-6));
  assert.match(m, /^element \[0\] differs: expected 0\.25, got 0\.5 \(allowed error 0\.000001\)/);
  // above 1 the tolerance is relative, and the message shows the error that was actually allowed
  assert.match(msgOf(() => T.close([1, -2, 3], [2, -4, 6], 1e-6)), /^element \[0\] differs: expected 2, got 1 \(allowed error 0\.000002\)/);
  assert.match(msgOf(() => T.close([[1, 2], [3, 4]], [[1, 2], [3, 5]])), /^element \[1\]\[1\] differs/);
  assert.match(msgOf(() => T.close(1, 2, 1e-6)), /^value differs: expected 2, got 1/);
});

test('NaN and Infinity messages name the beginner causes as well as overflow', () => {
  const T = makeT();
  const nan = msgOf(() => T.close([NaN], [0.5]));
  assert.match(nan, /reading an element that does not exist/);
  assert.match(nan, /an index past the end of an array or of a tensor's \.data \(an index formula error\)/);
  assert.match(nan, /indexing a tensor object itself instead of its \.data/);
  assert.match(nan, /placeholder that is not implemented yet/);
  assert.match(msgOf(() => T.close(Infinity, 1)), /dividing by zero, log\(0\)/);
});

test('"forgot to return" appears only when the function really returned undefined', () => {
  const starter = 'export function f(x) {\n  // TODO: step 1\n  return new Map();\n}\nexport function g(x) {\n  // TODO\n  return [];\n}\n';
  // the learner forgot the return
  const w1 = watchLearner({ f: (x) => { new Map([[x, 1]]); } }, { starter });
  const f1 = w1.module.f('a');
  assert.match(msgOf(() => makeT(w1).eq(f1, 3)), /f returned undefined: did the function forget to `return` a value\?/);
  // the learner returned an empty Map and the test looks a key up in it
  const w2 = watchLearner({ f: () => new Map() }, { starter });
  const m2 = msgOf(() => makeT(w2).eq(w2.module.f('a').get('a'), 3));
  assert.doesNotMatch(m2, /forget to `return`/);
  assert.match(m2, /f returned an empty Map, so looking anything up in it gives undefined/);
  // a function that changes its input and returns nothing by design is never blamed
  const w3 = watchLearner({ step: (p) => { p.x = 1; } }, { starter: 'export function step(p) {\n  // TODO\n}\n' });
  w3.module.step({});
  assert.doesNotMatch(msgOf(() => makeT(w3).ok(false, 'p did not move')), /forget/);
});

test('an untouched starter function is called "not written yet"; an edited one is not', () => {
  const starter = 'export function g(x) {\n  // TODO: step 2\n  return [];\n}\n';
  const mod = await_import(starter);
  const w = watchLearner(mod, { starter, source: starter });
  const got = w.module.g(1);
  assert.match(msgOf(() => makeT(w).eq(got, [1, 2])), /g is still the starter code \(its TODO is untouched\): looks like this function is not written yet/);
  const edited = watchLearner({ g: function g(x) { return []; } }, { starter });
  const got2 = edited.module.g(1);
  const m = msgOf(() => makeT(edited).eq(got2, [1, 2]));
  assert.doesNotMatch(m, /still the starter code/);
  assert.match(m, /empty array, the starter's usual placeholder/);
});

// A module's exported function whose source text is exactly the starter's (what the browser loads).
function await_import(src) {
  const body = /\{([\s\S]*)\}/.exec(src)[1];
  // eslint-disable-next-line no-new-func
  const g = new Function(`return function g(x) {${body}}`)();
  return { g };
}

test('for...in keys where words were expected are pointed out', () => {
  const m = msgOf(() => makeT().eq(['0', '1', '2'], ['the', 'cat', 'sat']));
  assert.match(m, /positions '0', '1', '2'/);
  assert.match(m, /for \(const w of words\)/);
  assert.doesNotMatch(msgOf(() => makeT().eq(['1', '2'], ['3', '4'])) ?? '', /positions/);
});

test('the recorder changes no value, identity or class behaviour', () => {
  class Box { constructor(v) { this.v = v; } static make(v) { return new this(v); } }
  const fn = (a, b) => a + b;
  const w = watchLearner({ fn, Box, n: 3 });
  assert.equal(w.module.fn(2, 3), 5);
  assert.equal(w.module.fn, w.module.fn);
  assert.equal(w.module.Box, Box);
  assert.ok(Box.make(1) instanceof w.module.Box);
  assert.equal(w.module.n, 3);
  assert.deepEqual(Object.keys(w.module), ['fn', 'Box', 'n']);
  assert.deepEqual(w.recent(), [['fn', 5]]);
});

test('raw JavaScript errors keep their text and gain a plain hint', () => {
  const source = 'export function f(words) {\n  for (const w of words) counts.set(w, 1);\n  return counts;\n}\n';
  let err;
  try { counts; } catch (e) { err = e; }   // eslint-disable-line no-undef
  const m = failureMessage(err, { source });
  assert.match(m, /^counts is not defined\n/);
  assert.match(m, /you used the name counts before creating it; declare it first, e\.g\. `const counts = …`/);

  const bad = 'export function norm(row) {\n  const mu = 0;\n  var = 0;\n  return var;\n}\n';
  const notes = explainError(Object.assign(new SyntaxError("Unexpected token '='"), {}), { source: bad }, { line: 3 });
  assert.match(notes.join(' '), /line 3 uses `var` as a name, but var is a reserved word.*`variance`/);
  const cls = explainError(new SyntaxError('Unexpected strict mode reserved word'), { source: 'const class = 1;\n' }, { line: 1 });
  assert.match(cls.join(' '), /`class` as a name/);

  assert.match(explainError(new TypeError('Assignment to constant variable.'), { source: 'const total = 0;\ntotal += 1;\n', file: 'L.js' }).join(' '),
    /declare it with let instead/);
  assert.match(explainError(new TypeError('pairs.length is not a function')).join(' '), /length is a number, not a function/);
  assert.match(explainError(new TypeError('m.topk is not a function'), { source: '', exports: new Set(['topK']), recent: () => [] }).join(' '),
    /your file does not export one/);
  assert.match(explainError(new TypeError("Cannot read properties of undefined (reading 'count')")).join(' '), /\.count on a value that is undefined/);
  assert.match(explainError(new ReferenceError('range is not defined')).join(' '), /JavaScript has no range\(\)/);
  const same = explainError(new ReferenceError('A is not defined'), { source: 'export function mm(a, b) {\n  return A;\n}\n' }).join(' ');
  assert.match(same, /your code creates a: names are case-sensitive/);
});

/** A learner module built from source text, recorded the way the sandbox worker records it. */
async function learner(source) {
  const file = `learner-${learner.n = (learner.n || 0) + 1}.js`;
  const mod = await import('data:text/javascript;base64,' + Buffer.from(`${source}\n//# sourceURL=${file}`).toString('base64'));
  return watchLearner(mod, { source, file });
}

test('a line-level note reaches only the tests that ran that line', async () => {
  const source = [
    'export function transpose(a) {',
    '  return a;',
    '}',
    'function same(a, b) {',
    '  return a.shape == b.shape;',
    '}',
    'export function add(a, b) {',
    '  if (!same(a, b)) throw new Error("bad shape");',
    '  return a;',
    '}',
    '',
  ].join('\n');
  const w = await learner(source);
  const t = { shape: [2], data: new Float32Array(2) };
  // a test that only called transpose: nothing about line 5
  w.reset();
  w.module.transpose(t);
  assert.doesNotMatch(msgOf(() => makeT(w).ok(false, 'transpose is wrong')), /line 5/);
  // a test that called add, which calls the helper holding the ==: the note appears, on T failures and on errors
  w.reset();
  let err;
  try { w.module.add(t, { shape: [2], data: new Float32Array(2) }); } catch (e) { err = e; }
  assert.match(failureMessage(err, w), /line 5 compares arrays with == \(`a\.shape == b\.shape`\)/);
  assert.match(msgOf(() => makeT(w).ok(false, 'x')), /\[2, 3\] == \[2, 3\] is false/);
  // no recorded calls and no stack: no claim about any line
  assert.doesNotMatch(msgOf(() => makeT({ source }).ok(false, 'x')), /line 5/);
});

test('^ used as a power is named as XOR, first, and only for the function that uses it', async () => {
  const source = 'export function variance(xs) {\n  let v = 0;\n  for (const x of xs) v += (x - 1)^2;\n  return v / xs.length;\n}\nexport function other(x) {\n  return x;\n}\n';
  const w = await learner(source);
  const got = w.module.variance([0.5, 1.5]);
  const m = msgOf(() => makeT(w).close(got, 0.25, 1e-6));
  assert.match(m.split('\n')[1], /^→ line 3 uses \^ as "to the power of" \(`\(x - 1\)\^2`\), but in JavaScript \^ is bitwise XOR/);
  assert.match(m, /write x \* x or x \*\* 2/);
  w.reset();
  w.module.other(1);
  assert.doesNotMatch(msgOf(() => makeT(w).eq(1, 2)), /XOR/);
  // x ^ 2, x ^ 0.5 are caught too; a hash's h ^ 2166136261 or h ^= x is not
  const w2 = await learner('export function f(x) {\n  return Math.sqrt(x ^ 2) + x ^ 0.5;\n}\nexport function h(s) {\n  let h = 7;\n  h ^= s;\n  return h ^ 2166136261;\n}\n');
  w2.module.f(3);
  assert.match(msgOf(() => makeT(w2).ok(false, 'x')), /line 2 uses \^/);
  w2.reset();
  w2.module.h(3);
  assert.doesNotMatch(msgOf(() => makeT(w2).ok(false, 'x')), /XOR/);
});

test('a Map filled like a Python dict is pointed out with its line and the .set/.get fix', async () => {
  const w = await learner('export function count(words) {\n  const counts = new Map();\n  for (const w of words) counts[w] = (counts[w] || 0) + 1;\n  return counts;\n}\n');
  const c = w.module.count(['a', 'b', 'a']);
  const m = msgOf(() => makeT(w).ok(c.has('a'), 'no entry for "a"'));
  assert.match(m, /count returned a Map with no entries but 2 ordinary properties \("a", "b"\)\. Line 3 writes `counts\[w\] = …`/);
  assert.match(m, /counts\.set\(w, …\) to store a value and counts\.get\(w\) to read one/);
  // the test's own message already says .set: no second explanation
  assert.doesNotMatch(msgOf(() => makeT(w).ok(false, 'use counts.set(w, n)')), /ordinary propert/);
  // a Map filled with .set is never blamed
  const ok = await learner('export function count(words) {\n  const counts = new Map();\n  for (const w of words) counts.set(w, 1);\n  return counts;\n}\n');
  ok.module.count(['a']);
  assert.doesNotMatch(msgOf(() => makeT(ok).ok(false, 'x')), /ordinary propert/);
});

test('a Python f-string in a syntax error gets the backtick version', () => {
  const source = 'export function label(letter, count) {\n  return f"{letter}: {count}";\n}\n';
  const notes = explainError(new SyntaxError('Unexpected string'), { source }, { line: 2 }).join(' ');
  assert.match(notes, /line 2 has a Python f-string \(`f"\{letter\}: \{count\}"`\), and JavaScript has no f-strings/);
  assert.match(notes, /slot: `\$\{letter\}: \$\{count\}`/);
  assert.doesNotMatch(explainError(new SyntaxError('Unexpected string'), { source: 'const s = "a" "b";\n' }, { line: 1 }).join(' '), /f-string/);
});

test('a function that is not defined because it sits inside another function is told to move out', () => {
  const source = 'export function add(a, b) {\n  function binary(x, y, fn) {\n    return fn(x, y);\n  }\n  return binary(a, b, (x, y) => x + y);\n}\nexport function mul(a, b) {\n  return binary(a, b, (x, y) => x * y);\n}\n';
  const notes = explainError(new ReferenceError('binary is not defined'), { source }).join(' ');
  assert.match(notes, /binary exists, but it is defined inside add \(line 2\)/);
  assert.match(notes, /Move the whole binary function out to the top level of the file, on its own \(for example just above `export function add`\)/);
  assert.doesNotMatch(notes, /declare it first/);
});

test('library matmul errors get plain shape advice', () => {
  const source = 'export class T {\n  mm(o) {\n    return fromOp(ops.matmul(this, o), (g) => {\n      accumulate(this, ops.matmul(g, o).data);\n    });\n  }\n}\n';
  const err = new Error('matmul: inner dims differ (4 vs 3) for [2,4] x [3,4]');
  err.stack = `Error: ${err.message}\n    at matmul (lib/ops.js:197:9)\n    at L.js:4:24\n`;
  const notes = explainError(err, { source, file: 'L.js' }).join('\n');
  assert.match(notes, /the last size of the left one must equal the first size of the right one/);
  assert.match(notes, /only one way round: ops\.matmul\(g, ops\.transpose\(o\)\) \(\[2, 4\] x \[4, 3\] gives \[2, 3\]\)/);
  assert.match(notes, /dA must be A's shape \[n, k\], and only dC · transpose\(B\)/);
  // without a line: the same advice in words
  assert.match(explainError(new Error('matmul: inner dims differ (3 vs 2) for [4,3] x [2,4]')).join(' '), /the transpose of the left one times the transpose of the right one|the transpose of both/);
  const flat = explainError(new Error('matmul: need at least 2D tensors')).join(' ');
  assert.match(flat, /must become a column of shape \[N, 1\].*xs\.map\(\(x\) => \[x\]\)/);
});

test('NaN from a .data index past the end is named without blaming the tensor object', async () => {
  const w = await learner('export function sum(a) {\n  let s = 0;\n  for (let j = 0; j <= a.data.length; j++) s += a.data[j];\n  return s;\n}\n');
  const got = w.module.sum({ shape: [2], data: new Float32Array([1, 2]) });
  const m = msgOf(() => makeT(w).close(got, 3));
  assert.match(m, /your code reads the numbers through \.data, so look for an index that runs past the end of a \.data array/);
  assert.doesNotMatch(m, /tensor object/);
  const w2 = await learner('export function first(a) {\n  const n = a.shape[0];\n  return a[0] + n;\n}\n');
  const got2 = w2.module.first({ shape: [2], data: new Float32Array([1, 2]) });
  assert.match(msgOf(() => makeT(w2).close(got2, 3)), /line 3 reads `a\[…\]`, but a looks like a tensor object/);
});

test('chart numbers have no trailing zeros and log axes use plain values', () => {
  assert.equal(format.fmtNum(9.3), '9.3');
  assert.equal(format.fmtNum(0.2), '0.2');
  assert.equal(format.fmtNum(0.30000000000000004), '0.3');
  assert.equal(format.fmtNum(NaN), '–');
  assert.equal(format.fmtTick(10000), '10,000');
  assert.equal(format.fmtTick(2.5e9), '2.5B');
  assert.deepEqual(format.logTicks(1, 1600), [1, 10, 100, 1000]);
  assert.deepEqual(format.logTicks(1, 40), [1, 2, 5, 10, 20]);
  assert.equal(format.preLogged('log10(rank): 0 is rank 1'), 'rank');
  assert.equal(format.preLogged('rank'), null);
});
