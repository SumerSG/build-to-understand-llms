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
  assert.match(nan, /indexing the tensor object instead of its \.data/);
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

test('code that compares arrays with == is pointed out on any failure', () => {
  const source = 'export function add(a, b) {\n  if (a.shape == b.shape) return a;\n  throw new Error("bad shape");\n}\n';
  assert.match(failureMessage(new Error('bad shape'), { source }), /line 2 compares arrays with ==/);
  assert.match(msgOf(() => makeT({ source }).ok(false, 'x')), /\[2, 3\] == \[2, 3\] is false/);
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
