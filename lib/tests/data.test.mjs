// Tests for lib/data.js — run with: node --test lib/tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  CHAT, CORPUS, INSTRUCTIONS, MATH_TASKS, PREFERENCES, PROSE,
  formatChat, getBatch, toyCorpus, trainValSplit,
} from '../data.js';
import { rng } from '../util.js';

test('toyCorpus is deterministic for a seed and different across seeds', () => {
  assert.equal(toyCorpus(100, 1), toyCorpus(100, 1));
  assert.notEqual(toyCorpus(100, 1), toyCorpus(100, 2));
  assert.equal(toyCorpus(100, 1).split('\n').length, 100);
  assert.equal(toyCorpus(0, 1), '');
});

test('toyCorpus sentences are readable English-like lines', () => {
  const lines = toyCorpus(200, 1).split('\n');
  for (const line of lines) {
    // A capital, or a digit for the templates that open with a count ("4 gardens are quiet...").
    assert.match(line, /^[A-Z\d]/, `line should start with a capital or a digit: ${line}`);
    assert.match(line, /[.?]$/, `line should end with punctuation: ${line}`);
    assert.ok(line.split(' ').length >= 3, `line should be a sentence: ${line}`);
  }
  assert.ok(lines.some((l) => /\d/.test(l)), 'some sentences should contain digits');
});

test('CORPUS is the toy corpus plus PROSE and about 40 KB of plain ASCII', () => {
  assert.ok(CORPUS.length >= 30000 && CORPUS.length <= 60000, `CORPUS is ${CORPUS.length} characters`);
  assert.ok(PROSE.length >= 10000 && PROSE.length <= 14000, `PROSE is ${PROSE.length} characters`);
  assert.ok(CORPUS.endsWith(PROSE));
  assert.ok(CORPUS.includes('\n\n' + PROSE.slice(0, 40)));
  const characters = [...new Set(CORPUS)];
  assert.ok(characters.every((c) => c.charCodeAt(0) < 127), 'CORPUS should stay ASCII');
  assert.ok(characters.length < 100, `character vocabulary is ${characters.length}`);
});

test('INSTRUCTIONS: at least 60 short, non-empty pairs', () => {
  assert.ok(INSTRUCTIONS.length >= 60, `got ${INSTRUCTIONS.length}`);
  for (const { prompt, response } of INSTRUCTIONS) {
    assert.equal(typeof prompt, 'string');
    assert.equal(typeof response, 'string');
    assert.ok(prompt.length > 0 && response.length > 0);
    assert.ok(prompt.length < 120 && response.length < 120);
  }
  assert.equal(new Set(INSTRUCTIONS.map((p) => p.prompt)).size, INSTRUCTIONS.length, 'prompts should be unique');
});

test('PREFERENCES: at least 40 triples whose chosen differs from rejected', () => {
  assert.ok(PREFERENCES.length >= 40, `got ${PREFERENCES.length}`);
  for (const { prompt, chosen, rejected } of PREFERENCES) {
    assert.ok(prompt.length > 0 && chosen.length > 0 && rejected.length > 0);
    assert.notEqual(chosen, rejected);
  }
  // Rejected answers come in two flavours: rambling (longer) and truncated/off-topic (shorter).
  const rambling = PREFERENCES.filter((p) => p.rejected.length > p.chosen.length + 20).length;
  const clipped = PREFERENCES.filter((p) => p.rejected.length < p.chosen.length).length;
  assert.ok(rambling >= 8, `only ${rambling} rambling rejections`);
  assert.ok(clipped >= 8, `only ${clipped} truncated rejections`);
  assert.ok(PREFERENCES.every((p) => p.chosen.length <= 80), 'chosen answers should stay concise');
});

test('MATH_TASKS: at least 100 one-step problems whose answers are correct', () => {
  assert.ok(MATH_TASKS.length >= 100, `got ${MATH_TASKS.length}`);
  for (const { question, answer } of MATH_TASKS) {
    const parts = question.match(/^What is (\d+) ([-+*]) (\d+)\?$/);
    assert.ok(parts, `unexpected question format: ${question}`);
    const a = Number(parts[1]);
    const b = Number(parts[3]);
    const expected = parts[2] === '+' ? a + b : parts[2] === '-' ? a - b : a * b;
    assert.equal(answer, String(expected));
    assert.ok(expected >= 0, 'answers should be non-negative');
  }
  const operators = new Set(MATH_TASKS.map((t) => t.question.split(' ')[3]));
  assert.deepEqual([...operators].sort(), ['*', '+', '-']);
});

test('CHAT markers and formatChat', () => {
  assert.deepEqual(CHAT, { system: '<|system|>', user: '<|user|>', assistant: '<|assistant|>', end: '<|end|>' });
  assert.equal(
    formatChat([{ role: 'user', content: 'hi' }]),
    '<|user|>hi<|end|><|assistant|>',
  );
  assert.equal(
    formatChat([
      { role: 'system', content: 'be brief' },
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
    ]),
    '<|system|>be brief<|end|><|user|>hi<|end|><|assistant|>hello<|end|>',
  );
  assert.equal(formatChat([]), '');
});

test('getBatch: shapes, valid ids, and y shifted one past x', () => {
  const ids = Array.from({ length: 500 }, (_, i) => i);
  const { x, y } = getBatch(ids, { blockSize: 8, batchSize: 4, next: rng(1) });
  assert.equal(x.length, 4);
  assert.equal(y.length, 4);
  for (let b = 0; b < 4; b++) {
    assert.equal(x[b].length, 8);
    assert.equal(y[b].length, 8);
    assert.ok(Array.isArray(x[b]) && Array.isArray(y[b]));
    for (let t = 0; t < 7; t++) assert.equal(y[b][t], x[b][t + 1], 'y is x shifted by one');
    assert.equal(y[b][7], x[b][7] + 1);
    assert.ok(x[b][0] >= 0 && y[b][7] < ids.length);
  }
});

test('getBatch: same seed gives the same batch, different seeds do not', () => {
  const ids = Array.from({ length: 500 }, (_, i) => i % 40);
  const opts = { blockSize: 16, batchSize: 3 };
  const a = getBatch(ids, { ...opts, next: rng(7) });
  const b = getBatch(ids, { ...opts, next: rng(7) });
  const c = getBatch(ids, { ...opts, next: rng(8) });
  assert.deepEqual(a, b);
  assert.notDeepEqual(a, c);
});

test('getBatch: refuses a sequence shorter than the block', () => {
  assert.throws(() => getBatch([1, 2, 3], { blockSize: 8, batchSize: 1, next: rng(1) }), /getBatch/);
});

test('trainValSplit: contiguous split that loses nothing', () => {
  const ids = Array.from({ length: 1000 }, (_, i) => i);
  const { train, val } = trainValSplit(ids);
  assert.equal(train.length, 900);
  assert.equal(val.length, 100);
  assert.deepEqual([...train, ...val], ids);
  const half = trainValSplit(ids, 0.5);
  assert.equal(half.train.length, 500);
  assert.equal(half.val[0], 500);
});
