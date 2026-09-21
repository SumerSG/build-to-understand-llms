// Tests for lib/sampling.js — run with: node --test lib/tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  applyTemperature, greedy, minPFilter, repetitionPenalty, sample, softmaxLogits, topKFilter, topPFilter,
} from '../sampling.js';
import { rng } from '../util.js';

const LOGITS = Float32Array.from([3, 2, 1, 0, -1]);

test('every processor returns a fresh Float32Array and leaves its input alone', () => {
  const original = [...LOGITS];
  const outputs = [
    applyTemperature(LOGITS, 0.5), topKFilter(LOGITS, 2), topPFilter(LOGITS, 0.5),
    minPFilter(LOGITS, 0.1), repetitionPenalty(LOGITS, [0, 1], 1.5), softmaxLogits(LOGITS),
  ];
  for (const out of outputs) {
    assert.ok(out instanceof Float32Array);
    assert.equal(out.length, LOGITS.length);
  }
  assert.deepEqual([...LOGITS], original);
});

test('processors accept plain number arrays too', () => {
  assert.deepEqual([...applyTemperature([2, 1], 2)], [1, 0.5]);
  assert.equal(greedy([0.1, 9, 3]), 1);
});

test('temperature: 1 is a copy, small sharpens, large flattens, 0 is greedy', () => {
  assert.deepEqual([...applyTemperature(LOGITS, 1)], [...LOGITS]);
  const sharp = softmaxLogits(applyTemperature(LOGITS, 0.1));
  const flat = softmaxLogits(applyTemperature(LOGITS, 100));
  assert.ok(sharp[0] > 0.99, `expected a near one-hot, got ${sharp[0]}`);
  for (const p of flat) assert.ok(Math.abs(p - 1 / LOGITS.length) < 0.01, 'high temperature is near uniform');
  const zero = applyTemperature(LOGITS, 0);
  assert.equal(zero[0], 0);
  assert.ok(zero.slice(1).every((v) => v === -Infinity));
  assert.equal(greedy(zero), greedy(LOGITS));
});

test('topKFilter keeps exactly k logits; k <= 0 or k >= vocab is a no-op', () => {
  const kept = topKFilter(LOGITS, 2);
  assert.deepEqual([...kept], [3, 2, -Infinity, -Infinity, -Infinity]);
  assert.equal(kept.filter((v) => Number.isFinite(v)).length, 2);
  assert.deepEqual([...topKFilter(LOGITS, 0)], [...LOGITS]);
  assert.deepEqual([...topKFilter(LOGITS, 99)], [...LOGITS]);
  // Ties are broken by index, so the filter is deterministic.
  assert.deepEqual([...topKFilter([1, 1, 1], 2)], [1, 1, -Infinity]);
});

test('topPFilter keeps the smallest prefix reaching p, and never everything-removed', () => {
  // probs = [0.5, 0.25, 0.25]
  const logits = Float32Array.from([Math.log(2), 0, 0]);
  assert.deepEqual([...topPFilter(logits, 0.4)].map(Number.isFinite), [true, false, false]);
  assert.deepEqual([...topPFilter(logits, 0.6)].map(Number.isFinite), [true, true, false]);
  assert.deepEqual([...topPFilter(logits, 0.9)].map(Number.isFinite), [true, true, true]);
  assert.deepEqual([...topPFilter(logits, 1)], [...logits]);
  // Even a tiny p keeps one token.
  assert.equal([...topPFilter(logits, 0.001)].filter(Number.isFinite).length, 1);
  // A flat distribution keeps more tokens than a peaked one at the same p.
  const flat = topPFilter([0, 0, 0, 0, 0], 0.5).filter(Number.isFinite).length;
  const peaked = topPFilter([10, 0, 0, 0, 0], 0.5).filter(Number.isFinite).length;
  assert.ok(flat > peaked, `flat kept ${flat}, peaked kept ${peaked}`);
});

test('minPFilter drops tokens below p times the top probability', () => {
  // probs = [0.64, 0.24, 0.09, 0.03] for these logits
  const probs = softmaxLogits(LOGITS.slice(0, 4));
  const filtered = minPFilter(LOGITS.slice(0, 4), 0.3);
  for (let i = 0; i < 4; i++) {
    const shouldKeep = probs[i] >= 0.3 * probs[0];
    assert.equal(Number.isFinite(filtered[i]), shouldKeep, `token ${i}`);
  }
  assert.deepEqual([...minPFilter(LOGITS, 0)], [...LOGITS]);
  assert.equal(minPFilter(LOGITS, 0.999).filter(Number.isFinite).length, 1, 'the argmax always survives');
});

test('repetitionPenalty divides positive logits and multiplies negative ones', () => {
  const penalised = repetitionPenalty([2, -2, 5], [0, 1], 2);
  assert.deepEqual([...penalised], [1, -4, 5]);
  assert.deepEqual([...repetitionPenalty([2, -2, 5], [0, 1], 1)], [2, -2, 5]);
  assert.deepEqual([...repetitionPenalty([2, -2, 5], [], 2)], [2, -2, 5]);
  // Repeating the same id twice is the same as once (it is a set, not a count).
  assert.deepEqual([...repetitionPenalty([2, -2, 5], [0, 0, 0], 2)], [1, -2, 5]);
  // Out-of-range ids are ignored rather than crashing.
  assert.deepEqual([...repetitionPenalty([2, -2, 5], [9, -1], 2)], [2, -2, 5]);
});

test('softmaxLogits is stable and normalised', () => {
  const probs = softmaxLogits([1000, 1001, 1002]);
  assert.ok(probs.every(Number.isFinite));
  assert.ok(Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-6);
  assert.ok(probs[2] > probs[1] && probs[1] > probs[0]);
  const masked = softmaxLogits([0, -Infinity, -Infinity]);
  assert.deepEqual([...masked], [1, 0, 0]);
  const empty = softmaxLogits([-Infinity, -Infinity]);
  assert.deepEqual([...empty], [0.5, 0.5], 'a fully masked row falls back to uniform');
});

test('sample: 20,000 draws match the expected distribution', () => {
  // probs = [0.5, 0.25, 0.125, 0.125]
  const logits = Float32Array.from([Math.log(4), Math.log(2), 0, 0]);
  const expected = [...softmaxLogits(logits)];
  const next = rng(3);
  const counts = new Array(4).fill(0);
  const draws = 20000;
  for (let i = 0; i < draws; i++) counts[sample(logits, { next })]++;
  // The sampling error at 20,000 draws is about 0.35 percentage points for p = 0.5, so 2 points
  // absolute is a comfortable bound; the 3% relative bound keeps the small probabilities honest too.
  for (let i = 0; i < 4; i++) {
    const observed = counts[i] / draws;
    const error = Math.abs(observed - expected[i]);
    assert.ok(
      error <= 0.02 && error <= 0.03 * expected[i],
      `token ${i}: observed ${observed.toFixed(4)} vs expected ${expected[i].toFixed(4)}`,
    );
  }
});

test('sample: filters are applied before drawing', () => {
  const logits = Float32Array.from([3, 2, 1, 0, -1]);
  const next = rng(11);
  const counts = new Map();
  const draws = 20000;
  for (let i = 0; i < draws; i++) {
    const id = sample(logits, { topK: 2, next });
    counts.set(id, (counts.get(id) ?? 0) + 1);
  }
  assert.deepEqual([...counts.keys()].sort(), [0, 1], 'top-k 2 must only ever draw the two best tokens');
  const expected = [...softmaxLogits(topKFilter(logits, 2))];
  assert.ok(Math.abs(counts.get(0) / draws - expected[0]) <= 0.02);
  // Temperature 0 is deterministic.
  const greedyDraws = new Set();
  for (let i = 0; i < 50; i++) greedyDraws.add(sample(logits, { temperature: 0, next }));
  assert.deepEqual([...greedyDraws], [0]);
});

test('sample: the same seed replays the same tokens', () => {
  const logits = Float32Array.from([1, 2, 3, 4]);
  const draw = (seed) => {
    const next = rng(seed);
    return Array.from({ length: 20 }, () => sample(logits, { temperature: 0.8, topP: 0.9, next }));
  };
  assert.deepEqual(draw(3), draw(3));
  assert.notDeepEqual(draw(3), draw(4));
});

test('greedy returns the argmax, ties going to the first index', () => {
  assert.equal(greedy(LOGITS), 0);
  assert.equal(greedy([0, 5, 5]), 1);
  assert.equal(greedy(Float32Array.from([-3, -1, -2])), 1);
});
