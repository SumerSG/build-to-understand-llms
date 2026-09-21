// Tests for lib/tokenizer.js — run with: node --test lib/tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BPETokenizer, CharTokenizer, PRETOKEN_RE, UNK, pretokenize } from '../tokenizer.js';
import { CORPUS } from '../data.js';

test('CharTokenizer: sorted unique characters, exact round-trip', () => {
  const tokenizer = new CharTokenizer('banana bread');
  assert.deepEqual(tokenizer.itos, [' ', 'a', 'b', 'd', 'e', 'n', 'r']);
  assert.equal(tokenizer.vocabSize, 7);
  assert.equal(tokenizer.decode(tokenizer.encode('bread and banana')), 'bread and banana');
  assert.equal(tokenizer.encode('zzz').length, 0); // characters outside the vocabulary are skipped
  assert.deepEqual(tokenizer.encode('ab'), [1, 2]); // ids are positions in the sorted vocabulary
});

test('CharTokenizer: toJSON/fromJSON round-trip', () => {
  const tokenizer = new CharTokenizer(CORPUS.slice(0, 2000));
  const copy = CharTokenizer.fromJSON(JSON.parse(JSON.stringify(tokenizer.toJSON())));
  assert.deepEqual(copy.itos, tokenizer.itos);
  assert.equal(copy.vocabSize, tokenizer.vocabSize);
  const text = CORPUS.slice(100, 400);
  assert.deepEqual(copy.encode(text), tokenizer.encode(text));
  assert.equal(copy.decode(copy.encode(text)), text);
});

test('PRETOKEN_RE splits words, numbers and punctuation, keeping leading spaces', () => {
  assert.deepEqual(pretokenize('the cat sat on 42 mats!'), ['the', ' cat', ' sat', ' on', ' 42', ' mats', '!']);
  // A whitespace run is greedy, so the following word does not also get a leading space.
  assert.deepEqual(pretokenize('a\n  b'), ['a', '\n  ', 'b']);
  // The shared global regex is reusable: match() resets lastIndex, so a second call agrees.
  assert.deepEqual('hi there'.match(PRETOKEN_RE), 'hi there'.match(PRETOKEN_RE));
});

test('BPE: exact round-trip on CORPUS slices', () => {
  const tokenizer = BPETokenizer.train(CORPUS, { vocabSize: 256 });
  assert.equal(tokenizer.vocabSize, 256);
  for (const start of [0, 5000, 19000, 33000, 41000]) {
    const text = CORPUS.slice(start, start + 900);
    const ids = tokenizer.encode(text);
    assert.ok(ids.every((id) => Number.isInteger(id) && id >= 0 && id < tokenizer.vocabSize));
    assert.equal(tokenizer.decode(ids), text, `round-trip failed at ${start}`);
  }
  assert.equal(tokenizer.decode(tokenizer.encode(CORPUS)), CORPUS);
});

test('BPE: characters outside the training text become the unknown token, without crashing', () => {
  const tokenizer = BPETokenizer.train('the cat sat on the mat. the cat ate the hat.', { vocabSize: 60 });
  const unkId = tokenizer.stoi.get(UNK);
  assert.ok(Number.isInteger(unkId));
  const ids = tokenizer.encode('the ☃ cat é');
  assert.equal(ids.filter((id) => id === unkId).length, 2);
  assert.equal(tokenizer.decode([unkId]), UNK);
  assert.equal(tokenizer.decode(tokenizer.encode('the cat')), 'the cat'); // known text still exact
});

test('BPE: special tokens are matched first and encode to one id each', () => {
  const specials = ['<|endoftext|>', '<|user|>', '<|assistant|>', '<|end|>'];
  const tokenizer = BPETokenizer.train(CORPUS.slice(0, 8000), { vocabSize: 200, specials });
  for (const special of specials) {
    assert.deepEqual(tokenizer.encode(special), [tokenizer.stoi.get(special)]);
  }
  assert.equal(tokenizer.eos, tokenizer.stoi.get('<|endoftext|>'));
  assert.equal(tokenizer.vocab.slice(-specials.length).join('|'), specials.join('|'));
  const ids = tokenizer.encode('<|user|>hi<|end|>');
  assert.equal(ids[0], tokenizer.stoi.get('<|user|>'));
  assert.equal(ids[ids.length - 1], tokenizer.stoi.get('<|end|>'));
  assert.equal(tokenizer.decode(ids), '<|user|>hi<|end|>');
});

test('BPE: a bigger vocabulary means fewer tokens per character', () => {
  const text = CORPUS.slice(0, 20000);
  const ratios = [96, 192, 384].map((vocabSize) => {
    const tokenizer = BPETokenizer.train(CORPUS, { vocabSize });
    return tokenizer.encode(text).length / text.length;
  });
  assert.ok(ratios[0] > ratios[1], `expected ${ratios[0]} > ${ratios[1]}`);
  assert.ok(ratios[1] > ratios[2], `expected ${ratios[1]} > ${ratios[2]}`);
  assert.ok(ratios[2] < 0.5, `expected fewer than 0.5 tokens per character, got ${ratios[2]}`);
});

test('BPE: training is deterministic', () => {
  const a = BPETokenizer.train(CORPUS.slice(0, 12000), { vocabSize: 180 });
  const b = BPETokenizer.train(CORPUS.slice(0, 12000), { vocabSize: 180 });
  assert.deepEqual(a.merges, b.merges);
  assert.deepEqual(a.vocab, b.vocab);
  assert.deepEqual(a.encode(CORPUS.slice(0, 500)), b.encode(CORPUS.slice(0, 500)));
});

test('BPE: toJSON/fromJSON round-trip', () => {
  const tokenizer = BPETokenizer.train(CORPUS.slice(0, 12000), { vocabSize: 200 });
  const json = JSON.parse(JSON.stringify(tokenizer.toJSON()));
  const copy = BPETokenizer.fromJSON(json);
  assert.deepEqual(copy.toJSON(), tokenizer.toJSON());
  assert.equal(copy.vocabSize, tokenizer.vocabSize);
  assert.equal(copy.eos, tokenizer.eos);
  const text = CORPUS.slice(2000, 3000);
  assert.deepEqual(copy.encode(text), tokenizer.encode(text));
  assert.equal(copy.decode(copy.encode(text)), text);
});

test('BPE: training ~40 KB to vocab 256 stays well under 3 seconds', () => {
  const started = Date.now();
  const tokenizer = BPETokenizer.train(CORPUS, { vocabSize: 256 });
  const elapsed = Date.now() - started;
  assert.equal(tokenizer.vocabSize, 256);
  assert.ok(elapsed < 3000, `training took ${elapsed} ms`);
});

test('BPE: stops early when no pair repeats, and keeps every base character', () => {
  const tokenizer = BPETokenizer.train('abcdef', { vocabSize: 100 });
  assert.deepEqual(tokenizer.merges, []);
  assert.equal(tokenizer.decode(tokenizer.encode('face')), 'face');
  assert.ok(tokenizer.vocabSize < 100); // 6 characters + <|unk|> + one special
});
