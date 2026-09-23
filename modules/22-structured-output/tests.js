// Tests for module 22. Every schema, vocabulary and random model here is deterministic.

const PERSON = {
  type: 'object',
  properties: {
    name: { type: 'string', maxLength: 8 },
    age: { type: 'integer', maxDigits: 3 },
    member: { type: 'boolean' },
    tier: { enum: ['gold', 'silver', 'bronze'] },
  },
  required: ['name', 'age', 'member', 'tier'],
};

// Every printable ASCII character as its own token, plus multi-character tokens that span JSON syntax,
// a raw newline, an empty special token (never allowed) and an end-of-sequence token.
function testVocab() {
  const vocab = [];
  for (let c = 32; c < 127; c++) vocab.push(String.fromCharCode(c));
  vocab.push('{"', '":', '",', '"}', 'true', 'false', 'tru', '12', '0,', 'name', ' the', '\\n', '\\u00', '"gold"', '},', '\n', '');
  vocab.push('<|endoftext|>');
  return { vocab, eos: vocab.length - 1 };
}

function randomModel(T, seed, V, spread = 3) {
  const next = T.rng(seed);
  return () => {
    const l = new Float32Array(V);
    for (let j = 0; j < V; j++) l[j] = (next() * 2 - 1) * spread;
    return l;
  };
}

// Independent schema check on the parsed value: no automaton involved.
function conforms(schema, value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const keys = Object.keys(schema.properties);
  for (const k of Object.keys(value)) if (!keys.includes(k)) return false;
  for (const k of schema.required ?? []) if (!(k in value)) return false;
  for (const k of Object.keys(value)) {
    const s = schema.properties[k], v = value[k];
    if (s.enum && !s.enum.includes(v)) return false;
    if (s.type === 'string' && (typeof v !== 'string' || v.length > (s.maxLength ?? 16))) return false;
    if (s.type === 'integer' && !Number.isInteger(v)) return false;
    if (s.type === 'boolean' && typeof v !== 'boolean') return false;
  }
  return true;
}

function allowedTokens(mask, vocab) {
  const out = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) out.push(vocab[i]);
  return out;
}

export const tests = [
  // ---------- step 1: strings and integers ----------
  { step: 'values', name: 'the string machine accepts JSON strings, escapes included', run(m, T) {
    const s = m.stringMachine(16);
    for (const ok of ['""', '"hello"', '"a b, c}"', '"say \\"hi\\""', '"tab\\tnew\\nline"', '"back\\\\slash"', '"\\u00e9t\\u00E9"', '"\\/"']) {
      T.ok(m.matches(s, ok), `${ok} is a valid JSON string and must be accepted`);
    }
  } },
  { step: 'values', name: 'the string machine rejects unclosed, badly escaped, control-character and over-long strings', run(m, T) {
    const s = m.stringMachine(4);
    T.ok(m.matches(s, '"abcd"'), '"abcd" has exactly maxLength = 4 characters and must be accepted');
    T.ok(m.matches(s, '"ab\\nc"'), 'an escape such as \\n counts as ONE character towards maxLength');
    T.ok(!m.matches(s, '"abc\\n\\t"'), '"abc\\n\\t" holds 5 characters (each escape is one), so with maxLength 4 it must be rejected: an escape counts as one character, not zero');
    T.ok(!m.matches(s, '"\\u00e9\\u00e9\\u00e9\\u00e9\\u00e9"'), 'five \\u escapes are five characters, one more than maxLength = 4');
    T.ok(m.matches(s, '"\\u00e9"') && !m.matches(s, '"\\u00e"') && !m.matches(s, '"\\u00"'), '\\u must be followed by EXACTLY four hex digits: "\\u00e9" is valid, "\\u00e" and "\\u00" are not');
    const bad = [['"abc', 'it is never closed'], ['abc"', 'it does not start with a quote'], ['"a\\x"', '\\x is not a JSON escape'],
      ['"\\u12G4"', 'G is not a hex digit'], ['"a\nb"', 'raw control characters (code < 0x20) are illegal inside JSON strings'],
      ['"abcde"', 'it has 5 characters and maxLength is 4'], ['"ab"c', 'nothing may follow the closing quote inside the value']];
    for (const [text, why] of bad) T.ok(!m.matches(s, text), `${JSON.stringify(text)} must be rejected: ${why}`);
    T.eq(m.advance(s, s.start, '"ab'), m.advance(s, s.start, '"ab'), 'step must be deterministic');
    T.ok(m.advance(s, s.start, '"ab') !== null, 'a proper prefix of a valid string must keep the machine alive (step returns a state, not null)');
    T.ok(m.advance(s, s.start, '"abcd') !== null && m.advance(s, s.start, '"abcde') === null, 'the fifth character must be rejected the moment it arrives, not only at the end');
  } },
  { step: 'values', name: 'the integer machine follows -?(0|[1-9][0-9]*) with a digit cap', run(m, T) {
    const n = m.integerMachine(3);
    for (const ok of ['0', '7', '42', '-5', '-0', '999', '-120']) T.ok(m.matches(n, ok), `${ok} is a valid JSON integer and must be accepted`);
    const bad = [['', 'the empty string is not a number'], ['-', 'a lone minus is not a number'], ['007', 'JSON forbids leading zeros'],
      ['1000', 'maxDigits is 3'], ['4.5', 'the schema asks for an integer'], ['+3', 'JSON has no leading plus'], ['1-', 'minus only at the start']];
    for (const [text, why] of bad) T.ok(!m.matches(n, text), `${JSON.stringify(text)} must be rejected: ${why}`);
    T.ok(m.advance(n, n.start, '-') !== null, 'after "-" the machine is not accepting but must stay alive: a digit can still follow');
  } },

  // ---------- step 2: the object grammar ----------
  { step: 'object', name: 'accepts every valid object and rejects near-misses', run(m, T) {
    const M = m.compileSchema(PERSON);
    const good = ['{"name":"Ada","age":36,"member":true,"tier":"gold"}', '{"name":"","age":-0,"member":false,"tier":"bronze"}',
      '{"name":"a\\"b}","age":0,"member":false,"tier":"silver"}'];
    for (const g of good) T.ok(m.matches(M, g), `${g} matches the schema and must be accepted`);
    const bad = [
      ['{"name":"Ada","age":36,"member":true,"tier":"gold",}', 'a trailing comma is not JSON'],
      ['{"name":"Ada","age":36,"member":true,"tier":"gold"', 'the object is never closed'],
      ['{"name":"Ada","member":true,"tier":"gold"}', 'the required key "age" is missing'],
      ['{"age":36,"name":"Ada","member":true,"tier":"gold"}', 'keys must come in declaration order'],
      ['{"name":"Ada","age":"36","member":true,"tier":"gold"}', 'age must be an integer, not a string'],
      ['{"name":"Ada","age":36,"member":1,"tier":"gold"}', 'member must be true or false'],
      ['{"name":"Ada","age":36,"member":true,"tier":"platinum"}', 'tier must be one of the enum values'],
      ['{"name":"Ada","age":36,"member":true,"tier":"gold","x":1}', 'no extra keys are allowed'],
      ['{"name":"Ada","age":36,"member":true,"tier":"gold"} ', 'nothing may follow the closing brace'],
      ['{"name":"Ada","age":-,"member":true,"tier":"gold"}', 'a lone "-" is not a complete integer, so the comma may not end the value (check accepts(sub) before taking "," or "}")'],
      ['{"name":"Ada","age":36,"member":tru,"tier":"gold"}', '"tru" is not a complete boolean, so the comma may not end the value'],
      ['{"name":"Ada","age":36,"member":true,"tier":"gol}', '"gol is not a complete enum value, so "}" may not close the object'],
    ];
    for (const [text, why] of bad) T.ok(!m.matches(M, text), `${text} must be rejected: ${why}`);
  } },
  { step: 'object', name: 'optional keys may be skipped, required ones may not', run(m, T) {
    const S = { type: 'object', properties: { a: { type: 'integer' }, b: { type: 'boolean' }, c: { type: 'string' } }, required: ['c'] };
    const M = m.compileSchema(S);
    for (const g of ['{"c":"x"}', '{"a":1,"c":"x"}', '{"b":true,"c":""}', '{"a":1,"b":false,"c":"x"}']) T.ok(m.matches(M, g), `${g}: a and b are optional, so this must be accepted`);
    for (const b of ['{}', '{"a":1}', '{"a":1,"b":true}', '{"a":1,}']) T.ok(!m.matches(M, b), `${b}: "c" is required, so this must be rejected`);
    const E = m.compileSchema({ type: 'object', properties: { a: { type: 'integer' } } });
    T.ok(m.matches(E, '{}') && m.matches(E, '{"a":5}'), 'with no required keys, {} and {"a":5} are both valid');
    const N = m.compileSchema({ type: 'object', properties: { p: { type: 'object', properties: { q: { type: 'boolean' } }, required: ['q'] } }, required: ['p'] });
    T.ok(m.matches(N, '{"p":{"q":true}}') && !m.matches(N, '{"p":{}}'), 'nested objects compile through compileValue and follow the same rules');
  } },
  { step: 'object', name: 'every prefix of a valid object keeps the machine alive and none of them accepts early', run(m, T) {
    const M = m.compileSchema(PERSON);
    const text = '{"name":"x,y","age":12,"member":false,"tier":"silver"}';
    let s = M.start;
    for (let i = 0; i < text.length; i++) {
      s = M.step(s, text[i]);
      T.ok(s !== null, `the prefix ${JSON.stringify(text.slice(0, i + 1))} can still become valid JSON, so step must not return null`);
      if (i < text.length - 1) T.ok(!M.accepts(s), `the prefix ${JSON.stringify(text.slice(0, i + 1))} is incomplete and must not be accepted`);
    }
    T.ok(M.accepts(s), 'the complete object must be accepted');
  } },

  // ---------- step 3: the token mask ----------
  { step: 'mask', name: 'a token is allowed exactly when every one of its characters keeps the machine alive', run(m, T) {
    const { vocab, eos } = testVocab();
    const M = m.compileSchema(PERSON);
    const at = (prefix) => new Set(allowedTokens(m.maskForState(M, m.advance(M, M.start, prefix), vocab, eos), vocab));
    const s0 = at('');
    T.ok(s0.has('{"') && s0.has('{'), 'at the start, "{" and the two-character token \'{"\' are both legal');
    T.eq(s0.size, 2, 'at the start only "{" and \'{"\' may be emitted; anything else can never become a valid object');
    const inName = at('{"name":"ab');
    T.ok(inName.has('",') && inName.has(' the') && inName.has('\\n'), 'inside the name string: \'",\' (close the string, then a comma), " the" and the escape "\\n" are all legal');
    T.ok(!inName.has('"}'), '\'"}\' would close the object while age, member and tier are still required: it must be masked');
    T.ok(!inName.has('\n'), 'a raw newline is illegal inside a JSON string');
    const inMember = at('{"name":"ab","age":1,"member":');
    T.ok(inMember.has('tru') && inMember.has('true') && inMember.has('false'), 'a token may end mid-literal ("tru") as long as the machine stays alive');
    T.ok(!inMember.has('12') && !inMember.has('name'), 'only the letters of true/false may follow "member":');
  } },
  { step: 'mask', name: 'eos is allowed only when the machine accepts; empty special tokens never are', run(m, T) {
    const { vocab, eos } = testVocab();
    const M = m.compileSchema(PERSON);
    const empty = vocab.indexOf('');
    const done = m.advance(M, M.start, '{"name":"a","age":1,"member":true,"tier":"gold"}');
    const maskDone = m.maskForState(M, done, vocab, eos);
    T.eq(allowedTokens(maskDone, vocab), ['<|endoftext|>'], 'after the closing brace the ONLY legal token is eos: that is how generation stops');
    const mid = m.advance(M, M.start, '{"name":"a"');
    const maskMid = m.maskForState(M, mid, vocab, eos);
    T.eq(maskMid[eos], 0, 'eos in the middle of an object would leave truncated JSON; it must be masked');
    T.eq(maskMid[empty], 0, 'an empty token string advances nothing and would let the sampler loop forever; mask it');
    T.eq(maskMid.length, vocab.length, 'the mask has one entry per vocabulary token');
  } },
  { step: 'mask', name: 'on random valid walks at least one token is always allowed', run(m, T) {
    const { vocab, eos } = testVocab();
    const M = m.compileSchema(PERSON);
    const next = T.rng(5);
    for (let walk = 0; walk < 10; walk++) {
      let s = M.start;
      for (let stepN = 0; stepN < 200; stepN++) {
        const mask = m.maskForState(M, s, vocab, eos);
        const ok = [];
        for (let i = 0; i < mask.length; i++) if (mask[i]) ok.push(i);
        T.ok(ok.length > 0, 'a reachable state with no allowed token is a dead end: the sampler would have nothing to pick');
        const id = ok[Math.floor(next() * ok.length)];
        if (id === eos) break;
        s = m.advance(M, s, vocab[id]);
        T.ok(s !== null, `the mask allowed ${JSON.stringify(vocab[id])} but advancing by it killed the machine`);
      }
    }
  } },

  // ---------- step 4: constrained sampling ----------
  { step: 'sample', name: 'applyMask sets disallowed logits to -Infinity and leaves the rest untouched', run(m, T) {
    const logits = new Float32Array([1.5, -2, 0.25, 7]);
    const out = m.applyMask(logits, new Uint8Array([1, 0, 1, 0]));
    T.eq(Array.from(out), [1.5, -Infinity, 0.25, -Infinity], 'allowed logits keep their values; masked ones become -Infinity, so softmax gives them exactly 0');
    T.eq(Array.from(logits), [1.5, -2, 0.25, 7], 'the input logits must not be modified');
    T.ok(out instanceof Float32Array, 'return a Float32Array, the same type every logit processor in lib/sampling.js returns');
  } },
  { step: 'sample', name: 'a random-logit model produces schema-valid JSON in 100 out of 100 trials', run(m, T) {
    const { vocab, eos } = testVocab();
    const M = m.compileSchema(PERSON);
    const model = randomModel(T, 11, vocab.length);
    const next = T.rng(12);
    const bad = [];
    for (let i = 0; i < 100; i++) {
      const r = m.constrainedGenerate(model, M, vocab, { eos, next, maxTokens: 256 });
      T.ok(r && typeof r.text === 'string' && Array.isArray(r.ids), 'constrainedGenerate returns { text, ids, finished }');
      let value = null;
      try { value = JSON.parse(r.text); } catch { bad.push(r.text); continue; }
      if (!conforms(PERSON, value) || !r.finished) bad.push(r.text);
      T.eq(r.ids.map((id) => vocab[id]).join(''), r.text, 'text must be the concatenation of the sampled tokens');
    }
    T.eq(bad.length, 0, `every output must parse and match the schema, got ${bad.length} failures, e.g. ${JSON.stringify(bad[0])}`);
  } },
  { step: 'sample', name: 'a model that wants to break the format is overruled, and runs are reproducible', run(m, T) {
    const { vocab, eos } = testVocab();
    const M = m.compileSchema(PERSON);
    const hostile = () => { // loves '}', eos and the word " the", which are almost never legal together
      const l = new Float32Array(vocab.length).fill(-5);
      l[vocab.indexOf('}')] = 10; l[eos] = 9; l[vocab.indexOf(' the')] = 8; l[vocab.indexOf('0,')] = 6;
      return l;
    };
    const r = m.constrainedGenerate(hostile, M, vocab, { eos, next: T.rng(3) });
    let v = null;
    try { v = JSON.parse(r.text); } catch { T.fail(`output must be valid JSON even when the model prefers invalid tokens; got ${JSON.stringify(r.text)}`); }
    T.ok(conforms(PERSON, v), `output must match the schema; got ${r.text}`);
    T.ok(/^(\}| the|0,)+$/.test(v.name), `inside the name string '}', ' the' and '0,' are all legal, so the model's favourites must still win there; got name ${JSON.stringify(v.name)}`);
    T.eq(v.age, 0, 'after "age": the favourite legal token is "0," (a zero, then the comma): the mask filters, it does not replace the model\'s preferences');
    const a = m.constrainedGenerate(randomModel(T, 21, vocab.length), M, vocab, { eos, next: T.rng(22) });
    const b = m.constrainedGenerate(randomModel(T, 21, vocab.length), M, vocab, { eos, next: T.rng(22) });
    T.eq(a.text, b.text, 'same model, same seed: the same output (draw randomness only through `next`)');
  } },

  { step: 'sample', name: 'tokens are drawn with sample(), temperature and next from the masked distribution', run(m, T) {
    // A two-option grammar: the text is "a" or "b", then eos. The model always scores a: 0, b: ln 3, c: 5, eos: 0.
    // After the mask removes "c", softmax gives a 1/4 and b 3/4 at temperature 1; at temperature 0.5 the
    // logits double, so b gets 9/10. Greedy decoding (always "b") or ignoring temperature fails this.
    const vocab = ['a', 'b', 'c', '<eos>'];
    const eos = 3;
    const M = m.literalMachine(['a', 'b']);
    const model = () => new Float32Array([0, Math.log(3), 5, 0]);
    for (const [temperature, want, tol] of [[1, 0.75, 0.04], [0.5, 0.9, 0.03]]) {
      const next = T.rng(40);
      let b = 0;
      const N = 2000;
      for (let i = 0; i < N; i++) {
        const r = m.constrainedGenerate(model, M, vocab, { eos, next, temperature });
        T.ok(r.finished && (r.text === 'a' || r.text === 'b'), `the grammar allows only "a" or "b" then eos; got ${JSON.stringify(r.text)}`);
        if (r.text === 'b') b++;
      }
      T.close(b / N, want, tol, `at temperature ${temperature}, "b" must be drawn with probability about ${want} (softmax over the masked logits); got ${(b / N).toFixed(3)}. Draw with sample(masked, { temperature, next }), not argmax`);
    }
  } },
  { step: 'sample', name: 'the model is called with the ids generated so far', run(m, T) {
    const { vocab, eos } = testVocab();
    const M = m.compileSchema(PERSON);
    const inner = randomModel(T, 41, vocab.length);
    const seen = [];
    const model = (ids) => { seen.push(Array.from(ids ?? [])); return inner(); };
    const r = m.constrainedGenerate(model, M, vocab, { eos, next: T.rng(42) });
    T.eq(seen.length, r.ids.length + 1, 'one model call per sampled token, the last one producing eos');
    for (let k = 0; k < seen.length; k++) {
      T.eq(seen[k], r.ids.slice(0, k), `call ${k} must receive the ${k} ids generated before it; a model that never sees its own output cannot condition on it`);
    }
  } },

  // ---------- step 5: caching masks ----------
  { step: 'cache', name: 'the cached masker returns the same masks as maskForState', run(m, T) {
    const { vocab, eos } = testVocab();
    const M = m.compileSchema(PERSON);
    const masker = m.cachedMasker(M, vocab, eos);
    for (const prefix of ['', '{"name":"', '{"name":"ab', '{"name":"ab"', '{"name":"ab","age":4', '{"name":"ab","age":4,"member":tr', '{"name":"a","age":1,"member":true,"tier":"gold"}']) {
      const s = m.advance(M, M.start, prefix);
      const want = Array.from(m.maskForState(M, s, vocab, eos));
      T.eq(Array.from(masker(s)), want, `cached mask differs from maskForState after ${JSON.stringify(prefix)}`);
      T.eq(Array.from(masker(s)), want, `second (cached) lookup after ${JSON.stringify(prefix)} must return the same mask`);
    }
  } },
  { step: 'cache', name: 'stats count calls, misses and token scans, and repeated states are not rescanned', run(m, T) {
    const { vocab, eos } = testVocab();
    const M = m.compileSchema(PERSON);
    const masker = m.cachedMasker(M, vocab, eos);
    T.ok(masker.stats && typeof masker.stats.calls === 'number', 'the masker function carries a `stats` object { calls, misses, tokenScans }');
    const s1 = m.advance(M, M.start, '{"name":"a');
    const s1again = m.advance(M, M.start, '{"name":"b'); // a different prefix that reaches an equal state
    masker(s1); masker(s1again); masker(M.start);
    T.eq(masker.stats.calls, 3, 'three lookups');
    T.eq(masker.stats.misses, 2, '"{\\"name\\":\\"a" and "{\\"name\\":\\"b" reach equal states (one character into the name string): the second lookup must hit the cache. Key on the state\'s value, e.g. JSON.stringify(state), not on object identity');
    T.eq(masker.stats.tokenScans, 2 * vocab.length, 'each miss scans the whole vocabulary once; hits scan nothing');
    const model = randomModel(T, 31, vocab.length);
    const next = T.rng(32);
    for (let i = 0; i < 20; i++) {
      const r = m.constrainedGenerate(model, M, vocab, { eos, next, masker });
      T.ok(r.finished, 'generation with the cached masker must still finish with valid JSON');
    }
    const { calls, misses } = masker.stats;
    T.ok(misses < calls / 5, `over 20 generations fewer than 20% of lookups should miss (the state space is small); got ${misses} misses of ${calls} calls`);
  } },
  { step: 'cache', name: 'a cache hit does no work: the machine is not stepped again', run(m, T) {
    const { vocab, eos } = testVocab();
    const M = m.compileSchema(PERSON);
    let steps = 0;
    const counting = { start: M.start, step: (s, ch) => { steps++; return M.step(s, ch); }, accepts: (s) => M.accepts(s) };
    const masker = m.cachedMasker(counting, vocab, eos);
    const s1 = m.advance(M, M.start, '{"name":"ab","age":4');
    const s2 = m.advance(M, M.start, '{"name":"xy","age":7'); // a different prefix, an equal state
    const first = masker(s1);
    T.ok(steps > 0, 'the first lookup of a state is a miss and must compute the mask by simulating tokens');
    const before = steps;
    const again = masker(s2);
    T.eq(steps, before, `the second lookup reaches an equal state and must be answered from the cache without stepping the machine; it made ${steps - before} step calls`);
    T.eq(Array.from(again), Array.from(first), 'a hit returns the stored mask');
  } },
];
