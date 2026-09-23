// Structured outputs & constrained decoding.
//
// A "machine" is a character-level automaton: { start, step(state, ch), accepts(state) }.
//   start            the initial state (plain data: numbers, strings, nested plain objects)
//   step(state, ch)  the next state after reading one character, or null if ch can never lead to valid output
//   accepts(state)   true if the text read so far is a complete, valid value
// States are never mutated; step returns a fresh object. Because states are plain data,
// JSON.stringify(state) is a faithful key for caching (step 5).

import { sample } from 'lib/sampling.js';

// ---------- worked examples (done for you; read them, they set the conventions) ----------

/**
 * Literal machine: the text must spell exactly one of `options`. Booleans are ['true', 'false'],
 * an enum is its values as JSON (['"gold"', '"silver"']), and step 2 reuses it for object keys.
 * The state is just the text typed so far; a character is allowed if some option still starts with it.
 */
export function literalMachine(options) {
  return {
    start: { text: '' },
    step(state, ch) {
      const text = state.text + ch;
      return options.some((o) => o.startsWith(text)) ? { text } : null;
    },
    accepts(state) {
      return options.includes(state.text);
    },
  };
}

/** Compile one schema node into a machine. */
export function compileValue(spec) {
  if (Array.isArray(spec.enum)) return literalMachine(spec.enum.map((v) => JSON.stringify(v)));
  if (spec.type === 'boolean') return literalMachine(['true', 'false']);
  if (spec.type === 'string') return stringMachine(spec.maxLength ?? 16);
  if (spec.type === 'integer') return integerMachine(spec.maxDigits ?? 9);
  if (spec.type === 'object') return compileSchema(spec);
  throw new Error(`compileValue: unsupported schema node ${JSON.stringify(spec)}`);
}

/** Feed every character of `text` through the machine from `state`; null as soon as one is rejected. */
export function advance(machine, state, text) {
  let s = state;
  for (const ch of text) {
    s = machine.step(s, ch);
    if (s === null) return null;
  }
  return s;
}

/** Does the machine accept exactly this text? matches(literalMachine(['true', 'false']), 'true') === true */
export function matches(machine, text) {
  const s = advance(machine, machine.start, text);
  return s !== null && machine.accepts(s);
}

// ---------- step 1: string and integer machines ----------

const ESCAPABLE = '"\\/bfnrt';   // the characters that may follow a backslash (besides 'u')
const HEX = /^[0-9a-fA-F]$/;

/**
 * A JSON string of at most `maxLength` characters (an escape such as \n or \u00e9 counts as one).
 * Phases: 'open' (want the opening quote), 'body', 'esc' (just read a backslash),
 * 'hex' (inside \uXXXX, `h` hex digits still to read), 'done' (closing quote read).
 * `n` is the number of characters in the string so far.
 */
export function stringMachine(maxLength = 16) {
  return {
    start: { p: 'open', n: 0, h: 0 },
    step(state, ch) {
      const { p, n, h } = state;
      if (p === 'open') return ch === '"' ? { p: 'body', n: 0, h: 0 } : null;
      // TODO: step 1 — the 'body', 'esc' and 'hex' phases.
      //   body: '"' closes; control characters (code < 0x20) are illegal; once n === maxLength only '"' is left;
      //         '\\' starts an escape; anything else is one more character.
      //   esc:  'u' starts four hex digits; one of ESCAPABLE is one character; anything else is illegal.
      //   hex:  a hex digit counts down h; the last one completes one character.
      return null; // 'done' (and, for now, everything else): nothing may follow
    },
    accepts(state) {
      return state.p === 'done';
    },
  };
}

/**
 * A JSON integer: -?(0|[1-9][0-9]*), with at most `maxDigits` digits.
 * Suggested phases: 'start', 'minus' (read '-'), 'zero' (a lone 0, which nothing may follow), 'digits'.
 */
export function integerMachine(maxDigits = 9) {
  return {
    start: { p: 'start', d: 0 },
    step(state, ch) {
      // TODO: step 1
      return null;
    },
    accepts(state) {
      // TODO: step 1
      return false;
    },
  };
}

// ---------- step 2: the object grammar ----------

/**
 * Compile { type: 'object', properties: { key: spec, ... }, required: [...] } into a machine for
 * compact JSON (no whitespace outside strings) with keys in declaration order. Keys not listed in
 * `required` may be skipped; required keys may not. No trailing commas, no extra keys.
 * Use compileValue for each property and literalMachine-style matching for the key text '"name":'.
 */
export function compileSchema(schema) {
  // TODO: step 2
  return {
    start: { p: 'open' },
    step(state, ch) { return null; },
    accepts(state) { return false; },
  };
}

// ---------- step 3: the token mask ----------

/**
 * Which vocabulary tokens may come next? Return a Uint8Array with mask[id] = 1 if every character of
 * vocab[id] keeps the machine alive from `state`, else 0. The end-of-sequence token (id `eos`) is allowed
 * exactly when the machine accepts. Empty strings (special tokens with no text) are never allowed.
 */
export function maskForState(machine, state, vocab, eos) {
  // TODO: step 3
  return new Uint8Array(vocab.length);
}

// ---------- step 4: constrained sampling ----------

/** Copy of `logits` (a new Float32Array) with every token whose mask entry is 0 set to -Infinity. */
export function applyMask(logits, mask) {
  // TODO: step 4
  return Float32Array.from(logits);
}

/**
 * Sample tokens from `model(ids) -> logits` so that the text is always a prefix of something the
 * machine accepts; stop when the sampler picks `eos` (allowed only once the machine accepts).
 * Draw with lib/sampling.js `sample(logits, { temperature, next })`. Use `masker(state)` for masks
 * when one is given, maskForState otherwise.
 * Returns { text, ids, finished }; finished is false only if maxTokens ran out first.
 */
export function constrainedGenerate(model, machine, vocab, { eos, next, temperature = 1, maxTokens = 256, masker = null } = {}) {
  // TODO: step 4
  return { text: '', ids: [], finished: false };
}

// ---------- step 5: caching masks ----------

/**
 * Return a function masker(state) -> mask that caches masks by JSON.stringify(state), with a property
 * masker.stats = { calls, misses, tokenScans } (tokenScans grows by vocab.length on every miss).
 */
export function cachedMasker(machine, vocab, eos) {
  // TODO: step 5
  const masker = (state) => maskForState(machine, state, vocab, eos);
  masker.stats = { calls: 0, misses: 0, tokenScans: 0 };
  return masker;
}
