// Structured outputs & constrained decoding (reference solution).
//
// A "machine" is a character-level automaton: { start, step(state, ch), accepts(state) }.
//   start            the initial state (plain data: numbers, strings, nested plain objects)
//   step(state, ch)  the next state after reading one character, or null if ch can never lead to valid output
//   accepts(state)   true if the text read so far is a complete, valid value
// States are never mutated; step returns a fresh object. Because states are plain data,
// JSON.stringify(state) is a faithful key for caching (step 5).

import { sample } from 'lib/sampling.js';

// ---------- worked examples ----------

/** Literal machine: the text must spell exactly one of `options` (booleans, enums, object keys). */
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

/** Does the machine accept exactly this text? */
export function matches(machine, text) {
  const s = advance(machine, machine.start, text);
  return s !== null && machine.accepts(s);
}

// ---------- step 1: string and integer machines ----------

const ESCAPABLE = '"\\/bfnrt';
const HEX = /^[0-9a-fA-F]$/;

/**
 * A JSON string of at most `maxLength` characters (an escape counts as one character).
 * Phases: 'open' (want the opening quote), 'body', 'esc' (just read a backslash),
 * 'hex' (inside \uXXXX, `h` digits still to read), 'done' (closing quote read).
 */
export function stringMachine(maxLength = 16) {
  return {
    start: { p: 'open', n: 0, h: 0 },
    step(state, ch) {
      const { p, n, h } = state;
      if (p === 'open') return ch === '"' ? { p: 'body', n: 0, h: 0 } : null;
      if (p === 'body') {
        if (ch === '"') return { p: 'done', n, h: 0 };
        if (ch.charCodeAt(0) < 0x20) return null; // JSON forbids raw control characters in strings
        if (n >= maxLength) return null;            // full: only the closing quote is left
        if (ch === '\\') return { p: 'esc', n, h: 0 };
        return { p: 'body', n: n + 1, h: 0 };
      }
      if (p === 'esc') {
        if (ch === 'u') return { p: 'hex', n, h: 4 };
        return ESCAPABLE.includes(ch) ? { p: 'body', n: n + 1, h: 0 } : null;
      }
      if (p === 'hex') {
        if (!HEX.test(ch)) return null;
        return h === 1 ? { p: 'body', n: n + 1, h: 0 } : { p: 'hex', n, h: h - 1 };
      }
      return null; // 'done': nothing may follow inside this value
    },
    accepts(state) {
      return state.p === 'done';
    },
  };
}

/**
 * A JSON integer: -?(0|[1-9][0-9]*), with at most `maxDigits` digits.
 * Phases: 'start', 'minus' (read '-'), 'zero' (a lone 0, which nothing may follow), 'digits'.
 */
export function integerMachine(maxDigits = 9) {
  const isDigit = (ch) => ch >= '0' && ch <= '9';
  return {
    start: { p: 'start', d: 0 },
    step(state, ch) {
      const { p, d } = state;
      if (p === 'start' && ch === '-') return { p: 'minus', d: 0 };
      if (p === 'start' || p === 'minus') {
        if (ch === '0') return { p: 'zero', d: 1 };
        return isDigit(ch) ? { p: 'digits', d: 1 } : null;
      }
      if (p === 'digits' && isDigit(ch) && d < maxDigits) return { p: 'digits', d: d + 1 };
      return null;
    },
    accepts(state) {
      return state.p === 'zero' || state.p === 'digits';
    },
  };
}

// ---------- step 2: the object grammar ----------

/**
 * Compile { type: 'object', properties: { key: spec, ... }, required: [...] } into a machine for
 * compact JSON (no whitespace outside strings) with keys in declaration order. Keys not listed in
 * `required` may be skipped; required keys may not.
 *
 * State: { p, last, cur, text, comma, sub }
 *   p = 'open'  | 'key' (typing a key literal like "age":) | 'value' (inside property `cur`) | 'done'
 *   last  index of the last property written (-1 before any)
 *   text  the key literal typed so far; comma: true if we just read ','; sub: the value machine's state
 */
export function compileSchema(schema) {
  const keys = Object.keys(schema.properties ?? {});
  const required = new Set(schema.required ?? []);
  const machines = keys.map((k) => compileValue(schema.properties[k]));
  const literals = keys.map((k) => JSON.stringify(k) + ':');

  // Keys that may come next after property `last`: every later key up to and including the first required one.
  const candidates = (last) => {
    const out = [];
    for (let i = last + 1; i < keys.length; i++) {
      out.push(i);
      if (required.has(keys[i])) break;
    }
    return out;
  };
  // The object may close after `last` only if no required key remains.
  const canClose = (last) => {
    for (let i = last + 1; i < keys.length; i++) if (required.has(keys[i])) return false;
    return true;
  };

  return {
    start: { p: 'open', last: -1, cur: -1, text: '', comma: false, sub: null },
    step(state, ch) {
      const { p, last, cur, text, comma, sub } = state;
      if (p === 'open') {
        return ch === '{' ? { p: 'key', last: -1, cur: -1, text: '', comma: false, sub: null } : null;
      }
      if (p === 'key') {
        if (text === '' && ch === '}' && !comma && canClose(last)) return { ...state, p: 'done', sub: null };
        const typed = text + ch;
        for (const i of candidates(last)) {
          if (literals[i] === typed) return { p: 'value', last, cur: i, text: '', comma: false, sub: machines[i].start };
        }
        return candidates(last).some((i) => literals[i].startsWith(typed)) ? { ...state, text: typed } : null;
      }
      if (p === 'value') {
        const inner = machines[cur].step(sub, ch);
        if (inner !== null) return { ...state, sub: inner };
        if (!machines[cur].accepts(sub)) return null;
        if (ch === ',' && candidates(cur).length > 0) return { p: 'key', last: cur, cur: -1, text: '', comma: true, sub: null };
        if (ch === '}' && canClose(cur)) return { p: 'done', last: cur, cur: -1, text: '', comma: false, sub: null };
        return null;
      }
      return null;
    },
    accepts(state) {
      return state.p === 'done';
    },
  };
}

// ---------- step 3: the token mask ----------

/**
 * Which vocabulary tokens may come next? mask[id] = 1 if every character of vocab[id] keeps the machine
 * alive from `state`, else 0. The end-of-sequence token (id `eos`) is allowed exactly when the machine
 * accepts. Empty strings (special tokens with no text) are never allowed.
 */
export function maskForState(machine, state, vocab, eos) {
  const mask = new Uint8Array(vocab.length);
  for (let id = 0; id < vocab.length; id++) {
    if (id === eos) { mask[id] = machine.accepts(state) ? 1 : 0; continue; }
    const tok = vocab[id];
    if (!tok) continue;
    mask[id] = advance(machine, state, tok) === null ? 0 : 1;
  }
  return mask;
}

// ---------- step 4: constrained sampling ----------

/** Copy of `logits` with every disallowed token set to -Infinity. */
export function applyMask(logits, mask) {
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = mask[i] ? logits[i] : -Infinity;
  return out;
}

/**
 * Sample tokens from `model(ids) -> logits` so that the text can only ever be a prefix of something
 * the machine accepts, and stop when the model picks `eos` (allowed only once the machine accepts).
 * Returns { text, ids, finished }; finished is false only if maxTokens ran out first.
 */
export function constrainedGenerate(model, machine, vocab, { eos, next, temperature = 1, maxTokens = 256, masker = null } = {}) {
  const maskOf = masker ?? ((s) => maskForState(machine, s, vocab, eos));
  let state = machine.start;
  const ids = [];
  let text = '';
  for (let t = 0; t < maxTokens; t++) {
    const logits = model(ids);
    const id = sample(applyMask(logits, maskOf(state)), { temperature, next });
    if (id === eos) return { text, ids, finished: true };
    state = advance(machine, state, vocab[id]);
    if (state === null) throw new Error(`constrainedGenerate: sampled a masked token ${JSON.stringify(vocab[id])}`);
    ids.push(id);
    text += vocab[id];
  }
  return { text, ids, finished: false };
}

// ---------- step 5: caching masks ----------

/**
 * A mask function with a cache keyed by JSON.stringify(state). stats counts calls, cache misses and
 * tokenScans (tokens simulated character by character, i.e. vocab.length per miss).
 */
export function cachedMasker(machine, vocab, eos) {
  const cache = new Map();
  const stats = { calls: 0, misses: 0, tokenScans: 0 };
  const masker = (state) => {
    stats.calls++;
    const key = JSON.stringify(state);
    let mask = cache.get(key);
    if (mask === undefined) {
      stats.misses++;
      stats.tokenScans += vocab.length;
      mask = maskForState(machine, state, vocab, eos);
      cache.set(key, mask);
    }
    return mask;
  };
  masker.stats = stats;
  masker.cache = cache;
  return masker;
}
