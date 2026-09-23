// Goal demo for module 22: the trained checkpoint (lib/checkpoints/tiny-gpt.json) has never seen JSON.
// YOUR schema compiler, token mask, constrained sampler and mask cache make it emit schema-valid objects
// anyway; the same model sampled without the mask shows what the grammar is doing for it.

import { rng } from 'lib/util.js';
import { loadModel, newCache, prefill, forwardStep } from 'lib/infer.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { sample, softmaxLogits } from 'lib/sampling.js';

const SCHEMA = {
  type: 'object',
  properties: {
    name: { type: 'string', maxLength: 12 },
    age: { type: 'integer', maxDigits: 2 },
    member: { type: 'boolean' },
    tier: { enum: ['gold', 'silver', 'bronze'] },
  },
  required: ['name', 'age', 'member', 'tier'],
};
const FIELDS = Object.keys(SCHEMA.properties);

// Independent check on the parsed value (JSON.parse plus plain type tests): no automaton involved.
function validate(text) {
  let v;
  try { v = JSON.parse(text); } catch { return false; }
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return false;
  if (Object.keys(v).join() !== FIELDS.join()) return false;
  return typeof v.name === 'string' && v.name.length <= 12 && Number.isInteger(v.age)
    && typeof v.member === 'boolean' && SCHEMA.properties.tier.enum.includes(v.tier);
}

// Which part of the object is the NEXT token going to write, judged from the text so far?
function fieldOf(prefix) {
  if (/"name":("[^"]*)?$/.test(prefix)) return 'name';
  if (/"age":-?\d*$/.test(prefix)) return 'age';
  if (/"member":[a-z]*$/.test(prefix)) return 'member';
  if (/"tier":("[a-z]*)?$/.test(prefix)) return 'tier';
  return 'keys & punctuation';
}

export default async function demo(m, lab) {
  const json = (await import('lib/checkpoints/tiny-gpt.json', { with: { type: 'json' } })).default;
  const tokJson = (await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default;
  const model = loadModel(json);
  const tok = BPETokenizer.fromJSON(tokJson);
  const V = model.config.vocabSize;
  const { blockSize } = model.config;

  // ---------- the decoding vocabulary ----------
  // The lab tokenizer was trained on stories: it has no '{' or '}'. We append them as two extra ids the
  // model cannot score and never sees (they are skipped when feeding its KV cache). They get a very low
  // but finite logit (-1e4): wherever a real token is also legal (a '}' inside the name string, say) it
  // wins, and where the grammar forces a brace the brace is the only finite logit left, so it is drawn.
  // (-Infinity would not work: with every logit at -Infinity the sampler falls back to uniform.)
  const vocab = tok.vocab.map((s, i) => (i === tok.eos ? s : (tok.specials.includes(s) || s === '<|unk|>' ? '' : s)));
  const OPEN = vocab.length; // the id of '{'
  vocab.push('{', '}');
  const eos = tok.eos;

  const promptIds = tok.encode('The quiet farmer:\n');

  // A model function ids -> logits over the extended vocab, backed by an incremental KV cache.
  function checkpointModel(record) {
    let cache = newCache(model);
    let context = promptIds.slice();
    let logits = prefill(model, cache, context);
    let fed = 0;
    return (ids) => {
      for (; fed < ids.length; fed++) {
        const id = ids[fed];
        if (id >= V) continue; // structural tokens the model cannot see
        context.push(id);
        if (cache.length >= blockSize) { // window full: keep the newest half and re-prefill
          context = context.slice(-blockSize / 2);
          cache = newCache(model);
          logits = prefill(model, cache, context);
        } else {
          logits = forwardStep(model, cache, id);
        }
      }
      const out = new Float32Array(vocab.length).fill(-1e4);
      out.set(logits);
      if (record) record.push(out);
      return out;
    };
  }

  const machine = m.compileSchema(SCHEMA);
  const masker = m.cachedMasker(machine, vocab, eos);
  lab.check(typeof masker === 'function' && masker.stats, 'cachedMasker must return a function with a stats object');

  // ---------- 1. constrained generation with the checkpoint ----------
  const N = 20;
  const next = rng(22);
  const constrained = [];
  const massByField = Object.fromEntries([...FIELDS, 'keys & punctuation'].map((f) => [f, []]));
  const t0 = performance.now();
  for (let i = 0; i < N; i++) {
    const record = [];
    const r = m.constrainedGenerate(checkpointModel(record), machine, vocab, { eos, next, temperature: 1, maxTokens: 160, masker });
    constrained.push(r);
    // Replay: how much of the model's own probability did the mask keep at each step?
    let state = machine.start, prefix = '';
    for (let s = 0; s < record.length; s++) {
      const probs = softmaxLogits(record[s].subarray(0, V));
      const mask = masker(state);
      let kept = 0;
      for (let id = 0; id < V; id++) if (mask[id]) kept += probs[id];
      massByField[fieldOf(prefix)].push(kept);
      if (s < r.ids.length) { state = m.advance(machine, state, vocab[r.ids[s]]); prefix += vocab[r.ids[s]]; }
    }
    lab.progress((i + 1) / (2 * N), `constrained ${i + 1}/${N}`);
    await lab.tick();
  }
  const tConstrained = (performance.now() - t0) / 1000;
  const validC = constrained.filter((r) => r.finished && validate(r.text)).length;
  lab.check(validC === N, `every constrained output must parse and match the schema; ${N - validC} did not, e.g. ${JSON.stringify(constrained.find((r) => !validate(r.text))?.text)}`);

  lab.table({
    title: 'The checkpoint, constrained by your grammar (first 8 of 20; every one parses)',
    columns: ['#', 'output', 'tokens', 'JSON.parse + schema'],
    rows: constrained.slice(0, 8).map((r, i) => [i + 1, r.text, r.ids.length, validate(r.text) ? 'valid' : 'INVALID']),
  });

  // ---------- 2. the same model without the mask ----------
  const scaffold = '{"name":"';
  const unconstrained = [];
  const breaks = Object.fromEntries([...FIELDS, 'keys & punctuation', 'still a valid prefix'].map((f) => [f, 0]));
  for (let i = 0; i < N; i++) {
    const f = checkpointModel(null);
    const ids = [OPEN, ...tok.encode(scaffold.slice(1))];
    let text = scaffold;
    for (let s = 0; s < 40; s++) {
      const id = sample(f(ids).subarray(0, V), { temperature: 1, next }); // the model's own vocab: no braces
      if (id === eos) break;
      ids.push(id);
      text += vocab[id];
    }
    unconstrained.push(text);
    // Walk YOUR machine over the text to find the first character that breaks the schema.
    let state = machine.start, at = text.length;
    for (let c = 0; c < text.length; c++) {
      const nextState = machine.step(state, text[c]);
      if (nextState === null) { at = c; break; }
      state = nextState;
    }
    if (at === text.length) breaks['still a valid prefix']++;
    else breaks[fieldOf(text.slice(0, at))]++;
    lab.progress(0.5 + (i + 1) / (2 * N), `unconstrained ${i + 1}/${N}`);
    await lab.tick();
  }
  const validU = unconstrained.filter(validate).length;
  lab.table({
    title: 'The same checkpoint, prompted with {"name":" and sampled freely (first 5 of 20)',
    columns: ['#', 'output', 'JSON.parse + schema'],
    rows: unconstrained.slice(0, 5).map((t, i) => [i + 1, t, validate(t) ? 'valid' : 'invalid']),
  });

  // ---------- 3. a model with no opinions at all: random logits ----------
  const R = 100;
  const nextR = rng(7);
  const randomModel = () => { const l = new Float32Array(vocab.length); for (let j = 0; j < l.length; j++) l[j] = (nextR() * 2 - 1) * 3; return l; };
  let validRC = 0, validRU = 0;
  for (let i = 0; i < R; i++) {
    const r = m.constrainedGenerate(randomModel, machine, vocab, { eos, next: nextR, maxTokens: 200, masker });
    if (r.finished && validate(r.text)) validRC++;
    let text = '';
    for (let s = 0; s < 60; s++) { const id = sample(randomModel(), { next: nextR }); if (id === eos) break; text += vocab[id]; }
    if (validate(text)) validRU++;
    if (i % 10 === 9) await lab.tick();
  }

  lab.bar({
    title: 'Schema-valid outputs (%)',
    labels: ['checkpoint, constrained', 'checkpoint, free', 'random logits, constrained', 'random logits, free'],
    values: [100 * validC / N, 100 * validU / N, 100 * validRC / R, 100 * validRU / R],
  });

  const meanMass = Object.fromEntries(Object.entries(massByField).map(([k, v]) => [k, v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0]));
  lab.bar({
    title: 'Mean probability the checkpoint itself put on tokens the mask allowed',
    labels: Object.keys(meanMass),
    values: Object.values(meanMass).map((x) => +x.toFixed(3)),
  });
  lab.bar({
    title: 'Free sampling: where the output first left the schema (20 samples)',
    labels: Object.keys(breaks),
    values: Object.values(breaks),
  });

  const { calls, misses, tokenScans } = masker.stats;
  const avoided = (calls - misses) * vocab.length;
  lab.table({
    title: 'Your mask cache over all constrained runs',
    columns: ['mask lookups', 'cache misses (distinct states)', 'token scans done', 'token scans avoided', 'hit rate'],
    rows: [[calls, misses, tokenScans, avoided, `${(100 * (1 - misses / calls)).toFixed(1)}%`]],
  });

  const ages = constrained.map((r) => JSON.parse(r.text).age);
  const members = constrained.filter((r) => JSON.parse(r.text).member).length;
  const worst = Object.entries(meanMass).sort((a, b) => a[1] - b[1])[0];
  lab.done(`Your constrained sampler turned a story model that has never seen JSON into a schema-valid generator: **${validC}/${N}** checkpoint outputs and **${validRC}/${R}** random-logit outputs parse and match \`{name, age, member, tier}\`, against **${validU}/${N}** and **${validRU}/${R}** without the mask (${tConstrained.toFixed(1)} s for the ${N} constrained runs). The format is guaranteed, the content is not: ages ranged from ${Math.min(...ages)} to ${Math.max(...ages)} (${ages.filter((a) => a < 0).length}/${N} negative: the schema said integer, not age), ${members}/${N} objects said member=true, and in **${worst[0]}** the model put only **${(100 * worst[1]).toFixed(1)}%** of its probability on legal tokens, so there the grammar, not the model, chose. Your cache answered ${calls.toLocaleString()} mask lookups with ${misses} full vocabulary scans, avoiding ${avoided.toLocaleString()} token simulations (${(100 * (1 - misses / calls)).toFixed(1)}% hit rate).`);
}
