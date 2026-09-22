import { loadModel, newCache, prefill, forwardStep } from 'lib/infer.js';
import { CharTokenizer, BPETokenizer } from 'lib/tokenizer.js';
import { GPT } from 'lib/gpt.js';

/** softmax([log p]) = p, so these logits have exactly these probabilities: 0.5, 0.3, 0.15, 0.05. */
const P4 = [0.5, 0.3, 0.15, 0.05];
const L4 = P4.map(Math.log);
const finiteIdx = (arr) => Array.from(arr).map((v, i) => (Number.isFinite(v) ? i : -1)).filter((i) => i >= 0);
const softmax = (l) => { const m = Math.max(...l); const e = l.map((v) => Math.exp(v - m)); const z = e.reduce((s, v) => s + v, 0); return e.map((v) => v / z); };
const randomLogits = (n, seed, scale = 3) => { const next = seed; return Array.from({ length: n }, () => (next() * 2 - 1) * scale); };

/** A tiny random-weight GPT plus a character tokenizer: enough to test the decode loop without a checkpoint. */
function tinySetup(T, text = 'abcdefgh', tokenizer = null) {
  const tok = tokenizer || new CharTokenizer(text);
  const model = loadModel(new GPT({ vocabSize: tok.vocabSize, blockSize: 32, nLayer: 1, nHead: 2, nEmbd: 8, seed: 1 }).toJSON());
  return { tok, model };
}

export const tests = [
  // ---------- step 1: temperature ----------
  { step: 'temperature', name: 'applyTemperature returns a fresh Float32Array and never mutates its input', run(m, T) {
    const logits = [2, 1, 0, -1];
    const out = m.applyTemperature(logits, 1);
    T.ok(out instanceof Float32Array, 'every processor returns a Float32Array so the next one can rely on it');
    T.close(out, [2, 1, 0, -1], 1e-6, 'temperature 1 must leave the logits unchanged');
    const typed = Float32Array.from(logits);
    const out2 = m.applyTemperature(typed, 0.5);
    T.ok(out2 !== typed, 'must return a new array, not the input');
    T.close(typed, [2, 1, 0, -1], 1e-6, 'the input must not be modified: the same logits are reused by other processors');
  } },
  { step: 'temperature', name: 't = 0 is greedy, t → ∞ is uniform, and t scales the logit gaps by 1/t', run(m, T) {
    const logits = [2, 1, 0, -1];
    T.close(m.applyTemperature(logits, 0.5), [4, 2, 0, -2], 1e-6, 'logits are DIVIDED by t: t = 0.5 doubles every gap (sharpens)');
    T.close(m.applyTemperature(logits, 2), [1, 0.5, 0, -0.5], 1e-6, 't = 2 halves every gap (flattens)');
    const cold = m.softmaxLogits(m.applyTemperature(logits, 0));
    T.ok(!Number.isNaN(cold[0]), 't = 0 must not divide by zero and produce NaN; take the limit (argmax) directly');
    T.close(cold, [1, 0, 0, 0], 1e-6, 'at t = 0 all the mass sits on the argmax');
    const hot = m.softmaxLogits(m.applyTemperature(logits, 1e6));
    T.close(hot, [0.25, 0.25, 0.25, 0.25], 1e-3, 'at t → ∞ every token is equally likely');
  } },
  { step: 'temperature', name: 'greedy returns the index of the largest logit (first on ties, negatives allowed)', run(m, T) {
    T.eq(m.greedy([1, 5, 5, 2]), 1, 'ties go to the earlier index, as argmax does everywhere in the lab');
    T.eq(m.greedy(Float32Array.from([-3, -1, -2])), 1, 'all-negative logits still have an argmax; do not start the search from 0');
    T.eq(m.greedy([0.1, 0.2, 0.3, 0.25]), 2);
  } },

  // ---------- step 2: top-k ----------
  { step: 'topk', name: 'keeps exactly the k largest logits unchanged and sets every other logit to -Infinity', run(m, T) {
    const logits = randomLogits(50, T.rng(2));
    const out = m.topKFilter(logits, 5);
    const kept = finiteIdx(out);
    T.eq(kept.length, 5, 'exactly k = 5 tokens survive');
    const top5 = logits.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 5).map((x) => x[1]).sort((a, b) => a - b);
    T.eq(kept, top5, 'the survivors are the 5 largest logits');
    for (const i of kept) T.close(out[i], logits[i], 1e-6, 'a kept logit keeps its value');
    for (let i = 0; i < 50; i++) if (!kept.includes(i)) T.ok(out[i] === -Infinity, `dropped token ${i} must be -Infinity, not ${out[i]}: a 0 would still get probability after softmax`);
  } },
  { step: 'topk', name: 'k ≤ 0 or k ≥ vocab means no filter; ties keep the earlier index; input untouched', run(m, T) {
    const logits = Float32Array.from([1, 3, 3, 2]);
    T.eq(finiteIdx(m.topKFilter(logits, 2)), [1, 2], 'k = 2 on [1, 3, 3, 2] keeps the two 3s');
    T.eq(finiteIdx(m.topKFilter(logits, 0)), [0, 1, 2, 3], 'k = 0 is "off": keep everything');
    T.eq(finiteIdx(m.topKFilter(logits, 4)), [0, 1, 2, 3], 'k equal to the vocabulary keeps everything');
    T.eq(finiteIdx(m.topKFilter(logits, 100)), [0, 1, 2, 3], 'k larger than the vocabulary keeps everything');
    T.close(logits, [1, 3, 3, 2], 1e-6, 'input must not be modified');
    T.close(m.softmaxLogits(m.topKFilter([2, 1, 0, -1], 2)), [0.73106, 0.26894, 0, 0], 1e-4, 'after filtering, softmax renormalises over the survivors');
  } },
  { step: 'topk', name: 'top-k keeps the same count on a peaked and on a flat distribution (the limitation top-p fixes)', run(m, T) {
    const peaked = [10, 0, 0, 0, 0, 0];
    const flat = [0, 0, 0, 0, 0, 0];
    T.eq(finiteIdx(m.topKFilter(peaked, 3)).length, 3, 'peaked: 3 survive even though two of them have probability ≈ 0');
    T.eq(finiteIdx(m.topKFilter(flat, 3)).length, 3, 'flat: 3 survive even though all six are equally plausible');
    T.eq(finiteIdx(m.topKFilter(flat, 3)), [0, 1, 2], 'on exact ties the earlier indices win');
  } },

  // ---------- step 3: top-p ----------
  { step: 'topp', name: 'keeps the smallest set of most-likely tokens whose mass reaches p, whatever their order', run(m, T) {
    T.eq(finiteIdx(m.topPFilter(L4, 0.9)), [0, 1, 2], 'probabilities 0.5, 0.3, 0.15, 0.05 with p = 0.9: 0.5 + 0.3 = 0.8 < 0.9, so the 0.15 token is needed too');
    T.eq(finiteIdx(m.topPFilter(L4, 0.79)), [0, 1], 'p = 0.79: 0.5 + 0.3 = 0.8 ≥ 0.79 already, so stop after two');
    T.eq(finiteIdx(m.topPFilter(L4, 0.81)), [0, 1, 2], 'p = 0.81: the token that crosses p is KEPT, not dropped (off-by-one)');
    const shuffled = [Math.log(0.15), Math.log(0.5), Math.log(0.05), Math.log(0.3)];
    T.eq(finiteIdx(m.topPFilter(shuffled, 0.9)), [0, 1, 3], 'sort by probability first: the order in the vocabulary is irrelevant');
    const out = m.topPFilter(shuffled, 0.9);
    T.close(out[1], Math.log(0.5), 1e-6, 'kept logits keep their values (softmax afterwards renormalises)');
    T.ok(out[2] === -Infinity, 'dropped tokens are -Infinity');
  } },
  { step: 'topp', name: 'always keeps at least one token; p ≥ 1 keeps everything; input untouched', run(m, T) {
    const flat = Float32Array.from([0, 0, 0, 0, 0]);
    T.eq(finiteIdx(m.topPFilter(flat, 0.01)), [0], 'a tiny p must still keep the single most likely token, or there is nothing to sample');
    T.eq(finiteIdx(m.topPFilter(flat, 1)), [0, 1, 2, 3, 4], 'p = 1 is "off"');
    T.close(flat, [0, 0, 0, 0, 0], 1e-6, 'input must not be modified');
    T.ok(m.topPFilter(L4, 0.9) instanceof Float32Array, 'returns a Float32Array');
  } },
  { step: 'topp', name: 'adapts to the shape: flat keeps many, peaked keeps one, and the same p keeps fewer after cooling', run(m, T) {
    const flat = new Array(100).fill(0);
    const n = finiteIdx(m.topPFilter(flat, 0.9)).length;
    T.ok(n === 90 || n === 91, `100 equally likely tokens at p = 0.9 keep about 90 (got ${n}); top-k would keep k no matter what`);
    const peaked = [10, ...new Array(99).fill(0)];
    T.eq(finiteIdx(m.topPFilter(peaked, 0.9)), [0], 'one token with 99.9% of the mass is the whole nucleus');
    const logits = [2, 1, 0, -1];
    T.eq(finiteIdx(m.topPFilter(logits, 0.9)).length, 3, 'at temperature 1, [2, 1, 0, -1] needs three tokens to reach 0.9');
    T.eq(finiteIdx(m.topPFilter(m.applyTemperature(logits, 0.3), 0.9)).length, 1, 'after temperature 0.3 the same logits need one: top-p measures the distribution it is given');
  } },

  // ---------- step 4: min-p and the penalties ----------
  { step: 'penalties', name: 'minPFilter drops tokens below p × (max probability) and always keeps the argmax', run(m, T) {
    T.eq(finiteIdx(m.minPFilter(L4, 0.2)), [0, 1, 2], 'max prob 0.5 → floor 0.1: 0.05 is dropped, 0.15 survives (an ABSOLUTE threshold of 0.2 would wrongly drop it)');
    T.eq(finiteIdx(m.minPFilter(L4, 0.5)), [0, 1], 'floor 0.25: only 0.5 and 0.3 survive');
    T.eq(finiteIdx(m.minPFilter(L4, 0.99)), [0], 'the argmax always passes its own floor');
    T.eq(finiteIdx(m.minPFilter(L4, 0)), [0, 1, 2, 3], 'p = 0 is "off"');
    const flat = [0, 0, 0, 0];
    T.eq(finiteIdx(m.minPFilter(flat, 0.9)), [0, 1, 2, 3], 'on a flat distribution every token is within p of the max, so all survive');
    const out = m.minPFilter(L4, 0.2);
    T.close(out[1], L4[1], 1e-6, 'kept logits keep their values');
  } },
  { step: 'penalties', name: 'repetitionPenalty divides positive logits and multiplies negative ones, for seen ids only', run(m, T) {
    const logits = Float32Array.from([2, -2, 1, 0.5]);
    T.close(m.repetitionPenalty(logits, [0, 1, 0], 2), [1, -4, 1, 0.5], 1e-6, 'CTRL rule: 2 → 2/2 = 1 and -2 → -2·2 = -4; unseen ids untouched (a plain subtraction or a plain multiply is wrong on one sign)');
    T.close(m.repetitionPenalty(logits, [0, 1], 1), [2, -2, 1, 0.5], 1e-6, 'penalty 1 is "off"');
    T.close(m.repetitionPenalty(logits, [], 2), [2, -2, 1, 0.5], 1e-6, 'nothing seen, nothing penalised');
    T.close(m.repetitionPenalty(logits, [99, -1], 2), [2, -2, 1, 0.5], 1e-6, 'ids outside the vocabulary are ignored');
    T.close(logits, [2, -2, 1, 0.5], 1e-6, 'input must not be modified');
  } },
  { step: 'penalties', name: 'frequencyPresencePenalty subtracts frequency × count plus presence once per seen id', run(m, T) {
    const logits = [1, 1, 1, 1];
    T.close(m.frequencyPresencePenalty(logits, [0, 0, 0, 2], { frequency: 0.5, presence: 1 }), [-1.5, 1, -0.5, 1], 1e-6,
      'id 0 seen 3×: 1 − 0.5·3 − 1 = −1.5; id 2 seen once: 1 − 0.5 − 1 = −0.5 (presence is paid ONCE, not per occurrence)');
    T.close(m.frequencyPresencePenalty(logits, [0, 0, 0, 2], { frequency: 1 }), [-2, 1, 0, 1], 1e-6, 'frequency alone scales with the count');
    T.close(m.frequencyPresencePenalty(logits, [0, 0, 0, 2], { presence: 2 }), [-1, 1, -1, 1], 1e-6, 'presence alone is a flat cost for having appeared');
    T.close(m.frequencyPresencePenalty(logits, [0, 0], {}), [1, 1, 1, 1], 1e-6, 'both zero is "off"');
    T.eq(logits, [1, 1, 1, 1], 'input must not be modified');
  } },

  // ---------- step 5: the pipeline and the sampler ----------
  { step: 'sample', name: 'processLogits applies penalties, then temperature, then top-k, top-p, min-p — in that order', run(m, T) {
    T.close(m.processLogits([2, 1, 0, -1]), [2, 1, 0, -1], 1e-6, 'with default options nothing changes');
    const l = [3, 2.5, 2.4, 0];
    T.eq(finiteIdx(m.processLogits(l, { prevIds: [0], repetitionPenalty: 2, topK: 2 })), [1, 2],
      'penalty first: 3/2 = 1.5 pushes token 0 out of the top-2. Truncating first would keep {0, 1} and then penalise inside the set');
    T.eq(finiteIdx(m.processLogits([2, 1, 0, -1], { temperature: 0.3, topP: 0.9 })), [0],
      'temperature before top-p: at t = 0.3 token 0 alone holds > 0.9 of the mass. Applying top-p on the raw logits would keep three tokens');
    T.eq(finiteIdx(m.processLogits([2, 1, 0, -1], { topK: 3, minP: 0.3 })), [0, 1],
      'top-k then min-p: after top-3 the probabilities are 0.665, 0.245, 0.090; floor 0.3 × 0.665 = 0.2 drops token 2 and keeps token 1');
    T.close(m.processLogits([2, 1, 0, -1], { temperature: 2, topK: 3 }), [1, 0.5, 0, -Infinity], 1e-6, 'the survivors carry the temperature-scaled logits');
  } },
  { step: 'sample', name: 'sample draws exactly one uniform per call, is reproducible under a seed, and is greedy at t = 0', run(m, T) {
    const logits = randomLogits(16, T.rng(4));
    let calls = 0;
    const counted = () => { calls++; return T.rng(9)(); };
    for (let i = 0; i < 10; i++) m.sample(logits, { next: counted });
    T.eq(calls, 10, 'one rng draw per sampled token: any other count breaks "same seed, same output" between implementations');
    const a = T.rng(11), b = T.rng(11);
    const seqA = Array.from({ length: 40 }, () => m.sample(logits, { temperature: 1.3, topK: 8, next: a }));
    const seqB = Array.from({ length: 40 }, () => m.sample(logits, { temperature: 1.3, topK: 8, next: b }));
    T.eq(seqA, seqB, 'the same seed must give the same tokens');
    T.ok(seqA.every((id) => Number.isInteger(id) && id >= 0 && id < 16), 'sample returns an integer token id in [0, vocab)');
    T.ok(new Set(seqA).size > 1, 'with topK 8 and t = 1.3 more than one token should appear in 40 draws');
    const best = logits.indexOf(Math.max(...logits));
    const cold = T.rng(5);
    for (let i = 0; i < 20; i++) T.eq(m.sample(logits, { temperature: 0, next: cold }), best, 'temperature 0 always returns the argmax');
  } },
  { step: 'sample', name: '20,000 draws match the processed distribution and never touch a filtered-out token', run(m, T) {
    const logits = randomLogits(8, T.rng(6), 2);
    const opts = { temperature: 0.7, topK: 4 };
    const expected = softmax(Array.from(m.processLogits(logits, opts)).map((v) => (v === -Infinity ? -1e9 : v)));
    const kept = finiteIdx(m.processLogits(logits, opts));
    const next = T.rng(7);
    const counts = new Array(8).fill(0);
    const N = 20000;
    for (let i = 0; i < N; i++) counts[m.sample(logits, { ...opts, next })]++;
    for (let i = 0; i < 8; i++) {
      if (!kept.includes(i)) T.eq(counts[i], 0, `token ${i} was filtered out by top-k but still sampled ${counts[i]} times: sample from the PROCESSED logits`);
    }
    const freq = counts.map((c) => c / N);
    T.close(freq, expected, 0.02, 'empirical frequencies must match softmax(processed logits) to within 2 points over 20,000 draws (missing temperature or a missing renormalisation shows up here)');
  } },

  // ---------- step 6: generate ----------
  { step: 'generate', name: 'generate = prefill, then repeat { sample → feed back }: identical to the reference loop with YOUR sample', run(m, T) {
    const { tok, model } = tinySetup(T);
    const opts = { maxNewTokens: 20, temperature: 1.2, topK: 6, repetitionPenalty: 1.5 };
    const out = m.generate(model, tok, 'ab', { ...opts, next: T.rng(3) });
    T.ok(out && Array.isArray(out.ids) && typeof out.text === 'string', 'generate returns { text, ids, finishReason }');
    T.eq(out.finishReason, 'length', 'no stop and no eos: it ran to maxNewTokens');
    T.eq(out.ids.length, 20, 'exactly maxNewTokens ids');
    T.eq(out.text, tok.decode(out.ids), 'text is the decoded generated ids (without the prompt)');
    // Reference loop using lib/infer.js and the learner's own sample().
    const promptIds = tok.encode('ab');
    const cache = newCache(model);
    let logits = prefill(model, cache, promptIds);
    const next = T.rng(3);
    const ref = [];
    for (let i = 0; i < 20; i++) {
      const id = m.sample(logits, { ...opts, prevIds: promptIds.concat(ref), next });
      ref.push(id);
      if (i < 19) logits = forwardStep(model, cache, id);
    }
    T.eq(out.ids, ref, 'each sampled token must be fed back through forwardStep before the next draw, with one rng draw per token and prevIds = prompt + generated so the repetition penalty sees the history');
    const again = m.generate(model, tok, 'ab', { ...opts, next: T.rng(3) });
    T.eq(again.ids, out.ids, 'same seed, same tokens');
    const other = m.generate(model, tok, 'ab', { ...opts, next: T.rng(4) });
    T.ok(other.ids.some((id, i) => id !== out.ids[i]), 'a different seed gives different tokens');
  } },
  { step: 'generate', name: 'stop sequences end generation, are cut from the text, and may straddle token boundaries', run(m, T) {
    const { tok, model } = tinySetup(T);
    const out = m.generate(model, tok, 'ab', { maxNewTokens: 40, temperature: 1.5, next: T.rng(1), stop: ['c'] });
    T.eq(out.finishReason, 'stop', 'a stop sequence was hit, so finishReason is "stop"');
    T.ok(out.ids.length < 40, 'generation stopped early');
    T.ok(!out.text.includes('c'), `the stop sequence is not part of the returned text (got "${out.text}")`);
    const full = tok.decode(out.ids);
    T.ok(full.includes('c') && full.startsWith(out.text), 'ids hold everything that was sampled; text is cut at the stop sequence');
    const multi = m.generate(model, tok, 'ab', { maxNewTokens: 40, temperature: 1.5, next: T.rng(1), stop: ['zz', 'hh'] });
    T.eq(multi.finishReason, 'stop');
    T.ok(!multi.text.includes('hh') && tok.decode(multi.ids).includes('hh'), 'a two-character stop is found in the decoded text even though each character is its own token');
    const none = m.generate(model, tok, 'ab', { maxNewTokens: 15, temperature: 1.5, next: T.rng(1), stop: ['zzz'] });
    T.eq(none.finishReason, 'length', 'a stop sequence that never appears does not stop anything');
    T.eq(none.ids.length, 15);
  } },
  { step: 'generate', name: 'eos ends generation when stopAtEos is on; maxNewTokens and the context window are hard limits', run(m, T) {
    const bpe = BPETokenizer.train('ab ab ba ab ba ab ab', { vocabSize: 8 });
    const { model } = tinySetup(T, '', bpe);
    T.ok(bpe.eos >= 0, 'the test tokenizer has an eos id');
    const out = m.generate(model, bpe, 'ab', { maxNewTokens: 40, temperature: 2, next: T.rng(1) });
    T.eq(out.finishReason, 'eos', 'the model sampled <|endoftext|>, so generation ended with finishReason "eos"');
    T.ok(!out.ids.includes(bpe.eos), 'eos is not part of the output ids');
    const through = m.generate(model, bpe, 'ab', { maxNewTokens: 40, temperature: 2, next: T.rng(1), stopAtEos: false });
    T.ok(through.ids.length > out.ids.length && through.ids.includes(bpe.eos), 'with stopAtEos: false the same seed runs past eos and keeps it as an ordinary token');
    const { tok, model: charModel } = tinySetup(T);
    const short = m.generate(charModel, tok, 'ab', { maxNewTokens: 5, next: T.rng(2) });
    T.ok(short.ids.length <= 5, 'maxNewTokens caps the output');
    const long = m.generate(charModel, tok, 'ab', { maxNewTokens: 500, temperature: 1.5, next: T.rng(2), stop: ['zzz'] });
    T.ok(long.ids.length <= charModel.config.blockSize && long.finishReason === 'length', `the cache holds blockSize = ${charModel.config.blockSize} positions; past that there is no position embedding, so stop with "length" (got ${long.ids.length} ids)`);
    const longPrompt = m.generate(charModel, tok, 'abcdefgh'.repeat(6), { maxNewTokens: 3, next: T.rng(2) });
    T.eq(longPrompt.ids.length, 1, 'a 48-token prompt is cut to its newest 32 tokens instead of throwing; that fills the window, so exactly one more token fits');
    T.eq(longPrompt.finishReason, 'length');
  } },
];
