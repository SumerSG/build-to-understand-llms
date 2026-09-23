import { BPETokenizer } from 'lib/tokenizer.js';
import { GPT } from 'lib/gpt.js';
import { Tensor, crossEntropy } from 'lib/tensor.js';
import { AdamW, clipGradNorm } from 'lib/optim.js';
import { CHAT } from 'lib/data.js';
import { randInt } from 'lib/util.js';

const FIXTURE_TEXT = 'say hello to the cat and the dog. hello! the cat sat on the mat, the dog ran to the barn. one two three. name a color: red or blue.';
const MARKERS = [CHAT.system, CHAT.user, CHAT.assistant, CHAT.end];

/** A small BPE tokenizer with only <|endoftext|> as a special, like the lab checkpoint's. */
function baseTokenizer() {
  return BPETokenizer.train(FIXTURE_TEXT, { vocabSize: 48 });
}

/** A chat-ready tokenizer built the reference way (so later steps do not depend on step 1 being done). */
function chatTokenizer() {
  const base = baseTokenizer();
  return new BPETokenizer({ vocab: [...base.vocab, ...MARKERS], merges: base.merges, specials: [...base.specials, ...MARKERS] });
}

function tinyModel(vocabSize, seed = 3) {
  return new GPT({ vocabSize, blockSize: 16, nLayer: 1, nHead: 2, nEmbd: 16, seed });
}

/** Reference example builder, so steps 4–6 can be tested without step 2. */
function refExample(tokenizer, prompt, response) {
  const promptIds = tokenizer.encode(CHAT.user + prompt + CHAT.end + CHAT.assistant);
  const responseIds = tokenizer.encode(response + CHAT.end);
  return { ids: promptIds.concat(responseIds), mask: promptIds.map(() => 0).concat(responseIds.map(() => 1)) };
}

function randomLogits(shape, next) {
  const n = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = (next() * 2 - 1) * 3;
  return new Tensor({ shape, data }, { requiresGrad: true });
}

function sum(arr) { let s = 0; for (const v of arr) s += v; return s; }

/** Reference masked cross-entropy, so the finetune test can replay the run the instructions describe. */
function refMaskedCE(logits, y, mask) {
  const V = logits.shape[logits.shape.length - 1];
  const targets = y.flat(Infinity), weights = mask.flat(Infinity);
  const pick = new Float32Array(targets.length * V);
  for (let i = 0; i < targets.length; i++) pick[i * V + targets[i]] = weights[i];
  return logits.logSoftmax().mul(new Tensor({ shape: logits.shape.slice(), data: pick })).sum().scale(-1 / sum(weights));
}

/** An optimizer stand-in that records what the gradients look like at the moment step() runs. */
function spyOptimizer(params) {
  return {
    t: 0, calls: [],
    step() {
      let ss = 0, hasGrad = true;
      for (const p of params) { if (!p.grad) { hasGrad = false; continue; } for (let i = 0; i < p.grad.length; i++) ss += p.grad[i] * p.grad[i]; }
      this.calls.push(['step', Math.sqrt(ss), hasGrad]);
      this.t++;
    },
    zeroGrad() { this.calls.push(['zeroGrad']); for (const p of params) p.zeroGrad(); },
  };
}

export const tests = [
  // ---------- step 1 ----------
  { step: 'template', name: 'formatChat renders markers and opens the assistant turn after a user message', run(m, T) {
    T.eq(m.formatChat([{ role: 'user', content: 'Say hello.' }]), '<|user|>Say hello.<|end|><|assistant|>',
      'a conversation that ends with a user message must end with the assistant marker so the model continues as the assistant');
    T.eq(m.formatChat([{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'Hi' }, { role: 'assistant', content: 'Hello!' }]),
      '<|system|>Be brief.<|end|><|user|>Hi<|end|><|assistant|>Hello!<|end|>',
      'every message is marker + content + end marker; a completed assistant turn gets no trailing marker');
    T.eq(m.formatChat([]), '', 'no messages renders to the empty string');
  } },
  { step: 'template', name: 'formatChat does not open a turn after an assistant message and treats unknown roles as user', run(m, T) {
    const out = m.formatChat([{ role: 'user', content: 'Hi' }, { role: 'assistant', content: 'Hello!' }]);
    T.ok(!out.endsWith(CHAT.assistant), 'after a finished assistant turn there is nothing for the model to continue, so no trailing assistant marker');
    T.eq(out, '<|user|>Hi<|end|><|assistant|>Hello!<|end|>');
    const multi = m.formatChat([{ role: 'user', content: 'Hi' }, { role: 'assistant', content: 'Hello!' }, { role: 'user', content: 'Name a color.' }]);
    T.eq(multi, '<|user|>Hi<|end|><|assistant|>Hello!<|end|><|user|>Name a color.<|end|><|assistant|>',
      'a second user turn re-opens the assistant turn: the template is the same for every turn of a multi-turn chat');
    T.eq(m.formatChat([{ role: 'tool', content: 'x' }]), '<|user|>x<|end|>', 'a role outside CHAT uses the user marker (the lib convention) but only a literal user role opens the assistant turn');
  } },
  { step: 'template', name: 'addChatTokens appends the four markers as single-id specials without touching the base vocabulary', run(m, T) {
    const base = baseTokenizer();
    const before = base.vocabSize;
    const tok = m.addChatTokens(base);
    T.eq(tok.vocabSize, before + 4, 'four new ids: system, user, assistant, end');
    T.eq(base.vocabSize, before, 'the base tokenizer must not be mutated');
    for (const marker of MARKERS) {
      const ids = tok.encode(marker);
      T.eq(ids.length, 1, `${marker} must encode to exactly one id (it is a special token, not text)`);
      T.ok(ids[0] >= before, `${marker} must get a NEW id at or above ${before}, not reuse an existing one`);
    }
    T.eq(tok.encode('<|user|>hello<|end|><|assistant|>'), [tok.encode('<|user|>')[0], ...base.encode('hello'), tok.encode('<|end|>')[0], tok.encode('<|assistant|>')[0]],
      'ordinary text between markers must tokenize exactly as the base tokenizer does');
    T.eq(tok.decode(tok.encode('<|user|>the cat<|end|>')), '<|user|>the cat<|end|>', 'markers must round-trip through decode');
  } },
  { step: 'template', name: 'addChatTokens keeps eos and existing ids, and is idempotent', run(m, T) {
    const base = baseTokenizer();
    const tok = m.addChatTokens(base);
    T.eq(tok.vocabSize, base.vocabSize + 4, 'the returned tokenizer must have the four markers added (returning the input unchanged is not enough)');
    T.eq(tok.eos, base.eos, '<|endoftext|> must keep its id: the checkpoint was trained with it');
    T.eq(tok.encode('the cat sat on the mat'), base.encode('the cat sat on the mat'), 'existing ids must not shift, or the checkpoint\'s embeddings would point at the wrong tokens');
    T.eq(m.addChatTokens(tok).vocabSize, tok.vocabSize, 'adding the markers twice must not add them again');
  } },

  // ---------- step 2 ----------
  { step: 'example', name: 'tokenizeExample: ids are the rendered chat plus response and end marker, mask is 1 only on assistant tokens', run(m, T) {
    const tok = chatTokenizer();
    const { ids, mask } = m.tokenizeExample(tok, 'say hello.', 'hello!');
    const expectIds = tok.encode(CHAT.user + 'say hello.' + CHAT.end + CHAT.assistant + 'hello!' + CHAT.end);
    T.eq(ids, expectIds, 'ids must be the tokenization of <|user|>prompt<|end|><|assistant|>response<|end|>');
    T.eq(mask.length, ids.length, 'one mask entry per token');
    const assistantId = tok.encode(CHAT.assistant)[0];
    const at = ids.indexOf(assistantId);
    T.eq(mask.slice(0, at + 1), new Array(at + 1).fill(0), 'every prompt token, including the markers, must be masked OUT (0)');
    T.eq(mask.slice(at + 1), new Array(ids.length - at - 1).fill(1), 'every response token must be masked IN (1)');
    T.eq(sum(mask), tok.encode('hello!').length + 1, 'the end marker after the response counts: the model must learn to stop');
  } },
  { step: 'example', name: 'buildExample shifts by one: mask follows y, and the first target is predicted from the assistant marker', run(m, T) {
    const tok = chatTokenizer();
    const { ids } = m.tokenizeExample(tok, 'name a color.', 'red.');
    const { x, y, mask } = m.buildExample(tok, 'name a color.', 'red.');
    T.eq(x, ids.slice(0, -1), 'x is every token but the last');
    T.eq(y, ids.slice(1), 'y is every token but the first');
    T.eq(mask.length, y.length, 'the mask must align with y (the targets), not with the full token stream');
    const first = mask.indexOf(1);
    T.eq(x[first], tok.encode(CHAT.assistant)[0], 'the first masked-in position is where x is <|assistant|> and y is the first response token (off-by-one check)');
    T.eq(y[y.length - 1], tok.encode(CHAT.end)[0], 'the last target is the end marker');
    T.eq(mask[mask.length - 1], 1, 'the end marker target is masked in');
    T.eq(sum(mask), tok.encode('red.').length + 1);
  } },

  // ---------- step 3 ----------
  { step: 'masked-loss', name: 'maskedCrossEntropy equals the mean NLL over masked-in positions only (hand-computed)', run(m, T) {
    const next = T.rng(11);
    const logits = randomLogits([1, 4, 5], next);
    const y = [[1, 3, 0, 4]];
    const mask = [[0, 1, 1, 0]];
    const loss = m.maskedCrossEntropy(logits, y, mask);
    T.ok(loss instanceof Tensor, 'return a Tensor (a scalar with a graph) so backward() can run through it');
    T.eq(loss.data.length, 1, 'the loss must be a single number');
    // hand computation
    let expect = 0;
    for (const t of [1, 2]) {
      const row = Array.from(logits.data.subarray(t * 5, t * 5 + 5));
      const mx = Math.max(...row);
      const lse = mx + Math.log(row.reduce((s, v) => s + Math.exp(v - mx), 0));
      expect += lse - row[y[0][t]];
    }
    expect /= 2;
    T.close(loss.item(), expect, 1e-4, `loss = sum(mask * nll) / sum(mask) = ${expect.toFixed(4)}: divide by the number of masked-in positions (here 2), not by the number of positions B·T = 4 (what lib crossEntropy does, which gives half of it) and not by the batch size B = 1 (which gives twice it)`);
    const all = m.maskedCrossEntropy(logits, y, [[1, 1, 1, 1]]);
    T.close(all.item(), crossEntropy(logits, y).item(), 1e-4, 'with an all-ones mask it must equal the ordinary mean cross-entropy');
  } },
  { step: 'masked-loss', name: 'changing the logits at masked-out (prompt) positions does not change the loss', run(m, T) {
    const next = T.rng(12);
    const logits = randomLogits([2, 3, 6], next);
    const y = [[0, 2, 5], [3, 3, 1]];
    const mask = [[0, 0, 1], [0, 1, 1]];
    const before = m.maskedCrossEntropy(logits, y, mask).item();
    const changed = new Tensor({ shape: [2, 3, 6], data: Float32Array.from(logits.data) }, { requiresGrad: true });
    for (let i = 0; i < 6; i++) { changed.data[0 * 18 + 0 * 6 + i] += 5; changed.data[1 * 18 + 0 * 6 + i] -= 3; changed.data[0 * 18 + 1 * 6 + i] *= -2; }
    const after = m.maskedCrossEntropy(changed, y, mask).item();
    T.close(after, before, 1e-5, 'the loss must not depend on what the model predicts at prompt positions');
    for (let i = 0; i < 6; i++) changed.data[1 * 18 + 2 * 6 + i] += 1.5 * (i % 2 ? 1 : -1);
    T.ok(Math.abs(m.maskedCrossEntropy(changed, y, mask).item() - before) > 1e-3, 'but changing a masked-in position must change it');
  } },
  { step: 'masked-loss', name: 'backward: zero gradient at masked-out positions, (softmax - onehot) / count at masked-in ones', run(m, T) {
    const next = T.rng(13);
    const logits = randomLogits([1, 3, 4], next);
    const y = [[2, 0, 3]];
    const mask = [[1, 0, 1]];
    m.maskedCrossEntropy(logits, y, mask).backward();
    T.ok(logits.grad !== null, 'logits.grad must be filled by backward()');
    const g = logits.grad;
    for (let i = 0; i < 4; i++) T.close(g[4 + i], 0, 1e-7, 'position 1 is masked out: its gradient must be exactly zero');
    for (const t of [0, 2]) {
      const row = Array.from(logits.data.subarray(t * 4, t * 4 + 4));
      const mx = Math.max(...row);
      const e = row.map((v) => Math.exp(v - mx));
      const z = e.reduce((s, v) => s + v, 0);
      for (let i = 0; i < 4; i++) T.close(g[t * 4 + i], (e[i] / z - (i === y[0][t] ? 1 : 0)) / 2, 1e-4, 'masked-in gradient must be (softmax - onehot) divided by the number of masked-in positions');
    }
    T.throws(() => m.maskedCrossEntropy(logits, y, [[0, 0, 0]]), 'an all-zero mask has no positions to average over and must throw rather than return NaN');
    // V = 4 here. Without a range check, target 4 at position 0 would silently write into position 1's row of pick.
    T.throws(() => m.maskedCrossEntropy(logits, [[4, 0, 3]], [[1, 0, 1]]), 'a target of V = 4 has no logit to pick (valid targets are 0..V-1) and must throw; without the check pick[t * V + 4] lands in the NEXT position\'s row and the loss is silently wrong');
    T.throws(() => m.maskedCrossEntropy(logits, [[2, 0, -1]], [[1, 0, 1]]), 'a negative target has no logit to pick and must throw');
  } },

  // ---------- step 4 ----------
  { step: 'packing', name: 'packs examples in order with an eos separator, and every pack has blockSize positions', run(m, T) {
    const tok = chatTokenizer();
    const pairs = [['say hello.', 'hello!'], ['name a color.', 'red.'], ['say hello to the dog.', 'hello, dog!'], ['count.', 'one two three.']];
    const examples = pairs.map(([p, r]) => refExample(tok, p, r));
    const blockSize = 40;
    const packs = m.packExamples(examples, { blockSize, eos: tok.eos });
    T.ok(Array.isArray(packs) && packs.length >= 1, 'return an array of packs');
    for (const pack of packs) {
      T.eq(pack.x.length, blockSize, 'x must have exactly blockSize entries');
      T.eq(pack.y.length, blockSize, 'y must have exactly blockSize entries');
      T.eq(pack.mask.length, blockSize, 'mask must have exactly blockSize entries');
      for (let i = 0; i + 1 < blockSize; i++) T.eq(pack.x[i + 1], pack.y[i], 'y must be x shifted left by one (one token stream, sliced twice)');
    }
    // reconstruct each pack's token stream and check the examples appear in order, each followed by eos
    const streams = packs.map((p) => p.x.concat([p.y[p.y.length - 1]]));
    let pi = 0, pos = 0;
    for (const ex of examples) {
      if (pos + ex.ids.length + 1 > blockSize + 1) { pi++; pos = 0; }
      T.ok(pi < packs.length, 'greedy in-order packing: start a new pack only when the next example plus its separator does not fit');
      T.eq(streams[pi].slice(pos, pos + ex.ids.length), ex.ids, `example tokens must appear contiguous inside pack ${pi} at position ${pos}`);
      T.eq(streams[pi][pos + ex.ids.length], tok.eos, 'each example must be followed by the eos separator');
      pos += ex.ids.length + 1;
    }
    T.eq(packs.length, pi + 1, `expected ${pi + 1} packs for these examples at blockSize ${blockSize}; put as many examples as fit into each window`);
    const total = examples.reduce((s, ex) => s + sum(ex.mask), 0);
    T.eq(packs.reduce((s, p) => s + sum(p.mask), 0), total, 'the packed masks must select exactly the assistant tokens of every example, no more and no fewer');
  } },
  { step: 'packing', name: 'the mask never crosses an example boundary and padding is eos with mask 0', run(m, T) {
    const tok = chatTokenizer();
    const examples = [refExample(tok, 'say hello.', 'hello!'), refExample(tok, 'name a color.', 'blue.'), refExample(tok, 'count.', 'one two three.')];
    const userId = tok.encode(CHAT.user)[0];
    const packs = m.packExamples(examples, { blockSize: 32, eos: tok.eos });
    for (const pack of packs) {
      for (let i = 0; i < pack.y.length; i++) {
        if (pack.y[i] === tok.eos) T.eq(pack.mask[i], 0, 'predicting the eos separator (or padding) must be masked out');
        if (pack.y[i] === userId) T.eq(pack.mask[i], 0, 'predicting the next example\'s <|user|> marker from eos must be masked out: that is a prompt token');
        if (pack.x[i] === tok.eos) T.eq(pack.mask[i], 0, 'nothing predicted from an eos separator is an assistant token');
      }
      // tail padding
      let tail = pack.y.length;
      while (tail > 0 && pack.y[tail - 1] === tok.eos) tail--;
      for (let i = tail; i < pack.y.length; i++) T.eq(pack.mask[i], 0, 'padding positions must be masked out');
    }
    const last = packs[packs.length - 1];
    T.eq(last.y[last.y.length - 1], tok.eos, 'the final pack is padded with eos up to blockSize');
  } },
  { step: 'packing', name: 'fits exactly-full windows (off-by-one) and rejects an example longer than the window', run(m, T) {
    const eos = 99;
    const ex = (n) => ({ ids: Array.from({ length: n }, (_, i) => i + 1), mask: Array.from({ length: n }, (_, i) => (i > 1 ? 1 : 0)) });
    // 20 + 1 + 20 + 1 + 22 + 1 = 65 = blockSize + 1 tokens: all three fit in ONE window of 64 positions
    const one = m.packExamples([ex(20), ex(20), ex(22)], { blockSize: 64, eos });
    T.eq(one.length, 1, '20+1, 20+1 and 22+1 tokens are exactly the 65 tokens a 64-position window holds; they must share one pack');
    T.eq(one[0].x.length, 64);
    // one more token and the third example must move to a new pack
    const two = m.packExamples([ex(20), ex(20), ex(23)], { blockSize: 64, eos });
    T.eq(two.length, 2, 'with 23 tokens the third example no longer fits and must start a second pack');
    const alone = m.packExamples([ex(64)], { blockSize: 64, eos });
    T.eq(alone.length, 1, 'an example of exactly blockSize tokens plus its separator fills one window');
    T.throws(() => m.packExamples([ex(65)], { blockSize: 64, eos }), 'an example longer than blockSize cannot be trained on and must throw');
    T.eq(m.packExamples([], { blockSize: 64, eos }), [], 'no examples, no packs');
  } },

  // ---------- step 5 ----------
  { step: 'resize', name: 'resizeEmbeddings grows wte, keeps trained rows, and fills new rows with the mean of the old ones', run(m, T) {
    const model = tinyModel(20, 5);
    const originalWte = Float32Array.from(model.wte.weight.data);
    const grown = m.resizeEmbeddings(model, 24);
    T.eq(grown.config.vocabSize, 24, 'config.vocabSize must be the new size');
    T.eq(grown.config.blockSize, model.config.blockSize);
    T.eq(grown.config.nEmbd, model.config.nEmbd);
    T.shape(grown.wte.weight, [24, 16], 'wte must have one row per token of the new vocabulary');
    T.close(Array.from(grown.wte.weight.data.subarray(0, 20 * 16)), Array.from(originalWte), 1e-7, 'the first 20 rows are the trained embeddings and must be copied unchanged');
    const mean = new Array(16).fill(0);
    for (let i = 0; i < 20; i++) for (let j = 0; j < 16; j++) mean[j] += originalWte[i * 16 + j] / 20;
    for (let i = 20; i < 24; i++) T.close(Array.from(grown.wte.weight.data.subarray(i * 16, i * 16 + 16)), mean, 1e-5, `new row ${i} must be the column-wise mean of the trained rows (not zeros, not the fresh random init)`);
    T.eq(Array.from(model.wte.weight.data), Array.from(originalWte), 'the original model must not be modified');
    T.eq(model.config.vocabSize, 20, 'the input model must not change: model.config.vocabSize was 20. Build the new config as { ...model.config, vocabSize: newVocabSize } instead of editing model.config, which is the same object the base model still uses');
  } },
  { step: 'resize', name: 'every other parameter is copied, so logits for the old tokens are unchanged', run(m, T) {
    const model = tinyModel(20, 7);
    const grown = m.resizeEmbeddings(model, 23);
    const src = model.parameters(), dst = grown.parameters();
    T.eq(dst.length, src.length, 'same parameter list');
    for (let k = 1; k < src.length; k++) T.eq(Array.from(dst[k].data), Array.from(src[k].data), `parameter ${k} (not wte) must be copied exactly`);
    for (let k = 0; k < src.length; k++) T.ok(dst[k].data.buffer !== src[k].data.buffer, `parameter ${k} must be a COPY: sharing the Float32Array with the base model means fine-tuning the new model silently rewrites the checkpoint you started from`);
    const ids = [[1, 5, 9, 2, 17, 3]];
    const a = model.forward(ids), b = grown.forward(ids);
    T.shape(b, [1, 6, 23]);
    for (let t = 0; t < 6; t++) {
      T.close(Array.from(b.data.subarray(t * 23, t * 23 + 20)), Array.from(a.data.subarray(t * 20, t * 20 + 20)), 1e-4, 'the tied head scores against the same rows, so the first 20 logits per position must match the base model');
    }
    T.throws(() => m.resizeEmbeddings(model, 10), 'shrinking would delete trained rows and must throw');
  } },

  // ---------- step 6 ----------
  { step: 'finetune', name: 'sftStep runs forward, masked loss, backward, clip, step, zeroGrad and lowers the loss on the same batch', run(m, T) {
    const model = tinyModel(20, 9);
    const optimizer = new AdamW(model.parameters(), { lr: 1e-2 });
    const batch = { x: [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]], y: [[2, 3, 4, 5, 6, 7, 8, 19], [10, 11, 12, 13, 14, 15, 16, 19]], mask: [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]] };
    const r1 = m.sftStep(model, optimizer, batch, { maxGradNorm: 1.0 });
    T.ok(r1 && typeof r1.loss === 'number' && Number.isFinite(r1.loss), 'return { loss, gradNorm } with a finite loss number');
    T.ok(typeof r1.gradNorm === 'number' && r1.gradNorm > 0, 'gradNorm is the pre-clip norm from clipGradNorm');
    T.eq(optimizer.t, 1, 'exactly one optimizer step per call');
    for (const p of model.parameters()) T.ok(p.grad === null, 'gradients must be cleared after the step, or the next backward accumulates onto stale values');
    const r2 = m.sftStep(model, optimizer, batch);
    T.ok(r2.loss < r1.loss, `a step at lr 1e-2 must reduce the loss on the same batch (got ${r1.loss.toFixed(4)} then ${r2.loss.toFixed(4)})`);
  } },
  { step: 'finetune', name: 'sftStep clips BEFORE the optimizer step, honours maxGradNorm, and clears gradients after it', run(m, T) {
    const model = tinyModel(20, 17);
    const spy = spyOptimizer(model.parameters());
    const batch = { x: [[1, 2, 3, 4, 5, 6]], y: [[2, 3, 4, 5, 6, 7]], mask: [[0, 0, 1, 1, 1, 1]] };
    const r = m.sftStep(model, spy, batch, { maxGradNorm: 0.01 });
    const steps = spy.calls.filter((c) => c[0] === 'step');
    T.eq(steps.length, 1, 'call optimizer.step() exactly once');
    T.ok(steps[0][2], 'every parameter must have a gradient when optimizer.step() runs: backward first, then step');
    T.ok(r.gradNorm > 0.01, `the returned gradNorm is the norm BEFORE clipping (got ${r.gradNorm}); this batch's norm is well above 0.01`);
    T.ok(steps[0][1] <= 0.01 * (1 + 1e-3), `when optimizer.step() runs the gradient norm must already be clipped to maxGradNorm = 0.01 (it was ${steps[0][1].toFixed(4)}): clip after backward and BEFORE step, and pass maxGradNorm through`);
    T.eq(spy.calls[spy.calls.length - 1][0], 'zeroGrad', 'optimizer.zeroGrad() comes after optimizer.step()');
  } },
  { step: 'finetune', name: 'sftStep ignores masked-out targets: two batches that differ only there produce identical parameters', run(m, T) {
    const batchA = { x: [[1, 2, 3, 4, 5, 6]], y: [[2, 3, 4, 5, 6, 7]], mask: [[0, 0, 0, 1, 1, 1]] };
    const batchB = { x: [[1, 2, 3, 4, 5, 6]], y: [[13, 17, 11, 5, 6, 7]], mask: [[0, 0, 0, 1, 1, 1]] };
    const modelA = tinyModel(20, 21), modelB = tinyModel(20, 21);
    m.sftStep(modelA, new AdamW(modelA.parameters(), { lr: 1e-2 }), batchA);
    m.sftStep(modelB, new AdamW(modelB.parameters(), { lr: 1e-2 }), batchB);
    const pa = modelA.parameters(), pb = modelB.parameters();
    for (let k = 0; k < pa.length; k++) T.close(Array.from(pa[k].data), Array.from(pb[k].data), 1e-7, 'the update must not depend on prompt-position targets: use the masked loss, not lib crossEntropy');
    const batchC = { x: [[1, 2, 3, 4, 5, 6]], y: [[2, 3, 4, 8, 6, 7]], mask: [[0, 0, 0, 1, 1, 1]] };
    const modelC = tinyModel(20, 21);
    m.sftStep(modelC, new AdamW(modelC.parameters(), { lr: 1e-2 }), batchC);
    const pc = modelC.parameters();
    let differs = false;
    for (let k = 0; k < pa.length && !differs; k++) for (let i = 0; i < pa[k].data.length; i++) if (Math.abs(pa[k].data[i] - pc[k].data[i]) > 1e-7) { differs = true; break; }
    T.ok(differs, 'a different masked-in target must change the update');
  } },
  { step: 'finetune', name: 'finetune returns one loss per step, calls onStep, trains the loss down and is deterministic given the seed', async run(m, T) {
    const eos = 19;
    const packs = [
      { x: [1, 2, 3, 4, 5, 6, 7, 8], y: [2, 3, 4, 5, 6, 7, 8, eos], mask: [0, 0, 1, 1, 1, 1, 1, 0] },
      { x: [9, 10, 11, 12, 13, 14, 15, 16], y: [10, 11, 12, 13, 14, 15, 16, eos], mask: [0, 0, 0, 1, 1, 1, 1, 0] },
      { x: [3, 1, 4, 1, 5, 9, 2, 6], y: [1, 4, 1, 5, 9, 2, 6, eos], mask: [0, 1, 1, 1, 1, 1, 1, 0] },
    ];
    const seen = [];
    const model = tinyModel(20, 31);
    const losses = await m.finetune(model, packs, { steps: 25, lr: 1e-2, batchSize: 2, next: T.rng(4), onStep: (step, loss) => { seen.push([step, loss]); } });
    T.eq(losses.length, 25, 'one loss per step');
    T.eq(seen.length, 25, 'onStep must be called once per step');
    T.eq(seen.map((s) => s[0]), losses.map((_, i) => i), 'onStep receives the step index 0..steps-1');
    T.close(seen.map((s) => s[1]), losses, 1e-9, 'onStep receives the same loss that is returned');
    const first = (losses[0] + losses[1] + losses[2]) / 3, last = (losses[22] + losses[23] + losses[24]) / 3;
    T.ok(last < first * 0.7, `25 steps at lr 1e-2 on 3 packs must cut the masked loss well below its start (first ${first.toFixed(3)}, last ${last.toFixed(3)})`);
    const again = await m.finetune(tinyModel(20, 31), packs, { steps: 25, lr: 1e-2, batchSize: 2, next: T.rng(4) });
    T.close(again, losses, 1e-6, 'the same seed must give the same run: batches come from `next`, nothing else');
  } },
  { step: 'finetune', name: 'finetune replays the specified run: one AdamW for the whole run with the given lr, betas, weightDecay and batchSize', async run(m, T) {
    const eos = 19;
    const packs = [
      { x: [1, 2, 3, 4, 5, 6, 7, 8], y: [2, 3, 4, 5, 6, 7, 8, eos], mask: [0, 0, 1, 1, 1, 1, 1, 0] },
      { x: [9, 10, 11, 12, 13, 14, 15, 16], y: [10, 11, 12, 13, 14, 15, 16, eos], mask: [0, 0, 0, 1, 1, 1, 1, 0] },
      { x: [3, 1, 4, 1, 5, 9, 2, 6], y: [1, 4, 1, 5, 9, 2, 6, eos], mask: [0, 1, 1, 1, 1, 1, 1, 0] },
      { x: [7, 7, 8, 8, 9, 9, 1, 1], y: [7, 8, 8, 9, 9, 1, 1, eos], mask: [0, 0, 0, 0, 1, 1, 1, 1] },
    ];
    const opts = { steps: 8, lr: 5e-3, batchSize: 3, weightDecay: 0.5, maxGradNorm: 0.5 };
    const got = await m.finetune(tinyModel(20, 41), packs, { ...opts, next: T.rng(8) });
    // The reference run, exactly as the instructions describe it.
    const ref = tinyModel(20, 41);
    const optimizer = new AdamW(ref.parameters(), { lr: opts.lr, betas: [0.9, 0.95], weightDecay: opts.weightDecay });
    const next = T.rng(8);
    const want = [];
    for (let step = 0; step < opts.steps; step++) {
      const x = [], y = [], mask = [];
      for (let b = 0; b < opts.batchSize; b++) { const p = packs[randInt(next, packs.length)]; x.push(p.x); y.push(p.y); mask.push(p.mask); }
      const loss = refMaskedCE(ref.forward(x), y, mask);
      loss.backward();
      clipGradNorm(ref.parameters(), opts.maxGradNorm);
      optimizer.step();
      optimizer.zeroGrad();
      want.push(loss.item());
    }
    T.eq(got.length, opts.steps, 'one loss per step');
    T.close(got, want, 1e-4, 'the losses must match the specified run: create ONE AdamW before the loop (its moments carry across steps) with betas [0.9, 0.95] and the given lr and weightDecay, draw sampleBatch(packs, batchSize, next) each step, and pass maxGradNorm to sftStep');
  } },
];
