import { BPETokenizer } from 'lib/tokenizer.js';
import { GPT } from 'lib/gpt.js';
import { loadModel, newCache, prefill, forwardStep, forward } from 'lib/infer.js';
import { sample, greedy } from 'lib/sampling.js';
import { rng } from 'lib/util.js';

const TEXT = [
  'the cat sat on the mat. the dog ran to the park. a bird sang in the tree.',
  'Say hello. Hello! Name a color. Red. What is two and two? It is four.',
  'Be brief. The king and the queen walked to the old mill by the river.',
  'It is 391 (or 17 times 23), and 10 is 2 times 5.',
].join('\n');

let baseTokenizer = null;
/** The chat tokenizer the tests use: a small BPE trained on TEXT, extended with the chat tokens. */
function testTokenizer(m) {
  if (!baseTokenizer) baseTokenizer = BPETokenizer.train(TEXT, { vocabSize: 70 });
  return m.chatTokenizer(baseTokenizer);
}

/**
 * A random-weight inference model. `boost` scales the position table and the qkv projections so the
 * logits change from position to position (a fresh GPT is initialised so small that its predictions
 * barely depend on the context, which would make cached-versus-fresh comparisons vacuous).
 */
function testModel(vocabSize, { blockSize = 64, seed = 5, boost = 8 } = {}) {
  const model = loadModel(new GPT({ vocabSize, blockSize, nLayer: 1, nHead: 2, nEmbd: 16, seed }).toJSON());
  for (const name of Object.keys(model.w)) {
    if (name === 'wpe.weight' || name === 'wte.weight' || name.endsWith('attn.qkv.weight')) {
      const d = model.w[name].data;
      for (let i = 0; i < d.length; i++) d[i] *= boost;
    }
  }
  return model;
}

function freshSession(model, ids) {
  const session = { cache: newCache(model), ids: ids.slice() };
  const logits = prefill(model, session.cache, ids);
  return { session, logits };
}

/** What the learner's decode loop must reproduce: sample, append, run through the cache — no stops. */
function referenceSamples(model, promptIds, n, opts) {
  const { session, logits: first } = freshSession(model, promptIds);
  let logits = first;
  const out = [];
  for (let i = 0; i < n; i++) {
    const id = sample(logits, opts);
    out.push(id);
    logits = forwardStep(model, session.cache, id);
  }
  return out;
}

function lastRow(raw) {
  const [T, V] = raw.shape;
  return Array.from(raw.data.subarray((T - 1) * V, T * V));
}

function sameCache(T, cache, model, ids, what) {
  const fresh = freshSession(model, ids).session.cache;
  T.eq(cache.length, ids.length, `${what}: cache.length must equal the number of ids it represents (${ids.length})`);
  for (let layer = 0; layer < fresh.k.length; layer++) {
    T.eq(cache.k[layer].shape, fresh.k[layer].shape, `${what}: layer ${layer} keys must have shape [H, ${ids.length}, dh]`);
    T.close(Array.from(cache.k[layer].data), Array.from(fresh.k[layer].data), 1e-4, `${what}: layer ${layer} keys must equal a fresh prefill of the same ids`);
    T.close(Array.from(cache.v[layer].data), Array.from(fresh.v[layer].data), 1e-4, `${what}: layer ${layer} values must equal a fresh prefill of the same ids`);
  }
}

const SYS = { role: 'system', content: 'Be brief.' };

export const tests = [
  // ---------- step 1: prompt ----------
  { step: 'prompt', name: 'a history that fits is rendered whole and the assistant turn is opened', run(m, T) {
    const tok = testTokenizer(m);
    const history = [SYS, { role: 'user', content: 'Say hello.' }];
    const before = JSON.stringify(history);
    const p = m.buildPrompt(tok, history, { budget: 200 });
    T.eq(p.ids, tok.encode(m.renderChat(history)), 'ids must be tokenizer.encode(renderChat(history)) when everything fits');
    T.eq(p.dropped, 0, 'nothing should be dropped when the prompt fits the budget');
    T.eq(p.ids[p.ids.length - 1], tok.encode(m.CHAT_TEMPLATE.assistant)[0], 'the prompt must end by opening the assistant turn, or the model does not know it is its turn to speak');
    T.eq(JSON.stringify(history), before, 'buildPrompt must not modify the history it was given');
    const q = m.buildPrompt(tok, history, { budget: 200, prefill: '<tool_call>calc(' });
    T.ok(q.text.endsWith(m.CHAT_TEMPLATE.assistant + '<tool_call>calc('), 'with a prefill the prompt must end with the assistant marker followed by the prefill text');
    T.eq(q.ids, tok.encode(q.text), 'ids must be the encoding of text');
  } },
  { step: 'prompt', name: 'over budget, the oldest whole turn is dropped and no more than needed', run(m, T) {
    const tok = testTokenizer(m);
    const u1 = { role: 'user', content: 'What is two and two?' };
    const a1 = { role: 'assistant', content: '<tool_call>calc(2+2)</tool_call>' };
    const t1 = { role: 'tool', name: 'calc', content: '4' };
    const a1b = { role: 'assistant', content: 'It is four.' };
    const u2 = { role: 'user', content: 'Name a color.' };
    const a2 = { role: 'assistant', content: 'Red.' };
    const u3 = { role: 'user', content: 'Say hello.' };
    const history = [SYS, u1, a1, t1, a1b, u2, a2, u3];
    const full = tok.encode(m.renderChat(history)).length;
    const withoutUserOnly = tok.encode(m.renderChat([SYS, a1, t1, a1b, u2, a2, u3])).length;
    const withoutTurn1 = tok.encode(m.renderChat([SYS, u2, a2, u3])).length;
    T.ok(withoutUserOnly < full && withoutTurn1 < withoutUserOnly, 'test setup');
    const p = m.buildPrompt(tok, history, { budget: withoutUserOnly });
    T.eq(p.messages.map((x) => x.role), ['system', 'user', 'assistant', 'user'], 'drop the oldest TURN (the user message and every assistant/tool message up to the next user message) as a unit: dropping only the user message leaves an assistant reply and a tool result with no question in front of them');
    T.eq(p.dropped, 4, 'dropped counts messages removed: turn 1 has 4 (user, tool call, tool result, reply)');
    T.ok(p.ids.length <= withoutUserOnly, `the prompt must fit the budget (${p.ids.length} > ${withoutUserOnly})`);
    const q = m.buildPrompt(tok, history, { budget: withoutTurn1 });
    T.eq(q.messages.length, 4, 'with a budget that fits after dropping one turn, exactly one turn must go: dropping more throws away context the model could have used');
    T.eq(q.ids.length, withoutTurn1, 'after dropping turn 1 the prompt costs exactly the rendering of [system, turn 2, turn 3]');
  } },
  { step: 'prompt', name: 'the system prompt and the current turn are never dropped; an impossible budget throws', run(m, T) {
    const tok = testTokenizer(m);
    const history = [SYS, { role: 'user', content: 'Say hello.' }, { role: 'assistant', content: 'Hello!' }, { role: 'user', content: 'Name a color.' }];
    const minimal = tok.encode(m.renderChat([SYS, history[3]])).length;
    const p = m.buildPrompt(tok, history, { budget: minimal });
    T.eq(p.messages.map((x) => x.role), ['system', 'user'], 'at the tightest workable budget only the system prompt and the current user message remain');
    T.eq(p.messages[1].content, 'Name a color.', 'the message kept must be the CURRENT (last) one, not the oldest');
    T.eq(p.ids[0], tok.encode(m.CHAT_TEMPLATE.system)[0], 'the system prompt must stay at the front: it is never dropped');
    T.throws(() => m.buildPrompt(tok, history, { budget: minimal - 1 }), 'when even the system prompt plus the current turn exceed the budget, throw (a real API returns "prompt too long") instead of silently cutting the question');
  } },
  { step: 'prompt', name: 'a prefill is part of the prompt and counts against the budget', run(m, T) {
    const tok = testTokenizer(m);
    const prefill = '<tool_call>calc(17*23)</tool_call>';
    const history = [SYS, { role: 'user', content: 'Say hello.' }, { role: 'assistant', content: 'Hello!' }, { role: 'user', content: 'What is 17*23?' }];
    const withPrefill = tok.encode(m.renderChat(history, prefill)).length;
    const withoutPrefill = tok.encode(m.renderChat(history)).length;
    const minimal = tok.encode(m.renderChat([SYS, history[3]], prefill)).length;
    T.ok(withoutPrefill < withPrefill - 1 && minimal <= withPrefill - 1, 'test setup');
    const p = m.buildPrompt(tok, history, { budget: withPrefill - 1, prefill });
    T.ok(p.ids.length <= withPrefill - 1, `with prefill ${JSON.stringify(prefill)} the prompt is ${p.ids.length} tokens, over the budget of ${withPrefill - 1}: measure the encoding of renderChat(messages, prefill), prefill included, since those tokens occupy cache positions too`);
    T.eq(p.messages.map((x) => x.role), ['system', 'user'], 'the prefill pushed the prompt one token over the budget, so the oldest turn must go');
    T.eq(p.dropped, 2, 'dropped counts the two messages of turn 1');
    T.ok(p.text.endsWith(prefill), 'the kept prompt still ends with the prefill');
  } },

  // ---------- step 2: prefix ----------
  { step: 'prefix', name: 'truncateCache keeps the first n positions and leaves a cache that still works', run(m, T) {
    const tok = testTokenizer(m);
    const model = testModel(tok.vocabSize);
    const ids = tok.encode('the cat sat on the mat.');
    const { session } = freshSession(model, ids);
    const out = m.truncateCache(session.cache, 4);
    T.ok(out === session.cache, 'truncateCache mutates and returns the same cache object');
    sameCache(T, session.cache, model, ids.slice(0, 4), 'after truncateCache(cache, 4)');
    const logits = forwardStep(model, session.cache, ids[4]);
    T.close(Array.from(logits), lastRow(forward(model, ids.slice(0, 5))), 1e-4, 'a truncated cache must accept the next token as if the removed positions had never been there');
  } },
  { step: 'prefix', name: 'syncCache on an empty session is a plain prefill', run(m, T) {
    const tok = testTokenizer(m);
    const model = testModel(tok.vocabSize);
    const ids = tok.encode('<|system|>Be brief.<|end|><|user|>Say hello.<|end|><|assistant|>');
    const session = { cache: newCache(model), ids: [] };
    const r = m.syncCache(model, session, ids);
    T.eq([r.reused, r.prefilled], [0, ids.length], 'nothing is cached yet: reused 0, prefilled every token');
    T.close(Array.from(r.logits), lastRow(forward(model, ids)), 1e-4, 'logits must be those after the last prompt token');
    T.eq(session.ids, ids, 'session.ids must record exactly the ids the cache now holds');
    T.ok(session.ids !== ids, 'store a copy of ids: the caller may reuse its array');
  } },
  { step: 'prefix', name: 'a longer prompt reuses the shared prefix and equals a fresh prefill', run(m, T) {
    const tok = testTokenizer(m);
    const model = testModel(tok.vocabSize);
    const a = tok.encode('<|system|>Be brief.<|end|><|user|>Say hello.<|end|><|assistant|>');
    const b = tok.encode('Hello!<|end|><|user|>Name a color.<|end|><|assistant|>');
    const session = { cache: newCache(model), ids: [] };
    m.syncCache(model, session, a);
    const r = m.syncCache(model, session, [...a, ...b]);
    T.eq([r.reused, r.prefilled], [a.length, b.length], 'the first turn is already in the cache: only the new suffix may be prefilled');
    T.close(Array.from(r.logits), lastRow(forward(model, [...a, ...b])), 1e-4, 'the cached path must give the same logits as recomputing everything');
    sameCache(T, session.cache, model, [...a, ...b], 'after extending');
  } },
  { step: 'prefix', name: 'a diverging prompt cuts the cache back to the shared prefix; an identical one still returns logits', run(m, T) {
    const tok = testTokenizer(m);
    const model = testModel(tok.vocabSize);
    const a = tok.encode('<|system|>Be brief.<|end|><|user|>');
    const b = tok.encode('Say hello.<|end|><|assistant|>');
    const c = tok.encode('Name a color.<|end|><|assistant|>');
    const session = { cache: newCache(model), ids: [] };
    m.syncCache(model, session, [...a, ...b]);
    const r = m.syncCache(model, session, [...a, ...c]);
    T.eq(r.reused, a.length, 'reused must be the longest common prefix of the cached ids and the new ids');
    sameCache(T, session.cache, model, [...a, ...c], 'after a diverging prompt (the stale suffix must be cut off before prefilling)');
    T.close(Array.from(r.logits), lastRow(forward(model, [...a, ...c])), 1e-4, 'logits after the diverging prompt');
    const again = m.syncCache(model, session, [...a, ...c]);
    T.eq([again.reused, again.prefilled], [a.length + c.length - 1, 1], 'an identical prompt is fully cached, but the last token must be run again because its logits are the output');
    T.close(Array.from(again.logits), lastRow(forward(model, [...a, ...c])), 1e-4, 'logits for an identical prompt');
  } },
  { step: 'prefix', name: 'the stored keys and values are actually reused, not recomputed', run(m, T) {
    const tok = testTokenizer(m);
    const model = testModel(tok.vocabSize);
    const a = tok.encode('<|system|>Be brief.<|end|><|user|>Say hello.');
    const b = tok.encode('<|end|><|assistant|>');
    const session = { cache: newCache(model), ids: [] };
    m.syncCache(model, session, a);
    // Poison what is stored. A pipeline that reuses the cache must now see different logits;
    // one that silently recomputes the prefix would not notice.
    for (const layer of session.cache.v) for (let i = 0; i < layer.data.length; i++) layer.data[i] *= 25;
    const r = m.syncCache(model, session, [...a, ...b]);
    const fresh = lastRow(forward(model, [...a, ...b]));
    let diff = 0;
    for (let i = 0; i < fresh.length; i++) diff = Math.max(diff, Math.abs(r.logits[i] - fresh[i]));
    T.ok(diff > 1e-3, 'syncCache recomputed the shared prefix: reuse the stored keys and values and prefill only ids.slice(reused), which is the entire point of keeping the cache across turns');
    T.eq(r.prefilled, b.length, 'only the new suffix may be prefilled');
  } },

  // ---------- step 3: decode ----------
  { step: 'decode', name: 'stops at the first stop token, which is not kept, with the cache in step', run(m, T) {
    const tok = testTokenizer(m);
    const model = testModel(tok.vocabSize);
    const prompt = tok.encode('<|user|>Say hello.<|end|><|assistant|>');
    const ref = referenceSamples(model, prompt, 12, { temperature: 1, next: rng(9) });
    let k = 3;
    while (ref.slice(0, k).includes(ref[k])) k++;
    const { session, logits } = freshSession(model, prompt);
    const out = m.decodeReply(model, tok, session, logits, { maxNewTokens: 12, stopIds: [ref[k]], next: rng(9) });
    T.eq(out.ids, ref.slice(0, k), `with stopIds [${ref[k]}] the reply must be the ${k} tokens sampled before it: sample with lib/sampling.js sample(logits, { temperature, topK, topP, next }), once per token`);
    T.eq(out.stopReason, 'stop', 'a stop token ends the reply with stopReason "stop"');
    T.eq(out.text, tok.decode(ref.slice(0, k)), 'text is the decoded ids, without the stop token');
    T.eq(session.ids, [...prompt, ...ref.slice(0, k)], 'every kept token is appended to session.ids; the stop token is not');
    T.eq(session.cache.length, session.ids.length, 'the cache must hold exactly session.ids: run every kept token through forwardStep');
  } },
  { step: 'decode', name: 'maxNewTokens ends with "length"; temperature 0 is greedy', run(m, T) {
    const tok = testTokenizer(m);
    const model = testModel(tok.vocabSize);
    const prompt = tok.encode('<|user|>Name a color.<|end|><|assistant|>');
    const ref = referenceSamples(model, prompt, 5, { temperature: 0.8, topK: 10, next: rng(4) });
    const { session, logits } = freshSession(model, prompt);
    const out = m.decodeReply(model, tok, session, logits, { maxNewTokens: 5, next: rng(4), temperature: 0.8, topK: 10 });
    T.eq(out.ids, ref, 'pass temperature, topK and topP through to sample(): the same seed must give the same tokens');
    T.eq(out.stopReason, 'length', 'running out of maxNewTokens ends the reply with "length"');
    sameCache(T, session.cache, model, [...prompt, ...ref], 'after 5 generated tokens');
    const g = freshSession(model, prompt);
    const greedyOut = m.decodeReply(model, tok, g.session, g.logits, { maxNewTokens: 4, next: rng(1), temperature: 0 });
    const expect = [];
    { const s = freshSession(model, prompt); let lg = s.logits; for (let i = 0; i < 4; i++) { const id = greedy(lg); expect.push(id); lg = forwardStep(model, s.session.cache, id); } }
    T.eq(greedyOut.ids, expect, 'temperature 0 must pick the argmax at every step');
    const nucleus = referenceSamples(model, prompt, 6, { temperature: 1.5, topP: 0.3, next: rng(12) });
    const t = freshSession(model, prompt);
    const topPOut = m.decodeReply(model, tok, t.session, t.logits, { maxNewTokens: 6, next: rng(12), temperature: 1.5, topP: 0.3 });
    T.eq(topPOut.ids, nucleus, 'pass topP through to sample() as well: with topP 0.3 the same seed must give the same tokens');
  } },
  { step: 'decode', name: 'a full context window ends the reply with "context" instead of throwing', run(m, T) {
    const tok = testTokenizer(m);
    const model = testModel(tok.vocabSize, { blockSize: 24 });
    const prompt = tok.encode('the cat sat on the mat. the dog ran to the park.').slice(0, 20);
    T.eq(prompt.length, 20, 'test setup');
    const { session, logits } = freshSession(model, prompt);
    const out = m.decodeReply(model, tok, session, logits, { maxNewTokens: 10, next: rng(2) });
    T.eq(out.ids.length, 4, 'a 24-token window with 20 prompt tokens has room for exactly 4 generated tokens: check cache.length against blockSize before each token');
    T.eq(out.stopReason, 'context', 'a full window ends the reply with "context": there is no position embedding left to use');
  } },

  // ---------- step 4: tools ----------
  { step: 'tools', name: 'findToolCalls extracts name and argument from complete calls only', run(m, T) {
    T.eq(m.findToolCalls('ok <tool_call>calc(17*23)</tool_call> done'), [{ name: 'calc', input: '17*23' }], 'one call: the name before "(" and the text between the parentheses');
    T.eq(m.findToolCalls('<tool_call> calc( (2+3)*4 ) </tool_call>'), [{ name: 'calc', input: '(2+3)*4' }], 'the argument runs from the FIRST "(" to the LAST ")" (it may contain parentheses), trimmed');
    T.eq(m.findToolCalls('<tool_call>calc(1+1)</tool_call>x<tool_call>clock()</tool_call>'), [{ name: 'calc', input: '1+1' }, { name: 'clock', input: '' }], 'every call, in order');
    T.eq(m.findToolCalls('<tool_call>calc 17</tool_call><tool_call>calc(2+2)'), [], 'skip a body without the name(...) shape, and stop at a block with no closing tag');
    T.eq(m.findToolCalls('The answer is 4.'), [], 'plain text contains no calls');
    T.eq(m.findToolCalls('<tool_call>calc(1+1) and then some</tool_call><tool_call>calc(3)</tool_call>'), [{ name: 'calc', input: '3' }], 'the body must END with ")": text after the call means it is not name(argument), so skip that block (anchor the pattern at both ends)');
  } },
  { step: 'tools', name: 'runToolCalls executes through the registry and returns tool messages', async run(m, T) {
    const tools = m.makeTools();
    T.eq(await m.runToolCalls('<tool_call>calc(17*23)</tool_call>', tools), [{ role: 'tool', name: 'calc', content: '391' }], 'one { role: "tool", name, content } message per call; call tools.call(name, { input })');
    const two = await m.runToolCalls('<tool_call>calc(1/0)</tool_call><tool_call>search(cats)</tool_call>', tools);
    T.eq(two.length, 2, 'two calls give two tool messages');
    T.ok(/^Error/.test(two[0].content), `a failing tool must come back as an "Error: ..." string the model can read, got ${JSON.stringify(two[0].content)}`);
    T.ok(/^Error/.test(two[1].content), 'an unknown tool must come back as an "Error: ..." string, not an exception that kills the chat');
    T.eq(await m.runToolCalls('no calls here', tools), [], 'no calls: no messages');
  } },

  // ---------- step 5: chat ----------
  { step: 'chat', name: 'two turns: the transcript grows, the second turn reuses the first, the cache stays exact', async run(m, T) {
    const tok = testTokenizer(m);
    const model = testModel(tok.vocabSize);
    const state = m.createChat({ model, tokenizer: tok, system: 'Be brief.' });
    const next = rng(3);
    const r1 = await m.chat(state, 'Say hello.', { maxNewTokens: 6, next });
    T.eq(state.history.map((x) => x.role), ['system', 'user', 'assistant'], 'chat appends the user message and the assistant reply to state.history');
    T.eq(r1.reply, state.history[2].content, 'reply is the content of the assistant message');
    T.ok(r1.stats && Array.isArray(r1.stats.rounds) && r1.stats.rounds.length === 1, 'stats.rounds has one entry per model call; a turn without tools is one round');
    const idsAfter1 = state.session.ids.slice();
    const r2 = await m.chat(state, 'Name a color.', { maxNewTokens: 6, next });
    T.eq(state.history.length, 5, 'second turn: two more messages');
    const prompt2 = tok.encode(m.renderChat(state.history.slice(0, 4)));
    let lcp = 0; while (lcp < idsAfter1.length && idsAfter1[lcp] === prompt2[lcp]) lcp++;
    T.eq(r2.stats.cachedTokens, Math.min(lcp, prompt2.length - 1), 'turn 2 must reuse every cached token its prompt shares with the cache (sync the cache with syncCache)');
    T.ok(r2.stats.cachedTokens >= tok.encode(m.renderChat(state.history.slice(0, 2))).length, 'at least the whole first prompt must be a cache hit on turn 2');
    sameCache(T, state.session.cache, model, state.session.ids, 'after two turns');
    T.eq(state.session.ids.slice(0, prompt2.length), prompt2, 'session.ids must begin with the prompt of the last round');
  } },
  { step: 'chat', name: 'stats add up and every prompt respects the budget, dropping turns when it must', async run(m, T) {
    const tok = testTokenizer(m);
    const model = testModel(tok.vocabSize, { blockSize: 48 });
    const state = m.createChat({ model, tokenizer: tok, system: 'Be brief.' });
    const next = rng(8);
    let dropped = 0;
    for (const u of ['Say hello.', 'Name a color.', 'What is two and two?', 'The king and the queen.']) {
      const r = await m.chat(state, u, { maxNewTokens: 12, next, temperature: 0.9 });
      for (const round of r.stats.rounds) {
        T.eq(round.promptTokens, round.cachedTokens + round.prefilledTokens, 'every prompt token is either reused from the cache or prefilled');
        T.ok(round.promptTokens <= 48 - 12, `a prompt of ${round.promptTokens} tokens leaves no room for 12 new tokens in a 48-token window: the budget is blockSize - maxNewTokens, so a reply is never cut short by the window`);
        T.ok(round.generatedTokens <= 12, 'never generate more than maxNewTokens');
      }
      T.eq(r.stats.generatedTokens, r.stats.rounds.reduce((s, x) => s + x.generatedTokens, 0), 'stats.generatedTokens is the sum over rounds');
      T.ok(typeof r.stats.ms === 'number' && r.stats.ms >= 0, 'stats.ms is the wall-clock time of the turn');
      T.eq(r.stats.dropped, Math.max(...r.stats.rounds.map((x) => x.dropped)), 'stats.dropped is the largest per-round value (every round rebuilds the prompt from the same history, so a sum would count one dropped message several times)');
      T.eq([r.stats.promptTokens, r.stats.cachedTokens, r.stats.prefilledTokens], ['promptTokens', 'cachedTokens', 'prefilledTokens'].map((k) => r.stats.rounds.reduce((s, x) => s + x[k], 0)), 'stats.promptTokens, cachedTokens and prefilledTokens are sums over rounds');
      dropped = Math.max(dropped, r.stats.dropped);
    }
    T.ok(dropped > 0, 'four turns cannot fit a 48-token window: stats.dropped must report the messages the budget removed');
    T.eq(state.history.length, 9, 'state.history is the full transcript; dropping happens only in the prompt');
  } },
  { step: 'chat', name: 'a forced tool call runs the calculator and a second round answers with the result in context', async run(m, T) {
    const tok = testTokenizer(m);
    const model = testModel(tok.vocabSize);
    const state = m.createChat({ model, tokenizer: tok, system: 'Be brief.' });
    const r = await m.chat(state, 'What is 17*23?', { maxNewTokens: 6, next: rng(6), prefill: '<tool_call>calc(17*23)</tool_call>' });
    T.eq(state.history.map((x) => x.role), ['system', 'user', 'assistant', 'tool', 'assistant'], 'the tool call, its result and the final answer are three messages after the user turn');
    T.eq(state.history[2].content, '<tool_call>calc(17*23)</tool_call>', 'a prefill that already holds a complete call is the whole first reply: nothing is generated after it');
    T.eq(state.history[3].content, '391', 'the calculator result goes into the transcript as a tool message');
    T.eq(r.stats.toolCalls, 1, 'stats.toolCalls counts executed calls');
    T.eq(r.stats.rounds.length, 2, 'one round for the call, one for the answer');
    T.eq(r.stats.rounds[0].generatedTokens, 0, 'the forced round generates nothing');
    T.eq(r.stats.rounds[1].cachedTokens, r.stats.rounds[0].promptTokens, 'the answer round must reuse the entire first-round prompt from the cache');
    T.eq(r.reply, state.history[4].content, 'the reply of the turn is the LAST assistant message');
    const capped = m.createChat({ model, tokenizer: tok, system: 'Be brief.' });
    const c = await m.chat(capped, 'What is 17*23?', { maxNewTokens: 6, next: rng(6), prefill: '<tool_call>calc(17*23)</tool_call>', maxToolRounds: 0 });
    T.eq(capped.history.map((x) => x.role), ['system', 'user', 'assistant'], 'maxToolRounds 0: the last round\'s tool calls are not run, so no tool message is left behind without an answer round');
    T.eq([c.stats.rounds.length, c.stats.toolCalls], [1, 0], 'maxToolRounds 0 means exactly one round and no executed calls');
    // Both rounds of a tool turn rebuild the prompt from the same history and drop the same old turn.
    const small = testModel(tok.vocabSize, { blockSize: 52 });
    const tight = m.createChat({ model: small, tokenizer: tok, system: 'Be brief.' });
    await m.chat(tight, 'Say hello.', { maxNewTokens: 6, next: rng(7) });
    const d = await m.chat(tight, 'What is 17*23?', { maxNewTokens: 6, next: rng(7), prefill: '<tool_call>calc(17*23)</tool_call>' });
    T.eq(d.stats.rounds.map((x) => x.dropped), [2, 2], 'test setup: both rounds drop turn 1 from the prompt');
    T.eq(d.stats.dropped, 2, 'stats.dropped is the largest per-round value, not the sum: turn 1\'s two messages were dropped once, from every round\'s prompt');
  } },
  { step: 'chat', name: 'chat decodes with the state\'s stop ids and the caller\'s sampling options', async run(m, T) {
    const tok = testTokenizer(m);
    const model = testModel(tok.vocabSize);
    const prompt = tok.encode(m.renderChat([SYS, { role: 'user', content: 'Say hello.' }]));
    // Every id is a stop id: the first sample must end the reply.
    const stopAll = m.createChat({ model, tokenizer: tok, system: 'Be brief.' });
    stopAll.stopIds = Array.from({ length: tok.vocabSize }, (_, i) => i);
    const s = await m.chat(stopAll, 'Say hello.', { maxNewTokens: 6, next: rng(2) });
    T.eq([s.reply, s.stats.stopReason, s.stats.generatedTokens], ['', 'stop', 0], 'pass state.stopIds to decodeReply: when every token is a stop token the reply is empty with stopReason "stop" (without them a model that writes <|user|> invents the user\'s next message)');
    T.eq(stopAll.session.ids, prompt, 'a stopped reply leaves only the prompt in the session');
    // No stop ids: the reply must be exactly what sample() gives with the caller's options.
    const opts = { temperature: 0.7, topK: 12, topP: 0.8 };
    const ref = referenceSamples(model, prompt, 6, { ...opts, next: rng(21) });
    const open = m.createChat({ model, tokenizer: tok, system: 'Be brief.' });
    open.stopIds = [];
    const o = await m.chat(open, 'Say hello.', { maxNewTokens: 6, next: rng(21), ...opts });
    T.eq(open.session.ids.slice(prompt.length), ref, 'pass temperature, topK and topP through to decodeReply: the same seed and options must give the same tokens as sample()');
    T.eq(o.reply, tok.decode(ref), 'reply is the decoded generated ids');
  } },
];
