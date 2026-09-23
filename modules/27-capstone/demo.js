// Goal demo for the capstone: YOUR chat pipeline (buildPrompt, syncCache, decodeReply, runToolCalls, chat)
// drives the checkpoint pre-trained in this lab through a scripted four-turn conversation, once with the
// KV cache kept across turns and once with a fresh cache every turn, and reports what each turn cost.
import { BPETokenizer } from 'lib/tokenizer.js';
import { loadModel, newCache } from 'lib/infer.js';
import { rng } from 'lib/util.js';

const SYSTEM = 'Be brief.';
const TURNS = [
  { user: 'Say hello.' },
  { user: 'Name a color.' },
  { user: 'What is 17*23?', prefill: '<tool_call>calc(17*23)</tool_call>' },
  { user: 'Tell me about the king.' },
];
const OPTS = { maxNewTokens: 10, temperature: 0.8, topK: 40 };

/** Run the scripted conversation. With reuse off, the session is replaced by an empty cache before every turn. */
async function converse(m, model, tokenizer, { reuse, lab }) {
  const state = m.createChat({ model, tokenizer, system: SYSTEM });
  const next = rng(1);
  const rows = [];
  for (let i = 0; i < TURNS.length; i++) {
    if (!reuse) state.session = { cache: newCache(model), ids: [] };
    const { user, prefill } = TURNS[i];
    const { reply, stats } = await m.chat(state, user, { ...OPTS, next, ...(prefill ? { prefill } : {}) });
    rows.push({ user, reply, stats });
    await lab.tick();
  }
  return { state, rows };
}

export default async function demo(m, lab) {
  // 1. The layers: the lab's pre-trained checkpoint and its tokenizer, extended with the chat tokens.
  const json = (await import('lib/checkpoints/tiny-gpt.json', { with: { type: 'json' } })).default;
  const tokJson = (await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default;
  const base = BPETokenizer.fromJSON(tokJson);
  const tokenizer = m.chatTokenizer(base);
  const model = m.extendVocab(loadModel(json), tokenizer.vocabSize);
  const { blockSize } = model.config;
  lab.log(`No SFT checkpoint ships with the lab, so this is the pre-trained tiny-gpt.json (${json.config.nLayer} layers, width ${json.config.nEmbd}, ${blockSize}-token window).`);
  lab.log(`vocabulary ${base.vocabSize} -> ${tokenizer.vocabSize}: ${m.EXTRA_TOKENS.join(' ')} are new, untrained rows.`);
  lab.log(`budget per prompt: blockSize ${blockSize} - maxNewTokens ${OPTS.maxNewTokens} = ${blockSize - OPTS.maxNewTokens} tokens.`);

  // 2. The conversation, with the KV cache kept across turns (your syncCache), and again with an empty
  // cache at the start of every turn: the baseline prefix reuse is measured against. Both are
  // deterministic, so the token counts come from one run; wall-clock time is noisy (JIT warm-up, GC),
  // so after one throwaway run each mode is repeated and the per-turn median is reported.
  const REPEATS = 5;
  await converse(m, model, tokenizer, { reuse: true, lab });
  const warmRuns = [];
  const coldRuns = [];
  for (let r = 0; r < REPEATS; r++) {
    warmRuns.push(await converse(m, model, tokenizer, { reuse: true, lab }));
    coldRuns.push(await converse(m, model, tokenizer, { reuse: false, lab }));
    lab.progress((r + 1) / REPEATS, `run ${r + 1} of ${REPEATS}`);
  }
  const median = (xs) => { const s = xs.slice().sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
  const medianMs = (runs) => TURNS.map((_, i) => median(runs.map((run) => run.rows[i].stats.ms)));
  const warm = warmRuns[0];
  const cold = coldRuns[0];
  const warmTurnMs = medianMs(warmRuns);
  const coldTurnMs = medianMs(coldRuns);

  const history = warm.state.history;
  const toolMessage = history.find((x) => x.role === 'tool');
  lab.check(toolMessage && toolMessage.content === '391', `the calculator turn must put a tool message "391" into the transcript, got ${JSON.stringify(toolMessage && toolMessage.content)}`);
  lab.check(warm.rows.every((r) => r.stats.rounds.every((x) => x.promptTokens <= blockSize - OPTS.maxNewTokens)), 'every prompt must fit the budget');

  const show = (s) => JSON.stringify(s.replace(/\n/g, ' ')).slice(1, -1);
  lab.table({
    title: 'Per turn (KV cache kept across turns)',
    columns: ['turn', 'user', 'rounds', 'prompt tokens', 'cached', 'prefilled', 'generated', 'dropped msgs', 'tool calls', 'stop', 'median ms', 'reply'],
    rows: warm.rows.map((r, i) => [
      i + 1, r.user, r.stats.rounds.length, r.stats.promptTokens, r.stats.cachedTokens, r.stats.prefilledTokens,
      r.stats.generatedTokens, r.stats.dropped, r.stats.toolCalls, r.stats.stopReason, +warmTurnMs[i].toFixed(1), show(r.reply),
    ]),
  });
  lab.bar({
    title: 'Tokens run through the model for prompts, per turn: reused cache vs fresh cache',
    labels: warm.rows.flatMap((_, i) => [`turn ${i + 1} reuse`, `turn ${i + 1} fresh`]),
    values: warm.rows.flatMap((r, i) => [r.stats.prefilledTokens, cold.rows[i].stats.prefilledTokens]),
  });
  lab.plot({
    title: `Latency per turn (prefill + decode, all rounds), median of ${REPEATS} runs, ms`,
    x: warm.rows.map((_, i) => i + 1),
    series: [
      { name: 'KV cache kept across turns', values: warmTurnMs.map((x) => +x.toFixed(2)) },
      { name: 'fresh cache every turn', values: coldTurnMs.map((x) => +x.toFixed(2)) },
    ],
    xlabel: 'turn', ylabel: 'ms',
  });
  lab.md(`### The transcript\n\n${m.renderTranscript(history)}`);

  const sum = (rows, key) => rows.reduce((s, r) => s + r.stats[key], 0);
  const warmPrefill = sum(warm.rows, 'prefilledTokens');
  const coldPrefill = sum(cold.rows, 'prefilledTokens');
  const promptTotal = sum(warm.rows, 'promptTokens');
  const cachedTotal = sum(warm.rows, 'cachedTokens');
  const generated = sum(warm.rows, 'generatedTokens');
  const warmMs = warmTurnMs.reduce((a, b) => a + b, 0);
  const coldMs = coldTurnMs.reduce((a, b) => a + b, 0);
  const t2 = warm.rows[1].stats;
  const t4 = warm.rows[3].stats;
  const stops = warm.rows.flatMap((r) => r.stats.rounds.map((x) => x.stopReason));
  const ended = stops.filter((s) => s === 'stop').length;
  const sameReplies = warm.rows.every((r, i) => r.reply === cold.rows[i].reply);
  lab.done(`Your pipeline held a **${TURNS.length}-turn** conversation (${stops.length} model rounds, 1 calculator call returning **${toolMessage.content}**) inside a ${blockSize}-token window. Across all rounds the prompts totalled **${promptTotal}** tokens, of which **${cachedTotal}** (${Math.round((100 * cachedTotal) / promptTotal)}%) came from the KV cache: you prefilled **${warmPrefill}** tokens instead of **${coldPrefill}**, and the four turns took **${warmMs.toFixed(0)} ms** (median over ${REPEATS} runs) against ${coldMs.toFixed(0)} ms with a fresh cache every turn${sameReplies ? ', with identical replies' : ''}. Turn 2 reused ${t2.cachedTokens} of its ${t2.promptTokens} prompt tokens; by turn 4 the budget had dropped ${t4.dropped} messages, which changed everything after the system prompt, so only ${t4.cachedTokens} tokens were reusable. The model generated ${generated} tokens, and ${ended} of ${stops.filter((s) => s !== 'forced').length} generated replies ended on a stop token: it was never trained on the chat markers, so it answers with corpus-like text rather than help.`);
}
