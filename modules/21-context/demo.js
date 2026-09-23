// Module 21 — goal demo: run YOUR ContextManager through a scripted 60-turn conversation with five facts
// planted in the first turns, tool results that bloat the history, and questions about the facts at
// the end. Count every message with the lab's BPE tokenizer, hold each policy to an 800-token budget,
// and measure how many of the late questions still have their fact somewhere in the context.

import { BPETokenizer } from 'lib/tokenizer.js';
import { rng, choice, randInt } from 'lib/util.js';

const BUDGET = 800;
const TURNS = 60;

const FACTS = [
  'the deploy key is in vault slot 7',
  'the staging database is called heron',
  'the release freeze starts on the 14th',
  'the on call engineer this week is Priya',
  'the budget code for the cluster is 4417',
];

// Two phrasings of a question per fact, asked in turns 50–59. Each shares some words with its fact,
// as a real follow-up usually does, which is what lexical retrieval relies on.
const PROBES = [
  [0, 'Which vault slot holds the deploy key?'],
  [1, 'What is the staging database called again?'],
  [2, 'When does the release freeze start?'],
  [3, 'Who is the on call engineer?'],
  [4, 'What budget code should the cluster job use?'],
  [0, 'Remind me where the deploy key lives.'],
  [1, 'Which name did we give the staging database?'],
  [2, 'Is the freeze before or after the release branch cut?'],
  [3, 'Can you page the engineer on call this week?'],
  [4, 'Which code do I bill the cluster to?'],
];

const TOPICS = ['the parser', 'the login page', 'the cache layer', 'the nightly job', 'the metrics panel', 'the search box', 'the upload form', 'the retry logic'];
const VERBS = ['refactor', 'speed up', 'test', 'document', 'simplify', 'profile', 'rename', 'review'];
const ASIDES = ['It should stay backwards compatible.', 'Keep the diff small.', 'We ship on Thursday.', 'The old version was slow.', 'Nobody has touched it in months.', 'Please explain your plan first.'];
const LOG_WORDS = ['GET', 'POST', 'ok', 'retry', 'timeout', 'cache', 'miss', 'hit', 'worker', 'queue', 'user', 'token', 'ms', 'error', 'warn'];

/** The scripted conversation: an array of turns, each a list of messages added in order. */
function script(next) {
  const turns = [];
  for (let t = 0; t < TURNS; t++) {
    const msgs = [];
    if (t < FACTS.length) {
      msgs.push({ role: 'user', content: `Before we start, a note for later.\nremember: ${FACTS[t]}` });
      msgs.push({ role: 'assistant', content: 'Noted. I will keep that in mind.' });
    } else if (t < TURNS - PROBES.length) {
      const topic = choice(next, TOPICS);
      msgs.push({ role: 'user', content: `Can you ${choice(next, VERBS)} ${topic}? ${choice(next, ASIDES)} ${choice(next, ASIDES)}` });
      if (t % 5 === 0) {
        const lines = [];
        for (let l = 0; l < 24; l++) lines.push(`${l} ${choice(next, LOG_WORDS)} ${choice(next, LOG_WORDS)} ${randInt(next, 900) + 100} ${choice(next, LOG_WORDS)}`);
        msgs.push({ role: 'assistant', content: `Let me search the logs for ${topic}.` });
        msgs.push({ role: 'tool', name: 'search_logs', content: lines.join('\n') });
      }
      msgs.push({ role: 'assistant', content: `Here is a plan for ${topic}: read it, change one function at a time, and run the tests after each change. ${choice(next, ASIDES)}` });
    } else {
      const [fact, q] = PROBES[t - (TURNS - PROBES.length)];
      msgs.push({ role: 'user', content: q, probe: fact });
      msgs.push({ role: 'assistant', content: 'Let me check what I have on that.' });
    }
    turns.push(msgs);
  }
  return turns;
}

export default async function demo(m, lab) {
  const tokenizer = BPETokenizer.fromJSON((await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default);
  const turns = script(rng(21));
  const system = 'You are a coding assistant for a small team. Answer briefly and use the notes you are given.';

  const policies = [
    { name: 'none (unmanaged)', opts: { policy: 'none' } },
    { name: 'drop oldest', opts: { policy: 'drop' } },
    { name: 'compact', opts: { policy: 'compact' } },
    { name: 'drop + BM25 memory', opts: { policy: 'drop', topK: 2 } },
  ];

  lab.log(`Budget ${BUDGET} tokens, ${TURNS} turns, ${FACTS.length} facts planted in turns 0–4, ${PROBES.length} questions about them in turns 50–59.`);
  const results = [];
  for (let p = 0; p < policies.length; p++) {
    const { name, opts } = policies[p];
    const cm = new m.ContextManager({ tokenizer, budget: BUDGET, system, keepLast: 6, maxToolTokens: 80, ...opts });
    const tokens = [];
    const probeHits = [];
    let toolTokens = 0;
    let finalFacts = 0;
    for (let t = 0; t < turns.length; t++) {
      const [userMsg, ...rest] = turns[t];
      cm.add(userMsg);
      // The model is called after every user message: this is the context it would see.
      const view = cm.assemble();
      tokens.push(view.tokens);
      if (userMsg.probe !== undefined) {
        const fact = FACTS[userMsg.probe];
        probeHits.push(view.messages.some((x) => String(x.content).includes(fact)) ? 1 : 0);
      }
      if (t === turns.length - 1) {
        finalFacts = FACTS.filter((f) => view.messages.some((x) => String(x.content).includes(f))).length;
        for (const x of view.messages) if (x.role === 'tool') toolTokens += m.messageTokens(tokenizer, x);
      }
      for (const msg of rest) cm.add(msg);
      if (t % 10 === 0) { lab.progress((p + t / turns.length) / policies.length, `${name}: turn ${t}`); await lab.tick(); }
    }
    const managed = opts.policy !== 'none';
    if (managed) lab.check(Math.max(...tokens) <= BUDGET, `policy "${name}" went over the ${BUDGET}-token budget (peak ${Math.max(...tokens)})`);
    results.push({ name, tokens, probeHits, hits: probeHits.reduce((a, b) => a + b, 0), finalFacts, toolTokens, managed,
      peak: Math.max(...tokens), mean: tokens.reduce((a, b) => a + b, 0) / tokens.length, final: tokens[tokens.length - 1] });
  }

  lab.plot({
    title: `Tokens in the assembled context per turn (budget ${BUDGET})`,
    x: Array.from({ length: TURNS }, (_, i) => i),
    series: results.map((r) => ({ name: r.name, values: r.tokens })),
    xlabel: 'turn', ylabel: 'tokens sent to the model', yscale: 'log',
  });
  const managed = results.filter((r) => r.managed);
  lab.bar({ title: `Questions answered from context (of ${PROBES.length}) under the ${BUDGET}-token budget`, labels: managed.map((r) => r.name), values: managed.map((r) => r.hits) });
  lab.heatmap({
    title: 'Was the asked-about fact in the context? (1 = yes)',
    rows: results.map((r) => r.probeHits),
    rowLabels: results.map((r) => r.name),
    colLabels: PROBES.map((_, i) => `t${TURNS - PROBES.length + i}`),
    min: 0, max: 1,
  });
  lab.table({
    title: 'Policies compared',
    columns: ['policy', 'peak tokens', 'mean tokens', 'tool tokens at turn 59', `facts in final context (of ${FACTS.length})`, `questions answerable (of ${PROBES.length})`],
    rows: results.map((r) => [r.name, r.peak, Math.round(r.mean), r.toolTokens, r.finalFacts, r.hits]),
  });

  const [none, drop, comp, mem] = results;
  lab.done(`Unmanaged, the conversation grew to **${none.final} tokens** by turn ${TURNS - 1}, ${(none.final / BUDGET).toFixed(1)}× the ${BUDGET}-token budget, and **${none.toolTokens}** of those tokens (${(100 * none.toolTokens / none.final).toFixed(0)}%) were raw tool output. ` +
    `Every managed policy stayed under budget (peaks ${drop.peak}, ${comp.peak} and ${mem.peak}). ` +
    `**Drop oldest** could answer **${drop.hits}/${PROBES.length}** of the late questions from its context${drop.hits === 0 ? ': the facts from turns 0–4 were gone' : ''}. ` +
    `**Compaction** kept ${comp.finalFacts}/${FACTS.length} facts pinned in its summary and answered **${comp.hits}/${PROBES.length}**; ` +
    `**BM25 memory** fetched the right note back for **${mem.hits}/${PROBES.length}** questions while paying for at most ${policies[3].opts.topK} notes per turn. ` +
    `Same tokenizer, same budget, same conversation: the policy decided what the model could know.`);
}
