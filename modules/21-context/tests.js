import { CharTokenizer } from 'lib/tokenizer.js';

// A whitespace "tokenizer": one id per word, so every count in these tests can be done by hand.
// Anything with encode(str) → ids and decode(ids) → str works with the learner's code.
function wordTokenizer() {
  const vocab = [];
  const stoi = new Map();
  return {
    encode(s) {
      return (String(s).match(/\S+/g) ?? []).map((w) => {
        if (!stoi.has(w)) { stoi.set(w, vocab.length); vocab.push(w); }
        return stoi.get(w);
      });
    },
    decode(ids) { return ids.map((i) => vocab[i]).join(' '); },
  };
}

const wordsOf = (n, prefix = 'w') => Array.from({ length: n }, (_, i) => `${prefix}${i}`).join(' ');

function sampleHistory() {
  return [
    { role: 'system', content: 'You are terse.' },                                   // 4 + 3 = 7
    { role: 'user', content: 'hello there\nremember: the deploy key is in vault slot 7' }, // 4 + 11 = 15
    { role: 'assistant', content: 'noted' },                                          // 4 + 1 = 5
    { role: 'user', content: 'Remember:  the build server is called atlas ' },        // 4 + 7 = 11
    { role: 'assistant', content: 'ok' },                                             // 5
    { role: 'user', content: 'what now' },                                            // 6
    { role: 'assistant', content: 'nothing' },                                        // 5
  ];
}

/** Drive a manager through `turns` turns; the fact is injected on turn 2, tool bloat every 7th turn. */
function converse(m, T, opts, turns = 50) {
  const tok = wordTokenizer();
  const cm = new m.ContextManager({ tokenizer: tok, budget: 120, system: 'You are a careful assistant.', keepLast: 4, maxToolTokens: 20, ...opts });
  const next = T.rng(21);
  for (let t = 0; t < turns; t++) {
    const filler = wordsOf(4 + Math.floor(next() * 6), `chat${t}_`);
    cm.add({ role: 'user', content: t === 2 ? 'remember: the deploy key is in vault slot 7' : `turn ${t} ${filler}` });
    if (t % 7 === 3) cm.add({ role: 'tool', name: 'search', content: wordsOf(60, `hit${t}_`) });
    cm.add({ role: 'assistant', content: `reply ${t} ${filler}` });
  }
  cm.add({ role: 'user', content: 'where is the deploy key?' });
  return { cm, tok };
}

export const tests = [
  // ---------- step 1 ----------
  { step: 'count', name: 'messageTokens = per-message overhead + content tokens (+ tool name)', run(m, T) {
    const tok = wordTokenizer();
    T.eq(m.messageTokens(tok, { role: 'user', content: 'a b c' }), m.TOKENS_PER_MESSAGE + 3, 'three words plus the framing overhead');
    T.eq(m.messageTokens(tok, { role: 'assistant', content: '' }), m.TOKENS_PER_MESSAGE, 'an empty message still costs its role framing');
    T.eq(m.messageTokens(tok, { role: 'tool', name: 'grep', content: 'a b c' }), m.TOKENS_PER_MESSAGE + 4, 'a tool message carries its tool name into the context too');
    const chars = new CharTokenizer('abcdefghijklmnopqrstuvwxyz ');
    T.eq(m.messageTokens(chars, { role: 'user', content: 'hello world' }), m.TOKENS_PER_MESSAGE + 11, 'the count must come from the tokenizer handed in, not from a fixed characters-per-token guess');
  } },
  { step: 'count', name: 'contextTokens sums every message; an empty array costs nothing', run(m, T) {
    const tok = wordTokenizer();
    T.eq(m.contextTokens(tok, []), 0);
    T.eq(m.contextTokens(tok, sampleHistory()), 7 + 15 + 5 + 11 + 5 + 6 + 5, 'sum of (4 + words) over the seven messages');
    const chars = new CharTokenizer('abcdefghijklmnopqrstuvwxyz ');
    const msgs = [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }];
    T.eq(m.contextTokens(chars, msgs), 2 * m.TOKENS_PER_MESSAGE + 7, 'two messages: 2 × overhead + 7 characters');
  } },
  { step: 'count', name: 'budgetReport says whether the array fits and by how much', run(m, T) {
    const tok = wordTokenizer();
    const msgs = sampleHistory(); // 54 tokens
    const r = m.budgetReport(tok, msgs, 60);
    T.eq(r.tokens, 54); T.eq(r.budget, 60); T.eq(r.remaining, 6); T.eq(r.fits, true);
    T.eq(m.budgetReport(tok, msgs, 54).fits, true, 'exactly on budget still fits (<=, not <)');
    const over = m.budgetReport(tok, msgs, 50);
    T.eq(over.fits, false); T.eq(over.remaining, -4, 'remaining goes negative when over budget so callers can see how much to cut');
  } },

  // ---------- step 2 ----------
  { step: 'truncate', name: 'truncateText keeps the head and the tail and fits maxTokens including the marker', run(m, T) {
    const tok = wordTokenizer();
    const text = wordsOf(20);
    T.eq(m.truncateText(tok, text, 20), text, 'text that fits (exactly) is returned unchanged');
    T.eq(m.truncateText(tok, text, 25), text);
    const cut = m.truncateText(tok, text, 11);
    const marker = m.countTokens(tok, m.TRUNCATION_MARKER); // 3 words on this tokenizer
    T.eq(marker, 3, 'marker cost under the word tokenizer');
    T.ok(cut.includes(m.TRUNCATION_MARKER), 'the model must be told something was cut: include TRUNCATION_MARKER');
    T.ok(cut.startsWith('w0 w1 w2 w3'), `head must be kept (the first ceil((11-3)/2) = 4 words); got ${JSON.stringify(cut)}`);
    T.ok(cut.endsWith('w16 w17 w18 w19'), `tail must be kept (the last 4 words); got ${JSON.stringify(cut)}`);
    T.eq(m.countTokens(tok, cut), 11, 'the result must cost at most maxTokens INCLUDING the marker, or the budget arithmetic upstream is wrong');
    T.eq(m.countTokens(tok, m.truncateText(tok, text, 10)), 10, 'odd keep counts still cost exactly maxTokens');
  } },
  { step: 'truncate', name: 'truncateText splits odd keeps head-first and handles an empty tail', run(m, T) {
    const tok = wordTokenizer();
    const text = wordsOf(20);
    const M = m.TRUNCATION_MARKER;
    T.eq(m.truncateText(tok, text, 10), 'w0 w1 w2 w3' + M + 'w17 w18 w19', 'keep = 10 − 3 = 7: the head gets ceil(7 / 2) = 4 tokens and the tail the other 3');
    T.eq(m.truncateText(tok, text, 4), 'w0' + M, 'keep = 1: one head token and an EMPTY tail; `ids.slice(-0)` is the whole array, so a tail of 0 needs its own case');
    T.eq(m.countTokens(tok, m.truncateText(tok, text, 3)), 3, 'keep = 0: only the marker remains');
  } },
  { step: 'truncate', name: 'truncateToolResults touches only long tool messages and never mutates the input', run(m, T) {
    const tok = wordTokenizer();
    const long = wordsOf(30, 'r');
    const msgs = [
      { role: 'system', content: 'sys' },
      { role: 'user', content: long },
      { role: 'tool', name: 'search', content: long },
      { role: 'tool', name: 'ls', content: 'a b' },
      { role: 'assistant', content: 'done' },
    ];
    const before = JSON.stringify(msgs);
    const out = m.truncateToolResults(tok, msgs, 12);
    T.eq(out.length, 5, 'one output message per input message');
    T.ok(out[1] === msgs[1], 'a long USER message is not a tool result and must be left alone');
    T.ok(out[3] === msgs[3] && out[0] === msgs[0] && out[4] === msgs[4], 'messages that fit are returned as the same objects');
    T.ok(out[2] !== msgs[2] && out[2].content.includes(m.TRUNCATION_MARKER), 'the 30-token tool result must be cut');
    T.eq(out[2].role, 'tool'); T.eq(out[2].name, 'search', 'role and name survive truncation');
    T.eq(m.countTokens(tok, out[2].content), 12);
    T.eq(JSON.stringify(msgs), before, 'the input array and its messages must not be modified');
  } },
  { step: 'truncate', name: 'dropOldest removes the oldest non-system messages and only as many as needed', run(m, T) {
    const tok = wordTokenizer();
    const msgs = sampleHistory(); // 54 tokens: 7, 15, 5, 11, 5, 6, 5
    const before = JSON.stringify(msgs);
    T.eq(m.dropOldest(tok, msgs, 54), msgs, 'an array that already fits is returned unchanged');
    const a = m.dropOldest(tok, msgs, 40);
    T.eq(a.map((x) => x.content), msgs.slice(0, 1).concat(msgs.slice(2)).map((x) => x.content), 'dropping the first user turn (15) brings 54 to 39 <= 40: stop there, do not drop more');
    const b = m.dropOldest(tok, msgs, 20);
    T.eq(b.map((x) => x.content), [msgs[0], msgs[5], msgs[6]].map((x) => x.content), '7 + 6 + 5 = 18 fits 20; the system prompt plus the last two turns');
    T.eq(JSON.stringify(msgs), before, 'the input must not be modified');
  } },
  { step: 'truncate', name: 'the system prompt is never dropped, even when it alone exceeds the budget', run(m, T) {
    const tok = wordTokenizer();
    const msgs = [
      { role: 'user', content: 'a b c' },
      { role: 'system', content: wordsOf(30, 's') },
      { role: 'user', content: 'd e' },
    ];
    const out = m.dropOldest(tok, msgs, 10);
    T.eq(out.length, 1, 'everything but the system prompt goes');
    T.eq(out[0].role, 'system', 'the system prompt carries the instructions and the safety rules; a policy that can drop it is not a policy');
    T.eq(m.contextTokens(tok, out), 34, 'the result may still be over budget; that is reported, not "fixed" by deleting the system prompt');
  } },

  // ---------- step 3 ----------
  { step: 'compact', name: 'extractFacts reads `remember:` lines, any case, trimmed, deduplicated', run(m, T) {
    T.eq(m.extractFacts('hello\nremember: the key is 7\nbye'), ['the key is 7']);
    T.eq(m.extractFacts('  Remember:   lunch at noon  \nREMEMBER: lunch at noon\nremember: lunch at noon'), ['lunch at noon'], 'case-insensitive prefix, whitespace trimmed, duplicates dropped');
    T.eq(m.extractFacts('please remember: this is not a fact line\nremember:\n'), [], 'the prefix must start the line and the fact must be non-empty');
    T.eq(m.extractFacts(''), []);
    T.eq(m.extractFacts(undefined), [], 'a message without content yields no facts rather than throwing');
    T.eq(m.extractFacts('remember: a\nremember: b\nremember: a'), ['a', 'b'], 'first occurrence order is kept');
  } },
  { step: 'compact', name: 'compact replaces the old span by one pinned summary that carries the facts', run(m, T) {
    const tok = wordTokenizer();
    const msgs = sampleHistory();
    const before = JSON.stringify(msgs);
    const out = m.compact(msgs, { keepLast: 2 });
    T.eq(out.length, 4, 'system prompt + summary + the last 2 non-system messages');
    T.eq(out[0].content, 'You are terse.', 'the system prompt stays first');
    T.eq(out[1].role, 'system', 'the summary is pinned with role "system" so dropOldest can never remove it');
    T.eq(out[1].summary, true, 'the summary is flagged so the next compaction can recognise and fold it');
    T.eq(out[2].content, 'what now'); T.eq(out[3].content, 'nothing');
    T.eq(m.extractFacts(out[1].content), ['the deploy key is in vault slot 7', 'the build server is called atlas'], 'facts are written back as `remember:` lines so the scripted summariser can read its own output');
    T.ok(out[1].content.includes('4 earlier messages'), `the summary states how many messages it stands for; got ${JSON.stringify(out[1].content)}`);
    T.ok(m.contextTokens(tok, out) < m.contextTokens(tok, msgs), 'compaction must make the context smaller');
    T.eq(JSON.stringify(msgs), before, 'the input must not be modified');
  } },
  { step: 'compact', name: 'compacting twice keeps every fact and yields a single, cumulative summary', run(m, T) {
    const once = m.compact(sampleHistory(), { keepLast: 2 });
    const more = [...once,
      { role: 'user', content: 'remember: lunch is at noon' }, { role: 'assistant', content: 'sure' },
      { role: 'user', content: 'thanks' }, { role: 'assistant', content: 'welcome' }];
    const twice = m.compact(more, { keepLast: 2 });
    T.eq(twice.length, 4, 'system prompt + one summary + the last 2 messages');
    T.eq(twice.filter((x) => x.summary).length, 1, 'an earlier summary is folded into the new one, not left as a second pinned message');
    T.eq(m.extractFacts(twice[1].content), ['the deploy key is in vault slot 7', 'the build server is called atlas', 'lunch is at noon'], 'facts from the first summary AND the newly compacted turns, oldest first; losing the old summary\'s facts is how agents forget after their second compaction');
    T.ok(twice[1].content.includes('8 earlier messages'), `the count accumulates (4 from the old summary + 4 new); got ${JSON.stringify(twice[1].content)}`);
    T.eq(twice.slice(2).map((x) => x.content), ['thanks', 'welcome']);
  } },
  { step: 'compact', name: 'nothing older than keepLast means nothing to compact', run(m, T) {
    const msgs = sampleHistory();
    T.eq(m.compact(msgs, { keepLast: 6 }), msgs, 'six non-system messages, keep six: unchanged');
    T.eq(m.compact(msgs, { keepLast: 10 }), msgs);
    const out = m.compact(msgs, { keepLast: 5 });
    T.eq(out.length, 7, 'one message compacted still produces a summary (6 → 1 + 5)');
    T.eq(out.filter((x) => x.summary).length, 1);
  } },

  { step: 'compact', name: 'compact leaves plain system messages in place and only folds summaries', run(m, T) {
    const msgs = [
      { role: 'system', content: 'You are terse.' },
      { role: 'user', content: 'remember: lunch is at noon' },
      { role: 'system', content: 'Never run rm -rf.' },
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
      { role: 'user', content: 'bye' },
    ];
    const out = m.compact(msgs, { keepLast: 2 });
    T.eq(out.map((x) => x.content.split('\n')[0]), ['You are terse.', 'Summary of 2 earlier messages.', 'Never run rm -rf.', 'hello', 'bye'],
      'a system message that is not a summary is an instruction, not conversation: it stays where it was and is not counted; the summary goes where the span began');
    T.eq(m.extractFacts(out[1].content), ['lunch is at noon']);
  } },

  // ---------- step 4 ----------
  { step: 'bm25', name: 'idf follows ln(1 + (N - df + 0.5) / (df + 0.5))', run(m, T) {
    const idx = new m.BM25Index(['the deploy key is in vault slot seven', 'the build server is called atlas', 'atlas runs the nightly build and the deploy of atlas', 'lunch is at noon']);
    T.close(idx.idf('atlas'), Math.log(2), 1e-6, 'df = 2 of N = 4: ln(1 + 2.5 / 2.5) = ln 2');
    T.close(idx.idf('the'), 0.3566749, 1e-6, 'df = 3 of 4: a term in almost every note is nearly worthless as evidence');
    T.close(idx.idf('zzz'), Math.log(10), 1e-6, 'an unseen term has df = 0: ln(1 + 4.5 / 0.5)');
    T.ok(idx.idf('the') > 0, 'this idf variant never goes negative (the +1 inside the log), unlike the original Robertson form');
  } },
  { step: 'bm25', name: 'score matches hand-computed values with k1 = 1.5, b = 0.75', run(m, T) {
    const idx = new m.BM25Index(['the deploy key is in vault slot seven', 'the build server is called atlas', 'atlas runs the nightly build and the deploy of atlas', 'lunch is at noon']);
    T.close(idx.avgdl, 7, 1e-9, 'average length of (8, 6, 10, 4) terms');
    // doc 1: tf=1, len 6: ln2 * (1 * 2.5) / (1 + 1.5 * (0.25 + 0.75 * 6/7)) = 0.740768
    T.close(idx.score('atlas', 1), 0.740768, 1e-5, 'tf = 1, length 6 (shorter than average, so the normaliser is < 1)');
    // doc 2: tf=2, len 10: ln2 * (2 * 2.5) / (2 + 1.5 * (0.25 + 0.75 * 10/7)) = 0.870319
    T.close(idx.score('atlas', 2), 0.870319, 1e-5, 'tf = 2 but length 10: the second occurrence is worth much less than the first');
    T.close(idx.score('atlas', 0), 0, 1e-9, 'a document without the term scores 0');
    T.close(idx.score('deploy key', 0), 1.782529, 1e-5, 'multi-term queries sum the per-term contributions');
    T.close(idx.score('Deploy, KEY!', 0), 1.782529, 1e-5, 'queries go through tokenizeTerms: case and punctuation do not matter');
  } },
  { step: 'bm25', name: 'length normalisation and tf saturation behave like BM25, not like raw counting', run(m, T) {
    const short = 'atlas builds nightly';
    const long = 'atlas builds nightly and also serves lunch to the whole team at noon';
    const idx = new m.BM25Index([short, long, 'unrelated note about cats']);
    T.ok(idx.score('atlas', 0) > idx.score('atlas', 1), 'same tf = 1, but the shorter note should score higher (b = 0.75 length normalisation)');
    const flat = new m.BM25Index([short, long, 'unrelated note about cats'], { b: 0 });
    T.close(flat.score('atlas', 0), flat.score('atlas', 1), 1e-9, 'with b = 0 length no longer matters');
    const sat = new m.BM25Index(['atlas', 'atlas '.repeat(100), 'cats']);
    const cap = (1.5 + 1) * sat.idf('atlas');
    T.ok(sat.score('atlas', 1) < cap, `tf = 100 must stay below the saturation ceiling (k1 + 1) · idf = ${cap.toFixed(4)}; a raw tf score would be ~100× larger`);
    T.ok(sat.score('atlas', 1) > sat.score('atlas', 0), 'more occurrences still score higher, just with diminishing returns');
  } },
  { step: 'bm25', name: 'search returns the top-k positive hits, best first, ties broken by index', run(m, T) {
    const docs = ['the deploy key is in vault slot seven', 'the build server is called atlas', 'atlas runs the nightly build and the deploy of atlas', 'lunch is at noon'];
    const idx = new m.BM25Index(docs);
    T.eq(idx.search('atlas', 3).map((h) => h.index), [2, 1], 'only notes with a positive score are returned, even when k is larger');
    T.eq(idx.search('deploy key', 1).map((h) => h.index), [0], 'k limits the result');
    T.eq(idx.search('the', 5).map((h) => h.index), [2, 1, 0], 'sorted by score descending');
    T.eq(idx.search('unicorns', 3), [], 'no matching term: empty, not an error');
    const tie = new m.BM25Index(['cat', 'dog', 'cat']);
    T.eq(tie.search('cat', 5).map((h) => h.index), [0, 2], 'equal scores keep index order so results are deterministic');
    T.close(idx.search('atlas', 1)[0].score, 0.870319, 1e-5, 'hits carry their score');
  } },

  // ---------- step 5 ----------
  { step: 'assemble', name: 'assemble puts the system prompt first and keeps everything when the budget allows', run(m, T) {
    const tok = wordTokenizer();
    const cm = new m.ContextManager({ tokenizer: tok, budget: 1000, system: 'You are terse.', policy: 'drop' });
    cm.add({ role: 'user', content: 'hello there' }).add({ role: 'assistant', content: 'hi' }).add({ role: 'user', content: 'how are you' });
    const { messages, tokens, retrieved } = cm.assemble();
    T.eq(messages.map((x) => x.role), ['system', 'user', 'assistant', 'user']);
    T.eq(messages[0].content, 'You are terse.');
    T.eq(tokens, m.contextTokens(tok, messages), 'reported tokens must be the cost of the messages actually returned');
    T.eq(tokens, 4 + 3 + 4 + 2 + 4 + 1 + 4 + 3);
    T.eq(retrieved, [], 'retrieval is off by default (topK = 0)');
    T.eq(cm.history.length, 3, 'assemble is a view: the full history is kept');
  } },
  { step: 'assemble', name: 'drop policy: every one of 50 turns fits the budget and keeps the latest user message', run(m, T) {
    const tok = wordTokenizer();
    const cm = new m.ContextManager({ tokenizer: tok, budget: 80, system: 'You are a careful assistant.', policy: 'drop', maxToolTokens: 20 });
    const raw = new m.ContextManager({ tokenizer: tok, budget: 80, system: 'You are a careful assistant.', policy: 'none' });
    let maxTokens = 0, rawMax = 0;
    for (let t = 0; t < 50; t++) {
      const user = { role: 'user', content: `turn ${t} ${wordsOf(5, 'q')}` };
      cm.add(user); raw.add(user);
      if (t % 5 === 0) { const tool = { role: 'tool', name: 'search', content: wordsOf(50, 'r') }; cm.add(tool); raw.add(tool); }
      const a = cm.assemble();
      maxTokens = Math.max(maxTokens, a.tokens);
      T.ok(a.tokens <= 80, `turn ${t}: assembled context costs ${a.tokens} > budget 80`);
      T.eq(a.messages[0].role, 'system', 'system prompt first, always');
      const newest = a.messages[a.messages.length - 1];
      T.ok(a.messages.includes(user), `turn ${t}: the newest user message is what the model must answer; it is never the one dropped`);
      T.ok(newest === user || (t % 5 === 0 && newest.role === 'tool'), `turn ${t}: order must be preserved, so the context ends with the newest history message (the user turn, or the tool result that followed it)`);
      for (const msg of a.messages) if (msg.role === 'tool') T.ok(m.countTokens(tok, msg.content) <= 20, 'tool results are truncated to maxToolTokens before anything else');
      rawMax = Math.max(rawMax, raw.assemble().tokens);
      cm.add({ role: 'assistant', content: `reply ${t}` }); raw.add({ role: 'assistant', content: `reply ${t}` });
    }
    T.ok(rawMax > 1000, `policy "none" must return the raw history (${rawMax} tokens at its largest) so you can see what unmanaged growth looks like`);
  } },
  { step: 'assemble', name: 'compact policy keeps a fact from turn 2 alive through 50 turns; drop alone loses it', run(m, T) {
    const fact = 'the deploy key is in vault slot 7';
    const { cm: compacted } = converse(m, T, { policy: 'compact' });
    const a = compacted.assemble();
    T.ok(a.tokens <= 120, `compact policy must still respect the budget (got ${a.tokens})`);
    T.ok(a.messages.some((x) => x.summary && x.content.includes(fact)), 'the fact must be present in a pinned summary message');
    T.eq(a.messages[a.messages.length - 1].content, 'where is the deploy key?');
    T.eq(a.messages.filter((x) => x.summary).length, 1, 'exactly one summary');
    const { cm: dropped } = converse(m, T, { policy: 'drop' });
    const b = dropped.assemble();
    T.ok(b.tokens <= 120);
    T.ok(!b.messages.some((x) => x.content.includes(fact)), 'with oldest-first dropping and no retrieval the fact is gone: that is the failure compaction exists to prevent');
  } },
  { step: 'assemble', name: 'retrieval injects the relevant notes just before the latest user message, inside the budget', run(m, T) {
    const tok = wordTokenizer();
    const notes = ['the deploy key is in vault slot 7', 'the build server is called atlas', 'lunch is at noon on fridays', 'the office plant is named fern'];
    const cm = new m.ContextManager({ tokenizer: tok, budget: 60, system: 'You are terse.', policy: 'drop', topK: 2, notes });
    for (let t = 0; t < 10; t++) cm.add({ role: 'user', content: `turn ${t} ${wordsOf(6, 'x')}` }).add({ role: 'assistant', content: `reply ${t}` });
    cm.add({ role: 'user', content: 'when do we eat lunch?' });
    const a = cm.assemble();
    T.eq(a.retrieved, ['lunch is at noon on fridays'], 'BM25 over the note store with the latest user message as the query; only the one matching note scores > 0');
    const n = a.messages.length;
    T.eq(a.messages[n - 1].content, 'when do we eat lunch?');
    T.ok(a.messages[n - 2].retrieved === true && a.messages[n - 2].content.includes('lunch is at noon on fridays'), 'the retrieval message sits immediately before the latest user message, where the model attends to it best and where it does not break the cached prefix');
    T.ok(a.tokens <= 60, `the retrieval message must be paid for out of the budget BEFORE the history is trimmed (got ${a.tokens} > 60)`);
    T.eq(a.messages[0].role, 'system');
    cm.add({ role: 'assistant', content: 'noon' }).add({ role: 'user', content: 'what is the plant named and where is the deploy key?' });
    T.eq(cm.assemble().retrieved, ['the office plant is named fern', 'the deploy key is in vault slot 7'], 'top-2 for a two-topic query, best first: the fern note matches two rare terms ("plant", "named") and is shorter, so it outranks the deploy-key note');
    cm.add({ role: 'assistant', content: 'checking' }).add({ role: 'user', content: 'when do we eat lunch?' }).add({ role: 'tool', name: 'search', content: 'plant fern plant fern office named' });
    T.eq(cm.assemble().retrieved, ['lunch is at noon on fridays'], 'the query is the newest USER message, not the newest message: a tool result that follows the question is not what the user asked about');
    const off = new m.ContextManager({ tokenizer: tok, budget: 60, system: 'You are terse.', policy: 'drop', topK: 0, notes });
    off.add({ role: 'user', content: 'when is lunch?' });
    T.eq(off.assemble().messages.length, 2, 'topK = 0 means no retrieval message at all');
  } },
  { step: 'assemble', name: 'memory: facts harvested from the conversation come back by retrieval after they were dropped', run(m, T) {
    const fact = 'the deploy key is in vault slot 7';
    const { cm } = converse(m, T, { policy: 'drop', topK: 2 });
    T.ok(cm.notes.includes(fact), 'add() must harvest `remember:` facts into the note store when memory is on');
    const a = cm.assemble();
    T.ok(a.tokens <= 120);
    T.eq(a.retrieved, [fact], 'the note store is what lets a dropped fact come back when the user asks about it');
    T.ok(a.messages.some((x) => x.retrieved && x.content.includes(fact)));
    const forgetful = new m.ContextManager({ tokenizer: wordTokenizer(), budget: 120, system: 's', policy: 'drop', topK: 2, memory: false });
    forgetful.add({ role: 'user', content: `remember: ${fact}` });
    T.eq(forgetful.notes, [], 'memory: false leaves the store alone');
    cm.remember(fact);
    T.eq(cm.notes.filter((x) => x === fact).length, 1, 'remember() deduplicates');
    cm.add({ role: 'assistant', content: 'vault slot 7' }).add({ role: 'user', content: 'ok.\nremember: the heron database runs on port 5432' });
    cm.add({ role: 'assistant', content: 'noted' }).add({ role: 'user', content: 'which port does heron use?' });
    T.eq(cm.assemble().retrieved, ['the heron database runs on port 5432'], 'a fact harvested AFTER an earlier assemble() must be searchable: rebuild the index whenever it is null (remember() resets it), do not build it once and keep it');
  } },
  { step: 'assemble', name: 'compact policy compacts only when the history does not fit, and still truncates tool results', run(m, T) {
    const tok = wordTokenizer();
    const roomy = new m.ContextManager({ tokenizer: tok, budget: 1000, system: 'You are terse.', policy: 'compact', keepLast: 2 });
    for (let t = 0; t < 6; t++) roomy.add({ role: 'user', content: `turn ${t}` }).add({ role: 'assistant', content: `reply ${t}` });
    const a = roomy.assemble();
    T.eq(a.messages.filter((x) => x.summary).length, 0, 'the history fits the budget, so nothing is compacted: compaction loses verbatim text and changes the cached prefix, so it only runs when needed');
    T.eq(a.messages.length, 13, 'system prompt + all 12 history messages');
    const tight = new m.ContextManager({ tokenizer: tok, budget: 60, system: 'You are terse.', policy: 'compact', keepLast: 4, maxToolTokens: 10 });
    for (let t = 0; t < 6; t++) tight.add({ role: 'user', content: `turn ${t} ${wordsOf(4, 'x')}` }).add({ role: 'assistant', content: `reply ${t}` });
    tight.add({ role: 'user', content: 'search please' }).add({ role: 'tool', name: 'search', content: wordsOf(30, 'r') });
    const b = tight.assemble();
    T.ok(b.tokens <= 60, `over budget: ${b.tokens} > 60`);
    T.eq(b.messages.filter((x) => x.summary).length, 1, 'this history does not fit, so it is compacted');
    const tool = b.messages.find((x) => x.role === 'tool');
    T.ok(tool && m.countTokens(tok, tool.content) <= 10, 'compact the TRUNCATED history: a kept tool result must still be cut to maxToolTokens');
  } },
];
