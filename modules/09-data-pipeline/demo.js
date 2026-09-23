// Module 09 — goal demo: run YOUR pipeline on a small corpus with known junk, exact copies and
// near-copies planted in it, and chart what every stage removed, how MinHash tracks the exact Jaccard,
// how the mixer holds the domain shares, and what the shards look like.

import { PROSE, toyCorpus } from 'lib/data.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { rng } from 'lib/util.js';

/** Replace the words at the given positions with `repl` (a controlled near-duplicate). */
function swapWords(text, positions, repl = 'something') {
  const ws = text.split(' ');
  for (const p of positions) if (p < ws.length) ws[p] = repl;
  return ws.join(' ');
}

/** Verse is short lines; prose in PROSE is hard-wrapped, so join its lines back into one line. */
function isVerse(p) {
  const ls = p.split('\n');
  return ls.length > 1 && ls.reduce((s, l) => s + l.length, 0) / ls.length < 50;
}

export default async function demo(m, lab) {
  const tokenizer = BPETokenizer.fromJSON((await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default);
  const next = rng(9);

  // ---------- 1. the corpus: real prose and verse, toy sentences, plus planted duplicates and junk ----------
  const docs = [];
  const paras = PROSE.split(/\n\n+/);
  paras.forEach((p, i) => docs.push({ id: `book-${i}`, text: isVerse(p) ? p : p.replace(/\n/g, ' '), domain: 'books' }));
  const toyLines = toyCorpus(800, 1).split('\n');
  const toyDocs = [];
  for (let i = 0; i < 24; i++) toyDocs.push({ id: `toy-${i}`, text: toyLines.slice(i * 8, i * 8 + 8).join('\n'), domain: 'toy' });
  docs.push(...toyDocs);
  const nBase = docs.length;
  const baseText = docs.map((d) => d.text);   // captured before the shuffle below
  const prose = (i) => baseText[i];
  // Exact copies that differ only in case, punctuation or whitespace.
  const exactCopies = [
    { id: 'copy-alice', text: prose(6).toUpperCase(), domain: 'books' },
    { id: 'copy-dickens', text: '   ' + prose(24).replace(/ /g, '  ') + '\n', domain: 'books' },
    { id: 'copy-oz', text: prose(27).replace(/,/g, ''), domain: 'books' },
    { id: 'copy-toy', text: toyDocs[3].text, domain: 'toy' },
  ];
  // Near copies: one or two words changed (high Jaccard), and one with many words changed (below threshold).
  const nearCopies = [
    { id: 'near-alice', text: swapWords(prose(6), [55]), domain: 'books' },
    { id: 'near-dickens', text: swapWords(prose(24), [3]), domain: 'books' },
    { id: 'near-ishmael', text: swapWords(prose(25), [2, 60, 120]), domain: 'books' },
    { id: 'near-toy', text: swapWords(toyDocs[5].text, [1]), domain: 'toy' },
    { id: 'far-oz', text: swapWords(prose(27), [2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86]), domain: 'books' },
  ];
  // Junk that a crawl is full of.
  const junk = [
    { id: 'junk-menu', text: Array.from({ length: 24 }, (_, i) => `Menu item number ${i}`).join('\n'), domain: 'web' },
    { id: 'junk-lorem', text: prose(7) + ' Lorem ipsum dolor sit amet, consectetur adipiscing elit.', domain: 'web' },
    { id: 'junk-hashtags', text: prose(15) + ' #summer #sun #beach #fun #travel #love #happy #life #style #photo #daily', domain: 'web' },
    { id: 'junk-spam', text: Array.from({ length: 12 }, () => 'Buy now and save today.').join('\n') + '\nOffer ends soon, so hurry.', domain: 'web' },
    { id: 'junk-base64', text: Array.from({ length: 30 }, (_, i) => 'QWxpY2Ugd2FzIGJlZ2lubmluZw' + i).join(' ') + '.', domain: 'web' },
    { id: 'junk-short', text: 'Click here to continue.', domain: 'web' },
    { id: 'junk-cookies', text: prose(16) + ' By continuing you accept our cookie policy.', domain: 'web' },
  ];
  docs.push(...exactCopies, ...nearCopies, ...junk);
  // Shuffle so the planted documents are not conveniently at the end.
  for (let i = docs.length - 1; i > 0; i--) { const j = Math.floor(next() * (i + 1)); [docs[i], docs[j]] = [docs[j], docs[i]]; }
  lab.log(`${docs.length} documents: ${nBase} base (${paras.length} from PROSE, ${toyDocs.length} toy), ${exactCopies.length} exact copies, ${nearCopies.length} near copies, ${junk.length} junk`);
  await lab.tick();

  // ---------- 2. run the pipeline ----------
  const config = {
    tokenizer,
    near: { k: 5, numHashes: 64, threshold: 0.8, seed: 1, minReport: 0.25 },
    mix: { weights: { books: 0.6, toy: 0.4 }, budget: 12000, maxEpochs: 2, seed: 1 },
    shardSize: 512,
  };
  const t0 = performance.now();
  const r = m.runPipeline(docs, config);
  const ms = performance.now() - t0;
  lab.log(`pipeline ran in ${ms.toFixed(0)} ms: ${r.tokens} tokens in ${r.shards.length} shards of ${config.shardSize}`);
  await lab.tick();

  // ---------- 3. what each stage removed ----------
  lab.bar({ title: 'Documents removed per stage', labels: r.report.map((s) => s.stage), values: r.report.map((s) => s.removed) });
  lab.table({ title: 'Stage report', columns: ['stage', 'in', 'out', 'removed'], rows: r.report.map((s) => [s.stage, s.in, s.out, s.removed]) });
  const reasons = new Map();
  for (const x of r.removed) if (x.stage === 'quality') reasons.set(x.reason, (reasons.get(x.reason) || 0) + 1);
  const reasonRows = [...reasons].sort((a, b) => b[1] - a[1]);
  lab.bar({ title: 'Quality filter: documents removed per rule', labels: reasonRows.map((x) => x[0]), values: reasonRows.map((x) => x[1]) });
  const verseIds = paras.map((p, i) => (isVerse(p) ? `book-${i}` : null)).filter(Boolean);
  const verseRemoved = r.removed.filter((x) => verseIds.includes(x.id) && x.reason === 'terminal_punct').length;
  const junkCaught = junk.filter((j) => r.removed.some((x) => x.id === j.id)).length;
  lab.log(`quality: caught ${junkCaught} of ${junk.length} planted junk documents, and also removed ${verseRemoved} of ${verseIds.length} verse stanzas for terminal_punct (a C4-style line rule deletes verse)`);
  lab.table({
    title: 'Every removed document and why',
    columns: ['id', 'stage', 'reason'],
    rows: r.removed.map((x) => [x.id, x.stage, x.reason]),
  });
  await lab.tick();

  // ---------- 4. near-duplicates: MinHash estimate vs exact Jaccard ----------
  const pairRows = r.pairs.slice(0, 12).map((p) => [p.a, p.b, +p.estimate.toFixed(3), +p.jaccard.toFixed(3), p.estimate >= config.near.threshold ? 'removed' : 'kept']);
  lab.table({ title: `Near-duplicate candidates (estimate >= ${config.near.minReport}); threshold ${config.near.threshold}`, columns: ['a', 'b', 'MinHash estimate', 'exact Jaccard', 'decision'], rows: pairRows });
  const maxErr = r.pairs.reduce((s, p) => Math.max(s, Math.abs(p.estimate - p.jaccard)), 0);
  const nearCaught = nearCopies.filter((d) => r.removed.some((x) => x.id === d.id && x.stage === 'near-dedup')).length;
  lab.log(`near-dedup: ${r.pairs.length} candidate pairs, largest |estimate − exact| = ${maxErr.toFixed(3)}; removed ${nearCaught} of the ${nearCopies.length} near copies (far-oz is planted below the threshold and must stay)`);
  // How the estimate converges with the number of hash functions, on one planted pair.
  const A = m.shingles(prose(25)), B = m.shingles(nearCopies[2].text);
  const exact = m.jaccard(A, B);
  const ks = [4, 8, 16, 32, 64, 128, 256, 512];
  const est = [], std = [];
  for (const k of ks) {
    est.push(m.estimateJaccard(m.minhash(A, k, 1), m.minhash(B, k, 1)));
    std.push(Math.sqrt(exact * (1 - exact) / k));
    await lab.tick();
  }
  lab.plot({
    title: `MinHash estimate vs number of hash functions (book-25 vs near-ishmael, exact Jaccard ${exact.toFixed(3)})`,
    x: ks,
    series: [
      { name: 'estimate', values: est },
      { name: 'exact', values: ks.map(() => exact) },
      { name: 'exact + 1 std', values: std.map((s) => exact + s) },
      { name: 'exact − 1 std', values: std.map((s) => exact - s) },
    ],
    xlabel: 'hash functions k', ylabel: 'Jaccard',
  });
  await lab.tick();

  // ---------- 5. mixing: running share of books over the draw order ----------
  const sizeOf = new Map();
  for (const e of r.index) sizeOf.set(e.docId, e.length);
  const domainOf = new Map(docs.map((d) => [d.id, d.domain]));
  const share = [], xs = [];
  let books = 0, total = 0;
  r.mix.order.forEach((id, i) => {
    const n = sizeOf.get(id) + 1;
    total += n;
    if (domainOf.get(id) === 'books') books += n;
    if (i % 2 === 0 || i === r.mix.order.length - 1) { xs.push(i + 1); share.push(books / total); }
  });
  lab.plot({ title: 'Mixer: running share of books tokens vs target 0.6', x: xs, series: [{ name: 'books share', values: share }, { name: 'target', values: xs.map(() => 0.6) }], xlabel: 'documents drawn', ylabel: 'share of tokens', });
  const mixRows = Object.entries(r.mix.report).map(([d, s]) => [d, s.docs, s.draws, +s.epochs.toFixed(2), s.size, +(100 * s.share).toFixed(1), s.weight]);
  lab.table({ title: 'Tokens per domain after mixing (budget 12,000, maxEpochs 2)', columns: ['domain', 'docs', 'draws', 'epochs', 'tokens', 'share %', 'weight'], rows: mixRows });
  lab.bar({ title: 'Final tokens per domain', labels: mixRows.map((x) => x[0]), values: mixRows.map((x) => x[4]) });

  // ---------- 6. shards and the index ----------
  const firstCross = r.index.find((e) => e.offset + e.length > config.shardSize);
  const back = firstCross ? m.readDoc(r.shards, firstCross) : null;
  const okBack = back ? tokenizer.decode(back) === docs.find((d) => d.id === firstCross.docId).text : false;
  lab.table({
    title: 'Shard index (first 10 entries)',
    columns: ['docId', 'shard', 'offset', 'length', 'domain'],
    rows: r.index.slice(0, 10).map((e) => [e.docId, e.shard, e.offset, e.length, domainOf.get(e.docId)]),
  });
  if (firstCross) lab.log(`${firstCross.docId} starts at shard ${firstCross.shard} offset ${firstCross.offset} and crosses into the next shard; readDoc gives back ${back.length} tokens which decode to the original text: ${okBack}`);
  const lastShard = r.shards[r.shards.length - 1];
  lab.log(`first shard begins: ${JSON.stringify(tokenizer.decode(r.shards[0].slice(0, 12)))}…; last shard holds ${lastShard.length} tokens`);

  const q = r.report[0], e = r.report[1], nd = r.report[2];
  const bookRep = r.mix.report.books, toyRep = r.mix.report.toy;
  lab.done(
    `Your pipeline took **${docs.length} documents** to **${r.tokens} tokens** in **${r.shards.length} shards** of ${config.shardSize}. ` +
    `Quality removed **${q.removed}** (${reasonRows.map((x) => `${x[0]} ${x[1]}`).join(', ')}), including ${verseRemoved} of ${verseIds.length} verse stanzas; ` +
    `exact dedup removed **${e.removed}** and near dedup **${nd.removed}** (${nearCaught}/${nearCopies.length} planted near copies), with the MinHash estimate within ${maxErr.toFixed(3)} of the exact Jaccard on every candidate pair. ` +
    `Mixing gave books ${(100 * bookRep.share).toFixed(1)}% of the tokens over ${bookRep.epochs.toFixed(2)} epochs and toy ${(100 * toyRep.share).toFixed(1)}% over ${toyRep.epochs.toFixed(2)} epochs, against weights 0.6 / 0.4. ` +
    `Every one of those numbers is a modelling decision you can now change and re-measure.`
  );
}
