// Goal demo for "JavaScript for this lab": your functions count, rank and tabulate the letters of the
// opening of Charles Dickens's A Tale of Two Cities (1859, public domain).
const TEXT = 'It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way…';

const ALPHABET = 'abcdefghijklmnopqrstuvwxyz';

export default async function demo(m, lab) {
  lab.md(`Your code reads this sentence:\n\n> ${TEXT}`);

  // Count and rank with your step 4 and step 5 functions.
  const counts = m.countLetters(TEXT);
  lab.check(counts instanceof Map && counts.size > 10, 'countLetters should return a Map with one key per letter that appears');
  const ranked = m.rankCounts(counts);
  lab.check(Array.isArray(ranked) && ranked.length === counts.size, 'rankCounts should return one [letter, count] pair per letter');
  for (let i = 1; i < ranked.length; i++) lab.check(ranked[i - 1][1] >= ranked[i][1], `rankCounts is not sorted largest first at position ${i}`);

  // Totals with your step 2 functions. The demo counts the letters its own way too, to check yours.
  const total = m.sum(ranked.map((pair) => pair[1]));
  const expected = TEXT.toLowerCase().replace(/[^a-z]/g, '').length;
  lab.check(total === expected, `sum of the counts is ${total}, but the sentence has ${expected} letters`);
  const top = m.firstK(ranked, 10);
  lab.check(top.length === 10, 'firstK(ranked, 10) should give 10 pairs');

  lab.bar({ title: 'The 10 most common letters (counted by your countLetters, ranked by your rankCounts)', labels: top.map((pair) => pair[0]), values: top.map((pair) => pair[1]) });

  // A table of labels, shares and surprises with your step 1 functions, numbered with your range.
  const ranks = m.range(top.length);
  lab.check(ranks.length === top.length && ranks[0] === 0, 'range(n) should give 0, 1, ..., n - 1');
  lab.table({
    title: 'Each letter as label(), share() and surprise() see it',
    columns: ['rank', 'label(letter, count)', 'share', 'surprise (nats)'],
    rows: ranks.map((i) => [i + 1, m.label(top[i][0], top[i][1]), +m.share(top[i][1], total).toFixed(3), +m.surprise(top[i][1], total).toFixed(3)]),
  });

  // Letters that never appear have a share of 0 and so an infinite surprise.
  const missing = [...ALPHABET].filter((ch) => !counts.has(ch));
  const missingSurprise = missing.length ? m.surprise(0, total) : null;
  if (missing.length) lab.check(missingSurprise === Infinity, 'surprise(0, total) should be Infinity');

  // Store the top 5 in a { shape, data } table and read it back with cell().
  const five = m.firstK(ranked, 5);
  const values = [...five.map((pair) => pair[1]), ...five.map((pair) => m.surprise(pair[1], total))];
  const t = m.makeTable(2, 5, values);
  lab.check(m.sameShape(t.shape, [2, 5]), `makeTable(2, 5, ...) should have shape [2, 5], got ${JSON.stringify(t.shape)}`);
  lab.check(t.data instanceof Float32Array, 'makeTable should store its numbers in a Float32Array');
  const cols = m.range(5);
  lab.table({
    title: 'makeTable(2, 5, values), read back with cell(t, row, col): row 0 holds counts, row 1 surprises (float32, about 7 digits)',
    columns: ['row', ...five.map((pair) => pair[0])],
    rows: [['0: count', ...cols.map((c) => m.cell(t, 0, c))], ['1: surprise', ...cols.map((c) => m.cell(t, 1, c))]],
  });
  lab.check(m.cell(t, 1, 0) === Math.fround(m.surprise(five[0][1], total)), 'cell(t, 1, 0) should read the first surprise back');
  let threw = false;
  try { m.makeTable(2, 5, values.slice(1)); } catch { threw = true; }
  lab.check(threw, 'makeTable should throw when it is given 9 values for a 2 x 5 table');

  const [first, second] = ranked;
  lab.md(`Module 00 does the same thing with **words** instead of letters, on a longer passage, and finds a pattern in the ranking called Zipf's law. Module 01 grows your \`{ shape, data }\` table into a full tensor library.`);
  lab.done(`Your functions counted **${total}** letters, **${counts.size}** different ones. The most common is "${first[0]}" (${first[1]} times, a share of ${m.share(first[1], total).toFixed(3)} and a surprise of **${m.surprise(first[1], total).toFixed(2)}** nats), then "${second[0]}" (${second[1]} times). ${missing.length ? `${missing.length} letters never appear (${missing.join(', ')}), so their surprise is ${missingSurprise}.` : 'Every letter appears at least once.'} Your 2 x 5 table holds ${t.data.length} numbers, and it refused a list of 9.`);
}
