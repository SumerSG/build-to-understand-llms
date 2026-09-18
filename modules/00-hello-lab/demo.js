// Module 00 demo: Zipf's law on the opening of Alice's Adventures in Wonderland (public domain).
const PASSAGE = `Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"
So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.
There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.
In another moment down went Alice after it, never once considering how in the world she was to get out again.
The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.
Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled "ORANGE MARMALADE", but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody underneath, so managed to put it into one of the cupboards as she fell past it.
"Well!" thought Alice to herself, "after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!" (Which was very likely true.)
Down, down, down. Would the fall never come to an end? "I wonder how many miles I've fallen by this time?" she said aloud. "I must be getting somewhere near the centre of the earth. Let me see: that would be four thousand miles down, I think—" (for, you see, Alice had learnt several things of this sort in her lessons in the schoolroom, and though this was not a very good opportunity for showing off her knowledge, as there was no one to listen to her, still it was good practice to say it over) "—yes, that's about the right distance—but then I wonder what Latitude or Longitude I've got to?" (Alice had no idea what Latitude was, or Longitude either, but thought they were nice grand words to say.)`;

export default async function demo(m, lab) {
  const words = m.tokenizeWords(PASSAGE);
  const counts = m.countFrequencies(words);
  lab.check(words.length > 100, 'tokenizeWords returned too few words');
  const top = m.topK(counts, 40);
  lab.check(top.length >= 20, 'topK returned too few entries');
  const actual = top.map(([, c]) => c);
  const predicted = m.zipfPredicted(actual[0], actual.length);
  const err = m.zipfLogError(actual, predicted);
  const once = [...counts.values()].filter((c) => c === 1).length;
  lab.table({ title: 'Top 10 words', columns: ['rank', 'word', 'count', 'Zipf prediction'], rows: top.slice(0, 10).map(([w, c], i) => [i + 1, w, c, +predicted[i].toFixed(1)]) });
  lab.plot({ title: 'Word frequency vs rank (log scale)', x: actual.map((_, i) => i + 1), series: [{ name: 'actual', values: actual }, { name: 'Zipf: f(1)/rank', values: predicted }], xlabel: 'rank', ylabel: 'count', yscale: 'log' });
  lab.bar({ title: 'How many words appear n times', labels: ['1', '2', '3', '4', '5+'], values: [1, 2, 3, 4].map((n) => [...counts.values()].filter((c) => c === n).length).concat([[...counts.values()].filter((c) => c >= 5).length]) });
  lab.done(`Your counter found **${words.length}** words, **${counts.size}** distinct. The top word "${top[0][0]}" appears ${top[0][1]} times; **${once}** words (${(100 * once / counts.size).toFixed(0)}% of the vocabulary) appear exactly once. Mean log-error from ideal Zipf over the top ${actual.length} ranks: **${err.toFixed(3)}**.`);
}
