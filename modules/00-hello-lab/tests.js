export const tests = [
  { step: 'tokenize', name: 'splits a sentence into lowercase words', run(m, T) {
    T.eq(m.tokenizeWords('Alice was beginning to get very tired.'), ['alice', 'was', 'beginning', 'to', 'get', 'very', 'tired'], 'words should be lowercased and punctuation dropped');
  } },
  { step: 'tokenize', name: 'keeps apostrophes inside words and splits on hyphens and digits', run(m, T) {
    T.eq(m.tokenizeWords("Alice's well-lit room, 3 chairs"), ["alice's", 'well', 'lit', 'room', 'chairs']);
  } },
  { step: 'tokenize', name: 'returns an empty array for text with no words', run(m, T) {
    T.eq(m.tokenizeWords('123 ... !!!'), [], 'no letters means no words (watch out: String.match returns null)');
  } },
  { step: 'count', name: 'counts repeated words', run(m, T) {
    const c = m.countFrequencies(['a', 'b', 'a', 'c', 'a', 'b']);
    T.ok(c instanceof Map, 'should return a Map');
    T.eq(c.get('a'), 3); T.eq(c.get('b'), 2); T.eq(c.get('c'), 1); T.eq(c.size, 3);
  } },
  { step: 'count', name: 'handles awkward keys like "constructor"', run(m, T) {
    const c = m.countFrequencies(['constructor', '__proto__', 'constructor']);
    T.eq(c.get('constructor'), 2); T.eq(c.get('__proto__'), 1);
  } },
  { step: 'topk', name: 'returns the k most frequent, most frequent first', run(m, T) {
    T.eq(m.topK(new Map([['b', 2], ['a', 2], ['c', 5], ['d', 1]]), 2), [['c', 5], ['a', 2]]);
  } },
  { step: 'topk', name: 'breaks ties alphabetically and copes with k larger than the vocabulary', run(m, T) {
    T.eq(m.topK(new Map([['pear', 1], ['apple', 1], ['fig', 1]]), 10), [['apple', 1], ['fig', 1], ['pear', 1]]);
  } },
  { step: 'zipf', name: 'predicts topCount / rank', run(m, T) {
    T.close(m.zipfPredicted(100, 4), [100, 50, 33.3333, 25], 1e-3);
    T.eq(m.zipfPredicted(7, 0), []);
  } },
  { step: 'zipf', name: 'log error is zero for a perfect Zipf sequence and positive otherwise', run(m, T) {
    const ideal = m.zipfPredicted(64, 6);
    T.close(m.zipfLogError(ideal, ideal), 0, 1e-9);
    T.close(m.zipfLogError([64, 64, 64], m.zipfPredicted(64, 3)), (Math.log(2) + Math.log(3)) / 3, 1e-6, 'mean of |log a - log p|');
  } },
];
