// Exhaustive check of pass@k: enumerate every k-subset of n samples, of which the first c succeed.
function bruteForcePassAtK(n, c, k) {
  let total = 0, hit = 0;
  const idx = Array.from({ length: k }, (_, i) => i);
  for (;;) {
    total++;
    if (idx[0] < c) hit++; // idx is ascending, so the subset contains a success iff its smallest index < c
    let i = k - 1;
    while (i >= 0 && idx[i] === n - k + i) i--;
    if (i < 0) break;
    idx[i]++;
    for (let j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
  }
  return hit / total;
}

function bruteForcePassPowK(n, c, k) {
  let total = 0, hit = 0;
  const idx = Array.from({ length: k }, (_, i) => i);
  for (;;) {
    total++;
    if (idx[k - 1] < c) hit++; // all chosen indices are successes iff the largest one is < c
    let i = k - 1;
    while (i >= 0 && idx[i] === n - k + i) i--;
    if (i < 0) break;
    idx[i]++;
    for (let j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
  }
  return hit / total;
}

const evenTasks = (n) => Array.from({ length: n }, (_, i) => ({ id: `q${i}`, question: `q${i}`, answer: String(i) }));

export const tests = [
  // ---------- step 1: graders ----------
  { step: 'graders', name: 'normalizeAnswer handles case, whitespace, trailing period and number formatting', run(m, T) {
    T.eq(m.normalizeAnswer('  The  Answer '), 'the answer', 'lowercase, trim, collapse internal whitespace');
    T.eq(m.normalizeAnswer('42.'), '42', 'a trailing period must be removed');
    T.eq(m.normalizeAnswer('1,000'), '1000', 'thousands separators must be removed');
    T.eq(m.normalizeAnswer('1,234,567'), '1234567', 'every thousands separator, not only the first');
    T.eq(m.normalizeAnswer('42.0'), '42', 'trailing zeros after the decimal point are not meaningful');
    T.eq(m.normalizeAnswer('3.50'), '3.5');
    T.eq(m.normalizeAnswer('10.05'), '10.05', 'an interior zero is meaningful: 10.05 must not become 10.5');
    T.eq(m.normalizeAnswer('12,3456'), '12,3456', 'a comma that is not a thousands separator must stay');
  } },
  { step: 'graders', name: 'exactMatch compares normalised strings and returns the number 0 or 1', run(m, T) {
    T.eq(m.exactMatch('42.', '42'), 1, 'normalise both sides before comparing');
    T.eq(m.exactMatch('The Answer', ' the  answer '), 1);
    T.eq(m.exactMatch('1,188', '1188'), 1, 'a model that writes 1,188 for 1188 is right');
    T.eq(m.exactMatch('420', '42'), 0, 'exact match is exact: 420 is not 42');
    T.eq(m.exactMatch('4 2', '42'), 0, 'internal whitespace is collapsed, not removed');
    T.eq(m.exactMatch('forty-two', '42'), 0);
  } },
  { step: 'graders', name: 'extractFinal takes the text after the last #### or else the last number', run(m, T) {
    T.eq(m.extractFinal('#### 1,188'), '1,188', 'the GSM8K convention: the final answer follows ####');
    T.eq(m.extractFinal('x = 5 #### 7 #### 9'), '9', 'the LAST #### marks the final answer');
    T.eq(m.extractFinal('The answer is 42.'), '42', 'without ####, take the last number');
    T.eq(m.extractFinal('I think 41, though maybe 42'), '42');
    T.eq(m.extractFinal('There are 3 boxes, so 3 * 12 = 36 apples'), '36', 'the last number, not the first');
    T.eq(m.extractFinal('none'), 'none', 'no number: return the trimmed text');
    T.eq(m.exactMatch(m.extractFinal('The answer is 1,188.'), '1188'), 1, 'extract then normalise then compare');
  } },
  { step: 'graders', name: 'regexMatch respects the pattern and returns 0/1', run(m, T) {
    T.eq(m.regexMatch('The answer is 42.', '\\b42\\b'), 1);
    T.eq(m.regexMatch('The answer is 420.', '\\b42\\b'), 0, 'a word boundary in the pattern must be honoured: 420 is not 42');
    T.eq(m.regexMatch('answer: 42', /^answer: \d+$/), 1, 'a RegExp object must work as well as a string');
    T.eq(m.regexMatch('nothing here', '\\d'), 0);
    T.eq(m.regexMatch('Forty-Two', 'forty-two'), 1, 'string patterns are case-insensitive');
  } },

  // ---------- step 2: pass@k ----------
  { step: 'passk', name: 'passAtK equals the exhaustive fraction of k-subsets containing a success', run(m, T) {
    for (const [n, c, k] of [[10, 3, 5], [10, 1, 1], [8, 2, 3], [12, 5, 4], [6, 6, 2], [9, 0, 3], [10, 7, 7]]) {
      T.close(m.passAtK(n, c, k), bruteForcePassAtK(n, c, k), 1e-9, `passAtK(${n}, ${c}, ${k}) must be 1 - C(n-c,k)/C(n,k), not the naive 1-(1-c/n)^k`);
    }
    T.close(m.passAtK(10, 3, 1), 0.3, 1e-9, 'pass@1 is the plain success rate c/n');
    T.ok(Math.abs(m.passAtK(10, 3, 5) - (1 - 0.7 ** 5)) > 0.05, 'the unbiased estimator differs from 1 - (1 - c/n)^k; here 0.917 vs 0.832');
  } },
  { step: 'passk', name: 'passAtK edge cases: c=0, c=n, k>n-c, invalid k', run(m, T) {
    T.eq(m.passAtK(10, 0, 5), 0, 'no correct samples: pass@k is 0');
    T.eq(m.passAtK(10, 10, 1), 1, 'all correct: pass@k is 1');
    T.eq(m.passAtK(10, 8, 3), 1, 'when n - c < k every k-subset contains a success');
    T.throws(() => m.passAtK(10, 3, 11), 'k > n must throw: you cannot draw 11 of 10 samples');
  } },
  { step: 'passk', name: 'passAtK is stable for n=1000 (naive binomials overflow to Infinity/NaN)', run(m, T) {
    const v = m.passAtK(1000, 1, 500);
    T.ok(Number.isFinite(v), 'C(1000, 500) is about 2.7e299: compute the ratio in log space');
    T.close(v, 0.5, 1e-6, 'one success in 1000 samples, 500 drawn: exactly 1/2');
    T.close(m.passAtK(1000, 500, 1), 0.5, 1e-9);
    T.close(m.logChoose(5, 2), Math.log(10), 1e-9, 'logChoose(5, 2) = log C(5,2) = log 10');
    T.close(m.logChoose(1000, 500), 689.4672615678504, 1e-6);
  } },
  { step: 'passk', name: 'passPowK is the unbiased C(c,k)/C(n,k), not (c/n)^k', run(m, T) {
    for (const [n, c, k] of [[10, 3, 2], [10, 3, 3], [8, 8, 3], [8, 2, 3], [12, 6, 1]]) {
      T.close(m.passPowK(n, c, k), bruteForcePassPowK(n, c, k), 1e-9, `passPowK(${n}, ${c}, ${k}) must be C(c,k)/C(n,k)`);
    }
    T.close(m.passPowK(10, 3, 3), 1 / 120, 1e-9, 'C(3,3)/C(10,3) = 1/120, whereas (0.3)^3 = 0.027 overstates it');
    T.eq(m.passPowK(10, 2, 3), 0, 'fewer successes than k: no k-subset is all-correct');
  } },

  // ---------- step 3: bootstrap CI ----------
  { step: 'bootstrap', name: 'bootstrapCI brackets the mean, is deterministic for a seed, and resamples with next', run(m, T) {
    const scores = Array.from({ length: 100 }, (_, i) => (i < 50 ? 1 : 0));
    const a = m.bootstrapCI(scores, { next: T.rng(1) });
    const b = m.bootstrapCI(scores, { next: T.rng(1) });
    T.close(a.mean, 0.5, 1e-9, 'mean must be the plain mean of the scores');
    T.ok(a.lo <= a.mean && a.mean <= a.hi, `interval [${a.lo}, ${a.hi}] must contain the mean ${a.mean}`);
    T.ok(a.lo < a.mean && a.mean < a.hi, 'with 50/100 correct the interval must have positive width on both sides');
    T.eq([a.lo, a.hi, a.se], [b.lo, b.hi, b.se], 'the same seed must give the same interval (reproducibility)');
    const c = m.bootstrapCI(scores, { next: T.rng(2) });
    T.ok(a.se !== c.se || a.lo !== c.lo || a.hi !== c.hi, 'a different seed should change the resamples: draw indices with the supplied next');
    const k = m.bootstrapCI([1, 1, 1, 1, 1], { next: T.rng(1) });
    T.eq([k.mean, k.lo, k.hi], [1, 1, 1], 'constant scores: every resample has the same mean, so the interval has zero width');
  } },
  { step: 'bootstrap', name: 'interval width matches the binomial standard error and shrinks like 1/sqrt(tasks)', run(m, T) {
    const half = (n) => Array.from({ length: n }, (_, i) => (i < n / 2 ? 1 : 0));
    const r100 = m.bootstrapCI(half(100), { next: T.rng(3) });
    T.ok(r100.se > 0.04 && r100.se < 0.06, `se for p=0.5, n=100 should be about sqrt(0.25/100)=0.05, got ${r100.se}`);
    const w100 = r100.hi - r100.lo;
    T.ok(w100 > 0.15 && w100 < 0.25, `95% interval width for p=0.5, n=100 should be about 2*1.96*0.05=0.196, got ${w100}`);
    const w20 = m.bootstrapCI(half(20), { next: T.rng(4) });
    const w320 = m.bootstrapCI(half(320), { next: T.rng(5) });
    const ratio = (w20.hi - w20.lo) / (w320.hi - w320.lo);
    T.ok(ratio > 3 && ratio < 5.5, `16x more tasks should shrink the interval about 4x, got ratio ${ratio.toFixed(2)}`);
  } },
  { step: 'bootstrap', name: 'alpha and B are honoured', run(m, T) {
    const scores = Array.from({ length: 60 }, (_, i) => (i % 3 === 0 ? 1 : 0));
    const wide = m.bootstrapCI(scores, { next: T.rng(7), alpha: 0.05 });
    const narrow = m.bootstrapCI(scores, { next: T.rng(7), alpha: 0.5 });
    T.ok(narrow.hi - narrow.lo < wide.hi - wide.lo, 'a 50% interval must be narrower than a 95% interval');
    const one = m.bootstrapCI(scores, { next: T.rng(7), B: 1 });
    T.eq(one.lo, one.hi, 'with B=1 there is a single resampled mean, so lo === hi');
    T.close(one.mean, 1 / 3, 1e-9);
  } },

  // ---------- step 4: judges ----------
  { step: 'judge', name: 'judgeGrader parses the verdict text; INCORRECT is not CORRECT', run(m, T) {
    const calls = [];
    const judge = (question, answer, reference) => { calls.push([question, answer, reference]); return answer === 'yes' ? 'Verdict: CORRECT.' : 'Verdict: INCORRECT.'; };
    const g = m.judgeGrader(judge);
    T.eq(g('yes', 'ref', 'q?'), 1, 'a CORRECT verdict scores 1');
    T.eq(g('no', 'ref', 'q?'), 0, 'INCORRECT contains the letters CORRECT: match the whole word, not a substring');
    T.eq(calls[0], ['q?', 'yes', 'ref'], 'the judge is called as judge(question, answer, reference)');
    T.eq(m.judgeGrader(() => 'the response is correct')('a', 'b', 'c'), 1, 'case-insensitive');
    T.eq(m.judgeGrader(() => 'I cannot decide')('a', 'b', 'c'), 0, 'no verdict word: score 0');
  } },
  { step: 'judge', name: 'agreementRate is the fraction of records on which two graders give the same verdict', run(m, T) {
    const records = [
      { question: 'q1', answer: '42', reference: '42' },
      { question: 'q2', answer: '41', reference: '42' },
      { question: 'q3', answer: 'The answer is 42', reference: '42' },
      { question: 'q4', answer: '7', reference: '8' },
    ];
    const strict = (a, r) => (a === r ? 1 : 0);
    const lenient = (a, r) => (a.includes(r) ? 1 : 0);
    T.close(m.agreementRate(records, strict, lenient), 0.75, 1e-9, 'they disagree only on q3, so 3 of 4 agree');
    T.close(m.agreementRate(records, strict, strict), 1, 1e-9, 'a grader always agrees with itself');
    T.close(m.agreementRate(records, strict, () => 1), 0.25, 1e-9, 'an always-1 grader agrees only on q1');
  } },
  { step: 'judge', name: 'positionBias runs every pair in both orders and measures consistency and first-slot wins', run(m, T) {
    const pairs = [
      { question: 'q1', a: 'long correct answer 42', b: '41', reference: '42' },
      { question: 'q2', a: '7', b: 'the answer is 8', reference: '8' },
      { question: 'q3', a: 'x', b: 'y', reference: '1' },
    ];
    const seen = [];
    const alwaysA = (q, a, b) => { seen.push([a, b]); return 'A'; };
    const r1 = m.positionBias(alwaysA, pairs);
    T.eq(seen.length, 6, 'each pair must be judged twice, once in each order');
    T.ok(seen.some((s) => s[0] === '41' && s[1] === 'long correct answer 42'), 'the swapped order (b, a) must be presented too');
    T.close(r1.consistency, 0, 1e-9, 'a judge that always picks A never agrees with itself after the swap');
    T.close(r1.firstWinRate, 1, 1e-9, 'it picks the first slot 100% of the time');
    const byLength = (q, a, b) => (a.length >= b.length ? (a.length === b.length ? 'tie' : 'A') : 'B');
    const r2 = m.positionBias(byLength, pairs);
    T.close(r2.consistency, 1, 1e-9, 'a judge that scores content only is consistent under swapping');
    T.close(r2.firstWinRate, 0.5, 1e-9, 'q3 is a tie both ways: ties count for neither side, so 2 of the 4 decided calls chose A');
    const r3 = m.positionBias(() => 'tie', pairs);
    T.close(r3.consistency, 1, 1e-9, 'tie both ways is a consistent verdict');
    T.close(r3.firstWinRate, 0.5, 1e-9, 'a judge that only ties has no slot preference; do not divide by 2*pairs and report 0');
  } },

  // ---------- step 5: contamination and the runner ----------
  { step: 'runner', name: 'ngrams returns the set of consecutive word n-grams, case- and punctuation-insensitive', run(m, T) {
    const g = m.ngrams('The cat sat, the CAT sat.', 2);
    T.eq([...g].sort(), ['cat sat', 'sat the', 'the cat'], 'lowercase tokens, punctuation dropped, duplicates collapsed');
    T.eq(m.ngrams('a b c d e', 3).size, 3, 'a 5-word text has 5 - 3 + 1 = 3 trigrams');
    T.eq(m.ngrams('a b c', 4).size, 0, 'text shorter than n has no n-grams');
    T.eq(m.ngrams('x y', 1).size, 2);
  } },
  { step: 'runner', name: 'contamination flags tasks whose prompt shares a 13-gram with the corpus', run(m, T) {
    const tasks = m.makeTasks(10, 1);
    const eleven = m.wordTokens(tasks[3].question).slice(0, 12).join(' ');
    const corpus = ['Some ordinary training text about apples and boxes and friends.', tasks[0].question, tasks[5].question, eleven].join(' ');
    const r = m.contamination(tasks, corpus);
    T.eq(r.flagged, ['t00', 't05'], 'exactly the two prompts copied verbatim are flagged; a 12-word overlap does not make a 13-gram');
    T.close(r.rate, 0.2, 1e-9, 'rate is flagged / total tasks');
    T.eq(m.contamination(tasks, corpus, 5).flagged.includes('t03'), true, 'with n=5 the 12-word overlap is caught');
    T.eq(m.contamination(tasks, '', 13).flagged, [], 'an empty corpus flags nothing');
  } },
  { step: 'runner', name: 'runEval samples n answers per task, counts c, and reports pass@k with CIs', run(m, T) {
    const tasks = evenTasks(8);
    let calls = 0;
    const model = (q) => { calls++; return `The answer is ${q.slice(1)}.`; };
    const grader = (a, ref) => m.exactMatch(m.extractFinal(a), ref);
    const rep = m.runEval({ tasks, model, grader, n: 4, ks: [1, 2, 4], next: T.rng(1), B: 100 });
    T.eq(calls, 32, '8 tasks x 4 samples = 32 model calls');
    T.eq(rep.results.length, 8);
    T.eq(rep.results.map((r) => r.c), [4, 4, 4, 4, 4, 4, 4, 4], 'every sample is correct, so c = n for each task');
    T.eq(rep.results[2].answers.length, 4, 'keep the sampled answers so graders can be compared later');
    T.eq(rep.metrics.map((x) => x.k), [1, 2, 4]);
    for (const x of rep.metrics) {
      T.close(x.value, 1, 1e-9, `pass@${x.k} must be 1 for an always-correct model`);
      T.close([x.lo, x.hi], [1, 1], 1e-9, 'the bootstrap interval of all-ones is [1, 1]');
    }
    T.throws(() => m.runEval({ tasks, model, grader, n: 4, ks: [8], next: T.rng(1) }), 'pass@8 with only 4 samples per task must throw');
  } },
  { step: 'runner', name: 'runEval with a 30%-accurate stochastic model: pass@10 far exceeds pass@1 and CIs bracket the estimate', run(m, T) {
    const tasks = evenTasks(40);
    const model = (q, next) => (next() < 0.3 ? q.slice(1) : 'no idea');
    const grader = (a, ref) => m.exactMatch(a, ref);
    const rep = m.runEval({ tasks, model, grader, n: 10, ks: [1, 5, 10], next: T.rng(9), B: 300 });
    const [p1, p5, p10] = rep.metrics.map((x) => x.value);
    T.ok(p1 > 0.2 && p1 < 0.4, `pass@1 should be near 0.3, got ${p1}`);
    T.ok(p10 > 0.85, `pass@10 should be near 1 - 0.7^10 = 0.97, got ${p10}`);
    T.ok(p1 < p5 && p5 < p10, 'pass@k must increase with k');
    for (const x of rep.metrics) {
      T.ok(x.lo <= x.value && x.value <= x.hi, `pass@${x.k}: CI [${x.lo}, ${x.hi}] must bracket ${x.value}`);
      T.ok(x.hi - x.lo < 0.4, 'a 40-task interval should not be absurdly wide');
    }
    T.ok(rep.metrics[0].hi - rep.metrics[0].lo > 0.05, 'pass@1 over 40 tasks has real uncertainty: the interval must not collapse');
    const expectedPass1 = rep.results.reduce((s, r) => s + r.c / r.n, 0) / rep.results.length;
    T.close(p1, expectedPass1, 1e-9, 'pass@1 is the mean over tasks of c/n');
    T.ok(rep.metrics[2].passPowK < 0.01, `pass^10 for a 30% model is about 0.3^10 = 6e-6, got ${rep.metrics[2].passPowK}`);
  } },
];
