// A BPE tokenizer demo: train your BPE tokenizer on the lab corpus at several vocabulary sizes, plot the
// compression curve, and look at what the tokens actually are.
import { CORPUS } from 'lib/data.js';

const show = (s) => s.replace(/ /g, '␣').replace(/\n/g, '⏎');

export default async function demo(m, lab) {
  const chars = CORPUS.length;
  const distinct = new Set(CORPUS).size;
  const pre = m.pretokenize(CORPUS);
  lab.check(pre.length > 1000, `pretokenize returned only ${pre.length} pre-tokens for a ${chars}-character corpus`);
  lab.check(pre.join('') === CORPUS, 'pre-tokens must concatenate back to the corpus');
  lab.log(`Corpus: ${chars} characters, ${distinct} distinct characters, ${pre.length} pre-tokens (${new Set(pre).size} distinct).`);

  // 1. Compression vs vocabulary size (one training run per size).
  const sizes = [96, 128, 192, 256, 384, 512];
  const curve = [];
  const trainMs = [];
  const tokenizers = [];
  for (let i = 0; i < sizes.length; i++) {
    const t0 = performance.now();
    const tok = m.BPETokenizer.train(CORPUS, { vocabSize: sizes[i] });
    trainMs.push(performance.now() - t0);
    tokenizers.push(tok);
    lab.check(tok.vocabSize === sizes[i], `asked for vocabSize ${sizes[i]} but got ${tok.vocabSize}`);
    lab.check(m.roundTripExact(tok, CORPUS), `decode(encode(corpus)) differs from the corpus at vocabSize ${sizes[i]}`);
    const tokens = tok.encode(CORPUS).length;
    curve.push({ vocabSize: sizes[i], tokens, charsPerToken: m.charsPerToken(tok, CORPUS) });
    lab.progress((i + 1) / sizes.length, `vocab ${sizes[i]}: ${tokens} tokens`);
    await lab.tick();
  }
  const fromCurve = m.compressionCurve(CORPUS, [128, 256]);
  lab.check(fromCurve.length === 2 && fromCurve[1].tokens === curve[3].tokens, 'compressionCurve must agree with training directly');

  lab.plot({
    title: 'Compression vs vocabulary size (trained on the lab corpus)',
    x: sizes,
    series: [{ name: 'characters per token', values: curve.map((c) => +c.charsPerToken.toFixed(3)) }],
    xlabel: 'vocabulary size', ylabel: 'chars / token',
  });
  lab.plot({
    title: `Tokens needed for the ${chars}-character corpus`,
    x: sizes,
    series: [{ name: 'tokens', values: curve.map((c) => c.tokens) }, { name: 'characters (no merges)', values: sizes.map(() => chars) }],
    xlabel: 'vocabulary size', ylabel: 'tokens',
  });
  lab.table({
    title: 'One training run per size',
    columns: ['vocab', 'merges', 'tokens', 'chars / token', 'tokens / char', 'train ms', 'round trip'],
    rows: curve.map((c, i) => [c.vocabSize, tokenizers[i].merges.length, c.tokens, +c.charsPerToken.toFixed(3), +(1 / c.charsPerToken).toFixed(3), Math.round(trainMs[i]), 'exact']),
  });

  // 2. The first 20 merges of the largest tokenizer.
  const tok = tokenizers[tokenizers.length - 1];
  lab.table({
    title: `First 20 merges (vocab ${tok.vocabSize}); ␣ marks a space`,
    columns: ['rank', 'left', 'right', 'new symbol', 'id'],
    rows: tok.merges.slice(0, 20).map(([a, b], i) => [i, show(a), show(b), show(a + b), tok.stoi.get(a + b)]),
  });
  const lengths = tok.vocab.filter((s) => s !== m.UNK && !tok.specials.includes(s)).map((s) => s.length);
  const maxLen = Math.max(...lengths);
  lab.bar({
    title: `Symbol length in the vocab-${tok.vocabSize} vocabulary`,
    labels: Array.from({ length: maxLen }, (_, i) => String(i + 1)),
    values: Array.from({ length: maxLen }, (_, i) => lengths.filter((l) => l === i + 1).length),
  });

  // 3. Token boundaries in a sentence, and the digit pathology.
  const sentence = 'The quick student dances gladly, and 12 hungry robots count 8 green baskets!';
  const ids = tok.encode(sentence);
  lab.check(tok.decode(ids) === sentence, 'the sentence must round-trip');
  lab.md(`**Token boundaries** (vocab ${tok.vocabSize}, ${ids.length} tokens for ${sentence.length} characters):\n\n\`${ids.map((id) => show(tok.vocab[id])).join('|')}\``);
  lab.table({
    title: 'Encoded sentence',
    columns: ['#', 'id', 'token', 'chars'],
    rows: ids.map((id, i) => [i, id, show(tok.vocab[id]), tok.vocab[id].length]),
  });
  const digits = '1234567';
  const digitTokens = tok.encode(digits).map((id) => tok.vocab[id]);
  const eosIds = tok.encode(`${sentence}<|endoftext|>`);
  const digitSymbols = tok.vocab.filter((v) => /\d/.test(v) && v.length > 1);
  const digitCount = (CORPUS.match(/\d/g) ?? []).length;
  lab.md(`The number \`${digits}\` becomes ${digitTokens.length} token(s): \`${digitTokens.join('|')}\`. The corpus has only ${digitCount} digit characters, so only ${digitSymbols.length} multi-character symbol(s) contain a digit (${digitSymbols.map((v) => '`' + show(v) + '`').join(', ') || 'none'}); a model on top of this tokenizer sees most numbers digit by digit. `
    + `Appending \`<|endoftext|>\` adds exactly ${eosIds.length - ids.length} id (${tok.eos}), matched before pre-tokenisation.`);

  const c128 = curve[1], c256 = curve[3], c512 = curve[5];
  lab.done(`Your tokenizer round-trips the ${chars}-character corpus exactly at every size. `
    + `Vocab **128** needs **${c128.tokens}** tokens (${c128.charsPerToken.toFixed(2)} chars/token), vocab **256** needs **${c256.tokens}** (${c256.charsPerToken.toFixed(2)}), vocab **512** needs **${c512.tokens}** (${c512.charsPerToken.toFixed(2)}): `
    + `doubling the vocabulary from 256 to 512 cut the token count by ${(100 * (1 - c512.tokens / c256.tokens)).toFixed(0)}%, not 50%. `
    + `The first merge was "${show(tok.merges[0][0] + tok.merges[0][1])}"; the longest symbol has ${maxLen} characters; the sample sentence took ${ids.length} tokens for ${sentence.length} characters. `
    + `Training the 512-entry vocabulary took ${Math.round(trainMs[5])} ms.`);
}
