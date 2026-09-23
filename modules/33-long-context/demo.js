// Long-context evaluation demo: run YOUR haystack generator, graders, grid runner and effective-context estimator on a
// synthetic model that advertises an 8,192-word window, draw the accuracy heatmap and the depth curves,
// price long prompts at Llama-3-8B dims with YOUR cost model, and finally probe what the 64-token lab
// checkpoint can do with a fact placed inside (and outside) its window.
import { rng } from 'lib/util.js';
import { loadModel, forward } from 'lib/infer.js';
import { BPETokenizer } from 'lib/tokenizer.js';
import { logSoftmax } from 'lib/ops.js';

const LENGTHS = [256, 512, 1024, 2048, 4096, 8192];
const DEPTHS = [0, 0.25, 0.5, 0.75, 1];
const TRIALS = 20;
const THRESHOLD = 0.85;   // RULER's bar: Llama-2-7B's score at 4k (85.6%)
const WINDOW = 8192;

/** Evaluate one length at a time so the page stays responsive, then assemble a full result. */
async function grid(m, lab, model, kind, seed, done, total) {
  const rows = { acc: [], lo: [], hi: [] };
  for (let i = 0; i < LENGTHS.length; i++) {
    const r = m.contextEval(model, { lengths: [LENGTHS[i]], depths: DEPTHS, trials: TRIALS, kind, seed: seed + i });
    lab.check(Array.isArray(r.acc) && r.acc.length === 1 && r.acc[0].length === DEPTHS.length, `contextEval must return a 1 × ${DEPTHS.length} grid for one length`);
    rows.acc.push(r.acc[0]); rows.lo.push(r.lo[0]); rows.hi.push(r.hi[0]);
    lab.progress((done + i + 1) / total, `${kind}: ${LENGTHS[i]} words`);
    await lab.tick();
  }
  return { kind, lengths: LENGTHS.slice(), depths: DEPTHS.slice(), trials: TRIALS, ...rows };
}

async function loadCheckpoint(lab) {
  try {
    const json = (await import('lib/checkpoints/tiny-gpt.json', { with: { type: 'json' } })).default;
    const tok = (await import('lib/checkpoints/tokenizer.json', { with: { type: 'json' } })).default;
    return { model: loadModel(json), tokenizer: BPETokenizer.fromJSON(tok) };
  } catch (e) {
    lab.log(`Checkpoint not available (${String(e.message).slice(0, 100)}); skipping the lab-model probe.`);
    return null;
  }
}

/** Sum of log p over the last nTail tokens of ids, keeping only the last blockSize tokens (the window). */
function tailLogProb(model, ids, nTail) {
  const B = model.config.blockSize;
  const kept = ids.length > B ? ids.slice(ids.length - B) : ids;
  const V = model.config.vocabSize;
  const ls = logSoftmax(forward(model, kept.slice(0, -1))).data;
  let s = 0;
  for (let t = kept.length - nTail; t < kept.length; t++) s += ls[(t - 1) * V + kept[t]];
  return s;
}

export default async function demo(m, lab) {
  // ---------- 1. the grid on the synthetic model ----------
  const model = m.makeFadingModel({ window: WINDOW });
  lab.md(`**The model under test** advertises an ${WINDOW.toLocaleString()}-word window. Its recall fades with distance from the question, the first few hundred words stay visible (a caricature of attention sinks), and similar keys interfere. You do not get to see those parameters in the result; your eval has to find them.`);
  const kinds = ['niah', 'multikey', 'multivalue'];
  const total = kinds.length * LENGTHS.length;
  const results = {};
  for (let k = 0; k < kinds.length; k++) results[kinds[k]] = await grid(m, lab, model, kinds[k], 100 * (k + 1), k * LENGTHS.length, total);

  const niah = results.niah;
  lab.heatmap({
    title: `Needle-in-a-haystack accuracy (${TRIALS} trials per cell): rows = context length, columns = needle depth`,
    rows: niah.acc,
    rowLabels: LENGTHS.map((L) => `${L} words`),
    colLabels: DEPTHS.map((d) => `${Math.round(d * 100)}%`),
    min: 0, max: 1,
  });

  const probeLen = 4096;
  lab.plot({
    title: `Accuracy vs needle depth at ${probeLen} words: lost in the middle`,
    x: DEPTHS.map((d) => Math.round(d * 100)),
    series: kinds.map((k) => ({ name: k, values: m.depthCurve(results[k], probeLen) })),
    xlabel: 'needle depth (% of the context)', ylabel: 'accuracy',
  });

  const summary = kinds.map((k) => {
    const r = results[k];
    const curve = m.lengthCurve(r);
    const idx = LENGTHS.indexOf(probeLen);
    const ciLo = r.lo[idx].reduce((a, b) => a + b, 0) / DEPTHS.length, ciHi = r.hi[idx].reduce((a, b) => a + b, 0) / DEPTHS.length;
    return { kind: k, eff: m.effectiveContext(r, THRESHOLD), atWindow: curve[curve.length - 1], at4k: curve[idx], ciLo, ciHi, drop: m.middleDrop(m.depthCurve(r, probeLen)) };
  });
  lab.table({
    title: `Effective context (mean accuracy ≥ ${THRESHOLD} at every length up to it) vs the advertised ${WINDOW}`,
    columns: ['task', 'effective context (words)', `accuracy at ${probeLen}`, `cell 95% CIs at ${probeLen} (mean lo, mean hi)`, `accuracy at ${WINDOW}`, `middle drop at ${probeLen}`],
    rows: summary.map((s) => [s.kind, s.eff, +s.at4k.toFixed(3), `[${s.ciLo.toFixed(2)}, ${s.ciHi.toFixed(2)}]`, +s.atWindow.toFixed(3), +s.drop.toFixed(3)]),
  });

  // ---------- 2. the price of long prompts ----------
  const costLengths = [4096, 8192, 16384, 32768, 65536, 131072];
  const costs = costLengths.map((T) => m.longPromptCost(m.LLAMA3_8B, T));
  lab.bar({ title: 'Llama-3-8B prefill compute per prompt (PFLOPs)', labels: costLengths.map((T) => `${T / 1024}k`), values: costs.map((c) => c.flops / 1e15), ylabel: 'PFLOP' });
  lab.bar({ title: 'Llama-3-8B KV cache per sequence (GiB, bf16)', labels: costLengths.map((T) => `${T / 1024}k`), values: costs.map((c) => c.kvBytes / 2 ** 30), ylabel: 'GiB' });
  lab.table({
    title: 'Price of one long prompt at Llama-3-8B dims (H100 at ~989 TFLOP/s dense bf16, 50% MFU)',
    columns: ['tokens', 'prefill PFLOPs', 'attention share', 'prefill seconds', 'KV cache GiB'],
    rows: costs.map((c) => [c.tokens, +(c.flops / 1e15).toFixed(3), `${(100 * c.attnShare).toFixed(1)}%`, +c.seconds.toFixed(2), +(c.kvBytes / 2 ** 30).toFixed(2)]),
  });
  await lab.tick();

  // ---------- 3. the lab checkpoint inside its 64-token window ----------
  const ckpt = await loadCheckpoint(lab);
  let probeNote = 'The lab checkpoint could not be loaded, so the in-window probe was skipped.';
  if (ckpt) {
    const next = rng(33);
    const probeRows = [];
    for (const words of [2, 6, 12, 32]) {
      let gain = 0, n = 0, inside = 0, tokens = 0;
      for (let t = 0; t < 12; t++) {
        const task = m.makeTask('niah', { length: words + 7, depth: 0, next });
        const key = /for (\w+)\?/.exec(task.question)[1], value = task.answers[0];
        const query = ` The secret code for ${key} is ${value}.`;
        const withNeedle = ckpt.tokenizer.encode(task.context + query);
        const control = ckpt.tokenizer.encode(task.context.replace(/^The secret code for \w+ is \d+\. ?/, '') + query);
        const nTail = ckpt.tokenizer.encode(` ${value}.`).length;
        const nNeedle = ckpt.tokenizer.encode(`The secret code for ${key} is ${value}.`).length;
        const cut = Math.max(0, withNeedle.length - ckpt.model.config.blockSize);
        if (cut <= nNeedle - nTail) inside++;   // the needle's copy of the code is still inside the window
        const a = tailLogProb(ckpt.model, withNeedle, nTail), b = tailLogProb(ckpt.model, control, nTail);
        gain += a - b; n++; tokens += withNeedle.length;
      }
      probeRows.push([words, Math.round(tokens / n), `${inside}/${n}`, +(gain / n).toFixed(3)]);
      lab.progress(1, `lab checkpoint: ${words} filler words`);
      await lab.tick();
    }
    lab.table({
      title: 'Lab checkpoint (64-token window): log-prob gain on the code when the needle is in the prompt',
      columns: ['filler words', 'prompt tokens (mean)', 'needle code still inside the window', 'gain (nats)'],
      rows: probeRows,
    });
    probeNote = `On the real lab checkpoint the needle raised the log-probability of the code by **${probeRows[0][3]} nats** with ${probeRows[0][0]} filler words, and by **${probeRows[probeRows.length - 1][3]} nats** once the prompt grew to ${probeRows[probeRows.length - 1][1]} tokens and pushed the needle out of the 64-token window: past the window there is nothing left to retrieve.`;
  }

  const n = summary.find((s) => s.kind === 'niah'), mk = summary.find((s) => s.kind === 'multikey'), mv = summary.find((s) => s.kind === 'multivalue');
  const c128 = costs[costs.length - 1];
  lab.done(`Advertised window: **${WINDOW} words**. Your eval measured an effective context of **${n.eff}** words on single-needle retrieval, **${mk.eff}** with three distractor keys, and **${mv.eff}** when all three values must be found (threshold ${THRESHOLD}). At ${probeLen} words the needle task scored ${n.at4k.toFixed(2)} and the middle of the context sat **${n.drop.toFixed(2)}** below its edges. A 128k-token prompt at Llama-3-8B dims costs about **${(c128.flops / 1e15).toFixed(1)} PFLOPs** to prefill (${(100 * c128.attnShare).toFixed(0)}% of it attention, ~${c128.seconds.toFixed(1)} s on one H100 at 50% MFU) and holds **${(c128.kvBytes / 2 ** 30).toFixed(0)} GiB** of KV cache. ${probeNote}`);
}
