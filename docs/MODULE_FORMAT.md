# Module format (authoring contract)

A module is a directory `modules/<NN>-<slug>/` containing exactly these files:

| File          | Purpose |
|---------------|---------|
| `module.js`   | Default-exports the module object (content, steps, hints, quiz, reflection). |
| `starter.js`  | The file the learner starts from. Loaded into the editor. |
| `solution.js` | Reference solution. Same exports as `starter.js`. Never shown until the learner asks. |
| `tests.js`    | `export const tests = [...]` — step-scoped, deterministic tests. |
| `demo.js`     | `export default async function demo(m, lab)` — the Goal demo that runs the learner's code. |

The module must also be listed in `modules/index.js` (`MODULES` array) with `status: 'ready'`.

## Import rules (important)

* Files in `modules/` import shared code **only** with the bare `lib/` prefix:
  `import { rng } from 'lib/util.js'` and `import * as ops from 'lib/ops.js'`.
  The sandbox rewrites `lib/...` to the real URL in the browser and to a file URL in Node.
  Do not use `./` or `../` imports in module files. Do not import between module directories.
* Learner code runs in a Web Worker: no DOM, no `fetch`, no `window`. `console.log` is captured.
* All randomness goes through `rng(seed)` from `lib/util.js` so results are reproducible.

## `module.js` schema

```js
export default {
  id: '05-attention',                  // must equal the directory name
  title: 'Attention from scratch',     // artifact-oriented title
  track: 'transformer',                // a track id from modules/index.js
  minutes: 90,                         // honest estimate for a focused learner
  threshold: 'One sentence naming the threshold concept of this module.',
  goal: 'One or two sentences naming the working artifact and its observable behaviour.',
  prereqs: ['01-tensors', '02-autograd'],
  recall: [                            // 3–5 questions about EARLIER modules (retrieval practice)
    { q: 'Which dimension does softmax normalise over in lib/ops.js?',
      options: ['The first', 'The last', 'All elements'], answer: 1,
      why: 'Row-wise softmax over the last dimension is the convention throughout the lab.' },
  ],
  review: [                            // 3–5 questions about THIS module, used by the spaced review queue
    { q: '…', options: ['…', '…', '…'], answer: 0, why: '…' },
  ],
  concept: `markdown (400–900 words) with at least two :::predict blocks`,
  steps: [
    { id: 'scores',                    // stable id; tests reference it
      title: 'Similarity scores',
      instructions: `markdown: what to build, the exact signature, and WHY it exists`,
      hints: ['nudge', 'strategy', 'near-solution'],   // exactly 3, escalating, never the full code
      predict: { question: 'optional POE prompt', answer: 'what actually happens and why' },
    },
  ],
  reflection: ['2–3 open prompts that target the threshold concept'],
  stretch: ['2–4 optional extensions, each pointing at a real system'],
  timeouts: { tests: 20000, demo: 120000 },   // ms, optional (defaults shown)
};
```

### Markdown dialect

Headings, paragraphs, `**bold**`, `*italic*`, `` `code` ``, fenced code blocks, links, lists,
blockquotes and pipe tables are supported. No LaTeX: write formulas in code spans, e.g.
`` `softmax(Q·Kᵀ / sqrt(d)) · V` ``. Two custom containers:

```
:::predict
Question the learner must answer before revealing (markdown).
---
The answer and the reason (markdown).
:::

:::note
A callout.
:::
```

## `tests.js`

```js
export const tests = [
  { step: 'scores', name: 'scores are q·kᵀ scaled by 1/sqrt(d)', run(m, T) { ... } },
];
```

* `m` is the learner's module namespace (their exports). `T` is the assertion helper.
* Every step needs at least 2 tests. Tests reference only exported names.
* Tests must fail on `starter.js` (at least one per step) and pass on `solution.js`.
* Deterministic: use `T.rng(seed)`; never `Math.random()`. Each test under 2 s in Node.
* Messages should say what was expected and why it matters, not just "wrong".

`T` API: `T.ok(cond, msg)`, `T.eq(actual, expected, msg)` (deep, exact),
`T.close(actual, expected, tol = 1e-4, msg)` (numbers, nested arrays, typed arrays, `{shape,data}`
tensors — shapes must match when both have one), `T.shape(t, [..], msg)`, `T.throws(fn, msg)`,
`T.rng(seed)`, `T.fail(msg)`, `T.arr(x)` (typed array or tensor → nested plain arrays).

## `demo.js`

```js
export default async function demo(m, lab) {
  lab.log('Training…');
  for (let step = 0; step < 200; step++) {
    // ... use m.* (the learner's code)
    if (step % 10 === 0) { lab.progress(step / 200, `step ${step}`); await lab.tick(); }
  }
  lab.plot({ title: 'Loss', series: [{ name: 'train', values: losses }], xlabel: 'step', ylabel: 'loss' });
  lab.done(`Your model reached loss **${last.toFixed(3)}** and generated: "${sample}"`);
}
```

`lab` API: `lab.log(...args)`, `lab.md(markdown)`, `lab.plot({title, x?, series:[{name, values}], xlabel?, ylabel?, yscale?})`,
`lab.bar({title, labels, values})`, `lab.heatmap({title, rows, rowLabels?, colLabels?})`,
`lab.table({title, columns, rows})`, `lab.progress(fraction, label?)`, `await lab.tick()`,
`lab.check(cond, message)` (throws on failure), `lab.done(summaryMarkdown)`.

* The demo must exercise the learner's code, produce at least one visual (plot/bar/heatmap/table),
  and call `lab.done` exactly once with a summary that contains real numbers from the run.
* Must finish in under 60 s in Node on the solution (the browser is roughly 2× slower).

## Pedagogy checklist (the verifier enforces the mechanical parts)

1. **Goal first.** `goal` names an artifact and observable behaviour; `threshold` is one sentence.
2. **One idea per step**, 3–6 steps, ordered by dependency, each step's tests fail on starter and pass on solution.
3. **Faded scaffolding.** `starter.js` contains at least one fully worked function that demonstrates the
   conventions; step 1 is a completion problem; later steps give less.
4. **Hint ladder.** Exactly 3 hints per step: (1) what to think about, (2) the algorithm in words,
   (3) pseudo-code or a partial snippet. Never the full solution.
5. **Predict–Observe–Explain.** At least two `:::predict` blocks in `concept` (or step `predict` fields).
6. **Retrieval.** 3–5 `recall` questions about earlier modules (modules 00–01 may ask about general JS/math),
   and 3–5 `review` questions about this module for the spaced review queue. Every question has a `why`.
7. **Reflection.** 2–3 self-explanation prompts that target the threshold concept.
8. **Stretch.** 2–4 optional extensions, each naming the real system where the idea lives
   (GPT-2, Llama, vLLM, SGLang, Megatron-LM, DeepSpeed, DeepSeek, TensorRT-LLM, …).
9. **Demo shows the goal** visually and calls `lab.done` with real numbers.
10. **Voice.** Second person, direct, no hype. Define every symbol before using it. Prefer a concrete
    number to an adjective. Say "approximately" for hardware figures and name the source
    (e.g. "NVIDIA's H100 datasheet lists ~3.35 TB/s HBM3 bandwidth").
11. **Honesty about scale.** State plainly where the toy differs from production (e.g. "real KV caches
    are paged in 16-token blocks; ours is a growing array").
12. **Runtime budget.** All tests < 10 s total and demo < 60 s in Node on the solution.

Run `node tools/verify.mjs --module <id>` before committing. It checks the schema, runs tests on
starter and solution, and runs the demo headlessly.
