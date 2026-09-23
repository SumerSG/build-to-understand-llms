# Build to Understand LLMs

A build-it-yourself curriculum for the whole LLM stack, running entirely in your browser.
Thirty-five self-contained projects: you write the code, the tests check it, and a goal demo
draws what your code just did. Nothing counts as learned until it runs.

| Track | You build |
|---|---|
| Foundations | a Zipf word counter (the lab loop), a tensor library, an autograd engine, a BPE tokenizer, a bigram language model |
| Transformer & pre-training | attention, a GPT, the training loop, a compute/scaling planner, a data pipeline, a mixture-of-experts layer, a tiny multimodal model |
| Post-training | SFT with loss masking, LoRA adapters, knowledge distillation, reward models and DPO, GRPO with verifiable rewards, an eval harness |
| Inference | sampling, a KV cache, continuous batching with paged attention, prefix/prompt caching, speculative decoding, test-time compute (voting, verifiers, reward-guided search), RoPE/GQA/MLA/sliding-window attention, long-context evaluation, quantisation |
| Harnesses & agents | an agent tool loop, context management with BM25 retrieval, grammar-constrained decoding |
| Systems & data centers | roofline analysis, data/tensor/pipeline parallelism simulators, node & interconnect placement, a serving cluster with disaggregated prefill/decode |
| Capstone | a chat with a model trained in this lab |

Planned next: a data-center deep dive: programming the collectives, a simulated network fabric, and
cluster operations (design in [docs/ROADMAP.md](docs/ROADMAP.md)).

## Run it

No build step and no npm dependencies: the only external asset is CodeMirror 5, loaded from a CDN for the
code editor (offline or with the CDN blocked, the editor falls back to a plain textarea; everything else
still works). Serve the folder over HTTP (module workers need it):

```bash
node tools/serve.mjs        # then open http://localhost:8000
```

or use any static server (`python3 -m http.server`). Progress is saved in your browser's localStorage;
the home page can export/import it as JSON. In the editor, Ctrl+Enter (Cmd+Enter on macOS) checks the
current step.

## How it teaches

Every module follows **Recall → Concept → Build → Goal → Reflect**, and each phase maps to an
evidence-based learning principle (retrieval practice, predict–observe–explain, constructionism, mastery
learning, faded worked examples, scaffolding, self-explanation, spaced review). See
[docs/PEDAGOGY.md](docs/PEDAGOGY.md).

## Repository layout

```
index.html          the app shell
app/                UI, sandbox worker, test kit, charts, markdown, storage
lib/                the shared reference implementation modules build on (tensor, autograd, tokenizer, GPT, …)
modules/NN-slug/    one directory per module: module.js, starter.js, solution.js, tests.js, demo.js
tools/verify.mjs    headless verifier: schema + pedagogy checklist, tests on starter & solution, demo run
tools/e2e.mjs       drives every module through the real UI in headless Chromium (needs playwright-core)
tools/pretrain.mjs  trains the chat playground's checkpoint
tools/serve.mjs     zero-dependency dev server
docs/               PEDAGOGY.md, MODULE_FORMAT.md, LIB_API.md
```

## Verify

```bash
node tools/verify.mjs                      # every module: schema, tests on starter & solution, demo
node tools/verify.mjs --module 05-attention
node --test 'lib/tests/*.test.mjs'                     # shared library tests
npm i -D playwright-core && npm run e2e    # drive every module through the real UI in headless Chromium
npm run e2e -- --base http://localhost:8080/build-to-understand-llms/   # same, against a server hosting a subfolder like GitHub Pages
```

## Chat playground

The sidebar's **Chat playground** runs the lab's own model in the page: the BPE tokenizer, the GPT trained
by `tools/pretrain.mjs`, a KV cache reused across turns, and the sampling pipeline, wrapped in the chat
template. It is a ~100k-parameter model trained on a toy corpus, so it produces corpus-like text rather than
answers; the point is that every piece of it is something you built in the modules. You can load a
checkpoint you trained yourself: the model JSON that `node tools/pretrain.mjs` writes.

## Writing a module

Read [docs/MODULE_FORMAT.md](docs/MODULE_FORMAT.md) (the authoring contract) and
[docs/LIB_API.md](docs/LIB_API.md) (what you can import). `modules/01-tensors` is the reference example.

## License

MIT.
