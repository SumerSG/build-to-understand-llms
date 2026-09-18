# Build to Understand LLMs

A build-it-yourself curriculum for the whole LLM stack, running entirely in your browser.
Twenty-eight self-contained projects: you write the code, the tests check it, and a goal demo
draws what your code just did. Nothing counts as learned until it runs.

| Track | You build |
|---|---|
| Foundations | a tensor library, an autograd engine, a BPE tokenizer, a bigram language model |
| Transformer & pre-training | attention, a GPT, the training loop, a compute/scaling planner, a data pipeline |
| Post-training | SFT with loss masking, reward models and DPO, GRPO with verifiable rewards, an eval harness |
| Inference | sampling, a KV cache, continuous batching with paged attention, prefix/prompt caching, speculative decoding, quantisation |
| Harnesses & agents | an agent tool loop, context management with BM25 retrieval, grammar-constrained decoding |
| Systems & data centers | roofline analysis, data/tensor/pipeline parallelism simulators, node & interconnect placement, a serving cluster with disaggregated prefill/decode |
| Capstone | a chat with a model trained in this lab |

## Run it

No build step, no dependencies. Serve the folder over HTTP (module workers need it):

```bash
node tools/serve.mjs        # then open http://localhost:8000
```

or use any static server (`python3 -m http.server`). Progress is saved in your browser's localStorage;
the home page can export/import it as JSON.

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
tools/serve.mjs     zero-dependency dev server
docs/               PEDAGOGY.md, MODULE_FORMAT.md, LIB_API.md
```

## Verify

```bash
node tools/verify.mjs                      # every module
node tools/verify.mjs --module 05-attention
node --test lib/tests/                     # shared library tests
```

## Writing a module

Read [docs/MODULE_FORMAT.md](docs/MODULE_FORMAT.md) (the authoring contract) and
[docs/LIB_API.md](docs/LIB_API.md) (what you can import). `modules/01-tensors` is the reference example.

## License

MIT.
