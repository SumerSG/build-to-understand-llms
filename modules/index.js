// modules/index.js — the curriculum registry. Order matters: it is the recommended path.

export const TRACKS = [
  { id: 'foundations', title: 'Foundations', blurb: 'The numerical substrate and the data interface: tensors, gradients, tokens, and the simplest language model.' },
  { id: 'transformer', title: 'Transformer & pre-training', blurb: 'Attention, the GPT architecture, the training loop, and the arithmetic of scale and data.' },
  { id: 'posttraining', title: 'Post-training', blurb: 'Turning a text predictor into an assistant: SFT, preference optimisation, RL with verifiable rewards, and evals.' },
  { id: 'inference', title: 'Inference', blurb: 'Serving tokens fast: decoding, KV caches, batching, prefix caching, speculative decoding, quantisation.' },
  { id: 'harness', title: 'Harnesses & agents', blurb: 'The code around the model: tool loops, context management, constrained decoding.' },
  { id: 'systems', title: 'Systems & data centers', blurb: 'Where it physically runs: GPUs, parallelism, nodes and interconnects, serving clusters.' },
  { id: 'capstone', title: 'Capstone', blurb: 'Assemble everything into a chat you can talk to.' },
];

export const MODULES = [
  { id: '00-hello-lab', track: 'foundations', title: "Zipf's law: learn the lab loop", minutes: 20, status: 'ready',
    goal: 'A word-frequency counter that plots Zipf\'s law on the training corpus, built through the Recall → Concept → Build → Goal → Reflect loop.' },
  { id: '01-tensors', track: 'foundations', title: 'Tensors from scratch', minutes: 90, status: 'ready',
    goal: 'A tiny tensor library (matmul, transpose, broadcasting, softmax, layernorm) whose outputs match the reference bit-for-bit within tolerance.' },
  { id: '02-autograd', track: 'foundations', title: 'Autograd from scratch', minutes: 120, status: 'ready',
    goal: 'A reverse-mode automatic differentiation engine that passes numerical gradient checks and trains a linear model by gradient descent.' },
  { id: '03-tokenizer', track: 'foundations', title: 'A BPE tokenizer', minutes: 90, status: 'ready',
    goal: 'A byte-pair-encoding tokenizer trained on the corpus that round-trips text exactly and compresses it, with a plot of compression vs vocabulary size.' },
  { id: '04-bigram', track: 'foundations', title: 'From counting to learning: the bigram model', minutes: 90, status: 'ready',
    goal: 'A count-based and a neural bigram language model; you measure perplexity for both and sample text from each.' },
  { id: '05-attention', track: 'transformer', title: 'Attention from scratch', minutes: 90, status: 'ready',
    goal: 'A causal multi-head self-attention layer that matches reference outputs and renders its own attention heatmap.' },
  { id: '06-transformer', track: 'transformer', title: 'The GPT architecture', minutes: 120, status: 'ready',
    goal: 'A complete GPT (embeddings, pre-LN blocks, MLP, tied LM head) with correct shapes and a parameter-count formula that matches the built model.' },
  { id: '07-pretraining', track: 'transformer', title: 'The pre-training loop', minutes: 120, status: 'ready',
    goal: 'A training loop (batches, AdamW, warmup+cosine schedule, gradient clipping, eval) that trains a small GPT in your browser until it generates recognisable text.' },
  { id: '08-scaling', track: 'transformer', title: 'Scaling laws & the arithmetic of compute', minutes: 75, status: 'ready',
    goal: 'A run planner that turns model size, tokens and hardware into FLOPs, memory, and wall-clock time, and finds the compute-optimal model for a budget.' },
  { id: '09-data-pipeline', track: 'transformer', title: 'The pre-training data pipeline', minutes: 90, status: 'ready',
    goal: 'A pipeline that filters, deduplicates, mixes and shards raw documents into token shards, with a report of what was kept and why.' },
  { id: '28-moe', track: 'transformer', title: 'Mixture of experts', minutes: 105, status: 'ready',
    goal: 'A sparse mixture-of-experts layer (top-k router, expert MLPs, load-balancing loss, capacity and dropped tokens) swapped into your GPT and compared with a dense block at equal active parameters.' },
  { id: '32-multimodal', track: 'transformer', title: 'Vision tokens: a tiny multimodal model', minutes: 105, status: 'ready',
    goal: 'A patch embedder and projector that feed image tokens into your GPT, a contrastive alignment head, and a caption trainer that reads synthetic shapes.' },
  { id: '10-sft', track: 'posttraining', title: 'Supervised fine-tuning', minutes: 90, status: 'ready',
    goal: 'A chat-formatted SFT trainer with assistant-only loss masking that turns the pre-trained model into one that follows the chat template.' },
  { id: '30-lora', track: 'posttraining', title: 'Parameter-efficient fine-tuning (LoRA)', minutes: 90, status: 'ready',
    goal: 'Low-rank adapters wrapped around your GPT\'s projections, trained on a few percent of the parameters, merged back into plain weights, and compared with full fine-tuning.' },
  { id: '31-distillation', track: 'posttraining', title: 'Knowledge distillation', minutes: 90, status: 'ready',
    goal: 'A distillation trainer (temperature-softened targets, KL loss, sequence-level and on-policy data) that trains a smaller student to track the checkpoint teacher faster than training from scratch.' },
  { id: '11-preference', track: 'posttraining', title: 'Reward models & DPO', minutes: 105, status: 'ready',
    goal: 'A Bradley–Terry reward model loss and a DPO loss that raise the implicit reward margin on preference pairs when trained.' },
  { id: '12-rlvr', track: 'posttraining', title: 'RL with verifiable rewards (GRPO)', minutes: 120, status: 'ready',
    goal: 'A GRPO trainer with group-normalised advantages and a KL penalty that raises a policy\'s accuracy on a verifiable toy task.' },
  { id: '13-evals', track: 'posttraining', title: 'An eval harness', minutes: 75, status: 'ready',
    goal: 'An eval runner with exact-match and judge graders, the unbiased pass@k estimator and bootstrap confidence intervals, producing a report table.' },
  { id: '14-decoding', track: 'inference', title: 'Decoding & sampling', minutes: 75, status: 'ready',
    goal: 'A logit-processor pipeline (temperature, top-k, top-p, min-p, repetition penalty, stop sequences) whose sampled distributions match the maths.' },
  { id: '15-kv-cache', track: 'inference', title: 'The KV cache', minutes: 90, status: 'ready',
    goal: 'Incremental decoding with a key/value cache that produces identical tokens to full recomputation while doing a fraction of the FLOPs.' },
  { id: '16-batching', track: 'inference', title: 'Continuous batching & paged attention', minutes: 105, status: 'ready',
    goal: 'A continuous-batching scheduler with a paged KV block allocator; you measure throughput, latency and memory fragmentation.' },
  { id: '17-prefix-caching', track: 'inference', title: 'Prefix caching & prompt caching', minutes: 90, status: 'ready',
    goal: 'A radix-tree prefix cache with LRU eviction and a prompt-caching cost model; you measure hit rates and cost savings on realistic traffic.' },
  { id: '18-speculative', track: 'inference', title: 'Speculative decoding', minutes: 90, status: 'ready',
    goal: 'Draft-and-verify decoding with the rejection-sampling acceptance rule that provably preserves the target distribution, with measured speedup.' },
  { id: '29-attention-variants', track: 'inference', title: 'Modern attention: RoPE, GQA, MLA, sliding windows', minutes: 105, status: 'ready',
    goal: 'Rotary positions, grouped-query and multi-head latent attention, and a sliding-window ring cache, each proven equivalent to plain attention where it should be, with the KV-cache bytes each one saves.' },
  { id: '33-long-context', track: 'inference', title: 'Long-context evaluation', minutes: 90, status: 'ready',
    goal: 'A needle-in-a-haystack and retrieval eval suite over lengths and depths, an effective-context estimator, and the compute and memory price of long prompts.' },
  { id: '19-quantization', track: 'inference', title: 'Quantisation', minutes: 90, status: 'ready',
    goal: 'Absmax, per-channel and group-wise int8/int4 quantisers with error analysis and a memory calculator for weights and KV cache.' },
  { id: '20-agent-loop', track: 'harness', title: 'The agent loop harness', minutes: 90, status: 'ready',
    goal: 'A tool-use harness (schemas, call parsing, execution, stop conditions, error handling) that completes a multi-step task with a scripted model.' },
  { id: '21-context', track: 'harness', title: 'Context management & retrieval', minutes: 90, status: 'ready',
    goal: 'A context manager with token budgeting, truncation, compaction and BM25 retrieval that keeps a long conversation under budget without losing key facts.' },
  { id: '22-structured-output', track: 'harness', title: 'Structured outputs & constrained decoding', minutes: 90, status: 'ready',
    goal: 'A grammar-constrained sampler that masks logits token by token so the model can only emit JSON matching a schema.' },
  { id: '23-gpu-roofline', track: 'systems', title: 'GPUs, memory bandwidth & the roofline', minutes: 90, status: 'ready',
    goal: 'A roofline calculator and a tiled matmul; you show why decoding is memory-bound and measure the tiling speedup in your browser.' },
  { id: '24-parallelism', track: 'systems', title: 'Data, tensor & pipeline parallelism', minutes: 105, status: 'ready',
    goal: 'A simulator for ring all-reduce, tensor-parallel sharding, pipeline bubbles and ZeRO memory that reports step time and memory per strategy.' },
  { id: '25-cluster', track: 'systems', title: 'Nodes, interconnects & clusters', minutes: 105, status: 'ready',
    goal: 'A topology-aware placement planner (NVLink inside a node, InfiniBand across nodes, alpha–beta collective cost model, MoE all-to-all) that picks the fastest parallel layout.' },
  { id: '26-serving', track: 'systems', title: 'Serving at scale', minutes: 105, status: 'ready',
    goal: 'A cluster simulator with prefix-aware routing, disaggregated prefill/decode and autoscaling that meets TTFT and TPOT SLOs under load.' },
  { id: '27-capstone', track: 'capstone', title: 'Capstone: chat with your own model', minutes: 120, status: 'ready',
    goal: 'Tokenizer + model + KV cache + sampler + harness wired together into a chat that runs in the page on a model trained in this lab.' },
];

export function moduleById(id) {
  return MODULES.find((m) => m.id === id) || null;
}

export function nextModule(id) {
  const i = MODULES.findIndex((m) => m.id === id);
  return i >= 0 && i + 1 < MODULES.length ? MODULES[i + 1] : null;
}

export async function loadModule(id) {
  const mod = await import(`./${id}/module.js`);
  return mod.default;
}
