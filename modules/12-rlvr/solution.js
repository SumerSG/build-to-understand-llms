// RL with verifiable rewards (GRPO) — reference solution: GRPO on a verifiable toy environment.
//
// The policy is deliberately small: a softmax over the integers 0..99 (one "token" per completion),
// whose logits are a sum of learned rows indexed by hashed features of the question. Everything the
// module is about (verifier, rollouts, group advantages, clipped policy gradient, KL to a reference,
// the training loop) is independent of the policy's size; the same code drives a GPT in production.

import { Tensor, noGrad, crossEntropy } from 'lib/tensor.js';
import { AdamW } from 'lib/optim.js';
import { hash32, sampleIndex, softmaxArray, argmaxArray, shuffle } from 'lib/util.js';

/** The policy's answer vocabulary: completion token t means the text `${t}`. */
export const ANSWER_VOCAB = 100;
/** Number of hashed feature buckets (the "hashing trick"): each feature string picks one row of W. */
export const N_FEATURES = 1024;

// ---------- given: the policy ----------

/** "What is 7 + 3?" -> { a: 7, op: '+', b: 3 }; null when the question does not have that shape. */
export function parseQuestion(question) {
  const m = /(-?\d+)\s*([+\-*])\s*(-?\d+)/.exec(question);
  return m ? { a: Number(m[1]), op: m[2], b: Number(m[3]) } : null;
}

/** Feature ids of a question: one bucket per feature string, hashed into N_FEATURES rows. */
export function features(question) {
  const q = parseQuestion(question);
  const names = q
    ? [`a=${q.a}`, `b=${q.b}`, `op=${q.op}`, `${q.op}a=${q.a}`, `${q.op}b=${q.b}`, `q=${q.a}${q.op}${q.b}`]
    : [`raw=${question}`];
  return names.map((name) => hash32(name) % N_FEATURES);
}

/**
 * A linear softmax policy over ANSWER_VOCAB answers: logits(q) = sum of W[f] over the feature ids f of q.
 * W starts at zero, so a fresh policy is uniform over the 100 answers.
 */
export class Policy {
  constructor() {
    this.W = Tensor.param({ shape: [N_FEATURES, ANSWER_VOCAB], data: new Float32Array(N_FEATURES * ANSWER_VOCAB) });
  }

  /** Logits for a list of questions: Tensor [N, ANSWER_VOCAB], differentiable in W. */
  logits(questions) {
    const ids = questions.map(features);            // [N, k] feature ids
    return this.W.embed(ids).sum(1);                 // [N, k, V] -> [N, V]
  }

  /** log softmax of the logits: Tensor [N, ANSWER_VOCAB]. */
  logProbs(questions) {
    return this.logits(questions).logSoftmax();
  }

  /** Probabilities of every answer for one question, as a plain Float32Array (no graph). */
  probs(question) {
    return noGrad(() => softmaxArray(Array.from(this.logits([question]).data)));
  }

  parameters() {
    return [this.W];
  }

  /** An independent copy (used as the frozen reference policy). */
  clone() {
    const p = new Policy();
    p.W.data.set(this.W.data);
    return p;
  }
}

/** Pick each row's chosen token: logProbs [N, V] (Tensor) and tokens [N] -> Tensor [N] with gradient. */
export function selectLogProbs(logProbs, tokens) {
  const [N, V] = logProbs.shape;
  const onehot = new Float32Array(N * V);
  for (let i = 0; i < N; i++) onehot[i * V + tokens[i]] = 1;
  return logProbs.mul(new Tensor({ shape: [N, V], data: onehot })).sum(1);
}

/** Greedy accuracy: fraction of tasks whose most likely answer verifies. */
export function accuracy(policy, tasks) {
  let correct = 0;
  for (const task of tasks) correct += verify(task, String(argmaxArray(policy.probs(task.question))));
  return tasks.length ? correct / tasks.length : 0;
}

/**
 * Worked example: supervised warm-up (the stand-in for SFT). Maximises the log-probability of the
 * labelled answer with cross-entropy, so the policy starts RL with a non-zero success rate.
 * Read this before step 5: forward -> scalar loss -> zeroGrad -> backward -> step is the whole ritual.
 */
export function sftWarmup(policy, tasks, { epochs = 10, lr = 0.05 } = {}) {
  const optimizer = new AdamW(policy.parameters(), { lr });
  const questions = tasks.map((t) => t.question);
  const targets = tasks.map((t) => Number(t.answer));
  let loss = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    const nll = crossEntropy(policy.logits(questions), targets);
    optimizer.zeroGrad();
    nll.backward();
    optimizer.step();
    loss = nll.item();
  }
  return loss;
}

// ---------- step 1: the environment ----------

/** The first number in a completion, or null. "The answer is 10." -> 10, "ten" -> null. */
export function parseAnswer(text) {
  const m = /-?\d+(\.\d+)?/.exec(String(text));
  return m ? Number(m[0]) : null;
}

/** Reward: 1 when the completion's number equals the task's answer, else 0. */
export function verify(task, completion) {
  const got = parseAnswer(completion);
  if (got === null) return 0;
  return Math.abs(got - Number(task.answer)) < 1e-6 ? 1 : 0;
}

/**
 * Sample G completions per task from the current policy and score each with the verifier.
 * Returns a flat array of tasks.length * G samples in group-major order (sample j of task i at i*G + j):
 * { group, task, token, text, reward, oldLogp } with oldLogp the log-probability at sampling time.
 */
export function rollout(policy, tasks, G, next, verifyFn = verify) {
  const samples = [];
  for (let i = 0; i < tasks.length; i++) {
    const task = tasks[i];
    const probs = policy.probs(task.question);
    for (let j = 0; j < G; j++) {
      const token = sampleIndex(probs, next());
      const text = String(token);
      samples.push({ group: i, task, token, text, reward: verifyFn(task, text), oldLogp: Math.log(probs[token]) });
    }
  }
  return samples;
}

// ---------- step 2: group-relative advantages ----------

/** (r - mean) / (std + eps) within each block of G consecutive rewards; population std (divide by G). */
export function groupAdvantages(rewards, G, eps = 1e-6) {
  if (rewards.length % G !== 0) throw new Error(`groupAdvantages: ${rewards.length} rewards do not split into groups of ${G}`);
  const adv = new Float32Array(rewards.length);
  for (let start = 0; start < rewards.length; start += G) {
    let mean = 0;
    for (let j = 0; j < G; j++) mean += rewards[start + j];
    mean /= G;
    let variance = 0;
    for (let j = 0; j < G; j++) variance += (rewards[start + j] - mean) ** 2;
    const std = Math.sqrt(variance / G);
    for (let j = 0; j < G; j++) adv[start + j] = (rewards[start + j] - mean) / (std + eps);
  }
  return adv;
}

// ---------- step 3: the policy-gradient loss ----------

/** REINFORCE: -mean(A * log pi). logp is a Tensor [N] (with gradient), adv a plain array [N] of constants. */
export function reinforceLoss(logp, adv) {
  const A = new Tensor({ shape: [adv.length], data: Float32Array.from(adv) });
  return logp.mul(A).mean().neg();
}

/**
 * PPO-style clipped objective: -mean(min(r * A, clip(r, 1 - eps, 1 + eps) * A)) with r = exp(logp - oldLogp).
 * Which branch of the min is active is a plain boolean per sample; the gradient flows only through r * A on
 * the samples where the unclipped branch is the smaller one.
 */
export function clippedLoss(logp, oldLogp, adv, clip = 0.2) {
  const N = adv.length;
  const ratio = logp.sub(new Tensor({ shape: [N], data: Float32Array.from(oldLogp) })).exp();
  const live = new Float32Array(N);      // A_i where the unclipped branch is active, else 0
  const frozen = new Float32Array(N);    // the clipped (constant) value where it is not
  for (let i = 0; i < N; i++) {
    const r = ratio.data[i], A = adv[i];
    const clamped = Math.min(Math.max(r, 1 - clip), 1 + clip);
    if (clamped * A < r * A) frozen[i] = clamped * A;   // clipped branch is smaller: constant, no gradient
    else live[i] = A;                                   // unclipped branch: r * A, gradient flows
  }
  const objective = ratio.mul(new Tensor({ shape: [N], data: live })).add(new Tensor({ shape: [N], data: frozen }));
  return objective.mean().neg();
}

// ---------- step 4: KL penalty to the reference ----------

/** Unbiased, non-negative "k3" estimator of KL(pi || ref): mean(exp(ref - logp) - (ref - logp) - 1). */
export function klPenalty(logp, refLogp) {
  const ref = refLogp instanceof Tensor ? refLogp.detach() : new Tensor({ shape: [logp.data.length], data: Float32Array.from(refLogp) });
  const diff = ref.sub(logp);                   // log(ref / pi) per sample
  return diff.exp().sub(diff).sub(1).mean();
}

// ---------- step 5: the training loop ----------

/** Shannon entropy in nats of a probability vector: -sum p log p (0 log 0 = 0). */
export function entropyOf(probs) {
  let h = 0;
  for (let i = 0; i < probs.length; i++) if (probs[i] > 0) h -= probs[i] * Math.log(probs[i]);
  return h;
}

/**
 * One GRPO iteration on `tasks`: roll out G samples per task, compute group advantages, then take `mu`
 * gradient steps on clippedLoss + beta * klPenalty. Returns statistics of the iteration.
 */
export function grpoStep(policy, ref, tasks, { G = 8, beta = 0.02, clip = 0.2, mu = 1, optimizer, next, verifyFn = verify }) {
  const samples = rollout(policy, tasks, G, next, verifyFn);
  const rewards = samples.map((s) => s.reward);
  const adv = groupAdvantages(rewards, G);
  const questions = samples.map((s) => s.task.question);
  const tokens = samples.map((s) => s.token);
  const oldLogp = samples.map((s) => s.oldLogp);
  const refLogp = noGrad(() => selectLogProbs(ref.logProbs(questions), tokens)).data;
  let entropy = 0;
  for (const task of tasks) entropy += entropyOf(policy.probs(task.question));
  entropy /= tasks.length;
  let signal = 0;                                  // groups whose rewards are not all equal
  for (let g = 0; g < tasks.length; g++) if (adv.subarray(g * G, (g + 1) * G).some((a) => a !== 0)) signal++;
  const signalFrac = signal / tasks.length;
  let loss = 0, kl = 0, clipFrac = 0;
  for (let step = 0; step < mu; step++) {
    const logp = selectLogProbs(policy.logProbs(questions), tokens);
    const pg = clippedLoss(logp, oldLogp, adv, clip);
    const klTerm = klPenalty(logp, refLogp);
    const total = pg.add(klTerm.scale(beta));
    optimizer.zeroGrad();
    total.backward();
    optimizer.step();
    loss = total.item();
    kl = klTerm.item();
    let clipped = 0;
    for (let i = 0; i < tokens.length; i++) {
      const r = Math.exp(logp.data[i] - oldLogp[i]);
      if ((adv[i] > 0 && r > 1 + clip) || (adv[i] < 0 && r < 1 - clip)) clipped++;
    }
    clipFrac = clipped / tokens.length;
  }
  const reward = rewards.reduce((s, r) => s + r, 0) / rewards.length;
  return { reward, kl, entropy, loss, clipFrac, signalFrac, samples, adv };
}

/**
 * The full loop: `iterations` GRPO steps, each on a random batch of `batchSize` tasks. Returns the
 * per-iteration statistics. `onIter(stats, i)` may be async (the demo yields to the browser there).
 */
export async function trainGRPO(policy, ref, tasks, { iterations = 40, G = 8, batchSize = 16, beta = 0.02, clip = 0.2, mu = 1, lr = 0.05, next, onIter = null, verifyFn = verify } = {}) {
  const optimizer = new AdamW(policy.parameters(), { lr });
  const history = [];
  for (let i = 0; i < iterations; i++) {
    const batch = shuffle(next, tasks.slice()).slice(0, Math.min(batchSize, tasks.length));
    const stats = grpoStep(policy, ref, batch, { G, beta, clip, mu, optimizer, next, verifyFn });
    const record = { iteration: i, reward: stats.reward, kl: stats.kl, entropy: stats.entropy, loss: stats.loss, clipFrac: stats.clipFrac, signalFrac: stats.signalFrac };
    history.push(record);
    if (onIter) await onIter(record, i);
  }
  return history;
}
