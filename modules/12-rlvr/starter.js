// RL with verifiable rewards (GRPO).
//
// The environment: one-step arithmetic questions ("What is 7 + 3?") with a known answer. A program can
// check any completion, so the reward is exact-match correctness and nothing is learned about "reward".
// The policy is deliberately small: a softmax over the integers 0..99 (one "token" per completion), whose
// logits are a sum of learned rows indexed by hashed features of the question. It is given to you below.
// Everything the module is about — the verifier, rollouts, group-relative advantages, the clipped policy
// gradient, the KL penalty to a frozen reference and the training loop — does not depend on the policy's
// size; the same code drives a GPT in production, where "token" becomes "every token of a sampled trace".

import { Tensor, noGrad, crossEntropy } from 'lib/tensor.js';
import { AdamW } from 'lib/optim.js';
import { hash32, sampleIndex, softmaxArray, argmaxArray, shuffle } from 'lib/util.js';

/** The policy's answer vocabulary: completion token t means the text `${t}`. */
export const ANSWER_VOCAB = 100;
/** Number of hashed feature buckets (the "hashing trick"): each feature string picks one row of W. */
export const N_FEATURES = 1024;

// ---------- given: the policy (read it, it sets the conventions) ----------

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

  /** Probabilities of every answer for one question, as a plain array (no graph). */
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

/** Greedy accuracy: fraction of tasks whose most likely answer verifies (uses your verify from step 1). */
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
  // TODO: step 1
  return null;
}

/** Reward: the number 1 when the completion's number equals the task's answer, else 0. Never throws. */
export function verify(task, completion) {
  // TODO: step 1
  return 0;
}

/**
 * Sample G completions per task from the current policy and score each with the verifier.
 * Returns a flat array of tasks.length * G samples in group-major order (sample j of task i at i*G + j):
 * { group, task, token, text, reward, oldLogp } with oldLogp the log-probability at sampling time.
 * `next` is the rng; draw one uniform per sample with sampleIndex(probs, next()).
 */
export function rollout(policy, tasks, G, next, verifyFn = verify) {
  // TODO: step 1
  return [];
}

// ---------- step 2: group-relative advantages ----------

/** (r - mean) / (std + eps) within each block of G consecutive rewards; population std (divide by G). */
export function groupAdvantages(rewards, G, eps = 1e-6) {
  // TODO: step 2
  return new Float32Array(rewards.length);
}

// ---------- step 3: the policy-gradient loss ----------

/** REINFORCE: -mean(A * log pi). logp is a Tensor [N] (with gradient), adv a plain array [N] of constants. */
export function reinforceLoss(logp, adv) {
  // TODO: step 3
  return logp.mean();
}

/**
 * PPO-style clipped objective: -mean(min(r * A, clip(r, 1 - eps, 1 + eps) * A)) with r = exp(logp - oldLogp).
 * Which branch of the min is active is a plain boolean per sample; the gradient flows only through r * A on
 * the samples where the unclipped branch is the smaller one.
 */
export function clippedLoss(logp, oldLogp, adv, clip = 0.2) {
  // TODO: step 3
  return logp.mean();
}

// ---------- step 4: KL penalty to the reference ----------

/**
 * Unbiased, non-negative "k3" estimator of KL(pi || ref): mean(exp(ref - logp) - (ref - logp) - 1).
 * refLogp may be a plain array or a Tensor; either way no gradient may reach the reference.
 */
export function klPenalty(logp, refLogp) {
  // TODO: step 4
  return logp.mean();
}

// ---------- step 5: the training loop ----------

/** Shannon entropy in nats of a probability vector: -sum p log p (0 log 0 = 0). */
export function entropyOf(probs) {
  // TODO: step 5
  return 0;
}

/**
 * One GRPO iteration on `tasks`: roll out G samples per task, compute group advantages, then take `mu`
 * gradient steps on clippedLoss + beta * klPenalty. Returns statistics of the iteration:
 * { reward, kl, entropy, loss, clipFrac, signalFrac, samples, adv } where reward is the mean reward of
 * the rollout, kl and loss come from the last gradient step, entropy is the mean policy entropy over the
 * tasks before the update, clipFrac the fraction of samples whose ratio was clipped in the last step and
 * signalFrac the fraction of groups whose rewards were not all equal. The loss is
 * clippedLoss(logp, oldLogp, adv, clip).add(klPenalty(logp, refLogp).scale(beta)), with oldLogp the
 * rollout's values on every one of the mu steps and refLogp from the frozen `ref` policy.
 */
export function grpoStep(policy, ref, tasks, { G = 8, beta = 0.02, clip = 0.2, mu = 1, optimizer, next, verifyFn = verify }) {
  // TODO: step 5
  return { reward: 0, kl: 0, entropy: 0, loss: 0, clipFrac: 0, signalFrac: 0, samples: [], adv: new Float32Array(0) };
}

/**
 * The full loop: `iterations` GRPO steps, each on a random batch of `batchSize` tasks (shuffle a copy of
 * `tasks` with `next`), using ONE AdamW over policy.parameters() for the whole run and passing verifyFn on
 * to grpoStep. Returns one record per iteration: { iteration, reward, kl, entropy, loss, clipFrac,
 * signalFrac }. `onIter(record, i)` may be async (the demo yields to the browser there): await it.
 */
export async function trainGRPO(policy, ref, tasks, { iterations = 40, G = 8, batchSize = 16, beta = 0.02, clip = 0.2, mu = 1, lr = 0.05, next, onIter = null, verifyFn = verify } = {}) {
  // TODO: step 5
  return [];
}
