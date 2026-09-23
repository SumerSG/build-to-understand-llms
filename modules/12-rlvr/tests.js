import { Tensor } from 'lib/tensor.js';
import { AdamW } from 'lib/optim.js';
import { MATH_TASKS } from 'lib/data.js';

/** A policy whose answer to `question` is `token` with probability about `p` (the rest spread evenly). */
function peakedPolicy(m, question, token, p = 0.999) {
  const policy = new m.Policy();
  const ids = m.features(question);
  // Put the whole logit on the first feature row; the other rows of this question stay zero.
  const V = m.ANSWER_VOCAB;
  const logit = Math.log((p * (V - 1)) / (1 - p));
  policy.W.data[ids[0] * V + token] = logit;
  return policy;
}

function scalarWithGrad(values) {
  return new Tensor({ shape: [values.length], data: Float32Array.from(values) }, { requiresGrad: true });
}

export const tests = [
  // ---------- step 1: environment ----------
  { step: 'env', name: 'parseAnswer finds the number in a completion and returns null when there is none', run(m, T) {
    T.eq(m.parseAnswer('10'), 10);
    T.eq(m.parseAnswer('The answer is 10.'), 10, 'a completion may wrap the number in words and punctuation');
    T.eq(m.parseAnswer(' 42\n'), 42, 'surrounding whitespace must be ignored');
    T.eq(m.parseAnswer('-3'), -3, 'negative numbers keep their sign');
    T.eq(m.parseAnswer('ten'), null, 'no digits means no parsable answer (return null, not 0 or NaN)');
    T.eq(m.parseAnswer(''), null, 'an empty completion has no answer');
  } },
  { step: 'env', name: 'verify returns the number 1 only for an exact numeric match', run(m, T) {
    const task = { question: 'What is 7 + 3?', answer: '10' };
    T.eq(m.verify(task, '10'), 1, 'the exact answer earns reward 1 (a number, not true)');
    T.eq(m.verify(task, 'The answer is 10.'), 1, 'tolerant parsing: the number inside a sentence counts');
    T.eq(m.verify(task, '1'), 0, 'a prefix of the answer is wrong');
    T.eq(m.verify(task, '100'), 0, 'a string that merely contains the answer is wrong (compare numbers, not substrings)');
    T.eq(m.verify(task, '10.0'), 1, '10.0 is numerically equal to 10');
    T.eq(m.verify(task, 'ten'), 0, 'unparsable completions earn 0, not an exception');
    T.eq(m.verify({ question: 'What is 2 - 2?', answer: '0' }, '0'), 1, 'zero is a valid answer');
  } },
  { step: 'env', name: 'rollout returns tasks×G samples in group-major order with rewards and log-probs', run(m, T) {
    const tasks = [{ question: 'What is 7 + 3?', answer: '10' }, { question: 'What is 2 * 4?', answer: '8' }];
    const policy = peakedPolicy(m, tasks[0].question, 10);
    const G = 4;
    const samples = m.rollout(policy, tasks, G, T.rng(1));
    T.eq(samples.length, tasks.length * G, 'one sample per (task, j) pair');
    T.eq(samples.map((s) => s.group), [0, 0, 0, 0, 1, 1, 1, 1], 'sample j of task i must sit at index i*G + j');
    for (let j = 0; j < G; j++) {
      const s = samples[j];
      T.eq(s.task, tasks[0], 'each sample carries its task');
      T.eq(s.token, 10, 'a policy with 99.9% mass on token 10 samples 10');
      T.eq(s.text, '10', 'the completion text of token t is the string of t');
      T.eq(s.reward, 1, 'reward must come from verify(task, text)');
      T.close(s.oldLogp, Math.log(policy.probs(tasks[0].question)[10]), 1e-5, 'oldLogp is log of the sampled token\'s probability at sampling time');
    }
    for (let j = G; j < 2 * G; j++) {
      const s = samples[j];
      T.eq(s.task, tasks[1]);
      T.close(s.oldLogp, Math.log(policy.probs(tasks[1].question)[s.token]), 1e-5, 'oldLogp of a nearly uniform policy is about log(1/100)');
      T.eq(s.reward, m.verify(tasks[1], s.text), 'reward must agree with verify on the sampled text');
    }
  } },
  { step: 'env', name: 'rollout samples from the policy distribution, deterministically for a given seed', run(m, T) {
    const task = { question: 'What is 5 + 5?', answer: '10' };
    const policy = peakedPolicy(m, task.question, 10, 0.5);
    const a = m.rollout(policy, [task], 400, T.rng(7));
    const b = m.rollout(policy, [task], 400, T.rng(7));
    T.eq(a.map((s) => s.token), b.map((s) => s.token), 'the same seed must give the same samples');
    const hits = a.filter((s) => s.token === 10).length;
    T.ok(hits > 150 && hits < 250, `with P(10) = 0.5, about 200 of 400 samples should be 10; got ${hits} (sample with sampleIndex(probs, next()), do not take the argmax)`);
    const distinct = new Set(a.map((s) => s.token)).size;
    T.ok(distinct > 20, `the other half of the mass is spread over 99 tokens, so many distinct tokens should appear; got ${distinct}`);
    const mean = a.reduce((s, x) => s + x.reward, 0) / a.length;
    T.close(mean, hits / 400, 1e-9, 'mean reward equals the fraction of correct samples');
    const c = m.rollout(policy, [task], 20, T.rng(8));
    T.ok(c.some((s, i) => s.token !== a[i].token), 'a different seed should give different samples');
  } },

  // ---------- step 2: group advantages ----------
  { step: 'advantages', name: 'one correct answer in a group of four gets +1.732, the others -0.577', run(m, T) {
    const adv = m.groupAdvantages([1, 0, 0, 0], 4);
    T.close(T.arr(adv), [1.73205, -0.57735, -0.57735, -0.57735], 1e-3, '(r - mean) / std with the population std (divide by G); mean-only or the n-1 std give 0.75 or 1.5 instead of 1.732');
    T.close(T.arr(m.groupAdvantages([1, 1, 0, 0], 4)), [1, 1, -1, -1], 1e-4, 'two of four correct: std is 0.5, so advantages are exactly ±1');
  } },
  { step: 'advantages', name: 'a group with all-equal rewards gets zero advantage (finite, not NaN)', run(m, T) {
    const zeros = T.arr(m.groupAdvantages([0, 0, 0, 0], 4));
    const ones = T.arr(m.groupAdvantages([1, 1, 1, 1], 4));
    for (const v of [...zeros, ...ones]) T.ok(Number.isFinite(v) && Math.abs(v) < 1e-9, `all-wrong and all-right groups carry no signal: their advantages must be exactly 0, got ${v} (add eps to the std before dividing)`);
    const mixed = T.arr(m.groupAdvantages([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], 4));
    T.close(mixed.slice(0, 8), [0, 0, 0, 0, 0, 0, 0, 0], 1e-6, 'the two all-equal groups are zero');
    T.ok(mixed[8] > 1.5 && mixed[9] < -0.5, `the mixed group in the same call must still get non-zero advantages (got ${mixed.slice(8)}): zero is the answer for equal rewards, not the answer for everything`);
    T.throws(() => m.groupAdvantages([1, 0, 1], 2), '3 rewards do not split into groups of 2: throw rather than silently mis-grouping');
  } },
  { step: 'advantages', name: 'groups are normalised independently, not over the whole batch', run(m, T) {
    const adv = T.arr(m.groupAdvantages([1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], 4));
    T.close(adv.slice(0, 4), [1.73205, -0.57735, -0.57735, -0.57735], 1e-3, 'group 0 must not see the rewards of groups 1 and 2');
    T.close(adv.slice(4, 8), [0, 0, 0, 0], 1e-6, 'group 1 is all-correct: zero advantage even though other groups have failures');
    T.close(adv.slice(8, 12), [-1, 1, -1, 1], 1e-4, 'group 2 has two of four correct');
    const next = T.rng(3);
    const rewards = Array.from({ length: 40 }, () => (next() < 0.3 ? 1 : 0));
    const a = T.arr(m.groupAdvantages(rewards, 8));
    T.eq(a.length, 40, 'one advantage per reward');
    for (let g = 0; g < 5; g++) {
      const grp = a.slice(g * 8, g * 8 + 8);
      const mean = grp.reduce((s, v) => s + v, 0) / 8;
      const sd = Math.sqrt(grp.reduce((s, v) => s + (v - mean) ** 2, 0) / 8);
      T.close(mean, 0, 1e-4, `group ${g} must have mean 0`);
      const raw = rewards.slice(g * 8, g * 8 + 8);
      if (raw.some((r) => r !== raw[0])) T.close(sd, 1, 1e-3, `group ${g} has mixed rewards so its advantages must have std 1`);
    }
  } },

  // ---------- step 3: policy-gradient losses ----------
  { step: 'pg', name: 'reinforceLoss is -mean(A · log π) and its gradient is -A / N', run(m, T) {
    const logp = scalarWithGrad([-1, -2, -3]);
    const loss = m.reinforceLoss(logp, [1, -1, 0]);
    T.ok(loss instanceof Tensor, 'return a Tensor so that backward() can run through it');
    T.close(loss.item(), -1 / 3, 1e-5, '-(1·(-1) + (-1)·(-2) + 0·(-3)) / 3 = -1/3; you must average over the N samples, not sum');
    loss.backward();
    T.close(T.arr(logp.grad), [-1 / 3, 1 / 3, 0], 1e-5, 'd loss / d logp_i = -A_i / N: positive advantages push log π up, negative ones push it down');
  } },
  { step: 'pg', name: 'reinforceLoss treats the advantages as constants', run(m, T) {
    const logp = scalarWithGrad([-0.5, -0.7, -1.2, -2]);
    const adv = new Tensor({ shape: [4], data: Float32Array.from([2, -1, 0.5, -1.5]) }, { requiresGrad: true });
    const loss = m.reinforceLoss(logp, Array.from(adv.data));
    T.close(loss.item(), -(2 * -0.5 + -1 * -0.7 + 0.5 * -1.2 + -1.5 * -2) / 4, 1e-5);
    loss.backward();
    T.close(T.arr(logp.grad), [-0.5, 0.25, -0.125, 0.375], 1e-5, 'gradient is -A/N for every sample');
    T.ok(adv.grad === null, 'advantages are detached constants (a plain array); no gradient may flow into them');
  } },
  { step: 'pg', name: 'clippedLoss with ratio 1 equals reinforceLoss (the first update after a rollout is never clipped)', run(m, T) {
    const values = [-0.3, -1.1, -2.5, -0.9];
    const adv = [1.5, -0.5, -0.5, -0.5];
    const logp = scalarWithGrad(values);
    const loss = m.clippedLoss(logp, values.slice(), adv, 0.2);
    T.close(loss.item(), -adv.reduce((s, a) => s + a, 0) / 4, 1e-5, 'exp(logp - oldLogp) = 1, so min(r·A, clip(r)·A) = A and the loss value is -mean(A)');
    const ref = scalarWithGrad(values);
    m.reinforceLoss(ref, adv).backward();
    loss.backward();
    T.close(T.arr(logp.grad), T.arr(ref.grad), 1e-5, 'at ratio 1 the clipped loss and REINFORCE have identical gradients');
    T.close(T.arr(logp.grad), adv.map((a) => -a / 4), 1e-5, 'at ratio 1 the gradient is -A/N, the same as REINFORCE (d r / d logp = r = 1)');
  } },
  { step: 'pg', name: 'clippedLoss stops the gradient where the ratio leaves [1-ε, 1+ε] in the favourable direction', run(m, T) {
    const old = [-1, -1, -1, -1];
    const ratios = [1.5, 0.5, 1.5, 0.5];             // new/old probability
    const adv = [1, -1, -1, 1];
    const logp = scalarWithGrad(old.map((o, i) => o + Math.log(ratios[i])));
    const loss = m.clippedLoss(logp, old, adv, 0.2);
    // sample 0: A>0, r=1.5 > 1.2 -> clipped to 1.2·1;   sample 1: A<0, r=0.5 < 0.8 -> clipped to 0.8·(-1)
    // sample 2: A<0, r=1.5 -> unclipped 1.5·(-1) (the pessimistic branch); sample 3: A>0, r=0.5 -> unclipped 0.5·1
    T.close(loss.item(), -(1.2 - 0.8 - 1.5 + 0.5) / 4, 1e-4, 'the objective takes min(r·A, clip(r)·A) per sample, then the loss is minus the mean');
    loss.backward();
    T.close(T.arr(logp.grad), [0, 0, 1.5 / 4, -0.5 / 4], 1e-4, 'clipped samples contribute zero gradient; unclipped ones contribute -r·A/N (d exp(x)/dx = exp(x))');
  } },

  // ---------- step 4: KL penalty ----------
  { step: 'kl', name: 'klPenalty is zero, with zero gradient, when the policy equals the reference', run(m, T) {
    const values = [-0.5, -1.5, -3, -0.1];
    const logp = scalarWithGrad(values);
    const kl = m.klPenalty(logp, values.slice());
    T.ok(kl instanceof Tensor, 'return a Tensor scalar so that it can join the loss');
    T.close(kl.item(), 0, 1e-7, 'exp(0) - 0 - 1 = 0 for every sample');
    kl.backward();
    T.close(T.arr(logp.grad), [0, 0, 0, 0], 1e-7, 'd k3 / d logp = 1 - exp(ref - logp), which vanishes at equality');
  } },
  { step: 'kl', name: 'klPenalty computes the mean of exp(ref - logp) - (ref - logp) - 1', run(m, T) {
    const logp = scalarWithGrad([-1, -2]);
    const kl = m.klPenalty(logp, [-1.5, -1]);
    const k3 = (d) => Math.exp(d) - d - 1;
    T.close(kl.item(), (k3(-0.5) + k3(1)) / 2, 1e-5, 'per sample d = ref - logp: k3 = exp(d) - d - 1, then average over samples (k1 = -d would give 0.25 here, k2 = d²/2 would give 0.3125)');
    kl.backward();
    T.close(T.arr(logp.grad), [(1 - Math.exp(-0.5)) / 2, (1 - Math.exp(1)) / 2], 1e-5, 'gradient per sample is (1 - exp(ref - logp)) / N');
  } },
  { step: 'kl', name: 'klPenalty is never negative and leaves the reference untouched', run(m, T) {
    const next = T.rng(5);
    for (let trial = 0; trial < 5; trial++) {
      const n = 6;
      const a = Array.from({ length: n }, () => -next() * 4);
      const b = Array.from({ length: n }, () => -next() * 4);
      const kl = m.klPenalty(scalarWithGrad(a), b);
      T.ok(kl.item() >= -1e-7, `k3 is non-negative for every sample (got ${kl.item()}); the plain difference logp - ref can be negative and would let the penalty reward drift`);
    }
    const ref = new Tensor({ shape: [3], data: Float32Array.from([-1, -2, -3]) }, { requiresGrad: true });
    const logp = scalarWithGrad([-1.2, -1.8, -3.5]);
    const kl = m.klPenalty(logp, ref);
    kl.backward();
    T.ok(logp.grad !== null, 'the policy log-probs must receive a gradient');
    T.ok(ref.grad === null || Array.from(ref.grad).every((g) => g === 0), 'the reference is frozen: no gradient may reach it (detach it or pass plain numbers)');
  } },

  // ---------- step 5: training loop ----------
  { step: 'train', name: 'entropyOf: uniform gives log V, a one-hot gives 0', run(m, T) {
    T.close(m.entropyOf(new Float32Array(100).fill(0.01)), Math.log(100), 1e-4, 'the uniform distribution over 100 answers has entropy ln 100 = 4.605 nats');
    const onehot = new Float32Array(100); onehot[7] = 1;
    T.close(m.entropyOf(onehot), 0, 1e-9, 'a deterministic policy has zero entropy (treat 0·log 0 as 0, do not produce NaN)');
    T.close(m.entropyOf([0.5, 0.5]), Math.log(2), 1e-6);
    T.close(m.entropyOf([0.9, 0.1]), -(0.9 * Math.log(0.9) + 0.1 * Math.log(0.1)), 1e-6);
  } },
  { step: 'train', name: 'grpoStep reports the rollout\'s reward, KL, entropy and signal, and moves the policy when there is signal', run(m, T) {
    const tasks = MATH_TASKS.slice(0, 6);
    const policy = new m.Policy();
    // Give the policy a 50% chance on task 0 so that some groups are mixed; the reference is this same policy.
    const V = m.ANSWER_VOCAB;
    policy.W.data[m.features(tasks[0].question)[0] * V + Number(tasks[0].answer)] = Math.log(V - 1);
    const ref = policy.clone();
    const before = Float32Array.from(policy.W.data);
    const optimizer = new AdamW(policy.parameters(), { lr: 0.05 });
    const stats = m.grpoStep(policy, ref, tasks, { G: 8, beta: 0.04, clip: 0.2, mu: 1, optimizer, next: T.rng(2) });
    for (const k of ['reward', 'kl', 'entropy', 'loss', 'clipFrac', 'signalFrac']) T.ok(typeof stats[k] === 'number' && Number.isFinite(stats[k]), `stats.${k} must be a finite number`);
    T.ok(stats.reward > 0 && stats.reward < 0.5, `mean reward over 48 samples should be between 0 and 0.5 here, got ${stats.reward}`);
    T.close(stats.kl, 0, 1e-6, 'the KL is measured before the update: policy and reference still agree on the sampled tokens');
    T.ok(stats.entropy > 3.5 && stats.entropy <= Math.log(V) + 1e-6, `the mean entropy over the 6 prompts should be just under ln 100 = 4.6, got ${stats.entropy}`);
    T.eq(stats.clipFrac, 0, 'with mu = 1 the ratio is exactly 1 for every sample, so nothing can be clipped');
    T.ok(stats.signalFrac > 0 && stats.signalFrac <= 1, `some groups must be mixed (task 0 is correct half the time), got signalFrac ${stats.signalFrac}`);
    const moved = policy.W.data.some((v, i) => v !== before[i]);
    T.ok(moved, 'a step with mixed groups must change the policy parameters (did you call optimizer.step()?)');
  } },
  { step: 'train', name: 'grpoStep leaves the policy unchanged when no group has signal (all-wrong rewards)', run(m, T) {
    const tasks = MATH_TASKS.slice(0, 4).map((t) => ({ question: t.question, answer: '999' }));   // unreachable answers
    const policy = new m.Policy();
    const ref = policy.clone();
    const before = Float32Array.from(policy.W.data);
    const optimizer = new AdamW(policy.parameters(), { lr: 0.1 });
    const stats = m.grpoStep(policy, ref, tasks, { G: 8, beta: 0.04, clip: 0.2, mu: 2, optimizer, next: T.rng(3) });
    T.eq(stats.reward, 0, 'no completion can be right when the answer is outside the vocabulary');
    T.eq(stats.signalFrac, 0, 'every group is all-wrong: zero advantage everywhere');
    T.eq(stats.samples.length, 32, 'the rollout of 4 tasks × G = 8 must still happen (32 samples) even though it yields no gradient');
    T.eq(T.arr(stats.adv), new Array(32).fill(0), 'stats.adv must hold the 32 group advantages, all zero here');
    T.ok(stats.entropy > 4.6 && stats.entropy <= Math.log(m.ANSWER_VOCAB) + 1e-6, `a uniform policy has entropy ln 100 = 4.605 nats before the update, got ${stats.entropy}`);
    T.ok(policy.W.data.every((v, i) => v === before[i]), 'zero advantages and zero KL gradient at the reference mean zero update: GRPO wastes prompts whose group is all-wrong or all-right');
  } },
  { step: 'train', name: 'trainGRPO raises accuracy on a handful of tasks from a uniform start, deterministically', async run(m, T) {
    const tasks = MATH_TASKS.slice(10, 16);
    const run = async (seed) => {
      const policy = new m.Policy();
      const ref = policy.clone();
      let calls = 0;
      const history = await m.trainGRPO(policy, ref, tasks, { iterations: 40, G: 8, batchSize: 6, beta: 0.04, clip: 0.2, mu: 2, lr: 0.2, next: T.rng(seed), onIter: () => { calls++; } });
      return { policy, history, calls };
    };
    const a = await run(11);
    T.eq(a.history.length, 40, 'one record per iteration');
    T.eq(a.calls, 40, 'onIter must be called once per iteration (the demo yields to the browser there)');
    for (const k of ['reward', 'kl', 'entropy', 'signalFrac']) T.ok(a.history.every((h) => typeof h[k] === 'number'), `each record needs a numeric ${k}`);
    const acc = m.accuracy(a.policy, tasks);
    T.ok(acc >= 0.5, `after 40 iterations of G=8 on 6 tasks greedy accuracy should be at least 0.5, got ${acc} (mean reward went ${a.history[0].reward.toFixed(2)} -> ${a.history.at(-1).reward.toFixed(2)})`);
    T.ok(a.history.at(-1).reward > a.history[0].reward + 0.2, 'mean reward must rise over training');
    T.ok(a.history.at(-1).kl > 0.1, `the policy must move away from the uniform reference (final KL ${a.history.at(-1).kl})`);
    T.ok(a.history.at(-1).entropy < a.history[0].entropy - 0.5, 'entropy must fall as the policy commits to answers');
    const b = await run(11);
    T.close(b.history.map((h) => h.reward), a.history.map((h) => h.reward), 1e-9, 'the same seed must reproduce the same run (all randomness through next())');
  } },
];
