export default {
  id: '12-rlvr',
  title: 'RL with verifiable rewards (GRPO)',
  track: 'posttraining',
  minutes: 120,
  threshold: 'When a program can check whether an answer is right, you can train directly on "was it right": sample several attempts per prompt, reward the correct ones, and push probability toward them relative to their siblings, with no learned reward model and no value network.',
  goal: 'A GRPO trainer with group-normalised advantages and a KL penalty that raises a policy\'s accuracy on a verifiable toy task.',
  prereqs: ['02-autograd', '07-pretraining', '10-sft', '11-preference'],
  recall: [
    { q: 'In DPO (module 11), what plays the role of the reward?', options: ['A separate reward head trained on the pairs', 'β times the policy\'s log-ratio against a frozen reference, log π(y|x) − log ref(y|x)', 'The length of the chosen response'], answer: 1,
      why: 'DPO folds the reward into the policy itself. RLVR goes one step further: the reward is a program, so nothing about "reward" is learned at all. The frozen reference survives, as the anchor of the KL penalty.' },
    { q: 'Which positions contribute to the SFT loss (module 10)?', options: ['Every token in the window', 'Only the assistant\'s response tokens and the end marker', 'Only the user\'s tokens'], answer: 1,
      why: 'The mask decides what the model imitates. In RL the equivalent question is which tokens carry the advantage: in this module a completion is one token, in a real trainer it is every token of the sampled trace.' },
    { q: 'A tensor `x` is used twice in a graph (module 02). After `backward()`, `x.grad` holds…', options: ['Only the gradient from the last use', 'The sum of the gradients from both uses', 'The product of the two gradients'], answer: 1,
      why: 'The policy\'s log-probability appears in both the clipped objective and the KL term of the GRPO loss; the two gradients accumulate into the same weights, and `beta` sets the balance.' },
    { q: 'In module 07, what does AdamW\'s division by `sqrt(v)` achieve?', options: ['It normalises the loss', 'Every parameter moves by roughly `lr` per step, whatever the scale of its gradient', 'It prevents overflow'], answer: 1,
      why: 'Your GRPO loop reuses AdamW. Because the step size is roughly `lr` regardless of gradient scale, even a tiny spurious gradient (say 1e-9 from a NaN-guard or a rounding error) would move a weight by about `lr`. Only an exactly zero gradient gives zero first and second moments and so a zero step, which is why an all-equal group must produce advantages of exactly 0 (one of the tests).' },
  ],
  review: [
    { q: 'In a group of 8 samples where exactly one is correct, the correct sample\'s advantage is…', options: ['1', '(1 − 1/8) / sqrt(7/64) ≈ 2.65', '1/8'], answer: 1,
      why: 'Mean reward is 1/8 and the population std is sqrt(1/8 · 7/8) ≈ 0.33, so (1 − 0.125) / 0.33 ≈ 2.65; the seven wrong samples each get about −0.38. The rarer a success, the harder it is pushed.' },
    { q: 'What does GRPO do with a prompt whose 8 samples are all wrong (or all right)?', options: ['Pushes all of them down (or up)', 'Nothing: the advantages are all zero', 'Falls back to the reference policy'], answer: 1,
      why: 'Equal rewards give r − mean = 0 for every sample. The prompt costs 8 rollouts and contributes no gradient, which is why curricula and prompt filtering matter in real runs (DAPO drops such prompts before the update).' },
    { q: 'Why does the clipped objective use `min(r·A, clip(r)·A)` rather than just `clip(r)·A`?', options: ['To make the loss differentiable everywhere', 'So the clip only ever removes incentive: it bounds how far the policy is rewarded for moving in the advantage\'s direction, but never hides a move in the wrong direction', 'To speed up the computation'], answer: 1,
      why: 'The min makes the objective pessimistic. For A > 0 the bonus stops growing beyond r = 1 + ε; for A < 0 the penalty stops shrinking below r = 1 − ε; a step that made things worse is always seen in full.' },
    { q: 'The k3 estimator `exp(ref − logp) − (ref − logp) − 1` is preferred over the plain difference `logp − ref` because…', options: ['It is cheaper', 'It is non-negative for every sample and unbiased for KL(π ‖ ref), so noise cannot make the penalty reward drift', 'It has no gradient'], answer: 1,
      why: 'The plain difference is unbiased too but can be negative on individual samples, so a few lucky draws can make the "penalty" push the policy away from the reference. k3 is a per-sample convex function that is zero exactly at agreement (Schulman 2020, "Approximating KL divergence").' },
    { q: 'A verifier that awards reward 1 when the completion *contains* the answer string is unsound because…', options: ['It is too slow', 'A policy can learn to emit many numbers ("1 2 3 … 100") and always be rewarded: the reward is hackable', 'It cannot handle negative numbers'], answer: 1,
      why: 'RLVR only trains "was it right" when the verifier truly measures rightness. Length, format and enumeration exploits are the classic ways a policy games a loose checker; the tests make your verifier compare numbers, not substrings.' },
  ],
  concept: `
## From a learned reward to a checked one

RLHF as used for InstructGPT trains three networks: the policy, a **reward model** fitted to human preference pairs (module 11), and a **value network** (the "critic") that PPO uses as its baseline. The critic is as large as the policy and its errors leak into every update.

**RL with verifiable rewards (RLVR)** replaces the reward model with a program: a script (arithmetic, unit tests, a proof checker) decides whether a completion is right. The reward is \`1\` or \`0\` and nothing about reward is learned. The name comes from AI2's Tülu 3 report (Lambert et al. 2024). DeepSeekMath (Shao et al. 2024) introduced the GRPO algorithm below, still with a learned reward model; DeepSeek-R1 (2025) ran GRPO on rule-based rewards alone (answer accuracy plus a format check) and showed that this, at scale, teaches long reasoning chains.

## GRPO: the group is the baseline

A policy gradient needs a **baseline** to compare each reward against, so that easy and hard prompts do not push equally hard. PPO uses the critic's value estimate. **Group Relative Policy Optimization** (GRPO) uses the other samples of the same prompt instead: sample \`G\` completions per prompt (DeepSeekMath reports \`G = 64\`), score them, and give sample \`i\` the advantage

\`A_i = (r_i − mean(r)) / (std(r) + eps)\`

computed within its group only. Subtracting a baseline that does not depend on the sample being scored leaves the expected gradient unchanged and removes the variance due to prompt difficulty. The group mean is almost that: it includes \`r_i\` itself, which only shrinks the expected gradient by a factor \`(1 − 1/G)\`. Dividing by the std is a choice, not a free lunch: it makes a rare success count for more, but it also reweights prompts by how uncertain the policy is on them, which Dr. GRPO (Liu et al. 2025) argues is a bias and removes. No critic to train, and roughly half the trainable parameters PPO keeps in memory.

:::predict
The policy starts uniform over 100 answers. With \`G = 8\`, what fraction of prompts get a group with at least one correct sample, and what advantage does an all-wrong group get?
---
\`1 − 0.99^8 ≈ 7.7%\` of prompts. In an all-wrong group every \`r_i − mean\` is 0, so every advantage is 0. Early in the demo about nine in ten groups are wasted and the few mixed groups do all the work; real runs filter prompts by measured pass rate.
:::

## The objective

With \`logp_i\` the policy's log-probability of sample \`i\`, REINFORCE minimises \`−mean(A_i · logp_i)\`. Rollouts are expensive, so trainers take several gradient steps per rollout, after which the samples are off-policy. PPO's fix, kept by GRPO, is the **ratio** \`r_i = exp(logp_i − oldLogp_i)\` and the clipped objective

\`min(r_i · A_i, clip(r_i, 1 − ε, 1 + ε) · A_i)\`

with \`ε = 0.2\`: a sample whose probability has already moved 20% the way its advantage wanted stops giving gradient.

A **KL penalty** to a frozen reference (the SFT model) keeps the policy from drifting somewhere the verifier is satisfied but the language is broken. GRPO's estimator is \`k3 = exp(ref − logp) − (ref − logp) − 1\`, averaged over samples: unbiased for \`KL(π ‖ ref)\` when the samples are drawn from \`π\`, and non-negative for every sample (Schulman 2020, "Approximating KL divergence"), so noise can never turn the penalty into a bonus. Your loss is \`clippedLoss + β · klPenalty\`.

:::predict
After a rollout you take exactly one gradient step (\`mu = 1\`). What is the ratio \`r_i\` when the loss is evaluated, and how many samples are clipped?
---
Exactly 1 for every sample: the parameters that sampled also compute the log-probabilities. \`min(A, A) = A\`, nothing is clipped, and the gradient equals REINFORCE's \`−A_i / N\`. The clip matters only from the second step on the same rollout, or when the rollout came from stale weights.
:::

## Reward hacking is only impossible if the verifier is sound

A checker that accepts any completion *containing* the answer is beaten by a policy that lists every number. Real pipelines add **format rewards** (the answer inside \`\\boxed{}\` or a tag) so parsing is unambiguous, and watch response length: GRPO averages the loss over each sequence's tokens, so a long wrong answer is penalised less per token than a short one and length creeps up (DAPO's token-level loss and Dr. GRPO's removal of the length normalisation answer exactly that). Your \`verify\` compares numbers, not strings, and returns \`0\` on garbage.

## Reasoning traces and test-time scaling

On a real model the completion is a chain of thought followed by an answer, and only the answer is verified. Nothing in the objective asks for long chains, yet R1-style training makes them grow, because longer deliberation raises the fraction of correct samples and the gradient follows. Spending more tokens per question to get more right answers is **test-time scaling**; \`pass@k\` in module 13 measures the same curve from the outside. Module 34 later spends that compute on purpose at inference: voting, verifiers, process reward models and thinking budgets.

## Systems note: rollouts dominate

In production most wall-clock goes to generation, not gradient steps: 16 samples of thousands of tokens per prompt. Trainers such as verl and OpenRLHF run an inference engine (vLLM or SGLang) inside the loop, sync weights into it after each update, and increasingly generate **asynchronously** from weights an update or two old. The clipped ratio is what makes that staleness tolerable.

## Where this toy differs from production

The policy here is a linear softmax over the 100 answers \`0..99\`, with logits summed from hashed question features, and a completion is a single token. It is fast, but the policy memorises prompts rather than learning arithmetic: its accuracy on unseen questions stays near zero (the demo's warm start shows it), and it starts from nothing, a 1-in-100 hit rate, where a real run starts from an SFT model that already solves a useful fraction of its prompts (prompts it always or never solves are often filtered out beforehand). In production the policy is the SFT transformer, \`logp_i\` is a sum over every token of a sampled trace, the group is 8 to 64 samples, the reference is a second copy of the weights, and the gains come over thousands of iterations. The algorithm you write is otherwise the same.
`,
  steps: [
    {
      id: 'env',
      title: 'The environment: verifier and rollouts',
      instructions: `
Three functions make the toy environment.

\`parseAnswer(text)\`: the first number in a completion as a JavaScript number, or \`null\` when there is none. \`"The answer is 10."\` gives \`10\`, \`"-3"\` gives \`-3\`, \`"ten"\` gives \`null\`. Tolerant parsing matters because a real model wraps its answer in words.

\`verify(task, completion)\`: the reward. Return the number \`1\` when the parsed number equals \`Number(task.answer)\` (compare numerically so \`"10.0"\` counts and \`"100"\` does not), otherwise \`0\`. Never throw; garbage earns \`0\`.

\`rollout(policy, tasks, G, next, verifyFn = verify)\`: for each task in order, get \`policy.probs(task.question)\` once and draw \`G\` tokens with \`sampleIndex(probs, next())\`. Return one flat array of \`tasks.length × G\` samples, sample \`j\` of task \`i\` at index \`i*G + j\`, each \`{ group: i, task, token, text: String(token), reward: verifyFn(task, text), oldLogp: Math.log(probs[token]) }\`. \`oldLogp\` is the log-probability *at sampling time*; step 3 needs it to form the ratio.

The \`Policy\` class above the TODO line is given; read \`probs\` and \`logProbs\` to see the two views of the same distribution (a plain array without a graph, and a Tensor with one).
`,
      predict: { question: 'A policy puts 50% on token 10 and spreads the rest over the other 99 answers. Over 400 rollouts of "What is 5 + 5?", roughly how many distinct tokens appear?', answer: 'Well over 20: about 200 draws land on 10 and the other 200 are spread over 99 tokens, most of which appear at least once. If your rollout takes the argmax instead of sampling, only one token appears and there is nothing for RL to compare.' },
      hints: [
        'A regular expression finds the number: an optional minus sign, digits, an optional fraction. `Number(match)` converts it. For verify, compare two numbers with a small tolerance, and remember that `parseAnswer` may return `null`.',
        'rollout is two nested loops: over tasks, then `G` times. Compute the probability array once per task (outside the inner loop), draw `token = sampleIndex(probs, next())`, and build the sample object with `Math.log(probs[token])`.',
        '`const m = /* regex: optional minus, digits, optional fraction */.exec(String(text)); return m ? Number(m[0]) : null;` for parseAnswer. rollout is shaped `for (i over tasks) { probs = policy.probs(question_i); for (j < G) { token = /* one fresh draw */; samples.push({ group, task, token, text, reward, oldLogp }); } }`: each field is a one-liner from the instructions, and `reward` must call `verifyFn`, not `verify`.',
      ],
    },
    {
      id: 'advantages',
      title: 'Group-relative advantages',
      instructions: `
Implement \`groupAdvantages(rewards, G, eps = 1e-6)\`: \`rewards\` is the flat array from \`rollout\`, so consecutive blocks of \`G\` entries belong to the same prompt. Return a \`Float32Array\` of the same length with

\`A_i = (r_i − mean_g) / (std_g + eps)\`

where \`mean_g\` and \`std_g\` are the mean and **population** standard deviation (divide the variance by \`G\`, not \`G − 1\`) of that sample's own group. Throw if \`rewards.length\` is not a multiple of \`G\`.

\`eps\` is not cosmetic: a group whose rewards are all equal has \`std = 0\`, and without \`eps\` you would divide \`0 / 0\` into \`NaN\`, which would then poison every weight in the policy. With it, such groups get exactly zero advantage, which is the correct answer: they carry no information about which sample was better.
`,
      predict: { question: 'Rewards `[1, 1, 0, 0]` with G = 4. What are the four advantages?', answer: '`[1, 1, −1, −1]`. Mean 0.5, variance 0.25, std 0.5, so each sample is exactly one std from the mean. With `[1, 0, 0, 0]` the lone success gets +1.73 and the failures −0.58: the rarer the success, the harder it is pushed.' },
      hints: [
        'Loop over group starts `start = 0, G, 2G, …`. Inside, two passes over the `G` entries: one for the mean, one for the variance around that mean. Then a third to write the advantages.',
        'Variance is the mean of squared deviations: `sum((r − mean)²) / G`. Take `Math.sqrt`, add `eps`, divide. Do not normalise over the whole batch: group 0 must not see group 1\'s rewards.',
        '`for (let start = 0; start < rewards.length; start += G) { const mean = /* average of rewards[start .. start+G) */; const std = /* population std of the same slice around mean */; for (let j = 0; j < G; j++) adv[start + j] = /* the formula */; }`, preceded by the multiple-of-G check.',
      ],
    },
    {
      id: 'pg',
      title: 'The policy-gradient loss, plain and clipped',
      instructions: `
Both functions take \`logp\`, a Tensor of shape \`[N]\` holding the policy's current log-probability of each sample (with a gradient path back to the weights), and \`adv\`, a plain array of \`N\` advantages that are **constants**: no gradient may flow into them.

\`reinforceLoss(logp, adv)\`: return the scalar Tensor \`−mean(adv · logp)\`. Its gradient with respect to \`logp_i\` is \`−A_i / N\`. Wrap the advantages in a \`Tensor\` without \`requiresGrad\` and use the Tensor ops (\`mul\`, \`mean\`, \`neg\`) so that \`backward()\` reaches the policy.

\`clippedLoss(logp, oldLogp, adv, clip = 0.2)\`: the PPO objective. Form the ratio \`r = exp(logp − oldLogp)\` as a Tensor (\`oldLogp\` is a plain array of constants). For each sample decide, with plain numbers, which branch of \`min(r·A, clamp(r, 1−clip, 1+clip)·A)\` is smaller. Where the unclipped branch wins, the contribution is \`r_i · A_i\` and the gradient flows through \`r\`; where the clipped branch wins, the contribution is the constant \`clamp(r_i)·A_i\` and there is no gradient. One way to build this in the graph: multiply \`r\` by a "live" vector holding \`A_i\` where the ratio is unclipped and \`0\` elsewhere, and add a "frozen" constant vector holding \`clamp(r_i)·A_i\` where it is clipped. Return \`−mean(...)\`.
`,
      hints: [
        'Both losses are three Tensor ops long once the constants are Tensors. Check the units first: at ratio 1 the clipped loss must equal REINFORCE exactly, value and gradient.',
        'For the clip, read `r.data[i]` (a plain number) to decide the branch per sample. Fill two Float32Arrays of length N: `live[i] = A_i` when unclipped (else 0) and `frozen[i] = clamp(r_i)·A_i` when clipped (else 0). Then `objective = r.mul(live).add(frozen)`.',
        '`const ratio = logp.sub(new Tensor({ shape: [N], data: Float32Array.from(oldLogp) })).exp(); for (i) { const r = ratio.data[i], A = adv[i]; const clamped = Math.min(Math.max(r, 1 - clip), 1 + clip); if (/* clipped branch is the smaller one */) frozen[i] = clamped * A; else live[i] = A; } return ratio.mul(new Tensor({ shape: [N], data: live })).add(new Tensor({ shape: [N], data: frozen })).mean().neg();`',
      ],
    },
    {
      id: 'kl',
      title: 'The KL penalty to the reference',
      instructions: `
Implement \`klPenalty(logp, refLogp)\`: the k3 estimator of \`KL(π ‖ ref)\` on the sampled tokens,

\`mean( exp(d) − d − 1 )\` with \`d = refLogp − logp\`

as a scalar Tensor with a gradient path through \`logp\` only. \`refLogp\` may arrive as a plain array or as a Tensor; in either case the reference is frozen, so detach it (or wrap the numbers in a Tensor without \`requiresGrad\`) before subtracting.

Check the two facts the tests rely on: at \`logp = refLogp\` the value is 0 and so is the gradient (\`d k3 / d logp = 1 − exp(d)\`), and every per-sample term is \`≥ 0\` because \`exp(d) − d − 1\` is convex with its minimum at \`d = 0\`. The plain estimator \`logp − refLogp\` has the same expectation but can be negative on a sample, which is why GRPO uses k3.
`,
      hints: [
        'Write `d` as a Tensor first: reference minus policy. Everything else is `exp`, `sub`, `sub(1)`, `mean`.',
        'If `refLogp instanceof Tensor`, use `refLogp.detach()`; otherwise `new Tensor({ shape: [N], data: Float32Array.from(refLogp) })`. Then `d = ref.sub(logp)`.',
        '`const ref = refLogp instanceof Tensor ? refLogp.detach() : new Tensor({ shape: [logp.data.length], data: Float32Array.from(refLogp) }); const d = ref.sub(logp); return /* k3 of d, averaged */;`',
      ],
    },
    {
      id: 'train',
      title: 'The GRPO training loop',
      instructions: `
Three functions.

\`entropyOf(probs)\`: Shannon entropy in nats, \`−Σ p log p\`, skipping zero entries. Uniform over 100 answers gives \`ln 100 ≈ 4.605\`; a one-hot gives 0. Entropy is the health signal of an RL run: it should fall as the policy commits, and a collapse to 0 means no more mixed groups.

\`grpoStep(policy, ref, tasks, { G, beta, clip, mu, optimizer, next, verifyFn })\`: one iteration. Roll out \`G\` samples per task; \`groupAdvantages\` on the rewards; collect \`questions\`, \`tokens\` and \`oldLogp\` from the samples; compute the reference log-probs once inside \`noGrad\` with \`selectLogProbs(ref.logProbs(questions), tokens)\` (a Float32Array via \`.data\`). Measure the mean \`entropyOf(policy.probs(q))\` over the tasks and \`signalFrac\`, the fraction of groups with any non-zero advantage, before updating. Then \`mu\` times: \`logp = selectLogProbs(policy.logProbs(questions), tokens)\`, \`total = clippedLoss(logp, oldLogp, adv, clip) + beta · klPenalty(logp, refLogp)\` (in Tensor ops: \`pg.add(klTerm.scale(beta))\`), \`zeroGrad → backward → step\` (the ritual in \`sftWarmup\`). \`oldLogp\` stays the rollout's value on every one of the \`mu\` steps: that is what lets the ratio move away from 1 and the clip engage. Return \`{ reward, kl, entropy, loss, clipFrac, signalFrac, samples, adv }\`, where \`reward\` is the mean rollout reward, \`kl\` and \`loss\` come from the last gradient step, and \`clipFrac\` is the fraction of samples whose ratio in that step lay outside \`[1 − clip, 1 + clip]\` on the side the advantage wanted.

\`trainGRPO(policy, ref, tasks, opts)\`: create one \`AdamW\` over \`policy.parameters()\` with \`opts.lr\`, then for \`iterations\`: take a batch of \`batchSize\` tasks from a shuffled copy (\`shuffle(next, tasks.slice())\`), call \`grpoStep\` (passing \`opts.verifyFn\` through), push \`{ iteration, reward, kl, entropy, loss, clipFrac, signalFrac }\`, and \`await onIter(record, i)\` when given. Return the history. All randomness must go through \`next\` so that a seed reproduces a run.
`,
      predict: { question: 'The reference equals the policy and the rollout\'s rewards are all 0. After one grpoStep, how much did the weights move?', answer: 'Not at all. Every advantage is 0, so the clipped loss has zero gradient; and at policy = reference the k3 gradient `1 − exp(0)` is 0 too. AdamW turns a zero gradient into a zero step. GRPO learns nothing from prompts it always fails, or always solves.' },
      hints: [
        'Build grpoStep in the order the data flows: samples → rewards → advantages → (questions, tokens, oldLogp) → refLogp (noGrad, once) → stats before the update → the mu-step loop. Keep the statistics as plain numbers, the loss as a Tensor.',
        'The clip fraction: for each sample compute `r = Math.exp(logp.data[i] − oldLogp[i])` after the forward pass and count `(A > 0 && r > 1 + clip) || (A < 0 && r < 1 − clip)`. With mu = 1 it is always 0, which is one of the tests.',
        '`for (let step = 0; step < mu; step++) { const logp = selectLogProbs(policy.logProbs(questions), tokens); const pg = clippedLoss(logp, oldLogp, adv, clip); const klTerm = klPenalty(logp, refLogp); const total = /* combine pg and klTerm with beta */; optimizer.zeroGrad(); total.backward(); optimizer.step(); loss = total.item(); kl = klTerm.item(); }` and in trainGRPO: `const batch = shuffle(next, tasks.slice()).slice(0, batchSize);`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why GRPO needs no value network: what does the group of G samples provide that PPO gets from the critic, and what does it cost (which prompts are wasted)?',
    'Your verifier is the whole reward. Describe two ways a policy could exploit a looser version of it, and what the fix to the verifier or to the reward would be.',
    'The KL penalty and the clip both limit how far the policy moves, but against different things. What does each one protect against, and what would you expect to go wrong with beta = 0?',
  ],
  stretch: [
    'Replace the answer policy with the tiny GPT from `lib/checkpoints/tiny-gpt.json`: a completion becomes a token sequence, `logp` a sum of per-token log-probs, and `rollout` a call to `generate`. This is the shape of TRL\'s `GRPOTrainer` and of verl; expect each iteration to take seconds rather than milliseconds.',
    'Implement the prompt filtering of DAPO (Yu et al. 2025): before the update, drop groups whose rewards are all equal and top the batch up with fresh prompts, so that every gradient step sees the same number of informative groups. Measure how many rollouts it costs per unit of accuracy.',
    'Add a format reward: only completions of the form `"answer: N"` earn credit, with a small bonus for the format alone as in DeepSeek-R1. Watch how quickly the format is learned compared with correctness, then remove the correctness term and observe reward hacking directly.',
    'Make the rollouts stale on purpose, as an asynchronous trainer with vLLM does: sample from a copy of the weights that is refreshed only every k iterations, and plot clipFrac and accuracy against k.',
  ],
  timeouts: { tests: 20000, demo: 60000 },
};
