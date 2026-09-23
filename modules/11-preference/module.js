export default {
  id: '11-preference',
  title: 'Reward models & DPO',
  track: 'posttraining',
  minutes: 105,
  threshold: 'A preference between two answers is a training signal: the Bradley–Terry model turns "A beats B" into a logistic loss on a score difference, and DPO shows the policy\'s own log-ratio against a frozen reference can play the role of that score, so no separate reward model is needed.',
  goal: 'A Bradley–Terry reward model loss and a DPO loss that raise the implicit reward margin on preference pairs when trained: a reward head that fits the training pairs and is scored on 10 held-out pairs (so each pair is 10 points and the held-out number is a rough estimate), and a DPO run whose margin climbs from exactly zero.',
  prereqs: ['10-sft', '07-pretraining', '02-autograd'],
  recall: [
    { q: 'In module 10, which positions did the SFT loss mask select?', options: ['Every token of the window', 'The assistant\'s response tokens, including the end marker', 'The prompt tokens'], answer: 1,
      why: 'The same mask defines "the response" here: the log-probability of a response is summed over exactly the positions that would have counted in SFT.' },
    { q: 'In module 07 you ran validation inside `noGrad(fn)` from lib/tensor.js. What does it change?', options: ['It zeroes every gradient', 'Ops inside fn record no graph, so the result cannot be backpropagated through and no memory is kept for it', 'It makes the forward pass faster by skipping layers'], answer: 1,
      why: 'The frozen reference model is only ever run under noGrad: its log-probs are constants, and the DPO gradient must reach the policy only.' },
    { q: 'From module 04: the log-probability a language model assigns to a whole sequence is…', options: ['The log-prob of its last token', 'The sum of the log-probs of each token given the tokens before it', 'The mean of the per-token log-probs'], answer: 1,
      why: 'The chain rule of probability: log p(y₁…yₙ) = Σ log p(yₜ | y₁…yₜ₋₁). DPO compares these sums; averaging instead would change what "preferred" means for responses of different lengths.' },
    { q: 'In module 06, the GPT\'s logits are `x · wteᵀ`. What is `x` at that point?', options: ['The token embeddings', 'The residual stream after the last block and the final LayerNorm, shape [B, T, C]', 'The attention weights'], answer: 1,
      why: 'A reward model reads exactly this tensor. Replacing the tied LM head by a `Linear(C, 1)` on the last position turns a language model into a scorer.' },
  ],
  review: [
    { q: 'With equal scores for chosen and rejected, the Bradley–Terry loss is…', options: ['0', 'log 2 ≈ 0.693', 'Infinity'], answer: 1,
      why: 'σ(0) = 1/2, so −log σ(0) = log 2. This is also the DPO loss on step 0, because the policy starts as the reference and every margin is exactly 0.' },
    { q: 'The gradient of `−log σ(m)` with respect to the margin `m` is `−(1 − σ(m))`. What does that mean for a pair the model already ranks correctly by a wide margin?', options: ['It contributes almost nothing to the update', 'It contributes the most', 'Its gradient is exactly zero'], answer: 0,
      why: '1 − σ(m) → 0 as m grows. The loss spends its gradient on the pairs it gets wrong or barely right, which is also why DPO keeps pushing the pairs whose margin is still small.' },
    { q: 'In DPO, the implicit reward of a response is…', options: ['Its log-probability under the policy', 'β · (log π(y|x) − log ref(y|x)), up to a per-prompt constant that cancels in the pair difference', 'The output of a separate reward head'], answer: 1,
      why: 'Inverting the optimal KL-regularised policy π* = ref · exp(r/β) / Z gives r = β log(π*/ref) + β log Z, and log Z(x) is the same for both responses to the same prompt.' },
    { q: 'What does a reward accuracy of 70% on held-out pairs tell you?', options: ['The model is 70% correct on any task', 'On 7 of 10 unseen pairs the chosen response gets the higher score; chance is 50%, and human annotators themselves agree only about 70–75% of the time', 'The reward model has failed'], answer: 1,
      why: 'Reward accuracy is pairwise ranking accuracy on unseen pairs. Ouyang et al. 2022 measured agreement between their labellers at approximately 73%, and Stiennon et al. 2020 report a similar figure for summaries; that caps how much a reward model can learn from the labels.' },
    { q: 'After DPO the margin on a pair is +2 while the chosen response\'s own log-ratio is −1.9. Is that possible?', options: ['No, a positive margin needs a positive chosen log-ratio', 'Yes: the loss only constrains the difference, so the rejected response fell by more than the chosen one', 'Only if β is negative'], answer: 1,
      why: 'DPO can raise the margin by lowering both log-probs unequally. Production runs log the chosen and rejected log-ratios separately (and variants such as DPO-Positive add a term to stop the chosen one from falling) for exactly this reason.' },
  ],
  concept: `
:::plain
Preference training teaches a model which of two answers people like better, since people can reliably pick between two answers even when they could not write the best one. This module builds a reward model, a scorer trained on such choices, and DPO, a shortcut that trains the assistant directly on preferred and rejected pairs. This stage shapes tone, helpfulness and refusals. Familiar quirks, such as overlong answers or refusing harmless but risky-looking requests, often trace back to it.
:::

## From a comparison to a gradient

SFT (module 10) gives you a model that answers in the right format. It does not say which of two plausible answers is *better*. Humans can rarely write the best answer, but they can usually pick between two. InstructGPT (Ouyang et al. 2022) made the recipe standard: **SFT**, then a **reward model** (RM) trained on human comparisons, then **reinforcement learning** (PPO) towards high reward. This module builds the RM stage, then DPO, which collapses the last two stages into one loss.

The Bradley–Terry model (1952) turns a comparison into a number. Give every response a scalar score \`r\`; the probability that A beats B is \`σ(r_A − r_B)\`, where \`σ(z) = 1 / (1 + exp(−z))\`. If a labeller chose \`y_c\` over \`y_r\`, the loss is the negative log-likelihood of that choice:

\`\`\`
L_BT = −log σ(r_c − r_r)
\`\`\`

Its gradient with respect to the margin \`m = r_c − r_r\` is \`−(1 − σ(m))\`: a pair already ranked with a wide margin contributes almost nothing, a pair ranked wrongly contributes almost a full unit. Chess Elo ratings use the same logistic model (in base 10, with scores scaled by 400), and the Chatbot Arena leaderboard fits it to human votes between chatbots.

:::predict
The reward model gives both responses of a pair the same score. What is the loss, and what is the gradient with respect to the chosen score?
---
The loss is \`−log σ(0) = log 2 ≈ 0.693\` and the gradient is \`−(1 − 1/2) = −0.5\` (divided by the batch size). This is not a corner case: DPO starts with the policy equal to the reference, so on the first update (step 0 in the training loop) every margin is exactly 0 and the whole first update comes from this gradient. An implementation whose subgradient at 0 is 0 never leaves the starting point.
:::

## The reward model

A reward model is the language model with its LM head replaced by \`Linear(C, 1)\` read at the last token, whose hidden state has attended to the whole prompt and response. In production the entire network is fine-tuned on the comparisons; here you keep the 120k-parameter checkpoint frozen and train the head alone. The number you report is **reward accuracy**: the fraction of held-out pairs where the chosen response scores higher. Chance is 50%. Published RMs reach approximately 65–75%: Ouyang et al. 2022 report that their 6B RM predicts the preferences of the labellers it was trained on about 72% of the time and of held-out labellers about 70%, while their labellers agree with *each other* only about 73% of the time. So 70% is close to the ceiling the labels allow, not a weak RM.

## Why a reference model and β

Once you have a reward, "maximise it" fails: the RM is only accurate near its training data, and an optimiser finds the responses it over-scores. This is **reward hacking**; its most common form is **length bias**, since RMs and LLM judges tend to score longer answers higher. The fix is to optimise the reward while staying close to where you started:

\`\`\`
maximise  E[ r(x, y) ]  −  β · KL( π(·|x) || ref(·|x) )
\`\`\`

\`ref\` is a frozen copy of the SFT model and \`β\` prices the drift: small β lets the policy move far for a little reward, large β keeps it near the reference.

## DPO: the policy is its own reward model

Rafailov et al. 2023 noticed that this objective has a closed-form optimum, \`π*(y|x) = ref(y|x) · exp(r(x, y) / β) / Z(x)\`, where \`Z(x)\` sums over all responses and is intractable. Invert it for the reward:

\`\`\`
r(x, y) = β · log( π*(y|x) / ref(y|x) ) + β · log Z(x)
\`\`\`

Substitute into Bradley–Terry. Both responses share the prompt, so \`log Z(x)\` cancels in the difference, and only the policy and the reference remain:

\`\`\`
L_DPO = −log σ( β · [ (log π(y_c|x) − log ref(y_c|x)) − (log π(y_r|x) − log ref(y_r|x)) ] )
\`\`\`

\`β · log(π/ref)\` is the **implicit reward** and the difference is the **margin**. DPO is Bradley–Terry with the policy's own log-ratios as scores; your \`dpoLoss\` will literally call your \`bradleyTerryLoss\`. Each log-probability is the sum over the response tokens of \`log softmax(logits)\`, selected with the mask from module 10.

:::predict
The first DPO step raises the chosen response's log-probability by 0.4 nats and lowers the rejected one's by 0.6. With β = 0.1, what is the margin now, and could it be positive even if the chosen log-probability had gone *down*?
---
Margin = 0.1 · (0.4 − (−0.6)) = 0.1. And yes: the loss constrains only the difference, so DPO can lower both log-probabilities as long as the rejected one falls faster. This happens in real runs, which is why trainers log both log-ratios separately.
:::

## Variants, in one paragraph

:::deeper Going deeper: DPO variants
**IPO** (Azar et al. 2023) replaces the log-sigmoid with a squared loss on the margin, so the policy stops pushing at a target margin instead of overfitting near-deterministic preferences. **KTO** (Ethayarajh et al. 2024) needs no pairs, only good/bad labels on single responses. **ORPO** (Hong et al. 2024) drops the reference and adds an odds-ratio term to the SFT loss. **SimPO** (Meng et al. 2024) also drops the reference and uses the *length-normalised* log-probability with a target margin, a direct attack on length bias. Meta's Llama 3 report (2024) describes rounds of rejection sampling plus DPO on top of SFT; many open pipelines use DPO or a close variant, with PPO or GRPO (module 12) reserved for rewards a verifier can compute.
:::

## Harmlessness is a preference too

Much of safety training reuses this machinery with different labels. Bai et al. 2022a (Anthropic) collected two kinds of comparisons, one for helpfulness and one from red-teaming conversations where the preferred reply is the less harmful one, and found the two pull against each other. Llama 2 (Touvron et al. 2023) trains a separate safety reward model next to the helpfulness one and switches to the safety score on prompts tagged as risky or when that score is low. **Constitutional AI** (Bai et al. 2022b) replaces the human harmlessness labels (human helpfulness labels stay): first the model critiques and revises its own answers against a written list of principles and is fine-tuned on the revisions, then a model's preference labels between pairs of answers train the reward model for RL, which is **RL from AI feedback** (RLAIF). The characteristic failure is **over-refusal**: a model that has learned "refusing scores well" also refuses safe prompts that merely look unsafe, such as how to kill a Python process. XSTest (Röttger et al. 2023) is a test set of such prompts plus unsafe contrast prompts, and a stretch goal in module 13 has you measure both sides. Deployed systems also put input and output classifiers on the request path (Llama Guard, Inan et al. 2023); a stretch goal in module 26 prices an input classifier's latency.

## Where this toy differs from production

Your reward model is a linear probe on a frozen 2-layer, 120k-parameter GPT; a production RM is a full fine-tune of a model with billions of parameters, and your held-out set is 10 pairs, so one pair moves the accuracy by 10 points. You train DPO on 32 pairs for 150 steps of 2 pairs at a learning rate of 1e-4; real runs use approximately 10⁴–10⁶ pairs, batches of 32–128 pairs, learning rates of roughly 5e-7 to 1e-6 with β ≈ 0.1, and one to three epochs, because DPO overfits quickly. You precompute the reference log-probs once, as TRL's \`precompute_ref_log_probs\` option does; large runs often keep the reference model resident instead. And your pairs use a plain \`prompt\\nresponse\` template rather than module 10's chat markers, which the checkpoint has never seen.
`,
  steps: [
    {
      id: 'bradley-terry',
      title: 'The Bradley–Terry loss',
      instructions: `
Two functions on batches of \`N\` scores. Inputs may be plain arrays or Tensors; the worked example \`asTensor\` turns either into a 1-D Tensor.

\`bradleyTerryLoss(rChosen, rRejected)\`: return a scalar Tensor equal to the mean over pairs of \`−log σ(rChosen − rRejected)\`. Two requirements that a naive \`log(1 + exp(−m))\` fails: it must stay finite for margins like −200 (\`exp(200)\` is Infinity in float32), and its gradient at margin exactly 0 must be \`−0.5 / N\` (DPO's first step has every margin at exactly 0). The identity \`log σ(m) = m − logsumexp(m, 0)\` gives you both, and \`Tensor.logSoftmax()\` is a stable logsumexp in disguise.

\`rewardAccuracy(rChosen, rRejected)\`: the fraction of pairs whose chosen score is *strictly* greater than the rejected one; a tie is not a win, and no pairs gives 0. A plain number, no graph needed.

The starter's \`asTensor\`, \`tokenizePair\`, \`shift\`, \`buildPair\`, \`padBatch\` and \`cloneModel\` are done; read them, later steps rely on their shapes.
`,
      predict: { question: 'What does bradleyTerryLoss([0], [5]) return, roughly?', answer: 'About 5.0067: −log σ(−5) = log(1 + e⁵). The loss is the size of the wrong margin plus a little; it is never negative.' },
      hints: [
        'The margin `m = rChosen − rRejected` is a Tensor [N]. You need `−log σ(m)` averaged over N, computed in a way that is stable for large |m| and differentiable at 0.',
        'Build an [N, 2] tensor whose rows are `[m, 0]` (reshape m to [N, 1] and multiply by `Tensor.from([[1, 0]])`; broadcasting does the rest). `logSoftmax()` of that row gives `[log σ(m), log σ(−m)]`; take column 0 with `slice(1, 0, 1)`, negate, mean.',
        '`const margin = asTensor(rChosen).sub(asTensor(rRejected)); const N = margin.shape[0]; const pair = margin.reshape([N, 1]).mul(Tensor.from([[1, 0]])); const logSigmoid = /* … the stable log σ(m) from pair … */; return logSigmoid.neg().mean();` For accuracy: `const c = asTensor(rChosen).data, r = asTensor(rRejected).data;` (plain arrays have no `.data`), then count `c[i] > r[i]`, divide by the length (guard against 0).',
      ],
    },
    {
      id: 'seqlogprob',
      title: 'The log-probability of a response',
      instructions: `
Implement \`sequenceLogProbs(model, batch)\`. \`batch\` is what \`padBatch\` returns: \`x\`, \`y\` and \`mask\` are \`B × T\` arrays. Return a Tensor of shape \`[B]\` whose entry \`b\` is

\`\`\`
Σ_t  mask[b][t] · log softmax(logits[b][t])[ y[b][t] ]
\`\`\`

the log-probability of the response tokens given everything before them. It is a **sum** over the response, not a mean (a mean would make a 3-token answer and a 30-token answer incomparable in the wrong way; SimPO's length normalisation is a deliberate choice, not the default). Masked-out targets (prompt tokens and padding) contribute nothing.

Build it from Tensor ops so \`backward()\` reaches the model: the module-10 trick works here too. Make a \`Float32Array\` of shape \`[B, T, V]\` that holds \`mask[b][t]\` at index \`y[b][t]\` of each row and 0 elsewhere, multiply it with \`logits.logSoftmax()\`, and sum each sequence.
`,
      hints: [
        'One forward pass gives logits [B, T, V]. `logSoftmax()` turns each row into log-probabilities. You need to pick one entry per (b, t), weighted by the mask, and sum over t.',
        'Fill `pick[(b * T + t) * V + y[b][t]] = mask[b][t]`, wrap it as `new Tensor({ shape: [B, T, V], data: pick })`, multiply, then reduce: `reshape([B, T * V]).sum(1)` collapses everything but the batch axis.',
        '`const logits = model.forward(batch.x); const [B, T, V] = logits.shape; const pick = new Float32Array(B * T * V); for (b) for (t) pick[/* … */] = batch.mask[b][t]; return logits.logSoftmax().mul(new Tensor({ shape: [B, T, V], data: pick })).reshape([B, T * V]).sum(1);`',
      ],
    },
    {
      id: 'dpo',
      title: 'The DPO loss',
      instructions: `
Two small functions that make the derivation concrete.

\`implicitRewards(logpPolicy, logpRef, beta)\`: \`β · (log π − log ref)\` elementwise, as a Tensor \`[N]\`. The policy log-probs come from \`sequenceLogProbs\` and carry a graph; the reference log-probs are plain numbers computed once.

\`dpoLoss(policyChosen, policyRejected, refChosen, refRejected, beta)\`: \`−log σ(β[(π_c − ref_c) − (π_r − ref_r)])\` averaged over the batch, as a scalar Tensor. Do not write a second log-sigmoid: compute the two implicit rewards and hand them to your \`bradleyTerryLoss\`. That the code is two lines is the point of the module.

Check by hand: \`dpoLoss([-10], [-12], [-11], [-11], 0.1)\` has margin 0.2 and loss 0.5981.
`,
      predict: { question: 'Policy and reference agree on every pair. What is dpoLoss, and does its gradient with respect to the policy log-probs vanish?', answer: 'log 2 ≈ 0.693, and no: the gradient with respect to log π(chosen) is −β/(2N), so the first step already moves the policy. This is why the Bradley–Terry step tested the gradient at margin exactly 0.' },
      hints: [
        'The implicit reward of a response is β times how much more likely the policy finds it than the reference does. DPO is Bradley–Terry on those two rewards.',
        'Subtract as Tensors (asTensor handles the constants) and `scale(beta)`. Then `bradleyTerryLoss(rewardChosen, rewardRejected)` is the whole loss.',
        '`export function implicitRewards(logpPolicy, logpRef, beta) { return asTensor(logpPolicy).sub(asTensor(logpRef))./* … */; }` and `export function dpoLoss(pc, pr, rc, rr, beta) { return bradleyTerryLoss(implicitRewards(pc, rc, beta), /* … */); }`',
      ],
    },
    {
      id: 'reward-head',
      title: 'A reward model on the final hidden state',
      instructions: `
Three functions. The class \`RewardHead\` (\`Linear(C, 1)\` reshaped to \`[B]\`) is already written.

\`hiddenStates(model, x)\`: the GPT forward pass *without* the LM head: token embeddings plus position embeddings, every block, then the final LayerNorm. Returns \`[B, T, C]\`. \`model.wte\`, \`model.wpe\`, \`model.blocks\` and \`model.lnF\` are public; \`lib/gpt.js\` \`forward\` is eight lines and yours is the same minus the last one. The test multiplies your result by \`wteᵀ\` and expects the model's logits, which catches a forgotten \`lnF\`.

\`lastTokenHidden(h, lengths)\`: from \`[B, T, C]\` pick row \`lengths[b] − 1\` of each sequence (its last real token; the rest is padding), returning \`[B, C]\` with gradient flowing to exactly that row. There is no gather op; a one-hot selector of shape \`[B, T, 1]\` multiplied in and summed over \`T\` does it. Throw if a length is outside \`1..T\`.

\`trainRewardHead(head, features, { steps, lr, weightDecay })\`: full-batch training. \`features\` is an array of \`{ chosen: Float32Array [C], rejected: Float32Array [C] }\`; stack each side into a \`[N, C]\` Tensor once, then each step: score both, \`bradleyTerryLoss\`, backward, AdamW step, zeroGrad, and record \`{ loss, accuracy }\` (both measured on this step's forward pass, before the update). Return the array of \`steps\` records.
`,
      hints: [
        'hiddenStates: which tensor does the LM head read, and which lines of `GPT.forward` in lib/gpt.js produce it? lastTokenHidden: in a right-padded sequence of n real tokens, which position has attended to all of them, and how could a sum over T keep only that one?',
        'hiddenStates is `GPT.forward` stopped before `x.matmul(this.wte.weight.transpose())`, with positions `[0, 1, …, T−1]`. Selector: `select[b * T + lengths[b] − 1] = 1` in a Float32Array of size B·T, wrapped as a Tensor of shape [B, T, 1]. `h.mul(selector)` broadcasts over C; `.sum(1)` removes the T axis. Training: `new AdamW(head.parameters(), { lr, weightDecay })`, then the usual four calls per step.',
        '`let h = model.wte.forward(x).add(model.wpe.forward(positions)); for (const block of model.blocks) h = block.forward(h); return /* … */;` and `return h.mul(new Tensor({ shape: [B, T, 1], data: select })).sum(1);` and in the loop `const loss = bradleyTerryLoss(head.forward(hChosen), head.forward(hRejected)); loss.backward(); /* … */; history.push({ loss: loss.item(), accuracy: rewardAccuracy(rChosen, rRejected) });`',
      ],
    },
    {
      id: 'dpo-train',
      title: 'The DPO training loop',
      instructions: `
\`dpoStep(policy, optimizer, batch, { beta, maxGradNorm })\`: \`batch\` holds \`chosen\` and \`rejected\` (two padded batches from \`padBatch\`) and \`refChosen\`, \`refRejected\` (plain arrays of reference log-probs). Run \`sequenceLogProbs\` on both, take \`dpoLoss\`, \`backward()\`, \`clipGradNorm\`, \`optimizer.step()\`, \`optimizer.zeroGrad()\`. Return \`{ loss, margin, accuracy, gradNorm }\`, where \`gradNorm\` is what \`clipGradNorm\` returns (the global norm *before* clipping), \`margin\` is the mean implicit reward margin and \`accuracy\` the reward accuracy, both measured on this step's forward pass (so the first step, step 0, reports margin 0 and accuracy 0 exactly). Computing those two numbers inside \`noGrad\` keeps them off the graph.

\`trainDPO(policy, pairs, refLogps, opts)\`: \`new AdamW(policy.parameters(), { lr, betas: [0.9, 0.95], weightDecay })\`, then for \`step = 0 … steps − 1\`: draw \`batchSize\` indices with \`randInt(next, pairs.length)\`, pad the chosen examples and the rejected examples separately, look up \`refLogps.chosen[i]\` and \`refLogps.rejected[i]\`, call \`dpoStep\` with \`{ beta, maxGradNorm }\`, push the record, \`await onStep(step, record)\` if given. Return the records.

The reference never appears in the loop: \`referenceLogProbs\` (worked example) computed its numbers once under \`noGrad\`, which is what "frozen" means in practice.
`,
      predict: { question: 'You run trainDPO with beta = 0. What happens to the loss and to the parameters?', answer: 'The loss is log 2 on every step and, with the default weightDecay = 0, the parameters never move: β multiplies every implicit reward, so with β = 0 the margin, and its gradient, are identically zero. (With weightDecay > 0, AdamW\'s decoupled decay would still shrink them, although no preference signal arrives.) β is the volume knob on the preference signal.' },
      hints: [
        'dpoStep is module 10\'s sftStep with two forward passes and a different loss. The only new bookkeeping is the batch: two padded sides plus two arrays of constants.',
        'Order inside dpoStep: forward chosen, forward rejected, dpoLoss, backward, clipGradNorm(policy.parameters(), maxGradNorm), step, zeroGrad. Then `noGrad(() => implicitRewards(policyChosen, batch.refChosen, beta).data)` and the same for rejected give you the margin and accuracy as plain numbers.',
        '`const idx = []; for (b < batchSize) idx.push(randInt(next, pairs.length)); const batch = { chosen: padBatch(idx.map((i) => pairs[i].chosen), padId), rejected: /* … */, refChosen: idx.map((i) => refLogps.chosen[i]), refRejected: /* … */ }; const record = dpoStep(policy, optimizer, batch, { beta, maxGradNorm }); history.push(record); if (onStep) await onStep(step, record);`',
      ],
    },
  ],
  reflection: [
    'Explain to a colleague why DPO needs no reward model: start from "the best policy under a KL penalty is ref · exp(r/β) / Z" and end at a loss that only mentions the policy and the reference.',
    'In the demo the margin grew while some chosen responses became less likely than under the reference. What does the DPO loss actually promise about the chosen response, and what would you add to the loss (or to the logging) before trusting a run?',
    'A reward model reaches 70% on held-out pairs. List two reasons that could be the best achievable number and two reasons it could be a bug in the pipeline.',
  ],
  stretch: [
    'Implement IPO: replace `−log σ(β·h)` with `(h − 1/(2τ))²`, where `h = (π_c − ref_c) − (π_r − ref_r)` is the log-ratio difference *before* any β, and compare how the chosen and rejected log-ratios move over 150 steps. TRL exposes this as `loss_type="ipo"` in its `DPOTrainer`, reusing `beta` as τ.',
    'Implement SimPO: divide each sequence log-prob by its response length, drop the reference, and add a target margin γ; check whether the length gap between chosen and rejected responses still predicts the margin. Meng et al. 2024 report it beats DPO on AlpacaEval 2 and Arena-Hard with no reference model in memory.',
    'Fine-tune the whole body for the reward model instead of only the head (backpropagate through `hiddenStates`) and compare held-out reward accuracy; this is how the InstructGPT and Llama 2 reward models were trained.',
    'Fit Bradley–Terry scores to a small tournament of models by maximum likelihood (the Chatbot Arena leaderboard does this over human votes) and check that the fitted score differences reproduce the observed win rates.',
  ],
  timeouts: { tests: 20000, demo: 120000 },
};
