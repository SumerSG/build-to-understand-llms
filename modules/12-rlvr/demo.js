import { MATH_TASKS } from 'lib/data.js';
import { rng, argmaxArray } from 'lib/util.js';

// The goal: YOUR GRPO loop, on YOUR verifier, raises a policy's accuracy on 30 arithmetic tasks from a
// uniform start, with no reward model and no value network: only "was it right", sampled G at a time.
// A second, shorter run starts from an over-confident warm-up so you can see what starves GRPO of signal.
export default async function demo(m, lab) {
  const tasks = MATH_TASKS.slice(0, 30);            // the RL prompts
  // Used only by the warm-start run's SFT stand-in. MATH_TASKS repeats some questions, so drop any that is
  // also an RL prompt: the warm start must not have seen the answers it is about to be rewarded for.
  const rlQuestions = new Set(tasks.map((t) => t.question));
  const otherTasks = MATH_TASKS.slice(60, 120).filter((t) => !rlQuestions.has(t.question));
  const G = 8, iterations = 40, batchSize = 16, beta = 0.02, clip = 0.2, lr = 0.1;
  const pct = (x) => `${(100 * x).toFixed(0)}%`;
  const topAnswer = (policy, task) => { const p = policy.probs(task.question); const t = argmaxArray(p); return { text: String(t), prob: p[t] }; };

  // 1. Cold start: a uniform policy over the 100 possible answers.
  const policy = new m.Policy();
  const reference = policy.clone();
  const accBefore = m.accuracy(policy, tasks);
  const entropyBefore = tasks.reduce((s, t) => s + m.entropyOf(policy.probs(t.question)), 0) / tasks.length;
  const showTasks = tasks.slice(0, 6);
  const before = showTasks.map((t) => topAnswer(policy, t));
  lab.log(`Cold start: ${tasks.length} tasks, policy uniform over ${m.ANSWER_VOCAB} answers. Greedy accuracy ${pct(accBefore)} (a uniform argmax picks answer 0, right for the ${tasks.filter((t) => Number(t.answer) === 0).length} prompt(s) of the form x - x), entropy ${entropyBefore.toFixed(2)} nats (ln ${m.ANSWER_VOCAB} = ${Math.log(m.ANSWER_VOCAB).toFixed(2)}).`);
  lab.log(`With G = ${G} samples per prompt and a 1% hit rate, the chance that a group has at least one correct sample is 1 - 0.99^${G} = ${pct(1 - 0.99 ** G)}: most groups carry no signal at first.`);

  const accuracyCurve = [];
  const history = await m.trainGRPO(policy, reference, tasks, {
    iterations, G, batchSize, beta, clip, mu: 1, lr, next: rng(1),
    onIter: async (record, i) => {
      accuracyCurve.push(m.accuracy(policy, tasks));
      if (i % 5 === 4) lab.log(`iter ${String(i + 1).padStart(2)}: reward ${record.reward.toFixed(2)}  accuracy ${pct(accuracyCurve.at(-1))}  KL ${record.kl.toFixed(3)}  entropy ${record.entropy.toFixed(2)}  mixed groups ${pct(record.signalFrac)}`);
      lab.progress(0.05 + 0.7 * ((i + 1) / iterations), `GRPO iteration ${i + 1}/${iterations}`);
      await lab.tick();
    },
  });
  const accAfter = m.accuracy(policy, tasks);
  lab.log('Greedy accuracy moves before the sampled reward does: on a near-uniform policy a single rewarded sample is enough to lift its answer to the argmax, while the mean sampled reward only rises once the probability mass itself has moved there, so it lags behind.');
  const after = showTasks.map((t) => topAnswer(policy, t));
  const last = history.at(-1);
  const meanSignal = history.reduce((s, h) => s + h.signalFrac, 0) / history.length;
  const x = history.map((h) => h.iteration + 1);
  lab.plot({
    title: 'Cold start: greedy accuracy, mean sampled reward and the fraction of groups with mixed rewards',
    x,
    series: [
      { name: 'greedy accuracy (30 tasks)', values: accuracyCurve },
      { name: 'mean reward of the rollout', values: history.map((h) => h.reward) },
      { name: 'groups with signal', values: history.map((h) => h.signalFrac) },
    ],
    xlabel: 'GRPO iteration', ylabel: 'fraction',
  });
  lab.plot({
    title: 'KL(policy || reference) and policy entropy',
    x,
    series: [
      { name: 'KL to the uniform reference (k3, nats)', values: history.map((h) => h.kl) },
      { name: 'entropy (nats)', values: history.map((h) => h.entropy) },
    ],
    xlabel: 'GRPO iteration', ylabel: 'nats',
  });
  lab.table({
    title: 'Six prompts: the policy\'s most likely answer before and after GRPO',
    columns: ['question', 'answer', 'before', 'p(before)', 'after', 'p(after)'],
    rows: showTasks.map((t, i) => [t.question, t.answer, before[i].text, +before[i].prob.toFixed(3), after[i].text, +after[i].prob.toFixed(3)]),
  });
  lab.progress(0.78, 'warm-start run');
  await lab.tick();

  // 2. Warm start: the SFT stand-in on *other* questions makes the policy confident about typical answers
  //    before it has seen these prompts. Same GRPO budget; does confidence help or hurt?
  const warm = new m.Policy();
  const sftLoss = m.sftWarmup(warm, otherTasks, { epochs: 20, lr: 0.05 });
  const warmRef = warm.clone();
  const warmAccBefore = m.accuracy(warm, tasks);
  const warmEntropyBefore = tasks.reduce((s, t) => s + m.entropyOf(warm.probs(t.question)), 0) / tasks.length;
  const warmCurve = [];
  const warmHistory = await m.trainGRPO(warm, warmRef, tasks, {
    iterations, G, batchSize, beta, clip, mu: 1, lr, next: rng(1),
    onIter: async (record, i) => {
      warmCurve.push(m.accuracy(warm, tasks));
      if (i % 10 === 9) { lab.progress(0.78 + 0.2 * ((i + 1) / iterations), `warm start ${i + 1}/${iterations}`); await lab.tick(); }
    },
  });
  const warmAccAfter = m.accuracy(warm, tasks);
  const warmSignal = warmHistory.reduce((s, h) => s + h.signalFrac, 0) / warmHistory.length;
  lab.log(`Warm start: 20 SFT epochs on ${otherTasks.length} other questions, none of them an RL prompt (final cross-entropy ${sftLoss.toFixed(2)}, so it has memorised them), give greedy accuracy ${pct(warmAccBefore)} and entropy ${warmEntropyBefore.toFixed(2)} nats on the 30 RL tasks before any RL; after ${iterations} iterations: ${pct(warmAccAfter)}, mixed groups on average ${pct(warmSignal)} vs ${pct(meanSignal)} for the cold start.`);
  lab.plot({
    title: 'Same GRPO budget, two starting points: greedy accuracy per iteration',
    x,
    series: [
      { name: `cold start (uniform, entropy ${entropyBefore.toFixed(1)})`, values: accuracyCurve },
      { name: `warm start (SFT on other tasks, entropy ${warmEntropyBefore.toFixed(1)})`, values: warmCurve },
    ],
    xlabel: 'GRPO iteration', ylabel: 'greedy accuracy',
  });
  lab.progress(1, 'done');

  const began = warmAccBefore > accBefore ? ', even though it began higher' : warmAccBefore < accBefore ? '' : ', from the same starting accuracy';
  const warmVerdict = warmAccAfter < accAfter
    ? `finished lower (${pct(warmAccAfter)}) than the cold start${began}: SFT on other questions taught it confident guesses (entropy ${warmEntropyBefore.toFixed(2)} nats) but not these answers (a linear policy over hashed features memorises, it does not generalise), so fewer of its groups had mixed rewards (${pct(warmSignal)} vs ${pct(meanSignal)} per iteration), and a group whose rewards all agree gives GRPO nothing to learn from`
    : `finished at ${pct(warmAccAfter)} vs ${pct(accAfter)} for the cold start: on this seed its prior helped, and its mixed-group rate was ${pct(warmSignal)} vs ${pct(meanSignal)}`;
  lab.done(`Your GRPO loop took greedy accuracy on ${tasks.length} arithmetic tasks from **${pct(accBefore)}** to **${pct(accAfter)}** in ${iterations} iterations of G = ${G} (mean sampled reward ${history[0].reward.toFixed(2)} -> ${last.reward.toFixed(2)}), with no reward model and no value network: only exact-match rewards, group-normalised inside each prompt's ${G} samples. Entropy fell from ${entropyBefore.toFixed(2)} to ${last.entropy.toFixed(2)} nats while the KL to the uniform reference grew to ${last.kl.toFixed(2)} nats (beta = ${beta}); on average only ${pct(meanSignal)} of groups per iteration had mixed rewards, and those groups did all the work. The warm-started policy, which began at ${pct(warmAccBefore)}, ${warmVerdict}.`);
}
