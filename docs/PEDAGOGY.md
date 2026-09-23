# Pedagogy: why the lab is built the way it is

This curriculum has one thesis: **you understand a system when you can build a working version of it.**
Every design choice in the UI traces back to a specific, well-evidenced learning principle. This document
is the map from principle to feature, so that anyone adding a module keeps the design coherent.

## The learning loop every module follows

```
 Recall  →  Concept  →  Build  →  Goal  →  Reflect
 (retrieve  (read +     (write    (run     (explain it
  earlier    predict)    code,     your     in your own
  modules)               tests)    thing)   words)
```

| Phase   | What the learner does                                                       | Principle behind it |
|---------|------------------------------------------------------------------------------|---------------------|
| Recall  | Answers 3–5 quick questions about *earlier* modules before starting          | Retrieval practice / testing effect; spacing |
| Concept | Reads a short explanation punctuated by "predict before you reveal" cards    | Predict–Observe–Explain; elaborative interrogation |
| Build   | Implements the module in 3–6 steps; each step has tests and a 3-level hint ladder | Constructionism; mastery learning; faded worked examples; scaffolding in the ZPD; productive failure |
| Goal    | Runs a demo that uses *their* code to produce a visible artifact (chart, heatmap, generated text) | Constructionism (public artifact); dual coding; backward design |
| Reflect | Writes a self-explanation; optional stretch goals                             | Self-explanation effect; Feynman technique; transfer |

## Principles and the features that implement them

**Constructionism (Papert 1980; Harel & Papert 1991).** People learn most deeply when they construct a
tangible, shareable artifact. *Feature:* every module ends with a working thing you built (a tokenizer, an
attention layer, a KV cache, a cluster scheduler), and the Goal demo makes it visible. Modules are named
after the artifact, not the topic.

**Backward design / Understanding by Design (Wiggins & McTighe 2005).** Decide the evidence of
understanding first, then design the activities. *Feature:* each module states its `goal` (the artifact and
its observable behaviour) and its `threshold` concept at the top, and every build step is traceable to that
goal. Nothing is taught that the goal does not need.

**Mastery learning (Bloom 1968; Guskey 2007).** Learners should reach mastery of a unit before moving on,
with feedback and correctives rather than a single grade. *Feature:* each step has tests that must pass;
the module completes only when all steps pass and the goal demo has run. Gating is *soft* (you can peek
ahead) because adult learners benefit from autonomy (self-determination theory, Deci & Ryan 2000), but the
UI makes the recommended path obvious.

**Cognitive load theory, worked examples and fading (Sweller 1988; Renkl & Atkinson 2003).** Novices learn
better from worked examples than from unguided problem solving, and scaffolding should fade as competence
grows. *Feature:* starter code ships with at least one fully worked function that shows the conventions;
early steps are completion problems (fill in one piece of a working skeleton); later steps leave more to
you. One new idea per step keeps intrinsic load bounded.

**Scaffolding in the zone of proximal development (Vygotsky 1978; Wood, Bruner & Ross 1976).** Help should
be just enough to let the learner do what they cannot yet do alone. *Feature:* the three-level hint ladder
(nudge → strategy → near-solution) reveals one rung at a time, and only after a first attempt.

**Productive failure (Kapur 2008).** Attempting a problem before instruction improves later learning even
when the attempt fails. *Feature:* hints are locked until you have run the tests at least once; the
"predict" cards ask for a committed answer before revealing.

**Retrieval practice and spacing (Roediger & Karpicke 2006; Cepeda et al. 2006).** Recalling information
strengthens memory far more than re-reading, and spreading recall over time beats massing it. *Feature:*
the Recall phase at the start of each module quizzes earlier modules, and the home page keeps a spaced
review queue (a Leitner box schedule: 1, 3, 7, 14, 30 days) built from every completed module's
questions. A review sitting promotes the module one box when at least three quarters of its questions
are answered correctly (all 3 of 3, or 3 of 4); otherwise it drops back to box 1 and is due again tomorrow.

**Predict–Observe–Explain (White & Gunstone 1992).** Committing to a prediction before observing an outcome
turns passive reading into hypothesis testing. *Feature:* `:::predict` cards and per-step `predict`
prompts; your prediction is saved next to the answer so you can see where your model of the system was wrong.

**Self-explanation (Chi et al. 1989) and the Feynman technique.** Explaining a mechanism in your own words
exposes gaps that recognition hides. *Feature:* the Reflect phase asks 2–3 open prompts and saves your
answers; the recommended practice is to write as if teaching a colleague.

**Deliberate practice with immediate feedback (Ericsson, Krampe & Tesch-Römer 1993).** *Feature:* tests run
in under a few seconds, in the browser, on exactly the code you wrote, with specific failure messages.

**Dual coding (Paivio 1986).** Pairing verbal explanation with a matching visual improves retention.
*Feature:* the Goal demo renders what your code did (loss curves, attention heatmaps, cache hit-rate
charts, cluster timelines).

**Threshold concepts (Meyer & Land 2003).** Some ideas are portals: once grasped they reorganise
everything else. *Feature:* each module names its threshold concept in one sentence, and the reflection
prompts target it.

**Isolation of failure.** A subtle bug in your module-2 autograd should not make module-7 pre-training
mysteriously fail. *Feature:* each module is a self-contained project that imports the vetted reference
implementation of earlier modules from `lib/`. Swapping in your own implementation is a stretch goal, not a
prerequisite.

## Sequencing logic

Tracks follow the life of a model: the numerical substrate (tensors, autograd), the data interface
(tokenizers), the architecture (attention, transformer), how it is built and trained (pre-training,
scaling, data, mixture of experts, image tokens), how it is shaped (SFT, LoRA, distillation, preference
optimisation, RL with verifiable rewards, evals), how it is served (decoding, KV cache, batching, prefix
caching, speculative decoding, attention variants, long context, quantisation), how it is wrapped
(agent harnesses, context management, constrained decoding), and where it physically runs (GPUs,
parallelism, nodes and interconnects, serving clusters). The capstone assembles the pieces into a chat
you can talk to.

Modules keep their original ids when the path is reordered, so a module's number is a name, not its
position. `prereqs` and recall questions must point backwards in the path: a module may build on, and
quiz, only modules that come before it in `MODULES` (modules/index.js). A pointer to a later module
belongs in concept text or a stretch goal, phrased as a forward pointer.

## Authoring rule of thumb

If a sentence in a module cannot be traced to a test the learner will pass, a prediction they will make, or
the goal they will run, cut it.
