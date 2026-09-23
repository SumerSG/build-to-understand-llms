import { Tensor, noGrad } from 'lib/tensor.js';
import * as ops from 'lib/ops.js';

// Helpers shared by the tests. Everything is seeded through T.rng so failures reproduce exactly.
function randTensor(shape, next, requiresGrad = false) {
  return new Tensor(ops.randn(shape, next, 1), { requiresGrad });
}

function makeExperts(m, C, H, E, next) {
  const experts = [];
  for (let e = 0; e < E; e++) experts.push(new m.Expert(C, H, { next }));
  // Larger weights than the 0.02 default so expert outputs are clearly different from each other.
  for (const ex of experts) for (const p of ex.parameters()) for (let i = 0; i < p.data.length; i++) p.data[i] *= 10;
  return experts;
}

function routerWithSpread(m, C, E, k, next) {
  const r = new m.Router(C, E, k, { next });
  // Spread the logits so the routing is decisive rather than near-uniform.
  const w = r.gate.weight;
  for (let i = 0; i < w.data.length; i++) w.data[i] *= 40;
  return r;
}

function row(t, i) {
  const d = t.shape[t.shape.length - 1];
  return Array.from(t.data.subarray(i * d, (i + 1) * d));
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

export const tests = [
  // ---------- step 1: router ----------
  { step: 'router', name: 'topKIndices returns the k largest, largest first, ties to the lower index', run(m, T) {
    T.eq(m.topKIndices([0.1, 0.5, 0.2, 0.2], 1), [1], 'k=1 is the argmax');
    T.eq(m.topKIndices([0.1, 0.5, 0.2, 0.2], 2), [1, 2], 'second place is a tie between experts 2 and 3; the lower index wins so routing is deterministic');
    T.eq(m.topKIndices([0.1, 0.5, 0.2, 0.2], 4), [1, 2, 3, 0], 'k=E returns every expert, sorted by probability');
    T.eq(m.topKIndices(new Float32Array([3, -1, 7, 0, 5]), 3), [2, 4, 0], 'must work on a Float32Array row (a subarray of the probs tensor)');
  } },
  { step: 'router', name: 'probs are a softmax over experts; gates keep only the top-k and renormalise them to sum to 1', run(m, T) {
    const next = T.rng(11);
    const N = 9, C = 6, E = 5, k = 2;
    const r = routerWithSpread(m, C, E, k, next);
    const x = randTensor([N, C], next);
    const { probs, topIdx, gates } = r.forward(x);
    T.shape(probs, [N, E], 'probs must be [N, E]: one distribution over experts per token');
    T.shape(gates, [N, E], 'gates must be [N, E] with zeros for experts a token does not use');
    T.eq(topIdx.length, N, 'topIdx needs one entry per token');
    // Reference softmax(x · W_g) in plain JS, independent of your code.
    const W = r.gate.weight.data;
    const refProbs = [];
    for (let t = 0; t < N; t++) {
      const z = Array.from({ length: E }, (_, e) => { let s = 0; for (let c = 0; c < C; c++) s += x.data[t * C + c] * W[c * E + e]; return s; });
      const mx = Math.max(...z), ex = z.map((v) => Math.exp(v - mx)), s = ex.reduce((a, b) => a + b, 0);
      refProbs.push(...ex.map((v) => v / s));
    }
    T.close(Array.from(probs.data), refProbs, 1e-4, 'probs must be the FULL softmax(x · W_g) over all E experts, not the top-k gates: the load-balancing loss (step 3) needs the probability mass on the experts that were NOT chosen');
    for (let t = 0; t < N; t++) {
      const p = row(probs, t), g = row(gates, t);
      T.close(p.reduce((s, v) => s + v, 0), 1, 1e-4, `probs row ${t} must sum to 1 (softmax of x · W_g)`);
      T.eq(topIdx[t], m.topKIndices(p, k), `token ${t}: topIdx must be the top-${k} of its probs row`);
      T.close(g.reduce((s, v) => s + v, 0), 1, 1e-4, `gates row ${t} must sum to 1 over the chosen experts (renormalise after top-k); otherwise the output shrinks by the discarded mass`);
      for (let e = 0; e < E; e++) if (!topIdx[t].includes(e)) T.eq(g[e], 0, `token ${t}: expert ${e} was not chosen, so its gate must be exactly 0`);
      const [a, b] = topIdx[t];
      T.close(g[a] / g[b], p[a] / p[b], 1e-3, `token ${t}: renormalising must keep the ratio between chosen experts (gate = p_e / Σ chosen p), not replace it with 1/k`);
    }
  } },
  { step: 'router', name: 'k = E reproduces the dense softmax weighting, and gates stay differentiable', run(m, T) {
    const next = T.rng(12);
    const N = 5, C = 4, E = 3;
    const dense = routerWithSpread(m, C, E, E, next);
    const x = randTensor([N, C], next);
    const out = dense.forward(x);
    T.close(Array.from(out.gates.data), Array.from(out.probs.data), 1e-5, 'with k = E nothing is discarded, so the gates must equal the softmax probabilities');
    const r = routerWithSpread(m, C, E, 2, next);
    const { gates } = r.forward(x);
    const w = new Tensor(ops.randn(gates.shape, next, 1));
    try { gates.mul(w).sum().backward(); }
    catch (e) { T.fail(`backward through gates failed (${e.message}): gates must be built from probs with Tensor ops (mul by a 0/1 mask, divide by the row sum), not copied out of probs.data into a new Tensor`); }
    const g = r.gate.weight.grad;
    T.ok(g && g.some((v) => Math.abs(v) > 1e-6), 'the gate weights must receive a gradient through gates: build them with Tensor ops (mul by a 0/1 mask, divide by the row sum), not by copying numbers out of probs.data');
  } },

  // ---------- step 2: dispatch / combine ----------
  { step: 'dispatch', name: 'groupByExpert lists each expert\'s tokens in token order', run(m, T) {
    T.eq(m.groupByExpert([[0, 2], [2, 1], [0, 1]], 4), [[0, 2], [1, 2], [0, 1], []], 'expert 3 receives no tokens and must still get an (empty) list');
    T.eq(m.groupByExpert([[1], [1], [0], [1]], 2), [[2], [0, 1, 3]], 'top-1: every token appears in exactly one list');
  } },
  { step: 'dispatch', name: 'dispatchCombine matches the per-token reference naiveMoE', run(m, T) {
    const next = T.rng(21);
    const N = 10, C = 6, H = 8, E = 4, k = 2;
    const r = routerWithSpread(m, C, E, k, next);
    const experts = makeExperts(m, C, H, E, next);
    const x = randTensor([N, C], next);
    const { topIdx, gates } = r.forward(x);
    const y = m.dispatchCombine(x, gates, m.groupByExpert(topIdx, E), experts);
    T.shape(y, [N, C], 'output must have one row per token');
    const ref = m.naiveMoE(x, gates, topIdx, experts);
    const d = maxAbsDiff(y.data, ref.data);
    T.ok(d < 1e-4, `max |dispatchCombine − naiveMoE| = ${d.toExponential(2)}; each token's output must be Σ over its experts of gate · expert(x). Check that you multiply by the gate and scatter to the right rows`);
  } },
  { step: 'dispatch', name: 'each expert runs once, on exactly its own tokens (N·k rows in total)', run(m, T) {
    const next = T.rng(22);
    const N = 12, C = 4, H = 6, E = 4, k = 2;
    const r = routerWithSpread(m, C, E, k, next);
    const base = makeExperts(m, C, H, E, next);
    const calls = new Array(E).fill(0), rows = new Array(E).fill(0);
    const experts = base.map((ex, e) => ({ forward(x) { calls[e]++; rows[e] += x.shape[0]; return ex.forward(x); }, parameters: () => ex.parameters() }));
    const x = randTensor([N, C], next);
    const { topIdx, gates } = r.forward(x);
    const lists = m.groupByExpert(topIdx, E);
    m.dispatchCombine(x, gates, lists, experts);
    for (let e = 0; e < E; e++) {
      T.ok(calls[e] === (lists[e].length ? 1 : 0), `expert ${e} was called ${calls[e]} times for ${lists[e].length} tokens; batch its tokens into ONE call (and skip experts with no tokens). This is what makes MoE a few big matmuls instead of N tiny ones`);
      T.eq(rows[e], lists[e].length, `expert ${e} must see exactly its ${lists[e].length} routed tokens, not the whole batch (running every expert on every token is a dense model)`);
    }
    T.eq(rows.reduce((a, b) => a + b, 0), N * k, 'in total the experts process N·k rows');
  } },
  { step: 'dispatch', name: 'k = E equals a dense, probability-weighted sum of every expert; gradients reach x and the router', run(m, T) {
    const next = T.rng(23);
    const N = 6, C = 4, H = 5, E = 3;
    const r = routerWithSpread(m, C, E, E, next);
    const experts = makeExperts(m, C, H, E, next);
    const x = randTensor([N, C], next, true);
    const { probs, topIdx, gates } = r.forward(x);
    const y = m.dispatchCombine(x, gates, m.groupByExpert(topIdx, E), experts);
    const expect = noGrad(() => {
      const out = new Float32Array(N * C);
      for (let e = 0; e < E; e++) {
        const ye = experts[e].forward(x).data;
        for (let t = 0; t < N; t++) for (let c = 0; c < C; c++) out[t * C + c] += probs.data[t * E + e] * ye[t * C + c];
      }
      return out;
    });
    T.ok(maxAbsDiff(y.data, expect) < 1e-4, 'with every expert chosen, the MoE layer must equal Σ_e p_e · expert_e(x): a dense mixture');
    y.mul(new Tensor(ops.randn([N, C], next, 1))).sum().backward();
    T.ok(x.grad && x.grad.some((v) => Math.abs(v) > 1e-6), 'x must receive a gradient: gather with gatherRows (embed), not by copying x.data');
    T.ok(r.gate.weight.grad && r.gate.weight.grad.some((v) => Math.abs(v) > 1e-6), 'the router must receive a gradient through the gate weights you multiplied in');
  } },

  // ---------- step 3: load-balancing loss ----------
  { step: 'balance', name: 'routingFractions counts the share of all N·k assignments per expert', run(m, T) {
    T.close(m.routingFractions([[0], [0], [1], [3]], 4), [0.5, 0.25, 0, 0.25], 1e-9, 'top-1: 2 of 4 tokens went to expert 0');
    T.close(m.routingFractions([[0, 1], [0, 2], [0, 1]], 3), [0.5, 1 / 3, 1 / 6], 1e-9, 'top-2: divide by N·k = 6 assignments so the fractions sum to 1');
  } },
  { step: 'balance', name: 'loss is E · Σ f_i · P_i: 1 when balanced, E when collapsed, exact on a hand example', run(m, T) {
    const E = 4;
    const uniform = new Tensor(ops.full([8, E], 1 / E));
    const balanced = [[0], [1], [2], [3], [0], [1], [2], [3]];
    const l1 = m.loadBalanceLoss(uniform, balanced);
    T.ok(l1 instanceof Tensor && l1.size === 1, 'loadBalanceLoss must return a scalar Tensor so it can be added to the LM loss');
    T.close(l1.item(), 1, 1e-5, 'perfect balance (f_i = P_i = 1/E) must give exactly 1; a missing factor E gives 1/E');
    const collapsedP = ops.zeros([8, E]);
    for (let t = 0; t < 8; t++) collapsedP.data[t * E] = 1;
    T.close(m.loadBalanceLoss(new Tensor(collapsedP), balanced.map(() => [0])).item(), E, 1e-5, 'everything on expert 0 with probability 1 must give E, the maximum');
    const P = new Tensor(ops.fromArray([[0.7, 0.1, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1]]));
    // f = [0.5, 0.25, 0, 0.25] over 4 assignments, P = mean rows = [0.55, 0.25, 0.1, 0.1]
    // loss = 4 · (0.5·0.55 + 0.25·0.25 + 0 + 0.25·0.1) = 1.45
    T.close(m.loadBalanceLoss(P, [[0, 3], [0, 1]]).item(), 1.45, 1e-5, 'hand example with top-2: f from counts over N·k, P = column mean of probs');
  } },
  { step: 'balance', name: 'the gradient pushes probability away from the overloaded expert', run(m, T) {
    const next = T.rng(31);
    const N = 16, E = 4;
    const logits = new Tensor(ops.randn([N, E], next, 0.1), { requiresGrad: true });
    for (let t = 0; t < N; t++) logits.data[t * E] += 2; // expert 0 favoured by every token
    const probs = logits.softmax();
    const topIdx = Array.from({ length: N }, () => [0]);
    m.loadBalanceLoss(probs, topIdx).backward();
    T.ok(logits.grad !== null, 'the loss must be differentiable with respect to the router logits (through P, the mean probability)');
    let g0 = 0, gOther = 0;
    for (let t = 0; t < N; t++) { g0 += logits.grad[t * E]; for (let e = 1; e < E; e++) gOther += logits.grad[t * E + e]; }
    T.ok(g0 > 1e-4, `gradient on the overloaded expert's logits should be positive (descent lowers them), got ${g0.toExponential(2)}`);
    T.ok(gOther < -1e-4, `gradient on the idle experts' logits should be negative (descent raises them), got ${gOther.toExponential(2)}`);
  } },

  // ---------- step 4: capacity ----------
  { step: 'capacity', name: 'expertCapacity = ceil(factor · N · k / E)', run(m, T) {
    T.eq(m.expertCapacity(100, 4, 1, 1.25), 32, 'Switch-style top-1: 1.25 · 100 / 4 = 31.25, rounded UP to 32 slots');
    T.eq(m.expertCapacity(100, 4, 2, 1), 50, 'top-2 doubles the assignments to place, so each expert needs 2 · 100 / 4 = 50 slots at factor 1');
    T.eq(m.expertCapacity(7, 3, 1, 1), 3, '7 / 3 = 2.33 must round up to 3; rounding down would drop a token even under perfect balance');
    T.eq(m.expertCapacity(64, 8, 2, 8), 128, 'factor = E gives N·k slots per expert: room for every assignment');
  } },
  { step: 'capacity', name: 'applyCapacity keeps the first tokens, counts the drops and does not mutate its input', run(m, T) {
    const lists = [[0, 1, 2, 3], [4], [5, 6]];
    const { kept, dropped } = m.applyCapacity(lists, 2);
    T.eq(kept, [[0, 1], [4], [5, 6]], 'token order is priority: keep the earliest tokens and drop the tail (Switch Transformer does the same)');
    T.eq(dropped, 2, 'expert 0 had 4 tokens for 2 slots: 2 assignments dropped');
    T.eq(lists, [[0, 1, 2, 3], [4], [5, 6]], 'the input lists must not be modified (the router statistics still need them)');
    T.eq(m.applyCapacity(lists, 4).dropped, 0, 'with capacity 4 nothing is dropped');
  } },
  { step: 'capacity', name: 'factor ≥ E never drops; a tight factor drops exactly the overflow', run(m, T) {
    const next = T.rng(41);
    for (const [N, E, k] of [[20, 4, 1], [33, 4, 2], [50, 8, 2]]) {
      const topIdx = [];
      for (let t = 0; t < N; t++) {
        const skew = Array.from({ length: E }, (_, e) => next() + (e === 0 ? 0.6 : 0)); // expert 0 is popular
        topIdx.push(skew.map((v, e) => [v, e]).sort((p, q) => q[0] - p[0]).slice(0, k).map((p) => p[1])); // independent of your router
      }
      const lists = m.groupByExpert(topIdx, E);
      T.eq(m.applyCapacity(lists, m.expertCapacity(N, E, k, E)).dropped, 0, `N=${N}, E=${E}, k=${k}: factor E must leave room for every assignment`);
      const cap = m.expertCapacity(N, E, k, 1);
      const expected = lists.reduce((s, l) => s + Math.max(0, l.length - cap), 0);
      T.eq(m.applyCapacity(lists, cap).dropped, expected, `N=${N}, E=${E}, k=${k}: at factor 1 (capacity ${cap}) the drops must equal the overflow Σ max(0, count − capacity)`);
    }
  } },

  // ---------- step 5: the MoE layer and accounting ----------
  { step: 'moe', name: 'MoE.forward keeps the [B, T, C] shape and equals router → dispatch → combine', run(m, T) {
    const next = T.rng(51);
    const B = 2, Tn = 5, C = 6, E = 4, k = 2;
    const layer = new m.MoE({ nEmbd: C, nExperts: E, k, hidden: 8, next });
    for (const p of layer.parameters()) for (let i = 0; i < p.data.length; i++) p.data[i] *= 20;
    const x = randTensor([B, Tn, C], next);
    const y = layer.forward(x);
    T.shape(y, [B, Tn, C], 'the MoE layer is a drop-in replacement for the MLP, so it must return the input shape');
    const flat = new Tensor(ops.reshape(x, [B * Tn, C]));
    const { topIdx, gates } = layer.router.forward(flat);
    const ref = m.naiveMoE(flat, gates, topIdx, layer.experts);
    T.ok(maxAbsDiff(y.data, ref.data) < 1e-3, 'with no capacity limit, MoE.forward must equal naiveMoE on the flattened tokens (did you return x unchanged, or forget to reshape?)');
  } },
  { step: 'moe', name: 'MoE.forward records the aux loss and routing statistics, and drops the overflow', run(m, T) {
    const next = T.rng(52);
    const N = 24, C = 4, E = 4, k = 2;
    const layer = new m.MoE({ nEmbd: C, nExperts: E, k, hidden: 6, capacityFactor: 0.75, next });
    for (let i = 0; i < layer.router.gate.weight.data.length; i++) layer.router.gate.weight.data[i] *= 50;
    for (const ex of layer.experts) for (const p of ex.parameters()) for (let i = 0; i < p.data.length; i++) p.data[i] *= 10;
    const x = randTensor([1, N, C], next);
    const y = layer.forward(x);
    T.ok(layer.lastAux instanceof Tensor && layer.lastAux.size === 1, 'forward must set this.lastAux to the scalar load-balancing loss');
    const { probs, topIdx } = layer.router.forward(new Tensor(ops.reshape(x, [N, C])));
    T.close(layer.lastAux.item(), m.loadBalanceLoss(probs, topIdx).item(), 1e-5, 'lastAux must be loadBalanceLoss of this forward pass\'s probs and routing');
    const s = layer.lastStats;
    T.ok(s && Array.isArray(s.counts), 'forward must set this.lastStats = { counts, dropped, assignments, capacity }');
    const lists = m.groupByExpert(topIdx, E);
    T.eq(s.counts, lists.map((l) => l.length), 'counts are tokens routed to each expert BEFORE capacity is applied');
    T.eq(s.assignments, N * k, 'assignments = N · k');
    const cap = m.expertCapacity(N, E, k, 0.75);
    T.eq(s.capacity, cap, 'capacity = expertCapacity(N, E, k, capacityFactor)');
    T.eq(s.dropped, m.applyCapacity(lists, cap).dropped, 'dropped must come from applyCapacity at that capacity');
    T.ok(s.dropped > 0, 'a factor of 0.75 must drop something; is the capacity limit applied before dispatch?');
    // The drops must be real: a dropped assignment contributes nothing to the output.
    const { kept } = m.applyCapacity(lists, cap);
    const keptIdx = topIdx.map((choices, t) => choices.filter((e) => kept[e].includes(t)));
    const flat = new Tensor(ops.reshape(x, [N, C]));
    const { gates } = layer.router.forward(flat);
    const capped = m.naiveMoE(flat, gates, keptIdx, layer.experts);
    const uncapped = m.naiveMoE(flat, gates, topIdx, layer.experts);
    T.ok(maxAbsDiff(capped.data, uncapped.data) > 1e-3, 'internal check: dropping must change the reference output');
    const d = maxAbsDiff(y.data, capped.data);
    T.ok(d < 1e-3, `max |MoE.forward − reference with the over-capacity assignments removed| = ${d.toExponential(2)}: dispatch the KEPT lists, not the full ones. Counting drops without removing them is a layer that reports a capacity it does not enforce`);
  } },
  { step: 'moe', name: 'countParams: total vs active parameters and FLOPs per token', run(m, T) {
    T.eq(m.countParams({ nEmbd: 32, hidden: 64, nExperts: 4, k: 2 }), { total: 16896, active: 8512, flopsPerToken: 17024 },
      'expert = 2·C·H + H + C = 4192; router = C·E = 128; total = 4·4192 + 128; active = 2·4192 + 128; FLOPs = 2·active');
    const layer = new m.MoE({ nEmbd: 12, nExperts: 3, k: 1, hidden: 20, next: T.rng(53) });
    const actual = layer.parameters().reduce((s, p) => s + p.size, 0);
    T.eq(m.countParams({ nEmbd: 12, hidden: 20, nExperts: 3, k: 1 }).total, actual, 'total must match the number of scalars in MoE.parameters()');
    const big = m.countParams({ nEmbd: 4096, hidden: 14336, nExperts: 8, k: 2 });
    T.close(big.active / big.total, 0.25, 1e-3, 'at Mixtral-like widths, top-2 of 8 activates about k/E = 1/4 of the expert parameters');
  } },
];
