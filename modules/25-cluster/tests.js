// Module 25 — tests. Everything here is arithmetic on plain numbers, so every expectation is a value
// you can check by hand. Where a number is not obvious, the comment shows the calculation.

// A toy cluster with round bandwidths and ZERO latency, so every expected time is exact:
// 2 GPUs per node, 2 nodes per pod (so a pod is GPUs 0-3, the next pod 4-7).
// node link 1 MB/s, pod link 100 KB/s, spine 10 KB/s.
const TOY = {
  name: 'toy cluster',
  gpu: { name: 'toy gpu', flops: 1e12, memory: 80e9, mfu: 0.5, watts: 1 },
  gpusPerNode: 2,
  nodesPerPod: 2,
  links: {
    self: { name: 'toy memory', tier: 'self', latency: 0, bandwidth: 1e12 },
    node: { name: 'toy node link', tier: 'node', latency: 0, bandwidth: 1e6 },
    pod: { name: 'toy pod link', tier: 'pod', latency: 0, bandwidth: 1e5 },
    cluster: { name: 'toy spine', tier: 'cluster', latency: 0, bandwidth: 1e4 },
  },
  overlap: 0,
};

// A model small enough that every term of a step can be multiplied out by hand.
const TINY = { name: 'tiny', params: 1e9, layers: 8, dModel: 512, dFF: 2048, seqLen: 128, batchSeqs: 8 };

const range = (n) => [...Array(n).keys()];

export const tests = [
  // ---------- step 1: the bandwidth hierarchy ----------
  {
    step: 'topology',
    name: 'location puts a GPU id in a node and a pod',
    run(m, T) {
      T.eq(m.location(0, TOY).node, 0);
      T.eq(m.location(1, TOY).node, 0, 'GPUs 0 and 1 share node 0 when gpusPerNode is 2');
      T.eq(m.location(2, TOY).node, 1, 'GPU 2 starts the second node');
      T.eq(m.location(3, TOY).pod, 0, 'a pod holds gpusPerNode x nodesPerPod = 4 GPUs, so 0-3 is pod 0');
      T.eq(m.location(4, TOY).pod, 1, 'GPU 4 starts the second pod');
      T.eq(m.location(5, TOY).slot, 1, 'GPU 5 is the second GPU of its node');
      // The real cluster: 8 per node, 8 nodes per pod = 64 per pod.
      T.eq(m.location(70, m.CLUSTER).node, 8, '70 / 8 = 8 (floor)');
      T.eq(m.location(70, m.CLUSTER).pod, 1, '70 / 64 = 1 (floor)');
      T.eq(m.location(70, m.CLUSTER).slot, 6, '70 mod 8 = 6');
      T.eq(m.location(6, TOY).slot, 0, 'GPU 6 is the FIRST GPU of node 3: slot is the position inside the node (6 mod 2), not inside the pod');
      T.eq(m.location(13, m.CLUSTER).slot, 5, '13 mod 8 = 5: slot counts inside the node of 8, not the pod of 64');
      // A topology where gpusPerNode != nodesPerPod, so mixing the two up cannot pass by coincidence.
      // 4 GPUs per node, 2 nodes per pod: a pod is 8 GPUs.
      const lopsided = { ...TOY, gpusPerNode: 4, nodesPerPod: 2 };
      T.eq(m.location(9, lopsided).node, 2, '9 / 4 = 2 (floor)');
      T.eq(m.location(9, lopsided).pod, 1, 'a pod is gpusPerNode x nodesPerPod = 8 GPUs, so 9 / 8 = 1. Dividing the node by gpusPerNode instead gives 0');
      T.eq(m.location(9, lopsided).slot, 1, '9 mod 4 = 1');
      T.eq(m.location(7, lopsided).pod, 0, 'GPUs 0-7 are pod 0');
      T.throws(() => m.location(-1, TOY), 'a negative GPU id is not a place in the machine and must throw');
      T.throws(() => m.location(1.5, TOY), 'a fractional GPU id must throw rather than silently floor');
    },
  },
  {
    step: 'topology',
    name: 'linkBetween picks the rung of the hierarchy the pair actually uses',
    run(m, T) {
      T.eq(m.linkBetween(0, 0, TOY).tier, 'self', 'a GPU talking to itself uses its own memory, not a network');
      T.eq(m.linkBetween(0, 1, TOY).tier, 'node', 'GPUs 0 and 1 share a node');
      T.eq(m.linkBetween(0, 2, TOY).tier, 'pod', 'different nodes, same pod: one leaf-switch hop');
      T.eq(m.linkBetween(0, 4, TOY).tier, 'cluster', 'different pods: through the spine');
      T.eq(m.linkBetween(4, 0, TOY).tier, 'cluster', 'the link must not depend on the order of the arguments');
      T.eq(m.linkBetween(7, 3, m.CLUSTER).tier, 'node', 'in the real cluster GPUs 0-7 are one node, so 3 and 7 are neighbours');
      T.eq(m.linkBetween(7, 8, m.CLUSTER).tier, 'pod', 'GPU 8 is the first GPU of node 1, one hop away');
      T.eq(m.linkBetween(7, 64, m.CLUSTER).tier, 'cluster', 'GPU 64 opens the second pod');
      T.ok(
        m.CLUSTER.links.self.bandwidth > m.CLUSTER.links.node.bandwidth &&
        m.CLUSTER.links.node.bandwidth > m.CLUSTER.links.pod.bandwidth &&
        m.CLUSTER.links.pod.bandwidth >= m.CLUSTER.links.cluster.bandwidth,
        'the four rungs must be strictly ordered: HBM > NVLink > leaf > spine. That ordering is the whole module',
      );
    },
  },
  {
    step: 'topology',
    name: 'slowestLink reports the worst pair in a group, not the first or the average',
    run(m, T) {
      T.eq(m.slowestLink([3], TOY).tier, 'self', 'a group of one communicates with nobody');
      T.eq(m.slowestLink([0, 1], TOY).tier, 'node');
      T.eq(m.slowestLink([0, 1, 2, 3], TOY).tier, 'pod', 'two nodes in one pod: the pod link sets the cost');
      T.eq(m.slowestLink([0, 1, 2, 3, 4, 5, 6, 7], TOY).tier, 'cluster');
      // The worst pair is NOT the adjacent one: 5 and 6 are in different nodes of pod 1, 1 is in pod 0.
      T.eq(m.slowestLink([5, 6], TOY).tier, 'pod');
      T.eq(m.slowestLink([1, 5, 6], TOY).tier, 'cluster', 'adding a GPU from another pod must drag the whole group down to the spine');
      T.eq(m.slowestLink([6, 5, 1], TOY).tier, 'cluster', 'the answer must not depend on the order of the group');
      // The worst pair can be in the MIDDLE of an unsorted group: comparing only the ends is not enough.
      T.eq(m.slowestLink([0, 4, 1], TOY).tier, 'cluster', 'GPU 4 is in the other pod even though the first and last ids (0, 1) share a node: check every GPU, not just the ends');
      T.eq(m.slowestLink([0, 8, 1], m.CLUSTER).tier, 'pod', 'GPU 8 is on node 1; the group is not one node just because 0 and 1 are');
      T.eq(m.slowestLink(range(8), m.CLUSTER).tier, 'node', 'the first 8 GPUs of the real cluster are exactly one node');
      T.eq(m.slowestLink(range(9), m.CLUSTER).tier, 'pod', 'one extra GPU crosses a node boundary and costs a switch hop');
      T.throws(() => m.slowestLink([], TOY), 'an empty group is a bug in the caller and must throw');
    },
  },

  // ---------- step 2: collectives on the hierarchy ----------
  {
    step: 'collective',
    name: 'the flat ring pays the slowest link for every one of its 2(n-1) steps',
    run(m, T) {
      // 8 GPUs spanning both pods: slowest link 1e4 B/s, chunk 8e6/8 = 1e6 B, 2(8-1) = 14 steps.
      T.close(m.flatAllReduceTime(8e6, range(8), TOY), 1400, 1e-9,
        'expected 14 steps x (1 MB / 10 KB/s) = 1400 s. Using the node link (1.4 s) or sending the whole buffer each step (11200 s) is wrong');
      // 4 GPUs inside one pod: 1e5 B/s, chunk 8e6/4 = 2e6, 6 steps.
      T.close(m.flatAllReduceTime(8e6, range(4), TOY), 120, 1e-9, 'expected 6 steps x (2 MB / 100 KB/s) = 120 s');
      T.close(m.flatAllReduceTime(8e6, [0, 1], TOY), 8, 1e-9, 'expected 2 steps x (4 MB / 1 MB/s) = 8 s on the node link');
      T.close(m.flatAllReduceTime(8e6, [3], TOY), 0, 1e-12, 'a single GPU does no communication at all');
    },
  },
  {
    step: 'collective',
    name: 'the hierarchical schedule puts only bytes/g on the slow link',
    run(m, T) {
      const r = m.hierarchicalAllReduce(8e6, range(8), TOY);
      T.eq(r.nodes, 4, 'eight GPUs at two per node is four nodes');
      T.eq(r.gpusPerNode, 2, 'the group has two GPUs on each node it touches');
      T.eq(r.crossLink.tier, 'cluster', 'the four nodes span two pods, so the cross-node phase runs over the spine');
      T.eq(r.phases.length, 3, 'reduce-scatter in node, all-reduce across nodes, all-gather in node');
      // chunk = 8e6 / 2 = 4e6 bytes.
      T.close(r.phases[0].time, 4, 1e-9, 'reduce-scatter over 2 GPUs in a node: 1 step x (4 MB / 1 MB/s) = 4 s');
      T.close(r.phases[1].time, 600, 1e-9, 'all-reduce of the 4 MB chunk over 4 nodes on the spine: 6 steps x (1 MB / 10 KB/s) = 600 s');
      T.close(r.phases[2].time, 4, 1e-9, 'all-gather mirrors the reduce-scatter');
      T.close(r.time, 608, 1e-9, 'the three phases are sequential: 4 + 600 + 4');
      T.close(r.phases[1].bytes, 4e6, 1e-9, 'the slow link must carry bytes/g = 8 MB / 2, not the whole 8 MB');
    },
  },
  {
    step: 'collective',
    name: 'inside one node the hierarchical schedule degenerates to the plain ring, and uneven groups throw',
    run(m, T) {
      const inNode = m.hierarchicalAllReduce(1e9, range(8), m.CLUSTER);
      T.eq(inNode.nodes, 1, 'the first 8 GPUs are one node');
      T.close(inNode.time, m.ringAllReduceTime(1e9, 8, m.NVLINK), 1e-9,
        'with one node there is nothing to do across nodes, and reduce-scatter + all-gather IS the ring all-reduce');
      const oneEach = m.hierarchicalAllReduce(8e6, [0, 2, 4, 6], TOY);
      T.eq(oneEach.gpusPerNode, 1, 'one GPU per node means no intra-node phase');
      T.eq(oneEach.phases.length, 1, 'with g = 1 only the cross-node all-reduce remains');
      T.close(oneEach.time, m.ringAllReduceTime(8e6, 4, TOY.links.cluster), 1e-9,
        'expected the plain 4-way ring on the spine: 6 steps x (2 MB / 10 KB/s) = 1200 s');
      T.throws(() => m.hierarchicalAllReduce(8e6, [0, 1, 2], TOY),
        'a group with 2 GPUs on node 0 and 1 on node 1 has no symmetric schedule; it must throw rather than quietly use the wrong g');
    },
  },
  {
    step: 'collective',
    name: 'allReduceTime never loses to the flat ring and wins by roughly g on a big buffer',
    run(m, T) {
      T.close(m.allReduceTime(8e6, range(8), TOY), 608, 1e-9, 'the hierarchical schedule (608 s) must be chosen over the flat ring (1400 s)');
      T.close(m.allReduceTime(1e9, [5], TOY), 0, 1e-12, 'one GPU communicates with nobody');
      let uneven;
      try { uneven = m.allReduceTime(8e6, [0, 1, 2], TOY); } catch (e) {
        T.fail(`allReduceTime over the uneven group [0, 1, 2] threw (${e.message}); with no symmetric hierarchical schedule it must fall back to the flat ring`);
      }
      T.close(uneven, m.flatAllReduceTime(8e6, [0, 1, 2], TOY), 1e-9,
        'an uneven group (2 GPUs on node 0, 1 on node 1) has no hierarchical schedule, so allReduceTime must return the flat ring time');
      const next = T.rng(25);
      for (let i = 0; i < 20; i++) {
        const bytes = Math.floor(next() * 4e9) + 1e6;
        const n = [2, 4, 8, 16, 32, 64, 128][Math.floor(next() * 7)];
        const gpus = range(n);
        const chosen = m.allReduceTime(bytes, gpus, m.CLUSTER);
        T.ok(chosen <= m.flatAllReduceTime(bytes, gpus, m.CLUSTER) + 1e-12,
          `allReduceTime must never be slower than the flat ring (${bytes} B over ${n} GPUs)`);
        T.ok(chosen > 0, 'a real collective over more than one GPU takes real time');
      }
      // 16 GB of gradients over 128 GPUs (16 nodes): the spine carries 1/8 of the buffer.
      const flat = m.flatAllReduceTime(16e9, range(128), m.CLUSTER);
      const best = m.allReduceTime(16e9, range(128), m.CLUSTER);
      T.ok(flat / best > 4, `a two-level all-reduce of 16 GB over 128 GPUs should be more than 4x faster than the flat ring; got ${(flat / best).toFixed(2)}x`);
    },
  },

  // ---------- step 3: placing ranks on the hierarchy ----------
  {
    step: 'placement',
    name: 'groupFor partitions the ranks: tp is contiguous, dp strides by tp, pp by tp*dp',
    run(m, T) {
      const L = { tp: 2, pp: 2, dp: 2 }; // 8 ranks
      T.eq(m.groupFor('tp', 0, L), [0, 1], 'tensor-parallel ranks are consecutive so they land inside one node');
      T.eq(m.groupFor('tp', 5, L), [4, 5]);
      T.eq(m.groupFor('dp', 0, L), [0, 2], 'data-parallel replicas sit tp apart');
      T.eq(m.groupFor('dp', 1, L), [1, 3]);
      T.eq(m.groupFor('pp', 0, L), [0, 4], 'pipeline stages sit tp*dp apart, the outermost axis');
      T.eq(m.groupFor('pp', 3, L), [3, 7]);
      // Every group contains its own rank, and membership is symmetric.
      for (let r = 0; r < 8; r++) {
        for (const kind of ['tp', 'dp', 'pp']) {
          const g = m.groupFor(kind, r, L);
          T.ok(g.includes(r), `the ${kind} group of rank ${r} must contain rank ${r}`);
          T.eq(g.length, L[kind], `a ${kind} group has exactly ${L[kind]} members`);
          for (const other of g) T.eq(m.groupFor(kind, other, L), g, `rank ${other} must report the same ${kind} group as rank ${r}`);
        }
      }
      T.throws(() => m.groupFor('cp', 0, L), "an unknown parallelism kind must throw, not return an empty group");
      T.throws(() => m.groupFor('tp', 8, L), 'rank 8 does not exist in a 2x2x2 layout and must throw');
    },
  },
  {
    step: 'placement',
    name: 'the ranks of a 128-GPU layout land where Megatron-LM puts them',
    run(m, T) {
      const L = { tp: 8, pp: 2, dp: 8 }; // 128 ranks
      T.eq(m.groupFor('tp', 0, L), range(8), 'tp = 8 is exactly one node of the real cluster');
      T.eq(m.groupFor('tp', 100, L), [96, 97, 98, 99, 100, 101, 102, 103], 'rank 100 sits in the tp group starting at 96');
      T.eq(m.groupFor('dp', 0, L), [0, 8, 16, 24, 32, 40, 48, 56], 'one data-parallel replica per node, 8 apart');
      T.eq(m.groupFor('pp', 0, L), [0, 64], 'the two pipeline stages are 64 ranks apart, which is a whole pod');
      const seen = new Set();
      for (let r = 0; r < 128; r++) seen.add(m.groupFor('dp', r, L).join(','));
      T.eq(seen.size, 16, '128 ranks in data-parallel groups of 8 must form exactly 16 distinct groups');
    },
  },
  {
    step: 'placement',
    name: 'groupLink is what makes tp=8 legal and tp=16 expensive',
    run(m, T) {
      const topo = m.CLUSTER;
      T.eq(m.groupLink('tp', { tp: 8, pp: 1, dp: 16 }, topo).tier, 'node',
        'eight consecutive ranks are one node, so tensor parallelism stays on NVLink');
      T.eq(m.groupLink('tp', { tp: 16, pp: 1, dp: 8 }, topo).tier, 'pod',
        'sixteen consecutive ranks straddle two nodes, so every tensor-parallel all-reduce drops to InfiniBand');
      T.eq(m.groupLink('dp', { tp: 8, pp: 1, dp: 8 }, topo).tier, 'pod',
        'data-parallel ranks 8 apart sit one per node, all inside the 64-GPU pod');
      T.eq(m.groupLink('dp', { tp: 8, pp: 1, dp: 16 }, topo).tier, 'cluster',
        '16 replicas 8 apart span 128 GPUs, which is two pods, so the gradient all-reduce crosses the spine');
      T.eq(m.groupLink('pp', { tp: 8, pp: 2, dp: 8 }, topo).tier, 'cluster', 'the two stages are 64 ranks apart: different pods');
      T.eq(m.groupLink('tp', { tp: 1, pp: 8, dp: 16 }, topo).tier, 'self', 'with tp = 1 there is no tensor-parallel group to communicate in');
      // A group that straddles a boundary must be found even when the group of rank 0 does not.
      // 3 GPUs per node: the dp group of rank 0 is [0, 2] (both in node 0) but the group of rank 1
      // is [1, 3], which crosses into node 1.
      const odd = { ...topo, gpusPerNode: 3, nodesPerPod: 4 };
      T.eq(m.slowestLink(m.groupFor('dp', 0, { tp: 2, pp: 1, dp: 2 }), odd).tier, 'node', 'the group of rank 0 stays inside node 0');
      T.eq(m.groupLink('dp', { tp: 2, pp: 1, dp: 2 }, odd).tier, 'pod',
        'the group of rank 1 is [1, 3] and crosses a node boundary: groupLink must check every group, not just the one containing rank 0');
    },
  },

  // ---------- step 4: the placement planner ----------
  {
    step: 'planner',
    name: 'a one-GPU step is pure compute, and every term multiplies out by hand',
    run(m, T) {
      // compute = 6 * 1e9 params * (8 * 128 tokens) / (1e12 * 0.5 FLOP/s) = 12.288 s
      const solo = m.layoutStepTime({ model: TINY, gpus: 1, layout: { tp: 1, pp: 1, dp: 1 }, topo: TOY });
      T.close(solo.compute, 12.288, 1e-9, 'expected 6 FLOPs per parameter per token: 6 x 1e9 x 1024 / (1e12 x 0.5)');
      T.close(solo.stepTime, 12.288, 1e-9, 'with tp = pp = dp = 1 there is no communication at all, so the step is exactly the compute');
      T.close(solo.tpComm, 0, 1e-12); T.close(solo.dpComm, 0, 1e-12); T.close(solo.ppComm, 0, 1e-12); T.close(solo.bubble, 0, 1e-12);
      // tp = 2 inside one node of TOY. m = 8 micro-batches, 8 layers, 4 all-reduces each,
      // of 2 bytes x 128 tokens x 512 dModel = 131072 B, ring over 2 GPUs at 1 MB/s = 0.131072 s.
      const tp2 = m.layoutStepTime({ model: TINY, gpus: 2, layout: { tp: 2, pp: 1, dp: 1 }, topo: TOY });
      T.close(tp2.compute, 6.144, 1e-9, 'two GPUs halve the compute');
      T.close(tp2.tpComm, 8 * 8 * 4 * 0.131072, 1e-7,
        'expected micro-batches x layers x 4 all-reduces x 0.131072 s = 33.554 s. Charging one all-reduce per layer instead of four gives 8.39 s');
      T.close(tp2.stepTime, 6.144 + 33.554432, 1e-7, 'the step is compute plus the tensor-parallel traffic it cannot hide');
      T.eq(tp2.m, 8, 'with dp = 1 and one sequence per micro-batch the pipeline runs all 8 sequences as 8 micro-batches');
    },
  },
  {
    step: 'planner',
    name: 'with all three axes on, every term lands on the right link and multiplies out by hand',
    run(m, T) {
      // TOY with half the gradient traffic hideable behind compute. 8 GPUs, tp = pp = dp = 2.
      const topo = { ...TOY, overlap: 0.5 };
      const r = m.layoutStepTime({ model: TINY, gpus: 8, layout: { tp: 2, pp: 2, dp: 2 }, topo });
      // m = 8 sequences / (dp 2 x 1) = 4 micro-batches; 4 layers per stage; actBytes = 2 x 128 x 512 = 131072 B.
      T.eq(r.m, 4, 'm = batchSeqs / (dp x microBatchSeqs) = 8 / 2 = 4 micro-batches');
      T.eq(r.links.tp.tier, 'node', 'tp group [0, 1] is one toy node');
      T.eq(r.links.pp.tier, 'cluster', 'pp group [0, 4] is tp x dp = 4 ranks apart: the other pod');
      T.eq(r.links.dp.tier, 'pod', 'dp group [0, 2] is two nodes of the same pod');
      T.close(r.compute, 1.536, 1e-9, 'expected 6 x 1e9 x 1024 / (1e12 x 0.5 x 8 GPUs) = 1.536 s');
      // ring over 2 GPUs on the node link: 2 steps x 65536 B / 1e6 = 0.131072 s; x 4 micro-batches x 4 layers x 4.
      T.close(r.tpComm, 64 * 0.131072, 1e-7, 'expected 4 micro-batches x 4 layers per stage x 4 all-reduces x 0.131072 s = 8.389 s');
      T.close(r.bubble, (1.536 + 8.388608) / 4, 1e-7,
        'expected (compute + tpComm) x (pp - 1) / m = 9.9246 x 1/4 = 2.481 s. Leaving tpComm out of the bubble (0.384 s) forgets that the idle stages also wait for its all-reduces');
      T.close(r.ppComm, 4 * 2 * 13.1072, 1e-6,
        'expected m x 2 x (pp - 1) x (131072 B / 10 KB/s) = 4 x 2 x 13.1072 = 104.86 s on the spine. Charging it once per step (module 24) gives 26.2 s; every micro-batch crosses every boundary, forward and backward');
      // dp all-reduce of 2 bytes x 2.5e8 owned params = 5e8 B over [0, 2] on the pod link: 2 x (2.5e8 / 1e5) = 5000 s.
      T.close(r.dpComm, 5000, 1e-6, 'expected a 2-way ring of 5e8 gradient bytes on the 100 KB/s pod link: 2 x 2.5e8 / 1e5 = 5000 s');
      T.close(r.dpExposed, 5000 - 0.5 * 1.536, 1e-6, 'expected dpComm - overlap x compute = 5000 - 0.768 s: half the compute hides some of the gradient traffic');
      T.close(r.stepTime, 1.536 + 8.388608 + 2.481152 + 104.8576 + 4999.232, 1e-5, 'stepTime = compute + tpComm + bubble + ppComm + dpExposed');
      T.close(r.memory.total, m.memoryPerGpu({ model: TINY, layout: { tp: 2, pp: 2, dp: 2 } }).total, 1e-3, 'memory must be the given memoryPerGpu for this layout');
      T.eq(r.fits, true, 'a quarter of a 1B model fits easily in 80 GB');
      T.eq(m.layoutStepTime({ model: TINY, gpus: 8, layout: { tp: 2, pp: 2, dp: 2 }, topo, memoryCap: 1e9 }).fits, false, 'the same layout must not fit under a 1 GB cap');
    },
  },
  {
    step: 'planner',
    name: 'the gradient all-reduce uses the hierarchical schedule, not a flat ring on the worst link',
    run(m, T) {
      // Pure data parallelism over all 8 toy GPUs: the dp group is 0..7, two per node, over two pods.
      const r = m.layoutStepTime({ model: TINY, gpus: 8, layout: { tp: 1, pp: 1, dp: 8 }, topo: TOY });
      // 2e9 gradient bytes. Hierarchical: chunk 1e9; 1 step in node (1000 s) + 6 x (2.5e8 / 1e4) across 4 nodes on the spine
      // (150000 s) + 1 step in node (1000 s) = 152000 s. The flat ring on the spine would be 14 x 2.5e8 / 1e4 = 350000 s.
      T.close(r.dpComm, 152000, 1e-3,
        'expected allReduceTime over the dp group = 152,000 s (1000 + 150000 + 1000). 350,000 s means you ran a flat ring on groupLink and threw away the hierarchy you built in step 2');
      T.close(r.dpExposed, r.dpComm, 1e-6, 'TOY has overlap 0, so all of the gradient traffic is exposed');
      T.close(r.tpComm + r.bubble + r.ppComm, 0, 1e-12, 'with tp = pp = 1 there is no tensor or pipeline traffic and no bubble');
    },
  },
  {
    step: 'planner',
    name: 'illegal layouts throw instead of returning a number',
    run(m, T) {
      T.throws(() => m.layoutStepTime({ model: TINY, gpus: 4, layout: { tp: 2, pp: 1, dp: 1 }, topo: TOY }),
        'tp x pp x dp must equal the GPU count; 2 != 4 must throw');
      T.throws(() => m.layoutStepTime({ model: TINY, gpus: 3, layout: { tp: 1, pp: 3, dp: 1 }, topo: TOY }),
        '8 layers do not divide into 3 pipeline stages');
      T.throws(() => m.layoutStepTime({ model: TINY, gpus: 3, layout: { tp: 3, pp: 1, dp: 1 }, topo: TOY }),
        'dModel = 512 is not divisible by tp = 3, so the matrices cannot be sharded');
      T.throws(() => m.layoutStepTime({ model: TINY, gpus: 16, layout: { tp: 1, pp: 1, dp: 16 }, topo: TOY }),
        'a batch of 8 sequences cannot be split over 16 data-parallel replicas');
    },
  },
  {
    step: 'planner',
    name: 'on 128 GPUs the planner keeps tensor parallelism inside a node',
    run(m, T) {
      const model = m.LLAMA3_70B;
      const ranked = m.rankLayouts({ model, gpus: 128 });
      // tp in {1,2,...,128} (powers of two divide 8192 and 28672), pp in {1,2,4,8,16} (divides 80) with
      // tp*pp <= 128: 5 + 5 + 5 + 5 + 4 + 3 + 2 + 1 = 30 layouts.
      T.eq(ranked.length, 30, `expected exactly 30 legal layouts of ${model.name} on 128 GPUs (every tp dividing 128, every pp dividing both 128/tp and 80); a planner that misses some can miss the best one`);
      for (let i = 1; i < ranked.length; i++) {
        const a = ranked[i - 1], b = ranked[i];
        T.ok(a.fits || !b.fits, 'every layout that fits in memory must be ranked ahead of every layout that does not');
        if (a.fits === b.fits) T.ok(a.stepTime <= b.stepTime + 1e-9, `rank ${i - 1} (${a.stepTime.toFixed(2)} s) must not be slower than rank ${i} (${b.stepTime.toFixed(2)} s)`);
      }
      const best = ranked[0];
      T.ok(best.fits, 'the best layout must be one that fits in the 80 GB of an H100');
      T.eq(best.layout.tp, 8, 'the fastest layout that fits uses tp = 8 — exactly the 8 GPUs joined by NVLink inside one node');
      T.eq(best.links.tp.tier, 'node', 'and its tensor-parallel group must therefore never leave the node');
      const tp16 = ranked.find((r) => r.layout.tp === 16 && r.layout.pp === 1);
      T.ok(tp16 && tp16.stepTime > best.stepTime * 1.2,
        `doubling tp to 16 pushes the group across a node boundary and should cost well over 20% of step time; got ${tp16 && (tp16.stepTime / best.stepTime).toFixed(2)}x`);
      const pureDp = ranked.find((r) => r.layout.tp === 1 && r.layout.pp === 1);
      T.ok(pureDp && !pureDp.fits, 'pure data parallelism keeps a full copy of the 70B optimizer state and 182 GB of activations per GPU; it cannot fit in 80 GB');
      T.ok(best.stepTime > best.compute, 'a 128-GPU step cannot be pure compute: something has to be communicated');
      T.ok(best.tokensPerSec > 0 && Math.abs(best.tokensPerSec * best.stepTime - model.batchSeqs * model.seqLen) < 1,
        'tokensPerSec must be the global batch divided by the step time');
    },
  },

  // ---------- step 5: mixture-of-experts all-to-all ----------
  {
    step: 'moe',
    name: 'expectedNodesPerToken counts distinct nodes, not expert copies',
    run(m, T) {
      T.close(m.expectedNodesPerToken(8, 1), 1, 1e-9, 'with one node every expert is local: the token reaches exactly one node');
      T.close(m.expectedNodesPerToken(1, 16), 1, 1e-9, 'top-1 routing sends the token to one expert, hence one node');
      // 8 choices over 8 nodes: 8 * (1 - (7/8)^8) = 8 * (1 - 0.343609) = 5.25113
      T.close(m.expectedNodesPerToken(8, 8), 5.251128, 1e-5,
        'expected 8 x (1 - (7/8)^8) = 5.251 distinct nodes. Answering 8 assumes every expert lands on a different node');
      T.close(m.expectedNodesPerToken(8, 16), 16 * (1 - Math.pow(15 / 16, 8)), 1e-9);
      T.close(m.expectedNodesPerToken(8, 16, 4), 4, 1e-9, "DeepSeek-V3's node-limited routing caps a token at 4 nodes");
      T.ok(m.expectedNodesPerToken(8, 64) <= 8, 'a token with 8 experts can never touch more than 8 nodes');
      T.ok(m.expectedNodesPerToken(16, 4) <= 4, 'a token can never touch more nodes than the cluster has');
      T.ok(m.expectedNodesPerToken(4, 8) > m.expectedNodesPerToken(2, 8), 'more experts per token means more nodes touched');
      T.throws(() => m.expectedNodesPerToken(0, 8), 'top-0 routing is not a thing and must throw');
      T.throws(() => m.expectedNodesPerToken(8, 0), 'a cluster with zero nodes must throw');
    },
  },
  {
    step: 'moe',
    name: 'all-to-all bytes are tokens x copies x dModel x bytes, twice (dispatch and combine)',
    run(m, T) {
      // TOY: 8 GPUs = 4 nodes. tokensPerGpu 1000, topK 2, dModel 100, 2 bytes.
      // perToken = 2 (dispatch+combine) x 100 x 2 = 400 B per copy.
      // intra-node: every one of the 2 copies -> 1000 x 400 x 2 = 800,000 B.
      const r = m.moeAllToAll({ tokensPerGpu: 1000, topK: 2, dModel: 100, gpus: range(8), topo: TOY });
      T.eq(r.nodes, 4, 'eight toy GPUs at two per node is four nodes of experts');
      T.close(r.intraNodeBytes, 800000, 1e-6,
        'expected 1000 tokens x 2 experts x 100 dims x 2 bytes x 2 (dispatch + combine) = 800,000 B. Forgetting the combine halves it');
      // nodesPerToken = 4 x (1 - (3/4)^2) = 1.75; remote share = 3/4.
      T.close(r.nodesPerToken, 1.75, 1e-9);
      T.close(r.interNodeBytes, 1000 * 400 * 1.75 * 0.75, 1e-6,
        'one copy per DESTINATION NODE, not per expert: 1000 x 400 B x 1.75 nodes x 3/4 remote = 525,000 B');
      T.close(r.intraNodeTime, 800000 / 1e6, 1e-9, 'the intra-node fan-out runs at the node link bandwidth');
      T.close(r.interNodeTime, 525000 / 1e4, 1e-9, 'the group spans two pods, so the remote copies go over the 10 KB/s spine');
      T.close(r.time, 0.8 + 52.5, 1e-9, 'dispatch crosses the network and then fans out inside the node: the two phases add');
      T.ok(r.interNodeTime > 50 * r.intraNodeTime, 'even though the network carries FEWER bytes here, it is the slower of the two phases');
    },
  },
  {
    step: 'moe',
    name: 'node-limited routing and capacity factor move the traffic the way they should',
    run(m, T) {
      const base = { tokensPerGpu: 4096, topK: 8, dModel: 7168, gpus: range(64), topo: m.CLUSTER };
      const free = m.moeAllToAll(base);
      const limited = m.moeAllToAll({ ...base, maxNodes: 4 });
      T.eq(free.nodes, 8, '64 GPUs at 8 per node is 8 nodes of experts');
      T.ok(limited.interNodeBytes < free.interNodeBytes,
        `capping a token at 4 nodes must cut inter-node traffic; got ${limited.interNodeBytes} vs ${free.interNodeBytes}`);
      T.close(limited.interNodeBytes / free.interNodeBytes, 4 / free.nodesPerToken, 1e-6,
        'the cut is exactly the ratio of node counts, 4 / 5.251');
      T.close(limited.intraNodeBytes, free.intraNodeBytes, 1e-9,
        'node-limited routing does not change how many experts a token visits, only how they are spread, so the NVLink traffic is unchanged');
      const heavy = m.moeAllToAll({ ...base, capacityFactor: 1.25 });
      T.close(heavy.interNodeBytes, free.interNodeBytes * 1.25, 1e-6, 'a capacity factor of 1.25 lets experts accept 25% more tokens, so 25% more bytes move');
      const oneNode = m.moeAllToAll({ ...base, gpus: range(8) });
      T.eq(oneNode.nodes, 1);
      T.close(oneNode.interNodeTime, 0, 1e-12, 'if every expert is in the node, nothing crosses the network');
      T.ok(oneNode.intraNodeTime > 0, 'a single-node group still pays the NVLink fan-out: intraNodeTime must be positive');
      T.close(oneNode.time, oneNode.intraNodeTime, 1e-12, 'on one node time = interNodeTime + intraNodeTime = 0 + intraNodeTime');
      T.ok(oneNode.time < free.time / 3, 'expert parallelism inside one node is several times cheaper than across eight');
      T.throws(() => m.moeAllToAll({ ...base, gpus: [] }), 'an empty expert-parallel group must throw');
    },
  },

  // ---------- step 6: failures and checkpoints ----------
  {
    step: 'reliability',
    name: 'failure rates add, so interruptions scale linearly with the cluster',
    run(m, T) {
      T.close(m.failuresPerDay(1, 24), 1, 1e-9, 'one GPU with a 24-hour MTBF fails once a day by definition');
      T.close(m.failuresPerDay(1000, 50000), 0.48, 1e-9, 'expected 1000 x 24 / 50000 = 0.48 per day');
      T.close(m.failuresPerDay(2000, 50000), 2 * m.failuresPerDay(1000, 50000), 1e-9, 'twice the GPUs, twice the interruptions: the rates add');
      T.close(m.failuresPerDay(0, 50000), 0, 1e-12);
      // Llama 3 405B: the Llama 3 paper reports 466 job interruptions in a 54-day snapshot on 16,384 H100s,
      // of which 47 were planned and 419 unexpected. 16384 x 24 x 54 / 419 = 50,677 hours per GPU.
      T.close(m.failuresPerDay(16384, 50677), 419 / 54, 2e-3,
        'a per-GPU MTBF of about 50,677 hours (5.8 years) reproduces the Llama 3 paper: 419 unexpected interruptions in 54 days, 7.8 a day, on 16,384 GPUs');
      T.throws(() => m.failuresPerDay(100, 0), 'an MTBF of zero is not a number of hours and must throw');
    },
  },
  {
    step: 'reliability',
    name: "Young's interval is the square root of 2 x MTBF x checkpoint time",
    run(m, T) {
      T.close(m.youngInterval(5000, 10), Math.sqrt(2 * 5000 * 10), 1e-9, 'expected sqrt(2 x 5000 x 10) = 316.2 s');
      T.close(m.youngInterval(20000, 10), 2 * m.youngInterval(5000, 10), 1e-9, 'a 4x longer MTBF doubles the interval, it does not quadruple it');
      T.close(m.youngInterval(5000, 40), 2 * m.youngInterval(5000, 10), 1e-9, 'a 4x more expensive checkpoint also only doubles the interval');
      T.close(m.youngInterval(5000, 0), 0, 1e-12, 'a free checkpoint should be taken continuously');
      T.throws(() => m.youngInterval(0, 10), 'a job that never runs has no optimal interval; MTBF must be positive');
      T.throws(() => m.youngInterval(5000, -1), 'a negative checkpoint time must throw');
    },
  },
  {
    step: 'reliability',
    name: 'wasted time is write cost plus lost work, and Young\'s interval minimises it',
    run(m, T) {
      const mtbfSeconds = 10000, checkpointSeconds = 120;
      // 1000 s interval: 120/1000 writing + 1000/20000 redoing = 0.12 + 0.05 = 0.17
      T.close(m.wastedFraction({ mtbfSeconds, checkpointSeconds, intervalSeconds: 1000 }), 0.17, 1e-9,
        'expected 120/1000 spent writing plus 1000/(2 x 10000) of work redone after a failure');
      const opt = m.youngInterval(mtbfSeconds, checkpointSeconds);
      const atOpt = m.wastedFraction({ mtbfSeconds, checkpointSeconds, intervalSeconds: opt });
      T.close(atOpt, Math.sqrt((2 * checkpointSeconds) / mtbfSeconds), 1e-9,
        "at Young's interval the waste is exactly sqrt(2 x checkpointTime / MTBF) = 15.5%");
      for (const factor of [0.25, 0.5, 0.8, 1.25, 2, 4]) {
        const other = m.wastedFraction({ mtbfSeconds, checkpointSeconds, intervalSeconds: opt * factor });
        T.ok(other >= atOpt - 1e-12, `interval ${(opt * factor).toFixed(0)} s wastes ${(other * 100).toFixed(2)}%, less than the supposed optimum ${(atOpt * 100).toFixed(2)}% — the optimum is wrong`);
      }
      T.ok(m.wastedFraction({ mtbfSeconds: 1000, checkpointSeconds, intervalSeconds: opt }) >
           m.wastedFraction({ mtbfSeconds: 100000, checkpointSeconds, intervalSeconds: opt }),
        'a flakier cluster wastes more time at the same checkpoint interval');
      T.throws(() => m.wastedFraction({ mtbfSeconds, checkpointSeconds, intervalSeconds: 0 }), 'an interval of zero means checkpointing forever and must throw');
    },
  },
];
