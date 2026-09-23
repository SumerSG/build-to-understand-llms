# Roadmap

What is built, what is in progress, and what is planned. The registry (`modules/index.js`) is the source
of truth: a module with `status: 'ready'` is part of the path and is verified by CI; `status: 'planned'`
modules appear in the lab as "planned" and are skipped by the verifier.

## In progress
The 34 modules in the registry are being authored and adversarially reviewed. A whole-curriculum review
(sequencing, cross-module facts, progression, coverage gaps) will produce further fixes and possibly new
modules, which take ids 34–39.

## Planned: data-center deep dive (groundwork laid, not scheduled)
Modules 40–42 in docs/CURRICULUM_BRIEFS.md: collectives from scratch, the data-center fabric, and
operating an AI cluster. They go deeper than modules 24–25 by making the learner program the collectives
and watch contention on a simulated fabric. Before any of them is written, two pieces of shared
infrastructure are needed.

### 1. `lib/netsim.js` — discrete-event network core (to build first)
```js
export class EventQueue { schedule(t, fn); run(untilT = Infinity); get now() }
export class Link { constructor({ from, to, bandwidth /* bytes/s */, latency /* s */ }); transmit(bytes, onDone) }  // serializes sends
export class World { constructor({ ranks, topology }); send(from, to, bytes, payload); recv(rank, from) → Promise; run() }
export function topology(kind /* 'ring'|'two-level'|'fat-tree'|'rail'|'torus' */, opts) → { nodes, links }
export function maxMinFair(flows /* [{ path: linkIds }] */, capacities) → rates   // progressive filling
```
Deterministic (seeded), no DOM, runs in the sandbox worker and in Node, tested like the rest of `lib/`.

### 2. `lab.graph` — a topology view for demos (app work)
```js
lab.graph({
  title,
  nodes: [{ id, label?, group?, x?, y? }],          // x/y optional; otherwise laid out by group (node, rail, pod)
  links: [{ from, to, value? /* 0..1 utilisation */ }],
  frames?: [{ t, links: [{ from, to, value }], note? }],   // optional animation over simulated time
})
```
Rendered as SVG in the atelier style: ink nodes, links whose stroke weight and darkness encode
utilisation, a time scrubber when `frames` is given, hover for per-link numbers, and a table view like
every other chart. The sandbox worker forwards it as `{ type: 'graph', spec }`; `app/charts.js` gains
`renderGraph`; `tools/verify.mjs` counts it as a visual. Reusable by module 16 (block tables), 24
(pipeline stages) and 26 (routing) once it exists.

### Order of work when this is picked up
1. `lib/netsim.js` with tests.  2. `lab.graph` renderer and worker plumbing.  3. Module 40, then 41, then 42,
through the same author → adversarial review → fix pipeline as the rest.
