#!/usr/bin/env node
// tools/verify.mjs — headless verifier for modules.
//   node tools/verify.mjs                 verify every module marked ready
//   node tools/verify.mjs --module 05-attention
//   node tools/verify.mjs --skip-demo     skip running demos (faster)
//   node tools/verify.mjs --demo-only     run only demos
// Checks: schema + pedagogy checklist (mechanical parts), tests pass on solution, tests fail on
// starter (at least one per step), starter imports cleanly, demo runs on the solution and calls lab.done.

import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { rewriteImports } from '../app/rewrite.js';
import { makeT } from '../app/testkit.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const BASE = pathToFileURL(ROOT).href.replace(/\/$/, '');
const args = process.argv.slice(2);
const only = args.includes('--module') ? args[args.indexOf('--module') + 1] : null;
const skipDemo = args.includes('--skip-demo');
const demoOnly = args.includes('--demo-only');
const quiet = args.includes('--quiet');
const skipIncomplete = args.includes('--skip-incomplete');   // CI while modules are still being written

const { MODULES, TRACKS } = await import(pathToFileURL(path.join(ROOT, 'modules/index.js')).href);

let importCounter = 0;
async function importSource(code) {
  const src = rewriteImports(code, BASE) + `\n//# sourceURL=verify-${importCounter++}.js`;
  return import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
}

function makeLab() {
  const events = [];
  let done = null;
  return {
    events,
    log: (...a) => events.push(['log', a.join(' ')]),
    md: (m) => events.push(['md', m]),
    plot: (s) => events.push(['plot', s]),
    bar: (s) => events.push(['bar', s]),
    heatmap: (s) => events.push(['heatmap', s]),
    table: (s) => events.push(['table', s]),
    progress: () => {},
    tick: () => new Promise((r) => setImmediate(r)),
    check: (c, m) => { if (!c) throw new Error('check failed: ' + m); },
    done: (s) => { done = String(s ?? ''); events.push(['done', done]); },
    get doneSummary() { return done; },
  };
}

function checkSchema(def, meta) {
  const errs = [], warns = [];
  const req = ['id', 'title', 'track', 'threshold', 'goal', 'concept', 'steps', 'reflection', 'stretch'];
  for (const k of req) if (def[k] === undefined || def[k] === null || def[k] === '') errs.push(`missing field "${k}"`);
  if (def.id !== meta.id) errs.push(`module.js id "${def.id}" != directory "${meta.id}"`);
  if (!TRACKS.some((t) => t.id === def.track)) errs.push(`unknown track "${def.track}"`);
  if (!Array.isArray(def.steps) || def.steps.length < 3 || def.steps.length > 7) errs.push(`need 3–7 steps, got ${def.steps?.length}`);
  const ids = new Set();
  for (const s of def.steps || []) {
    if (!s.id || !s.title || !s.instructions) errs.push(`step "${s.id}" missing id/title/instructions`);
    if (ids.has(s.id)) errs.push(`duplicate step id "${s.id}"`);
    ids.add(s.id);
    if (!Array.isArray(s.hints) || s.hints.length !== 3) errs.push(`step "${s.id}" must have exactly 3 hints (got ${s.hints?.length})`);
  }
  const num = parseInt(meta.id.slice(0, 2), 10);
  const recall = def.recall || [];
  if (recall.length < 3 || recall.length > 6) errs.push(`need 3–6 recall questions, got ${recall.length}`);
  for (const q of recall) {
    if (!q.q || !Array.isArray(q.options) || q.options.length < 2 || typeof q.answer !== 'number' || q.answer >= q.options.length) errs.push(`bad recall question: ${JSON.stringify(q).slice(0, 80)}`);
    if (!q.why) warns.push(`recall question without "why": ${String(q.q).slice(0, 60)}`);
  }
  const review = def.review || [];
  if (review.length < 3) errs.push(`need >= 3 review questions about this module, got ${review.length}`);
  for (const q of review) if (!q.q || !Array.isArray(q.options) || typeof q.answer !== 'number' || q.answer >= q.options.length) errs.push(`bad review question: ${JSON.stringify(q).slice(0, 80)}`);
  const predicts = (String(def.concept).match(/^:::predict/gm) || []).length + (def.steps || []).filter((s) => s.predict).length;
  if (predicts < 2) errs.push(`need >= 2 predict prompts (concept :::predict blocks or step.predict), got ${predicts}`);
  if (!Array.isArray(def.reflection) || def.reflection.length < 2) errs.push('need >= 2 reflection prompts');
  if (!Array.isArray(def.stretch) || def.stretch.length < 2) errs.push('need >= 2 stretch goals');
  const words = String(def.concept).split(/\s+/).length;
  if (words < 300) warns.push(`concept is short (${words} words; aim for 400–900)`);
  if (words > 1400) warns.push(`concept is long (${words} words; aim for 400–900)`);
  if (/\$[^$]+\$/.test(def.concept)) warns.push('concept seems to contain LaTeX ($…$); use code spans instead');
  if (num > 1 && !(def.prereqs || []).length) warns.push('no prereqs listed');
  return { errs, warns };
}

async function verifyModule(meta) {
  const dir = path.join(ROOT, 'modules', meta.id);
  const out = { id: meta.id, errors: [], warnings: [], testsOnSolution: null, testsOnStarter: null, demo: null, ms: 0 };
  const t0 = performance.now();
  for (const f of ['module.js', 'starter.js', 'solution.js', 'tests.js', 'demo.js']) {
    if (!existsSync(path.join(dir, f))) out.errors.push(`missing ${f}`);
  }
  if (out.errors.length) return out;
  let def;
  try { def = (await import(pathToFileURL(path.join(dir, 'module.js')).href)).default; }
  catch (e) { out.errors.push(`module.js failed to import: ${e.message}`); return out; }
  const { errs, warns } = checkSchema(def, meta);
  out.errors.push(...errs);
  out.warnings.push(...warns);
  const [starterSrc, solutionSrc, testsSrc, demoSrc] = await Promise.all(['starter.js', 'solution.js', 'tests.js', 'demo.js'].map((f) => readFile(path.join(dir, f), 'utf8')));
  for (const [name, src] of [['starter.js', starterSrc], ['solution.js', solutionSrc], ['tests.js', testsSrc], ['demo.js', demoSrc]]) {
    if (/from\s*['"]\.\.?\//.test(src)) out.errors.push(`${name} uses a relative import; use 'lib/...' instead`);
  }
  let tests;
  try { ({ tests } = await importSource(testsSrc)); } catch (e) { out.errors.push(`tests.js failed to import: ${e.message}`); return out; }
  if (!Array.isArray(tests) || !tests.length) { out.errors.push('tests.js exports no tests'); return out; }
  const stepIds = new Set(def.steps.map((s) => s.id));
  for (const t of tests) {
    if (!stepIds.has(t.step)) out.errors.push(`test "${t.name}" references unknown step "${t.step}"`);
    if (typeof t.run !== 'function') out.errors.push(`test "${t.name}" has no run()`);
  }
  for (const s of def.steps) {
    const n = tests.filter((t) => t.step === s.id).length;
    if (n < 2) out.errors.push(`step "${s.id}" has ${n} test(s); need >= 2`);
  }
  if (demoOnly) { /* fallthrough to demo */ } else {
    // starter must import cleanly
    let starterMod = null;
    try { starterMod = await importSource(starterSrc); } catch (e) { out.errors.push(`starter.js failed to import: ${e.message}`); }
    // solution must pass every test
    let solMod = null;
    try { solMod = await importSource(solutionSrc); } catch (e) { out.errors.push(`solution.js failed to import: ${e.message}`); }
    if (solMod) {
      const res = await runTests(tests, solMod);
      out.testsOnSolution = res;
      for (const r of res.results) if (!r.pass) out.errors.push(`solution fails test [${r.step}] "${r.name}": ${r.message}`);
      if (res.ms > 15000) out.warnings.push(`tests took ${Math.round(res.ms)} ms on solution (budget 10 s)`);
    }
    if (starterMod) {
      const res = await runTests(tests, starterMod);
      out.testsOnStarter = res;
      for (const s of def.steps) {
        const rs = res.results.filter((r) => r.step === s.id);
        if (rs.length && rs.every((r) => r.pass)) out.errors.push(`starter already passes every test of step "${s.id}" (nothing to build)`);
      }
    }
  }
  if (!skipDemo) {
    let solMod = null;
    try { solMod = await importSource(solutionSrc); } catch (e) { /* reported above */ }
    if (solMod) {
      const lab = makeLab();
      const td = performance.now();
      try {
        const { default: demo } = await importSource(demoSrc);
        await Promise.race([
          demo(solMod, lab),
          new Promise((_, rej) => setTimeout(() => rej(new Error('demo exceeded 120 s')), 120000)),
        ]);
        const ms = performance.now() - td;
        const visuals = lab.events.filter((e) => ['plot', 'bar', 'heatmap', 'table'].includes(e[0])).length;
        out.demo = { ms, visuals, done: lab.doneSummary };
        if (lab.doneSummary === null) out.errors.push('demo did not call lab.done(...)');
        else if (!/\d/.test(lab.doneSummary)) out.warnings.push('lab.done summary contains no numbers');
        if (!visuals) out.errors.push('demo produced no visual (plot/bar/heatmap/table)');
        if (ms > 60000) out.warnings.push(`demo took ${Math.round(ms / 1000)} s on the solution (budget 60 s)`);
      } catch (e) {
        out.errors.push(`demo threw: ${e.message}`);
        out.demo = { ms: performance.now() - td, error: e.message };
      }
    }
  }
  out.ms = performance.now() - t0;
  return out;
}

async function runTests(tests, m) {
  const results = [];
  const t0 = performance.now();
  for (const t of tests) {
    const ts = performance.now();
    try {
      await Promise.race([
        Promise.resolve().then(() => t.run(m, makeT())),
        new Promise((_, rej) => setTimeout(() => rej(new Error('test exceeded 10 s')), 10000)),
      ]);
      results.push({ step: t.step, name: t.name, pass: true, ms: performance.now() - ts });
    } catch (e) {
      results.push({ step: t.step, name: t.name, pass: false, message: e && e.message ? e.message : String(e), ms: performance.now() - ts });
    }
  }
  return { results, passed: results.filter((r) => r.pass).length, failed: results.filter((r) => !r.pass).length, ms: performance.now() - t0 };
}

const targets = MODULES.filter((m) => m.status === 'ready' && (!only || m.id === only));
if (!targets.length) { console.error(only ? `No ready module with id ${only}` : 'No ready modules'); process.exit(2); }
let failed = 0, skipped = 0;
const REQUIRED = ['module.js', 'starter.js', 'solution.js', 'tests.js', 'demo.js'];
for (const meta of targets) {
  if (skipIncomplete) {
    const missing = REQUIRED.filter((f) => !existsSync(path.join(ROOT, 'modules', meta.id, f)));
    if (missing.length) { skipped++; console.log(`SKIP  ${meta.id}  (in progress: missing ${missing.join(', ')})`); continue; }
  }
  const r = await verifyModule(meta);
  const ok = r.errors.length === 0;
  if (!ok) failed++;
  const bits = [];
  if (r.testsOnSolution) bits.push(`solution ${r.testsOnSolution.passed}/${r.testsOnSolution.results.length}`);
  if (r.testsOnStarter) bits.push(`starter ${r.testsOnStarter.passed}/${r.testsOnStarter.results.length}`);
  if (r.demo) bits.push(r.demo.error ? 'demo ✗' : `demo ${Math.round(r.demo.ms / 1000)}s, ${r.demo.visuals} visual(s)`);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${meta.id}  (${bits.join(', ')}; ${Math.round(r.ms)} ms)`);
  for (const e of r.errors) console.log(`      ✗ ${e}`);
  if (!quiet) for (const w of r.warnings) console.log(`      ! ${w}`);
  if (r.demo && r.demo.done && !quiet) console.log(`      done: ${r.demo.done.replace(/\s+/g, ' ').slice(0, 160)}`);
}
console.log(`\n${targets.length - failed - skipped}/${targets.length - skipped} modules verified${skipped ? ` (${skipped} in progress, skipped)` : ''}`);
process.exit(failed ? 1 : 0);
