// app/sandbox-worker.js — runs learner code, tests, and goal demos off the main thread.
import { rewriteImports } from './rewrite.js';
import { makeT } from './testkit.js';

let currentRun = null;

function post(msg) { self.postMessage({ runId: currentRun, ...msg }); }

// Capture console output from learner code.
const origLog = console.log;
for (const level of ['log', 'info', 'warn', 'error']) {
  console[level] = (...args) => { post({ type: 'log', level, text: args.map(fmtArg).join(' ') }); };
}
function fmtArg(a) {
  if (typeof a === 'string') return a;
  if (a && a.data instanceof Float32Array && a.shape) return `tensor(shape=[${a.shape}], data=[${Array.from(a.data.slice(0, 8)).map((x) => +x.toFixed(4)).join(', ')}${a.data.length > 8 ? ', …' : ''}])`;
  if (ArrayBuffer.isView(a)) return `[${Array.from(a.slice(0, 16)).map((x) => +x.toFixed(4)).join(', ')}${a.length > 16 ? ', …' : ''}]`;
  try { return JSON.stringify(a, (k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v)); } catch { return String(a); }
}

async function importSource(code, base) {
  const blob = new Blob([rewriteImports(code, base)], { type: 'text/javascript' });
  const url = URL.createObjectURL(blob);
  try { return await import(url); } finally { URL.revokeObjectURL(url); }
}

async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.text();
}

function makeLab() {
  let doneCalled = false;
  return {
    log: (...args) => post({ type: 'log', level: 'log', text: args.map(fmtArg).join(' ') }),
    md: (markdown) => post({ type: 'md', markdown }),
    plot: (spec) => post({ type: 'plot', spec: plain(spec) }),
    bar: (spec) => post({ type: 'bar', spec: plain(spec) }),
    heatmap: (spec) => post({ type: 'heatmap', spec: plain(spec) }),
    table: (spec) => post({ type: 'table', spec: plain(spec) }),
    progress: (fraction, label = '') => post({ type: 'progress', fraction, label }),
    tick: () => new Promise((r) => setTimeout(r, 0)),
    check: (cond, message) => { if (!cond) throw new Error('check failed: ' + message); },
    done: (summary) => { doneCalled = true; post({ type: 'demo-done', summary: String(summary ?? '') }); },
    get doneCalled() { return doneCalled; },
  };
}

// Deep-convert typed arrays so specs survive structured clone in a predictable shape.
function plain(x) {
  if (ArrayBuffer.isView(x)) return Array.from(x);
  if (Array.isArray(x)) return x.map(plain);
  if (x && typeof x === 'object') { const o = {}; for (const k in x) o[k] = plain(x[k]); return o; }
  return x;
}

self.onmessage = async (e) => {
  const { type, runId, mode, code, base, testsUrl, demoUrl, stepId } = e.data;
  if (type !== 'run') return;
  currentRun = runId;
  try {
    const learner = await importSource(code, base);
    if (mode === 'tests') {
      const testsSrc = await fetchText(testsUrl);
      const { tests } = await importSource(testsSrc, base);
      const selected = stepId ? tests.filter((t) => t.step === stepId) : tests;
      let passed = 0, failed = 0;
      for (const t of selected) {
        const t0 = performance.now();
        try {
          await t.run(learner, makeT());
          passed++;
          post({ type: 'test', step: t.step, name: t.name, pass: true, message: '', ms: performance.now() - t0 });
        } catch (err) {
          failed++;
          post({ type: 'test', step: t.step, name: t.name, pass: false, message: err && err.message ? err.message : String(err), stack: err && err.stack, ms: performance.now() - t0 });
        }
      }
      post({ type: 'tests-done', passed, failed, total: selected.length });
    } else if (mode === 'demo') {
      const demoSrc = await fetchText(demoUrl);
      const { default: demo } = await importSource(demoSrc, base);
      const lab = makeLab();
      await demo(learner, lab);
      if (!lab.doneCalled) post({ type: 'error', message: 'The demo finished without calling lab.done(...)' });
      post({ type: 'run-finished' });
    }
  } catch (err) {
    post({ type: 'error', message: err && err.message ? err.message : String(err), stack: err && err.stack });
    post({ type: 'run-finished' });
  }
};
