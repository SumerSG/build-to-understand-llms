// app/sandbox-worker.js — runs learner code, tests, and goal demos off the main thread.
import { rewriteImports } from './rewrite.js';
import { makeT, watchLearner, explainError, explainMessage, failureMessage } from './testkit.js';
import { format as chartFormat } from './charts.js';

let currentRun = null;

// ---- console capture: batched, capped, so a console.log inside a hot loop cannot stall the page ----
const LOG_MAX_LINES = 2000, LOG_MAX_CHARS = 4000, LOG_BATCH = 200;
let logBuf = [], logCount = 0, logTimer = null, logTruncated = false;

function flushLogs() {
  if (logTimer) { clearTimeout(logTimer); logTimer = null; }
  if (logBuf.length) { const lines = logBuf; logBuf = []; self.postMessage({ runId: currentRun, type: 'log', lines }); }
}
function resetLogs() { logBuf = []; logCount = 0; logTruncated = false; if (logTimer) { clearTimeout(logTimer); logTimer = null; } }
function emitLog(level, args) {
  if (logCount >= LOG_MAX_LINES) {
    if (!logTruncated) { logTruncated = true; logBuf.push({ level: 'warn', text: `console output truncated after ${LOG_MAX_LINES} lines (remove the console.log from the loop to see the rest)` }); flushLogs(); }
    return;
  }
  logCount++;
  let text = args.map(fmtArg).join(' ');
  if (text.length > LOG_MAX_CHARS) text = `${text.slice(0, LOG_MAX_CHARS)}… (${text.length} chars)`;
  logBuf.push({ level, text });
  if (logBuf.length >= LOG_BATCH) flushLogs();
  else if (!logTimer) logTimer = setTimeout(flushLogs, 0);
}
for (const level of ['log', 'info', 'warn', 'error']) console[level] = (...args) => emitLog(level, args);

// Every non-log message flushes pending console lines first so output stays in order.
function post(msg) { flushLogs(); self.postMessage({ runId: currentRun, ...msg }); }

function fmtArg(a) {
  if (typeof a === 'string') return a;
  if (a === undefined || a === null || typeof a === 'function' || typeof a === 'symbol' || typeof a === 'bigint' || typeof a === 'number' || typeof a === 'boolean') return String(a);
  if (a instanceof Error) return a.stack || `${a.name}: ${a.message}`;
  if (a.data instanceof Float32Array && a.shape) return `tensor(shape=[${a.shape}], data=[${Array.from(a.data.slice(0, 8)).map((x) => +x.toFixed(4)).join(', ')}${a.data.length > 8 ? ', …' : ''}])`;
  if (ArrayBuffer.isView(a)) return `[${Array.from(a.slice(0, 16)).map((x) => +x.toFixed(4)).join(', ')}${a.length > 16 ? ', …' : ''}]`;
  const seen = new WeakSet();
  const replacer = (k, v) => {
    if (ArrayBuffer.isView(v)) return Array.from(v);
    if (v instanceof Map) return { Map: [...v] };
    if (v instanceof Set) return { Set: [...v] };
    if (v instanceof Error) return `${v.name}: ${v.message}`;
    if (typeof v === 'bigint' || typeof v === 'symbol' || typeof v === 'function') return String(v);
    if (typeof v === 'number' && !Number.isFinite(v)) return String(v);
    if (v && typeof v === 'object') { if (seen.has(v)) return '[Circular]'; seen.add(v); }
    return v;
  };
  try {
    if (a instanceof Map) return `Map(${a.size}) ${JSON.stringify([...a], replacer)}`;
    if (a instanceof Set) return `Set(${a.size}) ${JSON.stringify([...a], replacer)}`;
    return JSON.stringify(a, replacer) ?? String(a);
  } catch { return String(a); }
}

async function importSource(code, base, onUrl = null) {
  const blob = new Blob([rewriteImports(code, base)], { type: 'text/javascript' });
  const url = URL.createObjectURL(blob);
  if (onUrl) onUrl(url);
  try { return await import(url); } finally { URL.revokeObjectURL(url); }
}

// A rejected import() of a module carries no position for a SyntaxError. To recover the line, strip the
// module syntax (keeping every newline so line numbers survive) and parse the result as a classic script
// in a nested worker: its ErrorEvent carries lineno/colno. Best effort; null if nothing arrives in ~1 s.
const blankOut = (m) => m.replace(/[^\n]/g, '');
function stripModuleSyntax(code) {
  return code
    .replace(/^[ \t]*import\s*(?:[\w$*{},\s]*?from\s*)?['"][^'"\n]*['"][ \t]*;?/mg, blankOut)
    .replace(/^[ \t]*export\s*\{[^}]*\}[^\n]*$/mg, blankOut)
    .replace(/^[ \t]*export\s+default\s+/mg, 'void ')
    .replace(/^[ \t]*export\s+/mg, '');
}
function locateSyntaxError(code, message) {
  return new Promise((resolve) => {
    if (typeof Worker === 'undefined') return resolve(null);
    let url = null, w = null, timer = null;
    const finish = (v) => { clearTimeout(timer); if (w) w.terminate(); if (url) URL.revokeObjectURL(url); resolve(v); };
    try {
      url = URL.createObjectURL(new Blob([stripModuleSyntax(code)], { type: 'text/javascript' }));
      w = new Worker(url);
      w.onerror = (e) => {
        e.preventDefault();
        const sameError = typeof e.message === 'string' && message && e.message.includes(message);
        finish(sameError && e.lineno > 0 ? { line: e.lineno, col: e.colno || 0 } : null);
      };
      timer = setTimeout(() => finish(null), 1000);
    } catch { finish(null); }
  });
}

async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.text();
}

// The untouched starter, so a failure can say "this function is not written yet". Best effort: '' if missing.
async function fetchStarter(testsUrl) {
  try { return testsUrl ? await fetchText(testsUrl.replace(/tests\.js$/, 'starter.js')) : ''; } catch { return ''; }
}

function makeLab() {
  let doneCalled = false;
  return {
    log: (...args) => emitLog('log', args),
    md: (markdown) => post({ type: 'md', markdown }),
    plot: (spec) => post({ type: 'plot', spec: plain(spec) }),
    bar: (spec) => post({ type: 'bar', spec: plain(spec) }),
    heatmap: (spec) => post({ type: 'heatmap', spec: plain(heatmapTexts(spec)) }),
    table: (spec) => post({ type: 'table', spec: plain(spec) }),
    progress: (fraction, label = '') => post({ type: 'progress', fraction, label }),
    tick: () => new Promise((r) => setTimeout(r, 0)),
    check: (cond, message) => { if (!cond) throw new Error('check failed: ' + message); },
    done: (summary) => { doneCalled = true; post({ type: 'demo-done', summary: String(summary ?? '') }); },
    get doneCalled() { return doneCalled; },
  };
}

// Deep-convert typed arrays so specs survive structured clone in a predictable shape. Functions cannot be
// cloned (postMessage would throw), so they are left out.
function plain(x) {
  if (ArrayBuffer.isView(x)) return Array.from(x);
  if (Array.isArray(x)) return x.map(plain);
  if (typeof x === 'function') return undefined;
  if (x && typeof x === 'object') { const o = {}; for (const k in x) if (typeof x[k] !== 'function') o[k] = plain(x[k]); return o; }
  return x;
}

// A heatmap `format` given as a function runs here, in the demo's worker: its output travels as text
// (legendText for the two ends of the colour scale, cellText for the tooltip and table). A named format
// ('log10-bytes', …) or a { prefix, suffix, scale } object is cloneable and is applied by app/charts.js.
function heatmapTexts(spec) {
  if (!spec || typeof spec.format !== 'function') return spec;
  const fmt = chartFormat.valueFormat(spec.format);
  const rows = (spec.rows || []).map((r) => Array.from(r));
  const [lo, hi] = chartFormat.heatRange({ ...spec, rows });
  return { ...spec, format: undefined, legendText: [fmt(lo), fmt(hi)], cellText: rows.map((r) => r.map(fmt)) };
}

const errMsg = (err) => (err && err.message ? err.message : String(err));

// Errors keep their original text; explainError adds plain-words notes after it (see app/testkit.js).
async function loadLearner(code, base) {
  let file = '';
  try { return { mod: await importSource(code, base, (u) => { file = u; }), file }; }
  catch (err) {
    if (err instanceof SyntaxError) {
      const loc = await locateSyntaxError(code, err.message);
      const where = loc ? ` on line ${loc.line}${loc.col ? `, column ${loc.col}` : ''}` : '';
      const notes = explainError(err, { source: code }, { line: loc ? loc.line : 0 });
      post({ type: 'error', syntax: true, line: loc ? loc.line : null, col: loc ? loc.col : null, message: explainMessage(`Your file could not be loaded (syntax error${where}): ${err.message}`, notes) });
    } else {
      const notes = explainError(err, { source: code, file });
      post({ type: 'error', message: explainMessage(`Your file threw while loading: ${errMsg(err)}`, notes), stack: err && err.stack });
    }
    return null;
  }
}

self.onmessage = async (e) => {
  const { type, runId, mode, code, base, testsUrl, demoUrl, stepId } = e.data;
  if (type !== 'run') return;
  currentRun = runId;
  resetLogs();
  let watch = null;
  try {
    const [loaded, starter] = await Promise.all([loadLearner(code, base), fetchStarter(testsUrl)]);
    if (!loaded) { post({ type: 'run-finished' }); return; }
    // The tests and the demo see the learner's module through a recorder, so a failure can name the
    // function that produced a wrong value (it changes no value and no behaviour).
    watch = watchLearner(loaded.mod, { source: code, starter, file: loaded.file });
    const learner = watch.module;
    if (mode === 'tests') {
      const testsSrc = await fetchText(testsUrl);
      const { tests } = await importSource(testsSrc, base);
      const selected = stepId ? tests.filter((t) => t.step === stepId) : tests;
      let passed = 0, failed = 0;
      for (const t of selected) {
        const t0 = performance.now();
        watch.reset();
        try {
          await t.run(learner, makeT(watch));
          passed++;
          post({ type: 'test', step: t.step, name: t.name, pass: true, message: '', ms: performance.now() - t0 });
        } catch (err) {
          failed++;
          post({ type: 'test', step: t.step, name: t.name, pass: false, message: failureMessage(err, watch), stack: err && err.stack, ms: performance.now() - t0 });
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
    post({ type: 'error', message: watch ? failureMessage(err, watch) : errMsg(err), stack: err && err.stack });
    post({ type: 'run-finished' });
  }
};
