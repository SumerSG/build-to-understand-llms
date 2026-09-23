// app/main.js — the lab UI: router, sidebar, home, module page (Recall → Concept → Build → Goal → Reflect), review queue.
import { TRACKS, MODULES, moduleById, nextModule, loadModule } from '../modules/index.js';
import { store, scheduleReview, enrollReview, dueReviews, reviewSummary, LEITNER_DAYS } from './storage.js';
import { render as md, inline, activatePredicts, escapeHtml as esc } from './markdown.js';
import { renderChart } from './charts.js';
import { createEditor } from './editor.js';
import { Runner } from './runner.js';

const BASE = new URL('..', import.meta.url).href.replace(/\/$/, '');
const runner = new Runner({ base: BASE });
const $app = document.getElementById('app');
const $sidebar = document.getElementById('sidebar');
const PHASES = [
  { id: 'recall', label: 'Recall' },
  { id: 'concept', label: 'Concept' },
  { id: 'build', label: 'Build' },
  { id: 'goal', label: 'Goal' },
  { id: 'reflect', label: 'Reflect' },
];
const DEFAULT_TIMEOUTS = { tests: 20000, demo: 120000 };
const LOG_DOM_MAX = 2000;          // console lines kept in the DOM per run
const cache = new Map();           // module id -> { def, starter }
const READY = MODULES.filter((m) => m.status === 'ready');   // planned modules are shown but not counted
const failedLoads = new Set();     // module ids whose module.js could not be imported (shown as planned)

let renderSeq = 0;                 // bumped on every route(); async renders bail out when superseded
let currentPage = null;            // { id, def, starter, phase, phasesEl } for the module page on screen
let flushPendingSave = null;       // set by renderBuild: writes a debounced edit immediately
let lastPhaseUI = null;            // { id, phase } of the last rendered phase strip, for the sliding indicator

// ---------- helpers ----------

function h(html) {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function isComplete(id) { return !!store.module(id).completedAt; }

function moduleStatus(id) {
  if (failedLoads.has(id)) return 'planned';
  const s = store.module(id);
  if (s.completedAt) return 'done';
  if (s.code || Object.keys(s.stepsDone).length || Object.keys(s.recall).length) return 'progress';
  return 'new';
}

async function getModule(id) {
  if (cache.has(id)) return cache.get(id);
  const def = await loadModule(id);
  const starter = await (await fetch(`${BASE}/modules/${id}/starter.js`)).text();
  const entry = { def, starter };
  cache.set(id, entry);
  return entry;
}

async function fetchSolution(id) {
  return (await fetch(`${BASE}/modules/${id}/solution.js`)).text();
}

// A pass is recorded together with a hash of the code it ran on, so editing the file afterwards
// shows the step as "passed on an earlier version" instead of silently keeping the tick (mastery learning).
function codeHash(s) {
  let x = 2166136261;
  for (let i = 0; i < s.length; i++) { x ^= s.charCodeAt(i); x = Math.imul(x, 16777619); }
  return (x >>> 0).toString(16);
}
const currentCode = (state, starter) => state.code ?? starter;
function stepStatus(state, sid, code) {
  const v = state.stepsDone[sid];
  return !v ? 'no' : v === codeHash(code) ? 'pass' : 'stale';
}
function stepsAllDone(def, state, code) { return def.steps.every((s) => stepStatus(state, s.id, code) === 'pass'); }
function demoStatus(state, code) { return !state.demoDone ? 'no' : state.demoDone === codeHash(code) ? 'pass' : 'stale'; }
const STALE_NOTE = 'passed on an earlier version of your code — check again';

/** Record per-step outcomes of a test run on `code`; steps with no results are left untouched. */
function applyTestResults(state, def, tests, code, { clearAll = false } = {}) {
  const byStep = {};
  for (const t of tests) (byStep[t.step] = byStep[t.step] || []).push(t);
  const hash = codeHash(code);
  for (const s of def.steps) {
    const list = byStep[s.id];
    if (clearAll) { delete state.stepsDone[s.id]; continue; }
    if (!list) continue;
    if (list.every((t) => t.pass)) state.stepsDone[s.id] = hash; else delete state.stepsDone[s.id];
  }
}

function debounce(fn, ms) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

function download(name, text) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/javascript' }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// Append a batch of console lines with one layout pass, keeping the DOM bounded.
function appendLogLines($c, lines) {
  $c.classList.remove('hidden');
  const frag = document.createDocumentFragment();
  for (const l of lines) {
    const d = document.createElement('div');
    d.className = `log-line ${l.level === 'error' ? 'log-error' : l.level === 'warn' ? 'log-warn' : ''}`;
    d.textContent = l.text;
    frag.appendChild(d);
  }
  $c.appendChild(frag);
  while ($c.childElementCount > LOG_DOM_MAX) $c.firstElementChild.remove();
  $c.scrollTop = $c.scrollHeight;
}
const logLinesOf = (msg) => msg.lines || [{ level: msg.level, text: msg.text }];

/** An error status line; a syntax error with a known line becomes a link that moves the editor there. */
function errorLine(msg, editor = null) {
  const el = h(`<div class="status-line bad"></div>`);
  if (msg.line && editor) {
    const before = msg.message.split(/\bon line \d+/)[0];
    el.append(before, h(`<a href="#" data-goto>on line ${msg.line}${msg.col ? `, column ${msg.col}` : ''}</a>`), msg.message.slice(before.length).replace(/^on line \d+(, column \d+)?/, ''));
    el.querySelector('[data-goto]').addEventListener('click', (e) => { e.preventDefault(); editor.goTo(msg.line); });
  } else {
    el.textContent = msg.message.startsWith('Your file') || msg.message.startsWith('Timed out') ? msg.message : `Error: ${msg.message}`;
  }
  return el;
}

function storageWarning() {
  return h(`<div class="badge warn" style="margin:0 0 16px">Your browser refused to save progress (private mode, storage disabled or full). Work will be lost on reload: use Download / Export progress.</div>`);
}

// ---------- sidebar ----------

function renderSidebar(activeId = null) {
  const done = MODULES.filter((m) => isComplete(m.id)).length;
  const parts = [`<div class="brand"><a href="#/">Build to Understand<span class="sub">the LLM stack, from scratch</span></a></div>`,
    `<button class="btn btn-small sidebar-toggle" id="sidebar-close" type="button">Close</button>`,
    `<div class="progress-bar" title="${done} of ${READY.length} modules complete"><i style="width:${(100 * done) / READY.length}%"></i></div>`,
    `<div class="small muted">${done} / ${READY.length} modules complete</div>`];
  for (const t of TRACKS) {
    const mods = MODULES.filter((m) => m.track === t.id);
    parts.push(`<div class="nav-track"><div class="nav-track-title">${esc(t.title)}</div>${mods.map((m) => {
      const st = m.status !== 'ready' ? 'planned' : moduleStatus(m.id);
      return `<a class="nav-item ${m.id === activeId ? 'active' : ''}" href="#/m/${m.id}"><span class="dot ${st}"></span><span class="num">${m.id.slice(0, 2)}</span><span>${esc(m.title)}</span></a>`;
    }).join('')}</div>`);
  }
  const due = dueReviews().length;
  parts.push(`<div class="nav-links"><a href="#/review">Review queue${due ? ` (${due} due)` : ''}</a><a href="#/chat">Chat playground</a><a href="#/about">How this lab teaches</a><a href="https://github.com/SumerSG/build-to-understand-llms" target="_blank" rel="noopener">Source</a><a href="#" id="theme-toggle">Theme</a></div>`);
  $sidebar.innerHTML = parts.join('');
  $sidebar.querySelector('#theme-toggle').addEventListener('click', (e) => {
    e.preventDefault();
    const cur = document.documentElement.dataset.theme;
    const dark = cur ? cur === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('btu:theme', document.documentElement.dataset.theme); } catch { /* ignore */ }
  });
  $sidebar.querySelector('#sidebar-close').addEventListener('click', () => $sidebar.classList.remove('open'));
}

function sidebarButton() {
  const b = h(`<button class="btn btn-small sidebar-toggle" type="button">Menu</button>`);
  b.addEventListener('click', () => $sidebar.classList.add('open'));
  return b;
}

// ---------- home ----------

function renderHome() {
  const due = dueReviews();
  const last = store.all().lastModule;
  const lastMod = last ? moduleById(last) : null;
  const next = MODULES.find((m) => m.status === 'ready' && !isComplete(m.id));
  $app.innerHTML = '';
  $app.appendChild(sidebarButton());
  if (!store.lastWriteOk) $app.appendChild(storageWarning());
  $app.appendChild(h(`<section class="hero">
    <h1>Build to understand the LLM stack</h1>
    <p class="lede">Thirty-four self-contained projects: build every layer of a modern language-model system in your browser,
    and nothing counts as done until <em>your</em> code passes the tests and runs the goal demo.</p>
    <div class="hero-actions">
      ${lastMod ? `<a class="btn btn-primary" href="#/m/${lastMod.id}">Continue: ${esc(lastMod.title)}</a>` : next ? `<a class="btn btn-primary" href="#/m/${next.id}">Start: ${esc(next.title)}</a>` : ''}
      ${lastMod && next && next.id !== lastMod.id ? `<a class="btn" href="#/m/${next.id}">Next up: ${esc(next.title)}</a>` : ''}
      ${due.length ? `<a class="btn" href="#/review">Review ${due.length} due item${due.length > 1 ? 's' : ''}</a>` : ''}
      <a class="btn btn-ghost" href="#/about">How this lab teaches</a>
    </div>
  </section>`));
  const grid = h(`<div class="grid-2 tracks"></div>`);
  for (const t of TRACKS) {
    const mods = MODULES.filter((m) => m.track === t.id);
    const done = mods.filter((m) => isComplete(m.id)).length;
    grid.appendChild(h(`<div class="card track-card">
      <h3>${esc(t.title)}</h3>
      <div class="blurb">${esc(t.blurb)}</div>
      <div class="progress-bar" title="${done} of ${mods.length} complete"><i style="width:${mods.length ? (100 * done) / mods.length : 0}%"></i></div>
      <div class="mods">${mods.map((m) => `<a href="#/m/${m.id}" class="${isComplete(m.id) ? 'done' : ''}"><span class="num">${m.id.slice(0, 2)}</span><span class="title">${esc(m.title)}</span><span class="mins">${m.minutes} min</span><span class="check" aria-hidden="true"></span></a>`).join('')}</div>
    </div>`));
  }
  $app.appendChild(grid);
  $app.appendChild(h(`<div class="card loop-card">
    <h3>The loop in every module</h3>
    <ol class="loop">
      <li><b>Recall</b><span>a few facts from earlier modules</span></li>
      <li><b>Concept</b><span>a short read with prediction checkpoints</span></li>
      <li><b>Build</b><span>3–6 tested steps with a hint ladder</span></li>
      <li><b>Goal</b><span>a demo that runs on your own code</span></li>
      <li><b>Reflect</b><span>explain it in your own words</span></li>
    </ol>
    <p class="small muted">Your work is saved in this browser only. <a href="#/about">Why it is designed this way.</a></p>
    <div class="row"><button class="btn btn-small" id="export-progress" type="button">Export progress</button><button class="btn btn-small" id="import-progress" type="button">Import progress</button><button class="btn btn-small" id="reset-progress" type="button">Reset progress</button></div>
  </div>`));
  $app.querySelector('#export-progress').addEventListener('click', () => download('btu-progress.json', store.export()));
  $app.querySelector('#import-progress').addEventListener('click', () => {
    const inp = document.createElement('input'); inp.type = 'file'; inp.accept = 'application/json';
    inp.onchange = async () => {
      const f = inp.files[0];
      if (!f) return;
      let ok = false;
      try { store.import(await f.text()); ok = true; } catch (e) { alert('Could not import: ' + e.message + '. Your existing progress is unchanged.'); }
      if (ok) route();   // only re-render once the new data is known to be valid and saved
    };
    inp.click();
  });
  $app.querySelector('#reset-progress').addEventListener('click', () => { if (confirm('Erase all saved code, answers and progress in this browser?')) { store.reset(); route(); } });
}

// ---------- about ----------

function renderAbout() {
  $app.innerHTML = '';
  $app.appendChild(sidebarButton());
  $app.appendChild(h(`<div class="concept prose measure">${md(`
# How this lab teaches

The thesis: **you understand a system when you can build a working version of it.** Every feature in the UI comes from a specific learning principle. The full map is in [docs/PEDAGOGY.md](https://github.com/SumerSG/build-to-understand-llms/blob/main/docs/PEDAGOGY.md); here is the short version.

| You do | Because |
|---|---|
| Answer recall questions about earlier modules before starting | Retrieval practice strengthens memory far more than re-reading (Roediger & Karpicke 2006); spacing it over modules beats cramming (Cepeda et al. 2006). |
| Write a prediction before revealing an answer | Predict–Observe–Explain (White & Gunstone 1992) turns reading into hypothesis testing; a committed wrong prediction is the most memorable kind of feedback. |
| Build in small tested steps, starting from a worked example | Constructionism (Papert): knowledge is built by making things. Cognitive load theory (Sweller): worked examples first, then faded scaffolding (Renkl & Atkinson). |
| Get hints one rung at a time, only after an attempt | Scaffolding in the zone of proximal development (Vygotsky; Wood, Bruner & Ross); productive failure (Kapur 2008) — struggling first improves later learning. |
| Pass all tests before the module counts as complete | Mastery learning (Bloom 1968): feedback and correctives until mastery, rather than moving on with gaps. Gates are soft, because autonomy matters (Deci & Ryan). |
| Run a goal demo that draws what your code did | Constructionism again (a public, inspectable artifact) and dual coding (Paivio): a picture next to the mechanism. |
| Explain the threshold concept in your own words | The self-explanation effect (Chi et al. 1989) and the Feynman technique expose gaps that recognition hides. Threshold concepts (Meyer & Land) are the ideas that reorganise everything else. |
| See a review queue on the home page | Leitner-box spaced repetition: 1, 3, 7, 14, 30 days. Getting at least three quarters of a module's questions right in one sitting moves it up a box; otherwise it goes back to box 1, due tomorrow. |

## Why each module imports the reference implementation of earlier modules

Isolation of failure. If module 7 (pre-training) ran on your module 2 autograd, a subtle gradient bug would surface three modules later as "loss does not go down", which is the least informative failure there is. So each module is a self-contained project built on \`lib/\`, the vetted reference. Swapping in your own implementation is a stretch goal.

## Honesty about scale

Everything runs in your browser on plain JavaScript typed arrays. The models are tiny (about a hundred thousand parameters), the corpora are kilobytes, and the "clusters" are simulators. The point is the mechanism, and each module says plainly where the toy differs from production.
`)}</div>`));
}

// ---------- review queue ----------

const reviewNeeded = (n) => Math.ceil(n * 0.75);

async function renderReview(seq) {
  $app.innerHTML = '';
  $app.appendChild(sidebarButton());
  const due = dueReviews();
  const all = reviewSummary();
  const wrap = h(`<div class="measure"><h1>Review queue</h1><p class="muted lede-p">Spaced retrieval on modules you completed. Get at least three quarters of a module's questions right in one sitting (all 3 of 3, or 3 of 4) to move it up a box (${LEITNER_DAYS.join(', ')} days); otherwise it goes back to box 1, due tomorrow.</p></div>`);
  $app.appendChild(wrap);
  if (!due.length) {
    wrap.appendChild(h(`<div class="card">Nothing due right now. ${all.length ? `${all.length} module${all.length > 1 ? 's' : ''} scheduled; next due ${new Date(Math.min(...all.map((r) => r.due))).toLocaleDateString()}.` : 'Complete a module to enrol it.'}</div>`));
    return;
  }
  for (const item of due) {
    const meta = moduleById(item.moduleId);
    if (!meta) continue;
    let def;
    try { ({ def } = await getModule(item.moduleId)); } catch { continue; }
    if (seq !== renderSeq) return;
    const qs = (def.review && def.review.length ? def.review : def.recall) || [];
    const need = reviewNeeded(qs.length);
    const card = h(`<div class="card review-card"><h3>${esc(meta.title)} <span class="badge">box ${item.box + 1}</span></h3><p class="muted small">${qs.length} questions; ${need} correct moves this module up a box.</p></div>`);
    let answered = 0, correct = 0;
    for (const q of qs) {
      const qEl = h(`<div class="quiz-q"><div class="q">${inline(q.q)}</div></div>`);
      q.options.forEach((opt, oi) => {
        const b = h(`<button class="quiz-opt" type="button">${inline(opt)}</button>`);
        b.addEventListener('click', () => {
          if (qEl.dataset.done) return;
          qEl.dataset.done = '1';
          answered++;
          if (oi === q.answer) { correct++; b.classList.add('correct'); } else { b.classList.add('wrong'); qEl.querySelectorAll('.quiz-opt')[q.answer].classList.add('correct'); }
          qEl.appendChild(h(`<div class="quiz-why">${md(q.why || '')}</div>`));
          if (answered === qs.length) {
            const ok = correct >= need;
            const r = scheduleReview(item.moduleId, ok);
            card.appendChild(h(`<div class="status-line ${ok ? 'ok' : 'bad'}">${correct}/${qs.length} correct (${need} needed). ${ok ? 'Moved up' : 'Back'} to box ${r.box + 1}; next review in ${LEITNER_DAYS[r.box]} day${LEITNER_DAYS[r.box] > 1 ? 's' : ''}. <a href="#/m/${item.moduleId}">Revisit the module</a></div>`));
            renderSidebar();
          }
        });
        qEl.appendChild(b);
      });
      card.appendChild(qEl);
    }
    wrap.appendChild(card);
  }
}

// ---------- module page ----------

async function renderModulePage(id, phaseArg, seq) {
  const meta = moduleById(id);
  if (!meta) { $app.innerHTML = '<p>Unknown module.</p>'; return; }
  $app.innerHTML = '';
  $app.appendChild(sidebarButton());
  if (meta.status !== 'ready') {
    $app.appendChild(h(`<div class="mod-head" data-num="${id.slice(0, 2)}"><nav class="crumbs"><span>Planned</span></nav><h1>${esc(meta.title)}</h1><div class="goal-banner"><p class="goal">${esc(meta.goal)}</p><p class="threshold">This module is planned and not yet written. Its design is in <code>docs/CURRICULUM_BRIEFS.md</code> and the shared pieces it needs are in <code>docs/ROADMAP.md</code>.</p></div></div>`));
    return;
  }
  let entry;
  try { entry = await getModule(id); } catch (err) {
    if (seq !== renderSeq) return;
    failedLoads.add(id);
    renderSidebar(id);
    $app.appendChild(h(`<div class="mod-head" data-num="${id.slice(0, 2)}"><nav class="crumbs"><span>In progress</span></nav><h1>${esc(meta.title)}</h1><div class="goal-banner"><p class="goal">${esc(meta.goal)}</p><p class="threshold">This module is still being written. It will appear here once it passes its checks.</p></div><p><a class="btn" href="#/">Back to the lab</a></p><p class="muted small">Load error: ${esc(err.message)}</p></div>`));
    return;
  }
  if (seq !== renderSeq) return;   // the learner navigated on while this module was loading
  const { def, starter } = entry;
  const state = store.module(id);
  const track = TRACKS.find((t) => t.id === meta.track);
  const hasRecall = def.recall && def.recall.length;
  let phase = phaseArg || state.tab || (hasRecall ? 'recall' : 'concept');
  if (phase === 'recall' && !hasRecall) phase = 'concept';
  const missingPrereqs = (def.prereqs || []).filter((p) => !isComplete(p));

  const head = h(`<div class="mod-head" data-num="${id.slice(0, 2)}">
    <nav class="crumbs" aria-label="Breadcrumb"><span>${esc(track ? track.title : meta.track)}</span><span class="sep"></span><span class="num">Module ${id.slice(0, 2)}</span><span class="sep"></span><span class="num">About ${meta.minutes} min</span></nav>
    <h1>${esc(def.title)}</h1>
    <div class="goal-banner"><p class="goal">${esc(def.goal)}</p><p class="threshold"><b>The idea to take away.</b> ${esc(def.threshold || '')}</p>${missingPrereqs.length ? `<p class="prereq-note">Builds on ${missingPrereqs.map((p) => `<a href="#/m/${p}">${esc(moduleById(p)?.title || p)}</a>`).join(', ')}.</p>` : ''}</div>
  </div>`);
  $app.appendChild(head);
  if (!store.lastWriteOk) $app.appendChild(storageWarning());

  const phasesEl = h(`<div class="phases"></div>`);
  $app.appendChild(phasesEl);
  currentPage = { id, def, starter, phase, phasesEl };
  document.dispatchEvent(new Event('btu:page'));
  refreshPhases();
  store.update(id, { tab: phase });
  const body = h(`<div class="phase-body"></div>`);
  $app.appendChild(body);
  const timeouts = Object.assign({}, DEFAULT_TIMEOUTS, def.timeouts || {});
  if (phase === 'recall') renderRecall(body, id, def);
  else if (phase === 'concept') renderConcept(body, id, def);
  else if (phase === 'build') renderBuild(body, id, def, starter, timeouts);
  else if (phase === 'goal') renderGoal(body, id, def, starter, timeouts);
  else if (phase === 'reflect') renderReflect(body, id, def, starter, timeouts);
}

/** Re-render the phase strip (ticks) from the current saved state; safe to call after any store.update. */
function refreshPhases() {
  if (!currentPage) return;
  const { id, def, starter, phase, phasesEl } = currentPage;
  const state = store.module(id);
  const code = currentCode(state, starter);
  const hasRecall = def.recall && def.recall.length;
  const build = stepsAllDone(def, state, code) ? 'pass' : def.steps.some((s) => stepStatus(state, s.id, code) === 'stale') && def.steps.every((s) => stepStatus(state, s.id, code) !== 'no') ? 'stale' : 'no';
  const status = {
    recall: hasRecall && Object.keys(state.recall).length >= def.recall.length ? 'pass' : 'no',
    concept: state.conceptRead ? 'pass' : 'no',
    build,
    goal: demoStatus(state, code),
    reflect: state.completedAt ? 'pass' : 'no',
  };
  phasesEl.innerHTML = '';
  const ind = h(`<span class="phases-ind" aria-hidden="true"></span>`);
  phasesEl.appendChild(ind);
  for (const p of PHASES) {
    if (p.id === 'recall' && !hasRecall) continue;
    const mark = status[p.id] === 'pass' ? '<span class="tick">✓</span>' : status[p.id] === 'stale' ? `<span class="hint-stale" title="${STALE_NOTE}">↻</span>` : '';
    const b = h(`<div class="phase ${p.id === phase ? 'active' : ''}" data-phase="${p.id}" role="tab" aria-selected="${p.id === phase}">${mark}${p.label}</div>`);
    b.addEventListener('click', () => { location.hash = `#/m/${id}/${p.id}`; });
    phasesEl.appendChild(b);
  }
  // Segmented control: the indicator slides from the previously selected phase of the same module.
  const from = lastPhaseUI && lastPhaseUI.id === id && lastPhaseUI.phase !== phase ? phasesEl.querySelector(`.phase[data-phase="${lastPhaseUI.phase}"]`) : null;
  if (from) { ind.style.transition = 'none'; placePhaseIndicator(from); void ind.offsetWidth; ind.style.transition = ''; }
  placePhaseIndicator();
  lastPhaseUI = { id, phase };
}

function placePhaseIndicator(target = null) {
  if (!currentPage) return;
  const ind = currentPage.phasesEl.querySelector('.phases-ind');
  const el = target || currentPage.phasesEl.querySelector('.phase.active');
  if (!ind || !el) return;
  const place = () => {
    ind.style.transform = `translateX(${el.offsetLeft}px)`;
    ind.style.width = `${el.offsetWidth}px`;
  };
  place();
  if (!target) {
    // Measure again once layout and fonts have settled, so the indicator never sits at a stale position.
    requestAnimationFrame(place);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(place);
    el.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
}
if (typeof ResizeObserver !== 'undefined') {
  const ro = new ResizeObserver(() => placePhaseIndicator());
  const watch = () => { if (currentPage && currentPage.phasesEl && !currentPage.phasesEl.dataset.watched) { currentPage.phasesEl.dataset.watched = '1'; ro.observe(currentPage.phasesEl); } };
  document.addEventListener('btu:page', watch);
}

function renderRecall(body, id, def) {
  const state = store.module(id);
  const about = (def.prereqs || []).length ? 'These questions are about <em>earlier</em> modules' : 'These questions are about basic JavaScript and how this lab works';
  body.appendChild(h(`<p class="muted measure">Before building, pull a few things back out of memory. ${about}; getting one wrong is useful information, not a penalty.</p>`));
  const wrap = h(`<div class="card measure"></div>`);
  let answered = Object.keys(state.recall).length;
  def.recall.forEach((q, qi) => {
    const qEl = h(`<div class="quiz-q"><div class="q">${qi + 1}. ${inline(q.q)}</div></div>`);
    const prev = state.recall[qi];
    q.options.forEach((opt, oi) => {
      const b = h(`<button class="quiz-opt" type="button">${inline(opt)}</button>`);
      if (prev !== undefined) {
        if (oi === q.answer) b.classList.add('correct');
        if (oi === prev && prev !== q.answer) b.classList.add('wrong');
      }
      b.addEventListener('click', () => {
        if (qEl.dataset.done) return;
        qEl.dataset.done = '1';
        state.recall[qi] = oi;
        store.update(id, { recall: state.recall });
        if (oi === q.answer) b.classList.add('correct'); else { b.classList.add('wrong'); qEl.querySelectorAll('.quiz-opt')[q.answer].classList.add('correct'); }
        qEl.appendChild(h(`<div class="quiz-why">${md(q.why || '')}</div>`));
        answered++;
        renderSidebar(id);
        refreshPhases();
        if (answered >= def.recall.length) showNext();
      });
      qEl.appendChild(b);
    });
    if (prev !== undefined) { qEl.dataset.done = '1'; qEl.appendChild(h(`<div class="quiz-why">${md(q.why || '')}</div>`)); }
    wrap.appendChild(qEl);
  });
  body.appendChild(wrap);
  const nextRow = h(`<div class="row" style="margin-top:24px"><a class="btn btn-primary" href="#/m/${id}/concept">Continue to Concept</a><a class="btn btn-ghost" href="#/m/${id}/concept">Skip recall</a></div>`);
  function showNext() {
    const correct = def.recall.filter((q, qi) => state.recall[qi] === q.answer).length;
    wrap.appendChild(h(`<div class="status-line ${correct === def.recall.length ? 'ok' : ''}">${correct}/${def.recall.length} correct.</div>`));
  }
  if (answered >= def.recall.length) showNext();
  body.appendChild(nextRow);
}

function renderConcept(body, id, def) {
  const state = store.module(id);
  const el = h(`<div class="concept card measure">${md(def.concept, { predictKey: (i) => `concept-${i}` })}</div>`);
  activatePredicts(el, { getSaved: (k) => state.predictions[k], setSaved: (k, v) => {
    state.predictions[k] = v;
    const cards = [...el.querySelectorAll('.predict')];
    const allRevealed = cards.length > 0 && cards.every((c) => state.predictions[c.dataset.key]?.revealed);
    store.update(id, { predictions: state.predictions, ...(allRevealed ? { conceptRead: true } : {}) });
    if (allRevealed) refreshPhases();
  } });
  body.appendChild(el);
  const row = h(`<div class="row" style="margin-top:24px"><a class="btn btn-primary" href="#/m/${id}/build">I'm ready to build</a></div>`);
  row.firstElementChild.addEventListener('click', () => store.update(id, { conceptRead: true }));
  body.appendChild(row);
}

function renderBuild(body, id, def, starter, timeouts) {
  const state = store.module(id);
  let stepIdx = Math.min(state.step || 0, def.steps.length - 1);
  const layout = h(`<div>
    <div class="steps-nav"></div>
    <div class="build">
      <div class="step-panel card"></div>
      <div>
        <div class="editor-wrap">
          <div class="editor-bar"><span class="mono">${esc(id)}/starter.js</span><span class="spacer"></span>
            <span class="muted" id="save-state">saved</span>
            <button class="btn btn-small" id="btn-reset" type="button">Reset to starter</button>
            <button class="btn btn-small" id="btn-solution" type="button">Show reference</button>
            <button class="btn btn-small" id="btn-download" type="button">Download</button>
          </div>
          <div class="editor-host"></div>
        </div>
        <div class="row check-row">
          <button class="btn btn-primary" id="btn-check" type="button">Check this step</button>
          <button class="btn" id="btn-check-all" type="button">Check all steps</button>
          <span class="kbd">⌘↩ / Ctrl+Enter checks the current step</span>
          <span class="muted small" id="run-state"></span>
        </div>
        <div class="results"></div>
        <div class="console hidden"></div>
      </div>
    </div>
  </div>`);
  body.appendChild(layout);
  const $nav = layout.querySelector('.steps-nav');
  const $panel = layout.querySelector('.step-panel');
  const $results = layout.querySelector('.results');
  const $console = layout.querySelector('.console');
  const $runState = layout.querySelector('#run-state');
  const $saveState = layout.querySelector('#save-state');
  const $check = layout.querySelector('#btn-check'), $checkAll = layout.querySelector('#btn-check-all');

  // Saving: debounced, honest about failure, and flushable (route change, tab hide, unload).
  let pendingCode = null, panelStatus = null;
  function showSaved(ok) {
    $saveState.textContent = ok ? 'saved' : 'not saved — browser storage unavailable (use Download / Export)';
    $saveState.className = ok ? 'muted' : 'save-warn';
  }
  function commit(code) {
    pendingCode = null;
    store.update(id, { code });
    showSaved(store.lastWriteOk);
    renderNav();
    if (stepStatus(state, def.steps[stepIdx].id, code) !== panelStatus) renderPanel();
    refreshPhases();
  }
  const save = debounce((code) => { if (pendingCode === code) commit(code); }, 400);
  flushPendingSave = () => { if (pendingCode !== null) commit(pendingCode); };
  const editor = createEditor(layout.querySelector('.editor-host'), {
    value: currentCode(state, starter),
    onChange: (v) => { pendingCode = v; $saveState.textContent = 'saving…'; $saveState.className = 'muted'; save(v); },
    onRun: () => { if (!runner.busy) check(def.steps[stepIdx].id); },
  });
  showSaved(store.lastWriteOk);

  function renderNav() {
    $nav.innerHTML = '';
    const code = currentCode(state, starter);
    def.steps.forEach((s, i) => {
      const st = stepStatus(state, s.id, code);
      const chip = h(`<button class="step-chip ${i === stepIdx ? 'active' : ''} ${st === 'pass' ? 'done' : ''}" type="button" ${st === 'stale' ? `title="${STALE_NOTE}"` : ''}>${st === 'pass' ? '✓ ' : st === 'stale' ? '↻ ' : ''}${i + 1}. ${esc(s.title)}</button>`);
      chip.addEventListener('click', () => { stepIdx = i; store.update(id, { step: i }); renderNav(); renderPanel(); });
      $nav.appendChild(chip);
    });
  }

  function renderPanel() {
    const s = def.steps[stepIdx];
    const attempts = (state.attempts && state.attempts[s.id]) || 0;
    $panel.innerHTML = `<div class="muted small">Step ${stepIdx + 1} of ${def.steps.length}</div><h2>${esc(s.title)}</h2><div class="instructions">${md(s.instructions, { predictKey: (i) => `step-${s.id}-${i}` })}</div>`;
    activatePredicts($panel, { getSaved: (k) => state.predictions[k], setSaved: (k, v) => { state.predictions[k] = v; store.update(id, { predictions: state.predictions }); } });
    if (s.predict) {
      const key = `steppredict-${s.id}`;
      const pe = h(`<div class="predict" data-key="${key}"><div class="predict-label">Predict before you run</div><div class="predict-q">${md(s.predict.question)}</div><textarea class="predict-input" rows="2" placeholder="Write your prediction first…"></textarea><button class="btn btn-small predict-reveal" type="button">Reveal</button><div class="predict-a hidden">${md(s.predict.answer)}</div></div>`);
      $panel.appendChild(pe);
      activatePredicts(pe, { getSaved: (k) => state.predictions[k], setSaved: (k, v) => { state.predictions[k] = v; store.update(id, { predictions: state.predictions }); } });
    }
    const hints = h(`<div class="hints"><div class="muted small">Hints unlock one at a time after your first check of this step.</div></div>`);
    const open = (state.hintsOpen && state.hintsOpen[s.id]) || 0;
    (s.hints || []).forEach((text, hi) => {
      const unlocked = attempts > 0 && hi <= open;
      const revealed = attempts > 0 && hi < open;
      const hint = h(`<div class="hint ${revealed ? 'open' : ''}"><button type="button" aria-expanded="${revealed}" ${unlocked ? '' : 'disabled'}><span>Hint ${hi + 1} of ${s.hints.length}${['', ' · a nudge', ' · the strategy', ' · nearly the code'][hi + 1] || ''}</span></button><div class="hint-body"><div class="hint-inner"><div>${md(text)}</div></div></div></div>`);
      hint.querySelector('button').addEventListener('click', (e) => {
        // Expand in place (animated) and unlock the next rung; a revealed hint can be folded again without losing it.
        const open = hint.classList.toggle('open');
        e.currentTarget.setAttribute('aria-expanded', String(open));
        if (!open) return;
        state.hintsOpen = state.hintsOpen || {};
        state.hintsOpen[s.id] = Math.max(state.hintsOpen[s.id] || 0, hi + 1);
        store.update(id, { hintsOpen: state.hintsOpen });
        const next = hint.nextElementSibling && hint.nextElementSibling.querySelector('button');
        if (next) next.disabled = false;
      });
      hints.appendChild(hint);
    });
    $panel.appendChild(hints);
    panelStatus = stepStatus(state, s.id, currentCode(state, starter));
    if (panelStatus === 'pass') $panel.appendChild(h(`<div class="status-line ok"><span>This step's tests pass.${stepIdx + 1 < def.steps.length ? ` <a href="#" data-next>Next step ›</a>` : ` <a href="#/m/${id}/goal">Run the goal ›</a>`}</span></div>`));
    else if (panelStatus === 'stale') $panel.appendChild(h(`<div class="status-line hint-stale">↻ This step ${STALE_NOTE}.</div>`));
    const nx = $panel.querySelector('[data-next]');
    if (nx) nx.addEventListener('click', (e) => { e.preventDefault(); stepIdx++; store.update(id, { step: stepIdx }); renderNav(); renderPanel(); });
  }

  let activeRun = 0;
  async function check(stepId) {
    const code = editor.getValue();
    pendingCode = null;
    store.update(id, { code });
    showSaved(store.lastWriteOk);
    $results.innerHTML = '';
    $console.innerHTML = '';
    $console.classList.add('hidden');
    $runState.textContent = 'running…';
    $check.disabled = true; $checkAll.disabled = true;
    const myRun = ++activeRun;
    let res;
    try {
      res = await runner.run({ mode: 'tests', code, moduleId: id, stepId, timeout: timeouts.tests, onMessage: (msg) => {
        if (msg.type === 'log') appendLogLines($console, logLinesOf(msg));
        if (msg.type === 'test') {
          const stepTitle = def.steps.find((s) => s.id === msg.step)?.title || msg.step;
          $results.appendChild(h(`<div class="test ${msg.pass ? 'pass' : 'fail'}"><span class="mark" role="img" aria-label="${msg.pass ? 'passed' : 'failed'}"></span><div><div>${stepId ? '' : `<span class="muted">${esc(stepTitle)} · </span>`}${esc(msg.name)}</div>${msg.pass ? '' : `<div class="msg">${esc(msg.message)}</div>`}</div><span class="ms">${msg.ms.toFixed(0)} ms</span></div>`));
        }
        if (msg.type === 'error') $results.appendChild(errorLine(msg, editor));
      } });
    } catch (err) {
      if (err.message !== 'superseded' && err.message !== 'navigated') $results.appendChild(h(`<div class="status-line bad">${esc(err.message)}</div>`));
      return;
    } finally {
      if (myRun === activeRun) { $check.disabled = false; $checkAll.disabled = false; $runState.textContent = ''; }
    }
    state.attempts = typeof state.attempts === 'object' && state.attempts ? state.attempts : {};
    // Hints unlock per step after a genuine attempt at that step: "Check all" counts only for the step you are on.
    const attempted = stepId || def.steps[stepIdx].id;
    state.attempts[attempted] = (state.attempts[attempted] || 0) + 1;
    if (res.summary) {
      applyTestResults(state, def, res.tests, code);
    } else {
      // The file did not load at all (syntax error or a throw at top level): nothing can be considered passing.
      applyTestResults(state, def, [], code, { clearAll: true });
      $results.appendChild(h(`<div class="status-line bad">No tests ran, so every step is marked as not passing until the file loads again.</div>`));
    }
    store.update(id, { attempts: state.attempts, stepsDone: state.stepsDone });
    if (res.summary) {
      const ok = res.summary.failed === 0 && res.summary.total > 0;
      $results.prepend(h(`<div class="status-line ${ok ? 'ok' : 'bad'}">${res.summary.passed}/${res.summary.total} tests passed${ok && stepsAllDone(def, state, code) ? ' — all steps done. Head to the Goal tab.' : ''}</div>`));
    }
    renderNav();
    renderPanel();
    renderSidebar(id);
    refreshPhases();
  }

  $check.addEventListener('click', () => check(def.steps[stepIdx].id));
  $checkAll.addEventListener('click', () => check(null));
  layout.querySelector('#btn-reset').addEventListener('click', () => { if (confirm('Replace your code with the starter file?')) { editor.setValue(starter); commit(starter); } });
  layout.querySelector('#btn-download').addEventListener('click', () => download(`${id}.js`, editor.getValue()));
  layout.querySelector('#btn-solution').addEventListener('click', async () => {
    const cur = def.steps[stepIdx].id;
    const attempts = (state.attempts && typeof state.attempts === 'object' && state.attempts[cur]) || 0;
    if (attempts < 2 && !confirm(`You have not tried this step's tests twice yet (${attempts} so far). The reference contains every step's solution; looking now will cost you most of the learning. Show it anyway?`)) return;
    const sol = await fetchSolution(id);
    const wrap = h(`<div class="card"><div class="row"><b>Reference solution</b><span class="spacer"></span><button class="btn btn-small" id="sol-copy" type="button">Load into editor</button><button class="btn btn-small" id="sol-close" type="button">Close</button></div><pre class="code"><code>${esc(sol)}</code></pre></div>`);
    wrap.querySelector('#sol-close').addEventListener('click', () => wrap.remove());
    wrap.querySelector('#sol-copy').addEventListener('click', () => { if (confirm('Replace your code with the reference? Your version will be lost.')) { editor.setValue(sol); commit(sol); } });
    $results.before(wrap);
  });
  renderNav();
  renderPanel();
  setTimeout(() => editor.refresh(), 0);
}

function renderGoal(body, id, def, starter, timeouts) {
  const state = store.module(id);
  const code = currentCode(state, starter);
  const ready = stepsAllDone(def, state, code);
  const demo = demoStatus(state, code);
  const wrap = h(`<div>
    <div class="card measure">
      <h2>Run the goal</h2>
      <p>${esc(def.goal)}</p>
      <p class="muted small">The demo runs <em>your</em> code from the Build tab. ${ready ? 'All steps pass on the current code.' : 'Not all steps pass on the current code yet; the demo may fail or show odd results, which is itself informative.'}</p>
      <div class="row"><button class="btn btn-primary" id="btn-run" type="button">Run the goal demo</button><button class="btn" id="btn-stop" type="button" disabled>Stop</button><span class="muted small" id="goal-state"></span></div>
      ${demo === 'pass' ? `<div class="done-banner"><b>Working goal achieved</b> (last run)<div>${md(state.demoSummary || '')}</div></div>` : ''}
      ${demo === 'stale' ? `<div class="status-line hint-stale">↻ The goal demo ${STALE_NOTE} — run it again on this version.</div>` : ''}
    </div>
    <div class="goal-out"></div>
  </div>`);
  body.appendChild(wrap);
  const $out = wrap.querySelector('.goal-out');
  const $state = wrap.querySelector('#goal-state');
  const $run = wrap.querySelector('#btn-run');
  const $stop = wrap.querySelector('#btn-stop');
  let progressEl = null, consoleEl = null;
  $stop.addEventListener('click', () => runner.cancel('stopped'));
  $run.addEventListener('click', async () => {
    $out.innerHTML = '';
    consoleEl = null; progressEl = null;
    $run.disabled = true; $stop.disabled = false;
    $state.textContent = 'running…';
    const runCode = currentCode(state, starter);
    try {
      const res = await runner.run({ mode: 'demo', code: runCode, moduleId: id, timeout: timeouts.demo, onMessage: (msg) => {
        if (msg.type === 'log') {
          if (!consoleEl) { consoleEl = h(`<div class="console"></div>`); $out.appendChild(consoleEl); }
          appendLogLines(consoleEl, logLinesOf(msg));
        }
        else if (msg.type === 'md') { const d = h(`<div class="card md">${md(msg.markdown)}</div>`); $out.appendChild(d); }
        else if (msg.type === 'plot' || msg.type === 'bar' || msg.type === 'heatmap' || msg.type === 'table') { consoleEl = null; renderChart($out, msg.type, msg.spec); }
        else if (msg.type === 'progress') {
          if (!progressEl) { progressEl = h(`<div class="progress-row"><div class="progress-bar"><i style="width:0%"></i></div><span class="plabel"></span></div>`); $out.appendChild(progressEl); }
          progressEl.querySelector('i').style.width = `${Math.round(100 * Math.max(0, Math.min(1, msg.fraction)))}%`;
          progressEl.querySelector('.plabel').textContent = msg.label || '';
        }
        else if (msg.type === 'demo-done') {
          store.update(id, { demoDone: codeHash(runCode), demoSummary: msg.summary });
          $out.appendChild(h(`<div class="done-banner"><b>Working goal achieved.</b><div>${md(msg.summary)}</div><div style="margin-top:8px"><a class="btn btn-primary btn-small" href="#/m/${id}/reflect">Continue to Reflect</a></div></div>`));
          renderSidebar(id);
          refreshPhases();
        }
        else if (msg.type === 'error') {
          const el = errorLine(msg);
          if (msg.stack) el.appendChild(h(`<pre class="code small">${esc(String(msg.stack).split('\n').slice(0, 6).join('\n'))}</pre>`));
          $out.appendChild(el);
        }
      } });
      $state.textContent = res.error ? 'finished with errors' : 'finished';
    } catch (err) {
      $state.textContent = err.message === 'stopped' ? 'stopped' : err.message;
    } finally {
      $run.disabled = false; $stop.disabled = true;
    }
  });
}

function renderReflect(body, id, def, starter, timeouts) {
  const state = store.module(id);
  const code = currentCode(state, starter);
  const mark = (st) => (st === 'pass' ? '✓' : st === 'stale' ? '↻' : '○');
  const cls = (st) => (st === 'pass' ? 'ok' : st === 'stale' ? 'hint-stale' : 'no');
  const buildSt = stepsAllDone(def, state, code) ? 'pass' : def.steps.some((s) => stepStatus(state, s.id, code) === 'stale') ? 'stale' : 'no';
  const demoSt = demoStatus(state, code);
  const wrap = h(`<div class="reflect measure">
    <div class="card">
      <h2>Explain it in your own words</h2>
      <p class="muted small">Write as if teaching a colleague who has not seen the module. Naming the thing you are least sure about is the most useful sentence you can write.</p>
      <div class="prompts"></div>
    </div>
    <div class="card">
      <h3>Stretch goals (optional)</h3>
      <ul>${(def.stretch || []).map((s) => `<li>${md(s).replace(/^<p>|<\/p>$/g, '')}</li>`).join('')}</ul>
    </div>
    <div class="card">
      <h3>Complete the module</h3>
      <ul class="checklist">
        <li class="${cls(buildSt)}" id="build-check">${mark(buildSt)} All build steps pass on the current code${buildSt === 'stale' ? ` (${STALE_NOTE})` : ''}</li>
        <li class="${cls(demoSt)}">${mark(demoSt)} Goal demo ran on the current code${demoSt === 'stale' ? ` (${STALE_NOTE})` : ''}</li>
        <li class="no" id="refl-check">○ At least one reflection written</li>
      </ul>
      <p class="muted small">Completing re-runs every test on the code in the editor first.</p>
      <div class="row"><button class="btn btn-primary" id="btn-complete" type="button" disabled>${state.completedAt ? 'Completed ✓' : 'Mark module complete'}</button>${state.completedAt ? `<span class="muted small">Completed ${new Date(state.completedAt).toLocaleDateString()}. Enrolled in the review queue.</span>` : ''}</div>
      <div id="complete-msg"></div>
    </div>
  </div>`);
  body.appendChild(wrap);
  const $prompts = wrap.querySelector('.prompts');
  const $reflCheck = wrap.querySelector('#refl-check');
  const $complete = wrap.querySelector('#btn-complete');
  const $msg = wrap.querySelector('#complete-msg');
  const anyReflection = () => Object.values(state.reflections).some((t) => t && t.trim().length > 20);
  function updateChecklist() {
    const any = anyReflection();
    $reflCheck.className = any ? 'ok' : 'no';
    $reflCheck.textContent = `${any ? '✓' : '○'} At least one reflection written (20+ characters)`;
    const ready = buildSt !== 'no' && demoSt === 'pass';   // stale steps are re-checked on completion; a stale demo must be re-run
    $complete.disabled = !(ready && any) || !!state.completedAt;
  }
  (def.reflection || []).forEach((p, i) => {
    const el = h(`<div class="quiz-q"><div class="q">${md(p).replace(/^<p>|<\/p>$/g, '')}</div><textarea placeholder="Your explanation…"></textarea></div>`);
    const ta = el.querySelector('textarea');
    ta.value = state.reflections[i] || '';
    ta.addEventListener('input', debounce(() => { state.reflections[i] = ta.value; store.update(id, { reflections: state.reflections }); updateChecklist(); }, 300));
    $prompts.appendChild(el);
  });
  updateChecklist();
  $complete.addEventListener('click', async () => {
    $complete.disabled = true;
    $complete.textContent = 'Re-checking your code…';
    $msg.innerHTML = '';
    let res = null;
    try { res = await runner.run({ mode: 'tests', code, moduleId: id, stepId: null, timeout: timeouts.tests, onMessage: () => {} }); }
    catch (err) { if (err.message === 'navigated') return; $msg.appendChild(h(`<div class="status-line bad">${esc(err.message)}</div>`)); }
    if (res) applyTestResults(state, def, res.tests, code, { clearAll: !res.summary });
    store.update(id, { stepsDone: state.stepsDone });
    const ok = res && res.summary && res.summary.failed === 0 && res.summary.total > 0 && stepsAllDone(def, state, code) && demoStatus(state, code) === 'pass';
    if (!ok) {
      const failed = res && res.summary ? `${res.summary.failed} of ${res.summary.total} tests fail on the current code` : res && res.error ? res.error.message : 'the tests could not run';
      $msg.appendChild(h(`<div class="status-line bad">Not complete yet: ${esc(failed)}. <a href="#/m/${id}/build">Back to Build</a></div>`));
      $complete.textContent = 'Mark module complete';
      renderSidebar(id); refreshPhases();
      return;
    }
    store.update(id, { completedAt: Date.now() });
    enrollReview(id);
    renderSidebar(id);
    refreshPhases();
    const nx = nextModule(id);
    $complete.textContent = 'Completed ✓';
    $complete.parentElement.appendChild(h(`<span>Enrolled in the review queue. ${nx ? `<a class="btn btn-small" href="#/m/${nx.id}">Next: ${esc(nx.title)}</a>` : 'That was the last module.'}</span>`));
  });
}


// ---------- chat playground ----------

let chatWorker = null;

function renderChat() {
  if (chatWorker) { chatWorker.terminate(); chatWorker = null; }
  $app.innerHTML = '';
  $app.appendChild(sidebarButton());
  const wrap = h(`<div class="measure">
    <h1>Chat playground</h1>
    <p class="muted lede-p">Talk to the lab's own model, running in this page: the BPE tokenizer (module 03), the GPT (module 06) trained by
    <code>tools/pretrain.mjs</code> (module 07), decoded through a KV cache that is reused across turns (modules 15 and 17) and the sampling pipeline (module 14),
    wrapped in the chat template from module 10. It is a ~100k-parameter model trained on a toy corpus, so expect corpus-like text, not answers. The point is that
    every piece of it is something you built.</p>
    <div class="card">
      <div class="row small" id="chat-info"><span class="muted">Loading model…</span></div>
      <div class="row chat-controls">
        <label class="small">temperature <input type="number" id="chat-temp" value="0.8" min="0" max="3" step="0.1"></label>
        <label class="small">top-p <input type="number" id="chat-topp" value="0.95" min="0" max="1" step="0.05"></label>
        <label class="small">top-k <input type="number" id="chat-topk" value="0" min="0" step="1"></label>
        <label class="small">max tokens <input type="number" id="chat-max" value="40" min="1" max="200" step="1"></label>
        <span class="spacer"></span>
        <button class="btn btn-small" id="chat-load" type="button">Load checkpoint JSON…</button>
        <button class="btn btn-small" id="chat-reset" type="button">New conversation</button>
      </div>
    </div>
    <div class="card chat-log" id="chat-log"></div>
    <div class="card">
      <div class="chat-composer"><input type="text" class="chat-input" id="chat-input" placeholder="Say something to the model… (the toy corpus is about cats, dogs, and the weather)" autocomplete="off">
      <button class="btn btn-primary" id="chat-send" type="button" disabled>Send</button><button class="btn" id="chat-stop" type="button" disabled>Stop</button></div>
      <div class="small muted" id="chat-stats"></div>
    </div>
  </div>`);
  $app.appendChild(wrap);
  const $log = wrap.querySelector('#chat-log'), $input = wrap.querySelector('#chat-input'), $send = wrap.querySelector('#chat-send');
  const $stop = wrap.querySelector('#chat-stop'), $stats = wrap.querySelector('#chat-stats'), $info = wrap.querySelector('#chat-info');
  const messages = [];
  let busy = false, current = null;
  function addBubble(role, text) {
    const b = h(`<div class="chat-msg chat-${role}"><div class="chat-role">${role}</div><div class="chat-text"></div></div>`);
    b.querySelector('.chat-text').textContent = text;
    $log.appendChild(b);
    $log.scrollTop = $log.scrollHeight;
    return b.querySelector('.chat-text');
  }
  function startWorker() {
    chatWorker = new Worker(new URL('./chat-worker.js', import.meta.url), { type: 'module' });
    chatWorker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'ready') { $info.innerHTML = `<span>Model: ${m.info.config.nLayer} layers · ${m.info.config.nEmbd} dims · ${m.info.config.nHead} heads · context ${m.info.config.blockSize} · vocab ${m.info.vocab} · <b>${m.info.params.toLocaleString()}</b> parameters</span>`; $send.disabled = false; }
      else if (m.type === 'prefill') { $stats.textContent = `prompt ${m.promptTokens} tokens, ${m.reusedTokens} reused from the KV cache, prefill ${m.ms.toFixed(0)} ms`; }
      else if (m.type === 'token') { if (current) { current.dataset.raw = (current.dataset.raw || '') + m.text; current.textContent = current.dataset.raw.replace(/\n{3,}/g, '\n\n').replace(/^\n+/, ''); $log.scrollTop = $log.scrollHeight; } }
      else if (m.type === 'done') { messages.push({ role: 'assistant', content: m.text }); $stats.textContent += ` · generated ${m.tokens} tokens in ${m.ms.toFixed(0)} ms (${(1000 * m.tokens / Math.max(1, m.ms)).toFixed(1)} tok/s)`; busy = false; current = null; $send.disabled = false; $stop.disabled = true; }
      else if (m.type === 'note') { $stats.textContent += ` · ${m.text}`; }
      else if (m.type === 'reset-done') { messages.length = 0; $log.innerHTML = ''; $stats.textContent = ''; }
      else if (m.type === 'error') { addBubble('error', m.message); busy = false; $send.disabled = false; $stop.disabled = true; }
    };
    chatWorker.onerror = (e) => { addBubble('error', e.message || 'worker error'); busy = false; $send.disabled = false; };
    chatWorker.postMessage({ type: 'init', base: BASE });
  }
  function send() {
    const text = $input.value.trim();
    if (!text || busy) return;
    $input.value = '';
    messages.push({ role: 'user', content: text });
    addBubble('user', text);
    current = addBubble('assistant', '');
    busy = true; $send.disabled = true; $stop.disabled = false;
    chatWorker.postMessage({ type: 'generate', messages: messages.slice(), opts: {
      temperature: +wrap.querySelector('#chat-temp').value, topP: +wrap.querySelector('#chat-topp').value,
      topK: +wrap.querySelector('#chat-topk').value, maxNewTokens: +wrap.querySelector('#chat-max').value, seed: messages.length,
    } });
  }
  $send.addEventListener('click', send);
  $input.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
  $stop.addEventListener('click', () => { if (chatWorker) { chatWorker.terminate(); } busy = false; current = null; $stop.disabled = true; $send.disabled = true; $info.innerHTML = '<span class="muted">Restarting model…</span>'; startWorker(); });
  wrap.querySelector('#chat-reset').addEventListener('click', () => { if (busy) return; chatWorker.postMessage({ type: 'reset' }); });
  wrap.querySelector('#chat-load').addEventListener('click', () => {
    const inp = document.createElement('input'); inp.type = 'file'; inp.accept = 'application/json';
    inp.onchange = async () => { const f = inp.files[0]; if (!f) return; try { const json = JSON.parse(await f.text()); $send.disabled = true; chatWorker.postMessage({ type: 'load', model: json }); messages.length = 0; $log.innerHTML = ''; } catch (e) { addBubble('error', 'Could not load checkpoint: ' + e.message); } };
    inp.click();
  });
  startWorker();
}

// ---------- router ----------

// No stored state may lock the learner out: any render failure shows a way to reset saved progress.
function showFatal(err) {
  console.error(err);
  $app.innerHTML = '';
  $app.appendChild(h(`<div class="card measure"><h2>Something went wrong</h2><pre class="code small">${esc(err && err.stack || String(err))}</pre>
    <div class="row"><a class="btn" href="#/">Reload the home page</a><button class="btn" id="fatal-export" type="button">Export saved progress</button><button class="btn" id="fatal-reset" type="button">Reset saved progress</button></div></div>`));
  $app.querySelector('#fatal-export').addEventListener('click', () => { try { download('btu-progress.json', store.export()); } catch (e) { alert('Could not export: ' + e.message); } });
  $app.querySelector('#fatal-reset').addEventListener('click', () => { if (confirm('Erase all saved code, answers and progress in this browser?')) { store.reset(); location.hash = '#/'; route(); } });
}

function leavePage() {
  if (flushPendingSave) { try { flushPendingSave(); } catch { /* ignore */ } flushPendingSave = null; }
  // Leaving the Concept phase by any route (a tab, a chip, the sidebar) counts as having read it.
  if (currentPage && currentPage.phase === 'concept') store.update(currentPage.id, { conceptRead: true });
  currentPage = null;
}

function route() {
  const seq = ++renderSeq;
  runner.cancel('navigated');
  if (chatWorker) { chatWorker.terminate(); chatWorker = null; }
  leavePage();
  $sidebar.classList.remove('open');
  const hash = location.hash || '#/';
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  window.scrollTo(0, 0);
  try {
    let p;
    if (parts[0] === 'm' && parts[1]) { renderSidebar(parts[1]); p = renderModulePage(parts[1], parts[2] || null, seq); }
    else {
      renderSidebar(null);
      if (parts[0] === 'review') p = renderReview(seq);
      else if (parts[0] === 'about') renderAbout();
      else if (parts[0] === 'chat') renderChat();
      else renderHome();
    }
    if (p) p.catch((err) => { if (seq === renderSeq) showFatal(err); });
  } catch (err) {
    showFatal(err);
  }
}

try { const t = localStorage.getItem('btu:theme'); if (t) document.documentElement.dataset.theme = t; } catch { /* ignore */ }
window.addEventListener('hashchange', route);
window.addEventListener('resize', () => placePhaseIndicator());
// The mobile sidebar is a sheet: a tap outside it (on the scrim) closes it.
document.addEventListener('click', (e) => {
  if ($sidebar.classList.contains('open') && !$sidebar.contains(e.target) && !e.target.closest('.sidebar-toggle')) $sidebar.classList.remove('open');
});
window.addEventListener('pagehide', () => { if (flushPendingSave) flushPendingSave(); });
window.addEventListener('beforeunload', () => { if (flushPendingSave) flushPendingSave(); });
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden' && flushPendingSave) flushPendingSave(); });
store.onExternalChange(() => { renderSidebar(currentPage ? currentPage.id : null); refreshPhases(); });
route();
