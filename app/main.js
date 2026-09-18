// app/main.js — the lab UI: router, sidebar, home, module page (Recall → Concept → Build → Goal → Reflect), review queue.
import { TRACKS, MODULES, moduleById, nextModule, loadModule } from '../modules/index.js';
import { store, scheduleReview, enrollReview, dueReviews, reviewSummary, LEITNER_DAYS } from './storage.js';
import { render as md, activatePredicts, escapeHtml as esc } from './markdown.js';
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
const cache = new Map();   // module id -> { def, starter }

// ---------- helpers ----------

function h(html) {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function isComplete(id) { return !!store.module(id).completedAt; }

function moduleStatus(id) {
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

function stepsAllDone(def, state) {
  return def.steps.every((s) => state.stepsDone[s.id]);
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

// ---------- sidebar ----------

function renderSidebar(activeId = null) {
  const done = MODULES.filter((m) => isComplete(m.id)).length;
  const parts = [`<div class="brand"><a href="#/">Build to Understand<span class="sub">the LLM stack, from scratch</span></a></div>`,
    `<button class="btn btn-small sidebar-toggle" id="sidebar-close" type="button">Close</button>`,
    `<div class="progress-bar" title="${done} of ${MODULES.length} modules complete"><i style="width:${(100 * done) / MODULES.length}%"></i></div>`,
    `<div class="small muted">${done} / ${MODULES.length} modules complete</div>`];
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
  $app.appendChild(h(`<div>
    <h1>Build to understand the LLM stack</h1>
    <p style="max-width:760px">Twenty-eight self-contained projects. In each one you build a real piece of a modern language-model
    system in your browser — from a tensor library to a data-center serving simulator — and it is not done until
    <em>your</em> code passes the tests and runs the goal demo. Reading is optional; building is not.</p>
    <div class="row" style="margin:14px 0 22px">
      ${lastMod ? `<a class="btn btn-primary" href="#/m/${lastMod.id}">Continue: ${esc(lastMod.title)}</a>` : ''}
      ${next && (!lastMod || next.id !== lastMod.id) ? `<a class="btn" href="#/m/${next.id}">${lastMod ? 'Next up' : 'Start'}: ${esc(next.title)}</a>` : ''}
      ${due.length ? `<a class="btn" href="#/review">Review ${due.length} due item${due.length > 1 ? 's' : ''}</a>` : ''}
      <a class="btn btn-ghost" href="#/about">How this lab teaches</a>
    </div>
  </div>`));
  const grid = h(`<div class="grid-2"></div>`);
  for (const t of TRACKS) {
    const mods = MODULES.filter((m) => m.track === t.id);
    const done = mods.filter((m) => isComplete(m.id)).length;
    grid.appendChild(h(`<div class="card track-card">
      <h3>${esc(t.title)}</h3>
      <div class="blurb">${esc(t.blurb)}</div>
      <div class="progress-bar"><i style="width:${mods.length ? (100 * done) / mods.length : 0}%"></i></div>
      <div class="mods">${mods.map((m) => `<a href="#/m/${m.id}" class="${isComplete(m.id) ? 'done' : ''}">${isComplete(m.id) ? '✓ ' : ''}${m.id.slice(0, 2)} · ${esc(m.title)} <span class="muted small">· ${m.minutes} min</span></a>`).join('')}</div>
    </div>`));
  }
  $app.appendChild(grid);
  $app.appendChild(h(`<div class="card" style="margin-top:18px;max-width:760px">
    <h3 style="margin-top:0">The loop in every module</h3>
    <p><b>Recall</b> a few facts from earlier modules (retrieval practice) → <b>Concept</b>: a short read with prediction checkpoints →
    <b>Build</b> it in 3–6 tested steps with a hint ladder → run the <b>Goal</b> demo on your own code → <b>Reflect</b> in your own words.
    Your work is saved in this browser only. <a href="#/about">Why it is designed this way.</a></p>
    <div class="row small"><button class="btn btn-small" id="export-progress" type="button">Export progress</button><button class="btn btn-small" id="import-progress" type="button">Import progress</button><button class="btn btn-small" id="reset-progress" type="button">Reset progress</button></div>
  </div>`));
  $app.querySelector('#export-progress').addEventListener('click', () => download('btu-progress.json', store.export()));
  $app.querySelector('#import-progress').addEventListener('click', () => {
    const inp = document.createElement('input'); inp.type = 'file'; inp.accept = 'application/json';
    inp.onchange = async () => { const f = inp.files[0]; if (!f) return; try { store.import(await f.text()); route(); } catch (e) { alert('Could not import: ' + e.message); } };
    inp.click();
  });
  $app.querySelector('#reset-progress').addEventListener('click', () => { if (confirm('Erase all saved code, answers and progress in this browser?')) { store.reset(); route(); } });
}

// ---------- about ----------

function renderAbout() {
  $app.innerHTML = '';
  $app.appendChild(sidebarButton());
  $app.appendChild(h(`<div class="concept">${md(`
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
| See a review queue on the home page | Leitner-box spaced repetition: 1, 3, 7, 14, 30 days. Correct answers move an item up a box; a miss sends it back. |

## Why each module imports the reference implementation of earlier modules

Isolation of failure. If module 7 (pre-training) ran on your module 2 autograd, a subtle gradient bug would surface three modules later as "loss does not go down", which is the least informative failure there is. So each module is a self-contained project built on \`lib/\`, the vetted reference. Swapping in your own implementation is a stretch goal.

## Honesty about scale

Everything runs in your browser on plain JavaScript typed arrays. The models are tiny (about a hundred thousand parameters), the corpora are kilobytes, and the "clusters" are simulators. The point is the mechanism, and each module says plainly where the toy differs from production.
`)}</div>`));
}

// ---------- review queue ----------

async function renderReview() {
  $app.innerHTML = '';
  $app.appendChild(sidebarButton());
  const due = dueReviews();
  const all = reviewSummary();
  const wrap = h(`<div><h1>Review queue</h1><p class="muted" style="max-width:720px">Spaced retrieval on modules you completed. Each correct answer moves the module up a box (${LEITNER_DAYS.join(', ')} days); a miss sends it back to tomorrow.</p></div>`);
  $app.appendChild(wrap);
  if (!due.length) {
    wrap.appendChild(h(`<div class="card">Nothing due right now. ${all.length ? `${all.length} module${all.length > 1 ? 's' : ''} scheduled; next due ${new Date(Math.min(...all.map((r) => r.due))).toLocaleDateString()}.` : 'Complete a module to enrol it.'}</div>`));
    return;
  }
  for (const item of due) {
    const meta = moduleById(item.moduleId);
    if (!meta) continue;
    const { def } = await getModule(item.moduleId);
    const qs = (def.review && def.review.length ? def.review : def.recall) || [];
    const card = h(`<div class="card review-card"><h3 style="margin-top:0">${esc(meta.title)} <span class="badge">box ${item.box + 1}</span></h3></div>`);
    let answered = 0, correct = 0;
    for (const q of qs) {
      const qEl = h(`<div class="quiz-q"><div class="q">${esc(q.q)}</div></div>`);
      q.options.forEach((opt, oi) => {
        const b = h(`<button class="quiz-opt" type="button">${esc(opt)}</button>`);
        b.addEventListener('click', () => {
          if (qEl.dataset.done) return;
          qEl.dataset.done = '1';
          answered++;
          if (oi === q.answer) { correct++; b.classList.add('correct'); } else { b.classList.add('wrong'); qEl.querySelectorAll('.quiz-opt')[q.answer].classList.add('correct'); }
          qEl.appendChild(h(`<div class="quiz-why">${md(q.why || '')}</div>`));
          if (answered === qs.length) {
            const ok = correct >= Math.ceil(qs.length * 0.75);
            const r = scheduleReview(item.moduleId, ok);
            card.appendChild(h(`<div class="status-line ${ok ? 'ok' : 'bad'}">${correct}/${qs.length} correct. ${ok ? 'Moved up' : 'Back'} to box ${r.box + 1}; next review in ${LEITNER_DAYS[r.box]} day${LEITNER_DAYS[r.box] > 1 ? 's' : ''}. <a href="#/m/${item.moduleId}">Revisit the module</a></div>`));
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

async function renderModulePage(id, phaseArg) {
  const meta = moduleById(id);
  if (!meta) { $app.innerHTML = '<p>Unknown module.</p>'; return; }
  $app.innerHTML = '';
  $app.appendChild(sidebarButton());
  if (meta.status !== 'ready') {
    $app.appendChild(h(`<div><h1>${esc(meta.title)}</h1><p class="muted">This module is planned but not yet written. Goal: ${esc(meta.goal)}</p></div>`));
    return;
  }
  let entry;
  try { entry = await getModule(id); } catch (err) { $app.appendChild(h(`<div class="card">Could not load module: ${esc(err.message)}</div>`)); return; }
  const { def, starter } = entry;
  const state = store.module(id);
  store.update(id, {});
  const track = TRACKS.find((t) => t.id === meta.track);
  const hasRecall = def.recall && def.recall.length;
  let phase = phaseArg || state.tab || (hasRecall ? 'recall' : 'concept');
  if (phase === 'recall' && !hasRecall) phase = 'concept';
  const missingPrereqs = (def.prereqs || []).filter((p) => !isComplete(p));

  const head = h(`<div class="mod-head">
    <div class="crumbs"><a href="#/">Lab</a> › ${esc(track ? track.title : meta.track)} › Module ${id.slice(0, 2)} · about ${meta.minutes} min</div>
    <h1>${esc(def.title)}</h1>
    ${missingPrereqs.length ? `<div class="badge warn">Recommended first: ${missingPrereqs.map((p) => `<a href="#/m/${p}">${esc(moduleById(p)?.title || p)}</a>`).join(', ')}</div>` : ''}
    <div class="goal-banner"><div class="label">Working goal</div><div class="goal">${esc(def.goal)}</div><div class="threshold"><b>Threshold concept:</b> ${esc(def.threshold || '')}</div></div>
  </div>`);
  $app.appendChild(head);

  const phasesEl = h(`<div class="phases"></div>`);
  const phaseDone = {
    recall: Object.keys(state.recall).length >= (def.recall || []).length && hasRecall,
    concept: !!state.conceptRead,
    build: stepsAllDone(def, state),
    goal: !!state.demoDone,
    reflect: !!state.completedAt,
  };
  for (const p of PHASES) {
    if (p.id === 'recall' && !hasRecall) continue;
    const b = h(`<div class="phase ${p.id === phase ? 'active' : ''}" data-phase="${p.id}">${phaseDone[p.id] ? '<span class="tick">✓</span>' : ''}${p.label}</div>`);
    b.addEventListener('click', () => { location.hash = `#/m/${id}/${p.id}`; });
    phasesEl.appendChild(b);
  }
  $app.appendChild(phasesEl);
  store.update(id, { tab: phase });
  const body = h(`<div class="phase-body"></div>`);
  $app.appendChild(body);
  if (phase === 'recall') renderRecall(body, id, def);
  else if (phase === 'concept') renderConcept(body, id, def);
  else if (phase === 'build') renderBuild(body, id, def, starter);
  else if (phase === 'goal') renderGoal(body, id, def, starter);
  else if (phase === 'reflect') renderReflect(body, id, def);
}

function renderRecall(body, id, def) {
  const state = store.module(id);
  body.appendChild(h(`<p class="muted" style="max-width:720px">Before building, pull a few things back out of memory. These questions are about <em>earlier</em> modules; getting one wrong is useful information, not a penalty.</p>`));
  const wrap = h(`<div class="card" style="max-width:760px"></div>`);
  let answered = Object.keys(state.recall).length;
  def.recall.forEach((q, qi) => {
    const qEl = h(`<div class="quiz-q"><div class="q">${qi + 1}. ${esc(q.q)}</div></div>`);
    const prev = state.recall[qi];
    q.options.forEach((opt, oi) => {
      const b = h(`<button class="quiz-opt" type="button">${esc(opt)}</button>`);
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
        if (answered >= def.recall.length) showNext();
      });
      qEl.appendChild(b);
    });
    if (prev !== undefined) { qEl.dataset.done = '1'; qEl.appendChild(h(`<div class="quiz-why">${md(q.why || '')}</div>`)); }
    wrap.appendChild(qEl);
  });
  body.appendChild(wrap);
  const nextRow = h(`<div class="row" style="margin-top:14px"><a class="btn btn-primary" href="#/m/${id}/concept">Continue to Concept →</a><a class="btn btn-ghost" href="#/m/${id}/concept">Skip recall</a></div>`);
  function showNext() {
    const correct = def.recall.filter((q, qi) => state.recall[qi] === q.answer).length;
    wrap.appendChild(h(`<div class="status-line ${correct === def.recall.length ? 'ok' : ''}">${correct}/${def.recall.length} correct.</div>`));
    renderSidebar(id);
  }
  if (answered >= def.recall.length) showNext();
  body.appendChild(nextRow);
}

function renderConcept(body, id, def) {
  const state = store.module(id);
  const el = h(`<div class="concept card">${md(def.concept, { predictKey: (i) => `concept-${i}` })}</div>`);
  activatePredicts(el, { getSaved: (k) => state.predictions[k], setSaved: (k, v) => { state.predictions[k] = v; store.update(id, { predictions: state.predictions }); } });
  body.appendChild(el);
  const row = h(`<div class="row" style="margin-top:14px"><a class="btn btn-primary" href="#/m/${id}/build">I'm ready to build →</a></div>`);
  row.firstElementChild.addEventListener('click', () => store.update(id, { conceptRead: true }));
  body.appendChild(row);
}

function renderBuild(body, id, def, starter) {
  const state = store.module(id);
  const timeouts = Object.assign({ tests: 20000, demo: 120000 }, def.timeouts || {});
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
        <div class="row" style="margin-top:10px">
          <button class="btn btn-primary" id="btn-check" type="button">Check this step (Ctrl+Enter)</button>
          <button class="btn" id="btn-check-all" type="button">Check all steps</button>
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

  const save = debounce((code) => { store.update(id, { code }); $saveState.textContent = 'saved'; }, 400);
  const editor = createEditor(layout.querySelector('.editor-host'), {
    value: state.code ?? starter,
    onChange: (v) => { $saveState.textContent = 'saving…'; save(v); },
    onRun: () => check(def.steps[stepIdx].id),
  });

  function renderNav() {
    $nav.innerHTML = '';
    def.steps.forEach((s, i) => {
      const chip = h(`<button class="step-chip ${i === stepIdx ? 'active' : ''} ${state.stepsDone[s.id] ? 'done' : ''}" type="button">${state.stepsDone[s.id] ? '✓ ' : ''}${i + 1}. ${esc(s.title)}</button>`);
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
      const hint = h(`<div class="hint"><button type="button" ${unlocked ? '' : 'disabled'}>Hint ${hi + 1} of ${s.hints.length}${['', ' · a nudge', ' · the strategy', ' · nearly the code'][hi + 1] || ''}</button><div class="hint-body ${revealed ? '' : 'hidden'}">${md(text)}</div></div>`);
      hint.querySelector('button').addEventListener('click', () => {
        hint.querySelector('.hint-body').classList.remove('hidden');
        state.hintsOpen = state.hintsOpen || {};
        state.hintsOpen[s.id] = Math.max(state.hintsOpen[s.id] || 0, hi + 1);
        store.update(id, { hintsOpen: state.hintsOpen });
        renderPanel();
      });
      hints.appendChild(hint);
    });
    $panel.appendChild(hints);
    if (state.stepsDone[s.id]) $panel.appendChild(h(`<div class="status-line ok">✓ This step's tests pass.${stepIdx + 1 < def.steps.length ? ` <a href="#" data-next>Next step →</a>` : ` <a href="#/m/${id}/goal">Run the goal →</a>`}</div>`));
    const nx = $panel.querySelector('[data-next]');
    if (nx) nx.addEventListener('click', (e) => { e.preventDefault(); stepIdx++; store.update(id, { step: stepIdx }); renderNav(); renderPanel(); });
  }

  function logLine(msg) {
    $console.classList.remove('hidden');
    const d = document.createElement('div');
    d.className = `log-line ${msg.level === 'error' ? 'log-error' : msg.level === 'warn' ? 'log-warn' : ''}`;
    d.textContent = msg.text;
    $console.appendChild(d);
    $console.scrollTop = $console.scrollHeight;
  }

  async function check(stepId) {
    const code = editor.getValue();
    store.update(id, { code });
    $results.innerHTML = '';
    $console.innerHTML = '';
    $console.classList.add('hidden');
    $runState.textContent = 'running…';
    layout.querySelector('#btn-check').disabled = true;
    layout.querySelector('#btn-check-all').disabled = true;
    const byStep = {};
    let res;
    try {
      res = await runner.run({ mode: 'tests', code, moduleId: id, stepId, timeout: timeouts.tests, onMessage: (msg) => {
        if (msg.type === 'log') logLine(msg);
        if (msg.type === 'test') {
          (byStep[msg.step] = byStep[msg.step] || []).push(msg);
          const stepTitle = def.steps.find((s) => s.id === msg.step)?.title || msg.step;
          const el = h(`<div class="test ${msg.pass ? 'pass' : 'fail'}"><span class="mark">${msg.pass ? '✓' : '✗'}</span><div><div>${stepId ? '' : `<span class="muted">${esc(stepTitle)} · </span>`}${esc(msg.name)}</div>${msg.pass ? '' : `<div class="msg">${esc(msg.message)}</div>`}</div><span class="ms">${msg.ms.toFixed(0)} ms</span></div>`);
          $results.appendChild(el);
        }
        if (msg.type === 'error') $results.appendChild(h(`<div class="status-line bad">Error: ${esc(msg.message)}</div>`));
      } });
    } catch (err) {
      if (err.message !== 'superseded') $results.appendChild(h(`<div class="status-line bad">${esc(err.message)}</div>`));
      return;
    } finally {
      layout.querySelector('#btn-check').disabled = false;
      layout.querySelector('#btn-check-all').disabled = false;
      $runState.textContent = '';
    }
    state.attempts = typeof state.attempts === 'object' && state.attempts ? state.attempts : {};
    const touched = stepId ? [stepId] : def.steps.map((s) => s.id);
    for (const sid of touched) {
      state.attempts[sid] = (state.attempts[sid] || 0) + 1;
      const list = byStep[sid] || [];
      const allPass = list.length > 0 && list.every((t) => t.pass);
      if (allPass) state.stepsDone[sid] = true; else delete state.stepsDone[sid];
    }
    store.update(id, { attempts: state.attempts, stepsDone: state.stepsDone });
    if (res.summary) {
      const ok = res.summary.failed === 0 && res.summary.total > 0;
      $results.prepend(h(`<div class="status-line ${ok ? 'ok' : 'bad'}">${res.summary.passed}/${res.summary.total} tests passed${ok && stepsAllDone(def, state) ? ' — all steps done. Head to the Goal tab.' : ''}</div>`));
    }
    renderNav();
    renderPanel();
    renderSidebar(id);
  }

  layout.querySelector('#btn-check').addEventListener('click', () => check(def.steps[stepIdx].id));
  layout.querySelector('#btn-check-all').addEventListener('click', () => check(null));
  layout.querySelector('#btn-reset').addEventListener('click', () => { if (confirm('Replace your code with the starter file?')) { editor.setValue(starter); store.update(id, { code: starter }); } });
  layout.querySelector('#btn-download').addEventListener('click', () => download(`${id}.js`, editor.getValue()));
  layout.querySelector('#btn-solution').addEventListener('click', async () => {
    const attempts = Object.values((state.attempts && typeof state.attempts === 'object') ? state.attempts : {}).reduce((a, b) => a + b, 0);
    if (attempts < 2 && !confirm('You have not tried the tests twice yet. Looking at the reference now will cost you most of the learning. Show it anyway?')) return;
    const sol = await fetchSolution(id);
    const wrap = h(`<div class="card" style="margin-top:12px"><div class="row"><b>Reference solution</b><span class="spacer"></span><button class="btn btn-small" id="sol-copy" type="button">Load into editor</button><button class="btn btn-small" id="sol-close" type="button">Close</button></div><pre class="code"><code>${esc(sol)}</code></pre></div>`);
    wrap.querySelector('#sol-close').addEventListener('click', () => wrap.remove());
    wrap.querySelector('#sol-copy').addEventListener('click', () => { if (confirm('Replace your code with the reference? Your version will be lost.')) { editor.setValue(sol); store.update(id, { code: sol }); } });
    $results.before(wrap);
  });
  renderNav();
  renderPanel();
  setTimeout(() => editor.refresh(), 0);
}

function renderGoal(body, id, def, starter) {
  const state = store.module(id);
  const timeouts = Object.assign({ tests: 20000, demo: 120000 }, def.timeouts || {});
  const ready = stepsAllDone(def, state);
  const wrap = h(`<div>
    <div class="card" style="max-width:820px">
      <h2 style="margin-top:0">Run the goal</h2>
      <p>${esc(def.goal)}</p>
      <p class="muted small">The demo runs <em>your</em> code from the Build tab. ${ready ? 'All steps pass.' : 'Not all steps pass yet; the demo may fail or show odd results, which is itself informative.'}</p>
      <div class="row"><button class="btn btn-primary" id="btn-run" type="button">Run the goal demo</button><button class="btn" id="btn-stop" type="button" disabled>Stop</button><span class="muted small" id="goal-state"></span></div>
      ${state.demoDone ? `<div class="done-banner"><b>Working goal achieved</b> (last run)<div>${md(state.demoSummary || '')}</div></div>` : ''}
    </div>
    <div class="goal-out"></div>
  </div>`);
  body.appendChild(wrap);
  const $out = wrap.querySelector('.goal-out');
  const $state = wrap.querySelector('#goal-state');
  const $run = wrap.querySelector('#btn-run');
  const $stop = wrap.querySelector('#btn-stop');
  let progressEl = null, consoleEl = null;
  function consoleLine(text, level) {
    if (!consoleEl) { consoleEl = h(`<div class="console"></div>`); $out.appendChild(consoleEl); }
    const d = document.createElement('div');
    d.className = `log-line ${level === 'error' ? 'log-error' : level === 'warn' ? 'log-warn' : ''}`;
    d.textContent = text;
    consoleEl.appendChild(d);
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }
  $stop.addEventListener('click', () => runner.cancel('stopped'));
  $run.addEventListener('click', async () => {
    $out.innerHTML = '';
    consoleEl = null; progressEl = null;
    $run.disabled = true; $stop.disabled = false;
    $state.textContent = 'running…';
    const code = state.code ?? starter;
    try {
      const res = await runner.run({ mode: 'demo', code, moduleId: id, timeout: timeouts.demo, onMessage: (msg) => {
        if (msg.type === 'log') consoleLine(msg.text, msg.level);
        else if (msg.type === 'md') { const d = h(`<div class="card md">${md(msg.markdown)}</div>`); $out.appendChild(d); }
        else if (msg.type === 'plot' || msg.type === 'bar' || msg.type === 'heatmap' || msg.type === 'table') { consoleEl = null; renderChart($out, msg.type, msg.spec); }
        else if (msg.type === 'progress') {
          if (!progressEl) { progressEl = h(`<div class="progress-row"><div class="progress-bar"><i style="width:0%"></i></div><span class="plabel"></span></div>`); $out.appendChild(progressEl); }
          progressEl.querySelector('i').style.width = `${Math.round(100 * Math.max(0, Math.min(1, msg.fraction)))}%`;
          progressEl.querySelector('.plabel').textContent = msg.label || '';
        }
        else if (msg.type === 'demo-done') {
          store.update(id, { demoDone: true, demoSummary: msg.summary });
          $out.appendChild(h(`<div class="done-banner"><b>Working goal achieved.</b><div>${md(msg.summary)}</div><div style="margin-top:8px"><a class="btn btn-primary btn-small" href="#/m/${id}/reflect">Continue to Reflect →</a></div></div>`));
          renderSidebar(id);
        }
        else if (msg.type === 'error') $out.appendChild(h(`<div class="status-line bad">Error: ${esc(msg.message)}${msg.stack ? `<pre class="code small">${esc(String(msg.stack).split('\n').slice(0, 6).join('\n'))}</pre>` : ''}</div>`));
      } });
      $state.textContent = res.error ? 'finished with errors' : 'finished';
    } catch (err) {
      $state.textContent = err.message === 'stopped' ? 'stopped' : err.message;
    } finally {
      $run.disabled = false; $stop.disabled = true;
    }
  });
}

function renderReflect(body, id, def) {
  const state = store.module(id);
  const ready = stepsAllDone(def, state) && state.demoDone;
  const wrap = h(`<div class="reflect">
    <div class="card" style="max-width:820px">
      <h2 style="margin-top:0">Explain it in your own words</h2>
      <p class="muted small">Write as if teaching a colleague who has not seen the module. Naming the thing you are least sure about is the most useful sentence you can write.</p>
      <div class="prompts"></div>
    </div>
    <div class="card" style="max-width:820px">
      <h3 style="margin-top:0">Stretch goals (optional)</h3>
      <ul>${(def.stretch || []).map((s) => `<li>${md(s).replace(/^<p>|<\/p>$/g, '')}</li>`).join('')}</ul>
    </div>
    <div class="card" style="max-width:820px">
      <h3 style="margin-top:0">Complete the module</h3>
      <ul class="checklist">
        <li class="${stepsAllDone(def, state) ? 'ok' : 'no'}">${stepsAllDone(def, state) ? '✓' : '○'} All build steps pass</li>
        <li class="${state.demoDone ? 'ok' : 'no'}">${state.demoDone ? '✓' : '○'} Goal demo ran on your code</li>
        <li class="no" id="refl-check">○ At least one reflection written</li>
      </ul>
      <div class="row"><button class="btn btn-primary" id="btn-complete" type="button" ${ready ? '' : 'disabled'}>${state.completedAt ? 'Completed ✓' : 'Mark module complete'}</button>${state.completedAt ? `<span class="muted small">Completed ${new Date(state.completedAt).toLocaleDateString()}. Enrolled in the review queue.</span>` : ''}</div>
    </div>
  </div>`);
  body.appendChild(wrap);
  const $prompts = wrap.querySelector('.prompts');
  const $reflCheck = wrap.querySelector('#refl-check');
  const $complete = wrap.querySelector('#btn-complete');
  function updateChecklist() {
    const any = Object.values(state.reflections).some((t) => t && t.trim().length > 20);
    $reflCheck.className = any ? 'ok' : 'no';
    $reflCheck.textContent = `${any ? '✓' : '○'} At least one reflection written (20+ characters)`;
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
  $complete.addEventListener('click', () => {
    store.update(id, { completedAt: Date.now() });
    enrollReview(id);
    renderSidebar(id);
    const nx = nextModule(id);
    $complete.textContent = 'Completed ✓';
    $complete.disabled = true;
    $complete.parentElement.appendChild(h(`<span>Enrolled in the review queue. ${nx ? `<a class="btn btn-small" href="#/m/${nx.id}">Next: ${esc(nx.title)} →</a>` : 'That was the last module.'}</span>`));
  });
}


// ---------- chat playground ----------

let chatWorker = null;

function renderChat() {
  if (chatWorker) { chatWorker.terminate(); chatWorker = null; }
  $app.innerHTML = '';
  $app.appendChild(sidebarButton());
  const wrap = h(`<div>
    <h1>Chat playground</h1>
    <p class="muted" style="max-width:760px">Talk to the lab's own model, running in this page: the BPE tokenizer (module 03), the GPT (module 06) trained by
    <code>tools/pretrain.mjs</code> (module 07), decoded through a KV cache that is reused across turns (modules 15 and 17) and the sampling pipeline (module 14),
    wrapped in the chat template from module 10. It is a ~100k-parameter model trained on a toy corpus, so expect corpus-like text, not answers. The point is that
    every piece of it is something you built.</p>
    <div class="card" style="max-width:860px">
      <div class="row small" id="chat-info"><span class="muted">Loading model…</span></div>
      <div class="row" style="margin-top:8px">
        <label class="small">temperature <input type="number" id="chat-temp" value="0.8" min="0" max="3" step="0.1" style="width:64px"></label>
        <label class="small">top-p <input type="number" id="chat-topp" value="0.95" min="0" max="1" step="0.05" style="width:64px"></label>
        <label class="small">top-k <input type="number" id="chat-topk" value="0" min="0" step="1" style="width:64px"></label>
        <label class="small">max tokens <input type="number" id="chat-max" value="40" min="1" max="200" step="1" style="width:64px"></label>
        <span class="spacer"></span>
        <button class="btn btn-small" id="chat-load" type="button">Load checkpoint JSON…</button>
        <button class="btn btn-small" id="chat-reset" type="button">New conversation</button>
      </div>
    </div>
    <div class="card chat-log" id="chat-log" style="max-width:860px;min-height:160px"></div>
    <div class="card" style="max-width:860px">
      <div class="row"><input type="text" id="chat-input" placeholder="Say something to the model… (the toy corpus is about cats, dogs, and the weather)" style="flex:1;font:inherit;padding:8px 10px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--ink)">
      <button class="btn btn-primary" id="chat-send" type="button" disabled>Send</button><button class="btn" id="chat-stop" type="button" disabled>Stop</button></div>
      <div class="small muted" id="chat-stats" style="margin-top:6px"></div>
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
      else if (m.type === 'token') { if (current) { current.textContent += m.text; $log.scrollTop = $log.scrollHeight; } }
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

function route() {
  runner.cancel('navigated');
  if (chatWorker) { chatWorker.terminate(); chatWorker = null; }
  $sidebar.classList.remove('open');
  const hash = location.hash || '#/';
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  window.scrollTo(0, 0);
  if (parts[0] === 'm' && parts[1]) { renderSidebar(parts[1]); renderModulePage(parts[1], parts[2] || null); return; }
  renderSidebar(null);
  if (parts[0] === 'review') return renderReview();
  if (parts[0] === 'about') return renderAbout();
  if (parts[0] === 'chat') return renderChat();
  renderHome();
}

try { const t = localStorage.getItem('btu:theme'); if (t) document.documentElement.dataset.theme = t; } catch { /* ignore */ }
window.addEventListener('hashchange', route);
route();
