// app/storage.js — progress persistence and the spaced-review scheduler (localStorage).
//
// One blob under KEY holds every module's progress. Writes merge at the module level against what is
// on disk, so a second tab (Concept in one, Build in another) never overwrites work saved by the other.

const KEY = 'btu:v1';
const VERSION = 1;
const DAY = 24 * 60 * 60 * 1000;
export const LEITNER_DAYS = [1, 3, 7, 14, 30];

function blank() {
  return { version: VERSION, modules: {}, review: {}, lastModule: null, createdAt: Date.now(), updatedAt: 0 };
}

function blankModule() {
  return { code: null, tab: 'recall', step: 0, passed: {}, stepsDone: {}, demoDone: false, demoSummary: '',
    reflections: {}, predictions: {}, recall: {}, hintsOpen: {}, attempts: {}, completedAt: null, conceptRead: false };
}

const isObj = (x) => !!x && typeof x === 'object' && !Array.isArray(x);

/** Coerce anything JSON.parse produced into a valid progress object, or return null if it is not one. */
function normalise(obj) {
  if (!isObj(obj) || !isObj(obj.modules)) return null;
  if (obj.version !== undefined && obj.version !== VERSION) return null;
  const d = blank();
  d.createdAt = typeof obj.createdAt === 'number' ? obj.createdAt : d.createdAt;
  d.updatedAt = typeof obj.updatedAt === 'number' ? obj.updatedAt : 0;
  d.lastModule = typeof obj.lastModule === 'string' ? obj.lastModule : null;
  for (const [id, m] of Object.entries(obj.modules)) {
    if (!isObj(m)) continue;
    const shape = blankModule();
    const mod = Object.assign(blankModule(), m);
    for (const k of Object.keys(shape)) if (isObj(shape[k]) && !isObj(mod[k])) mod[k] = {};   // e.g. a hand-edited stepsDone: null
    if (typeof mod.code !== 'string') mod.code = null;
    d.modules[id] = mod;
  }
  if (isObj(obj.review)) {
    for (const [id, r] of Object.entries(obj.review)) {
      if (!isObj(r)) continue;
      const box = Math.max(0, Math.min(LEITNER_DAYS.length - 1, Number.isInteger(r.box) ? r.box : 0));
      d.review[id] = { box, due: typeof r.due === 'number' ? r.due : Date.now(), history: Array.isArray(r.history) ? r.history : [] };
    }
  }
  return d;
}

let data = null;
let lastWriteOk = true;
const dirty = { modules: new Set(), review: new Set() };
const listeners = new Set();

function readDisk() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? normalise(JSON.parse(raw)) : null;
  } catch {
    return null;   // no storage, or a corrupt blob: never brick the app
  }
}

function read() {
  if (data) return data;
  data = readDisk() || blank();
  return data;
}

/** Pull another tab's changes into memory: entries this tab has not touched take the on-disk version. */
function mergeFromDisk(disk) {
  if (!disk) return;
  for (const [id, m] of Object.entries(disk.modules)) {
    if (dirty.modules.has(id)) continue;
    if (data.modules[id]) Object.assign(data.modules[id], m);   // keep object identity for UI references
    else data.modules[id] = m;
  }
  for (const [id, r] of Object.entries(disk.review)) if (!dirty.review.has(id)) data.review[id] = r;
}

/** Persist. Returns false when the browser refused (quota, private mode, storage disabled). */
function write({ replace = false } = {}) {
  if (!replace) mergeFromDisk(readDisk());
  dirty.modules.clear(); dirty.review.clear();
  data.updatedAt = Date.now();
  try {
    localStorage.setItem(KEY, JSON.stringify(data));
    lastWriteOk = true;
  } catch {
    lastWriteOk = false;
  }
  return lastWriteOk;
}

export const store = {
  all() { return read(); },
  /** Whether the most recent write reached localStorage. */
  get lastWriteOk() { return lastWriteOk; },
  module(id) {
    const d = read();
    if (!d.modules[id]) d.modules[id] = blankModule();
    return d.modules[id];
  },
  update(id, patch) {
    const m = this.module(id);
    Object.assign(m, patch);
    read().lastModule = id;
    dirty.modules.add(id);
    write();
    return m;
  },
  save() { write(); },
  reset() { data = blank(); write({ replace: true }); },
  export() { return JSON.stringify(read(), null, 2); },
  /** Replace all progress with an exported file. Throws (and changes nothing) if it is not a progress file. */
  import(json) {
    let obj;
    try { obj = JSON.parse(json); } catch (e) { throw new Error('the file is not valid JSON'); }
    const next = normalise(obj);
    if (!next) throw new Error(isObj(obj) && obj.version !== undefined && obj.version !== VERSION ? `unsupported progress-file version ${obj.version}` : 'not a Build-to-Understand progress file (no "modules" object)');
    data = next;
    if (!write({ replace: true })) throw new Error('imported, but the browser refused to save it (storage unavailable)');
  },
  /** Called when another tab changed the saved progress; fn receives nothing. */
  onExternalChange(fn) { listeners.add(fn); return () => listeners.delete(fn); },
};

if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
  window.addEventListener('storage', (e) => {
    if (e.key !== KEY || !data) return;
    mergeFromDisk(readDisk());
    for (const fn of listeners) { try { fn(); } catch { /* ignore */ } }
  });
}

/** Record a review outcome for a module and schedule the next one (Leitner boxes). */
export function scheduleReview(moduleId, correct, now = Date.now()) {
  const d = read();
  const r = d.review[moduleId] || { box: 0, due: now, history: [] };
  r.box = correct ? Math.min(r.box + 1, LEITNER_DAYS.length - 1) : 0;
  r.due = now + LEITNER_DAYS[r.box] * DAY;
  r.history.push({ at: now, correct });
  if (r.history.length > 50) r.history.shift();
  d.review[moduleId] = r;
  dirty.review.add(moduleId);
  write();
  return r;
}

/** Called when a module is completed: put it in the review queue (first review tomorrow). */
export function enrollReview(moduleId, now = Date.now()) {
  const d = read();
  if (!d.review[moduleId]) {
    d.review[moduleId] = { box: 0, due: now + LEITNER_DAYS[0] * DAY, history: [] };
    dirty.review.add(moduleId);
    write();
  }
}

export function dueReviews(now = Date.now()) {
  const d = read();
  return Object.entries(d.review)
    .filter(([, r]) => r.due <= now)
    .map(([moduleId, r]) => ({ moduleId, ...r }))
    .sort((a, b) => a.due - b.due);
}

export function reviewSummary() {
  const d = read();
  return Object.entries(d.review).map(([moduleId, r]) => ({ moduleId, ...r }));
}
