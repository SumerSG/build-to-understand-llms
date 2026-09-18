// app/storage.js — progress persistence and the spaced-review scheduler (localStorage).

const KEY = 'btu:v1';
const DAY = 24 * 60 * 60 * 1000;
export const LEITNER_DAYS = [1, 3, 7, 14, 30];

function blank() {
  return { modules: {}, review: {}, lastModule: null, createdAt: Date.now() };
}

let data = null;

function read() {
  if (data) return data;
  try {
    const raw = localStorage.getItem(KEY);
    data = raw ? JSON.parse(raw) : blank();
  } catch {
    data = blank();
  }
  return data;
}

function write() {
  try { localStorage.setItem(KEY, JSON.stringify(data)); } catch { /* private mode etc. */ }
}

export const store = {
  all() { return read(); },
  module(id) {
    const d = read();
    if (!d.modules[id]) {
      d.modules[id] = { code: null, tab: 'recall', step: 0, passed: {}, stepsDone: {}, demoDone: false,
        demoSummary: '', reflections: {}, predictions: {}, recall: {}, hintsOpen: {}, attempts: 0, completedAt: null };
    }
    return d.modules[id];
  },
  update(id, patch) {
    const m = this.module(id);
    Object.assign(m, patch);
    read().lastModule = id;
    write();
    return m;
  },
  save() { write(); },
  reset() { data = blank(); write(); },
  export() { return JSON.stringify(read(), null, 2); },
  import(json) { data = JSON.parse(json); write(); },
};

/** Record a review outcome for a module and schedule the next one (Leitner boxes). */
export function scheduleReview(moduleId, correct, now = Date.now()) {
  const d = read();
  const r = d.review[moduleId] || { box: 0, due: now, history: [] };
  r.box = correct ? Math.min(r.box + 1, LEITNER_DAYS.length - 1) : 0;
  r.due = now + LEITNER_DAYS[r.box] * DAY;
  r.history.push({ at: now, correct });
  if (r.history.length > 50) r.history.shift();
  d.review[moduleId] = r;
  write();
  return r;
}

/** Called when a module is completed: put it in the review queue (first review tomorrow). */
export function enrollReview(moduleId, now = Date.now()) {
  const d = read();
  if (!d.review[moduleId]) {
    d.review[moduleId] = { box: 0, due: now + LEITNER_DAYS[0] * DAY, history: [] };
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
