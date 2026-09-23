// app/runner.js — main-thread controller for the sandbox worker.

export class Runner {
  constructor({ base }) {
    this.base = base;          // absolute URL of the repo root, no trailing slash
    this.worker = null;
    this.timer = null;
    this.runId = 0;
    this._reject = null;
  }

  get busy() { return this.worker !== null; }

  cancel(reason = 'cancelled') {
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
      const rej = this._reject;
      this._reject = null;
      if (rej) rej(new Error(reason));
    }
  }

  /**
   * @param {object} o
   * @param {'tests'|'demo'} o.mode
   * @param {string} o.code       learner source
   * @param {string} o.moduleId
   * @param {string} [o.stepId]
   * @param {number} [o.timeout]  ms
   * @param {(msg: object) => void} o.onMessage
   */
  run({ mode, code, moduleId, stepId = null, timeout = 20000, onMessage }) {
    this.cancel('superseded');
    const runId = ++this.runId;
    const worker = new Worker(new URL('./sandbox-worker.js', import.meta.url), { type: 'module' });
    this.worker = worker;
    const testsUrl = `${this.base}/modules/${moduleId}/tests.js`;
    const demoUrl = `${this.base}/modules/${moduleId}/demo.js`;
    return new Promise((resolve, reject) => {
      this._reject = reject;
      const results = { tests: [], summary: null, error: null, demoSummary: null };
      const finish = () => {
        if (this.timer) { clearTimeout(this.timer); this.timer = null; }
        if (this.worker === worker) { worker.terminate(); this.worker = null; }
        this._reject = null;
        resolve(results);
      };
      worker.onmessage = (e) => {
        const msg = e.data;
        if (msg.runId !== runId) return;
        if (msg.type === 'test') results.tests.push(msg);
        if (msg.type === 'tests-done') results.summary = msg;
        if (msg.type === 'error') results.error = msg;
        if (msg.type === 'demo-done') results.demoSummary = msg.summary;
        onMessage(msg);
        if (msg.type === 'tests-done' || msg.type === 'run-finished') finish();
      };
      worker.onerror = (e) => {
        results.error = { message: e.message || 'The code runner stopped with an error it could not describe (often a syntax error in your code: check the brackets and quotes near your last edit).' };
        onMessage({ type: 'error', message: results.error.message });
        finish();
      };
      this.timer = setTimeout(() => {
        results.error = { message: `Timed out after ${Math.round(timeout / 1000)} s: your code was still running. Usually a loop never ends (a while loop whose condition never becomes false, or a for loop whose counter never reaches its limit, e.g. i-- where i++ was meant); otherwise it does far more work than needed.` };
        onMessage({ type: 'error', message: results.error.message });
        finish();
      }, timeout);
      worker.postMessage({ type: 'run', runId, mode, code, base: this.base, testsUrl, demoUrl, stepId });
    });
  }
}
