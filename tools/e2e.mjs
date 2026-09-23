#!/usr/bin/env node
// tools/e2e.mjs — drive every ready module through the real UI in headless Chromium.
// For each module: open Build, check all steps on the starter (must not all pass), load the
// reference solution, check all steps (must all pass), open Goal, run the demo (must reach the
// done banner). Reports timings and failures. Needs `npm i -D playwright-core` and a Chromium.
//   node tools/e2e.mjs [--module 05-attention] [--headed] [--base http://host/sub/path/]
// --base tests an already-running server, e.g. one that serves the repo under a subfolder the way
// GitHub Pages does; without it the script starts tools/serve.mjs at the site root.
import { spawn } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const only = args.includes('--module') ? args[args.indexOf('--module') + 1] : null;
// a free port, so several copies of this script (e.g. parallel reviewers) never share or kill one server
const PORT = process.env.E2E_PORT ? Number(process.env.E2E_PORT) : await new Promise((resolve) => {
  const srv = net.createServer().listen(0, '127.0.0.1', () => { const { port } = srv.address(); srv.close(() => resolve(port)); });
});
const baseArg = args.includes('--base') ? args[args.indexOf('--base') + 1] : null;

let chromium;
try { ({ chromium } = await import('playwright-core')); }
catch { console.error('playwright-core is not installed: npm i -D playwright-core'); process.exit(2); }

const { MODULES } = await import(pathToFileURL(path.join(ROOT, 'modules/index.js')).href);
const targets = MODULES.filter((m) => m.status === 'ready' && (!only || m.id === only));

const server = baseArg ? null : spawn('node', [path.join(ROOT, 'tools/serve.mjs'), String(PORT)], { stdio: 'ignore' });
if (server) await new Promise((r) => setTimeout(r, 800));
const exe = process.env.CHROMIUM_PATH || (process.env.PLAYWRIGHT_BROWSERS_PATH ? path.join(process.env.PLAYWRIGHT_BROWSERS_PATH, 'chromium') : undefined);
const browser = await chromium.launch({ headless: !args.includes('--headed'), executablePath: exe });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(e.message));
page.on('dialog', (d) => d.accept());
const base = baseArg ? baseArg.replace(/\/?$/, '/') : `http://localhost:${PORT}/`;
await page.goto(base);
await page.evaluate(() => localStorage.clear());

let failed = 0;
for (const m of targets) {
  const t0 = Date.now();
  const row = { id: m.id, starter: '?', solution: '?', demo: '?', ms: 0, error: null };
  try {
    await page.goto(`${base}#/m/${m.id}/build`);
    await page.waitForSelector('.step-chip', { timeout: 20000 });
    await page.click('#btn-check-all');
    await page.waitForSelector('.results .status-line', { timeout: 60000 });
    row.starter = (await page.locator('.results .status-line').first().evaluate((el) => el.textContent)).split(' tests')[0];
    if (/^(\d+)\/\1 /.test(row.starter + ' ')) throw new Error(`starter already passes everything (${row.starter})`);
    await page.click('#btn-solution');
    await page.waitForSelector('#sol-copy', { timeout: 10000 });
    await page.click('#sol-copy');
    await page.click('#btn-check-all');
    await page.waitForFunction(() => { const s = document.querySelector('.results .status-line'); return s && /tests passed/.test(s.textContent); }, null, { timeout: 90000 });
    row.solution = (await page.locator('.results .status-line').first().evaluate((el) => el.textContent)).split(' tests')[0];
    const [p, t] = row.solution.split('/').map(Number);
    if (p !== t) {
      const fails = await page.locator('.test.fail').allInnerTexts();
      throw new Error(`solution fails ${t - p} test(s): ${fails.slice(0, 3).join(' | ').slice(0, 300)}`);
    }
    await page.goto(`${base}#/m/${m.id}/goal`);
    await page.waitForSelector('#btn-run', { timeout: 20000 });
    const td = Date.now();
    await page.click('#btn-run');
    await Promise.race([
      page.waitForSelector('.goal-out .done-banner', { timeout: 180000 }),
      page.waitForSelector('.goal-out .status-line.bad', { timeout: 180000 }).then(async () => { throw new Error('demo error: ' + (await page.locator('.goal-out .status-line.bad').first().innerText()).slice(0, 300)); }),
    ]);
    row.demo = `${((Date.now() - td) / 1000).toFixed(1)}s, ${await page.locator('.goal-out .chart').count()} visual(s)`;
  } catch (e) {
    row.error = e.message.split('\n')[0].slice(0, 400);
    failed++;
  }
  row.ms = Date.now() - t0;
  console.log(`${row.error ? 'FAIL' : 'PASS'}  ${row.id}  starter ${row.starter}, solution ${row.solution}, demo ${row.demo}  (${(row.ms / 1000).toFixed(0)}s)${row.error ? '\n      ✗ ' + row.error : ''}`);
}
if (pageErrors.length) console.log(`page errors: ${[...new Set(pageErrors)].slice(0, 5).join(' | ')}`);
console.log(`\n${targets.length - failed}/${targets.length} modules pass end to end`);
await browser.close();
server?.kill();
process.exit(failed ? 1 : 0);
