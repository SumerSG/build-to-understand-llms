// app/charts.js — small SVG charts for the Goal demo output: line plot, bar, heatmap, table.
// Colours come from CSS variables (--chart-1 … --chart-8, --ink, --muted, --grid) so the charts
// follow light/dark mode. Every chart ships a hover tooltip and a table view toggle.

const SVG = 'http://www.w3.org/2000/svg';
const W = 640, H = 300, PAD = { l: 52, r: 16, t: 28, b: 40 };

function el(tag, attrs = {}, parent = null) {
  const n = document.createElementNS(SVG, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}

function fmtNum(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
  if (Number.isInteger(v)) return String(v);
  return a < 1 ? v.toFixed(4) : a < 100 ? v.toFixed(3) : v.toFixed(1);
}

function niceTicks(min, max, n = 5) {
  if (!(max > min)) { max = min + 1; }
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = (span / n) / step0;
  const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function frame(container, title) {
  const wrap = document.createElement('div');
  wrap.className = 'chart';
  const head = document.createElement('div');
  head.className = 'chart-head';
  head.innerHTML = `<span class="chart-title"></span><button class="chart-toggle" type="button">Table</button>`;
  head.querySelector('.chart-title').textContent = title || '';
  wrap.appendChild(head);
  const body = document.createElement('div');
  body.className = 'chart-body';
  wrap.appendChild(body);
  const table = document.createElement('div');
  table.className = 'chart-table hidden';
  wrap.appendChild(table);
  const tip = document.createElement('div');
  tip.className = 'chart-tip hidden';
  wrap.appendChild(tip);
  head.querySelector('.chart-toggle').addEventListener('click', (e) => {
    const showTable = table.classList.toggle('hidden') === false;
    body.classList.toggle('hidden', showTable);
    e.target.textContent = showTable ? 'Chart' : 'Table';
  });
  container.appendChild(wrap);
  return { wrap, body, table, tip };
}

function tableHtml(columns, rows) {
  return `<table><thead><tr>${columns.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows
    .map((r) => `<tr>${r.map((v) => `<td>${esc(typeof v === 'number' ? fmtNum(v) : v)}</td>`).join('')}</tr>`)
    .join('')}</tbody></table>`;
}

function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }

// A diverging loss or a 1/0 in learner code produces NaN/Infinity; say so instead of drawing a silent gap.
function noteSkipped(f, n) {
  if (!n) return;
  const d = document.createElement('div');
  d.className = 'chart-note small muted';
  d.textContent = `${n} value${n > 1 ? 's were' : ' was'} NaN or ±Infinity and ${n > 1 ? 'are' : 'is'} not drawn.`;
  f.body.appendChild(d);
}

function showTip(f, x, y, html) {
  f.tip.innerHTML = html;
  f.tip.classList.remove('hidden');
  const r = f.wrap.getBoundingClientRect();
  f.tip.style.left = Math.min(x - r.left + 12, r.width - f.tip.offsetWidth - 8) + 'px';
  f.tip.style.top = Math.max(0, y - r.top - f.tip.offsetHeight - 12) + 'px';
}

/** Line plot. spec: { title, x?, series: [{ name, values }], xlabel?, ylabel?, yscale?: 'log' } */
export function renderPlot(container, spec) {
  const f = frame(container, spec.title);
  const series = (spec.series || []).filter((s) => s && s.values && s.values.length);
  if (!series.length) { f.body.textContent = 'No data.'; return; }
  const log = spec.yscale === 'log';
  const n = Math.max(...series.map((s) => s.values.length));
  const xs = spec.x && spec.x.length ? spec.x : Array.from({ length: n }, (_, i) => i);
  const finiteXs = xs.filter(Number.isFinite);
  const xmin = finiteXs.length ? Math.min(...finiteXs) : 0, xmax = finiteXs.length ? Math.max(...finiteXs) : 1;
  let ymin = Infinity, ymax = -Infinity, skipped = 0;
  const drawable = (v, i) => Number.isFinite(v) && (!log || v > 0) && Number.isFinite(xs[i] ?? i);
  for (const s of series) s.values.forEach((v, i) => { if (drawable(v, i)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); } else skipped++; });
  if (!Number.isFinite(ymin)) { ymin = log ? 1 : 0; ymax = log ? 10 : 1; }
  if (ymin === ymax) { if (log) { ymin /= 10; ymax *= 10; } else { ymin -= 1; ymax += 1; } }
  const ty = (v) => (log ? Math.log10(v) : v);
  const lo = ty(ymin), hi = ty(ymax);
  const sx = (x) => PAD.l + ((x - xmin) / Math.max(1e-12, xmax - xmin)) * (W - PAD.l - PAD.r);
  const sy = (v) => PAD.t + (1 - (ty(v) - lo) / Math.max(1e-12, hi - lo)) * (H - PAD.t - PAD.b);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg', role: 'img' }, f.body);
  const yt = log ? niceTicks(lo, hi, 4).map((t) => Math.pow(10, t)) : niceTicks(ymin, ymax, 5);
  for (const t of yt) {
    const y = sy(t);
    if (y < PAD.t - 1 || y > H - PAD.b + 1) continue;
    el('line', { x1: PAD.l, x2: W - PAD.r, y1: y, y2: y, class: 'grid' }, svg);
    el('text', { x: PAD.l - 8, y: y + 4, class: 'tick', 'text-anchor': 'end' }, svg).textContent = fmtNum(t);
  }
  const xt = niceTicks(xmin, xmax, 6);
  for (const t of xt) {
    const x = sx(t);
    if (x < PAD.l - 1 || x > W - PAD.r + 1) continue;
    el('text', { x, y: H - PAD.b + 18, class: 'tick', 'text-anchor': 'middle' }, svg).textContent = fmtNum(t);
  }
  el('line', { x1: PAD.l, x2: W - PAD.r, y1: H - PAD.b, y2: H - PAD.b, class: 'axis' }, svg);
  if (spec.xlabel) el('text', { x: (PAD.l + W - PAD.r) / 2, y: H - 6, class: 'label', 'text-anchor': 'middle' }, svg).textContent = spec.xlabel;
  if (spec.ylabel) el('text', { x: 12, y: PAD.t - 10, class: 'label' }, svg).textContent = spec.ylabel;
  series.forEach((s, si) => {
    let d = '';
    s.values.forEach((v, i) => {
      if (!drawable(v, i)) return;
      const x = sx(xs[i] ?? i), y = sy(v);
      d += (d ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1);
    });
    if (d) el('path', { d, class: `series s${(si % 8) + 1}` }, svg);
    if (s.values.length <= 40) {
      s.values.forEach((v, i) => {
        if (!drawable(v, i)) return;
        el('circle', { cx: sx(xs[i] ?? i), cy: sy(v), r: 3, class: `marker s${(si % 8) + 1}` }, svg);
      });
    }
  });
  noteSkipped(f, skipped);
  // legend
  if (series.length > 1) {
    const leg = document.createElement('div');
    leg.className = 'legend';
    series.forEach((s, si) => {
      const item = document.createElement('span');
      item.className = 'legend-item';
      item.innerHTML = `<i class="swatch s${(si % 8) + 1}"></i>${esc(s.name || 'series ' + (si + 1))}`;
      leg.appendChild(item);
    });
    f.body.appendChild(leg);
  }
  // hover crosshair
  const cross = el('line', { y1: PAD.t, y2: H - PAD.b, class: 'crosshair hidden' }, svg);
  svg.addEventListener('mousemove', (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const xv = xmin + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (xmax - xmin);
    let best = -1;
    for (let i = 0; i < xs.length; i++) if (Number.isFinite(xs[i]) && (best < 0 || Math.abs(xs[i] - xv) < Math.abs(xs[best] - xv))) best = i;
    if (best < 0) return;
    cross.setAttribute('x1', sx(xs[best])); cross.setAttribute('x2', sx(xs[best]));
    cross.classList.remove('hidden');
    const rows = series.map((s, si) => `<div><i class="swatch s${(si % 8) + 1}"></i>${esc(s.name || 'series')}: <b>${fmtNum(s.values[best])}</b></div>`).join('');
    showTip(f, e.clientX, e.clientY, `<div class="tip-x">${esc(spec.xlabel || 'x')} = ${fmtNum(xs[best])}</div>${rows}`);
  });
  svg.addEventListener('mouseleave', () => { cross.classList.add('hidden'); f.tip.classList.add('hidden'); });
  f.table.innerHTML = tableHtml([spec.xlabel || 'x', ...series.map((s) => s.name || 'series')], xs.map((x, i) => [x, ...series.map((s) => s.values[i] ?? '')]));
}

/** Bar chart. spec: { title, labels, values, ylabel? } */
export function renderBar(container, spec) {
  const f = frame(container, spec.title);
  const labels = spec.labels || [], values = spec.values || [];
  if (!values.length) { f.body.textContent = 'No data.'; return; }
  const finite = values.filter(Number.isFinite);
  const vmin = Math.min(0, ...finite), vmax = Math.max(0, ...finite);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg', role: 'img' }, f.body);
  const sy = (v) => PAD.t + (1 - (v - vmin) / Math.max(1e-12, vmax - vmin)) * (H - PAD.t - PAD.b);
  for (const t of niceTicks(vmin, vmax, 5)) {
    const y = sy(t);
    el('line', { x1: PAD.l, x2: W - PAD.r, y1: y, y2: y, class: 'grid' }, svg);
    el('text', { x: PAD.l - 8, y: y + 4, class: 'tick', 'text-anchor': 'end' }, svg).textContent = fmtNum(t);
  }
  const bw = (W - PAD.l - PAD.r) / values.length;
  values.forEach((v, i) => {
    const x = PAD.l + i * bw + bw * 0.15, w = bw * 0.7;
    if (values.length <= 24) {
      const t = el('text', { x: x + w / 2, y: H - PAD.b + 16, class: 'tick', 'text-anchor': 'middle' }, svg);
      t.textContent = String(labels[i] ?? i).slice(0, 12);
    }
    if (!Number.isFinite(v)) {   // mark the slot instead of emitting height="NaN"
      el('text', { x: x + w / 2, y: sy(0) - 6, class: 'tick', 'text-anchor': 'middle' }, svg).textContent = String(v);
      return;
    }
    const y0 = sy(0), y1 = sy(v);
    const rect = el('rect', { x, y: Math.min(y0, y1), width: w, height: Math.max(1, Math.abs(y0 - y1)), rx: 3, class: 'bar s1' }, svg);
    rect.addEventListener('mousemove', (e) => showTip(f, e.clientX, e.clientY, `<div>${esc(labels[i] ?? i)}: <b>${fmtNum(v)}</b></div>`));
    rect.addEventListener('mouseleave', () => f.tip.classList.add('hidden'));
  });
  el('line', { x1: PAD.l, x2: W - PAD.r, y1: sy(0), y2: sy(0), class: 'axis' }, svg);
  if (spec.ylabel) el('text', { x: 12, y: PAD.t - 10, class: 'label' }, svg).textContent = spec.ylabel;
  noteSkipped(f, values.length - finite.length);
  f.table.innerHTML = tableHtml(['label', 'value'], values.map((v, i) => [labels[i] ?? i, v]));
}

/** Heatmap. spec: { title, rows: number[][], rowLabels?, colLabels?, min?, max? } */
export function renderHeatmap(container, spec) {
  const f = frame(container, spec.title);
  const rows = spec.rows || [];
  if (!rows.length) { f.body.textContent = 'No data.'; return; }
  const nr = rows.length, nc = Math.max(...rows.map((r) => r.length));
  let vmin = spec.min ?? Infinity, vmax = spec.max ?? -Infinity;
  if (spec.min === undefined || spec.max === undefined) {
    for (const r of rows) for (const v of r) if (Number.isFinite(v)) { vmin = Math.min(vmin, v); vmax = Math.max(vmax, v); }
  }
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) { vmin = 0; vmax = 1; }
  if (vmin === vmax) vmax = vmin + 1;
  const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches && document.documentElement.dataset.theme !== 'light' || document.documentElement.dataset.theme === 'dark';
  const lo = dark ? [24, 79, 149] : [205, 226, 251];
  const hi = dark ? [205, 226, 251] : [13, 54, 107];
  let skipped = 0;
  const color = (v) => {
    if (!Number.isFinite(v)) { skipped++; return 'var(--grid)'; }
    const t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)));
    const c = lo.map((a, i) => Math.round(a + (hi[i] - a) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
  const labelW = spec.rowLabels ? 60 : 8, labelH = spec.colLabels ? 22 : 8;
  const cell = Math.max(6, Math.min(28, Math.floor((W - labelW - 16) / nc), Math.floor((H - labelH - 8) / nr)));
  const width = labelW + nc * cell + 16, height = labelH + nr * cell + 8;
  const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, class: 'chart-svg heatmap', role: 'img', style: `max-width:${width}px` }, f.body);
  rows.forEach((r, i) => {
    if (spec.rowLabels) el('text', { x: labelW - 6, y: labelH + i * cell + cell * 0.7, class: 'tick', 'text-anchor': 'end' }, svg).textContent = String(spec.rowLabels[i]).slice(0, 8);
    r.forEach((v, j) => {
      const rect = el('rect', { x: labelW + j * cell, y: labelH + i * cell, width: cell - 1, height: cell - 1, fill: color(v), rx: 1 }, svg);
      rect.addEventListener('mousemove', (e) => showTip(f, e.clientX, e.clientY, `<div>row ${esc(spec.rowLabels ? spec.rowLabels[i] : i)}, col ${esc(spec.colLabels ? spec.colLabels[j] : j)}: <b>${fmtNum(v)}</b></div>`));
      rect.addEventListener('mouseleave', () => f.tip.classList.add('hidden'));
    });
  });
  if (spec.colLabels) rows[0].forEach((_, j) => { el('text', { x: labelW + j * cell + cell / 2, y: labelH - 6, class: 'tick', 'text-anchor': 'middle' }, svg).textContent = String(spec.colLabels[j]).slice(0, 4); });
  noteSkipped(f, skipped);
  f.table.innerHTML = tableHtml(['', ...(spec.colLabels || Array.from({ length: nc }, (_, j) => j))], rows.map((r, i) => [spec.rowLabels ? spec.rowLabels[i] : i, ...r]));
}

/** Table. spec: { title, columns, rows } */
export function renderTable(container, spec) {
  const f = frame(container, spec.title);
  f.wrap.querySelector('.chart-toggle').remove();
  f.body.innerHTML = tableHtml(spec.columns || [], spec.rows || []);
}

export function renderChart(container, type, spec) {
  try {
    if (type === 'plot') renderPlot(container, spec);
    else if (type === 'bar') renderBar(container, spec);
    else if (type === 'heatmap') renderHeatmap(container, spec);
    else if (type === 'table') renderTable(container, spec);
  } catch (err) {
    const d = document.createElement('div');
    d.className = 'log-line log-error';
    d.textContent = `Could not render ${type}: ${err.message}`;
    container.appendChild(d);
  }
}
