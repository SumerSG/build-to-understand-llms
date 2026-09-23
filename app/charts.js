// app/charts.js — small SVG charts for the Goal demo output: line plot, bar, heatmap, table.
// Colours come from CSS variables (--chart-1 … --chart-8, --heat-lo/--heat-hi, --ink, --muted, --grid) so
// the charts follow light/dark mode. Every chart ships a hover tooltip and a table view toggle.
//
// Legibility rules: the SVG is drawn at the width it is shown at (so an 11px tick label is 11px on a phone,
// and it is redrawn when that width changes), numbers print without trailing zeros, log axes are labelled
// with plain values (1, 10, 100), crowded axis labels are thinned so they never overlap, and every heatmap
// carries a min-to-max colour legend.

const SVG = 'http://www.w3.org/2000/svg';
const MAX_W = 640, MIN_W = 300;
const TICK_PX = 11;                 // tick labels: at least this many CSS pixels high, at any width
const CHAR_PX = 6.6;                // generous average width of one 11px tick character (tabular digits)

function el(tag, attrs = {}, parent = null) {
  const n = document.createElementNS(SVG, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}

/** Strip float noise and trailing zeros: 0.30000000000000004 -> 0.3, 9.300 -> 9.3. */
const clean = (v, digits) => String(+v.toFixed(digits));

/** A value in a tooltip or table: short, no trailing zeros (9.3, not 9.300). */
function fmtNum(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  if (typeof v !== 'number') return String(v);
  if (!Number.isFinite(v)) return v > 0 ? 'Infinity' : '-Infinity';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2).replace(/\.?0+e/, 'e').replace('e+', 'e');
  if (Number.isInteger(v)) return String(v);
  return a < 1 ? clean(v, 4) : a < 100 ? clean(v, 3) : clean(v, 1);
}

/** An axis tick: a plain value, with thousands separators and k/M/B for big round numbers. */
function fmtTick(v) {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 1e-4 || a >= 1e15) return v.toExponential().replace('e+', 'e');
  for (const [unit, s] of [[1e12, 'T'], [1e9, 'B'], [1e6, 'M']]) if (a >= unit) return `${+(v / unit).toPrecision(4)}${s}`;
  const p = +v.toPrecision(10);
  return a >= 1e4 ? p.toLocaleString('en-US', { maximumFractionDigits: 6 }) : String(p);
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

/** Ticks for a log axis between two positive values, in plain values: powers of ten (1, 10, 100) when the
 *  range spans them, else 1-2-5 steps (2, 5, 10, 20), else ordinary nice ticks. */
function logTicks(min, max, n = 5) {
  const inside = (t) => t >= min * (1 - 1e-9) && t <= max * (1 + 1e-9);
  const lo = Math.floor(Math.log10(min)), hi = Math.ceil(Math.log10(max));
  const every = Math.max(1, Math.ceil((hi - lo) / Math.max(1, n)));
  const decades = [];
  for (let e = lo; e <= hi; e += every) decades.push(+Math.pow(10, e).toPrecision(12));
  const tens = decades.filter(inside);
  if (tens.length >= 3) return tens;
  const steps = [];
  for (let e = lo; e <= hi; e++) for (const m of [1, 2, 5]) steps.push(+(m * Math.pow(10, e)).toPrecision(12));
  const fit = steps.filter(inside);
  if (fit.length >= 2 && fit.length <= n + 2) return fit;          // 1, 2, 5, 10, 20, 50 on a short range
  if (tens.length >= 2) return tens;
  return niceTicks(min, max, n).filter((t) => t > 0);
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

/** The width the chart body is shown at, in CSS pixels (0 while hidden or not yet laid out). */
function shownWidth(f) {
  return Math.floor(f.body.getBoundingClientRect().width || 0);
}

/**
 * Draw into f.body at the width it is shown at, and draw again when that width changes (a phone rotated,
 * a window resized, a hidden tab shown), so tick labels keep their pixel size instead of shrinking.
 * draw(width, canvas, shown) builds everything inside `canvas`; `shown` is the measured width (0 if unknown).
 */
function responsive(f, draw) {
  const canvas = document.createElement('div');
  f.body.appendChild(canvas);
  let drawnFor = -1;
  const redraw = () => {
    const shown = shownWidth(f);
    if (!shown && drawnFor > 0) return;                 // hidden (table view, other tab): keep what is drawn
    const width = shown ? Math.max(MIN_W, Math.min(MAX_W, shown)) : MAX_W;
    if (drawnFor > 0 && Math.abs(width - drawnFor) < 24) return;
    drawnFor = width;
    canvas.textContent = '';
    draw(width, canvas, shown);
  };
  redraw();
  if (typeof ResizeObserver === 'function') {
    let pending = false;
    new ResizeObserver(() => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => { pending = false; if (f.wrap.isConnected) redraw(); });
    }).observe(f.body);
  }
}

/** Keep tick text at TICK_PX on screen when an SVG `width` units wide is shown in `shown` pixels. */
function tickStyle(width, shown) {
  if (!shown || shown >= width) return {};
  return { style: `font-size:${(TICK_PX * width / shown).toFixed(1)}px` };
}

function tableHtml(columns, rows) {
  return `<table><thead><tr>${columns.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows
    .map((r) => `<tr>${r.map((v) => `<td>${esc(typeof v === 'number' ? fmtNum(v) : v)}</td>`).join('')}</tr>`)
    .join('')}</tbody></table>`;
}

function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }

/** Read a colour token (#rrggbb) from the current theme; fall back when it is missing or not hex. */
function cssRgb(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const m = /^#([0-9a-f]{6})$/i.exec(v);
  return m ? [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16)) : fallback;
}

// A diverging loss or a 1/0 in learner code produces NaN/Infinity; say so instead of drawing a silent gap.
function noteSkipped(parent, n) {
  if (!n) return;
  const d = document.createElement('div');
  d.className = 'chart-note small muted';
  d.textContent = `${n} value${n > 1 ? 's were' : ' was'} NaN or ±Infinity and ${n > 1 ? 'are' : 'is'} not drawn.`;
  parent.appendChild(d);
}

function showTip(f, x, y, html) {
  f.tip.innerHTML = html;
  f.tip.classList.remove('hidden');
  const r = f.wrap.getBoundingClientRect();
  f.tip.style.left = Math.max(0, Math.min(x - r.left + 12, r.width - f.tip.offsetWidth - 8)) + 'px';
  f.tip.style.top = Math.max(0, y - r.top - f.tip.offsetHeight - 12) + 'px';
}

/** An x label that starts "log10(rank)" means the x values are already log10 of the real ones: show the
 *  real ones (1, 10, 100) on the axis and in the tooltip instead of 0.2, 0.4 … (whatever follows the
 *  bracket explained the log values, so it is dropped with them). */
function preLogged(label) {
  const m = /^\s*log10\s*\(\s*([^()]+?)\s*\)/.exec(label || '');
  return m ? m[1] : null;
}

/** Line plot. spec: { title, x?, series: [{ name, values }], xlabel?, ylabel?, yscale?: 'log', xscale?: 'log' } */
export function renderPlot(container, spec) {
  const f = frame(container, spec.title);
  const series = (spec.series || []).filter((s) => s && s.values && s.values.length);
  if (!series.length) { f.body.textContent = 'No data.'; return; }
  const log = spec.yscale === 'log';
  const n = Math.max(...series.map((s) => s.values.length));
  const xs = spec.x && spec.x.length ? spec.x : Array.from({ length: n }, (_, i) => i);
  // x on a log scale: either the values are real and xscale is 'log', or they arrive as log10(values)
  const xReal = preLogged(spec.xlabel);
  const xLog = spec.xscale === 'log' || !!xReal;
  const xPos = (x) => (xReal ? x : spec.xscale === 'log' ? (x > 0 ? Math.log10(x) : NaN) : x);   // position on the axis
  const xVal = (x) => (xReal ? Math.pow(10, x) : x);                                                // value a reader sees
  const xLabel = xReal ? `${xReal} (log scale)` : spec.xscale === 'log' && spec.xlabel ? `${spec.xlabel} (log scale)` : spec.xlabel;
  const finiteXs = xs.map(xPos).filter(Number.isFinite);
  const xmin = finiteXs.length ? Math.min(...finiteXs) : 0, xmax = finiteXs.length ? Math.max(...finiteXs) : 1;
  let ymin = Infinity, ymax = -Infinity, skipped = 0;
  const drawable = (v, i) => Number.isFinite(v) && (!log || v > 0) && Number.isFinite(xPos(xs[i] ?? i));
  for (const s of series) s.values.forEach((v, i) => { if (drawable(v, i)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); } else skipped++; });
  if (!Number.isFinite(ymin)) { ymin = log ? 1 : 0; ymax = log ? 10 : 1; }
  if (ymin === ymax) { if (log) { ymin /= 10; ymax *= 10; } else { ymin -= 1; ymax += 1; } }
  const ty = (v) => (log ? Math.log10(v) : v);
  const lo = ty(ymin), hi = ty(ymax);
  const yt = log ? logTicks(ymin, ymax, 5) : niceTicks(ymin, ymax, 5);
  const yLabels = yt.map(fmtTick);

  responsive(f, (W, canvas, shown) => {
    const H = Math.round(Math.min(300, Math.max(230, W * 0.6)));
    const longest = Math.max(1, ...yLabels.map((s) => s.length));
    const PAD = { l: Math.max(36, Math.round(14 + CHAR_PX * longest)), r: 16, t: 28, b: 40 };
    const plotW = W - PAD.l - PAD.r;
    const sx = (x) => PAD.l + ((x - xmin) / Math.max(1e-12, xmax - xmin)) * plotW;
    const sy = (v) => PAD.t + (1 - (ty(v) - lo) / Math.max(1e-12, hi - lo)) * (H - PAD.t - PAD.b);
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg', role: 'img' }, canvas);
    const tick = { class: 'tick', ...tickStyle(W, shown) };
    yt.forEach((t, k) => {
      const y = sy(t);
      if (y < PAD.t - 1 || y > H - PAD.b + 1) return;
      el('line', { x1: PAD.l, x2: W - PAD.r, y1: y, y2: y, class: 'grid' }, svg);
      el('text', { x: PAD.l - 8, y: y + 4, 'text-anchor': 'end', ...tick }, svg).textContent = yLabels[k];
    });
    // x ticks: as many as fit, and a label is skipped rather than drawn over its neighbour
    let xt, labelOf;
    if (xLog) {
      xt = logTicks(Math.pow(10, xmin), Math.pow(10, xmax), Math.max(2, Math.floor(plotW / 70))).map((t) => Math.log10(t));
      labelOf = (t) => fmtTick(+Math.pow(10, t).toPrecision(12));
    } else {
      xt = niceTicks(xmin, xmax, Math.max(2, Math.min(6, Math.floor(plotW / 80))));
      labelOf = fmtTick;
    }
    let lastRight = -Infinity;
    for (const t of xt) {
      const x = sx(t);
      if (x < PAD.l - 1 || x > W - PAD.r + 1) continue;
      const text = labelOf(t), half = (text.length * CHAR_PX) / 2;
      if (x - half < lastRight + 6) continue;
      lastRight = x + half;
      el('text', { x, y: H - PAD.b + 18, 'text-anchor': 'middle', ...tick }, svg).textContent = text;
    }
    el('line', { x1: PAD.l, x2: W - PAD.r, y1: H - PAD.b, y2: H - PAD.b, class: 'axis' }, svg);
    if (xLabel) el('text', { x: (PAD.l + W - PAD.r) / 2, y: H - 6, class: 'label', 'text-anchor': 'middle' }, svg).textContent = xLabel;
    if (spec.ylabel) el('text', { x: 12, y: PAD.t - 10, class: 'label' }, svg).textContent = spec.ylabel;
    series.forEach((s, si) => {
      let d = '';
      s.values.forEach((v, i) => {
        if (!drawable(v, i)) return;
        d += (d ? 'L' : 'M') + sx(xPos(xs[i] ?? i)).toFixed(1) + ' ' + sy(v).toFixed(1);
      });
      if (d) el('path', { d, class: `series s${(si % 8) + 1}` }, svg);
      if (s.values.length <= 40) {
        s.values.forEach((v, i) => {
          if (!drawable(v, i)) return;
          el('circle', { cx: sx(xPos(xs[i] ?? i)), cy: sy(v), r: 3, class: `marker s${(si % 8) + 1}` }, svg);
        });
      }
    });
    noteSkipped(canvas, skipped);
    if (series.length > 1) {
      const leg = document.createElement('div');
      leg.className = 'legend';
      series.forEach((s, si) => {
        const item = document.createElement('span');
        item.className = 'legend-item';
        item.innerHTML = `<i class="swatch s${(si % 8) + 1}"></i>${esc(s.name || 'series ' + (si + 1))}`;
        leg.appendChild(item);
      });
      canvas.appendChild(leg);
    }
    // hover crosshair
    const cross = el('line', { y1: PAD.t, y2: H - PAD.b, class: 'crosshair hidden' }, svg);
    svg.addEventListener('mousemove', (e) => {
      const r = svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * W;
      const xv = xmin + ((px - PAD.l) / plotW) * (xmax - xmin);
      let best = -1;
      for (let i = 0; i < xs.length; i++) {
        const p = xPos(xs[i]);
        if (Number.isFinite(p) && (best < 0 || Math.abs(p - xv) < Math.abs(xPos(xs[best]) - xv))) best = i;
      }
      if (best < 0) return;
      cross.setAttribute('x1', sx(xPos(xs[best]))); cross.setAttribute('x2', sx(xPos(xs[best])));
      cross.classList.remove('hidden');
      const rows = series.map((s, si) => `<div><i class="swatch s${(si % 8) + 1}"></i>${esc(s.name || 'series')}: <b>${fmtNum(s.values[best])}</b></div>`).join('');
      showTip(f, e.clientX, e.clientY, `<div class="tip-x">${esc(xReal || spec.xlabel || 'x')} = ${fmtNum(+xVal(xs[best]).toPrecision(10))}</div>${rows}`);
    });
    svg.addEventListener('mouseleave', () => { cross.classList.add('hidden'); f.tip.classList.add('hidden'); });
  });
  f.table.innerHTML = tableHtml([xReal || spec.xlabel || 'x', ...series.map((s) => s.name || 'series')],
    xs.map((x, i) => [xReal ? +xVal(x).toPrecision(10) : x, ...series.map((s) => s.values[i] ?? '')]));
}

/** Bar chart. spec: { title, labels, values, ylabel? } */
export function renderBar(container, spec) {
  const f = frame(container, spec.title);
  const labels = spec.labels || [], values = spec.values || [];
  if (!values.length) { f.body.textContent = 'No data.'; return; }
  const finite = values.filter(Number.isFinite);
  const vmin = Math.min(0, ...finite), vmax = Math.max(0, ...finite);
  const yt = niceTicks(vmin, vmax, 5);
  const yLabels = yt.map(fmtTick);
  const texts = values.map((_, i) => String(labels[i] ?? i).slice(0, 12));
  responsive(f, (W, canvas, shown) => {
    const H = Math.round(Math.min(300, Math.max(230, W * 0.6)));
    const PAD = { l: Math.max(36, Math.round(14 + CHAR_PX * Math.max(1, ...yLabels.map((s) => s.length)))), r: 16, t: 28, b: 40 };
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg', role: 'img' }, canvas);
    const tick = { class: 'tick', ...tickStyle(W, shown) };
    const sy = (v) => PAD.t + (1 - (v - vmin) / Math.max(1e-12, vmax - vmin)) * (H - PAD.t - PAD.b);
    yt.forEach((t, k) => {
      const y = sy(t);
      el('line', { x1: PAD.l, x2: W - PAD.r, y1: y, y2: y, class: 'grid' }, svg);
      el('text', { x: PAD.l - 8, y: y + 4, 'text-anchor': 'end', ...tick }, svg).textContent = yLabels[k];
    });
    const bw = (W - PAD.l - PAD.r) / values.length;
    // label every bar when the labels fit side by side, otherwise every k-th one, so they never overlap
    const widest = Math.max(1, ...texts.map((t) => t.length)) * CHAR_PX + 6;
    const every = Math.max(1, Math.ceil(widest / bw));
    values.forEach((v, i) => {
      const x = PAD.l + i * bw + bw * 0.15, w = bw * 0.7;
      if (i % every === 0) el('text', { x: x + w / 2, y: H - PAD.b + 16, 'text-anchor': 'middle', ...tick }, svg).textContent = texts[i];
      if (!Number.isFinite(v)) {   // mark the slot instead of emitting height="NaN"
        el('text', { x: x + w / 2, y: sy(0) - 6, 'text-anchor': 'middle', ...tick }, svg).textContent = String(v);
        return;
      }
      const y0 = sy(0), y1 = sy(v);
      const rect = el('rect', { x, y: Math.min(y0, y1), width: w, height: Math.max(1, Math.abs(y0 - y1)), rx: 3, class: 'bar s1' }, svg);
      rect.addEventListener('mousemove', (e) => showTip(f, e.clientX, e.clientY, `<div>${esc(labels[i] ?? i)}: <b>${fmtNum(v)}</b></div>`));
      rect.addEventListener('mouseleave', () => f.tip.classList.add('hidden'));
    });
    el('line', { x1: PAD.l, x2: W - PAD.r, y1: sy(0), y2: sy(0), class: 'axis' }, svg);
    if (spec.ylabel) el('text', { x: 12, y: PAD.t - 10, class: 'label' }, svg).textContent = spec.ylabel;
    noteSkipped(canvas, values.length - finite.length);
  });
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
  // Sequential ramp from the theme tokens (--heat-lo → --heat-hi): one ramp, light to strong, in either mode.
  const lo = cssRgb('--heat-lo', [233, 230, 221]);
  const hi = cssRgb('--heat-hi', [16, 16, 16]);
  const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
  const color = (v) => {
    if (!Number.isFinite(v)) return 'var(--grid)';
    const t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)));
    return rgb(lo.map((a, i) => Math.round(a + (hi[i] - a) * t)));
  };
  let skipped = 0;
  for (const r of rows) for (const v of r) if (!Number.isFinite(v)) skipped++;
  const clip = (t, n) => { t = String(t); return t.length > n ? t.slice(0, n - 1) + '…' : t; };
  const rowText = spec.rowLabels ? rows.map((_, i) => clip(spec.rowLabels[i] ?? '', 22)) : null;
  const colText = spec.colLabels ? Array.from({ length: nc }, (_, j) => clip(spec.colLabels[j] ?? '', 8)) : null;

  responsive(f, (W, canvas, shown) => {
    const longest = rowText ? Math.max(...rowText.map((l) => l.length)) : 0;
    const labelW = rowText ? Math.min(150, 12 + CHAR_PX * longest) : 8, labelH = colText ? 22 : 8;
    const cell = Math.max(6, Math.min(28, Math.floor((W - labelW - 16) / nc), Math.floor((300 - labelH - 8) / nr)));
    const width = labelW + nc * cell + 16, height = labelH + nr * cell + 8;
    const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, class: 'chart-svg heatmap', role: 'img', style: `max-width:${width}px` }, canvas);
    const tick = { class: 'tick', ...tickStyle(width, shown && Math.min(shown, width)) };
    // thin crowded labels: a row label needs about one line of text height, a column label its own width
    const rowEvery = Math.max(1, Math.ceil((TICK_PX + 2) / cell));
    const colEvery = colText ? Math.max(1, Math.ceil((Math.max(...colText.map((t) => t.length)) * CHAR_PX + 6) / cell)) : 1;
    rows.forEach((r, i) => {
      if (rowText && i % rowEvery === 0) el('text', { x: labelW - 6, y: labelH + i * cell + cell / 2 + 4, 'text-anchor': 'end', ...tick }, svg).textContent = rowText[i];
      r.forEach((v, j) => {
        const rect = el('rect', { x: labelW + j * cell, y: labelH + i * cell, width: cell - 1, height: cell - 1, fill: color(v), rx: 1 }, svg);
        rect.addEventListener('mousemove', (e) => showTip(f, e.clientX, e.clientY, `<div>row ${esc(spec.rowLabels ? spec.rowLabels[i] : i)}, col ${esc(spec.colLabels ? spec.colLabels[j] : j)}: <b>${fmtNum(v)}</b></div>`));
        rect.addEventListener('mouseleave', () => f.tip.classList.add('hidden'));
      });
    });
    if (colText) {
      for (let j = 0; j < nc; j += colEvery) el('text', { x: labelW + j * cell + cell / 2, y: labelH - 6, 'text-anchor': 'middle', ...tick }, svg).textContent = colText[j];
    }
    // colour legend: what the lightest and the strongest cells mean
    const leg = document.createElement('div');
    leg.className = 'legend heat-legend';
    leg.setAttribute('aria-label', `colour scale from ${fmtNum(vmin)} (lightest) to ${fmtNum(vmax)} (strongest)`);
    const item = document.createElement('span');
    item.className = 'legend-item';
    const minT = document.createElement('span'), maxT = document.createElement('span');
    minT.textContent = fmtNum(vmin);
    maxT.textContent = fmtNum(vmax);
    const bar = document.createElement('i');
    bar.setAttribute('aria-hidden', 'true');
    bar.style.cssText = `display:inline-block;width:120px;height:10px;border:1px solid var(--border);background:linear-gradient(to right, ${rgb(lo)}, ${rgb(hi)})`;
    item.append(minT, bar, maxT);
    leg.appendChild(item);
    canvas.appendChild(leg);
    noteSkipped(canvas, skipped);
  });
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

/** Number formatting and tick choice, exported for unit tests (no DOM needed). */
export const format = { fmtNum, fmtTick, niceTicks, logTicks, preLogged };
