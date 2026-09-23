// app/markdown.js — a small, dependency-free Markdown renderer for module content.
// Supports: headings, paragraphs, bold/italic/code, links, fenced code, lists (nested one level),
// blockquotes, pipe tables, horizontal rules, and the custom containers :::predict, :::note, :::warn,
// :::plain ("In plain words") and :::deeper <title> (a collapsed <details>). Containers may nest.
//
// Module cross-references. Module text cites other modules by id ("module 15", "modules 24 and 25",
// "modules 04–06", "module-06"), because ids are names that survive reordering the path. At display time
// every such mention in prose (never inside code) becomes the number the learner sees on the path, linked to
// that module with its title as the tooltip. main.js supplies the path with configureModuleRefs().

export function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ---------- module cross-references ----------

let REFS = { byPrefix: new Map(), href: (id) => `#/m/${id}` };

/**
 * Tell the renderer which modules are on the path, in path order.
 * @param {{ modules: {id: string, title: string}[], href?: (id: string) => string }} o
 */
export function configureModuleRefs({ modules = [], href } = {}) {
  const byPrefix = new Map();
  modules.forEach((m, i) => { byPrefix.set(m.id.slice(0, 2), { id: m.id, title: m.title, pos: i }); });
  REFS = { byPrefix, href: href || REFS.href };
}

/** The number a module shows on the path ("00", "01", …), or '' when it is not on the path. */
export function pathNumberOf(idOrPrefix) {
  const r = REFS.byPrefix.get(String(idOrPrefix).slice(0, 2));
  return r ? String(r.pos).padStart(2, '0') : '';
}

/** Path entry for an id or two-digit id prefix: { id, title, pos, num } or null. */
export function moduleRef(idOrPrefix) {
  const r = REFS.byPrefix.get(String(idOrPrefix).slice(0, 2));
  return r ? { ...r, num: String(r.pos).padStart(2, '0') } : null;
}

/** HTML for one module reference: its path number (or `label`), linked, with the title as tooltip. */
export function moduleLink(idOrPrefix, label = null, { links = true } = {}) {
  const r = moduleRef(idOrPrefix);
  if (!r) return label == null ? escapeHtml(String(idOrPrefix)) : label;
  const text = label == null ? r.num : label;
  const tip = escapeHtml(r.title);
  return links ? `<a class="modref" href="${escapeHtml(REFS.href(r.id))}" title="${tip}">${text}</a>` : `<span class="modref" title="${tip}">${text}</span>`;
}

const NN = '(\\d{2})(?![\\d.,:]\\d)';
const RANGE_RE = new RegExp(`\\b(modules)(\\s+)${NN}(\\s*(?:–|—|-|to)\\s*)${NN}(?!\\w)`, 'gi');
const LIST_RE = new RegExp(`\\b(modules)(\\s+)((?:\\d{2}(?:,\\s*|,?\\s+(?:and|or)\\s+))*\\d{2})(?![\\d.,:]\\d)(?!\\w)`, 'gi');
const ONE_RE = new RegExp(`\\b(module)(\\s+|-)${NN}(?![\\w-]*\\d)`, 'gi');

/**
 * Rewrite module-id mentions in an HTML-escaped prose fragment (no tags inside that could contain the
 * pattern in an attribute) to path numbers. Unknown ids are left as written.
 */
export function linkModuleRefs(html, { links = true } = {}) {
  if (!REFS.byPrefix.size || !/module/i.test(html)) return html;
  const known = (nn) => REFS.byPrefix.has(nn);
  const link = (nn) => moduleLink(nn, null, { links });
  const held = [];
  const hold = (s) => { held.push(s); return `\u0002${held.length - 1}\u0002`; };
  let s = String(html);
  // "modules 04–06": every existing id in the range, as a range if its path positions are contiguous.
  s = s.replace(RANGE_RE, (m, word, sp, a, dash, b) => {
    const lo = Math.min(+a, +b), hi = Math.max(+a, +b);
    const ids = [];
    for (let k = lo; k <= hi; k++) { const nn = String(k).padStart(2, '0'); if (known(nn)) ids.push(nn); }
    if (!ids.length || !known(a) || !known(b)) return m;
    const pos = ids.map((nn) => REFS.byPrefix.get(nn).pos).sort((x, y) => x - y);
    const contiguous = pos.every((p, i) => i === 0 || p === pos[i - 1] + 1);
    if (contiguous) return hold(`${word}${sp}${link(a)}${dash}${link(b)}`);
    ids.sort((x, y) => REFS.byPrefix.get(x).pos - REFS.byPrefix.get(y).pos);
    return hold(`${word}${sp}${joinList(ids.map(link))}`);
  });
  // "modules 24 and 25", "modules 06, 08 and 15"
  s = s.replace(LIST_RE, (m, word, sp, list) => {
    const nums = list.match(/\d{2}/g);
    if (!nums.every(known)) return m;
    return hold(`${word}${sp}${list.replace(/\d{2}/g, (nn) => link(nn))}`);
  });
  // "module 15", "Module 07", "module-06"
  s = s.replace(ONE_RE, (m, word, sep, nn) => (known(nn) ? hold(moduleLink(nn, `${word}${sep}${moduleRef(nn).num}`, { links })) : m));
  return s.replace(/\u0002(\d+)\u0002/g, (_, i) => held[+i]);
}

function joinList(items) {
  if (items.length < 2) return items.join('');
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

// ---------- inline ----------

const CODE_MARK = '\u0001';
const LINK_MARK = '\u0003';

/** Inline markdown. `opts.links: false` renders module references as plain spans (for text inside buttons). */
export function inline(text, opts = {}) {
  const codes = [], anchors = [];
  let s = String(text).replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `${CODE_MARK}${codes.length - 1}${CODE_MARK}`; });
  s = escapeHtml(s);   // from here on `s` is escaped exactly once; URLs below must not be escaped again
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\w)/g, '$1<em>$2</em>');   // `2 * 3 * 4` in prose is not italic
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, t, u) => {
    if (/^\s*javascript:/i.test(u)) return m;
    const internal = u.startsWith('#');
    anchors.push(`<a href="${u}"${internal ? '' : ' target="_blank" rel="noopener"'}>${linkModuleRefs(t, { links: false })}</a>`);
    return `${LINK_MARK}${anchors.length - 1}${LINK_MARK}`;
  });
  s = linkModuleRefs(s, { links: opts.links !== false });
  s = s.replace(new RegExp(`${LINK_MARK}(\\d+)${LINK_MARK}`, 'g'), (_, i) => anchors[+i]);
  s = s.replace(new RegExp(`${CODE_MARK}(\\d+)${CODE_MARK}`, 'g'), (_, i) => `<code>${escapeHtml(codes[+i])}</code>`);
  return s;
}

/** Render markdown to HTML. `ctx.predictKey(index)` may customise keys for prediction cards. */
export function render(md, ctx = {}) {
  const state = ctx._state || (ctx._state = { predict: 0 });
  const lines = String(md || '').replace(/\r\n/g, '\n').split('\n');
  const out = [];
  let i = 0;
  const para = [];
  const flushPara = () => {
    if (para.length) { out.push(`<p>${inline(para.join(' ').trim())}</p>`); para.length = 0; }
  };
  while (i < lines.length) {
    const line = lines[i];
    // fenced code
    const fence = line.match(/^```([\w.+-]*)[^`]*$/);   // allows ```c-like and ```js title
    if (fence) {
      flushPara();
      const buf = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++;
      out.push(`<pre class="code lang-${fence[1] || 'text'}"><code>${escapeHtml(buf.join('\n'))}</code></pre>`);
      continue;
    }
    // custom containers (nestable: a :::deeper block may hold a :::note)
    const cont = line.match(/^:::(predict|note|warn|plain)\s*$/) || line.match(/^:::(deeper)(?:\s+(.*?))?\s*$/);
    if (cont) {
      flushPara();
      const buf = [];
      i++;
      let depth = 1, inFence = false;
      while (i < lines.length) {
        const l = lines[i];
        if (/^```/.test(l)) inFence = !inFence;
        else if (!inFence && /^:::\s*$/.test(l)) { if (--depth === 0) break; }
        else if (!inFence && /^:::\w/.test(l)) depth++;
        buf.push(l);
        i++;
      }
      i++;
      if (cont[1] === 'predict') {
        const body = buf.join('\n');
        const parts = body.split(/^\s*---\s*$/m);
        const q = parts[0] || '', a = parts.slice(1).join('\n---\n');
        const idx = state.predict++;
        const key = ctx.predictKey ? ctx.predictKey(idx) : `p${idx}`;
        out.push(`<div class="predict" data-key="${escapeHtml(key)}">
  <div class="predict-label">Predict before you reveal</div>
  <div class="predict-q">${render(q, ctx)}</div>
  <textarea class="predict-input" rows="2" placeholder="Write your prediction first…"></textarea>
  <button class="btn btn-small predict-reveal" type="button">Reveal</button>
  <div class="predict-a hidden">${render(a, ctx)}</div>
</div>`);
      } else if (cont[1] === 'plain') {
        out.push(`<div class="callout callout-plain"><div class="callout-label">In plain words</div>${render(buf.join('\n'), ctx)}</div>`);
      } else if (cont[1] === 'deeper') {
        const title = (cont[2] || '').trim() || 'Going deeper';
        out.push(`<details class="deeper"><summary><span>${inline(title)}</span></summary><div class="deeper-body">${render(buf.join('\n'), ctx)}</div></details>`);
      } else {
        out.push(`<div class="callout callout-${cont[1]}">${render(buf.join('\n'), ctx)}</div>`);
      }
      continue;
    }
    // headings
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { flushPara(); const lvl = Math.min(6, h[1].length + 1); out.push(`<h${lvl}>${inline(h[2])}</h${lvl}>`); i++; continue; }
    // hr
    if (/^\s*(---|\*\*\*)\s*$/.test(line)) { flushPara(); out.push('<hr>'); i++; continue; }
    // blockquote
    if (/^>\s?/.test(line)) {
      flushPara();
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ''));
      out.push(`<blockquote>${render(buf.join('\n'), ctx)}</blockquote>`);
      continue;
    }
    // table
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      flushPara();
      const header = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(splitRow(lines[i++]));
      out.push(`<div class="table-wrap"><table><thead><tr>${header.map((c) => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
      continue;
    }
    // lists
    const li = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
    if (li) {
      flushPara();
      const ordered = /\d/.test(li[2]);
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
        if (!m) break;
        const depth = m[1].length >= 2 ? 1 : 0;
        if (depth === 0 || !items.length) items.push({ text: m[3], children: [] });
        else items[items.length - 1].children.push(m[3]);
        i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
          const last = items[items.length - 1];
          if (last.children.length) last.children[last.children.length - 1] += ' ' + lines[i].trim();
          else last.text += ' ' + lines[i].trim();
          i++;
        }
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>${items.map((it) => `<li>${inline(it.text)}${it.children.length ? `<ul>${it.children.map((c) => `<li>${inline(c)}</li>`).join('')}</ul>` : ''}</li>`).join('')}</${tag}>`);
      continue;
    }
    if (!line.trim()) { flushPara(); i++; continue; }
    para.push(line);
    i++;
  }
  flushPara();
  return out.join('\n');
}

// Split a pipe-table row on `|`, ignoring pipes inside code spans (`a | b`) and escaped pipes (\|).
function splitRow(line) {
  const s = line.trim().replace(/^\|/, '').replace(/\|$/, '');
  const cells = [];
  let cur = '', inCode = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '\\' && s[i + 1] === '|') { cur += '|'; i++; continue; }
    if (c === '`') inCode = !inCode;
    if (c === '|' && !inCode) { cells.push(cur.trim()); cur = ''; continue; }
    cur += c;
  }
  cells.push(cur.trim());
  return cells;
}

/** Wire up predict cards inside a rendered container. getSaved/setSaved persist the learner's text. */
export function activatePredicts(root, { getSaved = () => null, setSaved = () => {} } = {}) {
  root.querySelectorAll('.predict').forEach((card) => {
    const key = card.dataset.key;
    const input = card.querySelector('.predict-input');
    const btn = card.querySelector('.predict-reveal');
    const ans = card.querySelector('.predict-a');
    const saved = getSaved(key);
    if (saved && saved.text) input.value = saved.text;
    // Once revealed, the prediction is committed: it stays visible next to the answer and can no longer be edited.
    const reveal = () => { ans.classList.remove('hidden'); btn.textContent = 'Revealed'; btn.disabled = true; input.readOnly = true; input.title = 'Your committed prediction'; };
    if (saved && saved.revealed) reveal();
    input.addEventListener('input', () => { if (!input.readOnly) setSaved(key, { text: input.value, revealed: false }); });
    btn.addEventListener('click', () => {
      if (!input.value.trim()) { input.focus(); input.classList.add('shake'); setTimeout(() => input.classList.remove('shake'), 400); return; }
      setSaved(key, { text: input.value, revealed: true });
      reveal();
    });
  });
}
