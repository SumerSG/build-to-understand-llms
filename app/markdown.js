// app/markdown.js — a small, dependency-free Markdown renderer for module content.
// Supports: headings, paragraphs, bold/italic/code, links, fenced code, lists (nested one level),
// blockquotes, pipe tables, horizontal rules, and the custom :::predict / :::note containers.

export function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

const CODE_MARK = '';

export function inline(text) {
  const codes = [];
  let s = String(text).replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `${CODE_MARK}${codes.length - 1}${CODE_MARK}`; });
  s = escapeHtml(s);   // from here on `s` is escaped exactly once; URLs below must not be escaped again
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\w)/g, '$1<em>$2</em>');   // `2 * 3 * 4` in prose is not italic
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, t, u) => (/^\s*javascript:/i.test(u) ? m : `<a href="${u}" target="_blank" rel="noopener">${t}</a>`));
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
    // custom containers
    const cont = line.match(/^:::(predict|note|warn)\s*$/);
    if (cont) {
      flushPara();
      const buf = [];
      i++;
      while (i < lines.length && !/^:::\s*$/.test(lines[i])) buf.push(lines[i++]);
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
      out.push(`<table><thead><tr>${header.map((c) => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`);
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
