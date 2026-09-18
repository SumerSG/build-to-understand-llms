// app/editor.js — code editor wrapper: CodeMirror 5 when the CDN loaded, plain textarea otherwise.

export function createEditor(container, { value = '', onChange = () => {}, onRun = () => {} } = {}) {
  container.innerHTML = '';
  if (typeof window !== 'undefined' && window.CodeMirror) {
    const cm = window.CodeMirror(container, {
      value,
      mode: 'javascript',
      theme: 'btu',
      lineNumbers: true,
      indentUnit: 2,
      tabSize: 2,
      lineWrapping: false,
      matchBrackets: true,
      viewportMargin: 50,
      extraKeys: {
        'Ctrl-Enter': () => onRun(),
        'Cmd-Enter': () => onRun(),
        Tab: (c) => { if (c.somethingSelected()) c.indentSelection('add'); else c.replaceSelection('  ', 'end'); },
      },
    });
    cm.on('change', () => onChange(cm.getValue()));
    return {
      kind: 'codemirror',
      getValue: () => cm.getValue(),
      setValue: (v) => { cm.setValue(v); },
      focus: () => cm.focus(),
      refresh: () => cm.refresh(),
      goTo: (line) => { cm.setCursor({ line: Math.max(0, line - 1), ch: 0 }); cm.focus(); },
    };
  }
  const ta = document.createElement('textarea');
  ta.className = 'editor-fallback';
  ta.spellcheck = false;
  ta.value = value;
  ta.addEventListener('input', () => onChange(ta.value));
  ta.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); onRun(); }
    if (e.key === 'Tab') { e.preventDefault(); const s = ta.selectionStart; ta.setRangeText('  ', s, ta.selectionEnd, 'end'); onChange(ta.value); }
  });
  container.appendChild(ta);
  return {
    kind: 'textarea',
    getValue: () => ta.value,
    setValue: (v) => { ta.value = v; },
    focus: () => ta.focus(),
    refresh: () => {},
    goTo: () => ta.focus(),
  };
}
