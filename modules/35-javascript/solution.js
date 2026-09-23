// JavaScript for this lab — reference solution.

export function share(count, total) {
  const fraction = count / total;
  return fraction;
}

export function label(letter, count) {
  return `${letter}: ${count}`;
}

export function surprise(count, total) {
  return -Math.log(share(count, total));
}

export function range(n) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(i);
  return out;
}

export function sum(numbers) {
  let total = 0;
  for (const x of numbers) total += x;
  return total;
}

export function firstK(items, k) {
  return items.slice(0, k);
}

export function sameShape(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export function countLetters(text) {
  const counts = new Map();
  const letters = text.toLowerCase().match(/[a-z]/g) || [];
  for (const ch of letters) counts.set(ch, (counts.get(ch) || 0) + 1);
  return counts;
}

export function rankCounts(counts) {
  const pairs = [...counts];
  pairs.sort((a, b) => b[1] - a[1]);
  return pairs;
}

export function makeTable(rows, cols, values) {
  if (values.length !== rows * cols) {
    throw new Error(`makeTable: a ${rows} x ${cols} table needs ${rows * cols} values, got ${values.length}`);
  }
  const shape = [rows, cols];
  const data = Float32Array.from(values);
  return { shape, data };
}

export function cell(t, row, col) {
  const { shape, data } = t;
  return data[row * shape[1] + col];
}
