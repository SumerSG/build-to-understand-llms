// app/rewrite.js — rewrite bare `lib/…` import specifiers to absolute URLs.
// Shared by the browser sandbox worker and the Node verifier.

const RE = /(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(['"])lib\/([^'"]+)\2/g;

/**
 * @param {string} code   ES module source
 * @param {string} base   absolute URL of the repo root, without trailing slash
 */
export function rewriteImports(code, base) {
  return code.replace(RE, (_, prefix, quote, path) => `${prefix}${quote}${base}/lib/${path}${quote}`);
}
