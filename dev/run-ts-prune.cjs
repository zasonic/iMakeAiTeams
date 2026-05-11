#!/usr/bin/env node
/*
 * dev/run-ts-prune.cjs — Layer 4.3 dead-code gate for the renderer.
 *
 * Symmetric with backend's vulture gate. Runs ts-prune over the
 * desktop-ui + desktop-shell trees (via tsconfig.web/.node) and fails
 * with non-zero exit if it reports any unused export NOT listed in
 * dev/ts-prune-allowlist.txt.
 *
 * Allowlist format: one entry per line, ``relative/path.ts - exportName``
 * (line numbers omitted on purpose — they shift on every edit and would
 * force allowlist churn even for unrelated changes). Lines starting
 * with ``#`` are comments. Add an entry only after verifying the
 * export is referenced through indirection ts-prune can't see
 * (test-only imports, dynamic imports, Vite/Electron entry points
 * re-imported through HTML).
 *
 * Why a wrapper instead of vanilla ``ts-prune --error``: ts-prune's
 * built-in ignore mechanism is regex-based and entry-point-aware but
 * has no allowlist file, so every re-introduction of a known false
 * positive would re-fail CI. The vulture gate solves the same problem
 * with dev/vulture_allowlist.py — this is the symmetric solution.
 */

const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const REPO_ROOT = path.resolve(__dirname, "..");
const ALLOWLIST_PATH = path.join(__dirname, "ts-prune-allowlist.txt");

function loadAllowlist() {
  if (!fs.existsSync(ALLOWLIST_PATH)) return new Set();
  return new Set(
    fs
      .readFileSync(ALLOWLIST_PATH, "utf8")
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#")),
  );
}

function runTsPrune(project) {
  // ts-prune writes findings to stdout; non-zero exit only on internal
  // crash. We capture the output and decide pass/fail ourselves so the
  // allowlist applies before the gate trips.
  try {
    return execFileSync(
      "npx",
      ["--no-install", "ts-prune", "--project", project],
      { cwd: REPO_ROOT, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] },
    );
  } catch (err) {
    // ts-prune itself crashed — surface that so CI fails loudly.
    process.stderr.write(`ts-prune crashed for ${project}: ${err.message}\n`);
    process.exit(2);
  }
}

const allowlist = loadAllowlist();
const projects = ["tsconfig.web.json", "tsconfig.node.json"];

// Paths excluded from the gate entirely. Generated files emit interfaces
// the renderer may not import yet — `desktop-ui/api/generated.d.ts` is
// the Pydantic → TS codegen output (Layer C5). Listing every entry in
// the allowlist would spam it; the gate that matters for generated files
// is the "did regenerating it change anything" check in CI, not the
// dead-export check.
const IGNORED_PATHS = new Set([
  "desktop-ui/api/generated.d.ts",
]);

const findings = [];
for (const project of projects) {
  const raw = runTsPrune(project);
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    // ts-prune lines look like:
    //   desktop-ui/api/client.ts:42 - foo
    //   desktop-ui/api/client.ts:42 - foo (used in module)
    // We strip the `:NN` line number before allowlist matching so an
    // unrelated edit that just shifts line numbers doesn't churn the
    // allowlist. Match the key against `path - name`.
    const dashIdx = trimmed.indexOf(" - ");
    if (dashIdx < 0) continue;
    const head = trimmed.slice(0, dashIdx).replace(/:\d+$/, "");
    const tail = trimmed.slice(dashIdx + 3).split(" ")[0];
    if (IGNORED_PATHS.has(head)) continue;
    const key = `${head} - ${tail}`;
    if (allowlist.has(key)) continue;
    findings.push(trimmed);
  }
}

if (findings.length === 0) {
  process.stdout.write("ts-prune: no unused exports outside the allowlist.\n");
  process.exit(0);
}

process.stderr.write(
  `ts-prune found ${findings.length} unused export(s) outside the allowlist:\n`,
);
for (const f of findings) process.stderr.write(`  ${f}\n`);
process.stderr.write(
  "\nIf an entry is a known false positive (test-only import, dynamic\n" +
    "import, framework entry point), add it to dev/ts-prune-allowlist.txt\n" +
    "in the form `path:line - name` with a brief comment explaining why.\n",
);
process.exit(1);
