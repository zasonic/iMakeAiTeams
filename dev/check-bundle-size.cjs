#!/usr/bin/env node
/*
 * dev/check-bundle-size.cjs — Layer C6 bundle-size delta gate.
 *
 * Runs after ``npm run build`` and fails the build when the renderer
 * bundle size grows by more than a configured percent over the
 * baseline. The full Layer C6 plan (first-paint < 500 ms via
 * Lighthouse-CI) is still deferred — a real first-paint measurement
 * needs a packaged Electron run, which doesn't exist in CI yet. The
 * bundle-size half of the pair IS tractable today (everything we need
 * is in ``out/renderer/`` after a vite build) and catches the most
 * common UX regression: someone pulls in a 2 MB mermaid plugin
 * without realising the renderer already bundles half of mermaid.
 *
 * Behaviour:
 *   - Walks out/renderer/ summing the byte size of every .js / .css /
 *     .html file. node_modules-style ratios on individual files are
 *     not policed — only the rolled-up total.
 *   - Reads dev/bundle-size-baseline.json for the prior total.
 *   - Fails when total > baseline * (1 + tolerance). Default
 *     tolerance is 10 percent — the goal is to catch step-changes,
 *     not bicker about kilobytes.
 *   - Always prints the absolute total + delta so PR reviewers see
 *     "+47 KB (0.6%)" in the build log even when the gate passes.
 *
 * Update the baseline:
 *     node dev/check-bundle-size.cjs --update
 *   ↑ writes the current measurement to the JSON file. Commit it
 *   alongside the change that caused the size move.
 */

const fs = require("node:fs");
const path = require("node:path");

const REPO_ROOT = path.resolve(__dirname, "..");
const RENDERER_DIR = path.join(REPO_ROOT, "out", "renderer");
const BASELINE_FILE = path.join(__dirname, "bundle-size-baseline.json");

// Allowed growth before the gate fails. 10% catches step-changes (a
// new chart library, a forgotten dynamic import) without burning PR
// time on kilobyte-sized drift.
const TOLERANCE_PCT = 10;

const COUNTED_EXTS = new Set([".js", ".css", ".html"]);

function walk(dir) {
  let total = 0;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      total += walk(full);
    } else if (COUNTED_EXTS.has(path.extname(entry.name))) {
      total += fs.statSync(full).size;
    }
  }
  return total;
}

function fmtKB(bytes) {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function fmtPct(num, den) {
  if (den === 0) return "n/a";
  const pct = ((num - den) / den) * 100;
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
}

function loadBaseline() {
  if (!fs.existsSync(BASELINE_FILE)) return null;
  try {
    return JSON.parse(fs.readFileSync(BASELINE_FILE, "utf8"));
  } catch (err) {
    process.stderr.write(`Could not parse ${BASELINE_FILE}: ${err.message}\n`);
    process.exit(2);
  }
}

function writeBaseline(total) {
  const payload = {
    _comment:
      "Layer C6 bundle-size baseline. Update via " +
      "`node dev/check-bundle-size.cjs --update` and commit the diff " +
      "alongside the change that moved the number.",
    total_bytes: total,
    measured_at: new Date().toISOString().slice(0, 10),
  };
  fs.writeFileSync(BASELINE_FILE, JSON.stringify(payload, null, 2) + "\n");
}

const args = process.argv.slice(2);
const updating = args.includes("--update");

if (!fs.existsSync(RENDERER_DIR)) {
  process.stderr.write(
    `Renderer build directory not found at ${RENDERER_DIR}.\n` +
    `Run \`npm run build\` first.\n`,
  );
  process.exit(2);
}

const total = walk(RENDERER_DIR);

if (updating) {
  writeBaseline(total);
  process.stdout.write(`Updated baseline → ${fmtKB(total)} (${total} bytes).\n`);
  process.exit(0);
}

const baseline = loadBaseline();
if (baseline === null) {
  process.stderr.write(
    `No baseline at ${BASELINE_FILE}. Run with --update once to seed it.\n`,
  );
  process.exit(2);
}

const baselineBytes = Number(baseline.total_bytes);
if (!Number.isFinite(baselineBytes) || baselineBytes <= 0) {
  process.stderr.write(
    `Malformed baseline: total_bytes = ${baseline.total_bytes}\n`,
  );
  process.exit(2);
}

const allowed = Math.floor(baselineBytes * (1 + TOLERANCE_PCT / 100));
const delta = fmtPct(total, baselineBytes);

process.stdout.write(
  `Renderer bundle: ${fmtKB(total)} (${total} bytes)\n` +
  `Baseline:        ${fmtKB(baselineBytes)} (${baselineBytes} bytes) ` +
  `measured ${baseline.measured_at ?? "unknown date"}\n` +
  `Delta:           ${delta}\n` +
  `Tolerance:       +${TOLERANCE_PCT}% (${fmtKB(allowed - baselineBytes)} headroom)\n`,
);

if (total > allowed) {
  process.stderr.write(
    `\nbundle-size: total grew beyond +${TOLERANCE_PCT}% tolerance.\n` +
    `If this growth is intentional, run:\n` +
    `  node dev/check-bundle-size.cjs --update\n` +
    `and commit the regenerated baseline alongside your change.\n`,
  );
  process.exit(1);
}

process.stdout.write(`bundle-size: within tolerance.\n`);
process.exit(0);
