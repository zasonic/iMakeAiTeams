// scripts/build-sidecar.cjs — node entry for `npm run build:sidecar`.
//
// Delegates to the platform-appropriate build_sidecar script. Useful for CI
// matrices that want a single `npm run build:sidecar` regardless of OS.

const { spawnSync } = require("node:child_process");
const { resolve } = require("node:path");
const { existsSync } = require("node:fs");

const repoRoot = resolve(__dirname, "..");
const isWindows = process.platform === "win32";

const script = isWindows
  ? resolve(repoRoot, "backend", "build_sidecar.bat")
  : resolve(repoRoot, "backend", "build_sidecar.sh");

if (!existsSync(script)) {
  console.error(`build-sidecar: ${script} not found`);
  process.exit(1);
}

const result = isWindows
  ? spawnSync("cmd.exe", ["/c", script], { stdio: "inherit" })
  : spawnSync("bash", [script], { stdio: "inherit" });

if (result.error) {
  console.error("build-sidecar:", result.error.message);
  process.exit(1);
}

process.exit(result.status ?? 0);
