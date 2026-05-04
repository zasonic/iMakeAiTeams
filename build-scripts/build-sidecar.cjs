// build-scripts/build-sidecar.cjs — node entry for `npm run build:sidecar`.
//
// 1. Delegates PyInstaller to the platform-appropriate build-sidecar script
//    in the same folder (writes backend/dist/server/).
// 2. Mirrors that output to branding/sidecar-bundle/ so electron-builder's
//    extraResources rule can package it. This step lives here (not in the
//    .bat/.sh) so `npm run dist` works on any OS without a separate copy step.

const { spawnSync } = require("node:child_process");
const { resolve } = require("node:path");
const { existsSync, rmSync, mkdirSync, cpSync } = require("node:fs");

const isWindows = process.platform === "win32";

const script = isWindows
  ? resolve(__dirname, "build-sidecar.bat")
  : resolve(__dirname, "build-sidecar.sh");

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

const exitCode = result.status ?? (result.signal ? 1 : 0);
if (exitCode !== 0) {
  process.exit(exitCode);
}

// Mirror PyInstaller output → branding/sidecar-bundle/ for electron-builder.
const projectRoot = resolve(__dirname, "..");
const pyinstallerOut = resolve(projectRoot, "backend", "dist", "server");
const bundleDir = resolve(projectRoot, "branding", "sidecar-bundle");

if (!existsSync(pyinstallerOut)) {
  console.error(`build-sidecar: expected PyInstaller output at ${pyinstallerOut} but it does not exist`);
  process.exit(1);
}

const exeName = isWindows ? "server.exe" : "server";
const expectedBinary = resolve(pyinstallerOut, exeName);
if (!existsSync(expectedBinary)) {
  console.error(`build-sidecar: PyInstaller did not produce ${expectedBinary}`);
  process.exit(1);
}

if (existsSync(bundleDir)) {
  rmSync(bundleDir, { recursive: true, force: true });
}
mkdirSync(bundleDir, { recursive: true });
cpSync(pyinstallerOut, bundleDir, { recursive: true });

console.log(`build-sidecar: mirrored ${pyinstallerOut} → ${bundleDir}`);
