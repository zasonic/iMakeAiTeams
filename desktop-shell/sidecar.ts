// desktop-shell/sidecar.ts — manages the Python FastAPI sidecar process.
//
// Responsibilities:
//   1. Spawn the Python sidecar (dev: backend/.venv/Scripts/python.exe →
//      system python; prod: <process.resourcesPath>/backend/server.exe —
//      the "resources" segment is Electron's own packaged-app convention,
//      unrelated to this repo's source layout)
//   2. Pass `--token <uuid>` and `--user-data <path>` so the sidecar binds
//      a free port, prints PORT=<n>, then prints READY when serving.
//   3. Read stdout line-by-line until PORT= appears (15s timeout), then
//      poll GET /health every 500ms (30 retries) before declaring ready.
//   4. Pipe stdout/stderr to `{userData}/sidecar.log` with 10MB rotation.
//   5. Emit lifecycle events ('starting' | 'ready' | 'crashed' | 'stopped')
//      for the main process to forward to the renderer over IPC.
//   6. On graceful shutdown: POST /shutdown then wait 3s; if still alive,
//      taskkill /f /t (Windows) or SIGKILL.
//
// All sidecar binds happen on 127.0.0.1 — never 0.0.0.0.

import { app } from "electron";
import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { createWriteStream, existsSync, mkdirSync, statSync, renameSync, WriteStream } from "node:fs";
import { EventEmitter } from "node:events";
import { join } from "node:path";

const HEALTH_POLL_INTERVAL_MS = 500;
const HEALTH_POLL_MAX_RETRIES = 30;          // 30 * 500ms = 15s
const PORT_LINE_TIMEOUT_MS = 15_000;
const SIDECAR_LOG_MAX_BYTES = 10 * 1024 * 1024; // 10MB
const SHUTDOWN_GRACE_MS = 3_000;
// Caps on the in-memory pipe buffers. The stdout buffer accumulates
// between newlines; the stderr buffer holds the tail used for the
// crash-summary status. Without a cap, a misbehaving sidecar that
// writes a multi-MB blob without a newline would pin all of it in
// memory.
const STDOUT_BUFFER_MAX_BYTES = 256 * 1024;  // 256 KiB
const STDERR_BUFFER_MAX_BYTES = 64 * 1024;   // 64 KiB (tail-only)

function redactArgs(args: readonly string[]): string[] {
  // Mask --token <value> and --token=<value> so sidecar.log never contains
  // the bearer secret. Defensive against future flag aliases.
  const SECRET_FLAGS = new Set(["--token", "--auth-token"]);
  return args.map((arg, i) => {
    const eq = arg.indexOf("=");
    if (eq > 0 && SECRET_FLAGS.has(arg.slice(0, eq))) {
      return `${arg.slice(0, eq)}=***`;
    }
    if (i > 0 && SECRET_FLAGS.has(args[i - 1])) {
      return "***";
    }
    return arg;
  });
}

export type SidecarStatus =
  | { status: "starting" }
  | { status: "ready"; port: number; token: string }
  | { status: "crashed"; code: number | null; signal: NodeJS.Signals | null; error?: string }
  | { status: "stopped" };

export interface SidecarInfo {
  port: number;
  token: string;
}

export class SidecarManager extends EventEmitter {
  private child: ChildProcessWithoutNullStreams | null = null;
  private port: number | null = null;
  private readonly token: string;
  private logStream: WriteStream | null = null;
  private readonly logPath: string;
  private stdoutBuffer = "";
  private stderrBuffer = "";
  private explicitShutdown = false;
  private startPromise: Promise<SidecarInfo> | null = null;
  private stopPromise: Promise<void> | null = null;
  private resolveStart: ((info: SidecarInfo) => void) | null = null;
  private rejectStart: ((err: Error) => void) | null = null;

  constructor(private readonly projectRoot: string, private readonly userDataDir: string) {
    super();
    this.token = randomUUID();
    if (!existsSync(this.userDataDir)) {
      mkdirSync(this.userDataDir, { recursive: true });
    }
    this.logPath = join(this.userDataDir, "sidecar.log");
  }

  getInfo(): SidecarInfo | null {
    if (this.port == null) return null;
    return { port: this.port, token: this.token };
  }

  /** Start (or restart) the sidecar. Resolves once /health returns 200. */
  async start(): Promise<SidecarInfo> {
    if (this.startPromise) return this.startPromise;

    this.explicitShutdown = false;
    this.rotateLogIfLarge();
    this.logStream = createWriteStream(this.logPath, { flags: "a" });
    this.logToFile(`\n=== sidecar starting at ${new Date().toISOString()} ===\n`);

    this.emit("status", { status: "starting" } satisfies SidecarStatus);

    this.startPromise = new Promise<SidecarInfo>((resolve, reject) => {
      this.resolveStart = resolve;
      this.rejectStart = reject;
    });

    try {
      const { command, args } = this.resolveSpawnArgs();
      this.logToFile(`spawn: ${command} ${redactArgs(args).join(" ")}\n`);

      this.child = spawn(command, args, {
        cwd: this.sidecarCwd(),
        env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
        windowsHide: true,
      });

      this.child.stdout.on("data", (chunk) => this.handleStdout(chunk.toString("utf-8")));
      this.child.stderr.on("data", (chunk) => this.handleStderr(chunk.toString("utf-8")));
      this.child.on("error", (err) => this.handleSpawnError(err));
      this.child.on("exit", (code, signal) => this.handleExit(code, signal));

      // Hard timeout for the PORT= line.
      const portTimeout = setTimeout(() => {
        if (this.port == null) {
          const msg = `Sidecar did not announce a port within ${PORT_LINE_TIMEOUT_MS}ms`;
          this.logToFile(`ERROR: ${msg}\n`);
          this.failStart(new Error(msg));
        }
      }, PORT_LINE_TIMEOUT_MS);

      this.startPromise.finally(() => clearTimeout(portTimeout));
    } catch (err) {
      this.failStart(err as Error);
    }

    return this.startPromise;
  }

  /** Gracefully stop the sidecar; falls back to SIGKILL after SHUTDOWN_GRACE_MS.
   *  Reentrant: concurrent calls share the same in-flight teardown. */
  async stop(): Promise<void> {
    if (this.stopPromise) return this.stopPromise;

    this.explicitShutdown = true;
    const child = this.child;
    if (!child) return;

    this.stopPromise = (async () => {
      const info = this.getInfo();
      if (info) {
        try {
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), 1000);
          await fetch(`http://127.0.0.1:${info.port}/shutdown`, {
            method: "POST",
            headers: { Authorization: `Bearer ${info.token}` },
            signal: controller.signal,
          }).catch(() => {});
          clearTimeout(timeout);
        } catch {
          /* sidecar already gone — no-op */
        }
      }

      await new Promise<void>((resolve) => {
        const timer = setTimeout(() => {
          if (!child.killed) {
            this.logToFile("graceful shutdown timed out; forcing kill\n");
            if (process.platform === "win32" && child.pid != null) {
              spawn("taskkill", ["/pid", String(child.pid), "/f", "/t"], {
                windowsHide: true,
              });
            } else {
              child.kill("SIGKILL");
            }
          }
        }, SHUTDOWN_GRACE_MS);
        child.once("exit", () => {
          clearTimeout(timer);
          resolve();
        });
      });
    })().finally(() => {
      this.stopPromise = null;
    });

    return this.stopPromise;
  }

  /** Restart the sidecar (kill, wait, spawn). */
  async restart(): Promise<SidecarInfo> {
    await this.stop();
    this.child = null;
    this.port = null;
    this.startPromise = null;
    this.resolveStart = null;
    this.rejectStart = null;
    return this.start();
  }

  // ── private helpers ────────────────────────────────────────────────────────

  private resolveSpawnArgs(): { command: string; args: string[] } {
    const args = ["--token", this.token, "--user-data", this.userDataDir];
    if (app.isPackaged) {
      // <process.resourcesPath>/backend/server.exe (PyInstaller onedir).
      // process.resourcesPath is Electron's runtime path inside the packaged
      // app — distinct from the repo's source `branding/` folder.
      const exe =
        process.platform === "win32"
          ? "server.exe"
          : "server";
      const command = join(process.resourcesPath, "backend", exe);
      return { command, args };
    }
    // Dev: prefer the project venv, fall back to system python.
    const venvPython =
      process.platform === "win32"
        ? join(this.projectRoot, "backend", ".venv", "Scripts", "python.exe")
        : join(this.projectRoot, "backend", ".venv", "bin", "python");
    const command = existsSync(venvPython) ? venvPython : process.platform === "win32" ? "python" : "python3";
    return { command, args: ["server.py", ...args] };
  }

  private sidecarCwd(): string {
    if (app.isPackaged) {
      return join(process.resourcesPath, "backend");
    }
    return join(this.projectRoot, "backend");
  }

  private handleStdout(chunk: string): void {
    this.stdoutBuffer += chunk;
    this.logToFile(chunk);

    // Cap the buffer so a sidecar that emits a giant blob without a
    // newline can't pin unbounded memory. We trim from the head and
    // keep the tail since the only line we care about (PORT=...) is
    // recent.
    if (this.stdoutBuffer.length > STDOUT_BUFFER_MAX_BYTES) {
      this.stdoutBuffer = this.stdoutBuffer.slice(-STDOUT_BUFFER_MAX_BYTES);
    }

    while (true) {
      const newlineIdx = this.stdoutBuffer.indexOf("\n");
      if (newlineIdx < 0) break;
      const line = this.stdoutBuffer.slice(0, newlineIdx).trimEnd();
      this.stdoutBuffer = this.stdoutBuffer.slice(newlineIdx + 1);

      if (line.startsWith("PORT=")) {
        const parsed = Number.parseInt(line.slice(5), 10);
        if (Number.isFinite(parsed) && parsed > 0 && parsed < 65536) {
          this.port = parsed;
          // Don't resolve yet — wait for /health to come up.
          this.pollHealth().catch((err) => this.failStart(err));
        }
      }
    }
  }

  private handleStderr(chunk: string): void {
    this.stderrBuffer += chunk;
    // Only the last few KB matter (used for the crash-tail status
    // message). Trim aggressively so a runaway log can't grow forever.
    if (this.stderrBuffer.length > STDERR_BUFFER_MAX_BYTES) {
      this.stderrBuffer = this.stderrBuffer.slice(-STDERR_BUFFER_MAX_BYTES);
    }
    this.logToFile(chunk, "stderr");
  }

  private handleSpawnError(err: Error): void {
    this.logToFile(`spawn error: ${err.message}\n`);
    // When `child.on("error")` fires (vs spawn() throwing) the process
    // never started, but `this.child` is still the dead handle. Drop it
    // here so a later stop() doesn't await `child.once("exit")` forever
    // — exit never fires because the child never started.
    this.child = null;
    this.port = null;
    this.failStart(err);
  }

  private handleExit(code: number | null, signal: NodeJS.Signals | null): void {
    this.logToFile(`exit code=${code} signal=${signal}\n`);

    if (!this.explicitShutdown) {
      const tail = this.stderrBuffer.split("\n").slice(-20).join("\n").trim();
      this.emit("status", {
        status: "crashed",
        code,
        signal,
        error: tail || `Sidecar exited unexpectedly (code=${code}, signal=${signal})`,
      } satisfies SidecarStatus);
      if (this.rejectStart) {
        this.failStart(new Error(`Sidecar exited before becoming ready (code=${code})`));
      }
    } else {
      this.emit("status", { status: "stopped" } satisfies SidecarStatus);
    }

    this.child = null;
    this.port = null;
    // Drop the cached startPromise so a post-crash start() can build a fresh
    // one. Without this, a previously-rejected promise would be returned to
    // every subsequent caller, breaking auto-recovery.
    this.startPromise = null;
    if (this.logStream) {
      this.logStream.end();
      this.logStream = null;
    }
  }

  private async pollHealth(): Promise<void> {
    if (this.port == null) {
      throw new Error("pollHealth called before PORT was known");
    }
    const url = `http://127.0.0.1:${this.port}/health`;
    let lastErr: unknown = null;

    for (let attempt = 0; attempt < HEALTH_POLL_MAX_RETRIES; attempt++) {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 800);
        const resp = await fetch(url, { signal: controller.signal });
        clearTimeout(timeout);
        if (resp.ok) {
          const info: SidecarInfo = { port: this.port, token: this.token };
          this.emit("status", { status: "ready", ...info } satisfies SidecarStatus);
          this.resolveStart?.(info);
          this.resolveStart = null;
          this.rejectStart = null;
          return;
        }
        lastErr = new Error(`health returned ${resp.status}`);
      } catch (err) {
        lastErr = err;
      }
      await new Promise((r) => setTimeout(r, HEALTH_POLL_INTERVAL_MS));
    }

    throw new Error(
      `Sidecar /health never responded after ${HEALTH_POLL_MAX_RETRIES} retries (last: ${
        lastErr instanceof Error ? lastErr.message : String(lastErr)
      })`,
    );
  }

  private failStart(err: Error): void {
    if (this.rejectStart) {
      this.rejectStart(err);
      this.rejectStart = null;
      this.resolveStart = null;
    }
    this.emit("status", {
      status: "crashed",
      code: null,
      signal: null,
      error: err.message,
    } satisfies SidecarStatus);
  }

  private logToFile(text: string, _stream: "stdout" | "stderr" = "stdout"): void {
    if (!this.logStream) return;
    try {
      this.logStream.write(text);
    } catch {
      // Logging failure must never crash the manager.
    }
  }

  private rotateLogIfLarge(): void {
    try {
      if (!existsSync(this.logPath)) return;
      const size = statSync(this.logPath).size;
      if (size <= SIDECAR_LOG_MAX_BYTES) return;
      const archived = `${this.logPath}.1`;
      try {
        renameSync(this.logPath, archived);
      } catch {
        /* ignore — next write will create a fresh file */
      }
    } catch {
      /* ignore */
    }
  }
}
