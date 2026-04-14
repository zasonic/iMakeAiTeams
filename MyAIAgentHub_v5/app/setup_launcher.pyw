"""
setup_launcher.pyw
==================
Called by the START_HERE launcher scripts.
Runs with pythonw — NO terminal window ever appears.

What it does on first run:
  1. Shows a dark-themed setup window
  2. Creates a .venv (or detects portable Python and skips this)
  3. pip-installs requirements.txt with real progress
  4. Pre-downloads the embedding model
  5. Launches main.py, then exits

On subsequent runs the environment already exists so setup is
skipped and the app launches immediately with a brief splash.
"""

import os
import sys
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.resolve()
VENV     = ROOT / ".venv"
REQS     = ROOT / "requirements.txt"
MAIN     = ROOT / "main.py"
PORTABLE = ROOT / ".python" / ("python.exe" if sys.platform == "win32" else "python")
MARKER   = ROOT / ".setup_done"

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

if PORTABLE.exists():
    MODE       = "portable"
    RUNTIME_PY = PORTABLE
else:
    MODE       = "venv"
    RUNTIME_PY = VENV / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python")

def _needs_setup() -> bool:
    return not MARKER.exists() if MODE == "portable" else not RUNTIME_PY.exists()

# ── Design tokens ──────────────────────────────────────────────────────────────
BG        = "#0a0a0c"
SURFACE   = "#131316"
BORDER    = "#222228"
ACCENT    = "#8b7cf6"
ACCENT2   = "#a78bfa"
TEXT      = "#ececf0"
TEXT2     = "#9d9db0"
TEXT3     = "#5c5c72"
GREEN     = "#3dd68c"
RED       = "#f0564a"

WIN_W, WIN_H = 480, 360

# ── Font stack ─────────────────────────────────────────────────────────────────
if IS_MAC:
    FONT     = "SF Pro Display"
    FONT_M   = "SF Mono"
else:
    FONT     = "Segoe UI"
    FONT_M   = "Consolas"

# ── Root window ────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("MyAI Agent Hub")
root.geometry(f"{WIN_W}x{WIN_H}")
root.resizable(False, False)
root.configure(bg=BG)
root.update_idletasks()
x = (root.winfo_screenwidth()  - WIN_W) // 2
y = (root.winfo_screenheight() - WIN_H) // 2
root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

if IS_WIN:
    root.overrideredirect(True)
    _drag = {"x": 0, "y": 0}
    def _ds(e): _drag["x"] = e.x; _drag["y"] = e.y
    def _dm(e): root.geometry(f"+{root.winfo_x()+e.x-_drag['x']}+{root.winfo_y()+e.y-_drag['y']}")
    root.bind("<ButtonPress-1>", _ds)
    root.bind("<B1-Motion>", _dm)

# ── Canvas ─────────────────────────────────────────────────────────────────────
c = tk.Canvas(root, width=WIN_W, height=WIN_H, bg=BG, highlightthickness=0, bd=0)
c.place(x=0, y=0)

# Outer border
c.create_rectangle(0, 0, WIN_W-1, WIN_H-1, outline=BORDER, width=1)

# ── Header ─────────────────────────────────────────────────────────────────────
# Accent gradient bar at top
for i in range(3):
    alpha = 1.0 - i * 0.3
    r = int(0x8b * alpha + 0x0a * (1 - alpha))
    g = int(0x7c * alpha + 0x0a * (1 - alpha))
    b = int(0xf6 * alpha + 0x0c * (1 - alpha))
    c.create_line(0, i, WIN_W, i, fill=f"#{r:02x}{g:02x}{b:02x}")

# App name — centered, clean
c.create_text(
    WIN_W // 2, 50,
    text="MyAI Agent Hub",
    fill=TEXT, font=(FONT, 20, "bold"), anchor="center",
)
c.create_text(
    WIN_W // 2, 76,
    text="Setting up your workspace",
    fill=TEXT3, font=(FONT, 11), anchor="center",
)

# Divider
c.create_line(40, 100, WIN_W - 40, 100, fill=BORDER, width=1)

# ── Status area ────────────────────────────────────────────────────────────────
_status_id = c.create_text(
    WIN_W // 2, 140,
    text="Preparing...", fill=TEXT, font=(FONT, 13), anchor="center",
)
_detail_id = c.create_text(
    WIN_W // 2, 164,
    text="", fill=TEXT3, font=(FONT, 10), anchor="center",
)

# ── Step indicators ────────────────────────────────────────────────────────────
_STEPS = ["Environment", "Packages", "AI Model", "Launch"]
_step_ids = []
_step_y = 200
_step_spacing = 90
_step_start_x = (WIN_W - (_step_spacing * (len(_STEPS) - 1))) // 2

for i, label in enumerate(_STEPS):
    sx = _step_start_x + i * _step_spacing
    # Circle
    cid = c.create_oval(sx - 12, _step_y - 12, sx + 12, _step_y + 12,
                        fill=SURFACE, outline=BORDER, width=1)
    # Number inside
    nid = c.create_text(sx, _step_y, text=str(i + 1), fill=TEXT3, font=(FONT, 9, "bold"))
    # Label below
    lid = c.create_text(sx, _step_y + 26, text=label, fill=TEXT3, font=(FONT, 9), anchor="center")
    # Connector line (except last)
    if i < len(_STEPS) - 1:
        nx = _step_start_x + (i + 1) * _step_spacing
        connector = c.create_line(sx + 14, _step_y, nx - 14, _step_y, fill=BORDER, width=1)
        _step_ids.append((cid, nid, lid, connector))
    else:
        _step_ids.append((cid, nid, lid, None))

def _set_step(idx, state="active"):
    """state: 'done', 'active', 'pending'"""
    for i, (cid, nid, lid, conn) in enumerate(_step_ids):
        if i < idx:
            c.itemconfigure(cid, fill=GREEN, outline=GREEN)
            c.itemconfigure(nid, text="\u2713", fill="#0a0a0c")
            c.itemconfigure(lid, fill=GREEN)
            if conn:
                c.itemconfigure(conn, fill=GREEN)
        elif i == idx:
            c.itemconfigure(cid, fill=ACCENT, outline=ACCENT)
            c.itemconfigure(nid, fill="#fff")
            c.itemconfigure(lid, fill=TEXT)
            if conn:
                c.itemconfigure(conn, fill=BORDER)
        else:
            c.itemconfigure(cid, fill=SURFACE, outline=BORDER)
            c.itemconfigure(nid, fill=TEXT3)
            c.itemconfigure(lid, fill=TEXT3)
            if conn:
                c.itemconfigure(conn, fill=BORDER)

# ── Progress bar ───────────────────────────────────────────────────────────────
_BAR_X = 40
_BAR_Y = 260
_BAR_W = WIN_W - 80
_BAR_H = 6
_BAR_R = 3

# Track (rounded via overlapping elements)
c.create_rectangle(_BAR_X, _BAR_Y, _BAR_X + _BAR_W, _BAR_Y + _BAR_H, fill=BORDER, outline="")
# Fill
_bar_fill = c.create_rectangle(_BAR_X, _BAR_Y, _BAR_X, _BAR_Y + _BAR_H, fill=ACCENT, outline="")

# Percentage text
_pct_id = c.create_text(
    WIN_W // 2, _BAR_Y + 20,
    text="", fill=TEXT3, font=(FONT, 9), anchor="center",
)

# Error text (hidden)
_error_id = c.create_text(
    WIN_W // 2, 310,
    text="", fill=RED, font=(FONT, 10),
    width=WIN_W - 80, anchor="center", justify="center",
)

# Retry button (hidden)
_BTN_W, _BTN_H = 120, 32
_BTN_X = (WIN_W - _BTN_W) // 2
_BTN_Y = 316
_retry_bg = c.create_rectangle(_BTN_X, _BTN_Y, _BTN_X + _BTN_W, _BTN_Y + _BTN_H,
                                fill=ACCENT, outline="", state="hidden")
_retry_text = c.create_text(_BTN_X + _BTN_W // 2, _BTN_Y + _BTN_H // 2,
                             text="Retry", fill="#fff", font=(FONT, 10, "bold"), state="hidden")

def _show_retry():
    c.itemconfigure(_retry_bg, state="normal")
    c.itemconfigure(_retry_text, state="normal")
def _hide_retry():
    c.itemconfigure(_retry_bg, state="hidden")
    c.itemconfigure(_retry_text, state="hidden")

c.tag_bind(_retry_bg, "<Button-1>", lambda _: _start_setup())
c.tag_bind(_retry_text, "<Button-1>", lambda _: _start_setup())
c.tag_bind(_retry_bg, "<Enter>", lambda _: c.itemconfigure(_retry_bg, fill=ACCENT2))
c.tag_bind(_retry_bg, "<Leave>", lambda _: c.itemconfigure(_retry_bg, fill=ACCENT))

# ── Thread-safe UI helpers ─────────────────────────────────────────────────────
def _set_bar(pct):
    w = int(_BAR_W * max(0.0, min(1.0, pct / 100.0)))
    c.coords(_bar_fill, _BAR_X, _BAR_Y, _BAR_X + w, _BAR_Y + _BAR_H)

def _ui(status=None, detail=None, pct=None, step=None):
    def _do():
        if status is not None:
            c.itemconfigure(_status_id, text=status, fill=TEXT)
        if detail is not None:
            c.itemconfigure(_detail_id, text=detail)
        if pct is not None:
            _set_bar(pct)
            c.itemconfigure(_pct_id, text=f"{int(pct)}%")
        if step is not None:
            _set_step(step)
    root.after(0, _do)

def _fail(msg):
    def _do():
        c.itemconfigure(_status_id, text="Setup failed", fill=RED)
        c.itemconfigure(_detail_id, text="")
        c.itemconfigure(_error_id, text=msg)
        c.itemconfigure(_pct_id, text="")
        _set_bar(0)
        _show_retry()
    root.after(0, _do)

# ── Indeterminate sweep for quick launches ─────────────────────────────────────
_sweep_pos = 0.0
_sweep_dir = 1
_sweep_timer = None

def _tick_sweep():
    global _sweep_pos, _sweep_dir, _sweep_timer
    _sweep_pos += _sweep_dir * 0.025
    if _sweep_pos >= 0.75: _sweep_dir = -1
    elif _sweep_pos <= 0.0: _sweep_dir = 1
    x1 = _BAR_X + int(_BAR_W * _sweep_pos)
    x2 = _BAR_X + int(_BAR_W * min(1.0, _sweep_pos + 0.25))
    c.coords(_bar_fill, x1, _BAR_Y, x2, _BAR_Y + _BAR_H)
    _sweep_timer = root.after(20, _tick_sweep)

def _stop_sweep():
    global _sweep_timer
    if _sweep_timer:
        root.after_cancel(_sweep_timer)
        _sweep_timer = None
    c.coords(_bar_fill, _BAR_X, _BAR_Y, _BAR_X, _BAR_Y + _BAR_H)

# ── Launch app ─────────────────────────────────────────────────────────────────
def _launch_app():
    _stop_sweep()
    flags = {}
    if IS_WIN:
        flags["creationflags"] = subprocess.DETACHED_PROCESS | 0x08000000
    else:
        flags["start_new_session"] = True
    subprocess.Popen([str(RUNTIME_PY), str(MAIN)], cwd=str(ROOT), **flags)
    root.after(500, root.destroy)

# ── Setup logic ────────────────────────────────────────────────────────────────
def _run_setup():
    try:
        root.after(0, _hide_retry)
        root.after(0, lambda: c.itemconfigure(_error_id, text=""))

        # Step 1: Environment
        if MODE == "venv" and not RUNTIME_PY.exists():
            _ui(status="Creating Python environment...",
                detail="This only happens once", pct=5, step=0)
            result = subprocess.run(
                [sys.executable, "-m", "venv", str(VENV)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create virtual environment.\n{result.stderr.strip()}")

        _ui(status="Upgrading pip...", detail="", pct=10, step=0)
        py = str(RUNTIME_PY if MODE == "venv" else PORTABLE)
        subprocess.run([py, "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
                       capture_output=True)

        # Step 2: Packages
        _ui(step=1)
        _install_requirements(py)

        # Step 3: Embedding model
        _ui(status="Downloading AI model...",
            detail="Embedding model for document search (~90 MB)",
            pct=92, step=2)
        try:
            subprocess.run(
                [py, "-c",
                 "from sentence_transformers import SentenceTransformer; "
                 "SentenceTransformer('all-MiniLM-L6-v2')"],
                capture_output=True, timeout=300,
            )
            _ui(detail="Model ready", pct=96)
        except Exception:
            _ui(detail="Model will download on first use", pct=96)

        # Step 4: Done
        if MODE == "portable":
            MARKER.write_text("ok", encoding="utf-8")

        _ui(status="Ready — opening Agent Hub...",
            detail="", pct=100, step=3)
        root.after(800, _launch_app)

    except Exception as exc:
        _fail(str(exc))


def _install_requirements(py_exe):
    try:
        with open(str(REQS)) as f:
            total = sum(1 for ln in f if ln.strip() and not ln.strip().startswith("#")
                        and not ln.strip().startswith("--"))
    except Exception:
        total = 13

    _ui(status="Installing packages...",
        detail=f"0 of {total} packages", pct=15, step=1)

    proc = subprocess.Popen(
        [py_exe, "-m", "pip", "install", "-r", str(REQS), "--progress-bar", "off"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(ROOT),
    )
    installed = 0
    start_time = time.time()

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("Collecting", "Installing", "Downloading", "Building")):
            pkg = line.split()[-1].split("==")[0].split(">=")[0].split("<")[0]
            if len(pkg) > 35:
                pkg = pkg[:32] + "..."
            if line.startswith(("Collecting", "Installing")):
                installed += 1
            ratio = min(installed / max(total, 1), 1.0)
            pct = 15 + ratio * 73  # 15% to 88%
            elapsed = time.time() - start_time
            if installed > 2 and ratio > 0:
                eta = int((elapsed / ratio) * (1 - ratio))
                eta_s = f"{eta // 60}m {eta % 60}s left" if eta >= 60 else f"{eta}s left"
            else:
                eta_s = "estimating..."
            _ui(status=f"Installing packages... ({installed}/{total})",
                detail=f"{pkg}  \u00b7  {eta_s}", pct=pct)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("Package installation failed.\nCheck your internet connection and try again.")
    _ui(status="Packages installed", detail="", pct=90, step=1)


def _start_setup():
    threading.Thread(target=_run_setup, daemon=True).start()


def _main():
    if not _needs_setup():
        c.itemconfigure(_status_id, text="Opening Agent Hub...")
        c.itemconfigure(_detail_id, text="")
        _set_step(3)
        _tick_sweep()
        root.after(400, _launch_app)
    else:
        _start_setup()


root.after(120, _main)
root.mainloop()
