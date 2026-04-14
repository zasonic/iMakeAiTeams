"""
setup_launcher.pyw
==================
Called by the START_HERE launcher scripts.
Runs with pythonw — NO terminal window ever appears.

What it does on first run:
  1. Shows a dark-themed setup window with canvas-drawn progress
  2. Creates a .venv (or detects portable Python and skips this)
  3. pip-installs requirements.txt
  4. Launches main.py, then exits

On subsequent runs the environment already exists so setup is
skipped and the app launches in under a second (indeterminate sweep).

Design notes:
  - Zero ttk usage — ttk lets the OS theme bleed in regardless of colors.
  - All progress bars, dividers, and accents are drawn on tk.Canvas.
  - Amber (#f5a623) accent matches the main app's design system.
  - No emoji icons — canvas-drawn logo mark instead.
  - Outfit font with Segoe UI / SF Pro Display fallback.
"""

import os
import sys
import subprocess
import threading
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
BG        = "#0a0a0c"   # match app background
SURFACE   = "#111114"   # card surface
BORDER    = "#2a2a33"   # divider
ACCENT    = "#8b7cf6"   # purple accent — matches main app
ACCENT_DIM = "#0f0d1f"  # accent wash
TEXT      = "#ececf0"   # primary text
TEXT_2    = "#9d9db0"   # secondary text
TEXT_3    = "#5c5c72"   # muted text
GREEN     = "#3dd68c"
RED       = "#f0564a"

WIN_W, WIN_H = 460, 310

# ── Font stack ─────────────────────────────────────────────────────────────────
if IS_MAC:
    F_TITLE  = ("SF Pro Display", 16, "bold")
    F_BODY   = ("SF Pro Display", 12)
    F_SMALL  = ("SF Pro Display", 10)
    F_MONO   = ("Menlo",          10)
else:
    F_TITLE  = ("Segoe UI",       16, "bold")
    F_BODY   = ("Segoe UI",       12)
    F_SMALL  = ("Segoe UI",       10)
    F_MONO   = ("Consolas",       10)

# ── Root window ───────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("MyAI Agent Hub")
root.geometry(f"{WIN_W}x{WIN_H}")
root.resizable(False, False)
root.configure(bg=BG)

# Centre on screen
root.update_idletasks()
x = (root.winfo_screenwidth()  - WIN_W) // 2
y = (root.winfo_screenheight() - WIN_H) // 2
root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

# Remove title bar chrome on Windows for a cleaner look (optional — comment out
# if you want the standard OS chrome)
if IS_WIN:
    root.overrideredirect(True)
    # Re-enable dragging since we removed the title bar
    _drag_data = {"x": 0, "y": 0}
    def _on_drag_start(e): _drag_data["x"] = e.x; _drag_data["y"] = e.y
    def _on_drag(e):
        dx = e.x - _drag_data["x"]; dy = e.y - _drag_data["y"]
        root.geometry(f"+{root.winfo_x()+dx}+{root.winfo_y()+dy}")
    root.bind("<ButtonPress-1>",   _on_drag_start)
    root.bind("<B1-Motion>",       _on_drag)

# ── Canvas logo mark (amber square with white grid — no emoji) ────────────────
def _draw_logo(canvas: tk.Canvas, x: int, y: int, size: int = 32):
    """Draw a minimal geometric logo mark: amber rounded square + white AI lines."""
    r = 6  # corner radius approximation via polygon
    s = size
    # Background square (amber)
    canvas.create_rectangle(x, y, x+s, y+s, fill=ACCENT, outline="", tags="logo")
    # Three horizontal lines (representing neural layers / text lines) — dark
    lc = "#0a0600"
    for i, frac in enumerate([0.30, 0.50, 0.70]):
        ly = y + int(s * frac)
        lw = s - 12 if i == 1 else s - 8   # middle line shorter
        lx = x + (s - lw) // 2
        canvas.create_line(lx, ly, lx+lw, ly, fill=lc, width=2, tags="logo")

# ── Main canvas (replaces all tkinter widgets for the visual frame) ───────────
c = tk.Canvas(root, width=WIN_W, height=WIN_H, bg=BG, highlightthickness=0)
c.place(x=0, y=0)

# Card background
_PAD = 24
c.create_rectangle(
    _PAD, _PAD, WIN_W - _PAD, WIN_H - _PAD,
    fill=SURFACE, outline=BORDER, width=1,
)

# Logo mark
_draw_logo(c, x=_PAD + 20, y=_PAD + 20, size=32)

# App name
c.create_text(
    _PAD + 64, _PAD + 28,
    text="MyAI Agent Hub",
    fill=TEXT, font=F_TITLE,
    anchor="w",
)
c.create_text(
    _PAD + 64, _PAD + 48,
    text="Local AI Workspace",
    fill=TEXT_3, font=F_SMALL,
    anchor="w",
)

# Thin amber accent line under header
_DIVIDER_Y = _PAD + 68
c.create_line(_PAD + 1, _DIVIDER_Y, WIN_W - _PAD - 1, _DIVIDER_Y, fill=BORDER, width=1)

# ── Progress bar (fully canvas-drawn — no ttk) ─────────────────────────────────
_BAR_X  = _PAD + 20
_BAR_Y  = WIN_H - _PAD - 50
_BAR_W  = WIN_W - (_PAD * 2) - 40
_BAR_H  = 4

# Track
c.create_rectangle(_BAR_X, _BAR_Y, _BAR_X + _BAR_W, _BAR_Y + _BAR_H,
                   fill=BORDER, outline="")
# Fill (starts at 0 width)
_bar_fill = c.create_rectangle(_BAR_X, _BAR_Y, _BAR_X, _BAR_Y + _BAR_H,
                                fill=ACCENT, outline="")

def _set_bar(pct: float):
    """Set progress bar to pct (0–100). Thread-safe via root.after."""
    w = int(_BAR_W * max(0.0, min(1.0, pct / 100.0)))
    c.coords(_bar_fill, _BAR_X, _BAR_Y, _BAR_X + w, _BAR_Y + _BAR_H)

# ── Step pills ────────────────────────────────────────────────────────────────
_PILL_Y = _DIVIDER_Y + 22
_PILLS  = ["env", "packages", "done"]
_PILL_IDS: dict = {}   # label -> (bg_rect_id, text_id)
_PILL_COLORS = {
    "done":   (ACCENT_DIM, ACCENT),
    "active": ("#0d1a33",  "#60a5fa"),
    "todo":   (SURFACE,    TEXT_3),
}

_total_pill_w = 0
_pill_padding_x = 12
_pill_h = 20
for _p in _PILLS:
    _total_pill_w += len(_p) * 7 + _pill_padding_x * 2 + 8

_px = (WIN_W - _total_pill_w) // 2

for pill in _PILLS:
    pw = len(pill) * 7 + _pill_padding_x * 2
    bg = c.create_rectangle(_px, _PILL_Y, _px + pw, _PILL_Y + _pill_h,
                             fill=SURFACE, outline=BORDER, width=1)
    tx = c.create_text(_px + pw // 2, _PILL_Y + _pill_h // 2,
                       text=pill, fill=TEXT_3, font=F_SMALL)
    _PILL_IDS[pill] = (bg, tx)
    _px += pw + 8

def _set_pills(pills: list):
    """pills = [(label, state), ...].  States: 'done', 'active', 'todo'."""
    for lbl, state in pills:
        if lbl not in _PILL_IDS:
            continue
        bg_id, tx_id = _PILL_IDS[lbl]
        bg_color, fg_color = _PILL_COLORS.get(state, _PILL_COLORS["todo"])
        c.itemconfigure(bg_id, fill=bg_color)
        c.itemconfigure(tx_id, fill=fg_color)

# ── Status text ───────────────────────────────────────────────────────────────
_STATUS_Y = _PILL_Y + _pill_h + 22
_status_id = c.create_text(
    WIN_W // 2, _STATUS_Y,
    text="Preparing…", fill=TEXT, font=F_BODY, anchor="center",
)
_detail_id = c.create_text(
    WIN_W // 2, _STATUS_Y + 24,
    text="", fill=TEXT_3, font=F_SMALL, anchor="center",
)

# Error text (hidden by default)
_error_id = c.create_text(
    WIN_W // 2, _STATUS_Y + 52,
    text="", fill=RED, font=F_SMALL,
    width=WIN_W - (_PAD * 2) - 20, anchor="center", justify="center",
)

# ── Retry button (canvas button — only shown on failure) ──────────────────────
_BTN_W, _BTN_H = 100, 30
_BTN_X = (WIN_W - _BTN_W) // 2
_BTN_Y = _BAR_Y - 44
_retry_bg   = c.create_rectangle(_BTN_X, _BTN_Y, _BTN_X + _BTN_W, _BTN_Y + _BTN_H,
                                  fill=ACCENT, outline="", state="hidden")
_retry_text = c.create_text(_BTN_X + _BTN_W // 2, _BTN_Y + _BTN_H // 2,
                             text="Retry", fill="#0a0600", font=(*F_BODY[:2], "bold"),
                             state="hidden")

def _show_retry():
    c.itemconfigure(_retry_bg,   state="normal")
    c.itemconfigure(_retry_text, state="normal")

def _hide_retry():
    c.itemconfigure(_retry_bg,   state="hidden")
    c.itemconfigure(_retry_text, state="hidden")

c.tag_bind(_retry_bg,   "<Button-1>", lambda _e: _start_setup())
c.tag_bind(_retry_text, "<Button-1>", lambda _e: _start_setup())
c.tag_bind(_retry_bg,   "<Enter>",    lambda _e: c.itemconfigure(_retry_bg, fill="#f0b640"))
c.tag_bind(_retry_bg,   "<Leave>",    lambda _e: c.itemconfigure(_retry_bg, fill=ACCENT))

# ── Thread-safe UI update ─────────────────────────────────────────────────────
def _set(status=None, detail=None, pct=None, pills=None):
    def _do():
        if status is not None:
            c.itemconfigure(_status_id, text=status, fill=TEXT)
        if detail is not None:
            c.itemconfigure(_detail_id, text=detail)
        if pct is not None:
            _set_bar(pct)
        if pills is not None:
            _set_pills(pills)
    root.after(0, _do)

def _fail(msg: str):
    def _do():
        c.itemconfigure(_status_id, text="Setup failed",   fill=RED)
        c.itemconfigure(_error_id,  text=msg)
        _set_bar(0)
        _show_retry()
    root.after(0, _do)

# ── Indeterminate sweep for quick launches ────────────────────────────────────
_sweep_pos  = 0.0
_sweep_dir  = 1
_sweep_w    = 0.25   # sweep highlight width as fraction
_sweep_timer = None

def _tick_sweep():
    global _sweep_pos, _sweep_dir, _sweep_timer
    _sweep_pos += _sweep_dir * 0.03
    if _sweep_pos + _sweep_w >= 1.0:
        _sweep_dir = -1
    elif _sweep_pos <= 0.0:
        _sweep_dir = 1
    x1 = _BAR_X + int(_BAR_W * _sweep_pos)
    x2 = _BAR_X + int(_BAR_W * min(1.0, _sweep_pos + _sweep_w))
    c.coords(_bar_fill, x1, _BAR_Y, x2, _BAR_Y + _BAR_H)
    _sweep_timer = root.after(20, _tick_sweep)

def _stop_sweep():
    global _sweep_timer
    if _sweep_timer is not None:
        root.after_cancel(_sweep_timer)
        _sweep_timer = None
    c.coords(_bar_fill, _BAR_X, _BAR_Y, _BAR_X, _BAR_Y + _BAR_H)

# ── Launch app ────────────────────────────────────────────────────────────────
def _launch_app():
    _stop_sweep()
    if IS_WIN:
        _CREATE_NO_WINDOW = 0x08000000
        subprocess.Popen(
            [str(RUNTIME_PY), str(MAIN)],
            cwd=str(ROOT),
            creationflags=subprocess.DETACHED_PROCESS | _CREATE_NO_WINDOW,
        )
    else:
        subprocess.Popen(
            [str(RUNTIME_PY), str(MAIN)],
            cwd=str(ROOT),
            start_new_session=True,
        )
    root.after(700, root.destroy)

# ── Setup logic ───────────────────────────────────────────────────────────────
def _run_setup():
    try:
        root.after(0, _hide_retry)
        root.after(0, lambda: c.itemconfigure(_error_id, text=""))
        root.after(0, lambda: c.itemconfigure(_status_id, fill=TEXT))

        if MODE == "venv":
            _setup_venv()
        else:
            _setup_portable()

        _set(
            status="Setup complete — opening Agent Hub…",
            detail="",
            pct=100,
            pills=[("env", "done"), ("packages", "done"), ("done", "done")],
        )
        root.after(600, _launch_app)

    except Exception as exc:
        _fail(str(exc))


def _setup_venv():
    if not RUNTIME_PY.exists():
        _set(
            status="Setting up for the first time…",
            detail="Creating isolated Python environment",
            pct=5,
            pills=[("env", "active"), ("packages", "todo"), ("done", "todo")],
        )
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(VENV)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create virtual environment.\n{result.stderr.strip()}"
            )

    _set(pct=15, detail="Upgrading pip…",
         pills=[("env", "done"), ("packages", "active"), ("done", "todo")])
    subprocess.run(
        [str(RUNTIME_PY), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        capture_output=True,
    )
    _install_requirements(str(RUNTIME_PY))


def _setup_portable():
    _set(
        status="Setting up for the first time…",
        detail="Preparing portable Python",
        pct=5,
        pills=[("env", "active"), ("packages", "todo"), ("done", "todo")],
    )
    _set(pct=10, detail="Upgrading pip…",
         pills=[("env", "done"), ("packages", "active"), ("done", "todo")])
    subprocess.run(
        [str(PORTABLE), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        capture_output=True,
    )
    _install_requirements(str(PORTABLE))
    MARKER.write_text("ok", encoding="utf-8")


def _install_requirements(py_exe: str):
    import time as _time

    # Count total packages from requirements.txt for accurate progress
    try:
        with open(str(REQS), "r") as f:
            total_pkgs = sum(1 for ln in f if ln.strip() and not ln.strip().startswith("#"))
    except Exception:
        total_pkgs = 9  # fallback estimate

    _set(
        status="Installing packages…",
        detail=f"Preparing to install {total_pkgs} packages (~1.5 GB total)",
        pct=20,
        pills=[("env", "done"), ("packages", "active"), ("done", "todo")],
    )
    proc = subprocess.Popen(
        [py_exe, "-m", "pip", "install", "-r", str(REQS), "--progress-bar", "off"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(ROOT),
    )
    installed = 0
    start_time = _time.time()

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("Collecting", "Installing", "Downloading", "Building")):
            # Extract package name (last word, strip version specifiers)
            pkg = line.split()[-1].split("==")[0].split(">=")[0].split("<")[0]
            if len(pkg) > 30:
                pkg = pkg[:27] + "…"

            if line.startswith(("Collecting", "Installing")):
                installed += 1

            # Calculate progress (20% to 85% range)
            ratio = min(installed / max(total_pkgs, 1), 1.0)
            pct = 20 + ratio * 65

            # Calculate ETA
            elapsed = _time.time() - start_time
            if installed > 1 and ratio > 0:
                eta_secs = int((elapsed / ratio) * (1 - ratio))
                if eta_secs >= 60:
                    eta_str = f"~{eta_secs // 60}m {eta_secs % 60}s remaining"
                else:
                    eta_str = f"~{eta_secs}s remaining"
            else:
                eta_str = "Estimating time…"

            _set(
                status=f"Installing packages ({installed}/{total_pkgs})",
                detail=f"{pkg}  ·  {eta_str}",
                pct=pct,
            )

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            "Package installation failed.\n"
            "Check your internet connection and try again."
        )
    _set(pct=90, detail="All packages installed",
         pills=[("env", "done"), ("packages", "done"), ("done", "active")])

    # Pre-download the embedding model so first launch doesn't hang
    _set(detail="Downloading AI embedding model (~90 MB)…", pct=92)
    try:
        subprocess.run(
            [py_exe, "-c",
             "from sentence_transformers import SentenceTransformer; "
             "SentenceTransformer('all-MiniLM-L6-v2')"],
            capture_output=True, timeout=300,
        )
        _set(detail="Embedding model ready", pct=95)
    except Exception:
        # Non-fatal — model will download on first use
        _set(detail="Embedding model will download on first use", pct=95)


def _start_setup():
    threading.Thread(target=_run_setup, daemon=True).start()


def _main():
    if not _needs_setup():
        c.itemconfigure(_status_id, text="Opening Agent Hub…")
        c.itemconfigure(_detail_id, text="Everything is ready")
        _tick_sweep()
        root.after(300, _launch_app)
    else:
        _start_setup()


root.after(120, _main)
root.mainloop()
