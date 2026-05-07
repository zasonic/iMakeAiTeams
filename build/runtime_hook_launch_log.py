"""
PyInstaller runtime hook. Runs inside the bundled interpreter before any
Analysis.scripts entry point — the earliest Python hook point available
in a frozen build.

Writes one line to %LOCALAPPDATA%\\MyAIAgentHub\\launch.log on every launch
and installs a sys.excepthook that appends uncaught tracebacks to the same
file. Stdlib only, so it survives a half-broken bundle (missing DLL,
missing package data) that would kill a richer import graph.

Rationale: when a clean-VM install crashes before app/main.py gets far
enough to configure its own logging, this is the only artifact we have
for postmortem.
"""

import datetime
import os
import sys
import tempfile
import traceback
from pathlib import Path


def _launch_log_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    d = Path(base) / "MyAIAgentHub"
    d.mkdir(parents=True, exist_ok=True)
    return d / "launch.log"


def _append(line: str) -> None:
    try:
        with _launch_log_path().open("a", encoding="utf-8") as fh:
            fh.write(line)
            if not line.endswith("\n"):
                fh.write("\n")
    except OSError:
        pass


_ts = datetime.datetime.now().isoformat(timespec="seconds")
_meipass = getattr(sys, "_MEIPASS", "")
_frozen = getattr(sys, "frozen", False)
_append(
    f"[{_ts}] launch pid={os.getpid()} frozen={_frozen} "
    f"exe={sys.executable!r} meipass={_meipass!r} "
    f"py={sys.version.split()[0]}"
)


def _excepthook(exc_type, exc, tb):
    _append(
        f"[{datetime.datetime.now().isoformat(timespec='seconds')}] UNCAUGHT "
        f"{exc_type.__name__}: {exc}"
    )
    _append("".join(traceback.format_exception(exc_type, exc, tb)))
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _excepthook

if _meipass:
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME", str(Path(_meipass) / "models")
    )
