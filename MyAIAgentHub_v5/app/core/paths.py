"""
core/paths.py — Single source of truth for writable paths.

The install directory (C:\\Program Files\\..., /Applications/..., /usr/bin/...)
is read-only or UAC-protected on production deploys. All persistent state
(database, settings, logs, caches, indexes) must live under the per-user
writable data dir resolved by platformdirs.

This module also ships a one-shot migrator that moves legacy files from the
install directory to the user data directory the first time the app runs
after upgrading. The migrator is idempotent: it writes a sentinel and is a
no-op on subsequent runs.

Ordering invariant: migrate_legacy_install() MUST run before
logging.basicConfig() configures its FileHandler, otherwise the log file
handle pins the old location.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "iMakeAiTeams"
APP_AUTHOR = "iMakeAiTeams"
MIGRATION_SENTINEL = ".migrated_v5"

# Legacy artifacts that lived next to the executable in v5.0.x.
# Order matters: SQLite WAL/SHM must move with the main DB file.
LEGACY_ARTIFACTS: tuple[str, ...] = (
    "myai.db",
    "myai.db-wal",
    "myai.db-shm",
    "settings.json",
    "app.log",
    "rag_cache",
    "myai_vector_store",
)


def user_dir() -> Path:
    """Resolve the per-user writable data directory and ensure it exists."""
    path = Path(user_data_dir(APP_NAME, APP_AUTHOR, roaming=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def install_root() -> Path:
    """
    Resolve the read-only install root.

    In a PyInstaller frozen build, ``Path(__file__).parent`` points at the
    temporary extraction dir (``sys._MEIPASS``), not the install dir — so use
    ``sys.executable`` instead. In a source checkout, ``__file__`` is correct.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def db_path() -> Path:
    return user_dir() / "myai.db"


def settings_path() -> Path:
    return user_dir() / "settings.json"


def log_path() -> Path:
    return user_dir() / "app.log"


def rag_cache_dir() -> Path:
    d = user_dir() / "rag_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def vector_store_dir() -> Path:
    d = user_dir() / "myai_vector_store"
    d.mkdir(parents=True, exist_ok=True)
    return d


def extensions_dir() -> Path:
    d = user_dir() / "extensions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def migrate_legacy_install(app_root: Path, target_user_dir: Path) -> None:
    """
    One-shot move of legacy install-dir files into the user data dir.

    Runs exactly once per user data dir (guarded by a sentinel file).
    Never raises: any failure is reported to stderr and swallowed so a botched
    migration cannot brick the app.

    Must be called before logging is configured.
    """
    try:
        target_user_dir.mkdir(parents=True, exist_ok=True)
        sentinel = target_user_dir / MIGRATION_SENTINEL
        if sentinel.exists():
            return

        moved: list[str] = []
        skipped: list[str] = []
        for name in LEGACY_ARTIFACTS:
            src = app_root / name
            if not src.exists():
                continue
            dst = target_user_dir / name
            if dst.exists():
                skipped.append(name)
                print(
                    f"paths.migrate: destination already exists, leaving legacy in place: {name}",
                    file=sys.stderr,
                )
                continue
            try:
                shutil.move(str(src), str(dst))
                moved.append(name)
            except OSError as move_exc:
                try:
                    if src.is_dir():
                        shutil.copytree(src, dst)
                        shutil.rmtree(src, ignore_errors=True)
                    else:
                        shutil.copy2(src, dst)
                        try:
                            src.unlink()
                        except OSError:
                            pass
                    moved.append(name)
                except OSError as copy_exc:
                    print(
                        f"paths.migrate: failed to migrate {name}: {move_exc} / {copy_exc}",
                        file=sys.stderr,
                    )

        sentinel.write_text(
            json.dumps(
                {
                    "migrated_at": time.time(),
                    "from": str(app_root),
                    "moved": moved,
                    "skipped": skipped,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"paths.migrate: unexpected error: {exc}", file=sys.stderr)
