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

APP_NAME = "MyAIAgentHub"
APP_AUTHOR = "iMakeAiTeams"
MIGRATION_SENTINEL = ".migrated_v5"
V5_RENAME_SENTINEL = ".migrated_v6_rename"
LEGACY_APP_NAME = "iMakeAiTeams"

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


def legacy_user_dir() -> Path:
    """v5 user data directory (before the APP_NAME rename to MyAIAgentHub)."""
    return Path(user_data_dir(LEGACY_APP_NAME, APP_AUTHOR, roaming=False))


def migrate_v5_user_dir() -> None:
    """
    One-shot move of v5 user data from the legacy 'iMakeAiTeams' dir to the
    new 'MyAIAgentHub' dir after the APP_NAME rename. Called once at startup
    from app/main.py, before logging is configured.

    Sentinel-guarded, swallows errors, never raises — a broken migration must
    not brick the app. Keyring entries are stored in the OS keychain and are
    not touched here.
    """
    try:
        target = user_dir()
        sentinel = target / V5_RENAME_SENTINEL
        if sentinel.exists():
            return
        legacy = legacy_user_dir()
        if not legacy.exists() or legacy == target:
            sentinel.write_text("{}", encoding="utf-8")
            return
        # Only migrate if the new dir is effectively empty (ignoring our own
        # sentinels). Respects users who already have data in the new dir.
        existing = [p for p in target.iterdir() if p.name not in {
            V5_RENAME_SENTINEL, MIGRATION_SENTINEL,
        }]
        if existing:
            sentinel.write_text(
                json.dumps({"skipped": "target not empty", "at": time.time()}),
                encoding="utf-8",
            )
            return
        moved: list[str] = []
        for entry in list(legacy.iterdir()):
            dst = target / entry.name
            if dst.exists():
                continue
            try:
                shutil.move(str(entry), str(dst))
                moved.append(entry.name)
            except OSError as exc:
                print(
                    f"paths.migrate_v5: failed to move {entry.name}: {exc}",
                    file=sys.stderr,
                )
        sentinel.write_text(
            json.dumps(
                {"migrated_at": time.time(), "from": str(legacy), "moved": moved},
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"paths.migrate_v5: unexpected error: {exc}", file=sys.stderr)


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


def mcp_servers_dir() -> Path:
    d = user_dir() / "mcp_servers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def bundled_model_dir(name: str = "all-MiniLM-L6-v2") -> Path:
    """
    Resolve the sentence-transformers model bundled with the installer.

    Frozen builds place the model at install_root()/_internal/models/<name>/
    (PyInstaller onedir layout). Source checkouts fall back to
    build/models/<name>/ at the repo root, populated by build/fetch_model.py.
    """
    frozen_path = install_root() / "_internal" / "models" / name
    if frozen_path.exists():
        return frozen_path
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        mp = Path(meipass) / "models" / name
        if mp.exists():
            return mp
    # Source checkout
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "build" / "models" / name


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
