"""
Entry point. Sets up core services, checks first run, opens PyWebView window.
"""

import logging
import os
import sys
import threading
from pathlib import Path

import webview

from core import paths
from core.settings import Settings
from core.events import EventBus
from core.api import API
from core.first_run import needs_first_run

APP_ROOT = Path(__file__).parent
# v5→v6 APP_NAME rename: move %LOCALAPPDATA%\iMakeAiTeams\* to ...\MyAIAgentHub\*
# before anyone resolves a path that would pin the old location.
paths.migrate_v5_user_dir()
USER_DIR = paths.user_dir()
# Must run before logging.basicConfig: a FileHandler pointed at APP_ROOT/app.log
# would pin the old location and orphan the migrated log file.
paths.migrate_legacy_install(APP_ROOT, USER_DIR)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(paths.log_path(), encoding="utf-8"),
    ],
)
log = logging.getLogger("MyAIAgentHub")

settings = Settings(paths.settings_path())
bus = EventBus()
api = API(settings, bus, USER_DIR, log)

window = webview.create_window(
    title="iMakeAiTeams",
    url=str(APP_ROOT / "frontend" / "index.html"),
    js_api=api,
    width=1280,
    height=820,
    min_size=(1024, 660),
    background_color="#0f0f0f",
)

api.set_window(window)


def _on_loaded():
    if needs_first_run(settings):
        window.evaluate_js("window.showFirstRun()")
    else:
        start = settings.get("start_tab", "chat")
        window.evaluate_js(f"window.navigate('{start}')")
    # Kick off the deferred (slow) services now that the window has painted.
    # sentence-transformers, ChromaDB, and the indexer thread all run here so
    # a blank PyWebView frame is never shown during a 60s first-run download.
    api.start_deferred_init()


def _on_closing():
    def _cleanup():
        log.info("Window closing — shutting down services…")
        try:
            api.shutdown()
            log.info("Shutdown complete.")
        except Exception as exc:
            log.warning("api.shutdown() raised: %s", exc, exc_info=True)

    t = threading.Thread(target=_cleanup, daemon=True)
    t.start()
    # 15s covers ChromaDB flush + WAL checkpoint on slow machines. If we still
    # time out, surface the non-daemon threads still alive so zombies are
    # diagnosable instead of silent.
    t.join(timeout=15)
    if t.is_alive():
        alive = [
            th.name for th in threading.enumerate()
            if th is not threading.current_thread() and not th.daemon
        ]
        log.warning("Shutdown did not complete within 15s; live threads: %s", alive)


window.events.loaded += _on_loaded
window.events.closing += _on_closing

webview.start(debug=os.environ.get("MYAI_DEBUG", "").lower() in ("1", "true", "yes"))
