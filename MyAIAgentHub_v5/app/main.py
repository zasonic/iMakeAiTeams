"""
Entry point. Sets up core services, checks first run, opens PyWebView window.
"""

import logging
import os
import sys
import threading
from pathlib import Path

import webview

from core.settings import Settings
from core.events import EventBus
from core.api import API
from core.first_run import needs_first_run

APP_ROOT = Path(__file__).parent
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(APP_ROOT / "app.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("MyAIAgentHub")

settings = Settings(APP_ROOT / "settings.json")
bus = EventBus()
api = API(settings, bus, APP_ROOT, log)

window = webview.create_window(
    title="MyAI Agent Hub",
    url=str(APP_ROOT / "frontend" / "index.html"),
    js_api=api,
    width=1280,
    height=820,
    min_size=(1024, 660),
    background_color="#0f0f0f",
)

api.set_window(window)


def _start_channel_manager():
    """Start the channel manager in a background thread after services are ready."""
    try:
        from channels.channel_manager import ChannelManager
        from services.guardrails_gate import GuardrailsGate

        # Guardrails gate (optional — degrades gracefully if nemoguardrails not installed)
        guardrails = GuardrailsGate(settings, local_client=getattr(api, '_local', None))

        cm = ChannelManager(
            settings=settings,
            bus=bus,
            chat_orchestrator=api._chat if hasattr(api, '_chat') else None,
            claude_client=api._claude if hasattr(api, '_claude') else None,
            local_client=api._local if hasattr(api, '_local') else None,
            memory=api._memory if hasattr(api, '_memory') else None,
            safety_gate=None,
            guardrails_gate=guardrails,
            project_root=APP_ROOT,
        )
        cm.start()
        api.set_channel_manager(cm)
        log.info("Channel manager started and wired into API")
    except Exception as exc:
        log.error("Channel manager failed to start: %s", exc, exc_info=True)


def _on_loaded():
    if needs_first_run(settings):
        window.evaluate_js("window.showFirstRun()")
    else:
        start = settings.get("start_tab", "chat")
        window.evaluate_js(f"window.navigate('{start}')")
    # Start channel manager after GUI is loaded (services are fully initialised)
    threading.Thread(target=_start_channel_manager, name="channel-manager-start", daemon=True).start()


def _on_closing():
    def _cleanup():
        log.info("Window closing — shutting down services…")
        api.shutdown()
        log.info("Shutdown complete.")

    t = threading.Thread(target=_cleanup, daemon=True)
    t.start()
    t.join(timeout=4)


window.events.loaded += _on_loaded
window.events.closing += _on_closing

webview.start(debug=os.environ.get("MYAI_DEBUG", "").lower() in ("1", "true", "yes"))
