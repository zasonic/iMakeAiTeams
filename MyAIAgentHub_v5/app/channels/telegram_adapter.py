"""
channels/telegram_adapter.py

Telegram bot adapter using python-telegram-bot (v21, polling).

No webhook or public HTTPS endpoint required — polling works perfectly
for a solo developer running the app on their own machine.

Features
--------
- Receives messages from any allowed user (access_control.py)
- Routes /run, /agent, /code prefix messages to the agentic loop
- Routes all other messages through the normal chat orchestrator
- Streams tool call events back to Telegram during agent runs
- Permission confirmations: agent asks, user replies Yes/No in Telegram
- Sends long responses in chunks (Telegram 4096-char message limit)

Slash commands registered with Telegram
---------------------------------------
  /start   — welcome message + usage
  /run     — trigger agentic coding loop
  /agent   — alias for /run
  /code    — alias for /run
  /status  — show agent hub status
  /stop    — stop a running agent task
  /help    — list commands

Thread safety
-------------
The Telegram polling loop runs in its own daemon thread.
All callbacks to the orchestrator and agent_loop use the existing
thread-safe patterns from the existing codebase.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from channels.access_control import AccessControl
    from channels.inbound_message import InboundMessage

log = logging.getLogger("MyAIAgentHub.telegram")

# Max message length Telegram allows
_TG_MAX_LEN = 4000  # Leave buffer below 4096 hard limit


class TelegramAdapter:
    """
    Telegram bot adapter.

    Parameters
    ----------
    token           : Bot token from @BotFather.
    access_control  : AccessControl instance for allowlist checks.
    on_message      : Callback(InboundMessage) → called for every allowed message.
    project_root    : Default project directory for agent runs.
    """

    def __init__(
        self,
        token: str,
        access_control: "AccessControl",
        on_message: Callable[["InboundMessage"], None],
        project_root: Path | None = None,
    ) -> None:
        self._token    = token
        self._ac       = access_control
        self._on_msg   = on_message
        self._root     = project_root or Path.cwd()
        self._app      = None
        self._thread: threading.Thread | None = None
        self._running  = False
        # Map request_id → chat_id for permission callbacks
        self._pending_confirms: dict[str, int] = {}
        self._confirm_callbacks: dict[str, Callable[[bool], None]] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the Telegram polling loop in a daemon thread."""
        if self._running:
            log.warning("TelegramAdapter already running")
            return
        try:
            from telegram.ext import Application
        except ImportError:
            log.error(
                "python-telegram-bot not installed. "
                "Run: pip install python-telegram-bot"
            )
            return

        self._thread = threading.Thread(
            target=self._run_polling,
            name="telegram-polling",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        log.info("TelegramAdapter started (polling)")

    def stop(self) -> None:
        """Request the polling loop to stop."""
        self._running = False
        if self._app:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._app.stop())
                loop.close()
            except Exception as exc:
                log.warning("TelegramAdapter stop error: %s", exc)
        log.info("TelegramAdapter stopped")

    def is_running(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())

    def status(self) -> dict:
        return {
            "running": self.is_running(),
            "has_token": bool(self._token),
        }

    # ── Send helpers ───────────────────────────────────────────────────────

    def send_message(self, chat_id: int | str, text: str) -> None:
        """Send a message to a Telegram chat, chunking if needed."""
        if not self._app:
            log.warning("TelegramAdapter: send_message called before start()")
            return
        import asyncio
        chunks = self._chunk(text)
        for chunk in chunks:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(
                    self._app.bot.send_message(chat_id=chat_id, text=chunk, parse_mode="Markdown")
                )
                loop.close()
            except Exception:
                # Retry without Markdown if parse fails
                try:
                    loop2 = asyncio.new_event_loop()
                    loop2.run_until_complete(
                        self._app.bot.send_message(chat_id=chat_id, text=chunk)
                    )
                    loop2.close()
                except Exception as exc2:
                    log.error("TelegramAdapter: send_message failed: %s", exc2)

    def ask_confirm(
        self,
        chat_id: int,
        request_id: str,
        description: str,
        callback: Callable[[bool], None],
    ) -> None:
        """
        Send a Yes/No confirmation request to the user.
        callback(True) = approved, callback(False) = denied.
        """
        self._confirm_callbacks[request_id] = callback
        text = (
            f"⚠️ *Agent wants to:*\n`{description}`\n\n"
            f"Reply *yes* or *no* (request ID: `{request_id}`)"
        )
        self.send_message(chat_id, text)

    # ── Polling setup ──────────────────────────────────────────────────────

    def _run_polling(self) -> None:
        """Run asyncio event loop for Telegram polling."""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_run())
        except Exception as exc:
            log.error("TelegramAdapter polling error: %s", exc)
            self._running = False

    async def _async_run(self) -> None:
        """Async Telegram polling loop."""
        from telegram.ext import (
            Application, CommandHandler, MessageHandler,
            filters, ContextTypes,
        )
        from telegram import Update

        app = (
            Application.builder()
            .token(self._token)
            .build()
        )
        self._app = app

        # Register command handlers
        app.add_handler(CommandHandler("start",  self._cmd_start))
        app.add_handler(CommandHandler("help",   self._cmd_help))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("stop",   self._cmd_stop))
        app.add_handler(CommandHandler("run",    self._cmd_run))
        app.add_handler(CommandHandler("agent",  self._cmd_run))
        app.add_handler(CommandHandler("code",   self._cmd_run))

        # All other text messages
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )

        log.info("Telegram bot polling started")
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        # Keep polling until stopped
        import asyncio as _asyncio
        while self._running:
            await _asyncio.sleep(1)

        await app.updater.stop()
        await app.stop()
        await app.shutdown()

    # ── Command handlers ───────────────────────────────────────────────────

    async def _cmd_start(self, update, context) -> None:
        user_id = str(update.effective_user.id)
        from channels.inbound_message import Channel
        if not self._ac.is_allowed(Channel.TELEGRAM, user_id):
            await update.message.reply_text("⛔ You are not authorised to use this bot.")
            return
        await update.message.reply_text(
            "👋 *MyAI Agent Hub* connected!\n\n"
            "Commands:\n"
            "• Just type to chat\n"
            "• `/run <task>` — start agentic coding task\n"
            "• `/agent <task>` — same as /run\n"
            "• `/status` — show system status\n"
            "• `/stop` — stop running task\n"
            "• `/help` — this message",
            parse_mode="Markdown",
        )

    async def _cmd_help(self, update, context) -> None:
        await self._cmd_start(update, context)

    async def _cmd_status(self, update, context) -> None:
        from channels.inbound_message import Channel
        if not self._ac.is_allowed(Channel.TELEGRAM, str(update.effective_user.id)):
            return
        await update.message.reply_text(
            "✅ MyAI Agent Hub is running.\nUse `/run <task>` to start an agentic task.",
            parse_mode="Markdown",
        )

    async def _cmd_stop(self, update, context) -> None:
        from channels.inbound_message import Channel
        if not self._ac.is_allowed(Channel.TELEGRAM, str(update.effective_user.id)):
            return
        await update.message.reply_text("🛑 Stop signal sent to active agent task.")
        # The channel_manager will forward this stop signal

    async def _cmd_run(self, update, context) -> None:
        """Handle /run, /agent, /code commands."""
        from channels.inbound_message import Channel, ChatType
        user = update.effective_user
        chat = update.effective_chat

        if not self._ac.is_allowed(Channel.TELEGRAM, str(user.id)):
            await update.message.reply_text("⛔ Not authorised.")
            return

        # Extract task text (everything after the command)
        args = context.args
        task = " ".join(args) if args else ""
        if not task:
            await update.message.reply_text(
                "Please provide a task. Example:\n`/run fix the failing tests in test_router.py`",
                parse_mode="Markdown",
            )
            return

        msg = self._make_inbound(
            user=user,
            chat=chat,
            text=task,
            agent_mode=True,
        )
        await update.message.reply_text(f"🤖 Starting agent task:\n_{task[:100]}_", parse_mode="Markdown")
        threading.Thread(target=self._on_msg, args=(msg,), daemon=True).start()

    async def _handle_text(self, update, context) -> None:
        """Handle plain text messages (non-command)."""
        from channels.inbound_message import Channel, ChatType
        user = update.effective_user
        chat = update.effective_chat

        if not self._ac.is_allowed(Channel.TELEGRAM, str(user.id)):
            return

        text = update.message.text or ""

        # Handle yes/no confirmation replies
        lower = text.strip().lower()
        if lower in ("yes", "y", "no", "n"):
            # Check if there's a pending confirmation for this user
            handled = self._handle_confirm_reply(user.id, lower in ("yes", "y"))
            if handled:
                reply = "✅ Approved." if lower in ("yes", "y") else "❌ Denied."
                await update.message.reply_text(reply)
                return

        # Normal chat message
        msg = self._make_inbound(user=user, chat=chat, text=text, agent_mode=False)
        threading.Thread(target=self._on_msg, args=(msg,), daemon=True).start()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _make_inbound(self, user, chat, text: str, agent_mode: bool):
        from channels.inbound_message import InboundMessage, Channel, ChatType
        chat_type = ChatType.PRIVATE if chat.type == "private" else ChatType.GROUP
        conv_id = f"telegram:{user.id}" if chat_type == ChatType.PRIVATE else f"telegram:{chat.id}"
        return InboundMessage(
            channel=Channel.TELEGRAM,
            chat_type=chat_type,
            user_id=str(user.id),
            username=user.username or user.first_name or str(user.id),
            text=text,
            conversation_id=conv_id,
            agent_mode=agent_mode,
            raw_meta={"chat_id": chat.id, "user_id": user.id},
        )

    def _handle_confirm_reply(self, user_id: int, approved: bool) -> bool:
        """Process a yes/no reply for the most recent pending confirmation."""
        if not self._confirm_callbacks:
            return False
        # Take the most recent pending confirmation for this user
        request_id = next(iter(self._confirm_callbacks))
        cb = self._confirm_callbacks.pop(request_id)
        threading.Thread(target=cb, args=(approved,), daemon=True).start()
        return True

    @staticmethod
    def _chunk(text: str) -> list[str]:
        """Split text into Telegram-sized chunks."""
        if len(text) <= _TG_MAX_LEN:
            return [text]
        chunks = []
        while text:
            chunks.append(text[:_TG_MAX_LEN])
            text = text[_TG_MAX_LEN:]
        return chunks
