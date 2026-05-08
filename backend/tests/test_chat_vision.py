"""tests/test_chat_vision.py — PR 11 image-input tests.

Covers the vision dispatch added to ChatOrchestrator.send() and the
attachment route's image handling:

  - Claude route with an image attachment passes Anthropic image blocks
    on the user message.
  - Local route with a vision-capable model passes images through
    LocalClient.chat_with_images.
  - Local route with a non-vision model raises LocalVisionUnavailable
    and surfaces a friendly error that names a fallback family.
  - Image attachments larger than 20MB return 400 from the route.
  - Image rows are forced to persist=false even when the request body
    sets persist=true (RAG is text-only in this codebase).
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes import attachments as attachments_routes
from server import BearerAuthMiddleware


TOKEN = "test-token-vision"


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def attachments_root(tmp_path, monkeypatch):
    root = tmp_path / "attachments"
    root.mkdir()
    from core import paths as _paths
    monkeypatch.setattr(_paths, "attachments_dir", lambda: root)
    monkeypatch.setattr(attachments_routes.paths, "attachments_dir", lambda: root)
    return root


@pytest.fixture
def fake_rag():
    rag = MagicMock()
    rag.add_text = MagicMock(return_value=1)
    return rag


@pytest.fixture
def app(in_memory_db, attachments_root, fake_rag):
    a = FastAPI()
    a.add_middleware(BearerAuthMiddleware, expected_token=TOKEN)
    a.include_router(attachments_routes.router, prefix="/api")

    fake_api = MagicMock()
    fake_api._rag = fake_rag
    fake_container = MagicMock()
    fake_container.api = fake_api
    a.state.container = fake_container
    return a


def _seed_conversation(in_memory_db, *, cid: str = "conv-vis") -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    in_memory_db.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (cid, "Vision test", now, now),
    )
    in_memory_db.commit()


def _png_bytes() -> bytes:
    # Smallest valid PNG: 1x1 red pixel. Captured via PIL once and
    # hard-coded so the test suite has no Pillow dependency.
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
        b"nGNgYGBgAAAABQABXvMqOgAAAABJRU5ErkJggg=="
    )


def _orchestrator(in_memory_db, claude_client, local_client, settings, routing="claude"):
    from services.chat_orchestrator import ChatOrchestrator
    from models import RouteDecision
    from services.memory import MemoryManager

    router = MagicMock()
    router.classify.return_value = RouteDecision(
        model=routing, complexity="simple", reasoning="vision test",
    )
    mem = MemoryManager(rag_index=None, semantic_search_mod=None,
                        local_client=local_client)
    return ChatOrchestrator(claude_client, local_client, router, mem, settings)


def _seed_image_attachment(
    in_memory_db, attachments_dir: Path, conversation_id: str,
    *, filename: str = "shot.png", mime: str = "image/png",
) -> str:
    """Insert an image row + write its bytes to disk under attachments_dir."""
    import uuid
    from datetime import datetime, timezone
    aid = str(uuid.uuid4())
    raw = _png_bytes()
    ext = Path(filename).suffix.lower()
    (attachments_dir / f"{aid}{ext}").write_bytes(raw)
    now = datetime.now(timezone.utc).isoformat()
    in_memory_db.execute(
        "INSERT INTO attachments (id, conversation_id, filename, mime_type, "
        "size_bytes, persist, rag_doc_id, content_extract, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (aid, conversation_id, filename, mime, len(raw), 0, None,
         f"[image: {filename}]", now),
    )
    in_memory_db.commit()
    return aid


# ── Orchestrator: Claude path attaches image blocks ─────────────────────────

class TestClaudeVisionDispatch:
    def test_image_attachment_becomes_anthropic_block(
        self, in_memory_db, claude_client, local_client_available,
        settings, attachments_root,
    ):
        orch = _orchestrator(
            in_memory_db, claude_client, local_client_available, settings,
            routing="claude",
        )
        conv_id = orch.create_conversation()
        _seed_image_attachment(in_memory_db, attachments_root, conv_id)

        captured: list = []

        def capture(system, msgs, **kwargs):
            captured.extend(msgs)
            return {"text": "I see a tiny red square.",
                    "input_tokens": 1, "output_tokens": 1}

        claude_client.chat_multi_turn = capture

        result = orch.send(conv_id, "what does this show?")

        assert "tiny red square" in result.text
        # The last user message should now have a content list with an
        # image block prepended and the typed text as a trailing text block.
        last_user = next(
            (m for m in reversed(captured) if m.get("role") == "user"),
            None,
        )
        assert last_user is not None
        content = last_user["content"]
        assert isinstance(content, list), \
            f"expected content list with image blocks, got {type(content)}"
        types = [b.get("type") for b in content]
        assert "image" in types, f"expected image block in {types}"
        # Find the image block and assert the Anthropic shape.
        img = next(b for b in content if b.get("type") == "image")
        assert img["source"]["type"] == "base64"
        assert img["source"]["media_type"] == "image/png"
        assert isinstance(img["source"]["data"], str)
        # base64 should round-trip back to the original PNG bytes.
        assert base64.b64decode(img["source"]["data"]) == _png_bytes()


# ── Orchestrator: local + vision-capable model passes images via Ollama ─────

class TestLocalVisionDispatch:
    def test_vision_model_calls_chat_with_images(
        self, in_memory_db, claude_client, local_client_available,
        settings, attachments_root,
    ):
        # is_vision_model returns True; chat_with_images is the entry point.
        local_client_available.is_vision_model = MagicMock(return_value=True)
        local_client_available.chat_with_images = MagicMock(
            return_value="local vision says: red pixel",
        )
        # Active local model name shows up in the call payload.
        settings.set("default_local_model", "qwen2.5-vl:7b")

        orch = _orchestrator(
            in_memory_db, claude_client, local_client_available, settings,
            routing="local",
        )
        conv_id = orch.create_conversation()
        _seed_image_attachment(in_memory_db, attachments_root, conv_id)

        result = orch.send(conv_id, "describe the image")

        assert "red pixel" in result.text
        assert local_client_available.chat_with_images.called
        # Inspect the kwargs/args: third positional is the base64 list.
        call = local_client_available.chat_with_images.call_args
        images_arg = call.args[2] if len(call.args) >= 3 else call.kwargs.get("images_b64")
        assert isinstance(images_arg, list) and len(images_arg) == 1
        assert base64.b64decode(images_arg[0]) == _png_bytes()
        # Claude should NOT have been touched.
        claude_client.chat_multi_turn.assert_not_called()


# ── Orchestrator: local + non-vision model returns the friendly error ───────

class TestLocalVisionUnavailable:
    def test_non_vision_model_returns_friendly_error_with_hint(
        self, in_memory_db, claude_client, local_client_available,
        settings, attachments_root,
    ):
        local_client_available.is_vision_model = MagicMock(return_value=False)
        local_client_available.chat_with_images = MagicMock(
            side_effect=AssertionError(
                "should not be called when is_vision_model is False",
            ),
        )
        settings.set("default_local_model", "llama3:8b")

        orch = _orchestrator(
            in_memory_db, claude_client, local_client_available, settings,
            routing="local",
        )
        conv_id = orch.create_conversation()
        _seed_image_attachment(in_memory_db, attachments_root, conv_id)

        result = orch.send(conv_id, "describe the image")

        # Friendly error mentions a vision family from the default settings list.
        assert result.route_reason == "vision_unavailable_local"
        assert "qwen2.5-vl" in result.text or "llava" in result.text
        local_client_available.chat_with_images.assert_not_called()
        claude_client.chat_multi_turn.assert_not_called()


# ── Route: oversized image returns 400 ──────────────────────────────────────

class TestImageSizeCap:
    def test_oversized_image_returns_400(
        self, app, in_memory_db, attachments_root,
    ):
        _seed_conversation(in_memory_db)
        client = TestClient(app)

        # 21 MB of bytes — over the 20 MB image cap. Use a PNG signature
        # so _is_supported() picks the image branch instead of bouncing
        # on extension-mismatch validation.
        body = b"\x89PNG\r\n\x1a\n" + (b"\x00" * (21 * 1024 * 1024))
        files = {"file": ("huge.png", BytesIO(body), "image/png")}
        resp = client.post(
            "/api/chat/conv-vis/attach",
            files=files, data={"persist": "false"}, headers=_auth(),
        )
        assert resp.status_code == 400, resp.text
        assert "20 MB" in resp.json().get("detail", "") or \
               "too large" in resp.json().get("detail", "").lower()

        # No attachment row should have been written.
        rows = in_memory_db.fetchall("SELECT * FROM attachments")
        assert rows == []


# ── Route: image persist defaults to false even when body says true ─────────

class TestImagePersistEphemeralOnly:
    def test_image_persist_true_silently_downgraded(
        self, app, in_memory_db, attachments_root, fake_rag,
    ):
        _seed_conversation(in_memory_db)
        client = TestClient(app)

        files = {"file": ("photo.png", BytesIO(_png_bytes()), "image/png")}
        resp = client.post(
            "/api/chat/conv-vis/attach",
            files=files, data={"persist": "true"}, headers=_auth(),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # The route forces persist=false for images regardless of input.
        assert body["persist"] is False

        row = in_memory_db.fetchone(
            "SELECT * FROM attachments WHERE id = ?", (body["id"],),
        )
        assert row is not None
        assert row["persist"] == 0
        assert row["rag_doc_id"] is None
        # And RAG must NOT have been touched even though the user said true.
        fake_rag.add_text.assert_not_called()
