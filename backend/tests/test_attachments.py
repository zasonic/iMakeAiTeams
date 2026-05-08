"""tests/test_attachments.py — HTTP-level tests for the chat-input file
attachment routes added in PR 8.

The attachment route registers itself under /api with three endpoints:

    POST   /api/chat/{conversation_id}/attach    multipart upload
    GET    /api/chat/{conversation_id}/attachments
    DELETE /api/chat/attachments/{id}

Tests mount only the attachments router on a minimal FastAPI app — the
sidecar's full container isn't needed, but we do hand the app a tiny stub
container so the persist-to-RAG path can flip the rag_doc_id without
spinning up sentence-transformers.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes import attachments as attachments_routes
from server import BearerAuthMiddleware


TOKEN = "test-token-attach"


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def attachments_root(tmp_path, monkeypatch):
    """Redirect attachments_dir() to a temp path so tests don't pollute
    the real userData directory.
    """
    root = tmp_path / "attachments"
    root.mkdir()
    from core import paths as _paths
    monkeypatch.setattr(_paths, "attachments_dir", lambda: root)
    # The route module captured ``paths`` at import time; override the
    # attribute on the module-level reference too.
    monkeypatch.setattr(attachments_routes.paths, "attachments_dir", lambda: root)
    return root


@pytest.fixture
def fake_rag():
    """Stub RAG index with an add_text spy so we can assert ingestion."""
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


def _seed_conversation(in_memory_db, *, cid: str = "conv-A") -> None:
    """Conversation row needed because the attachments table references it."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    in_memory_db.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (cid, "Attach test", now, now),
    )
    in_memory_db.commit()


def _upload(client: TestClient, conversation_id: str, *,
            content: bytes, filename: str, persist: str) -> dict:
    files = {"file": (filename, BytesIO(content), "text/plain")}
    data = {"persist": persist}
    resp = client.post(
        f"/api/chat/{conversation_id}/attach",
        files=files, data=data, headers=_auth(),
    )
    return resp


# ── POST .txt with persist=false ────────────────────────────────────────────


class TestUploadEphemeral:
    def test_inserts_row_and_writes_file_without_rag_doc_id(
        self, app, in_memory_db, attachments_root, fake_rag,
    ):
        _seed_conversation(in_memory_db)
        client = TestClient(app)

        resp = _upload(
            client, "conv-A",
            content=b"hello world\nthis is the body",
            filename="notes.txt",
            persist="false",
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["filename"] == "notes.txt"
        assert body["persist"] is False
        assert body["size_bytes"] == len(b"hello world\nthis is the body")
        assert body["extract_chars"] > 0

        # Row exists, persist=0, rag_doc_id is NULL, content_extract is set.
        row = in_memory_db.fetchone(
            "SELECT * FROM attachments WHERE id = ?", (body["id"],),
        )
        assert row is not None
        assert row["conversation_id"] == "conv-A"
        assert row["persist"] == 0
        assert row["rag_doc_id"] is None
        assert "hello world" in (row["content_extract"] or "")

        # File on disk under attachments_dir().
        files = list(Path(attachments_root).iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".txt"
        assert files[0].read_bytes() == b"hello world\nthis is the body"

        # No RAG ingest on the ephemeral path.
        fake_rag.add_text.assert_not_called()


# ── POST .txt with persist=true ─────────────────────────────────────────────


class TestUploadPersistent:
    def test_inserts_row_with_rag_doc_id_and_calls_rag(
        self, app, in_memory_db, fake_rag,
    ):
        _seed_conversation(in_memory_db)
        client = TestClient(app)

        resp = _upload(
            client, "conv-A",
            content=b"persist this please",
            filename="doc.md",
            persist="true",
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["persist"] is True

        row = in_memory_db.fetchone(
            "SELECT * FROM attachments WHERE id = ?", (body["id"],),
        )
        assert row is not None
        assert row["persist"] == 1
        assert row["rag_doc_id"]  # non-null
        assert row["rag_doc_id"] == body["id"]
        assert "persist this please" in (row["content_extract"] or "")

        # RAG add_text was called with the extracted text.
        fake_rag.add_text.assert_called_once()
        call_kwargs = fake_rag.add_text.call_args
        assert "persist this please" in call_kwargs.args[0]
        assert call_kwargs.kwargs.get("source") == "doc.md"


# ── POST unsupported extension ──────────────────────────────────────────────


class TestUploadUnsupported:
    def test_pdf_returns_400_with_clear_error(self, app, in_memory_db):
        _seed_conversation(in_memory_db)
        client = TestClient(app)

        files = {"file": ("report.pdf", BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
        resp = client.post(
            "/api/chat/conv-A/attach",
            files=files, data={"persist": "false"}, headers=_auth(),
        )

        assert resp.status_code == 400
        body = resp.json()
        assert "pdf" in body.get("detail", "").lower()

        rows = in_memory_db.fetchall("SELECT * FROM attachments")
        assert rows == []

    def test_unknown_ext_returns_400(self, app, in_memory_db):
        _seed_conversation(in_memory_db)
        client = TestClient(app)

        files = {"file": ("blob.xyz", BytesIO(b"data"), "application/octet-stream")}
        resp = client.post(
            "/api/chat/conv-A/attach",
            files=files, data={"persist": "false"}, headers=_auth(),
        )

        assert resp.status_code == 400
        body = resp.json()
        assert "supported" in body.get("detail", "").lower()


# ── DELETE removes row and file ─────────────────────────────────────────────


class TestDelete:
    def test_delete_drops_row_and_file(
        self, app, in_memory_db, attachments_root,
    ):
        _seed_conversation(in_memory_db)
        client = TestClient(app)

        upload = _upload(
            client, "conv-A",
            content=b"throwaway",
            filename="ephemeral.txt",
            persist="false",
        ).json()
        files_before = list(Path(attachments_root).iterdir())
        assert len(files_before) == 1

        resp = client.delete(
            f"/api/chat/attachments/{upload['id']}", headers=_auth(),
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        row = in_memory_db.fetchone(
            "SELECT id FROM attachments WHERE id = ?", (upload["id"],),
        )
        assert row is None
        files_after = list(Path(attachments_root).iterdir())
        assert files_after == []

    def test_delete_persisted_clears_rag_documents_table(
        self, app, in_memory_db,
    ):
        _seed_conversation(in_memory_db)
        client = TestClient(app)

        upload = _upload(
            client, "conv-A",
            content=b"keep this in RAG",
            filename="paper.md",
            persist="true",
        ).json()

        # Simulate the row that ingest_document would have written.
        in_memory_db.execute(
            "INSERT INTO documents (id, content, source, doc_type, "
            "embedding_status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'file', 'clean', '2026-05-01', '2026-05-01')",
            ("doc-1", "keep this in RAG", "paper.md"),
        )
        in_memory_db.commit()

        resp = client.delete(
            f"/api/chat/attachments/{upload['id']}", headers=_auth(),
        )
        assert resp.status_code == 200

        # documents row removed by the delete handler.
        rag_row = in_memory_db.fetchone(
            "SELECT id FROM documents WHERE source = ?", ("paper.md",),
        )
        assert rag_row is None

    def test_delete_missing_row_returns_404(self, app):
        client = TestClient(app)
        resp = client.delete(
            "/api/chat/attachments/does-not-exist", headers=_auth(),
        )
        assert resp.status_code == 404


# ── GET list ────────────────────────────────────────────────────────────────


class TestList:
    def test_returns_attachments_for_the_right_conversation(
        self, app, in_memory_db,
    ):
        _seed_conversation(in_memory_db, cid="conv-A")
        _seed_conversation(in_memory_db, cid="conv-B")
        client = TestClient(app)

        a1 = _upload(
            client, "conv-A", content=b"a1", filename="a1.txt", persist="false",
        ).json()
        a2 = _upload(
            client, "conv-A", content=b"a2", filename="a2.md", persist="true",
        ).json()
        b1 = _upload(
            client, "conv-B", content=b"b1", filename="b1.txt", persist="false",
        ).json()

        resp = client.get("/api/chat/conv-A/attachments", headers=_auth())
        assert resp.status_code == 200
        rows = resp.json()
        ids = [r["id"] for r in rows]
        assert a1["id"] in ids
        assert a2["id"] in ids
        assert b1["id"] not in ids
        # persist flag round-trips as a bool
        for r in rows:
            assert isinstance(r["persist"], bool)
            if r["filename"] == "a2.md":
                assert r["persist"] is True
            elif r["filename"] == "a1.txt":
                assert r["persist"] is False

    def test_returns_empty_list_for_unknown_conversation(self, app):
        client = TestClient(app)
        resp = client.get("/api/chat/nope/attachments", headers=_auth())
        assert resp.status_code == 200
        assert resp.json() == []


# ── Auth ────────────────────────────────────────────────────────────────────


class TestAuth:
    def test_unauthenticated_upload_returns_401(self, app, in_memory_db):
        _seed_conversation(in_memory_db)
        client = TestClient(app)

        files = {"file": ("nope.txt", BytesIO(b"nope"), "text/plain")}
        resp = client.post(
            "/api/chat/conv-A/attach",
            files=files, data={"persist": "false"},
        )
        assert resp.status_code == 401

    def test_unauthenticated_list_returns_401(self, app):
        client = TestClient(app)
        resp = client.get("/api/chat/conv-A/attachments")
        assert resp.status_code == 401

    def test_unauthenticated_delete_returns_401(self, app):
        client = TestClient(app)
        resp = client.delete("/api/chat/attachments/whatever")
        assert resp.status_code == 401
