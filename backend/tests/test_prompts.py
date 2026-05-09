"""tests/test_prompts.py — HTTP-level tests for /api/prompt-templates routes.

Exercises the full CRUD surface plus list ordering, validation, 404
handling and bearer-auth gating.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes import prompt_templates as prompt_templates_routes
from server import BearerAuthMiddleware


TOKEN = "test-token-prompt-templates"


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def app(in_memory_db):
    a = FastAPI()
    a.add_middleware(BearerAuthMiddleware, expected_token=TOKEN)
    a.include_router(
        prompt_templates_routes.router, prefix="/api/prompt-templates"
    )
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


def _create(client: TestClient, **overrides) -> dict:
    payload = {
        "title": "My snippet",
        "body": "Hello from a snippet.",
        "kind": "snippet",
        "tags": "intro,greeting",
    }
    payload.update(overrides)
    resp = client.post("/api/prompt-templates", json=payload, headers=_auth())
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── CRUD round-trip ───────────────────────────────────────────────────────────


class TestCrudRoundTrip:
    def test_create_then_get_then_update_then_delete(self, client):
        created = _create(client)
        assert created["id"]
        assert created["title"] == "My snippet"
        assert created["body"] == "Hello from a snippet."
        assert created["kind"] == "snippet"
        assert created["tags"] == "intro,greeting"
        assert created["use_count"] == 0

        # GET single
        resp = client.get(
            f"/api/prompt-templates/{created['id']}", headers=_auth()
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == created["id"]

        # PUT update — only title and tags
        resp = client.put(
            f"/api/prompt-templates/{created['id']}",
            json={"title": "Updated title", "tags": "newtag"},
            headers=_auth(),
        )
        assert resp.status_code == 200
        updated = resp.json()
        assert updated["title"] == "Updated title"
        assert updated["body"] == "Hello from a snippet."  # unchanged
        assert updated["tags"] == "newtag"
        assert updated["kind"] == "snippet"

        # DELETE
        resp = client.delete(
            f"/api/prompt-templates/{created['id']}", headers=_auth()
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        # GET 404 after delete
        resp = client.get(
            f"/api/prompt-templates/{created['id']}", headers=_auth()
        )
        assert resp.status_code == 404


# ── List ordering ─────────────────────────────────────────────────────────────


class TestListOrdering:
    def test_list_orders_by_use_count_desc(self, client):
        a = _create(client, title="A")
        b = _create(client, title="B")
        c = _create(client, title="C")

        # Bump B twice and C once so order should be B, C, A.
        client.post(
            f"/api/prompt-templates/{b['id']}/use", headers=_auth()
        )
        client.post(
            f"/api/prompt-templates/{b['id']}/use", headers=_auth()
        )
        client.post(
            f"/api/prompt-templates/{c['id']}/use", headers=_auth()
        )

        resp = client.get("/api/prompt-templates", headers=_auth())
        assert resp.status_code == 200
        rows = resp.json()
        ids = [r["id"] for r in rows]
        # Most-used first: B (2), C (1), A (0)
        assert ids == [b["id"], c["id"], a["id"]]
        assert rows[0]["use_count"] == 2
        assert rows[1]["use_count"] == 1
        assert rows[2]["use_count"] == 0


# ── Validation ────────────────────────────────────────────────────────────────


class TestValidation:
    def test_rejects_too_long_body(self, client):
        oversized = "x" * 10_001
        resp = client.post(
            "/api/prompt-templates",
            json={"title": "Big", "body": oversized, "kind": "snippet"},
            headers=_auth(),
        )
        assert resp.status_code == 422

    def test_rejects_too_long_title(self, client):
        resp = client.post(
            "/api/prompt-templates",
            json={
                "title": "x" * 101,
                "body": "ok",
                "kind": "snippet",
            },
            headers=_auth(),
        )
        assert resp.status_code == 422

    def test_rejects_empty_title(self, client):
        resp = client.post(
            "/api/prompt-templates",
            json={"title": "", "body": "ok", "kind": "snippet"},
            headers=_auth(),
        )
        assert resp.status_code == 422

    def test_rejects_invalid_kind(self, client):
        resp = client.post(
            "/api/prompt-templates",
            json={"title": "ok", "body": "ok", "kind": "wat"},
            headers=_auth(),
        )
        assert resp.status_code == 422

    def test_accepts_system_prompt_kind(self, client):
        out = _create(client, kind="system_prompt")
        assert out["kind"] == "system_prompt"

    def test_update_rejects_invalid_kind(self, client):
        created = _create(client)
        resp = client.put(
            f"/api/prompt-templates/{created['id']}",
            json={"kind": "nonsense"},
            headers=_auth(),
        )
        assert resp.status_code == 422


# ── 404 ───────────────────────────────────────────────────────────────────────


class TestNotFound:
    def test_get_missing_returns_404(self, client):
        resp = client.get(
            "/api/prompt-templates/does-not-exist", headers=_auth()
        )
        assert resp.status_code == 404

    def test_update_missing_returns_404(self, client):
        resp = client.put(
            "/api/prompt-templates/does-not-exist",
            json={"title": "x"},
            headers=_auth(),
        )
        assert resp.status_code == 404

    def test_delete_missing_returns_404(self, client):
        resp = client.delete(
            "/api/prompt-templates/does-not-exist", headers=_auth()
        )
        assert resp.status_code == 404

    def test_use_missing_returns_404(self, client):
        resp = client.post(
            "/api/prompt-templates/does-not-exist/use", headers=_auth()
        )
        assert resp.status_code == 404


# ── Auth ──────────────────────────────────────────────────────────────────────


class TestAuth:
    def test_unauthenticated_list_returns_401(self, client):
        resp = client.get("/api/prompt-templates")
        assert resp.status_code == 401

    def test_unauthenticated_create_returns_401(self, client):
        resp = client.post(
            "/api/prompt-templates",
            json={"title": "x", "body": "x", "kind": "snippet"},
        )
        assert resp.status_code == 401
