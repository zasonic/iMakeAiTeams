"""tests/test_system_routes.py — HTTP-level tests for /api/system routes.

Exercises the auth middleware, local-model browser endpoints, the
active-model setting, and the Phase 9 bundled llama.cpp server endpoints
via FastAPI's TestClient. The fixture builds a minimal app with a faked
API facade so we don't touch ChatOrchestrator or any of the deferred init
paths that would slow tests down.
"""

import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import sse_events
from routes import system as system_routes
from server import BearerAuthMiddleware


TOKEN = "test-token-xyz"


@pytest.fixture
def app_with_fake_api(tmp_path):
    """Build a minimal FastAPI app wired to a fake API facade."""
    from core.settings import Settings
    settings = Settings(tmp_path / "settings.json")

    fake_local = MagicMock()
    fake_local.list_local_models.return_value = [
        {
            "id":             "qwen3-30b-a3b-q4_k_m",
            "size_bytes":     8_500_000_000,
            "context_length": 32_768,
            "quantization":   "Q4_K_M",
            "backend":        "ollama",
            "loaded":         False,
        },
        {
            "id":             "lmstudio-community/Qwen3-7B",
            "size_bytes":     None,
            "context_length": None,
            "quantization":   None,
            "backend":        "lm_studio",
            "loaded":         False,
        },
    ]

    fake_bundled = MagicMock()
    fake_bundled.is_running.return_value = False
    fake_bundled.port.return_value = None
    fake_bundled.model_id.return_value = None

    fake_api = MagicMock()
    fake_api.local_client = fake_local
    fake_api.bundled_server = fake_bundled
    fake_api._settings = settings

    fake_container = MagicMock()
    fake_container.api = fake_api

    app = FastAPI()
    app.add_middleware(BearerAuthMiddleware, expected_token=TOKEN)
    app.state.container = fake_container
    app.include_router(system_routes.router, prefix="/api/system")
    return app, settings, fake_bundled


def _drain_sse_queue() -> list[dict]:
    """Pop everything currently queued in the module-level sse_events queue.

    Bypasses ``drain()`` (which awaits an asyncio loop) by reaching into
    the test-only internals — the queue exposes its lock and items so we
    can flush synchronously between operations in a TestClient run.
    """
    q = sse_events._queue
    with q._lock:
        out = list(q._items)
        q._items.clear()
    return out


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


class TestLocalModelsList:
    def test_returns_models_and_current(self, app_with_fake_api):
        app, settings, _bundled = app_with_fake_api
        settings.set("default_local_model", "qwen3-30b-a3b-q4_k_m")
        client = TestClient(app)

        resp = client.get("/api/system/local_models", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["current"] == "qwen3-30b-a3b-q4_k_m"
        assert isinstance(body["models"], list)
        assert len(body["models"]) == 2
        first = body["models"][0]
        assert set(first.keys()) >= {
            "id", "size_bytes", "context_length",
            "quantization", "backend", "loaded",
        }
        assert first["backend"] == "ollama"
        assert body["models"][1]["backend"] == "lm_studio"

    def test_returns_empty_when_local_client_unavailable(self, app_with_fake_api):
        app, _, _bundled = app_with_fake_api
        # Replace the api.local_client handle with None to simulate init failure.
        app.state.container.api.local_client = None
        client = TestClient(app)

        resp = client.get("/api/system/local_models", headers=_auth_headers())

        assert resp.status_code == 200
        assert resp.json()["models"] == []

    def test_rejects_without_bearer_auth(self, app_with_fake_api):
        app, _, _bundled = app_with_fake_api
        client = TestClient(app)
        resp = client.get("/api/system/local_models")
        assert resp.status_code == 401


class TestActiveLocalModel:
    def test_post_updates_setting(self, app_with_fake_api):
        app, settings, _bundled = app_with_fake_api
        client = TestClient(app)

        resp = client.post(
            "/api/system/local_model/active",
            json={"model_id": "qwen3-30b-a3b-q4_k_m"},
            headers=_auth_headers(),
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["current"] == "qwen3-30b-a3b-q4_k_m"
        assert settings.get("default_local_model") == "qwen3-30b-a3b-q4_k_m"

    def test_post_overwrites_previous_value(self, app_with_fake_api):
        app, settings, _bundled = app_with_fake_api
        settings.set("default_local_model", "old-model")
        client = TestClient(app)

        resp = client.post(
            "/api/system/local_model/active",
            json={"model_id": "new-model"},
            headers=_auth_headers(),
        )

        assert resp.status_code == 200
        assert resp.json()["current"] == "new-model"
        assert settings.get("default_local_model") == "new-model"

    def test_rejects_without_bearer_auth(self, app_with_fake_api):
        app, _, _bundled = app_with_fake_api
        client = TestClient(app)
        resp = client.post(
            "/api/system/local_model/active",
            json={"model_id": "any"},
        )
        assert resp.status_code == 401


# ── Phase 9: Bundled llama.cpp endpoints ────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_download_lock():
    """The /bundled/download handler keeps a module-level "in-flight" guard.
    Reset it between tests so a previous test's leak doesn't 503 the next.
    """
    system_routes._bundled_download_running = False
    yield
    system_routes._bundled_download_running = False


class TestBundledStatus:
    def test_returns_running_false_initially(self, app_with_fake_api):
        app, _, _bundled = app_with_fake_api
        client = TestClient(app)
        resp = client.get("/api/system/bundled/status",
                          headers=_auth_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["running"] is False
        assert body["available"] is True

    def test_reports_available_false_when_handle_is_none(self, app_with_fake_api):
        app, _, _ = app_with_fake_api
        app.state.container.api.bundled_server = None
        client = TestClient(app)
        resp = client.get("/api/system/bundled/status",
                          headers=_auth_headers())
        assert resp.status_code == 200
        assert resp.json()["available"] is False

    def test_rejects_without_bearer_auth(self, app_with_fake_api):
        app, _, _ = app_with_fake_api
        client = TestClient(app)
        resp = client.get("/api/system/bundled/status")
        assert resp.status_code == 401


class TestBundledDownload:
    def _wait_for_complete(self, fake_bundled, timeout: float = 2.0) -> None:
        """Spin until the background-thread download_model has fired."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if fake_bundled.download_model.called:
                # Give the background thread a beat to publish the SSE event
                # *and* flip the in-flight guard before tests inspect either.
                time.sleep(0.05)
                if not system_routes._bundled_download_running:
                    return
            time.sleep(0.01)

    def test_kicks_off_background_download_and_emits_complete(
        self, app_with_fake_api,
    ):
        app, settings, fake_bundled = app_with_fake_api
        fake_bundled.download_model.return_value = {
            "file_path": "/fake/path.gguf",
            "expected_sha256":     "abc",
            "expected_size_bytes": 1234,
        }
        _drain_sse_queue()

        client = TestClient(app)
        resp = client.post(
            "/api/system/bundled/download",
            json={"model_id": "Qwen3-4B-Instruct-Q4_K_M"},
            headers=_auth_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json()["model_id"] == "Qwen3-4B-Instruct-Q4_K_M"

        self._wait_for_complete(fake_bundled)
        events = _drain_sse_queue()
        names = [e["event"] for e in events]
        assert "bundled_download_complete" in names
        assert settings.get("local_backend_mode") == "bundled"
        assert settings.get("bundled_model_id") == "Qwen3-4B-Instruct-Q4_K_M"

    def test_emits_error_event_on_bundled_server_error(self, app_with_fake_api):
        from services.bundled_server import BundledServerError

        app, settings, fake_bundled = app_with_fake_api
        fake_bundled.download_model.side_effect = BundledServerError("boom")
        _drain_sse_queue()

        client = TestClient(app)
        resp = client.post(
            "/api/system/bundled/download",
            json={"model_id": "Qwen3-4B-Instruct-Q4_K_M"},
            headers=_auth_headers(),
        )
        assert resp.status_code == 200

        self._wait_for_complete(fake_bundled)
        events = _drain_sse_queue()
        names = [e["event"] for e in events]
        assert "bundled_download_error" in names
        # local_backend_mode must NOT be flipped on a failed download.
        assert settings.get("local_backend_mode") == "auto"

    def test_concurrent_downloads_rejected(self, app_with_fake_api):
        app, _, fake_bundled = app_with_fake_api
        # Simulate a slow download by blocking the worker until we say so.
        block = MagicMock()
        block.call_count = 0
        def _slow(*args, **kwargs):
            block.call_count += 1
            time.sleep(0.5)
            return {"file_path": "/x", "expected_sha256": "a",
                    "expected_size_bytes": 1}
        fake_bundled.download_model.side_effect = _slow
        client = TestClient(app)

        first = client.post(
            "/api/system/bundled/download",
            json={"model_id": "x"},
            headers=_auth_headers(),
        )
        assert first.status_code == 200
        assert first.json()["ok"] is True

        second = client.post(
            "/api/system/bundled/download",
            json={"model_id": "x"},
            headers=_auth_headers(),
        )
        assert second.status_code == 200
        assert second.json()["ok"] is False
        assert "in progress" in second.json()["error"]

        # Drain the wait so the lock clears for subsequent tests.
        deadline = time.time() + 5.0
        while time.time() < deadline and system_routes._bundled_download_running:
            time.sleep(0.05)

    def test_rejects_without_bearer_auth(self, app_with_fake_api):
        app, _, _ = app_with_fake_api
        client = TestClient(app)
        resp = client.post("/api/system/bundled/download", json={})
        assert resp.status_code == 401


class TestBundledStartStop:
    def test_start_returns_404_when_no_db_row(self, app_with_fake_api, tmp_path):
        app, _, _ = app_with_fake_api
        # No row in bundled_models for "missing-id" → should report failure.
        import db
        db.init_db(tmp_path / "test.db")
        try:
            client = TestClient(app)
            resp = client.post(
                "/api/system/bundled/start",
                json={"model_id": "missing-id"},
                headers=_auth_headers(),
            )
            assert resp.status_code == 200
            assert resp.json()["ok"] is False
            assert "not downloaded" in resp.json()["error"]
        finally:
            db._conn.close()
            db._conn = None
            db._db_path = None

    def test_start_calls_bundled_server_when_row_exists(
        self, app_with_fake_api, tmp_path,
    ):
        app, settings, fake_bundled = app_with_fake_api
        fake_bundled.start.return_value = 56789
        import db
        db.init_db(tmp_path / "test.db")
        try:
            db.get_db().execute(
                """INSERT INTO bundled_models
                   (model_id, file_path, size_bytes, sha256, downloaded_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("xyz", "/fake.gguf", 100, "sha", "2026-05-01T00:00:00Z"),
            )
            db.get_db().commit()
            client = TestClient(app)
            resp = client.post(
                "/api/system/bundled/start",
                json={"model_id": "xyz"},
                headers=_auth_headers(),
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["ok"] is True
            assert body["port"] == 56789
            assert settings.get("local_backend_mode") == "bundled"
            assert settings.get("bundled_model_id") == "xyz"
        finally:
            db._conn.close()
            db._conn = None
            db._db_path = None

    def test_stop_invokes_bundled_server_stop(self, app_with_fake_api):
        app, _, fake_bundled = app_with_fake_api
        client = TestClient(app)
        resp = client.post("/api/system/bundled/stop",
                           headers=_auth_headers())
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        fake_bundled.stop.assert_called_once()

    def test_rejects_without_bearer_auth(self, app_with_fake_api):
        app, _, _ = app_with_fake_api
        client = TestClient(app)
        for path in ("/api/system/bundled/start", "/api/system/bundled/stop"):
            resp = client.post(path, json={})
            assert resp.status_code == 401, f"path {path} did not 401"
