"""tests/test_system_routes.py — HTTP-level tests for /api/system routes.

Exercises the auth middleware, local-model browser endpoints, and the
active-model setting via FastAPI's TestClient. The fixture builds a minimal
app with a faked API facade so we don't touch ChatOrchestrator or any of the
deferred init paths that would slow tests down.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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

    fake_api = MagicMock()
    fake_api.local_client = fake_local
    fake_api._settings = settings

    fake_container = MagicMock()
    fake_container.api = fake_api

    app = FastAPI()
    app.add_middleware(BearerAuthMiddleware, expected_token=TOKEN)
    app.state.container = fake_container
    app.include_router(system_routes.router, prefix="/api/system")
    return app, settings


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


class TestLocalModelsList:
    def test_returns_models_and_current(self, app_with_fake_api):
        app, settings = app_with_fake_api
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
        app, _ = app_with_fake_api
        # Replace the api.local_client handle with None to simulate init failure.
        app.state.container.api.local_client = None
        client = TestClient(app)

        resp = client.get("/api/system/local_models", headers=_auth_headers())

        assert resp.status_code == 200
        assert resp.json()["models"] == []

    def test_rejects_without_bearer_auth(self, app_with_fake_api):
        app, _ = app_with_fake_api
        client = TestClient(app)
        resp = client.get("/api/system/local_models")
        assert resp.status_code == 401


class TestActiveLocalModel:
    def test_post_updates_setting(self, app_with_fake_api):
        app, settings = app_with_fake_api
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
        app, settings = app_with_fake_api
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
        app, _ = app_with_fake_api
        client = TestClient(app)
        resp = client.post(
            "/api/system/local_model/active",
            json={"model_id": "any"},
        )
        assert resp.status_code == 401
