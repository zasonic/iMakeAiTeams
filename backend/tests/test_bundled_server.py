"""tests/test_bundled_server.py — exercise BundledServer download + lifecycle.

We don't actually fetch from Hugging Face. ``download_model`` accepts a
``_requests`` kwarg so tests can pass a fake namespace whose ``get`` returns
a context-managed response that yields canned bytes. The catalog's expected
sha256 is monkeypatched to match (or mismatch, in the negative test).

``start`` / ``stop`` / ``is_running`` are tested by replacing
``subprocess.Popen`` with a fake whose ``poll`` returns None until ``stop``
is called, since spawning a real binary in CI is out of scope.
"""

from __future__ import annotations

import hashlib
import io
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.settings import Settings
from services import bundled_server as bundled_module
from services.bundled_server import BundledServer, BundledServerError


# ── Helpers / fakes ──────────────────────────────────────────────────────────


class _FakeStreamResponse:
    """Mimics requests.Response when used as a context manager + iter_content."""

    def __init__(self, content: bytes, status_code: int = 200,
                 content_length: int | None = None) -> None:
        self._content = content
        self.status_code = status_code
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def iter_content(self, chunk_size: int = 1):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


class _FakeRequestsModule:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.calls: list[str] = []

    def get(self, url, *, stream=False, timeout=None):
        self.calls.append(url)
        return _FakeStreamResponse(self._payload,
                                    content_length=len(self._payload))


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def patched_dirs(tmp_path, monkeypatch):
    """Redirect bundled paths into tmp_path so tests don't pollute userData."""
    models = tmp_path / "models"
    models.mkdir()
    port_file = tmp_path / "bundled_server.port"
    monkeypatch.setattr(bundled_module.paths, "bundled_models_dir",
                        lambda: models)
    monkeypatch.setattr(bundled_module.paths, "bundled_server_port_file",
                        lambda: port_file)
    monkeypatch.setattr(bundled_module.paths, "bundled_models_catalog_path",
                        lambda: tmp_path / "missing_catalog.json")
    return tmp_path


@pytest.fixture
def server(patched_dirs, tmp_path):
    s = Settings(tmp_path / "settings.json")
    return BundledServer(s)


# ── Catalog ──────────────────────────────────────────────────────────────────


class TestResolveModel:
    def test_unknown_id_raises(self, server):
        with pytest.raises(BundledServerError):
            server.resolve_model("does-not-exist")

    def test_default_id_resolves_via_live_lookup(self, server, monkeypatch):
        called = {"n": 0}
        def _fake_lookup(repo, filename, *, timeout=10.0):
            called["n"] += 1
            return ("sha-from-hf", 12345)
        monkeypatch.setattr(bundled_module, "_hf_metadata_lookup", _fake_lookup)
        entry = server.resolve_model(bundled_module.DEFAULT_MODEL_ID)
        assert entry["expected_sha256"] == "sha-from-hf"
        assert entry["expected_size_bytes"] == 12345
        assert called["n"] == 1


# ── Download ─────────────────────────────────────────────────────────────────


class TestDownloadModel:
    def test_success_writes_partial_then_renames(
        self, server, patched_dirs, monkeypatch, in_memory_db,
    ):
        payload = b"GGUF" + b"\x00" * 1024
        sha = hashlib.sha256(payload).hexdigest()
        monkeypatch.setattr(
            server, "resolve_model",
            lambda mid: {
                "repo":     "Qwen/Qwen3-4B-Instruct-GGUF",
                "filename": "Qwen3-4B-Instruct-Q4_K_M.gguf",
                "expected_sha256":     sha,
                "expected_size_bytes": len(payload),
            },
        )
        fake_req = _FakeRequestsModule(payload)
        progress: list[tuple[int, int]] = []
        def _on_progress(d, t):
            progress.append((d, t))

        result = server.download_model(
            "Qwen3-4B-Instruct-Q4_K_M",
            on_progress=_on_progress,
            _requests=fake_req,
        )

        target = patched_dirs / "models" / "Qwen3-4B-Instruct-Q4_K_M.gguf"
        partial = target.with_suffix(target.suffix + ".partial")
        assert target.exists()
        assert not partial.exists()
        assert result["expected_sha256"] == sha
        assert result["skipped"] is False
        assert progress  # at least one chunk

        # DB row inserted exactly once
        rows = in_memory_db.get_db().execute(
            "SELECT model_id, sha256, size_bytes FROM bundled_models"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "Qwen3-4B-Instruct-Q4_K_M"
        assert rows[0][1] == sha

    def test_sha_mismatch_drops_partial_and_no_db_row(
        self, server, patched_dirs, monkeypatch, in_memory_db,
    ):
        payload = b"GGUF" + b"\xff" * 32
        wrong_sha = "0" * 64
        monkeypatch.setattr(
            server, "resolve_model",
            lambda mid: {
                "repo":     "Qwen/Qwen3-4B-Instruct-GGUF",
                "filename": "Qwen3-4B-Instruct-Q4_K_M.gguf",
                "expected_sha256":     wrong_sha,
                "expected_size_bytes": len(payload),
            },
        )
        fake_req = _FakeRequestsModule(payload)

        with pytest.raises(BundledServerError, match="sha256 mismatch"):
            server.download_model(
                "Qwen3-4B-Instruct-Q4_K_M",
                _requests=fake_req,
            )

        target = patched_dirs / "models" / "Qwen3-4B-Instruct-Q4_K_M.gguf"
        partial = target.with_suffix(target.suffix + ".partial")
        assert not partial.exists()
        assert not target.exists()

        rows = in_memory_db.get_db().execute(
            "SELECT model_id FROM bundled_models"
        ).fetchall()
        assert rows == []

    def test_existing_valid_file_short_circuits(
        self, server, patched_dirs, monkeypatch, in_memory_db,
    ):
        payload = b"already there"
        sha = hashlib.sha256(payload).hexdigest()
        target = patched_dirs / "models" / "Qwen3-4B-Instruct-Q4_K_M.gguf"
        target.write_bytes(payload)
        monkeypatch.setattr(
            server, "resolve_model",
            lambda mid: {
                "repo":     "Qwen/Qwen3-4B-Instruct-GGUF",
                "filename": "Qwen3-4B-Instruct-Q4_K_M.gguf",
                "expected_sha256":     sha,
                "expected_size_bytes": len(payload),
            },
        )
        fake_req = _FakeRequestsModule(b"should not be requested")

        result = server.download_model(
            "Qwen3-4B-Instruct-Q4_K_M",
            _requests=fake_req,
        )

        assert result["skipped"] is True
        assert fake_req.calls == []  # cached path skipped HTTP entirely


# ── Subprocess lifecycle ─────────────────────────────────────────────────────


class _FakePopen:
    """Stand-in for subprocess.Popen used by start()."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.pid = 4242
        self._alive = True
        self.stdout = io.StringIO("")  # already EOF, drain returns immediately
        self.terminated = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False
        self.terminated = True

    def kill(self):
        self._alive = False

    def wait(self, timeout=None):
        return 0


class TestLifecycle:
    def test_start_writes_port_file_and_is_running(
        self, server, patched_dirs, monkeypatch, tmp_path,
    ):
        binary = tmp_path / "llama-server.exe"
        binary.write_bytes(b"\x00")
        model = tmp_path / "models" / "fake.gguf"
        model.write_bytes(b"\x00")
        monkeypatch.setattr(bundled_module.paths, "bundled_server_binary",
                            lambda: binary)
        monkeypatch.setattr(bundled_module, "_bind_free_port", lambda: 56789)
        with patch.object(bundled_module.subprocess, "Popen", _FakePopen):
            port = server.start(str(model), model_id="x")

        assert port == 56789
        assert server.is_running()
        assert server.port() == 56789
        assert server.model_id() == "x"
        port_file = patched_dirs / "bundled_server.port"
        assert port_file.read_text(encoding="utf-8") == "56789"

    def test_stop_clears_port_file_and_state(
        self, server, patched_dirs, monkeypatch, tmp_path,
    ):
        binary = tmp_path / "llama-server.exe"
        binary.write_bytes(b"\x00")
        model = tmp_path / "models" / "fake.gguf"
        model.write_bytes(b"\x00")
        monkeypatch.setattr(bundled_module.paths, "bundled_server_binary",
                            lambda: binary)
        monkeypatch.setattr(bundled_module, "_bind_free_port", lambda: 56790)
        with patch.object(bundled_module.subprocess, "Popen", _FakePopen):
            server.start(str(model), model_id="x")
            assert server.is_running()
            server.stop()

        assert not server.is_running()
        assert server.port() is None
        port_file = patched_dirs / "bundled_server.port"
        assert not port_file.exists()

    def test_start_missing_binary_raises(self, server, monkeypatch, tmp_path):
        monkeypatch.setattr(
            bundled_module.paths, "bundled_server_binary",
            lambda: tmp_path / "no-such-file.exe",
        )
        with pytest.raises(BundledServerError, match="not found"):
            server.start(str(tmp_path / "any.gguf"))

    def test_start_missing_model_raises(self, server, monkeypatch, tmp_path):
        binary = tmp_path / "llama-server.exe"
        binary.write_bytes(b"\x00")
        monkeypatch.setattr(bundled_module.paths, "bundled_server_binary",
                            lambda: binary)
        with pytest.raises(BundledServerError, match="model file missing"):
            server.start(str(tmp_path / "no-such.gguf"))
