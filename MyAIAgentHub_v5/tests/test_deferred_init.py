"""
tests/test_deferred_init.py — Cover the _run_deferred_init flow without
dragging in the full core.api module (anthropic is not installed in CI).

The test constructs a bare object, attaches only the handful of attributes
_run_deferred_init actually touches (_status, _rag, _safe_init, _emit, _log,
_load_shared_embedder), and invokes the method bound from core.api.
"""

import sys
import types
from unittest.mock import MagicMock


def _get_run_deferred_init():
    """Pull just _run_deferred_init off core.api without executing its
    top-level side effects. We stub the heavy imports so `import core.api`
    succeeds under a bare Python install without anthropic/webview.

    We additionally stub a handful of services that core.api imports at
    module load time and which either require anthropic or contain
    unrelated type-annotation issues — those bugs live in adversarial
    code paths this test doesn't exercise."""
    from unittest.mock import MagicMock as _MM

    # External packages
    for name in ("anthropic", "anthropic._models", "anthropic.types", "webview"):
        sys.modules.setdefault(name, _MM(name=name))
    # App services that fail to import in a bare env (pre-existing issues
    # unrelated to deferred-init).
    for name in (
        "services.claude_client",
        "services.adversarial_debate",
        "services.chat_orchestrator",
        "services.guardrails_gate",
    ):
        sys.modules.setdefault(name, _MM(name=name))

    import core.api as api_module
    return api_module.API._run_deferred_init, api_module


class _StubRag:
    def __init__(self):
        self._model = None
        self.loaded = False

    def load(self, path):
        self.loaded = True

    def chunk_count(self):
        return 7


def _make_fake_api(run_deferred, api_module, monkeypatch, tmp_path,
                   embedder_raises=False, has_cache=False):
    """Build a stand-in for API with the minimum surface _run_deferred_init needs."""

    # Redirect paths.* to tmp so we never touch the real user dir.
    monkeypatch.setattr(api_module.paths, "rag_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module.paths, "vector_store_dir", lambda: tmp_path / "vec")

    # Stub semantic_search to observe calls
    ss_calls = []

    class _SS:
        @staticmethod
        def init_vector_store(*a, **kw):
            ss_calls.append(("init", a, kw))
            return True

        @staticmethod
        def start_background_indexer(*a, **kw):
            ss_calls.append(("indexer", a, kw))
            return True

    monkeypatch.setattr(api_module, "semantic_search", _SS)

    if has_cache:
        (tmp_path / "index.npz").write_bytes(b"fake-cache")

    # Build a throwaway object with just the attributes _run_deferred_init touches.
    stub = types.SimpleNamespace()
    stub._log = MagicMock()
    stub._window = None
    stub._rag = _StubRag()
    stub._status = {
        "embedder": {"ok": False, "error": None, "pending": True},
        "rag_load": {"ok": False, "error": None, "pending": True},
        "semantic_search": {"ok": False, "error": None, "pending": True},
        "semantic_search_indexer": {"ok": False, "error": None, "pending": True},
    }
    stub.emitted = []

    def _fake_safe_init(name, factory, required=False, fallback=None):
        try:
            result = factory()
            stub._status[name] = {"ok": True, "error": None}
            return result
        except Exception as exc:
            stub._status[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            return fallback

    def _fake_emit(event, payload=None):
        stub.emitted.append((event, payload or {}))

    def _fake_emit_service_update(name):
        entry = stub._status.get(name, {})
        _fake_emit("service_status_update", {"service": name, **entry})

    def _fake_load_shared_embedder():
        if embedder_raises:
            raise RuntimeError("model download blocked")
        return object()  # opaque sentinel — represents the loaded model

    stub._safe_init = _fake_safe_init
    stub._emit = _fake_emit
    stub._emit_service_update = _fake_emit_service_update
    stub._load_shared_embedder = _fake_load_shared_embedder

    return stub, ss_calls


def test_run_deferred_init_happy_path(monkeypatch, tmp_path):
    run_deferred, api_module = _get_run_deferred_init()
    stub, ss_calls = _make_fake_api(run_deferred, api_module, monkeypatch, tmp_path,
                                     has_cache=True)

    run_deferred(stub)

    # All four deferred steps now report ok.
    for name in ("embedder", "rag_load", "semantic_search", "semantic_search_indexer"):
        assert stub._status[name]["ok"] is True, f"{name} not ok after deferred init"

    # Embedder was attached to the RAG index, and the cache was loaded.
    assert stub._rag._model is not None
    assert stub._rag.loaded is True

    # semantic_search received the shared model.
    init_call = next(c for c in ss_calls if c[0] == "init")
    assert init_call[2]["shared_model"] is stub._rag._model

    # Live updates emitted in order.
    events = [p["service"] for e, p in stub.emitted if e == "service_status_update"]
    assert events == ["embedder", "rag_load", "semantic_search",
                      "semantic_search_indexer"]


def test_run_deferred_init_embedder_failure_degrades_gracefully(monkeypatch, tmp_path):
    run_deferred, api_module = _get_run_deferred_init()
    stub, ss_calls = _make_fake_api(run_deferred, api_module, monkeypatch, tmp_path,
                                     embedder_raises=True, has_cache=True)

    run_deferred(stub)  # must not raise

    assert stub._status["embedder"]["ok"] is False
    assert "RuntimeError" in stub._status["embedder"]["error"]
    # RAG cache load is skipped because embedder wasn't loaded.
    assert stub._rag.loaded is False
    assert stub._status["rag_load"]["ok"] is False
    # semantic_search still runs (can use its own embedding fn if available).
    assert any(c[0] == "init" for c in ss_calls)


def test_run_deferred_init_no_cache_marks_rag_load_ok_when_embedder_present(
        monkeypatch, tmp_path):
    run_deferred, api_module = _get_run_deferred_init()
    stub, _ = _make_fake_api(run_deferred, api_module, monkeypatch, tmp_path,
                             has_cache=False)

    run_deferred(stub)

    # No cache file exists — rag_load should be ok=True with no error (fresh install).
    assert stub._status["rag_load"]["ok"] is True
    assert stub._status["rag_load"]["error"] is None
