"""
tests/test_requires_decorator.py — Verify the _requires short-circuit decorator.

The decorator reads self._status and short-circuits a method when the
backing service reported ok=False during API._safe_init. It also emits a
service_unavailable event so the frontend can surface a toast.
"""

from core.service_guard import requires as _requires


class _FakeAPI:
    """Minimal stand-in that mimics the API attributes _requires touches."""

    def __init__(self, status: dict):
        self._status = status
        self.emitted = []

    def _emit(self, event, payload):
        self.emitted.append((event, payload))


def test_decorator_calls_through_when_service_ok():
    api = _FakeAPI({"chat_orchestrator": {"ok": True, "error": None}})

    @_requires("chat_orchestrator", default=[])
    def chat_list(self):
        return ["a", "b"]

    assert chat_list(api) == ["a", "b"]
    assert api.emitted == []


def test_decorator_returns_default_when_service_failed():
    api = _FakeAPI({"chat_orchestrator": {"ok": False, "error": "boom"}})
    called = []

    @_requires("chat_orchestrator", default=[])
    def chat_list(self):
        called.append(True)
        return ["a", "b"]

    result = chat_list(api)
    assert result == []
    assert called == []  # body must not execute


def test_decorator_emits_service_unavailable_event():
    api = _FakeAPI({"hook_manager": {"ok": False, "error": "import failed"}})

    @_requires("hook_manager", default=[])
    def hook_list(self):
        return ["should-not-run"]

    hook_list(api)

    assert len(api.emitted) == 1
    event, payload = api.emitted[0]
    assert event == "service_unavailable"
    assert payload["service"] == "hook_manager"
    assert payload["error"] == "import failed"
    assert payload["method"] == "hook_list"


def test_decorator_treats_missing_status_as_unavailable():
    api = _FakeAPI({})  # no entry at all — treat as failed

    @_requires("rag_index", default={"error": "rag unavailable"})
    def rag_search(self, q):
        return ["real-result"]

    assert rag_search(api, "hello") == {"error": "rag unavailable"}
    assert api.emitted[0][0] == "service_unavailable"


def test_decorator_preserves_args_kwargs_and_metadata():
    api = _FakeAPI({"chat_orchestrator": {"ok": True}})

    @_requires("chat_orchestrator", default=None)
    def chat_get_messages(self, conversation_id, limit=10):
        """Return messages."""
        return [conversation_id, limit]

    assert chat_get_messages(api, "c1", limit=5) == ["c1", 5]
    # functools.wraps preserves name + docstring
    assert chat_get_messages.__name__ == "chat_get_messages"
    assert chat_get_messages.__doc__ == "Return messages."


def test_decorator_is_quiet_for_pending_services():
    """A service that's still booting (deferred_init) must not fire the
    service_unavailable toast — that would alarm the user over a normal
    first-run state."""
    api = _FakeAPI({"embedder": {"ok": False, "error": None, "pending": True}})

    @_requires("embedder", default=[])
    def rag_search(self, q):
        return ["real"]

    assert rag_search(api, "q") == []
    # Crucially: no toast event emitted while pending.
    assert api.emitted == []
