"""
services/mcp_credentials.py — Phase 2: per-MCP-server secret broker.

The hub mediates every MCP server credential. No other module under
app/services/ may call ``keyring`` for ``KEYRING_SERVICE_PREFIX`` entries —
this is enforced by an AST guard test in tests/test_mcp_credentials.py.

Secrets are stored in the OS keyring (DPAPI on Windows, Keychain on macOS,
SecretService on Linux) under the service name
``iMakeAiTeams.mcp.<server_id>``. The set of declared environment-variable
keys lives in each server's mcp.json manifest under ``env_keys``; this module
only persists/retrieves values, it does not declare schemas.

Failure mode: a broken keyring backend is silently treated as "secret not
set". Phase 2 does not execute MCP tools, so a missing secret only matters
in the (deferred) execution path.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

log = logging.getLogger("MyAIEnv.mcp_credentials")

KEYRING_SERVICE_PREFIX = "iMakeAiTeams.mcp."


def _service_name(server_id: str) -> str:
    if not server_id or not server_id.strip():
        raise ValueError("server_id is required")
    return KEYRING_SERVICE_PREFIX + server_id.strip()


def set_secret(server_id: str, key: str, value: str) -> bool:
    """Store one secret value for a server. Returns False if keyring is unusable.

    Programmer errors (empty server_id) raise ValueError; only keyring backend
    failures are swallowed.
    """
    service = _service_name(server_id)
    try:
        import keyring
        keyring.set_password(service, key, value)
        return True
    except BaseException as exc:
        log.warning(
            "mcp_credentials.set_secret(%s/%s) failed: %s",
            server_id, key, exc,
        )
        return False


def get_secret(server_id: str, key: str) -> str | None:
    """Retrieve one secret value. Returns None if missing or keyring unusable."""
    service = _service_name(server_id)
    try:
        import keyring
        return keyring.get_password(service, key)
    except BaseException as exc:
        log.debug(
            "mcp_credentials.get_secret(%s/%s) failed: %s",
            server_id, key, exc,
        )
        return None


def delete_secret(server_id: str, key: str) -> None:
    service = _service_name(server_id)
    try:
        import keyring
        keyring.delete_password(service, key)
    except BaseException as exc:
        log.debug(
            "mcp_credentials.delete_secret(%s/%s) failed: %s",
            server_id, key, exc,
        )


def list_set_secrets(server_id: str, declared_keys: list[str]) -> dict[str, bool]:
    """Return {key: True/False} for each declared env_key, indicating if set.

    Never returns the secret values themselves — only presence. Used by the
    UI to show which credentials still need to be supplied.
    """
    return {k: bool(get_secret(server_id, k)) for k in declared_keys}


@contextmanager
def with_secrets(server_id: str, declared_keys: list[str]) -> Iterator[dict[str, str]]:
    """
    Yield a {env_key: value} dict scoped to the with-block.

    The dict goes out of scope when the block exits, minimizing the window in
    which secrets are present in the process's variable space. Missing
    secrets are simply absent from the dict — callers must check.

    The (deferred) MCP execution layer is the only intended consumer.
    """
    secrets: dict[str, str] = {}
    for key in declared_keys:
        val = get_secret(server_id, key)
        if val is not None:
            secrets[key] = val
    try:
        yield secrets
    finally:
        # Best-effort wipe so the dict can't be inspected after the block.
        for k in list(secrets):
            secrets[k] = ""
        secrets.clear()
