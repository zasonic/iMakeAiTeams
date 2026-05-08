"""
backend/tests/agentdojo/ — AgentDojo benchmark harness for the iMakeAiTeams stack.

This package is intentionally separate from the runtime test suite. Importing
``agentdojo`` is gated on the package being installed; if it is not, this
module is a no-op so that ``cd backend && pytest tests/`` keeps working in
production environments where only ``backend/requirements.txt`` has been
applied.

The bench-only dependency lives in ``backend/requirements-bench.txt`` and is
installed by the ``security-bench`` GitHub Actions workflow and by the local
``dev/run-bench.bat`` helper. ``backend/pytest.ini`` excludes this directory
from the default collection (``norecursedirs = agentdojo``) so a regular
``pytest tests/`` run never imports it.
"""

from __future__ import annotations

# Importing agentdojo here makes ``run_suites.py`` fail fast with a clear
# message when the bench dependency is missing, rather than producing a
# confusing import error deep inside the runner. The presence of this
# attribute is the single source of truth used by ``run_suites.main()`` to
# decide whether to abort with a usage hint or proceed with the full run.
#
# We import a specific submodule (``agentdojo.agent_pipeline``) rather than
# the bare top-level package because the real PyPI distribution and this
# directory share a name. If a contributor accidentally adds
# ``backend/tests/`` to sys.path directly (instead of importing through
# the qualified ``backend.tests.agentdojo`` name) a bare ``import agentdojo``
# would resolve to THIS package and falsely report availability.
try:  # pragma: no cover — depends on whether the bench dep is installed
    from agentdojo import agent_pipeline as _agent_pipeline  # type: ignore  # noqa: F401
    AGENTDOJO_AVAILABLE = True
except Exception:  # noqa: BLE001 — any import-time failure means "skip"
    AGENTDOJO_AVAILABLE = False
