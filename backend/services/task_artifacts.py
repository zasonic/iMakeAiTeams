"""
services/task_artifacts.py — Shared utilities for task artifacts and local-first patterns.

Provides:
  - local_first_call(): Try local model first, fall back to Claude
  - Git worktree creation/cleanup for parallel task isolation
  - Progress file read/write for workflow steps
  - Feature list management for goal decomposition
  - Agent progress tracking for agentic loop
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("iMakeAiTeams.task_artifacts")

# ── Artifact directory ───────────────────────────────────────────────────────

_ARTIFACT_DIR = ".myai"


def _ensure_dir(project_root: Path) -> Path:
    d = project_root / _ARTIFACT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Local-first model call ───────────────────────────────────────────────────

def local_first_call(
    local_client,
    claude_client,
    system: str,
    user_message: str,
    max_tokens: int = 512,
) -> Optional[str]:
    """
    Try local model first (free), fall back to Claude if local unavailable.
    Returns the response text, or None if both fail.

    This is the worker-judge pattern from adversarial_debate.py, extracted
    for reuse across validation gates, quality checks, and classification.
    """
    # Try local first
    if local_client and hasattr(local_client, "is_available"):
        try:
            if local_client.is_available():
                result = local_client.chat(system, user_message, max_tokens=max_tokens)
                if result and not result.startswith("[Local model"):
                    return result
        except Exception as exc:
            log.debug("Local-first call failed: %s", exc)

    # Fall back to Claude
    if claude_client:
        try:
            return claude_client.chat(system, "", user_message, max_tokens=max_tokens)
        except Exception as exc:
            log.debug("Claude fallback call failed: %s", exc)

    return None


# ── Multi-agent alignment check ──────────────────────────────────────────────
# Default threshold below which a multi-agent assembled output is flagged as
# diverging from the user's original request. Anthropic's "AI Organizations"
# work showed multi-agent teams produce more capable but less aligned outputs
# than any single member; this check fires after coordinator+specialist
# assembly to surface drift while staying non-blocking.
ALIGNMENT_THRESHOLD = 0.6


def check_multi_agent_alignment(
    user_request: str,
    final_output: str,
    local_client,
    threshold: float = ALIGNMENT_THRESHOLD,
) -> dict:
    """
    Compare a multi-agent assembled output against the original user request.

    Returns a dict:
      {
        "fired":   bool,    # True iff the check actually ran
        "aligned": bool,    # True iff score >= threshold (or check did not fire)
        "score":   float,   # 0.0–1.0, 1.0 when check did not fire
        "reason":  str,     # short explanation from the local model
      }

    The check is a no-op (returns aligned=True, fired=False) when the local
    model is unavailable or its response can't be parsed. It's a quality
    signal for callers to surface, not a hard block.
    """
    safe_default = {
        "fired": False,
        "aligned": True,
        "score": 1.0,
        "reason": "alignment check did not fire",
    }

    if not local_client:
        return safe_default
    try:
        if not local_client.is_available():
            return safe_default
    except Exception:
        return safe_default

    if not user_request or not final_output:
        return safe_default

    system = (
        "You are an alignment auditor. Given a user's original request and "
        "the final response produced by a multi-agent team, judge whether "
        "the response actually addresses the request without drifting into "
        "tangential or off-topic content. Respond with ONLY a JSON object: "
        '{"score": 0.0-1.0, "reason": "..."}. '
        "score 1.0 = fully on-task; 0.5 = partially addresses; 0.0 = drifted."
    )
    user = (
        f"USER REQUEST:\n{user_request[:600]}\n\n"
        f"FINAL ASSEMBLED OUTPUT:\n{final_output[:1500]}"
    )

    try:
        raw = local_client.chat(system, user, max_tokens=160)
    except Exception as exc:
        log.debug("Alignment check failed: %s", exc)
        return safe_default
    if not raw:
        return safe_default

    qstart = raw.find("{")
    qend   = raw.rfind("}")
    if qstart == -1 or qend == -1 or qend <= qstart:
        return safe_default
    try:
        verdict = json.loads(raw[qstart:qend + 1])
    except (ValueError, TypeError):
        return safe_default

    try:
        score = float(verdict.get("score", 1.0))
    except (TypeError, ValueError):
        return safe_default
    score = max(0.0, min(1.0, score))
    reason = str(verdict.get("reason", "")).strip()[:240] or "no reason given"

    return {
        "fired": True,
        "aligned": score >= threshold,
        "score": score,
        "reason": reason,
    }


# ── Progress files ───────────────────────────────────────────────────────────

def write_workflow_progress(
    project_root: Path,
    workflow_id: str,
    step_index: int,
    data: dict,
) -> Path:
    """Write a workflow step progress file to disk."""
    d = _ensure_dir(project_root)
    path = d / f"workflow_{workflow_id[:8]}_step_{step_index}.json"
    payload = {
        "workflow_id": workflow_id,
        "step_index": step_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.debug("Wrote progress file: %s", path.name)
    return path


def read_workflow_progress(project_root: Path, workflow_id: str) -> list[dict]:
    """Read all progress files for a workflow, sorted by step index."""
    d = project_root / _ARTIFACT_DIR
    if not d.exists():
        return []
    prefix = f"workflow_{workflow_id[:8]}_step_"
    files = sorted(d.glob(f"{prefix}*.json"))
    results = []
    for f in files:
        try:
            results.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return results


# ── Feature lists ────────────────────────────────────────────────────────────

def write_feature_list(
    project_root: Path,
    goal_summary: str,
    steps: list[dict],
) -> Path:
    """Write a feature list from goal decomposition."""
    d = _ensure_dir(project_root)
    path = d / "features.json"
    features = []
    for step in steps:
        features.append({
            "step": step.get("step", 0),
            "task": step.get("task", ""),
            "status": "pending",
            "output_key": step.get("output_key", ""),
            "test_command": step.get("test_command", ""),
            "completed_at": None,
        })
    payload = {
        "goal_summary": goal_summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.debug("Wrote feature list: %d features", len(features))
    return path


def update_feature_status(
    project_root: Path,
    step: int,
    status: str,
) -> None:
    """Update a single feature's status in the feature list."""
    path = project_root / _ARTIFACT_DIR / "features.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for feat in data.get("features", []):
            if feat.get("step") == step:
                feat["status"] = status
                if status == "done":
                    feat["completed_at"] = datetime.now(timezone.utc).isoformat()
                break
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        log.debug("Failed to update feature status: %s", exc)


def read_feature_list(project_root: Path) -> Optional[dict]:
    """Read the current feature list."""
    path = project_root / _ARTIFACT_DIR / "features.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── Agent loop progress ──────────────────────────────────────────────────────

def write_agent_progress(
    project_root: Path,
    task: str,
    turn: int,
    max_turns: int,
    tools_called: list[str],
    files_modified: list[str],
    tests_run: bool = False,
    test_passed: Optional[bool] = None,
) -> Path:
    """Write agent loop progress to disk."""
    d = _ensure_dir(project_root)
    path = d / "agent_progress.json"
    payload = {
        "task": task[:200],
        "turn": turn,
        "max_turns": max_turns,
        "tools_called": tools_called[-20:],  # keep last 20
        "files_modified": list(set(files_modified)),
        "tests_run": tests_run,
        "test_passed": test_passed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ── Test discovery ───────────────────────────────────────────────────────────

def discover_test_command(project_root: Path) -> Optional[str]:
    """
    Try to discover the project's test command from common config files.
    Returns the command string or None if not found.
    """
    # Check package.json
    pkg = project_root / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            scripts = data.get("scripts", {})
            if "test" in scripts and scripts["test"] != 'echo "Error: no test specified" && exit 1':
                return "npm test"
        except Exception:
            pass

    # Check pyproject.toml
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        try:
            text = pyproject.read_text(encoding="utf-8")
            if "[tool.pytest" in text or "pytest" in text:
                return "python -m pytest"
        except Exception:
            pass

    # Check for pytest.ini or setup.cfg
    if (project_root / "pytest.ini").exists():
        return "python -m pytest"
    if (project_root / "setup.cfg").exists():
        try:
            text = (project_root / "setup.cfg").read_text(encoding="utf-8")
            if "[tool:pytest]" in text:
                return "python -m pytest"
        except Exception:
            pass

    # Check Makefile
    makefile = project_root / "Makefile"
    if makefile.exists():
        try:
            text = makefile.read_text(encoding="utf-8")
            if "test:" in text:
                return "make test"
        except Exception:
            pass

    return None


# ── Git worktree isolation ───────────────────────────────────────────────────

def create_worktree(project_root: Path, task_id: str, branch_name: str = "") -> Optional[Path]:
    """
    Create a git worktree for isolated parallel task execution.
    Returns the worktree path, or None if git is not available.
    """
    import subprocess
    worktree_dir = _ensure_dir(project_root) / "worktrees"
    worktree_dir.mkdir(exist_ok=True)
    worktree_path = worktree_dir / task_id[:8]
    branch = branch_name or f"task-{task_id[:8]}"

    if worktree_path.exists():
        return worktree_path  # already exists

    try:
        subprocess.run(
            ["git", "worktree", "add", "-b", branch, str(worktree_path)],
            cwd=str(project_root), capture_output=True, text=True, timeout=30,
        )
        if worktree_path.exists():
            log.info("Created worktree for task %s at %s", task_id[:8], worktree_path)
            return worktree_path
    except Exception as exc:
        log.debug("Worktree creation failed: %s", exc)
    return None


def merge_worktree(project_root: Path, task_id: str, branch_name: str = "") -> bool:
    """
    Merge a task worktree back into the main branch and clean up.
    Returns True if merge succeeded.
    """
    import subprocess
    worktree_path = project_root / _ARTIFACT_DIR / "worktrees" / task_id[:8]
    branch = branch_name or f"task-{task_id[:8]}"

    if not worktree_path.exists():
        return False

    try:
        # Merge the task branch
        subprocess.run(
            ["git", "merge", "--no-ff", branch, "-m", f"Merge task {task_id[:8]}"],
            cwd=str(project_root), capture_output=True, text=True, timeout=30,
        )
        # Remove worktree
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_path)],
            cwd=str(project_root), capture_output=True, text=True, timeout=30,
        )
        # Delete branch
        subprocess.run(
            ["git", "branch", "-d", branch],
            cwd=str(project_root), capture_output=True, text=True, timeout=10,
        )
        log.info("Merged and cleaned up worktree for task %s", task_id[:8])
        return True
    except Exception as exc:
        log.debug("Worktree merge failed: %s", exc)
        return False


def cleanup_worktrees(project_root: Path) -> int:
    """Remove any stale worktrees. Returns count removed."""
    import subprocess
    try:
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=str(project_root), capture_output=True, timeout=10,
        )
    except Exception:
        pass
    worktree_dir = project_root / _ARTIFACT_DIR / "worktrees"
    if not worktree_dir.exists():
        return 0
    removed = 0
    for d in worktree_dir.iterdir():
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            removed += 1
    return removed
