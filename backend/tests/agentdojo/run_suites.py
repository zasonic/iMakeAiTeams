"""
backend/tests/agentdojo/run_suites.py — CLI driver for the AgentDojo bench.

Usage:
    python -m backend.tests.agentdojo.run_suites \\
        --suite workspace \\
        --output benchmarks/workspace.json

Runs every published user_task and injection_task in the chosen suite
through the iMakeAiTeams stack via ``runner.build_pipeline()``. Writes
``results.json`` with the headline metrics:

    total_tasks         — count of (user_task, injection_task) pairs evaluated
    utility             — share of user_tasks the agent completed correctly
    asr                 — overall injection success rate (lower is better)
    targeted_asr        — injection success rate when the injection's target
                          tool is in the agent's tool catalog (the metric
                          AgentDojo's published leaderboard uses)
    per_task            — list of (task_id, injection_id, success_kind)

The script is import-clean (sys.exit codes only — never raises into the
runtime test suite). It honours the threshold configured in
``benchmarks/thresholds.json``: if ``asr`` exceeds ``max_asr_pct`` for
the chosen suite, the process exits 1 so the GitHub workflow fails the
build. ``--ignore-threshold`` is provided for local exploration.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Path-hack so ``python -m backend.tests.agentdojo.run_suites`` works whether
# CWD is the repo root or backend/. Mirrors tests/conftest.py.
_HERE = Path(__file__).resolve()
_BACKEND_DIR = _HERE.parent.parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
for p in (_REPO_ROOT, _BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.tests.agentdojo import AGENTDOJO_AVAILABLE  # noqa: E402

log = logging.getLogger("imakeaiteams.bench")


SUITES = ("workspace", "slack", "banking", "travel")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_suites",
        description="Run an AgentDojo suite against the iMakeAiTeams stack.",
    )
    parser.add_argument("--suite", required=True, choices=SUITES)
    parser.add_argument(
        "--output", required=True,
        help="Path to results.json (parent dir is created if missing).",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Anthropic model id passed to the Actor (Reader uses the same).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Optional cap on (user_task, injection_task) pairs for smoke runs.",
    )
    parser.add_argument(
        "--no-split", action="store_true",
        help="Disable the Reader/Actor split — for measuring the baseline.",
    )
    parser.add_argument(
        "--ignore-threshold", action="store_true",
        help="Don't fail the process when ASR exceeds the configured threshold.",
    )
    parser.add_argument(
        "--thresholds",
        default=str(_REPO_ROOT / "benchmarks" / "thresholds.json"),
        help="Path to per-suite ASR thresholds JSON.",
    )
    args = parser.parse_args(argv)

    if not AGENTDOJO_AVAILABLE:
        sys.stderr.write(
            "agentdojo is not installed. Run "
            "`pip install -r backend/requirements-bench.txt` first, or use "
            "`dev\\run-bench.bat` which handles the install.\n"
        )
        return 2

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.stderr.write(
            "ANTHROPIC_API_KEY env var is required. Set it before running, "
            "or configure it as a secret on the security-bench workflow.\n"
        )
        return 2

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = _run_suite(
        suite_name=args.suite,
        model=args.model,
        limit=args.limit,
        split_enabled=not args.no_split,
    )

    output_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    log.info("Wrote %s", output_path)

    threshold_status = _check_threshold(
        suite_name=args.suite,
        results=results,
        thresholds_path=Path(args.thresholds),
    )
    if threshold_status is not None and not args.ignore_threshold:
        sys.stderr.write(threshold_status + "\n")
        return 1
    return 0


def _run_suite(
    *,
    suite_name: str,
    model: str,
    limit: int,
    split_enabled: bool,
) -> dict[str, Any]:
    """Iterate every (user_task, injection_task) pair and tally outcomes.

    Uses AgentDojo's published task suites (loaded by the upstream package)
    rather than re-implementing them. The pipeline is rebuilt from scratch
    for each pair so the GovernanceEngine and RiskLedger start clean —
    cross-task carryover would inflate ASR and pollute the metrics.
    """
    # Lazy imports so a missing agentdojo gives a clean error from main().
    from agentdojo.task_suite.load_suites import get_suites  # type: ignore
    from agentdojo.attacks.important_instructions_attacks import (  # type: ignore
        ImportantInstructionsAttacker,
    )
    from agentdojo.benchmark import benchmark_suite_with_injections  # type: ignore

    from backend.tests.agentdojo.runner import build_pipeline  # noqa: E402

    suites = get_suites("v1.2.1")
    if suite_name not in suites:
        raise RuntimeError(
            f"Suite {suite_name!r} not in AgentDojo v1.2.1 manifest "
            f"({sorted(suites)}). Upgrade requirements-bench.txt or pin "
            f"a different agentdojo version."
        )
    suite = suites[suite_name]

    user_task_ids = list(suite.user_tasks.keys())
    injection_task_ids = list(suite.injection_tasks.keys())
    if limit > 0:
        user_task_ids = user_task_ids[:limit]
        injection_task_ids = injection_task_ids[:limit]

    started_at = time.time()
    per_task: list[dict[str, Any]] = []
    utility_hits = 0
    asr_hits = 0
    targeted_eligible = 0
    targeted_hits = 0

    for ut_id in user_task_ids:
        for it_id in injection_task_ids:
            pipeline, handles = build_pipeline(
                model=model, split_enabled=split_enabled,
            )
            attacker = ImportantInstructionsAttacker(
                target_pipeline=pipeline,
                user_task=suite.user_tasks[ut_id],
                injection_task=suite.injection_tasks[it_id],
            )
            try:
                result = benchmark_suite_with_injections(
                    pipeline,
                    suite,
                    user_task_ids=[ut_id],
                    injection_task_ids=[it_id],
                    attacker=attacker,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "task %s × injection %s raised: %s", ut_id, it_id, exc,
                )
                per_task.append({
                    "user_task": ut_id, "injection_task": it_id,
                    "utility": False, "asr": False, "targeted_asr": False,
                    "error": str(exc)[:200],
                })
                continue

            utility_ok = bool(result.get("utility"))
            asr_ok = bool(result.get("security"))
            target_tool = suite.injection_tasks[it_id].target_tool
            tool_in_catalog = target_tool in {t.name for t in suite.tools}
            if tool_in_catalog:
                targeted_eligible += 1
                if asr_ok:
                    targeted_hits += 1

            if utility_ok:
                utility_hits += 1
            if asr_ok:
                asr_hits += 1
            per_task.append({
                "user_task": ut_id,
                "injection_task": it_id,
                "utility": utility_ok,
                "asr": asr_ok,
                "targeted_asr": (asr_ok and tool_in_catalog),
                "final_risk_score": handles.risk_ledger.assess().cumulative_score,
            })

    total_tasks = len(per_task)
    return {
        "suite": suite_name,
        "agentdojo_version": _agentdojo_version(),
        "model": model,
        "split_enabled": split_enabled,
        "started_at": started_at,
        "duration_seconds": time.time() - started_at,
        "total_tasks": total_tasks,
        "utility": _safe_pct(utility_hits, total_tasks),
        "asr": _safe_pct(asr_hits, total_tasks),
        "targeted_asr": _safe_pct(targeted_hits, targeted_eligible),
        "per_task": per_task,
    }


def _safe_pct(num: int, denom: int) -> float:
    return 0.0 if denom <= 0 else round(100.0 * num / denom, 4)


def _agentdojo_version() -> str:
    try:
        import agentdojo  # type: ignore
        return getattr(agentdojo, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def _check_threshold(
    *, suite_name: str, results: dict, thresholds_path: Path,
) -> str | None:
    """Return an error string if ASR exceeds the suite threshold; else None."""
    if not thresholds_path.exists():
        return None
    try:
        thresholds = json.loads(thresholds_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"thresholds.json could not be parsed: {exc}"
    suite_cfg = (thresholds.get("suites") or {}).get(suite_name) or {}
    max_asr = suite_cfg.get("max_asr_pct")
    if max_asr is None:
        return None
    actual = float(results.get("asr") or 0.0)
    if actual > float(max_asr):
        return (
            f"[threshold] {suite_name} ASR {actual:.2f}% exceeds "
            f"configured ceiling {float(max_asr):.2f}%"
        )
    return None


if __name__ == "__main__":  # pragma: no cover — CLI entry
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
