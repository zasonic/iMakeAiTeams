"""
build-scripts/generate_benchmarks_md.py — assemble BENCHMARKS.md from the
four per-suite results.json files written by run_suites.py.

Usage:
    python build-scripts/generate_benchmarks_md.py \\
        --benchmarks-dir benchmarks \\
        --thresholds benchmarks/thresholds.json \\
        --output BENCHMARKS.md

The generator is dependency-free (only stdlib) so it runs identically in
the GitHub Actions workflow and in the dev/run-bench.bat path. It is
defensive about missing files: a suite without a results.json renders as
a row of `—` placeholders rather than crashing the step.

The published markdown table reports four numbers per suite plus the
Hackett et al. (ACL 2025) monolithic baseline. The methodology block
records the model id, the bench timestamp, and which defences were active
during the run so the marketing claim is reproducible.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

SUITES = ("workspace", "slack", "banking", "travel")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks-dir", default="benchmarks",
        help="Directory containing <suite>.json files.",
    )
    parser.add_argument(
        "--thresholds", default="benchmarks/thresholds.json",
        help="Per-suite ASR ceilings (used to display the baseline column).",
    )
    parser.add_argument(
        "--output", default="BENCHMARKS.md",
        help="Markdown file to write at repo root.",
    )
    args = parser.parse_args(argv)

    bench_dir = Path(args.benchmarks_dir).resolve()
    thresholds = _load_thresholds(Path(args.thresholds))

    rows: list[dict[str, Any]] = []
    methodology_seen: dict[str, Any] = {}
    for suite in SUITES:
        path = bench_dir / f"{suite}.json"
        if not path.exists():
            rows.append({
                "suite": suite, "tasks": "—", "utility": "—",
                "asr": "—", "targeted_asr": "—",
                "baseline": _baseline_for(thresholds, suite),
            })
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            rows.append({
                "suite": suite, "tasks": "—", "utility": "—",
                "asr": "—", "targeted_asr": "—",
                "baseline": _baseline_for(thresholds, suite),
            })
            continue
        rows.append({
            "suite": suite,
            "tasks": data.get("total_tasks", "—"),
            "utility": _fmt_pct(data.get("utility")),
            "asr": _fmt_pct(data.get("asr")),
            "targeted_asr": _fmt_pct(data.get("targeted_asr")),
            "baseline": _baseline_for(thresholds, suite),
        })
        # The methodology block uses the first non-empty record we see —
        # all four suites are run in the same workflow against the same
        # pipeline, so any one of them is authoritative.
        if not methodology_seen:
            methodology_seen = {
                "model": data.get("model"),
                "agentdojo_version": data.get("agentdojo_version"),
                "split_enabled": data.get("split_enabled"),
                "started_at": data.get("started_at"),
            }

    out_path = Path(args.output).resolve()
    out_path.write_text(_render(rows, methodology_seen))
    sys.stdout.write(f"Wrote {out_path}\n")
    return 0


def _baseline_for(thresholds: dict, suite: str) -> str:
    cfg = (thresholds.get("suites") or {}).get(suite) or {}
    val = cfg.get("baseline_asr_pct")
    if val is None:
        return "—"
    return f"{float(val):.1f}%"


def _fmt_pct(val: Any) -> str:
    if val is None or val == "—":
        return "—"
    try:
        return f"{float(val):.2f}%"
    except (TypeError, ValueError):
        return "—"


def _load_thresholds(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _render(rows: list[dict[str, Any]], methodology: dict[str, Any]) -> str:
    """Render the final markdown document.

    Headers:
      Suite | Tasks | Utility | ASR | Targeted ASR | Baseline ASR (Hackett 2025)
    """
    header = (
        "# Security Benchmarks — AgentDojo\n\n"
        "Continuous evaluation of the iMakeAiTeams security stack against "
        "the four [AgentDojo](https://github.com/ethz-spylab/agentdojo) "
        "published suites. The numbers below are produced by\n"
        "`.github/workflows/security-bench.yml` (weekly on Mondays at 06:00 "
        "UTC plus on-demand via `workflow_dispatch`) and committed back into "
        "this repository, so the headline claim (\"7×–230× ASR reduction "
        "vs the Hackett et al. monolithic baseline\") is auditable from "
        "the commit history.\n\n"
        "## Results\n\n"
        "| Suite | Tasks | Utility | ASR | Targeted ASR | "
        "Baseline ASR (Hackett 2025) |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    body_rows = []
    for r in rows:
        body_rows.append(
            f"| {r['suite']} | {r['tasks']} | {r['utility']} | "
            f"{r['asr']} | {r['targeted_asr']} | {r['baseline']} |"
        )
    body = "\n".join(body_rows) + "\n\n"

    started_at = methodology.get("started_at")
    if started_at:
        try:
            ts = _dt.datetime.fromtimestamp(
                float(started_at), tz=_dt.timezone.utc,
            ).isoformat(timespec="seconds")
        except (TypeError, ValueError):
            ts = "(unknown)"
    else:
        ts = "(no run yet)"

    methodology_md = (
        "## Methodology\n\n"
        "- **Architecture under test:** `security_engine` 5 defences "
        "(quarantine, risk ledger, memory firewall, skill scanner, "
        "deterministic rules) + `governance` Reader/Actor split + Wiser-Human "
        "escalation channel.\n"
        f"- **Reader/Actor split enabled:** "
        f"{methodology.get('split_enabled', '(unknown)')}\n"
        f"- **Model:** {methodology.get('model', '(unknown)')}\n"
        f"- **AgentDojo version:** "
        f"{methodology.get('agentdojo_version', '(unknown)')}\n"
        f"- **Last run (UTC):** {ts}\n\n"
        "Each suite runs every (user_task × injection_task) pair from the "
        "AgentDojo v1.2.1 manifest with the `ImportantInstructionsAttacker`. "
        "`utility` is the share of user_tasks completed correctly; `ASR` is "
        "the overall injection success rate; `Targeted ASR` is the success "
        "rate restricted to injections whose target tool is in the agent's "
        "catalog (the metric the AgentDojo leaderboard reports).\n\n"
        "## Reproducing\n\n"
        "```\n"
        "pip install -r backend/requirements-bench.txt\n"
        "python -m backend.tests.agentdojo.run_suites "
        "--suite workspace --output benchmarks/workspace.json\n"
        "python build-scripts/generate_benchmarks_md.py\n"
        "```\n\n"
        "On Windows, `dev\\run-bench.bat` chains the install + all four "
        "suites + the regeneration of this file.\n"
    )
    return header + body + methodology_md


if __name__ == "__main__":  # pragma: no cover — CLI entry
    raise SystemExit(main())
