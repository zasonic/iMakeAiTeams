"""
services/adversarial_debate.py — Adversarial Debate Round Engine.

Priority 6 (new module — no existing files modified).

Synchronous — uses concurrent.futures.ThreadPoolExecutor for parallel
challenge calls (matching task_scheduler.py's threading pattern).
Uses claude_client.chat() (synchronous) for each challenge.

Flow
----
1. After run_workflow() completes, api.py calls run_debate_round()
2. Each agent sees all other agents' outputs and lists:
   - assumption_diffs  (where interpretations differ)
   - fact_conflicts    (specific factual contradictions)
   - missing_analysis  (coverage gaps no agent addressed)
3. If an agent revises their position, the revision is noted
4. Results persisted to debate_log and returned to api.py
5. api.py emits debate events to the frontend via self._emit()

Default tier behaviour (stored in settings table):
  debate_enabled = "1"
  debate_tier_threshold = "claude"   (only for claude-model workflows)

Cost: N agents × ~400 output tokens per challenge call.
"""

import concurrent.futures
import json
import logging
import time
import uuid
from datetime import datetime, timezone

import db

log = logging.getLogger("adversarial_debate")

# ── Challenge prompt ──────────────────────────────────────────────────────────

_CHALLENGE_PROMPT = """You are {agent_name}.

In a multi-agent workflow you produced this output:

---
SUBTASK: {my_subtask}
YOUR OUTPUT: {my_artifact}
YOUR CONFIDENCE: {my_confidence}
YOUR UNCERTAINTIES: {my_uncertainties}
---

Other agents produced these outputs for their subtasks:

{other_outputs}

---

Your job is to find problems — NOT to agree.

Respond ONLY with the JSON object below:

{{
  "assumption_diffs": ["Every case where your assumptions differ from another agent's. Name the assumption, the agent, and the conflict."],
  "fact_conflicts": ["Every factual claim that conflicts between agents. Cite both positions: 'Agent A says X, Agent B says Y.'"],
  "missing_analysis": ["Gaps no agent covered — questions unanswered, context unverified, risks unconsidered."],
  "changed_position": false,
  "revised_conclusion": null,
  "overall_assessment": "One paragraph: how much do you trust the combined outputs? What should the coordinator verify?"
}}

Rules:
- If you genuinely find none in a category, return []. Look hard first.
- changed_position: true only if reviewing others caused you to revise your conclusion.
- Be specific. Vague entries like "methodology unclear" are not allowed.
"""

# ── Settings ──────────────────────────────────────────────────────────────────

def _debate_settings() -> dict:
    try:
        rows = db.fetchall(
            "SELECT key, value FROM settings WHERE key IN ('debate_enabled','debate_tier_threshold')"
        )
        return {r["key"]: r["value"] for r in rows}
    except Exception:
        return {}


def is_debate_enabled() -> bool:
    s = _debate_settings()
    return s.get("debate_enabled", "1") == "1"


def get_debate_tier_threshold() -> str:
    """Returns "claude" (default) or "local" (always) or "never"."""
    return _debate_settings().get("debate_tier_threshold", "claude")


def set_debate_settings(enabled: bool, tier_threshold: str = "claude") -> None:
    now = datetime.now(timezone.utc).isoformat()
    for key, val in [
        ("debate_enabled",        "1" if enabled else "0"),
        ("debate_tier_threshold", tier_threshold),
    ]:
        try:
            db.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, val, now),
            )
        except Exception as exc:
            log.warning("set_debate_settings failed for %s: %s", key, exc)
    db.commit()
    log.info("Debate settings: enabled=%s threshold=%s", enabled, tier_threshold)


def should_debate(use_local: bool) -> bool:
    """Determine if debate should run for this workflow."""
    if not is_debate_enabled():
        return False
    threshold = get_debate_tier_threshold()
    if threshold == "never":
        return False
    if threshold == "local":   # always debate
        return True
    if threshold == "claude":  # only for Claude-API workflows
        return not use_local
    return not use_local       # default: claude only


# ── Single-agent challenge ────────────────────────────────────────────────────

def _challenge_one_agent(
    agent_name:    str,
    agent_id:      str,
    my_subtask:    str,
    my_artifact:   str,
    my_confidence: str,
    my_uncertainties: str,
    other_outputs: str,
    claude_client,
    workflow_id:   str,
    debate_id:     str,
    local_client=None,
) -> dict:
    """
    Send one agent's challenge prompt. Returns a challenge dict.
    Runs in a ThreadPoolExecutor worker.

    v4.1 — Worker-judge pattern: tries the local model first for challenge
    calls (structured JSON extraction that smaller models handle well).
    Falls back to Claude if local fails or is unavailable. This makes
    debate rounds nearly free when a capable local model is running.
    """
    t0 = time.monotonic()
    challenge_id = str(uuid.uuid4())

    prompt = _CHALLENGE_PROMPT.format(
        agent_name       = agent_name,
        my_subtask       = my_subtask[:500],
        my_artifact      = my_artifact[:1200],
        my_confidence    = my_confidence,
        my_uncertainties = my_uncertainties,
        other_outputs    = other_outputs,
    )

    system = "You are a precise analytical agent. Find real problems in the combined outputs."
    raw = None
    used_local = False

    # ── Try local model first (worker-judge: cheap challenges) ────────────
    if local_client and local_client.is_available():
        try:
            raw = local_client.chat(system, prompt, max_tokens=1024)
            used_local = True
            # Quick validation: must parse as JSON with expected keys
            _test = raw.strip()
            if _test.startswith("```"):
                _test = "\n".join(_test.split("\n")[1:])
            if _test.endswith("```"):
                _test = _test.rsplit("```", 1)[0].strip()
            _parsed = json.loads(_test)
            if "assumption_diffs" not in _parsed:
                raise ValueError("Missing expected key — falling back to Claude")
        except Exception as exc:
            log.debug("Local challenge failed for %s (%s), falling back to Claude",
                      agent_name, exc)
            raw = None
            used_local = False

    # ── Fall back to Claude ───────────────────────────────────────────────
    if raw is None:
        try:
            raw = claude_client.chat(system, "", prompt, max_tokens=1024)
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            log.error("Challenge call failed for %s: %s", agent_name, exc)
            return {
                "challenge_id": challenge_id, "debate_id": debate_id, "workflow_id": workflow_id,
                "agent_id": agent_id, "agent_name": agent_name,
                "assumption_diffs": [], "fact_conflicts": [], "missing_analysis": [],
                "changed_position": False, "revised_conclusion": None,
                "overall_assessment": f"Challenge call failed: {exc}",
                "duration_ms": round(duration_ms, 1), "parse_failed": True,
                "input_tokens": 0, "output_tokens": 0, "used_local": False,
            }

    try:
        duration_ms = (time.monotonic() - t0) * 1000

        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0].strip()

        data = json.loads(raw)

        return {
            "challenge_id":       challenge_id,
            "debate_id":          debate_id,
            "workflow_id":        workflow_id,
            "agent_id":           agent_id,
            "agent_name":         agent_name,
            "assumption_diffs":   [str(a) for a in data.get("assumption_diffs", []) if a],
            "fact_conflicts":     [str(f) for f in data.get("fact_conflicts",   []) if f],
            "missing_analysis":   [str(m) for m in data.get("missing_analysis", []) if m],
            "changed_position":   bool(data.get("changed_position", False)),
            "revised_conclusion": data.get("revised_conclusion") or None,
            "overall_assessment": str(data.get("overall_assessment", ""))[:1000],
            "duration_ms":        round(duration_ms, 1),
            "parse_failed":       False,
            "input_tokens":       0,
            "output_tokens":      0,
            "used_local":         used_local,
        }

    except json.JSONDecodeError as exc:
        duration_ms = (time.monotonic() - t0) * 1000
        log.warning("Debate JSON parse failed for %s: %s", agent_name, exc)

        # ── If local parse failed, retry with Claude (judge escalation) ───
        if used_local:
            log.info("Local parse failed for %s, retrying with Claude", agent_name)
            try:
                raw2 = claude_client.chat(system, "", prompt, max_tokens=1024)
                raw2 = raw2.strip()
                if raw2.startswith("```"):
                    raw2 = "\n".join(raw2.split("\n")[1:])
                if raw2.endswith("```"):
                    raw2 = raw2.rsplit("```", 1)[0].strip()
                data = json.loads(raw2)
                duration_ms = (time.monotonic() - t0) * 1000
                return {
                    "challenge_id": challenge_id, "debate_id": debate_id,
                    "workflow_id": workflow_id, "agent_id": agent_id,
                    "agent_name": agent_name,
                    "assumption_diffs": [str(a) for a in data.get("assumption_diffs", []) if a],
                    "fact_conflicts": [str(f) for f in data.get("fact_conflicts", []) if f],
                    "missing_analysis": [str(m) for m in data.get("missing_analysis", []) if m],
                    "changed_position": bool(data.get("changed_position", False)),
                    "revised_conclusion": data.get("revised_conclusion") or None,
                    "overall_assessment": str(data.get("overall_assessment", ""))[:1000],
                    "duration_ms": round(duration_ms, 1), "parse_failed": False,
                    "input_tokens": 0, "output_tokens": 0, "used_local": False,
                }
            except Exception:
                pass  # fall through to error return

        return {
            "challenge_id": challenge_id, "debate_id": debate_id, "workflow_id": workflow_id,
            "agent_id": agent_id, "agent_name": agent_name,
            "assumption_diffs": [], "fact_conflicts": [], "missing_analysis": [],
            "changed_position": False, "revised_conclusion": None,
            "overall_assessment": f"Parse failed — raw:\n{raw[:400]}",
            "duration_ms": round(duration_ms, 1), "parse_failed": True,
            "input_tokens": 0, "output_tokens": 0, "used_local": used_local,
        }


# ── Debate log persistence ────────────────────────────────────────────────────

def _log_challenge(challenge: dict) -> None:
    try:
        db.execute(
            """
            INSERT INTO debate_log
                (challenge_id, debate_id, workflow_id, agent_id, agent_name,
                 assumption_diffs_json, fact_conflicts_json, missing_analysis_json,
                 changed_position, revised_conclusion, overall_assessment,
                 input_tokens, output_tokens, duration_ms, parse_failed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                challenge["challenge_id"], challenge["debate_id"], challenge["workflow_id"],
                challenge["agent_id"], challenge["agent_name"],
                json.dumps(challenge["assumption_diffs"]),
                json.dumps(challenge["fact_conflicts"]),
                json.dumps(challenge["missing_analysis"]),
                1 if challenge["changed_position"] else 0,
                challenge["revised_conclusion"],
                challenge["overall_assessment"],
                challenge["input_tokens"],
                challenge["output_tokens"],
                challenge["duration_ms"],
                1 if challenge["parse_failed"] else 0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        db.commit()
    except Exception as exc:
        log.warning("debate_log write failed: %s", exc)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_debate_round(
    workflow_id:   str,
    claude_client,
    on_event:      callable | None = None,
    use_local:     bool = False,
    local_client=None,
) -> dict:
    """
    Run the adversarial debate phase for a completed workflow.

    Called from api.py's plan_and_run_workflow() after run_workflow() succeeds.

    Parameters
    ----------
    workflow_id   : Completed workflow to debate
    claude_client : ClaudeClient instance
    on_event      : Optional callback(event_name, payload_dict) — api.py passes self._emit
    use_local     : True if workflow ran on local model only
    local_client  : Optional LocalClient — when provided, challenge rounds try
                    local first (worker-judge pattern: free challenges, Claude synthesis)

    Returns
    -------
    dict with: debate_id, challenges (list), totals, skipped (bool), skip_reason (str)
    """
    debate_id = str(uuid.uuid4())

    # ── Eligibility check ─────────────────────────────────────────────────────
    if not should_debate(use_local):
        reason = (
            "Debate disabled in settings"
            if not is_debate_enabled()
            else f"Debate not configured for {'local' if use_local else 'Claude'} workflows"
        )
        log.debug("Debate skipped for wf=%s: %s", workflow_id[:8], reason)
        return {"debate_id": debate_id, "challenges": [], "skipped": True, "skip_reason": reason,
                "total_fact_conflicts": 0, "total_assumption_diffs": 0, "total_gaps": 0, "position_changes": 0}

    # ── Load all HandoffPackets for this workflow ──────────────────────────────
    from services.task_scheduler import get_workflow_handoffs  # noqa: PLC0415
    handoffs = get_workflow_handoffs(workflow_id)

    if len(handoffs) < 2:
        return {"debate_id": debate_id, "challenges": [], "skipped": True,
                "skip_reason": "Only one agent completed — debate needs ≥2 agents.",
                "total_fact_conflicts": 0, "total_assumption_diffs": 0, "total_gaps": 0, "position_changes": 0}

    if on_event:
        try:
            on_event("debate_round", {
                "event": "started", "workflow_id": workflow_id, "debate_id": debate_id,
                "agent_count": len(handoffs), "icon": "⚔️",
                "label": f"Adversarial Debate — {len(handoffs)} agents",
                "detail": "Each agent reviews all other agents' outputs for conflicts and gaps.",
                "status": "running",
            })
        except Exception:
            pass

    # ── Build challenge inputs ────────────────────────────────────────────────
    challenge_inputs = []
    for h in handoffs:
        others = [o for o in handoffs if o["agent_id"] != h["agent_id"]]
        other_sections = []
        for o in others:
            section = (
                f"**{o['agent_name']}** (subtask: {o['subtask_completed']})\n"
                f"Output: {(o.get('artifact_summary') or '')[:600]}\n"
                f"Confidence: {o.get('confidence', 1.0):.0%}\n"
                f"Uncertainties: {'; '.join(json.loads(o.get('uncertainties_json','[]'))[:3]) or 'None stated'}"
            )
            other_sections.append(section)

        challenge_inputs.append({
            "agent_name":      h["agent_name"],
            "agent_id":        h["agent_id"],
            "my_subtask":      h["subtask_completed"],
            "my_artifact":     h.get("artifact_summary", ""),
            "my_confidence":   f"{h.get('confidence', 1.0):.0%}",
            "my_uncertainties": "; ".join(json.loads(h.get("uncertainties_json", "[]"))[:3]) or "None stated",
            "other_outputs":   "\n\n---\n\n".join(other_sections),
        })

    # ── Run all challenges in parallel ────────────────────────────────────────
    challenges: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(
                _challenge_one_agent,
                ci["agent_name"], ci["agent_id"],
                ci["my_subtask"], ci["my_artifact"],
                ci["my_confidence"], ci["my_uncertainties"],
                ci["other_outputs"],
                claude_client, workflow_id, debate_id,
                local_client,
            ): ci["agent_name"]
            for ci in challenge_inputs
        }

        for fut in concurrent.futures.as_completed(futures):
            agent_name = futures[fut]
            try:
                challenge = fut.result()
            except Exception as exc:
                log.error("Challenge future failed for %s: %s", agent_name, exc)
                continue

            challenges.append(challenge)
            _log_challenge(challenge)

            if on_event:
                try:
                    on_event("debate_round", {
                        "event":      "agent_challenged",
                        "workflow_id": workflow_id,
                        "debate_id":  debate_id,
                        "agent_name": challenge["agent_name"],
                        "agent_id":   challenge["agent_id"],
                        "icon":       "⚔️" if (challenge["fact_conflicts"] or challenge["assumption_diffs"]) else "✅",
                        "label":      f"{challenge['agent_name']} — debate complete",
                        "detail": (
                            f"{len(challenge['fact_conflicts'])} conflict(s) · "
                            f"{len(challenge['assumption_diffs'])} assumption diff(s) · "
                            f"{len(challenge['missing_analysis'])} gap(s)"
                            + (" · Position revised" if challenge["changed_position"] else "")
                        ),
                        "status": "complete",
                        "packet": {
                            "agent_name":       challenge["agent_name"],
                            "fact_conflicts":   challenge["fact_conflicts"],
                            "assumption_diffs": challenge["assumption_diffs"],
                            "missing_analysis": challenge["missing_analysis"],
                            "changed_position": challenge["changed_position"],
                            "revised_conclusion": challenge["revised_conclusion"],
                            "overall_assessment": challenge["overall_assessment"],
                            "duration_ms":      challenge["duration_ms"],
                        },
                    })
                except Exception:
                    pass

    # ── Aggregate ─────────────────────────────────────────────────────────────
    fc = sum(len(c["fact_conflicts"])   for c in challenges)
    ad = sum(len(c["assumption_diffs"]) for c in challenges)
    g  = sum(len(c["missing_analysis"]) for c in challenges)
    pc = sum(1 for c in challenges if c["changed_position"])
    local_count = sum(1 for c in challenges if c.get("used_local", False))

    if local_count:
        log.info("Debate: %d/%d challenges ran on local model (free)",
                 local_count, len(challenges))

    if on_event:
        try:
            on_event("debate_round", {
                "event":      "complete",
                "workflow_id": workflow_id,
                "debate_id":  debate_id,
                "icon":       "⚔️" if (fc + ad) > 0 else "✅",
                "label":      "Debate Round Complete",
                "detail": (
                    f"{fc} fact conflict(s) · {ad} assumption diff(s) · "
                    f"{g} gap(s) · {pc} position change(s)"
                ),
                "status": "complete",
                "fact_conflicts":    fc,
                "assumption_diffs":  ad,
                "gaps":              g,
                "position_changes":  pc,
                "challenges": [
                    {
                        "agent_name":       c["agent_name"],
                        "fact_conflicts":   len(c["fact_conflicts"]),
                        "assumption_diffs": len(c["assumption_diffs"]),
                        "gaps":             len(c["missing_analysis"]),
                        "changed_position": c["changed_position"],
                    }
                    for c in challenges
                ],
            })
        except Exception:
            pass

    log.info("Debate complete wf=%s: %d challenges, %d conflicts, %d diffs, %d gaps",
             workflow_id[:8], len(challenges), fc, ad, g)

    return {
        "debate_id":             debate_id,
        "challenges":            challenges,
        "skipped":               False,
        "skip_reason":           "",
        "total_fact_conflicts":  fc,
        "total_assumption_diffs": ad,
        "total_gaps":            g,
        "position_changes":      pc,
        "challenges_on_local":   local_count,
    }


def build_debate_synthesis_addendum(debate_result: dict) -> str:
    """
    Format all challenge packets as text to inject into the coordinator synthesis prompt.
    Called from plan_and_run_workflow() after debate completes.
    """
    if debate_result.get("skipped") or not debate_result.get("challenges"):
        return ""

    fc = debate_result["total_fact_conflicts"]
    ad = debate_result["total_assumption_diffs"]
    g  = debate_result["total_gaps"]
    pc = debate_result["position_changes"]

    header = (
        f"\n---\n## Adversarial Debate Round Results\n\n"
        f"Summary: {fc} fact conflict(s) · {ad} assumption diff(s) · {g} gap(s) · {pc} position change(s)\n\n"
        "You MUST address every fact_conflict below — either resolve it with evidence or flag it "
        "explicitly in the final response as an unresolved conflict the user should investigate.\n\n"
    )

    sections = []
    for c in debate_result["challenges"]:
        lines = [f"### {c['agent_name']} Review"]
        if c["assumption_diffs"]:
            lines.append("**Assumption conflicts:**")
            lines.extend(f"- {a}" for a in c["assumption_diffs"])
        if c["fact_conflicts"]:
            lines.append("**Fact conflicts:**")
            lines.extend(f"- {f}" for f in c["fact_conflicts"])
        if c["missing_analysis"]:
            lines.append("**Coverage gaps:**")
            lines.extend(f"- {m}" for m in c["missing_analysis"])
        if c["changed_position"] and c["revised_conclusion"]:
            lines.append(f"**⚠️ Position revised:** {c['revised_conclusion']}")
        if c["overall_assessment"]:
            lines.append(f"**Assessment:** {c['overall_assessment']}")
        if not (c["fact_conflicts"] or c["assumption_diffs"] or c["missing_analysis"]):
            lines.append("*No conflicts or gaps identified.*")
        sections.append("\n".join(lines))

    return header + "\n\n".join(sections) + "\n---\n"


def get_workflow_debate(workflow_id: str) -> list[dict]:
    """Return all ChallengePackets for a workflow from debate_log."""
    try:
        rows = db.fetchall(
            "SELECT * FROM debate_log WHERE workflow_id = ? ORDER BY created_at ASC",
            (workflow_id,),
        )
        result = []
        for r in rows:
            d = dict(r)
            d["assumption_diffs"]  = json.loads(d.pop("assumption_diffs_json",  "[]"))
            d["fact_conflicts"]    = json.loads(d.pop("fact_conflicts_json",    "[]"))
            d["missing_analysis"]  = json.loads(d.pop("missing_analysis_json", "[]"))
            result.append(d)
        return result
    except Exception as exc:
        log.warning("get_workflow_debate failed: %s", exc)
        return []
