"""
services/router.py

Task router — classifies message complexity and picks Claude vs local.

The router itself runs on the LOCAL model (free). Only complex tasks get
sent to Claude. This is the core token optimization mechanism.

Stage 5 changes:
  - Replaced local Route dataclass with RouteDecision from models.py.

v4.1 — Uncertainty-Aware Routing (inspired by AUQ/UAR research):
  - Router now returns a confidence score (0.0-1.0) alongside the route.
  - Low confidence (< ESCALATION_THRESHOLD) triggers two behaviours:
    1. Route escalation: local -> Claude when the local model isn't sure.
    2. Context expansion signal: needs_context=True tells the orchestrator
       to widen RAG retrieval before generating.
  - This replaces the binary "complex -> Claude" heuristic with a continuous
    signal that captures *epistemic uncertainty*, not just task type.
  - The confidence threshold adapts based on observed router error rates:
    if local routes are producing errors, the threshold tightens.
"""

import logging
import re

from models import RouteDecision

log = logging.getLogger("iMakeAiTeams.router")

# ── Deterministic keyword fallback ────────────────────────────────────────
# Used when no local model is available for routing classification.

import re as _re

_SIMPLE_PATTERNS = _re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no|bye|good morning|"
    r"good night|how are you|what'?s up)\b",
    _re.IGNORECASE,
)

_COMPLEX_SIGNALS = _re.compile(
    r"(analyz|compar|explain in detail|write.*essay|write.*report|"
    r"refactor|architect|design.*system|debate|critique|"
    r"multiple.*step|step.by.step|think.*through|research|"
    r"```|code review|security|vulnerability|optimize|"
    r"chart|plot|graph|trend|correlat|distribut|visuali[zs]|histogram|"
    r"csv|spreadsheet|data.*analy|pivot|aggregat|statistic|regression|"
    r"outlier|cluster|forecast|time.series|breakdown.*by|group.*by|"
    r"show.*data|average.*by|total.*by|count.*by)",
    _re.IGNORECASE,
)


def _keyword_classify(message: str) -> RouteDecision:
    """Deterministic fallback when no local model is available for routing."""
    text = message.strip()
    if not text:
        return RouteDecision(
            model="claude", complexity="simple", reasoning="empty message",
            confidence=0.9,
        )

    if _SIMPLE_PATTERNS.match(text) or len(text) < 20:
        return RouteDecision(
            model="claude", complexity="simple",
            reasoning="keyword match: greeting/short (no local model)",
            confidence=0.8,
        )

    if _COMPLEX_SIGNALS.search(text):
        return RouteDecision(
            model="claude", complexity="complex",
            reasoning="keyword match: complexity signals (no local model)",
            confidence=0.7,
        )

    word_count = len(text.split())
    if word_count > 100:
        return RouteDecision(
            model="claude", complexity="complex",
            reasoning="keyword match: long message (no local model)",
            confidence=0.6,
        )

    return RouteDecision(
        model="claude", complexity="medium",
        reasoning="keyword fallback (no local model)",
        confidence=0.5,
    )


# ── Confidence thresholds ─────────────────────────────────────────────────────
# Below this confidence, local routes escalate to Claude.
ESCALATION_THRESHOLD = 0.6
# Below this confidence, the orchestrator is told to expand RAG context.
CONTEXT_EXPANSION_THRESHOLD = 0.5
# Minimum error rate on local routes before we tighten the threshold.
ADAPTIVE_ERROR_FLOOR = 0.15

ROUTER_SYSTEM = """You are a task classifier. Given a user message and conversation context, classify it.
Return ONLY a JSON object:
{
  "complexity": "simple" | "medium" | "complex",
  "model": "local" | "claude",
  "confidence": 0.0 to 1.0,
  "needs_context": true | false,
  "reasoning": "one sentence"
}

Classification rules:
- simple -> local: greetings, simple Q&A, summarization, formatting, data extraction,
  classification, translation, simple math, definitions, list generation
- medium -> local (if 13B+ model available) or claude: multi-step analysis,
  code generation, moderate reasoning, comparisons
- complex -> claude: planning, evaluation, creative writing, nuanced judgment,
  multi-document synthesis, debugging complex code, anything requiring deep reasoning
- If the user says "use Claude" or "@claude", always return "claude".

Confidence scoring:
- 1.0 = you are completely certain about both the classification AND that the
  chosen model can handle it well.
- 0.7+ = confident. Routine request, clear fit.
- 0.4-0.7 = uncertain. The request is ambiguous, or you're unsure the chosen
  model has the knowledge. Set needs_context=true if the answer likely depends
  on specific documents or facts the user has stored.
- <0.4 = very uncertain. Route to claude and set needs_context=true.

needs_context: set true when the question references specific documents, prior
conversations, stored knowledge, or domain facts that aren't general knowledge.
"""


# Legacy alias so any external code importing Route still works.
Route = RouteDecision


class TaskRouter:
    def __init__(self, local_client, settings):
        self.local = local_client
        self._settings = settings
        self._enabled = True  # User can disable routing (always use Claude)

    def classify(self, message: str, history: list | None = None,
                 memory_context=None) -> RouteDecision:
        """Classify a message and return a RouteDecision with confidence."""
        # Deterministic fallback when local model is offline
        if not self.local.is_available():
            log.info("Local model unavailable — using keyword classifier")
            return _keyword_classify(message)
        # If routing is disabled or local unavailable, always use Claude
        if not self._enabled or not self.local.is_available():
            return RouteDecision(model="claude", complexity="complex",
                                reasoning="routing disabled or local unavailable",
                                confidence=1.0)

        # Fast path: explicit user overrides (no model call needed)
        lower = message.lower().strip()
        if any(kw in lower for kw in ("@claude", "use claude", "ask claude")):
            return RouteDecision(model="claude", complexity="complex",
                                reasoning="user requested claude",
                                confidence=1.0)
        if any(kw in lower for kw in ("@local", "use local")):
            return RouteDecision(model="local", complexity="simple",
                                reasoning="user requested local",
                                confidence=1.0)

        # Slow path: ask local model to classify with confidence
        try:
            context = ""
            if history and len(history) > 0:
                last = history[-2:] if len(history) >= 2 else history
                context = "\n".join(
                    f"{m['role']}: {m['content'][:200]}" for m in last
                )

            # Include memory availability hint so the classifier knows
            # whether context expansion is even possible.
            mem_hint = ""
            if memory_context:
                has_rag = bool(getattr(memory_context, "rag_chunks", None))
                has_facts = bool(getattr(memory_context, "session_facts", None))
                if has_rag or has_facts:
                    mem_hint = (
                        "\n[System note: The user has indexed documents and "
                        "stored facts. Set needs_context=true if their question "
                        "might benefit from searching these.]"
                    )

            prompt = f"User message: {message}"
            if context:
                prompt = f"Recent conversation:\n{context}\n\n{prompt}"
            if mem_hint:
                prompt += mem_hint

            result = self.local.chat(ROUTER_SYSTEM, prompt, max_tokens=250)
            route = RouteDecision.from_json(result)

            # ── UAR escalation: low confidence local -> Claude ────────────────
            # Per-complexity adaptive thresholds: error rates differ sharply
            # between simple/medium/complex buckets, so a single aggregate
            # rate over-tightens simple queries and under-tightens medium ones.
            esc_threshold = self._adaptive_threshold_for(route.complexity) if route.complexity else self._adaptive_threshold()
            if route.model == "local" and route.confidence < esc_threshold:
                log.info(
                    "UAR escalation: confidence %.2f < threshold %.2f, "
                    "upgrading local -> claude",
                    route.confidence, esc_threshold,
                )
                return RouteDecision(
                    model="claude",
                    complexity=route.complexity,
                    reasoning=f"low confidence ({route.confidence:.0%}) — escalated to Claude",
                    confidence=route.confidence,
                    needs_context=route.confidence < CONTEXT_EXPANSION_THRESHOLD,
                )

            # ── Heuristic safety net (unchanged) ─────────────────────────────
            if route.model == "local" and self._looks_complex(message):
                log.info("Router override: heuristic says complex, upgrading to Claude")
                return RouteDecision(
                    model="claude", complexity="complex",
                    reasoning="heuristic override — message looks complex",
                    confidence=route.confidence,
                    needs_context=route.needs_context,
                )

            # ── Tag context expansion for any low-confidence route ────────────
            if route.confidence < CONTEXT_EXPANSION_THRESHOLD and not route.needs_context:
                route = RouteDecision(
                    model=route.model,
                    complexity=route.complexity,
                    reasoning=route.reasoning,
                    confidence=route.confidence,
                    needs_context=True,
                )

            log.info(
                "Router -> %s (%s, conf=%.2f, ctx=%s): %s",
                route.model, route.complexity, route.confidence,
                route.needs_context, route.reasoning,
            )
            return route
        except Exception as exc:
            log.warning(f"Router classification failed: {exc} — defaulting to Claude")
            return RouteDecision(model="claude", complexity="complex",
                                reasoning=f"router error: {exc}",
                                confidence=0.0, needs_context=True)

    def _adaptive_threshold_for(self, complexity: str) -> float:
        """
        Per-bucket adaptive escalation threshold.

        Same logic as `_adaptive_threshold()` but restricted to a single
        complexity bucket and with a lower minimum-sample floor (5 rows
        instead of 10). Falls back to the aggregate `_adaptive_threshold()`
        when the per-bucket sample is too thin to be meaningful.
        """
        try:
            import db as _db
            row = _db.fetchone(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN had_error = 1 OR response_empty = 1 THEN 1 ELSE 0 END) as bad "
                "FROM router_log WHERE route_taken = 'local' "
                "AND complexity = ? "
                "AND created_at > datetime('now', '-24 hours')",
                (complexity,),
            )
            if not row or not row["total"] or row["total"] < 5:
                return self._adaptive_threshold()

            error_rate = row["bad"] / row["total"]
            if error_rate > ADAPTIVE_ERROR_FLOOR:
                adjusted = min(0.85, ESCALATION_THRESHOLD + (error_rate - ADAPTIVE_ERROR_FLOOR))
                log.debug(
                    "Adaptive threshold (%s bucket): local error rate %.1f%% -> threshold %.2f",
                    complexity, error_rate * 100, adjusted,
                )
                return adjusted
            return ESCALATION_THRESHOLD
        except Exception:
            return ESCALATION_THRESHOLD

    def _adaptive_threshold(self, complexity: str | None = None) -> float:
        """
        Adjust escalation threshold based on observed local-route error rates.

        If local routes are failing often (> ADAPTIVE_ERROR_FLOOR), tighten
        the threshold so more borderline queries go to Claude. This is a
        feedback loop: the router_log table records errors, and we use that
        signal to self-correct.

        When `complexity` is provided, the error rate is computed for that
        bucket only — simple, medium, and complex have very different
        baseline failure rates, so a single aggregate over-tightens simple
        queries and under-tightens medium ones. Falls back to the aggregate
        rate when the per-bucket sample is too small (< 10 rows).
        """
        try:
            import db as _db
            row = None
            if complexity:
                row = _db.fetchone(
                    "SELECT COUNT(*) as total, "
                    "SUM(CASE WHEN had_error = 1 OR response_empty = 1 THEN 1 ELSE 0 END) as bad "
                    "FROM router_log WHERE route_taken = 'local' "
                    "AND complexity = ? "
                    "AND created_at > datetime('now', '-24 hours')",
                    (complexity,),
                )
            if not row or not row["total"] or row["total"] < 10:
                # Per-bucket sample too thin (or no complexity given) — fall
                # back to the aggregate rate so we still self-correct on
                # overall router health.
                row = _db.fetchone(
                    "SELECT COUNT(*) as total, "
                    "SUM(CASE WHEN had_error = 1 OR response_empty = 1 THEN 1 ELSE 0 END) as bad "
                    "FROM router_log WHERE route_taken = 'local' "
                    "AND created_at > datetime('now', '-24 hours')"
                )
            if not row or not row["total"] or row["total"] < 10:
                return ESCALATION_THRESHOLD  # not enough data

            error_rate = row["bad"] / row["total"]
            if error_rate > ADAPTIVE_ERROR_FLOOR:
                # Tighten: raise threshold proportionally (max 0.85)
                adjusted = min(0.85, ESCALATION_THRESHOLD + (error_rate - ADAPTIVE_ERROR_FLOOR))
                log.debug(
                    "Adaptive threshold (%s): local error rate %.1f%% -> threshold %.2f",
                    complexity or "aggregate", error_rate * 100, adjusted,
                )
                return adjusted
            return ESCALATION_THRESHOLD
        except Exception:
            return ESCALATION_THRESHOLD

    @staticmethod
    def _looks_complex(message: str) -> bool:
        """Heuristic complexity check as a safety net for bad local classifiers."""
        indicators = [
            len(message) > 500,                          # long messages
            message.count('?') > 2,                      # multi-question
            any(w in message.lower() for w in [
                "analyze", "compare", "evaluate", "design", "architect",
                "debug", "optimize", "explain why", "trade-off", "pros and cons",
                "write a", "create a", "build a", "implement", "refactor",
                "prove", "derive", "synthesize", "critique",
            ]),
            bool(re.search(r'```', message)),            # contains code blocks
            bool(re.search(r'\b(if|else|for|while|def|class|function)\b', message)),
        ]
        return sum(indicators) >= 2

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
