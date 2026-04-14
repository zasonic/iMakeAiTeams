"""
services/retrieval_refiner.py

Iterative Retrieval Refinement Engine
=====================================

Transforms the app's RAG from single-pass "retrieve then generate" into an
agentic loop:

    1. Evaluate whether retrieved context is sufficient for the user's query
    2. If not, reformulate the search query to target what's missing
    3. Retrieve again with the new query
    4. Merge and deduplicate results
    5. Repeat until sufficient or max iterations reached

The evaluator uses the local model when available (free) and falls back to a
lightweight heuristic when no local model is running. This keeps the cost of
the reasoning loop at zero for most users.

Design decisions:
  - Refinement runs BEFORE generation, not after. This avoids wasting API
    tokens on a response built from bad context.
  - Max 2 refinement passes (3 total retrieval attempts). Diminishing returns
    beyond that, and latency matters.
  - Each step emits an event so the frontend thinking timeline can show what's
    happening: "Searching documents → Found 2 chunks → Refining search →
    Found 3 more chunks → Context ready"
  - Chunk deduplication by content hash prevents the same paragraph from
    appearing multiple times in the system prompt.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field

log = logging.getLogger("MyAIEnv.retrieval_refiner")

MAX_REFINEMENT_PASSES = 2
MIN_CHUNKS_SUFFICIENT = 1
MIN_RELEVANCE_SCORE = 0.45

# Queries shorter than this rarely benefit from iterative retrieval
SHORT_QUERY_THRESHOLD = 15

_EVAL_SYSTEM = (
    "You are a retrieval evaluator. Given a user's question and the retrieved "
    "document chunks, determine whether the chunks contain enough information "
    "to answer the question well.\n\n"
    "Respond with ONLY a JSON object, no markdown, no backticks:\n"
    '{"sufficient": true/false, "reason": "brief explanation", '
    '"refined_query": "a better search query if not sufficient, else empty string"}\n\n'
    "Rules:\n"
    "- sufficient=true if the chunks directly address the core of the question\n"
    "- sufficient=false if the chunks are tangential, too vague, or missing key info\n"
    "- When sufficient=false, write a refined_query that targets the MISSING info\n"
    "- The refined_query should use different keywords than the original query\n"
    "- Keep the refined_query under 10 words\n"
)


@dataclass
class RefinementStep:
    """One pass of the retrieval refinement loop."""
    query: str
    chunks_found: int
    scores: list[float] = field(default_factory=list)
    sufficient: bool = False
    reason: str = ""
    refined_query: str = ""


@dataclass
class RefinementResult:
    """Final output of the retrieval refiner."""
    chunks: list[tuple[str, float]]  # (text, score) pairs, deduplicated
    steps: list[RefinementStep] = field(default_factory=list)
    total_passes: int = 1
    was_refined: bool = False

    @property
    def texts(self) -> list[str]:
        """Just the chunk texts, for easy injection into MemoryContext."""
        return [text for text, _ in self.chunks]


class RetrievalRefiner:
    """
    Wraps the RAG index with iterative refinement logic.

    Usage:
        refiner = RetrievalRefiner(rag_index, local_client)
        result = refiner.refine(user_message, on_event=callback)
        # result.chunks = deduplicated (text, score) pairs
        # result.steps = reasoning trace for the thinking timeline
    """

    def __init__(self, rag_index, local_client=None, claude_client=None):
        self.rag = rag_index
        self.local = local_client
        self.claude = claude_client

    def refine(
        self,
        user_message: str,
        initial_chunks: list[tuple[str, float]] | None = None,
        top_k: int = 3,
        on_event=None,
    ) -> RefinementResult:
        """
        Run iterative retrieval refinement.

        Args:
            user_message: The user's original question
            initial_chunks: Pre-fetched chunks from memory.get_context() — if
                provided, we skip the first retrieval and evaluate these directly
            top_k: Number of chunks per retrieval pass
            on_event: Optional callback(event_type, data_dict) for thinking timeline

        Returns:
            RefinementResult with deduplicated chunks and reasoning trace
        """

        def _emit(event_type: str, data: dict):
            if on_event:
                try:
                    on_event(event_type, data)
                except Exception:
                    pass

        # Short queries or very simple messages don't benefit from refinement
        if len(user_message.strip()) < SHORT_QUERY_THRESHOLD:
            chunks = initial_chunks or self._retrieve(user_message, top_k)
            step = RefinementStep(
                query=user_message,
                chunks_found=len(chunks),
                scores=[s for _, s in chunks],
                sufficient=True,
                reason="short query — skipping refinement",
            )
            return RefinementResult(
                chunks=chunks,
                steps=[step],
                total_passes=1,
                was_refined=False,
            )

        # ── Pass 1: Initial retrieval ───────────────────────────────────────
        all_chunks: dict[str, tuple[str, float]] = {}  # content_hash -> (text, score)
        steps: list[RefinementStep] = []

        if initial_chunks is not None:
            chunks = initial_chunks
        else:
            chunks = self._retrieve(user_message, top_k)

        self._merge_chunks(all_chunks, chunks)

        step = RefinementStep(
            query=user_message,
            chunks_found=len(chunks),
            scores=[s for _, s in chunks],
        )

        _emit("retrieval_pass", {
            "pass": 1,
            "query": user_message,
            "chunks_found": len(chunks),
            "total_chunks": len(all_chunks),
        })

        # ── Evaluate sufficiency ────────────────────────────────────────────
        if len(all_chunks) == 0:
            # No documents indexed at all — nothing to refine
            step.sufficient = True
            step.reason = "no documents indexed"
            steps.append(step)
            return RefinementResult(
                chunks=list(all_chunks.values()),
                steps=steps,
                total_passes=1,
                was_refined=False,
            )

        eval_result = self._evaluate_sufficiency(
            user_message, list(all_chunks.values())
        )
        step.sufficient = eval_result.get("sufficient", True)
        step.reason = eval_result.get("reason", "")
        step.refined_query = eval_result.get("refined_query", "")
        steps.append(step)

        _emit("retrieval_evaluated", {
            "pass": 1,
            "sufficient": step.sufficient,
            "reason": step.reason,
            "refined_query": step.refined_query,
        })

        # ── Refinement passes ───────────────────────────────────────────────
        current_pass = 1
        while (
            not step.sufficient
            and step.refined_query
            and current_pass < MAX_REFINEMENT_PASSES + 1
        ):
            current_pass += 1
            query = step.refined_query

            log.info(
                "Retrieval refinement pass %d: query=%r (reason: %s)",
                current_pass, query, step.reason,
            )

            new_chunks = self._retrieve(query, top_k)
            new_unique = self._merge_chunks(all_chunks, new_chunks)

            step = RefinementStep(
                query=query,
                chunks_found=len(new_chunks),
                scores=[s for _, s in new_chunks],
            )

            _emit("retrieval_pass", {
                "pass": current_pass,
                "query": query,
                "chunks_found": len(new_chunks),
                "new_unique": new_unique,
                "total_chunks": len(all_chunks),
            })

            if new_unique == 0:
                # Refinement found nothing new — stop
                step.sufficient = True
                step.reason = "refinement found no new chunks"
                steps.append(step)
                break

            eval_result = self._evaluate_sufficiency(
                user_message, list(all_chunks.values())
            )
            step.sufficient = eval_result.get("sufficient", True)
            step.reason = eval_result.get("reason", "")
            step.refined_query = eval_result.get("refined_query", "")
            steps.append(step)

            _emit("retrieval_evaluated", {
                "pass": current_pass,
                "sufficient": step.sufficient,
                "reason": step.reason,
                "refined_query": step.refined_query,
            })

        # ── Sort by score descending and return ─────────────────────────────
        final_chunks = sorted(
            all_chunks.values(), key=lambda x: x[1], reverse=True
        )

        was_refined = current_pass > 1

        if was_refined:
            _emit("retrieval_refined", {
                "total_passes": current_pass,
                "initial_chunks": steps[0].chunks_found,
                "final_chunks": len(final_chunks),
            })

        return RefinementResult(
            chunks=final_chunks,
            steps=steps,
            total_passes=current_pass,
            was_refined=was_refined,
        )

    # ── Internal methods ────────────────────────────────────────────────────

    def _retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Single retrieval pass against the RAG index."""
        try:
            results = self.rag.search(query, top_k=top_k)
            # Normalize to (text, score) tuples
            normalized = []
            for r in results:
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    normalized.append((str(r[0]), float(r[1])))
                elif isinstance(r, str):
                    normalized.append((r, 0.5))
                else:
                    normalized.append((str(r), 0.5))
            return normalized
        except Exception as exc:
            log.debug("Retrieval failed: %s", exc)
            return []

    def _merge_chunks(
        self,
        target: dict[str, tuple[str, float]],
        new_chunks: list[tuple[str, float]],
    ) -> int:
        """
        Merge new chunks into the target dict, deduplicating by content hash.
        If a duplicate is found with a higher score, update the score.
        Returns the number of genuinely new (non-duplicate) chunks added.
        """
        added = 0
        for text, score in new_chunks:
            if score < MIN_RELEVANCE_SCORE:
                continue
            h = self._hash(text)
            if h not in target:
                target[h] = (text, score)
                added += 1
            else:
                # Keep the higher score
                existing_text, existing_score = target[h]
                if score > existing_score:
                    target[h] = (text, score)
        return added

    def _hash(self, text: str) -> str:
        """Content hash for deduplication."""
        return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]

    def _evaluate_sufficiency(
        self,
        user_message: str,
        chunks: list[tuple[str, float]],
    ) -> dict:
        """
        Evaluate whether the retrieved chunks are sufficient to answer the query.

        Tries the local model first (free), falls back to heuristics if
        no local model is available.
        """
        # ── Try local model evaluation ──────────────────────────────────────
        if self.local and self.local.is_available():
            return self._evaluate_with_model(self.local, user_message, chunks)

        # ── Heuristic fallback ──────────────────────────────────────────────
        return self._evaluate_heuristic(user_message, chunks)

    def _evaluate_with_model(
        self, client, user_message: str, chunks: list[tuple[str, float]]
    ) -> dict:
        """Use local model to evaluate retrieval sufficiency."""
        chunk_texts = "\n---\n".join(
            f"[Score: {score:.2f}] {text[:500]}"
            for text, score in chunks[:5]
        )

        prompt = (
            f"User question: {user_message}\n\n"
            f"Retrieved document chunks:\n{chunk_texts}\n\n"
            f"Evaluate whether these chunks are sufficient to answer the question."
        )

        try:
            result = client.chat(_EVAL_SYSTEM, prompt, max_tokens=200)
            return self._parse_eval_json(result)
        except Exception as exc:
            log.debug("Model evaluation failed: %s", exc)
            return self._evaluate_heuristic(user_message, chunks)

    def _evaluate_heuristic(
        self,
        user_message: str,
        chunks: list[tuple[str, float]],
    ) -> dict:
        """
        Fast heuristic evaluation when no local model is available.

        Checks:
        1. Do we have enough chunks?
        2. Are the scores high enough?
        3. Do the chunks share vocabulary with the query?
        """
        if len(chunks) < MIN_CHUNKS_SUFFICIENT:
            return {
                "sufficient": False,
                "reason": f"only {len(chunks)} chunk(s) found",
                "refined_query": self._keyword_reformulate(user_message),
            }

        avg_score = sum(s for _, s in chunks) / len(chunks) if chunks else 0
        if avg_score < 0.35:
            return {
                "sufficient": False,
                "reason": f"low relevance (avg score {avg_score:.2f})",
                "refined_query": self._keyword_reformulate(user_message),
            }

        # Check vocabulary overlap between query and top chunk
        query_words = set(user_message.lower().split())
        top_chunk_words = set(chunks[0][0].lower().split()[:200])
        overlap = len(query_words & top_chunk_words)
        if overlap < 2 and len(query_words) > 3:
            return {
                "sufficient": False,
                "reason": "low vocabulary overlap with top result",
                "refined_query": self._keyword_reformulate(user_message),
            }

        return {"sufficient": True, "reason": "heuristic pass", "refined_query": ""}

    def _keyword_reformulate(self, query: str) -> str:
        """
        Simple keyword-based query reformulation.

        Strips common question words and function words to extract the core
        search terms, then reorders by likely importance.
        """
        stop_words = {
            "what", "is", "are", "was", "were", "how", "do", "does", "did",
            "can", "could", "would", "should", "the", "a", "an", "in", "on",
            "at", "to", "for", "of", "with", "by", "from", "about", "tell",
            "me", "my", "your", "this", "that", "it", "and", "or", "but",
            "i", "we", "they", "you", "he", "she", "please", "explain",
            "describe", "show", "find", "get", "give", "help", "know",
            "want", "need", "like", "think", "any", "some", "there",
        }

        words = query.lower().split()
        keywords = [w.strip("?.,!\"'") for w in words if w.lower().strip("?.,!\"'") not in stop_words]

        if len(keywords) == 0:
            return query  # Can't reformulate, use original

        # Take the most distinctive terms (longer words tend to be more specific)
        keywords.sort(key=lambda w: len(w), reverse=True)
        return " ".join(keywords[:5])

    def _parse_eval_json(self, raw: str) -> dict:
        """Parse the model's JSON evaluation response, with fallbacks."""
        cleaned = raw.strip()
        # Strip markdown fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )

        try:
            result = json.loads(cleaned)
            return {
                "sufficient": bool(result.get("sufficient", True)),
                "reason": str(result.get("reason", "")),
                "refined_query": str(result.get("refined_query", "")),
            }
        except (json.JSONDecodeError, TypeError):
            # If the model output isn't valid JSON, treat as sufficient
            # to avoid blocking the response
            log.debug("Eval JSON parse failed: %r", raw[:200])
            return {"sufficient": True, "reason": "parse error — defaulting to sufficient", "refined_query": ""}
