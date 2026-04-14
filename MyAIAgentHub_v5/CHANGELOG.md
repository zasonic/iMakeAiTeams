# iMakeAiTeams — Changelog

## v5.0.2 (2026 — current)

### Bug Fixes
- Fix 11 settings keys missing from schema — feature toggles now persist correctly
- Fix IndexError crash in fact extraction when local model returns malformed JSON
- Fix active memory (OpenClaw) lost during system prompt rebuilds
- Fix race condition in task lock acquisition (atomic UPDATE)
- Fix duplicate streaming tokens on local-to-Claude escalation
- Fix pre-existing SyntaxError in validation gate

## v4.0.0 (2025)

### New Features
- **Goal Decomposition Engine** — complex multi-part requests are automatically split into sequential steps, each executed with prior context. Step tracker renders inline in chat between your message and the response.
- **Knowledge Graph Memory** — entity relationships are extracted after every conversation and stored in a triple store. Relevant relationships are injected into system context automatically.
- **Interleaved Reasoning Visibility** — Claude's extended thinking is surfaced as a collapsible timeline inside the chat, showing routing decisions, memory recalled, reasoning steps, and cost — all inline.
- **Studio Mode** — a sidebar toggle that expands the nav to reveal Agents, Teams, Workflows, and Prompts. Hidden by default for a clean chat-first experience.

### UI Overhaul
- Sidebar collapses to 60px icon-only strip in simple mode; expands to 220px labeled nav in Studio Mode
- Conversation list moved to a slide-over drawer in simple mode; static panel in Studio Mode
- Assistant bubbles now use glassmorphic styling (backdrop blur + translucent background)
- User bubbles use accent gradient
- Chat input: auto-growing textarea (min 52px, max 200px) with attach button and purple focus glow
- Full Markdown rendering in assistant messages (marked.js + highlight.js)
- First-run wizard redesigned: 4 steps, cost framing pill, verify-on-paste API key flow
- Goal decomposition step tracker with animated state dots
- Thinking timeline inline between messages, collapsed by default

### Settings additions
- Studio Mode toggle (prominent banner at top of settings)
- Goal Decomposition toggle
- Extended Reasoning toggle
- Knowledge Graph toggle
- All toggles use plain-English labels

## v3.1.0
- Iterative Retrieval Refiner (RAG quality)
- Adversarial debate engine
- Prompt library with versioning and A/B experiments
- Workflow checkpoints (saga pattern)
- Semantic search alongside BM25 + hybrid

## v3.0.0
- Theory of Mind agent modeling
- BM25 + hybrid search
- Handoff protocol with trust scoring
- Security scanner / firewall

## v2.0.0 — v2.7.0
- Multi-agent teams with coordinator roles
- Workflow engine with task graph
- Smart routing (Claude ↔ local)
- RAG document indexing
- 3-tier memory (buffer / facts / RAG)
- Hook system for pre/post processing

## v1.0.0
- Initial release

## v4.1.0 — Research-Informed Improvements

Three changes grounded in frontier AI/ML research, adapted to the orchestration layer:

### 1. Uncertainty-Aware Routing (UAR) — `router.py`, `models.py`
*Inspired by: AUQ/UAR framework (arXiv:2601.15703)*

- Router now asks the local classifier for a **confidence score** (0.0–1.0) alongside the route decision.
- **Low confidence local routes escalate to Claude** — replaces the binary complex/simple heuristic with a continuous epistemic signal.
- **Adaptive threshold** — the escalation threshold tightens automatically when `router_log` shows high local error rates (self-correcting feedback loop).
- **Context expansion signal** — low-confidence routes set `needs_context=True`, telling the orchestrator to widen RAG retrieval *before* generating.

### 2. Confidence-Driven Context Expansion — `chat_orchestrator.py`
*Inspired by: Engram U-shaped allocation (arXiv:2601.07372)*

- When the router signals `needs_context=True`, runs a **second retrieval pass** with doubled `top_k` (6 instead of 3), merges and deduplicates.
- **Adaptive memory injection budget** — caps RAG chunks at 2/4/8 for simple/medium/complex queries. Prevents irrelevant retrieved context from overwhelming simple Q&A (the "25% memory, 75% reasoning" heuristic from Engram research).

### 3. Local-Model Debate Rounds — `adversarial_debate.py`
*Inspired by: Worker-judge cost parity research (multi-agent verification literature)*

- Challenge rounds **try the local model first** — structured JSON extraction that 13B+ models handle well.
- Falls back to Claude on parse failure (**judge escalation**).
- Final synthesis stays on Claude (the "judge" in the worker-judge pattern).
- Tracks `used_local` per challenge and `challenges_on_local` in aggregate stats.

### 4. Frontend Confidence Indicator — `app.js`
- Thinking timeline shows **confidence percentage** on route decisions.
- Escalation events display with a warning icon.
- Context expansion triggers a dedicated "Expanding context window" step.

### Files changed
- `app/models.py` — `RouteDecision` gains `confidence: float` and `needs_context: bool`
- `app/services/router.py` — Rewritten with UAR prompt, escalation logic, adaptive threshold
- `app/services/chat_orchestrator.py` — Post-route context expansion + memory budget
- `app/services/adversarial_debate.py` — Local-first challenge rounds with Claude fallback
- `app/frontend/app.js` — Confidence indicator in thinking timeline
- `tests/test_router.py` — 10 new tests for confidence scoring

## v4.2.0 — Structural Security Engine

Five defenses based on empirical benchmark data. Every defense uses structural
constraints — not LLM classifiers (bypassed at 57-93%) or guardrail models
(100% bypass via emoji smuggling). Structural isolation achieves 7.5% residual
ASR, the best measured result in the literature.

### Defense 1: Context Quarantine (`security_engine.py`)
*Based on: Anthropic Citations (10%→0% hallucination), Microsoft Spotlight (<2% ASR)*
- Every RAG chunk tagged with provenance: source type, document ID, retrieval time, similarity score
- Per-source-type influence caps (web_search: 2 chunks, user_document: 6)
- Structural delimiters separating data from instructions in the system prompt
- Model sees WHERE each chunk came from, enabling trust-aware reasoning

### Defense 2: Risk Ledger (`security_engine.py`)
*Calibrated from: AgentDojo (92% Slack ASR), InjecAgent (100% Stage 2), SafetyDrift (85% communication)*
- Every operation records a risk entry with empirically calibrated weight
- Communication: 0.85 (SafetyDrift), Code exec: 0.75 (InjecAgent), Data write: 0.70
- Cumulative scoring per conversation with hard abort at 3.0 and warn at 1.5
- User-facing risk display: "Risk: MODERATE (1.2/3.0) · 4 operations"

### Defense 3: Memory Firewall (`security_engine.py` + `memory.py`)
*Based on: MINJA (98.2% injection), SpAIware (persistent ChatGPT injection), EchoLeak (CVSS 9.3)*
- Fact length caps (300 chars max — injection payloads need space)
- Structural blocklist: role reassignment, system prompt override, base64 smuggling, unicode tags, markdown exfiltration
- Special character density check (injection payloads have high symbol ratios)
- Per-conversation fact cap (50 max — prevents memory flooding)
- TTL enforcement (90 days — even successful injections expire)
- Source attestation with SHA-256 hash for integrity auditing

### Defense 4: Skill Scanner (`security_engine.py`)
*Based on: Snyk ToxicSkills (36% ClawHub malware, 90-100% recall), ClawHavoc (341 malicious skills)*
- Static regex analysis — no ML, can't be evaded by prompt manipulation
- Detects: shell exec, env/credential access, embedded injection, unicode smuggling, external fetch, dynamic eval, base64 payloads, markdown exfiltration, filesystem writes
- Returns severity-rated findings with line numbers
- `scan_skill_is_safe()` for quick pass/fail gating

### Defense 5: Deterministic Rule Engine (`security_engine.py` + `chat_orchestrator.py`)
*Based on: Hackett et al. ACL 2025 (100% emoji bypass), Unicode tag smuggling (90%+ bypass)*
- Runs between context assembly and model call — the exact insertion point the paper identified
- Strips: instruction delimiters ([INST], <system>), role reassignment, unicode tag characters, markdown exfiltration URLs
- Warns: base64 payloads, system prompt extraction attempts
- Deterministic regex — runs in <1ms, can't be prompt-injected because it doesn't use an LLM
- Multiple violations in single context all caught and acted on independently

### Integration
- `chat_orchestrator.py`: Security engine runs after hooks, before model call. Risk abort returns user-friendly message. Security assessment emitted to thinking timeline.
- `memory.py`: Memory firewall validates every fact before storage. Runs before existing trust scan — facts that fail structural validation never reach the LLM-based scanner.
- `app.js`: Security assessment rendered in thinking timeline with shield icon.

### What this does NOT solve (honest limitations)
- Sophisticated paraphrased injection that avoids all blocked patterns
- Novel attack techniques not in the structural blocklist
- Adversarial content that looks like legitimate business writing
- Zero-day attack patterns unknown to the regex rules
- The fundamental unsolvability of indirect prompt injection (no one has solved this)

### What this DOES solve
- Known structural attack signatures (unicode smuggling, markdown exfil, role reassignment)
- Memory flooding and persistence attacks (TTL + caps)
- Untrusted content influence (provenance tagging + source caps)
- Unlimited blast radius (cumulative risk scoring + hard abort)
- Malicious skill/template import (static analysis before execution)

### Files
- NEW: `app/services/security_engine.py` (420 lines)
- NEW: `tests/test_security_engine.py` (330 lines)
- Modified: `app/services/chat_orchestrator.py` (security enforcement insertion)
- Modified: `app/services/memory.py` (memory firewall integration)
- Modified: `app/frontend/app.js` (security assessment event)
