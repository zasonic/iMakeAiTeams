# Codebase Review Report

Findings from a full read of the iMakeAiTeams codebase, organized by severity (most likely to cause real user-facing problems first). Each finding has been verified against the source at the cited file:line locations.

---

## BUG 1 — Dead `_active_mem_suffix` variable causes silent context loss on RAG trimming

**File:** `backend/services/chat_orchestrator.py:552`
**Severity:** Medium

`_active_mem_suffix` is initialized to `""` and never reassigned, but on line 625 (inside the RAG-trimming branch) it is checked with `if _active_mem_suffix:` and would be appended to `full_system` if truthy. Because it is always empty, any "active memory suffix" content that was intended to be re-injected after trimming is silently dropped.

This looks like an incomplete refactor — upstream code that set this variable was removed, but the consumer was not.

**Practical impact:** when RAG chunks exceed `max_context_items` and get trimmed, the system prompt is rebuilt from `system_prompt + mem_suffix` but whatever `_active_mem_suffix` was supposed to carry is gone.

**Fix:** either remove the dead branch, or restore the upstream assignment.

---

## BUG 2 — `_invoke_claude` and `_invoke_local` in `hub_router.py` are dead code

**File:** `backend/services/hub_router.py:431-498`
**Severity:** Low (cleanup)

`_invoke_claude` (lines 431-461) and `_invoke_local` (lines 463-498) are private methods on `HubRouter`. The actual `invoke()` method at line 352 calls `client.stream_unified()` and `client.chat_unified()` directly — never these private methods. A `grep` over `backend/` confirms no production callers; the only references are in test names (which test `hub.invoke()`, not these private methods) and one stale comment in `qwen_thinking.py:79`.

**Practical impact:** ~70 lines of dead code, with two parallel dispatch paths existing in the same module — a maintenance hazard.

**Fix:** delete `_invoke_claude`, `_invoke_local`, and the stale comment in `qwen_thinking.py`.

---

## BUG 3 — `stream_multi_turn` return value inconsistency between clients

**Files:**
- `backend/services/local_client.py:179-180` — returns `str`
- `backend/services/claude_client.py:190-196` — returns `tuple[str, object]`

**Severity:** Low (latent)

The unified interface (`stream_unified`/`chat_unified`) papers over this, but any direct caller of `stream_multi_turn` polymorphically would blow up on tuple unpacking against the local client. The dead `_invoke_claude` (Bug 2) does exactly this — if anyone revives it to handle `local`, it breaks.

**Fix:** make both clients return the same shape, ideally a `dict` like the `*_unified` methods, or document the contract explicitly.

---

## BUG 4 — `response_empty` evaluated before escalation can replace the response

**File:** `backend/services/chat_orchestrator.py:929` (computed) → `:1003` (logged)
**Severity:** Medium (analytics corruption)

`response_empty` is computed as `len((response_text or "").strip()) < 20` at line 929. Lines 930-992 can escalate to Claude and replace `response_text` entirely (line 984: `response_text = esc_result.text`). Line 1003 then logs `response_empty` to `router_log` — but it is still the stale value from before escalation.

**Practical impact:** the `router_log` records "response was empty" even when escalation produced a full response, corrupting router-accuracy analytics.

**Note:** there is also a side-issue at line 933 — the escalation gate is `not response_empty`, so a fully empty local response **skips** escalation, even though that is the case where it is most needed.

**Fix:** recompute `response_empty` after escalation, and reconsider the gate logic.

---

## BUG 5 — Budget check uses pre-response `spent` instead of post-response total

**File:** `backend/services/chat_orchestrator.py:454-470, 1070-1074`
**Severity:** Medium (race condition)

`spent` is fetched at line 461 inside `_db._lock`, but the lock is released after the user-message INSERT at line 470. The LLM call then runs (possibly tens of seconds), and the budget warning at line 1071 calculates `new_spent = spent + cost` using the stale `spent`.

**Practical impact:** if two concurrent sends on the same conversation overlap (one starts before the other's `token_usage` INSERT commits), both will use the same `spent` value, and the sum can exceed the budget without either one triggering the warning.

**Fix:** re-read the running total just before the warning check, inside `_db._lock`.

---

## BUG 6 — Two separate `commit()` calls for one logical operation in the send path

**File:** `backend/services/chat_orchestrator.py:1028, 1061`
**Severity:** Medium (durability)

Line 1028 commits the assistant message + conversation update, then line 1061 commits the `token_usage` row separately. A crash (or process kill) between the two commits leaves the conversation with a message but no corresponding `token_usage` row, causing under-counting in budget checks and the token-stats view.

**Fix:** issue both INSERTs and the UPDATE in a single transaction, then `commit()` once.

---

## BUG 7 — `_risk_history` dict grows unbounded across conversations

**File:** `backend/services/chat_orchestrator.py:173, 251, 745-748`
**Severity:** Low (slow leak)

`_risk_history` is keyed by `conversation_id`. Each list is capped at 5 entries by `del history[:-5]` (line 748), but the dict itself has no eviction — entries are removed only in `delete_conversation` (line 251). Active conversations that go quiet but are never deleted leave entries forever.

**Practical impact:** a slow memory leak in long-running sessions with many conversations.

**Fix:** add an LRU bound on the dict, or evict on conversation archival.

---

## BUG 8 — `app://-` CORS origin does not match Electron production loader

**File:** `backend/server.py:225` and `desktop-shell/main.ts:166`
**Severity:** Low

`allow_origins` includes `"app://-"` commented as "electron-vite production." But `desktop-shell/main.ts:166` shows production loads via `mainWindow.loadFile(...)`, which serves from the `file://` protocol — not `app://`.

**Practical impact:** the documented production origin in CORS is wrong. In practice this is moot today because the renderer's outbound calls are to `localhost`, not the renderer origin, and `BearerAuthMiddleware` is the actual auth gate. But the dead allow-list entry is misleading and would matter the moment the renderer issued a true cross-origin request.

**Fix:** remove the `app://-` entry, or replace it with the actual loader origin.

---

## BUG 9 — `@app.on_event("startup")` is deprecated

**File:** `backend/server.py:238`
**Severity:** Low (forward-compat)

`@app.on_event("startup")` has been deprecated since FastAPI 0.93 in favor of lifespan context managers. It still works in current FastAPI, but it is a ticking clock for a future framework upgrade.

**Fix:** migrate to `@asynccontextmanager` lifespan handlers via `FastAPI(lifespan=...)`.

---

## BUG 10 — `session_facts.status` column added in both `_create_schema` AND migrations

**File:** `backend/db.py:190` (schema) and `backend/db.py:662-664` (migration)
**Severity:** Low (noise)

`_create_schema` issues `ALTER TABLE session_facts ADD COLUMN status` (with a `try/except OperationalError` to swallow duplicate-column errors). The migration `"session_facts.status"` does the same `ALTER TABLE`. For a fresh database the schema creates the column, then the migration tries to add it again, hitting the duplicate-column error path on every fresh-database startup.

**Fix:** remove the migration entry once the schema covers it, or remove it from the schema (preferred — let migrations own column adds).

---

## BUG 11 — `agent_performance` table created in both `_create_schema` AND migrations

**File:** `backend/db.py:230-242` (schema) and `backend/db.py:667-678` (migration)
**Severity:** Low (redundancy)

Same pattern as Bug 10. Lines 230-242 `CREATE TABLE IF NOT EXISTS agent_performance` and the index. Migration `"agent_performance.1.0"` does the same. Harmless because of `IF NOT EXISTS`, but redundant.

**Fix:** consolidate into one source of truth.

---

## BUG 12 — Savings calculation in `get_token_stats` uses hardcoded Sonnet input price

**File:** `backend/services/chat_orchestrator.py:1213-1215`
**Severity:** Low (analytics accuracy)

`estimated_savings_usd` multiplies local-token counts by `3.0 / 1_000_000` (Sonnet input price). If the user has custom model prices set, or default routing would have gone to Haiku ($0.80/MTok) or Opus ($15.00/MTok), the savings estimate is wrong.

**Fix:** use the same `_estimate_cost` logic with the appropriate fallback model, or expose a configurable "comparison model" setting.

---

## BUG 13 — `ExecutionClassifier` constructed via `getattr(self.api, "_claude", None)`

**File:** `backend/server.py:160-162`
**Severity:** Low (encapsulation)

`getattr(self.api, "_claude", None)` reaches into the API facade's private attribute. If `_claude` init failed (it is wrapped in `_safe_init`), this silently passes `None` to the classifier, which may not handle it gracefully in all paths.

**Fix:** expose a public accessor on the API facade, and validate in `ExecutionClassifier.__init__` that the client is non-None or document the None-safe behavior.

---

## MINOR — Operator-precedence ambiguity in `hub_router.py:437-438`

```python
tokens_in = getattr(usage, "input_tokens", 0) or 0 if usage else 0
```

Parsed as `(getattr(...) or 0) if usage else 0` due to Python's ternary precedence — happens to work, but reads like it might mean `getattr(...) or (0 if usage else 0)`. (Inside dead code per Bug 2, so it goes away if that is deleted; otherwise add parens.)

---

## ARCHITECTURE NOTE — Single SQLite connection shared across all threads

**File:** `backend/db.py`
**Severity:** Informational

The module uses one `sqlite3.Connection` with `check_same_thread=False` and a module-level `threading.Lock`. Correct for WAL mode with serialized writes, but every read also acquires `_lock`, so under concurrent load (multiple chat sends, Docker health checks, background indexer) lock contention is the ceiling. For a single-user desktop app this is fine — worth noting if the app ever serves multiple sessions.

---

## CORRECTIONS / FINDINGS THAT DID NOT HOLD UP

The following were considered during review but rejected after closer inspection:

### NOT A BUG — `startChatStream` race against `appendChatToken`

The concern was that `chat_token` SSE events could arrive before `startChatStream` set `activeChat`. **Verified false:** `desktop-ui/components/ChatView.tsx:208` calls `startChatStream(activeId)` synchronously **before** `await Chat.send(...)` on line 210. The store is populated before the POST is even issued, so no race window exists.

### NOT A BUG — Double `container.shutdown()` on SIGTERM

The concern was that the SIGTERM handler at `backend/server.py:371` and the `finally` block at line 386 would both run. **Verified false:** the SIGTERM handler ends with `os._exit(0)` (line 373), which bypasses Python cleanup, including the `finally` block. Only the `KeyboardInterrupt` / normal-exit path reaches the `finally` block, and that path does not run the signal handler. No double-shutdown is possible.

---

## Summary

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | Medium | `chat_orchestrator.py:552` | Dead `_active_mem_suffix` drops trimmed-RAG context |
| 2 | Low | `hub_router.py:431-498` | Dead private invoke methods |
| 3 | Low | `local_client.py` / `claude_client.py` | Inconsistent `stream_multi_turn` return shape |
| 4 | Medium | `chat_orchestrator.py:929,1003` | `response_empty` logged stale after escalation |
| 5 | Medium | `chat_orchestrator.py:1071` | Budget check races on stale `spent` |
| 6 | Medium | `chat_orchestrator.py:1028,1061` | Two commits for one logical operation |
| 7 | Low | `chat_orchestrator.py:173` | `_risk_history` dict not evicted |
| 8 | Low | `server.py:225` | CORS `app://-` does not match production loader |
| 9 | Low | `server.py:238` | Deprecated `@app.on_event("startup")` |
| 10 | Low | `db.py:190,662` | `session_facts.status` added in schema and migration |
| 11 | Low | `db.py:230,667` | `agent_performance` created in schema and migration |
| 12 | Low | `chat_orchestrator.py:1214` | Savings calc hardcodes Sonnet price |
| 13 | Low | `server.py:161` | Private-attribute access on API facade |

The Medium-severity items (1, 4, 5, 6) are the priority targets — they affect correctness in normal use. The rest are cleanup, forward-compat, or analytics accuracy.
