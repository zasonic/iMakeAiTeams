# iMakeAiTeams

A local-first desktop AI platform where Claude and local models work as a coordinated team.

## What it does

- **Smart routing** — automatically routes messages to Claude or local models (Ollama / LM Studio) based on complexity, saving tokens on simple tasks
- **Multi-agent teams** — create custom AI agents with individual system prompts, model preferences, and token budgets, then compose them into coordinated teams
- **Workflows** — describe a goal and the coordinator decomposes it into tasks for the right agents, with structured handoffs and saga-based checkpoints
- **Three-tier memory** — conversation buffer, auto-extracted session facts (local model, $0), and long-term RAG + semantic search over your documents
- **RAG & knowledge graph** — index documents and files; agents search them for context with hybrid BM25 + semantic retrieval and automatic relationship extraction
- **Goal decomposition** — complex multi-part requests are automatically split into sequential steps with inline progress tracking
- **Extended reasoning visibility** — Claude's thinking is surfaced as a collapsible timeline showing routing decisions, memory recalled, and reasoning steps
- **Prompt library** — version-controlled system prompts with A/B testing
- **Security engine** — multi-layered defenses including context quarantine, input sanitization, adversarial debate, and memory trust scoring
- **Agentic coding** — built-in agent loop with file read/write/edit, bash, git tools, and permission gating

## Setup

1. Install Python 3.11+
2. Install dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```
3. Run:
   ```bash
   # Windows — double-click:
   START_HERE_Windows.vbs

   # Mac — double-click:
   START_HERE_Mac.command
   ```
4. Enter your Anthropic API key in Settings or on first launch

## Optional: Local models (recommended)

Install [Ollama](https://ollama.ai) for free local inference. The smart router will automatically send simple tasks to your local model, saving Claude tokens.

```bash
ollama pull llama3:8b   # or any model you prefer
```

Then in Settings, set your default local model.

## Architecture

```
app/
  core/
    api.py              <- PyWebView JS bridge (wires all services)
    events.py           <- Event bus
    settings.py         <- JSON settings manager
    worker.py           <- Background thread helper
  services/
    agent_loop.py       <- Agentic coding loop (while tool_calls)
    agent_registry.py   <- Agent and team CRUD + Theory of Mind
    chat_orchestrator.py<- Unified conversation loop
    claude_client.py    <- Anthropic SDK wrapper
    local_client.py     <- Ollama / LM Studio client
    memory.py           <- Three-tier memory manager
    router.py           <- Uncertainty-aware task classifier
    rag_index.py        <- Sentence-transformer RAG
    semantic_search.py  <- ChromaDB hybrid search
    task_scheduler.py   <- Multi-agent workflow engine with saga checkpoints
    security_engine.py  <- Context quarantine, risk ledger, rule engine
    input_sanitizer.py  <- Firewall + validation
    goal_decomposer.py  <- Task breakdown engine
    knowledge_graph.py  <- Entity relationship extraction
    context_compressor.py <- Token-aware context trimming
    prompt_library.py   <- Versioned prompt storage + A/B testing
  channels/
    telegram_adapter.py <- Telegram bot integration
    channel_manager.py  <- Multi-channel dispatch
  frontend/
    index.html          <- App shell
    app.js              <- All UI logic (vanilla JS)
  db.py                 <- SQLite schema and helpers
  main.py               <- Entry point
```

## Token savings

The uncertainty-aware router classifies every message before sending it:
- **Simple** (greetings, summarization, formatting) -> local model ($0)
- **Medium** (analysis, code) -> local if 13B+ available, otherwise Claude
- **Complex** (planning, judgment, creativity) -> Claude

Low-confidence local routes automatically escalate to Claude. Override any time with `@claude` or `@local` in your message.

## Requirements

- Python 3.11+
- pywebview, anthropic, sentence-transformers, chromadb, requests, psutil, numpy, tenacity, pydantic
- Anthropic API key (required)
- Ollama or LM Studio (optional, for local inference)
