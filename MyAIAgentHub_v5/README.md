# MyAI Agent Hub

A local-first desktop AI platform where Claude and local models work as a team.

## What it does

- **Multi-agent chat** — route messages to Claude or local models (Ollama / LM Studio) automatically based on complexity
- **RAG** — index your documents and files; all agents can search them for context
- **Agent builder** — create custom AI agents with their own system prompts, model preferences, and token budgets
- **Teams** — compose agents into coordinated teams with a coordinator
- **Workflows** — describe a goal and let the coordinator decompose it into tasks for the right agents
- **Prompt library** — version-controlled system prompts with A/B testing
- **Three-tier memory** — conversation buffer, auto-extracted session facts (local model, $0), long-term RAG + semantic search

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

Then in Settings → Local Models, set your default model.

## Architecture

```
app/
  core/
    api.py              ← PyWebView JS bridge (wires all services)
    events.py           ← Event bus
    settings.py         ← JSON settings manager
    worker.py           ← Background thread helper
  services/
    agent_registry.py   ← Agent and team CRUD
    chat_orchestrator.py← Unified conversation loop
    claude_client.py    ← Anthropic SDK wrapper
    error_classifier.py ← Structured error logging
    health_monitor.py   ← Pre-flight checks
    local_client.py     ← Ollama / LM Studio client
    memory.py           ← Three-tier memory manager
    prompt_library.py   ← Versioned prompt storage
    rag_index.py        ← Sentence-transformer RAG
    router.py           ← Task complexity classifier
    semantic_search.py  ← ChromaDB semantic search
    task_scheduler.py   ← Multi-agent workflow engine
  frontend/
    index.html          ← App shell
    app.js              ← All UI logic
  db.py                 ← SQLite schema and helpers
  main.py               ← Entry point
```

## Token savings

The smart router classifies every message before sending it:
- **Simple** (greetings, summarization, formatting) → local model ($0)
- **Medium** (analysis, code) → local if 13B+ available, otherwise Claude
- **Complex** (planning, judgment, creativity) → Claude

Override any time with `@claude` or `@local` in your message.

## Requirements

- Python 3.11+
- pywebview, anthropic, sentence-transformers, chromadb, requests, psutil, numpy, tenacity, pydantic
- Anthropic API key (required)
- Ollama or LM Studio (optional, for local inference)
