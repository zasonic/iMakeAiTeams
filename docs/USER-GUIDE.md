# User Guide

What each feature does and where to find it.

## Chat

Open a conversation, type a message, watch tokens stream in. Messages are
classified before they leave: simple requests (greetings, formatting,
short answers) go to your local model when one is available; complex
requests (analysis, planning, code review) go to Claude. If no local
model is running, every message routes to Claude and a "Local model
offline" indicator appears in the status bar so you know costs are
higher.

The Conversation Engine drives the per-turn loop: it persists user
messages, recalls memory, asks the Smart Router which model to use,
runs the Safety Layer, dispatches to the worker, and saves the reply.
Per-conversation budget guard, sliding-window risk scoring, and
auto-titling all happen inside that loop.

## Agents

Create specialists with a name, a system prompt, and a model
preference. An agent can be configured to always use Claude, always use
a local model, or let the router decide. Agents can declare skills
(used for routing) and tool restrictions (used to limit what they can
do). Group agents into teams; the team coordinator decomposes a turn
and dispatches sub-tasks to the specialists.

## Memory

Three layers, all on disk:

- **Recent Context** — buffer of the last few turns; used as immediate
  context.
- **Saved Facts** — promoted facts the assistant decided are worth
  remembering across sessions.
- **Document Memory** — files and folders you've added; embedded with
  fastembed (ONNX, no network) and indexed in sqlite-vec.

Search across saved facts and documents from the Memory panel.

## Power Mode (opt-in)

Delegate execution-style messages (write code, run shell, edit files,
browse the web) to OpenClaw running in Docker. Chat keeps working as
before; Power Mode only triggers for messages classified as execution.
Workspace folder, model provider, API key, and gateway port are
configured in Settings → Power Mode.

## Studio Mode

Toggle at the bottom of the sidebar for advanced panels: prompt
engineering, MCP tool servers, security scan log, and diagnostics. By
default these are hidden so the main interface stays clean for
non-technical users.
