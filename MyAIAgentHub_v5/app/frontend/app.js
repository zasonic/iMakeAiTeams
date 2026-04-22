/**
 * app.js — iMakeAiTeams v5.0.2
 * Complete frontend logic. No build tools, no frameworks — vanilla JS.
 * Backend: window.pywebview.api.*   Events: window.__emit → handleEvent()
 *
 * ┌────────────────────────────────────────────────────────────────────┐
 * │  FILE ORGANIZATION (search for ══ SECTION headings)               │
 * ├────────────────────────────────────────────────────────────────────┤
 * │  §1  CONFIGURATION & GLOBALS        — marked.js, state object     │
 * │  §2  API BRIDGE & EVENT BUS         — api(), __emit, handleEvent  │
 * │  §3  CHAT UI                        — send, stream, render msgs   │
 * │  §4  CONVERSATION MANAGEMENT        — list, create, delete, branch│
 * │  §5  AGENT & TEAM MANAGEMENT        — CRUD, ToM                  │
 * │  §6  RAG & DOCUMENTS                — file indexing, search       │
 * │  §7  PROMPT LIBRARY                 — versions                    │
 * │  §8  SETTINGS & DIAGNOSTICS         — config panels, health       │
 * │  §9  SECURITY & SAFETY              — scan display, firewall UI   │
 * │  §10 SETUP WIZARD                   — first-run onboarding        │
 * │  §11 INITIALIZATION                 — DOMContentLoaded, nav       │
 * └────────────────────────────────────────────────────────────────────┘
 *
 * REFACTORING NOTE: This file should eventually be split into ES modules
 * per section above. For now, search by section number (e.g. "§3") to
 * navigate quickly.
 */
"use strict";

// ── Configure marked.js ────────────────────────────────────────────────────────
if (typeof marked !== "undefined") {
  marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function(code, lang) {
      if (typeof hljs !== "undefined" && lang && hljs.getLanguage(lang)) {
        try { return hljs.highlight(code, { language: lang }).value; } catch(e){}
      }
      return typeof hljs !== "undefined" ? hljs.highlightAuto(code).value : code;
    }
  });
}

function renderMarkdown(text) {
  try { return marked.parse(text || ""); }
  catch(e) { return (text || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
}

// ── Global state ──────────────────────────────────────────────────────────────
const state = {
  activeView: "chat",
  conversations: [],
  activeConvoId: null,
  messages: [],
  streamBuffer: "",
  isStreaming: false,
  agents: [],
  teams: [],
  prompts: [],
  editingPromptId: null,
  ragChunks: 0,
  lastRoute: { model: "claude", reason: "" },
  tokenStats: { total_cost_usd: 0, estimated_savings_usd: 0 },
  settings: {},
  studioMode: false,
  // Current thinking block state
  _thinkingBlock: null,
  _thinkingSteps: [],
  _thinkingSummaryParts: [],
  _decompositionSteps: {},  // step_num → state (pending/running/done/error)
  _totalDecompSteps: 0,
  _convoSearchFilter: "",
  _searchMethod: "hybrid",
};

// ── API helper ────────────────────────────────────────────────────────────────
async function api(method, ...args) {
  try { return await window.pywebview.api[method](...args); }
  catch(e) { console.error("API error:", method, e); return null; }
}

// ── Event bus ─────────────────────────────────────────────────────────────────
window.__emit = function(event, jsonString) {
  try { handleEvent(event, JSON.parse(jsonString)); }
  catch(e) { console.error("__emit parse error:", e, jsonString); }
};

function handleEvent(event, payload) {
  switch(event) {
    case "chat_token":
      if(payload.conversation_id === state.activeConvoId) {
        // Sentinel token: clear buffer when backend escalates to a different model
        if(payload.token === "\x00__CLEAR__") {
          state.streamBuffer = "";
          updateStreamingBubble("");
          break;
        }
        state.streamBuffer += payload.token;
        updateStreamingBubble(state.streamBuffer);
      }
      break;
    case "chat_done": finalizeStreamingMessage(payload); break;
    case "chat_stopped":
      finalizeStreamingMessage({ text: state.streamBuffer, model: "", route_reason: "stopped",
        tokens_in:0, tokens_out:0, cost_usd:0, message_id:"" });
      break;
    case "chat_error":
      clearTypingIndicator();
      appendErrorMessage(payload.error || "Something went wrong");
      setStreamingState(false);
      break;
    case "service_unavailable":
      // Rate-limit toasts — a broken subsystem can fire many events per page load.
      window.__svcToastShown = window.__svcToastShown || {};
      if(!window.__svcToastShown[payload.service]) {
        window.__svcToastShown[payload.service] = true;
        showToast(
          `${payload.service} is unavailable — see Settings → Subsystem status`,
          "error"
        );
      }
      break;
    case "service_status_update":
      // Live update from deferred-init: a previously-pending service has
      // finished booting. Refresh the Settings + wizard status lists if
      // they're rendered; show a one-time toast only when a service came up
      // that the user likely cares about.
      renderServiceStatusIfVisible();
      if(payload.ok && payload.service === "embedder") {
        showToast("Document search is now ready", "success");
      }
      break;
    case "chat_event": handleStructuredEvent(payload); break;
    case "rag_progress":
      document.getElementById("rag-subtitle").textContent = payload.status || "";
      wizUpdateProgress(payload.status, payload.pct);
      break;
    case "rag_done":
      state.ragChunks = payload.chunks || 0;
      updateRagStats();
      showToast("Indexed " + (payload.chunks||0).toLocaleString() + " chunks", "success");
      wizIndexDone(payload.chunks);
      break;
    case "rag_error":
      showToast("Index error: " + payload.error, "error");
      wizIndexError(payload.error);
      break;
    case "health_check_done":
      renderHealthResults(payload.results || []);
      updateHealthDot(payload.has_failures ? "bad" : "ok");
      break;
    case "diagnostics_ready":
      document.getElementById("diag-status").textContent = "Saved: " + payload.path;
      break;
    case "diagnostics_error":
      document.getElementById("diag-status").textContent = "Export failed: " + payload.error;
      break;
    case "connection_result":
      showConnResult(payload.backend, payload.ok);
      break;
    case "security_scan": handleSecurityScanEvent(payload); break;
    default:
      console.debug("Unhandled event:", event, payload);
  }
}

// ── Structured event handler ──────────────────────────────────────────────────
function handleStructuredEvent(payload) {
  if(payload.type==="security_scan"){ handleSecurityScanEvent(payload); return; }

  switch(payload.type) {
    case "message_start":
      startThinkingBlock();
      break;

    case "route_decided":
      state.lastRoute = { model: payload.model, reason: payload.reasoning||"" };
      updateModelChip(payload.model, payload.complexity);
      {
        const conf = payload.confidence != null ? payload.confidence : 1.0;
        const confPct = Math.round(conf * 100);
        let confLabel = "";
        if (conf < 0.5) confLabel = " · ⚠ low confidence";
        else if (conf < 0.7) confLabel = " · uncertain";
        const escalated = payload.reasoning && payload.reasoning.includes("escalated");
        addThinkingStep({
          icon: payload.model==="claude" ? "✦" : "⚡",
          label: (escalated ? "Escalated to Claude" : (payload.model==="claude" ? "Routing to Claude" : "Routing to local model"))
                 + (conf < 1.0 ? ` (${confPct}% confident)` : ""),
          detail: (payload.reasoning || "") + confLabel,
          status: escalated ? "warn" : "ok",
          summaryChip: payload.model==="claude" ? "Claude" : "Local",
        });
        if (payload.needs_context) {
          addThinkingStep({
            icon: "🔎",
            label: "Expanding context window",
            detail: "Low confidence triggered wider document search",
            status: "spin",
          });
        }
      }
      break;

    case "security_assessment":
      addThinkingStep({
        icon: payload.icon || "🛡️",
        label: payload.label || "Security check",
        detail: payload.detail || "",
        status: payload.status || "ok",
      });
      break;

    case "memory_recalled":
      if(payload.facts_count>0||payload.rag_chunks>0||payload.memories>0) {
        const total = (payload.facts_count||0)+(payload.rag_chunks||0)+(payload.memories||0);
        const detail = total + " memor"+(total===1?"y":"ies")+" recalled";
        addThinkingStep({ icon:"🧠", label:"Memory recalled", detail, status:"ok",
          summaryChip: "📚 " + total });
      }
      break;

    // ── Extended Reasoning events (#4) ──
    case "reasoning_started":
      addThinkingStep({ icon:"🤔", label:payload.label||"Extended reasoning…", detail:payload.detail||"", status:"spin",
        summaryChip:"🤔 Reasoning" });
      break;
    case "reasoning_complete":
      addThinkingStep({
        icon:"💡", label:"Reasoning complete",
        detail: payload.detail || "",
        status:"ok",
        expandContent: payload.thinking_preview ? "<div style='font-size:11px;color:var(--text2);white-space:pre-wrap;padding:8px;background:var(--bg3);border-radius:6px;margin-top:6px;'>" + (payload.thinking_preview||"").replace(/</g,"&lt;") + "…</div>" : "",
      });
      break;

    default:
      console.debug("Unhandled structured event:", payload.type, payload);
  }
}

// ── Thinking block management ─────────────────────────────────────────────────
function startThinkingBlock() {
  state._thinkingBlock = null;
  state._thinkingSteps = [];
  state._thinkingSummaryParts = [];
  state._decompositionSteps = {};
  state._totalDecompSteps = 0;
}

function addThinkingStep({ icon="•", label="", detail="", status="", summaryChip="", expandContent="" }) {
  state._thinkingSteps.push({ icon, label, detail, status, summaryChip, expandContent });
  if(summaryChip) state._thinkingSummaryParts.push(summaryChip);
  renderThinkingBlock();
}

function renderThinkingBlock() {
  let block = state._thinkingBlock;
  if(!block) {
    block = document.createElement("div");
    block.className = "thinking-block";
    const msgs = document.getElementById("messages");
    const ti = document.getElementById("typing-indicator");
    msgs.insertBefore(block, ti);
    state._thinkingBlock = block;
    // Register the toggle listener ONCE at creation, not on every render
    block.addEventListener("click", e => {
      const summary = block.querySelector(".thinking-summary");
      const steps = block.querySelector(".thinking-steps");
      if(summary && e.target.closest(".thinking-summary")) {
        summary.classList.toggle("open");
        if(steps) steps.classList.toggle("open");
      }
    });
  }

  const chips = state._thinkingSummaryParts.filter(Boolean);
  const summaryHTML = chips.map(c => `<span class="t-chip">${escHtml(c)}</span>`).join("");
  const stepsHTML = state._thinkingSteps.map(s => `
    <div class="ts">
      <div class="ts-ic ${s.status||""}">${escHtml(s.icon)}</div>
      <div class="ts-body">
        <div class="ts-lbl">${escHtml(s.label)}</div>
        ${s.detail ? `<div class="ts-dtl">${escHtml(s.detail)}</div>` : ""}
        ${s.expandContent ? s.expandContent : ""}
      </div>
    </div>`).join("");

  block.innerHTML = `
    <div class="thinking-summary">
      <div class="thinking-chips">${summaryHTML||'<span class="t-chip">Thinking…</span>'}</div>
      <span class="t-expand">▼</span>
    </div>
    <div class="thinking-steps">${stepsHTML}</div>`;

  scrollToBottom();
}

function buildDecompTracker(steps) {
  return `<div class="decomp-track" id="decomp-track">${
    steps.map((s,i) => [
      `<div class="decomp-step" data-step="${s.step}" title="${escHtml(s.task)}">${s.step}. ${escHtml((s.task||"").substring(0,20))}…</div>`,
      i < steps.length-1 ? '<span class="decomp-arrow">→</span>' : ""
    ].join("")).join("")
  }</div>`;
}

function renderDecompTracker() {
  const track = document.getElementById("decomp-track");
  if(!track) return;
  track.querySelectorAll(".decomp-step").forEach(el => {
    const step = parseInt(el.dataset.step);
    el.className = "decomp-step " + (state._decompositionSteps[step]||"pending");
  });
}

function clearThinkingBlock() {
  state._thinkingBlock = null;
  state._thinkingSteps = [];
  state._thinkingSummaryParts = [];
}

// ── Navigation ────────────────────────────────────────────────────────────────
window.navigate = function(view) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  const el = document.getElementById("view-" + view);
  if(el) el.classList.add("active");
  const navEl = document.querySelector(`.nav-item[data-nav="${view}"]`);
  if(navEl) navEl.classList.add("active");
  state.activeView = view;
  onViewActivated(view);
};

document.querySelectorAll(".nav-item[data-nav]").forEach(item => {
  item.addEventListener("click", () => {
    navigate(item.dataset.nav);
    closeDrawer();
  });
});

function onViewActivated(view) {
  if(view==="chat") { loadConversations(); loadAgentsForSelect(); }
  else if(view==="agents") loadAgents();
  else if(view==="teams") loadTeams();
  else if(view==="prompts") loadPrompts();
  else if(view==="docs") loadRagStats();
  else if(view==="settings") loadSettings();
}

// ── Studio Mode ───────────────────────────────────────────────────────────────
async function initStudioMode() {
  const result = await api("studio_mode_get");
  const enabled = result && result.enabled;
  setStudioMode(enabled, false);
}

function setStudioMode(enabled, save=true) {
  state.studioMode = enabled;
  const sidebar = document.getElementById("sidebar");
  if(enabled) { sidebar.classList.add("studio"); }
  else { sidebar.classList.remove("studio"); }
  const toggle = document.getElementById("studio-toggle");
  if(toggle) toggle.checked = enabled;
  if(save) api("studio_mode_set", enabled);
}

document.getElementById("studio-toggle").addEventListener("change", function() {
  setStudioMode(this.checked);
});

// ── Drawer / Hamburger (simple mode) ─────────────────────────────────────────
function openDrawer() {
  document.getElementById("drawer-overlay").classList.add("open");
  document.getElementById("convo-drawer").classList.add("open");
  renderConvoList("drawer-items", state.conversations);
}
function closeDrawer() {
  document.getElementById("drawer-overlay").classList.remove("open");
  document.getElementById("convo-drawer").classList.remove("open");
}
document.getElementById("hamburger-btn").addEventListener("click", openDrawer);
document.getElementById("drawer-overlay").addEventListener("click", closeDrawer);
document.getElementById("drawer-close-btn").addEventListener("click", closeDrawer);
document.getElementById("drawer-new-btn").addEventListener("click", () => { closeDrawer(); newConversation(); });
document.getElementById("studio-new-btn").addEventListener("click", newConversation);

// Drawer search
document.getElementById("drawer-search").addEventListener("input", e => {
  state._convoSearchFilter = e.target.value.toLowerCase();
  renderConvoList("drawer-items", state.conversations);
});
document.getElementById("studio-search").addEventListener("input", e => {
  state._convoSearchFilter = e.target.value.toLowerCase();
  renderConvoList("studio-items", state.conversations);
});

// ── Conversation list ─────────────────────────────────────────────────────────
async function loadConversations() {
  const convos = await api("chat_list_conversations");
  state.conversations = convos || [];
  renderConvoList("drawer-items", state.conversations);
  renderConvoList("studio-items", state.conversations);
}

function renderConvoList(containerId, convos) {
  const el = document.getElementById(containerId);
  if(!el) return;
  const filter = state._convoSearchFilter;
  const filtered = filter ? convos.filter(c => (c.title||"").toLowerCase().includes(filter)) : convos;
  if(!filtered.length) { el.innerHTML = '<div style="padding:16px;font-size:12px;color:var(--text3);text-align:center;">No conversations</div>'; return; }
  el.innerHTML = filtered.map(c => {
    const date = c.updated_at ? new Date(c.updated_at).toLocaleDateString() : "";
    const active = c.id === state.activeConvoId ? " active" : "";
    return `<div class="convo-item${active}" data-id="${c.id}">
      <div class="convo-item-main">
        <div class="convo-title">${escHtml(c.title||"Untitled")}</div>
        <div class="convo-meta">${escHtml(date)}</div>
      </div>
      <div class="convo-actions">
        <button class="convo-action-btn rename-btn" data-id="${c.id}" title="Rename">✎</button>
        <button class="convo-action-btn delete-btn" data-id="${c.id}" data-title="${escAttr(c.title||"Untitled")}" title="Delete">✕</button>
      </div>
    </div>`;
  }).join("");

  el.querySelectorAll(".convo-item").forEach(item => {
    item.addEventListener("click", e => {
      if(e.target.closest(".convo-actions")) return; // let action buttons handle their own clicks
      selectConversation(item.dataset.id);
      closeDrawer();
    });
  });

  el.querySelectorAll(".rename-btn").forEach(btn => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      renameConversation(btn.dataset.id);
    });
  });

  el.querySelectorAll(".delete-btn").forEach(btn => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      deleteConversation(btn.dataset.id, btn.dataset.title);
    });
  });
}

async function selectConversation(id) {
  state.activeConvoId = id;
  const convo = state.conversations.find(c => c.id === id);
  document.getElementById("chat-title").textContent = convo ? (convo.title||"Untitled") : "Conversation";
  renderConvoList("drawer-items", state.conversations);
  renderConvoList("studio-items", state.conversations);
  const msgs = await api("chat_get_messages", id, 100);
  state.messages = msgs || [];
  renderMessages();
  clearThinkingBlock();
}

async function newConversation() {
  const result = await api("chat_new_conversation");
  if(!result) return;
  state.activeConvoId = result.id;
  state.messages = [];
  state.streamBuffer = "";
  document.getElementById("chat-title").textContent = "New conversation";
  clearMessages();
  clearThinkingBlock();
  await loadConversations();
}

async function deleteConversation(id, title) {
  showModal("Delete conversation", `
    <p style="font-size:13px;color:var(--text2);line-height:1.6;">
      Delete <strong>${escHtml(title)}</strong>? This cannot be undone.
    </p>
  `, async () => {
    await api("chat_delete_conversation", id);
    if(state.activeConvoId === id) {
      state.activeConvoId = null;
      state.messages = [];
      clearMessages();
      clearThinkingBlock();
      document.getElementById("chat-title").textContent = "No conversation selected";
    }
    await loadConversations();
    showToast("Conversation deleted");
  });
}

async function renameConversation(id) {
  const convo = state.conversations.find(c => c.id === id);
  const currentTitle = convo ? (convo.title || "Untitled") : "Untitled";
  showModal("Rename conversation", `
    <div class="form-group">
      <label class="form-label">Title</label>
      <input class="form-input" id="m-rename-input" value="${escAttr(currentTitle)}" maxlength="120">
    </div>
  `, async () => {
    const newTitle = document.getElementById("m-rename-input").value.trim();
    if(!newTitle) { showToast("Title cannot be empty", "error"); return false; }
    await api("chat_rename_conversation", id, newTitle);
    if(state.activeConvoId === id) {
      document.getElementById("chat-title").textContent = newTitle;
    }
    await loadConversations();
  });
  // Focus and select the input after the modal renders
  setTimeout(() => {
    const inp = document.getElementById("m-rename-input");
    if(inp) { inp.focus(); inp.select(); }
  }, 50);
}

// ── Message rendering ─────────────────────────────────────────────────────────
function renderMessages() {
  clearMessages();
  state.messages.forEach(m => {
    if(m.role==="user") appendUserMessage(m.content);
    else if(m.role==="assistant") appendAssistantMessage(m.content, m.model_used, m.cost_usd, m.tokens_out);
  });
  scrollToBottom();
}

function clearMessages() {
  const msgs = document.getElementById("messages");
  const ti = document.getElementById("typing-indicator");
  while(msgs.firstChild && msgs.firstChild !== ti) msgs.removeChild(msgs.firstChild);
  clearTypingIndicator();
}

function appendUserMessage(text) {
  const msgs = document.getElementById("messages");
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="msg-bubble">${escHtml(text)}</div>`;
  msgs.insertBefore(el, document.getElementById("typing-indicator"));
  scrollToBottom();
}

function appendAssistantMessage(text, model, cost, tokens) {
  const msgs = document.getElementById("messages");
  const el = document.createElement("div");
  el.className = "msg assistant";
  const modelLabel = model ? escHtml(model.split("-").slice(0,2).join(" ")) : "";
  const costStr = cost ? ` · $${parseFloat(cost).toFixed(4)}` : "";
  el.innerHTML = `
    <div class="msg-bubble">${renderMarkdown(text)}</div>
    <div class="msg-meta">
      <span>${modelLabel}${costStr}</span>
      <button class="msg-copy-btn" title="Copy response">Copy</button>
    </div>`;
  msgs.insertBefore(el, document.getElementById("typing-indicator"));
  el.querySelectorAll("pre code").forEach(b => { try { hljs.highlightElement(b); } catch(e){} });
  el.querySelector(".msg-copy-btn").addEventListener("click", () => copyMessageText(text, el.querySelector(".msg-copy-btn")));
  scrollToBottom();
  return el;
}

function appendErrorMessage(text) {
  const msgs = document.getElementById("messages");
  const el = document.createElement("div");
  el.className = "msg assistant msg-error";
  el.innerHTML = `<div class="msg-bubble">⚠️ ${escHtml(text)}</div>`;
  msgs.insertBefore(el, document.getElementById("typing-indicator"));
  scrollToBottom();
}

// Streaming bubble
let _streamingEl = null;
function showTypingIndicator(label="Thinking…") {
  document.getElementById("typing-label").textContent = label;
  document.getElementById("typing-indicator").classList.add("visible");
  scrollToBottom();
}
function clearTypingIndicator() {
  document.getElementById("typing-indicator").classList.remove("visible");
}

function updateStreamingBubble(text) {
  if(!_streamingEl) {
    clearTypingIndicator();
    const msgs = document.getElementById("messages");
    _streamingEl = document.createElement("div");
    _streamingEl.className = "msg assistant";
    _streamingEl.innerHTML = `<div class="msg-bubble"></div>`;
    msgs.insertBefore(_streamingEl, document.getElementById("typing-indicator"));
  }
  _streamingEl.querySelector(".msg-bubble").innerHTML = renderMarkdown(text);
  scrollToBottom();
}

function finalizeStreamingMessage(payload) {
  clearTypingIndicator();
  setStreamingState(false);

  if(_streamingEl) {
    const text = payload.text || state.streamBuffer || "";
    const model = payload.model || "";
    const cost = payload.cost_usd || 0;
    const costStr = cost ? ` · $${parseFloat(cost).toFixed(4)}` : "";
    const modelLabel = model ? escHtml(model.split("-").slice(0,2).join(" ")) : "";
    _streamingEl.querySelector(".msg-bubble").innerHTML = renderMarkdown(text);
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    meta.innerHTML = `<span>${modelLabel}${costStr}</span><button class="msg-copy-btn" title="Copy response">Copy</button>`;
    meta.querySelector(".msg-copy-btn").addEventListener("click", () => copyMessageText(text, meta.querySelector(".msg-copy-btn")));
    _streamingEl.appendChild(meta);
    _streamingEl.querySelectorAll("pre code").forEach(b => { try { hljs.highlightElement(b); } catch(e){} });
    _streamingEl = null;
  }

  // Finalize thinking block — collapse it
  if(state._thinkingBlock) {
    // Just leave it there — it's already rendered between user msg and assistant msg
    // Build a final summary chip line
    const cost = payload.cost_usd || 0;
    const costStr = cost > 0 ? ` · $${parseFloat(cost).toFixed(4)}` : " · $0.00";
    state._thinkingSummaryParts.push(costStr);
    renderThinkingBlock();
    clearThinkingBlock();
  }

  state.streamBuffer = "";
  state.tokenStats.total_cost_usd = (state.tokenStats.total_cost_usd||0) + (payload.cost_usd||0);
  document.getElementById("cost-display").textContent = "$" + state.tokenStats.total_cost_usd.toFixed(4);

  if(payload.budget_warning) showToast(payload.budget_warning, "error");
  loadConversations();
  scrollToBottom();
}

function setStreamingState(streaming) {
  state.isStreaming = streaming;
  document.getElementById("send-btn").style.display = streaming ? "none" : "flex";
  document.getElementById("stop-btn").style.display = streaming ? "flex" : "none";
  document.getElementById("chat-input").disabled = streaming;
}

function scrollToBottom() {
  const msgs = document.getElementById("messages");
  msgs.scrollTop = msgs.scrollHeight;
}

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage() {
  if(state.isStreaming) return;
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if(!text) return;
  if(!state.activeConvoId) {
    const res = await api("chat_new_conversation");
    if(!res) return;
    state.activeConvoId = res.id;
    await loadConversations();
  }
  input.value = "";
  input.style.height = "52px";
  state.streamBuffer = "";
  startThinkingBlock();
  setStreamingState(true);
  appendUserMessage(text);
  showTypingIndicator("Thinking…");
  const agentId = document.getElementById("agent-select").value || "";
  api("chat_send", state.activeConvoId, text, agentId);
}

document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("stop-btn").addEventListener("click", () => api("chat_stop"));
document.getElementById("chat-input").addEventListener("keydown", e => {
  if(e.key==="Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// Auto-growing textarea
document.getElementById("chat-input").addEventListener("input", function() {
  this.style.height = "52px";
  this.style.height = Math.min(this.scrollHeight, 200) + "px";
});

// Attach button
document.getElementById("attach-btn").addEventListener("click", async () => {
  const files = await api("pick_files");
  if(files && files.length) {
    for(const f of files) {
      const r = await api("rag_add_file", f);
      if(r && !r.error) showToast("Added: " + f.split(/[\\/]/).pop(), "success");
      else showToast("Error adding file", "error");
    }
    loadRagStats();
  }
});

// ── Model chip update ─────────────────────────────────────────────────────────
function updateModelChip(model, complexity) {
  const chip = document.getElementById("model-chip");
  const dot = document.getElementById("chip-dot");
  const label = document.getElementById("chip-label");
  if(model==="claude") {
    dot.className = "chip-dot";
    label.textContent = "Claude Sonnet";
    chip.title = complexity || "";
  } else {
    dot.className = "chip-dot local";
    label.textContent = "Local Model";
    chip.title = complexity || "";
  }
}

// ── Agents for chat selector ──────────────────────────────────────────────────
async function loadAgentsForSelect() {
  const agents = await api("agent_list");
  state.agents = agents || [];
  const sel = document.getElementById("agent-select");
  const current = sel.value;
  sel.innerHTML = '<option value="">No agent</option>' +
    state.agents.map(a => `<option value="${a.id}">${escHtml(a.name)}</option>`).join("");
  if(current) sel.value = current;
  updateAgentLabel(sel.value);
}

// Registered once at startup — not inside loadAgentsForSelect
document.getElementById("agent-select").addEventListener("change", function(){ updateAgentLabel(this.value); });

function updateAgentLabel(agentId) {
  const agent = state.agents.find(a => a.id===agentId);
  document.getElementById("agent-label").textContent = agent ? "Agent: " + agent.name : "No agent";
}

// ── Avatar color helper ───────────────────────────────────────────────────────
const AVATAR_COLORS = ["#7c3aed","#0891b2","#0d9488","#dc2626","#d97706","#059669","#7c3aed","#db2777","#2563eb"];
function avatarColor(name) {
  let h = 0; for(let i=0;i<name.length;i++) h = (h*31 + name.charCodeAt(i)) & 0xffffffff;
  return AVATAR_COLORS[Math.abs(h) % AVATAR_COLORS.length];
}
function initials(name) { return (name||"?").split(/\s+/).map(w=>w[0]).slice(0,2).join("").toUpperCase(); }

// ── Agents view ───────────────────────────────────────────────────────────────
async function loadAgents() {
  const grid = document.getElementById("agents-grid");
  grid.innerHTML = '<div class="loading-state">Loading agents…</div>';
  const agents = await api("agent_list");
  state.agents = agents || [];
  if(!state.agents.length) { grid.innerHTML = '<div class="empty-state">No agents yet. Click <strong>New Agent</strong> to create one.</div>'; return; }
  grid.innerHTML = state.agents.map(a => {
    const bg = avatarColor(a.name||"");
    const ini = initials(a.name||"");
    const model = a.model_preference || "auto";
    const spec = a.domain || a.role || "General assistant";
    return `<div class="agent-card" data-agent-id="${a.id}">
      <div class="agent-card-top">
        <div class="agent-avatar" style="background:${bg};">${ini}</div>
        <div style="flex:1;min-width:0;"><div class="agent-name">${escHtml(a.name||"")}</div><div class="agent-spec">${escHtml(spec)}</div></div>
      </div>
      <div class="agent-footer">
        <span class="badge-sm ${model}">${model}</span>
        ${a.is_builtin ? '<span class="badge-builtin">Built-in</span>' : ''}
        <div style="margin-left:auto;display:flex;gap:5px;">
          ${!a.is_builtin ? `<button class="btn-sm" data-action="edit-agent" data-id="${a.id}">Edit</button>` : ""}
          <button class="btn-sm danger" data-action="delete-agent" data-id="${a.id}" data-name="${escAttr(a.name||"")}">Delete</button>
        </div>
      </div>
    </div>`;
  }).join("");
}

document.getElementById("new-agent-btn").addEventListener("click", () => openAgentModal());

function openAgentModal(agentId) {
  const agent = agentId ? state.agents.find(a=>a.id===agentId) : null;
  const initialBudget = (agent && agent.thinking_budget != null) ? agent.thinking_budget : 2048;
  showModal(agent ? "Edit Agent" : "New Agent", `
    <div class="form-group"><label class="form-label">Name</label><input class="form-input" id="m-agent-name" value="${escAttr(agent?.name||"")}"></div>
    <div class="form-group"><label class="form-label">Specialty / Domain</label><input class="form-input" id="m-agent-domain" value="${escAttr(agent?.domain||"")}" placeholder="e.g. Code review, Legal analysis…"></div>
    <div class="form-group"><label class="form-label">System Prompt</label><textarea class="form-input" id="m-agent-prompt" style="height:120px;resize:vertical;font-family:var(--mono);font-size:12px;">${escHtml(agent?.system_prompt||"")}</textarea></div>
    <div class="form-group"><label class="form-label">Model</label><select class="form-input" id="m-agent-model"><option value="auto">Auto (smart routing)</option><option value="claude">Always Claude</option><option value="local">Always Local (free)</option></select></div>
    <div class="form-group">
      <label class="form-label">Thinking budget <span style="font-weight:400;color:var(--text3);">(local Qwen3 only — 0 disables /think)</span></label>
      <div style="display:flex;align-items:center;gap:10px;">
        <input type="range" id="m-agent-thinking" min="0" max="8192" step="256" value="${initialBudget}" style="flex:1;">
        <span id="m-agent-thinking-val" style="font-family:var(--mono);font-size:12px;color:var(--text2);min-width:60px;text-align:right;">${initialBudget} tok</span>
      </div>
    </div>
  `, async () => {
    const name = document.getElementById("m-agent-name").value.trim();
    if(!name) { showToast("Name required","error"); return false; }
    const budget = parseInt(document.getElementById("m-agent-thinking").value, 10);
    const data = {
      name,
      domain: document.getElementById("m-agent-domain").value.trim(),
      system_prompt: document.getElementById("m-agent-prompt").value.trim(),
      model_preference: document.getElementById("m-agent-model").value,
      description: document.getElementById("m-agent-domain").value.trim(),
      thinking_budget: isNaN(budget) ? 2048 : budget,
    };
    if(agentId) await api("agent_update", agentId, data);
    else await api("agent_create", data.name, data.description, data.system_prompt, data.model_preference);
    loadAgents();
  });
  if(agent) { setTimeout(()=>{ const sel=document.getElementById("m-agent-model"); if(sel) sel.value=agent.model_preference||"auto"; },50); }
  setTimeout(() => {
    const slider = document.getElementById("m-agent-thinking");
    const out = document.getElementById("m-agent-thinking-val");
    if(slider && out) {
      slider.addEventListener("input", () => { out.textContent = `${slider.value} tok`; });
    }
  }, 50);
}

function editAgent(id) { openAgentModal(id); }

async function deleteAgent(id, name) {
  showModal("Delete agent", `
    <p style="font-size:13px;color:var(--text2);line-height:1.6;">
      Delete agent <strong>${escHtml(name)}</strong>? This cannot be undone.
    </p>
  `, async () => {
    await api("agent_delete", id);
    loadAgents();
    showToast("Agent deleted");
  });
}

// ── Teams view ────────────────────────────────────────────────────────────────
async function loadTeams() {
  const list = document.getElementById("teams-list");
  list.innerHTML = '<div class="loading-state">Loading teams…</div>';
  const teams = await api("team_list");
  state.teams = teams || [];
  const agents = await api("agent_list");
  state.agents = agents || [];
  if(!state.teams.length) { list.innerHTML = '<div class="empty-state">No teams yet. Click <strong>New Team</strong> to create one.</div>'; return; }
  list.innerHTML = state.teams.map(t => {
    const members = t.members || [];
    const membersHtml = members.map(m => {
      const isCoord = m.role === "coordinator";
      const bg = avatarColor(m.agent_name||"");
      const ini = initials(m.agent_name||"");
      return `<div class="member-pill ${isCoord?"coord":""}">
        <div class="member-avatar" style="background:${bg};">${ini}</div>
        <span>${escHtml(m.agent_name||"")}</span>
        ${isCoord ? '<span class="member-role">coordinator</span>' : ''}
        <button class="btn-sm danger" style="padding:1px 5px;font-size:10px;"
          data-action="remove-member" data-team-id="${t.id}" data-agent-id="${m.agent_id}">✕</button>
      </div>`;
    }).join("");
    const agentOptions = state.agents.map(a => `<option value="${a.id}">${escHtml(a.name)}</option>`).join("");
    return `<div class="team-card" data-team-id="${t.id}">
      <div class="team-card-top"><div class="team-name">${escHtml(t.name||"")}</div>
        <button class="btn-sm danger" data-action="delete-team" data-id="${t.id}" data-name="${escAttr(t.name||"")}">Delete</button>
      </div>
      <div class="team-desc">${escHtml(t.description||"")}</div>
      <div class="team-roster">${membersHtml||'<span style="font-size:12px;color:var(--text3);">No members yet</span>'}</div>
      <div class="team-add-row">
        <select class="team-add-agent" data-team-id="${t.id}">${agentOptions}</select>
        <select class="team-add-role" data-team-id="${t.id}" style="width:120px;background:var(--bg3);border:1px solid var(--border);border-radius:7px;padding:6px 10px;color:var(--text);font-size:12px;">
          <option value="member">Member</option><option value="coordinator">Coordinator</option>
        </select>
        <button class="btn btn-primary" style="padding:6px 12px;font-size:12px;"
          data-action="add-member" data-team-id="${t.id}">Add</button>
      </div>
    </div>`;
  }).join("");
}

document.getElementById("new-team-btn").addEventListener("click", () => {
  showModal("New Team", `
    <div class="form-group"><label class="form-label">Team Name</label><input class="form-input" id="m-team-name"></div>
    <div class="form-group"><label class="form-label">Description</label><input class="form-input" id="m-team-desc" placeholder="What does this team do?"></div>
  `, async () => {
    const name = document.getElementById("m-team-name").value.trim();
    if(!name) { showToast("Name required","error"); return false; }
    await api("team_create", name, document.getElementById("m-team-desc").value.trim(), []);
    loadTeams();
  });
});

async function addTeamMemberDirect(teamId, agentId, role) {
  if(!agentId) return;
  await api("team_add_member", teamId, agentId, role || "member");
  loadTeams();
}
// Keep old name as alias in case called elsewhere
async function addTeamMember(teamId) {
  const sel = document.getElementById("team-add-select-"+teamId);
  const roleSel = document.getElementById("team-add-role-"+teamId);
  if(!sel||!roleSel) return;
  await addTeamMemberDirect(teamId, sel.value, roleSel.value);
}
async function removeTeamMember(teamId, agentId) {
  await api("team_remove_member", teamId, agentId);
  loadTeams();
}
async function deleteTeam(id, name) {
  showModal("Delete team", `
    <p style="font-size:13px;color:var(--text2);line-height:1.6;">
      Delete team <strong>${escHtml(name)}</strong>? This cannot be undone.
    </p>
  `, async () => {
    await api("team_delete", id);
    loadTeams();
    showToast("Team deleted");
  });
}

// ── Prompts view ──────────────────────────────────────────────────────────────
async function loadPrompts() {
  const list = document.getElementById("prompts-list");
  list.innerHTML = '<div class="loading-state">Loading prompts…</div>';
  const prompts = await api("prompt_list");
  state.prompts = prompts || [];
  if(!state.prompts.length) { list.innerHTML = '<div class="empty-state">No prompts yet. Click <strong>New Prompt</strong> to create one.</div>'; return; }
  list.innerHTML = state.prompts.map(p => `
    <div class="prompt-item ${p.id===state.editingPromptId?"editing":""}" data-prompt-id="${p.id}">
      <div class="prompt-name">${escHtml(p.name||"")}</div>
      <div class="prompt-meta"><span>${escHtml(p.category||"")}</span><span>v${p.current_version||1}</span></div>
    </div>`).join("");
}

function editPrompt(id) {
  state.editingPromptId = id;
  const prompt = state.prompts.find(p=>p.id===id);
  if(!prompt) return;
  loadPrompts();
  const editorEl = document.getElementById("prompt-editor");
  editorEl.style.display = "block";
  editorEl.innerHTML = `
    <div class="prompt-edit-area">
      <div style="font-size:13px;font-weight:600;margin-bottom:10px;">${escHtml(prompt.name||"")}</div>
      <textarea id="prompt-edit-text">${escHtml(prompt.current_text||"")}</textarea>
      <div style="display:flex;gap:8px;margin-top:10px;">
        <button class="btn btn-primary" id="prompt-save-btn">Save</button>
        <button class="btn" id="prompt-cancel-btn">Cancel</button>
        <button class="btn btn-danger" id="prompt-delete-btn">Delete</button>
      </div>
    </div>`;
  document.getElementById("prompt-save-btn").addEventListener("click",   () => savePrompt(id));
  document.getElementById("prompt-cancel-btn").addEventListener("click", () => closePromptEditor());
  document.getElementById("prompt-delete-btn").addEventListener("click", () => deletePrompt(id));
}

function closePromptEditor() {
  state.editingPromptId = null;
  document.getElementById("prompt-editor").style.display = "none";
  loadPrompts();
}

async function savePrompt(id) {
  const text = document.getElementById("prompt-edit-text").value;
  await api("prompt_save", id, text, "");
  showToast("Prompt saved","success");
  closePromptEditor();
}

async function deletePrompt(id) {
  showModal("Delete prompt", `
    <p style="font-size:13px;color:var(--text2);line-height:1.6;">
      Delete this prompt permanently? This cannot be undone.
    </p>
  `, async () => {
    await api("prompt_delete", id);
    closePromptEditor();
    loadPrompts();
  });
}

document.getElementById("new-prompt-btn").addEventListener("click", () => {
  showModal("New Prompt", `
    <div class="form-group"><label class="form-label">Name</label><input class="form-input" id="m-prompt-name"></div>
    <div class="form-group"><label class="form-label">Category</label><input class="form-input" id="m-prompt-cat" placeholder="e.g. writing, coding, analysis"></div>
    <div class="form-group"><label class="form-label">Description</label><input class="form-input" id="m-prompt-desc"></div>
    <div class="form-group"><label class="form-label">Prompt Text</label><textarea class="form-input" id="m-prompt-text" style="height:120px;font-family:var(--mono);font-size:12px;resize:vertical;"></textarea></div>
  `, async () => {
    const name = document.getElementById("m-prompt-name").value.trim();
    if(!name) { showToast("Name required","error"); return false; }
    await api("prompt_create", name, document.getElementById("m-prompt-cat").value.trim(),
      document.getElementById("m-prompt-desc").value.trim(), document.getElementById("m-prompt-text").value.trim());
    loadPrompts();
  });
});

// ── Documents view ────────────────────────────────────────────────────────────
async function loadRagStats() {
  const status = await api("rag_status");
  if(status) {
    document.getElementById("rag-chunks").textContent = (status.total_chunks||0).toLocaleString();
    document.getElementById("rag-sources").textContent = (status.total_sources||0).toLocaleString();
    state.ragChunks = status.total_chunks || 0;
  }
}

function updateRagStats() { loadRagStats(); }

// Search method buttons
document.querySelectorAll(".method-btn").forEach(btn => {
  btn.addEventListener("click", function() {
    document.querySelectorAll(".method-btn").forEach(b=>b.classList.remove("active"));
    this.classList.add("active");
    state._searchMethod = this.dataset.method;
  });
});

async function doDocSearch() {
  const q = document.getElementById("doc-search").value.trim();
  if(!q) return;
  let results = [];
  if(state._searchMethod==="hybrid") results = await api("rag_search_hybrid", q, 8) || [];
  else if(state._searchMethod==="semantic") results = await api("rag_search", q, 8) || [];
  else results = (await api("rag_search_hybrid", q, 8, "bm25")) || [];

  const el = document.getElementById("doc-results");
  if(!results.length) { el.innerHTML = '<div style="color:var(--text3);font-size:13px;text-align:center;padding:16px;">No results</div>'; return; }
  el.innerHTML = results.map(r => {
    const text = typeof r==="string" ? r : (r.text||r.content||JSON.stringify(r));
    const src = typeof r==="object" ? (r.source||r.metadata?.source||"") : "";
    return `<div class="search-result">
      ${src ? `<div class="search-result-src">${escHtml(src)}</div>` : ""}
      <div class="search-result-txt">${escHtml((text||"").substring(0,300))}</div>
    </div>`;
  }).join("");
}

function clearDocResults() { document.getElementById("doc-results").innerHTML = ""; }

document.getElementById("doc-search-btn").addEventListener("click", doDocSearch);
document.getElementById("doc-clear-btn").addEventListener("click", clearDocResults);
document.getElementById("doc-search").addEventListener("keydown", e => { if(e.key==="Enter") doDocSearch(); });

document.getElementById("add-file-btn").addEventListener("click", async () => {
  const files = await api("pick_files");
  if(!files || !files.length) return;
  for(const f of files) {
    const r = await api("rag_add_file", f);
    if(r && !r.error) showToast("Added: " + f.split(/[\\/]/).pop(), "success");
    else showToast("Error: " + (r?.error||"unknown"), "error");
  }
  loadRagStats();
});

document.getElementById("index-folder-btn").addEventListener("click", async () => {
  const folder = await api("pick_folder");
  if(!folder) return;
  document.getElementById("rag-subtitle").textContent = "Indexing…";
  api("build_rag_index", folder);
});

// Drop zone
// Drop zone — pywebview intercepts native drag-drop, so we trigger the file picker instead.
// We do support the visual drag-over state for UX feedback, then fall through to the picker.
const dz = document.getElementById("drop-zone");
dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("drag-over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
dz.addEventListener("drop", async e => {
  e.preventDefault();
  dz.classList.remove("drag-over");
  // pywebview sandboxes the renderer — native File objects from drag-drop don't have real paths.
  // Open the file picker instead so the backend gets a real filesystem path.
  const files = await api("pick_files");
  if(!files || !files.length) return;
  document.getElementById("rag-subtitle").textContent = "Indexing…";
  for(const f of files) {
    const r = await api("rag_add_file", f);
    if(r && !r.error) showToast("Added: " + f.split(/[\\\/]/).pop(), "success");
    else showToast("Error: " + (r?.error || "unknown"), "error");
  }
  loadRagStats();
});
dz.addEventListener("click", () => document.getElementById("add-file-btn").click());

// ── Settings view ─────────────────────────────────────────────────────────────
async function loadSettings() {
  const s = await api("get_settings");
  if(!s) return;
  state.settings = s;
  const get = k => s[k];

  // API key: never fill the input with the masked value — use placeholder instead
  const apiKeyEl = document.getElementById("s-api-key");
  if(apiKeyEl) {
    apiKeyEl.value = "";
    apiKeyEl.placeholder = get("claude_api_key_set")
      ? "sk-ant-••••••••••••  (saved — paste new key to replace)"
      : "sk-ant-api03-…";
  }
  const modelEl = document.getElementById("s-model");
  if(modelEl && get("claude_model")) modelEl.value = get("claude_model");
  const localUrlEl = document.getElementById("s-local-url");
  if(localUrlEl && get("local_model_url")) localUrlEl.value = get("local_model_url");
  const localModelEl = document.getElementById("s-local-model");
  if(localModelEl && get("default_local_model")) localModelEl.value = get("default_local_model");
  const sysEl = document.getElementById("s-system");
  if(sysEl && get("system_prompt")) sysEl.value = get("system_prompt");
  const budgetEl = document.getElementById("s-budget");
  if(budgetEl && get("max_conversation_budget_usd") !== undefined) budgetEl.value = get("max_conversation_budget_usd");

  // Toggles — all keys now returned by get_settings()
  setToggle("s-routing",   get("smart_routing_enabled")         !== false);
  setToggle("s-reasoning", get("interleaved_reasoning_enabled") !== false);
  setToggle("s-firewall",  get("firewall_enabled")              !== false);

  renderServiceStatus();
  renderMcpServers();
}

// ── MCP Servers (Phase 2) ────────────────────────────────────────────────────
async function renderMcpServers() {
  const list = document.getElementById("mcp-servers-list");
  if(!list) return;
  list.innerHTML = "";
  const r = await api("list_mcp_servers");
  const servers = (r && r.servers) || [];
  if(servers.length === 0) {
    const empty = document.createElement("div");
    empty.style.cssText = "font-size:12px;color:var(--text3);padding:8px;text-align:center;background:var(--bg3);border-radius:6px;";
    empty.textContent = "No MCP servers installed yet.";
    list.appendChild(empty);
    return;
  }
  for(const s of servers) {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:10px;padding:10px;border-radius:6px;background:var(--bg3);";
    const info = document.createElement("div");
    info.style.cssText = "flex:1;min-width:0;";
    const title = document.createElement("div");
    title.style.cssText = "font-weight:600;color:var(--text);font-size:13px;";
    title.textContent = `${s.name} `;
    const ver = document.createElement("span");
    ver.style.cssText = "font-weight:400;color:var(--text3);font-size:11px;";
    ver.textContent = `v${s.version} · ${s.tool_count} tool${s.tool_count===1?"":"s"}`;
    title.appendChild(ver);
    const sub = document.createElement("div");
    sub.style.cssText = "font-size:11px;color:var(--text3);font-family:var(--mono);";
    sub.textContent = s.server_id;
    info.appendChild(title);
    info.appendChild(sub);
    const toggle = document.createElement("label");
    toggle.className = "toggle";
    toggle.innerHTML = `<input type="checkbox"${s.enabled?" checked":""}><span class="toggle-slider"></span>`;
    toggle.querySelector("input").addEventListener("change", async (ev) => {
      await api("set_mcp_server_enabled", s.server_id, ev.target.checked);
    });
    const remove = document.createElement("button");
    remove.className = "btn";
    remove.style.cssText = "padding:6px 10px;font-size:12px;";
    remove.textContent = "Remove";
    remove.addEventListener("click", async () => {
      if(!confirm(`Remove MCP server "${s.name}"? This deletes its installed folder.`)) return;
      const out = await api("remove_mcp_server", s.server_id);
      if(out && out.ok) renderMcpServers();
      else showToast(out?.error || "Failed to remove server", "error");
    });
    row.appendChild(info);
    row.appendChild(toggle);
    row.appendChild(remove);
    list.appendChild(row);
  }
}

document.getElementById("mcp-add-btn")?.addEventListener("click", async () => {
  const resultEl = document.getElementById("mcp-add-result");
  resultEl.textContent = "Opening folder picker…";
  resultEl.style.color = "var(--text3)";
  let r = await api("pick_mcp_server_folder", false);
  if(r && !r.ok && r.needs_overwrite_confirm) {
    if(confirm(r.error + " Overwrite?")) {
      r = await api("pick_mcp_server_folder", true);
    } else {
      resultEl.textContent = "";
      return;
    }
  }
  if(!r || (!r.ok && !r.cancelled)) {
    resultEl.textContent = "✗ " + (r?.error || "Failed to install server");
    resultEl.style.color = "#f44336";
    return;
  }
  if(r.cancelled) { resultEl.textContent = ""; return; }
  resultEl.textContent = `✓ Installed ${r.name}${r.overwritten ? " (replaced existing)" : ""}`;
  resultEl.style.color = "#4caf50";
  renderMcpServers();
});

// Shared across the Settings → Subsystem status block and the first-run
// wizard summary so both surfaces use identical human-readable names.
const SERVICE_LABELS = {
  claude_client: "Claude API client",
  local_client: "Local model (Ollama / LM Studio)",
  embedder: "Shared embedding model",
  rag_index: "RAG index",
  rag_load: "RAG cache load",
  database: "SQLite database",
  prompts_seed: "Prompt library",
  agents_seed: "Built-in agents",
  theory_of_mind: "Theory of Mind",
  firewall: "Input firewall",
  semantic_search: "Semantic search (ChromaDB)",
  semantic_search_indexer: "Semantic search indexer",
  memory_manager: "Memory manager",
  router: "Task router",
  chat_orchestrator: "Chat orchestrator",
};

function renderServiceStatusIfVisible() {
  // Settings panel is only mounted when the user navigates to that view.
  // Skip when hidden to avoid a useless api() round-trip per event.
  if(document.getElementById("service-status-list")) renderServiceStatus();
}

async function renderServiceStatus() {
  const list = document.getElementById("service-status-list");
  if(!list) return;
  const status = await api("service_status");
  list.innerHTML = "";
  if(!status || typeof status !== "object") {
    list.textContent = "Service status unavailable.";
    return;
  }
  const names = Object.keys(status).sort();
  for(const name of names) {
    const entry = status[name] || {};
    const ok = !!entry.ok;
    const pending = !!entry.pending;
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:8px;padding:6px 10px;border-radius:6px;background:var(--bg3);font-size:12px;";
    const dot = document.createElement("span");
    const color = pending ? "#f0ad4e" : (ok ? "#4caf50" : "#f44336");
    dot.style.cssText = `width:8px;height:8px;border-radius:50%;background:${color};flex-shrink:0;`;
    const label = document.createElement("span");
    label.textContent = SERVICE_LABELS[name] || name;
    label.style.cssText = "flex:1;color:var(--text);";
    const detail = document.createElement("span");
    detail.style.cssText = `color:${pending ? "#f0ad4e" : (ok ? "var(--text3)" : "#f44336")};font-size:11px;`;
    detail.textContent = pending ? "starting…" : (ok ? "ok" : (entry.error || "unavailable"));
    row.appendChild(dot);
    row.appendChild(label);
    row.appendChild(detail);
    list.appendChild(row);
  }
}

document.getElementById("service-status-refresh")?.addEventListener("click", renderServiceStatus);

function setToggle(id, val) {
  const el = document.getElementById(id);
  if(el) el.checked = !!val;
}

async function saveApiSettings() {
  const key = document.getElementById("s-api-key").value.trim();
  const model = document.getElementById("s-model").value;
  await api("save_setting", "claude_model", model);
  if(key) {
    const r = await api("verify_api_key", key);
    showToast(r && r.ok ? "✓ API key verified — " + (r.message||"") : "✗ " + (r?.message||"Verification failed"), r&&r.ok?"success":"error");
  } else {
    showToast("Model saved", "success");
  }
}

async function saveAllSettings() {
  // Budget: validate before use — empty or non-numeric input must not write NaN
  const budgetRaw = document.getElementById("s-budget").value.trim();
  const budget = parseFloat(budgetRaw);
  if(budgetRaw !== "" && (isNaN(budget) || budget < 0)) {
    showToast("Budget must be a positive number (e.g. 5.00)", "error");
    return;
  }

  // API key: only save if the user typed something new (not empty, not still the placeholder)
  const keyInput = document.getElementById("s-api-key").value.trim();
  if(keyInput) {
    await api("save_setting", "claude_api_key", keyInput);
    // Re-verify silently and update the Claude client
    const r = await api("verify_api_key", keyInput);
    if(!r || !r.ok) {
      showToast("⚠ API key may be invalid: " + (r?.message || "check your key"), "error");
    }
  }

  const settings = {
    claude_model:                   document.getElementById("s-model").value,
    local_model_url:                document.getElementById("s-local-url").value.trim(),
    default_local_model:            document.getElementById("s-local-model").value.trim(),
    system_prompt:                  document.getElementById("s-system").value.trim(),
    smart_routing_enabled:          document.getElementById("s-routing").checked,
    interleaved_reasoning_enabled:  document.getElementById("s-reasoning").checked,
    firewall_enabled:               document.getElementById("s-firewall").checked,
  };
  if(budgetRaw !== "") settings.max_conversation_budget_usd = budget;

  for(const [k, v] of Object.entries(settings)) {
    if(v !== null && v !== undefined && v !== "") await api("save_setting", k, v);
  }

  showToast("All settings saved", "success");
}

document.getElementById("save-api-btn").addEventListener("click", saveApiSettings);
document.getElementById("save-all-btn").addEventListener("click", saveAllSettings);

async function testLocalConn() {
  const url = document.getElementById("s-local-url").value.trim();
  if(!url) { showToast("Enter a local model URL first", "error"); return; }
  await api("save_setting", "local_model_url", url);
  const resultEl = document.getElementById("s-local-result");
  resultEl.textContent = "Testing connection…";
  resultEl.style.color = "var(--text3)";
  // Result arrives via connection_result event → showConnResult()
  api("test_connection", "local");
}
document.getElementById("test-local-btn").addEventListener("click", testLocalConn);

function showConnResult(backend, ok) {
  const r = document.getElementById("s-local-result");
  if(!r) return;
  r.textContent = ok ? "✓ Connected to " + backend : "✗ " + backend + " not reachable — check the URL and that the server is running";
  r.style.color = ok ? "var(--green)" : "var(--red)";
}

async function runHealthCheck() {
  const btn = document.getElementById("health-btn");
  btn.disabled = true;
  btn.textContent = "Running…";
  api("run_health_check");
  setTimeout(()=>{ btn.disabled=false; btn.textContent="Run Health Check"; }, 5000);
}
document.getElementById("health-btn").addEventListener("click", runHealthCheck);

function renderHealthResults(results) {
  const el = document.getElementById("health-results");
  if(!results || !results.length) { el.innerHTML=""; return; }
  el.innerHTML = results.map(r => `
    <div style="display:flex;align-items:center;gap:8px;padding:7px 10px;background:var(--bg3);border-radius:7px;margin-bottom:5px;font-size:12px;">
      <span style="color:${r.ok?"var(--green)":"var(--red)"};">${r.ok?"✓":"✗"}</span>
      <span style="font-weight:600;flex:1;">${escHtml(r.name||"")}</span>
      <span style="color:var(--text3);">${escHtml(r.message||"")}</span>
    </div>`).join("");
}

async function exportConversation() {
  if(!state.activeConvoId) { showToast("No conversation selected", "error"); return; }
  const result = await api("chat_export_conversation", state.activeConvoId, "markdown");
  if(!result || result.error) { showToast("Export failed", "error"); return; }

  // Try native save dialog first; fall back to copy-modal if unavailable
  const saved = await api("save_file_dialog", result.content, result.filename || "conversation.md");
  if(saved && saved.ok) {
    showToast("Saved to " + saved.path.split(/[\\\/]/).pop(), "success");
  } else if(saved && saved.cancelled) {
    // user cancelled — no toast
  } else {
    // Fallback: show content in modal for manual copy
    showModal("Export conversation", `
      <p style="font-size:12px;color:var(--text3);margin-bottom:8px;">Copy the markdown below:</p>
      <textarea style="width:100%;height:260px;font-family:var(--mono);font-size:11px;resize:vertical;background:var(--bg3);border:1px solid var(--border);border-radius:7px;padding:8px;color:var(--text);" readonly>${escHtml(result.content)}</textarea>
    `, null);
    // Replace confirm button with a close-only button
    const btn = document.getElementById("modal-confirm-btn");
    if(btn) { btn.textContent = "Close"; btn.onclick = closeModal; }
  }
}

function updateHealthDot(state) {
  const dot = document.getElementById("health-dot");
  if(!dot) return;
  dot.className = "health-dot" + (state==="bad"?" bad":state==="warn"?" warn":"");
  document.getElementById("health-text").textContent = state==="bad" ? "Issues found" : state==="warn" ? "Warnings" : "System OK";
}

// ── Security scan event ───────────────────────────────────────────────────────
function handleSecurityScanEvent(payload) {
  const status = payload.verdict==="pass" ? "ok" : payload.verdict==="warn" ? "warn" : "error";
  addThinkingStep({
    icon: payload.icon || (status==="ok"?"🛡":"⚠️"),
    label: payload.label || "Security scan",
    detail: payload.detail || "",
    status,
  });
}

// ── First-Run Wizard ──────────────────────────────────────────────────────────
const wizard = { step:1, apiKeyVerified:false, folderIndexed:false, chunks:0, localData:null, ollamaReady:false };

window.showFirstRun = function() {
  document.getElementById("first-run").classList.add("visible");
  wizGoToStep(1);
  // Background: detect local model setup while user enters API key
  api("detect_local_setup").then(data => {
    wizard.localData = data;
    if(wizard.step === 2) wizPopulateLocalStep(data);
  });
};

function wizGoToStep(n) {
  wizard.step = n;
  // Animate panel transition
  document.querySelectorAll(".wz-panel").forEach(p => p.classList.remove("active"));
  const panel = document.getElementById("wz-panel-" + n);
  if(panel) panel.classList.add("active");
  // Update segmented progress bar
  for(let i = 1; i <= 4; i++) {
    const seg = document.getElementById("wz-seg-" + i);
    seg.classList.remove("active", "done");
    if(i === n) seg.classList.add("active");
    else if(i < n) seg.classList.add("done");
  }
  if(n === 2) wizPopulateLocalStep(wizard.localData);
  if(n === 4) wizPopulateReadyStep();
}

// ── Step 1: API Key ──
document.getElementById("wz-verify-btn").addEventListener("click", async () => {
  const key = document.getElementById("wz-api-key").value.trim();
  const btn = document.getElementById("wz-verify-btn");
  const msg = document.getElementById("wz-verify-msg");
  if(!key) { msg.textContent = "Please enter your API key"; msg.className = "wz-msg fail"; return; }
  btn.disabled = true; btn.textContent = "Verifying...";
  msg.innerHTML = ""; msg.className = "wz-msg loading"; msg.textContent = "Connecting to Anthropic...";
  const r = await api("verify_api_key", key);
  btn.disabled = false; btn.textContent = "Verify";
  if(r && r.ok) {
    wizard.apiKeyVerified = true;
    msg.className = "wz-msg ok";
    msg.textContent = "Connected to Claude successfully";
    document.getElementById("wz-next-1").disabled = false;
    await api("save_setting", "claude_api_key", key);
  } else {
    wizard.apiKeyVerified = false;
    msg.className = "wz-msg fail";
    msg.textContent = r?.message || "Verification failed — check your key";
    document.getElementById("wz-next-1").disabled = true;
  }
});
document.getElementById("wz-api-key").addEventListener("keydown", e => { if(e.key === "Enter") document.getElementById("wz-verify-btn").click(); });
document.getElementById("wz-next-1").addEventListener("click", () => wizGoToStep(2));
document.getElementById("wz-console-link").addEventListener("click", e => { e.preventDefault(); api("open_url", "https://console.anthropic.com/"); });

// ── Step 2: Local AI ──
function wizPopulateLocalStep(data) {
  const detail = document.getElementById("wz-local-detail");
  const badge = document.getElementById("wz-local-badge");
  const actionRow = document.getElementById("wz-ollama-action");
  const svc = document.getElementById("wz-ollama-svc");
  const ramHint = document.getElementById("wz-ram-hint");
  if(!data) { detail.textContent = "Detecting..."; badge.textContent = "Checking"; badge.className = "wz-svc-badge checking"; return; }
  if(data.ollama_running) {
    const models = (data.ollama_models || []).join(", ");
    detail.textContent = models ? "Running: " + models.substring(0, 50) : "Running (no models pulled yet)";
    badge.textContent = "Ready"; badge.className = "wz-svc-badge ok";
    svc.classList.add("detected");
    actionRow.style.display = "none";
    wizard.ollamaReady = true;
  } else {
    detail.textContent = "Not running on this machine";
    badge.textContent = "Not found"; badge.className = "wz-svc-badge miss";
    svc.classList.remove("detected");
    actionRow.style.display = "block";
  }
  if(data.ram_gb) {
    ramHint.style.display = "block";
    ramHint.textContent = "System RAM: " + data.ram_gb + " GB — recommended model: " + (data.recommended_model || "phi3:mini");
  }
  // Phase 3: surface Qwen3-30B-A3B detection / fallback notice in plain English.
  const qwenStatus = data.qwen_status;
  if(qwenStatus) {
    let qwenEl = document.getElementById("wz-qwen-status");
    if(!qwenEl) {
      qwenEl = document.createElement("div");
      qwenEl.id = "wz-qwen-status";
      qwenEl.style.cssText = "margin-top:10px;padding:10px;border-radius:6px;font-size:12px;line-height:1.5;";
      ramHint.parentNode.insertBefore(qwenEl, ramHint.nextSibling);
    }
    if(qwenStatus.detected) {
      qwenEl.style.cssText += "background:rgba(76,175,80,0.12);color:#4caf50;border:1px solid rgba(76,175,80,0.3);";
      qwenEl.textContent = "✓ Qwen3-30B-A3B detected (" + qwenStatus.model_id + ") — hybrid thinking ready.";
    } else {
      qwenEl.style.cssText += "background:rgba(240,173,78,0.12);color:#f0ad4e;border:1px solid rgba(240,173,78,0.3);";
      qwenEl.textContent = qwenStatus.fallback_reason || "Qwen3-30B-A3B not detected.";
    }
  }
}
document.getElementById("wz-next-2").addEventListener("click", () => wizGoToStep(3));
document.getElementById("wz-skip-2").addEventListener("click", () => wizGoToStep(3));
document.getElementById("wz-ollama-link").addEventListener("click", () => api("open_url", "https://ollama.ai"));

// ── Step 3: Documents ──
document.getElementById("wz-folder-btn").addEventListener("click", async () => {
  const folder = await api("pick_folder");
  if(!folder) return;
  // Show progress bar
  document.getElementById("wz-index-progress").style.display = "block";
  document.getElementById("wz-index-msg").textContent = "";
  document.getElementById("wz-index-msg").className = "wz-msg";
  document.getElementById("wz-progress-fill").style.width = "2%";
  document.getElementById("wz-index-status").textContent = "Scanning files...";
  document.getElementById("wz-index-pct").textContent = "0%";
  api("build_rag_index", folder);
});

function wizUpdateProgress(status, pct) {
  const fill = document.getElementById("wz-progress-fill");
  const statusEl = document.getElementById("wz-index-status");
  const pctEl = document.getElementById("wz-index-pct");
  if(fill) fill.style.width = Math.min(pct || 0, 100) + "%";
  if(statusEl) statusEl.textContent = status || "Processing...";
  if(pctEl) pctEl.textContent = Math.round(pct || 0) + "%";
}

function wizIndexDone(chunks) {
  wizard.chunks = chunks || 0;
  wizard.folderIndexed = true;
  document.getElementById("wz-index-progress").style.display = "none";
  const msg = document.getElementById("wz-index-msg");
  msg.className = "wz-msg ok";
  msg.textContent = "Indexed " + wizard.chunks.toLocaleString() + " document chunks";
}

function wizIndexError(err) {
  document.getElementById("wz-index-progress").style.display = "none";
  const msg = document.getElementById("wz-index-msg");
  msg.className = "wz-msg fail";
  msg.textContent = err || "Indexing failed";
}

document.getElementById("wz-next-3").addEventListener("click", () => wizGoToStep(4));
document.getElementById("wz-skip-3").addEventListener("click", () => wizGoToStep(4));

// ── Step 4: Ready checklist ──
async function wizPopulateReadyStep() {
  const list = document.getElementById("wz-checklist");
  const items = [];
  // Claude connection
  const modelName = (state.settings?.claude_model || "").includes("opus") ? "Opus" : (state.settings?.claude_model || "").includes("haiku") ? "Haiku" : "Sonnet";
  items.push({ ok: wizard.apiKeyVerified, label: "Claude " + modelName + " connected", detail: "Paid (per-use)" });
  // Local AI
  items.push({ ok: wizard.ollamaReady, label: wizard.ollamaReady ? "Ollama local AI active" : "Local AI skipped", detail: wizard.ollamaReady ? "Free for simple tasks" : "Claude handles everything" });
  // Documents
  if(wizard.chunks > 0) {
    items.push({ ok: true, label: wizard.chunks.toLocaleString() + " document chunks indexed", detail: "Searchable in every chat" });
  } else {
    items.push({ ok: false, label: "No documents added yet", detail: "Add anytime in Settings" });
  }
  list.innerHTML = items.map(it => `
    <div class="wz-check-item">
      <div class="wz-check-icon ${it.ok ? 'ok' : 'skip'}">${it.ok ? '&#10003;' : '&#8212;'}</div>
      <div class="wz-check-label">${escHtml(it.label)}</div>
      <div class="wz-check-detail">${escHtml(it.detail)}</div>
    </div>
  `).join("");

  // Surface any subsystems that failed to start so the user sees the degraded
  // state before clicking into the app. Don't render healthy ones — at 16
  // services a full grid would overwhelm the "you're all set" moment.
  const wzStatus = document.getElementById("wz-subsystem-status");
  if(wzStatus) {
    wzStatus.innerHTML = "";
    const status = await api("service_status");
    if(status && typeof status === "object") {
      const failed = Object.entries(status).filter(([, e]) => e && !e.ok);
      if(failed.length) {
        const heading = document.createElement("div");
        heading.className = "wz-sub";
        heading.style.cssText = "margin:14px 0 6px;font-size:12px;text-align:left;";
        heading.textContent = "Unavailable subsystems (features degrade, chat still works):";
        wzStatus.appendChild(heading);
        wzStatus.innerHTML += failed.map(([name, entry]) => `
          <div class="wz-check-item">
            <div class="wz-check-icon skip">&#8212;</div>
            <div class="wz-check-label">${escHtml(SERVICE_LABELS[name] || name)}</div>
            <div class="wz-check-detail">${escHtml(entry.error || "unavailable")}</div>
          </div>
        `).join("");
      }
    }
  }
}

document.getElementById("wz-finish-btn").addEventListener("click", async () => {
  await api("complete_first_run", "chat");
  document.getElementById("first-run").classList.remove("visible");
  navigate("chat");
  newConversation();
  showToast("Welcome to iMakeAiTeams", "success");
});

// ── Wizard event bridge (called from backend) ──
window.wizUpdateIndexProgress = function(status, pct) { wizUpdateProgress(status, pct); };
window.wizIndexDone = wizIndexDone;
window.wizIndexError = wizIndexError;

// ── Modal helper ──────────────────────────────────────────────────────────────
function showModal(title, bodyHtml, onConfirm) {
  const overlay = document.getElementById("modal-overlay");
  document.getElementById("modal-content").innerHTML = `
    <div class="modal-title">${escHtml(title)}</div>
    <div id="modal-body">${bodyHtml}</div>
    <div class="modal-actions">
      <button class="btn" id="modal-cancel-btn">Cancel</button>
      <button class="btn btn-primary" id="modal-confirm-btn">Save</button>
    </div>`;
  overlay.classList.add("open");
  document.getElementById("modal-cancel-btn").addEventListener("click", closeModal);
  document.getElementById("modal-confirm-btn").addEventListener("click", async () => {
    const result = onConfirm ? await onConfirm() : true;
    if(result !== false) closeModal();
  });
}
function closeModal() { document.getElementById("modal-overlay").classList.remove("open"); }
document.getElementById("modal-overlay").addEventListener("click", e => { if(e.target===e.currentTarget) closeModal(); });

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg, type="") {
  const container = document.getElementById("toast");
  const item = document.createElement("div");
  item.className = "toast-item " + type;
  item.textContent = msg;
  container.appendChild(item);
  setTimeout(() => { item.style.opacity="0"; item.style.transition="opacity .3s"; setTimeout(()=>item.remove(),300); }, 3500);
}

// ── Utility ───────────────────────────────────────────────────────────────────
function escHtml(s) { return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
function escAttr(s) { return String(s||"").replace(/"/g,"&quot;").replace(/'/g,"&#39;"); }

async function copyMessageText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    const orig = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = orig; }, 1800);
  } catch(e) {
    showToast("Copy failed — try selecting text manually", "error");
  }
}

function openUrl(url) { api("open_url", url); }
window.openUrl = openUrl;

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  // Wait for pywebview
  await new Promise(resolve => {
    if(window.pywebview && window.pywebview.api) { resolve(); return; }
    window.addEventListener("pywebviewready", resolve, {once:true});
    setTimeout(resolve, 3000); // fallback
  });

  await initStudioMode();
  await loadConversations();
  await loadAgentsForSelect();

  // Check first run
  const settings = await api("get_settings");
  state.settings = settings || {};
  if(settings && settings.is_first_run) {
    window.showFirstRun();
  } else {
    if(state.conversations.length > 0) {
      await selectConversation(state.conversations[0].id);
    } else {
      await newConversation();
    }
  }

  // ── Permanent delegated listeners ─────────────────────────────────────────
  // These sit on stable container elements whose innerHTML gets rebuilt by
  // load*() functions. Registering once here means they survive every rebuild.

  document.getElementById("agents-grid").addEventListener("click", e => {
    const btn = e.target.closest("[data-action]");
    if(!btn) return;
    const { action, id, name } = btn.dataset;
    if(action === "edit-agent")   editAgent(id);
    if(action === "delete-agent") deleteAgent(id, name);
  });

  document.getElementById("teams-list").addEventListener("click", e => {
    const btn = e.target.closest("[data-action]");
    if(!btn) return;
    const list = document.getElementById("teams-list");
    const action = btn.dataset.action;
    if(action === "delete-team")   deleteTeam(btn.dataset.id, btn.dataset.name);
    if(action === "remove-member") removeTeamMember(btn.dataset.teamId, btn.dataset.agentId);
    if(action === "add-member") {
      const tid = btn.dataset.teamId;
      const agentSel = list.querySelector(`.team-add-agent[data-team-id="${tid}"]`);
      const roleSel  = list.querySelector(`.team-add-role[data-team-id="${tid}"]`);
      if(agentSel && roleSel) addTeamMemberDirect(tid, agentSel.value, roleSel.value);
    }
  });

  document.getElementById("prompts-list").addEventListener("click", e => {
    const item = e.target.closest("[data-prompt-id]");
    if(item) editPrompt(item.dataset.promptId);
  });

  // Export conversation button
  document.getElementById("export-btn").addEventListener("click", exportConversation);

  // Diagnostics button
  document.getElementById("diag-btn").addEventListener("click", () => api("export_diagnostics"));

  // ── Keyboard shortcuts ─────────────────────────────────────────────────────
  document.addEventListener("keydown", e => {
    // Escape: close modal, then drawer, then nothing
    if(e.key === "Escape") {
      const modal = document.getElementById("modal-overlay");
      if(modal && modal.classList.contains("open")) { closeModal(); return; }
      const drawer = document.getElementById("convo-drawer");
      if(drawer && drawer.classList.contains("open")) { closeDrawer(); return; }
    }

    // Cmd/Ctrl + N: new conversation (only when not typing in an input)
    if((e.metaKey || e.ctrlKey) && e.key === "n") {
      const tag = document.activeElement?.tagName;
      if(tag !== "INPUT" && tag !== "TEXTAREA" && tag !== "SELECT") {
        e.preventDefault();
        newConversation();
      }
    }

    // Cmd/Ctrl + Shift + E: export current conversation
    if((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === "e") {
      e.preventDefault();
      exportConversation();
    }

    // Enter submits modal confirm when modal is open and focus is on an input
    if(e.key === "Enter" && !e.shiftKey) {
      const modal = document.getElementById("modal-overlay");
      if(modal && modal.classList.contains("open")) {
        const active = document.activeElement?.tagName;
        if(active === "INPUT") {
          e.preventDefault();
          document.getElementById("modal-confirm-btn")?.click();
        }
      }
    }
  });
}

// Handle pywebview ready
if(document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}

