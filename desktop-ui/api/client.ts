// desktop-ui/api/client.ts — typed fetch wrapper for the FastAPI sidecar.
//
// Caches the sidecar info (port + token) on first lookup; if the sidecar
// crashes and respawns with a new port, callers should re-call resetSidecarInfo()
// (the StatusBar does this when the 'ready' status arrives).
//
// All requests are http://127.0.0.1:<port> with a Bearer token.

import type { SidecarInfo } from "@/env";
import type { SafetySummary } from "@/types/safety";
import type { UsageGroupBy, UsageSummary } from "@/types/usage";

let cached: SidecarInfo | null = null;
let inflight: Promise<SidecarInfo> | null = null;

export function resetSidecarInfo(info: SidecarInfo | null): void {
  cached = info;
  inflight = null;
}

async function getSidecarInfo(): Promise<SidecarInfo> {
  if (cached) return cached;
  if (inflight) return inflight;

  inflight = (async () => {
    for (let attempt = 0; attempt < 60; attempt++) {
      const info = await window.electronAPI.getSidecarInfo();
      if (info) {
        cached = info;
        return info;
      }
      await new Promise((r) => setTimeout(r, 250));
    }
    throw new Error("Backend never reported ready. Check the status bar for details.");
  })();

  try {
    return await inflight;
  } finally {
    inflight = null;
  }
}

function baseUrl(info: SidecarInfo): string {
  return `http://127.0.0.1:${info.port}`;
}

export interface ApiError extends Error {
  status?: number;
  body?: unknown;
}

async function request<T>(
  method: "GET" | "POST" | "PUT" | "DELETE",
  path: string,
  body?: unknown,
  query?: Record<string, string | number | boolean | undefined>,
): Promise<T> {
  const info = await getSidecarInfo();
  const params = new URLSearchParams();
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== null) params.set(k, String(v));
    }
  }
  const qs = params.toString() ? `?${params.toString()}` : "";
  const url = `${baseUrl(info)}${path}${qs}`;

  const init: RequestInit = {
    method,
    headers: {
      Authorization: `Bearer ${info.token}`,
      ...(body != null ? { "Content-Type": "application/json" } : {}),
    },
  };
  if (body != null) init.body = JSON.stringify(body);

  let resp: Response;
  try {
    resp = await fetch(url, init);
  } catch (err) {
    const e: ApiError = new Error(
      err instanceof Error ? err.message : "Network request failed",
    );
    throw e;
  }

  if (!resp.ok) {
    let parsed: unknown = null;
    try {
      parsed = await resp.json();
    } catch {
      /* not JSON */
    }
    const e: ApiError = new Error(
      typeof parsed === "object" && parsed && "error" in parsed
        ? String((parsed as { error: unknown }).error)
        : `Request failed with ${resp.status}`,
    );
    e.status = resp.status;
    e.body = parsed;
    throw e;
  }

  if (resp.status === 204) return undefined as unknown as T;
  return (await resp.json()) as T;
}

export const api = {
  get: <T>(path: string, query?: Record<string, string | number | boolean | undefined>) =>
    request<T>("GET", path, undefined, query),
  post: <T>(path: string, body?: unknown) => request<T>("POST", path, body),
  put: <T>(path: string, body?: unknown) => request<T>("PUT", path, body),
  delete: <T>(path: string) => request<T>("DELETE", path),
};

// Multipart helper for the attachment upload endpoint. The standard
// `request` wrapper sets Content-Type: application/json which would
// confuse FastAPI's multipart parser; this version lets the browser
// pick the boundary for us.
async function postMultipart<T>(path: string, form: FormData): Promise<T> {
  const info = await getSidecarInfo();
  const url = `${baseUrl(info)}${path}`;
  let resp: Response;
  try {
    resp = await fetch(url, {
      method: "POST",
      headers: { Authorization: `Bearer ${info.token}` },
      body: form,
    });
  } catch (err) {
    const e: ApiError = new Error(
      err instanceof Error ? err.message : "Network request failed",
    );
    throw e;
  }
  if (!resp.ok) {
    let parsed: unknown = null;
    try {
      parsed = await resp.json();
    } catch {
      /* not JSON */
    }
    const detail =
      typeof parsed === "object" && parsed && "detail" in parsed
        ? String((parsed as { detail: unknown }).detail)
        : `Request failed with ${resp.status}`;
    const e: ApiError = new Error(detail);
    e.status = resp.status;
    e.body = parsed;
    throw e;
  }
  return (await resp.json()) as T;
}

// ── High-level helpers (one entry per route, typed for the renderer) ──────

export interface SettingsPayload {
  lm_studio_url: string;
  ollama_url: string;
  claude_api_key: string;
  claude_api_key_set: boolean;
  claude_model: string;
  claude_prompt_caching: boolean;
  default_local_backend: string;
  default_local_model: string;
  system_prompt: string;
  start_tab: string;
  routing_enabled: boolean;
  smart_routing_enabled: boolean;
  interleaved_reasoning_enabled: boolean;
  firewall_enabled: boolean;
  is_first_run: boolean;
  first_run_complete: boolean;
  max_conversation_budget_usd: number | null;
  budget_warning_threshold_pct: number | null;
  // ── Power Mode (v3) ─────────────────────────────────────────────────────
  power_mode_enabled: boolean;
  power_mode_workspace: string;
  power_mode_model_provider: string;
  power_mode_model_name: string;
  power_mode_api_key: string;
  power_mode_api_key_set: boolean;
  power_mode_autostart: boolean;
  power_mode_gateway_port: number;
  auto_update_enabled: boolean;
  // PR 17: voice input (Whisper.cpp) + voice output (Piper)
  voice_input_enabled: boolean;
  voice_output_enabled: boolean;
  stt_model_id: string;
  tts_voice_id: string;
}

export const Settings = {
  get: () => api.get<SettingsPayload>("/api/settings"),
  save: (key: string, value: unknown) =>
    api.post<{ ok: true }>("/api/settings/save", { key, value }),
  set: (key: string, value: unknown) =>
    api.post<{ ok: true }>("/api/settings/set", { key, value }),
  completeFirstRun: (start_tab: string) =>
    api.post<{ ok: true }>("/api/settings/complete_first_run", { start_tab }),
  verifyApiKey: (key: string) =>
    api.post<{ ok: boolean; message: string }>("/api/settings/verify_api_key", { key }),
  detectLocal: () => api.get<unknown>("/api/settings/detect_local"),
  getModelPrices: () => api.get<Record<string, { input: number; output: number }>>(
    "/api/settings/model_prices",
  ),
  setModelPrices: (prices: Record<string, [number, number] | { input: number; output: number }>) =>
    api.post<{ ok: true; prices: unknown }>("/api/settings/model_prices", { prices }),
};

// ── Models catalog (single source of truth, served from backend/config/models.json)
export interface ModelCatalogEntry {
  id:                       string;
  family:                   "opus" | "sonnet" | "haiku";
  display_name:             string;
  input_price_per_mtok:     number;
  output_price_per_mtok:    number;
  context_window_tokens:    number;
  vision:                   boolean;
  available_via:            string[];
}

export interface ModelCatalogResponse {
  default_claude_id: string;
  models:            ModelCatalogEntry[];
}

export const Models = {
  catalog: () => api.get<ModelCatalogResponse>("/api/models/catalog"),
};

// Plain-text GET helper. The export routes return raw markdown / JSON /
// HTML — not the JSON envelope `request<T>()` parses — so this skips the
// JSON.parse step and returns the response body verbatim.
async function fetchText(path: string): Promise<string> {
  const info = await getSidecarInfo();
  const url = `${baseUrl(info)}${path}`;
  let resp: Response;
  try {
    resp = await fetch(url, {
      method: "GET",
      headers: { Authorization: `Bearer ${info.token}` },
    });
  } catch (err) {
    const e: ApiError = new Error(
      err instanceof Error ? err.message : "Network request failed",
    );
    throw e;
  }
  if (!resp.ok) {
    const e: ApiError = new Error(`Request failed with ${resp.status}`);
    e.status = resp.status;
    throw e;
  }
  return resp.text();
}

export type ConversationExportFormat = "md" | "json" | "pdf-html";

export interface SearchResult {
  message_id: string;
  conversation_id: string;
  conversation_title: string;
  role: "user" | "assistant" | "system";
  snippet: string;
  created_at: string;
  rank: number;
}

export const Chat = {
  send: (conversation_id: string, user_message: string, agent_id = "") =>
    api.post<{ ok: true }>("/api/chat/send", { conversation_id, user_message, agent_id }),
  stop: () => api.post<{ ok: true }>("/api/chat/stop"),
  newConversation: (agent_id = "", title = "New conversation") =>
    api.post<{ id: string }>("/api/chat/new_conversation", { agent_id, title }),
  list: (limit = 30) => api.get<unknown[]>("/api/chat/conversations", { limit }),
  messages: (conversation_id: string, limit = 100) =>
    api.get<unknown[]>(`/api/chat/messages/${encodeURIComponent(conversation_id)}`, { limit }),
  rename: (conversation_id: string, title: string) =>
    api.post<{ ok: true }>("/api/chat/rename_conversation", { conversation_id, title }),
  delete: (conversation_id: string) =>
    api.post<{ ok: true }>(`/api/chat/delete_conversation/${encodeURIComponent(conversation_id)}`),
  branch: (conversation_id: string, from_message_id: string) =>
    api.post<unknown>("/api/chat/branch_conversation", { conversation_id, from_message_id }),
  export: (conversation_id: string, fmt = "markdown") =>
    api.post<unknown>("/api/chat/export_conversation", { conversation_id, fmt }),
  exportConversation: (id: string, format: ConversationExportFormat) =>
    fetchText(
      `/api/chat/conversations/${encodeURIComponent(id)}/export.${format}`,
    ),
  // PR 13: cross-conversation FTS5 search. Returns up to ``limit`` matches
  // ordered by bm25 rank. ``days`` scopes results to the last N days.
  searchConversations: (q: string, limit = 50, days?: number) =>
    api.get<SearchResult[]>("/api/conversations/search", {
      q,
      limit,
      ...(days != null ? { days } : {}),
    }),
  tokenStats: () => api.get<unknown>("/api/chat/token_stats"),
  routerStats: () => api.get<unknown>("/api/chat/router_stats"),
};

export const Agents = {
  list: () => api.get<unknown[]>("/api/agents"),
  get: (id: string) => api.get<unknown>(`/api/agents/${encodeURIComponent(id)}`),
  create: (input: {
    name: string;
    description: string;
    system_prompt: string;
    model_preference?: string;
    temperature?: number;
    max_tokens?: number;
  }) => api.post<unknown>("/api/agents/create", input),
  update: (agent_id: string, fields: Record<string, unknown>) =>
    api.post<unknown>("/api/agents/update", { agent_id, fields }),
  delete: (agent_id: string) =>
    api.post<unknown>(`/api/agents/delete/${encodeURIComponent(agent_id)}`),
  duplicate: (agent_id: string, new_name: string) =>
    api.post<unknown>("/api/agents/duplicate", { agent_id, new_name }),
};

export interface PendingWriteRow {
  id: string;
  conversation_id: string | null;
  write_type: string;
  content: string;
  contradicts_id: string | null;
  contradicts_content: string | null;
  proposed_at: string;
  decision: string | null;
  decided_at: string | null;
}

export const Memory = {
  save: (content: string, category = "fact") =>
    api.post<unknown>("/api/memory/save", { content, category }),
  searchMemories: (query: string, top_k = 5) =>
    api.post<unknown[]>("/api/memory/search_memories", { query, top_k }),
  searchDocuments: (query: string, top_k = 10, doc_type = "") =>
    api.post<unknown[]>("/api/memory/search_documents", { query, top_k, doc_type }),
  semanticAvailable: () => api.get<{ available: boolean }>("/api/memory/semantic_available"),
  pending: (limit = 100) =>
    api.get<PendingWriteRow[]>("/api/memory/pending", { limit }),
  approvePending: (id: string) =>
    api.post<{ ok: boolean; decision?: string; error?: string }>(
      `/api/memory/pending/${encodeURIComponent(id)}/approve`,
    ),
  denyPending: (id: string) =>
    api.post<{ ok: boolean; decision?: string; error?: string }>(
      `/api/memory/pending/${encodeURIComponent(id)}/deny`,
    ),
};

export interface Attachment {
  id: string;
  conversation_id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  persist: boolean;
  rag_doc_id: string | null;
  created_at: string;
}

export interface AttachmentUploadResult {
  id: string;
  filename: string;
  size_bytes: number;
  persist: boolean;
  extract_chars: number;
}

async function fetchBlob(path: string): Promise<Blob> {
  const info = await getSidecarInfo();
  const url = `${baseUrl(info)}${path}`;
  let resp: Response;
  try {
    resp = await fetch(url, {
      method: "GET",
      headers: { Authorization: `Bearer ${info.token}` },
    });
  } catch (err) {
    const e: ApiError = new Error(
      err instanceof Error ? err.message : "Network request failed",
    );
    throw e;
  }
  if (!resp.ok) {
    const e: ApiError = new Error(`Request failed with ${resp.status}`);
    e.status = resp.status;
    throw e;
  }
  return resp.blob();
}

export const Attachments = {
  upload: (
    conversationId: string,
    file: File,
    persist: boolean,
  ): Promise<AttachmentUploadResult> => {
    const form = new FormData();
    form.append("file", file);
    form.append("persist", persist ? "true" : "false");
    return postMultipart<AttachmentUploadResult>(
      `/api/chat/${encodeURIComponent(conversationId)}/attach`,
      form,
    );
  },
  list: (conversationId: string): Promise<Attachment[]> =>
    api.get<Attachment[]>(
      `/api/chat/${encodeURIComponent(conversationId)}/attachments`,
    ),
  delete: (id: string): Promise<{ ok: boolean }> =>
    api.delete<{ ok: boolean }>(
      `/api/chat/attachments/${encodeURIComponent(id)}`,
    ),
  // PR 11: image chip rehydration after conversation reload. The raw
  // File object only exists at upload time; subsequent renders fetch
  // the bytes back through the bearer-authenticated endpoint and wrap
  // them in an object URL.
  fetchBlob: (id: string): Promise<Blob> =>
    fetchBlob(`/api/chat/attachments/${encodeURIComponent(id)}/blob`),
};

export const Rag = {
  indexFolder: (folder_path: string) =>
    api.post<unknown>("/api/rag/index_folder", { folder_path }),
  addFile: (file_path: string) => api.post<unknown>("/api/rag/add_file", { file_path }),
  addText: (text: string, source = "manual") =>
    api.post<unknown>("/api/rag/add_text", { text, source }),
  clear: () => api.post<unknown>("/api/rag/clear"),
  status: () => api.get<unknown>("/api/rag/status"),
  search: (query: string, top_k = 5) =>
    api.post<unknown[]>("/api/rag/search", { query, top_k }),
  searchHybrid: (query: string, top_k = 5, method = "hybrid", doc_type = "") =>
    api.post<unknown[]>("/api/rag/search_hybrid", { query, top_k, method, doc_type }),
};

export interface McpServerSummary {
  server_id: string;
  name: string;
  version?: string;
  tool_count: number;
  enabled: boolean;
  env_keys: string[];
  env_set?: Record<string, boolean>;
  tools?: Array<{
    name: string;
    description?: string;
    skill_tags?: string[];
    scopes?: string[];
  }>;
}

export interface McpListResponse {
  servers: McpServerSummary[];
  root: string;
}

export const Mcp = {
  list: () => api.get<McpListResponse>("/api/mcp/servers"),
  install: (folder_path: string, overwrite = false) =>
    api.post<unknown>("/api/mcp/install", { folder_path, overwrite }),
  remove: (server_id: string) =>
    api.post<unknown>(`/api/mcp/remove/${encodeURIComponent(server_id)}`),
  setEnabled: (server_id: string, enabled: boolean) =>
    api.post<unknown>("/api/mcp/enabled", { server_id, enabled }),
  setSecret: (server_id: string, key: string, value: string) =>
    api.post<unknown>("/api/mcp/secrets/set", { server_id, key, value }),
  clearSecret: (server_id: string, key: string) =>
    api.post<unknown>("/api/mcp/secrets/clear", { server_id, key }),
  refresh: () => api.post<unknown>("/api/mcp/refresh"),
};

export const Prompts = {
  list: () => api.get<unknown[]>("/api/prompts"),
  versions: (id: string) => api.get<unknown[]>(`/api/prompts/${encodeURIComponent(id)}/versions`),
  save: (prompt_id: string, text: string, notes = "") =>
    api.post<unknown>("/api/prompts/save", { prompt_id, text, notes }),
  create: (input: {
    name: string;
    category: string;
    description: string;
    text: string;
    model_target?: string;
  }) => api.post<unknown>("/api/prompts/create", input),
};

// ── PR 18: User-saved prompt templates ─────────────────────────────────────

export type PromptTemplateKind = "snippet" | "system_prompt";

export interface PromptTemplate {
  id: string;
  title: string;
  body: string;
  kind: PromptTemplateKind;
  tags: string;
  created_at: string;
  updated_at: string;
  use_count: number;
}

export interface PromptTemplateCreatePayload {
  title: string;
  body: string;
  kind: PromptTemplateKind;
  tags?: string;
}

export interface PromptTemplateUpdatePayload {
  title?: string;
  body?: string;
  kind?: PromptTemplateKind;
  tags?: string;
}

export const PromptTemplates = {
  list: (): Promise<PromptTemplate[]> =>
    api.get<PromptTemplate[]>("/api/prompt-templates"),
  get: (id: string): Promise<PromptTemplate> =>
    api.get<PromptTemplate>(`/api/prompt-templates/${encodeURIComponent(id)}`),
  create: (payload: PromptTemplateCreatePayload): Promise<PromptTemplate> =>
    api.post<PromptTemplate>("/api/prompt-templates", payload),
  update: (
    id: string,
    payload: PromptTemplateUpdatePayload,
  ): Promise<PromptTemplate> =>
    api.put<PromptTemplate>(
      `/api/prompt-templates/${encodeURIComponent(id)}`,
      payload,
    ),
  delete: (id: string): Promise<{ ok: boolean }> =>
    api.delete<{ ok: boolean }>(
      `/api/prompt-templates/${encodeURIComponent(id)}`,
    ),
  use: (id: string): Promise<PromptTemplate> =>
    api.post<PromptTemplate>(
      `/api/prompt-templates/${encodeURIComponent(id)}/use`,
    ),
};

// Convenience aliases that match the function-name shape called out in the
// PR spec. Re-exported so callers can use either ``PromptTemplates.list()``
// or ``listPromptTemplates()`` interchangeably.
export const listPromptTemplates = PromptTemplates.list;
export const getPromptTemplate = PromptTemplates.get;
export const createPromptTemplate = PromptTemplates.create;
export const updatePromptTemplate = PromptTemplates.update;
export const deletePromptTemplate = PromptTemplates.delete;
export const usePromptTemplate = PromptTemplates.use;

export interface LocalModelRow {
  id: string;
  size_bytes: number | null;
  context_length: number | null;
  quantization: string | null;
  backend: "ollama" | "lm_studio";
  loaded: boolean;
}

export interface LocalModelsResponse {
  models: LocalModelRow[];
  current: string;
}

export const System = {
  serviceStatus: () => api.get<Record<string, { ok: boolean; error?: string | null }>>(
    "/api/system/service_status",
  ),
  listLocalModels: () => api.get<LocalModelsResponse>("/api/system/local_models"),
  setActiveLocalModel: (model_id: string) =>
    api.post<{ current: string; ok: boolean }>(
      "/api/system/local_model/active",
      { model_id },
    ),
  probeHardware: () => api.post<{ ok: true }>("/api/system/probe_hardware"),
  testConnection: (backend: "ollama" | "lmstudio") =>
    api.post<{ ok: true }>("/api/system/test_connection", { backend }),
  fetchModels: (backend: "ollama" | "lmstudio") =>
    api.post<{ ok: true }>("/api/system/fetch_chat_models", { backend }),
  runHealthCheck: (skip_api = false) =>
    api.post<{ ok: true }>("/api/system/run_health_check", { skip_api }),
  errorLogs: (limit = 50) => api.get<unknown[]>("/api/system/error_logs", { limit }),
  changelog: () => api.get<unknown>("/api/system/changelog"),
  changelogSeen: () => api.post<unknown>("/api/system/changelog/seen"),
  exportDiagnostics: () => api.post<{ ok: true }>("/api/system/export_diagnostics"),
  securityStatus: () => api.get<unknown>("/api/system/security/status"),
  toggleFirewall: (enabled: boolean) =>
    api.post<unknown>("/api/system/security/firewall", { enabled }),
  scanLog: (limit = 50, verdict_filter = "") =>
    api.get<unknown[]>("/api/system/security/scan_log", { limit, verdict_filter }),
  openUrl: (url: string) => api.post<unknown>("/api/system/open_url", { url }),
  resetCanary: (model_id: string) =>
    api.post<{ ok: boolean; deleted?: number; error?: string }>(
      `/api/system/canary/reset/${encodeURIComponent(model_id)}`,
    ),

  // ── Phase 9: Bundled llama.cpp server ──────────────────────────────────
  bundledDownload: (model_id?: string) =>
    api.post<{ ok: boolean; model_id?: string; error?: string }>(
      "/api/system/bundled/download",
      { model_id: model_id ?? "" },
    ),
  bundledStart: (model_id?: string) =>
    api.post<{ ok: boolean; port?: number; model_id?: string; error?: string }>(
      "/api/system/bundled/start",
      { model_id: model_id ?? "" },
    ),
  bundledStop: () =>
    api.post<{ ok: boolean; error?: string }>("/api/system/bundled/stop"),
  bundledStatus: () =>
    api.get<{
      running: boolean;
      port: number | null;
      model_id: string | null;
      available: boolean;
    }>("/api/system/bundled/status"),
};

export const Lifecycle = {
  audit: (limit = 100) =>
    api.get<{ events: unknown[]; path: string }>("/api/lifecycle/audit", { limit }),
  confirm: (token: string) => api.post<unknown>("/api/lifecycle/confirm", { token }),
  deny: (token: string) => api.post<unknown>("/api/lifecycle/deny", { token }),
};

export interface EscalationRow {
  id: string;
  conversation_id: string;
  triggered_at: string;
  trigger_type: string;
  trigger_detail: string;
  model_input?: string;
  proposed_action?: string | null;
  decision?: string | null;
  decided_at?: string | null;
}

export const Escalation = {
  pending: () => api.get<EscalationRow[]>("/api/escalation/pending"),
  approve: (id: string) =>
    api.post<{ ok: boolean; decision?: string; error?: string }>(
      `/api/escalation/${encodeURIComponent(id)}/approve`,
    ),
  deny: (id: string) =>
    api.post<{ ok: boolean; decision?: string; error?: string }>(
      `/api/escalation/${encodeURIComponent(id)}/deny`,
    ),
};

export const Echo = {
  reverse: (text: string) => api.post<{ text: string; reversed: string }>("/api/echo", { text }),
};

export const Safety = {
  getSafetySummary: (days: number = 30) =>
    api.get<SafetySummary>("/api/safety/summary", { days }),
};

export const Usage = {
  getUsageSummary: (days: number = 30, groupBy: UsageGroupBy = "day") =>
    api.get<UsageSummary>("/api/usage/summary", { days, group_by: groupBy }),
};

// ── Power Mode (v3) — Docker / OpenClaw delegation ────────────────────────

export interface DockerStatus {
  wsl_installed: boolean;
  docker_installed: boolean;
  docker_running: boolean;
  openclaw_running: boolean;
  openclaw_healthy: boolean;
  gpu_available: boolean;
  platform: string;
  detail: string;
  last_error: string;
  gateway_url: string;
  workspace_dir: string;
}

export interface ClassifyResult {
  route: "chat" | "execution";
  confidence: number;
  reasoning: string;
  source: string;
}

// ── PR 17: Voice (Whisper.cpp STT + Piper TTS) ─────────────────────────────

export interface VoiceAssetsStatus {
  stt_ready: boolean;
  tts_ready: boolean;
  stt_model_id: string;
  tts_voice_id: string;
}

export interface VoiceAssetsDownloadResult {
  ok: boolean;
  stt_model_id?: string;
  tts_voice_id?: string;
  error?: string;
}

async function postMultipartJson<T>(path: string, form: FormData): Promise<T> {
  return postMultipart<T>(path, form);
}

async function fetchBlobPost(path: string, body: unknown): Promise<Blob> {
  const info = await getSidecarInfo();
  const url = `${baseUrl(info)}${path}`;
  let resp: Response;
  try {
    resp = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${info.token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
  } catch (err) {
    const e: ApiError = new Error(
      err instanceof Error ? err.message : "Network request failed",
    );
    throw e;
  }
  if (!resp.ok) {
    let parsed: unknown = null;
    try {
      parsed = await resp.json();
    } catch {
      /* not JSON */
    }
    const detail =
      typeof parsed === "object" && parsed && "detail" in parsed
        ? String((parsed as { detail: unknown }).detail)
        : `Request failed with ${resp.status}`;
    const e: ApiError = new Error(detail);
    e.status = resp.status;
    e.body = parsed;
    throw e;
  }
  return resp.blob();
}

export const Voice = {
  transcribe: (audio: Blob, filename = "clip.wav"): Promise<{ text: string }> => {
    const form = new FormData();
    form.append("file", audio, filename);
    return postMultipartJson<{ text: string }>("/api/voice/transcribe", form);
  },
  synthesize: (text: string, voice_id = ""): Promise<Blob> =>
    fetchBlobPost("/api/voice/synthesize", { text, voice_id }),
  assetsStatus: (): Promise<VoiceAssetsStatus> =>
    api.get<VoiceAssetsStatus>("/api/voice/assets/status"),
  assetsDownload: (
    stt_model_id?: string,
    tts_voice_id?: string,
  ): Promise<VoiceAssetsDownloadResult> =>
    api.post<VoiceAssetsDownloadResult>("/api/voice/assets/download", {
      stt_model_id: stt_model_id ?? "",
      tts_voice_id: tts_voice_id ?? "",
    }),
};

export const Docker = {
  status: () => api.get<DockerStatus>("/api/docker/status"),
  start: () => api.post<{ ok: boolean; gateway_url?: string; detail?: string }>(
    "/api/docker/start",
  ),
  stop: () => api.post<{ ok: boolean; detail?: string }>("/api/docker/stop"),
  restart: () => api.post<{ ok: boolean; gateway_url?: string }>("/api/docker/restart"),
  health: () => api.get<{ ok: boolean; gateway_url: string }>("/api/docker/health"),
  classify: (user_message: string, conversation_id = "") =>
    api.post<ClassifyResult>("/api/docker/classify", { user_message, conversation_id }),
  execute: (conversation_id: string, user_message: string) =>
    api.post<{ ok: boolean; task_id?: string; error?: string }>(
      "/api/docker/execute",
      { conversation_id, user_message },
    ),
  cancel: (task_id: string) =>
    api.post<{ ok: boolean }>("/api/docker/cancel", { task_id }),
  approve: (task_id: string, approval_id: string, allow: boolean) =>
    api.post<{ ok: boolean }>("/api/docker/approve", {
      task_id,
      approval_id,
      allow,
    }),
};
