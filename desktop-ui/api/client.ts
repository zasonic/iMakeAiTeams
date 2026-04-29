// desktop-ui/api/client.ts — typed fetch wrapper for the FastAPI sidecar.
//
// Caches the sidecar info (port + token) on first lookup; if the sidecar
// crashes and respawns with a new port, callers should re-call resetSidecarInfo()
// (the StatusBar does this when the 'ready' status arrives).
//
// All requests are http://127.0.0.1:<port> with a Bearer token.

import type { SidecarInfo } from "@/env";

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
  method: "GET" | "POST",
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
};

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

export const Memory = {
  save: (content: string, category = "fact") =>
    api.post<unknown>("/api/memory/save", { content, category }),
  searchMemories: (query: string, top_k = 5) =>
    api.post<unknown[]>("/api/memory/search_memories", { query, top_k }),
  searchDocuments: (query: string, top_k = 10, doc_type = "") =>
    api.post<unknown[]>("/api/memory/search_documents", { query, top_k, doc_type }),
  semanticAvailable: () => api.get<{ available: boolean }>("/api/memory/semantic_available"),
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

export const System = {
  serviceStatus: () => api.get<Record<string, { ok: boolean; error?: string | null }>>(
    "/api/system/service_status",
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
};

export const Lifecycle = {
  audit: (limit = 100) =>
    api.get<{ events: unknown[]; path: string }>("/api/lifecycle/audit", { limit }),
  confirm: (token: string) => api.post<unknown>("/api/lifecycle/confirm", { token }),
  deny: (token: string) => api.post<unknown>("/api/lifecycle/deny", { token }),
};

export const Echo = {
  reverse: (text: string) => api.post<{ text: string; reversed: string }>("/api/echo", { text }),
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
