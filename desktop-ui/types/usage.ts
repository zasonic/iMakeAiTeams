// desktop-ui/types/usage.ts — types for the read-only Usage panel.
//
// Mirrors the backend payload from GET /api/usage/summary. Keep this in
// lock-step with backend/routes/usage.py — when a section gains a new
// field on the server, add it here too so the renderer can read it.

export type UsageGroupBy = "day" | "model" | "agent";

export interface UsageTotal {
  input_tokens:  number;
  output_tokens: number;
  cost_usd:      number;
  turns:         number;
}

export interface UsageRow {
  key:           string;
  input_tokens:  number;
  output_tokens: number;
  cost_usd:      number;
  turns:         number;
}

export interface UsageByModel {
  model:    string;
  cost_usd: number;
  turns:    number;
}

export interface UsageByAgent {
  agent_id: string;
  cost_usd: number;
  turns:    number;
}

export interface UsageSummary {
  window_days: number;
  group_by:    UsageGroupBy;
  total:       UsageTotal;
  rows:        UsageRow[];
  by_model:    UsageByModel[];
  by_agent:    UsageByAgent[];
}
