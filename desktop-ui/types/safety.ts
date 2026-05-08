// desktop-ui/types/safety.ts — types for the read-only Safety panel.
//
// Mirrors the backend payload from GET /api/safety/summary. Keep this in
// lock-step with backend/routes/safety.py — when a section gains a new
// field on the server, add it here too so the renderer can read it.

export interface SafetyEscalations {
  triggered: number;
  approved:  number;
  denied:    number;
  pending:   number;
}

export interface SafetyMemoryGate {
  facts_proposed: number;
  auto_accepted:  number;
  user_approved:  number;
  user_denied:    number;
  pending:        number;
}

export interface SafetyCanary {
  baselines:     number;
  alerts_fired:  number;
  last_alert_at: string | null;
}

export interface SafetyDenialReason {
  reason: string;
  count:  number;
}

export interface SafetyGovernance {
  tool_calls_total:   number;
  tool_calls_denied:  number;
  denial_top_reasons: SafetyDenialReason[];
}

export interface SafetyMastEntry {
  category: string;
  count:    number;
}

export interface SafetyRouting {
  turns_total:    number;
  turns_failed:   number;
  mast_breakdown: SafetyMastEntry[];
}

export interface SafetyVoting {
  high_stakes_turns:  number;
  consensus_reached:  number;
}

export interface SafetySummary {
  window_days: number;
  escalations: SafetyEscalations;
  memory_gate: SafetyMemoryGate;
  canary:      SafetyCanary;
  governance:  SafetyGovernance;
  routing:     SafetyRouting;
  voting:      SafetyVoting;
}
