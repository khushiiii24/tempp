/**
 * The snapshot.
 *
 * Every figure the site renders is read from these files, which are written by
 * `python -m deduction_desk export-web` from the same SQLite tables the grader reads. There
 * are no illustrative numbers anywhere in this app and no hardcoded ones — a lesson that
 * cost us a whole regenerated PDF (BROKE entry 16), where funnel figures typed into a
 * diagram went stale the moment a double-payment bug was fixed underneath them.
 *
 * If a number needs to appear on screen, it gets added to the exporter first.
 */
import leakRaw from "../data/leak.json";
import pipelineRaw from "../data/pipeline.json";
import scoreboardRaw from "../data/scoreboard.json";
import guardrailsRaw from "../data/guardrails.json";
import taxonomyRaw from "../data/taxonomy.json";
import metaRaw from "../data/meta.json";
import caseIndexRaw from "../data/cases_index.json";
import { BRAND } from "./brand";

export interface Leak {
  invoices: number;
  invoice_value_paise: number;
  payments: number;
  deductions: number;
  short_paid_paise: number;
  leak_rate_bp: number;
  recoverable_paise: number;
  reachable_paise: number;
  valid_paise: number;
  by_family: { family: string; count: number; paise: number; recoverable_paise: number; valid_count: number }[];
  by_code: {
    code: string;
    label: string;
    family: string;
    chaseable: boolean;
    count: number;
    paise: number;
    recoverable_paise: number;
  }[];
  by_segment: { segment: string; count: number; paise: number }[];
  top_buyers: {
    id: string;
    name: string;
    segment: string;
    behaviour: string;
    count: number;
    paise: number;
    recoverable_paise: number;
  }[];
  histogram: { label: string; count: number; paise: number }[];
  largest_deduction: { paise: number; code: string; reason_text: string } | null;
  smallest_bucket_count: number;
  sample_advices: {
    id: string;
    format: string;
    received_at: string;
    raw_text: string;
    truncated: boolean;
  }[];
}

export interface FunnelStage {
  key: string;
  label: string;
  count: number;
  paise: number;
  note: string;
}

export interface Pipeline {
  funnel: FunnelStage[];
  curve: { day: number; recovered_paise: number; contacts: number; closed: number }[];
  matching: {
    payments: number;
    matched: number;
    match_rate: number;
    source: string;
    by_method: { method: string; count: number }[];
    exceptions: { kind: string; count: number }[];
    exceptions_total: number;
    invoices_with_allocations?: number;
    fully_settled?: number;
    with_shortfall?: number;
    awaiting_settlement?: number;
  };
  states: { state: string; count: number }[];
  abstained: number;
  outbox: number;
  outbox_sent_for_real: number;
  approvals_requested: number;
  approvals_granted: number;
}

export interface PolicyCard {
  policy: string;
  money: Record<string, number | null>;
  accuracy: Record<string, number>;
  harm: Record<string, number | null>;
  operations: Record<string, number | null>;
  compliance: { violations?: number; [k: string]: unknown };
  curve: { day: number; recovered_paise: number; contacts: number }[];
}

export interface Scoreboard {
  seed: number;
  days: number;
  policies: PolicyCard[];
}

export interface Guardrails {
  contact_window: { start: string; end: string };
  contact_days: string[];
  max_contacts_per_case: number;
  max_contacts_per_buyer_per_week: number;
  min_gap_hours: number;
  channel_ladder: string[];
  escalation_ladder: string[];
  forbidden_phrases: string[];
  require_consent_for: string[];
  stopping_rules: Record<string, boolean | number>;
  economics: Record<string, number | Record<string, number>>;
  confidence: Record<string, number>;
  settlement: Record<string, number | boolean>;
  verification: Record<string, number | boolean>;
}

export interface ReasonCode {
  code: string;
  label: string;
  family: string;
  verifier: string;
  chaseable: boolean;
  default_valid: boolean | null;
  default_action: string;
  expected_rate_bp: number | null;
  note: string;
}

export interface Meta {
  generated_at: string;
  run_id: string | null;
  seed: number;
  days: number;
  db_content_hash: string;
  cases: number;
  config_files: string[];
  scoreboard_present: boolean;
}

export interface CaseIndexRow {
  id: string;
  buyer: string;
  segment: string;
  amount_paise: number;
  code: string;
  label: string;
  family: string;
  confidence: number;
  verdict: string;
  state: string;
  contacts: number;
  recovered_paise: number;
  showcase_id: string | null;
}

export interface CaseContact {
  ts: string;
  channel: string;
  role: string;
  template_id: string;
  subject: string;
  body: string;
  drafted_by: string;
  checks: string[];
  rejections: string[];
  cost_paise: number;
  response_kind: string | null;
  response_text: string | null;
  response_at: string | null;
}

export interface TraceRow {
  seq: number;
  sim_date: string;
  stage: string;
  rules: string[];
  decision: Record<string, unknown>;
  action: Record<string, unknown>;
  outcome: Record<string, unknown>;
}

/**
 * The full record. It does not extend `CaseIndexRow` — the index flattens `contacts` to a
 * count for the list, while the full record carries the messages themselves, and pretending
 * one is a superset of the other only buys a name collision.
 */
export interface CaseRecord {
  id: string;
  state: string;
  recovered_paise: number;
  showcase_id?: string | null;
  deduction_id: string;
  stop_reason: string | null;
  opened_at: string;
  closed_at: string | null;
  contacts_used: number;
  current_role: string;
  cost_paise: number;
  written_off_paise: number;
  credit_note_paise: number;
  awaiting_human: boolean;
  human_reason: string | null;
  buyer: any;
  contract: {
    delivery_terms: string;
    payment_terms_days: number;
    tds_section_expected: string;
    tds_rate_expected_bp: number;
    tcs_applicable: boolean;
  };
  invoice: {
    no: string;
    issue_date: string;
    due_date: string;
    taxable_paise: number;
    gst_paise: number;
    tcs_paise: number;
    total_paise: number;
  };
  payment: { utr: string; value_date: string; amount_paise: number; narration: string };
  advice: { format: string; received_at: string; raw_text: string; truncated: boolean } | null;
  deduction: {
    amount_paise: number;
    claimed_reason_text: string | null;
    predicted_code: string;
    predicted_label: string;
    family: string;
    confidence: number;
    rationale: string;
    predicted_by: string;
    verdict: string;
    recoverable_paise: number;
    verification: {
      verdict?: string;
      recoverable_paise?: number;
      evidence?: Record<string, unknown>;
      rules_fired?: string[];
      recheck_after_days?: number | null;
      evidence_needed?: string[];
    };
  };
  contacts: CaseContact[];
  trace: TraceRow[];
  truth: {
    true_reason_code: string;
    is_valid: boolean;
    recoverable_paise: number;
    will_pay_if_chased: boolean;
    pays_after_n_contacts: number;
    responds_to_channels: string[];
    responds_only_at_role: string;
    will_dispute: boolean;
    promise_then_default: boolean;
    opt_out: boolean;
    showcase_id: string | null;
    notes: string;
    code_correct: boolean;
  } | null;
}

export const leak = leakRaw as Leak;
export const pipeline = pipelineRaw as unknown as Pipeline;
export const scoreboard = scoreboardRaw as unknown as Scoreboard;
export const guardrails = guardrailsRaw as unknown as Guardrails;
export const taxonomy = taxonomyRaw as ReasonCode[];
export const meta = metaRaw as Meta;
export const caseIndex = caseIndexRaw as CaseIndexRow[];

export const agent = scoreboard.policies.find((p) => p.policy === "agent")!;
export const baselines = scoreboard.policies.filter((p) => p.policy !== "agent");
export const byPolicy = Object.fromEntries(
  scoreboard.policies.map((p) => [p.policy, p]),
) as Record<string, PolicyCard>;

/** What each baseline actually is. The scoreboard is meaningless without this. */
export const POLICY_BLURB: Record<string, { name: string; what: string }> = {
  agent: {
    name: BRAND.name,
    what: "Classify it, check it against the source systems, then chase only what is actually owed.",
  },
  b0: { name: "Do nothing", what: "Write every shortfall off. What most finance teams do today." },
  b1: {
    name: "Write to everyone",
    what: "Chase every shortfall on a fixed ladder. No classification, no checking.",
  },
  b2: {
    name: "Only the big ones",
    what: "Chase anything above a rupee threshold. Still no checking.",
  },
  b3: {
    name: "Skip the checking",
    what: "The full agent with verification removed — the ablation that prices what checking is worth.",
  },
};

/** Loaded on demand: 2 MB of raw advice text, draft bodies and decision traces. */
export async function loadCases(): Promise<CaseRecord[]> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/cases.json`);
  if (!res.ok) throw new Error(`cases.json: ${res.status}`);
  return (await res.json()) as CaseRecord[];
}
