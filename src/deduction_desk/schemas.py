"""The data model. SQLModel tables over SQLite, mirrored by the enums the pipeline uses.

Three conventions that hold everywhere:

* **All money is an integer number of paise.** Field names end in `_paise` so a grep can
  police it. See `money.py`.
* **All timestamps are ISO-8601 strings carrying a +05:30 offset**, not native datetimes.
  SQLite silently drops tzinfo on round-trip, and a timezone-naive AR system will compute
  the contact window wrong for exactly the hour either side of the boundary — which is
  precisely where a compliance violation would hide. Strings survive round-trips
  unchanged, sort correctly at a fixed offset, and hash deterministically. `clock.py`
  owns all conversion.
* **`DeductionTruth` is quarantined.** It is what the agent is graded against, so no
  module under ingest/, matching/, classify/, verify/, policy/ or actions/ may import it.
  `tests/test_no_truth_leakage.py` enforces that by grepping the source.
"""

from __future__ import annotations

from enum import StrEnum

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

# ==================================================================================
# Enums
# ==================================================================================


class Segment(StrEnum):
    ENTERPRISE = "enterprise"
    MIDMARKET = "midmarket"
    SME = "sme"


class BehaviourTag(StrEnum):
    PROMPT = "prompt"
    AVERAGE = "average"
    SLOW = "slow"
    DIFFICULT = "difficult"


class DeliveryTerms(StrEnum):
    FOR_DESTINATION = "for_destination"  # seller bears freight
    EX_WORKS = "ex_works"  # buyer bears freight


class Channel(StrEnum):
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    SMS = "sms"
    CALL = "call"


class Role(StrEnum):
    AP_CLERK = "ap_clerk"
    AP_MANAGER = "ap_manager"
    PROCUREMENT = "procurement"
    ACCOUNT_MANAGER = "account_manager"


class AdviceFormat(StrEnum):
    EMAIL = "email"
    PDF_TEXT = "pdf_text"
    XLSX = "xlsx"
    NONE = "none"


class MatchMethod(StrEnum):
    EXACT = "exact"
    NORMALISED = "normalised"
    ADVICE = "advice"
    FUZZY = "fuzzy"
    SUBSET_SUM = "subset_sum"
    LLM = "llm"
    MANUAL = "manual"


class Verdict(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    PROVISIONAL_VALID = "provisional_valid"  # 26AS lag: believe it, but re-check later


class CaseState(StrEnum):
    NEW = "new"
    AWAITING_SETTLEMENT = "awaiting_settlement"  # may yet be a split payment, not a short one
    VERIFYING = "verifying"
    CHASING = "chasing"
    ESCALATED = "escalated"
    PROMISED = "promised"
    RESOLVED_RECOVERED = "resolved_recovered"
    RESOLVED_WRITTEN_OFF = "resolved_written_off"
    RESOLVED_CREDIT_NOTE = "resolved_credit_note"
    RESOLVED_CLOSED_VALID = "resolved_closed_valid"
    HUMAN_QUEUE = "human_queue"
    STOPPED = "stopped"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATES


_TERMINAL_STATES = frozenset(
    {
        CaseState.RESOLVED_RECOVERED,
        CaseState.RESOLVED_WRITTEN_OFF,
        CaseState.RESOLVED_CREDIT_NOTE,
        CaseState.RESOLVED_CLOSED_VALID,
        CaseState.HUMAN_QUEUE,
        CaseState.STOPPED,
    }
)


class ActionType(StrEnum):
    """The complete bounded action set. Nothing outside this may ever be executed."""

    NO_ACTION = "no_action"
    CLOSE_VALID = "close_valid"
    WRITE_OFF = "write_off"
    ISSUE_CREDIT_NOTE = "issue_credit_note"
    REQUEST_DOCUMENT = "request_document"
    CHASE = "chase"
    ESCALATE_ROLE = "escalate_role"
    RECORD_PROMISE_TO_PAY = "record_promise_to_pay"
    PROPOSE_CREDIT_HOLD = "propose_credit_hold"  # never auto-executes
    ROUTE_TO_HUMAN = "route_to_human"


class InboundKind(StrEnum):
    PAYMENT = "payment"
    DEFLECTION = "deflection"
    DISPUTE = "dispute"
    PROMISE = "promise"
    DOCUMENT = "document"
    OPT_OUT = "opt_out"
    SILENCE = "silence"


# ==================================================================================
# Core entities
# ==================================================================================


class Buyer(SQLModel, table=True):
    __tablename__ = "buyer"

    id: str = Field(primary_key=True)
    name: str
    gstin: str
    pan: str
    segment: str
    payment_behaviour_tag: str
    credit_limit_paise: int
    relationship_value_paise: int
    preferred_channel: str
    contact_email: str
    contact_phone: str
    ap_manager_email: str
    procurement_email: str
    account_manager_email: str
    consent_whatsapp: bool = False
    dnd: bool = False


class Contract(SQLModel, table=True):
    __tablename__ = "contract"

    id: str = Field(primary_key=True)
    buyer_id: str = Field(index=True)
    delivery_terms: str
    payment_terms_days: int
    early_payment_discount_bp: int = 0
    early_payment_window_days: int = 0
    tds_section_expected: str
    # The section alone is not enough to adjudicate a rate mismatch. Several sections have
    # more than one legitimate rate — 194J is 10% for professional fees and 2% for
    # technical services — so "expected section 194J" does not tell you whether a 5%
    # deduction is an over-deduction or an under-deduction. A real vendor master pins the
    # applicable rate, and so does this.
    tds_rate_expected_bp: int = 200
    tcs_applicable: bool = False
    freight_borne_by: str  # "seller" | "buyer" — derived from delivery_terms, stored for clarity
    rate_card: dict = Field(default_factory=dict, sa_column=Column(JSON))


class Invoice(SQLModel, table=True):
    __tablename__ = "invoice"

    id: str = Field(primary_key=True)
    invoice_no: str = Field(index=True)
    buyer_id: str = Field(index=True)
    issue_date: str
    due_date: str
    taxable_paise: int
    gst_paise: int
    tcs_paise: int
    total_paise: int
    line_items: list = Field(default_factory=list, sa_column=Column(JSON))
    status: str = "open"


class PaymentEvent(SQLModel, table=True):
    __tablename__ = "payment_event"

    id: str = Field(primary_key=True)
    utr: str = Field(index=True)
    value_date: str
    amount_paise: int
    narration_raw: str
    source: str = "bank_statement"
    buyer_id_resolved: str | None = Field(default=None, index=True)
    # Set when the counterparty simulation emits a recovery payment, so the eval can tell
    # money the agent recovered apart from money that was always going to arrive.
    recovery_for_case_id: str | None = Field(default=None, index=True)


class RemittanceAdvice(SQLModel, table=True):
    __tablename__ = "remittance_advice"

    id: str = Field(primary_key=True)
    buyer_id: str = Field(index=True)
    received_at: str
    format: str
    raw_text: str
    parsed: dict | None = Field(default=None, sa_column=Column(JSON))
    links_to_payment: str | None = Field(default=None, index=True)


class Allocation(SQLModel, table=True):
    __tablename__ = "allocation"

    id: str = Field(primary_key=True)
    payment_event_id: str = Field(index=True)
    invoice_id: str = Field(index=True)
    allocated_paise: int
    method: str
    confidence: float = 1.0


class Deduction(SQLModel, table=True):
    __tablename__ = "deduction"

    id: str = Field(primary_key=True)
    invoice_id: str = Field(index=True)
    payment_event_id: str = Field(index=True)
    amount_paise: int
    claimed_reason_text: str | None = None

    predicted_code: str = "NEEDS_HUMAN"
    predicted_confidence: float = 0.0
    predicted_rationale: str = ""
    predicted_by: str = "none"  # "llm" | "stub" | "none"

    verification: dict = Field(default_factory=dict, sa_column=Column(JSON))
    verdict: str = Verdict.UNKNOWN.value
    recoverable_paise: int = 0
    state: str = "new"
    created_at: str = ""


class DeductionTruth(SQLModel, table=True):
    """Pre-committed ground truth. The agent NEVER sees this table.

    Written by the generator before the agent runs, read only by the counterparty
    simulation (which needs to know whether the money arrives) and by the eval harness
    (which needs to know whether the agent was right). This quarantine is the entire
    reason the scoreboard is defensible: the agent cannot influence its own grade.
    """

    __tablename__ = "deduction_truth"

    deduction_id: str = Field(primary_key=True)
    true_reason_code: str
    is_valid: bool
    recoverable_paise: int
    will_pay_if_chased: bool
    pays_after_n_contacts: int  # 1..4, or 99 = never
    responds_to_channels: list = Field(default_factory=list, sa_column=Column(JSON))
    responds_only_at_role: str
    will_dispute: bool = False
    promise_then_default: bool = False
    opt_out: bool = False
    latency_days: int = 3
    showcase_id: str | None = None
    notes: str = ""


class Case(SQLModel, table=True):
    __tablename__ = "case"

    id: str = Field(primary_key=True)
    deduction_id: str = Field(index=True)
    buyer_id: str = Field(index=True)
    state: str = CaseState.NEW.value
    contacts_used: int = 0
    current_role: str = Role.AP_CLERK.value
    documents_requested: int = 0
    cost_incurred_paise: int = 0
    opened_at: str = ""
    closed_at: str | None = None
    recovered_paise: int = 0
    written_off_paise: int = 0
    credit_note_paise: int = 0
    stop_reason: str | None = None
    last_contact_at: str | None = None
    promise_due_date: str | None = None
    awaiting_human: bool = False
    human_reason: str | None = None


class ContactLog(SQLModel, table=True):
    __tablename__ = "contact_log"

    id: str = Field(primary_key=True)
    case_id: str = Field(index=True)
    buyer_id: str = Field(index=True)
    ts: str = Field(index=True)
    channel: str
    recipient_role: str
    recipient_address: str
    template_id: str
    subject: str = ""
    body: str = ""
    drafted_by: str = "template"  # "llm" | "template" (after validator rejection)
    validator_rejections: list = Field(default_factory=list, sa_column=Column(JSON))
    policy_checks_passed: list = Field(default_factory=list, sa_column=Column(JSON))
    cost_paise: int = 0
    response_received_at: str | None = None
    response_text: str | None = None
    response_kind: str | None = None


class DecisionLog(SQLModel, table=True):
    """Append-only. Nothing in this system may UPDATE or DELETE a row here.

    Must contain enough to reconstruct any case with zero access to live services —
    `replay` proves it by reading this table and nothing else.
    """

    __tablename__ = "decision_log"

    id: str = Field(primary_key=True)
    run_id: str = Field(index=True)
    seq: int = Field(index=True)
    ts: str
    sim_date: str
    case_id: str | None = Field(default=None, index=True)
    deduction_id: str | None = Field(default=None, index=True)
    stage: str
    inputs_hash: str
    observation: dict = Field(default_factory=dict, sa_column=Column(JSON))
    hypothesis: dict = Field(default_factory=dict, sa_column=Column(JSON))
    policy_rules_fired: list = Field(default_factory=list, sa_column=Column(JSON))
    decision: dict = Field(default_factory=dict, sa_column=Column(JSON))
    action_taken: dict | None = Field(default=None, sa_column=Column(JSON))
    outcome: dict | None = Field(default=None, sa_column=Column(JSON))
    llm_calls: list = Field(default_factory=list, sa_column=Column(JSON))
    human_approval: dict | None = Field(default=None, sa_column=Column(JSON))


class Outbox(SQLModel, table=True):
    __tablename__ = "outbox"

    id: str = Field(primary_key=True)
    run_id: str = Field(index=True)
    case_id: str = Field(index=True)
    channel: str
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))
    scheduled_for: str
    sent_at: str | None = None
    dry_run: bool = True


class InboundEvent(SQLModel, table=True):
    """A counterparty response scheduled by the simulation, delivered on a future tick."""

    __tablename__ = "inbound_event"

    id: str = Field(primary_key=True)
    run_id: str = Field(index=True)
    case_id: str = Field(index=True)
    deliver_on: str = Field(index=True)
    kind: str
    amount_paise: int = 0
    text: str = ""
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))
    delivered: bool = False


class HumanApproval(SQLModel, table=True):
    """The approval queue. `propose_credit_hold` lands here and stays here."""

    __tablename__ = "human_approval"

    id: str = Field(primary_key=True)
    run_id: str = Field(index=True)
    case_id: str = Field(index=True)
    requested_at: str
    action_type: str
    amount_paise: int
    reason: str
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))
    decided_at: str | None = None
    approved: bool | None = None


class Run(SQLModel, table=True):
    __tablename__ = "run"

    id: str = Field(primary_key=True)
    policy_name: str
    seed: int
    days: int
    started_at: str
    finished_at: str | None = None
    llm_backend: str = "stub"
    config_snapshot: dict = Field(default_factory=dict, sa_column=Column(JSON))
    stats: dict = Field(default_factory=dict, sa_column=Column(JSON))


class Exception_(SQLModel, table=True):
    """Everything the agent refused to resolve, with the reason. Published, not hidden."""

    __tablename__ = "exception_report"

    id: str = Field(primary_key=True)
    run_id: str = Field(default="", index=True)
    kind: str  # unmatched_payment | ambiguous_match | unparsed_advice | abstained | ...
    subject_id: str
    detail: str
    amount_paise: int = 0
    created_at: str = ""


# Tables the generator writes and the agent may read (i.e. the observable world).
OBSERVABLE_TABLES = (
    "buyer",
    "contract",
    "invoice",
    "payment_event",
    "remittance_advice",
)

# Tables the agent may never read.
QUARANTINED_TABLES = ("deduction_truth",)
