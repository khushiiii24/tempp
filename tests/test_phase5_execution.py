"""Phase 5 acceptance: the bounded action set, compliance, audit and replay.

The claims this file makes checkable:

* nothing sends for real
* `propose_credit_hold` never executes
* the compliance gate blocks what it should, and an **independent** auditor confirms it
* every case is reconstructable from the decision log with every other table dropped

That last one is the strongest form of the audit claim and the reason `replay.py` reads
only `decision_log` — a replay that consulted the invoice table would still produce a
convincing trace after the log had lost something.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlmodel import Session, select

from deduction_desk.actions.drafts import make_draft, render_template
from deduction_desk.actions.validator import validate_draft
from deduction_desk.audit.log import DecisionRecorder, hash_inputs
from deduction_desk.audit.replay import render_trace, replay_case
from deduction_desk.clock import at_time, parse_date, to_iso
from deduction_desk.config import load_policy
from deduction_desk.db import init_db, make_engine
from deduction_desk.eval.compliance_audit import audit
from deduction_desk.policy.compliance import check_contact
from deduction_desk.schemas import (
    ActionType,
    Buyer,
    Case,
    CaseState,
    ContactLog,
    HumanApproval,
    Outbox,
)


@pytest.fixture(scope="module")
def policy():
    return load_policy()


def _buyer(**overrides) -> Buyer:
    base = dict(
        id="BUY-0001", name="Test Traders Pvt Ltd", gstin="27AAACT1234K1Z5",
        pan="AAACT1234K", segment="midmarket", payment_behaviour_tag="average",
        credit_limit_paise=10_000_000, relationship_value_paise=100_000_000,
        preferred_channel="email", contact_email="ap@test.example",
        contact_phone="+919800000000", ap_manager_email="apm@test.example",
        procurement_email="proc@test.example", account_manager_email="km@ours.example",
        consent_whatsapp=False, dnd=False,
    )
    base.update(overrides)
    return Buyer(**base)


def _case(**overrides) -> Case:
    base = dict(
        id="CASE-0001-0", deduction_id="DED-0001-0", buyer_id="BUY-0001",
        state=CaseState.NEW.value, opened_at="2026-07-01",
    )
    base.update(overrides)
    return Case(**base)


# ======================================================================================
# Compliance gate
# ======================================================================================


def test_contact_outside_the_window_is_blocked(policy) -> None:
    when = at_time(parse_date("2026-07-01"), 22, 0)  # 10pm
    decision = check_contact(
        buyer=_buyer(), case=_case(), channel="email", when=when,
        buyer_history=[], policy=policy,
    )
    assert not decision.allowed
    assert "compliance.outside_contact_window" in decision.violations


def test_contact_at_the_weekend_is_blocked(policy) -> None:
    saturday = parse_date("2026-07-04")
    assert saturday.weekday() == 5
    decision = check_contact(
        buyer=_buyer(), case=_case(), channel="email",
        when=at_time(saturday, 11, 0), buyer_history=[], policy=policy,
    )
    assert not decision.allowed
    assert "compliance.outside_contact_days" in decision.violations


def test_whatsapp_without_consent_is_blocked(policy) -> None:
    decision = check_contact(
        buyer=_buyer(consent_whatsapp=False), case=_case(), channel="whatsapp",
        when=at_time(parse_date("2026-07-01"), 11, 0), buyer_history=[], policy=policy,
    )
    assert not decision.allowed
    assert "compliance.no_consent_for_whatsapp" in decision.violations


def test_dnd_buyer_is_never_contacted(policy) -> None:
    decision = check_contact(
        buyer=_buyer(dnd=True), case=_case(), channel="email",
        when=at_time(parse_date("2026-07-01"), 11, 0), buyer_history=[], policy=policy,
    )
    assert not decision.allowed
    assert "compliance.dnd_registered" in decision.violations


def test_contact_cap_per_case_is_enforced(policy) -> None:
    cap = int(policy.compliance["max_contacts_per_case"])
    decision = check_contact(
        buyer=_buyer(), case=_case(contacts_used=cap), channel="email",
        when=at_time(parse_date("2026-07-01"), 11, 0), buyer_history=[], policy=policy,
    )
    assert not decision.allowed
    assert "compliance.max_contacts_per_case_reached" in decision.violations


def test_minimum_gap_between_contacts_is_enforced(policy) -> None:
    yesterday = at_time(parse_date("2026-06-30"), 11, 0)
    decision = check_contact(
        buyer=_buyer(),
        case=_case(contacts_used=1, last_contact_at=to_iso(yesterday)),
        channel="email",
        when=at_time(parse_date("2026-07-01"), 11, 0),
        buyer_history=[], policy=policy,
    )
    assert not decision.allowed
    assert "compliance.min_gap_not_elapsed" in decision.violations


# ======================================================================================
# Draft validator
# ======================================================================================


def test_static_templates_all_pass_their_own_validator(policy) -> None:
    """A template edited to add a forbidden phrase would otherwise sail straight through."""
    slots = {
        "buyer_name": "Test Traders", "invoice_no": "INV/2026/0042",
        "amount": "Rs 12,500", "case_ref": "CASE-0001-0",
        "evidence": "debit note", "section": "194C", "deduction_reason": "FREIGHT",
        "contact_role": "your accounts payable team",
    }
    for template_id in (
        "chase_1", "chase_2", "chase_3", "chase_4", "chase_escalated",
        "request_document", "request_tds_certificate",
    ):
        subject, body = render_template(template_id, slots)
        result = validate_draft(
            subject=subject, body=body, policy=policy,
            case_amount_paise=1_250_000, invoice_total_paise=1_250_000,
            escalation_authorised=True,
        )
        assert result.ok, f"{template_id} fails validation: {result.rejections}"


def test_threatening_language_is_rejected(policy) -> None:
    result = validate_draft(
        subject="Payment overdue",
        body="Pay immediately or else we will take legal action. Case CASE-1.",
        policy=policy, case_amount_paise=1_250_000,
    )
    assert not result.ok
    assert any("forbidden_phrase" in r for r in result.rejections)


def test_an_invented_amount_is_rejected(policy) -> None:
    """A wrong figure in a demand letter is authoritative-looking and wrong."""
    result = validate_draft(
        subject="Query on INV/2026/0042",
        body="Rs 99,999 remains outstanding. Case CASE-1.",
        policy=policy, case_amount_paise=1_250_000, invoice_total_paise=1_250_000,
    )
    assert not result.ok
    assert any("unverified_amount" in r for r in result.rejections)


def test_a_payment_link_is_rejected_when_policy_forbids_it(policy) -> None:
    result = validate_draft(
        subject="Payment", body="Pay at https://pay.example/abc. Case CASE-1.",
        policy=policy, case_amount_paise=1_250_000, allow_payment_link=False,
    )
    assert not result.ok
    assert "payment_link_not_authorised" in result.rejections


def test_unauthorised_escalation_language_is_rejected(policy) -> None:
    result = validate_draft(
        subject="Final notice", body="This is our final demand. Case CASE-1.",
        policy=policy, case_amount_paise=1_250_000, escalation_authorised=False,
    )
    assert not result.ok


def test_a_rejected_llm_draft_falls_back_to_the_template(policy) -> None:
    """The system cannot emit an unchecked message even if the model misbehaves."""

    class BadModel:
        def complete_detailed(self, *_args, **_kwargs):
            from deduction_desk.llm.client import LLMResponse
            from deduction_desk.llm.schema import DraftMessage

            return LLMResponse(
                value=DraftMessage(
                    subject="FINAL NOTICE",
                    body="We will take legal action and place you on blacklist. Rs 88,888 due.",
                ),
                task="draft", provider="fake", model="fake", key="k",
                prompt_sha256="h", cached=False,
            )

    draft = make_draft(
        template_id="chase_1",
        slots={
            "buyer_name": "Test", "invoice_no": "INV/2026/0042", "amount": "Rs 12,500",
            "case_ref": "CASE-0001-0", "evidence": "x", "section": "194C",
            "deduction_reason": "FREIGHT", "contact_role": "AP",
        },
        policy=policy, case_amount_paise=1_250_000, invoice_total_paise=1_250_000,
        llm_client=BadModel(),
    )

    assert draft.rejected
    assert draft.drafted_by == "template_after_rejection"
    assert draft.validation.rejections
    assert "legal action" not in draft.body.lower()


# ======================================================================================
# Audit log and replay
# ======================================================================================


def test_decision_log_is_append_only(tmp_path) -> None:
    """Enforced by a database trigger, not by convention."""
    from deduction_desk.schemas import DecisionLog

    engine = init_db(tmp_path / "audit.db", reset=True)
    with Session(engine) as session:
        session.add(
            DecisionLog(
                id="DL-1", run_id="R", seq=1, ts="2026-07-01T09:30:00+05:30",
                sim_date="2026-07-01", stage="decide", inputs_hash="x",
            )
        )
        session.commit()

    with pytest.raises(Exception, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("UPDATE decision_log SET stage='tampered'"))


def test_inputs_hash_changes_when_the_observation_changes() -> None:
    a = hash_inputs({"verdict": "invalid", "recoverable_paise": 1000})
    b = hash_inputs({"verdict": "invalid", "recoverable_paise": 1001})
    c = hash_inputs({"recoverable_paise": 1000, "verdict": "invalid"})
    assert a != b
    assert a == c, "key order must not change the hash"


def test_replay_reconstructs_a_case_from_the_log_alone(tmp_path) -> None:
    """The strongest form of the audit claim: every other table is dropped first."""

    engine = init_db(tmp_path / "replay.db", reset=True)
    with Session(engine) as session:
        recorder = DecisionRecorder(session=session, run_id="R")
        recorder.record(
            stage="decide", ts="2026-07-01T09:30:00+05:30", sim_date="2026-07-01",
            case_id="CASE-X", deduction_id="DED-X",
            observation={"verdict": "invalid", "recoverable_paise": 250000},
            policy_rules_fired=["action.chase"],
            decision={"type": "chase", "reason": "verified recoverable"},
            action_taken={"executed": True, "detail": "chase to ap_clerk via email"},
        )
        recorder.record(
            stage="inbound", ts="2026-07-06T10:00:00+05:30", sim_date="2026-07-06",
            case_id="CASE-X", deduction_id="DED-X",
            observation={"kind": "payment", "amount_paise": 250000},
            outcome={"case_state": "resolved_recovered"},
        )
        recorder.flush()

    # Drop everything except the decision log.
    with engine.begin() as conn:
        for table in ("case", "contact_log", "deduction", "invoice", "buyer",
                      "payment_event", "outbox", "deduction_truth", "contract"):
            conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))

    with Session(engine) as session:
        result = replay_case(session, "CASE-X")

    assert result.found
    assert len(result.steps) == 2
    assert "action.chase" in result.rules_fired
    assert result.contacts_made == 1
    assert result.final_state == "resolved_recovered"

    trace = render_trace(result)
    assert "CASE-X" in trace and "action.chase" in trace


# ======================================================================================
# Independent compliance audit
# ======================================================================================


def test_auditor_catches_a_violation_the_gate_would_have_blocked(policy) -> None:
    """The auditor must be able to fail, or its clean verdict means nothing."""
    late = to_iso(at_time(parse_date("2026-07-01"), 23, 30))
    contacts = [
        ContactLog(
            id="CT-1", case_id="CASE-1", buyer_id="BUY-0001", ts=late,
            channel="email", recipient_role="ap_clerk", recipient_address="ap@x",
            template_id="chase_1", subject="Query", body="Case CASE-1.",
        )
    ]
    report = audit(
        contacts, policy=policy,
        buyer_consent={"BUY-0001": False}, buyer_dnd={"BUY-0001": False},
    )
    assert not report.clean
    assert "contact_outside_window" in report.by_rule()


def test_auditor_catches_a_forbidden_phrase_in_a_sent_message(policy) -> None:
    """Something the pre-execution gate never sees, because it happens after."""
    ok_time = to_iso(at_time(parse_date("2026-07-01"), 11, 0))
    contacts = [
        ContactLog(
            id="CT-1", case_id="CASE-1", buyer_id="BUY-0001", ts=ok_time,
            channel="email", recipient_role="ap_clerk", recipient_address="ap@x",
            template_id="chase_1", subject="Overdue",
            body="We will take legal action. Case CASE-1.",
        )
    ]
    report = audit(
        contacts, policy=policy,
        buyer_consent={"BUY-0001": False}, buyer_dnd={"BUY-0001": False},
    )
    assert "forbidden_phrase_in_message" in report.by_rule()


def test_auditor_does_not_import_the_gate_it_checks() -> None:
    """If the same code enforced and verified, 'zero violations' would be a tautology."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "deduction_desk" / "eval" / "compliance_audit.py"
    ).read_text(encoding="utf-8")

    # Check the IMPORT lines, not the prose. The module docstring names
    # `policy.compliance` precisely to explain why it does not import it, and a naive
    # substring search over the whole file flags that explanation as the violation.
    import_lines = [
        line for line in source.splitlines()
        if line.startswith(("import ", "from ")) or line.strip().startswith(("import ", "from "))
    ]
    joined = "\n".join(import_lines)

    assert "policy" not in joined, f"auditor imports from policy: {joined}"
    assert "check_contact" not in joined


def test_credit_hold_marked_approved_inside_a_run_is_a_violation(policy) -> None:
    approvals = [
        HumanApproval(
            id="HA-1", run_id="R", case_id="CASE-1", requested_at="2026-07-01",
            action_type="propose_credit_hold", amount_paise=500_000,
            reason="ladder exhausted", approved=True,
        )
    ]
    report = audit(
        [], policy=policy, buyer_consent={}, buyer_dnd={}, approvals=approvals
    )
    assert "credit_hold_executed_without_human" in report.by_rule()


# ======================================================================================
# Nothing sends
# ======================================================================================


def test_every_outbox_message_is_dry_run() -> None:
    """`dry_run=True` by construction. Only `--live` plus an env var can change it, and
    neither is set anywhere in the demo path."""
    with Session(make_engine()) as session:
        messages = list(session.exec(select(Outbox)).all())

    if not messages:
        pytest.skip("no run yet; execute `python -m deduction_desk run` first")

    for message in messages:
        assert message.dry_run is True, f"{message.id} is not dry-run"
        assert message.sent_at is None, f"{message.id} claims to have been sent"


def test_no_credit_hold_is_ever_executed() -> None:
    """The strongest lever always waits for a person, regardless of ladder exhaustion."""
    with Session(make_engine()) as session:
        approvals = list(session.exec(select(HumanApproval)).all())

    for approval in approvals:
        if approval.action_type == ActionType.PROPOSE_CREDIT_HOLD.value:
            assert approval.approved is not True, (
                f"{approval.id} was auto-approved; credit holds must never self-execute"
            )
