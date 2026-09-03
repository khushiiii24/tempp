"""Phase 6 acceptance: baselines, scoring, and the scoreboard.

These tests do not assert that the agent wins. They assert that the comparison is *fair* —
same batch, same clock, same cost model, same counterparty, one decision function swapped —
and that the scorecard measures what it claims to. Whether the agent beats B1 is a result,
and a test that demanded it would be a test that could only be satisfied by tuning until it
passed.

What is asserted is the structural property that makes the result meaningful: **B1 must
produce false chases and the agent must not**, because B1 cannot tell a statutory
withholding from an invalid claim and the agent verifies before acting. If that ever
stopped being true, either the verification layer had broken or the harm metric had stopped
measuring anything.
"""

from __future__ import annotations

import pytest
from sqlmodel import Session, select

from deduction_desk.config import load_policy
from deduction_desk.db import make_engine
from deduction_desk.eval.baselines import BASELINES, b0_do_nothing, b1_blanket_dunning
from deduction_desk.eval.harness import Scorecard
from deduction_desk.eval.report import render_scoreboard
from deduction_desk.policy.engine import DecisionContext
from deduction_desk.schemas import ActionType, Buyer, Case, CaseState, Deduction, Verdict


@pytest.fixture(scope="module")
def policy():
    return load_policy()


def _ctx(**overrides) -> DecisionContext:
    deduction = Deduction(
        id="DED-1-0", invoice_id="INV-1", payment_event_id="PAY-1",
        amount_paise=overrides.pop("amount_paise", 1_000_000),
        predicted_code=overrides.pop("predicted_code", "FREIGHT"),
        predicted_confidence=overrides.pop("confidence", 0.9),
        verdict=overrides.pop("verdict", Verdict.INVALID.value),
        recoverable_paise=overrides.pop("recoverable_paise", 1_000_000),
    )
    buyer = Buyer(
        id="BUY-1", name="Test", gstin="g", pan="p", segment="midmarket",
        payment_behaviour_tag="average", credit_limit_paise=1, relationship_value_paise=10_000_000,
        preferred_channel="email", contact_email="a@b", contact_phone="+91",
        ap_manager_email="c@d", procurement_email="e@f", account_manager_email="g@h",
    )
    case = Case(
        id="CASE-1-0", deduction_id="DED-1-0", buyer_id="BUY-1",
        state=CaseState.NEW.value, opened_at="2026-07-01",
        contacts_used=overrides.pop("contacts_used", 0),
    )
    return DecisionContext(
        deduction=deduction, buyer=buyer, case=case,
        today=overrides.pop("today", "2026-07-20"),
        days_past_due=overrides.pop("days_past_due", 30),
        **overrides,
    )


# ======================================================================================
# Baselines behave as described
# ======================================================================================


def test_b0_never_chases_anything(policy) -> None:
    action = b0_do_nothing(_ctx(), policy)
    assert action.type == ActionType.WRITE_OFF
    assert action.channel is None


def test_b1_chases_a_valid_tds_deduction(policy) -> None:
    """The whole point of B1, and the source of the harm number.

    It has no classification and no verification, so a statutory withholding — money the
    buyer was legally obliged to keep — looks exactly like an invalid claim.
    """
    action = b1_blanket_dunning(
        _ctx(predicted_code="TDS_194C", verdict=Verdict.VALID.value, recoverable_paise=0),
        policy,
    )
    assert action.type == ActionType.CHASE, "B1 must chase indiscriminately; that is the baseline"


def test_the_agent_does_not_chase_a_verified_valid_deduction(policy) -> None:
    """The contrast that the scoreboard's harm row is built on."""
    from deduction_desk.policy.engine import decide

    action = decide(
        _ctx(predicted_code="TDS_194C", verdict=Verdict.VALID.value, recoverable_paise=0),
        policy,
    )
    assert action.type != ActionType.CHASE
    assert action.channel is None
    assert "stopping.verified_valid" in action.rules_fired


def test_agent_does_not_close_an_unverified_deduction_as_valid(policy) -> None:
    """'We never looked' must not be recorded as 'we checked and it was fine'.

    Testing `recoverable_paise <= 0` alone also matches every deduction that was never
    classified, which produced a perfect harm score and a 100% auto-resolution rate for an
    agent that had done no work at all.
    """
    from deduction_desk.policy.engine import decide

    action = decide(
        _ctx(verdict=Verdict.UNKNOWN.value, recoverable_paise=0, predicted_code="NEEDS_HUMAN",
             classification_abstained=True),
        policy,
    )
    assert action.type != ActionType.CLOSE_VALID
    assert action.next_state == CaseState.HUMAN_QUEUE.value


def test_b2_writes_off_below_its_threshold(policy) -> None:
    from deduction_desk.eval.baselines import b2_threshold

    action = b2_threshold(_ctx(amount_paise=100_000, recoverable_paise=100_000), policy)
    assert action.type == ActionType.WRITE_OFF


def test_b3_closes_on_the_taxonomy_default_without_checking_source_data(policy) -> None:
    """The ablation: it trusts the label and never opens the contract or 26AS."""
    from deduction_desk.eval.baselines import b3_classify_only

    action = b3_classify_only(_ctx(predicted_code="TDS_194C"), policy)
    assert action.type == ActionType.CLOSE_VALID
    assert "baseline.b3.taxonomy_default_valid" in action.rules_fired

    # FREIGHT's validity genuinely depends on the contract, which B3 never reads.
    freight = b3_classify_only(_ctx(predicted_code="FREIGHT"), policy)
    assert freight.type == ActionType.CHASE


def test_every_baseline_respects_the_settlement_grace_period(policy) -> None:
    """Otherwise their false-chase counts would be an artefact of the clock, not the policy."""
    for name, (fn, _label) in BASELINES.items():
        if name == "b0":
            continue  # B0 never contacts anyone, so the guard is moot
        action = fn(_ctx(days_past_due=1), policy)
        assert action.type != ActionType.CHASE, f"{name} chased inside the grace period"


def test_every_baseline_respects_the_contact_cap(policy) -> None:
    cap = int(policy.compliance["max_contacts_per_case"])
    for name, (fn, _label) in BASELINES.items():
        if name in {"b0"}:
            continue
        action = fn(_ctx(contacts_used=cap), policy)
        assert action.type != ActionType.CHASE, f"{name} exceeded the contact cap"


# ======================================================================================
# Scoreboard rendering
# ======================================================================================


def _card(name: str, **overrides) -> Scorecard:
    money = {
        "total_short_paid_paise": 12_203_318, "recoverable_paise": 4_732_790,
        "reachable_ceiling_paise": 2_249_826, "recovered_paise": 0,
        "recovery_rate_vs_ceiling": 0.0, "correctly_closed_valid_paise": 0,
        "wrongly_written_off_paise": 0, "cost_paise": 0,
        "net_recovery_paise": 0, "rupees_per_rupee_spent": None,
    }
    money.update(overrides.pop("money", {}))
    harm = {
        "false_chase_contacts": 0, "false_chase_cases": 0, "contacts_total": 0,
        "contacts_per_rupee_recovered": None, "escalations_to_senior_roles": 0,
        "credit_holds_executed": 0, "credit_holds_proposed": 0,
    }
    harm.update(overrides.pop("harm", {}))
    return Scorecard(
        policy=name, money=money, harm=harm,
        accuracy={"graded": 0, "answered": 0, "abstained": 0, "abstention_rate": 0.0,
                  "code_accuracy_answered": 0.0, "verdict_accuracy": 0.0},
        operations={"cases": 0, "auto_resolved": 0, "auto_resolved_rate": 0.0,
                    "human_queue": 0, "approvals_pending": 0, "messages_queued": 0,
                    "messages_sent_for_real": 0, "mean_days_to_resolution": None},
    )


def test_scoreboard_leads_with_net_recovery_then_harm() -> None:
    """A reader who stops after two sections must have seen the claim and its qualifier."""
    markdown = render_scoreboard([_card("agent"), _card("b1")])
    money_at = markdown.index("## Money")
    harm_at = markdown.index("## Harm")
    accuracy_at = markdown.index("## Accuracy")

    assert money_at < harm_at < accuracy_at
    assert "Net recovery" in markdown
    assert "False chases" in markdown


def test_scoreboard_states_that_nothing_was_sent() -> None:
    markdown = render_scoreboard([_card("agent")])
    assert "dry_run=True" in markdown
    assert "Messages actually sent" in markdown


def test_scoreboard_quotes_recovery_against_the_reachable_ceiling() -> None:
    """Quoting against total short-paid counts statutory TDS as a missed opportunity."""
    markdown = render_scoreboard([_card("agent")])
    assert "reachable ceiling" in markdown


def test_scoreboard_shows_compliance_violations_when_audited() -> None:
    markdown = render_scoreboard(
        [_card("agent")], compliance={"agent": {"violations": 0}}
    )
    assert "Compliance violations" in markdown


# ======================================================================================
# Run wiring
# ======================================================================================


def test_all_baselines_are_registered() -> None:
    assert set(BASELINES) == {"b0", "b1", "b2", "b3"}
    for fn, label in BASELINES.values():
        assert callable(fn)
        assert label


def test_scored_run_reports_money_harm_accuracy_and_operations() -> None:
    from deduction_desk.eval.harness import score_run
    from deduction_desk.schemas import Run

    with Session(make_engine()) as session:
        runs = list(session.exec(select(Run)).all())
        if not runs:
            pytest.skip("no run yet; execute `python -m deduction_desk run` first")
        card = score_run(session, run_id=runs[0].id, policy_name=runs[0].policy_name)

    for section in (card.money, card.harm, card.accuracy, card.operations):
        assert isinstance(section, dict) and section

    assert card.operations["messages_sent_for_real"] == 0


def test_the_counterparty_loop_actually_pays(tmp_path) -> None:
    """A chasing baseline must recover *something*, or the feedback loop is dead.

    This exists because it once was. `state.cases` was keyed by deduction id while
    `_deliver_inbound` looked cases up by case id, so every lookup silently returned None
    and no inbound event was ever applied — no payments, no disputes, no promises.

    Every policy then scored exactly ₹0, which reads as "the agent is conservative" rather
    than "the wiring is broken". What exposed it was a baseline behaving implausibly: B1
    sent 677 letters and recovered nothing, and no real dunning process is that bad.

    So the assertion is deliberately about a *baseline*, not the agent. The agent
    legitimately recovers nothing when its inputs are unclassified; B1 chases regardless of
    classification, so if it recovers nothing the machinery is broken.
    """
    # The decision log is append-only, so a fixed run id collides on re-run.
    import uuid

    from sqlmodel import Session as _S

    from deduction_desk.db import make_engine as _engine
    from deduction_desk.eval.baselines import b1_blanket_dunning
    from deduction_desk.runner import run_batch
    from deduction_desk.schemas import Deduction as _D

    run_id = f"test-b1-loop-{uuid.uuid4().hex[:8]}"

    with _S(_engine()) as session:
        if not session.exec(select(_D)).all():
            pytest.skip("no generated batch")

        stats = run_batch(
            session,
            run_id=run_id,
            policy_fn=b1_blanket_dunning,
            policy_name="b1",
            days=45,
            seed=42,
        )

    assert stats["contacts"] > 0, "B1 sent no contacts at all"
    assert stats["recovered_paise"] > 0, (
        f"B1 sent {stats['contacts']} contacts and recovered nothing — the counterparty "
        f"feedback loop is not delivering inbound events"
    )
