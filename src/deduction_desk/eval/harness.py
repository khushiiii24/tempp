"""Scoring one run against ground truth: money, accuracy, harm, operations.

Ground truth is read here and nowhere in the agent. This module is the grader.

The harm section is the one that distinguishes this from a submission that reports only
what it caught. A **false chase** is a contact sent about a deduction that ground truth
says was legitimate — a letter to a customer who did nothing wrong, usually about a
statutory withholding they were legally obliged to make. It costs money, it costs goodwill,
and it is invisible to every recovery metric. Blanket dunning generates them by the
hundred; that contrast is the argument.

Recovery is measured against the **reachable ceiling**, not against total short-paid.
Quoting it against everything deducted counts statutory TDS as a missed opportunity and
flatters every policy equally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlmodel import Session, select

from ..money import format_inr
from ..schemas import (
    Case,
    CaseState,
    ContactLog,
    Deduction,
    DeductionTruth,
    HumanApproval,
    Outbox,
)


@dataclass
class Scorecard:
    policy: str
    money: dict[str, Any]
    accuracy: dict[str, Any]
    harm: dict[str, Any]
    operations: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "money": self.money,
            "accuracy": self.accuracy,
            "harm": self.harm,
            "operations": self.operations,
        }


def score_run(
    session: Session,
    *,
    run_id: str,
    policy_name: str,
) -> Scorecard:
    """Grade one run. Reads ground truth; the agent never does."""
    cases = [c for c in session.exec(select(Case)).all()]
    contacts = [c for c in session.exec(select(ContactLog)).all()]
    approvals = [a for a in session.exec(select(HumanApproval)).all() if a.run_id == run_id]
    outbox = [o for o in session.exec(select(Outbox)).all() if o.run_id == run_id]
    deductions = {d.id: d for d in session.exec(select(Deduction)).all()}
    truths = {t.deduction_id: t for t in session.exec(select(DeductionTruth)).all()}

    truth_for_case = {
        c.id: truths.get(c.deduction_id) for c in cases if c.deduction_id in truths
    }

    # ---- money --------------------------------------------------------------------
    total_short = sum(int(d.amount_paise) for d in deductions.values())
    recoverable = sum(int(t.recoverable_paise) for t in truths.values())
    reachable = sum(
        int(t.recoverable_paise) for t in truths.values() if t.will_pay_if_chased
    )
    recovered = sum(int(c.recovered_paise) for c in cases)
    cost = sum(int(c.cost_incurred_paise) for c in cases)

    # Money correctly NOT chased: valid deductions closed as valid. A success, and one
    # that no recovery metric would ever show.
    correctly_closed = sum(
        int(deductions[c.deduction_id].amount_paise)
        for c in cases
        if c.state == CaseState.RESOLVED_CLOSED_VALID.value
        and (t := truth_for_case.get(c.id)) is not None
        and t.is_valid
    )

    # Recoverable money abandoned: written off despite being genuinely reachable.
    wrongly_written_off = sum(
        int(t.recoverable_paise)
        for c in cases
        if c.state == CaseState.RESOLVED_WRITTEN_OFF.value
        and (t := truth_for_case.get(c.id)) is not None
        and t.will_pay_if_chased
        and t.recoverable_paise > 0
    )

    # Reachable money the agent handed to a person rather than chasing itself.
    #
    # Counting this as zero, which the scoreboard did until now, scores every correct
    # hand-off as a failure. An abstention, a claim above the human-review threshold and a
    # buyer with no lawful contact channel are all cases the design *intends* to escalate;
    # the money is queued, not lost, and a blanket-dunning baseline that blindly chases the
    # same cases looks better purely because it never defers.
    #
    # It is reported as its own line rather than folded into "recovered", because the agent
    # did not recover it — a human still has to. Adding it to the recovery figure would be
    # claiming credit for work nobody has done yet.
    queued_for_human = sum(
        int(t.recoverable_paise)
        for c in cases
        if c.state == CaseState.HUMAN_QUEUE.value
        and (t := truth_for_case.get(c.id)) is not None
        and t.recoverable_paise > 0
    )
    queued_reachable = sum(
        int(t.recoverable_paise)
        for c in cases
        if c.state == CaseState.HUMAN_QUEUE.value
        and (t := truth_for_case.get(c.id)) is not None
        and t.will_pay_if_chased
        and t.recoverable_paise > 0
    )

    # ---- harm ---------------------------------------------------------------------
    # A false chase: a contact about a deduction ground truth says was legitimate.
    false_chase_contacts = 0
    false_chase_cases: set[str] = set()
    for contact in contacts:
        truth = truth_for_case.get(contact.case_id)
        if truth is not None and truth.is_valid:
            false_chase_contacts += 1
            false_chase_cases.add(contact.case_id)

    escalations = sum(
        1 for c in contacts if c.recipient_role in {"procurement", "account_manager"}
    )

    # ---- accuracy ------------------------------------------------------------------
    graded = [
        (deductions[c.deduction_id], truth_for_case[c.id])
        for c in cases
        if c.id in truth_for_case and truth_for_case[c.id] is not None
    ]
    correct_code = sum(1 for d, t in graded if d.predicted_code == t.true_reason_code)
    abstained = sum(1 for d, _ in graded if d.predicted_code == "NEEDS_HUMAN")
    answered = len(graded) - abstained
    verdict_right = sum(
        1 for d, t in graded if (int(d.recoverable_paise) > 0) == (int(t.recoverable_paise) > 0)
    )

    # ---- operations ------------------------------------------------------------------
    terminal = [c for c in cases if CaseState(c.state).is_terminal]
    auto_resolved = [
        c for c in terminal if c.state != CaseState.HUMAN_QUEUE.value and not c.awaiting_human
    ]
    days_to_resolution = [
        max(0, (int(c.closed_at.replace("-", "")) - int(c.opened_at.replace("-", ""))))
        for c in terminal
        if c.closed_at and c.opened_at
    ]

    return Scorecard(
        policy=policy_name,
        money={
            "total_short_paid_paise": total_short,
            "recoverable_paise": recoverable,
            "reachable_ceiling_paise": reachable,
            "recovered_paise": recovered,
            "recovery_rate_vs_ceiling": round(recovered / reachable, 4) if reachable else 0.0,
            "correctly_closed_valid_paise": correctly_closed,
            "wrongly_written_off_paise": wrongly_written_off,
            "queued_for_human_paise": queued_for_human,
            "queued_reachable_paise": queued_reachable,
            # What the agent settled itself PLUS what it correctly escalated. Not a
            # recovery claim — a measure of money the pipeline did not drop.
            "addressed_paise": recovered + queued_reachable,
            "cost_paise": cost,
            "net_recovery_paise": recovered - cost,
            "rupees_per_rupee_spent": round(recovered / cost, 2) if cost else None,
        },
        accuracy={
            "graded": len(graded),
            "answered": answered,
            "abstained": abstained,
            "abstention_rate": round(abstained / len(graded), 4) if graded else 0.0,
            "code_accuracy_answered": round(correct_code / answered, 4) if answered else 0.0,
            "verdict_accuracy": round(verdict_right / len(graded), 4) if graded else 0.0,
        },
        harm={
            "false_chase_contacts": false_chase_contacts,
            "false_chase_cases": len(false_chase_cases),
            "contacts_total": len(contacts),
            "contacts_per_rupee_recovered": (
                round(len(contacts) / (recovered / 100), 6) if recovered else None
            ),
            "escalations_to_senior_roles": escalations,
            "credit_holds_executed": sum(
                1 for a in approvals if a.action_type == "propose_credit_hold" and a.approved
            ),
            "credit_holds_proposed": sum(
                1 for a in approvals if a.action_type == "propose_credit_hold"
            ),
        },
        operations={
            "cases": len(cases),
            "auto_resolved": len(auto_resolved),
            "auto_resolved_rate": round(len(auto_resolved) / len(cases), 4) if cases else 0.0,
            "human_queue": sum(1 for c in cases if c.state == CaseState.HUMAN_QUEUE.value),
            "approvals_pending": sum(1 for a in approvals if a.approved is None),
            "messages_queued": len(outbox),
            "messages_sent_for_real": sum(1 for o in outbox if not o.dry_run),
            "mean_days_to_resolution": (
                round(sum(days_to_resolution) / len(days_to_resolution), 1)
                if days_to_resolution
                else None
            ),
        },
    )


def format_money_row(label: str, paise: int) -> str:
    return f"{label}: {format_inr(paise)}"
