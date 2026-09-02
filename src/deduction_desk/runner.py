"""The tick loop: run the agent over a simulated 45 days.

One day at a time, and within a day, cases in sorted id order. Determinism is not decorative
here — the scoreboard is compared against baselines run over the same clock, and a loop
whose ordering drifted would produce differences that look like policy effects.

Each tick, per case:

1. **Deliver inbound events due today.** Payments, disputes, promises, opt-outs — all
   scheduled earlier by the counterparty state machine, which read pre-committed truth.
2. **Re-evaluate stopping rules**, before considering any action. Most are terminal, and
   the cheapest correct action is very often none.
3. **Execute at most one action.** One per case per day is what keeps the ladder a ladder;
   without the cap a case could burn its entire contact allowance in a single tick.
4. **Record the decision.** Every stage writes to the append-only log, whether or not it
   acted, because "the agent considered this case and declined" is the most interesting
   thing it does and it leaves no other trace.

The `--policy` flag selects the decision function, which is what lets the baselines
(§11) run through this identical loop. Baselines that had their own runner would differ
from the agent in ways nobody could account for, and the comparison would be worthless.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlmodel import Session, select

from .actions.executor import ExecutionResult, execute
from .audit.log import DecisionRecorder
from .clock import IST, SimClock, add_days, at_time, date_str, days_between, to_iso
from .config import Policy, load_policy
from .money import Paise
from .policy.compliance import next_role
from .policy.engine import Action, DecisionContext, decide, propose_credit_hold
from .schemas import (
    ActionType,
    Buyer,
    Case,
    CaseState,
    ContactLog,
    Deduction,
    DeductionTruth,
    HumanApproval,
    InboundEvent,
    InboundKind,
    Invoice,
    Outbox,
    PaymentEvent,
    Role,
    Run,
)
from .simulate.counterparty import respond
from .simulate.replies import render_reply, reply_slots

# A decision function: given the context and policy, choose one action.
PolicyFn = Callable[[DecisionContext, Policy], Action]


@dataclass
class RunState:
    """Everything mutated during a run, kept together so a tick is easy to reason about."""

    cases: dict[str, Case] = field(default_factory=dict)
    contacts: list[ContactLog] = field(default_factory=list)
    outbox: list[Outbox] = field(default_factory=list)
    approvals: list[HumanApproval] = field(default_factory=list)
    inbound: list[InboundEvent] = field(default_factory=list)
    recovery_payments: list[PaymentEvent] = field(default_factory=list)
    # Per-case flags the policy engine reads but that live outside the Case row.
    disputed: set[str] = field(default_factory=set)
    opted_out: set[str] = field(default_factory=set)
    contacts_by_buyer: dict[str, list[ContactLog]] = field(default_factory=dict)
    sequence: int = 0

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence


def _open_cases_for(
    session: Session,
    *,
    run_id: str,
    today: str,
    policy: Policy,
) -> dict[str, Case]:
    """Create one case per deduction that is ripe to act on.

    A case is opened only once its invoice is past due by the settlement grace period.
    Before that a shortfall may simply be the first half of a split payment, and opening a
    case would start a clock on money that was always going to arrive.
    """
    deductions = sorted(session.exec(select(Deduction)).all(), key=lambda d: d.id)
    invoices = {i.id: i for i in session.exec(select(Invoice)).all()}
    grace = int(policy.settlement["treat_as_short_after_days"])

    # Keyed by CASE id, not deduction id.
    #
    # These were keyed by deduction id while `_deliver_inbound` looked cases up by case id,
    # so every lookup silently returned None and **no inbound event was ever applied**.
    # The entire counterparty feedback loop was dead: no payments, no disputes, no
    # promises. Every policy recovered exactly ₹0, which reads as "the agent is
    # conservative" rather than "the wiring is broken", and B1 sending 677 contacts for
    # zero recovery is what finally made it obvious.
    cases: dict[str, Case] = {}
    for deduction in deductions:
        invoice = invoices.get(deduction.invoice_id)
        if invoice is None:
            continue
        case_id = f"CASE-{deduction.id.split('-', 1)[1]}"
        cases[case_id] = Case(
            id=case_id,
            deduction_id=deduction.id,
            buyer_id=invoice.buyer_id,
            state=(
                CaseState.NEW.value
                if days_between(invoice.due_date, today) >= grace
                else CaseState.AWAITING_SETTLEMENT.value
            ),
            current_role=Role.AP_CLERK.value,
            opened_at=today,
        )
    return cases


def _deliver_inbound(
    state: RunState,
    *,
    today: str,
    deduction_by_case: dict[str, Deduction],
    recorder: DecisionRecorder,
    run_id: str,
) -> None:
    """Apply everything the counterparty scheduled for today."""
    for event in sorted(state.inbound, key=lambda e: (e.deliver_on, e.id)):
        if event.delivered or event.deliver_on != today:
            continue
        event.delivered = True

        case = state.cases.get(event.case_id)
        if case is None:
            continue

        kind = InboundKind(event.kind)

        # A settled case stays settled. Two contacts made before the first payment lands
        # schedule two credits, and applying both accumulated `recovered_paise` twice —
        # four cases collected exactly double their true recoverable amount, which pushed
        # "addressed" above the reachable ceiling. Impossible arithmetic is the only reason
        # it was noticed; the inflated total on its own looked entirely plausible.
        if (
            kind == InboundKind.PAYMENT
            and case.state == CaseState.RESOLVED_RECOVERED.value
        ):
            continue

        if kind == InboundKind.PAYMENT:
            case.recovered_paise = Paise(int(case.recovered_paise) + int(event.amount_paise))
            case.state = CaseState.RESOLVED_RECOVERED.value
            case.closed_at = today
            state.recovery_payments.append(
                PaymentEvent(
                    # Scoped by the inbound event, not just the case. Two contacts made
                    # before the first payment lands schedule two credits, and keying on
                    # the case alone collides on the primary key.
                    id=f"REC-{run_id}-{event.id}",
                    utr=f"REC{abs(hash(case.id)) % 10**12:012d}",
                    value_date=today,
                    amount_paise=Paise(int(event.amount_paise)),
                    narration_raw=f"NEFT-RECOVERY-{case.buyer_id}-{case.id}",
                    source="bank_statement",
                    buyer_id_resolved=case.buyer_id,
                    recovery_for_case_id=case.id,
                )
            )
        elif kind == InboundKind.DISPUTE:
            state.disputed.add(case.id)
            case.state = CaseState.STOPPED.value
            case.stop_reason = "stopping.dispute_raised"
            case.closed_at = today
        elif kind == InboundKind.OPT_OUT:
            state.opted_out.add(case.id)
            case.state = CaseState.STOPPED.value
            case.stop_reason = "stopping.opt_out"
            case.closed_at = today
        elif kind == InboundKind.PROMISE:
            case.promise_due_date = str(event.payload.get("promise_date") or today)
            case.state = CaseState.PROMISED.value

        recorder.record(
            stage="inbound",
            ts=to_iso(at_time(SimClock.from_strings(today, 1).start, 10, 0)),
            sim_date=today,
            case_id=case.id,
            deduction_id=case.deduction_id,
            observation={"kind": event.kind, "amount_paise": int(event.amount_paise)},
            outcome={"case_state": case.state, "text": event.text[:200]},
        )


def _clear_prior_run_state(session: Session, run_id: str) -> None:
    """Remove state from any earlier run so this one starts clean.

    Cases and contacts are global rather than run-scoped, so they are cleared outright;
    outbox, approvals and inbound events are scoped by run id. Recovery payments are
    identified by `recovery_for_case_id` — they are agent-created credits, not part of the
    generated batch, so removing them cannot touch the source data.
    """
    from sqlmodel import delete

    session.exec(delete(ContactLog))
    session.exec(delete(Case))
    session.exec(delete(Outbox).where(Outbox.run_id == run_id))
    session.exec(delete(HumanApproval).where(HumanApproval.run_id == run_id))
    session.exec(delete(InboundEvent).where(InboundEvent.run_id == run_id))
    session.exec(
        delete(PaymentEvent).where(PaymentEvent.recovery_for_case_id.is_not(None))
    )
    for existing in session.exec(select(Run).where(Run.id == run_id)).all():
        session.delete(existing)
    session.commit()


def agent_policy(ctx: DecisionContext, policy: Policy) -> Action:
    """The full pipeline's decision function. Just the policy engine."""
    return decide(ctx, policy)


def run_batch(
    session: Session,
    *,
    run_id: str,
    policy_fn: PolicyFn = agent_policy,
    policy_name: str = "agent",
    days: int = 45,
    seed: int = 42,
    llm_client=None,
    razorpay_client=None,
    policy: Policy | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Run one policy over the simulated clock and return its summary.

    Idempotent: prior run state is cleared first, so calling this twice is safe and every
    caller gets the same behaviour. Leaving that to the CLI meant a direct call collided on
    `Case` primary keys, which is a footgun for exactly the tests most likely to exercise
    the loop. `decision_log` is exempt — it is append-only by design and scoped by run id.
    """
    policy = policy or load_policy()
    _clear_prior_run_state(session, run_id)

    deductions = {d.id: d for d in session.exec(select(Deduction)).all()}
    invoices = {i.id: i for i in session.exec(select(Invoice)).all()}
    buyers = {b.id: b for b in session.exec(select(Buyer)).all()}
    truths = {t.deduction_id: t for t in session.exec(select(DeductionTruth)).all()}

    # The clock starts the day after the last credit landed: the agent works the backlog
    # that the matching stage produced, it does not run alongside the payments arriving.
    payments = list(session.exec(select(PaymentEvent)).all())
    start = add_days(max((p.value_date for p in payments), default="2026-06-01"), 1)
    clock = SimClock.from_strings(start, days)

    state = RunState()
    state.cases = _open_cases_for(session, run_id=run_id, today=start, policy=policy)
    deduction_by_case = {c.id: deductions[c.deduction_id] for c in state.cases.values()}

    recorder = DecisionRecorder(session=session, run_id=run_id)
    escalation_ladder = list(policy.compliance["escalation_ladder"])

    for day in clock:
        today = date_str(day)

        _deliver_inbound(
            state,
            today=today,
            deduction_by_case=deduction_by_case,
            recorder=recorder,
            run_id=run_id,
        )

        for case_id in sorted(state.cases):
            case = state.cases[case_id]
            if CaseState(case.state).is_terminal:
                continue

            deduction = deductions[case.deduction_id]
            invoice = invoices[deduction.invoice_id]
            buyer = buyers[invoice.buyer_id]

            # A case parked awaiting settlement becomes actionable once the grace period
            # has passed with no further credit.
            grace = int(policy.settlement["treat_as_short_after_days"])
            past_due = days_between(invoice.due_date, today)
            if case.state == CaseState.AWAITING_SETTLEMENT.value and past_due >= grace:
                case.state = CaseState.NEW.value

            ctx = DecisionContext(
                deduction=deduction,
                buyer=buyer,
                case=case,
                today=today,
                verification=deduction.verification or {},
                classification_confidence=float(deduction.predicted_confidence),
                classification_abstained=deduction.predicted_code == "NEEDS_HUMAN",
                dispute_raised=case.id in state.disputed,
                opted_out=case.id in state.opted_out,
                human_takeover=case.awaiting_human,
                document_already_requested=case.documents_requested > 0,
                days_past_due=past_due,
            )

            action = policy_fn(ctx, policy)

            # Ladder exhausted on a real, chaseable amount: propose a credit hold, which
            # only ever queues for a human.
            if (
                action.type == ActionType.NO_ACTION
                and "stopping.max_contacts_reached" in action.rules_fired
                and int(deduction.recoverable_paise) > 0
            ):
                action = propose_credit_hold(ctx, policy)

            result = execute(
                action,
                case=case,
                deduction=deduction,
                invoice=invoice,
                buyer=buyer,
                policy=policy,
                sim_date=today,
                run_id=run_id,
                buyer_contacts=state.contacts_by_buyer.get(buyer.id, []),
                llm_client=llm_client,
                razorpay_client=razorpay_client,
                sequence=state.next_sequence(),
            )

            _apply(state, case, action, result, today=today)

            # A contact that actually went out gets a counterparty response scheduled.
            if result.contact is not None:
                truth = truths.get(case.deduction_id)
                if truth is not None:
                    reaction = respond(
                        truth=truth,
                        case=case,
                        contact=result.contact,
                        escalation_ladder=escalation_ladder,
                        seed=seed,
                    )
                    if not reaction.is_silence:
                        state.inbound.append(
                            InboundEvent(
                                id=f"IN-{run_id}-{case.id}-{state.next_sequence():03d}",
                                run_id=run_id,
                                case_id=case.id,
                                deliver_on=reaction.deliver_on,
                                kind=reaction.kind.value,
                                amount_paise=Paise(int(reaction.amount_paise)),
                                text=render_reply(
                                    kind=reaction.kind,
                                    text_key=reaction.text_key,
                                    seed=seed,
                                    case_id=case.id,
                                    slots=reply_slots(
                                        invoice_no=invoice.invoice_no,
                                        amount_paise=int(deduction.recoverable_paise),
                                        buyer_name=buyer.name,
                                        promise_date=str(
                                            reaction.payload.get("promise_date") or ""
                                        ),
                                        section=(deduction.predicted_code or "").replace(
                                            "TDS_", ""
                                        ),
                                    ),
                                ),
                                payload=dict(reaction.payload),
                            )
                        )

            recorder.record(
                stage="decide",
                ts=to_iso(at_time(day, 9, 30).astimezone(IST)),
                sim_date=today,
                case_id=case.id,
                deduction_id=case.deduction_id,
                observation={
                    "verdict": deduction.verdict,
                    "recoverable_paise": int(deduction.recoverable_paise),
                    "contacts_used": case.contacts_used,
                    "role": case.current_role,
                    "days_past_due": past_due,
                },
                hypothesis={
                    "predicted_code": deduction.predicted_code,
                    "confidence": float(deduction.predicted_confidence),
                },
                policy_rules_fired=action.rules_fired,
                decision=action.as_dict(),
                action_taken=result.as_dict(),
                human_approval=(
                    {"queued": True, "action": result.approval.action_type}
                    if result.approval
                    else None
                ),
                llm_calls=[result.draft.llm_call] if (result.draft and result.draft.llm_call) else [],
            )

    # ---- persist ------------------------------------------------------------------
    for row in (
        *state.cases.values(),
        *state.contacts,
        *state.outbox,
        *state.approvals,
        *state.inbound,
        *state.recovery_payments,
    ):
        session.add(row)

    stats = summarise(state, deductions, policy)
    session.add(
        Run(
            id=run_id,
            policy_name=policy_name,
            seed=seed,
            days=days,
            started_at=start,
            finished_at=date_str(clock.end),
            llm_backend="template" if llm_client is None else "llm",
            config_snapshot={"days": days, "policy": policy_name},
            stats=stats,
        )
    )
    recorder.flush()
    session.commit()
    return stats


def _apply(
    state: RunState,
    case: Case,
    action: Action,
    result: ExecutionResult,
    *,
    today: str,
) -> None:
    """Fold an execution result back into the case."""
    case.cost_incurred_paise = Paise(int(case.cost_incurred_paise) + int(result.cost_paise))

    if result.contact is not None:
        state.contacts.append(result.contact)
        state.contacts_by_buyer.setdefault(case.buyer_id, []).append(result.contact)
        case.contacts_used += 1
        case.last_contact_at = result.contact.ts

    if result.outbox is not None:
        state.outbox.append(result.outbox)

    if result.approval is not None:
        state.approvals.append(result.approval)
        case.awaiting_human = True
        case.human_reason = action.reason

    if action.type == ActionType.REQUEST_DOCUMENT and result.executed:
        case.documents_requested += 1

    if action.type == ActionType.ESCALATE_ROLE and result.executed:
        promoted = next_role(case.current_role, load_policy())
        if promoted:
            case.current_role = promoted

    if action.type == ActionType.WRITE_OFF and result.executed:
        case.written_off_paise = Paise(int(action.amount_paise))
    if action.type == ActionType.ISSUE_CREDIT_NOTE and result.executed:
        case.credit_note_paise = Paise(int(action.amount_paise))

    if result.next_state:
        case.state = result.next_state
        if CaseState(result.next_state).is_terminal and not case.closed_at:
            case.closed_at = today
            if not case.stop_reason and action.rules_fired:
                case.stop_reason = action.rules_fired[0]


def summarise(state: RunState, deductions: dict[str, Deduction], policy: Policy) -> dict[str, Any]:
    cases = list(state.cases.values())
    by_state: dict[str, int] = {}
    for case in cases:
        by_state[case.state] = by_state.get(case.state, 0) + 1

    return {
        "cases": len(cases),
        "by_state": dict(sorted(by_state.items())),
        "contacts": len(state.contacts),
        "recovered_paise": sum(int(c.recovered_paise) for c in cases),
        "written_off_paise": sum(int(c.written_off_paise) for c in cases),
        "credit_notes_paise": sum(int(c.credit_note_paise) for c in cases),
        "cost_paise": sum(int(c.cost_incurred_paise) for c in cases),
        "approvals_queued": len(state.approvals),
        "human_queue": sum(1 for c in cases if c.state == CaseState.HUMAN_QUEUE.value),
        "disputes": len(state.disputed),
        "opt_outs": len(state.opted_out),
    }
