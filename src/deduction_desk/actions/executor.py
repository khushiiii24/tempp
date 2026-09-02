"""Execute one policy decision. The bounded action set, and nothing outside it.

Structure worth noting: this module has one branch per `ActionType` and no default that
does anything. An action the executor does not recognise raises rather than falling
through to a chase, because "do something reasonable with an unknown instruction" is how
an agent ends up sending a letter nobody authorised.

Three hard rules live here:

* **Nothing sends for real.** Every outbound message lands in `Outbox` with
  `dry_run=True`. Only `--live` *and* an environment variable together can change that,
  and the demo never sets either.
* **`propose_credit_hold` never executes.** It creates a `HumanApproval` row and stops.
  Refusing to ship over a disputed deduction can end a customer relationship, so no amount
  of ladder exhaustion authorises the agent to do it.
* **The compliance gate runs before every contact**, and a blocked contact is recorded
  rather than retried. `eval/compliance_audit.py` independently re-derives violations from
  the resulting `ContactLog` without calling anything here — if the same code both enforced
  and verified, "zero violations" would be the function agreeing with itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..clock import IST, at_time, next_contact_slot, parse_date, to_iso
from ..config import Policy
from ..generator.buyers import contact_for_role
from ..money import Paise, format_inr
from ..policy.compliance import check_contact
from ..policy.engine import Action
from ..schemas import (
    ActionType,
    Buyer,
    Case,
    CaseState,
    ContactLog,
    Deduction,
    HumanApproval,
    Invoice,
    Outbox,
)
from .drafts import Draft, build_slots, make_draft
from .razorpay_client import describe_link_for_message, payment_link_allowed


@dataclass
class ExecutionResult:
    """What actually happened, for the decision log."""

    executed: bool
    action_type: str
    detail: str = ""
    contact: ContactLog | None = None
    outbox: Outbox | None = None
    approval: HumanApproval | None = None
    draft: Draft | None = None
    compliance_passed: list[str] = field(default_factory=list)
    compliance_blocked: list[str] = field(default_factory=list)
    cost_paise: int = 0
    next_state: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "action_type": self.action_type,
            "detail": self.detail,
            "cost_paise": self.cost_paise,
            "compliance_passed": self.compliance_passed,
            "compliance_blocked": self.compliance_blocked,
            "drafted_by": self.draft.drafted_by if self.draft else None,
            "validator_rejections": (
                self.draft.validation.rejections if self.draft else []
            ),
            "next_state": self.next_state,
        }


def _contact_instant(sim_date: str, policy: Policy) -> datetime:
    """When today's contact would go out.

    Scheduled at the start of the permitted window rather than "now", because there is no
    "now" in a simulated day — and picking an arbitrary hour would make the compliance
    window test depend on nothing.
    """
    window = policy.compliance["contact_window_ist"]
    hour, minute = (int(x) for x in window["start"].split(":"))
    return at_time(parse_date(sim_date), hour, minute)


def execute(
    action: Action,
    *,
    case: Case,
    deduction: Deduction,
    invoice: Invoice,
    buyer: Buyer,
    policy: Policy,
    sim_date: str,
    run_id: str,
    buyer_contacts: list[ContactLog],
    llm_client=None,
    razorpay_client=None,
    sequence: int = 0,
) -> ExecutionResult:
    """Carry out one decision. Pure with respect to the database — the caller persists."""
    kind = action.type

    # ---- actions that touch nobody -------------------------------------------------
    if kind == ActionType.NO_ACTION:
        return ExecutionResult(
            executed=True, action_type=kind.value, detail=action.reason,
            next_state=action.next_state,
        )

    if kind == ActionType.CLOSE_VALID:
        return ExecutionResult(
            executed=True,
            action_type=kind.value,
            detail=f"closed as valid: {action.reason}",
            next_state=action.next_state or CaseState.RESOLVED_CLOSED_VALID.value,
        )

    if kind == ActionType.WRITE_OFF:
        return ExecutionResult(
            executed=True,
            action_type=kind.value,
            detail=f"written off {format_inr(action.amount_paise)}: {action.reason}",
            next_state=action.next_state or CaseState.RESOLVED_WRITTEN_OFF.value,
        )

    if kind == ActionType.ISSUE_CREDIT_NOTE:
        if action.requires_human:
            approval = HumanApproval(
                id=f"HA-{run_id}-{case.id}-cn",
                run_id=run_id,
                case_id=case.id,
                requested_at=sim_date,
                action_type=kind.value,
                amount_paise=Paise(action.amount_paise),
                reason=action.reason,
                payload={"invoice_no": invoice.invoice_no},
            )
            return ExecutionResult(
                executed=False,
                action_type=kind.value,
                detail="credit note above approval threshold; queued for a human",
                approval=approval,
                next_state=CaseState.HUMAN_QUEUE.value,
            )
        return ExecutionResult(
            executed=True,
            action_type=kind.value,
            detail=f"credit note issued for {format_inr(action.amount_paise)}",
            next_state=action.next_state or CaseState.RESOLVED_CREDIT_NOTE.value,
        )

    if kind == ActionType.ROUTE_TO_HUMAN:
        return ExecutionResult(
            executed=True,
            action_type=kind.value,
            detail=action.reason,
            cost_paise=policy.contact_cost_paise("human_analyst"),
            next_state=CaseState.HUMAN_QUEUE.value,
        )

    if kind == ActionType.RECORD_PROMISE_TO_PAY:
        return ExecutionResult(
            executed=True, action_type=kind.value, detail=action.reason,
            next_state=CaseState.PROMISED.value,
        )

    # ---- the one that always waits for a person -------------------------------------
    if kind == ActionType.PROPOSE_CREDIT_HOLD:
        approval = HumanApproval(
            id=f"HA-{run_id}-{case.id}-hold",
            run_id=run_id,
            case_id=case.id,
            requested_at=sim_date,
            action_type=kind.value,
            amount_paise=Paise(action.amount_paise),
            reason=action.reason,
            payload={
                "buyer": buyer.name,
                "relationship_value_paise": int(buyer.relationship_value_paise),
                "note": "Never auto-executes. Refusing to ship can end a relationship.",
            },
        )
        return ExecutionResult(
            executed=False,
            action_type=kind.value,
            detail="credit hold PROPOSED and left in the approval queue, unexecuted",
            approval=approval,
            next_state=CaseState.HUMAN_QUEUE.value,
        )

    # ---- everything below contacts the buyer ----------------------------------------
    if kind not in {ActionType.CHASE, ActionType.ESCALATE_ROLE, ActionType.REQUEST_DOCUMENT}:
        raise ValueError(f"executor has no branch for {kind!r}")

    channel = action.channel or "email"
    role = action.recipient_role or case.current_role
    when = _contact_instant(sim_date, policy)

    gate = check_contact(
        buyer=buyer,
        case=case,
        channel=channel,
        when=when,
        buyer_history=buyer_contacts,
        policy=policy,
    )

    if not gate.allowed:
        # Blocked, and recorded as blocked. The next permitted slot is computed so the
        # caller can see the agent is deferring rather than abandoning.
        try:
            next_slot = next_contact_slot(
                parse_date(sim_date),
                policy.compliance["contact_days"],
                policy.compliance["contact_window_ist"]["start"],
            )
            deferred = f"; next permitted slot {to_iso(next_slot)}"
        except ValueError:
            deferred = ""
        return ExecutionResult(
            executed=False,
            action_type=kind.value,
            detail=f"blocked by compliance: {', '.join(gate.violations)}{deferred}",
            compliance_passed=gate.checks_passed,
            compliance_blocked=gate.violations,
            next_state=None,
        )

    slots = build_slots(
        buyer=buyer,
        invoice=invoice,
        deduction=deduction,
        case_ref=case.id,
        amount_paise=action.amount_paise or int(deduction.recoverable_paise),
        evidence=action.evidence_requested,
        section=(deduction.predicted_code or "").replace("TDS_", ""),
    )

    draft = make_draft(
        template_id=action.template_id or "chase_1",
        slots=slots,
        policy=policy,
        case_amount_paise=action.amount_paise or int(deduction.recoverable_paise),
        invoice_total_paise=int(invoice.total_paise),
        llm_client=llm_client,
        escalation_authorised=(kind == ActionType.ESCALATE_ROLE),
    )

    # Phase 7: attach a payable link, when BOTH the integration is configured and the
    # compliance policy permits links. Off by default on both counts. The draft validator
    # independently rejects any URL that policy has not authorised, so a link cannot reach
    # a customer through this path or any other.
    payment_link = None
    if (
        kind == ActionType.CHASE
        and razorpay_client is not None
        and getattr(razorpay_client, "enabled", False)
        and payment_link_allowed(policy)
    ):
        payment_link = razorpay_client.create_payment_link(
            amount_paise=action.amount_paise or int(deduction.recoverable_paise),
            reference_id=case.id,
            description=f"Short payment on {invoice.invoice_no}",
            customer_email=contact_for_role(buyer, role),
        )
        draft.body = f"{draft.body}\n\n{describe_link_for_message(payment_link)}"

    recipient = contact_for_role(buyer, role)
    cost = policy.contact_cost_paise(channel)
    ts = to_iso(when.astimezone(IST))

    contact = ContactLog(
        id=f"CT-{run_id}-{case.id}-{sequence:03d}",
        case_id=case.id,
        buyer_id=buyer.id,
        ts=ts,
        channel=channel,
        recipient_role=role,
        recipient_address=recipient,
        template_id=draft.template_id,
        subject=draft.subject,
        body=draft.body,
        drafted_by=draft.drafted_by,
        validator_rejections=draft.validation.rejections,
        policy_checks_passed=gate.checks_passed,
        cost_paise=Paise(cost),
    )

    outbox = Outbox(
        id=f"OB-{run_id}-{case.id}-{sequence:03d}",
        run_id=run_id,
        case_id=case.id,
        channel=channel,
        payload={
            "to": recipient,
            "role": role,
            "subject": draft.subject,
            "body": draft.body,
            "template_id": draft.template_id,
            "payment_link": payment_link.as_dict() if payment_link else None,
        },
        scheduled_for=ts,
        sent_at=None,
        # Never sends. `--live` plus DEDUCTION_DESK_ALLOW_LIVE=1 are both required, and
        # neither is set anywhere in the demo path.
        dry_run=True,
    )

    return ExecutionResult(
        executed=True,
        action_type=kind.value,
        detail=(
            f"{kind.value} to {role} via {channel} "
            f"({draft.drafted_by}, {len(draft.validation.rejections)} rejection(s))"
        ),
        contact=contact,
        outbox=outbox,
        draft=draft,
        compliance_passed=gate.checks_passed,
        cost_paise=cost,
        next_state=action.next_state,
    )
