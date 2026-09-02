"""The policy engine. Deterministic, inspectable, and the only thing that decides money.

**No LLM.** If you find yourself wanting to ask a model "should we chase this?", stop —
that question is this file's entire job, and the answer has to be reproducible and
arguable. The model classifies and drafts prose; plain Python and plain YAML decide.

`decide()` is a pure function of `(deduction, verification, buyer, case, policy, today)`
returning exactly one `Action` from a fixed set. Rules are evaluated in a fixed priority
order and **every rule that fires is named**, so the decision log answers "why" without
re-running anything.

The priority order is itself a policy statement, and it is deliberate:

1. **Stopping rules first.** Most are terminal, and the cheapest correct action is very
   often none. Checking economics before checking "has this buyer opted out" would be
   backwards.
2. **Route to a human before acting**, on abstention or on size. An uncertain call and a
   large call both belong to a person.
3. **Own faults before other people's.** A rate difference is the seller's error; it gets
   a credit note, never a chase.
4. **De-minimis before chasing.** Below the write-off threshold, looking at it costs more
   than it recovers.
5. **Evidence before demands.** Ask for the debit note before asserting the claim is bad.
6. Only then the chase ladder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Policy
from ..money import format_inr
from ..schemas import ActionType, Buyer, Case, CaseState, Deduction, Role, Verdict
from . import compliance, stopping


@dataclass
class Action:
    """One bounded decision. Nothing outside `ActionType` can ever be produced."""

    type: ActionType
    rules_fired: list[str] = field(default_factory=list)
    reason: str = ""
    channel: str | None = None
    recipient_role: str | None = None
    template_id: str | None = None
    amount_paise: int = 0
    requires_human: bool = False
    next_state: str | None = None
    evidence_requested: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "rules_fired": self.rules_fired,
            "reason": self.reason,
            "channel": self.channel,
            "recipient_role": self.recipient_role,
            "template_id": self.template_id,
            "amount_paise": self.amount_paise,
            "requires_human": self.requires_human,
            "next_state": self.next_state,
            "evidence_requested": self.evidence_requested,
            "metadata": self.metadata,
        }


@dataclass
class DecisionContext:
    """Everything the engine is allowed to see. Notably absent: ground truth."""

    deduction: Deduction
    buyer: Buyer
    case: Case
    today: str
    verification: dict[str, Any] = field(default_factory=dict)
    classification_confidence: float = 0.0
    classification_abstained: bool = False
    dispute_raised: bool = False
    opted_out: bool = False
    human_takeover: bool = False
    document_already_requested: bool = False
    days_past_due: int = 0


def decide(ctx: DecisionContext, policy: Policy) -> Action:
    """Choose exactly one action. Deterministic; same inputs, same output, always."""
    fired: list[str] = []
    d = ctx.deduction
    verdict = d.verdict
    recoverable = int(d.recoverable_paise)

    # ------------------------------------------------------------------ 0. settlement
    # A split payment looks exactly like a short payment on day one. Chasing it means
    # chasing money that was always going to arrive, and it would inflate the false-chase
    # number that the entire harm argument rests on.
    grace = int(policy.settlement["treat_as_short_after_days"])
    if ctx.days_past_due < grace:
        fired.append("settlement.within_grace_period")
        return Action(
            type=ActionType.NO_ACTION,
            rules_fired=fired,
            reason=(
                f"Only {ctx.days_past_due} day(s) past due; a further remittance may still "
                f"arrive. Treating as short-paid after {grace} days."
            ),
            next_state=CaseState.AWAITING_SETTLEMENT.value,
        )

    # ------------------------------------------------------------------ 1. stopping
    next_channel = compliance.channel_for_attempt(ctx.case.contacts_used, policy, ctx.buyer)
    stop = stopping.evaluate(
        case=ctx.case,
        deduction=d,
        buyer=ctx.buyer,
        policy=policy,
        today=ctx.today,
        dispute_raised=ctx.dispute_raised,
        opted_out=ctx.opted_out,
        human_takeover=ctx.human_takeover,
        next_channel=next_channel,
    )
    if stop.stop:
        fired.append(stop.rule or "stopping.unspecified")
        # A verified-valid stop is a *close*, not a giving-up. The distinction matters in
        # the scoreboard: money correctly not chased is a success, not an abandonment.
        action_type = (
            ActionType.CLOSE_VALID
            if stop.rule == "stopping.verified_valid"
            else ActionType.NO_ACTION
        )
        return Action(
            type=action_type,
            rules_fired=fired,
            reason=stop.reason,
            next_state=stop.next_state,
            amount_paise=d.amount_paise if action_type == ActionType.CLOSE_VALID else 0,
            metadata={"terminal": stop.terminal},
        )

    # ------------------------------------------------------------------ 2. routing
    if ctx.classification_abstained:
        fired.append("routing.classifier_abstained")
        return Action(
            type=ActionType.ROUTE_TO_HUMAN,
            rules_fired=fired,
            reason="Classifier abstained; a human decides.",
            requires_human=True,
            next_state=CaseState.HUMAN_QUEUE.value,
        )

    floor = float(policy.confidence["abstain_to_human_below"])
    if ctx.classification_confidence < floor:
        fired.append("routing.confidence_below_floor")
        return Action(
            type=ActionType.ROUTE_TO_HUMAN,
            rules_fired=fired,
            reason=(
                f"Classification confidence {ctx.classification_confidence:.2f} is below "
                f"the floor {floor:.2f}."
            ),
            requires_human=True,
            next_state=CaseState.HUMAN_QUEUE.value,
        )

    if verdict == Verdict.UNKNOWN.value:
        fired.append("routing.verification_undetermined")
        return Action(
            type=ActionType.ROUTE_TO_HUMAN,
            rules_fired=fired,
            reason="Verification could not determine a verdict.",
            requires_human=True,
            next_state=CaseState.HUMAN_QUEUE.value,
        )

    threshold = policy.threshold("human_review_threshold_paise")
    if recoverable >= threshold:
        fired.append("routing.above_human_review_threshold")
        return Action(
            type=ActionType.ROUTE_TO_HUMAN,
            rules_fired=fired,
            reason=(
                f"{format_inr(recoverable)} is at or above the human review threshold "
                f"{format_inr(threshold)}."
            ),
            amount_paise=recoverable,
            requires_human=True,
            next_state=CaseState.HUMAN_QUEUE.value,
        )

    # ------------------------------------------------------------------ 3. own fault
    overbilled = int(ctx.verification.get("evidence", {}).get("overbilled_paise", 0))
    if d.predicted_code == "RATE_DIFFERENCE" and overbilled > 0:
        fired.append("action.self_correct_billing_error")
        cn_threshold = policy.threshold("cn_approval_threshold_paise")
        needs_human = overbilled >= cn_threshold
        if needs_human:
            fired.append("action.credit_note_requires_approval")
        return Action(
            type=ActionType.ISSUE_CREDIT_NOTE,
            rules_fired=fired,
            reason=(
                f"We billed {format_inr(overbilled)} above the contracted rate. "
                f"Issue a credit note; the buyer is correct."
            ),
            amount_paise=overbilled,
            requires_human=needs_human,
            next_state=(
                CaseState.HUMAN_QUEUE.value if needs_human else CaseState.RESOLVED_CREDIT_NOTE.value
            ),
        )

    # ------------------------------------------------------------------ 4. de minimis
    if stopping.should_write_off(d, policy):
        fired.append("action.below_write_off_threshold")
        return Action(
            type=ActionType.WRITE_OFF,
            rules_fired=fired,
            reason=(
                f"{format_inr(recoverable)} is below the write-off threshold "
                f"{format_inr(policy.threshold('write_off_threshold_paise'))}; "
                f"pursuing it costs more than it returns."
            ),
            amount_paise=recoverable,
            next_state=CaseState.RESOLVED_WRITTEN_OFF.value,
        )

    # ------------------------------------------------------------------ 5. evidence
    evidence_needed = list(ctx.verification.get("evidence_needed") or [])
    if evidence_needed and not ctx.document_already_requested:
        fired.append("action.request_supporting_document")
        return Action(
            type=ActionType.REQUEST_DOCUMENT,
            rules_fired=fired,
            reason=f"Need {', '.join(evidence_needed)} before asserting the claim.",
            channel="email",
            recipient_role=ctx.case.current_role or Role.AP_CLERK.value,
            template_id="request_document",
            evidence_requested=evidence_needed,
            amount_paise=recoverable,
            next_state=CaseState.VERIFYING.value,
        )

    # ------------------------------------------------------------------ 6. chase
    if verdict in {Verdict.INVALID.value, Verdict.PARTIAL.value} and recoverable > 0:
        # Escalate one rung if this buyer has already been contacted at the current level.
        if ctx.case.contacts_used > 0:
            promoted = compliance.next_role(ctx.case.current_role, policy)
            if promoted and promoted != ctx.case.current_role:
                fired.append("action.escalate_one_role")
                return Action(
                    type=ActionType.ESCALATE_ROLE,
                    rules_fired=fired,
                    reason=f"No resolution at {ctx.case.current_role}; escalating to {promoted}.",
                    channel=next_channel,
                    recipient_role=promoted,
                    template_id="chase_escalated",
                    amount_paise=recoverable,
                    next_state=CaseState.ESCALATED.value,
                )

        fired.append("action.chase")
        return Action(
            type=ActionType.CHASE,
            rules_fired=fired,
            reason=f"{format_inr(recoverable)} verified recoverable; chasing.",
            channel=next_channel,
            recipient_role=ctx.case.current_role or Role.AP_CLERK.value,
            template_id=f"chase_{min(ctx.case.contacts_used + 1, 4)}",
            amount_paise=recoverable,
            next_state=CaseState.CHASING.value,
        )

    # ------------------------------------------------------------------ 7. default
    fired.append("routing.no_rule_matched")
    return Action(
        type=ActionType.ROUTE_TO_HUMAN,
        rules_fired=fired,
        reason="No policy rule produced an action; a human decides.",
        requires_human=True,
        next_state=CaseState.HUMAN_QUEUE.value,
    )


def propose_credit_hold(ctx: DecisionContext, policy: Policy) -> Action:
    """The strongest lever, and the one that always waits for a person.

    Refusing to ship over a disputed deduction can end a customer relationship, so this is
    never auto-executed regardless of amount or how many contacts have failed. It is
    produced as a *proposal* that lands in the approval queue and stays there.
    """
    return Action(
        type=ActionType.PROPOSE_CREDIT_HOLD,
        rules_fired=["action.propose_credit_hold", "compliance.credit_hold_requires_human"],
        reason=(
            f"Chase ladder exhausted on {format_inr(ctx.deduction.recoverable_paise)}. "
            f"Proposing credit hold for human approval — never auto-executed."
        ),
        amount_paise=ctx.deduction.recoverable_paise,
        requires_human=True,
        next_state=CaseState.HUMAN_QUEUE.value,
    )
