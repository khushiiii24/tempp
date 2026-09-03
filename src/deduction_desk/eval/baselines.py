"""Baseline policies. Same batch, same clock, same cost model, same counterparty.

Each is a `PolicyFn` and runs through the identical tick loop as the agent. That matters
more than it sounds: a baseline with its own runner would differ from the agent in ways
nobody could account for, and every gap in the scoreboard would be arguable. Here the only
thing that changes is the decision function.

* **B0 — do nothing.** Everything short-paid becomes a write-off. The floor.
* **B1 — blanket dunning.** Email every open delta, no classification, no verification.
  This is what most AR teams actually run, and it is the interesting comparison because it
  will chase statutory TDS deductions from customers who did nothing wrong. That harm
  number is the point of the whole exercise.
* **B2 — threshold rule.** Chase anything above ₹5,000, ignore the rest. Slightly smarter
  than B1 and still blind to *why* the money is missing.
* **B3 — classification without verification.** The ablation. It trusts the classifier's
  label and acts on the taxonomy's default, never checking 26AS or the contract. This is
  what answers "what is the deterministic verification layer actually worth?" with a
  number rather than an assertion.

B1 and B2 deliberately ignore compliance-driven stopping rules that depend on knowing
whether a deduction is valid — they cannot know. They still obey the contact window, the
caps and consent, because those are legal constraints an AR team operates under regardless
of how naive its policy is; a baseline that broke the law would be a strawman.
"""

from __future__ import annotations

from ..config import Policy, Taxonomy, load_taxonomy
from ..money import format_inr
from ..policy import compliance
from ..policy.engine import Action, DecisionContext
from ..schemas import ActionType, CaseState, Role

_TAXONOMY: Taxonomy | None = None


def _taxonomy() -> Taxonomy:
    global _TAXONOMY
    if _TAXONOMY is None:
        _TAXONOMY = load_taxonomy()
    return _TAXONOMY


def _settlement_guard(ctx: DecisionContext, policy: Policy) -> Action | None:
    """Even a naive policy waits for the split-payment grace period.

    Not charity toward the baselines: without it they would chase part-payments that were
    always going to complete, and the resulting false-chase count would be an artefact of
    the clock rather than of the policy. The comparison has to isolate the policy.
    """
    grace = int(policy.settlement["treat_as_short_after_days"])
    if ctx.days_past_due < grace:
        return Action(
            type=ActionType.NO_ACTION,
            rules_fired=["settlement.within_grace_period"],
            reason=f"only {ctx.days_past_due} day(s) past due",
            next_state=CaseState.AWAITING_SETTLEMENT.value,
        )
    return None


def _contact_cap_reached(ctx: DecisionContext, policy: Policy) -> Action | None:
    cap = int(policy.compliance["max_contacts_per_case"])
    if ctx.case.contacts_used >= cap:
        return Action(
            type=ActionType.NO_ACTION,
            rules_fired=["stopping.max_contacts_reached"],
            reason=f"contacted {ctx.case.contacts_used} times; cap is {cap}",
            next_state=CaseState.STOPPED.value,
        )
    return None


# ======================================================================================
# B0 — do nothing
# ======================================================================================
def b0_do_nothing(ctx: DecisionContext, policy: Policy) -> Action:
    """Never chase. Every shortfall becomes a write-off.

    The honest floor, and not a stupid one: it is exactly what happens today to every
    deduction below a team's attention threshold.
    """
    return Action(
        type=ActionType.WRITE_OFF,
        rules_fired=["baseline.b0.never_chase"],
        reason="B0 writes off every shortfall without investigation",
        amount_paise=int(ctx.deduction.amount_paise),
        next_state=CaseState.RESOLVED_WRITTEN_OFF.value,
    )


# ======================================================================================
# B1 — blanket dunning
# ======================================================================================
def b1_blanket_dunning(ctx: DecisionContext, policy: Policy) -> Action:
    """Email every open delta every 7 days, up to the cap. No classification.

    The realistic comparison, and the one that produces the harm number: it cannot tell a
    statutory TDS withholding from an invalid freight claim, so it chases both.
    """
    if (guard := _settlement_guard(ctx, policy)) is not None:
        return guard
    if (stop := _contact_cap_reached(ctx, policy)) is not None:
        return stop

    # Its own 7-day cadence, on top of the compliance minimum gap.
    if ctx.case.last_contact_at:
        from ..clock import days_between

        if days_between(ctx.case.last_contact_at[:10], ctx.today) < 7:
            return Action(
                type=ActionType.NO_ACTION,
                rules_fired=["baseline.b1.within_dunning_cycle"],
                reason="B1 dunning cycle is 7 days",
            )

    return Action(
        type=ActionType.CHASE,
        rules_fired=["baseline.b1.chase_every_delta"],
        reason=f"B1 chases every open shortfall of {format_inr(ctx.deduction.amount_paise)}",
        channel="email",
        recipient_role=ctx.case.current_role or Role.AP_CLERK.value,
        template_id=f"chase_{min(ctx.case.contacts_used + 1, 4)}",
        amount_paise=int(ctx.deduction.amount_paise),
        next_state=CaseState.CHASING.value,
    )


# ======================================================================================
# B2 — threshold rule
# ======================================================================================
def b2_threshold(ctx: DecisionContext, policy: Policy) -> Action:
    """Chase above ₹5,000, ignore below. Still blind to why the money is missing."""
    threshold = 500_000  # Rs 5,000 in paise

    if (guard := _settlement_guard(ctx, policy)) is not None:
        return guard

    if int(ctx.deduction.amount_paise) < threshold:
        return Action(
            type=ActionType.WRITE_OFF,
            rules_fired=["baseline.b2.below_threshold"],
            reason=f"below {format_inr(threshold)}; B2 writes it off unexamined",
            amount_paise=int(ctx.deduction.amount_paise),
            next_state=CaseState.RESOLVED_WRITTEN_OFF.value,
        )

    if (stop := _contact_cap_reached(ctx, policy)) is not None:
        return stop

    channel = compliance.channel_for_attempt(ctx.case.contacts_used, policy, ctx.buyer) or "email"
    return Action(
        type=ActionType.CHASE,
        rules_fired=["baseline.b2.above_threshold"],
        reason=f"{format_inr(ctx.deduction.amount_paise)} is above B2's chase threshold",
        channel=channel,
        recipient_role=ctx.case.current_role or Role.AP_CLERK.value,
        template_id=f"chase_{min(ctx.case.contacts_used + 1, 4)}",
        amount_paise=int(ctx.deduction.amount_paise),
        next_state=CaseState.CHASING.value,
    )


# ======================================================================================
# B3 — classification, no verification
# ======================================================================================
def b3_classify_only(ctx: DecisionContext, policy: Policy) -> Action:
    """Trust the classifier's label; never check it against source data.

    The ablation that prices the verification layer. It acts on the taxonomy's *default*
    validity, which is right on average and wrong exactly where it costs money: freight is
    "depends on the contract", and without reading the contract this policy has to guess.
    """
    if (guard := _settlement_guard(ctx, policy)) is not None:
        return guard

    code = ctx.deduction.predicted_code

    if ctx.classification_abstained:
        return Action(
            type=ActionType.ROUTE_TO_HUMAN,
            rules_fired=["baseline.b3.abstained"],
            reason="classifier abstained",
            requires_human=True,
            next_state=CaseState.HUMAN_QUEUE.value,
        )

    taxonomy = _taxonomy()
    reason_code = taxonomy.codes.get(code)

    # Default validity from the taxonomy, with no source data consulted. `None` means
    # "depends on evidence we have not looked at" — B3 guesses chaseable, which is the
    # whole point of the ablation.
    if reason_code is not None and reason_code.default_valid is True:
        return Action(
            type=ActionType.CLOSE_VALID,
            rules_fired=["baseline.b3.taxonomy_default_valid"],
            reason=f"{code} is valid by default; closed WITHOUT checking source data",
            amount_paise=int(ctx.deduction.amount_paise),
            next_state=CaseState.RESOLVED_CLOSED_VALID.value,
        )

    if (stop := _contact_cap_reached(ctx, policy)) is not None:
        return stop

    channel = compliance.channel_for_attempt(ctx.case.contacts_used, policy, ctx.buyer) or "email"
    return Action(
        type=ActionType.CHASE,
        rules_fired=["baseline.b3.taxonomy_default_chaseable"],
        reason=f"{code} is chaseable by default; chasing WITHOUT verification",
        channel=channel,
        recipient_role=ctx.case.current_role or Role.AP_CLERK.value,
        template_id=f"chase_{min(ctx.case.contacts_used + 1, 4)}",
        amount_paise=int(ctx.deduction.amount_paise),
        next_state=CaseState.CHASING.value,
    )


BASELINES = {
    "b0": (b0_do_nothing, "B0 — do nothing"),
    "b1": (b1_blanket_dunning, "B1 — blanket dunning"),
    "b2": (b2_threshold, "B2 — threshold rule"),
    "b3": (b3_classify_only, "B3 — classify, no verification"),
}
