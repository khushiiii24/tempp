"""Stopping rules. When the agent must not act, and which rule says so.

These are evaluated **before** any action is considered, because most of them are terminal
and the cheapest correct action is very often no action at all.

The rule worth pointing at in a demo is `stop_if_relationship_value_ratio_exceeds`. An
agent that maximises recovery will happily chase ₹1,200 from a ₹3Cr account, win the
₹1,200, and put the account at risk. The ratio rule makes that trade explicit and refuses
it — and because the rule is named in the decision log, the refusal is inspectable rather
than an absence of behaviour. A system that can only be shown doing things cannot be shown
to be bounded.

Every function here is a pure function of `(case, deduction, verification, buyer, policy)`.
No LLM, no I/O, no clock beyond what is passed in.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..clock import days_between
from ..config import Policy
from ..schemas import Buyer, Case, CaseState, Deduction, Verdict
from .compliance import usable_channels


@dataclass
class StopDecision:
    """Why the agent is not acting, in a form the audit log can carry."""

    stop: bool
    rule: str | None = None
    reason: str = ""
    terminal: bool = True  # False for a temporary freeze, e.g. an open promise
    next_state: str | None = None


NO_STOP = StopDecision(stop=False)


def expected_recovery_paise(
    deduction: Deduction, *, contacts_remaining: int, base_success_rate: float = 0.55
) -> int:
    """Expected value of continuing to chase.

    A deliberately simple model — the recoverable amount discounted by a success rate that
    decays with each contact already spent. It is not trying to be a forecast; it is
    trying to be an *auditable* number that a human can argue with, which a learned
    propensity model would not be. The decay reflects the obvious: a buyer who has ignored
    three letters is less likely to pay than one who has ignored none.
    """
    if contacts_remaining <= 0:
        return 0
    decay = base_success_rate ** max(1, (4 - contacts_remaining) + 1)
    return int(deduction.recoverable_paise * decay)


def cost_to_continue_paise(case: Case, policy: Policy, *, channel: str) -> int:
    """What one more contact costs, including the eventual human touch if it escalates."""
    return policy.contact_cost_paise(channel)


def evaluate(
    *,
    case: Case,
    deduction: Deduction,
    buyer: Buyer,
    policy: Policy,
    today: str,
    dispute_raised: bool = False,
    opted_out: bool = False,
    human_takeover: bool = False,
    next_channel: str = "email",
) -> StopDecision:
    """Should the agent stop? Returns the first rule that says so, by severity."""
    rules = policy.stopping

    # ---- hard legal / behavioural stops ---------------------------------------
    if rules.get("stop_on_dispute_raised", True) and dispute_raised:
        return StopDecision(
            stop=True,
            rule="stopping.dispute_raised",
            reason="Buyer formally disputed the claim; all collection contact ceases.",
            next_state=CaseState.STOPPED.value,
        )

    if rules.get("stop_on_opt_out", True) and opted_out:
        return StopDecision(
            stop=True,
            rule="stopping.opt_out",
            reason="Buyer asked not to be contacted about this.",
            next_state=CaseState.STOPPED.value,
        )

    if rules.get("stop_on_human_takeover", True) and human_takeover:
        return StopDecision(
            stop=True,
            rule="stopping.human_takeover",
            reason="A human has taken the case; the agent stands down.",
            next_state=CaseState.HUMAN_QUEUE.value,
        )

    # ---- nobody we may lawfully contact ----------------------------------------
    # A buyer on DND, or with no consented channel, cannot be reached at all. Without this
    # the case is not stopped — it is *retried every single tick*, blocked identically each
    # time, for the whole 45-day run. Measured: 315 DND blocks and 136 consent blocks
    # against 33 contacts actually sent. The compliance gate was doing its job perfectly
    # and the agent simply never took the hint.
    if not usable_channels(buyer, policy):
        return StopDecision(
            stop=True,
            rule="stopping.no_permissible_channel",
            reason=(
                "Buyer is on DND or has consented to no channel; there is no lawful way to "
                "contact them. Routing to a human rather than retrying."
            ),
            next_state=CaseState.HUMAN_QUEUE.value,
        )

    # ---- nothing to chase ------------------------------------------------------
    # Only close when the deduction was actually ADJUDICATED as legitimate. Testing
    # `recoverable_paise <= 0` alone also catches everything that was never verified —
    # an unclassified deduction has zero recoverable simply because nobody looked at it,
    # and closing it as valid means the agent reports "correctly did not chase" for a case
    # it never examined. That is the most flattering possible bug: it produces a perfect
    # harm score and a perfect auto-resolution rate by doing no work at all.
    verified_valid = deduction.verdict in {
        Verdict.VALID.value,
        Verdict.PROVISIONAL_VALID.value,
    }
    if rules.get("stop_on_verified_valid", True) and verified_valid:
        return StopDecision(
            stop=True,
            rule="stopping.verified_valid",
            reason="Deduction verified as legitimate; there is no money to recover.",
            next_state=CaseState.RESOLVED_CLOSED_VALID.value,
        )

    # Never adjudicated at all — the agent has no basis to act or to close.
    if deduction.verdict == Verdict.UNKNOWN.value:
        return StopDecision(
            stop=True,
            rule="stopping.not_verified",
            reason="Deduction was never classified or verified; routing to a human.",
            next_state=CaseState.HUMAN_QUEUE.value,
        )

    # ---- promise freeze (temporary, not terminal) ------------------------------
    if case.promise_due_date:
        grace = int(rules.get("stop_on_promise_until_days_after", 3))
        days_past = days_between(case.promise_due_date, today)
        if days_past <= grace:
            return StopDecision(
                stop=True,
                rule="stopping.promise_to_pay_open",
                reason=(
                    f"Buyer promised payment by {case.promise_due_date}; "
                    f"holding until {grace} day(s) past that date."
                ),
                terminal=False,
                next_state=CaseState.PROMISED.value,
            )

    # ---- exhausted the ladder --------------------------------------------------
    if rules.get("stop_after_max_contacts", True):
        cap = int(policy.compliance["max_contacts_per_case"])
        if case.contacts_used >= cap:
            return StopDecision(
                stop=True,
                rule="stopping.max_contacts_reached",
                reason=f"Contacted {case.contacts_used} times; the cap is {cap}.",
                next_state=CaseState.STOPPED.value,
            )

    # ---- relationship value ----------------------------------------------------
    # Never chase small change from a large account.
    ratio_limit = rules.get("stop_if_relationship_value_ratio_exceeds")
    if ratio_limit and deduction.recoverable_paise > 0:
        ratio = buyer.relationship_value_paise / deduction.recoverable_paise
        if ratio > float(ratio_limit):
            return StopDecision(
                stop=True,
                rule="stopping.relationship_value_ratio_exceeded",
                reason=(
                    f"Account is worth {ratio:.0f}x the amount in dispute "
                    f"(limit {ratio_limit}x). Not worth the relationship risk."
                ),
                next_state=CaseState.STOPPED.value,
            )

    # ---- economics -------------------------------------------------------------
    if rules.get("stop_when_expected_recovery_below_cost", True):
        cap = int(policy.compliance["max_contacts_per_case"])
        remaining = max(0, cap - case.contacts_used)
        expected = expected_recovery_paise(deduction, contacts_remaining=remaining)
        cost = cost_to_continue_paise(case, policy, channel=next_channel)
        floor = policy.threshold("min_expected_value_to_chase_paise")

        if expected < max(cost, floor):
            return StopDecision(
                stop=True,
                rule="stopping.expected_recovery_below_cost",
                reason=(
                    f"Expected recovery {expected} paise is below the floor "
                    f"(cost {cost}, minimum {floor})."
                ),
                next_state=CaseState.STOPPED.value,
            )

    return NO_STOP


def should_write_off(deduction: Deduction, policy: Policy) -> bool:
    """Below the de-minimis threshold, booking it is cheaper than looking at it."""
    return 0 < deduction.recoverable_paise < policy.threshold("write_off_threshold_paise")
