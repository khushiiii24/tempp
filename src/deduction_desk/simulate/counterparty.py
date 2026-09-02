"""The simulated buyer. A deterministic state machine reading pre-committed ground truth.

**This is the component that makes the scoreboard defensible, and it contains no model.**

Every outcome — whether they pay, after how many contacts, at which seniority, on which
channel, whether they dispute, whether they promise and default — was drawn by the
generator before the agent existed and stored in `DeductionTruth`. This module only looks
the answer up.

The alternative, which most submissions of this shape end up building, is to ask a
language model how a buyer would react to the agent's message. That is circular: a more
persuasive email produces more "recovered" money because something downstream was
persuaded, not because a real buyer was, and the agent ends up grading its own homework.
Here a beautifully written chase to a buyer whose `will_pay_if_chased` is False recovers
exactly nothing. That is the correct answer, and it is an uncomfortable one for a demo,
which is rather the point.

Reply *text* comes from static templates with slot-filling
(`config/templates/counterparty_replies.yaml`) — see `replies.py`. Even that is not model
output, because an LLM writing the buyer's words is one refactor away from an LLM
deciding them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..clock import add_days
from ..generator.seed import rng_for
from ..schemas import Case, ContactLog, DeductionTruth, InboundKind, Role

# A junior contact cannot authorise anything, however many times you ask.
_JUNIOR_ROLES = (Role.AP_CLERK.value,)


@dataclass
class CounterpartyResponse:
    """What the buyer does, and when it lands."""

    kind: InboundKind
    deliver_on: str
    amount_paise: int = 0
    text_key: str = "generic"
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def is_silence(self) -> bool:
        return self.kind == InboundKind.SILENCE


def _role_rank(role: str, ladder: list[str]) -> int:
    return ladder.index(role) if role in ladder else 0


def respond(
    *,
    truth: DeductionTruth,
    case: Case,
    contact: ContactLog,
    escalation_ladder: list[str],
    seed: int,
) -> CounterpartyResponse:
    """Decide the buyer's reaction to one contact.

    Pure function of the truth record and the contact. No randomness beyond a seeded
    stream used for cosmetic template choice — never for the outcome.
    """
    today = contact.ts[:10]
    rng = rng_for(seed, "counterparty", f"{case.id}:{contact.id}")

    # ---- opt-out is immediate and unconditional --------------------------------
    if truth.opt_out and case.contacts_used >= 1:
        return CounterpartyResponse(
            kind=InboundKind.OPT_OUT,
            deliver_on=add_days(today, 1),
            text_key="opt_out",
        )

    # ---- wrong channel: the message is simply not seen --------------------------
    if contact.channel not in (truth.responds_to_channels or []):
        return CounterpartyResponse(kind=InboundKind.SILENCE, deliver_on=today)

    # ---- a document request is answered on its own terms ------------------------
    if contact.template_id == "request_document":
        return CounterpartyResponse(
            kind=InboundKind.DOCUMENT,
            deliver_on=add_days(today, max(1, truth.latency_days)),
            text_key="document",
            payload={"requested": True},
        )

    # ---- disputes fire as soon as the buyer is actually engaged -----------------
    if truth.will_dispute and case.contacts_used >= 1:
        return CounterpartyResponse(
            kind=InboundKind.DISPUTE,
            deliver_on=add_days(today, truth.latency_days),
            text_key="dispute",
        )

    # ---- seniority gate ---------------------------------------------------------
    # Some buyers only act once the request reaches someone who can authorise it.
    # Asking the AP clerk four times is not persistence, it is four wasted contacts —
    # which is exactly what the escalation ladder exists to avoid, and what a naive
    # agent that never escalates will do.
    required_rank = _role_rank(truth.responds_only_at_role, escalation_ladder)
    current_rank = _role_rank(contact.recipient_role, escalation_ladder)
    if current_rank < required_rank:
        return CounterpartyResponse(
            kind=InboundKind.DEFLECTION,
            deliver_on=add_days(today, max(1, truth.latency_days // 2)),
            text_key="junior_role" if contact.recipient_role in _JUNIOR_ROLES else "generic",
        )

    # ---- not yet ready to pay ---------------------------------------------------
    if case.contacts_used < truth.pays_after_n_contacts:
        # A promise-then-default buyer makes their promise on the contact before the one
        # they would otherwise have paid on, then goes quiet.
        if truth.promise_then_default and case.contacts_used == max(
            1, truth.pays_after_n_contacts - 1
        ):
            promise_date = add_days(today, truth.latency_days + 5)
            return CounterpartyResponse(
                kind=InboundKind.PROMISE,
                deliver_on=add_days(today, truth.latency_days),
                text_key="promise",
                payload={"promise_date": promise_date, "will_honour": False},
            )
        return CounterpartyResponse(
            kind=InboundKind.DEFLECTION,
            deliver_on=add_days(today, max(1, truth.latency_days)),
            text_key=_deflection_key(truth, rng),
        )

    # ---- threshold met ----------------------------------------------------------
    if truth.will_pay_if_chased and truth.recoverable_paise > 0:
        return CounterpartyResponse(
            kind=InboundKind.PAYMENT,
            deliver_on=add_days(today, truth.latency_days),
            amount_paise=truth.recoverable_paise,
            text_key="payment",
        )

    # Contacted enough times, but this buyer was never going to pay. Silence is the
    # honest outcome and the expensive one: cost incurred, nothing recovered.
    return CounterpartyResponse(
        kind=InboundKind.DEFLECTION,
        deliver_on=add_days(today, max(1, truth.latency_days)),
        text_key=_deflection_key(truth, rng),
    )


def _deflection_key(truth: DeductionTruth, rng) -> str:
    """Which flavour of brush-off. Cosmetic only — never affects the outcome."""
    code = truth.true_reason_code
    if code.startswith("TDS_") or code == "GST_TDS":
        return "tds_pushback"
    if code == "SCHEME_REBATE":
        return "scheme_pushback"
    if code == "FREIGHT":
        return "freight_pushback"
    return "generic"


def would_ever_pay(truth: DeductionTruth) -> bool:
    """Used by the eval to separate 'the agent failed' from 'nothing was available'.

    Reporting recovery against the total short-paid rather than against what was actually
    reachable would make every agent look bad and would hide which ones were actually
    working.
    """
    return truth.will_pay_if_chased and truth.recoverable_paise > 0 and not truth.will_dispute
