"""Matching pipeline: payments in, allocations and deltas out, exceptions published.

Deterministic throughout. The LLM has no role here — the spec allows it as a last-resort
fallback on genuinely ambiguous narrations, and it is not used because the deterministic
ladder already resolves everything it can safely resolve. What is left over is genuinely
ambiguous, and the honest output for genuinely ambiguous is an exception, not a model's
best guess about where someone's money should go.
"""

from __future__ import annotations

import re
from typing import Any

from sqlmodel import Session, select

from ..clock import days_between
from ..config import Policy, load_policy
from ..ingest.advice_parser import apportion, parse_advice
from ..ingest.normalize import normalise_ref
from ..money import Paise
from ..schemas import (
    Allocation,
    Buyer,
    Exception_,
    Invoice,
    PaymentEvent,
    RemittanceAdvice,
)
from .deltas import InvoiceDelta, isolate_deltas, reconciles_with
from .deltas import summarise as summarise_deltas
from .exceptions import build_exception
from .exceptions import summarise as summarise_exceptions
from .matcher import MatchIndex, MatchOutcome, allocate, match_payment, resolve_buyer
from .matcher import summarise as summarise_matches

__all__ = [
    "MatchIndex",
    "MatchOutcome",
    "InvoiceDelta",
    "run_matching",
]

# Invoice references as they appear inside advice text. The advice is our own document
# format quoted back at us, so a targeted pattern is appropriate here — unlike bank
# narrations, which are arbitrary and go through the normaliser.
_ADVICE_REF = re.compile(r"INV[/\-]\d{4}[/\-]\d{3,6}")


def extract_advice_refs(raw_text: str) -> list[str]:
    """Pull invoice references out of a remittance advice.

    Deliberately conservative: it recognises our own invoice-number format and nothing
    else. A reference it cannot find becomes a matcher rung-2 problem rather than a
    speculative extraction, and the LLM advice parser (stage [1]) is what handles genuinely
    unstructured layouts.
    """
    return list(dict.fromkeys(_ADVICE_REF.findall(raw_text or "")))


def link_advices_to_payments(
    payments: list[PaymentEvent],
    advices: list[RemittanceAdvice],
    buyer_of_payment: dict[str, str | None],
    *,
    tolerance_paise: int = 100,
) -> dict[str, RemittanceAdvice]:
    """Work out which remittance advice belongs to which credit.

    Nothing states this link. The generator does not stamp it because in reality nobody
    does — an advice arrives by email, usually a few days after the money, and an AR clerk
    pairs the two by eye.

    **The amount is the evidence; the date is only a tie-breaker.** An advice's net lines
    sum to the credit it describes, and that sum matching a payment to the rupee is close
    to conclusive. Date proximity is not: a buyer with four payments in the same fortnight
    has four equally plausible candidates, and picking the nearest attaches an advice
    naming INV-0062 to a payment that settled INV-0003. That mis-link then propagates as a
    high-confidence "the advice named this invoice" match, which is the worst kind of wrong
    — confident, traceable to a real document, and completely fabricated.

    Two passes, strongest evidence first, consumed one-to-one:

    1. amount reconciles exactly (within tolerance), nearest date breaking ties
    2. date proximity alone, for advices we could not parse

    Pass 2 results are still used, but only the *references* they name are trusted, never
    their apportionment — see `run_matching`.
    """
    parsed_totals: dict[str, int | None] = {}
    for advice in advices:
        parsed = parse_advice(advice.raw_text, advice.format)
        nets = parsed.net_by_ref
        parsed_totals[advice.id] = sum(nets.values()) if nets else None

    by_buyer: dict[str, list[RemittanceAdvice]] = {}
    for advice in sorted(advices, key=lambda a: (a.received_at, a.id)):
        by_buyer.setdefault(advice.buyer_id, []).append(advice)

    consumed: set[str] = set()
    linked: dict[str, RemittanceAdvice] = {}
    ordered_payments = sorted(payments, key=lambda p: (p.value_date, p.id))

    # -- pass 1: the advice's own totals reconcile to this credit --------------------
    for payment in ordered_payments:
        buyer_id = buyer_of_payment.get(payment.id)
        if not buyer_id:
            continue

        best: tuple[int, RemittanceAdvice] | None = None
        for advice in by_buyer.get(buyer_id, []):
            if advice.id in consumed:
                continue
            total = parsed_totals.get(advice.id)
            if total is None:
                continue
            if abs(total - int(payment.amount_paise)) > tolerance_paise:
                continue
            lag = days_between(payment.value_date, advice.received_at)
            if lag < 0 or lag > 12:
                continue
            if best is None or lag < best[0]:
                best = (lag, advice)

        if best is not None:
            linked[payment.id] = best[1]
            consumed.add(best[1].id)

    # -- pass 2: date proximity, for advices whose amounts could not be read ----------
    for payment in ordered_payments:
        if payment.id in linked:
            continue
        buyer_id = buyer_of_payment.get(payment.id)
        if not buyer_id:
            continue

        best = None
        for advice in by_buyer.get(buyer_id, []):
            if advice.id in consumed or parsed_totals.get(advice.id) is not None:
                # A parseable advice that did not reconcile belongs to a different credit.
                continue
            lag = days_between(payment.value_date, advice.received_at)
            if lag < 0 or lag > 12:
                continue
            if best is None or lag < best[0]:
                best = (lag, advice)

        if best is not None:
            linked[payment.id] = best[1]
            consumed.add(best[1].id)

    return linked


def run_matching(
    session: Session,
    *,
    run_id: str = "match",
    today: str | None = None,
    policy: Policy | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Match every payment, allocate, isolate deltas, and record exceptions."""
    policy = policy or load_policy()
    tolerance = int(policy.verification["amount_tolerance_paise"])
    grace = int(policy.settlement["treat_as_short_after_days"])

    buyers = list(session.exec(select(Buyer)).all())
    invoices = list(session.exec(select(Invoice)).all())
    # Only genuine bank-statement credits. Recovery payments are created by the agent's own
    # tick loop when a chase succeeds, so feeding them back into matching asks the matcher
    # to allocate money it generated itself — they correspond to no invoice, land in the
    # exceptions report, and drag the match rate down by several points for no reason.
    payments = sorted(
        (p for p in session.exec(select(PaymentEvent)).all() if not p.recovery_for_case_id),
        key=lambda p: p.id,
    )
    advices = list(session.exec(select(RemittanceAdvice)).all())

    index = MatchIndex.build(buyers, invoices)
    invoice_by_id = {i.id: i for i in invoices}

    # Resolve buyers once, then pair each credit with its own advice, one-to-one.
    buyer_of_payment = {p.id: resolve_buyer(p, index)[0] for p in payments}
    advice_for_payment = link_advices_to_payments(payments, advices, buyer_of_payment)

    outcomes: list[MatchOutcome] = []
    allocations: list[Allocation] = []
    exceptions: list[Exception_] = []
    # Cumulative allocation per invoice, so a second credit cannot re-settle it.
    allocated_so_far: dict[str, int] = {}

    for payment in sorted(payments, key=lambda p: (p.value_date, p.id)):
        buyer_id = buyer_of_payment.get(payment.id)
        candidate_advice = advice_for_payment.get(payment.id)
        advice_refs: list[str] = []

        if candidate_advice is not None and buyer_id:
            # Keep only references that name an invoice actually belonging to this buyer.
            # An advice naming someone else's invoice is a parsing artefact, not evidence.
            for ref in extract_advice_refs(candidate_advice.raw_text):
                iid = index.variant_to_invoice.get(normalise_ref(ref))
                if iid and invoice_by_id[iid].buyer_id == buyer_id:
                    advice_refs.append(ref)

        outcome = match_payment(
            payment,
            index,
            advice=candidate_advice,
            advice_refs=advice_refs,
            tolerance_paise=tolerance,
        )
        outcomes.append(outcome)

        # The advice's own per-invoice net figures, when they reconcile to the credit.
        apportionment = None
        if candidate_advice is not None:
            parsed = parse_advice(candidate_advice.raw_text, candidate_advice.format)
            apportionment = apportion(parsed, int(payment.amount_paise), tolerance)
            if persist and parsed.lines and candidate_advice.parsed is None:
                candidate_advice.parsed = {
                    "lines": [
                        {
                            "invoice_ref": line.invoice_ref,
                            "gross_paise": line.gross_paise,
                            "deduction_paise": line.deduction_paise,
                            "net_paise": line.net_paise,
                            "stated_reason": line.stated_reason,
                        }
                        for line in parsed.lines
                    ],
                    "parser": parsed.parser,
                    "confident": parsed.confident,
                }
                session.add(candidate_advice)

        if outcome.matched:
            fresh = allocate(outcome, payment, index, apportionment=apportionment)

            # Never allocate an invoice beyond its own total.
            #
            # Two credits can legitimately land on one invoice — that is a split payment —
            # but their sum cannot exceed what was billed. Without this cap, a bundle and a
            # later subset-sum match both claimed INV-0000 and allocated Rs 60,562 against a
            # Rs 36,394 invoice, which shows up as a zero delta and silently erases a real
            # deduction. Capping keeps the arithmetic honest and turns the surplus into a
            # visible exception rather than a vanished shortfall.
            for alloc in fresh:
                invoice_total = int(invoice_by_id[alloc.invoice_id].total_paise)
                already = allocated_so_far.get(alloc.invoice_id, 0)
                headroom = max(0, invoice_total - already)
                if headroom == 0:
                    exceptions.append(
                        build_exception(
                            run_id=run_id,
                            kind="over_allocation_blocked",
                            subject_id=alloc.invoice_id,
                            seq=len(exceptions),
                            detail=(
                                f"{payment.id} would allocate "
                                f"{int(alloc.allocated_paise)} paise to an invoice already "
                                f"settled in full; refusing to double-apply"
                            ),
                            amount_paise=int(alloc.allocated_paise),
                            created_at=payment.value_date,
                        )
                    )
                    continue
                capped = min(int(alloc.allocated_paise), headroom)
                alloc.allocated_paise = Paise(capped)
                allocated_so_far[alloc.invoice_id] = already + capped
                allocations.append(alloc)
            if persist:
                payment.buyer_id_resolved = outcome.buyer_id
                session.add(payment)
                if candidate_advice is not None and candidate_advice.links_to_payment is None:
                    candidate_advice.links_to_payment = payment.id
                    session.add(candidate_advice)
        else:
            exceptions.append(
                build_exception(
                    run_id=run_id,
                    kind=outcome.exception or "unmatched_payment",
                    subject_id=payment.id,
                    seq=len(exceptions),
                    detail=outcome.detail,
                    amount_paise=int(payment.amount_paise),
                    created_at=payment.value_date,
                )
            )

    payment_dates = {p.id: p.value_date for p in payments}
    deltas = isolate_deltas(allocations, invoice_by_id, payment_dates)

    as_of = today or max((p.value_date for p in payments), default="")

    if persist:
        # Idempotent, like `run_batch`. Allocation ids are derived from the payment id, so
        # a second `match` on the same batch collided on the primary key and aborted
        # mid-transaction — leaving the table however far it got. Matching is a pure
        # function of the generated data; running it twice must be a no-op, not a crash.
        from sqlmodel import delete

        session.exec(delete(Allocation))
        session.exec(delete(Exception_))
        for alloc in allocations:
            session.add(alloc)
        for exc in exceptions:
            session.add(exc)
        session.commit()

    return {
        "matching": summarise_matches(outcomes),
        "deltas": summarise_deltas(deltas, invoice_by_id, as_of, grace),
        "exceptions": summarise_exceptions(exceptions),
        "as_of": as_of,
        "_outcomes": outcomes,
        "_deltas": deltas,
        "_exceptions": exceptions,
    }


def check_fabricated_matches(
    outcomes: list[MatchOutcome],
    deltas: dict[str, InvoiceDelta],
    deduction_totals: dict[str, int],
    tolerance: int = 100,
) -> list[str]:
    """Invoices whose derived shortfall disagrees with their itemised deductions.

    This is the fabricated-match detector. A payment matched to the wrong invoice still
    produces an allocation and still counts toward the match rate; the only thing that
    exposes it is the delta failing to reconcile with what was actually deducted.
    """
    offenders: list[str] = []
    for invoice_id, delta in sorted(deltas.items()):
        expected = deduction_totals.get(invoice_id, 0)
        if not reconciles_with(delta, expected, tolerance):
            offenders.append(
                f"{invoice_id}: derived delta {delta.delta_paise} vs deductions {expected}"
            )
    return offenders
