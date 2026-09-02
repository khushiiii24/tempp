"""Verify the contract-governed deductions: freight, rate difference, unearned discount.

The freight case is the clearest illustration of why this layer cannot be a language
model. The buyer's behaviour is *identical* in both worlds — they deduct ₹8,000 of
transport cost and write "freight as per our terms" — and the deduction is legitimate
under FOR-destination and invalid under ex-works. Nothing in the payment, the narration or
the advice distinguishes them. Only the contract does. So the verdict is a lookup, and it
is right every time, which no amount of prompt engineering can promise.

`RATE_DIFFERENCE` is the case that most tests whether an agent is actually aligned with
the seller's interests or merely aggressive. The buyer is *right*: they were billed above
the contracted rate. There is nothing to recover, the correct action is to issue a credit
note unprompted, and an agent that chases here is chasing its own employer's billing
error into a customer relationship.
"""

from __future__ import annotations

from ..clock import days_between
from ..config import Policy
from ..money import apply_rate_bp
from ..schemas import Contract, Deduction, DeliveryTerms, Invoice, PaymentEvent, Verdict
from .base import VerificationResult, invalid, unknown, valid
from .store import FixtureStore


def verify_freight(
    deduction: Deduction,
    invoice: Invoice,
    contract: Contract,
    store: FixtureStore,
    policy: Policy,
) -> VerificationResult:
    """Who owes the freight? The contract decides, and only the contract."""
    amount = int(deduction.amount_paise)

    lookup = store.contract_for_buyer(invoice.buyer_id)
    if not lookup.found:
        return unknown("no contract on file for this buyer", evidence_needed=["contract"])

    terms = str(lookup.row["delivery_terms"])

    if terms == DeliveryTerms.FOR_DESTINATION.value:
        # Seller bears freight to destination. The buyer paid it and is recovering it.
        return valid(
            "verify.freight.for_destination_seller_bears_cost",
            delivery_terms=terms,
            freight_borne_by=lookup.row.get("freight_borne_by"),
            deducted_paise=amount,
        )

    return invalid(
        "verify.freight.ex_works_buyer_bears_cost",
        recoverable_paise=amount,
        delivery_terms=terms,
        freight_borne_by=lookup.row.get("freight_borne_by"),
        deducted_paise=amount,
        note=(
            "Contract is ex-works, so transport is the buyer's cost. They are not entitled "
            "to recover it from us."
        ),
    )


def verify_rate_difference(
    deduction: Deduction,
    invoice: Invoice,
    contract: Contract,
    store: FixtureStore,
    policy: Policy,
) -> VerificationResult:
    """Recompute the invoice against the contracted rate card.

    Returns VALID with zero recoverable whenever the seller really did overbill — because
    the buyer is correct and there is no money to chase. The policy engine turns this into
    a credit note, which is the one action in the set that costs the seller money and is
    still unambiguously the right thing to do.
    """
    amount = int(deduction.amount_paise)

    excess = 0
    lines: list[dict[str, int | str]] = []
    for line in invoice.line_items:
        billed = int(line["unit_rate_paise"])
        contracted = int(line.get("contracted_rate_paise", billed))
        if billed > contracted:
            line_excess = (billed - contracted) * int(line["qty"])
            excess += line_excess
            lines.append(
                {
                    "sku": str(line["sku"]),
                    "billed_rate_paise": billed,
                    "contracted_rate_paise": contracted,
                    "qty": int(line["qty"]),
                    "excess_paise": line_excess,
                }
            )

    if excess <= 0:
        # We billed correctly, so the buyer's claim is not supported by the rate card.
        return invalid(
            "verify.rate_difference.invoice_matches_rate_card",
            recoverable_paise=amount,
            overbilled_paise=0,
            note="No line was billed above the contracted rate; the claim is unsupported.",
        )

    # The buyer is right. Recoverable is zero — there is no money to chase, only a credit
    # note to issue. `overbilled_paise` is what the credit note should be worth.
    return VerificationResult(
        verdict=Verdict.VALID,
        recoverable_paise=0,
        evidence={
            "overbilled_paise": excess,
            "claimed_paise": amount,
            "lines": lines,
            "note": (
                "Seller billed above the contracted rate. This is our error: issue a "
                "credit note, do not chase."
            ),
        },
        rules_fired=["verify.rate_difference.seller_overbilled"],
    )


def verify_unearned_discount(
    deduction: Deduction,
    invoice: Invoice,
    contract: Contract,
    store: FixtureStore,
    policy: Policy,
    *,
    payment: PaymentEvent | None = None,
) -> VerificationResult:
    """Was the early-payment discount actually earned?

    Earned means paid inside the contractual window, counted from the invoice date. Taken
    on day 45 against a 10-day window, it is recoverable in full.
    """
    amount = int(deduction.amount_paise)

    lookup = store.contract_for_buyer(invoice.buyer_id)
    if not lookup.found:
        return unknown("no contract on file", evidence_needed=["contract"])

    window_days = int(lookup.row.get("early_payment_window_days") or 0)
    discount_pct = float(lookup.row.get("early_payment_discount_pct") or 0.0)

    if discount_pct <= 0 or window_days <= 0:
        return invalid(
            "verify.discount.no_discount_in_contract",
            recoverable_paise=amount,
            note="The contract offers no early-payment discount at all.",
        )

    if payment is None:
        return unknown(
            "cannot establish the payment date for this deduction",
            evidence_needed=["payment_history"],
        )

    days_taken = days_between(invoice.issue_date, payment.value_date)

    if days_taken <= window_days:
        # Paid in time. Confirm they took the right amount, not more.
        entitled = apply_rate_bp(int(invoice.taxable_paise), int(round(discount_pct * 100)))
        tolerance = int(policy.verification["amount_tolerance_paise"])
        if amount - entitled > tolerance:
            # Paid in time, but took more discount than the contract allows. Only the
            # over-claim is recoverable; the earned portion is theirs.
            return VerificationResult(
                verdict=Verdict.PARTIAL,
                recoverable_paise=amount - entitled,
                evidence={
                    "days_taken": days_taken,
                    "window_days": window_days,
                    "entitled_paise": entitled,
                    "claimed_paise": amount,
                },
                rules_fired=["verify.discount.earned_but_overclaimed"],
            )
        return valid(
            "verify.discount.earned_within_window",
            days_taken=days_taken,
            window_days=window_days,
            entitled_paise=entitled,
        )

    return invalid(
        "verify.discount.taken_outside_window",
        recoverable_paise=amount,
        days_taken=days_taken,
        window_days=window_days,
        note=(
            f"Discount window is {window_days} days; payment arrived on day {days_taken}. "
            f"The discount was not earned."
        ),
    )
