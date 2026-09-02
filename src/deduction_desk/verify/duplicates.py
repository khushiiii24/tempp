"""Verify 'we already paid this' claims, and the de-minimis codes.

A duplicate-payment claim is the one case where the buyer is asserting a fact about *our*
bank account, so it is cheap to settle and expensive to get wrong in either direction:
chase a buyer who genuinely did pay twice and you look like you cannot read your own
ledger; concede one who did not and you have written off the invoice.

The de-minimis codes live here too. `ROUNDING` and `BANK_CHARGES` are technically
recoverable and never worth recovering — the economics are decided later by the policy
engine's write-off threshold, but the verifier records them honestly as small valid
deductions rather than pretending they are free.
"""

from __future__ import annotations

from ..config import Policy
from ..money import rupees_to_paise, within_tolerance
from ..schemas import Deduction, Invoice, Verdict
from .base import VerificationResult, invalid, valid
from .store import FixtureStore

# Above this, a "rounding" difference is not rounding.
ROUNDING_CEILING_PAISE = 1_000  # Rs 10


def verify_duplicate_claim(
    deduction: Deduction,
    invoice: Invoice,
    store: FixtureStore,
    policy: Policy,
) -> VerificationResult:
    amount = int(deduction.amount_paise)
    tolerance = int(policy.verification["amount_tolerance_paise"])

    lookup = store.payments_for_invoice(invoice.invoice_no)

    if not lookup.found:
        return invalid(
            "verify.duplicate.no_prior_payment_on_record",
            recoverable_paise=amount,
            invoice_no=invoice.invoice_no,
            note=(
                "Payment history shows no earlier settlement of this invoice, so the "
                "duplicate claim is unsupported."
            ),
        )

    total_prior = sum(rupees_to_paise(r["amount_inr"]) for r in lookup.rows)

    if within_tolerance(total_prior, amount, tolerance) or total_prior >= amount:
        return valid(
            "verify.duplicate.prior_payment_confirmed",
            prior_payments=[r["utr"] for r in lookup.rows],
            prior_total_paise=total_prior,
            claimed_paise=amount,
        )

    return VerificationResult(
        verdict=Verdict.PARTIAL,
        recoverable_paise=amount - total_prior,
        evidence={
            "prior_payments": [r["utr"] for r in lookup.rows],
            "prior_total_paise": total_prior,
            "claimed_paise": amount,
            "note": "An earlier payment exists but does not cover the full amount claimed.",
        },
        rules_fired=["verify.duplicate.prior_payment_partial"],
    )


def verify_deminimis(
    deduction: Deduction,
    invoice: Invoice,
    policy: Policy,
    *,
    code: str | None = None,
) -> VerificationResult:
    """Rounding and bank charges. Small, legitimate-ish, and never worth chasing."""
    code = code or deduction.predicted_code
    amount = int(deduction.amount_paise)

    if code == "ROUNDING":
        if amount <= ROUNDING_CEILING_PAISE:
            return valid(
                "verify.deminimis.rounding",
                deducted_paise=amount,
                note="Sub-Rs 10 difference; book it and move on.",
            )
        # Labelled rounding, too large to be rounding.
        return invalid(
            "verify.deminimis.rounding_too_large",
            recoverable_paise=amount,
            deducted_paise=amount,
            ceiling_paise=ROUNDING_CEILING_PAISE,
        )

    if code == "BANK_CHARGES":
        return valid(
            "verify.deminimis.bank_charges",
            deducted_paise=amount,
            note=(
                "Transfer charges withheld by the remitting or intermediary bank. "
                "Technically recoverable, economically never worth it — the write-off "
                "threshold decides, not this layer."
            ),
        )

    return valid("verify.deminimis.unclassified_small", deducted_paise=amount)


def verify_unexplained(deduction: Deduction, invoice: Invoice) -> VerificationResult:
    """No reason given, and nothing in the data explains the gap.

    Invalid by default — the buyer owes the invoice and has not said why they short-paid —
    but the first action is to ask, not to demand.
    """
    amount = int(deduction.amount_paise)
    return VerificationResult(
        verdict=Verdict.INVALID,
        recoverable_paise=amount,
        evidence={
            "deducted_paise": amount,
            "note": "No reason stated and no source system explains the shortfall.",
        },
        rules_fired=["verify.unexplained.no_supporting_evidence"],
        evidence_needed=["remittance_detail"],
    )
