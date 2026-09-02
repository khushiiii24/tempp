"""Verify statutory withholding: TDS sections, GST TDS, and the 194Q/TCS overlap.

This is the highest-volume verifier and the one where being wrong is most expensive in
both directions. Roughly two-thirds of the batch is legitimate statutory withholding that
must **never** be chased — the seller claims it as a tax credit against Form 26AS — while
a minority is deducted at the wrong rate, under the wrong section, or on top of TCS the
seller already collected, and that excess is genuinely recoverable.

Three behaviours worth stating plainly:

* **A valid TDS deduction is closed, not chased.** The money is not lost; it is a tax
  credit. The only legitimate ask is for the certificate.
* **A missing 26AS row is not evidence of anything.** The form is filed quarterly and
  lags. Roughly 15% of entirely legitimate deductions are invisible at any moment, so a
  miss produces `PROVISIONAL_VALID` with a re-check date, never a chase.
* **On a rate mismatch, only the excess is recoverable.** The buyer was obliged to
  withhold something. Chasing the whole deduction would be demanding money they were
  legally required to keep.
"""

from __future__ import annotations

from typing import Any

from ..config import Policy, Taxonomy
from ..money import apply_rate_bp, implied_rate_bp, rupees_to_paise, within_tolerance
from ..schemas import Contract, Deduction, Invoice
from .base import (
    VerificationResult,
    invalid,
    partial,
    provisional_valid,
    unknown,
    valid,
)
from .store import FixtureStore

TDS_CODES = {"TDS_194C", "TDS_194J", "TDS_194H", "TDS_194Q"}


def _matches_a_plausible_rate(taxonomy: Taxonomy, code: str, implied_bp: int | None) -> int | None:
    """Which of the section's legitimate rates does the arithmetic match, if any?"""
    if implied_bp is None or code not in taxonomy:
        return None
    for rate in taxonomy[code].plausible_rates_bp():
        if abs(rate - implied_bp) <= 2:  # absorb paise truncation
            return rate
    return None


def verify_tds(
    deduction: Deduction,
    invoice: Invoice,
    contract: Contract,
    store: FixtureStore,
    policy: Policy,
    taxonomy: Taxonomy,
    *,
    code: str | None = None,
) -> VerificationResult:
    """Adjudicate a deduction the classifier assigned to the TDS/GST family."""
    code = code or deduction.predicted_code
    tolerance = int(policy.verification["amount_tolerance_paise"])
    recheck_days = int(policy.verification["tds_recheck_after_days"])
    amount = int(deduction.amount_paise)
    taxable = int(invoice.taxable_paise)
    implied_bp = implied_rate_bp(amount, taxable)

    # ------------------------------------------------------------------ GST TDS
    if code == "GST_TDS":
        lookup = store.gst_tds_for_invoice(invoice.invoice_no)
        if lookup.found:
            declared = rupees_to_paise(lookup.row["amount_deducted_inr"])
            if within_tolerance(declared, amount, tolerance):
                return valid(
                    "verify.gst_tds.matches_gstr7",
                    gstr7_amount_paise=declared,
                    deducted_paise=amount,
                )
            return partial(
                "verify.gst_tds.amount_mismatch",
                recoverable_paise=amount - declared,
                gstr7_amount_paise=declared,
                deducted_paise=amount,
            )
        return provisional_valid(
            "verify.gst_tds.not_yet_in_gstr7",
            recheck_after_days=recheck_days,
            note="GSTR-7 is filed monthly and lags; absence is not disproof.",
        )

    # -------------------------------------------------- 194Q on top of seller TCS
    if code == "TCS_194Q_OVERLAP":
        if not (contract.tcs_applicable and invoice.tcs_paise > 0):
            # The premise fails: there is no TCS for the 194Q to overlap with.
            return unknown(
                "classified as TCS/194Q overlap but the seller charged no TCS on this invoice",
                evidence_needed=["contract"],
            )
        return invalid(
            "verify.tcs_194q_overlap.seller_already_collected_tcs",
            recoverable_paise=amount,
            seller_tcs_paise=int(invoice.tcs_paise),
            buyer_194q_paise=amount,
            note=(
                "Seller collected TCS u/s 206C(1H); the buyer is not also entitled to "
                "deduct u/s 194Q on the same transaction."
            ),
        )

    # ----------------------------------------------------------- wrong rate/section
    if code == "TDS_RATE_MISMATCH":
        expected_code = contract.tds_section_expected
        correct_bp = int(contract.tds_rate_expected_bp)
        if not correct_bp or implied_bp is None:
            return unknown(
                "cannot establish the correct rate for this contract",
                evidence_needed=["contract", "tds_certificate"],
            )

        # The buyer should have withheld at the rate pinned on the contract. Anything
        # above it is the recoverable excess; at or below is not a mismatch at all.
        correct_amount = apply_rate_bp(taxable, correct_bp)
        excess = amount - correct_amount

        if excess <= tolerance:
            return valid(
                "verify.tds.rate_within_expected",
                implied_rate_bp=implied_bp,
                expected_rate_bp=correct_bp,
            )

        return partial(
            "verify.tds.rate_mismatch_excess_only",
            recoverable_paise=excess,
            implied_rate_bp=implied_bp,
            expected_section=expected_code,
            expected_rate_bp=correct_bp,
            correct_amount_paise=correct_amount,
            deducted_paise=amount,
            computation=(
                f"deducted {amount} = {implied_bp}bp of {taxable}; "
                f"correct {correct_amount} = {correct_bp}bp; excess {excess}"
            ),
        )

    # ------------------------------------------------------------ plain sections
    if code in TDS_CODES:
        matched_rate = _matches_a_plausible_rate(taxonomy, code, implied_bp)

        # The arithmetic disagrees with the label. Trust the arithmetic: recompute against
        # what the contract expects and recover only the excess.
        if matched_rate is None:
            correct_bp = int(contract.tds_rate_expected_bp)
            if correct_bp and implied_bp is not None:
                correct_amount = apply_rate_bp(taxable, correct_bp)
                excess = amount - correct_amount
                if excess > tolerance:
                    return partial(
                        "verify.tds.rate_not_statutory_excess_recoverable",
                        recoverable_paise=excess,
                        implied_rate_bp=implied_bp,
                        expected_rate_bp=correct_bp,
                        deducted_paise=amount,
                    )
            return unknown(
                f"deduction implies {implied_bp}bp, which matches no statutory rate for {code}",
                evidence_needed=["tds_certificate", "form_26as"],
            )

        # The rate is legitimate. Now confirm the buyer actually deposited it — against
        # the row for THIS section, since an invoice can carry more than one withholding.
        lookup = store.tds_for_invoice(invoice.invoice_no, section=code)
        if not lookup.found:
            # THE case that separates a careful agent from a naive one.
            return provisional_valid(
                "verify.tds.not_yet_in_26as",
                recheck_after_days=recheck_days,
                implied_rate_bp=implied_bp,
                matched_rate_bp=matched_rate,
                note=(
                    "Rate is statutory and consistent with the contract, but 26AS is filed "
                    "quarterly and has not caught up. Provisionally close and re-check; "
                    "do NOT chase."
                ),
            )

        declared = rupees_to_paise(lookup.row["amount_deducted_inr"])
        if within_tolerance(declared, amount, tolerance):
            return valid(
                "verify.tds.matches_26as",
                form_26as_amount_paise=declared,
                deducted_paise=amount,
                section=lookup.row.get("section"),
                matched_rate_bp=matched_rate,
            )

        if declared < amount:
            # They withheld more than they deposited. The gap is genuinely ours.
            return partial(
                "verify.tds.deducted_more_than_deposited",
                recoverable_paise=amount - declared,
                form_26as_amount_paise=declared,
                deducted_paise=amount,
            )

        return valid(
            "verify.tds.deposited_at_least_deducted",
            form_26as_amount_paise=declared,
            deducted_paise=amount,
        )

    return unknown(f"{code} is not a TDS-family code", evidence_needed=["remittance_detail"])


def tds_credit_note(result: VerificationResult) -> dict[str, Any]:
    """What to record when a TDS deduction closes as valid.

    The money is not written off — it is a tax credit the seller will claim. Booking it as
    a loss would understate the seller's position and overstate the leak the system is
    supposed to be measuring.
    """
    return {
        "treatment": "tax_credit_expected",
        "provisional": result.is_provisional,
        "recheck_after_days": result.recheck_after_days,
    }
