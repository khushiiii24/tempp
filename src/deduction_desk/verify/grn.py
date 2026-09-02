"""Verify goods claims — damage, shortage, quality rejection, buyer debit notes.

A goods claim is legitimate when a discrepancy was actually recorded against the delivery.
Unlike the tax fixtures, the GRN store is our own warehouse system: if no discrepancy was
logged at receipt, none was reported, and the claim is unsupported.

The one genuine subtlety is `DEBIT_NOTE_BUYER`. A buyer-raised debit note is not evidence
of anything by itself — it is the buyer asserting a claim, not proving one. Where no GRN
discrepancy backs it, the right move is to **request the documentation**, not to chase and
not to concede. That distinction is why this returns `UNKNOWN` with an evidence request
rather than `INVALID`: we have not disproved it, we simply have not been shown it, and
firing a chase letter at a customer over a claim we have not actually examined is how an
AR function loses an account.
"""

from __future__ import annotations

from ..config import Policy
from ..money import rupees_to_paise, within_tolerance
from ..schemas import Deduction, Invoice
from .base import VerificationResult, invalid, partial, unknown, valid
from .store import FixtureStore

_EXPECTED_DISCREPANCY = {
    "DAMAGE_SHORTAGE": {"shortage", "damage"},
    "QUALITY_REJECTION": {"quality_reject"},
    "DEBIT_NOTE_BUYER": {"damage", "shortage", "quality_reject"},
}


def verify_goods_claim(
    deduction: Deduction,
    invoice: Invoice,
    store: FixtureStore,
    policy: Policy,
    *,
    code: str | None = None,
) -> VerificationResult:
    code = code or deduction.predicted_code
    amount = int(deduction.amount_paise)
    tolerance = int(policy.verification["amount_tolerance_paise"])

    lookup = store.grn_for_invoice(invoice.invoice_no)

    if not lookup.found:
        if code == "DEBIT_NOTE_BUYER":
            # The GRN log is OUR warehouse record. If no discrepancy was raised at
            # receipt, we accepted the goods as delivered, and that is disconfirming
            # evidence rather than mere silence — unlike 26AS, this system does not lag.
            #
            # So the money verdict is INVALID. What stops the agent firing a demand letter
            # is `evidence_needed`: the taxonomy routes this code to `request_document`
            # first, so the ladder asks for the debit note before it asks for the money.
            # Determining the amount and choosing the action are separate jobs, and this
            # layer only does the first.
            result = invalid(
                "verify.goods.debit_note_unsupported_by_grn",
                recoverable_paise=amount,
                invoice_no=invoice.invoice_no,
                note=(
                    "Buyer raised a debit note but no discrepancy was recorded against "
                    "this delivery. Request the debit note copy before escalating."
                ),
            )
            result.evidence_needed = ["debit_note", "grn"]
            return result
        return invalid(
            f"verify.goods.no_grn_discrepancy.{code.lower()}",
            recoverable_paise=amount,
            invoice_no=invoice.invoice_no,
            note=(
                "No discrepancy was logged against this delivery, so the claim is "
                "unsupported by our own goods-receipt records."
            ),
        )

    # A discrepancy exists. Does it match the kind of claim being made?
    expected = _EXPECTED_DISCREPANCY.get(code, set())
    matching = [r for r in lookup.rows if r.get("discrepancy_type") in expected]

    if not matching:
        kinds = sorted({r.get("discrepancy_type", "") for r in lookup.rows})
        return unknown(
            f"a discrepancy exists ({', '.join(kinds)}) but not of the type claimed ({code})",
            evidence_needed=["grn", "debit_note"],
        )

    row = matching[0]
    recorded = rupees_to_paise(row["value_inr"])

    if within_tolerance(recorded, amount, tolerance):
        return valid(
            f"verify.goods.grn_supports_claim.{code.lower()}",
            grn_no=row["grn_no"],
            discrepancy_type=row.get("discrepancy_type"),
            recorded_value_paise=recorded,
            deducted_paise=amount,
        )

    if amount > recorded:
        # The claim is real but inflated. Only the excess is ours to recover.
        return partial(
            "verify.goods.claim_exceeds_recorded_discrepancy",
            recoverable_paise=amount - recorded,
            grn_no=row["grn_no"],
            recorded_value_paise=recorded,
            deducted_paise=amount,
        )

    return valid(
        "verify.goods.claim_within_recorded_discrepancy",
        grn_no=row["grn_no"],
        recorded_value_paise=recorded,
        deducted_paise=amount,
    )
