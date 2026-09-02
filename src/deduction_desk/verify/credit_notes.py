"""Verify credit-note offsets against our own ledger.

An offset is valid when the credit note exists, belongs to this buyer, is unapplied, and
matches the amount. Each of those four is a separate way to be wrong, and the two that
matter are:

* **Not found.** The buyer netted off a credit note we never issued. Unlike 26AS, the CN
  ledger is *our own system* — if a note is not in it, it does not exist, and absence here
  really is disproof. That asymmetry with `tds.py` is deliberate and is the whole reason
  `Lookup` carries a `may_lag` flag.
* **Already applied.** The note is real but was consumed against another invoice. This is
  double-dipping, it is invisible in the advice text, and it is only findable by checking
  the applied flag — which is exactly the sort of thing an AR analyst under time pressure
  does not do.
"""

from __future__ import annotations

import re

from ..config import Policy
from ..money import rupees_to_paise, within_tolerance
from ..schemas import Deduction, Invoice
from .base import VerificationResult, invalid, partial, valid
from .store import FixtureStore

# Credit-note references as they appear in advice prose: "CN/1234", "cn 1234", "CN-1234".
_CN_PATTERN = re.compile(r"\bCN[\s/\-]?(\d{3,6})\b", re.IGNORECASE)


def _referenced_numbers(text: str | None) -> list[str]:
    """Pull candidate CN references out of the buyer's stated reason.

    This is reading *our own identifier format* out of a string the buyer wrote, then
    looking it up in a ledger — not extracting a decision from prose. Anything not found
    in the ledger is treated as absent, so a bad extraction fails safe.
    """
    if not text:
        return []
    return [f"CN/{m.group(1)}" for m in _CN_PATTERN.finditer(text)]


def verify_credit_note_offset(
    deduction: Deduction,
    invoice: Invoice,
    store: FixtureStore,
    policy: Policy,
) -> VerificationResult:
    amount = int(deduction.amount_paise)
    tolerance = int(policy.verification["amount_tolerance_paise"])
    buyer_id = invoice.buyer_id

    # 1. If the buyer named a credit note, adjudicate that one specifically.
    for number in _referenced_numbers(deduction.claimed_reason_text):
        lookup = store.credit_note(number)
        if not lookup.found:
            return invalid(
                "verify.credit_note.not_found",
                recoverable_paise=amount,
                referenced=number,
                note="The buyer netted off a credit note that does not exist in our ledger.",
            )

        row = lookup.row
        if row["buyer_id"] != buyer_id:
            return invalid(
                "verify.credit_note.belongs_to_another_buyer",
                recoverable_paise=amount,
                referenced=number,
                belongs_to=row["buyer_id"],
            )

        if row.get("applied") == "Y":
            return invalid(
                "verify.credit_note.already_applied",
                recoverable_paise=amount,
                referenced=number,
                applied_against=row.get("applied_against"),
                note="Credit note is real but was already consumed against another invoice.",
            )

        ledger_amount = rupees_to_paise(row["amount_inr"])
        if within_tolerance(ledger_amount, amount, tolerance):
            return valid(
                "verify.credit_note.exists_and_unapplied",
                referenced=number,
                ledger_amount_paise=ledger_amount,
                deducted_paise=amount,
            )

        if amount > ledger_amount:
            return partial(
                "verify.credit_note.over_offset",
                recoverable_paise=amount - ledger_amount,
                referenced=number,
                ledger_amount_paise=ledger_amount,
                deducted_paise=amount,
            )

        return valid(
            "verify.credit_note.under_offset",
            referenced=number,
            ledger_amount_paise=ledger_amount,
            deducted_paise=amount,
        )

    # 2. No reference given: look for any unapplied note of the right size.
    match = store.unapplied_credit_notes(buyer_id, amount, tolerance)
    if match.found:
        return valid(
            "verify.credit_note.matched_unapplied_by_amount",
            credit_note_no=match.row["credit_note_no"],
            ledger_amount_paise=rupees_to_paise(match.row["amount_inr"]),
        )

    return invalid(
        "verify.credit_note.no_matching_unapplied_note",
        recoverable_paise=amount,
        deducted_paise=amount,
        note=(
            "No unapplied credit note of this value exists for this buyer. The CN ledger "
            "is our own system, so absence is conclusive."
        ),
    )
