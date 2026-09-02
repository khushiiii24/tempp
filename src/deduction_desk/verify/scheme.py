"""Verify trade-scheme and quarterly-purchase-scheme claims against the scheme master.

Distributors accrue incentives and self-deduct them, which means the seller finds out a
scheme was claimed only when the money is already gone. Two failure modes dominate, and
neither is visible in the advice text:

* **Expired.** The scheme period closed before the invoice was even raised. The claim
  reads exactly like a valid one.
* **Slab unmet.** The scheme is live but the buyer never reached the volume that unlocks
  the rebate. Again, indistinguishable from the outside.

Both are recoverable in full, and both require looking the scheme up by period *and* by
achieved volume. Checking only that the scheme exists — which is the obvious
implementation — would pass every expired and unearned claim in the batch.
"""

from __future__ import annotations

from ..clock import parse_date
from ..config import Policy
from ..money import apply_rate_bp, rupees_to_paise, within_tolerance
from ..schemas import Deduction, Invoice
from .base import VerificationResult, invalid, partial, unknown, valid
from .store import FixtureStore


def _period_covers(row: dict, when: str) -> bool:
    start = parse_date(row["period_start"])
    end = parse_date(row["period_end"])
    return start <= parse_date(when) <= end


def verify_scheme_rebate(
    deduction: Deduction,
    invoice: Invoice,
    store: FixtureStore,
    policy: Policy,
) -> VerificationResult:
    amount = int(deduction.amount_paise)
    tolerance = int(policy.verification["amount_tolerance_paise"])

    lookup = store.schemes_for_buyer(invoice.buyer_id)
    if not lookup.found:
        return invalid(
            "verify.scheme.no_scheme_for_buyer",
            recoverable_paise=amount,
            note="No trade scheme exists for this buyer at all.",
        )

    active = [row for row in lookup.rows if _period_covers(row, invoice.issue_date)]

    if not active:
        # Every scheme this buyer has ever had was closed by the invoice date.
        nearest = max(lookup.rows, key=lambda r: r["period_end"])
        return invalid(
            "verify.scheme.expired",
            recoverable_paise=amount,
            invoice_date=invoice.issue_date,
            nearest_scheme_id=nearest["scheme_id"],
            nearest_period_end=nearest["period_end"],
            note=(
                f"Claim relates to a scheme that closed on {nearest['period_end']}, before "
                f"the invoice date {invoice.issue_date}."
            ),
        )

    # Among live schemes, did the buyer actually reach the volume slab?
    qualified = []
    for row in active:
        required = rupees_to_paise(row["min_volume_inr"])
        achieved = rupees_to_paise(row["achieved_volume_inr"])
        if achieved >= required:
            qualified.append((row, required, achieved))

    if not qualified:
        row = active[0]
        required = rupees_to_paise(row["min_volume_inr"])
        achieved = rupees_to_paise(row["achieved_volume_inr"])
        return invalid(
            "verify.scheme.volume_slab_not_met",
            recoverable_paise=amount,
            scheme_id=row["scheme_id"],
            required_volume_paise=required,
            achieved_volume_paise=achieved,
            shortfall_paise=required - achieved,
            note=(
                "Scheme is live but the buyer did not reach the qualifying volume, so no "
                "rebate accrued."
            ),
        )

    # Live and qualified. Confirm they claimed the right slab percentage.
    row, required, achieved = qualified[0]
    slab_bp = int(round(float(row["slab_pct"]) * 100))
    entitled = apply_rate_bp(int(invoice.taxable_paise), slab_bp)

    if within_tolerance(entitled, amount, tolerance):
        return valid(
            "verify.scheme.active_and_slab_met",
            scheme_id=row["scheme_id"],
            slab_bp=slab_bp,
            entitled_paise=entitled,
            achieved_volume_paise=achieved,
        )

    if amount > entitled:
        return partial(
            "verify.scheme.overclaimed_against_slab",
            recoverable_paise=amount - entitled,
            scheme_id=row["scheme_id"],
            slab_bp=slab_bp,
            entitled_paise=entitled,
            claimed_paise=amount,
        )

    return valid(
        "verify.scheme.underclaimed",
        scheme_id=row["scheme_id"],
        entitled_paise=entitled,
        claimed_paise=amount,
    )


def verify_scheme_by_id(
    scheme_id: str, invoice: Invoice, store: FixtureStore
) -> VerificationResult:
    """Direct lookup, used when the advice names a scheme explicitly."""
    lookup = store.scheme(scheme_id)
    if not lookup.found:
        return unknown(f"scheme {scheme_id} is not in the master", evidence_needed=["scheme_master"])
    row = lookup.row
    if not _period_covers(row, invoice.issue_date):
        return invalid(
            "verify.scheme.expired",
            recoverable_paise=0,
            scheme_id=scheme_id,
            period_end=row["period_end"],
        )
    return valid("verify.scheme.active", scheme_id=scheme_id)
