"""Plan every deduction: its true reason code, whether it is valid, and what it is worth.

This is the truth-first step, and the ordering matters more than anything else in the
generator. We decide *why* the buyer short-paid, then derive the amount from that reason.
A 194C deduction is `taxable x 200bp` **because** the truth record says 194C — never the
reverse. Fixtures are then written to match (`fixtures.py`), and only at the very end is
the observable mess produced (`noise.py`, `remittance.py`).

Doing it the other way round — sampling an amount and labelling it afterwards — produces
a dataset where the label is arbitrary, verification is a lookup that always succeeds, and
the accuracy numbers mean nothing.

**Feasibility constraints are enforced, not sampled.** Several codes are only physically
possible under certain contracts, and violating that would make the batch incoherent:

* `FREIGHT` validity is *derived from the delivery terms*, not drawn from a weight. Under
  FOR-destination the seller owes the freight and the buyer's deduction is legitimate;
  under ex-works it is not. The identical behaviour is valid or invalid depending only on
  the contract, which is exactly why the verification layer has to be a deterministic
  lookup rather than a language model's impression.
* `TCS_194Q_OVERLAP` can only occur where the seller actually charges TCS.
* `RATE_DIFFERENCE` can only occur on an invoice that really was billed above contract.

Where a drawn code is infeasible for the chosen invoice, we fall back rather than force
it, and the realised mix is reported by `generate` so any drift from `generator.yaml` is
visible rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Taxonomy
from ..money import apply_rate_bp
from ..schemas import Contract, DeliveryTerms, Invoice
from .invoices import rate_difference_paise
from .seed import chance, pick, rng_for, round_to_rupee, weighted_choice

# Rates a buyer plausibly applies when they get the section wrong. Each is a real TDS rate
# from a *different* section, which is what makes the mistake believable: an AP clerk
# picking the wrong row from a rate master, not inventing a number.
WRONG_RATE_CHOICES_BP = (100, 200, 500, 1000)

# Trade-scheme slabs, in basis points of taxable value.
SCHEME_SLABS_BP = (100, 150, 200, 250, 300, 500)

GST_TDS_RATE_BP = 200  # 2% u/s 51, government and PSU buyers

# TDS u/s 194Q applies only once a buyer's aggregate purchases from a seller exceed
# Rs 50 lakh in a financial year. It is a BUYER-level test, not a per-invoice one: below
# the threshold nobody deducts, above it the buyer deducts 0.1% on every subsequent
# invoice including small ones. Staging 194Q on a buyer who never crosses the threshold
# would put a deduction in the batch that no competent AP department would ever raise —
# and the same threshold governs the TCS/194Q overlap, which is the whole reason that
# confusion exists.
SECTION_194Q_THRESHOLD_PAISE = 500_000_000  # Rs 50,00,000


@dataclass
class PlannedDeduction:
    """One deduction, fully decided, before any observable artefact exists."""

    id: str
    invoice_id: str
    buyer_id: str
    code: str
    is_valid: bool
    amount_paise: int
    recoverable_paise: int
    # Everything a fixture writer or a message template needs, keyed by code.
    extra: dict[str, Any] = field(default_factory=dict)
    showcase_id: str | None = None

    @property
    def is_deminimis(self) -> bool:
        return self.code in {"BANK_CHARGES", "ROUNDING"}


def _tds_rate_for_section(taxonomy: Taxonomy, code: str, rng) -> int:
    rates = taxonomy[code].plausible_rates_bp()
    if not rates:
        return 200
    # The headline rate dominates; the alternate rate (1% for individuals, 2% for
    # technical services) shows up as a minority so the classifier cannot simply memorise
    # one number per section.
    return rates[0] if len(rates) == 1 or chance(rng, 0.78) else pick(rng, rates[1:])


def _freight_amount(rng, invoice: Invoice) -> int:
    """Transport cost scaled loosely to consignment size, then rounded to a rupee."""
    base = max(150_000, int(invoice.taxable_paise * rng.uniform(0.004, 0.022)))
    return round_to_rupee(min(base, 2_500_000))


def plan_deduction(
    seed: int,
    taxonomy: Taxonomy,
    invoice: Invoice,
    contract: Contract,
    code: str,
    *,
    ordinal: int = 0,
    forced_valid: bool | None = None,
    forced_amount: int | None = None,
    showcase_id: str | None = None,
    validity_split: dict[str, float] | None = None,
    buyer_volume_paise: int = 0,
) -> PlannedDeduction | None:
    """Turn a chosen reason code into a fully-specified deduction.

    Returns None when the code is infeasible for this invoice/contract pair, so the caller
    can draw again rather than fabricate an incoherent case.

    `buyer_volume_paise` is the buyer's aggregate purchase value across the batch, needed
    for the s.194Q threshold test.
    """
    ded_id = f"DED-{invoice.id.split('-')[1]}-{ordinal}"
    rng = rng_for(seed, "deduction", ded_id)
    split = validity_split or {}
    taxable = int(invoice.taxable_paise)
    extra: dict[str, Any] = {}

    def valid_by_split(default: float) -> bool:
        if forced_valid is not None:
            return forced_valid
        return chance(rng, float(split.get(code, default)))

    # ---------------------------------------------------------------- statutory TDS
    if code in {"TDS_194C", "TDS_194J", "TDS_194H", "TDS_194Q"}:
        if code == "TDS_194Q" and buyer_volume_paise < SECTION_194Q_THRESHOLD_PAISE:
            return None  # buyer never crossed the Rs 50L aggregate; 194Q does not apply

        # A legitimate withholding uses the section AND rate pinned on the contract.
        #
        # An earlier version drew the section from the mix independently of the contract,
        # which made the label unidentifiable: 75 of 107 valid TDS deductions ended up
        # under a section the contract did not expect, and a 10% deduction on a
        # 194C-at-2% contract was labelled "valid TDS_194J" in some rows and "invalid
        # TDS_RATE_MISMATCH" in others — with byte-identical observable data. No
        # classifier can beat chance on that, and the benchmark duly measured 0.43 macro-F1
        # against a task that was partly impossible.
        #
        # Pinning it to the contract is also the more faithful model: a vendor master
        # records the applicable section, and a buyer applying a different one is exactly
        # what TDS_RATE_MISMATCH is for.
        #
        # A drawn section that does not match the contract is SUBSTITUTED, not rejected.
        # Rejecting it sent the draw back to the mix and the TDS-family share collapsed
        # from ~38% to 23% — which quietly guts the batch's central premise, since the
        # "do not chase" statutory majority is the thing a naive agent fails on. Only the
        # within-family split moves, and it moves to follow the contract distribution,
        # which is the realistic one.
        code = contract.tds_section_expected
        if code == "TDS_194Q" and buyer_volume_paise < SECTION_194Q_THRESHOLD_PAISE:
            return None
        rate_bp = int(contract.tds_rate_expected_bp)
        amount = apply_rate_bp(taxable, rate_bp)
        if amount <= 0:
            return None
        extra = {"section": code.replace("TDS_", ""), "rate_bp": rate_bp, "base_paise": taxable}
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=True, amount_paise=amount, recoverable_paise=0,
            extra=extra, showcase_id=showcase_id,
        )

    if code == "GST_TDS":
        amount = apply_rate_bp(taxable, GST_TDS_RATE_BP)
        if amount <= 0:
            return None
        extra = {"rate_bp": GST_TDS_RATE_BP, "base_paise": taxable}
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=True, amount_paise=amount, recoverable_paise=0,
            extra=extra, showcase_id=showcase_id,
        )

    # ------------------------------------------------------- wrong rate / wrong section
    if code == "TDS_RATE_MISMATCH":
        correct_code = contract.tds_section_expected
        # The correct rate is the one PINNED ON THE CONTRACT, not a fresh draw. If the
        # generator picked a rate here independently, the verifier could not possibly
        # reproduce it and every rate-mismatch excess would be wrong.
        correct_bp = int(contract.tds_rate_expected_bp)
        # The amount is derived from the wrong RATE, never from a forced amount — a rate
        # mismatch whose amount does not equal a real rate times the base is not a rate
        # mismatch, and the classifier's arithmetic check would rightly fail to see one.
        candidates = [r for r in WRONG_RATE_CHOICES_BP if r > correct_bp]
        if not candidates:
            return None
        wrong_bp = pick(rng, tuple(candidates))
        amount = apply_rate_bp(taxable, wrong_bp)
        correct_amount = apply_rate_bp(taxable, correct_bp)
        excess = amount - correct_amount
        if excess <= 0:
            return None
        extra = {
            "section": correct_code.replace("TDS_", ""),
            "correct_rate_bp": correct_bp,
            "applied_rate_bp": wrong_bp,
            "correct_amount_paise": correct_amount,
            "base_paise": taxable,
        }
        # Only the EXCESS is recoverable. Chasing the whole deduction would be chasing
        # money the buyer was legally obliged to withhold.
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=False, amount_paise=amount, recoverable_paise=excess,
            extra=extra, showcase_id=showcase_id,
        )

    if code == "TCS_194Q_OVERLAP":
        if not contract.tcs_applicable:
            return None  # impossible unless the seller is actually collecting TCS
        if buyer_volume_paise < SECTION_194Q_THRESHOLD_PAISE:
            return None  # and impossible unless 194Q applies to the buyer at all
        rate_bp = 10
        amount = apply_rate_bp(taxable, rate_bp)
        if amount <= 0:
            return None
        extra = {
            "rate_bp": rate_bp,
            "base_paise": taxable,
            "seller_tcs_paise": int(invoice.tcs_paise),
        }
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=False, amount_paise=amount, recoverable_paise=amount,
            extra=extra, showcase_id=showcase_id,
        )

    # ------------------------------------------------------------------------ freight
    if code == "FREIGHT":
        amount = forced_amount or _freight_amount(rng, invoice)
        # Validity is a FACT ABOUT THE CONTRACT, not a coin flip.
        is_valid = contract.delivery_terms == DeliveryTerms.FOR_DESTINATION.value
        extra = {"delivery_terms": contract.delivery_terms}
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=is_valid, amount_paise=amount,
            recoverable_paise=0 if is_valid else amount,
            extra=extra, showcase_id=showcase_id,
        )

    # ------------------------------------------------------------------ scheme rebate
    if code == "SCHEME_REBATE":
        # Scheme validity is a property of the BUYER'S SCHEME, not of the individual
        # claim. A buyer sits on one scheme per period; giving them a live scheme and an
        # expired one simultaneously meant an invalid claim could be validated against
        # their other, healthy scheme — the verifier returning "valid" for something
        # ground truth says is worthless, and no way to tell which scheme was meant.
        scheme_rng = rng_for(seed, "scheme_terms", invoice.buyer_id)
        is_valid = (
            forced_valid
            if forced_valid is not None
            else chance(scheme_rng, float(split.get("SCHEME_REBATE", 0.5)))
        )
        slab_bp = pick(scheme_rng, SCHEME_SLABS_BP)
        amount = forced_amount or round_to_rupee(apply_rate_bp(taxable, slab_bp))
        if amount <= 0:
            return None
        # An invalid claim is invalid for a *reason a verifier can find*: the scheme had
        # expired, or the volume slab was never met. Written into scheme_master.csv.
        failure = (
            None if is_valid else weighted_choice(scheme_rng, {"expired": 0.6, "slab_unmet": 0.4})
        )
        # One scheme id per buyer. Real scheme masters carry one scheme per buyer per
        # period, and so does this.
        extra = {
            "scheme_id": f"SCH-{invoice.buyer_id.split('-')[1]}",
            "slab_bp": slab_bp,
            "failure_mode": failure,
        }
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=is_valid, amount_paise=amount,
            recoverable_paise=0 if is_valid else amount,
            extra=extra, showcase_id=showcase_id,
        )

    # -------------------------------------------------------------- credit note offset
    if code == "CREDIT_NOTE_OFFSET":
        is_valid = valid_by_split(0.78)
        amount = forced_amount or round_to_rupee(
            int(taxable * rng.uniform(0.01, 0.08))
        )
        if amount <= 0:
            return None
        extra = {
            "credit_note_no": f"CN/{rng.randrange(2000, 9999)}",
            # An invalid offset references a CN that does not exist, or one already applied
            # elsewhere. Both are findable in the ledger; neither is findable in the text.
            "failure_mode": None if is_valid else weighted_choice(
                rng, {"not_found": 0.55, "already_applied": 0.45}
            ),
        }
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=is_valid, amount_paise=amount,
            recoverable_paise=0 if is_valid else amount,
            extra=extra, showcase_id=showcase_id,
        )

    # ------------------------------------------------------------------ goods disputes
    if code in {"DAMAGE_SHORTAGE", "QUALITY_REJECTION", "DEBIT_NOTE_BUYER"}:
        default_split = {"DAMAGE_SHORTAGE": 0.65, "QUALITY_REJECTION": 0.60, "DEBIT_NOTE_BUYER": 0.50}
        is_valid = valid_by_split(default_split[code])
        amount = forced_amount or round_to_rupee(int(taxable * rng.uniform(0.02, 0.15)))
        if amount <= 0:
            return None
        line = pick(rng, invoice.line_items) if invoice.line_items else {}
        extra = {
            "grn_no": f"GRN-{rng.randrange(10000, 99999)}",
            "sku": line.get("sku"),
            "debit_note_no": f"DN/{rng.randrange(100, 999)}" if code == "DEBIT_NOTE_BUYER" else None,
        }
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=is_valid, amount_paise=amount,
            recoverable_paise=0 if is_valid else amount,
            extra=extra, showcase_id=showcase_id,
        )

    # ---------------------------------------------------------------- rate difference
    if code == "RATE_DIFFERENCE":
        excess = rate_difference_paise(invoice)
        if excess <= 0:
            return None  # the seller billed correctly; there is nothing to claim
        extra = {"billing_error_paise": excess}
        # Valid, and it is the SELLER's fault. Recoverable is zero: there is no money to
        # chase, only a credit note to issue.
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=True, amount_paise=excess, recoverable_paise=0,
            extra=extra, showcase_id=showcase_id,
        )

    # -------------------------------------------------------------- unearned discount
    if code == "UNEARNED_DISCOUNT":
        rate_bp = contract.early_payment_discount_bp or pick(rng, (100, 200, 250))
        amount = apply_rate_bp(taxable, rate_bp)
        if amount <= 0:
            return None
        extra = {
            "discount_bp": rate_bp,
            "window_days": contract.early_payment_window_days or 10,
        }
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=False, amount_paise=amount, recoverable_paise=amount,
            extra=extra, showcase_id=showcase_id,
        )

    # --------------------------------------------------------------- duplicate claim
    if code == "DUPLICATE_CLAIM":
        is_valid = valid_by_split(0.25)
        amount = forced_amount or round_to_rupee(int(taxable * rng.uniform(0.05, 0.30)))
        extra = {"asserted_utr": f"UTR{rng.randrange(10**11, 10**12)}"}
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=is_valid, amount_paise=amount,
            recoverable_paise=0 if is_valid else amount,
            extra=extra, showcase_id=showcase_id,
        )

    # ------------------------------------------------------------------- de minimis
    if code == "BANK_CHARGES":
        amount = round_to_rupee(rng.randint(2_000, 60_000))
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=True, amount_paise=amount, recoverable_paise=0,
            extra={}, showcase_id=showcase_id,
        )

    if code == "ROUNDING":
        amount = rng.randint(1, 999)  # under Rs 10
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=True, amount_paise=amount, recoverable_paise=0,
            extra={}, showcase_id=showcase_id,
        )

    # ------------------------------------------------------------------- unexplained
    if code == "UNEXPLAINED":
        amount = forced_amount or round_to_rupee(int(taxable * rng.uniform(0.005, 0.06)))
        if amount <= 0:
            return None
        return PlannedDeduction(
            ded_id, invoice.id, invoice.buyer_id, code,
            is_valid=False, amount_paise=amount, recoverable_paise=amount,
            extra={}, showcase_id=showcase_id,
        )

    return None
