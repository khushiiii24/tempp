"""Supporting evidence stores — the "source systems" the verification layer reads.

Every row here is a *projection of the truth record*, which is what makes verification a
real check. A credit-note offset is invalid precisely because this module declined to
write the ledger row; an expired scheme is expired because the period written here ends
before the invoice date. Nothing about the invalidity is visible in the payment or the
advice text. You have to go and look, which is the point.

**Some fixtures are deliberately incomplete.** Form 26AS lags by a quarter, so roughly
15% of entirely legitimate TDS deductions are simply not visible yet. This is the single
most important piece of realism in the batch, because it separates two agents that look
identical on every other case:

* the naive agent sees "claimed TDS, no 26AS row" and chases a customer who did nothing
  wrong;
* the correct agent recognises the lag, **provisionally closes and flags for re-check**,
  and chases nobody.

The metrics reward the second explicitly. An absent fixture row is not evidence of
absence, and an agent that cannot tell the difference between "disproved" and "not yet
knowable" is not safe to point at a customer base.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..clock import add_days, parse_date
from ..money import paise_to_rupees_str
from ..schemas import Buyer, Contract, Invoice
from .deductions import PlannedDeduction
from .seed import chance, rng_for, sample_range

SELLER_PAN = "AAACD1234K"
SELLER_GSTIN = "27AAACD1234K1Z9"
# Deliberately not the product name: this is a party inside the generated data, and
# changing the string changes every fixture that embeds it — and the batch hash with it.
SELLER_NAME = "DeductionDesk Demo Seller Pvt Ltd"


def _quarter_of(date_text: str) -> str:
    """Indian financial-year quarter label, e.g. 'Q1 FY2026-27' for April-June 2026."""
    d = parse_date(date_text)
    fy_start = d.year if d.month >= 4 else d.year - 1
    q = ((d.month - 4) % 12) // 3 + 1
    return f"Q{q} FY{fy_start}-{str(fy_start + 1)[-2:]}"


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" and a fixed line terminator so the file is byte-identical on every
    # platform — otherwise Windows writes \r\n, the determinism hash differs from CI, and
    # you spend an afternoon on it.
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_fixtures(
    seed: int,
    cfg: dict[str, Any],
    out_dir: Path,
    *,
    planned: list[PlannedDeduction],
    invoices: dict[str, Invoice],
    buyers: dict[str, Buyer],
    contracts: dict[str, Contract],
) -> dict[str, int]:
    """Write all seven fixture stores. Returns row counts for the generation report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fx = cfg["fixtures"]

    # Aggregate purchase value per buyer, used for scheme volume slabs.
    buyer_volume: dict[str, int] = {}
    for inv in invoices.values():
        buyer_volume[inv.buyer_id] = buyer_volume.get(inv.buyer_id, 0) + int(inv.taxable_paise)

    # Earliest and latest invoice date each buyer raises a scheme claim against, so the
    # scheme period can be written to cover all of them coherently.
    buyer_claim_window: dict[str, tuple[str, str]] = {}
    for p in planned:
        if p.code != "SCHEME_REBATE":
            continue
        issue = invoices[p.invoice_id].issue_date
        first, last = buyer_claim_window.get(p.buyer_id, (issue, issue))
        buyer_claim_window[p.buyer_id] = (min(first, issue), max(last, issue))

    form_26as: list[dict[str, Any]] = []
    gstr7: list[dict[str, Any]] = []
    credit_notes: list[dict[str, Any]] = []
    schemes: list[dict[str, Any]] = []
    grn: list[dict[str, Any]] = []
    payment_history: list[dict[str, Any]] = []
    lag_flags: dict[str, bool] = {}
    seen_schemes: set[str] = set()

    for p in planned:
        invoice = invoices[p.invoice_id]
        buyer = buyers[p.buyer_id]
        rng = rng_for(seed, "fixture", p.id)

        # ------------------------------------------------------------------ TDS / 26AS
        if p.code in {"TDS_194C", "TDS_194J", "TDS_194H", "TDS_194Q", "TDS_RATE_MISMATCH", "TCS_194Q_OVERLAP"}:
            # The lag is forced on for the showcase case so the demo always has one.
            lagged = (
                True
                if p.extra.get("force_26as_absent")
                else chance(rng, float(fx["form_26as_lag_rate"]))
            )
            lag_flags[p.id] = lagged
            if not lagged:
                section = p.extra.get("section") or "194Q"
                form_26as.append(
                    {
                        "deductee_pan": SELLER_PAN,
                        "deductor_gstin": buyer.gstin,
                        "deductor_name": buyer.name,
                        "section": section,
                        "invoice_no": invoice.invoice_no,
                        # 26AS reports what the buyer ACTUALLY deposited, including when
                        # that was computed at the wrong rate. That is the evidence.
                        "amount_deducted_inr": paise_to_rupees_str(p.amount_paise),
                        "base_amount_inr": paise_to_rupees_str(p.extra.get("base_paise", 0)),
                        "rate_pct": f"{p.extra.get('applied_rate_bp', p.extra.get('rate_bp', 0)) / 100:.2f}",
                        "quarter": _quarter_of(invoice.issue_date),
                        "status": "F",  # final
                    }
                )

        # ------------------------------------------------------------------ GST TDS
        if p.code == "GST_TDS":
            lagged = chance(rng, float(fx["gstr7_lag_rate"]))
            lag_flags[p.id] = lagged
            if not lagged:
                gstr7.append(
                    {
                        "deductee_gstin": SELLER_GSTIN,
                        "deductor_gstin": buyer.gstin,
                        "invoice_no": invoice.invoice_no,
                        "amount_deducted_inr": paise_to_rupees_str(p.amount_paise),
                        "taxable_value_inr": paise_to_rupees_str(invoice.taxable_paise),
                        "period": _quarter_of(invoice.issue_date),
                    }
                )

        # ------------------------------------------------------------- credit notes
        if p.code == "CREDIT_NOTE_OFFSET":
            failure = p.extra.get("failure_mode")
            if failure != "not_found":
                # A valid offset has an unapplied CN. 'already_applied' writes a real row
                # that has been consumed elsewhere — the buyer is double-dipping, which is
                # only findable by checking the applied flag.
                credit_notes.append(
                    {
                        "credit_note_no": p.extra["credit_note_no"],
                        "buyer_id": buyer.id,
                        "buyer_name": buyer.name,
                        "amount_inr": paise_to_rupees_str(p.amount_paise),
                        "issued_date": add_days(invoice.issue_date, -sample_range(rng, [5, 60])),
                        "applied": "Y" if failure == "already_applied" else "N",
                        "applied_against": (
                            f"INV/{parse_date(invoice.issue_date).year}/"
                            f"{rng.randrange(0, 9999):04d}"
                            if failure == "already_applied"
                            else ""
                        ),
                        "reason": "quality claim settlement",
                    }
                )

        # ------------------------------------------------------------------ schemes
        # One row per scheme, not per claim. Several invoices can be claimed under the
        # same scheme, and writing a row each time would put contradictory terms for the
        # same scheme id into the master.
        if p.code == "SCHEME_REBATE" and p.extra["scheme_id"] not in seen_schemes:
            seen_schemes.add(p.extra["scheme_id"])
            failure = p.extra.get("failure_mode")
            # The scheme period must be consistent with EVERY claim this buyer makes under
            # it, not just the first one encountered. Deriving it from a single invoice
            # meant a live scheme could fail to cover the buyer's other claim, and the
            # verifier would call a legitimate rebate expired.
            first_claim, last_claim = buyer_claim_window.get(
                buyer.id, (invoice.issue_date, invoice.issue_date)
            )
            if failure == "expired":
                # Closed before the buyer's earliest claim, so every claim under it is late.
                period_start = add_days(first_claim, -sample_range(rng, [150, 300]))
                period_end = add_days(first_claim, -sample_range(rng, [20, 90]))
            else:
                # Live across the buyer's whole claim window, with margin either side.
                period_start = add_days(first_claim, -sample_range(rng, [20, 80]))
                period_end = add_days(last_claim, sample_range(rng, [20, 90]))

            # A quarterly purchase scheme is measured on the buyer's AGGREGATE purchases,
            # not on the one invoice the claim happens to sit against. Using a single
            # invoice would make the qualifying volume incoherent the moment a buyer
            # claimed the same scheme twice.
            achieved = buyer_volume.get(buyer.id, int(invoice.taxable_paise))
            min_volume = (
                int(achieved * rng.uniform(1.6, 3.2))  # slab never met
                if failure == "slab_unmet"
                else int(achieved * rng.uniform(0.25, 0.85))
            )
            schemes.append(
                {
                    "scheme_id": p.extra["scheme_id"],
                    "buyer_id": buyer.id,
                    "period_start": period_start,
                    "period_end": period_end,
                    "slab_pct": f"{p.extra['slab_bp'] / 100:.2f}",
                    "min_volume_inr": paise_to_rupees_str(min_volume),
                    "achieved_volume_inr": paise_to_rupees_str(achieved),
                    "sku_scope": "ALL",
                }
            )

        # ---------------------------------------------------------------------- GRN
        if p.code in {"DAMAGE_SHORTAGE", "QUALITY_REJECTION", "DEBIT_NOTE_BUYER"}:
            if p.is_valid:
                grn.append(
                    {
                        "grn_no": p.extra["grn_no"],
                        "invoice_no": invoice.invoice_no,
                        "buyer_id": buyer.id,
                        "sku": p.extra.get("sku") or "",
                        "received_date": add_days(invoice.issue_date, sample_range(rng, [2, 12])),
                        "discrepancy_type": {
                            "DAMAGE_SHORTAGE": "shortage",
                            "QUALITY_REJECTION": "quality_reject",
                            "DEBIT_NOTE_BUYER": "damage",
                        }[p.code],
                        "value_inr": paise_to_rupees_str(p.amount_paise),
                        "debit_note_no": p.extra.get("debit_note_no") or "",
                    }
                )

        # ---------------------------------------------------------- payment history
        if p.code == "DUPLICATE_CLAIM" and p.is_valid:
            # The buyer is right: there really is an earlier payment against this invoice.
            payment_history.append(
                {
                    "utr": p.extra["asserted_utr"],
                    "buyer_id": buyer.id,
                    "invoice_no": invoice.invoice_no,
                    "amount_inr": paise_to_rupees_str(p.amount_paise),
                    "value_date": add_days(invoice.issue_date, sample_range(rng, [5, 40])),
                    "source": "bank_statement",
                }
            )

    # ------------------------------------------------------------------ contracts.json
    contracts_payload = {
        "seller": {"name": SELLER_NAME, "pan": SELLER_PAN, "gstin": SELLER_GSTIN},
        "contracts": [
            {
                "contract_id": c.id,
                "buyer_id": c.buyer_id,
                "buyer_name": buyers[c.buyer_id].name,
                "delivery_terms": c.delivery_terms,
                "freight_borne_by": c.freight_borne_by,
                "payment_terms_days": c.payment_terms_days,
                "early_payment_discount_pct": c.early_payment_discount_bp / 100,
                "early_payment_window_days": c.early_payment_window_days,
                "tds_section_expected": c.tds_section_expected,
                "tcs_applicable": c.tcs_applicable,
                "rate_card_inr": {
                    sku: paise_to_rupees_str(v) for sku, v in sorted(c.rate_card.items())
                },
            }
            for c in sorted(contracts.values(), key=lambda x: x.id)
        ],
    }

    (out_dir / "contracts.json").write_text(
        json.dumps(contracts_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Sorting every fixture by its natural key keeps the files stable across runs even if
    # the planning order changes — a diff between two generations then shows real changes.
    _write_csv(
        out_dir / "form_26as.csv",
        sorted(form_26as, key=lambda r: (r["invoice_no"], r["section"])),
        ["deductee_pan", "deductor_gstin", "deductor_name", "section", "invoice_no",
         "amount_deducted_inr", "base_amount_inr", "rate_pct", "quarter", "status"],
    )
    _write_csv(
        out_dir / "gstr7.csv",
        sorted(gstr7, key=lambda r: r["invoice_no"]),
        ["deductee_gstin", "deductor_gstin", "invoice_no", "amount_deducted_inr",
         "taxable_value_inr", "period"],
    )
    _write_csv(
        out_dir / "credit_note_ledger.csv",
        sorted(credit_notes, key=lambda r: r["credit_note_no"]),
        ["credit_note_no", "buyer_id", "buyer_name", "amount_inr", "issued_date",
         "applied", "applied_against", "reason"],
    )
    _write_csv(
        out_dir / "scheme_master.csv",
        sorted(schemes, key=lambda r: r["scheme_id"]),
        ["scheme_id", "buyer_id", "period_start", "period_end", "slab_pct",
         "min_volume_inr", "achieved_volume_inr", "sku_scope"],
    )
    _write_csv(
        out_dir / "grn_discrepancies.csv",
        sorted(grn, key=lambda r: r["grn_no"]),
        ["grn_no", "invoice_no", "buyer_id", "sku", "received_date", "discrepancy_type",
         "value_inr", "debit_note_no"],
    )
    _write_csv(
        out_dir / "payment_history.csv",
        sorted(payment_history, key=lambda r: r["utr"]),
        ["utr", "buyer_id", "invoice_no", "amount_inr", "value_date", "source"],
    )

    return {
        "form_26as": len(form_26as),
        "gstr7": len(gstr7),
        "credit_note_ledger": len(credit_notes),
        "scheme_master": len(schemes),
        "grn_discrepancies": len(grn),
        "payment_history": len(payment_history),
        "contracts": len(contracts_payload["contracts"]),
        "tds_lagged": sum(1 for v in lag_flags.values() if v),
    }
