"""Guaranteed demo scenarios.

The pitch video needs a case where the buyer *says* 2% and *deducts* 5%, a case where the
agent declines to act and names the rule, and a `propose_credit_hold` sitting unapproved
in a queue. Leaving those to the RNG means rehearsing a demo that a re-seed can silently
destroy — and the one thing worse than a weak demo is a demo that worked yesterday.

So the batch reserves a handful of invoices up front and forces the scenario onto them.
Each showcase declares what it needs from the invoice (an enterprise buyer, a contract
that charges TCS, an invoice that really was mis-billed), and the picker finds the first
match in deterministic order. If nothing matches, generation fails loudly rather than
quietly producing a batch the demo script does not fit.
"""

from __future__ import annotations

from typing import Any

from ..config import Taxonomy
from ..schemas import Buyer, Contract, Invoice, Segment
from .deductions import SECTION_194Q_THRESHOLD_PAISE, PlannedDeduction, plan_deduction
from .invoices import has_rate_error


def _matches(
    spec: dict[str, Any],
    invoice: Invoice,
    buyer: Buyer,
    contract: Contract,
    buyer_volume_paise: int,
) -> bool:
    """Can this scenario be staged on this invoice?"""
    reason = spec["reason"]

    if spec.get("force_segment") and buyer.segment != spec["force_segment"]:
        return False
    if spec.get("force_behaviour") and buyer.payment_behaviour_tag != spec["force_behaviour"]:
        return False

    # Feasibility constraints, identical to the ones deductions.py enforces. A showcase
    # cannot bend physics: there is no TCS overlap without TCS, and no 194Q below the
    # Rs 50L aggregate threshold.
    if reason == "TCS_194Q_OVERLAP":
        if not contract.tcs_applicable:
            return False
        if buyer_volume_paise < SECTION_194Q_THRESHOLD_PAISE:
            return False
    if reason == "RATE_DIFFERENCE" and not has_rate_error(invoice):
        return False

    # A legitimate withholding only occurs under the section the contract pins, so a
    # showcase asking for a specific section needs a buyer whose contract expects it.
    if reason in {"TDS_194C", "TDS_194J", "TDS_194H", "TDS_194Q"}:
        if contract.tds_section_expected != reason:
            return False

    if reason == "TDS_RATE_MISMATCH":
        # Needs headroom for a *higher* wrong rate to exist above the contracted one.
        if contract.tds_rate_expected_bp >= 1000:
            return False
    if reason == "SCHEME_REBATE" and invoice.taxable_paise < 20_000_000:
        return False
    if reason == "DAMAGE_SHORTAGE" and invoice.taxable_paise < 50_000_000:
        return False

    # A showcase whose amount is trivially small is not a showcase. The first version of
    # this staged the headline "says 2%, deducted 5%" case on a Rs 1,542 deduction and the
    # TCS overlap on Rs 38 — both economically beneath the write-off threshold, so the
    # correct agent behaviour would have been to ignore them, and the demo would have been
    # showing a case the policy engine throws away.
    min_amount = spec.get("min_deduction_paise")
    if min_amount and invoice.taxable_paise * _MIN_RATE_BP_FOR.get(reason, 100) // 10_000 < min_amount:
        return False

    if spec.get("force_amount_paise") and invoice.total_paise <= spec["force_amount_paise"] * 3:
        # Room for the forced deduction to be a plausible fraction of the invoice rather
        # than nearly all of it.
        return False
    return True


# Lowest rate each rate-driven code could plausibly apply, used to check a showcase can
# reach its minimum amount on a given invoice.
_MIN_RATE_BP_FOR = {
    "TCS_194Q_OVERLAP": 10,     # 0.10%
    "TDS_RATE_MISMATCH": 300,   # smallest realistic excess over the correct rate
    "RATE_DIFFERENCE": 300,
}


def reserve_showcases(
    seed: int,
    cfg: dict[str, Any],
    taxonomy: Taxonomy,
    invoices: list[Invoice],
    buyers: dict[str, Buyer],
    contracts: dict[str, Contract],
    buyer_volume: dict[str, int] | None = None,
) -> tuple[dict[str, PlannedDeduction], list[str]]:
    """Assign each showcase to an invoice.

    Returns `(deduction_by_invoice_id, reserved_invoice_ids)`. Reserved invoices are
    excluded from ordinary deduction planning so a showcase never has to share.
    """
    specs = cfg.get("showcase") or []
    volumes = buyer_volume or {}
    planned: dict[str, PlannedDeduction] = {}
    reserved: list[str] = []
    unmatched: list[str] = []

    for spec in specs:
        chosen: Invoice | None = None
        # Largest invoices first, so a showcase lands on a case big enough to be worth
        # demonstrating rather than the first arbitrary match in id order.
        for invoice in sorted(invoices, key=lambda i: -i.taxable_paise):
            if invoice.id in reserved:
                continue
            buyer = buyers[invoice.buyer_id]
            contract = contracts[invoice.buyer_id]
            if _matches(spec, invoice, buyer, contract, volumes.get(invoice.buyer_id, 0)):
                chosen = invoice
                break

        if chosen is None:
            unmatched.append(spec["id"])
            continue

        contract = contracts[chosen.buyer_id]
        deduction = plan_deduction(
            seed,
            taxonomy,
            chosen,
            contract,
            spec["reason"],
            ordinal=0,
            forced_valid=spec.get("valid"),
            forced_amount=spec.get("force_amount_paise"),
            showcase_id=spec["id"],
            validity_split=cfg.get("validity_split"),
            buyer_volume_paise=volumes.get(chosen.buyer_id, 0),
        )
        if deduction is None:
            unmatched.append(spec["id"])
            continue

        if spec.get("force_26as_absent"):
            deduction.extra["force_26as_absent"] = True

        planned[chosen.id] = deduction
        reserved.append(chosen.id)

    if unmatched:
        raise RuntimeError(
            "could not stage showcase scenario(s): "
            + ", ".join(unmatched)
            + ". Either widen the batch (--n), relax the constraint in config/generator.yaml, "
            "or drop the scenario. Failing loudly here beats discovering it mid-demo."
        )

    return planned, reserved


def showcase_summary(planned: dict[str, PlannedDeduction]) -> list[dict[str, Any]]:
    """For the generation report, so you can see which case id to point the demo at."""
    return [
        {
            "showcase_id": p.showcase_id,
            "deduction_id": p.id,
            "invoice_id": p.invoice_id,
            "code": p.code,
            "valid": p.is_valid,
            "amount_paise": p.amount_paise,
            "recoverable_paise": p.recoverable_paise,
        }
        for p in sorted(planned.values(), key=lambda x: x.showcase_id or "")
    ]


# Segment constant re-exported so config-driven specs can reference it in tests without
# importing schemas directly.
ENTERPRISE = Segment.ENTERPRISE.value
