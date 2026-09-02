"""Invoices: line items, tax stack, and the seller's own billing errors.

Two details that the rest of the pipeline depends on:

**The tax stack is built in the order Indian invoicing actually uses** — taxable value,
then GST on the taxable value, then TCS u/s 206C(1H) on the *invoice value including GST*.
Getting that order wrong would not look like a bug; it would look like every TCS-overlap
case being misclassified, because the classifier's strongest signal is the ratio of the
deduction to the taxable value and the ratio would be subtly off.

**~5% of invoices are deliberately billed above the contracted rate.** This is the seed
for `RATE_DIFFERENCE`, and it is the one deduction type where the buyer is right and the
seller is wrong. An agent that treats every deduction as something to recover will chase
its own billing error, annoy a customer who did nothing wrong, and lose. The correct
action is to issue a credit note without being asked.
"""

from __future__ import annotations

import math
from typing import Any

from ..clock import add_days, date_str, parse_date
from ..money import Paise, apply_rate_bp
from ..schemas import Contract, Invoice
from .contracts import SKU_DESCRIPTION
from .seed import chance, rng_for, sample_range

GST_RATE_BP = 1800  # 18% — the standard rate for this catalogue
TCS_206C_RATE_BP = 10  # 0.10%

# Share of invoices where the seller bills above the contracted rate. Small, because a
# seller that mis-bills 20% of the time has a bigger problem than deductions.
RATE_ERROR_SHARE = 0.05

# Order size classes, as (weight, taxable-value range in rupees).
#
# An earlier version drew 1-5 lines at random quantities and produced a distribution that
# topped out at Rs 1.66L with a median of Rs 35k — and a minimum of Rs 34, which is not an
# invoice, it is a rounding error. That distribution quietly broke three things: TDS 194Q
# only applies to purchases above Rs 50L and so could never legitimately appear; the
# Rs 50,000 human-review threshold would essentially never fire; and no case was ever large
# enough for a credit-hold conversation to be plausible.
#
# Real B2B invoice-to-cash is heavy-tailed: routine replenishment orders alongside
# occasional bulk consignments. Sampling a target value log-uniformly within a class and
# then solving for quantities gives direct control over that shape, rather than hoping it
# emerges from quantity draws.
ORDER_SIZE_CLASSES: tuple[tuple[str, float, int, int, tuple[int, int]], ...] = (
    # (name, weight, target_low_rupees, target_high_rupees, (min_lines, max_lines))
    ("small",  0.45,     25_000,    150_000, (1, 3)),
    ("medium", 0.33,    150_000,    900_000, (2, 5)),
    ("large",  0.17,    900_000,  4_500_000, (3, 6)),
    ("bulk",   0.05,  4_500_000, 16_000_000, (2, 4)),
)


def _invoice_no(index: int, issue: str) -> str:
    """`INV/2026/0042`. The canonical form; `noise.py` mangles it for bank narrations."""
    return f"INV/{parse_date(issue).year}/{index:04d}"


def _draw_order_size(rng) -> tuple[str, int, tuple[int, int]]:
    """Choose a size class and a target taxable value within it.

    Log-uniform within the class so the mass sits toward the lower end of each band —
    which is how order books actually look — rather than spreading evenly and
    over-representing the expensive end.
    """
    names = [c[0] for c in ORDER_SIZE_CLASSES]
    weights = [c[1] for c in ORDER_SIZE_CLASSES]
    idx = rng.choices(range(len(names)), weights=weights, k=1)[0]
    _, _, low_rupees, high_rupees, line_bounds = ORDER_SIZE_CLASSES[idx]

    log_low, log_high = math.log(low_rupees), math.log(high_rupees)
    target_rupees = math.exp(rng.uniform(log_low, log_high))
    return names[idx], int(target_rupees) * 100, line_bounds


def build_invoices(
    seed: int,
    cfg: dict[str, Any],
    buyer_ids: list[str],
    contracts: dict[str, Contract],
) -> list[Invoice]:
    n = int(cfg["batch"]["n_invoices"])
    start = cfg["batch"]["start_date"]
    span = int(cfg["batch"]["invoice_span_days"])

    invoices: list[Invoice] = []

    for i in range(n):
        inv_id = f"INV-{i:04d}"
        rng = rng_for(seed, "invoice", inv_id)

        buyer_id = buyer_ids[rng.randrange(len(buyer_ids))]
        contract = contracts[buyer_id]

        issue = add_days(start, sample_range(rng, [0, span]))
        due = add_days(issue, contract.payment_terms_days)

        # Pick an order size class, then a target value inside it, then solve for
        # quantities. Doing it in this order is what gives the batch a realistic heavy
        # tail instead of a narrow band clustered around one arbitrary mode.
        size_name, target_paise, line_bounds = _draw_order_size(rng)

        # Bill from the contract's rate card, which is what makes a rate difference
        # detectable — an invoice for a SKU with no contracted price is not adjudicable.
        card_skus = sorted(contract.rate_card)
        n_lines = rng.randint(line_bounds[0], min(line_bounds[1], len(card_skus)))
        chosen = rng.sample(card_skus, n_lines)

        mis_bills = chance(rng, RATE_ERROR_SHARE)
        # If the seller mis-bills, it is one line, not the whole invoice — that is how it
        # happens in practice and it makes the arithmetic a partial recovery, not a total.
        mis_billed_sku = chosen[0] if mis_bills else None

        # Split the target across lines using random weights, so a five-line invoice is
        # not five equal lines.
        weights = [rng.uniform(0.5, 1.5) for _ in chosen]
        weight_total = sum(weights)

        line_items: list[dict[str, Any]] = []
        taxable = 0
        for sku, weight in zip(chosen, weights, strict=True):
            rate = int(contract.rate_card[sku])
            share = target_paise * weight / weight_total
            qty = max(1, round(share / rate))
            billed_rate = rate
            if sku == mis_billed_sku:
                # 3-12% above contract: large enough for the buyer's AP to notice, small
                # enough to be a plausible stale-price-master error rather than fraud.
                billed_rate = int(rate * rng.uniform(1.03, 1.12))
            line_total = billed_rate * qty
            taxable += line_total
            line_items.append(
                {
                    "sku": sku,
                    "description": SKU_DESCRIPTION[sku],
                    "qty": qty,
                    "unit_rate_paise": billed_rate,
                    "contracted_rate_paise": rate,
                    "line_total_paise": line_total,
                }
            )

        gst = apply_rate_bp(taxable, GST_RATE_BP)
        # TCS u/s 206C(1H) is collected on the invoice value including GST. Only where the
        # contract says the seller is the one collecting — this flag is what creates the
        # 194Q/TCS overlap cases later.
        tcs = apply_rate_bp(taxable + gst, TCS_206C_RATE_BP) if contract.tcs_applicable else 0

        invoices.append(
            Invoice(
                id=inv_id,
                invoice_no=_invoice_no(i, issue),
                buyer_id=buyer_id,
                issue_date=date_str(parse_date(issue)),
                due_date=date_str(parse_date(due)),
                taxable_paise=Paise(taxable),
                gst_paise=Paise(gst),
                tcs_paise=Paise(tcs),
                total_paise=Paise(taxable + gst + tcs),
                line_items=line_items,
                status="open",
            )
        )

    return invoices


def rate_difference_paise(invoice: Invoice) -> int:
    """How much of this invoice was billed above contract. Zero for a correct invoice.

    Used by the generator to size a `RATE_DIFFERENCE` deduction and, independently, by
    `verify/contract.py` to adjudicate one. Both sides compute it the same way from the
    stored line items, so a verified rate difference reconciles to the paise.
    """
    excess = 0
    for line in invoice.line_items:
        billed = int(line["unit_rate_paise"])
        contracted = int(line.get("contracted_rate_paise", billed))
        if billed > contracted:
            excess += (billed - contracted) * int(line["qty"])
    return excess


def has_rate_error(invoice: Invoice) -> bool:
    return rate_difference_paise(invoice) > 0
