"""Stage [3]: isolate the shortfall on each matched invoice.

`delta = invoice.total_paise - sum(allocations)`. Arithmetic, not inference.

Two subtleties decide whether the number downstream means anything.

**A delta is not yet a deduction.** An invoice paid across two UTRs three weeks apart shows
a large delta after the first credit and none after the second. Treating that as a short
payment on day one means chasing money that was always going to arrive, and it would
inflate the false-chase count that the whole harm argument rests on. So a delta becomes
chaseable only once `settlement.treat_as_short_after_days` has elapsed past the due date
with no further credit — the `awaiting_settlement` state.

**A single delta can be several deductions.** A ₹9,600 gap may be ₹2,000 of TDS and ₹7,600
of freight, and those have different verdicts, different owners and different actions.
Where the remittance advice itemises the components, the delta is split accordingly;
where it does not, one lump deduction is the honest representation of what is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..clock import days_between
from ..schemas import Allocation, Invoice


@dataclass
class InvoiceDelta:
    """The shortfall on one invoice, and whether it is ripe to act on."""

    invoice_id: str
    invoice_no: str
    buyer_id: str
    total_paise: int
    allocated_paise: int
    payment_event_ids: list[str] = field(default_factory=list)
    last_value_date: str = ""

    @property
    def delta_paise(self) -> int:
        return max(0, self.total_paise - self.allocated_paise)

    @property
    def fully_settled(self) -> bool:
        return self.delta_paise == 0

    @property
    def overpaid(self) -> bool:
        return self.allocated_paise > self.total_paise

    def days_past_due(self, invoice: Invoice, today: str) -> int:
        return days_between(invoice.due_date, today)

    def is_chaseable(self, invoice: Invoice, today: str, grace_days: int) -> bool:
        """Has enough time passed to call this a short payment rather than a part payment?"""
        return self.delta_paise > 0 and self.days_past_due(invoice, today) >= grace_days


def isolate_deltas(
    allocations: list[Allocation],
    invoices: dict[str, Invoice],
    payment_dates: dict[str, str],
) -> dict[str, InvoiceDelta]:
    """Aggregate allocations per invoice and compute the shortfall.

    Aggregating across *all* payments for an invoice is the point: a split payment nets to
    zero delta once both tranches land, and only aggregation makes that visible.
    """
    deltas: dict[str, InvoiceDelta] = {}

    for alloc in sorted(allocations, key=lambda a: a.id):
        invoice = invoices.get(alloc.invoice_id)
        if invoice is None:
            continue

        delta = deltas.get(alloc.invoice_id)
        if delta is None:
            delta = InvoiceDelta(
                invoice_id=invoice.id,
                invoice_no=invoice.invoice_no,
                buyer_id=invoice.buyer_id,
                total_paise=int(invoice.total_paise),
                allocated_paise=0,
            )
            deltas[alloc.invoice_id] = delta

        delta.allocated_paise += int(alloc.allocated_paise)
        delta.payment_event_ids.append(alloc.payment_event_id)
        value_date = payment_dates.get(alloc.payment_event_id, "")
        if value_date > delta.last_value_date:
            delta.last_value_date = value_date

    return deltas


def reconciles_with(delta: InvoiceDelta, deduction_total_paise: int, tolerance: int = 100) -> bool:
    """Does the derived shortfall agree with the itemised deductions for this invoice?

    The honest measure of whether matching worked. A high match rate paired with deltas
    that do not reconcile means invoices were matched to the wrong payments — which looks
    like success on every metric except this one.
    """
    return abs(delta.delta_paise - deduction_total_paise) <= tolerance


def summarise(
    deltas: dict[str, InvoiceDelta],
    invoices: dict[str, Invoice],
    today: str,
    grace_days: int,
) -> dict[str, Any]:
    ripe = [
        d for d in deltas.values()
        if d.is_chaseable(invoices[d.invoice_id], today, grace_days)
    ]
    awaiting = [
        d for d in deltas.values()
        if d.delta_paise > 0
        and not d.is_chaseable(invoices[d.invoice_id], today, grace_days)
    ]

    return {
        "invoices_with_allocations": len(deltas),
        "fully_settled": sum(1 for d in deltas.values() if d.fully_settled),
        "with_shortfall": sum(1 for d in deltas.values() if d.delta_paise > 0),
        "chaseable_now": len(ripe),
        "awaiting_settlement": len(awaiting),
        "total_delta_paise": sum(d.delta_paise for d in deltas.values()),
        "chaseable_delta_paise": sum(d.delta_paise for d in ripe),
    }
