"""Stage [2]: match each payment to the invoice(s) it settles.

A deterministic ladder, cheapest and safest first. Every rung either resolves or hands
down; nothing guesses:

1. **Advice reference** — the remittance advice names the invoice. Highest confidence.
2. **Exact / normalised narration** — the reference survives the bank's mangling.
3. **Fuzzy narration** — rapidfuzz against known reference variants, above a threshold.
4. **Subset-sum on amount** — reconcile against the buyer's open invoices.

**Reporting what could not be matched is worth more than a fabricated match.** A wrong
allocation posts real money against the wrong invoice and produces a plausible-looking
deduction that never existed; every number downstream is then wrong with nothing to
indicate it. So the exceptions report is a first-class output of this stage, and the
acceptance test asserts zero fabricated matches — not merely a high match rate, which any
sufficiently reckless matcher achieves.

Buyer resolution happens first and constrains everything after it. Bank narrations carry a
truncated payer name, and knowing the buyer shrinks the candidate set from four hundred
invoices to a dozen, which is what makes both fuzzy matching and subset-sum tractable and
safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz, process

from ..ingest.normalize import (
    MIN_FUZZY_LENGTH,
    candidate_refs,
    declared_bundle_size,
    fuzzy_candidates,
    narration_name_fragment,
    normalise_ref,
    payer_key,
    ref_variants,
    truncation_suspect,
)
from ..money import Paise
from ..schemas import Allocation, Buyer, Invoice, MatchMethod, PaymentEvent, RemittanceAdvice
from .subset_sum import reconcile

# rapidfuzz score (0-100) below which a narration match is not trusted.
FUZZY_REF_THRESHOLD = 88
FUZZY_NAME_THRESHOLD = 80


@dataclass
class MatchOutcome:
    """What happened to one payment."""

    payment_id: str
    buyer_id: str | None = None
    invoice_ids: list[str] = field(default_factory=list)
    method: str = MatchMethod.MANUAL.value
    confidence: float = 0.0
    exception: str | None = None
    detail: str = ""

    @property
    def matched(self) -> bool:
        return bool(self.invoice_ids) and self.exception is None


@dataclass
class MatchIndex:
    """Pre-built lookups. Constructed once per run; the ladder is a pure function of it."""

    buyers: dict[str, Buyer]
    invoices: dict[str, Invoice]
    invoices_by_buyer: dict[str, list[str]]
    variant_to_invoice: dict[str, str]
    ambiguous_variants: set[str]
    buyer_keys: dict[str, str]  # payer_key -> buyer_id

    @classmethod
    def build(cls, buyers: list[Buyer], invoices: list[Invoice]) -> MatchIndex:
        by_buyer: dict[str, list[str]] = {}
        variant_map: dict[str, str] = {}
        ambiguous: set[str] = set()

        for inv in sorted(invoices, key=lambda i: i.id):
            by_buyer.setdefault(inv.buyer_id, []).append(inv.id)
            for variant in ref_variants(inv.invoice_no):
                if variant in variant_map and variant_map[variant] != inv.id:
                    # Two invoices share this form. It can never identify either of them,
                    # so it is retired rather than resolving to whichever was seen first.
                    ambiguous.add(variant)
                else:
                    variant_map[variant] = inv.id

        for variant in ambiguous:
            variant_map.pop(variant, None)

        return cls(
            buyers={b.id: b for b in buyers},
            invoices={i.id: i for i in invoices},
            invoices_by_buyer=by_buyer,
            variant_to_invoice=variant_map,
            ambiguous_variants=ambiguous,
            buyer_keys={payer_key(b.name): b.id for b in buyers},
        )

    def open_invoices_for(self, buyer_id: str) -> list[tuple[str, int]]:
        return [
            (iid, int(self.invoices[iid].total_paise))
            for iid in self.invoices_by_buyer.get(buyer_id, [])
        ]


# ----------------------------------------------------------------------------------
# Buyer resolution
# ----------------------------------------------------------------------------------
def resolve_buyer(payment: PaymentEvent, index: MatchIndex) -> tuple[str | None, float]:
    """Which buyer sent this payment?

    The narration carries a truncated, space-stripped payer name. An exact prefix match
    against a known buyer key is conclusive; otherwise rapidfuzz decides, and below the
    threshold we return None rather than picking the closest.
    """
    fragment = narration_name_fragment(payment.narration_raw)
    if not fragment:
        return None, 0.0

    prefix_hits = [bid for key, bid in index.buyer_keys.items() if key.startswith(fragment)]
    if len(prefix_hits) == 1:
        return prefix_hits[0], 1.0
    if len(prefix_hits) > 1:
        # The truncation is too short to identify one buyer. Fall through to fuzzy, which
        # scores the full key rather than just the prefix.
        pass

    match = process.extractOne(
        fragment, list(index.buyer_keys.keys()), scorer=fuzz.partial_ratio
    )
    if match and match[1] >= FUZZY_NAME_THRESHOLD:
        return index.buyer_keys[match[0]], match[1] / 100.0

    return None, 0.0


# ----------------------------------------------------------------------------------
# The ladder
# ----------------------------------------------------------------------------------
def match_payment(
    payment: PaymentEvent,
    index: MatchIndex,
    *,
    advice: RemittanceAdvice | None = None,
    advice_refs: list[str] | None = None,
    tolerance_paise: int = 100,
) -> MatchOutcome:
    """Run the ladder for one payment."""
    buyer_id, buyer_confidence = resolve_buyer(payment, index)
    outcome = MatchOutcome(payment_id=payment.id, buyer_id=buyer_id)

    if buyer_id is None:
        outcome.exception = "unresolved_buyer"
        outcome.detail = f"could not identify the payer from narration: {payment.narration_raw!r}"
        return outcome

    buyer_invoice_ids = set(index.invoices_by_buyer.get(buyer_id, []))

    # -- rung 1: the advice names the invoice --------------------------------------
    if advice_refs:
        hits: list[str] = []
        for ref in advice_refs:
            iid = index.variant_to_invoice.get(normalise_ref(ref))
            if iid and iid in buyer_invoice_ids:
                hits.append(iid)
        if hits:
            outcome.invoice_ids = sorted(set(hits))
            outcome.method = MatchMethod.ADVICE.value
            outcome.confidence = 0.99
            outcome.detail = f"remittance advice named {len(outcome.invoice_ids)} invoice(s)"
            return outcome

    tokens = candidate_refs(payment.narration_raw)
    buyer_variants = {
        v for v, iid in index.variant_to_invoice.items() if iid in buyer_invoice_ids
    }

    # -- truncation check, BEFORE matching -------------------------------------------
    # A token that is a strict prefix of a longer known reference is evidence the bank cut
    # the number short. It identifies nothing, and it must not be allowed to match — even
    # exactly. `INV20260+3MORE` is a seven-invoice bundle whose reference list did not fit;
    # the surviving fragment happens to equal a short invoice's variant, and matching on it
    # posted a bundle worth Rs 54,89,008 against a Rs 1,45,316 invoice. Discarding the
    # token costs a match; trusting it fabricates one.
    truncated = sorted(t for t in tokens if truncation_suspect(t, buyer_variants))
    usable = tokens - set(truncated)

    # -- rung 2: exact / normalised narration reference -----------------------------
    exact_hits = {
        index.variant_to_invoice[t]
        for t in usable
        if t in index.variant_to_invoice and index.variant_to_invoice[t] in buyer_invoice_ids
    }

    # A narration ending `+5MORE` is the bank telling us the reference list did not fit.
    # It names one invoice and admits to five it dropped. Taking the named one and stopping
    # allocates the entire credit to it — capped at its own total, so the rest of the money
    # simply vanishes and five invoices are never matched at all. That was 97 of 400
    # invoices left unallocated.
    #
    # The count is usable evidence: we know exactly how many invoices to look for, which
    # turns an open-ended subset-sum into a tightly constrained one.
    declared = declared_bundle_size(payment.narration_raw)

    if exact_hits and (declared is None or len(exact_hits) >= declared):
        outcome.invoice_ids = sorted(exact_hits)
        outcome.method = MatchMethod.NORMALISED.value
        outcome.confidence = 0.97
        outcome.detail = "invoice reference recovered from the narration"
        return outcome

    if exact_hits and declared:
        # Seed the search with what we know and find the remainder by amount.
        anchor = sorted(exact_hits)
        anchored_total = sum(int(index.invoices[i].total_paise) for i in anchor)
        others = [
            (iid, int(index.invoices[iid].total_paise))
            for iid in sorted(buyer_invoice_ids - set(anchor))
        ]
        bundle = reconcile(
            max(0, int(payment.amount_paise) - anchored_total),
            others,
            tolerance_paise=tolerance_paise,
            max_subset_size=max(1, declared - len(anchor)),
        )
        if bundle.resolved:
            outcome.invoice_ids = sorted(set(anchor) | set(bundle.invoice_ids))
            outcome.method = MatchMethod.SUBSET_SUM.value
            outcome.confidence = 0.85
            outcome.detail = (
                f"narration declared {declared} invoices and named {len(anchor)}; "
                f"the remaining {len(bundle.invoice_ids)} reconciled by amount"
            )
            return outcome

        # Could not complete the bundle. The named invoice alone is not the answer — the
        # credit is far larger than it — so this is an exception rather than a partial match.
        outcome.exception = "incomplete_bundle"
        outcome.detail = (
            f"narration declares {declared} invoices, only {len(anchor)} recoverable, and "
            f"the balance does not reconcile to any combination"
        )
        return outcome

    # -- rung 3: fuzzy on the narration ---------------------------------------------
    # Only long-enough tokens against long-enough variants. Short strings are similar to
    # everything: `INV/2026/0003` yields the variant `20263`, and the bare year `2026`
    # scores 88.9 against it.
    fuzzy_tokens = fuzzy_candidates(usable)
    # **Sorted, not set order.** `buyer_variants` is a set, and Python randomises string
    # hashing per process, so iterating it directly gives a different order in every run.
    # When two variants tie on fuzzy score, `extractOne` returns whichever it saw first —
    # so the same batch matched to different invoices from one process to the next, and
    # the measured "money located" swung between 78% and 98% with an identical database
    # content hash. Non-determinism here is invisible: every individual run looks fine.
    fuzzy_variants = sorted(v for v in buyer_variants if len(v) >= MIN_FUZZY_LENGTH)

    if fuzzy_variants and fuzzy_tokens:
        best: tuple[str, float] | None = None
        for token in fuzzy_tokens:
            hit = process.extractOne(token, fuzzy_variants, scorer=fuzz.ratio)
            if hit and (best is None or hit[1] > best[1]):
                best = (hit[0], hit[1])
        if best and best[1] >= FUZZY_REF_THRESHOLD:
            iid = index.variant_to_invoice.get(best[0])
            if iid:
                outcome.invoice_ids = [iid]
                outcome.method = MatchMethod.FUZZY.value
                outcome.confidence = best[1] / 100.0
                outcome.detail = f"fuzzy narration match at {best[1]:.0f}"
                return outcome

    # -- rung 4: reconcile on amount -------------------------------------------------
    result = reconcile(
        int(payment.amount_paise),
        index.open_invoices_for(buyer_id),
        tolerance_paise=tolerance_paise,
    )
    if result.resolved:
        outcome.invoice_ids = sorted(result.invoice_ids)
        outcome.method = MatchMethod.SUBSET_SUM.value
        # Deliberately below the advice and reference rungs: this is an inference from an
        # amount, not a stated fact, and the policy layer should treat it as such.
        outcome.confidence = 0.80
        outcome.detail = (
            f"amount reconciled against {len(outcome.invoice_ids)} invoice(s) "
            f"from {result.candidates_considered} candidates"
        )
        return outcome

    if result.ambiguous:
        outcome.exception = "ambiguous_amount_match"
        outcome.detail = (
            f"{result.solutions_found}+ invoice combinations reconcile to this amount; "
            f"refusing to guess"
        )
        return outcome

    outcome.exception = "unmatched_payment"
    outcome.detail = (
        "no reference recovered and no combination reconciles"
        + (f" (reference appears truncated: {truncated[0]})" if truncated else "")
    )
    return outcome


# ----------------------------------------------------------------------------------
# Allocation
# ----------------------------------------------------------------------------------
def allocate(
    outcome: MatchOutcome,
    payment: PaymentEvent,
    index: MatchIndex,
    *,
    apportionment: dict[str, int] | None = None,
) -> list[Allocation]:
    """Split a matched payment across its invoices.

    **Uses the advice's per-invoice net figures when they are available**, because on a
    bundled payment nothing else can say how the shortfall divides. Filling invoices
    sequentially instead puts the entire gap on whichever invoice comes last — inventing a
    large deduction on one and none on the others, while reconciling perfectly at the
    payment level so that nothing downstream notices. That was measured: 119 of 266
    allocated invoices had deltas that disagreed with their actual deductions.

    Sequential filling remains the fallback for a payment with no readable advice. For a
    single-invoice payment the two are identical, and single invoices are the majority.
    """
    if not outcome.matched:
        return []

    allocations: list[Allocation] = []

    if apportionment:
        # Keyed by invoice_no in the advice; translate to invoice ids we matched.
        by_id: dict[str, int] = {}
        for invoice_id in outcome.invoice_ids:
            invoice_no = index.invoices[invoice_id].invoice_no
            if invoice_no in apportionment:
                by_id[invoice_id] = int(apportionment[invoice_no])

        if len(by_id) == len(outcome.invoice_ids):
            for position, invoice_id in enumerate(sorted(by_id)):
                allocations.append(
                    Allocation(
                        id=f"ALL-{payment.id}-{position}",
                        payment_event_id=payment.id,
                        invoice_id=invoice_id,
                        allocated_paise=Paise(max(0, by_id[invoice_id])),
                        method=MatchMethod.ADVICE.value,
                        confidence=outcome.confidence,
                    )
                )
            return allocations

    ordered = sorted(
        outcome.invoice_ids, key=lambda iid: -int(index.invoices[iid].total_paise)
    )
    remaining = int(payment.amount_paise)

    for position, invoice_id in enumerate(ordered):
        if remaining <= 0:
            break
        invoice_total = int(index.invoices[invoice_id].total_paise)
        amount = min(remaining, invoice_total)
        remaining -= amount
        allocations.append(
            Allocation(
                id=f"ALL-{payment.id}-{position}",
                payment_event_id=payment.id,
                invoice_id=invoice_id,
                allocated_paise=Paise(amount),
                method=outcome.method,
                confidence=outcome.confidence,
            )
        )

    return allocations


def summarise(outcomes: list[MatchOutcome]) -> dict[str, Any]:
    total = len(outcomes)
    matched = sum(1 for o in outcomes if o.matched)
    by_method: dict[str, int] = {}
    by_exception: dict[str, int] = {}
    for o in outcomes:
        if o.matched:
            by_method[o.method] = by_method.get(o.method, 0) + 1
        else:
            by_exception[o.exception or "unknown"] = (
                by_exception.get(o.exception or "unknown", 0) + 1
            )

    return {
        "payments": total,
        "matched": matched,
        "match_rate": round(matched / total, 4) if total else 0.0,
        "by_method": dict(sorted(by_method.items())),
        "exceptions": dict(sorted(by_exception.items())),
    }
