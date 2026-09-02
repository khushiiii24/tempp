"""Messiness injection: the part that makes this a matching problem rather than a join.

Real bank narrations are 30-odd characters of truncated, delimiter-stripped, upper-cased
free text produced by whatever the buyer's bank felt like sending. `INV/2026/0042` arrives
as `INV20260042`, `2026-42`, `INV 42`, `RAZ/INV/26/42`, or truncated mid-number to
`NEFT-SHREESTE-INV202600`. That last one is the interesting case: the reference is not
merely reformatted, it is *incomplete*, and no amount of normalisation recovers the digits
that the bank threw away. Those cases have to be resolved by amount reconciliation or
declared unmatched — and declaring them unmatched is the correct answer, which is why the
exceptions report is a first-class output rather than an error log.

Three structural distortions matter as much as the textual ones:

* **Bundled payments** — one UTR covering three to seven invoices, each with its own
  deduction. The payment amount alone reconciles to nothing.
* **Split payments** — one invoice paid across two UTRs weeks apart. On day one this is
  indistinguishable from a short payment, and an agent that chases it immediately is
  chasing money that was always going to arrive. This is the trap the `awaiting_settlement`
  state exists for.
* **Absent advices** — 25% of the time nobody tells you anything at all.
"""

from __future__ import annotations

import re
from typing import Any

from .seed import chance, pick, rng_for, weighted_choice

BANK_PREFIXES = ("NEFT", "RTGS", "IMPS", "NEFT", "RTGS")


def mangle_invoice_ref(rng, invoice_no: str) -> str:
    """Distort a canonical invoice number the way a bank narration would.

    Returns something that a regex on the canonical format will not match, but which a
    normalising matcher usually can — except for the truncation variant, which is
    genuinely lossy on purpose.
    """
    m = re.match(r"INV/(\d{4})/(\d+)", invoice_no)
    if not m:
        return invoice_no
    year, num = m.group(1), m.group(2)
    yy = year[-2:]
    stripped = num.lstrip("0") or "0"

    style = weighted_choice(
        rng,
        {
            "squashed": 0.26,     # INV20260042
            "short_year": 0.16,   # 2026-42
            "spaced": 0.14,       # INV 42
            "prefixed": 0.12,     # RAZ/INV/26/42
            "truncated": 0.14,    # INV202600  <- lossy, may be unrecoverable
            "bare": 0.10,         # 0042
            "slashless": 0.08,    # INV-2026-0042
        },
    )

    if style == "squashed":
        return f"INV{year}{num}"
    if style == "short_year":
        return f"{year}-{stripped}"
    if style == "spaced":
        return f"INV {stripped}"
    if style == "prefixed":
        return f"RAZ/INV/{yy}/{stripped}"
    if style == "truncated":
        # Chop the tail: the last digits of the invoice number are simply gone.
        full = f"INV{year}{num}"
        return full[: max(8, len(full) - rng.randint(1, 3))]
    if style == "bare":
        return num
    return f"INV-{year}-{num}"


def build_narration(rng, buyer_name: str, refs: list[str], utr: str) -> str:
    """A bank-statement narration line.

    Upper-cased and length-capped, because that is what actually comes off a statement and
    the fuzzy matcher has to cope with it.
    """
    prefix = pick(rng, BANK_PREFIXES)
    # Bank narrations carry a squashed, truncated version of the payer name.
    name_token = re.sub(r"[^A-Za-z]", "", buyer_name).upper()[: rng.randint(6, 12)]

    if len(refs) == 1:
        body = f"{prefix}-{utr[:8]}-{name_token}-{refs[0]}"
    elif len(refs) <= 3:
        body = f"{prefix}-{utr[:8]}-{name_token}-{'/'.join(refs[:3])}"
    else:
        # A bundle of seven invoices does not fit; the bank truncates and the reference
        # list is simply lost. Amount reconciliation is the only route left.
        body = f"{prefix}-{utr[:8]}-{name_token}-{refs[0]}+{len(refs) - 1}MORE"

    return body.upper()[:90]


def make_utr(rng) -> str:
    """A UTR-shaped reference. Real ones are 16 or 22 chars depending on the rail."""
    return f"{pick(rng, ('SBIN', 'HDFC', 'ICIC', 'UTIB', 'PUNB'))}{rng.randrange(10**12):012d}"


def plan_payment_structure(
    seed: int,
    cfg: dict[str, Any],
    invoice_ids: list[str],
    buyer_of: dict[str, str],
) -> list[dict[str, Any]]:
    """Group invoices into payment events: singles, bundles, and splits.

    Bundling only ever groups invoices belonging to the SAME buyer — a UTR spanning two
    different customers would be a data error, not messiness, and would teach the matcher
    something false.
    """
    mess = cfg["messiness"]
    groups: list[dict[str, Any]] = []

    # Deterministic order, then group within buyer.
    by_buyer: dict[str, list[str]] = {}
    for inv_id in sorted(invoice_ids):
        by_buyer.setdefault(buyer_of[inv_id], []).append(inv_id)

    for buyer_id in sorted(by_buyer):
        rng = rng_for(seed, "payment_structure", buyer_id)
        pending = list(by_buyer[buyer_id])

        while pending:
            if len(pending) >= 3 and chance(rng, float(mess["bundled_payment_rate"])):
                size = min(len(pending), rng.randint(*mess["bundle_size"]))
                members = pending[:size]
                pending = pending[size:]
                groups.append({"buyer_id": buyer_id, "invoice_ids": members, "kind": "bundle"})
            else:
                inv = pending.pop(0)
                kind = "split" if chance(rng, float(mess["split_payment_rate"])) else "single"
                groups.append({"buyer_id": buyer_id, "invoice_ids": [inv], "kind": kind})

    return groups
