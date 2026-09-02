"""The exceptions report: everything the agent refused to resolve, and why.

A first-class output, not an error log. The spec's line is exact — *reporting what you
couldn't match is worth more than a fabricated match* — and the reason is that the two are
indistinguishable from any other metric. A matcher that guesses scores a higher match rate
than one that abstains, right up until someone checks where the money actually went.

Published in the scoreboard rather than hidden, because an AR system that cannot say what
it did not understand is not auditable.
"""

from __future__ import annotations

from typing import Any

from ..schemas import Exception_

# Human-readable explanations, so the report reads as a work queue rather than a stack of
# error codes. An analyst should be able to action a row without reading the source.
EXCEPTION_MEANINGS: dict[str, str] = {
    "unresolved_buyer": (
        "The payer could not be identified from the bank narration. Someone must look at "
        "the statement line and say who sent it."
    ),
    "unmatched_payment": (
        "The invoice reference was absent or truncated beyond recovery, and no combination "
        "of open invoices reconciles to the amount."
    ),
    "ambiguous_amount_match": (
        "Several invoice combinations reconcile to this amount equally well. Guessing "
        "would post real money against the wrong invoice."
    ),
    "unparsed_advice": (
        "A remittance advice arrived but no invoice line could be extracted from it."
    ),
    "classifier_abstained": (
        "The classifier declined to assign a reason code. Handing over beats guessing."
    ),
    "verification_undetermined": (
        "Source systems did not settle whether the deduction is valid."
    ),
    "delta_does_not_reconcile": (
        "The shortfall computed from allocations disagrees with the itemised deductions, "
        "which suggests the payment was matched to the wrong invoice."
    ),
}


def build_exception(
    *,
    run_id: str,
    kind: str,
    subject_id: str,
    detail: str,
    amount_paise: int = 0,
    created_at: str = "",
    seq: int | None = None,
) -> Exception_:
    """Build one exception row.

    `seq` disambiguates repeats. The same subject can legitimately raise the same kind
    more than once — an invoice can have two later credits blocked as over-allocation —
    and without it the id collides on the primary key and the whole run fails at commit.
    """
    suffix = f"-{seq}" if seq is not None else ""
    return Exception_(
        id=f"EXC-{run_id}-{kind}-{subject_id}{suffix}",
        run_id=run_id,
        kind=kind,
        subject_id=subject_id,
        detail=detail,
        amount_paise=amount_paise,
        created_at=created_at,
    )


def render_report(exceptions: list[Exception_]) -> str:
    """Markdown for `reports/<run>/exceptions.md`."""
    if not exceptions:
        return "# Exceptions\n\nNone. Every payment resolved.\n"

    by_kind: dict[str, list[Exception_]] = {}
    for exc in exceptions:
        by_kind.setdefault(exc.kind, []).append(exc)

    lines = ["# Exceptions", ""]
    lines.append(
        f"{len(exceptions)} item(s) the agent declined to resolve. This is published "
        f"rather than hidden: a matcher that guesses scores better on every metric except "
        f"the one that matters."
    )
    lines.append("")

    for kind in sorted(by_kind):
        items = by_kind[kind]
        lines.append(f"## `{kind}` — {len(items)}")
        lines.append("")
        meaning = EXCEPTION_MEANINGS.get(kind)
        if meaning:
            lines.append(f"_{meaning}_")
            lines.append("")
        lines.append("| subject | detail |")
        lines.append("|---|---|")
        for exc in sorted(items, key=lambda e: e.subject_id)[:40]:
            lines.append(f"| `{exc.subject_id}` | {exc.detail} |")
        if len(items) > 40:
            lines.append(f"| … | _and {len(items) - 40} more_ |")
        lines.append("")

    return "\n".join(lines)


def summarise(exceptions: list[Exception_]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    for exc in exceptions:
        by_kind[exc.kind] = by_kind.get(exc.kind, 0) + 1
    return {
        "total": len(exceptions),
        "by_kind": dict(sorted(by_kind.items())),
        "amount_paise": sum(e.amount_paise for e in exceptions),
    }
