"""Stage [1]: turn a remittance advice into per-invoice amounts.

Why this matters more than it looks. A bundled payment covers several invoices and each
carries its own deduction. The payment amount alone cannot tell you how the shortfall
splits — allocating sequentially puts the entire gap on whichever invoice happens to be
last, inventing a large deduction on one invoice and none on the others. Both are
fabrications, and they reconcile perfectly at the payment level, so nothing downstream
notices.

The advice is the only thing that says how the money was apportioned. Parsing it is
therefore not a convenience; it is what makes per-invoice deltas true.

**Deterministic first, LLM only where the format defeats it.** The email and spreadsheet
layouts are delimited and parse exactly, so using a model on them would spend ~50 seconds
per call to reproduce what a `split()` already gets right — and would introduce a chance of
being wrong. The PDF-extracted format deliberately runs its columns together
(`INV/2026/0061538696.6353706.00484990.63`), which is where a parser needs to reason about
where one number ends and the next begins. That is the case the LLM earns its place on.

Where parsing fails, the honest output is nothing — the caller then treats the bundle's
split as unknowable rather than guessing at it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..money import Paise, rupees_to_paise
from ..schemas import AdviceFormat

# Our own invoice-number format, quoted back at us in our own document.
_REF = re.compile(r"INV[/\-]\d{4}[/\-]\d{3,6}")
# An amount with exactly two decimal places, optionally comma-grouped.
_AMOUNT = re.compile(r"\d[\d,]*\.\d{2}")


@dataclass
class AdviceLine:
    """One invoice's row on a remittance advice."""

    invoice_ref: str
    gross_paise: int | None = None
    deduction_paise: int | None = None
    net_paise: int | None = None
    stated_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.net_paise is not None


@dataclass
class ParsedAdvice:
    lines: list[AdviceLine] = field(default_factory=list)
    format: str = ""
    parser: str = "deterministic"
    confident: bool = True

    @property
    def net_by_ref(self) -> dict[str, int]:
        return {
            line.invoice_ref: line.net_paise
            for line in self.lines
            if line.net_paise is not None
        }


def _amount(text: str) -> int | None:
    try:
        return int(rupees_to_paise(text))
    except (ValueError, TypeError):
        return None


def _parse_email(raw: str) -> list[AdviceLine]:
    """`Invoice INV/2026/0031 | Gross 218988.58 | Less 9279.09 (reason) | Net 191149.31`"""
    lines: list[AdviceLine] = []
    for row in raw.splitlines():
        if "|" not in row:
            continue
        ref_match = _REF.search(row)
        if not ref_match:
            continue

        line = AdviceLine(invoice_ref=ref_match.group(0))
        reasons: list[str] = []
        for cell in (c.strip() for c in row.split("|")):
            low = cell.lower()
            amounts = _AMOUNT.findall(cell)
            if low.startswith("gross") and amounts:
                line.gross_paise = _amount(amounts[0])
            elif low.startswith("net") and amounts:
                line.net_paise = _amount(amounts[0])
            elif low.startswith("less") and amounts:
                line.deduction_paise = (line.deduction_paise or 0) + (_amount(amounts[0]) or 0)
                if "(" in cell:
                    reasons.append(cell[cell.index("(") + 1 : cell.rindex(")")])
        line.stated_reason = "; ".join(reasons) or None
        lines.append(line)
    return lines


def _parse_xlsx(raw: str) -> list[AdviceLine]:
    """Tab-delimited with merged headers and stray blank rows."""
    lines: list[AdviceLine] = []
    for row in raw.splitlines():
        if "\t" not in row:
            continue
        cells = [c.strip() for c in row.split("\t")]
        ref_match = _REF.search(cells[0]) if cells else None
        if not ref_match:
            continue
        # Layout: ref, gross, deduction, net, remarks
        line = AdviceLine(invoice_ref=ref_match.group(0))
        if len(cells) > 1:
            line.gross_paise = _amount(cells[1])
        if len(cells) > 2:
            line.deduction_paise = _amount(cells[2])
        if len(cells) > 3:
            line.net_paise = _amount(cells[3])
        if len(cells) > 4 and cells[4]:
            line.stated_reason = cells[4]
        lines.append(line)
    return lines


def _parse_pdf_text(raw: str) -> list[AdviceLine]:
    """Columns run together: `INV/2026/0061538696.6353706.00484990.63  remarks`.

    The reference ends where the digits of the first amount begin, and each amount is
    identifiable by its two decimal places. Three amounts follow in order — gross,
    deduction, net — so the run can be split without knowing the column widths.

    This is the format the LLM parser exists for; the regex handles the common shape and
    returns nothing when it does not fit, rather than emitting a half-read row.
    """
    lines: list[AdviceLine] = []
    for row in raw.splitlines():
        ref_match = _REF.search(row)
        if not ref_match:
            continue
        tail = row[ref_match.end() :]
        amounts = _AMOUNT.findall(tail)
        if len(amounts) < 3:
            continue  # cannot tell gross from net; say nothing
        remarks = tail.split("  ", 1)[1].strip() if "  " in tail else ""
        lines.append(
            AdviceLine(
                invoice_ref=ref_match.group(0),
                gross_paise=_amount(amounts[0]),
                deduction_paise=_amount(amounts[1]),
                net_paise=_amount(amounts[2]),
                stated_reason=remarks or None,
            )
        )
    return lines


_PARSERS = {
    AdviceFormat.EMAIL.value: _parse_email,
    AdviceFormat.XLSX.value: _parse_xlsx,
    AdviceFormat.PDF_TEXT.value: _parse_pdf_text,
}


def parse_advice(raw_text: str, fmt: str) -> ParsedAdvice:
    """Parse one advice deterministically. Empty result means 'could not read it'."""
    parser = _PARSERS.get(fmt)
    if parser is None:
        return ParsedAdvice(format=fmt, confident=False)

    lines = parser(raw_text or "")

    # A row whose stated arithmetic does not hold is kept but flagged: the buyer's own
    # spreadsheet rounds inconsistently, and `amount_tolerance_paise` exists to absorb it.
    confident = bool(lines) and all(line.complete for line in lines)
    return ParsedAdvice(lines=lines, format=fmt, confident=confident)


def apportion(
    parsed: ParsedAdvice, payment_paise: int, tolerance_paise: int = 100
) -> dict[str, int] | None:
    """Per-invoice allocation implied by the advice, if it reconciles to the payment.

    Returns None when the advice's own net figures do not add up to what actually arrived.
    That mismatch means the advice belongs to a different credit, or was misread — either
    way the apportionment is not trustworthy and sequential allocation is the safer
    fallback.
    """
    nets = parsed.net_by_ref
    if not nets:
        return None

    total = sum(nets.values())
    if abs(total - payment_paise) > max(tolerance_paise, len(nets) * tolerance_paise):
        return None

    return {ref: Paise(v) for ref, v in nets.items()}
