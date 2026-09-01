"""Integer-paise money arithmetic. The only place in this codebase that does rate maths.

Two rules, both load-bearing:

1. **Money is an integer number of paise, everywhere.** No floats, no Decimal, no
   rupee-denominated variables. `tests/test_money_discipline.py` greps the money paths
   and fails the build if a float creeps in. Floats in an AR system are not a style
   preference — a 0.1% TDS computation on a Rs 60L invoice is exactly the shape of
   arithmetic that binary floating point gets wrong, and the whole submission rests on
   the claim that the money numbers are exact.

2. **The generator and the verifier call the SAME rate function.** This is the subtle
   one. If `generator/deductions.py` computes 2% of the taxable value by one rounding
   rule and `verify/tds.py` recomputes it by another, every single TDS verification comes
   back off by a paise or two, the verdict layer looks comprehensively broken, and it
   costs a day to find because each individual number looks right. One function, one
   rounding rule, imported by both sides.
"""

from __future__ import annotations

from typing import NewType

# A quantity of money, in paise. 100 paise = 1 rupee.
Paise = NewType("Paise", int)

PAISE_PER_RUPEE = 100
BP_DENOMINATOR = 10_000  # basis points: 10_000 bp = 100%


def rupees_to_paise(rupees: int | str) -> Paise:
    """Convert a whole-rupee or decimal-string amount to paise.

    Accepts a string so that callers reading fixture files never route a money value
    through a float on the way in. `rupees_to_paise("1234.56") == 123456`.
    """
    if isinstance(rupees, int):
        return Paise(rupees * PAISE_PER_RUPEE)

    text = str(rupees).strip().replace(",", "")
    negative = text.startswith("-")
    if negative:
        text = text[1:]

    if "." in text:
        whole, _, frac = text.partition(".")
        frac = (frac + "00")[:2]  # truncate/pad to exactly 2 decimal places
    else:
        whole, frac = text, "00"

    whole = whole or "0"
    value = int(whole) * PAISE_PER_RUPEE + int(frac)
    return Paise(-value if negative else value)


def paise_to_rupees_str(amount: int) -> str:
    """Render paise as a plain decimal rupee string. Presentation only — never parsed back."""
    negative = amount < 0
    amount = abs(int(amount))
    text = f"{amount // PAISE_PER_RUPEE}.{amount % PAISE_PER_RUPEE:02d}"
    return f"-{text}" if negative else text


def format_inr(amount: int, *, paise_precision: bool = False) -> str:
    """Indian-grouped rupee string for terminal output and message bodies.

    `format_inr(12345678)` -> 'Rs 1,23,456.78' with precision, 'Rs 1,23,457' without.
    Rounds half-up for display only; the underlying integer is never modified.
    """
    negative = amount < 0
    amount = abs(int(amount))

    if paise_precision:
        whole, frac = divmod(amount, PAISE_PER_RUPEE)
        tail = f".{frac:02d}"
    else:
        whole = (amount + PAISE_PER_RUPEE // 2) // PAISE_PER_RUPEE
        tail = ""

    digits = str(whole)
    if len(digits) > 3:
        head, last3 = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [last3])

    return f"{'-' if negative else ''}Rs {digits}{tail}"


def apply_rate_bp(amount_paise: int, rate_bp: int) -> Paise:
    """Apply a basis-point rate to a paise amount. THE canonical rate computation.

    Truncates toward zero, which is what Indian AP systems do when computing statutory
    withholding — they do not round up to the seller's benefit. Both the generator (which
    decides what the buyer deducted) and the verifier (which recomputes what the buyer
    *should* have deducted) must call this, or verification will chase phantom paise.
    """
    return Paise(int(amount_paise) * int(rate_bp) // BP_DENOMINATOR)


def implied_rate_bp(deducted_paise: int, base_paise: int) -> int | None:
    """Reverse of `apply_rate_bp`: what rate does this deduction imply?

    This is the single most informative feature the classifier gets. A deduction that is
    0.10% of the taxable value is 194Q almost regardless of what the buyer wrote in the
    advice, and a deduction that is 2.00% of taxable on a contract whose expected section
    is 194J at 10% is a rate mismatch worth exactly the difference.

    Returns None when the base is zero — never raises, because it runs over parsed data.
    """
    if not base_paise:
        return None
    return round(int(deducted_paise) * BP_DENOMINATOR / int(base_paise))


def bp_to_pct_str(rate_bp: int | None) -> str:
    """'200' -> '2.00%'. Presentation only."""
    if rate_bp is None:
        return "n/a"
    return f"{rate_bp / 100:.2f}%"


def within_tolerance(a: int, b: int, tolerance_paise: int) -> bool:
    """Are two paise amounts equal to within the configured tolerance?

    Used to absorb the buyer's own off-by-a-rupee arithmetic in a stated deduction
    breakdown. A real AP clerk's spreadsheet rounds inconsistently; refusing to reconcile
    over one rupee would put a quarter of the batch into the exceptions report for no
    economic reason.
    """
    return abs(int(a) - int(b)) <= int(tolerance_paise)
