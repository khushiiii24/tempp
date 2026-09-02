"""Which reason codes are physically possible for a given deduction.

Computed in Python from observable data only — invoice, contract, amount. No ground truth,
no model. It does two jobs, and the second one is the more valuable.

## 1. Narrowing the choice

A 19-way classification is hard for a 7B model. But most of those 19 are *impossible* for
any given case: `UNEARNED_DISCOUNT` cannot apply to a contract with no early-payment
discount clause, `TCS_194Q_OVERLAP` cannot apply where the seller charged no TCS, and
`ROUNDING` cannot describe a ₹3,000 deduction. Presenting only the feasible codes turns
recall into selection, which small models do far better.

## 2. A real abstention signal, replacing a useless one

Measured across 40 classifications, the model's self-reported confidence was **0.95 on 34
of them** — essentially constant, and identical for easy and hard cases alike. It carries
almost no information, so a confidence threshold never fires and the whole abstention
mechanism the policy engine depends on is dead on arrival.

Feasibility gives an abstention signal that is *real*, because it is derived rather than
self-reported. A model that returns `ROUNDING` for a ₹3,000 deduction has demonstrably not
understood the case, and that is worth far more than it telling us it is 95% sure. Such an
answer becomes `NEEDS_HUMAN` — a safe hand-off — instead of a confident wrong action.

The filter is deterministic and would apply equally to any classifier, so its contribution
is reported separately in the ablation rather than folded into the model's score.
"""

from __future__ import annotations

from ..money import implied_rate_bp
from ..schemas import Contract, Deduction, Invoice

# A "rounding" difference is by definition trivial. Anything above this is something else
# wearing the label.
ROUNDING_CEILING_PAISE = 1_000  # Rs 10

# Interbank transfer charges are small and bounded. A "bank charge" of Rs 5,000 is not one.
BANK_CHARGES_CEILING_PAISE = 100_000  # Rs 1,000

# Statutory sections. Only the one the contract pins can be a *legitimate* withholding;
# any other rate is a mismatch, which has its own code.
_PLAIN_TDS = ("TDS_194C", "TDS_194J", "TDS_194H", "TDS_194Q")

# Codes that are always available: they depend on nothing we can rule out in advance.
_ALWAYS_POSSIBLE = (
    "FREIGHT",
    "SCHEME_REBATE",
    "CREDIT_NOTE_OFFSET",
    "DAMAGE_SHORTAGE",
    "QUALITY_REJECTION",
    "DEBIT_NOTE_BUYER",
    "DUPLICATE_CLAIM",
    "UNEXPLAINED",
    "TDS_RATE_MISMATCH",
    "GST_TDS",
    # A buyer can always *wrongly* claim a discount. An earlier version gated this on the
    # contract actually offering one, which filtered the correct answer out of 10 real
    # cases: taking a discount the contract never offered is not impossible, it is the
    # most clear-cut form of unearned discount there is. A feasibility filter that
    # excludes the right answer does not make the model safer, it makes the case
    # unanswerable and forces a false abstention.
    "UNEARNED_DISCOUNT",
    "NEEDS_HUMAN",
)


def invoice_has_rate_error(invoice: Invoice) -> bool:
    """Was any line billed above its contracted rate?

    Reads `contracted_rate_paise` off the invoice line, which is our own contract price —
    legitimately observable, not ground truth.
    """
    for line in invoice.line_items:
        billed = int(line["unit_rate_paise"])
        contracted = int(line.get("contracted_rate_paise", billed))
        if billed > contracted:
            return True
    return False


def feasible_codes(
    *, deduction: Deduction, invoice: Invoice, contract: Contract
) -> set[str]:
    """The codes that could possibly describe this deduction."""
    amount = int(deduction.amount_paise)
    codes: set[str] = set(_ALWAYS_POSSIBLE)

    # Only the contract's own section is a legitimate withholding. Any other rate is a
    # mismatch, and TDS_RATE_MISMATCH is already always available.
    if contract.tds_section_expected in _PLAIN_TDS:
        codes.add(contract.tds_section_expected)

    if amount <= ROUNDING_CEILING_PAISE:
        codes.add("ROUNDING")

    if amount <= BANK_CHARGES_CEILING_PAISE:
        codes.add("BANK_CHARGES")

    # A 194Q overlap requires the seller to have actually collected TCS.
    if contract.tcs_applicable and invoice.tcs_paise > 0:
        codes.add("TCS_194Q_OVERLAP")

    # We can only have over-billed if a line really is above the contracted rate.
    if invoice_has_rate_error(invoice):
        codes.add("RATE_DIFFERENCE")

    return codes


def infeasibility_reason(
    code: str, *, deduction: Deduction, invoice: Invoice, contract: Contract
) -> str | None:
    """Why this code cannot apply, for the audit log. None if it is feasible."""
    amount = int(deduction.amount_paise)

    if code == "ROUNDING" and amount > ROUNDING_CEILING_PAISE:
        return f"rounding is under Rs 10; this deduction is {amount} paise"
    if code == "BANK_CHARGES" and amount > BANK_CHARGES_CEILING_PAISE:
        return f"bank charges are small; this deduction is {amount} paise"
    if code == "TCS_194Q_OVERLAP" and not (contract.tcs_applicable and invoice.tcs_paise > 0):
        return "the seller charged no TCS on this invoice, so there is nothing to overlap"
    if code == "RATE_DIFFERENCE" and not invoice_has_rate_error(invoice):
        return "no line was billed above the contracted rate"
    if code in _PLAIN_TDS and code != contract.tds_section_expected:
        return (
            f"the contract pins {contract.tds_section_expected}; a deduction under {code} "
            f"would be a rate/section mismatch"
        )
    return None


def arithmetic_hints(
    *, deduction: Deduction, invoice: Invoice, contract: Contract
) -> list[str]:
    """Computed facts about the case. **Never names a reason code.**

    That constraint is the whole design of this function, and it was learned the expensive
    way. An earlier version wrote helpful-sounding conditionals like "contract offers no
    early-payment discount, so any discount taken is unearned by definition" and "deduction
    rate exceeds the contracted rate — if this is TDS, it is TDS_RATE_MISMATCH".

    Measured effect: macro-F1 fell from 0.554 to 0.512, and the confusion matrix collapsed
    onto exactly the two codes those hints named — 11 wrong answers of `UNEARNED_DISCOUNT`
    and 6 of `TDS_RATE_MISMATCH`, on cases that were nothing of the sort.

    Mentioning a code makes a small model reach for it. The conditional framing ("if this
    is TDS...") is simply not carried; what survives is the label. So these lines state the
    *fact* and let the decision rules in the preamble do the mapping from fact to code.
    """
    amount = int(deduction.amount_paise)
    hints: list[str] = []

    if amount > ROUNDING_CEILING_PAISE:
        hints.append(f"amount is {amount} paise, far above the Rs 10 rounding threshold")
    if amount > BANK_CHARGES_CEILING_PAISE:
        hints.append("amount is far larger than any interbank transfer fee")

    if contract.early_payment_discount_bp <= 0:
        hints.append("contract contains no early-payment discount clause")
    else:
        hints.append(
            f"contract allows a {contract.early_payment_discount_bp / 100:.2f}% discount "
            f"only if paid within {contract.early_payment_window_days} days"
        )

    if not (contract.tcs_applicable and invoice.tcs_paise > 0):
        hints.append("seller charged no TCS on this invoice")
    if not invoice_has_rate_error(invoice):
        hints.append("every line was billed at exactly the contracted rate")

    implied = implied_rate_bp(amount, int(invoice.taxable_paise))
    expected = int(contract.tds_rate_expected_bp)
    if implied is not None and expected:
        relation = "equals" if implied == expected else ("exceeds" if implied > expected else "is below")
        hints.append(
            f"deduction rate {implied}bp {relation} the contracted TDS rate of {expected}bp"
        )

    return hints
