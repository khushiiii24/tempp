"""Prompt construction for the classifier.

## The measurement that shaped this file

The obvious assumption for local inference is that output tokens dominate — generation is
one forward pass per token, prompt evaluation is batched. Measured on this workload, that
is **false**, and by a wide margin:

| | tokens | time |
|---|---:|---:|
| prompt evaluation | ~1084 | ~85s |
| generation | ~60 | ~10s |

Prompt evaluation is roughly 90% of the cost. So the schema stays terse (it costs nothing
to keep it that way) but the *real* lever is the prompt, and it gets two optimisations
that follow directly from that number.

## 1. Static content first, variable content last

llama.cpp caches the KV state of a prompt prefix and reuses it when the next prompt starts
with the same tokens. The first version of this file put the invoice, buyer and contract
at the top and the reason-code list and decision rules at the bottom — meaning every
prompt diverged after about twenty tokens and the entire 1084 tokens were re-evaluated
every single call.

Inverting that order puts ~600 tokens of identical preamble at the front of every prompt
in the batch, so only the ~400 tokens of case data are new work. The content is unchanged;
only the order is, and the order was costing more than everything else in the pipeline
combined.

## 2. Say each thing once

The reason-code list carried a prose label for every code. The codes are already
self-describing (`TDS_194C`, `UNEARNED_DISCOUNT`), so the labels were paying tokens to
restate the identifier. They are kept only where a code is genuinely ambiguous from its
name.

## What did not change

**Give it the arithmetic; do not make it do the arithmetic.** The implied rate is computed
in Python and handed over. A deduction that is 0.10% of taxable is 194Q almost regardless
of what the buyer wrote. A 7B model asked to divide two nine-digit paise figures will
sometimes get it wrong and will never tell you it did.

**Say plainly which evidence wins.** When the buyer writes "TDS @ 2%" and the arithmetic
says 5%, that is the highest-value case in the batch, and a model that splits the
difference gets it exactly backwards.
"""

from __future__ import annotations

import functools

from ..config import Taxonomy, load_taxonomy
from ..money import bp_to_pct_str, format_inr, implied_rate_bp
from ..schemas import Buyer, Contract, Deduction, Invoice

SYSTEM_PROMPT = (
    "You are an accounts-receivable analyst in India classifying why a B2B buyer paid "
    "less than the invoice amount. You are precise, you cite arithmetic, and you say "
    "NEEDS_HUMAN when the evidence genuinely does not decide the answer. "
    "Guessing is worse than abstaining: a wrong reason code sends a chase letter to a "
    "customer who did nothing wrong."
)

# Codes whose meaning is not obvious from the identifier. The rest are self-describing,
# and a gloss on them would be paying prompt tokens to restate the name.
_CODE_GLOSS = {
    "TDS_RATE_MISMATCH": "wrong rate or wrong section",
    "TCS_194Q_OVERLAP": "buyer deducted 194Q where we already charged TCS 206C(1H)",
    "RATE_DIFFERENCE": "WE billed above the contracted rate (our error)",
    "CREDIT_NOTE_OFFSET": "netted an existing credit note",
    "DEBIT_NOTE_BUYER": "buyer raised their own debit note",
    "SCHEME_REBATE": "trade scheme / QPS claim",
    "UNEARNED_DISCOUNT": "early-payment discount taken late",
    "DUPLICATE_CLAIM": "'already paid this invoice'",
    "UNEXPLAINED": "short paid, no reason given",
    "NEEDS_HUMAN": "genuinely unsure — abstain",
}


@functools.lru_cache(maxsize=4)
def _static_preamble(confidence_floor_x100: int) -> str:
    """The identical prefix of every classification prompt.

    Cached and placed first so llama.cpp can reuse its KV state across the batch. Keyed on
    the confidence floor as an integer so the cache key is hashable and stable.
    """
    taxonomy = load_taxonomy()
    floor = confidence_floor_x100 / 100

    codes = []
    for code in taxonomy.all_codes:
        gloss = _CODE_GLOSS.get(code)
        codes.append(f"{code} = {gloss}" if gloss else code)

    return f"""Classify ONE deduction a buyer took against ONE invoice. Return JSON only.

## Reason codes (the full space; each case lists which are POSSIBLE for it)
{chr(10).join(codes)}

## How to decide
1. ARITHMETIC BEATS THE STATED REASON. If the buyer writes "TDS 194C @ 2%" but the
   deduction is 5.00% of taxable value, that is TDS_RATE_MISMATCH, not TDS_194C.
   Put the decisive number in `check`.
2. Percentage matches a statutory rate AND the contract expects that section
   -> use the plain section code.
3. Percentage matches a statutory rate HIGHER than the contract's expected rate
   -> TDS_RATE_MISMATCH.
4. Freight is FREIGHT whoever owes it. Whether it is legitimate is decided later against
   the contract, not by you.
5. Under Rs 10 -> ROUNDING. Small round transfer fees -> BANK_CHARGES.
6. No reason stated and the arithmetic matches nothing -> UNEXPLAINED.
7. Genuinely unsure -> NEEDS_HUMAN with confidence below {floor:.2f}. Abstaining is a
   correct and expected answer, not a failure.

`confidence` is a decimal between 0.0 and 1.0, never a percentage.
Keep `rationale` under 200 characters and cite the number that decided it.
Do not restate the invoice.

---
"""


def _candidate_sections(taxonomy: Taxonomy, implied_bp: int | None) -> list[str]:
    """Which statutory sections are consistent with the observed rate?

    Turning recall into selection, which small models do far better than recalling that
    194H is 5%. Tolerance is 2bp to absorb paise truncation.
    """
    if implied_bp is None:
        return []
    hits: list[str] = []
    for code in taxonomy.all_codes:
        for rate in taxonomy[code].plausible_rates_bp():
            if abs(rate - implied_bp) <= 2:
                hits.append(f"{code} ({bp_to_pct_str(rate)})")
                break
    return hits


def _history_line(history: dict[str, int] | None) -> str:
    if not history:
        return "none recorded"
    top = sorted(history.items(), key=lambda kv: -kv[1])[:3]
    return ", ".join(f"{k} x{v}" for k, v in top)


def build_classification_prompt(
    *,
    taxonomy: Taxonomy,
    deduction: Deduction,
    invoice: Invoice,
    buyer: Buyer,
    contract: Contract,
    buyer_history: dict[str, int] | None = None,
    confidence_floor: float = 0.72,
) -> str:
    """Assemble the prompt: shared static preamble, then this case's data."""
    taxable = int(invoice.taxable_paise)
    amount = int(deduction.amount_paise)

    implied_taxable_bp = implied_rate_bp(amount, taxable)
    candidates = _candidate_sections(taxonomy, implied_taxable_bp)

    stated = deduction.claimed_reason_text
    stated_line = (
        f'stated reason (VERBATIM, may be wrong): "{stated}"'
        if stated
        else "stated reason: NONE GIVEN"
    )

    arithmetic = [
        f"deduction is {bp_to_pct_str(implied_taxable_bp)} of TAXABLE value",
        (
            "statutory rates matching that percentage: " + "; ".join(candidates)
            if candidates
            else "NO statutory rate matches that percentage (argues against the TDS family "
            "unless a wrong rate was applied)"
        ),
    ]
    if invoice.tcs_paise:
        arithmetic.append(
            f"WE already charged TCS {format_inr(invoice.tcs_paise)} u/s 206C(1H). If the "
            f"buyer also deducted 194Q at 0.10%, that is TCS_194Q_OVERLAP."
        )
    # NOTE: computed feasibility hints and a per-case "possible codes" list were tried
    # here and MEASURED AS HARMFUL. See the ablation in docs/MODEL_SELECTION.md:
    #
    #   no feasibility block            macro-F1 0.554, 61.3s/call   <- shipped
    #   + code-naming hints             macro-F1 0.512, 69.9s/call
    #   + fact-only hints               macro-F1 0.458, 81.5s/call
    #
    # Every addition to the per-case block cost both accuracy and time. A 7B model given
    # more context to weigh does not weigh it better; it gets distracted, and the extra
    # tokens are pure latency. The feasibility logic is still used — but as a
    # *post-hoc gate* in classify/classifier.py, where it converts an impossible answer
    # into an abstention without ever entering the prompt.

    discount = (
        f"{bp_to_pct_str(contract.early_payment_discount_bp)} within "
        f"{contract.early_payment_window_days}d"
        if contract.early_payment_discount_bp
        else "none"
    )

    case_block = f"""## Invoice
{invoice.invoice_no} | taxable {format_inr(taxable)} | total {format_inr(invoice.total_paise)}
issued {invoice.issue_date}, due {invoice.due_date}

## Contract
delivery {contract.delivery_terms} (freight borne by {contract.freight_borne_by}) |
expected TDS {contract.tds_section_expected} at {bp_to_pct_str(contract.tds_rate_expected_bp)} |
we charge TCS: {'yes' if contract.tcs_applicable else 'no'} | early-payment discount: {discount}

## Buyer
{buyer.name} ({buyer.segment}) | prior deductions: {_history_line(buyer_history)}

## The deduction
amount {format_inr(amount)}
{stated_line}

## Arithmetic (computed for you — trust these)
{chr(10).join('- ' + line for line in arithmetic)}"""

    return _static_preamble(int(round(confidence_floor * 100))) + case_block
