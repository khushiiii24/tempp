"""The feasibility filter: it must narrow the choice without ever excluding the answer.

The filter is what turns a 19-way classification into roughly a 13-way one, and what
converts the model's most common failure — returning a physically impossible code — into a
safe abstention instead of a confident wrong action.

**The invariant that matters is the soundness one.** A filter that excludes the correct
answer does not make the system safer; it makes the case unanswerable and forces a false
abstention, and it does so invisibly. The first version of this filter did exactly that on
10 real cases by assuming a buyer cannot take an early-payment discount under a contract
that offers none — when in fact that is the most clear-cut unearned discount there is.
"""

from __future__ import annotations

import pytest

from deduction_desk.classify.feasibility import (
    BANK_CHARGES_CEILING_PAISE,
    ROUNDING_CEILING_PAISE,
    feasible_codes,
    infeasibility_reason,
)
from deduction_desk.eval.slices import load_labelled_cases


@pytest.fixture(scope="module")
def cases():
    loaded = load_labelled_cases()
    if not loaded:
        pytest.skip("no generated batch; run `python -m deduction_desk generate` first")
    return loaded


def test_filter_never_excludes_the_true_code(cases) -> None:
    """Soundness. Violating this makes a case unanswerable, silently."""
    excluded = [
        (c.id, c.true_code)
        for c in cases
        if c.true_code
        not in feasible_codes(deduction=c.deduction, invoice=c.invoice, contract=c.contract)
    ]
    assert not excluded, f"feasibility filter excluded the correct answer on: {excluded[:10]}"


def test_filter_actually_narrows_the_choice(cases) -> None:
    """If it excluded nothing it would be costing tokens for no benefit."""
    sizes = [
        len(feasible_codes(deduction=c.deduction, invoice=c.invoice, contract=c.contract))
        for c in cases
    ]
    mean = sum(sizes) / len(sizes)
    assert mean < 18, f"filter barely narrows anything (mean {mean:.1f} codes)"


def test_rounding_is_impossible_above_the_threshold(cases) -> None:
    """`ROUNDING` on a four-figure deduction was one of the observed failures."""
    big = next(c for c in cases if c.deduction.amount_paise > ROUNDING_CEILING_PAISE * 10)
    codes = feasible_codes(deduction=big.deduction, invoice=big.invoice, contract=big.contract)

    assert "ROUNDING" not in codes
    assert infeasibility_reason(
        "ROUNDING", deduction=big.deduction, invoice=big.invoice, contract=big.contract
    )


def test_bank_charges_impossible_above_the_threshold(cases) -> None:
    big = next(
        c for c in cases if c.deduction.amount_paise > BANK_CHARGES_CEILING_PAISE * 5
    )
    codes = feasible_codes(deduction=big.deduction, invoice=big.invoice, contract=big.contract)
    assert "BANK_CHARGES" not in codes


def test_tcs_overlap_impossible_without_seller_tcs(cases) -> None:
    """There is nothing to overlap with if we never charged TCS."""
    no_tcs = next(c for c in cases if c.invoice.tcs_paise == 0)
    codes = feasible_codes(
        deduction=no_tcs.deduction, invoice=no_tcs.invoice, contract=no_tcs.contract
    )
    assert "TCS_194Q_OVERLAP" not in codes


def test_unearned_discount_stays_possible_without_a_discount_clause(cases) -> None:
    """The regression that motivated this file.

    Taking a discount the contract never offered is not impossible — it is the clearest
    case of an unearned one. Gating on the clause filtered the right answer out of 10
    genuine cases.
    """
    no_clause = [c for c in cases if c.contract.early_payment_discount_bp == 0]
    assert no_clause, "no contracts without a discount clause; the case is untested"

    for c in no_clause[:20]:
        codes = feasible_codes(deduction=c.deduction, invoice=c.invoice, contract=c.contract)
        assert "UNEARNED_DISCOUNT" in codes
        assert (
            infeasibility_reason(
                "UNEARNED_DISCOUNT", deduction=c.deduction, invoice=c.invoice, contract=c.contract
            )
            is None
        )


def test_only_the_contracted_tds_section_is_feasible(cases) -> None:
    """Any other rate is a mismatch, which has its own code."""
    tds = next(c for c in cases if c.contract.tds_section_expected == "TDS_194C")
    codes = feasible_codes(deduction=tds.deduction, invoice=tds.invoice, contract=tds.contract)

    assert "TDS_194C" in codes
    for other in ("TDS_194J", "TDS_194H"):
        assert other not in codes
    # ...but a mismatch is always available, so the case is never unanswerable.
    assert "TDS_RATE_MISMATCH" in codes


def test_abstention_is_always_available(cases) -> None:
    """The model must always be able to say it does not know."""
    for c in cases[:40]:
        codes = feasible_codes(deduction=c.deduction, invoice=c.invoice, contract=c.contract)
        assert "NEEDS_HUMAN" in codes


def test_hints_never_name_a_reason_code(cases) -> None:
    """Naming a code in a hint acts as a standing suggestion to a small model.

    Measured: hints phrased as "...so any discount taken is unearned by definition" and
    "...if this is TDS, it is TDS_RATE_MISMATCH" dropped macro-F1 from 0.554 to 0.512 and
    collapsed the confusion matrix onto exactly those two codes — 11 spurious
    UNEARNED_DISCOUNT and 6 spurious TDS_RATE_MISMATCH. The conditional framing is not
    carried; only the label survives.

    Hints state facts. The decision rules in the preamble map facts to codes.
    """
    from deduction_desk.classify.feasibility import arithmetic_hints
    from deduction_desk.config import load_taxonomy

    codes = [c for c in load_taxonomy().all_codes if c != "NEEDS_HUMAN"]

    offenders: list[str] = []
    for c in cases[:80]:
        for hint in arithmetic_hints(
            deduction=c.deduction, invoice=c.invoice, contract=c.contract
        ):
            for code in codes:
                if code in hint:
                    offenders.append(f"{code} in {hint!r}")

    assert not offenders, f"hints name reason codes: {offenders[:5]}"


def test_feasibility_reads_no_ground_truth() -> None:
    """The filter runs inside the agent, so it must not touch the answer key."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "deduction_desk" / "classify" / "feasibility.py"
    ).read_text(encoding="utf-8")

    for needle in ("DeductionTruth", "deduction_truth", "true_reason_code", "is_valid"):
        assert needle not in source, f"feasibility.py references {needle}"


def test_gate_uses_each_deductions_own_contract(cases) -> None:
    """Regression: the feasibility gate must resolve the contract per deduction.

    A first version relied on `invoice` leaking out of an earlier loop in
    `classify_batch`, so every gate ran against whichever invoice happened to be last —
    each case judged on a stranger's contract. It failed silently and plausibly: cases
    were abstained for real-sounding reasons ("the contract pins TDS_194C") that belonged
    to a different buyer entirely, and the abstention rate read 30% where the benchmark,
    which resolves contracts correctly, measured 15%.

    Two independent measurements of the same quantity disagreeing is what surfaced it.
    """
    from deduction_desk.classify.feasibility import infeasibility_reason

    # Find two cases whose contracts expect different TDS sections.
    by_section: dict[str, list] = {}
    for c in cases:
        by_section.setdefault(c.contract.tds_section_expected, []).append(c)

    sections = [s for s, v in by_section.items() if v]
    assert len(sections) >= 2, "need contracts with differing TDS sections to test this"

    a = by_section[sections[0]][0]
    b = by_section[sections[1]][0]

    # A's own section is feasible under A's contract...
    assert (
        infeasibility_reason(
            a.contract.tds_section_expected,
            deduction=a.deduction, invoice=a.invoice, contract=a.contract,
        )
        is None
    )
    # ...and NOT under B's, which is exactly the confusion the bug produced.
    assert infeasibility_reason(
        a.contract.tds_section_expected,
        deduction=a.deduction, invoice=a.invoice, contract=b.contract,
    )
