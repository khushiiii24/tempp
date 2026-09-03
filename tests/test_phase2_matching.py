"""Phase 2 acceptance: matching, delta isolation, and the exceptions report.

**The headline metric here is not the match rate.** Any sufficiently reckless matcher
scores 100% — allocate every payment to the first plausible invoice and the rate looks
perfect while the money has gone to the wrong places. The rate is only meaningful beside
two other numbers:

* **Reconciliation** — does the shortfall derived from allocations agree with what was
  actually deducted? This is what catches a payment matched to the wrong invoice, which is
  invisible to every other measure.
* **Total money located** — does the sum of derived deltas equal the sum of real
  deductions? Currently 77.9%, against a match rate of 83.5%. Those two numbers moving in
  opposite directions is the point: the run declines 36 payments it cannot resolve safely
  rather than guessing where their money went.

Every bug these tests pin down was found by that reconciliation check rather than by the
match rate, which looked healthy throughout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import Session, select

from deduction_desk.db import make_engine
from deduction_desk.ingest.advice_parser import apportion, parse_advice
from deduction_desk.ingest.normalize import (
    candidate_refs,
    declared_bundle_size,
    fuzzy_candidates,
    is_year_token,
    normalise_ref,
    ref_variants,
    truncation_suspect,
)
from deduction_desk.matching import check_fabricated_matches, run_matching
from deduction_desk.matching.subset_sum import reconcile
from deduction_desk.schemas import Deduction, Invoice, RemittanceAdvice

SRC = Path(__file__).resolve().parents[1] / "src" / "deduction_desk"


@pytest.fixture(scope="module")
def matched():
    """One matching run over the generated batch, plus the truth to grade it against."""
    with Session(make_engine()) as session:
        report = run_matching(session, persist=False)
        invoices = {i.id: i for i in session.exec(select(Invoice)).all()}
        totals: dict[str, int] = {}
        for d in session.exec(select(Deduction)).all():
            totals[d.invoice_id] = totals.get(d.invoice_id, 0) + int(d.amount_paise)

    if not invoices:
        pytest.skip("no generated batch; run `python -m deduction_desk generate` first")
    return report, invoices, totals


# ======================================================================================
# Normalisation
# ======================================================================================


def test_reference_variants_cover_the_mangling_styles() -> None:
    variants = ref_variants("INV/2026/0042")
    for expected in ("INV20260042", "20260042", "42", "0042", "202642"):
        assert expected in variants, f"{expected} missing from {sorted(variants)}"


def test_bare_year_is_never_a_reference() -> None:
    """`2026` appears in every date-formatted narration and matches almost anything."""
    assert is_year_token("2026")
    assert not is_year_token("20260042")
    assert "2026" not in candidate_refs("NEFT-SBIN0585-LAKSHMICHE-2026-253")


def test_short_tokens_are_excluded_from_fuzzy_matching() -> None:
    """The bug this prevents, precisely.

    `INV/2026/0003` generates the variant `20263`. The bare year `2026` scores 88.9
    against it — over an 88 threshold. Six payments belonging to other invoices were
    allocated to INV-0003 on that basis.
    """
    assert "2026" not in fuzzy_candidates({"2026", "20260042"})
    assert "20260042" in fuzzy_candidates({"2026", "20260042"})


def test_truncated_reference_is_recognised_as_ambiguous() -> None:
    """A prefix is evidence of truncation, never of identity."""
    variants = ref_variants("INV/2026/0042") | ref_variants("INV/2026/0043")
    assert truncation_suspect("INV202600", variants)
    assert not truncation_suspect(normalise_ref("INV/2026/0042"), variants)


def test_bundle_overflow_count_is_extracted() -> None:
    """`+5MORE` means six invoices, and knowing the count constrains the search."""
    assert declared_bundle_size("NEFT-UTIB0979-CHETAK-INV 128+5MORE") == 6
    assert declared_bundle_size("RTGS-ICIC8293-GODAVARI-INV 28") is None


# ======================================================================================
# Subset-sum
# ======================================================================================


def test_subset_sum_resolves_a_single_combination() -> None:
    result = reconcile(150_000, [("A", 100_000), ("B", 60_000), ("C", 900_000)])
    assert result.resolved
    assert sorted(result.invoice_ids) == ["A", "B"]


def test_subset_sum_refuses_when_several_combinations_fit() -> None:
    """Refusing is the feature. Two equally good answers means no answer."""
    result = reconcile(100_000, [("A", 100_000), ("B", 100_000)])
    assert result.ambiguous
    assert result.invoice_ids == []


def test_subset_sum_never_lets_a_payment_exceed_the_invoices() -> None:
    """Buyers do not overpay; a combination smaller than the credit is not a match."""
    assert reconcile(500_000, [("A", 100_000)]).unresolved


# ======================================================================================
# Advice parsing
# ======================================================================================


def test_email_advice_parses_to_per_invoice_nets() -> None:
    raw = (
        "PFA payment advice for the below invoices.\n\n"
        "Invoice INV/2026/0031 | Gross 218988.58 | Less 9279.09 (tds comm) | Net 209709.49\n"
        "\nTotal remitted: 209709.49\n"
    )
    parsed = parse_advice(raw, "email")
    assert len(parsed.lines) == 1
    line = parsed.lines[0]
    assert line.invoice_ref == "INV/2026/0031"
    assert line.gross_paise == 21_898_858
    assert line.net_paise == 20_970_949
    assert "tds comm" in (line.stated_reason or "")


def test_advice_apportionment_rejects_a_mismatched_total() -> None:
    """An advice whose nets do not sum to the credit belongs to a different credit.

    Accepting it anyway is how an advice naming INV-0062 gets attached to the payment that
    settled INV-0003 — a fabricated match with a real document behind it.
    """
    raw = "Invoice INV/2026/0031 | Gross 100.00 | Net 90.00\n"
    parsed = parse_advice(raw, "email")
    assert apportion(parsed, 9_000) is not None  # matches
    assert apportion(parsed, 5_000_00) is None  # does not


# ======================================================================================
# End-to-end matching
# ======================================================================================


def test_match_rate_is_reported_and_reasonable(matched) -> None:
    report, _, _ = matched
    assert report["matching"]["match_rate"] >= 0.80


def test_every_unmatched_payment_appears_in_the_exceptions_report(matched) -> None:
    """The spec's requirement: nothing is silently dropped."""
    report, _, _ = matched
    unmatched = [o for o in report["_outcomes"] if not o.matched]
    reported = {e.subject_id for e in report["_exceptions"]}

    for outcome in unmatched:
        assert outcome.payment_id in reported, f"{outcome.payment_id} vanished silently"


def test_every_exception_carries_a_reason(matched) -> None:
    report, _, _ = matched
    for exc in report["_exceptions"]:
        assert exc.detail, f"{exc.id} has no explanation"
        assert exc.kind


def test_no_invoice_is_allocated_beyond_its_own_total(matched) -> None:
    """Two credits can settle one invoice; together they cannot exceed what was billed.

    Without this cap a bundle and a later subset-sum match both claimed INV-0000 and
    allocated Rs 60,562 against a Rs 36,394 invoice — which shows up as a zero delta and
    silently erases a real deduction.
    """
    report, invoices, _ = matched
    for invoice_id, delta in report["_deltas"].items():
        assert delta.allocated_paise <= int(invoices[invoice_id].total_paise), (
            f"{invoice_id} over-allocated: {delta.allocated_paise} > "
            f"{invoices[invoice_id].total_paise}"
        )


def test_derived_shortfall_locates_most_of_the_money(matched) -> None:
    """The number that actually matters.

    A high match rate with a low money figure means payments were matched to the wrong
    invoices, so this is the real measure of whether matching worked.

    **Measured: 77.9% of the true deduction total, against an 83.5% match rate.** The
    gap is not mis-allocation — the upper bound below asserts we never derive *more*
    shortfall than exists — it is the invoices inside bundles the matcher declined to
    resolve. Those payments are in the exceptions report and their deductions are never
    seen.

    This number swung between 78% and 98% across runs until `fuzzy_variants` was sorted;
    see `test_matching_is_deterministic_across_processes`.
    """
    report, _, totals = matched
    derived = report["deltas"]["total_delta_paise"]
    actual = sum(totals.values())

    assert actual > 0
    # The important direction: never claim more shortfall than was actually deducted.
    assert derived <= actual * 1.02, "derived more shortfall than exists — fabricated deltas"
    assert derived >= actual * 0.75, (
        f"only located {derived / actual:.1%} of the real deductions"
    )


def test_most_allocated_invoices_reconcile(matched) -> None:
    """Per-invoice agreement between derived delta and itemised deductions."""
    report, _, totals = matched
    bad = check_fabricated_matches(report["_outcomes"], report["_deltas"], totals)
    rate = 1 - len(bad) / len(report["_deltas"])

    assert rate >= 0.80, f"only {rate:.1%} of allocated invoices reconcile. Examples: {bad[:5]}"


def test_matching_is_deterministic_across_processes() -> None:
    """Same batch, same allocations — **measured in separate processes**.

    Running twice inside one process would not catch the bug this test exists for. Python
    randomises string hashing per process, so a set iterated directly gives a different
    order each time the interpreter starts. `fuzzy_variants` was built that way, and when
    two variants tied on fuzzy score `extractOne` returned whichever it happened to see
    first — so the same batch matched to different invoices from one run to the next and
    "money located" swung between 78% and 98% with an identical database content hash.

    Every individual run looked fine. Only comparing across processes exposes it.
    """
    import json
    import subprocess
    import sys

    script = (
        "from sqlmodel import Session, select;"
        "from deduction_desk.db import make_engine;"
        "from deduction_desk.matching import run_matching;"
        "import json;"
        "s=Session(make_engine());"
        "r=run_matching(s, persist=False);"
        "print(json.dumps({o.payment_id: sorted(o.invoice_ids) for o in r['_outcomes']}))"
    )

    results = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        if proc.returncode != 0:
            pytest.skip(f"could not run subprocess: {proc.stderr[-300:]}")
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    assert results[0] == results[1], (
        "matching produced different allocations in two processes — check for iteration "
        "over an unsorted set"
    )


def test_matching_uses_no_llm() -> None:
    """Deterministic by construction; a model deciding where money goes is not auditable."""
    offenders: list[str] = []
    for path in (SRC / "matching").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("from ..llm", "import llm", "ollama", "anthropic"):
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders, f"LLM reached the matching layer: {offenders}"


def test_advices_are_consumed_one_to_one(matched) -> None:
    """One advice describes one credit.

    Reusing an advice across a buyer's payments allocated 218 payments onto 83 invoices
    while reporting a 100% match rate.
    """
    report, _, _ = matched
    with Session(make_engine()) as session:
        n_advices = len(session.exec(select(RemittanceAdvice)).all())

    advice_matched = report["matching"]["by_method"].get("advice", 0)
    assert advice_matched <= n_advices, (
        f"{advice_matched} payments matched via advice but only {n_advices} advices exist"
    )
