"""Phase 3 acceptance for the deterministic half: does verification get the money right?

**The oracle-classifier test is the important one.** It feeds the *true* reason code into
the verifier and asks whether the recoverable rupees come out matching ground truth. That
deliberately removes the LLM from the measurement, which matters because the two failure
modes need entirely different fixes: a classifier that mislabels needs a better prompt or
a better model, while a verifier that misreads a fixture needs a code change. Measured
together they are indistinguishable, and you end up tuning prompts against a bug in a CSV
lookup.

With the oracle the verifier scores exactly 100%, so every rupee of error in the end-to-end
scoreboard is attributable to classification. That is a much stronger claim than a good
end-to-end number, and it is the reason this file exists separately.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import Session, select

from deduction_desk.config import load_policy, load_taxonomy
from deduction_desk.db import make_engine
from deduction_desk.eval.slices import load_labelled_cases
from deduction_desk.generator import generate
from deduction_desk.schemas import PaymentEvent, Verdict
from deduction_desk.verify import FixtureStore, verify_deduction

N = 400

SRC = Path(__file__).resolve().parents[1] / "src" / "deduction_desk"


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("verify_batch")
    db = tmp / "v.db"
    fixtures = tmp / "fixtures"
    generate(seed=42, n=N, db_file=db, fixtures_dir=fixtures)

    cases = load_labelled_cases(db)
    with Session(make_engine(db)) as s:
        payments = {p.id: p for p in s.exec(select(PaymentEvent)).all()}

    return cases, FixtureStore(fixtures), payments


def _verify_all_with_oracle(batch):
    cases, store, payments = batch
    policy, taxonomy = load_policy(), load_taxonomy()
    out = []
    for c in cases:
        result = verify_deduction(
            c.deduction, c.invoice, c.contract, store, policy, taxonomy,
            payment=payments.get(c.deduction.payment_event_id),
            code=c.true_code,  # oracle: isolate verification from classification
        )
        out.append((c, result))
    return out


# ======================================================================================
# The money
# ======================================================================================


def test_oracle_verification_recovers_exact_rupees(batch) -> None:
    """Given the right label, the verifier must compute the right recoverable amount.

    Spec floor is 90%. Anything below that and the money layer, not the model, is the
    thing to fix.
    """
    results = _verify_all_with_oracle(batch)
    exact = sum(1 for c, r in results if r.recoverable_paise == c.recoverable_paise)
    rate = exact / len(results)

    mismatches = [
        f"{c.id} {c.true_code}: got {r.recoverable_paise} want {c.recoverable_paise} "
        f"({r.rules_fired})"
        for c, r in results
        if r.recoverable_paise != c.recoverable_paise
    ][:8]

    assert rate >= 0.90, f"only {rate:.1%} exact. Examples: {mismatches}"


def test_oracle_verification_never_invents_money(batch) -> None:
    """Claiming money that ground truth says is not there is the expensive error.

    It sends a chase letter to a customer who did nothing wrong, which is precisely the
    harm the whole design is built to avoid.
    """
    results = _verify_all_with_oracle(batch)
    invented = [
        (c.id, c.true_code, r.recoverable_paise)
        for c, r in results
        if c.recoverable_paise == 0 and r.recoverable_paise > 0
    ]
    assert not invented, f"verifier invented recoverable money on: {invented[:8]}"


def test_oracle_verification_does_not_abandon_money(batch) -> None:
    """The opposite error: silently writing off something genuinely recoverable."""
    results = _verify_all_with_oracle(batch)
    abandoned = [
        (c.id, c.true_code, c.recoverable_paise)
        for c, r in results
        if c.recoverable_paise > 0 and r.recoverable_paise == 0
    ]
    assert not abandoned, f"verifier abandoned recoverable money on: {abandoned[:8]}"


# ======================================================================================
# The 26AS lag — the behaviour that separates a careful agent from a naive one
# ======================================================================================


def test_lagging_26as_produces_provisional_close_not_a_chase(batch) -> None:
    """A legitimate TDS deduction missing from 26AS must be provisionally closed.

    26AS is filed quarterly. A missing row is not evidence of anything, and an agent that
    treats absence as disproof will chase customers who did nothing wrong — confidently,
    and at scale.
    """
    results = _verify_all_with_oracle(batch)
    provisional = [
        (c, r) for c, r in results if r.verdict == Verdict.PROVISIONAL_VALID
    ]

    assert provisional, "no provisional closes; the 26AS-lag behaviour is untested"

    for c, r in provisional:
        assert r.recoverable_paise == 0, f"{c.id} provisional but claims recovery"
        assert r.recheck_after_days, f"{c.id} provisional with no re-check date"
        # And every one of them must genuinely be a valid deduction.
        assert c.is_valid, f"{c.id} provisionally closed but ground truth says invalid"


def test_freight_verdict_is_decided_only_by_the_contract(batch) -> None:
    """Identical buyer behaviour, opposite verdicts, decided by delivery terms alone."""
    results = _verify_all_with_oracle(batch)
    freight = [(c, r) for c, r in results if c.true_code == "FREIGHT"]
    assert freight

    for c, r in freight:
        expected_valid = c.contract.delivery_terms == "for_destination"
        got_valid = r.recoverable_paise == 0
        assert got_valid is expected_valid, (
            f"{c.id}: delivery_terms={c.contract.delivery_terms} but "
            f"recoverable={r.recoverable_paise}"
        )


def test_rate_mismatch_recovers_only_the_excess(batch) -> None:
    """The buyer was obliged to withhold something. Chasing the whole deduction would be
    demanding money they were legally required to keep."""
    results = _verify_all_with_oracle(batch)
    mismatches = [(c, r) for c, r in results if c.true_code == "TDS_RATE_MISMATCH"]
    assert mismatches

    for c, r in mismatches:
        assert 0 < r.recoverable_paise < c.deduction.amount_paise, (
            f"{c.id}: recovered {r.recoverable_paise} of {c.deduction.amount_paise}; "
            f"the correctly-withheld portion must not be chased"
        )


def test_seller_billing_error_is_never_chased(batch) -> None:
    """RATE_DIFFERENCE is the seller's own error. There is nothing to recover."""
    results = _verify_all_with_oracle(batch)
    rate_diffs = [(c, r) for c, r in results if c.true_code == "RATE_DIFFERENCE"]
    assert rate_diffs

    for c, r in rate_diffs:
        assert r.recoverable_paise == 0, f"{c.id}: chasing our own billing error"
        assert r.evidence.get("overbilled_paise", 0) > 0, (
            f"{c.id}: should quantify the credit note owed"
        )


def test_abstention_is_not_verified_into_a_verdict(batch) -> None:
    """A NEEDS_HUMAN classification must not acquire a confident money answer.

    Verifying an abstention would launder uncertainty into a number, which is exactly what
    the abstention exists to prevent.
    """
    cases, store, payments = batch
    policy, taxonomy = load_policy(), load_taxonomy()
    c = cases[0]

    result = verify_deduction(
        c.deduction, c.invoice, c.contract, store, policy, taxonomy,
        payment=payments.get(c.deduction.payment_event_id),
        code="NEEDS_HUMAN",
    )

    assert result.verdict == Verdict.UNKNOWN
    assert result.recoverable_paise == 0


def test_every_result_names_the_rule_that_fired(batch) -> None:
    """`DecisionLog.policy_rules_fired` is only meaningful if the rules are named."""
    results = _verify_all_with_oracle(batch)
    for c, r in results:
        assert r.rules_fired, f"{c.id} produced a verdict with no named rule"


# ======================================================================================
# Layer discipline
# ======================================================================================


def test_no_llm_import_in_verify_or_policy() -> None:
    """Spec rule 1: the LLM never decides money.

    A model in these layers would make the recoverable amount vary between runs of the
    same seed, destroying both the scoreboard and the audit trail.
    """
    offenders: list[str] = []
    for package in ("verify", "policy"):
        pkg = SRC / package
        if not pkg.exists():
            continue
        for path in pkg.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for needle in ("from ..llm", "from deduction_desk.llm", "import llm", "anthropic", "ollama"):
                if needle in text:
                    offenders.append(f"{path.name}: {needle}")

    assert not offenders, f"LLM reached the decision layers: {offenders}"


def test_verification_is_deterministic(batch) -> None:
    """Same inputs, same rupees — every time, or the audit trail means nothing."""
    first = _verify_all_with_oracle(batch)
    second = _verify_all_with_oracle(batch)

    for (c1, r1), (c2, r2) in zip(first, second, strict=True):
        assert c1.id == c2.id
        assert r1.recoverable_paise == r2.recoverable_paise
        assert r1.verdict == r2.verdict
        assert r1.rules_fired == r2.rules_fired
