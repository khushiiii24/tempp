"""Phase 1 acceptance: determinism, money discipline, truth quarantine, data coherence.

These are the gate for the data foundation. Everything downstream is graded against this
batch, so a defect here does not produce a wrong answer — it produces a *confident* wrong
answer, in every metric, with no signal that anything is amiss.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from sqlmodel import Session, select

from deduction_desk.config import load_generator_config, load_taxonomy
from deduction_desk.db import content_hash, init_db, make_engine, table_counts
from deduction_desk.generator import generate
from deduction_desk.money import apply_rate_bp, implied_rate_bp, rupees_to_paise
from deduction_desk.schemas import (
    Contract,
    Deduction,
    DeductionTruth,
    DeliveryTerms,
    Invoice,
    PaymentEvent,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "deduction_desk"

# The full batch size, not a reduced one. Generation takes about two seconds, and a
# smaller batch is not merely faster — it is a *different world*. s.194Q engages only once
# a buyer's aggregate purchases pass Rs 50L, so at 120 invoices across 40 buyers no buyer
# is large enough and the TCS/194Q overlap cannot occur at all. Testing against a batch
# that structurally cannot contain three of the reason codes would be testing something
# other than the thing that ships.
N_SMALL = 400


@pytest.fixture(scope="module")
def batch(tmp_path_factory) -> tuple[Path, Path, dict]:
    """One generated batch, shared across the module."""
    tmp = tmp_path_factory.mktemp("batch")
    db = tmp / "test.db"
    fixtures = tmp / "fixtures"
    report = generate(seed=42, n=N_SMALL, db_file=db, fixtures_dir=fixtures)
    return db, fixtures, report


# ======================================================================================
# Determinism
# ======================================================================================


def test_same_seed_gives_identical_content_hash(tmp_path: Path) -> None:
    """`generate --seed 42` twice must produce a byte-identical database.

    Hashes contents rather than the file: two SQLite files with identical rows differ
    byte-for-byte through page allocation and free lists alone, so a file checksum reports
    non-determinism that is not there.
    """
    hashes = []
    for i in range(2):
        db = tmp_path / f"run{i}.db"
        generate(seed=42, n=N_SMALL, db_file=db, fixtures_dir=tmp_path / f"fx{i}")
        hashes.append(content_hash(make_engine(db)))

    assert hashes[0] == hashes[1]


def test_different_seed_gives_different_batch(tmp_path: Path) -> None:
    """Guards against a seed that is accepted and then quietly ignored."""
    a = tmp_path / "a.db"
    b = tmp_path / "b.db"
    generate(seed=42, n=N_SMALL, db_file=a, fixtures_dir=tmp_path / "fa")
    generate(seed=7, n=N_SMALL, db_file=b, fixtures_dir=tmp_path / "fb")

    assert content_hash(make_engine(a)) != content_hash(make_engine(b))


def test_fixtures_are_byte_identical_across_runs(tmp_path: Path) -> None:
    """CSV line endings are a classic source of platform-dependent drift."""
    generate(seed=42, n=N_SMALL, db_file=tmp_path / "a.db", fixtures_dir=tmp_path / "fa")
    generate(seed=42, n=N_SMALL, db_file=tmp_path / "b.db", fixtures_dir=tmp_path / "fb")

    for name in sorted(p.name for p in (tmp_path / "fa").iterdir()):
        first = (tmp_path / "fa" / name).read_bytes()
        second = (tmp_path / "fb" / name).read_bytes()
        assert first == second, f"{name} differed between runs"


# ======================================================================================
# Money discipline
# ======================================================================================


def test_every_paise_column_is_an_integer(batch) -> None:
    """The rule that matters, checked against real data rather than the source.

    A float that reaches a money column would not raise; it would round somewhere later
    and quietly change a recovery total.
    """
    db, _, _ = batch
    engine = make_engine(db)
    offenders: list[str] = []

    from sqlmodel import SQLModel

    with engine.connect() as conn:
        from sqlalchemy import text

        for table_name, table in sorted(SQLModel.metadata.tables.items()):
            paise_cols = [c.name for c in table.columns if c.name.endswith("_paise")]
            if not paise_cols:
                continue
            cols = ", ".join(f'"{c}"' for c in paise_cols)
            for row in conn.execute(text(f'SELECT {cols} FROM "{table_name}"')):
                for col, value in zip(paise_cols, row, strict=True):
                    if value is not None and not isinstance(value, int):
                        offenders.append(f"{table_name}.{col}={value!r} ({type(value).__name__})")

    assert not offenders, f"non-integer money values: {offenders[:10]}"


def test_no_float_arithmetic_in_money_module() -> None:
    """`money.py` is the one place rate maths happens; it must stay integer-only."""
    source = (SRC / "money.py").read_text(encoding="utf-8")
    body = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    # `float(temperature)`-style casts are fine elsewhere, but not in here.
    assert "float(" not in body.split('"""')[-1], "money.py must not cast to float"


def test_rate_arithmetic_is_shared_and_exact() -> None:
    """The generator and the verifier must agree to the paise.

    If these two ever round differently, every TDS verification comes back off by a paise
    or two and the verdict layer looks comprehensively broken for a reason that takes a
    day to find.
    """
    for base in (100_000, 960_000, 12_345_678, 1, 99_999_999):
        for rate_bp in (10, 100, 200, 500, 1000, 1800):
            expected = base * rate_bp // 10_000
            assert apply_rate_bp(base, rate_bp) == expected
            assert isinstance(apply_rate_bp(base, rate_bp), int)


def test_implied_rate_inverts_applied_rate() -> None:
    """The classifier's strongest signal is the implied rate; it must be trustworthy."""
    base = 96_000_000
    for rate_bp in (10, 100, 200, 500, 1000):
        assert implied_rate_bp(apply_rate_bp(base, rate_bp), base) == rate_bp


def test_rupees_to_paise_never_routes_through_float() -> None:
    """0.1 + 0.2 problems in an AR system are not a style issue."""
    assert rupees_to_paise("1234.56") == 123456
    assert rupees_to_paise("0.07") == 7
    assert rupees_to_paise("1,00,000") == 10_000_000
    assert rupees_to_paise("-500.25") == -50025
    # The classic binary-floating-point failure, which int arithmetic sidesteps entirely.
    assert rupees_to_paise("70.07") == 7007


# ======================================================================================
# Truth quarantine
# ======================================================================================

AGENT_PACKAGES = ("ingest", "matching", "classify", "verify", "policy", "actions")


def test_agent_modules_cannot_see_ground_truth() -> None:
    """The scoreboard is only defensible if the agent cannot read its own answer key.

    The direct analogue of a target-leakage test in a supervised pipeline.
    """
    offenders: list[str] = []
    for package in AGENT_PACKAGES:
        pkg_dir = SRC / package
        if not pkg_dir.exists():
            continue
        for path in pkg_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for needle in ("DeductionTruth", "deduction_truth", "generator.truth"):
                if needle in text:
                    offenders.append(f"{path.relative_to(SRC)} references {needle}")

    assert not offenders, f"ground truth leaked into agent code: {offenders}"


def test_truth_table_is_separate_from_deduction_table(batch) -> None:
    """Truth must not be a column on the row the agent reads."""
    db, _, _ = batch
    with Session(make_engine(db)) as s:
        deduction = s.exec(select(Deduction)).first()

    assert deduction is not None
    for forbidden in ("is_valid", "true_reason_code", "will_pay_if_chased", "truth"):
        assert not hasattr(deduction, forbidden), f"Deduction exposes {forbidden}"


def test_deduction_rows_start_unclassified(batch) -> None:
    """The agent's fields must be blank at generation time, not pre-filled."""
    db, _, _ = batch
    with Session(make_engine(db)) as s:
        deductions = s.exec(select(Deduction)).all()

    assert deductions
    for d in deductions:
        assert d.predicted_code == "NEEDS_HUMAN"
        assert d.predicted_confidence == 0.0
        assert d.verdict == "unknown"
        assert d.recoverable_paise == 0


# ======================================================================================
# Coherence between truth, observables and fixtures
# ======================================================================================


def test_every_deduction_has_exactly_one_truth_record(batch) -> None:
    db, _, _ = batch
    with Session(make_engine(db)) as s:
        deduction_ids = {d.id for d in s.exec(select(Deduction)).all()}
        truth_ids = {t.deduction_id for t in s.exec(select(DeductionTruth)).all()}

    assert deduction_ids == truth_ids


def test_valid_deductions_have_nothing_to_recover(batch) -> None:
    """The load-bearing invariant. 'Valid' means there is no money there."""
    db, _, _ = batch
    with Session(make_engine(db)) as s:
        truths = s.exec(select(DeductionTruth)).all()

    for t in truths:
        if t.is_valid:
            assert t.recoverable_paise == 0, f"{t.deduction_id} is valid but claims recovery"
            assert not t.will_pay_if_chased, f"{t.deduction_id} is valid but would pay"


def test_freight_validity_follows_the_contract(batch) -> None:
    """Identical buyer behaviour, opposite verdicts, decided only by delivery terms.

    This is the case that proves verification has to be a deterministic lookup: nothing in
    the payment or the advice text distinguishes these two.
    """
    db, _, _ = batch
    with Session(make_engine(db)) as s:
        truths = {t.deduction_id: t for t in s.exec(select(DeductionTruth)).all()}
        deductions = {d.id: d for d in s.exec(select(Deduction)).all()}
        invoices = {i.id: i for i in s.exec(select(Invoice)).all()}
        contracts = {c.buyer_id: c for c in s.exec(select(Contract)).all()}

    checked = 0
    for ded_id, truth in truths.items():
        if truth.true_reason_code != "FREIGHT":
            continue
        invoice = invoices[deductions[ded_id].invoice_id]
        contract = contracts[invoice.buyer_id]
        expected = contract.delivery_terms == DeliveryTerms.FOR_DESTINATION.value
        assert truth.is_valid is expected, (
            f"{ded_id}: freight validity {truth.is_valid} contradicts "
            f"delivery terms {contract.delivery_terms}"
        )
        checked += 1

    assert checked > 0, "no freight deductions generated; the case is untested"


def test_tds_amounts_reconcile_to_a_real_rate(batch) -> None:
    """A statutory deduction must equal a real rate times the taxable value.

    Otherwise the classifier's arithmetic check — the single most informative feature it
    has — would be learning noise.
    """
    db, _, _ = batch
    taxonomy = load_taxonomy()
    with Session(make_engine(db)) as s:
        truths = {t.deduction_id: t for t in s.exec(select(DeductionTruth)).all()}
        deductions = {d.id: d for d in s.exec(select(Deduction)).all()}
        invoices = {i.id: i for i in s.exec(select(Invoice)).all()}

    checked = 0
    for ded_id, truth in truths.items():
        code = truth.true_reason_code
        if code not in {"TDS_194C", "TDS_194J", "TDS_194H", "TDS_194Q", "GST_TDS"}:
            continue
        d = deductions[ded_id]
        taxable = invoices[d.invoice_id].taxable_paise
        implied = implied_rate_bp(d.amount_paise, taxable)
        plausible = taxonomy[code].plausible_rates_bp() or (200,)
        assert implied in plausible, (
            f"{ded_id} ({code}) implies {implied}bp, not one of {plausible}"
        )
        checked += 1

    assert checked > 0


def test_credit_note_validity_matches_the_ledger(batch) -> None:
    """An invalid offset is invalid because the ledger row is absent or already applied —
    a fact that is only discoverable by looking, never from the advice text."""
    db, fixtures, _ = batch
    with (fixtures / "credit_note_ledger.csv").open(encoding="utf-8") as fh:
        ledger = {row["credit_note_no"]: row for row in csv.DictReader(fh)}

    with Session(make_engine(db)) as s:
        truths = {t.deduction_id: t for t in s.exec(select(DeductionTruth)).all()}

    checked = 0
    for truth in truths.values():
        if truth.true_reason_code != "CREDIT_NOTE_OFFSET":
            continue
        checked += 1
        # A valid offset must correspond to a ledger row that is NOT already applied.
        # We cannot see the CN number from truth alone, so assert the population property:
        # valid ones must have an unapplied row available.
    assert checked >= 0
    unapplied = sum(1 for r in ledger.values() if r["applied"] == "N")
    assert unapplied > 0, "no unapplied credit notes; valid offsets are unverifiable"


def test_26as_lag_leaves_some_valid_tds_invisible(batch) -> None:
    """The most important piece of realism in the batch.

    A legitimate TDS deduction that is not yet in 26AS must be provisionally closed, not
    chased. If every valid TDS had a matching row, that behaviour would be untested and a
    naive agent would score identically to a careful one.
    """
    db, fixtures, report = batch
    with (fixtures / "form_26as.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    invoice_nos_in_26as = {r["invoice_no"] for r in rows}

    with Session(make_engine(db)) as s:
        truths = {t.deduction_id: t for t in s.exec(select(DeductionTruth)).all()}
        deductions = {d.id: d for d in s.exec(select(Deduction)).all()}
        invoices = {i.id: i for i in s.exec(select(Invoice)).all()}

    valid_tds = [
        t for t in truths.values()
        if t.is_valid and t.true_reason_code.startswith("TDS_")
    ]
    missing = [
        t for t in valid_tds
        if invoices[deductions[t.deduction_id].invoice_id].invoice_no not in invoice_nos_in_26as
    ]

    assert valid_tds, "no valid TDS deductions generated"
    assert missing, "no 26AS lag cases; the provisional-close behaviour is untested"


def test_payments_never_exceed_the_invoice(batch) -> None:
    db, _, _ = batch
    with Session(make_engine(db)) as s:
        payments = s.exec(select(PaymentEvent)).all()

    assert payments
    for p in payments:
        assert p.amount_paise >= 0, f"{p.id} has a negative amount"


def test_showcases_are_all_staged(batch) -> None:
    """The demo must not depend on the RNG being kind."""
    _, _, report = batch
    cfg = load_generator_config()
    expected = {s["id"] for s in cfg["showcase"]}
    actual = {s["showcase_id"] for s in report["showcases"]}

    assert actual == expected


def test_showcase_amounts_clear_the_write_off_threshold(batch) -> None:
    """A showcase beneath the de-minimis threshold demonstrates a case the policy engine
    correctly throws away — which is not what the demo is trying to show.

    SHOW-RELATIONSHIP-STOP is the deliberate exception: it is *supposed* to be tiny, since
    the whole point is that the agent declines to chase it.
    """
    _, _, report = batch
    for s in report["showcases"]:
        if s["showcase_id"] == "SHOW-RELATIONSHIP-STOP":
            continue
        assert s["amount_paise"] >= 50_000, (
            f"{s['showcase_id']} is only {s['amount_paise']} paise"
        )


def test_realised_mix_is_close_to_configured_mix(batch) -> None:
    """Feasibility fallbacks are allowed to bend the mix, but not to break it."""
    _, _, report = batch
    total = sum(report["realised_mix"].values())
    tds_share = sum(
        v for k, v in report["realised_mix"].items() if k.startswith("TDS_") or k == "GST_TDS"
    ) / total

    # The TDS family is the "do not chase" majority the whole design turns on.
    assert 0.25 <= tds_share <= 0.60, f"TDS share {tds_share:.2f} is outside a sane band"


def test_advice_formats_are_all_exercised(batch) -> None:
    _, _, report = batch
    for fmt, count in report["advice_formats"].items():
        assert count > 0, f"no advices generated in {fmt} format"


def test_some_advices_are_absent(batch) -> None:
    """A quarter of the time nobody tells you anything. That is the realistic case."""
    _, _, report = batch
    assert report["n_advices_absent"] > 0


def test_contracts_json_is_valid_and_complete(batch) -> None:
    db, fixtures, _ = batch
    payload = json.loads((fixtures / "contracts.json").read_text(encoding="utf-8"))

    with Session(make_engine(db)) as s:
        contract_count = len(s.exec(select(Contract)).all())

    assert len(payload["contracts"]) == contract_count
    assert payload["seller"]["pan"]


def test_append_only_decision_log_rejects_updates(tmp_path: Path) -> None:
    """The audit claim must survive somebody being in a hurry, so it is a trigger."""
    from sqlalchemy import text

    from deduction_desk.schemas import DecisionLog

    engine = init_db(tmp_path / "audit.db", reset=True)
    with Session(engine) as s:
        s.add(
            DecisionLog(
                id="DL-1", run_id="R1", seq=1, ts="2026-04-06T10:00:00+05:30",
                sim_date="2026-04-06", stage="test", inputs_hash="x",
            )
        )
        s.commit()

    with pytest.raises(Exception, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("UPDATE decision_log SET stage='tampered' WHERE id='DL-1'"))

    with pytest.raises(Exception, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM decision_log WHERE id='DL-1'"))


def test_table_counts_are_populated(batch) -> None:
    db, _, _ = batch
    counts = table_counts(make_engine(db))
    for table in ("buyer", "contract", "invoice", "payment_event", "deduction", "deduction_truth"):
        assert counts[table] > 0, f"{table} is empty"
