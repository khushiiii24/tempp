"""Labelled evaluation slices.

Benchmarking a model on 40 records drawn uniformly at random would be close to useless
here: the batch is roughly two-thirds valid TDS, so a uniform sample lands ~27 TDS cases
and one or two of everything else, and macro-F1 over that is dominated by sampling noise
on the classes that actually matter. A model could miss every recoverable case and still
score respectably.

So the slice is **stratified**: take up to `per_code` examples of each reason code present,
in deterministic id order, round-robin until the budget is filled. Every code that occurs
at all is represented, which is what makes a macro average meaningful at n=40.

This is the one place in `eval/` that reads `DeductionTruth`, which is allowed — the eval
harness is the grader, not the agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from ..db import make_engine
from ..schemas import Buyer, Contract, Deduction, DeductionTruth, Invoice


@dataclass
class LabelledCase:
    """One deduction plus everything a classifier needs, and the answer it is graded on."""

    deduction: Deduction
    invoice: Invoice
    buyer: Buyer
    contract: Contract
    true_code: str
    is_valid: bool
    recoverable_paise: int

    @property
    def id(self) -> str:
        return self.deduction.id


def load_labelled_cases(db_file: Path | None = None) -> list[LabelledCase]:
    """Every deduction in the batch, joined to its context and its ground-truth label."""
    engine = make_engine(db_file)
    with Session(engine) as s:
        deductions = {d.id: d for d in s.exec(select(Deduction)).all()}
        truths = {t.deduction_id: t for t in s.exec(select(DeductionTruth)).all()}
        invoices = {i.id: i for i in s.exec(select(Invoice)).all()}
        buyers = {b.id: b for b in s.exec(select(Buyer)).all()}
        contracts = {c.buyer_id: c for c in s.exec(select(Contract)).all()}

    cases: list[LabelledCase] = []
    for ded_id in sorted(deductions):
        truth = truths.get(ded_id)
        if truth is None:
            continue
        invoice = invoices[deductions[ded_id].invoice_id]
        cases.append(
            LabelledCase(
                deduction=deductions[ded_id],
                invoice=invoice,
                buyer=buyers[invoice.buyer_id],
                contract=contracts[invoice.buyer_id],
                true_code=truth.true_reason_code,
                is_valid=truth.is_valid,
                recoverable_paise=truth.recoverable_paise,
            )
        )
    return cases


def stratified_slice(
    cases: list[LabelledCase], *, size: int = 40, per_code: int = 4
) -> list[LabelledCase]:
    """A deterministic, class-balanced subset.

    Round-robin across codes so that a 40-record slice covers the taxonomy rather than
    re-sampling the majority class. Deterministic: no RNG, just sorted ids, so a benchmark
    re-run compares models on exactly the same records.
    """
    by_code: dict[str, list[LabelledCase]] = {}
    for case in sorted(cases, key=lambda c: c.id):
        by_code.setdefault(case.true_code, []).append(case)

    chosen: list[LabelledCase] = []
    round_index = 0
    while len(chosen) < size and round_index < per_code:
        added_this_round = False
        for code in sorted(by_code):
            if len(chosen) >= size:
                break
            bucket = by_code[code]
            if round_index < len(bucket):
                chosen.append(bucket[round_index])
                added_this_round = True
        if not added_this_round:
            break
        round_index += 1

    return sorted(chosen, key=lambda c: c.id)


def buyer_history(cases: list[LabelledCase], buyer_id: str, exclude_id: str) -> dict[str, int]:
    """Prior deduction codes seen from this buyer.

    Uses the *claimed* reason text bucket rather than the true code — the agent may not
    look at truth. In practice an AR analyst does carry this prior, and it is genuinely
    informative, so withholding it entirely would understate what the classifier can know.
    Here it is derived only from what was previously observed and stated.
    """
    history: dict[str, int] = {}
    for case in cases:
        if case.buyer.id != buyer_id or case.deduction.id == exclude_id:
            continue
        stated = (case.deduction.claimed_reason_text or "").lower()
        if not stated:
            continue
        bucket = (
            "TDS-like" if "tds" in stated or "194" in stated
            else "freight" if "fr" in stated or "transport" in stated or "lorry" in stated
            else "scheme" if "scheme" in stated or "qps" in stated
            else "credit-note" if "cn" in stated or "credit note" in stated
            else "quality/damage" if "damag" in stated or "reject" in stated or "short" in stated
            else "other"
        )
        history[bucket] = history.get(bucket, 0) + 1
    return history


def slice_summary(slice_: list[LabelledCase]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for case in slice_:
        counts[case.true_code] = counts.get(case.true_code, 0) + 1
    return {
        "n": len(slice_),
        "n_codes": len(counts),
        "by_code": dict(sorted(counts.items())),
        "n_valid": sum(1 for c in slice_ if c.is_valid),
        "n_recoverable": sum(1 for c in slice_ if c.recoverable_paise > 0),
    }
