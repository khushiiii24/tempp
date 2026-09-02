"""Stage [5]: verification. Deterministic adjudication against source data.

**No LLM below this line.** This is where rupees are determined, so the number must be
reproducible, auditable and identical on every run of the same seed. A model here would
mean the amount the agent chases could vary between runs, which destroys the scoreboard
and the audit trail at once.

The dispatcher routes each classified code to its verifier via the `verifier:` field in
`config/reason_codes.yaml` — so adding a reason code is a config change plus a verifier,
never a change to this file's control flow.

One deliberate asymmetry runs through the whole layer: **absence means different things
in different stores.** The credit-note ledger and the GRN log are our own systems, so a
missing row is conclusive. Form 26AS and GSTR-7 are filed quarterly by someone else and
lag, so a missing row means nothing at all. Collapsing those two into a single "not found"
is the single most expensive mistake available here, because it chases customers who did
nothing wrong and does it with complete confidence.
"""

from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from ..config import Policy, Taxonomy, load_policy, load_taxonomy
from ..schemas import Contract, Deduction, Invoice, PaymentEvent, Verdict
from .base import VerificationResult, unknown
from .contract import verify_freight, verify_rate_difference, verify_unearned_discount
from .credit_notes import verify_credit_note_offset
from .duplicates import verify_deminimis, verify_duplicate_claim, verify_unexplained
from .grn import verify_goods_claim
from .scheme import verify_scheme_rebate
from .store import FixtureStore
from .tds import verify_tds

__all__ = [
    "FixtureStore",
    "VerificationResult",
    "verify_deduction",
    "verify_all",
]

_TDS_FAMILY = {
    "TDS_194C", "TDS_194J", "TDS_194H", "TDS_194Q",
    "GST_TDS", "TDS_RATE_MISMATCH", "TCS_194Q_OVERLAP",
}
_GOODS_FAMILY = {"DAMAGE_SHORTAGE", "QUALITY_REJECTION", "DEBIT_NOTE_BUYER"}
_DEMINIMIS = {"ROUNDING", "BANK_CHARGES"}


def verify_deduction(
    deduction: Deduction,
    invoice: Invoice,
    contract: Contract,
    store: FixtureStore,
    policy: Policy,
    taxonomy: Taxonomy,
    *,
    payment: PaymentEvent | None = None,
    code: str | None = None,
) -> VerificationResult:
    """Adjudicate one deduction. Pure function of its inputs."""
    code = code or deduction.predicted_code

    # An abstention is not something to verify. Forcing a verdict here would convert
    # "the classifier did not know" into a confident answer, which is exactly the
    # laundering of uncertainty the abstention exists to prevent.
    if code == "NEEDS_HUMAN":
        return unknown("classifier abstained", evidence_needed=["remittance_detail"])

    if code in _TDS_FAMILY:
        return verify_tds(deduction, invoice, contract, store, policy, taxonomy, code=code)

    if code == "FREIGHT":
        return verify_freight(deduction, invoice, contract, store, policy)

    if code == "RATE_DIFFERENCE":
        return verify_rate_difference(deduction, invoice, contract, store, policy)

    if code == "UNEARNED_DISCOUNT":
        return verify_unearned_discount(
            deduction, invoice, contract, store, policy, payment=payment
        )

    if code == "CREDIT_NOTE_OFFSET":
        return verify_credit_note_offset(deduction, invoice, store, policy)

    if code == "SCHEME_REBATE":
        return verify_scheme_rebate(deduction, invoice, store, policy)

    if code in _GOODS_FAMILY:
        return verify_goods_claim(deduction, invoice, store, policy, code=code)

    if code == "DUPLICATE_CLAIM":
        return verify_duplicate_claim(deduction, invoice, store, policy)

    if code in _DEMINIMIS:
        return verify_deminimis(deduction, invoice, policy, code=code)

    if code == "UNEXPLAINED":
        return verify_unexplained(deduction, invoice)

    return unknown(f"no verifier registered for {code}", evidence_needed=["remittance_detail"])


def verify_all(
    session: Session,
    *,
    store: FixtureStore | None = None,
    policy: Policy | None = None,
    taxonomy: Taxonomy | None = None,
    deduction_ids: list[str] | None = None,
) -> dict[str, VerificationResult]:
    """Verify every classified deduction and write the verdicts back."""
    store = store or FixtureStore()
    policy = policy or load_policy()
    taxonomy = taxonomy or load_taxonomy()

    deductions = list(session.exec(select(Deduction)).all())
    if deduction_ids is not None:
        wanted = set(deduction_ids)
        deductions = [d for d in deductions if d.id in wanted]
    deductions.sort(key=lambda d: d.id)

    invoices = {i.id: i for i in session.exec(select(Invoice)).all()}
    contracts = {c.buyer_id: c for c in session.exec(select(Contract)).all()}
    payments = {p.id: p for p in session.exec(select(PaymentEvent)).all()}

    results: dict[str, VerificationResult] = {}
    for d in deductions:
        invoice = invoices[d.invoice_id]
        result = verify_deduction(
            d,
            invoice,
            contracts[invoice.buyer_id],
            store,
            policy,
            taxonomy,
            payment=payments.get(d.payment_event_id),
        )
        results[d.id] = result

        d.verification = result.as_dict()
        d.verdict = result.verdict.value
        d.recoverable_paise = result.recoverable_paise
        session.add(d)

    session.commit()
    return results


def summarise(results: dict[str, VerificationResult]) -> dict[str, Any]:
    by_verdict: dict[str, int] = {}
    for r in results.values():
        by_verdict[r.verdict.value] = by_verdict.get(r.verdict.value, 0) + 1

    return {
        "n": len(results),
        "by_verdict": dict(sorted(by_verdict.items())),
        "recoverable_paise": sum(r.recoverable_paise for r in results.values()),
        "provisional": sum(1 for r in results.values() if r.is_provisional),
        "chaseable": sum(1 for r in results.values() if r.is_chaseable),
        "unknown": sum(1 for r in results.values() if r.verdict == Verdict.UNKNOWN),
    }
