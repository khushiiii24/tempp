"""Stage [4]: classify each deduction into a reason code.

The LLM's entire job here is to read messy text and pick a label. It does not decide
whether the deduction is valid, it does not decide what is recoverable, and it does not
decide whether to chase — those are `verify/` and `policy/`, both deterministic. What it
produces is a *hypothesis with a confidence*, and everything downstream treats it as such.

Three properties this module guarantees to the rest of the pipeline:

* **Abstention is preserved, never repaired away.** A classification that comes back
  `NEEDS_HUMAN`, or below the configured confidence floor, is recorded as an abstention and
  routed to a human. It is never quietly upgraded to the model's second choice.
* **Order-independent results.** Calls run on a small worker pool and complete out of
  order; results are keyed by deduction id and consumed sorted, so the batch produces the
  same database regardless of thread scheduling.
* **Every call is auditable.** The prompt hash, cache key, repair attempts and token counts
  are returned for `DecisionLog`, and the full prompt and raw response sit in the committed
  cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlmodel import Session, select

from ..config import Taxonomy, load_policy, load_taxonomy
from ..llm.batch import LLMTask, run_llm_batch
from ..llm.client import LLMClient, max_tokens_for
from ..llm.schema import ABSTAIN_CODE, Classification
from ..schemas import Buyer, Contract, Deduction, Invoice
from .feasibility import infeasibility_reason
from .prompts import SYSTEM_PROMPT, build_classification_prompt


@dataclass
class ClassificationOutcome:
    """One classified deduction, plus the audit record for the decision log."""

    deduction_id: str
    code: str
    confidence: float
    rationale: str
    check: str | None
    evidence_needed: list[str]
    abstained: bool
    below_floor: bool
    llm_call: dict[str, Any]

    @property
    def actionable(self) -> bool:
        """Is this safe to act on, or must it go to a human?"""
        return not self.abstained and not self.below_floor


def _observed_history(
    deductions: list[Deduction], buyer_invoices: set[str], exclude_id: str
) -> dict[str, int]:
    """Coarse prior over what this buyer has deducted before.

    Built from *stated reason text* on other deductions — observable data, not truth. A
    real AR analyst carries this prior and it is genuinely informative, so withholding it
    would understate what the classifier can legitimately know.
    """
    history: dict[str, int] = {}
    for d in deductions:
        if d.id == exclude_id or d.invoice_id not in buyer_invoices:
            continue
        stated = (d.claimed_reason_text or "").lower()
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


def classify_batch(
    session: Session,
    client: LLMClient,
    *,
    taxonomy: Taxonomy | None = None,
    deduction_ids: list[str] | None = None,
    progress: bool = True,
) -> dict[str, ClassificationOutcome]:
    """Classify every deduction (or a named subset). Writes predictions back to the rows."""
    taxonomy = taxonomy or load_taxonomy()
    policy = load_policy()
    floor = float(policy.confidence["classify_floor"])

    deductions = list(session.exec(select(Deduction)).all())
    if deduction_ids is not None:
        wanted = set(deduction_ids)
        deductions = [d for d in deductions if d.id in wanted]
    deductions.sort(key=lambda d: d.id)

    all_deductions = list(session.exec(select(Deduction)).all())
    invoices = {i.id: i for i in session.exec(select(Invoice)).all()}
    buyers = {b.id: b for b in session.exec(select(Buyer)).all()}
    contracts = {c.buyer_id: c for c in session.exec(select(Contract)).all()}

    invoices_by_buyer: dict[str, set[str]] = {}
    for inv in invoices.values():
        invoices_by_buyer.setdefault(inv.buyer_id, set()).add(inv.id)

    tasks: list[LLMTask] = []
    for d in deductions:
        invoice = invoices[d.invoice_id]
        buyer = buyers[invoice.buyer_id]
        tasks.append(
            LLMTask(
                ref=d.id,
                prompt=build_classification_prompt(
                    taxonomy=taxonomy,
                    deduction=d,
                    invoice=invoice,
                    buyer=buyer,
                    contract=contracts[invoice.buyer_id],
                    buyer_history=_observed_history(
                        all_deductions, invoices_by_buyer.get(buyer.id, set()), d.id
                    ),
                    confidence_floor=floor,
                ),
                task="classify",
                schema=Classification,
                max_tokens=max_tokens_for("classify"),
                system=SYSTEM_PROMPT,
            )
        )

    results, _stats = run_llm_batch(client, tasks, label="classify", progress=progress)

    outcomes: dict[str, ClassificationOutcome] = {}
    for d in deductions:  # sorted; consumption order is independent of completion order
        response = results.get(d.id)
        if response is None:
            continue

        value = response.value
        code = value.code.value if isinstance(value, Classification) else ABSTAIN_CODE
        confidence = float(value.confidence) if isinstance(value, Classification) else 0.0
        rationale = value.rationale if isinstance(value, Classification) else ""
        check = value.check if isinstance(value, Classification) else None
        evidence = (
            [e.value for e in value.evidence_needed] if isinstance(value, Classification) else []
        )

        abstained = bool(response.abstained) or code == ABSTAIN_CODE

        # Deterministic feasibility gate. A code that is physically impossible for this
        # case — ROUNDING on a Rs 3,000 deduction, UNEARNED_DISCOUNT on a contract with no
        # discount clause — is treated as an abstention, not acted on.
        #
        # This is the project's real abstention signal. The model's self-reported
        # confidence was measured at 0.95 on 34 of 40 cases, identical for easy and hard
        # ones, so a confidence threshold never fires and would give a false sense of
        # safety. A demonstrably impossible answer is evidence the case was not understood,
        # which is worth considerably more than the model saying it is 95% sure.
        # Re-resolve per deduction. An earlier version relied on `invoice` leaking from the
        # task-building loop above, so every feasibility check was run against whichever
        # invoice happened to be last — gating cases on a stranger's contract. It produced
        # plausible-looking abstentions and a 30% abstention rate where the benchmark,
        # which resolves the contract correctly, measured 15%.
        invoice_for_case = invoices[d.invoice_id]
        contract_for_case = contracts[invoice_for_case.buyer_id]

        infeasible_reason = None
        if not abstained:
            infeasible_reason = infeasibility_reason(
                code, deduction=d, invoice=invoice_for_case, contract=contract_for_case
            )
            if infeasible_reason:
                abstained = True
                code = ABSTAIN_CODE
                rationale = f"Model returned an impossible code: {infeasible_reason}"

        below_floor = (not abstained) and confidence < floor

        # A prediction the model is not confident enough about is recorded as what it is.
        # It is never promoted to an actionable answer, and the confidence is kept so the
        # eval can report precision among non-abstained cases honestly.
        d.predicted_code = code
        d.predicted_confidence = confidence
        d.predicted_rationale = rationale or ""
        d.predicted_by = response.provider
        d.state = "classified" if (not abstained and not below_floor) else "needs_human"
        session.add(d)

        outcomes[d.id] = ClassificationOutcome(
            deduction_id=d.id,
            code=code,
            confidence=confidence,
            rationale=rationale,
            check=check,
            evidence_needed=evidence,
            abstained=abstained,
            below_floor=below_floor,
            llm_call=response.as_llm_call_record(),
        )

    session.commit()
    return outcomes


def summarise(outcomes: dict[str, ClassificationOutcome]) -> dict[str, Any]:
    """Counts for the run report. Abstention and low-confidence are reported separately —
    they are different failures and conflating them hides which one is happening."""
    total = len(outcomes)
    abstained = sum(1 for o in outcomes.values() if o.abstained)
    below = sum(1 for o in outcomes.values() if o.below_floor)
    by_code: dict[str, int] = {}
    for o in outcomes.values():
        by_code[o.code] = by_code.get(o.code, 0) + 1
    repairs = sum(len(o.llm_call.get("repairs") or []) for o in outcomes.values())
    cached = sum(1 for o in outcomes.values() if o.llm_call.get("cached"))

    return {
        "n": total,
        "abstained": abstained,
        "below_floor": below,
        "actionable": total - abstained - below,
        "abstention_rate": round(abstained / total, 4) if total else 0.0,
        "repairs": repairs,
        "cached": cached,
        "by_code": dict(sorted(by_code.items())),
    }
