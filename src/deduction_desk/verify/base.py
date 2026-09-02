"""Shared types for the verification layer.

**No LLM may be imported anywhere under `verify/`.** This is where rupees are determined,
and the number has to be reproducible, auditable and identical on every run. A model in
this layer would mean the amount the agent chases could change between runs of the same
seed, which destroys both the scoreboard and the audit trail.
`tests/test_no_llm_in_decision_layers.py` enforces it.

Every verifier returns the same shape: a verdict, a recoverable amount, the evidence it
relied on, and the named rules that fired. The rule names flow into
`DecisionLog.policy_rules_fired`, so "why did the agent decide this" is answerable from
the log alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas import Verdict


@dataclass
class VerificationResult:
    """The outcome of adjudicating one deduction against source data."""

    verdict: Verdict
    recoverable_paise: int
    evidence: dict[str, Any] = field(default_factory=dict)
    rules_fired: list[str] = field(default_factory=list)
    # Set when the verdict is provisional because a source system lags. The policy engine
    # uses this to schedule a re-check rather than a chase.
    recheck_after_days: int | None = None
    # What a human would need to settle it, when we genuinely cannot.
    evidence_needed: list[str] = field(default_factory=list)

    @property
    def is_provisional(self) -> bool:
        return self.verdict == Verdict.PROVISIONAL_VALID

    @property
    def is_chaseable(self) -> bool:
        return self.recoverable_paise > 0 and self.verdict in {
            Verdict.INVALID,
            Verdict.PARTIAL,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "recoverable_paise": self.recoverable_paise,
            "evidence": self.evidence,
            "rules_fired": self.rules_fired,
            "recheck_after_days": self.recheck_after_days,
            "evidence_needed": self.evidence_needed,
        }


def unknown(reason: str, *, evidence_needed: list[str] | None = None) -> VerificationResult:
    """We could not decide. Distinct from 'we decided it is invalid'."""
    return VerificationResult(
        verdict=Verdict.UNKNOWN,
        recoverable_paise=0,
        evidence={"reason": reason},
        rules_fired=["verify.undetermined"],
        evidence_needed=evidence_needed or [],
    )


def valid(rule: str, **evidence: Any) -> VerificationResult:
    """The deduction is legitimate. There is no money to recover — that is what valid means."""
    return VerificationResult(
        verdict=Verdict.VALID,
        recoverable_paise=0,
        evidence=evidence,
        rules_fired=[rule],
    )


def invalid(rule: str, recoverable_paise: int, **evidence: Any) -> VerificationResult:
    return VerificationResult(
        verdict=Verdict.INVALID,
        recoverable_paise=max(0, int(recoverable_paise)),
        evidence=evidence,
        rules_fired=[rule],
    )


def partial(rule: str, recoverable_paise: int, **evidence: Any) -> VerificationResult:
    """Part of the deduction is legitimate and part is not.

    The rate-mismatch case: the buyer was obliged to withhold *something*, just not this
    much. Chasing the whole deduction would be chasing money they were legally required to
    keep, which is the mistake that makes an AR team look incompetent to its own customer.
    """
    return VerificationResult(
        verdict=Verdict.PARTIAL,
        recoverable_paise=max(0, int(recoverable_paise)),
        evidence=evidence,
        rules_fired=[rule],
    )


def provisional_valid(
    rule: str, *, recheck_after_days: int, **evidence: Any
) -> VerificationResult:
    """Believe it for now, and look again later.

    The 26AS-lag case. The source system is quarterly and simply has not caught up, so a
    missing row is not evidence of anything. The correct behaviour is to close
    provisionally and flag for re-check — never to chase. An agent that cannot tell
    "disproved" from "not yet knowable" is not safe to point at a customer base.
    """
    return VerificationResult(
        verdict=Verdict.PROVISIONAL_VALID,
        recoverable_paise=0,
        evidence=evidence,
        rules_fired=[rule],
        recheck_after_days=recheck_after_days,
    )
