"""The draft validator: what the model is allowed to say, and what happens when it strays.

The policy engine chooses the template class, the channel, the recipient and the timing.
The model only writes prose inside those bounds. This module is the thing that checks it
stayed inside them, and it rejects on four grounds:

1. **Forbidden phrases** — threats, legal claims, blacklisting. From `policy.yaml`, so a
   compliance officer can add one without touching code.
2. **Numbers that are not in the case record.** A model that invents or mistypes a figure
   in a demand letter is worse than one that says nothing: the number looks authoritative
   and the customer will act on it.
3. **Payment links when policy said none.** A live link in an unauthorised message is a
   security question, not a style question.
4. **Escalation language the policy did not authorise** — a first reminder must not read
   like a final notice.

**Rejection is not failure; it is the control working.** A rejected draft falls back to the
static template, the case still progresses, and the rejection is counted and reported.
Hiding the rejection rate would make the LLM look better than it is and remove the only
signal that the drafting prompt needs work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import Policy
from ..money import format_inr

# Any http(s) URL. Payment links are the specific concern, but an unauthorised link of any
# kind in an AR letter is a phishing vector.
_URL = re.compile(r"https?://\S+", re.IGNORECASE)

# Rupee figures the model may have written, in any of the forms it tends to produce.
_MONEY = re.compile(r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE)

# Language that implies a consequence the agent has no authority to threaten. Distinct
# from `forbidden_phrases` in policy.yaml, which the operator owns; these are structural.
_UNAUTHORISED_ESCALATION = (
    "final notice",
    "final demand",
    "legal proceedings",
    "court",
    "arbitration",
    "credit hold",
    "stop supply",
    "suspend your account",
    "recovery proceedings",
)


@dataclass
class ValidationResult:
    ok: bool
    rejections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.ok = False
        self.rejections.append(reason)


def _normalise_amount(text: str) -> str:
    """Canonical form of a rupee figure, for comparison.

    Trailing zeros are stripped **only after a decimal point**. Stripping them
    unconditionally turns `12500` into `125`, which made the validator reject every
    correctly-drafted message — including all seven static templates. Every draft then
    fell back to the template and the rejection metric became pure noise, while looking
    like the control working.
    """
    cleaned = text.replace(",", "").strip()
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned


def _numbers_in(text: str) -> set[str]:
    """Rupee amounts mentioned in a draft, normalised for comparison."""
    return {_normalise_amount(m.group(1)) for m in _MONEY.finditer(text)}


def _permitted_numbers(*amounts_paise: int) -> set[str]:
    """The same amounts, in every form a model might legitimately render them."""
    permitted: set[str] = set()
    for paise in amounts_paise:
        if paise is None:
            continue
        rupees = int(paise) // 100
        permitted.add(str(rupees))
        permitted.add(_normalise_amount(f"{rupees}.{int(paise) % 100:02d}"))
        permitted.add(_normalise_amount(format_inr(paise).replace("Rs ", "")))
        permitted.add(str(int(paise)))
        # `format_inr` rounds half-up for display, so a template rendering the amount can
        # legitimately differ from the floor by one rupee.
        permitted.add(str(rupees + 1))
    return permitted


def validate_draft(
    *,
    subject: str,
    body: str,
    policy: Policy,
    case_amount_paise: int,
    invoice_total_paise: int | None = None,
    allow_payment_link: bool | None = None,
    escalation_authorised: bool = False,
) -> ValidationResult:
    """Check one drafted message against the bounds the policy engine set."""
    result = ValidationResult(ok=True)
    text = f"{subject}\n{body}"
    lowered = text.lower()

    # -- 1. forbidden phrases -------------------------------------------------------
    for phrase in policy.compliance.get("forbidden_phrases", []):
        if phrase.lower() in lowered:
            result.reject(f"forbidden_phrase:{phrase}")

    # -- 2. numbers not in the case record ------------------------------------------
    permitted = _permitted_numbers(case_amount_paise, invoice_total_paise)
    for mentioned in _numbers_in(text):
        if mentioned and mentioned not in permitted:
            # An invented figure in a demand letter is authoritative-looking and wrong,
            # which is worse than saying nothing at all.
            result.reject(f"unverified_amount:{mentioned}")

    # -- 3. links --------------------------------------------------------------------
    if allow_payment_link is None:
        allow_payment_link = bool(policy.drafting.get("allow_payment_link", False))
    if not allow_payment_link and _URL.search(text):
        result.reject("payment_link_not_authorised")

    # -- 4. escalation language -------------------------------------------------------
    if not escalation_authorised:
        for phrase in _UNAUTHORISED_ESCALATION:
            if phrase in lowered:
                result.reject(f"unauthorised_escalation:{phrase}")

    # -- housekeeping ------------------------------------------------------------------
    max_chars = int(policy.drafting.get("max_body_chars", 1400))
    if len(body) > max_chars:
        result.reject(f"body_too_long:{len(body)}>{max_chars}")

    if not body.strip():
        result.reject("empty_body")

    if policy.drafting.get("require_case_reference", True) and "case" not in lowered:
        # A warning, not a rejection: a missing reference is untidy, not dangerous, and
        # rejecting on it would discard otherwise-good drafts over formatting.
        result.warnings.append("no_case_reference")

    return result
