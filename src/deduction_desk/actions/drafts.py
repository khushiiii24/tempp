"""Message drafting: templates by default, LLM optionally, validator always.

The policy engine has already fixed the template class, the channel, the recipient and the
timing before anything here runs. Drafting only chooses words.

**Templates are the default and the LLM is opt-in**, which is the reverse of what the spec
sketches, for a measured reason: a 300-case run at ~50 seconds per generation is over four
hours of inference to write prose the validator constrains tightly anyway. The value the
model adds here is fluency, and fluency is not what makes an AR letter work — being
correct, mild and on time is. So `--draft llm` exists for the demo and for showing the
validator catching a bad draft, and the batch runs on templates.

Whichever path produced the words, `validator.py` checks them, and a rejected draft falls
back to the template. That fallback is the point: the system cannot emit an unchecked
message even if the model misbehaves, and the rejection is counted rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import CONFIG_DIR, Policy, load_yaml
from ..money import format_inr
from ..schemas import Buyer, Deduction, Invoice
from .validator import ValidationResult, validate_draft

_TEMPLATE_FILE = CONFIG_DIR / "templates" / "outbound.yaml"

DRAFT_SYSTEM_PROMPT = (
    "You write short, courteous accounts-receivable emails for an Indian B2B seller. "
    "You never threaten, never mention legal action or credit holds, never invent "
    "figures, and never include links. You state the invoice, the amount, and a clear "
    "ask. Four sentences at most."
)


@dataclass
class Draft:
    subject: str
    body: str
    template_id: str
    drafted_by: str  # "template" | "llm" | "template_after_rejection"
    validation: ValidationResult
    llm_call: dict[str, Any] | None = None

    @property
    def rejected(self) -> bool:
        return self.drafted_by == "template_after_rejection"


def _templates() -> dict[str, Any]:
    return load_yaml(_TEMPLATE_FILE)


def build_slots(
    *,
    buyer: Buyer,
    invoice: Invoice,
    deduction: Deduction,
    case_ref: str,
    amount_paise: int,
    evidence: list[str] | None = None,
    section: str | None = None,
) -> dict[str, str]:
    return {
        "buyer_name": buyer.name,
        "invoice_no": invoice.invoice_no,
        "amount": format_inr(amount_paise),
        "case_ref": case_ref,
        "evidence": ", ".join(evidence or []) or "supporting documentation",
        "section": section or "the relevant section",
        "deduction_reason": deduction.predicted_code,
        "contact_role": "your accounts payable team",
    }


def render_template(template_id: str, slots: dict[str, str]) -> tuple[str, str]:
    """Fill a static template. Falls back to `chase_1` for an unknown class."""
    templates = _templates()
    template = templates.get(template_id) or templates.get("chase_1") or {}
    subject = str(template.get("subject", "Payment query: {invoice_no}"))
    body = str(template.get("body", "Dear Sir/Madam,\n\nReference: case {case_ref}\n"))

    for slot, value in slots.items():
        subject = subject.replace("{" + slot + "}", value)
        body = body.replace("{" + slot + "}", value)
    return subject.strip(), body.strip()


def build_draft_prompt(template_id: str, slots: dict[str, str], tone: str) -> str:
    """Prompt for the LLM path. States the bounds explicitly, since the validator enforces
    them and a draft that violates them is wasted inference."""
    return f"""Write one accounts-receivable email.

Buyer: {slots['buyer_name']}
Invoice: {slots['invoice_no']}
Amount outstanding: {slots['amount']}
Case reference: {slots['case_ref']}
Stage: {template_id} ({tone})

Rules:
- Mention ONLY the amount {slots['amount']} and the invoice {slots['invoice_no']}.
  Any other figure will be rejected.
- Do NOT threaten, mention legal action, credit holds, or stopping supply.
- Do NOT include any URL or payment link.
- Include the case reference.
- Four sentences maximum. Courteous and factual.

Return JSON with `subject` and `body`."""


_TONE = {
    "chase_1": "first, neutral enquiry",
    "chase_2": "polite reminder",
    "chase_3": "follow-up, still courteous",
    "chase_4": "persistent but not harsh",
    "chase_escalated": "addressed to a more senior contact",
    "request_document": "asking for paperwork",
    "request_tds_certificate": "asking for a tax certificate, not for money",
}


def make_draft(
    *,
    template_id: str,
    slots: dict[str, str],
    policy: Policy,
    case_amount_paise: int,
    invoice_total_paise: int | None = None,
    llm_client=None,
    escalation_authorised: bool = False,
) -> Draft:
    """Produce a validated message.

    The template path is validated too. It should always pass — and asserting that in a
    test is worth more than assuming it, because a template edited to add a phrase the
    policy forbids would otherwise sail straight through.
    """
    subject, body = render_template(template_id, slots)
    llm_call: dict[str, Any] | None = None
    drafted_by = "template"

    if llm_client is not None:
        from ..llm.client import max_tokens_for
        from ..llm.schema import DraftMessage

        response = llm_client.complete_detailed(
            build_draft_prompt(template_id, slots, _TONE.get(template_id, "neutral")),
            schema=DraftMessage,
            max_tokens=max_tokens_for("draft"),
            temperature=0.0,
            task="draft",
            system=DRAFT_SYSTEM_PROMPT,
        )
        llm_call = response.as_llm_call_record()

        if isinstance(response.value, DraftMessage):
            candidate = validate_draft(
                subject=response.value.subject,
                body=response.value.body,
                policy=policy,
                case_amount_paise=case_amount_paise,
                invoice_total_paise=invoice_total_paise,
                escalation_authorised=escalation_authorised,
            )
            if candidate.ok:
                return Draft(
                    subject=response.value.subject,
                    body=response.value.body,
                    template_id=template_id,
                    drafted_by="llm",
                    validation=candidate,
                    llm_call=llm_call,
                )
            # Rejected. Fall back to the template and keep the reasons for the report.
            drafted_by = "template_after_rejection"
            fallback = validate_draft(
                subject=subject,
                body=body,
                policy=policy,
                case_amount_paise=case_amount_paise,
                invoice_total_paise=invoice_total_paise,
                escalation_authorised=escalation_authorised,
            )
            fallback.rejections = candidate.rejections
            return Draft(
                subject=subject,
                body=body,
                template_id=template_id,
                drafted_by=drafted_by,
                validation=fallback,
                llm_call=llm_call,
            )

    validation = validate_draft(
        subject=subject,
        body=body,
        policy=policy,
        case_amount_paise=case_amount_paise,
        invoice_total_paise=invoice_total_paise,
        escalation_authorised=escalation_authorised,
    )
    return Draft(
        subject=subject,
        body=body,
        template_id=template_id,
        drafted_by=drafted_by,
        validation=validation,
        llm_call=llm_call,
    )
