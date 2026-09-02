"""Reply text for the simulated buyer: static templates with slot-filling. No LLM.

Two reasons, and the second is the one that matters.

**Cost.** Reply text would be several hundred extra generations per run on a machine doing
~6 tokens/second, for text that no part of the pipeline reads for meaning. The agent
reacts to the response *kind* — payment, dispute, promise, deflection — which the state
machine has already decided.

**Defensibility.** If a model wrote the buyer's words, it is one small refactor from a
model deciding them, and at that point the agent is grading its own homework. Keeping the
counterparty entirely template-driven makes that mistake impossible to make by accident.

Selection is deterministic: the same case and contact always draw the same phrasing, so a
re-run produces a byte-identical transcript.
"""

from __future__ import annotations

from typing import Any

from ..config import CONFIG_DIR, load_yaml
from ..generator.seed import pick, rng_for
from ..money import format_inr
from ..schemas import InboundKind

_TEMPLATE_FILE = CONFIG_DIR / "templates" / "counterparty_replies.yaml"

# Which template group answers which inbound kind.
_KIND_TO_GROUP = {
    InboundKind.DEFLECTION: "deflection",
    InboundKind.PROMISE: "promise",
    InboundKind.PAYMENT: "payment",
    InboundKind.DISPUTE: "dispute",
    InboundKind.OPT_OUT: "opt_out",
    InboundKind.DOCUMENT: "document",
}


def _templates() -> dict[str, Any]:
    return load_yaml(_TEMPLATE_FILE)


def render_reply(
    *,
    kind: InboundKind,
    text_key: str,
    seed: int,
    case_id: str,
    slots: dict[str, Any],
) -> str:
    """Pick and fill a reply template. Deterministic for a given (case, kind, key)."""
    if kind == InboundKind.SILENCE:
        return ""

    templates = _templates()
    group_name = _KIND_TO_GROUP.get(kind)
    if group_name is None:
        return ""

    group = templates.get(group_name)
    if group is None:
        return ""

    # Some groups are a flat list; others are keyed by flavour (tds_pushback, junior_role).
    if isinstance(group, dict):
        options = group.get(text_key) or group.get("generic") or next(iter(group.values()), [])
    else:
        options = group

    if not options:
        return ""

    rng = rng_for(seed, "reply", f"{case_id}:{kind.value}:{text_key}")
    template = pick(rng, tuple(options))

    filled = template
    for slot, value in slots.items():
        filled = filled.replace("{" + slot + "}", str(value))
    # Collapse the YAML block-scalar line wrapping so replies read as one paragraph.
    return " ".join(filled.split())


def reply_slots(
    *,
    invoice_no: str,
    amount_paise: int,
    buyer_name: str,
    promise_date: str | None = None,
    section: str | None = None,
    scheme_id: str | None = None,
    credit_note_no: str | None = None,
    utr: str | None = None,
    case_ref: str | None = None,
) -> dict[str, Any]:
    """Everything a reply template might reference, with safe fallbacks."""
    return {
        "invoice_no": invoice_no,
        "amount": format_inr(amount_paise),
        "buyer_name": buyer_name,
        "promise_date": promise_date or "",
        "section": section or "194C",
        "scheme_id": scheme_id or "",
        "credit_note_no": credit_note_no or "",
        "utr": utr or "",
        "case_ref": case_ref or "",
        "days": "",
        "contact_name": "Accounts Payable",
        "dn": "",
        "cn": credit_note_no or "",
    }
