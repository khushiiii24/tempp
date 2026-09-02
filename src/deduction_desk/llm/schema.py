"""Structured-output contracts, and the validate-then-repair loop.

Everything the model returns is constrained by a Pydantic model whose
`model_json_schema()` is handed to the provider as a grammar (Ollama's `format`
parameter). We do not ask for JSON in the prompt and hope — a prompt-only instruction is
advisory, a schema is enforced during decoding.

**Why these schemas are so small.** On a CPU-only box, generation is the entire cost:
prompt evaluation is fast and batched, but every output token is a full forward pass at
single-digit tokens/second. A rationale field that averages 40 tokens instead of 150
takes roughly ninety minutes off a full 400-invoice batch. So:

* codes are short enum members, not sentences
* `rationale` is capped at 200 characters and told to skip restating the input
* `evidence_needed` is an enum list, not free text — same information, a fifth of the tokens
* amounts come back as *verbatim strings*, never as computed paise

That last one is the important design call. The source documents are denominated in
rupees; asking a 7B model to multiply by 100 and hand back paise is asking it to do
arithmetic it will silently get wrong, in a money pipeline. Instead it copies the digits
it can see, and `money.rupees_to_paise` does the conversion deterministically in Python.
A model that cannot see a number returns null and the case becomes an exception, which is
the honest outcome.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

from ..config import load_taxonomy
from .errors import SchemaValidationFailed

# The label space is generated from config/reason_codes.yaml so that the taxonomy has
# exactly one definition. Adding a code to the YAML adds it to the grammar automatically.
_TAXONOMY = load_taxonomy()
ReasonCodeEnum = StrEnum("ReasonCodeEnum", {c: c for c in _TAXONOMY.all_codes})

ABSTAIN_CODE = "NEEDS_HUMAN"


class EvidenceKind(StrEnum):
    """What a human would need to fetch to settle this case. Enum, not prose."""

    TDS_CERTIFICATE = "tds_certificate"
    FORM_26AS = "form_26as"
    GSTR7 = "gstr7"
    CONTRACT = "contract"
    CREDIT_NOTE = "credit_note"
    GRN = "grn"
    SCHEME_MASTER = "scheme_master"
    PAYMENT_HISTORY = "payment_history"
    DEBIT_NOTE = "debit_note"
    REMITTANCE_DETAIL = "remittance_detail"
    NONE = "none"


# ======================================================================================
# Task schemas
# ======================================================================================


class Classification(BaseModel):
    """Output of stage [4]. Roughly 90-120 output tokens when full.

    The two `before` validators here are not defensive padding — each removes a *repair
    round trip that was firing on every single call*. On a CPU box at ~7 tokens/second a
    repair is a second full generation, so eliminating a systematic one halves the cost of
    the entire classification stage.

    This is the thing to understand about grammar-constrained decoding: it enforces
    **structure and type**, not **value**. A JSON Schema `maximum: 1` or `maxLength: 200`
    cannot be expressed in the GBNF grammar Ollama compiles the schema into, so the model
    is perfectly free to emit `"confidence": 100` or a 400-character rationale, and it
    will. The schema constrains the shape; only Python can constrain the range.
    """

    model_config = ConfigDict(extra="forbid")

    code: ReasonCodeEnum = Field(description="Reason code. NEEDS_HUMAN if genuinely unsure.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Decimal between 0.0 and 1.0 (e.g. 0.85). NOT a percentage.",
    )
    rationale: str = Field(
        max_length=200,
        description="Under 200 chars. Cite the arithmetic. Do not restate the input.",
    )
    check: str | None = Field(
        default=None,
        max_length=120,
        description="The arithmetic that decides it, e.g. '19200/960000 = 2.00% -> 194C'.",
    )
    evidence_needed: list[EvidenceKind] = Field(
        default_factory=list, max_length=2, description="At most 2."
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalise_confidence(cls, value: Any) -> Any:
        """Accept a percentage and convert it, rather than paying for a repair.

        The model reliably returns `100` meaning "certain". Rejecting that costs a second
        generation to be told the same thing in different units. Anything above 1 is read
        as a percentage; the only genuinely ambiguous input is exactly `1`, which is taken
        as full confidence because a model reporting "1% confident" is not a thing that
        happens, whereas one reporting "1" meaning certain is.
        """
        if isinstance(value, (int, float)) and value > 1:
            return float(value) / 100.0
        return value

    @field_validator("rationale", "check", mode="before")
    @classmethod
    def _truncate_prose(cls, value: Any, info: ValidationInfo) -> Any:
        """Trim over-long prose to that field's own cap instead of rejecting it.

        The cap exists to hold down output tokens, and by the time we can see the text
        those tokens are already spent — rejecting it buys nothing and costs a whole
        second generation. These fields are explanatory, never decision-relevant: no
        policy rule reads them. Truncating keeps the record honest and the clock moving.
        """
        limits = {"rationale": 200, "check": 120}
        limit = limits.get(info.field_name or "", 200)
        if isinstance(value, str) and len(value) > limit:
            return value[: limit - 3] + "..."
        return value


class AdviceLine(BaseModel):
    """One row of a remittance advice.

    Amounts are copied verbatim from the source text — the model must not convert,
    total or reconcile them. Nulls are strongly preferred over guesses.
    """

    model_config = ConfigDict(extra="forbid")

    invoice_ref: str | None = Field(
        default=None, max_length=40, description="Exactly as written. Null if absent."
    )
    gross_amount: str | None = Field(
        default=None, max_length=24, description="Digits exactly as written. No conversion."
    )
    deduction_amount: str | None = Field(default=None, max_length=24)
    stated_reason: str | None = Field(default=None, max_length=120)


class ParsedAdvice(BaseModel):
    """Output of stage [1]. Must never invent an invoice number absent from the text."""

    model_config = ConfigDict(extra="forbid")

    lines: list[AdviceLine] = Field(default_factory=list, max_length=12)


class DraftMessage(BaseModel):
    """Output of stage [7]. The policy engine has already fixed the channel, recipient,
    timing and template class; the model only writes prose inside those bounds."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(max_length=120)
    body: str = Field(max_length=1200)


class DoctorPing(BaseModel):
    """Trivial round-trip used by `doctor` to prove schema-constrained decoding works."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    code: ReasonCodeEnum


TASK_SCHEMAS: dict[str, type[BaseModel]] = {
    "classify": Classification,
    "parse": ParsedAdvice,
    "draft": DraftMessage,
    "doctor": DoctorPing,
}


def abstention() -> Classification:
    """The result when the model could not produce a valid answer.

    A first-class outcome, not a failure. It is deliberately impossible to reach a
    non-abstain classification through the failure path — see `validate_or_repair`.
    """
    return Classification(
        code=ReasonCodeEnum(ABSTAIN_CODE),
        confidence=0.0,
        rationale="Model output failed schema validation after repair attempts.",
        check=None,
        evidence_needed=[EvidenceKind.REMITTANCE_DETAIL],
    )


# ======================================================================================
# Validation and repair
# ======================================================================================


@dataclass
class RepairAttempt:
    """One failed validation, recorded for DecisionLog."""

    attempt: int
    error: str
    raw_excerpt: str

    def as_dict(self) -> dict[str, Any]:
        return {"attempt": self.attempt, "error": self.error, "raw_excerpt": self.raw_excerpt}


@dataclass
class ParseOutcome:
    value: BaseModel | None
    repairs: list[RepairAttempt] = field(default_factory=list)
    abstained: bool = False

    def repairs_as_dicts(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self.repairs]


def _excerpt(raw: str, limit: int = 400) -> str:
    raw = (raw or "").strip()
    return raw if len(raw) <= limit else raw[:limit] + f"... [{len(raw)} chars]"


def _format_validation_error(exc: ValidationError) -> str:
    """A compact, model-readable rendering of what was wrong.

    Pydantic's default repr is long and includes URLs; on a local model those tokens are
    expensive and unhelpful. This keeps the field path and the message only.
    """
    parts = []
    for err in exc.errors()[:6]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def build_repair_prompt(original_prompt: str, raw: str, error: str, schema: dict[str, Any]) -> str:
    """Re-ask, showing the model exactly what it got wrong.

    Deliberately does not repeat the full original prompt's supporting data — only the
    task line, the bad output and the error. Repair prompts are pure overhead, so they
    are kept short.
    """
    return (
        "Your previous JSON response was rejected by the schema validator.\n\n"
        f"VALIDATION ERROR: {error}\n\n"
        f"YOUR PREVIOUS OUTPUT:\n{_excerpt(raw, 600)}\n\n"
        "Return corrected JSON that satisfies this schema exactly. "
        "Respect every maximum length. Use null for anything you cannot determine. "
        "Do not add fields.\n\n"
        f"SCHEMA:\n{json.dumps(schema, separators=(',', ':'), sort_keys=True)}\n\n"
        "ORIGINAL TASK (abridged):\n"
        f"{_excerpt(original_prompt, 900)}"
    )


def parse_response(raw: str, model_cls: type[BaseModel]) -> BaseModel:
    """Validate a raw response string against the schema.

    `json.loads` then `model_validate` — no regex, no brace-hunting, no string slicing.
    If the provider's grammar constraint worked, this is exact. If it did not, this
    raises and the repair loop takes over. Fishing a JSON object out of prose with a
    regex is precisely how a malformed answer becomes a confident wrong one.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError.from_exception_data(
            model_cls.__name__,
            [
                {
                    "type": "json_invalid",
                    "loc": ("<root>",),
                    "input": raw,
                    "ctx": {"error": str(exc)},
                }
            ],
        ) from exc
    return model_cls.model_validate(payload)


def validate_or_repair(
    raw: str,
    model_cls: type[BaseModel],
    *,
    reinvoke: Callable[[str], str],
    original_prompt: str,
    max_attempts: int = 2,
) -> ParseOutcome:
    """Validate; on failure re-ask with the error; give up into an abstention.

    `reinvoke` takes a repair prompt and returns a fresh raw response. It is injected
    rather than reached for so this function stays provider-agnostic and unit-testable
    without a model.

    On exhaustion this raises `SchemaValidationFailed`. The caller converts that into an
    abstention. It never returns a partially-parsed or guessed object: an unparseable
    response carries zero information, and manufacturing a classification from it would
    put an invented reason code into a pipeline that decides money.
    """
    schema = model_cls.model_json_schema()
    repairs: list[RepairAttempt] = []
    current = raw

    for attempt in range(max_attempts + 1):
        try:
            return ParseOutcome(value=parse_response(current, model_cls), repairs=repairs)
        except ValidationError as exc:
            error = _format_validation_error(exc)
            repairs.append(
                RepairAttempt(attempt=attempt + 1, error=error, raw_excerpt=_excerpt(current))
            )
            if attempt >= max_attempts:
                raise SchemaValidationFailed(
                    attempts=attempt + 1, last_error=error, raw=_excerpt(current)
                ) from exc
            current = reinvoke(build_repair_prompt(original_prompt, current, error, schema))

    raise AssertionError("unreachable")  # pragma: no cover
