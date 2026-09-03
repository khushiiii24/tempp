"""Tests for the provider abstraction, cache, offline mode and the repair loop.

None of these need a model running. That is the point: the LLM layer's contracts are
testable without inference, so a broken cache or a broken repair path is caught in
milliseconds rather than four hours into an overnight batch.

The determinism assertion that requires a live model is in `test_llm_determinism.py` and
is marked `live_llm`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from deduction_desk.llm.cache import LLMCache, make_key, schema_hash
from deduction_desk.llm.client import LLMClient, size_context_window
from deduction_desk.llm.errors import LLMCacheMiss, SchemaValidationFailed
from deduction_desk.llm.schema import (
    Classification,
    ParsedAdvice,
    build_repair_prompt,
    parse_response,
    validate_or_repair,
)


class ScriptedClient(LLMClient):
    """A client that returns pre-scripted raw strings. Records how often it was invoked."""

    provider_name = "scripted"

    def __init__(self, cache: LLMCache, script: list[str], *, offline: bool = False) -> None:
        super().__init__({"repair": {"max_attempts": 2}}, cache=cache, offline=offline)
        self.script = list(script)
        self.calls = 0

    @property
    def model(self) -> str:
        return "scripted-model"

    def _invoke(self, prompt, *, schema, max_tokens, temperature, system):
        self.calls += 1
        raw = self.script.pop(0) if self.script else "{}"
        return raw, {"eval_count": 10, "prompt_eval_count": 5, "num_ctx": 8192}

    def health(self) -> dict[str, Any]:
        return {"provider": self.provider_name, "reachable": True}


VALID_CLASSIFICATION = json.dumps(
    {
        "code": "TDS_194C",
        "confidence": 0.91,
        "rationale": "19200/960000 = 2.00%, matches 194C contract rate.",
        "check": "19200/960000 = 2.00%",
        "evidence_needed": ["form_26as"],
    }
)


# ======================================================================================
# Cache
# ======================================================================================


def test_cache_key_changes_with_model(tmp_path: Path) -> None:
    """A scoreboard produced by one model must never be attributed to another."""
    digest = schema_hash(Classification.model_json_schema())
    a = make_key(provider="ollama", model="model-a", prompt="p", temperature=0.0, schema_digest=digest)
    b = make_key(provider="ollama", model="model-b", prompt="p", temperature=0.0, schema_digest=digest)
    assert a != b


def test_cache_key_changes_with_schema(tmp_path: Path) -> None:
    a = schema_hash(Classification.model_json_schema())
    b = schema_hash(ParsedAdvice.model_json_schema())
    assert a != b


def test_cache_round_trip_and_hit(tmp_path: Path) -> None:
    cache = LLMCache(tmp_path)
    client = ScriptedClient(cache, [VALID_CLASSIFICATION])

    first = client.complete_detailed("prompt", schema=Classification, task="classify")
    assert first.cached is False
    assert first.value.code == "TDS_194C"
    assert client.calls == 1

    second = client.complete_detailed("prompt", schema=Classification, task="classify")
    assert second.cached is True
    assert second.value.code == "TDS_194C"
    assert client.calls == 1, "a cache hit must not reach the provider"


def test_cache_writes_are_readable_json(tmp_path: Path) -> None:
    """The cache is a committed audit artefact, so entries must be human-inspectable."""
    cache = LLMCache(tmp_path)
    client = ScriptedClient(cache, [VALID_CLASSIFICATION])
    client.complete_detailed("prompt", schema=Classification, task="classify")

    files = list((tmp_path / "classify").glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8"))
    # The full prompt and the raw response are both retained — a decision log that
    # records only the conclusion is not an audit trail.
    for required in ("prompt", "response_text", "parsed", "model", "schema_hash", "meta"):
        assert required in record, f"cache entry missing {required}"


# ======================================================================================
# Offline mode
# ======================================================================================


def test_offline_raises_on_miss_and_never_calls_provider(tmp_path: Path) -> None:
    cache = LLMCache(tmp_path, offline=True)
    client = ScriptedClient(cache, [VALID_CLASSIFICATION], offline=True)

    with pytest.raises(LLMCacheMiss) as exc:
        client.complete_detailed("uncached prompt", schema=Classification, task="classify")

    assert client.calls == 0, "offline mode must be incapable of reaching the provider"
    assert exc.value.prompt_sha256
    assert exc.value.prompt_sha256 in str(exc.value), "the miss must name the prompt hash"


def test_offline_serves_a_warm_cache(tmp_path: Path) -> None:
    """The claim the README makes: clone, run --offline, get the scoreboard."""
    warm = ScriptedClient(LLMCache(tmp_path), [VALID_CLASSIFICATION])
    warm.complete_detailed("prompt", schema=Classification, task="classify")

    offline = ScriptedClient(LLMCache(tmp_path, offline=True), [], offline=True)
    response = offline.complete_detailed("prompt", schema=Classification, task="classify")

    assert response.cached is True
    assert response.value.code == "TDS_194C"
    assert offline.calls == 0


# ======================================================================================
# Schema validation and repair
# ======================================================================================


def test_parse_response_rejects_prose_without_regex_rescue() -> None:
    """Fishing JSON out of prose with a regex is how a malformed answer becomes a
    confident wrong one. We validate or we fail."""
    with pytest.raises(ValidationError):
        parse_response("Sure! Here is the JSON: {\"code\": \"TDS_194C\"}", Classification)


def test_repair_loop_recovers_on_second_attempt() -> None:
    bad = json.dumps({"code": "NOT_A_REAL_CODE", "confidence": 0.5, "rationale": "x"})
    attempts: list[str] = []

    def reinvoke(prompt: str) -> str:
        attempts.append(prompt)
        return VALID_CLASSIFICATION

    outcome = validate_or_repair(
        bad, Classification, reinvoke=reinvoke, original_prompt="orig", max_attempts=2
    )

    assert outcome.value.code == "TDS_194C"
    assert len(outcome.repairs) == 1
    assert len(attempts) == 1
    # The model must be told what was actually wrong, not merely asked again.
    assert "VALIDATION ERROR" in attempts[0]
    assert "code" in attempts[0]


def test_repair_exhaustion_raises_rather_than_guessing() -> None:
    bad = json.dumps({"code": "NOT_A_REAL_CODE"})

    with pytest.raises(SchemaValidationFailed) as exc:
        validate_or_repair(
            bad, Classification, reinvoke=lambda _: bad, original_prompt="orig", max_attempts=2
        )

    assert exc.value.attempts == 3


def test_exhausted_validation_becomes_abstention_not_a_guess(tmp_path: Path) -> None:
    """The load-bearing one. An unparseable response carries zero information; turning it
    into a reason code would put a fabricated number into a money pipeline."""
    garbage = json.dumps({"code": "INVENTED", "confidence": 2.0})
    cache = LLMCache(tmp_path)
    client = ScriptedClient(cache, [garbage, garbage, garbage])

    response = client.complete_detailed("prompt", schema=Classification, task="classify")

    assert response.abstained is True
    assert response.value.code == "NEEDS_HUMAN"
    assert response.value.confidence == 0.0


def test_overlong_prose_is_truncated_not_rejected() -> None:
    """The cap is enforced on the stored object, but never by paying for a repair.

    Rejecting an over-long rationale would trigger a second full generation — about 30
    seconds on this hardware — to be told the same thing more briefly. The output tokens
    are already spent by the time we can measure the string, so rejection buys nothing.
    These fields are explanatory and no policy rule reads them, so truncating is both
    cheaper and equally honest.
    """
    payload = json.dumps(
        {
            "code": "TDS_194C",
            "confidence": 0.9,
            "rationale": "x" * 400,
            "check": "y" * 300,
            "evidence_needed": [],
        }
    )
    result = parse_response(payload, Classification)

    assert len(result.rationale) == 200
    assert len(result.check) == 120
    assert result.rationale.endswith("...")


def test_percentage_confidence_is_normalised_not_rejected() -> None:
    """The single most expensive schema bug found in this build.

    The model reliably returns `"confidence": 100`, meaning certain. Grammar-constrained
    decoding enforces structure and type but *cannot* enforce a numeric range — `maximum:
    1` is not expressible in GBNF — so this fired a repair on literally every
    classification call, doubling the cost of the whole stage.
    """
    payload = json.dumps({"code": "TDS_194C", "confidence": 100, "rationale": "2.00% -> 194C"})
    assert parse_response(payload, Classification).confidence == 1.0

    payload = json.dumps({"code": "TDS_194C", "confidence": 85, "rationale": "x"})
    assert parse_response(payload, Classification).confidence == 0.85

    # Values already in range are untouched.
    payload = json.dumps({"code": "TDS_194C", "confidence": 0.72, "rationale": "x"})
    assert parse_response(payload, Classification).confidence == 0.72


def test_confidence_still_rejects_nonsense() -> None:
    """Tolerance is not the same as accepting anything."""
    payload = json.dumps({"code": "TDS_194C", "confidence": -1, "rationale": "x"})
    with pytest.raises(ValidationError):
        parse_response(payload, Classification)


def test_extra_fields_are_rejected() -> None:
    payload = json.dumps(
        {
            "code": "TDS_194C",
            "confidence": 0.9,
            "rationale": "ok",
            "evidence_needed": [],
            "invented_field": "surprise",
        }
    )
    with pytest.raises(ValidationError):
        parse_response(payload, Classification)


def test_advice_amounts_stay_verbatim_strings() -> None:
    """The model copies digits; Python converts to paise. Asking a 7B model to multiply
    by 100 in a money pipeline is asking for silent arithmetic errors."""
    payload = json.dumps(
        {"lines": [{"invoice_ref": "INV/2026/0042", "gross_amount": "1,00,000", "deduction_amount": "7,600", "stated_reason": "TDS 194C @2% + freight"}]}
    )
    parsed = parse_response(payload, ParsedAdvice)
    assert parsed.lines[0].gross_amount == "1,00,000"
    assert isinstance(parsed.lines[0].deduction_amount, str)


def test_repair_prompt_is_shorter_than_naive_resend() -> None:
    """Repair prompts are pure overhead on a CPU box; they must not resend everything."""
    original = "A" * 5000
    prompt = build_repair_prompt(original, "{bad}", "code: invalid", Classification.model_json_schema())
    assert len(prompt) < len(original)


# ======================================================================================
# Context sizing
# ======================================================================================


def test_context_window_grows_with_prompt() -> None:
    """Ollama truncates an overlong prompt from the front SILENTLY, producing a confident
    answer to a question it was never fully shown."""
    small = size_context_window(1_000, 300, 8192)
    large = size_context_window(80_000, 300, 8192)
    assert small == 8192, "short prompts should sit at the configured floor"
    assert large > small
    assert large % 1024 == 0


def test_context_window_is_deterministic_for_a_given_prompt() -> None:
    """num_ctx affects output, so it must be a pure function of the prompt or the cache
    would be recording one sample of several possible behaviours."""
    assert size_context_window(12_345, 300, 8192) == size_context_window(12_345, 300, 8192)


def test_context_window_is_capped() -> None:
    assert size_context_window(10_000_000, 400, 8192) <= 32_768
