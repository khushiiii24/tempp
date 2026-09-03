"""Determinism, stated precisely: what is reproducible, under what conditions, and why.

The naive version of this file asserted "same prompt twice → byte-identical output" and
failed. Investigating that produced a more useful and more honest picture, so the claim is
now decomposed into three separate assertions rather than one that is only sometimes true.

**What was measured.** Repeating one classification prompt against a *resident* model gives
byte-identical output every time (4/4, same SHA-256 of the raw response). Repeating it
against a *cold* model does not: the first call after a load differs from the second. In
every observed case the divergence was confined to the free-text `rationale` — `code`,
`confidence`, `check` and `evidence_needed` were identical.

**Why.** llama.cpp on CPU parallelises matrix multiplies across threads, and
floating-point addition is not associative, so reduction order changes the last bits.
While the model is still being memory-mapped from disk and the thread pool is spinning up,
that order is not stable; once resident it is. `temperature: 0` and a fixed `seed` remove
*sampling* randomness, which is a different thing from *numerical* reproducibility, and no
amount of seeding fixes the latter. The only true fix is single-threaded inference, which
would take this batch from hours to days.

**Why it does not undermine the project.** Reproducibility here rests on the committed
cache, not on the model. A call is made once, its exact response is stored in
`.llm_cache/`, and every subsequent run — including `--offline` on a machine with no model
at all — replays those bytes. Model-level determinism is a nice-to-have on top of that;
cache-level determinism is the guarantee, and it is absolute.

So this file asserts, separately:

1. Decision-relevant fields are stable even from cold — the claim that actually matters,
   since no policy rule reads the prose.
2. Raw output is byte-identical once the model is resident — the operating condition,
   given `keep_alive: 30m` and a batch of hundreds of sequential calls.
3. Context sizing is a pure function of the prompt, so it cannot drift between runs.

Cache-level byte determinism is covered in `test_llm_layer.py`, which needs no model.
"""

from __future__ import annotations

import pytest

from deduction_desk.llm.client import build_client, load_llm_config, max_tokens_for
from deduction_desk.llm.errors import LLMError
from deduction_desk.llm.schema import Classification

pytestmark = pytest.mark.live_llm

PROBE_PROMPT = (
    "A buyer paid Rs 9,60,000 against an invoice with a taxable value of Rs 9,60,000 "
    "and deducted Rs 19,200. The remittance advice says: 'less TDS as per our records'. "
    "The contract's expected TDS section is 194C at 2.00%. "
    "The deduction is 2.00% of the taxable value. Classify the deduction."
)

# Fields a policy rule may read. These are the ones that must not move.
DECISION_FIELDS = ("code", "confidence")


def _client(cache_dir, cfg=None):
    cfg = cfg or load_llm_config()
    client = build_client(cfg=cfg, cache_root=cache_dir)
    health = client.health()
    if not health.get("reachable"):
        pytest.skip(f"provider not reachable: {health.get('error')}")
    if not health.get("model_present"):
        pytest.skip(f"model {client.model} not pulled")
    return client, cfg


def _classify(client, cfg, prompt=PROBE_PROMPT):
    try:
        return client.complete_detailed(
            prompt,
            schema=Classification,
            max_tokens=max_tokens_for("classify", cfg),
            temperature=0.0,
            task="classify",
        )
    except LLMError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"provider error: {exc}")


def test_decision_relevant_fields_are_stable(tmp_path) -> None:
    """The claim that matters: the *answer* does not move, even across a cold start.

    Two separate empty caches, so both calls are genuinely live — running twice against
    one cache would pass trivially on a cache hit and prove nothing about the model.
    """
    client_a, cfg = _client(tmp_path / "a")
    client_b = build_client(cfg=cfg, cache_root=tmp_path / "b")

    first = _classify(client_a, cfg)
    second = _classify(client_b, cfg)

    assert first.cached is False and second.cached is False, "both calls must be live"

    for field in DECISION_FIELDS:
        assert getattr(first.value, field) == getattr(second.value, field), (
            f"{field} drifted between identical calls: "
            f"{getattr(first.value, field)!r} vs {getattr(second.value, field)!r}. "
            f"Prose drift is expected on CPU; a moving classification is not."
        )


def test_raw_output_is_byte_identical_once_resident(tmp_path) -> None:
    """Byte-level determinism, under the condition it actually holds.

    The warm-up call is the point of this test, not an accident: it is what separates
    "the model is non-deterministic" from "the first call after a 4.7 GB memory-map is
    non-deterministic". `keep_alive: 30m` means every call in a real batch after the first
    runs in this state.
    """
    client_warm, cfg = _client(tmp_path / "warm")
    _classify(client_warm, cfg, PROBE_PROMPT + " (warm-up)")  # settle the runner

    client_a = build_client(cfg=cfg, cache_root=tmp_path / "a")
    client_b = build_client(cfg=cfg, cache_root=tmp_path / "b")

    first = _classify(client_a, cfg)
    second = _classify(client_b, cfg)

    raw_a = client_a.cache.get("classify", first.key)["response_text"]
    raw_b = client_b.cache.get("classify", second.key)["response_text"]

    assert raw_a == raw_b, (
        "raw output differed between identical calls against a resident model. "
        "If this fails, check that temperature is 0 and seed is set in config/llm.yaml."
    )


def test_context_window_is_identical_across_identical_calls(tmp_path) -> None:
    """`num_ctx` influences output, so it must be a pure function of the prompt.

    If it drifted, the cache would be recording one sample of several possible
    behaviours rather than the behaviour.
    """
    client_a, cfg = _client(tmp_path / "a")
    client_b = build_client(cfg=cfg, cache_root=tmp_path / "b")

    a = _classify(client_a, cfg)
    b = _classify(client_b, cfg)

    assert a.meta["num_ctx"] == b.meta["num_ctx"]


def test_prompt_fits_the_context_window(tmp_path) -> None:
    """Ollama truncates an over-long prompt from the front, silently.

    The model then answers a question it was never fully shown, fluently and wrongly —
    which is far worse than an error, because nothing surfaces.
    """
    client, cfg = _client(tmp_path / "ctx")
    response = _classify(client, cfg)
    assert response.meta.get("truncation_risk") is False


def test_no_repair_needed_on_a_routine_classification(tmp_path) -> None:
    """A repair is a second full generation — ~30s here. It must not be routine.

    This caught a systematic bug: the model returns `"confidence": 100`, and grammar-
    constrained decoding cannot enforce a numeric range, so every single call was paying
    for a repair round trip. The schema now normalises instead.
    """
    client, cfg = _client(tmp_path / "repair")
    response = _classify(client, cfg)

    assert not response.repairs, (
        f"routine classification needed {len(response.repairs)} repair(s): "
        f"{response.repairs}. That doubles the cost of the classification stage."
    )
