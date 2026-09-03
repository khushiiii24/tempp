"""The reproducibility guarantee, asserted end-to-end on a real batch of LLM calls.

This is the claim the README makes to anyone who wants to check the numbers without
spending an evening of CPU: clone the repo, run `--offline`, get the identical result with
no model installed. It is also the claim that makes the project's headline figures
checkable rather than merely asserted, so it gets tested rather than described.

Three separate properties, because they fail in different ways:

1. **Warm re-run is free and identical.** A second pass over the same work touches the
   provider zero times and produces the same classifications.
2. **Interrupting loses at most the calls in flight.** Cache writes are atomic and
   per-item, so an overnight batch killed at any point resumes without redoing completed
   work. This is what makes a multi-hour run practical to iterate on.
3. **Offline cannot silently fall through.** A missing entry raises `LLMCacheMiss` naming
   the prompt hash, rather than quietly reaching for a model — otherwise "offline" would
   be a preference rather than a guarantee.

These run against the committed cache and need no model, so they stay green on any
machine.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from deduction_desk.classify.prompts import SYSTEM_PROMPT, build_classification_prompt
from deduction_desk.config import ROOT, load_taxonomy
from deduction_desk.eval.slices import (
    buyer_history,
    load_labelled_cases,
    stratified_slice,
)
from deduction_desk.llm.batch import LLMTask, plan_batch, run_llm_batch
from deduction_desk.llm.cache import LLMCache
from deduction_desk.llm.client import build_client, load_llm_config, max_tokens_for
from deduction_desk.llm.errors import LLMCacheMiss
from deduction_desk.llm.schema import Classification

CACHE_DIR = ROOT / ".llm_cache"
SLICE_SIZE = 40


def _tasks(limit: int | None = None) -> list[LLMTask]:
    taxonomy = load_taxonomy()
    cases = load_labelled_cases()
    if not cases:
        pytest.skip("no generated batch; run `python -m deduction_desk generate` first")

    chosen = stratified_slice(cases, size=SLICE_SIZE)
    if limit:
        chosen = chosen[:limit]

    cfg = load_llm_config()
    return [
        LLMTask(
            ref=c.id,
            prompt=build_classification_prompt(
                taxonomy=taxonomy,
                deduction=c.deduction,
                invoice=c.invoice,
                buyer=c.buyer,
                contract=c.contract,
                buyer_history=buyer_history(cases, c.buyer.id, c.id),
            ),
            task="classify",
            schema=Classification,
            max_tokens=max_tokens_for("classify", cfg),
            system=SYSTEM_PROMPT,
        )
        for c in chosen
    ]


def _require_warm_cache(tasks: list[LLMTask]):
    """Skip unless the committed cache actually covers this slice."""
    client = build_client(offline=True)
    plan = plan_batch(client, tasks)
    if plan.cached < len(tasks):
        pytest.skip(
            f"cache covers only {plan.cached}/{len(tasks)} calls; "
            f"run `python tests/bench_models.py` to populate it"
        )
    return client


def test_offline_run_needs_no_model(tmp_path: Path) -> None:
    """The README's central promise: clone, run offline, get the answers."""
    tasks = _tasks()
    client = _require_warm_cache(tasks)

    results, stats = run_llm_batch(client, tasks, progress=False)

    assert len(results) == len(tasks)
    assert stats["live"] == 0, "offline mode made a live call"
    assert all(r.cached for r in results.values())
    # And the answers are real classifications, not placeholders.
    assert all(isinstance(r.value, Classification) for r in results.values())


def test_offline_run_is_byte_identical_across_repeats() -> None:
    """Same cache, same answers — the property the scoreboard rests on."""
    tasks = _tasks()
    client = _require_warm_cache(tasks)

    first, _ = run_llm_batch(client, tasks, progress=False)
    second, _ = run_llm_batch(client, tasks, progress=False)

    for ref in sorted(first):
        assert first[ref].value.model_dump() == second[ref].value.model_dump()


def test_offline_miss_raises_and_names_the_prompt_hash(tmp_path: Path) -> None:
    """Offline must be incapable of quietly doing something else.

    A cache miss is an operator error to be fixed, not a reason to reach for a model, so
    it raises and names exactly which call is missing.
    """
    tasks = _tasks(limit=1)
    _require_warm_cache(tasks)

    # An empty cache directory: the entry cannot possibly be there.
    empty = LLMCache(tmp_path / "empty", offline=True)
    client = build_client(offline=True, cache_root=tmp_path / "empty")
    assert empty.inventory() == {}

    with pytest.raises(LLMCacheMiss) as exc:
        client.complete_detailed(
            tasks[0].prompt, schema=Classification, task="classify", max_tokens=300
        )

    assert exc.value.prompt_sha256 in str(exc.value)
    assert exc.value.model in str(exc.value)


def test_interrupted_batch_resumes_without_redoing_work(tmp_path: Path) -> None:
    """Simulates Ctrl-C partway through an overnight run.

    Copies part of the committed cache into a fresh directory, then plans the full batch
    against it. The completed calls must be recognised as done — that is what makes a
    five-hour run something you can interrupt and iterate on rather than a single
    all-or-nothing shot.
    """
    tasks = _tasks()
    _require_warm_cache(tasks)

    partial = tmp_path / "partial" / "classify"
    partial.mkdir(parents=True)

    source = sorted((CACHE_DIR / "classify").glob("*.json"))
    half = len(source) // 2
    for path in source[:half]:
        shutil.copy(path, partial / path.name)

    client = build_client(offline=True, cache_root=tmp_path / "partial")
    plan = plan_batch(client, tasks)

    assert plan.cached > 0, "resume recognised none of the completed work"
    assert plan.live == plan.total - plan.cached
    assert plan.cached + plan.live == len(tasks)


def test_cache_entries_are_atomic_and_complete() -> None:
    """An interrupted write must never leave a half-written entry that poisons the resume."""
    classify_dir = CACHE_DIR / "classify"
    if not classify_dir.exists():
        pytest.skip("no committed cache yet")

    assert not list(classify_dir.glob("*.tmp")), "temp files left behind by an interrupted write"

    for path in sorted(classify_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for required in ("prompt", "response_text", "model", "schema_hash", "meta"):
            assert required in record, f"{path.name} missing {required}"


def test_cache_is_pinned_to_the_configured_model() -> None:
    """A scoreboard produced by one model must never be re-attributed to another."""
    classify_dir = CACHE_DIR / "classify"
    if not classify_dir.exists():
        pytest.skip("no committed cache yet")

    configured = load_llm_config()["ollama"]["model"]
    models = {
        json.loads(p.read_text(encoding="utf-8"))["model"]
        for p in classify_dir.glob("*.json")
    }

    assert models <= {configured}, (
        f"cache contains entries from {models - {configured}}, but config names "
        f"{configured}. Changing the model invalidates the cache by design; these entries "
        f"are orphaned and should be removed."
    )
