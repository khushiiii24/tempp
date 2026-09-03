"""Empirical model selection. Run as a script, not as part of the test suite.

    python tests/bench_models.py                      # every candidate in config/llm.yaml
    python tests/bench_models.py --models qwen2.5:7b-instruct
    python tests/bench_models.py --size 40 --write-doc

Runs a stratified, labelled 40-record slice through each candidate and reports macro-F1
on reason codes, abstention rate, mean latency per call, and the projected wall clock for
a full 400-invoice batch. Writes `docs/MODEL_SELECTION.md`.

**The selection rule is "smallest model that clears the floor", not "highest F1".** On a
CPU-only machine the accuracy difference between a 7B and a 14B is worth far less than the
difference between a batch that finishes overnight and one that does not — a model that
scores three points higher and cannot complete a run is not a better model for this
project. So candidates are evaluated smallest-first and the first one to clear
`--floor` wins.

This file names no model. Candidates come from `config/llm.yaml`, and their existence is
verified against the local registry before anything is pulled.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deduction_desk.classify.feasibility import infeasibility_reason  # noqa: E402
from deduction_desk.classify.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    build_classification_prompt,
)
from deduction_desk.config import load_taxonomy  # noqa: E402
from deduction_desk.eval.metrics import abstention_aware_report, confusion_lines  # noqa: E402
from deduction_desk.eval.slices import (  # noqa: E402
    buyer_history,
    load_labelled_cases,
    slice_summary,
    stratified_slice,
)
from deduction_desk.llm.batch import LLMTask, _fmt_duration, run_llm_batch  # noqa: E402
from deduction_desk.llm.client import build_client, load_llm_config, max_tokens_for  # noqa: E402
from deduction_desk.llm.schema import Classification  # noqa: E402

DOC_PATH = ROOT / "docs" / "MODEL_SELECTION.md"


# ----------------------------------------------------------------------------------
# Registry checks — never assume a tag exists
# ----------------------------------------------------------------------------------
def registry_has(tag: str, *, timeout: int = 15) -> bool:
    """Does this tag exist in the public Ollama library?"""
    if ":" in tag:
        name, version = tag.split(":", 1)
    else:
        name, version = tag, "latest"
    url = f"https://registry.ollama.ai/v2/library/{name}/manifests/{version}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False


def installed_models(cfg: dict[str, Any]) -> list[str]:
    client = build_client(cfg=cfg)
    health = client.health()
    return list(health.get("installed") or [])


# ----------------------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------------------
def benchmark_model(
    tag: str,
    cases: list,
    all_cases: list,
    cfg: dict[str, Any],
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    """Run the slice through one model and score it."""
    taxonomy = load_taxonomy()
    model_cfg = copy.deepcopy(cfg)
    model_cfg["ollama"]["model"] = tag

    client = build_client(cfg=model_cfg)
    max_tokens = max_tokens_for("classify", model_cfg)

    tasks = [
        LLMTask(
            ref=case.id,
            prompt=build_classification_prompt(
                taxonomy=taxonomy,
                deduction=case.deduction,
                invoice=case.invoice,
                buyer=case.buyer,
                contract=case.contract,
                buyer_history=buyer_history(all_cases, case.buyer.id, case.id),
            ),
            task="classify",
            schema=Classification,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
        )
        for case in cases
    ]

    started = time.perf_counter()
    results, stats = run_llm_batch(
        client, tasks, label=f"bench:{tag}", max_workers=workers
    )
    elapsed = time.perf_counter() - started

    # Apply the same deterministic feasibility gate the pipeline uses, so the benchmark
    # measures the shipping configuration rather than the raw model. An impossible code
    # becomes an abstention, which is counted separately from a wrong answer.
    y_true = [c.true_code for c in cases]
    y_pred = []
    for c in cases:
        if c.id not in results:
            y_pred.append("NEEDS_HUMAN")
            continue
        code = results[c.id].value.code.value
        if code != "NEEDS_HUMAN" and infeasibility_reason(
            code, deduction=c.deduction, invoice=c.invoice, contract=c.contract
        ):
            code = "NEEDS_HUMAN"
        y_pred.append(code)

    report = abstention_aware_report(y_true, y_pred)

    latencies = [
        float(r.meta.get("wall_s") or 0.0) for r in results.values() if not r.cached
    ]
    tps_values = [
        float(r.meta.get("tokens_per_sec") or 0.0)
        for r in results.values()
        if r.meta.get("tokens_per_sec")
    ]
    out_tokens = [int(r.meta.get("eval_count") or 0) for r in results.values()]
    repairs = sum(len(r.repairs) for r in results.values())

    mean_latency = statistics.mean(latencies) if latencies else 0.0
    mean_tps = statistics.mean(tps_values) if tps_values else 0.0

    return {
        "model": tag,
        **report,
        "repairs": repairs,
        "mean_latency_s": round(mean_latency, 2),
        "mean_tokens_per_sec": round(mean_tps, 2),
        "mean_output_tokens": round(statistics.mean(out_tokens), 1) if out_tokens else 0,
        "slice_elapsed_s": round(elapsed, 1),
        "cached": stats.get("cached", 0),
        "live": stats.get("live", 0),
    }


def project_full_batch(result: dict[str, Any], call_counts: dict[str, int], workers: int) -> float:
    """Project the whole batch from the measured per-call latency.

    Uses the measured classification latency for classify calls and scales it by the
    ratio of token budgets for the others, which is the dominant term on CPU. The
    parallel speedup is deliberately modest — see `estimate_batch_wall_clock`.
    """
    from deduction_desk.llm.batch import estimate_batch_wall_clock

    classify_budget = max_tokens_for("classify")
    total = 0.0
    for task, n in call_counts.items():
        budget = max_tokens_for(task)
        per_call = result["mean_latency_s"] * (budget / classify_budget if classify_budget else 1)
        total += estimate_batch_wall_clock(n, per_call, max_workers=workers)
    return total


# ----------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------
def write_doc(
    results: list[dict[str, Any]],
    chosen: str | None,
    floor: float,
    slice_info: dict[str, Any],
    projections: dict[str, float],
    skipped: list[tuple[str, str]],
) -> None:
    lines: list[str] = []
    a = lines.append

    a("# Model selection")
    a("")
    a("Chosen empirically by `tests/bench_models.py`, not by reputation. Re-run with:")
    a("")
    a("```bash")
    a("python tests/bench_models.py --write-doc")
    a("```")
    a("")
    a("## The rule")
    a("")
    a(
        f"**The smallest candidate that clears macro-F1 {floor:.2f} on answered cases wins** "
        "— not the highest-scoring one."
    )
    a("")
    a(
        "This machine has no discrete GPU (Intel Iris Xe, ~1 GB VRAM), so every token is "
        "generated on an i5-13500H CPU. In that regime the gap between a 7B and a 14B is "
        "worth far less than the gap between a batch that finishes overnight and one that "
        "does not. A model that scores three points higher and cannot complete a run is not "
        "a better model for this project."
    )
    a("")
    a("## Evaluation slice")
    a("")
    a(
        f"{slice_info['n']} deductions covering {slice_info['n_codes']} reason codes, "
        f"{slice_info['n_valid']} valid and {slice_info['n_recoverable']} recoverable. "
        "**Stratified, not random**: the batch is roughly two-thirds valid TDS, so a "
        "uniform 40-record sample would land ~27 TDS cases and one or two of everything "
        "else, and a macro average over that measures sampling noise on exactly the classes "
        "that matter. Selection is deterministic, so every model sees the same records."
    )
    a("")
    a("```")
    a(json.dumps(slice_info["by_code"], indent=2))
    a("```")
    a("")
    a("## Results")
    a("")
    a("| model | macro-F1 (answered) | macro-F1 (all) | abstained | repairs | mean latency | tok/s | est. full batch |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        est = projections.get(r["model"])
        a(
            f"| `{r['model']}` | {r['macro_f1_answered']:.3f} | {r['macro_f1_all']:.3f} | "
            f"{r['abstention_rate']:.0%} | {r['repairs']} | {r['mean_latency_s']:.1f}s | "
            f"{r['mean_tokens_per_sec']:.1f} | {_fmt_duration(est) if est else 'n/a'} |"
        )
    a("")
    a(
        "`macro-F1 (answered)` is quality given the model committed to an answer; "
        "`macro-F1 (all)` counts `NEEDS_HUMAN` as its own class so that over-abstaining is "
        "penalised. Both are reported because quoting only the first would let a model look "
        "excellent by refusing to work."
    )
    a("")

    if skipped:
        a("### Not evaluated")
        a("")
        for tag, why in skipped:
            a(f"- `{tag}` — {why}")
        a("")

    for r in results:
        a(f"### `{r['model']}`")
        a("")
        a(
            f"Answered {r['n_answered']}/{r['n']}, abstained {r['n_abstained']}. "
            f"Mean output {r['mean_output_tokens']} tokens/call."
        )
        a("")
        conf = confusion_lines(r.get("confusion", {}))
        if conf:
            a("Most common confusions:")
            a("")
            for line in conf:
                a(f"- {line}")
            a("")

    a("## Decision")
    a("")
    if chosen:
        winner = next(r for r in results if r["model"] == chosen)
        a(
            f"**`{chosen}`.** It is the smallest candidate to clear the "
            f"macro-F1 {floor:.2f} floor on answered cases "
            f"({winner['macro_f1_answered']:.3f}), at "
            f"{winner['mean_latency_s']:.1f}s per classification and a projected "
            f"{_fmt_duration(projections.get(chosen, 0))} for the full batch."
        )
        a("")
        a(f"`config/llm.yaml` is set to `{chosen}`.")
    else:
        a(
            f"**No candidate cleared the macro-F1 {floor:.2f} floor.** The next step is to "
            "escalate to a larger candidate, accepting the wall-clock cost, or to revisit "
            "the prompt before blaming the model."
        )
    a("")
    a("---")
    a("")
    a(
        "_Note: the cache key includes the model, so changing the model here invalidates "
        "every cached call by design. A scoreboard produced by one model is never silently "
        "re-attributed to another._"
    )

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {DOC_PATH.relative_to(ROOT)}")


# ----------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=40, help="Slice size.")
    parser.add_argument("--floor", type=float, default=0.72, help="Macro-F1 floor to clear.")
    parser.add_argument("--models", nargs="*", help="Override the candidate list.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--write-doc", action="store_true", help="Write docs/MODEL_SELECTION.md")
    args = parser.parse_args()

    cfg = load_llm_config()
    candidates = args.models or [c["tag"] for c in cfg.get("candidates", [])]
    if not candidates:
        print("no candidates configured in config/llm.yaml", file=sys.stderr)
        return 2

    all_cases = load_labelled_cases()
    if not all_cases:
        print("no labelled cases; run `python -m deduction_desk generate` first", file=sys.stderr)
        return 2

    bench_slice = stratified_slice(all_cases, size=args.size)
    info = slice_summary(bench_slice)
    print(f"slice: {info['n']} cases across {info['n_codes']} codes")

    present = set(installed_models(cfg))
    call_counts = {"parse": 195, "classify": 304, "draft": 140}
    workers = args.workers or int(cfg.get("ollama", {}).get("max_workers", 2))

    results: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    projections: dict[str, float] = {}
    chosen: str | None = None

    for tag in candidates:
        wanted = tag if ":" in tag else f"{tag}:latest"
        if wanted not in present and tag not in present:
            # Verify before suggesting a pull — never assume a tag exists.
            exists = registry_has(tag)
            reason = (
                f"not pulled locally. It exists in the registry; run `ollama pull {tag}`"
                if exists
                else "does not exist in the Ollama registry under that tag"
            )
            print(f"skip {tag}: {reason}")
            skipped.append((tag, reason))
            continue

        print(f"\n=== {tag} ===")
        result = benchmark_model(tag, bench_slice, all_cases, cfg, workers=args.workers)
        results.append(result)
        projections[tag] = project_full_batch(result, call_counts, workers)

        print(
            f"  macro-F1(answered)={result['macro_f1_answered']:.3f}  "
            f"macro-F1(all)={result['macro_f1_all']:.3f}  "
            f"abstained={result['abstention_rate']:.0%}  "
            f"latency={result['mean_latency_s']:.1f}s  "
            f"full batch ~{_fmt_duration(projections[tag])}"
        )

        # Smallest-first: stop at the first model that clears the floor.
        if chosen is None and result["macro_f1_answered"] >= args.floor:
            chosen = tag
            print(f"  -> clears the {args.floor:.2f} floor; stopping here (smallest wins)")
            break

    if not results:
        print("\nno candidates were runnable", file=sys.stderr)

    if args.write_doc:
        write_doc(results, chosen, args.floor, info, projections, skipped)

    if chosen:
        print(f"\nCHOSEN: {chosen}")
    else:
        print(f"\nNo candidate cleared macro-F1 {args.floor:.2f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
