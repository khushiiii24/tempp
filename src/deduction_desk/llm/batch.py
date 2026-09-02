"""Resumable batch runner for LLM calls.

Resumability is not a feature bolted on here — it falls out of the cache being
content-addressed and written atomically per item. Interrupt an overnight run with
Ctrl-C and you lose at most the two calls in flight; restart and every completed call is
a hit. That matters because on a CPU-only laptop a full batch is measured in hours, and
an all-or-nothing runner would make the project undemoable on the day something needs
changing.

Two deliberate choices:

* **Results are keyed by caller reference, never by completion order.** The pool finishes
  work out of order; if downstream consumed it in that order the scoreboard would shift
  between runs for no reason. Callers get a dict and iterate it sorted.
* **The plan is computed before any work starts**, by asking the cache which keys already
  exist. That is what makes `cached=x live=y` and the ETA truthful rather than a guess
  that improves as it goes.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .cache import make_key, schema_hash
from .client import LLMClient, LLMResponse


@dataclass
class LLMTask:
    """One unit of work. `ref` is the caller's own id — a deduction id, an advice id."""

    ref: str
    prompt: str
    task: str
    schema: type[BaseModel] | None = None
    max_tokens: int = 300
    temperature: float = 0.0
    system: str | None = None


@dataclass
class BatchPlan:
    total: int
    cached: int
    live: int
    cached_refs: set[str] = field(default_factory=set)

    @property
    def pct_cached(self) -> float:
        return 100.0 * self.cached / self.total if self.total else 100.0


def plan_batch(client: LLMClient, tasks: Sequence[LLMTask]) -> BatchPlan:
    """Work out what is already done, without touching the hit/miss counters."""
    cached_refs: set[str] = set()
    for t in tasks:
        json_schema = t.schema.model_json_schema() if t.schema is not None else None
        key = make_key(
            provider=client.provider_name,
            model=client.model,
            prompt=t.prompt,
            temperature=t.temperature,
            schema_digest=schema_hash(json_schema),
        )
        if client.cache.contains(t.task, key):
            cached_refs.add(t.ref)
    return BatchPlan(
        total=len(tasks),
        cached=len(cached_refs),
        live=len(tasks) - len(cached_refs),
        cached_refs=cached_refs,
    )


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class _ProgressReporter:
    """Prints 'n/total, cached=x, live=y, elapsed, ETA' to stderr.

    stderr, not stdout, so that piping a report to a file does not interleave progress
    chatter with the artefact.
    """

    def __init__(self, label: str, plan: BatchPlan, *, enabled: bool = True) -> None:
        self.label = label
        self.plan = plan
        self.enabled = enabled
        self.done = 0
        self.cached_seen = 0
        self.live_seen = 0
        self.live_seconds = 0.0
        self.started = time.perf_counter()
        self._lock = threading.Lock()
        self._last_emit = 0.0

    def record(self, response: LLMResponse) -> None:
        with self._lock:
            self.done += 1
            if response.cached:
                self.cached_seen += 1
            else:
                self.live_seen += 1
                self.live_seconds += float(response.meta.get("wall_s") or 0.0)
            self._maybe_emit()

    def _eta_seconds(self) -> float | None:
        if self.live_seen == 0:
            return None if self.plan.live else 0.0
        mean_live = self.live_seconds / self.live_seen
        remaining_live = max(0, self.plan.live - self.live_seen)
        # Parallelism is real but sublinear on CPU; the runner reports the honest
        # single-stream estimate rather than dividing by worker count and being wrong low.
        return remaining_live * mean_live

    def _maybe_emit(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_emit < 2.0 and self.done < self.plan.total:
            return
        self._last_emit = now
        if not self.enabled:
            return
        elapsed = now - self.started
        eta = self._eta_seconds()
        eta_text = _fmt_duration(eta) if eta is not None else "--"
        tail = ""
        if self.live_seen:
            tail = f", {self.live_seconds / self.live_seen:.1f}s/live call"
        sys.stderr.write(
            f"\r[{self.label}] {self.done}/{self.plan.total}  "
            f"cached={self.cached_seen} live={self.live_seen}  "
            f"elapsed={_fmt_duration(elapsed)} eta={eta_text}{tail}   "
        )
        sys.stderr.flush()

    def finish(self) -> dict[str, Any]:
        self._maybe_emit(force=True)
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()
        return {
            "label": self.label,
            "total": self.plan.total,
            "cached": self.cached_seen,
            "live": self.live_seen,
            "elapsed_s": round(time.perf_counter() - self.started, 2),
            "mean_live_s": round(self.live_seconds / self.live_seen, 2) if self.live_seen else None,
        }


def run_llm_batch(
    client: LLMClient,
    tasks: Sequence[LLMTask],
    *,
    max_workers: int | None = None,
    label: str = "llm",
    progress: bool = True,
    on_result: Callable[[str, LLMResponse], None] | None = None,
) -> tuple[dict[str, LLMResponse], dict[str, Any]]:
    """Run every task, returning {ref: LLMResponse} and a stats dict.

    Pool size defaults to `ollama.max_workers` from config. Two is the documented start
    because these are CPU-bound generations sharing one machine: past a small number of
    concurrent requests the cores are oversubscribed, per-call latency rises faster than
    throughput improves, and total wall clock gets *worse*. Measure before raising it.
    """
    if not tasks:
        return {}, {"label": label, "total": 0, "cached": 0, "live": 0, "elapsed_s": 0.0}

    if max_workers is None:
        max_workers = int(client.cfg.get("ollama", {}).get("max_workers", 2))
    max_workers = max(1, int(max_workers))

    plan = plan_batch(client, tasks)
    reporter = _ProgressReporter(label, plan, enabled=progress)

    if progress:
        sys.stderr.write(
            f"[{label}] {plan.total} calls: {plan.cached} cached ({plan.pct_cached:.0f}%), "
            f"{plan.live} to run, {max_workers} worker(s)\n"
        )
        sys.stderr.flush()

    results: dict[str, LLMResponse] = {}
    results_lock = threading.Lock()

    def work(t: LLMTask) -> tuple[str, LLMResponse]:
        response = client.complete_detailed(
            t.prompt,
            schema=t.schema,
            max_tokens=t.max_tokens,
            temperature=t.temperature,
            task=t.task,
            system=t.system,
        )
        return t.ref, response

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=label) as pool:
        futures = {pool.submit(work, t): t for t in tasks}
        try:
            for fut in as_completed(futures):
                ref, response = fut.result()
                with results_lock:
                    results[ref] = response
                reporter.record(response)
                if on_result is not None:
                    on_result(ref, response)
        except KeyboardInterrupt:  # pragma: no cover - operator action
            sys.stderr.write(
                f"\n[{label}] interrupted. {len(results)} call(s) completed and cached; "
                f"re-run to resume from here.\n"
            )
            for fut in futures:
                fut.cancel()
            raise

    stats = reporter.finish()
    stats["cache"] = client.cache.stats.as_dict()
    return results, stats


def estimate_batch_wall_clock(
    n_calls: int, seconds_per_call: float, max_workers: int = 2, speedup: float = 1.35
) -> float:
    """Rough wall-clock estimate for `doctor`.

    `speedup` is deliberately well under `max_workers`: on a CPU-only machine two
    concurrent generations do not run twice as fast, they contend for the same cores.
    1.35x for two workers matches what this class of hardware actually delivers, and an
    estimate that promises 2x would be a promise the machine cannot keep.
    """
    if n_calls <= 0:
        return 0.0
    effective = speedup if max_workers > 1 else 1.0
    return n_calls * seconds_per_call / effective
