"""Disk cache for LLM calls. A committed build artifact, not a temp directory.

`.llm_cache/` is checked into git on purpose. Two things fall out of that:

* **Reproduction without inference.** Anyone can clone the repo and run
  `run --offline` to rebuild the entire scoreboard with no model installed, no GPU and
  no API key. The judges do not have to take the numbers on trust or spend eleven hours
  of CPU to check them.
* **The cache is the audit trail for everything the model said.** Each entry stores the
  full prompt, the raw response, the parsed object and any repair attempts. `replay` can
  therefore show not just which decision was made but the exact text the model was shown
  and the exact text it returned.

The key is `sha256(provider, model, prompt, temperature, schema_hash)`. Note what that
implies: **changing the model invalidates the entire cache**, by design. A scoreboard
produced by one model must never be silently attributed to another.

Layout is `.llm_cache/<task>/<key>.json` — one JSON file per key, grouped by task so the
directory stays browsable and per-task statistics are a directory listing away.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import LLMCacheMiss

CACHE_FORMAT_VERSION = 1


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def schema_hash(schema: dict[str, Any] | None) -> str:
    """Stable hash of a JSON Schema.

    Sorted keys, so a cosmetic reordering of Pydantic fields does not invalidate the
    cache, but a genuine change to the contract does.
    """
    if schema is None:
        return "noschema"
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return sha256_text(canonical)[:16]


def make_key(
    *,
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    schema_digest: str,
) -> str:
    """The cache key. Order and separator are fixed forever — changing them orphans the
    committed cache, which is a breaking change to the repo, not a refactor."""
    material = "\x1f".join(
        [
            f"v{CACHE_FORMAT_VERSION}",
            provider,
            model,
            f"{float(temperature):.4f}",
            schema_digest,
            prompt,
        ]
    )
    return sha256_text(material)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def hit(self) -> None:
        with self._lock:
            self.hits += 1

    def miss(self) -> None:
        with self._lock:
            self.misses += 1

    def write(self) -> None:
        with self._lock:
            self.writes += 1

    def as_dict(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}


class LLMCache:
    """Read-through disk cache. Thread-safe for the small worker pool."""

    def __init__(self, root: Path, *, offline: bool = False) -> None:
        self.root = Path(root)
        self.offline = offline
        self.stats = CacheStats()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def path_for(self, task: str, key: str) -> Path:
        safe_task = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in task) or "generic"
        return self.root / safe_task / f"{key}.json"

    def get(self, task: str, key: str) -> dict[str, Any] | None:
        path = self.path_for(task, key)
        if not path.exists():
            self.stats.miss()
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                record = json.load(fh)
        except (OSError, json.JSONDecodeError):
            # A corrupt entry is treated as absent rather than fatal. In offline mode the
            # subsequent miss still raises, so corruption can never masquerade as a hit.
            self.stats.miss()
            return None
        self.stats.hit()
        return record

    def require(self, task: str, key: str, *, prompt_sha256: str, model: str) -> dict[str, Any]:
        """Offline-mode fetch. Raises rather than falling through to a live call."""
        record = self.get(task, key)
        if record is None:
            raise LLMCacheMiss(key=key, prompt_sha256=prompt_sha256, task=task, model=model)
        return record

    def put(self, task: str, key: str, record: dict[str, Any]) -> None:
        """Atomically write an entry.

        Written to a temp file in the same directory and then replaced, so an interrupted
        run (Ctrl-C during an overnight batch is expected, not exceptional) can never
        leave a half-written JSON file that would poison the next resume.
        """
        path = self.path_for(task, key)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, ensure_ascii=False, indent=2, sort_keys=True)
                    fh.write("\n")
                os.replace(tmp_name, path)
            except BaseException:
                Path(tmp_name).unlink(missing_ok=True)
                raise
        self.stats.write()

    # ------------------------------------------------------------------
    def contains(self, task: str, key: str) -> bool:
        """Existence check that does not disturb the hit/miss counters.

        The batch runner uses this to report `cached=x live=y` and a truthful ETA before
        it starts work, which it cannot do if merely counting the plan mutates the stats.
        """
        return self.path_for(task, key).exists()

    def inventory(self) -> dict[str, int]:
        """Entry count per task. Used by `doctor` and the README's offline claim."""
        if not self.root.exists():
            return {}
        counts: dict[str, int] = {}
        for task_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            counts[task_dir.name] = len(list(task_dir.glob("*.json")))
        return counts
