"""Loaders for config/*.yaml.

Config over code (spec rule 9): thresholds, costs, ladders, windows and the taxonomy all
live in YAML so that a judge can change a number during the panel and watch the scoreboard
move. Nothing in this module holds a policy value of its own — it only reads, validates
and caches.

The loaders are cached by (path, mtime) so a run reads each file once, but a long-lived
process that edits `policy.yaml` on disk picks up the change on the next read.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
FIXTURES_DIR = DATA_DIR / "fixtures"
REPORTS_DIR = ROOT / "reports"
TEMPLATES_DIR = CONFIG_DIR / "templates"
LLM_CACHE_DIR = ROOT / ".llm_cache"

IST = "Asia/Kolkata"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return loaded


@functools.lru_cache(maxsize=32)
def _cached_yaml(path_str: str, mtime: float) -> dict[str, Any]:
    return _read_yaml(Path(path_str))


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file, cached on its modification time."""
    return _cached_yaml(str(path), path.stat().st_mtime)


# ----------------------------------------------------------------------------------
# policy.yaml
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Policy:
    """Typed accessor over policy.yaml. Deliberately thin — the YAML is the source."""

    raw: dict[str, Any]

    @property
    def economics(self) -> dict[str, Any]:
        return self.raw["economics"]

    @property
    def confidence(self) -> dict[str, Any]:
        return self.raw["confidence"]

    @property
    def compliance(self) -> dict[str, Any]:
        return self.raw["compliance"]

    @property
    def stopping(self) -> dict[str, Any]:
        return self.raw["stopping_rules"]

    @property
    def settlement(self) -> dict[str, Any]:
        return self.raw["settlement"]

    @property
    def verification(self) -> dict[str, Any]:
        return self.raw["verification"]

    @property
    def drafting(self) -> dict[str, Any]:
        return self.raw["drafting"]

    def contact_cost_paise(self, channel: str) -> int:
        return int(self.economics["contact_cost_paise"][channel])

    def threshold(self, name: str) -> int:
        """Fetch an economics threshold in paise, failing loudly if it is missing."""
        try:
            return int(self.economics[name])
        except KeyError as exc:
            raise KeyError(f"policy.yaml economics.{name} is not set") from exc


def load_policy(path: Path | None = None) -> Policy:
    return Policy(load_yaml(path or CONFIG_DIR / "policy.yaml"))


# ----------------------------------------------------------------------------------
# reason_codes.yaml
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class ReasonCode:
    code: str
    label: str
    family: str
    verifier: str
    default_valid: bool | None
    chaseable: bool
    default_action: str
    expected_rate_bp: int | None = None
    alt_rate_bp: tuple[int, ...] = ()
    note: str = ""

    def plausible_rates_bp(self) -> tuple[int, ...]:
        """Every rate this section is legitimately charged at."""
        if self.expected_rate_bp is None:
            return tuple(self.alt_rate_bp)
        return (self.expected_rate_bp, *self.alt_rate_bp)


@dataclass(frozen=True)
class Taxonomy:
    codes: dict[str, ReasonCode]

    def __contains__(self, code: str) -> bool:
        return code in self.codes

    def __getitem__(self, code: str) -> ReasonCode:
        return self.codes[code]

    @property
    def all_codes(self) -> list[str]:
        return sorted(self.codes)

    @property
    def predictable_codes(self) -> list[str]:
        """Codes the classifier may emit. Everything, including the abstain sentinel."""
        return self.all_codes

    def family(self, family: str) -> list[ReasonCode]:
        return [c for c in self.codes.values() if c.family == family]

    def is_tds_family(self, code: str) -> bool:
        return code in self.codes and self.codes[code].family in {"tds", "gst"}


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    raw = load_yaml(path or CONFIG_DIR / "reason_codes.yaml")
    codes: dict[str, ReasonCode] = {}
    for code, spec in raw["codes"].items():
        codes[code] = ReasonCode(
            code=code,
            label=spec["label"],
            family=spec["family"],
            verifier=spec["verifier"],
            default_valid=spec.get("default_valid"),
            chaseable=bool(spec.get("chaseable", False)),
            default_action=spec["default_action"],
            expected_rate_bp=spec.get("expected_rate_bp"),
            alt_rate_bp=tuple(spec.get("alt_rate_bp") or ()),
            note=spec.get("note", ""),
        )
    return Taxonomy(codes)


# ----------------------------------------------------------------------------------
# generator.yaml
# ----------------------------------------------------------------------------------
def load_generator_config(path: Path | None = None) -> dict[str, Any]:
    cfg = load_yaml(path or CONFIG_DIR / "generator.yaml")
    total = sum(cfg["mix"].values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"generator.yaml mix must sum to 1.0, got {total:.6f}")
    return cfg


def config_snapshot() -> dict[str, Any]:
    """Everything a judge might want to challenge, in one dict, written into every run."""
    return {
        "policy": load_policy().raw,
        "generator": load_generator_config(),
        "reason_codes": sorted(load_taxonomy().all_codes),
    }
