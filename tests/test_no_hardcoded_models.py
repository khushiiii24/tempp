"""No model tag may appear anywhere except config/llm.yaml.

The point of the provider abstraction is that swapping the whole project onto a different
model — or onto a hosted API — is a one-line edit to one file. A model tag that has leaked
into a prompt module, a default argument or a test fixture silently defeats that: the
config says one thing, some call site does another, and the scoreboard is attributed to
the wrong model.

What this test deliberately does NOT forbid is an adapter class naming its own identity
(`OllamaClient.provider_name = "ollama"`). That is the adapter's name, not a
configuration choice — any registry needs it, and forbidding it would only push the same
string into a stringly-typed lookup elsewhere. What matters is that nothing outside
config/llm.yaml *decides* which provider or model to use, and that is what the
`build_client` seam enforces.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONFIG_FILE = ROOT / "config" / "llm.yaml"

# Model families and tag shapes. If a future model is added to config/llm.yaml, add its
# family here too — the test should keep pace with the candidate list.
MODEL_TAG_PATTERNS = [
    re.compile(r"\bqwen[\w.]*", re.IGNORECASE),
    re.compile(r"\bllama[\w.]*[:\-]\s*\d", re.IGNORECASE),
    re.compile(r"\bmistral[\w.\-]*", re.IGNORECASE),
    re.compile(r"\bmixtral[\w.\-]*", re.IGNORECASE),
    re.compile(r"\bgemma[\w.\-]*", re.IGNORECASE),
    re.compile(r"\bphi-?\d", re.IGNORECASE),
    re.compile(r"\bdeepseek[\w.\-]*", re.IGNORECASE),
    re.compile(r"\bclaude-[\w.\-]+", re.IGNORECASE),
    re.compile(r"\bgpt-[\w.\-]+", re.IGNORECASE),
    re.compile(r":\d{1,3}b\b", re.IGNORECASE),  # ':7b', ':12b', ':70b'
]

# Files allowed to mention a model, and why.
ALLOWED = {
    CONFIG_FILE,                                   # the single source of truth
    ROOT / "tests" / "test_no_hardcoded_models.py",  # this file, which must name patterns
    ROOT / "docs" / "MODEL_SELECTION.md",          # the benchmark record
    ROOT / "docs" / "BROKE.md",                    # the incident log
    ROOT / "README.md",                            # quickstart instructions
    ROOT / "tests" / "bench_models.py",            # reads candidates from config, names none
}


def _python_sources() -> list[Path]:
    return [
        p
        for p in SRC.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


@pytest.mark.parametrize("path", _python_sources(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_model_tag_in_source(path: Path) -> None:
    if path in ALLOWED:
        pytest.skip("explicitly allowed to name a model")

    text = path.read_text(encoding="utf-8")
    offenders: list[str] = []
    for pattern in MODEL_TAG_PATTERNS:
        offenders.extend(pattern.findall(text))

    assert not offenders, (
        f"{path.relative_to(ROOT)} names a model ({sorted(set(offenders))}). "
        f"Model choice belongs in config/llm.yaml and nowhere else."
    )


def test_config_names_exactly_one_active_model() -> None:
    """The active model must be resolvable from config alone, with no code default."""
    import yaml

    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    assert cfg["provider"] in {"ollama", "openai_compat", "anthropic"}
    assert cfg["ollama"]["model"], "config/llm.yaml must name the ollama model"


def test_client_has_no_default_model() -> None:
    """`build_client` must fail loudly if config omits the model rather than substituting
    a built-in default — a silent fallback would attribute results to the wrong model."""
    from deduction_desk.llm.cache import LLMCache
    from deduction_desk.llm.client import OllamaClient

    with pytest.raises(KeyError):
        OllamaClient({"ollama": {"host": "http://x"}}, cache=LLMCache(Path("/tmp/nope")))
