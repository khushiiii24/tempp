"""Provider abstraction and the Ollama adapter.

Call sites never name a provider or a model. They ask for a task ("classify") and a
schema, and this module resolves everything else from `config/llm.yaml`. Swapping the
whole project onto a hosted API is a one-line edit to that file.

The Ollama-specific care is concentrated in `OllamaClient._invoke`, and it is there
because these are the failure modes that produce *confidently wrong answers* rather than
errors:

* **`num_ctx` is set explicitly on every call.** Ollama's default context is small, and
  a prompt that overflows it is silently truncated from the front — the model then
  answers a question it was never fully shown, fluently and wrongly. A remittance advice
  with eight bundled invoices is exactly the prompt that overflows. We size the window
  from the measured prompt length, then check the provider's own `prompt_eval_count`
  afterwards and warn at 80% occupancy.
* **`keep_alive` holds the model resident.** Without it Ollama unloads after five minutes
  and a batch pays a 10-30 second cold load on most calls, which on this workload
  dominates the actual inference.
* **`temperature: 0` and a fixed `seed` on every call**, so the same prompt returns the
  same bytes and the committed cache is a faithful record rather than one sample.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..config import CONFIG_DIR, ROOT, load_yaml
from .cache import LLMCache, make_key, schema_hash, sha256_text
from .errors import (
    LLMError,
    ProviderNotImplemented,
    ProviderUnavailable,
    SchemaValidationFailed,
)
from .schema import ParseOutcome, abstention, validate_or_repair

log = logging.getLogger("deduction_desk.llm")

# Characters per token, used only to SIZE the context window before the call. The actual
# count comes back from the provider as `prompt_eval_count` and is what we check against.
# 3.5 is conservative for the romanised Hinglish and run-together numeric text in these
# advices, which tokenise worse than clean English prose.
CHARS_PER_TOKEN_ESTIMATE = 3.5
CTX_GRANULARITY = 1024
CTX_HEADROOM = 1.25
CTX_MAX = 32_768
CTX_WARN_OCCUPANCY = 0.80


@dataclass
class LLMResponse:
    """Everything a caller needs to act on the answer AND to log it for audit."""

    value: BaseModel | str
    task: str
    provider: str
    model: str
    key: str
    prompt_sha256: str
    cached: bool
    abstained: bool = False
    repairs: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_llm_call_record(self) -> dict[str, Any]:
        """The shape `DecisionLog.llm_calls` expects."""
        return {
            "task": self.task,
            "provider": self.provider,
            "model": self.model,
            "prompt_hash": self.prompt_sha256,
            "cache_key": self.key,
            "cached": self.cached,
            "abstained": self.abstained,
            "repairs": self.repairs,
            "tokens_per_sec": self.meta.get("tokens_per_sec"),
            "eval_count": self.meta.get("eval_count"),
            "prompt_eval_count": self.meta.get("prompt_eval_count"),
            "num_ctx": self.meta.get("num_ctx"),
        }


# ======================================================================================
# Configuration
# ======================================================================================


def load_llm_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "llm.yaml")


def estimate_prompt_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 1)


def size_context_window(prompt_chars: int, max_tokens: int, floor: int) -> int:
    """Choose `num_ctx` for one call.

    A pure function of (prompt length, output budget, configured floor), so the same
    prompt always gets the same window and the run stays reproducible.
    """
    needed = (prompt_chars / CHARS_PER_TOKEN_ESTIMATE + max_tokens) * CTX_HEADROOM
    rounded = int(((needed + CTX_GRANULARITY - 1) // CTX_GRANULARITY) * CTX_GRANULARITY)
    return max(int(floor), min(rounded, CTX_MAX))


# ======================================================================================
# Base
# ======================================================================================


class LLMClient(ABC):
    """One method, one contract: give me text and optionally a schema, get back a
    validated object or a string."""

    provider_name: str = "abstract"

    def __init__(self, cfg: dict[str, Any], *, cache: LLMCache, offline: bool = False) -> None:
        self.cfg = cfg
        self.cache = cache
        self.offline = offline

    # -- provider hook --------------------------------------------------------------
    @abstractmethod
    def _invoke(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> tuple[str, dict[str, Any]]:
        """Make one live call. Returns (raw_text, meta)."""

    @property
    @abstractmethod
    def model(self) -> str:
        ...

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Used by `doctor`. Must not raise; report status in the returned dict."""

    # -- public API -----------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 300,
        temperature: float = 0.0,
        task: str = "generic",
        system: str | None = None,
    ) -> BaseModel | str:
        """The signature every call site uses."""
        return self.complete_detailed(
            prompt,
            schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
            task=task,
            system=system,
        ).value

    def complete_detailed(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 300,
        temperature: float = 0.0,
        task: str = "generic",
        system: str | None = None,
    ) -> LLMResponse:
        """As `complete`, but also returns cache status, repairs and timing for the audit log."""
        json_schema = schema.model_json_schema() if schema is not None else None
        digest = schema_hash(json_schema)
        prompt_hash = sha256_text(prompt)
        key = make_key(
            provider=self.provider_name,
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            schema_digest=digest,
        )

        cached = self.cache.get(task, key)
        if cached is not None:
            return self._response_from_cache(cached, task, key, prompt_hash, schema)

        if self.offline:
            # Never silently fall through to a live call. The point of offline mode is
            # that it is incapable of reaching a model.
            self.cache.require(task, key, prompt_sha256=prompt_hash, model=self.model)
            raise AssertionError("unreachable")  # pragma: no cover

        return self._live(
            prompt,
            schema=schema,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            task=task,
            system=system,
            key=key,
            prompt_hash=prompt_hash,
        )

    # -- internals ------------------------------------------------------------------
    def _response_from_cache(
        self,
        record: dict[str, Any],
        task: str,
        key: str,
        prompt_hash: str,
        schema: type[BaseModel] | None,
    ) -> LLMResponse:
        if record.get("abstained"):
            value: BaseModel | str = abstention()
        elif schema is not None:
            value = schema.model_validate(record["parsed"])
        else:
            value = record.get("response_text", "")
        return LLMResponse(
            value=value,
            task=task,
            provider=self.provider_name,
            model=self.model,
            key=key,
            prompt_sha256=prompt_hash,
            cached=True,
            abstained=bool(record.get("abstained")),
            repairs=record.get("repairs", []),
            meta=record.get("meta", {}),
        )

    def _live(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None,
        json_schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float,
        task: str,
        system: str | None,
        key: str,
        prompt_hash: str,
    ) -> LLMResponse:
        started = time.perf_counter()
        raw, meta = self._invoke(
            prompt,
            schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )

        outcome = ParseOutcome(value=None)
        abstained = False

        if schema is not None:
            def reinvoke(repair_prompt: str) -> str:
                log.warning("llm repair: task=%s model=%s", task, self.model)
                repaired, _ = self._invoke(
                    repair_prompt,
                    schema=json_schema,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                )
                return repaired

            max_attempts = int(self.cfg.get("repair", {}).get("max_attempts", 2))
            try:
                outcome = validate_or_repair(
                    raw,
                    schema,
                    reinvoke=reinvoke,
                    original_prompt=prompt,
                    max_attempts=max_attempts,
                )
                value: BaseModel | str = outcome.value  # type: ignore[assignment]
            except SchemaValidationFailed as exc:
                # Exhausted. Abstain — never guess. A fabricated reason code in a money
                # pipeline is strictly worse than an honest hand-off to a human.
                log.warning(
                    "llm abstain after %d attempts: task=%s error=%s",
                    exc.attempts,
                    task,
                    exc.last_error,
                )
                value = abstention()
                abstained = True
                outcome = ParseOutcome(value=None, repairs=[], abstained=True)
        else:
            value = raw

        meta["wall_s"] = round(time.perf_counter() - started, 3)
        repairs = outcome.repairs_as_dicts() if outcome.repairs else []

        self.cache.put(
            task,
            key,
            {
                "key": key,
                "task": task,
                "provider": self.provider_name,
                "model": self.model,
                "temperature": temperature,
                "schema_name": schema.__name__ if schema else None,
                "schema_hash": schema_hash(json_schema),
                "prompt_sha256": prompt_hash,
                "prompt": prompt,
                "system": system,
                "response_text": raw,
                "parsed": value.model_dump(mode="json") if isinstance(value, BaseModel) else None,
                "abstained": abstained,
                "repairs": repairs,
                "meta": meta,
            },
        )

        return LLMResponse(
            value=value,
            task=task,
            provider=self.provider_name,
            model=self.model,
            key=key,
            prompt_sha256=prompt_hash,
            cached=False,
            abstained=abstained,
            repairs=repairs,
            meta=meta,
        )


# ======================================================================================
# Ollama
# ======================================================================================


class OllamaClient(LLMClient):
    provider_name = "ollama"

    def __init__(self, cfg: dict[str, Any], *, cache: LLMCache, offline: bool = False) -> None:
        super().__init__(cfg, cache=cache, offline=offline)
        self.settings = cfg.get("ollama", {})
        self.host = str(self.settings.get("host", "http://localhost:11434")).rstrip("/")
        self._model = str(self.settings["model"])
        self.keep_alive = str(self.settings.get("keep_alive", "30m"))
        self.num_ctx_floor = int(self.settings.get("num_ctx", 8192))
        self.seed = int(self.settings.get("seed", 42))
        self.timeout_s = int(self.settings.get("request_timeout_s", 900))

    @property
    def model(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    def _post(self, path: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise ProviderUnavailable(f"ollama HTTP {exc.code} on {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderUnavailable(
                f"cannot reach ollama at {self.host} ({exc.reason}). "
                f"Is `ollama serve` running?"
            ) from exc

    def _get(self, path: str, *, timeout: int = 10) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.host}{path}", timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise ProviderUnavailable(f"cannot reach ollama at {self.host}: {exc}") from exc

    # ------------------------------------------------------------------
    def _invoke(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> tuple[str, dict[str, Any]]:
        num_ctx = size_context_window(len(prompt), max_tokens, self.num_ctx_floor)

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": float(temperature),
                "seed": self.seed,
                "num_ctx": num_ctx,
                "num_predict": int(max_tokens),
            },
        }
        if system:
            payload["system"] = system
        if schema is not None:
            # Grammar-constrained decoding. Not a prompt instruction — the decoder
            # physically cannot emit a token that breaks the schema.
            payload["format"] = schema

        body = self._post("/api/generate", payload, timeout=self.timeout_s)

        prompt_tokens = int(body.get("prompt_eval_count") or 0)
        eval_count = int(body.get("eval_count") or 0)
        eval_ns = int(body.get("eval_duration") or 0)
        tps = round(eval_count / (eval_ns / 1e9), 2) if eval_ns > 0 else None

        # The real check, using the provider's own count rather than our estimate.
        if prompt_tokens and prompt_tokens > CTX_WARN_OCCUPANCY * num_ctx:
            log.warning(
                "prompt occupies %d/%d tokens (%.0f%% of num_ctx) — raise num_ctx or shorten "
                "the prompt; Ollama truncates from the front SILENTLY and the answer will "
                "look confident and be wrong",
                prompt_tokens,
                num_ctx,
                100 * prompt_tokens / num_ctx,
            )

        meta = {
            "num_ctx": num_ctx,
            "prompt_eval_count": prompt_tokens,
            "eval_count": eval_count,
            "tokens_per_sec": tps,
            "total_duration_s": round(int(body.get("total_duration") or 0) / 1e9, 3),
            "truncation_risk": bool(prompt_tokens > CTX_WARN_OCCUPANCY * num_ctx),
        }
        return body.get("response", ""), meta

    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        body = self._get("/api/tags")
        return sorted(m.get("name", "") for m in body.get("models", []))

    def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "provider": self.provider_name,
            "host": self.host,
            "model": self._model,
            "reachable": False,
            "model_present": False,
            "installed": [],
            "error": None,
        }
        try:
            installed = self.list_models()
        except ProviderUnavailable as exc:
            info["error"] = str(exc)
            return info

        info["reachable"] = True
        info["installed"] = installed
        # Ollama reports "name:tag"; a bare "name" in config means ":latest".
        wanted = self._model if ":" in self._model else f"{self._model}:latest"
        info["model_present"] = wanted in installed or self._model in installed
        return info


# ======================================================================================
# Hosted providers — seams, not yet exercised
# ======================================================================================


class OpenAICompatClient(LLMClient):
    """Seam for any OpenAI-compatible endpoint (vLLM, LM Studio, Together, ...).

    Left unimplemented on purpose. The abstraction is proven by the Ollama and Anthropic
    adapters; shipping a third untested code path would be pretending to a capability the
    build has never run.
    """

    provider_name = "openai_compat"

    @property
    def model(self) -> str:
        return str(self.cfg.get("openai_compat", {}).get("model") or "unset")

    def _invoke(self, prompt, *, schema, max_tokens, temperature, system):
        raise ProviderNotImplemented(
            "openai_compat is a seam only. Implement _invoke against /v1/chat/completions "
            "with response_format=json_schema, then set provider: openai_compat in "
            "config/llm.yaml."
        )

    def health(self) -> dict[str, Any]:
        return {"provider": self.provider_name, "reachable": False, "error": "not implemented"}


class AnthropicClient(LLMClient):
    """Hosted fallback. Schema adherence via a single forced tool call.

    Present so the project can move to a hosted API without touching a call site. Not
    exercised in the measured build — every number in the scoreboard comes from the local
    model — so it is marked accordingly in `doctor`.
    """

    provider_name = "anthropic"

    def __init__(self, cfg: dict[str, Any], *, cache: LLMCache, offline: bool = False) -> None:
        super().__init__(cfg, cache=cache, offline=offline)
        self.settings = cfg.get("anthropic", {})
        self._model = str(self.settings.get("model", "unset"))

    @property
    def model(self) -> str:
        return self._model

    def _client(self):
        import os

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderUnavailable("anthropic SDK not installed") from exc

        key = os.environ.get(str(self.settings.get("api_key_env", "ANTHROPIC_API_KEY")), "")
        if not key.startswith("sk-ant-"):
            raise ProviderUnavailable(
                "no usable Anthropic API key. Note that a Claude Code harness credential "
                "(aero_live_...) is not an API key and will not authenticate here."
            )
        return anthropic.Anthropic(api_key=key)

    def _invoke(self, prompt, *, schema, max_tokens, temperature, system):
        client = self._client()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if schema is not None:
            kwargs["tools"] = [
                {"name": "emit", "description": "Return the result.", "input_schema": schema}
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "emit"}

        msg = client.messages.create(**kwargs)
        meta = {
            "prompt_eval_count": msg.usage.input_tokens,
            "eval_count": msg.usage.output_tokens,
            "tokens_per_sec": None,
            "num_ctx": None,
        }
        if schema is not None:
            for block in msg.content:
                if getattr(block, "type", None) == "tool_use":
                    return json.dumps(block.input), meta
            raise LLMError("anthropic returned no tool_use block")
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return text, meta

    def health(self) -> dict[str, Any]:
        info = {
            "provider": self.provider_name,
            "model": self._model,
            "reachable": False,
            "error": None,
            "note": "seam only; not exercised in the measured build",
        }
        try:
            self._client()
            info["reachable"] = True
        except ProviderUnavailable as exc:
            info["error"] = str(exc)
        return info


PROVIDERS: dict[str, type[LLMClient]] = {
    "ollama": OllamaClient,
    "openai_compat": OpenAICompatClient,
    "anthropic": AnthropicClient,
}


def build_client(
    *,
    offline: bool = False,
    cfg: dict[str, Any] | None = None,
    cache_root: Path | None = None,
) -> LLMClient:
    """The only way the pipeline obtains a client. Reads config/llm.yaml and nothing else."""
    cfg = cfg or load_llm_config()
    provider = str(cfg.get("provider", "ollama"))
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}")
    root = cache_root or (ROOT / str(cfg.get("cache", {}).get("dir", ".llm_cache")))
    cache = LLMCache(root, offline=offline)
    return PROVIDERS[provider](cfg, cache=cache, offline=offline)


def max_tokens_for(task: str, cfg: dict[str, Any] | None = None) -> int:
    cfg = cfg or load_llm_config()
    return int(cfg.get("per_task_overrides", {}).get(task, {}).get("max_tokens", 300))
