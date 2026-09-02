"""Failure modes of the LLM layer, as distinct types.

These are separate exceptions rather than a single `LLMError` because the pipeline treats
them completely differently. A cache miss in offline mode is a build error the operator
must fix. A schema validation failure that survives repair is a *routine* outcome that
becomes an abstention. Collapsing them would make the second look like a crash and the
first look like a bad answer.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base for everything this package raises."""


class LLMCacheMiss(LLMError):
    """Offline mode was asked for a call that is not in the committed cache.

    Carries the prompt hash so the operator can find exactly which call is missing and
    re-run that slice online. Never swallowed and never silently converted into a live
    call — the whole value of `--offline` is that it cannot reach the network.
    """

    def __init__(self, key: str, prompt_sha256: str, task: str, model: str) -> None:
        self.key = key
        self.prompt_sha256 = prompt_sha256
        self.task = task
        self.model = model
        super().__init__(
            f"offline cache miss: task={task} model={model} "
            f"key={key} prompt_sha256={prompt_sha256}\n"
            f"  The committed cache does not contain this call. Re-run the batch online "
            f"to populate it, or check that config/llm.yaml still names the model the "
            f"cache was built with (changing the model changes every key)."
        )


class ProviderUnavailable(LLMError):
    """The configured provider could not be reached or is not usable."""


class ProviderNotImplemented(LLMError):
    """A provider adapter exists as a seam but has not been exercised in this build."""


class SchemaValidationFailed(LLMError):
    """The model's output did not validate, and repair attempts were exhausted.

    The caller converts this into an abstention. It must NEVER be converted into a
    guessed classification — an unparseable answer carries no information, and inventing
    one would put fabricated numbers into a money pipeline.
    """

    def __init__(self, attempts: int, last_error: str, raw: str) -> None:
        self.attempts = attempts
        self.last_error = last_error
        self.raw = raw
        super().__init__(f"schema validation failed after {attempts} attempt(s): {last_error}")
