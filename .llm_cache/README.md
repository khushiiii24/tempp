# `.llm_cache/` is committed to git on purpose

This directory is **not** gitignored. That is deliberate, and it is doing three jobs.

### 1. Anyone can reproduce the scoreboard without running a model

Every number in this project's report comes from a local model doing several hundred
inference calls on a CPU — roughly an overnight run on a laptop with no discrete GPU.
Nobody should have to repeat that to check the work. With this directory committed:

```bash
git clone <repo> && cd deduction-desk
python -m deduction_desk run --policy agent --days 45 --offline
python -m deduction_desk report --compare agent,b0,b1,b2,b3
```

reproduces the identical scoreboard with **Ollama stopped, uninstalled, or never present**.
`--offline` serves every call from this directory and raises `LLMCacheMiss` rather than
reaching for a model, so it is structurally incapable of quietly doing something else.

### 2. It is the audit trail for everything the model was shown and said

Each entry stores the full prompt, the raw response, the parsed object, any schema-repair
attempts, and the timing and token counts. `replay --case CASE-0173` can therefore show
not merely which decision was taken but the exact text the model saw and the exact text
it returned. A decision log that records only the conclusion is not an audit trail.

### 3. It pins the numbers to a specific model

The cache key is `sha256(provider, model, prompt, temperature, schema_hash)`. **Changing
the model in `config/llm.yaml` invalidates every entry**, by design. A scoreboard
produced by one model can never be silently re-attributed to another — you either have
the cache that produced it or you re-run and generate a new one.

---

## Layout

```
.llm_cache/<task>/<sha256>.json
```

One JSON file per key, grouped by task (`classify`, `parse`, `draft`, `doctor`) so the
directory stays browsable and per-task counts are a directory listing away.

Writes are atomic — a temp file in the same directory, then `os.replace`. Interrupting an
overnight batch with Ctrl-C cannot leave a half-written entry that would poison the
resume.

## Size

Entries are a few kilobytes each and a full batch is on the order of a thousand calls, so
this directory runs to single-digit megabytes. That is an acceptable price for a
repository whose central claim is that its numbers are checkable.

## When to clear it

Only when you intend to invalidate results: a changed prompt, a changed schema or a
changed model already produce different keys, so stale entries are inert rather than
wrong. Deleting the directory does not corrupt anything — it just means the next run has
to do the inference again.
