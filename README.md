# Recoup

**An AI agent that recovers revenue lost to B2B short payments and deductions.**

Razorpay AI Buildathon 2026 · Track 03 — AI Revenue Recovery

A ₹1,00,000 invoice comes back as ₹92,400. Some of that gap is a statutory TDS deduction
the seller must never chase. Some is a freight claim the contract actually allows. Some is
a scheme rebate that expired last quarter. Some is the buyer simply paying short with no
explanation. Recoup isolates every rupee of the shortfall, works out which bucket
it belongs to, verifies the claim against source data, and picks a bounded intervention —
then reports exactly how much money it recovered against a baseline policy, and how much
harm it caused getting there.

### What is measured today

| | |
|---|---:|
| Verification exactness vs ground truth (oracle classifier) | **100%** of 308 deductions |
| Classification macro-F1, answered cases | **0.778** |
| Abstention rate | 15% |
| Payment match rate | 83.5% |
| Deduction value located by matching | 77.9% |
| Compliance violations (independent audit) | **0** |
| Messages actually sent | **0** — every message is `dry_run=True` |
| Live-money code paths | **0** — a non-`rzp_test_` key is refused at construction |

---

## Runs entirely on a laptop, with no API spend

Every model call in this project runs against a **local Ollama model**. There is no
hosted API in the measured path and no key required. The provider is chosen in one file
(`config/llm.yaml`) behind an abstraction, so moving to a hosted API later is a one-line
edit and no call-site changes.

### Path A — reproduce the results with no model at all

**This is the path to use if you just want to check the numbers.** The LLM cache is
committed to this repository, so the entire scoreboard reproduces with Ollama stopped,
uninstalled, or never present:

```bash
git clone <repo> && cd deduction-desk
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
```

```bash
python -m deduction_desk run --policy agent --days 45 --offline
```

`--offline` serves every call from `.llm_cache/` and raises `LLMCacheMiss` (naming the
prompt hash) rather t git push --set-upstream origin main git push --set-upstream origin mainhan reaching for a model. It is structurally incapable of quietly
doing something else. See [.llm_cache/README.md](.llm_cache/README.md) for why that
directory is committed rather than ignored.

### Path B — run the inference yourself

You need [Ollama](https://ollama.com/download). Then:

```bash
ollama serve
```

Pull the model named in `config/llm.yaml` (see [docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md)
for how it was chosen), then confirm the stack before committing to a long run:

```bash
python -m deduction_desk doctor
```

`doctor` checks the server is reachable, checks the model is actually pulled, runs one
schema-validated round-trip, measures tokens/sec **on your machine**, and turns that into
a per-task wall-clock estimate for the full batch. On a CPU-only laptop a full
400-invoice batch is an overnight job, so it is worth ten seconds to find out before
starting one.

```bash
python -m deduction_desk generate --seed 42 --n 400    # synthetic batch + ground truth
python -m deduction_desk match                          # parse advices, match payments, isolate deltas
python -m deduction_desk classify --offline             # classify + verify (cache only)
python -m deduction_desk run --policy agent --days 45   # the 45-day tick loop
python -m deduction_desk report --compare agent,b0,b1,b2,b3
python -m deduction_desk replay --case CASE-0173-0      # full decision trace
python -m deduction_desk export-web                     # freeze the run into web/src/data/
```

The front end is a separate Vite/React app:

```bash
cd web && npm install && npm run dev                    # localhost:5173
```

It reads the **static snapshot** written by `export-web`, not the database. That is
deliberate: a live reader holds an open handle on the SQLite file, which made `generate`
fail with a `PermissionError` quietly enough to look like non-determinism. It also means
the site builds and deploys with no Python at all — `npm run build` produces a static
bundle, and the numbers on it carry the batch content hash they were exported from.

**The batch is resumable.** Interrupt it with Ctrl-C and restart; every completed call is
served from cache and the run picks up where it stopped. Progress is reported as
`n/total, cached=x, live=y, elapsed, ETA`.

---

## Why local inference, and what it costs

The honest trade is accuracy for reproducibility and zero cost. A 7B model classifying
Hinglish remittance text is weaker than a frontier model, and
[docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md) reports the measured numbers rather
than asserting it is fine. What the local path buys is that **anybody can re-run this and
get the same numbers**, without a key, a budget, or trust in a screenshot.

Measured on the selection slice (40 stratified cases, `qwen2.5:7b-instruct`, CPU-only
i5-13500H):

| | |
|---|---:|
| macro-F1 on answered cases | **0.778** |
| macro-F1 counting abstention as a class | 0.665 |
| abstention rate | 15% |
| classification latency | 46.3 s/call |
| projected full batch | ~7h |

The abstention rate is a feature and is reported beside the accuracy, never folded into
it. An agent that answers 85% of cases well and hands the rest to a person beats one that
answers everything with silent errors, because a silent error here posts a demand letter
to a customer who did nothing wrong.

**The money layer does not depend on the model's accuracy.** Verification is deterministic,
and given a correct label it computes the recoverable rupees with **100% exact agreement**
against ground truth across all 308 deductions (`tests/test_phase3_verification.py`). So a
classification error costs a misrouted case, not a wrong number.

Three properties make that stick:

| Property | How |
|---|---|
| Same prompt → same bytes | `temperature: 0` and a fixed `seed` on every call, asserted by a test that makes two genuinely live calls against two empty caches |
| Same model → same scoreboard | Cache key is `sha256(provider, model, prompt, temperature, schema_hash)`; changing the model invalidates every entry by design |
| No silent truncation | `num_ctx` is sized per call from measured prompt length, and the provider's own `prompt_eval_count` is checked against it afterwards |

That last one is the subtle failure mode of local inference. Ollama's default context is
small, and a prompt that overflows it is truncated **from the front, silently** — the
model then answers a question it was never fully shown, fluently and wrongly. A bundled
remittance advice covering seven invoices is exactly the prompt that overflows. Every call
sets `num_ctx` explicitly and warns at 80% occupancy.

---

## Where the LLM is, and where it deliberately is not

The LLM parses, classifies, and drafts. It never decides money.

| Stage | LLM? | Why |
|---|---|---|
| Ingest / remittance parsing | **yes** | Unstructured Hinglish prose, broken PDF columns, merged Excel headers |
| Matching | no | Deterministic ladder; LLM only as a last-resort fallback with an explicit "none of these" option |
| Classification | **yes** | Schema-constrained, with confidence and a real abstention |
| Verification | **no** | This is where rupees are determined. Deterministic, auditable, reproducible |
| Policy / stopping rules | **no** | Plain Python and plain YAML. If you find yourself asking a model "should we chase this?", stop |
| Drafting | **yes** | Prose only; the policy engine fixes channel, recipient, timing and template class, and a validator polices the output |
| Counterparty simulation | **no** | Static templates with slot-filling. Outcomes are pre-committed by the data generator |
| Razorpay payment links | **no** | Optional, test-mode only, off by default — see below |

That last row is what makes the scoreboard defensible. Whether a buyer pays is decided by
a deterministic state machine reading ground truth that was written **before the agent
ran**. The agent cannot influence its own grade, and no model is anywhere near that
decision.

Amounts are a related case. The model copies amounts from source documents as **verbatim
strings**; Python converts them to integer paise. Asking a 7B model to multiply by 100
inside a money pipeline is asking for silent arithmetic errors, so it is never asked.

---

## Repository layout

```
config/          policy.yaml, generator.yaml, reason_codes.yaml, llm.yaml, templates/
data/fixtures/   form_26as, gstr7, credit_note_ledger, scheme_master, grn, contracts, payment_history
src/deduction_desk/
  money.py       integer-paise arithmetic; the single rate function both sides call
  generator/     synthetic world + pre-committed ground truth
  ingest/        bank statement + remittance parsing
  matching/      exact -> normalised -> fuzzy -> subset-sum ladder
  classify/      schema-constrained classifier
  verify/        deterministic adjudication against fixtures
  policy/        the decision engine, stopping rules, compliance
  actions/       bounded action set, draft validator, outbox
  simulate/      deterministic counterparty
  audit/         append-only decision log + replay
  llm/           provider abstraction, committed cache, repair loop
  eval/          baselines, metrics, scoreboard
.llm_cache/      committed; makes the offline path possible
docs/            ARCHITECTURE.md, MODEL_SELECTION.md, BROKE.md, DEMO_SCRIPT.md
```

## Tests

```bash
.venv/Scripts/python -m pytest
```

Tests requiring a live model are marked `live_llm` and skip cleanly when none is
reachable, so the suite is green on a machine that has never run inference.

```bash
.venv/Scripts/python -m pytest -m "not live_llm"
```
