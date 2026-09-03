"""Freeze the run into static JSON for the web front end.

The dashboard used to hold an open handle on the SQLite file and re-query it on every
rerun. That coupling cost real time — a running dashboard made `generate` fail with a
`PermissionError` because the database was locked, and the failure was silent enough to
look like non-determinism (see BROKE entry 12).

So the front end reads a **snapshot**, not the database. This module writes that snapshot.
The claim the site makes — "every number here was measured" — survives because the
snapshot is produced by this file from the same tables the grader reads, and it carries
the database content hash and the run id it came from. If the numbers on the page and the
numbers in `reports/scoreboard.md` ever disagree, the meta block says which run each came
from.

Ground truth is exported, but **fenced**: every truth-derived field on a case sits under a
single `truth` key, and the front end keeps it behind an explicit toggle. It is the answer
key; putting it beside the agent's own output by default would blur the line the whole
project rests on.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from ..config import (
    CONFIG_DIR,
    ROOT,
    load_generator_config,
    load_policy,
    load_taxonomy,
)
from ..schemas import (
    Allocation,
    Buyer,
    Case,
    ContactLog,
    Contract,
    DecisionLog,
    Deduction,
    DeductionTruth,
    Exception_,
    HumanApproval,
    Invoice,
    Outbox,
    PaymentEvent,
    RemittanceAdvice,
    Run,
)

WEB_DATA_DIR = ROOT / "web" / "src" / "data"

# How many decision-log rows to keep per case. Traces are the largest thing in the
# snapshot; a case that ran the full ladder produces well under this, so the cap only
# ever bites on pathological rows.
MAX_TRACE_ROWS = 40


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return path


# ======================================================================================
# BROKE.md
# ======================================================================================
_ENTRY_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2}) — Entry (\d+): (.+)$", re.MULTILINE)
# `Result` and `Why it took so long` are not decoration — two entries use them instead of,
# or alongside, `Fix`, and without them here those entries render on the site with an empty
# panel where the resolution should be. The log is written by hand as prose; the parser
# accommodates the prose rather than the prose being bent to fit the parser.
_FIELD_RE = re.compile(
    r"\*\*(Symptom|Wrong hypothesis|What it actually was|Fix|Result|"
    r"Why it took so long)[.:]\*\*"
)


def _plain(md: str) -> str:
    """Markdown to something a <p> can hold without a renderer.

    Fenced code blocks are dropped rather than flattened. A four-line snippet collapsed
    onto one line reads as noise in a paragraph, and the prose around it always says what
    the code did — the log is written to be read, not to be executed.
    """
    text = re.sub(r"```.*?```", "", md, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def parse_broke(path: Path) -> list[dict[str, Any]]:
    """The failure log, structured. Each entry is symptom → wrong guess → cause → fix."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    marks = list(_ENTRY_RE.finditer(raw))
    entries: list[dict[str, Any]] = []
    for i, m in enumerate(marks):
        body = raw[m.end() : marks[i + 1].start() if i + 1 < len(marks) else len(raw)]
        fields: dict[str, str] = {}
        parts = _FIELD_RE.split(body)
        for j in range(1, len(parts) - 1, 2):
            fields[parts[j]] = _plain(parts[j + 1].split("---")[0])
        entries.append(
            {
                "date": m.group(1),
                "n": int(m.group(2)),
                "title": m.group(3).strip(),
                "symptom": fields.get("Symptom", ""),
                "wrong_hypothesis": fields.get("Wrong hypothesis", ""),
                "cause": " ".join(
                    p
                    for p in (
                        fields.get("What it actually was", ""),
                        fields.get("Why it took so long", ""),
                    )
                    if p
                ),
                "fix": fields.get("Fix", ""),
                "result": fields.get("Result", ""),
            }
        )
    return entries


# ======================================================================================
# Snapshot
# ======================================================================================
def _sample_advices(
    advices: list[RemittanceAdvice], deductions: dict[str, Deduction]
) -> list[dict[str, Any]]:
    """One representative advice per format, chosen deterministically.

    Two constraints. It must be an advice that *explains a deduction* — an advice for a
    fully-paid invoice is three lines of nothing and shows none of what makes this hard.
    And it must be short, because a 4,000-character spreadsheet dump is unreadable on a web
    page while carrying no more of the register the buyer writes in than a short one does.
    """
    explaining = {
        d.payment_event_id
        for d in deductions.values()
        if d.claimed_reason_text and d.claimed_reason_text.strip()
    }
    by_format: dict[str, list[RemittanceAdvice]] = defaultdict(list)
    for a in advices:
        if a.raw_text and a.raw_text.strip():
            by_format[a.format].append(a)

    out = []
    for fmt in sorted(by_format):
        pool = [a for a in by_format[fmt] if a.links_to_payment in explaining] or by_format[fmt]
        pick = sorted(pool, key=lambda a: (len(a.raw_text), a.id))
        chosen = next((a for a in pick if len(a.raw_text) > 200), pick[0])
        out.append(
            {
                "id": chosen.id,
                "format": fmt,
                "received_at": chosen.received_at,
                "raw_text": chosen.raw_text[:900],
                "truncated": len(chosen.raw_text) > 900,
            }
        )
    return out


def _matching_stats(
    *,
    payments: int,
    allocated_payments: int,
    by_method: dict[str, int],
    exceptions_by_kind: dict[str, int],
    exceptions_total: int,
) -> dict[str, Any]:
    """Matcher figures, preferring the matcher's own report over a re-derivation.

    `reports/matching/matching.json` is written by `match` itself. The allocation table
    is a lossy view of what the matcher did — a payment resolved to a buyer but blocked
    from allocating (over-allocation, incomplete bundle) is matched and has no allocation
    row — so deriving the match rate from allocations reports it four payments low. The
    derived numbers stay as a fallback for a snapshot taken before `match` was ever run.
    """
    stats: dict[str, Any] = {
        "payments": payments,
        "matched": allocated_payments,
        "match_rate": round(allocated_payments / payments, 4) if payments else 0.0,
        "source": "derived_from_allocations",
        "by_method": [
            {"method": k, "count": v} for k, v in sorted(by_method.items(), key=lambda r: -r[1])
        ],
        "exceptions": sorted(
            ({"kind": k, "count": v} for k, v in exceptions_by_kind.items()),
            key=lambda r: -r["count"],
        ),
        "exceptions_total": exceptions_total,
    }

    report_path = ROOT / "reports" / "matching" / "matching.json"
    if not report_path.exists():
        return stats
    try:
        authoritative = json.loads(report_path.read_text("utf-8"))
    except json.JSONDecodeError:
        return stats

    m = authoritative.get("matching", {})
    d = authoritative.get("deltas", {})
    e = authoritative.get("exceptions", {})
    stats.update(
        {
            "payments": m.get("payments", stats["payments"]),
            "matched": m.get("matched", stats["matched"]),
            "match_rate": m.get("match_rate", stats["match_rate"]),
            "source": "reports/matching/matching.json",
            "by_method": [
                {"method": k, "count": v}
                for k, v in sorted(m.get("by_method", {}).items(), key=lambda r: -r[1])
            ]
            or stats["by_method"],
            "exceptions": [
                {"kind": k, "count": v}
                for k, v in sorted(e.get("by_kind", {}).items(), key=lambda r: -r[1])
            ]
            or stats["exceptions"],
            "exceptions_total": e.get("total", stats["exceptions_total"]),
            "invoices_with_allocations": d.get("invoices_with_allocations"),
            "fully_settled": d.get("fully_settled"),
            "with_shortfall": d.get("with_shortfall"),
            "awaiting_settlement": d.get("awaiting_settlement"),
        }
    )
    return stats


def _day_index(iso: str | None, start: date) -> int | None:
    if not iso:
        return None
    try:
        return (date.fromisoformat(iso[:10]) - start).days
    except ValueError:
        return None


def build_snapshot(session: Session, *, run_id: str | None = None) -> dict[str, Any]:
    """Read every table the site displays, once, and shape it for the browser."""
    # Resolve the run first — the outbox, the approvals and the decision log are all scoped
    # by it. The decision log is append-only by design and accumulates every execution ever
    # made (338,000 rows across 67 runs at the time of writing), so an unscoped export is
    # both twelve interleaved histories per case and a 14 MB file. Falling back to the most
    # recent agent run keeps `--no-rerun` meaningful rather than silently exporting the
    # whole build history.
    #
    # Insertion order, not `started_at`: every run shares the same simulated start date, so
    # sorting by it picks an arbitrary one of sixty-seven.
    runs = list(session.exec(select(Run)).all())
    if run_id is None:
        agent_runs = [r.id for r in runs if r.policy_name == "agent"]
        run_id = agent_runs[-1] if agent_runs else None

    buyers = {b.id: b for b in session.exec(select(Buyer)).all()}
    contracts = {c.buyer_id: c for c in session.exec(select(Contract)).all()}
    invoices = {i.id: i for i in session.exec(select(Invoice)).all()}
    # Genuine bank credits only. The agent creates a PaymentEvent when a chase succeeds;
    # counting those as inbound credits would inflate the funnel's top with money the
    # agent itself produced at the bottom.
    payments = {
        p.id: p
        for p in session.exec(select(PaymentEvent)).all()
        if not p.recovery_for_case_id
    }
    all_payments = {p.id: p for p in session.exec(select(PaymentEvent)).all()}
    allocations = list(session.exec(select(Allocation)).all())
    advices = list(session.exec(select(RemittanceAdvice)).all())
    deductions = {d.id: d for d in session.exec(select(Deduction)).all()}
    truths = {t.deduction_id: t for t in session.exec(select(DeductionTruth)).all()}
    cases = list(session.exec(select(Case)).all())

    # Does the case table actually belong to the run we are labelling?
    #
    # `Case` and `ContactLog` are global, not run-scoped — each execution clears and
    # rewrites them. `report` runs the agent first and b3 last, so exporting straight
    # afterwards without re-running produces a snapshot with the agent's decision log and
    # **b3's outcomes**, which renders perfectly and is wrong. It got as far as the page
    # once, quoting 13 recoveries where the agent made 18.
    #
    # Every run stores its own summary, so the two can be compared. Mismatch is a hard
    # failure: a snapshot that silently mixes two policies is worse than no snapshot.
    run_row = next((r for r in runs if r.id == run_id), None)
    if run_row is not None and run_row.stats:
        expected = int(run_row.stats.get("recovered_paise", -1))
        actual = sum(int(c.recovered_paise) for c in cases)
        if expected >= 0 and expected != actual:
            raise RuntimeError(
                f"Case table does not belong to run {run_id!r}: it recovered "
                f"{actual} paise, that run recorded {expected}. Another policy has "
                f"overwritten the case table since — re-run the agent before exporting "
                f"(`export-web` does this by default; `--no-rerun` does not)."
            )
    contacts = list(session.exec(select(ContactLog)).all())
    exceptions = list(session.exec(select(Exception_)).all())
    # Scoped by run. Outbox, approvals and inbound events accumulate across every
    # execution ever made — the unscoped counts here read 22,150 queued messages against
    # 147 contacts, which is not a metric, it is a history of the build.
    outbox = [
        o for o in session.exec(select(Outbox)).all() if run_id is None or o.run_id == run_id
    ]
    approvals = [
        a
        for a in session.exec(select(HumanApproval)).all()
        if run_id is None or a.run_id == run_id
    ]

    taxonomy = load_taxonomy()
    policy = load_policy()

    advice_by_payment = {a.links_to_payment: a for a in advices if a.links_to_payment}
    contacts_by_case: dict[str, list[ContactLog]] = defaultdict(list)
    for c in contacts:
        contacts_by_case[c.case_id].append(c)
    for rows in contacts_by_case.values():
        rows.sort(key=lambda c: c.ts)

    # ---- traces ---------------------------------------------------------------------
    log_q = select(DecisionLog)
    if run_id:
        log_q = log_q.where(DecisionLog.run_id == run_id)
    trace_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in session.exec(log_q).all():
        if not row.case_id:
            continue
        trace_by_case[row.case_id].append(
            {
                "seq": row.seq,
                "sim_date": row.sim_date,
                "stage": row.stage,
                "rules": list(row.policy_rules_fired or []),
                "decision": row.decision or {},
                "action": row.action_taken or {},
                "outcome": row.outcome or {},
            }
        )
    for rows in trace_by_case.values():
        rows.sort(key=lambda r: r["seq"])
        del rows[MAX_TRACE_ROWS:]

    # ---- cases ----------------------------------------------------------------------
    sim_dates = sorted({c.opened_at[:10] for c in cases if c.opened_at})
    start = date.fromisoformat(sim_dates[0]) if sim_dates else date.today()

    case_rows: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda c: c.id):
        ded = deductions.get(case.deduction_id)
        if ded is None:
            continue
        inv = invoices.get(ded.invoice_id)
        buyer = buyers.get(case.buyer_id)
        pay = all_payments.get(ded.payment_event_id)
        advice = advice_by_payment.get(ded.payment_event_id)
        truth = truths.get(ded.id)
        contract = contracts.get(case.buyer_id)
        code = taxonomy.codes.get(ded.predicted_code)

        case_rows.append(
            {
                "id": case.id,
                "deduction_id": ded.id,
                "state": case.state,
                "stop_reason": case.stop_reason,
                "opened_at": case.opened_at,
                "closed_at": case.closed_at,
                "contacts_used": case.contacts_used,
                "current_role": case.current_role,
                "cost_paise": case.cost_incurred_paise,
                "recovered_paise": case.recovered_paise,
                "written_off_paise": case.written_off_paise,
                "credit_note_paise": case.credit_note_paise,
                "awaiting_human": case.awaiting_human,
                "human_reason": case.human_reason,
                "buyer": {
                    "id": buyer.id if buyer else "",
                    "name": buyer.name if buyer else "",
                    "segment": buyer.segment if buyer else "",
                    "behaviour": buyer.payment_behaviour_tag if buyer else "",
                    "relationship_value_paise": buyer.relationship_value_paise if buyer else 0,
                    "consent_whatsapp": bool(buyer.consent_whatsapp) if buyer else False,
                    "dnd": bool(buyer.dnd) if buyer else False,
                },
                "contract": {
                    "delivery_terms": contract.delivery_terms if contract else "",
                    "payment_terms_days": contract.payment_terms_days if contract else 0,
                    "tds_section_expected": contract.tds_section_expected if contract else "",
                    "tds_rate_expected_bp": contract.tds_rate_expected_bp if contract else 0,
                    "tcs_applicable": bool(contract.tcs_applicable) if contract else False,
                },
                "invoice": {
                    "no": inv.invoice_no if inv else "",
                    "issue_date": inv.issue_date if inv else "",
                    "due_date": inv.due_date if inv else "",
                    "taxable_paise": inv.taxable_paise if inv else 0,
                    "gst_paise": inv.gst_paise if inv else 0,
                    "tcs_paise": inv.tcs_paise if inv else 0,
                    "total_paise": inv.total_paise if inv else 0,
                },
                "payment": {
                    "utr": pay.utr if pay else "",
                    "value_date": pay.value_date if pay else "",
                    "amount_paise": pay.amount_paise if pay else 0,
                    "narration": pay.narration_raw if pay else "",
                },
                "advice": (
                    {
                        "format": advice.format,
                        "received_at": advice.received_at,
                        # Truncated: some advices are long spreadsheets dumped to text and
                        # the page shows an excerpt, not the file.
                        "raw_text": advice.raw_text[:1800],
                        "truncated": len(advice.raw_text) > 1800,
                    }
                    if advice
                    else None
                ),
                "deduction": {
                    "amount_paise": ded.amount_paise,
                    "claimed_reason_text": ded.claimed_reason_text,
                    "predicted_code": ded.predicted_code,
                    "predicted_label": code.label if code else ded.predicted_code,
                    "family": code.family if code else "unknown",
                    "confidence": round(float(ded.predicted_confidence), 3),
                    "rationale": ded.predicted_rationale,
                    "predicted_by": ded.predicted_by,
                    "verdict": ded.verdict,
                    "recoverable_paise": ded.recoverable_paise,
                    "verification": ded.verification or {},
                },
                "contacts": [
                    {
                        "ts": c.ts,
                        "channel": c.channel,
                        "role": c.recipient_role,
                        "template_id": c.template_id,
                        "subject": c.subject,
                        "body": c.body,
                        "drafted_by": c.drafted_by,
                        "checks": list(c.policy_checks_passed or []),
                        "rejections": list(c.validator_rejections or []),
                        "cost_paise": c.cost_paise,
                        "response_kind": c.response_kind,
                        "response_text": c.response_text,
                        "response_at": c.response_received_at,
                    }
                    for c in contacts_by_case.get(case.id, [])
                ],
                "trace": trace_by_case.get(case.id, []),
                # The answer key. Fenced under one field on purpose — the UI hides it
                # behind a toggle and nothing else in the payload leaks it.
                "truth": (
                    {
                        "true_reason_code": truth.true_reason_code,
                        "is_valid": bool(truth.is_valid),
                        "recoverable_paise": truth.recoverable_paise,
                        "will_pay_if_chased": bool(truth.will_pay_if_chased),
                        "pays_after_n_contacts": truth.pays_after_n_contacts,
                        "responds_to_channels": list(truth.responds_to_channels or []),
                        "responds_only_at_role": truth.responds_only_at_role,
                        "will_dispute": bool(truth.will_dispute),
                        "promise_then_default": bool(truth.promise_then_default),
                        "opt_out": bool(truth.opt_out),
                        "showcase_id": truth.showcase_id,
                        "notes": truth.notes,
                        "code_correct": truth.true_reason_code == ded.predicted_code,
                    }
                    if truth
                    else None
                ),
            }
        )

    # ---- the leak -------------------------------------------------------------------
    by_family: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "paise": 0, "recoverable_paise": 0, "valid_count": 0}
    )
    by_code: dict[str, dict[str, Any]] = {}
    for ded in deductions.values():
        truth = truths.get(ded.id)
        if truth is None:
            continue
        code_def = taxonomy.codes.get(truth.true_reason_code)
        family = code_def.family if code_def else "unknown"
        bucket = by_family[family]
        bucket["count"] += 1
        bucket["paise"] += int(ded.amount_paise)
        bucket["recoverable_paise"] += int(truth.recoverable_paise)
        bucket["valid_count"] += 1 if truth.is_valid else 0

        slot = by_code.setdefault(
            truth.true_reason_code,
            {
                "code": truth.true_reason_code,
                "label": code_def.label if code_def else truth.true_reason_code,
                "family": family,
                "chaseable": bool(code_def.chaseable) if code_def else False,
                "count": 0,
                "paise": 0,
                "recoverable_paise": 0,
            },
        )
        slot["count"] += 1
        slot["paise"] += int(ded.amount_paise)
        slot["recoverable_paise"] += int(truth.recoverable_paise)

    by_segment: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "paise": 0})
    by_buyer: dict[str, dict[str, Any]] = {}
    for ded in deductions.values():
        inv = invoices.get(ded.invoice_id)
        buyer = buyers.get(inv.buyer_id) if inv else None
        if buyer is None:
            continue
        seg = by_segment[buyer.segment]
        seg["count"] += 1
        seg["paise"] += int(ded.amount_paise)
        row = by_buyer.setdefault(
            buyer.id,
            {
                "id": buyer.id,
                "name": buyer.name,
                "segment": buyer.segment,
                "behaviour": buyer.payment_behaviour_tag,
                "count": 0,
                "paise": 0,
                "recoverable_paise": 0,
            },
        )
        row["count"] += 1
        row["paise"] += int(ded.amount_paise)
        truth = truths.get(ded.id)
        if truth:
            row["recoverable_paise"] += int(truth.recoverable_paise)

    # Deduction sizes, log-spaced. A linear histogram of Indian AR deductions is one
    # tall bar at zero and nothing else.
    edges = [0, 50_000, 200_000, 1_000_000, 5_000_000, 20_000_000, 10**12]
    labels = ["<Rs 500", "Rs 500–2k", "Rs 2k–10k", "Rs 10k–50k", "Rs 50k–2L", ">Rs 2L"]
    histogram = [{"label": lab, "count": 0, "paise": 0} for lab in labels]
    for ded in deductions.values():
        amount = int(ded.amount_paise)
        for i in range(len(edges) - 1):
            if edges[i] <= amount < edges[i + 1]:
                histogram[i]["count"] += 1
                histogram[i]["paise"] += amount
                break

    invoice_total = sum(int(i.total_paise) for i in invoices.values())
    short_paid = sum(int(d.amount_paise) for d in deductions.values())
    largest = max(deductions.values(), key=lambda d: int(d.amount_paise), default=None)

    leak = {
        "invoices": len(invoices),
        "invoice_value_paise": invoice_total,
        "payments": len(payments),
        "deductions": len(deductions),
        "short_paid_paise": short_paid,
        "leak_rate_bp": round(short_paid / invoice_total * 10_000) if invoice_total else 0,
        "recoverable_paise": sum(int(t.recoverable_paise) for t in truths.values()),
        "reachable_paise": sum(
            int(t.recoverable_paise) for t in truths.values() if t.will_pay_if_chased
        ),
        "valid_paise": sum(
            int(deductions[t.deduction_id].amount_paise)
            for t in truths.values()
            if t.is_valid and t.deduction_id in deductions
        ),
        "largest_deduction": (
            {
                "paise": int(largest.amount_paise),
                "code": (truths[largest.id].true_reason_code if largest.id in truths else ""),
                "reason_text": largest.claimed_reason_text or "",
            }
            if largest is not None
            else None
        ),
        "smallest_bucket_count": sum(h["count"] for h in histogram[:2]),
        "by_family": [{"family": k, **v} for k, v in sorted(by_family.items())],
        "by_code": sorted(by_code.values(), key=lambda r: -r["paise"]),
        "by_segment": [{"segment": k, **v} for k, v in sorted(by_segment.items())],
        "top_buyers": sorted(by_buyer.values(), key=lambda r: -r["paise"])[:12],
        "histogram": histogram,
        # Real remittance advices, one per format. The single most convincing thing on the
        # page is not a chart — it is the actual text a buyer sends, in the actual register
        # they send it in. Picked deterministically (shortest usable text per format) so the
        # page does not change character between exports.
        "sample_advices": _sample_advices(advices, deductions),
    }

    # ---- pipeline funnel -------------------------------------------------------------
    matched_invoices = {a.invoice_id for a in allocations}
    matched_payments = {a.payment_event_id for a in allocations}
    by_method: dict[str, int] = defaultdict(int)
    for a in allocations:
        by_method[a.method] += 1
    exceptions_by_kind: dict[str, int] = defaultdict(int)
    for e in exceptions:
        exceptions_by_kind[e.kind] += 1

    # Resolved before the funnel is built, because the funnel's "matched" stage has to quote
    # the same number the matching panel quotes. Deriving it from the allocation table gives
    # 178 while the matcher itself reports 182 — four payments were matched to a buyer and
    # then blocked from allocating — and two different match counts on one page is the kind
    # of thing a judge finds in ten seconds.
    matching_stats = _matching_stats(
        payments=len(payments),
        allocated_payments=len(matched_payments),
        by_method=by_method,
        exceptions_by_kind=exceptions_by_kind,
        exceptions_total=len(exceptions),
    )

    classified = [d for d in deductions.values() if d.predicted_code != "NEEDS_HUMAN"]
    abstained = len(deductions) - len(classified)
    adjudicated = [d for d in deductions.values() if d.verdict != "unknown"]

    state_counts: dict[str, int] = defaultdict(int)
    for c in cases:
        state_counts[c.state] += 1

    funnel = [
        {
            "key": "invoices",
            "label": "Invoices raised",
            "count": len(invoices),
            "paise": invoice_total,
            "note": "Generated batch, seed 42.",
        },
        {
            "key": "payments",
            "label": "Bank credits received",
            "count": len(payments),
            "paise": sum(int(p.amount_paise) for p in payments.values()),
            "note": "Mangled narrations, split payments, bundled UTRs.",
        },
        {
            "key": "matched",
            "label": "Credits matched to invoices",
            "count": int(matching_stats["matched"]),
            "paise": sum(int(a.allocated_paise) for a in allocations),
            "note": (
                f"{len(matched_invoices)} invoices allocated across "
                f"{len(matched_payments)} credits. Advice, then exact, normalised, fuzzy, "
                "subset-sum. Anything left over is published as an exception rather than "
                "guessed."
            ),
        },
        {
            "key": "deductions",
            "label": "Shortfalls isolated",
            "count": len(deductions),
            "paise": short_paid,
            "note": "Every rupee of invoice-to-cash gap, split by component.",
        },
        {
            "key": "cases",
            "label": "Cases opened",
            "count": len(cases),
            "paise": sum(
                int(deductions[c.deduction_id].amount_paise)
                for c in cases
                if c.deduction_id in deductions
            ),
            "note": "Split payments wait out the settlement grace period first.",
        },
        {
            "key": "classified",
            "label": "Classified by the model",
            "count": len(classified),
            "paise": sum(int(d.amount_paise) for d in classified),
            "note": f"{abstained} abstained to a human rather than guess.",
        },
        {
            "key": "verified",
            "label": "Verified against source data",
            "count": len(adjudicated),
            "paise": sum(int(d.amount_paise) for d in adjudicated),
            "note": "26AS, credit-note ledger, contract, scheme master, GRN.",
        },
        {
            "key": "contacted",
            "label": "Cases chased",
            "count": sum(1 for c in cases if c.contacts_used > 0),
            "paise": sum(
                int(deductions[c.deduction_id].amount_paise)
                for c in cases
                if c.contacts_used > 0 and c.deduction_id in deductions
            ),
            "note": (
                f"{len(contacts)} contacts in total — inside the IST window, on a channel "
                "the buyer consented to, every one queued dry-run."
            ),
        },
        {
            "key": "recovered",
            "label": "Recovered",
            "count": sum(1 for c in cases if c.recovered_paise > 0),
            "paise": sum(int(c.recovered_paise) for c in cases),
            "note": "Cash back against the shortfall.",
        },
    ]

    # ---- daily curve ------------------------------------------------------------------
    horizon = 46
    recovered_by_day = [0] * horizon
    contacts_by_day = [0] * horizon
    closed_by_day = [0] * horizon
    for c in cases:
        d = _day_index(c.closed_at, start)
        if d is not None and 0 <= d < horizon:
            recovered_by_day[d] += int(c.recovered_paise)
            closed_by_day[d] += 1
    for c in contacts:
        d = _day_index(c.ts, start)
        if d is not None and 0 <= d < horizon:
            contacts_by_day[d] += 1

    curve, running, running_c, running_cl = [], 0, 0, 0
    for i in range(horizon):
        running += recovered_by_day[i]
        running_c += contacts_by_day[i]
        running_cl += closed_by_day[i]
        curve.append(
            {
                "day": i,
                "recovered_paise": running,
                "contacts": running_c,
                "closed": running_cl,
            }
        )

    pipeline = {
        "funnel": funnel,
        "curve": curve,
        "matching": matching_stats,
        "states": [{"state": k, "count": v} for k, v in sorted(state_counts.items())],
        "abstained": abstained,
        "outbox": len(outbox),
        "outbox_sent_for_real": sum(1 for o in outbox if not o.dry_run),
        "approvals_requested": len(approvals),
        "approvals_granted": sum(1 for a in approvals if a.approved),
    }

    # ---- guardrails -----------------------------------------------------------------
    comp = policy.compliance
    guardrails = {
        "contact_window": comp["contact_window_ist"],
        "contact_days": comp["contact_days"],
        "max_contacts_per_case": comp["max_contacts_per_case"],
        "max_contacts_per_buyer_per_week": comp["max_contacts_per_buyer_per_week"],
        "min_gap_hours": comp["min_gap_hours"],
        "channel_ladder": comp["channel_ladder"],
        "escalation_ladder": comp["escalation_ladder"],
        "forbidden_phrases": comp["forbidden_phrases"],
        "require_consent_for": comp["require_consent_for"],
        "stopping_rules": policy.stopping,
        "economics": policy.economics,
        "confidence": policy.confidence,
        "settlement": policy.settlement,
        "verification": policy.verification,
    }

    return {
        "run_id": run_id,
        "cases": case_rows,
        "leak": leak,
        "pipeline": pipeline,
        "guardrails": guardrails,
        "taxonomy": [
            {
                "code": c.code,
                "label": c.label,
                "family": c.family,
                "verifier": c.verifier,
                "chaseable": c.chaseable,
                "default_valid": c.default_valid,
                "default_action": c.default_action,
                "expected_rate_bp": c.expected_rate_bp,
                "note": c.note,
            }
            for c in sorted(taxonomy.codes.values(), key=lambda c: (c.family, c.code))
        ],
    }


def export_web_data(
    session: Session,
    *,
    run_id: str | None = None,
    out_dir: Path | None = None,
    db_hash: str = "",
    seed: int = 42,
    days: int = 45,
) -> list[Path]:
    """Write the whole snapshot. Returns the files written."""
    out = out_dir or WEB_DATA_DIR
    snap = build_snapshot(session, run_id=run_id)

    # Full case records carry the raw advice text, every draft body and the decision
    # trace — about 2 MB. Bundling that into the JS would put it on the critical path for
    # a hero the visitor sees before they ever open a case. It goes to `public/` and is
    # fetched when the explorer mounts; the index below is what renders immediately.
    public = out.parents[1] / "public" / "data"
    index = [
        {
            "id": c["id"],
            "buyer": c["buyer"]["name"],
            "segment": c["buyer"]["segment"],
            "amount_paise": c["deduction"]["amount_paise"],
            "code": c["deduction"]["predicted_code"],
            "label": c["deduction"]["predicted_label"],
            "family": c["deduction"]["family"],
            "confidence": c["deduction"]["confidence"],
            "verdict": c["deduction"]["verdict"],
            "state": c["state"],
            "contacts": len(c["contacts"]),
            "recovered_paise": c["recovered_paise"],
            "showcase_id": (c["truth"] or {}).get("showcase_id"),
        }
        for c in snap["cases"]
    ]

    written = [
        _write(public / "cases.json", snap["cases"]),
        _write(out / "cases_index.json", index),
        _write(out / "leak.json", snap["leak"]),
        _write(out / "pipeline.json", snap["pipeline"]),
        _write(out / "guardrails.json", snap["guardrails"]),
        _write(out / "taxonomy.json", snap["taxonomy"]),
        _write(out / "broke.json", parse_broke(ROOT / "docs" / "BROKE.md")),
        # The showcase manifest is public config, not the answer key: `config/generator.yaml`
        # declares in advance which seven scenarios must appear in every batch, precisely so
        # the demo does not depend on the RNG happening to produce a good case. Exporting it
        # is not a truth leak — the note says what to look at, not what the verdict is.
        _write(
            out / "showcase.json",
            [
                {
                    "id": s["id"],
                    "reason": s.get("reason", ""),
                    "note": s.get("note", ""),
                }
                for s in load_generator_config().get("showcase", [])
            ],
        ),
    ]

    # The scoreboard is produced by `report`, which re-runs every policy. Copying it
    # rather than recomputing keeps one source of truth for the comparison table.
    scoreboard_src = ROOT / "reports" / "scoreboard.json"
    if scoreboard_src.exists():
        written.append(
            _write(out / "scoreboard.json", json.loads(scoreboard_src.read_text("utf-8")))
        )

    written.append(
        _write(
            out / "meta.json",
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "run_id": snap["run_id"],
                "seed": seed,
                "days": days,
                "db_content_hash": db_hash,
                "cases": len(snap["cases"]),
                "config_files": sorted(p.name for p in CONFIG_DIR.glob("*.yaml")),
                "scoreboard_present": scoreboard_src.exists(),
            },
        )
    )
    return written
