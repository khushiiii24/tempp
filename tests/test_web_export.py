"""Acceptance for the web snapshot.

The front end no longer reads the database — it reads static JSON written by
`export-web`. That decouples it usefully, and it introduces a failure mode the old
dashboard did not have: **the site can be wrong while the pipeline is right**, because a
stale or mis-scoped snapshot still renders perfectly.

So these tests police the boundary rather than the rendering. There is deliberately no test
of the React components; testing a component that reads `leak.json` and puts it in a `<div>`
tests React, not this project. What is worth asserting is that the JSON is the right run,
that it does not contradict itself, and that the answer key stays fenced.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from sqlmodel import Session, select

from deduction_desk.config import ROOT
from deduction_desk.db import make_engine
from deduction_desk.eval.web_export import WEB_DATA_DIR, build_snapshot, parse_broke
from deduction_desk.schemas import Run


@pytest.fixture(scope="module")
def agent_run() -> str:
    """Put a known agent run in the database before anything reads it.

    `Case` and `ContactLog` are global rather than run-scoped, so every other test in the
    suite that exercises a policy leaves its own outcomes behind. Without this the snapshot
    under test belongs to whichever test ran last — which is the exact confusion the
    exporter's guard exists to catch, so the fixture would trip it.
    """
    from deduction_desk.runner import agent_policy, run_batch

    engine = make_engine()
    with Session(engine) as session:
        if not session.exec(select(Run)).first():
            pytest.skip("empty database; run `generate` first")
        # Unique per invocation. `decision_log` is append-only and its row ids are derived
        # from the run id, so a fixed one collides on the primary key the second time the
        # suite runs against the same database.
        run_id = f"agent-webtest-{uuid4().hex[:8]}"
        run_batch(
            session,
            run_id=run_id,
            policy_fn=agent_policy,
            policy_name="agent",
            days=45,
            seed=42,
        )
    return run_id


@pytest.fixture(scope="module")
def snapshot(agent_run: str):
    engine = make_engine()
    with Session(engine) as session:
        return build_snapshot(session, run_id=agent_run)


# ======================================================================================
# The answer key stays fenced
# ======================================================================================
TRUTH_FIELDS = {
    "true_reason_code",
    "is_valid",
    "will_pay_if_chased",
    "pays_after_n_contacts",
    "responds_only_at_role",
    "will_dispute",
    "promise_then_default",
    "opt_out",
    "code_correct",
}


def test_truth_is_confined_to_one_key(snapshot):
    """Ground truth appears under `truth` and nowhere else on a case.

    The UI hides that one key behind a toggle. If a truth-derived field ever leaked into
    the case body — `is_valid` next to the verdict, say — the toggle would still work and
    the page would quietly be showing the answer beside the question.
    """
    for case in snapshot["cases"]:
        body = {k: v for k, v in case.items() if k != "truth"}
        leaked = TRUTH_FIELDS & set(body)
        assert not leaked, f"{case['id']} leaks {sorted(leaked)} outside `truth`"
        assert TRUTH_FIELDS & set(case["truth"] or {}), "truth block is missing its fields"


def test_case_index_carries_no_truth():
    """The list rendered before the case file loads must not carry the answer.

    `showcase_id` is the one exception and is not truth: `config/generator.yaml` declares
    the seven pinned scenarios in advance, so it is public configuration.
    """
    path = WEB_DATA_DIR / "cases_index.json"
    if not path.exists():
        pytest.skip("no exported index; run `export-web` first")
    index = json.loads(path.read_text("utf-8"))
    for row in index:
        leaked = (TRUTH_FIELDS & set(row)) - {"showcase_id"}
        assert not leaked, f"{row['id']} leaks {sorted(leaked)}"


# ======================================================================================
# The snapshot does not contradict itself
# ======================================================================================
def test_funnel_matched_agrees_with_the_matching_panel(snapshot):
    """One match count per page.

    These come from two sources — the funnel stage and the matcher's own report — and they
    disagreed by four payments on the first build, because a payment can be matched to a
    buyer and still be blocked from allocating. Two different match rates on one page is
    the sort of thing a judge finds in ten seconds.
    """
    pipeline = snapshot["pipeline"]
    stage = next(s for s in pipeline["funnel"] if s["key"] == "matched")
    assert stage["count"] == pipeline["matching"]["matched"]


def test_funnel_narrows_monotonically(snapshot):
    """Every stage counts a subset of the one before it, from cases onward.

    The first stages count different things (invoices, then credits, then credits again),
    so the check starts where the unit stops changing. A funnel that widens is a funnel
    counting the same thing twice.
    """
    keys = ["cases", "classified", "verified"]
    stages = {s["key"]: s["count"] for s in snapshot["pipeline"]["funnel"]}
    counts = [stages[k] for k in keys if k in stages]
    assert counts == sorted(counts, reverse=True), dict(zip(keys, counts, strict=True))


def test_recovered_never_exceeds_what_was_chased(snapshot):
    """A case cannot recover more than its deduction.

    This is the invariant that caught BROKE entry 15, where four cases collected exactly
    twice because two contacts each scheduled a credit before the first one landed. It is
    cheap and it is the only guard against the whole scoreboard silently inflating.
    """
    for case in snapshot["cases"]:
        assert case["recovered_paise"] <= case["deduction"]["amount_paise"], case["id"]


def test_leak_totals_reconcile(snapshot):
    """Family totals add up to the headline, to the paise."""
    leak = snapshot["leak"]
    assert sum(f["paise"] for f in leak["by_family"]) == leak["short_paid_paise"]
    assert sum(c["paise"] for c in leak["by_code"]) == leak["short_paid_paise"]
    assert leak["reachable_paise"] <= leak["recoverable_paise"] <= leak["short_paid_paise"]


def test_the_snapshot_is_one_run(snapshot):
    """Traces come from a single execution.

    The decision log is append-only and accumulates every run ever made. Without scoping,
    a case shows a dozen interleaved histories and the file is seven times larger — which
    is how the first export shipped, at 14 MB.
    """
    assert snapshot["run_id"], "no run resolved"
    for case in snapshot["cases"]:
        seqs = [row["seq"] for row in case["trace"]]
        assert seqs == sorted(seqs), f"{case['id']} trace is out of order"


def test_nothing_was_actually_sent(snapshot):
    """The outbox is dry-run by construction and the site says so."""
    assert snapshot["pipeline"]["outbox_sent_for_real"] == 0


def test_snapshot_refuses_a_case_table_from_another_policy(agent_run: str):
    """Exporting after `report` without re-running must fail, not publish b3 as the agent.

    `Case` and `ContactLog` are global rather than run-scoped, so whichever policy ran last
    owns them. `report` runs the agent first and b3 last; exporting straight afterwards
    produced a snapshot carrying the agent's decision log and b3's outcomes, and it reached
    the page quoting 13 recoveries where the agent made 18. It renders perfectly, which is
    what makes it dangerous.
    """
    engine = make_engine()
    with Session(engine) as session:
        runs = list(session.exec(select(Run)).all())
        agent_runs = [r for r in runs if r.id == agent_run and r.stats]
        if not agent_runs:
            pytest.skip("no agent run with stats recorded")

        # Point the exporter at some *other* policy's run while the case table holds the
        # agent's. The recovered totals differ, so the guard has to notice.
        others = [
            r
            for r in runs
            if r.policy_name != "agent"
            and r.stats
            and int(r.stats.get("recovered_paise", -1))
            != int(agent_runs[-1].stats.get("recovered_paise", -2))
        ]
        if not others:
            pytest.skip("no baseline run with a different recovery total")

        with pytest.raises(RuntimeError, match="does not belong to run"):
            build_snapshot(session, run_id=others[-1].id)


# ======================================================================================
# BROKE.md parses
# ======================================================================================
def test_broke_log_parses_with_all_four_fields():
    """The failure log is rendered from the markdown, so the markdown has to stay parseable.

    A heading typed with a hyphen instead of an em dash silently drops an entry from the
    site while leaving the document itself perfectly readable.
    """
    entries = parse_broke(ROOT / "docs" / "BROKE.md")
    assert len(entries) >= 5, "the submission asks for at least five dated entries"
    assert [e["n"] for e in entries] == sorted(e["n"] for e in entries)
    for entry in entries:
        assert entry["title"] and entry["date"]
        assert entry["symptom"], f"entry {entry['n']} has no symptom"
        # `Fix` or `Result` — entry 9 describes its six sub-fixes inline and closes on the
        # measured outcome instead, which is the same beat under a different heading.
        assert entry["fix"] or entry["result"], f"entry {entry['n']} has no resolution"
        assert "**" not in entry["symptom"], "markdown leaked into the plain text"
