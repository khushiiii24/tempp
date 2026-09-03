"""Typer CLI. `python -m deduction_desk <command>`.

Commands land here as the phases complete. `doctor` is first because on a local-inference
project the most common failure is environmental — the server is down, the model was
never pulled, or the machine is far slower than the operator assumed — and all three
should be diagnosed in ten seconds rather than discovered four hours into a batch.
"""

from __future__ import annotations

import json
import time

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import REPORTS_DIR, load_generator_config, load_policy, load_taxonomy
from .llm.batch import _fmt_duration, estimate_batch_wall_clock
from .llm.client import build_client, load_llm_config, max_tokens_for
from .llm.errors import LLMError, ProviderUnavailable
from .llm.schema import Classification
from .money import format_inr

app = typer.Typer(
    add_completion=False,
    help="Katauti — recover revenue lost to B2B short payments.",
    no_args_is_help=True,
)
console = Console()


# Call-volume model for the runtime estimate. Derived from generator.yaml rather than
# hardcoded, so changing the batch size changes the estimate.
def _expected_call_counts() -> dict[str, int]:
    gen = load_generator_config()
    n_invoices = int(gen["batch"]["n_invoices"])
    deducting = int(n_invoices * float(gen["batch"]["deduction_rate"]))
    multi = 1.0 + float(gen["batch"]["multi_component_rate"])
    advices = int(deducting * (1.0 - float(gen["messiness"]["advice_absent_rate"])))
    deductions = int(deducting * multi)
    # Drafts only happen for cases that actually get chased — a minority, since most
    # deductions are valid TDS and close without contact.
    drafts = int(deductions * 0.45)
    return {"parse": advices, "classify": deductions, "draft": drafts}


def _representative_probe_prompt() -> str:
    """A prompt the same shape and size as a real classification.

    Built from the actual static preamble plus a synthetic case block, so the probe pays
    the same prompt-evaluation cost the batch will. A short prompt would measure
    generation speed and quietly ignore the 90% of the work that is prompt evaluation.
    """
    from .classify.prompts import _static_preamble

    return _static_preamble(72) + """## Invoice
INV/2026/0042 | taxable Rs 9,60,000 | total Rs 11,32,800
issued 2026-05-01, due 2026-06-15

## Contract
delivery ex_works (freight borne by buyer) |
expected TDS TDS_194C at 2.00% |
we charge TCS: no | early-payment discount: none

## Buyer
Probe Industries Pvt Ltd (midmarket) | prior deductions: TDS-like x3

## The deduction
amount Rs 19,200
stated reason (VERBATIM, may be wrong): "less TDS as per our records"

## Arithmetic (computed for you - trust these)
- deduction is 2.00% of TAXABLE value
- statutory rates matching that percentage: TDS_194C (2.00%)"""


@app.command()
def doctor(
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Check the local inference stack and estimate full-batch runtime.

    Verifies the provider is reachable, the configured model is actually pulled, and a
    schema-constrained round-trip validates — then measures tokens/sec on this machine
    and turns that into an honest wall-clock estimate for the whole batch.
    """
    cfg = load_llm_config()
    provider = cfg.get("provider")
    report: dict[str, object] = {"provider": provider, "checks": {}}

    client = build_client(cfg=cfg)
    console.print(f"[bold]provider[/bold]  {provider}")
    console.print(f"[bold]model[/bold]     {client.model}")

    # -- reachability & model presence ------------------------------------------
    health = client.health()
    report["health"] = health

    if not health.get("reachable"):
        console.print(
            Panel(
                f"[red]Cannot reach the provider.[/red]\n\n{health.get('error')}\n\n"
                "Start it with:  [bold]ollama serve[/bold]",
                title="doctor: FAILED",
                border_style="red",
            )
        )
        if json_out:
            console.print_json(json.dumps(report, default=str))
        raise typer.Exit(code=1)

    console.print("[green]OK[/green]        provider reachable")

    if not health.get("model_present"):
        installed = health.get("installed") or []
        console.print(
            Panel(
                f"[red]Model '{client.model}' is not pulled.[/red]\n\n"
                f"Installed: {', '.join(installed) if installed else '(none)'}\n\n"
                f"Pull it with:  [bold]ollama pull {client.model}[/bold]",
                title="doctor: FAILED",
                border_style="red",
            )
        )
        if json_out:
            console.print_json(json.dumps(report, default=str))
        raise typer.Exit(code=1)

    console.print(f"[green]OK[/green]        model present ({len(health.get('installed', []))} installed)")

    # -- schema-validated round trip --------------------------------------------
    # Deliberately a REALISTIC classification prompt, not a toy ping.
    #
    # An earlier version probed with a 20-token prompt and projected the batch from
    # generation speed alone. Measurement showed that to be badly wrong-headed: on this
    # workload prompt evaluation is ~85 of every ~95 seconds, and generation is the small
    # part. A probe that skips prompt evaluation is measuring the wrong 10% of the work,
    # and would happily promise an overnight run that takes two days.
    prompt = _representative_probe_prompt()
    started = time.perf_counter()
    try:
        response = client.complete_detailed(
            prompt,
            schema=Classification,
            max_tokens=max_tokens_for("classify", cfg),
            temperature=0.0,
            task="doctor",
        )
    except (LLMError, ProviderUnavailable) as exc:
        console.print(Panel(f"[red]{exc}[/red]", title="doctor: FAILED", border_style="red"))
        raise typer.Exit(code=1) from exc

    elapsed = time.perf_counter() - started
    ok = isinstance(response.value, Classification)
    console.print(
        f"[{'green' if ok else 'yellow'}]{'OK' if ok else 'WARN'}[/]        "
        f"schema round-trip {'validated' if ok else 'returned unexpected content'}"
        f"{' (from cache)' if response.cached else ''}"
    )
    if response.repairs:
        console.print(f"[yellow]note[/yellow]      {len(response.repairs)} schema repair(s) needed")

    # -- throughput --------------------------------------------------------------
    tps = response.meta.get("tokens_per_sec")
    eval_count = response.meta.get("eval_count") or 0
    report["checks"] = {"round_trip_ok": bool(ok), "tokens_per_sec": tps, "elapsed_s": elapsed}

    table = Table(title="Measured on this machine", show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    prompt_tokens = response.meta.get("prompt_eval_count") or 0
    table.add_row("tokens/sec (generation)", f"{tps:.1f}" if tps else "n/a")
    table.add_row("prompt tokens in probe", str(prompt_tokens))
    table.add_row("output tokens in probe", str(eval_count))
    table.add_row("wall clock for probe", f"{elapsed:.1f}s")
    table.add_row("num_ctx used", str(response.meta.get("num_ctx", "n/a")))
    console.print(table)

    if prompt_tokens and eval_count:
        console.print(
            f"[dim]Prompt evaluation is the dominant cost on CPU "
            f"({prompt_tokens} prompt tokens vs {eval_count} generated) — which is why the "
            f"classifier prompt puts its static preamble first, so llama.cpp can reuse the "
            f"KV cache across calls.[/dim]"
        )

    # -- full batch estimate -----------------------------------------------------
    counts = _expected_call_counts()
    workers = int(cfg.get("ollama", {}).get("max_workers", 2))

    # The probe's measured end-to-end latency is the unit of projection — it already
    # contains prompt evaluation, generation and the round trip, which is the whole cost.
    probe_s = float(response.meta.get("wall_s") or elapsed)

    if response.cached:
        console.print(
            "[yellow]note[/yellow]      probe served from cache; "
            "clear .llm_cache/doctor for a fresh timing"
        )

    est = Table(title="Estimated full-batch runtime", header_style="bold")
    est.add_column("task")
    est.add_column("calls", justify="right")
    est.add_column("s/call", justify="right")
    est.add_column("est. wall clock", justify="right")

    total_s = 0.0
    for task, n in counts.items():
        # Scale the measured classification latency by each task's output budget. Output
        # is the minority of the cost, so this is a gentle adjustment rather than a
        # proportional one.
        budget_ratio = max_tokens_for(task, cfg) / max(1, max_tokens_for("classify", cfg))
        per_call_s = probe_s * (0.85 + 0.15 * budget_ratio)
        task_s = estimate_batch_wall_clock(n, per_call_s, max_workers=workers)
        total_s += task_s
        est.add_row(task, str(n), f"{per_call_s:.0f}s", _fmt_duration(task_s))

    est.add_row(
        "[bold]total[/bold]",
        f"[bold]{sum(counts.values())}[/bold]",
        "",
        f"[bold]{_fmt_duration(total_s)}[/bold]",
    )
    console.print(est)
    report["estimate_s"] = total_s

    verdict = (
        "comfortably overnight" if total_s < 8 * 3600
        else "an overnight run" if total_s < 14 * 3600
        else "[yellow]longer than one night — consider a smaller model or --n[/yellow]"
    )
    console.print(f"\nFull batch is {verdict}.")

    # -- Phase 7 integration status ---------------------------------------------
    from .actions.razorpay_client import build_razorpay_client

    rzp = build_razorpay_client().health()
    if rzp.get("enabled"):
        console.print(
            f"\nrazorpay: [green]test mode[/green] ({rzp.get('key_id')}), "
            f"reachable={rzp.get('reachable')}"
        )
    else:
        console.print(
            f"\nrazorpay: offline — {rzp.get('reason')}\n"
            f"[dim]This is the default and the path every measured number was produced on. "
            f"No network call is possible through the null client.[/dim]"
        )

    inventory = client.cache.inventory()
    if inventory:
        console.print(
            f"\ncache: {sum(inventory.values())} entries "
            f"({', '.join(f'{k}={v}' for k, v in inventory.items())})"
        )
    else:
        console.print("\ncache: empty — the first run does the inference, later runs are free")

    if json_out:
        console.print_json(json.dumps(report, default=str))


@app.command()
def generate(
    seed: int = typer.Option(42, help="Master seed. Same seed must give the same database."),
    n: int = typer.Option(None, "--n", help="Number of invoices. Defaults to generator.yaml."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress the summary tables."),
) -> None:
    """Build the synthetic batch: invoices, payments, advices, fixtures and ground truth.

    Truth-first: the reason code and its validity are decided before the amount, the
    fixtures are written to match, and only then is the observable mess produced. Run it
    twice with the same seed and the database content hash is identical.
    """
    from .db import content_hash, make_engine
    from .generator import generate as run_generate

    report = run_generate(seed=seed, n=n)

    if quiet:
        console.print(content_hash(make_engine()))
        return

    ceiling = report["ceiling"]

    overview = Table(title=f"Batch generated (seed={seed})", header_style="bold")
    overview.add_column("metric")
    overview.add_column("value", justify="right")
    overview.add_row("invoices", str(report["n_invoices"]))
    overview.add_row("buyers", str(report["n_buyers"]))
    overview.add_row("invoices with a deduction", str(report["n_invoices_with_deduction"]))
    overview.add_row("deductions", str(report["n_deductions"]))
    overview.add_row("payment events", str(report["n_payments"]))
    overview.add_row("remittance advices", str(report["n_advices"]))
    overview.add_row("  ...of which absent", str(report["n_advices_absent"]))
    console.print(overview)

    money = Table(title="The opportunity, from ground truth", header_style="bold")
    money.add_column("metric")
    money.add_column("value", justify="right")
    money.add_row("deductions", str(ceiling["n_deductions"]))
    money.add_row("  valid (must NOT be chased)", str(ceiling["n_valid"]))
    money.add_row("  invalid & recoverable", str(ceiling["n_recoverable"]))
    money.add_row("  will raise a dispute if chased", str(ceiling["n_will_dispute"]))
    money.add_row("recoverable ceiling", format_inr(ceiling["recoverable_paise"]))
    money.add_row("  reachable by chasing", format_inr(ceiling["reachable_paise"]))
    money.add_row("  unreachable (buyer never pays)", format_inr(ceiling["unreachable_paise"]))
    console.print(money)

    fx = Table(title="Fixtures written", header_style="bold")
    fx.add_column("store")
    fx.add_column("rows", justify="right")
    for name, count in report["fixtures"].items():
        fx.add_row(name, str(count))
    console.print(fx)

    show = Table(title="Showcase cases (pinned for the demo)", header_style="bold")
    for col in ("scenario", "deduction", "code", "valid", "amount"):
        show.add_column(col)
    for s in report["showcases"]:
        show.add_row(
            s["showcase_id"], s["deduction_id"], s["code"],
            "yes" if s["valid"] else "no", format_inr(s["amount_paise"]),
        )
    console.print(show)

    console.print(f"\ncontent hash: [bold]{content_hash(make_engine())}[/bold]")


@app.command()
def match(
    quiet: bool = typer.Option(False, "--quiet", help="Print the summary only."),
) -> None:
    """Run stages [1]-[3]: parse advices, match payments to invoices, isolate deltas.

    Deterministic. Publishes an exceptions report for everything it declined to resolve —
    a matcher that guesses scores better on every metric except the one that matters.
    """
    from sqlmodel import Session

    from .db import make_engine
    from .matching import run_matching
    from .matching.exceptions import render_report

    with Session(make_engine()) as session:
        report = run_matching(session)

    m, d, e = report["matching"], report["deltas"], report["exceptions"]

    t = Table(title="Matching", header_style="bold")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("payments", str(m["payments"]))
    t.add_row("matched", f"{m['matched']} ({m['match_rate']:.1%})")
    for method, count in m["by_method"].items():
        t.add_row(f"  via {method}", str(count))
    console.print(t)

    dt = Table(title="Deltas", header_style="bold")
    dt.add_column("metric")
    dt.add_column("value", justify="right")
    dt.add_row("invoices with allocations", str(d["invoices_with_allocations"]))
    dt.add_row("settled in full", str(d["fully_settled"]))
    dt.add_row("with a shortfall", str(d["with_shortfall"]))
    dt.add_row("awaiting settlement", str(d["awaiting_settlement"]))
    dt.add_row("total shortfall", format_inr(d["total_delta_paise"]))
    console.print(dt)

    if e["total"]:
        et = Table(title="Exceptions (published, not hidden)", header_style="bold")
        et.add_column("kind")
        et.add_column("count", justify="right")
        for kind, count in e["by_kind"].items():
            et.add_row(kind, str(count))
        console.print(et)

    out = REPORTS_DIR / "matching"
    out.mkdir(parents=True, exist_ok=True)
    (out / "exceptions.md").write_text(
        render_report(report["_exceptions"]), encoding="utf-8"
    )
    # The web snapshot reads this rather than re-deriving the match rate from the
    # allocation table. Those two numbers are not the same: a payment can be matched to a
    # buyer and still produce no allocation (an over-allocation block, an incomplete
    # bundle), so counting distinct allocation rows understates the matcher by four
    # payments. One source of truth, written by the code that did the matching.
    (out / "matching.json").write_text(
        json.dumps({k: v for k, v in report.items() if not k.startswith("_")}, indent=1),
        encoding="utf-8",
    )
    console.print(f"\nexceptions report: {out / 'exceptions.md'}")


@app.command()
def classify(
    offline: bool = typer.Option(False, "--offline", help="Serve only from the committed cache."),
    limit: int = typer.Option(
        None, "--limit", help="Classify only the first N deductions (by id)."
    ),
    slice_only: bool = typer.Option(
        False, "--slice", help="Classify only the stratified benchmark slice."
    ),
) -> None:
    """Run stages [4] and [5]: classify each deduction, then verify it against source data.

    `--offline` serves every call from `.llm_cache/` and refuses to touch a model, so this
    reproduces exactly on a machine that has never run inference.
    """
    from sqlmodel import Session

    from .classify.classifier import classify_batch
    from .classify.classifier import summarise as summarise_classification
    from .db import make_engine
    from .eval.slices import load_labelled_cases, stratified_slice
    from .llm.client import build_client
    from .verify import summarise as summarise_verification
    from .verify import verify_all

    ids: list[str] | None = None
    if slice_only:
        ids = [c.id for c in stratified_slice(load_labelled_cases(), size=40)]
    elif limit:
        ids = [c.id for c in load_labelled_cases()][:limit]

    client = build_client(offline=offline)
    engine = make_engine()

    with Session(engine) as session:
        outcomes = classify_batch(session, client, deduction_ids=ids)
        cls_stats = summarise_classification(outcomes)
        ver = verify_all(session, deduction_ids=ids)
        ver_stats = summarise_verification(ver)

    table = Table(title="Classification", header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("classified", str(cls_stats["n"]))
    table.add_row("actionable", str(cls_stats["actionable"]))
    table.add_row("abstained", f"{cls_stats['abstained']} ({cls_stats['abstention_rate']:.0%})")
    table.add_row("below confidence floor", str(cls_stats["below_floor"]))
    table.add_row("schema repairs", str(cls_stats["repairs"]))
    table.add_row("served from cache", str(cls_stats["cached"]))
    console.print(table)

    vt = Table(title="Verification", header_style="bold")
    vt.add_column("verdict")
    vt.add_column("count", justify="right")
    for verdict, count in ver_stats["by_verdict"].items():
        vt.add_row(verdict, str(count))
    vt.add_row("[bold]recoverable[/bold]", f"[bold]{format_inr(ver_stats['recoverable_paise'])}[/bold]")
    console.print(vt)


@app.command()
def run(
    policy: str = typer.Option("agent", help="agent | b0 | b1 | b2 | b3"),
    days: int = typer.Option(45, help="Length of the simulated clock."),
    seed: int = typer.Option(42, help="Seed for the counterparty simulation."),
    draft: str = typer.Option("template", help="template | llm — who writes the prose."),
    offline: bool = typer.Option(False, "--offline", help="LLM drafting from cache only."),
) -> None:
    """Run one policy over the simulated clock.

    Every policy — the agent and all four baselines — goes through the identical tick loop,
    so the only difference in the scoreboard is the decision function.
    """
    from sqlmodel import Session

    from .db import make_engine
    from .eval.baselines import BASELINES
    from .runner import agent_policy, run_batch

    if policy == "agent":
        policy_fn, label = agent_policy, "agent"
    elif policy in BASELINES:
        policy_fn, label = BASELINES[policy][0], policy
    else:
        console.print(f"[red]Unknown policy {policy!r}.[/red] Use agent, or one of {sorted(BASELINES)}.")
        raise typer.Exit(code=1)

    llm_client = None
    if draft == "llm":
        from .llm.client import build_client

        llm_client = build_client(offline=offline)

    # The decision log is append-only, so a re-run cannot reuse its row ids. Each
    # execution therefore gets its own run id; the log accumulates them, which is what
    # "append-only" means. `replay` resolves a case to its most recent execution.
    run_id = f"{label}-{seed}-{days}d-{_execution_nonce()}"
    with Session(make_engine()) as session:
        stats = run_batch(
            session,
            run_id=run_id,
            policy_fn=policy_fn,
            policy_name=label,
            days=days,
            seed=seed,
            llm_client=llm_client,
        )

    t = Table(title=f"Run: {label} ({days} days)", header_style="bold")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("cases", str(stats["cases"]))
    t.add_row("contacts sent", str(stats["contacts"]))
    t.add_row("recovered", format_inr(stats["recovered_paise"]))
    t.add_row("written off", format_inr(stats["written_off_paise"]))
    t.add_row("credit notes", format_inr(stats["credit_notes_paise"]))
    t.add_row("intervention cost", format_inr(stats["cost_paise"]))
    t.add_row("human queue", str(stats["human_queue"]))
    t.add_row("approvals queued", str(stats["approvals_queued"]))
    t.add_row("disputes", str(stats["disputes"]))
    console.print(t)

    st = Table(title="Case outcomes", header_style="bold")
    st.add_column("state")
    st.add_column("count", justify="right")
    for state, count in stats["by_state"].items():
        st.add_row(state, str(count))
    console.print(st)


def _execution_nonce() -> str:
    """A short, monotonic suffix distinguishing one execution from the next."""
    import time as _time

    return _time.strftime("%m%d%H%M%S")


@app.command()
def report(
    compare: str = typer.Option("agent,b0,b1,b2,b3", help="Comma-separated policies."),
    days: int = typer.Option(45),
    seed: int = typer.Option(42),
    emit_json: bool = typer.Option(
        True,
        "--json/--no-json",
        help="Also write reports/scoreboard.json for the web front end.",
    ),
) -> None:
    """Score every policy against ground truth and write the scoreboard.

    Re-runs each policy in turn so they are graded on identical data, then writes
    `reports/scoreboard.md`.
    """
    from datetime import date

    from sqlmodel import Session, select

    from .db import make_engine
    from .eval.baselines import BASELINES
    from .eval.compliance_audit import audit
    from .eval.harness import score_run
    from .eval.report import write_scoreboard
    from .runner import agent_policy, run_batch
    from .schemas import Buyer, Case, ContactLog, HumanApproval

    wanted = [p.strip() for p in compare.split(",") if p.strip()]
    cards = []
    compliance_by_policy: dict[str, dict] = {}
    curves_by_policy: dict[str, list[dict]] = {}
    pol = load_policy()

    def _daily_curve(cases: list, contacts: list) -> list[dict]:
        """Cumulative recovery and contacts per simulated day, for the web chart.

        Derived from the same `Case.closed_at` and `ContactLog.ts` the scorecard reads,
        so the last point of the curve equals the scoreboard's recovered figure by
        construction rather than by coincidence.
        """
        opened = sorted(c.opened_at[:10] for c in cases if c.opened_at)
        if not opened:
            return []
        origin = date.fromisoformat(opened[0])
        horizon = days + 1
        rec = [0] * horizon
        con = [0] * horizon
        for c in cases:
            if not c.closed_at:
                continue
            i = (date.fromisoformat(c.closed_at[:10]) - origin).days
            if 0 <= i < horizon:
                rec[i] += int(c.recovered_paise)
        for c in contacts:
            i = (date.fromisoformat(c.ts[:10]) - origin).days
            if 0 <= i < horizon:
                con[i] += 1
        out, r, k = [], 0, 0
        for i in range(horizon):
            r += rec[i]
            k += con[i]
            out.append({"day": i, "recovered_paise": r, "contacts": k})
        return out

    for name in wanted:
        if name == "agent":
            policy_fn = agent_policy
        elif name in BASELINES:
            policy_fn = BASELINES[name][0]
        else:
            console.print(f"[yellow]skipping unknown policy {name!r}[/yellow]")
            continue

        run_id = f"{name}-{seed}-{days}d-{_execution_nonce()}"
        with Session(make_engine()) as session:
            run_batch(
                session,
                run_id=run_id,
                policy_fn=policy_fn,
                policy_name=name,
                days=days,
                seed=seed,
            )
            cards.append(score_run(session, run_id=run_id, policy_name=name))

            buyers = list(session.exec(select(Buyer)).all())
            contacts = list(session.exec(select(ContactLog)).all())
            approvals = [
                a for a in session.exec(select(HumanApproval)).all() if a.run_id == run_id
            ]
            report_obj = audit(
                contacts,
                policy=pol,
                buyer_consent={b.id: b.consent_whatsapp for b in buyers},
                buyer_dnd={b.id: b.dnd for b in buyers},
                approvals=approvals,
            )
            compliance_by_policy[name] = report_obj.as_dict()
            curves_by_policy[name] = _daily_curve(
                list(session.exec(select(Case)).all()), contacts
            )

        console.print(f"[green]scored[/green] {name}")

    path = write_scoreboard(
        cards, REPORTS_DIR, compliance=compliance_by_policy, seed=seed, days=days
    )

    if emit_json:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = REPORTS_DIR / "scoreboard.json"
        json_path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "days": days,
                    "policies": [
                        {
                            **card.as_dict(),
                            "compliance": compliance_by_policy.get(card.policy, {}),
                            "curve": curves_by_policy.get(card.policy, []),
                        }
                        for card in cards
                    ],
                },
                indent=1,
                default=str,
            ),
            encoding="utf-8",
        )
        console.print(f"[dim]scoreboard json: {json_path}[/dim]")

    t = Table(title="Scoreboard", header_style="bold")
    t.add_column("policy")
    t.add_column("net recovery", justify="right")
    t.add_column("recovered", justify="right")
    t.add_column("false chases", justify="right")
    t.add_column("contacts", justify="right")
    t.add_column("violations", justify="right")
    for card in cards:
        t.add_row(
            card.policy,
            format_inr(card.money["net_recovery_paise"]),
            format_inr(card.money["recovered_paise"]),
            str(card.harm["false_chase_contacts"]),
            str(card.harm["contacts_total"]),
            str(compliance_by_policy.get(card.policy, {}).get("violations", "—")),
        )
    console.print(t)
    console.print(f"\nscoreboard: {path}")


@app.command()
def replay(
    case: str = typer.Option(..., "--case", help="Case id, e.g. CASE-0173-0."),
) -> None:
    """Reconstruct a case from the append-only decision log alone."""
    from sqlmodel import Session

    from .audit.replay import render_trace, replay_case
    from .db import make_engine

    with Session(make_engine()) as session:
        result = replay_case(session, case)

    console.print(render_trace(result))


@app.command("export-web")
def export_web(
    days: int = typer.Option(45),
    seed: int = typer.Option(42),
    rerun: bool = typer.Option(
        True,
        "--rerun/--no-rerun",
        help="Re-run the agent policy first, so the snapshot is the agent's own state.",
    ),
) -> None:
    """Freeze the run into `web/src/data/*.json` for the front end.

    The front end reads a snapshot rather than the database. That is deliberate: a live
    reader held the SQLite file open and made `generate` fail silently (BROKE entry 12),
    and a static snapshot also means the site builds and deploys with no Python at all.

    `--rerun` matters. `report` leaves the database holding whichever policy it scored
    last — b3 — so exporting straight after it would show baseline cases on a page that
    says "agent".
    """
    from sqlmodel import Session

    from .db import content_hash, make_engine
    from .eval.web_export import export_web_data
    from .runner import agent_policy, run_batch

    # Hash the generated batch, not the run state. The decision log is append-only and
    # grows with every execution ever made, so hashing it would produce a different value
    # on every export and say nothing about whether the data is reproducible.
    runtime_tables = (
        "case",
        "contact_log",
        "decision_log",
        "human_approval",
        "inbound_event",
        "outbox",
        "run",
        "exception_report",
    )

    engine = make_engine()
    run_id = f"agent-web-{_execution_nonce()}"
    with Session(engine) as session:
        if rerun:
            run_batch(
                session,
                run_id=run_id,
                policy_fn=agent_policy,
                policy_name="agent",
                days=days,
                seed=seed,
            )
            console.print(f"[green]ran[/green] agent as {run_id}")
        written = export_web_data(
            session,
            run_id=run_id if rerun else None,
            db_hash=content_hash(engine, exclude=runtime_tables),
            seed=seed,
            days=days,
        )

    for path in written:
        console.print(f"  web/src/data/{path.name:<18} {path.stat().st_size // 1024:>5} KB")
    console.print(f"\n[green]{len(written)} files written.[/green] Now: cd web && npm run dev")


@app.command("taxonomy")
def show_taxonomy() -> None:
    """Print the reason-code taxonomy the classifier is constrained to."""
    tax = load_taxonomy()
    table = Table(header_style="bold")
    for col in ("code", "family", "verifier", "default", "chaseable", "action"):
        table.add_column(col)
    for code in tax.all_codes:
        rc = tax[code]
        default = {True: "valid", False: "invalid", None: "depends"}[rc.default_valid]
        table.add_row(
            rc.code, rc.family, rc.verifier, default, "yes" if rc.chaseable else "no",
            rc.default_action,
        )
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
