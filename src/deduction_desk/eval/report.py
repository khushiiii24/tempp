"""The scoreboard: A versus B0–B3, side by side, on the same batch and clock.

Ordering is deliberate. **Net recovery leads, harm follows immediately** — before accuracy,
before operations. A reader who stops after two rows should still have seen the number that
makes the case and the number that qualifies it, because a system that recovers well by
writing to everyone is not a system anyone should deploy.

`false_chase_contacts` is the row the whole design exists to move: letters sent about
deductions ground truth says were legitimate, usually statutory tax withholding the buyer
was obliged to make. Blanket dunning cannot avoid them; it does not know which is which.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..money import format_inr
from .harness import Scorecard


def _pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


def _money(paise: int | None) -> str:
    return format_inr(paise) if paise is not None else "—"


def _num(value: Any) -> str:
    return "—" if value is None else str(value)


def render_scoreboard(
    cards: list[Scorecard],
    *,
    compliance: dict[str, dict[str, Any]] | None = None,
    seed: int = 42,
    days: int = 45,
) -> str:
    """Markdown scoreboard for `reports/<run>/scoreboard.md`."""
    compliance = compliance or {}
    if not cards:
        return "# Scoreboard\n\nNo runs to compare.\n"

    names = [c.policy for c in cards]
    header = "| metric | " + " | ".join(f"**{n}**" for n in names) + " |"
    divider = "|---|" + "|".join(["---:"] * len(names)) + "|"

    def row(label: str, values: list[str]) -> str:
        return f"| {label} | " + " | ".join(values) + " |"

    lines: list[str] = []
    lines.append("# Scoreboard")
    lines.append("")
    lines.append(
        f"Same batch, same {days}-day clock, same cost model, same counterparty. "
        f"Seed {seed}. Every policy runs through the identical tick loop; only the "
        f"decision function differs."
    )
    lines.append("")

    ceiling = cards[0].money["reachable_ceiling_paise"]
    total_short = cards[0].money["total_short_paid_paise"]
    recoverable = cards[0].money["recoverable_paise"]
    lines.append(
        f"**The opportunity.** {_money(total_short)} was short-paid, of which "
        f"{_money(recoverable)} is genuinely recoverable and {_money(ceiling)} would "
        f"actually be paid if chased correctly. Recovery is quoted against that reachable "
        f"ceiling — measuring it against everything deducted would count statutory tax "
        f"withholding as a missed opportunity."
    )
    lines.append("")

    # ---- money ---------------------------------------------------------------------
    lines.append("## Money")
    lines.append("")
    lines.append(header)
    lines.append(divider)
    lines.append(row("**Net recovery** (recovered − cost)",
                     [f"**{_money(c.money['net_recovery_paise'])}**" for c in cards]))
    lines.append(row("Recovered", [_money(c.money["recovered_paise"]) for c in cards]))
    lines.append(row("Recovery vs reachable ceiling",
                     [_pct(c.money["recovery_rate_vs_ceiling"]) for c in cards]))
    lines.append(row("Money correctly *not* chased",
                     [_money(c.money["correctly_closed_valid_paise"]) for c in cards]))
    lines.append(row("Reachable money escalated to a human",
                     [_money(c.money.get("queued_reachable_paise")) for c in cards]))
    lines.append(row("**Addressed** (recovered + escalated)",
                     [f"**{_money(c.money.get('addressed_paise'))}**" for c in cards]))
    lines.append(row("Recoverable money abandoned",
                     [_money(c.money["wrongly_written_off_paise"]) for c in cards]))
    lines.append(row("Intervention cost", [_money(c.money["cost_paise"]) for c in cards]))
    lines.append(row("₹ recovered per ₹ spent",
                     [_num(c.money["rupees_per_rupee_spent"]) for c in cards]))
    lines.append("")
    lines.append(
        "_**Escalated** is money the agent handed to a person — a classifier abstention, a "
        "claim above the human-review threshold, or a buyer with no lawful contact channel. "
        "It is listed separately and never added to \"recovered\", because the agent did not "
        "collect it and a human still has to. Scoring it as zero, though, penalises the "
        "system for every correct hand-off and flatters a baseline that simply never defers._"
    )
    lines.append("")

    # ---- harm ----------------------------------------------------------------------
    lines.append("## Harm")
    lines.append("")
    lines.append(
        "_Most submissions do not report this section at all. A false chase is a letter "
        "sent about a deduction that was legitimate — usually statutory tax withholding "
        "the buyer was legally required to make. It recovers nothing and costs goodwill._"
    )
    lines.append("")
    lines.append(header)
    lines.append(divider)
    lines.append(row("**False chases** (contacts about valid deductions)",
                     [f"**{_num(c.harm['false_chase_contacts'])}**" for c in cards]))
    lines.append(row("Customers wrongly contacted",
                     [_num(c.harm["false_chase_cases"]) for c in cards]))
    lines.append(row("Total contacts", [_num(c.harm["contacts_total"]) for c in cards]))
    lines.append(row("Escalations to senior roles",
                     [_num(c.harm["escalations_to_senior_roles"]) for c in cards]))
    lines.append(row("Credit holds proposed",
                     [_num(c.harm["credit_holds_proposed"]) for c in cards]))
    lines.append(row("Credit holds **executed**",
                     [_num(c.harm["credit_holds_executed"]) for c in cards]))

    if compliance:
        lines.append(row("Compliance violations (independent audit)",
                         [_num(compliance.get(c.policy, {}).get("violations", "—"))
                          for c in cards]))
    lines.append("")

    # ---- accuracy -------------------------------------------------------------------
    lines.append("## Accuracy")
    lines.append("")
    lines.append(header)
    lines.append(divider)
    lines.append(row("Verdict accuracy (chase / do not chase)",
                     [_pct(c.accuracy["verdict_accuracy"]) for c in cards]))
    lines.append(row("Reason-code accuracy (answered only)",
                     [_pct(c.accuracy["code_accuracy_answered"]) for c in cards]))
    lines.append(row("Abstention rate", [_pct(c.accuracy["abstention_rate"]) for c in cards]))
    lines.append("")

    # ---- operations -------------------------------------------------------------------
    lines.append("## Operations")
    lines.append("")
    lines.append(header)
    lines.append(divider)
    lines.append(row("Cases", [_num(c.operations["cases"]) for c in cards]))
    lines.append(row("Auto-resolved, no human touch",
                     [_pct(c.operations["auto_resolved_rate"]) for c in cards]))
    lines.append(row("Human queue", [_num(c.operations["human_queue"]) for c in cards]))
    lines.append(row("Messages queued", [_num(c.operations["messages_queued"]) for c in cards]))
    lines.append(row("Messages actually sent",
                     [_num(c.operations["messages_sent_for_real"]) for c in cards]))
    lines.append(row("Mean days to resolution",
                     [_num(c.operations["mean_days_to_resolution"]) for c in cards]))
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_Every outbound message is queued with `dry_run=True`. \"Messages actually sent\" "
        "is 0 by construction and stays 0 unless both `--live` and an environment variable "
        "are set._"
    )

    return "\n".join(lines)


def write_scoreboard(
    cards: list[Scorecard],
    out_dir: Path,
    *,
    compliance: dict[str, dict[str, Any]] | None = None,
    seed: int = 42,
    days: int = 45,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scoreboard.md"
    path.write_text(
        render_scoreboard(cards, compliance=compliance, seed=seed, days=days),
        encoding="utf-8",
    )
    return path
