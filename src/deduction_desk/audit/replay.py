"""Reconstruct a case from the decision log alone.

The submission's audit claim is that any case can be rebuilt with **zero access to live
services** — no fixtures, no model, no invoice table. This module is what makes that
checkable rather than asserted, so it reads `decision_log` and nothing else.

That constraint is load-bearing, not stylistic. A replay that consulted the invoice table
would still produce a convincing trace after the log had lost something, and the gap would
never surface. `tests/test_phase5_execution.py` drops every other table and asserts this
still works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlmodel import Session, select

from ..money import format_inr
from ..schemas import DecisionLog


@dataclass
class ReplayStep:
    seq: int
    sim_date: str
    stage: str
    rules_fired: list[str] = field(default_factory=list)
    action: str = ""
    reason: str = ""
    executed: bool | None = None
    detail: str = ""
    observation: dict[str, Any] = field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    human_approval: dict[str, Any] | None = None
    next_state: str = ""


@dataclass
class ReplayResult:
    case_id: str
    steps: list[ReplayStep] = field(default_factory=list)
    found: bool = True

    @property
    def rules_fired(self) -> list[str]:
        seen: list[str] = []
        for step in self.steps:
            for rule in step.rules_fired:
                if rule not in seen:
                    seen.append(rule)
        return seen

    @property
    def contacts_made(self) -> int:
        return sum(
            1
            for s in self.steps
            if s.executed and s.action in {"chase", "escalate_role", "request_document"}
        )

    @property
    def final_state(self) -> str:
        """Where the case ended up.

        An inbound event records the resulting state directly; a decision records the
        state it moved the case *to*. Checking both means a case that was never contacted
        still reports its outcome rather than 'unknown'.
        """
        for step in reversed(self.steps):
            state = step.observation.get("case_state") or step.next_state
            if state:
                return str(state)
        return "unknown"


def replay_case(session: Session, case_id: str) -> ReplayResult:
    """Rebuild one case's history. Reads `decision_log` and nothing else."""
    rows = session.exec(select(DecisionLog).where(DecisionLog.case_id == case_id)).all()
    if not rows:
        return ReplayResult(case_id=case_id, found=False)

    # The log accumulates every execution. Show the most recent one rather than
    # concatenating several runs into a single incoherent history.
    latest_run = max(r.run_id for r in rows)
    rows = [r for r in rows if r.run_id == latest_run]

    steps: list[ReplayStep] = []
    for row in sorted(rows, key=lambda r: r.seq):
        decision = row.decision or {}
        action_taken = row.action_taken or {}
        steps.append(
            ReplayStep(
                seq=row.seq,
                sim_date=row.sim_date,
                stage=row.stage,
                rules_fired=list(row.policy_rules_fired or []),
                action=str(decision.get("type") or ""),
                reason=str(decision.get("reason") or ""),
                executed=action_taken.get("executed"),
                detail=str(action_taken.get("detail") or ""),
                observation={**(row.observation or {}), **(row.outcome or {})},
                llm_calls=list(row.llm_calls or []),
                human_approval=row.human_approval,
                next_state=str(decision.get("next_state") or ""),
            )
        )

    return ReplayResult(case_id=case_id, steps=steps)


def render_trace(result: ReplayResult) -> str:
    """A human-readable trace, for `replay --case CASE-0173`."""
    if not result.found:
        return f"No decision log found for {result.case_id}."

    lines = [f"# Replay: {result.case_id}", ""]
    lines.append(
        f"{len(result.steps)} decision(s), {result.contacts_made} contact(s), "
        f"final state `{result.final_state}`."
    )
    lines.append("")
    lines.append(
        "_Reconstructed from the append-only decision log alone — no fixtures, no model, "
        "no other table._"
    )
    lines.append("")

    for step in result.steps:
        header = f"## {step.sim_date} — {step.stage} (#{step.seq})"
        lines.append(header)
        if step.action:
            executed = (
                "executed" if step.executed else "not executed" if step.executed is False else ""
            )
            lines.append(f"**Action:** `{step.action}` {executed}".rstrip())
        if step.reason:
            lines.append(f"**Why:** {step.reason}")
        if step.rules_fired:
            lines.append("**Rules fired:** " + ", ".join(f"`{r}`" for r in step.rules_fired))
        if step.detail:
            lines.append(f"**Detail:** {step.detail}")

        obs = step.observation
        if obs:
            interesting = {
                k: v
                for k, v in obs.items()
                if k in {"verdict", "recoverable_paise", "contacts_used", "role", "kind",
                         "amount_paise", "case_state", "days_past_due"}
            }
            if interesting:
                rendered = ", ".join(
                    f"{k}={format_inr(v) if k.endswith('_paise') else v}"
                    for k, v in sorted(interesting.items())
                )
                lines.append(f"**Observed:** {rendered}")

        for call in step.llm_calls:
            if not call:
                continue
            lines.append(
                f"**LLM:** `{call.get('task')}` on `{call.get('model')}` "
                f"prompt `{str(call.get('prompt_hash'))[:12]}` "
                f"{'(cached)' if call.get('cached') else '(live)'}"
            )

        if step.human_approval:
            lines.append(f"**Human approval:** {step.human_approval}")

        lines.append("")

    return "\n".join(lines)
