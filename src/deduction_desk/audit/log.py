"""The append-only decision log, and the hash that makes it checkable.

Every stage writes one row per case per decision. The claim the submission makes is that
**any case can be reconstructed from this table alone, with no access to live services** —
`replay.py` proves it by reading nothing else.

Two properties are enforced rather than promised:

* **Append-only at the database level.** `db.py` installs SQLite triggers that abort any
  UPDATE or DELETE on `decision_log`. A convention would be a comment; a trigger survives
  somebody being in a hurry.
* **Inputs are hashed.** Each row records a digest of the observation it acted on, so a
  replay can detect that the world has changed underneath it rather than silently
  reporting a decision that no longer follows from the data.

The sequence number is per-run and monotonic, which is what lets `replay` order a case's
history without relying on timestamps that can collide inside a single simulated day.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from sqlmodel import Session, select

from ..schemas import DecisionLog


def hash_inputs(payload: dict[str, Any]) -> str:
    """A stable digest of whatever the decision was based on.

    Sorted keys so that a cosmetic reordering does not change the hash, and a short digest
    because it is for change-detection, not security.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class DecisionRecorder:
    """Writes decision rows for one run, keeping the sequence monotonic.

    Held for the life of a run rather than constructed per call, because the sequence
    number has to be unique across the whole run and handing that responsibility to each
    call site is how gaps and collisions appear.
    """

    session: Session
    run_id: str
    _seq: int = 0
    _pending: list[DecisionLog] = field(default_factory=list)

    def record(
        self,
        *,
        stage: str,
        ts: str,
        sim_date: str,
        case_id: str | None = None,
        deduction_id: str | None = None,
        observation: dict[str, Any] | None = None,
        hypothesis: dict[str, Any] | None = None,
        policy_rules_fired: list[str] | None = None,
        decision: dict[str, Any] | None = None,
        action_taken: dict[str, Any] | None = None,
        outcome: dict[str, Any] | None = None,
        llm_calls: list[dict[str, Any]] | None = None,
        human_approval: dict[str, Any] | None = None,
    ) -> DecisionLog:
        self._seq += 1
        observation = observation or {}

        row = DecisionLog(
            id=f"DL-{self.run_id}-{self._seq:06d}",
            run_id=self.run_id,
            seq=self._seq,
            ts=ts,
            sim_date=sim_date,
            case_id=case_id,
            deduction_id=deduction_id,
            stage=stage,
            inputs_hash=hash_inputs(observation),
            observation=observation,
            hypothesis=hypothesis or {},
            policy_rules_fired=policy_rules_fired or [],
            decision=decision or {},
            action_taken=action_taken,
            outcome=outcome,
            llm_calls=llm_calls or [],
            human_approval=human_approval,
        )
        self._pending.append(row)
        return row

    def flush(self) -> int:
        """Persist buffered rows.

        Buffered rather than written per call because a 45-day run over 300 cases produces
        thousands of rows, and one INSERT per decision makes the tick loop I/O-bound for no
        benefit. Nothing reads the log mid-run.
        """
        if not self._pending:
            return 0
        for row in self._pending:
            self.session.add(row)
        self.session.commit()
        written = len(self._pending)
        self._pending.clear()
        return written

    @property
    def buffered(self) -> int:
        return len(self._pending)


def case_history(session: Session, case_id: str) -> list[DecisionLog]:
    """Every decision for one case, in order. The unit `replay` works on."""
    rows = session.exec(
        select(DecisionLog).where(DecisionLog.case_id == case_id)
    ).all()
    return sorted(rows, key=lambda r: r.seq)


def run_history(session: Session, run_id: str) -> list[DecisionLog]:
    rows = session.exec(select(DecisionLog).where(DecisionLog.run_id == run_id)).all()
    return sorted(rows, key=lambda r: r.seq)


def summarise(session: Session, run_id: str) -> dict[str, Any]:
    rows = run_history(session, run_id)
    by_stage: dict[str, int] = {}
    rules: dict[str, int] = {}
    for row in rows:
        by_stage[row.stage] = by_stage.get(row.stage, 0) + 1
        for rule in row.policy_rules_fired or []:
            rules[rule] = rules.get(rule, 0) + 1

    return {
        "rows": len(rows),
        "cases": len({r.case_id for r in rows if r.case_id}),
        "by_stage": dict(sorted(by_stage.items())),
        "rules_fired": dict(sorted(rules.items(), key=lambda kv: -kv[1])),
        "llm_calls": sum(len(r.llm_calls or []) for r in rows),
    }
