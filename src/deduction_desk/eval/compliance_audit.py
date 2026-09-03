"""Independent compliance audit: re-derive every violation from the logs alone.

**This module must never import `policy.compliance`.** That is the entire point of it.

The compliance gate in `policy/compliance.py` decides whether a contact may go out. If the
same code then verified that no violations occurred, "zero compliance violations across the
batch" would mean nothing more than that a function agrees with itself — and a bug in the
gate would be invisible to exactly the check meant to catch it. A fintech panel spots that
in about ten seconds.

So this reads `ContactLog` and `policy.yaml`, reimplements each rule from the config, and
reports what it finds. Two independent implementations agreeing is evidence; one
implementation asserting is not.

It also audits things the gate never sees, because they happen after it: forbidden phrases
in the message that was actually sent, and whether any credit hold was executed rather than
queued.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

from ..clock import IST, parse_iso
from ..config import Policy
from ..schemas import ContactLog, HumanApproval

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass
class Violation:
    rule: str
    contact_id: str
    case_id: str
    buyer_id: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "contact_id": self.contact_id,
            "case_id": self.case_id,
            "buyer_id": self.buyer_id,
            "detail": self.detail,
        }


@dataclass
class AuditReport:
    violations: list[Violation] = field(default_factory=list)
    contacts_examined: int = 0
    approvals_examined: int = 0

    @property
    def clean(self) -> bool:
        return not self.violations

    def by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for v in self.violations:
            counts[v.rule] = counts.get(v.rule, 0) + 1
        return dict(sorted(counts.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "contacts_examined": self.contacts_examined,
            "approvals_examined": self.approvals_examined,
            "violations": len(self.violations),
            "by_rule": self.by_rule(),
            "examples": [v.as_dict() for v in self.violations[:10]],
        }


def _within_window(when: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    """Reimplemented here on purpose — see the module docstring."""
    local = when.astimezone(IST).time()
    sh, sm = (int(x) for x in start_hhmm.split(":"))
    eh, em = (int(x) for x in end_hhmm.split(":"))
    return time(sh, sm) <= local <= time(eh, em)


def audit(
    contacts: list[ContactLog],
    *,
    policy: Policy,
    buyer_consent: dict[str, bool],
    buyer_dnd: dict[str, bool],
    approvals: list[HumanApproval] | None = None,
) -> AuditReport:
    """Re-derive violations from the contact log and the policy file."""
    rules = policy.compliance
    report = AuditReport(contacts_examined=len(contacts))

    window = rules["contact_window_ist"]
    allowed_days = {d.lower() for d in rules["contact_days"]}
    needs_consent = set(rules.get("require_consent_for", []))
    forbidden = [p.lower() for p in rules.get("forbidden_phrases", [])]
    max_per_case = int(rules["max_contacts_per_case"])
    max_per_buyer_week = int(rules["max_contacts_per_buyer_per_week"])
    min_gap_hours = float(rules["min_gap_hours"])

    per_case: dict[str, list[ContactLog]] = defaultdict(list)
    per_buyer: dict[str, list[ContactLog]] = defaultdict(list)

    for contact in sorted(contacts, key=lambda c: (c.ts, c.id)):
        when = parse_iso(contact.ts)
        per_case[contact.case_id].append(contact)
        per_buyer[contact.buyer_id].append(contact)

        # -- timing ------------------------------------------------------------------
        if WEEKDAYS[when.astimezone(IST).weekday()] not in allowed_days:
            report.violations.append(
                Violation(
                    "contact_outside_permitted_days", contact.id, contact.case_id,
                    contact.buyer_id, f"sent on {WEEKDAYS[when.astimezone(IST).weekday()]}",
                )
            )

        if not _within_window(when, window["start"], window["end"]):
            report.violations.append(
                Violation(
                    "contact_outside_window", contact.id, contact.case_id, contact.buyer_id,
                    f"sent at {when.astimezone(IST).strftime('%H:%M')} IST",
                )
            )

        # -- consent and opt-out ------------------------------------------------------
        if rules.get("respect_dnd", True) and buyer_dnd.get(contact.buyer_id):
            report.violations.append(
                Violation(
                    "contacted_dnd_buyer", contact.id, contact.case_id, contact.buyer_id,
                    "buyer is on DND",
                )
            )

        if contact.channel in needs_consent and not buyer_consent.get(contact.buyer_id):
            report.violations.append(
                Violation(
                    f"no_consent_for_{contact.channel}", contact.id, contact.case_id,
                    contact.buyer_id, f"{contact.channel} without recorded consent",
                )
            )

        # -- message content ----------------------------------------------------------
        text = f"{contact.subject}\n{contact.body}".lower()
        for phrase in forbidden:
            if phrase in text:
                report.violations.append(
                    Violation(
                        "forbidden_phrase_in_message", contact.id, contact.case_id,
                        contact.buyer_id, f"contains {phrase!r}",
                    )
                )

    # -- frequency ---------------------------------------------------------------------
    for case_id, case_contacts in sorted(per_case.items()):
        if len(case_contacts) > max_per_case:
            last = case_contacts[-1]
            report.violations.append(
                Violation(
                    "exceeded_max_contacts_per_case", last.id, case_id, last.buyer_id,
                    f"{len(case_contacts)} contacts, cap is {max_per_case}",
                )
            )

        ordered = sorted(case_contacts, key=lambda c: c.ts)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            gap = (parse_iso(later.ts) - parse_iso(earlier.ts)).total_seconds() / 3600
            if gap < min_gap_hours:
                report.violations.append(
                    Violation(
                        "min_gap_not_respected", later.id, case_id, later.buyer_id,
                        f"{gap:.1f}h since the previous contact, minimum is {min_gap_hours}h",
                    )
                )

    for buyer_id, buyer_contacts in sorted(per_buyer.items()):
        ordered = sorted(buyer_contacts, key=lambda c: c.ts)
        for index, anchor in enumerate(ordered):
            anchor_date = parse_iso(anchor.ts).date()
            window_count = sum(
                1
                for other in ordered[index:]
                if 0 <= (parse_iso(other.ts).date() - anchor_date).days < 7
            )
            if window_count > max_per_buyer_week:
                report.violations.append(
                    Violation(
                        "exceeded_max_contacts_per_buyer_per_week", anchor.id,
                        anchor.case_id, buyer_id,
                        f"{window_count} contacts in the 7 days from {anchor_date}, "
                        f"cap is {max_per_buyer_week}",
                    )
                )
                break

    # -- credit holds ---------------------------------------------------------------
    approvals = approvals or []
    report.approvals_examined = len(approvals)
    if rules.get("credit_hold_requires_human", True):
        for approval in approvals:
            if approval.action_type == "propose_credit_hold" and approval.approved:
                report.violations.append(
                    Violation(
                        "credit_hold_executed_without_human", approval.id, approval.case_id,
                        "", "a credit hold was marked approved inside the run",
                    )
                )

    return report
