"""The compliance gate. Every outbound contact passes through here before it exists.

This is a *pre-execution* check: it decides whether an action may be taken, and the
executor refuses to proceed if it says no. It is deliberately **not** the thing that
proves compliance afterwards. `eval/compliance_audit.py` re-derives every violation from
`ContactLog` and `policy.yaml` alone, without calling anything in this module, and the
test asserts it finds zero. If the same function both enforced and verified, "zero
violations across the batch" would be a tautology — the function agreeing with itself —
and a fintech panel would say so within about ten seconds.

Every check returns a *named* rule, passed or failed, and the names flow into
`ContactLog.policy_checks_passed` and `DecisionLog.policy_rules_fired`. "Why was this
contact allowed" has to be answerable from the log without re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..clock import IST, days_between, is_contact_day, is_within_window, parse_iso
from ..config import Policy
from ..schemas import Buyer, Case, Channel, ContactLog


@dataclass
class ComplianceDecision:
    allowed: bool
    checks_passed: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def deny(self, rule: str) -> ComplianceDecision:
        self.allowed = False
        self.violations.append(rule)
        return self

    def allow(self, rule: str) -> ComplianceDecision:
        self.checks_passed.append(rule)
        return self


def _contacts_in_trailing_week(history: list[ContactLog], when: datetime) -> int:
    """Contacts to this buyer in the seven days ending at `when`."""
    cutoff = when.date()
    count = 0
    for entry in history:
        delta = (cutoff - parse_iso(entry.ts).date()).days
        if 0 <= delta < 7:
            count += 1
    return count


def check_contact(
    *,
    buyer: Buyer,
    case: Case,
    channel: str,
    when: datetime,
    buyer_history: list[ContactLog],
    policy: Policy,
) -> ComplianceDecision:
    """May we contact this buyer, on this channel, at this moment?

    Order matters only for readability — every applicable rule is evaluated so that the
    log records the full picture rather than the first objection.
    """
    rules = policy.compliance
    decision = ComplianceDecision(allowed=True)

    # ---- consent and opt-out --------------------------------------------------
    if rules.get("respect_dnd", True) and buyer.dnd:
        decision.deny("compliance.dnd_registered")
    else:
        decision.allow("compliance.dnd_ok")

    if channel in set(rules.get("require_consent_for", [])):
        if channel == Channel.WHATSAPP.value and not buyer.consent_whatsapp:
            decision.deny("compliance.no_consent_for_whatsapp")
        elif channel == Channel.SMS.value and not buyer.consent_whatsapp:
            # SMS consent is modelled by the same flag; a buyer who never opted into
            # messaging has not opted into either channel.
            decision.deny("compliance.no_consent_for_sms")
        else:
            decision.allow(f"compliance.consent_present_for_{channel}")
    else:
        decision.allow(f"compliance.consent_not_required_for_{channel}")

    # ---- timing ---------------------------------------------------------------
    local = when.astimezone(IST)

    if is_contact_day(local.date(), rules["contact_days"]):
        decision.allow("compliance.contact_day_ok")
    else:
        decision.deny("compliance.outside_contact_days")

    window = rules["contact_window_ist"]
    if is_within_window(local, window["start"], window["end"]):
        decision.allow("compliance.contact_window_ok")
    else:
        decision.deny("compliance.outside_contact_window")

    # ---- frequency ------------------------------------------------------------
    if case.contacts_used < int(rules["max_contacts_per_case"]):
        decision.allow("compliance.case_contact_cap_ok")
    else:
        decision.deny("compliance.max_contacts_per_case_reached")

    weekly = _contacts_in_trailing_week(buyer_history, local)
    if weekly < int(rules["max_contacts_per_buyer_per_week"]):
        decision.allow("compliance.buyer_weekly_cap_ok")
    else:
        decision.deny("compliance.max_contacts_per_buyer_per_week_reached")

    if case.last_contact_at:
        gap_hours = (local - parse_iso(case.last_contact_at)).total_seconds() / 3600.0
        if gap_hours >= float(rules["min_gap_hours"]):
            decision.allow("compliance.min_gap_ok")
        else:
            decision.deny("compliance.min_gap_not_elapsed")
    else:
        decision.allow("compliance.first_contact")

    return decision


def channel_for_attempt(attempt_index: int, policy: Policy, buyer: Buyer | None = None) -> str | None:
    """Which channel the ladder specifies for this contact number.

    The ladder escalates by design — email, email, WhatsApp, call — so pressure increases
    gradually rather than opening with a phone call about ₹4,000.

    **Restricted to channels this buyer can lawfully be contacted on.** Walking the raw
    ladder produced a deadlock: at attempt 2 it selects WhatsApp, the compliance gate
    blocks it for want of consent, and a blocked contact never increments `contacts_used`
    — so the case sits on that rung retrying an impermissible channel every day for the
    whole 45-day run. Measured: 136 consent blocks and 315 DND blocks against 33 contacts
    actually sent.

    Returns None when the buyer has no permissible channel at all, which is a reason to
    close the case rather than to keep trying.
    """
    ladder: list[str] = policy.compliance["channel_ladder"]
    if buyer is None:
        return ladder[min(attempt_index, len(ladder) - 1)] if ladder else Channel.EMAIL.value

    permitted = usable_channels(buyer, policy)
    if not permitted:
        return None

    # Preserve the ladder's escalation order, but only over channels that can actually be
    # used, so the rung index counts attempts rather than dead ends.
    ordered = [c for c in ladder if c in permitted] or permitted
    return ordered[min(attempt_index, len(ordered) - 1)]


def next_role(current_role: str, policy: Policy) -> str | None:
    """One step up the escalation ladder, or None at the top.

    One step per tick. Jumping from the AP clerk straight to the account manager over an
    unanswered email is how a routine query becomes a relationship problem.
    """
    ladder: list[str] = policy.compliance["escalation_ladder"]
    if current_role not in ladder:
        return ladder[0] if ladder else None
    index = ladder.index(current_role)
    return ladder[index + 1] if index + 1 < len(ladder) else None


def usable_channels(buyer: Buyer, policy: Policy) -> list[str]:
    """Channels this buyer may lawfully be contacted on, in ladder order."""
    rules = policy.compliance
    needs_consent = set(rules.get("require_consent_for", []))
    out: list[str] = []
    for channel in rules["channel_ladder"]:
        if rules.get("respect_dnd", True) and buyer.dnd:
            continue
        if channel in needs_consent and not buyer.consent_whatsapp:
            continue
        if channel not in out:
            out.append(channel)
    return out


def contains_forbidden_phrase(text: str, policy: Policy) -> list[str]:
    """Forbidden phrases found in a drafted message.

    Used by the draft validator and, independently, by the post-hoc auditor. Matching is
    case-insensitive because a model that writes "LEGAL ACTION" has not found a loophole.
    """
    lowered = (text or "").lower()
    return [p for p in policy.compliance.get("forbidden_phrases", []) if p.lower() in lowered]


def days_since_last_contact(case: Case, when: datetime) -> int | None:
    if not case.last_contact_at:
        return None
    return days_between(case.last_contact_at[:10], when.astimezone(IST).date().isoformat())
