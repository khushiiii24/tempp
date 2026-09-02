"""Ground truth: what is actually recoverable, and how the buyer will actually behave.

**This is the scoreboard, written before the agent exists.** Every behavioural outcome —
whether they pay, after how many contacts, at which seniority, on which channel, whether
they dispute, whether they promise and default — is drawn here, from the seed, and stored
in a table the agent is forbidden to read. The counterparty simulation later just looks
the answer up.

That ordering is the whole defensibility argument. If the outcome were decided at contact
time, or by a language model reading the agent's message, the agent would be grading its
own homework: a more persuasive email would "recover" more money because something
downstream was persuaded, not because a real buyer was. Here a beautifully written chase
to a buyer whose `will_pay_if_chased` is False recovers exactly nothing, which is the
correct answer and an uncomfortable one for a demo. That is the point.

Two invariants hold by construction:

* **A valid deduction has `recoverable_paise == 0` and `will_pay_if_chased == False`.**
  There is no money there. An agent that chases it burns cost and relationship for a
  guaranteed zero — which is exactly what the blanket-dunning baseline does, and exactly
  what the harm metric is built to expose.
* **Behaviour is conditioned on the buyer, not the deduction.** A difficult buyer is
  difficult about everything. This is what makes buyer-level policy (contact caps,
  relationship ratios) meaningful rather than noise.
"""

from __future__ import annotations

from typing import Any

from ..schemas import Buyer, DeductionTruth
from .deductions import PlannedDeduction
from .seed import chance, rng_for, sample_range, weighted_choice, weighted_index

NEVER_PAYS = 99

# Channels a buyer will actually respond on. Email is universal in Indian B2B AR; phone
# reaches the ones who ignore email; WhatsApp only matters where consent exists.
_CHANNEL_SETS: dict[str, list[str]] = {
    "email_only": ["email"],
    "email_call": ["email", "call"],
    "all": ["email", "whatsapp", "call"],
    "call_only": ["call"],
}
_CHANNEL_WEIGHTS = {"email_only": 0.34, "email_call": 0.38, "all": 0.20, "call_only": 0.08}


def build_truth(
    seed: int,
    cfg: dict[str, Any],
    planned: PlannedDeduction,
    buyer: Buyer,
) -> DeductionTruth:
    rng = rng_for(seed, "truth", planned.id)
    beh = cfg["behaviour"]
    tag = buyer.payment_behaviour_tag

    # ---- the money -------------------------------------------------------------
    # Valid deductions carry no recoverable rupees, by definition. This is not a
    # modelling choice; it is what "valid" means.
    recoverable = 0 if planned.is_valid else int(planned.recoverable_paise)

    # ---- will they pay? --------------------------------------------------------
    if recoverable <= 0:
        will_pay = False
        pays_after = NEVER_PAYS
    else:
        will_pay = chance(rng, float(beh["will_pay_if_chased"][tag]))
        pays_after = (
            weighted_index(rng, beh["pays_after_n_contacts_weights"][tag]) + 1
            if will_pay
            else NEVER_PAYS
        )

    # ---- how do they behave? ---------------------------------------------------
    channels = list(_CHANNEL_SETS[weighted_choice(rng, _CHANNEL_WEIGHTS)])
    # A buyer who never consented to WhatsApp cannot respond on it. Leaving it in would
    # let a compliance violation look like a successful recovery.
    if not buyer.consent_whatsapp and "whatsapp" in channels:
        channels.remove("whatsapp")
    if not channels:
        channels = ["email"]

    responds_only_at = weighted_choice(rng, beh["responds_only_at_role_weights"])

    # A dispute is a hard stop for the agent, so it can only arise where there is
    # something to argue about — nobody formally disputes a statutory withholding.
    will_dispute = recoverable > 0 and chance(rng, float(beh["will_dispute_rate"]))
    # Disputing and paying are mutually exclusive; a buyer who escalates to a formal
    # dispute is not also quietly settling.
    if will_dispute:
        will_pay = False
        pays_after = NEVER_PAYS

    promise_then_default = (
        recoverable > 0 and not will_dispute and chance(rng, float(beh["promise_then_default_rate"]))
    )

    return DeductionTruth(
        deduction_id=planned.id,
        true_reason_code=planned.code,
        is_valid=planned.is_valid,
        recoverable_paise=recoverable,
        will_pay_if_chased=will_pay,
        pays_after_n_contacts=pays_after,
        responds_to_channels=channels,
        responds_only_at_role=responds_only_at,
        will_dispute=will_dispute,
        promise_then_default=promise_then_default,
        opt_out=chance(rng, float(beh["opt_out_rate"])),
        latency_days=sample_range(rng, beh["latency_days"]),
        showcase_id=planned.showcase_id,
        notes=_describe(planned, will_pay, pays_after, responds_only_at),
    )


def _describe(planned: PlannedDeduction, will_pay: bool, pays_after: int, role: str) -> str:
    """A one-line human summary, for reading the generated data and for eval output.

    Purely descriptive — nothing in the pipeline parses this.
    """
    if planned.is_valid:
        return f"{planned.code}: legitimate, nothing to recover. Correct action is close/verify."
    if not will_pay:
        return f"{planned.code}: invalid but the buyer will not pay regardless of chasing."
    return (
        f"{planned.code}: invalid and recoverable; pays after {pays_after} contact(s) "
        f"once escalated to {role}."
    )


def recoverable_ceiling(truths: list[DeductionTruth]) -> dict[str, int]:
    """The most any agent could possibly recover, and the shape of the opportunity.

    Reported at the top of the scoreboard so that "recovered ₹X" is always read against
    what was actually available. A recovery rate quoted against total short-paid rather
    than against the recoverable ceiling flatters the agent by counting statutory TDS as a
    missed opportunity.
    """
    total_short = sum(t.recoverable_paise for t in truths)
    reachable = sum(t.recoverable_paise for t in truths if t.will_pay_if_chased)
    valid_money = sum(t.recoverable_paise for t in truths if t.is_valid)
    return {
        "recoverable_paise": total_short,
        "reachable_paise": reachable,
        "unreachable_paise": total_short - reachable,
        "valid_deduction_paise": valid_money,
        "n_deductions": len(truths),
        "n_valid": sum(1 for t in truths if t.is_valid),
        "n_recoverable": sum(1 for t in truths if t.recoverable_paise > 0),
        "n_will_dispute": sum(1 for t in truths if t.will_dispute),
    }
