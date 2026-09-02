"""Buyers: the counterparties, and the behavioural priors that decide what is recoverable.

The buyer is where the economics of the whole simulation are anchored. Two fields do most
of the work:

* `payment_behaviour_tag` conditions every behavioural draw in `truth.py` — whether they
  pay when chased, after how many contacts, and at which seniority they start responding.
* `relationship_value_paise` is what makes `stop_if_relationship_value_ratio_exceeds`
  bite. A ₹1,200 shortfall from a ₹3Cr account is not worth an email, and an agent that
  cannot represent that has no way to be *bounded* rather than merely effective.

Names, GSTINs and PANs are synthetic but structurally valid, because the classifier and
the fixtures both key off them and a malformed PAN would fail for the wrong reason.
"""

from __future__ import annotations

from typing import Any

from ..money import Paise
from ..schemas import Buyer, Channel, Segment
from .seed import chance, pick, rng_for, sample_range, weighted_choice

# Name components. Deliberately mundane — these read like an AR ledger, not a brand deck.
_PREFIX = (
    "Shree", "Sri", "Maa", "Jai", "Nav", "Om", "Raj", "Ganesh", "Krishna", "Lakshmi",
    "Bharat", "Hind", "Deccan", "Konkan", "Malabar", "Aravalli", "Sutlej", "Godavari",
    "Vindhya", "Coromandel", "Anand", "Vikram", "Surya", "Chetak", "Pioneer",
)
_CORE = (
    "Steel", "Poly", "Agro", "Pharma", "Textile", "Auto", "Electro", "Chem", "Packaging",
    "Ceramics", "Cables", "Fasteners", "Bearings", "Lubricants", "Paints", "Glass",
    "Plastics", "Foods", "Beverages", "Cement", "Timber", "Paper", "Rubber", "Optics",
)
_SUFFIX = (
    "Industries Pvt Ltd", "Enterprises", "Traders", "Distributors Pvt Ltd",
    "& Sons", "Corporation", "Trading Co", "Agencies", "Marketing Pvt Ltd",
    "Udyog Pvt Ltd", "Impex", "Solutions Pvt Ltd",
)

# GSTIN state codes for a plausible spread of Indian metros and industrial belts.
_STATE_CODES = ("27", "29", "33", "07", "24", "06", "19", "36", "23", "09", "32", "08")

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _make_pan(rng, entity_letter: str = "C") -> str:
    """Structurally valid PAN: 5 letters, 4 digits, 1 letter.

    The fourth character encodes entity type — 'C' for company — and the fixtures key
    26AS rows off the PAN, so the shape has to be right even though the value is fake.
    """
    head = "".join(pick(rng, _LETTERS) for _ in range(3))
    surname_initial = pick(rng, _LETTERS)
    digits = f"{rng.randrange(0, 10000):04d}"
    check = pick(rng, _LETTERS)
    return f"{head}{entity_letter}{surname_initial}{digits}{check}"


def _make_gstin(rng, pan: str) -> str:
    """15-char GSTIN: state code + PAN + entity number + 'Z' + checksum char."""
    state = pick(rng, _STATE_CODES)
    entity = str(rng.randint(1, 9))
    check = pick(rng, _LETTERS + "0123456789")
    return f"{state}{pan}{entity}Z{check}"


def _slug(name: str) -> str:
    keep = [ch.lower() for ch in name if ch.isalnum() or ch == " "]
    words = "".join(keep).split()
    return "".join(words[:2]) if words else "buyer"


def build_buyers(seed: int, cfg: dict[str, Any]) -> list[Buyer]:
    n = int(cfg["batch"]["n_buyers"])
    bcfg = cfg["buyers"]
    buyers: list[Buyer] = []
    used_names: set[str] = set()

    for i in range(n):
        bid = f"BUY-{i:04d}"
        rng = rng_for(seed, "buyer", bid)

        # Distinct names matter: the fuzzy matcher works on bank narration, and two buyers
        # sharing a name would make a genuinely ambiguous match look like a matcher bug.
        for _ in range(50):
            name = f"{pick(rng, _PREFIX)} {pick(rng, _CORE)} {pick(rng, _SUFFIX)}"
            if name not in used_names:
                break
        used_names.add(name)

        segment = weighted_choice(rng, bcfg["segment_weights"])
        behaviour = weighted_choice(rng, bcfg["behaviour_weights"])
        relationship = sample_range(rng, bcfg["relationship_value_paise"][segment])

        pan = _make_pan(rng)
        domain = f"{_slug(name)}.co.in"

        # Enterprises answer email; smaller distributors are reachable on WhatsApp. This
        # only sets a preference — the policy engine still has to check consent and DND.
        preferred = (
            Channel.EMAIL.value
            if segment == Segment.ENTERPRISE.value
            else weighted_choice(rng, {"email": 0.6, "whatsapp": 0.4})
        )

        buyers.append(
            Buyer(
                id=bid,
                name=name,
                gstin=_make_gstin(rng, pan),
                pan=pan,
                segment=segment,
                payment_behaviour_tag=behaviour,
                # Credit limit tracks relationship value but is not the same number; it is
                # what a credit hold would actually block.
                credit_limit_paise=Paise(int(relationship * rng.uniform(0.08, 0.25))),
                relationship_value_paise=Paise(relationship),
                preferred_channel=preferred,
                contact_email=f"accounts@{domain}",
                contact_phone=f"+9198{rng.randrange(10**8):08d}",
                ap_manager_email=f"ap.manager@{domain}",
                procurement_email=f"procurement@{domain}",
                account_manager_email=f"km.{_slug(name)[:8]}@ourcompany.example",
                consent_whatsapp=chance(rng, bcfg["consent_whatsapp_rate"]),
                dnd=chance(rng, bcfg["dnd_rate"]),
            )
        )

    return buyers


def contact_for_role(buyer: Buyer, role: str) -> str:
    """Resolve the escalation ladder's role to an address.

    Centralised so the executor cannot accidentally email the AP clerk when policy said
    escalate to procurement — that would be a compliance violation the ladder was
    specifically designed to prevent.
    """
    return {
        "ap_clerk": buyer.contact_email,
        "ap_manager": buyer.ap_manager_email,
        "procurement": buyer.procurement_email,
        "account_manager": buyer.account_manager_email,
    }.get(role, buyer.contact_email)
