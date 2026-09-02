"""Contracts, rate cards, and the SKU catalogue.

The contract is what makes several deduction types adjudicable at all. Freight is the
clearest case: the identical buyer behaviour — deducting ₹8,000 of transport cost — is
*legitimate* under FOR-destination and *invalid* under ex-works. Nothing in the payment,
the narration or the advice text distinguishes them. Only the contract does, which is
precisely why verification has to be a deterministic lookup and not a language model's
impression.

The same applies to `early_payment_discount_bp` (an unearned discount is only unearned
relative to a stated window) and to the rate card (a rate difference is only a difference
against a contracted price).
"""

from __future__ import annotations

from typing import Any

from ..money import Paise, rupees_to_paise
from ..schemas import Contract, DeliveryTerms
from .seed import chance, pick, rng_for, weighted_choice

# SKU catalogue. Industrial consumables at plausible Indian B2B price points, wide enough
# that a bundled invoice looks like a real order rather than one line repeated.
SKU_CATALOGUE: tuple[tuple[str, str, int], ...] = (
    # (sku, description, list price in paise)
    ("HR-COIL-2MM", "HR coil 2mm", rupees_to_paise("5850.00")),
    ("CR-SHEET-1MM", "CR sheet 1mm", rupees_to_paise("6420.00")),
    ("GI-PIPE-25", "GI pipe 25mm", rupees_to_paise("1180.00")),
    ("PVC-GRAN-K67", "PVC granules K67", rupees_to_paise("9450.00")),
    ("HDPE-GRAN", "HDPE granules", rupees_to_paise("8720.00")),
    ("BRG-6205", "Bearing 6205 ZZ", rupees_to_paise("245.00")),
    ("BRG-6305", "Bearing 6305 2RS", rupees_to_paise("410.00")),
    ("FSTN-M12", "Hex bolt M12 x 60", rupees_to_paise("18.50")),
    ("FSTN-M16", "Hex bolt M16 x 80", rupees_to_paise("34.00")),
    ("LUB-EP90", "Gear oil EP90 20L", rupees_to_paise("4260.00")),
    ("LUB-HYD68", "Hydraulic oil 68 26L", rupees_to_paise("5100.00")),
    ("PNT-EPX-20", "Epoxy primer 20L", rupees_to_paise("7350.00")),
    ("CBL-4C-16", "4C x 16sqmm armoured cable", rupees_to_paise("1290.00")),
    ("CBL-2C-4", "2C x 4sqmm cable", rupees_to_paise("310.00")),
    ("CRT-5PLY", "5-ply carton 600x400", rupees_to_paise("62.00")),
    ("STRP-PET", "PET strapping roll", rupees_to_paise("2180.00")),
    ("CEM-OPC53", "OPC 53 grade cement bag", rupees_to_paise("395.00")),
    ("GLS-4MM", "Float glass 4mm sqm", rupees_to_paise("880.00")),
    ("RBR-SHT-6", "Rubber sheet 6mm sqm", rupees_to_paise("1540.00")),
    ("PPR-KRAFT", "Kraft paper reel", rupees_to_paise("3900.00")),
)

SKUS = tuple(s[0] for s in SKU_CATALOGUE)
SKU_DESCRIPTION = {s[0]: s[1] for s in SKU_CATALOGUE}
SKU_LIST_PRICE = {s[0]: s[2] for s in SKU_CATALOGUE}

# TDS sections a seller of goods and services plausibly sits under, with the rate the
# buyer is expected to apply. `reason_codes.yaml` holds the authoritative rates; this is
# the per-contract expectation the verifier compares against.
TDS_SECTIONS = ("TDS_194C", "TDS_194J", "TDS_194H", "TDS_194Q")
TDS_SECTION_WEIGHTS = {"TDS_194C": 0.50, "TDS_194J": 0.22, "TDS_194H": 0.13, "TDS_194Q": 0.15}

# Legitimate rates per section, headline first. Mirrors `reason_codes.yaml`; kept here as
# a plain tuple because the generator needs to pick one and pin it on the contract.
SECTION_RATES_BP: dict[str, tuple[int, ...]] = {
    "TDS_194C": (200, 100),   # 2% companies, 1% individual/HUF
    "TDS_194J": (1000, 200),  # 10% professional, 2% technical services
    "TDS_194H": (500, 200),
    "TDS_194Q": (10,),        # 0.10%
}


def build_contracts(seed: int, cfg: dict[str, Any], buyer_ids: list[str]) -> list[Contract]:
    bcfg = cfg["buyers"]
    contracts: list[Contract] = []

    for buyer_id in buyer_ids:
        rng = rng_for(seed, "contract", buyer_id)

        for_destination = chance(rng, bcfg["for_destination_rate"])
        delivery = (
            DeliveryTerms.FOR_DESTINATION.value if for_destination else DeliveryTerms.EX_WORKS.value
        )

        has_discount = chance(rng, bcfg["early_discount_rate"])

        # A rate card covering a subset of the catalogue, priced slightly under list —
        # a negotiated contract price is the whole point of having one.
        n_skus = rng.randint(6, 12)
        chosen = sorted(rng.sample(SKUS, n_skus))
        rate_card = {
            sku: int(SKU_LIST_PRICE[sku] * rng.uniform(0.86, 0.98) // 1) for sku in chosen
        }

        section = weighted_choice(rng, TDS_SECTION_WEIGHTS)
        # Pin the applicable rate, not just the section. Which of a section's rates
        # applies depends on the nature of the supply, and the vendor master records it —
        # without it, a rate mismatch is not adjudicable.
        section_rates = SECTION_RATES_BP[section]
        rate_bp = section_rates[0] if len(section_rates) == 1 or chance(rng, 0.75) else section_rates[1]

        contracts.append(
            Contract(
                id=f"CTR-{buyer_id.split('-')[1]}",
                buyer_id=buyer_id,
                delivery_terms=delivery,
                payment_terms_days=pick(rng, (15, 30, 30, 45, 45, 60, 90)),
                early_payment_discount_bp=pick(rng, (100, 150, 200, 200, 250)) if has_discount else 0,
                early_payment_window_days=pick(rng, (7, 10, 10, 15)) if has_discount else 0,
                tds_section_expected=section,
                tds_rate_expected_bp=rate_bp,
                tcs_applicable=chance(rng, bcfg["tcs_applicable_rate"]),
                freight_borne_by="seller" if for_destination else "buyer",
                rate_card=rate_card,
            )
        )

    return contracts


def contract_summary_for_prompt(contract: Contract) -> str:
    """A compact contract rendering for the classifier prompt.

    Compact because output tokens are cheap to *read* but the prompt still has to fit the
    context window alongside the advice text and the buyer's deduction history. Only the
    fields that change a classification are included.
    """
    parts = [
        f"delivery_terms={contract.delivery_terms}",
        f"freight_borne_by={contract.freight_borne_by}",
        f"payment_terms_days={contract.payment_terms_days}",
        f"expected_tds_section={contract.tds_section_expected}",
        f"tcs_applicable={'yes' if contract.tcs_applicable else 'no'}",
    ]
    if contract.early_payment_discount_bp:
        parts.append(
            f"early_payment_discount={contract.early_payment_discount_bp / 100:.2f}%"
            f" within {contract.early_payment_window_days}d"
        )
    else:
        parts.append("early_payment_discount=none")
    return ", ".join(parts)


def contracted_rate(contract: Contract, sku: str) -> Paise | None:
    value = contract.rate_card.get(sku)
    return Paise(int(value)) if value is not None else None
