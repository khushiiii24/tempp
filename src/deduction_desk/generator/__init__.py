"""Generator orchestration: build the whole synthetic world, in truth-first order.

The sequence is the design. Buyers and contracts first, because they constrain what
deductions are even possible. Then invoices. Then the *truth* — the reason code, its
validity and the buyer's pre-committed behaviour. Only then the fixtures that make the
truth checkable, and only then the observable mess: payment amounts, mangled bank
narrations, and remittance advices that may be late, malformed or absent.

Nothing downstream of the truth step is allowed to change it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..clock import add_days, date_str, parse_date
from ..config import FIXTURES_DIR, load_generator_config, load_taxonomy
from ..db import init_db, session_scope
from ..money import Paise
from ..schemas import (
    AdviceFormat,
    Contract,
    Deduction,
    Invoice,
    PaymentEvent,
    RemittanceAdvice,
)
from .buyers import build_buyers
from .contracts import build_contracts
from .deductions import PlannedDeduction, plan_deduction
from .fixtures import write_fixtures
from .invoices import build_invoices, has_rate_error
from .noise import build_narration, make_utr, mangle_invoice_ref, plan_payment_structure
from .remittance import build_advice, stated_reason_for
from .seed import chance, rng_for, sample_range, weighted_choice
from .showcase import reserve_showcases, showcase_summary
from .truth import build_truth, recoverable_ceiling

# Codes that are always feasible on any invoice, used when a drawn code cannot be staged.
_FALLBACK_CODES = ("UNEXPLAINED", "FREIGHT", "BANK_CHARGES")

# How late buyers actually pay, by behaviour tag: days after the due date.
_PAYMENT_DELAY_DAYS = {
    "prompt": [-3, 5],
    "average": [0, 14],
    "slow": [7, 30],
    "difficult": [15, 45],
}


def _assign_codes(
    seed: int,
    cfg: dict[str, Any],
    taxonomy,
    invoice: Invoice,
    contract: Contract,
    buyer_volume_paise: int,
) -> list[PlannedDeduction]:
    """Choose and stage the deduction(s) on one invoice."""
    rng = rng_for(seed, "assign", invoice.id)
    mix = cfg["mix"]
    split = cfg.get("validity_split", {})

    n_components = 2 if chance(rng, float(cfg["batch"]["multi_component_rate"])) else 1

    # An invoice the seller mis-billed almost always draws the buyer's attention to it.
    forced_first = "RATE_DIFFERENCE" if has_rate_error(invoice) and chance(rng, 0.55) else None

    staged: list[PlannedDeduction] = []
    used_codes: set[str] = set()

    for ordinal in range(n_components):
        planned = None
        for _attempt in range(12):
            code = forced_first if (ordinal == 0 and forced_first) else weighted_choice(rng, mix)
            forced_first = None
            if code in used_codes:
                continue
            planned = plan_deduction(
                seed, taxonomy, invoice, contract, code,
                ordinal=ordinal, validity_split=split,
                buyer_volume_paise=buyer_volume_paise,
            )
            if planned is not None:
                break

        if planned is None:
            for code in _FALLBACK_CODES:
                if code in used_codes:
                    continue
                planned = plan_deduction(
                    seed, taxonomy, invoice, contract, code,
                    ordinal=ordinal, validity_split=split,
                )
                if planned is not None:
                    break

        if planned is None:
            continue

        # A multi-component gap must not exceed the invoice; a buyer paying a negative
        # amount is not messiness, it is a broken batch.
        if sum(d.amount_paise for d in staged) + planned.amount_paise >= invoice.total_paise:
            break

        used_codes.add(planned.code)
        staged.append(planned)

    return staged


def generate(
    seed: int = 42,
    n: int | None = None,
    *,
    db_file: Path | None = None,
    fixtures_dir: Path | None = None,
    reset: bool = True,
) -> dict[str, Any]:
    """Build the batch. Returns a report dict; writes the database and the fixtures."""
    cfg = load_generator_config()
    taxonomy = load_taxonomy()
    if n is not None:
        cfg = {**cfg, "batch": {**cfg["batch"], "n_invoices": int(n)}}

    fixtures_dir = fixtures_dir or FIXTURES_DIR

    # ---- entities -------------------------------------------------------------
    buyers = build_buyers(seed, cfg)
    buyer_by_id = {b.id: b for b in buyers}
    contracts = build_contracts(seed, cfg, [b.id for b in buyers])
    contract_by_buyer = {c.buyer_id: c for c in contracts}
    invoices = build_invoices(seed, cfg, [b.id for b in buyers], contract_by_buyer)
    invoice_by_id = {i.id: i for i in invoices}

    # Aggregate purchase value per buyer across the batch. This is the s.194Q test:
    # the section only engages once a buyer's purchases from us pass Rs 50L in the year,
    # and it is a buyer-level fact, not a property of any one invoice.
    buyer_volume: dict[str, int] = {}
    for invoice in invoices:
        buyer_volume[invoice.buyer_id] = (
            buyer_volume.get(invoice.buyer_id, 0) + int(invoice.taxable_paise)
        )

    # ---- truth ----------------------------------------------------------------
    showcase_planned, reserved = reserve_showcases(
        seed, cfg, taxonomy, invoices, buyer_by_id, contract_by_buyer, buyer_volume
    )

    planned_by_invoice: dict[str, list[PlannedDeduction]] = {
        inv_id: [p] for inv_id, p in showcase_planned.items()
    }

    for invoice in invoices:
        if invoice.id in planned_by_invoice:
            continue
        rng = rng_for(seed, "has_deduction", invoice.id)
        if not chance(rng, float(cfg["batch"]["deduction_rate"])):
            continue
        staged = _assign_codes(
            seed, cfg, taxonomy, invoice, contract_by_buyer[invoice.buyer_id],
            buyer_volume.get(invoice.buyer_id, 0),
        )
        if staged:
            planned_by_invoice[invoice.id] = staged

    all_planned = [p for lst in planned_by_invoice.values() for p in lst]
    truths = [
        build_truth(seed, cfg, p, buyer_by_id[p.buyer_id])
        for p in sorted(all_planned, key=lambda x: x.id)
    ]

    # ---- fixtures (a projection of the truth) ---------------------------------
    fixture_counts = write_fixtures(
        seed, cfg, fixtures_dir,
        planned=sorted(all_planned, key=lambda x: x.id),
        invoices=invoice_by_id,
        buyers=buyer_by_id,
        contracts=contract_by_buyer,
    )

    # ---- observable world -----------------------------------------------------
    groups = plan_payment_structure(
        seed, cfg, [i.id for i in invoices], {i.id: i.buyer_id for i in invoices}
    )

    reason_texts: dict[str, str | None] = {
        p.id: stated_reason_for(seed, cfg, p) for p in all_planned
    }

    payments: list[PaymentEvent] = []
    advices: list[RemittanceAdvice] = []
    deduction_rows: list[Deduction] = []

    for gi, group in enumerate(groups):
        gid = f"GRP-{gi:04d}"
        rng = rng_for(seed, "payment", gid)
        buyer = buyer_by_id[group["buyer_id"]]
        members = [invoice_by_id[i] for i in group["invoice_ids"]]

        # Value date: buyers pay around their terms, late in proportion to how difficult
        # they are. Most land at or after the due date, which is what makes an
        # early-payment discount taken at day 45 genuinely unearned.
        latest_due = max(parse_date(m.due_date) for m in members)
        delay = sample_range(rng, _PAYMENT_DELAY_DAYS[buyer.payment_behaviour_tag])
        value_date = date_str(latest_due) if delay == 0 else add_days(date_str(latest_due), delay)

        group_deductions = [
            d for m in members for d in planned_by_invoice.get(m.id, [])
        ]
        gross = sum(m.total_paise for m in members)
        deducted = sum(d.amount_paise for d in group_deductions)
        net = gross - deducted

        refs = [mangle_invoice_ref(rng, m.invoice_no) for m in members]

        if group["kind"] == "split" and net > 0:
            # One invoice, two credits, weeks apart. On day one this is indistinguishable
            # from a short payment — which is precisely the trap `awaiting_settlement`
            # exists to avoid walking into.
            first = int(net * rng.uniform(0.55, 0.75))
            gap = sample_range(rng, cfg["messiness"]["split_gap_days"])
            tranches = [(first, value_date), (net - first, add_days(value_date, gap))]
        else:
            tranches = [(net, value_date)]

        payment_ids: list[str] = []
        for ti, (amount, vdate) in enumerate(tranches):
            utr = make_utr(rng)
            pid = f"PAY-{gi:04d}-{ti}"
            payment_ids.append(pid)
            payments.append(
                PaymentEvent(
                    id=pid,
                    utr=utr,
                    value_date=vdate,
                    amount_paise=Paise(max(0, amount)),
                    narration_raw=build_narration(rng, buyer.name, refs, utr),
                    source="bank_statement",
                    buyer_id_resolved=None,  # the matcher's job, not a given
                )
            )

        # Deduction rows carry the *claimed* reason only. Predicted code, verdict and
        # recoverable amount are all left blank — they are the agent's to fill in.
        for d in group_deductions:
            deduction_rows.append(
                Deduction(
                    id=d.id,
                    invoice_id=d.invoice_id,
                    payment_event_id=payment_ids[-1],
                    amount_paise=Paise(d.amount_paise),
                    claimed_reason_text=reason_texts.get(d.id),
                    created_at=value_date,
                    state="new",
                )
            )

        advice = build_advice(
            seed, cfg,
            advice_id=gid,
            buyer=buyer,
            payment_id=payment_ids[0],
            value_date=value_date,
            invoices=members,
            deductions_by_invoice=planned_by_invoice,
            reason_texts=reason_texts,
            total_paid_paise=net,
        )
        if advice is not None:
            fmt, raw_text, received_at = advice
            advices.append(
                RemittanceAdvice(
                    id=f"ADV-{gi:04d}",
                    buyer_id=buyer.id,
                    received_at=received_at,
                    format=fmt,
                    raw_text=raw_text,
                    parsed=None,
                    links_to_payment=None,  # also the matcher's job
                )
            )

    # ---- persist --------------------------------------------------------------
    engine = init_db(db_file, reset=reset)
    with session_scope(engine) as session:
        for row in (*buyers, *contracts, *invoices, *payments, *advices, *deduction_rows, *truths):
            session.add(row)

    realised_mix: dict[str, int] = {}
    for p in all_planned:
        realised_mix[p.code] = realised_mix.get(p.code, 0) + 1

    return {
        "seed": seed,
        "n_invoices": len(invoices),
        "n_buyers": len(buyers),
        "n_payments": len(payments),
        "n_advices": len(advices),
        "n_advices_absent": len(groups) - len(advices),
        "n_deductions": len(deduction_rows),
        "n_invoices_with_deduction": len(planned_by_invoice),
        "fixtures": fixture_counts,
        "ceiling": recoverable_ceiling(truths),
        "realised_mix": dict(sorted(realised_mix.items())),
        "showcases": showcase_summary(showcase_planned),
        "advice_formats": {
            fmt: sum(1 for a in advices if a.format == fmt)
            for fmt in (AdviceFormat.EMAIL.value, AdviceFormat.PDF_TEXT.value, AdviceFormat.XLSX.value)
        },
    }
