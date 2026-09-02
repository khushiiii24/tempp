"""Remittance advices: the unstructured text the LLM parser actually has to survive.

Written from templates with seeded slot-filling — **never by a language model**. The
generator must be byte-reproducible and must run offline, and a batch whose input data
was itself sampled from a model would make "same seed, same scoreboard" impossible to
guarantee.

The text is where the AI earns its place. Three families of difficulty:

* **Register.** Indian AP correspondence is telegraphic and code-switched:
  `"short paying 7.6k agst Mar invs, damaged carton as per mail"`, `"less: 194Q 0.1%"`,
  `"bal amt after CN adj"`. A regex over `TDS|freight|scheme` gets the easy half and
  silently mislabels the rest.
* **Format.** The same content arrives as email prose, as text extracted from a PDF with
  the columns run together, or as a spreadsheet with merged headers and stray blank rows.
* **Unreliability.** A quarter of the time there is no advice at all. Some arrive days
  after the credit. Some state a reason that *contradicts the arithmetic* — the buyer
  writes "TDS @ 2%" and deducts 5%. That last case is the highest-value test in the batch,
  because the stated reason and the computed rate disagree and only one of them is
  evidence. The correct behaviour is to trust the arithmetic and say so.
"""

from __future__ import annotations

from typing import Any

from ..clock import add_days
from ..money import paise_to_rupees_str
from ..schemas import AdviceFormat, Buyer, Invoice
from .deductions import PlannedDeduction
from .seed import chance, pick, rng_for, sample_range, weighted_choice

# Per-code phrasings. Each tuple mixes clean labels with telegraphic and Hinglish forms,
# because a real AP department contains both a careful clerk and a hurried one.
REASON_PHRASES: dict[str, tuple[str, ...]] = {
    "TDS_194C": ("TDS 194C @2%", "less TDS u/s 194C", "TDS deducted as per 194C", "tds 194c", "less: TDS contractor"),
    "TDS_194J": ("TDS 194J @10%", "less TDS professional fees", "TDS u/s 194J", "tds prof 194j"),
    "TDS_194H": ("TDS 194H commission", "less TDS comm @5%", "TDS 194H"),
    "TDS_194Q": ("less: 194Q 0.1%", "TDS 194Q on purchase", "194q ded", "TDS u/s 194Q @0.1%"),
    "GST_TDS": ("GST TDS 2% u/s 51", "gst tds deducted", "less GST TDS"),
    "TDS_RATE_MISMATCH": ("TDS deducted as applicable", "less TDS as per our records", "tds ded"),
    "TCS_194Q_OVERLAP": ("less: 194Q 0.1%", "TDS 194Q applicable on us", "194q as per sec 194Q"),
    "FREIGHT": (
        "freight as per our terms", "frt deducted", "less transport charges",
        "freight paid by us, recovering", "lorry frt adj",
    ),
    "SCHEME_REBATE": (
        "scheme claim Q4 adj", "QPS incentive adjusted", "scheme rebate as per circular",
        "trade scheme adj", "qtrly scheme claim",
    ),
    "CREDIT_NOTE_OFFSET": (
        "bal amt after CN adj", "less CN {cn}", "adjusted against credit note {cn}",
        "cn adj", "net of CN",
    ),
    "DAMAGE_SHORTAGE": (
        "damaged carton as per mail", "short recd qty", "shortage in delivery",
        "material short received", "damage claim as per GRN",
    ),
    "QUALITY_REJECTION": (
        "qty rejected on quality", "material rejected", "QC rejection adj", "rejected lot deduction",
    ),
    "DEBIT_NOTE_BUYER": ("as per our DN {dn}", "debit note raised", "DN adjusted"),
    "RATE_DIFFERENCE": (
        "rate diff as per PO", "rate difference adj", "billed above contracted rate",
        "rate variance",
    ),
    "UNEARNED_DISCOUNT": ("cash disc availed", "less 2% CD", "early payment discount", "disc taken"),
    "BANK_CHARGES": ("bank charges", "less NEFT chgs", "remittance charges"),
    "ROUNDING": ("round off", "r/o", "rounding adj"),
    "DUPLICATE_CLAIM": (
        "already paid vide {utr}", "duplicate invoice, paid earlier", "this inv already settled",
    ),
    "UNEXPLAINED": (),  # deliberately empty: the whole point is that nothing is said
}

_EMAIL_OPENERS = (
    "Please find below the payment details for your invoices.",
    "PFA payment advice for the below invoices.",
    "Payment released today. Details below.",
    "Dear Sir/Madam,\n\nWe have released the following payment.",
    "Kindly note payment made as per below breakup.",
)

_EMAIL_CLOSERS = (
    "Regards,\nAccounts Payable",
    "Thanks & regards,\nAP Team",
    "Regards,\nFinance Dept",
    "Rgds,\nAccounts",
)


def _phrase_for(rng, planned: PlannedDeduction) -> str:
    """Pick a stated reason for one deduction, filling any template slots."""
    options = REASON_PHRASES.get(planned.code, ())
    if not options:
        return ""
    phrase = pick(rng, options)
    return (
        phrase.replace("{cn}", str(planned.extra.get("credit_note_no", "CN/0000")))
        .replace("{dn}", str(planned.extra.get("debit_note_no") or "DN/000"))
        .replace("{utr}", str(planned.extra.get("asserted_utr", "UTR")))
    )


def stated_reason_for(
    seed: int, cfg: dict[str, Any], planned: PlannedDeduction
) -> str | None:
    """The reason text the buyer claims, or None when they said nothing.

    Two distortions applied here, both configured:

    * `stated_reason_absent_rate` — the advice exists but the reason column is blank.
    * `contradicting_reason_rate` — the stated reason disagrees with the arithmetic.
      For a rate mismatch this is automatic: the buyer cites the section they *think*
      they applied while the numbers show a different rate entirely.
    """
    mess = cfg["messiness"]
    rng = rng_for(seed, "stated_reason", planned.id)

    if planned.code == "UNEXPLAINED":
        return None
    if chance(rng, float(mess["stated_reason_absent_rate"])):
        return None

    # A rate mismatch is a contradiction by construction: state the correct-looking rate,
    # having deducted a different one.
    if planned.code == "TDS_RATE_MISMATCH":
        section = planned.extra.get("section", "194C")
        correct_pct = planned.extra.get("correct_rate_bp", 200) / 100
        if chance(rng, 0.72):
            return f"TDS {section} @{correct_pct:.0f}%"
        return _phrase_for(rng, planned)

    phrase = _phrase_for(rng, planned)

    # A general contradiction: attach a plausible-but-wrong label to a different code.
    if phrase and chance(rng, float(mess["contradicting_reason_rate"])):
        other = weighted_choice(rng, {"TDS_194C": 0.4, "FREIGHT": 0.3, "SCHEME_REBATE": 0.3})
        alternatives = REASON_PHRASES.get(other, ())
        if alternatives:
            return pick(rng, alternatives)

    return phrase or None


def _amount_text(rng, cfg: dict[str, Any], amount_paise: int) -> str:
    """Render an amount, occasionally with the buyer's own arithmetic slightly wrong.

    A real AP spreadsheet rounds inconsistently and transposes the odd digit. The
    `amount_tolerance_paise` setting in policy.yaml exists to absorb exactly this; without
    the noise there would be nothing for it to absorb and the tolerance would be untested.
    """
    if chance(rng, float(cfg["messiness"]["off_by_a_rupee_rate"])):
        amount_paise = max(0, amount_paise + pick(rng, (-100, -100, 100, 200)))
    return paise_to_rupees_str(amount_paise)


def _render_email(rng, cfg, buyer: Buyer, rows: list[dict[str, Any]]) -> str:
    lines = [pick(rng, _EMAIL_OPENERS), ""]
    for r in rows:
        bits = [f"Invoice {r['ref']}", f"Gross {r['gross']}"]
        if r["deductions"]:
            for d in r["deductions"]:
                label = f" ({d['reason']})" if d["reason"] else ""
                bits.append(f"Less {d['amount']}{label}")
        bits.append(f"Net {r['net']}")
        lines.append(" | ".join(bits))
    lines += ["", f"Total remitted: {rows[-1]['total']}", "", pick(rng, _EMAIL_CLOSERS)]
    return "\n".join(lines)


def _render_pdf_text(rng, cfg, buyer: Buyer, rows: list[dict[str, Any]]) -> str:
    """Text extracted from a PDF: columns collapse, spacing is arbitrary, headers repeat.

    This is what `pdftotext` actually produces, and it is the format most likely to defeat
    a positional parser.
    """
    out = [f"{buyer.name}", "PAYMENT ADVICE", "-" * 46, "InvoiceGrossDeductionNetRemarks"]
    for r in rows:
        ded_total = r["deduction_total"]
        reasons = "; ".join(d["reason"] for d in r["deductions"] if d["reason"])
        # Columns run together with no reliable delimiter — the defining feature.
        out.append(f"{r['ref']}{r['gross']}{ded_total}{r['net']}  {reasons}")
    out += ["-" * 46, f"TOTAL{rows[-1]['total']}"]
    return "\n".join(out)


def _render_xlsx(rng, cfg, buyer: Buyer, rows: list[dict[str, Any]]) -> str:
    """A spreadsheet flattened to text: merged headers, blank rows, a stray note."""
    out = [
        f"\t{buyer.name}\t\t",
        "Payment Advice\t\t\t",
        "",
        "Invoice No\tGross Amt\tDeduction\tNet Paid\tRemarks",
    ]
    for r in rows:
        reasons = "; ".join(d["reason"] for d in r["deductions"] if d["reason"])
        out.append(f"{r['ref']}\t{r['gross']}\t{r['deduction_total']}\t{r['net']}\t{reasons}")
        if chance(rng, 0.18):
            out.append("\t\t\t\t")  # stray blank row
    out += ["", f"\t\tTotal\t{rows[-1]['total']}\t"]
    if chance(rng, 0.25):
        out.append("Note: pls share TDS certificate for the qtr")
    return "\n".join(out)


def build_advice(
    seed: int,
    cfg: dict[str, Any],
    *,
    advice_id: str,
    buyer: Buyer,
    payment_id: str,
    value_date: str,
    invoices: list[Invoice],
    deductions_by_invoice: dict[str, list[PlannedDeduction]],
    reason_texts: dict[str, str | None],
    total_paid_paise: int,
) -> tuple[str, str, str] | None:
    """Build one remittance advice.

    Returns `(format, raw_text, received_at)`, or None when no advice is sent at all —
    which happens a quarter of the time and is not an error.
    """
    mess = cfg["messiness"]
    rng = rng_for(seed, "advice", advice_id)

    if chance(rng, float(mess["advice_absent_rate"])):
        return None

    fmt = weighted_choice(rng, mess["advice_format_weights"])

    rows: list[dict[str, Any]] = []
    for invoice in invoices:
        ded = deductions_by_invoice.get(invoice.id, [])
        ded_total = sum(d.amount_paise for d in ded)
        rows.append(
            {
                # The advice usually cites the invoice number properly — it is the bank
                # narration that mangles it. Occasionally AP mistypes it too.
                "ref": invoice.invoice_no if not chance(rng, 0.12) else invoice.invoice_no.replace("/", "-"),
                "gross": _amount_text(rng, cfg, invoice.total_paise),
                "deduction_total": _amount_text(rng, cfg, ded_total) if ded_total else "0.00",
                "net": paise_to_rupees_str(invoice.total_paise - ded_total),
                "deductions": [
                    {
                        "amount": _amount_text(rng, cfg, d.amount_paise),
                        "reason": reason_texts.get(d.id) or "",
                    }
                    for d in ded
                ],
                "total": paise_to_rupees_str(total_paid_paise),
            }
        )

    renderer = {
        AdviceFormat.EMAIL.value: _render_email,
        AdviceFormat.PDF_TEXT.value: _render_pdf_text,
        AdviceFormat.XLSX.value: _render_xlsx,
    }[fmt]
    raw_text = renderer(rng, cfg, buyer, rows)

    # Late advices arrive days after the credit has already landed, so the agent has to
    # decide what to do with an unexplained shortfall in the meantime.
    delay = (
        sample_range(rng, mess["advice_late_days"])
        if chance(rng, float(mess["advice_late_rate"]))
        else 0
    )
    received_at = add_days(value_date, delay)

    return fmt, raw_text, received_at
