"""Read-only access to the fixture stores — the "source systems" verification checks.

Loaded once and indexed, because the verifiers run over every deduction and a linear scan
of six CSVs per deduction turns a two-second stage into a two-minute one.

**Absence is not disproof.** The single most important thing this module does is
distinguish "I looked and it is not there" from "I looked and I could not have seen it
yet". Form 26AS lags a quarter behind, so a perfectly legitimate TDS deduction may simply
not be visible. A verifier that treats a missing row as evidence of invalidity will chase
customers who did nothing wrong, and it will do so confidently. Every lookup here returns
a `Lookup` that says which of the two it is.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import FIXTURES_DIR
from ..money import rupees_to_paise


@dataclass
class Lookup:
    """The result of consulting a source system."""

    found: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    # True when the store itself is known to be incomplete for this period, so a miss
    # carries no information either way.
    may_lag: bool = False

    @property
    def row(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class FixtureStore:
    """Indexed access to every fixture the verification layer reads."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or FIXTURES_DIR)

        self.form_26as = _read_csv(self.root / "form_26as.csv")
        self.gstr7 = _read_csv(self.root / "gstr7.csv")
        self.credit_notes = _read_csv(self.root / "credit_note_ledger.csv")
        self.schemes = _read_csv(self.root / "scheme_master.csv")
        self.grn = _read_csv(self.root / "grn_discrepancies.csv")
        self.payment_history = _read_csv(self.root / "payment_history.csv")

        contracts_path = self.root / "contracts.json"
        self.contracts_doc: dict[str, Any] = (
            json.loads(contracts_path.read_text(encoding="utf-8"))
            if contracts_path.exists()
            else {"contracts": [], "seller": {}}
        )

        # ---- indexes ----
        self._26as_by_invoice: dict[str, list[dict[str, Any]]] = {}
        for row in self.form_26as:
            self._26as_by_invoice.setdefault(row["invoice_no"], []).append(row)

        self._gstr7_by_invoice: dict[str, list[dict[str, Any]]] = {}
        for row in self.gstr7:
            self._gstr7_by_invoice.setdefault(row["invoice_no"], []).append(row)

        self._cn_by_number: dict[str, dict[str, Any]] = {
            row["credit_note_no"]: row for row in self.credit_notes
        }
        self._cn_by_buyer: dict[str, list[dict[str, Any]]] = {}
        for row in self.credit_notes:
            self._cn_by_buyer.setdefault(row["buyer_id"], []).append(row)

        self._scheme_by_id: dict[str, dict[str, Any]] = {
            row["scheme_id"]: row for row in self.schemes
        }
        self._scheme_by_buyer: dict[str, list[dict[str, Any]]] = {}
        for row in self.schemes:
            self._scheme_by_buyer.setdefault(row["buyer_id"], []).append(row)

        self._grn_by_invoice: dict[str, list[dict[str, Any]]] = {}
        for row in self.grn:
            self._grn_by_invoice.setdefault(row["invoice_no"], []).append(row)

        self._payments_by_invoice: dict[str, list[dict[str, Any]]] = {}
        for row in self.payment_history:
            self._payments_by_invoice.setdefault(row["invoice_no"], []).append(row)

        self._contract_by_buyer: dict[str, dict[str, Any]] = {
            c["buyer_id"]: c for c in self.contracts_doc.get("contracts", [])
        }

    # ------------------------------------------------------------------ TDS
    def tds_for_invoice(self, invoice_no: str, *, section: str | None = None) -> Lookup:
        """26AS rows for an invoice, optionally narrowed to one section.

        The `section` filter is load-bearing. An invoice can carry more than one
        withholding — 194H on a commission component and 194J on a service component is
        entirely ordinary — and matching on invoice number alone returns whichever row
        happens to be first. Comparing a ₹18,558 194J deduction against the ₹9,279 194H
        row makes it look as though the buyer withheld twice what they deposited, and the
        agent goes off to chase a customer who did nothing wrong.
        """
        rows = self._26as_by_invoice.get(invoice_no, [])
        if section:
            wanted = section.replace("TDS_", "")
            narrowed = [r for r in rows if str(r.get("section", "")).replace("TDS_", "") == wanted]
            # If the invoice has 26AS rows but none for this section, that is still a
            # lag-ambiguous miss rather than a contradiction: the other section was filed,
            # this one may not have been.
            return Lookup(found=bool(narrowed), rows=narrowed, may_lag=True)
        # 26AS is filed quarterly and lags. A miss here is genuinely ambiguous, and the
        # policy layer is told so rather than being handed a bare False.
        return Lookup(found=bool(rows), rows=rows, may_lag=True)

    def gst_tds_for_invoice(self, invoice_no: str) -> Lookup:
        rows = self._gstr7_by_invoice.get(invoice_no, [])
        return Lookup(found=bool(rows), rows=rows, may_lag=True)

    # ---------------------------------------------------------- credit notes
    def credit_note(self, number: str) -> Lookup:
        row = self._cn_by_number.get(number)
        # The CN ledger is our own system. If it is not there, it does not exist.
        return Lookup(found=row is not None, rows=[row] if row else [], may_lag=False)

    def credit_notes_for_buyer(self, buyer_id: str) -> Lookup:
        rows = self._cn_by_buyer.get(buyer_id, [])
        return Lookup(found=bool(rows), rows=rows, may_lag=False)

    def unapplied_credit_notes(self, buyer_id: str, amount_paise: int, tolerance: int) -> Lookup:
        """Any unapplied CN for this buyer that matches the deducted amount."""
        matches = [
            row
            for row in self._cn_by_buyer.get(buyer_id, [])
            if row.get("applied") == "N"
            and abs(rupees_to_paise(row["amount_inr"]) - amount_paise) <= tolerance
        ]
        return Lookup(found=bool(matches), rows=matches, may_lag=False)

    # --------------------------------------------------------------- schemes
    def scheme(self, scheme_id: str) -> Lookup:
        row = self._scheme_by_id.get(scheme_id)
        return Lookup(found=row is not None, rows=[row] if row else [], may_lag=False)

    def schemes_for_buyer(self, buyer_id: str) -> Lookup:
        rows = self._scheme_by_buyer.get(buyer_id, [])
        return Lookup(found=bool(rows), rows=rows, may_lag=False)

    # ------------------------------------------------------------------- GRN
    def grn_for_invoice(self, invoice_no: str) -> Lookup:
        rows = self._grn_by_invoice.get(invoice_no, [])
        return Lookup(found=bool(rows), rows=rows, may_lag=False)

    # ------------------------------------------------------- payment history
    def payments_for_invoice(self, invoice_no: str) -> Lookup:
        rows = self._payments_by_invoice.get(invoice_no, [])
        return Lookup(found=bool(rows), rows=rows, may_lag=False)

    # --------------------------------------------------------------- contract
    def contract_for_buyer(self, buyer_id: str) -> Lookup:
        row = self._contract_by_buyer.get(buyer_id)
        return Lookup(found=row is not None, rows=[row] if row else [], may_lag=False)

    # ------------------------------------------------------------------ misc
    @property
    def seller_pan(self) -> str:
        return str(self.contracts_doc.get("seller", {}).get("pan", ""))

    def counts(self) -> dict[str, int]:
        return {
            "form_26as": len(self.form_26as),
            "gstr7": len(self.gstr7),
            "credit_note_ledger": len(self.credit_notes),
            "scheme_master": len(self.schemes),
            "grn_discrepancies": len(self.grn),
            "payment_history": len(self.payment_history),
            "contracts": len(self._contract_by_buyer),
        }
