"""Amount reconciliation: which combination of open invoices does this payment settle?

The last resort in the matching ladder, used when the narration's reference is truncated
or absent. Given a payment of ₹X and a buyer's open invoices, find the subset whose totals
minus plausible deductions reconcile to X.

**The important behaviour is refusing to answer.** Subset-sum over a buyer's ledger will
almost always find *some* combination that fits within tolerance, especially once a
deduction allowance is permitted. If two or more combinations fit, the honest output is an
exception, not the first one found. A fabricated allocation posts real money against the
wrong invoice and every number downstream is then wrong with nothing to indicate it.

So this returns a result that distinguishes three states — one solution, several, or none —
and the caller must handle all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

# Hard caps. Subset-sum is exponential, and a buyer with 40 open invoices would otherwise
# hang the batch. Bundles in the wild are 3-7 invoices; beyond that the reference is the
# only realistic route.
MAX_CANDIDATES = 18
MAX_SUBSET_SIZE = 7
MAX_COMBINATIONS = 60_000


@dataclass
class SubsetSumResult:
    """One solution, several, or none — and the caller must tell them apart."""

    invoice_ids: list[str] = field(default_factory=list)
    candidates_considered: int = 0
    solutions_found: int = 0
    exhausted: bool = False  # hit a cap before finishing the search

    @property
    def resolved(self) -> bool:
        return self.solutions_found == 1

    @property
    def ambiguous(self) -> bool:
        return self.solutions_found > 1

    @property
    def unresolved(self) -> bool:
        return self.solutions_found == 0


def reconcile(
    payment_paise: int,
    invoices: list[tuple[str, int]],
    *,
    max_deduction_rate: float = 0.35,
    tolerance_paise: int = 100,
    max_subset_size: int = MAX_SUBSET_SIZE,
) -> SubsetSumResult:
    """Find the invoice subset this payment settles.

    `invoices` is `[(invoice_id, total_paise)]`. A payment may fall short of the subset
    total by up to `max_deduction_rate` — that is the deduction — but may never exceed it,
    since a buyer does not overpay.

    Stops as soon as a second solution is found: the answer is already "ambiguous" and
    continuing only costs time.
    """
    if payment_paise <= 0 or not invoices:
        return SubsetSumResult()

    # Consider only invoices the payment could plausibly cover, largest first.
    upper = int(payment_paise / (1.0 - max_deduction_rate)) + tolerance_paise
    viable = sorted(
        [(iid, total) for iid, total in invoices if 0 < total <= upper],
        key=lambda x: -x[1],
    )[:MAX_CANDIDATES]

    if not viable:
        return SubsetSumResult(candidates_considered=0)

    solutions: list[list[str]] = []
    combos_examined = 0
    exhausted = False

    for size in range(1, min(max_subset_size, len(viable)) + 1):
        for combo in combinations(viable, size):
            combos_examined += 1
            if combos_examined > MAX_COMBINATIONS:
                exhausted = True
                break

            gross = sum(total for _, total in combo)
            if gross + tolerance_paise < payment_paise:
                continue  # cannot reach the payment even with no deduction
            shortfall = gross - payment_paise
            if shortfall < -tolerance_paise:
                continue  # payment exceeds the subset; buyers do not overpay
            if shortfall > gross * max_deduction_rate + tolerance_paise:
                continue  # implies an implausibly large deduction

            solutions.append([iid for iid, _ in combo])
            if len(solutions) > 1:
                # Already ambiguous. The answer will be an exception either way.
                return SubsetSumResult(
                    candidates_considered=len(viable),
                    solutions_found=len(solutions),
                    exhausted=exhausted,
                )
        if exhausted:
            break

    return SubsetSumResult(
        invoice_ids=solutions[0] if len(solutions) == 1 else [],
        candidates_considered=len(viable),
        solutions_found=len(solutions),
        exhausted=exhausted,
    )
