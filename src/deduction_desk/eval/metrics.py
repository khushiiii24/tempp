"""Classification metrics, implemented directly rather than pulled from scikit-learn.

Two reasons. The dependency is heavy for four functions, and — more usefully — abstention
has to be handled explicitly rather than inherited from a library's default, because the
default is wrong for this problem.

**Abstention is not an error, and it is not a free pass.** An agent that answers 70% of
cases correctly and routes 30% to a human is better than one that answers 95% with 12%
silent mistakes, because the silent mistakes send letters to customers who did nothing
wrong. But an agent that abstains on everything is useless. So both numbers are reported
and neither is allowed to hide:

* `macro_f1_answered` — quality *given* it committed to an answer. This is the number the
  model-selection floor is applied to.
* `macro_f1_all` — abstention counted as its own class, so over-abstaining is penalised.
* `abstention_rate` — reported beside both, never folded into either.

Quoting only the first would let a model look excellent by refusing to work.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

ABSTAIN = "NEEDS_HUMAN"


@dataclass
class PerClass:
    label: str
    support: int
    predicted: int
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "support": self.support,
            "predicted": self.predicted,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class ClassificationReport:
    per_class: list[PerClass]
    macro_f1: float
    micro_accuracy: float
    n: int
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "macro_f1": round(self.macro_f1, 4),
            "accuracy": round(self.micro_accuracy, 4),
            "n": self.n,
            "per_class": [c.as_dict() for c in self.per_class],
        }


def classification_report(
    y_true: list[str], y_pred: list[str], *, labels: list[str] | None = None
) -> ClassificationReport:
    """Macro-averaged report over the labels actually present.

    Macro rather than weighted, deliberately: the batch is dominated by valid TDS, and a
    weighted average would let a model that only ever recognises TDS look strong while
    missing every recoverable case — which is the entire point of the system.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")

    present = labels or sorted(set(y_true) | set(y_pred))
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    per_class: list[PerClass] = []
    for label in present:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label)
        per_class.append(
            PerClass(
                label=label,
                support=true_counts.get(label, 0),
                predicted=pred_counts.get(label, 0),
                tp=tp, fp=fp, fn=fn,
            )
        )

    # Classes with no support and no predictions contribute nothing; including them would
    # dilute the macro average with meaningless zeros.
    scored = [c for c in per_class if c.support or c.predicted]
    macro_f1 = sum(c.f1 for c in scored) / len(scored) if scored else 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)

    confusion: dict[str, dict[str, int]] = {}
    for t, p in zip(y_true, y_pred, strict=True):
        confusion.setdefault(t, {})
        confusion[t][p] = confusion[t].get(p, 0) + 1

    return ClassificationReport(
        per_class=sorted(scored, key=lambda c: -c.support),
        macro_f1=macro_f1,
        micro_accuracy=correct / len(y_true) if y_true else 0.0,
        n=len(y_true),
        confusion=confusion,
    )


def abstention_aware_report(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """The full picture: quality when it answers, quality overall, and how often it declines."""
    n = len(y_true)
    abstained = [i for i, p in enumerate(y_pred) if p == ABSTAIN]
    answered = [i for i, p in enumerate(y_pred) if p != ABSTAIN]

    all_report = classification_report(y_true, y_pred)
    answered_report = (
        classification_report([y_true[i] for i in answered], [y_pred[i] for i in answered])
        if answered
        else None
    )

    return {
        "n": n,
        "n_answered": len(answered),
        "n_abstained": len(abstained),
        "abstention_rate": round(len(abstained) / n, 4) if n else 0.0,
        # Quality given it committed. The model-selection floor applies here.
        "macro_f1_answered": round(answered_report.macro_f1, 4) if answered_report else 0.0,
        "accuracy_answered": round(answered_report.micro_accuracy, 4) if answered_report else 0.0,
        # Abstention as its own class, so refusing to work is not rewarded.
        "macro_f1_all": round(all_report.macro_f1, 4),
        "accuracy_all": round(all_report.micro_accuracy, 4),
        "per_class": [c.as_dict() for c in all_report.per_class],
        "confusion": all_report.confusion,
    }


def confusion_lines(confusion: dict[str, dict[str, int]], *, limit: int = 12) -> list[str]:
    """The most common confusions, as readable lines. For the model-selection doc."""
    rows: list[tuple[int, str]] = []
    for true_label, preds in confusion.items():
        for pred_label, count in preds.items():
            if true_label != pred_label:
                rows.append((count, f"{true_label} -> {pred_label} ({count})"))
    rows.sort(key=lambda r: -r[0])
    return [text for _, text in rows[:limit]]
