"""Deterministic randomness.

`generate --seed 42 --n 400` must produce a byte-identical database every time, on any
machine, on any day. That rules out `random` module-level state, `datetime.now()`, set
iteration order, and dict ordering that depends on insertion from an unordered source.

**Why per-entity streams rather than one threaded RNG.** The obvious design is a single
seeded generator passed down through buyers → contracts → invoices → deductions. It is
deterministic, and it is also brittle in a way that costs real time: adding one extra
draw in `invoices.py` shifts every subsequent number in the entire batch, so a one-line
change to invoice generation silently rewrites every deduction, every truth record and
every fixture. Diffing two runs to see what a change did becomes impossible.

Instead each entity gets its own stream, derived from
`sha256(seed : namespace : entity_id)`. Buyer BUY-0007's attributes depend on the seed and
on its own id, and on nothing else. Change how invoices are drawn and the buyers are
untouched; add a new field to deductions and the existing fields keep their values. The
determinism test still passes, and the diff between two runs is readable.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def derive_seed(seed: int, namespace: str, entity_id: str | int = "") -> int:
    """A stable 64-bit seed for one (namespace, entity) pair."""
    material = f"{seed}:{namespace}:{entity_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def rng_for(seed: int, namespace: str, entity_id: str | int = "") -> random.Random:
    """An independent RNG stream.

    `random.Random` (Mersenne Twister) is used rather than numpy because its sequence for
    a given integer seed is stable across Python versions and platforms, which is exactly
    the guarantee the determinism test needs.
    """
    return random.Random(derive_seed(seed, namespace, entity_id))


# ----------------------------------------------------------------------------------
# Sampling helpers
# ----------------------------------------------------------------------------------
def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    """Pick a key by weight.

    Keys are sorted before sampling. Without that, a dict built in a different order
    would produce a different draw from the same RNG state — a genuinely nasty
    non-determinism, because the dict compares equal either way.
    """
    keys = sorted(weights)
    values = [float(weights[k]) for k in keys]
    return rng.choices(keys, weights=values, k=1)[0]


def weighted_index(rng: random.Random, weights: Sequence[float]) -> int:
    return rng.choices(range(len(weights)), weights=list(weights), k=1)[0]


def pick(rng: random.Random, items: Sequence[T]) -> T:
    return items[rng.randrange(len(items))]


def sample_range(rng: random.Random, bounds: Sequence[int]) -> int:
    """Inclusive integer draw from a `[low, high]` pair as written in the YAML."""
    low, high = int(bounds[0]), int(bounds[1])
    return rng.randint(low, high) if high > low else low


def chance(rng: random.Random, probability: float) -> bool:
    return rng.random() < float(probability)


def jitter_paise(rng: random.Random, amount: int, pct: float = 0.15) -> int:
    """Vary an amount by +/- pct, staying an integer number of paise."""
    delta = int(amount * pct)
    return max(0, amount + rng.randint(-delta, delta)) if delta else amount


def round_to_rupee(amount_paise: int) -> int:
    """Snap to a whole rupee. Buyers deduct round numbers far more often than not."""
    return (amount_paise // 100) * 100
