"""Normalising the things banks mangle: invoice references and payer names.

A bank narration is thirty-odd characters of upper-cased, delimiter-stripped free text
produced by whatever system the buyer's bank happens to run. `INV/2026/0042` arrives as
`INV20260042`, `2026-42`, `INV 42`, `RAZ/INV/26/42` — or truncated mid-number to
`INV202600`, at which point the digits are simply gone and no amount of normalisation
brings them back.

That last case is the one that matters. The honest response to an unrecoverable reference
is to say so and put the payment in the exceptions report, not to pick the nearest
candidate and hope. A fabricated match silently mis-allocates real money and then every
downstream number is wrong with no signal that anything happened.

So this module produces *candidate keys*, and the matcher decides. Nothing here guesses.
"""

from __future__ import annotations

import re

# Everything that is not alphanumeric is noise as far as a bank narration is concerned.
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")
_DIGITS = re.compile(r"\d+")

# Tokens that appear in narrations and carry no identifying information.
_NOISE_TOKENS = frozenset(
    {"NEFT", "RTGS", "IMPS", "UPI", "CHQ", "TRF", "PAYMENT", "PYMT", "INV", "RAZ", "MORE"}
)


def normalise_ref(ref: str) -> str:
    """Case-fold and strip every separator. `INV/2026/0042` -> `INV20260042`."""
    return _NON_ALNUM.sub("", (ref or "")).upper()


def digits_only(ref: str) -> str:
    """All digits, concatenated. `INV/2026/0042` -> `20260042`."""
    return "".join(_DIGITS.findall(ref or ""))


def ref_variants(invoice_no: str) -> set[str]:
    """Every form an invoice number might legitimately take in a narration.

    Generated from the canonical number so the matcher can look up an observed token
    directly rather than fuzzy-matching when it does not have to. Exact beats fuzzy every
    time: it cannot produce a false positive.
    """
    canonical = normalise_ref(invoice_no)
    digits = digits_only(invoice_no)

    variants = {canonical, digits}

    m = re.match(r"INV/(\d{4})/(\d+)", invoice_no or "")
    if m:
        year, num = m.group(1), m.group(2)
        stripped = num.lstrip("0") or "0"
        variants.update(
            {
                f"INV{year}{num}",
                f"{year}{num}",
                f"{year}{stripped}",
                f"INV{stripped}",
                stripped,
                num,
                f"RAZINV{year[-2:]}{stripped}",
                f"INV{year[-2:]}{stripped}",
            }
        )

    return {v for v in variants if v}


def narration_tokens(narration: str) -> list[str]:
    """Split a narration into candidate tokens, dropping the obvious noise words."""
    raw = re.split(r"[^A-Za-z0-9]+", (narration or "").upper())
    return [t for t in raw if t and t not in _NOISE_TOKENS]


# A bare four-digit year is never an invoice reference. It appears in every narration that
# carries a date-formatted number, and it is short enough to fuzzy-match almost anything.
_YEAR = re.compile(r"^(19|20)\d{2}$")

# Below this length a token carries too little information for FUZZY comparison. Exact
# lookup on a short token is still safe, because the variant map has already rejected any
# form that identifies more than one invoice.
MIN_FUZZY_LENGTH = 6


def is_year_token(token: str) -> bool:
    return bool(_YEAR.match(token or ""))


def candidate_refs(narration: str) -> set[str]:
    """Tokens from a narration that could plausibly be an invoice reference.

    Anything containing a digit, except a bare year. Deliberately generous otherwise — the
    matcher rejects what does not resolve, and being generous here only costs lookups.
    """
    out: set[str] = set()
    for token in narration_tokens(narration):
        if not any(ch.isdigit() for ch in token):
            continue
        if is_year_token(token):
            continue
        out.add(token)
        digits = digits_only(token)
        if digits and not is_year_token(digits):
            out.add(digits)
    return {t for t in out if t}


def fuzzy_candidates(tokens: set[str]) -> list[str]:
    """Tokens long enough to fuzzy-match safely.

    Measured consequence of not doing this: `INV/2026/0003` generates the short variant
    `20263`, and the bare year token `2026` scores 88.9 against it — over an 88 threshold.
    Six payments belonging to other invoices were allocated to INV-0003 on that basis, and
    115 invoices ended up with no allocation at all because their payments had been
    consumed elsewhere. Short strings are similar to everything.
    """
    return sorted(t for t in tokens if len(t) >= MIN_FUZZY_LENGTH)


def payer_key(name: str) -> str:
    """A comparable form of a company name.

    Bank narrations truncate the payer name to a dozen-odd characters and strip spaces, so
    `Shree Steel Industries Pvt Ltd` arrives as `SHREESTEELIN`. Comparing prefixes of this
    key is what resolves a payment to a buyer.
    """
    cleaned = _NON_ALNUM.sub("", (name or "")).upper()
    for suffix in ("PVTLTD", "PRIVATELIMITED", "LIMITED", "LTD", "ANDSONS", "CO"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned


def narration_name_fragment(narration: str) -> str:
    """The longest alphabetic token in a narration — nearly always the payer name.

    Bank formats vary, but the pattern `PREFIX-REF-NAME-INVOICE` is near-universal, and the
    name is the longest run of letters.
    """
    alpha = [t for t in narration_tokens(narration) if t.isalpha()]
    return max(alpha, key=len) if alpha else ""


def truncation_suspect(token: str, variants: set[str]) -> bool:
    """Is this token a prefix of a known reference, i.e. probably truncated?

    Distinguishes "the bank cut the number short" from "this is a different invoice".
    A prefix match is evidence of truncation, not of identity — the matcher treats it as
    ambiguous rather than as a hit, because several invoices can share a prefix.
    """
    if len(token) < 6:
        return False
    return any(v.startswith(token) and v != token for v in variants)


# A bank narration that could not fit the reference list says so: `INV 128+5MORE`.
_BUNDLE_OVERFLOW = re.compile(r"\+\s*(\d+)\s*MORE", re.IGNORECASE)


def declared_bundle_size(narration: str) -> int | None:
    """How many invoices the narration claims to cover, if it says.

    `INV 128+5MORE` means six invoices: the one named plus five the bank dropped. That
    count is hard evidence and it is worth using — it converts an open-ended subset-sum
    over a buyer's whole ledger into a search for an exactly known number of invoices.

    Returns None when the narration makes no such claim.
    """
    match = _BUNDLE_OVERFLOW.search(narration or "")
    if not match:
        return None
    return int(match.group(1)) + 1
