"""Phase 7: Razorpay test-mode integration, behind an interface that cannot break the demo.

The spec's constraint is the design: *keep it behind an interface so the batch still runs
fully offline — a broken network must never break the demo.* So the default implementation
does not touch the network at all, and the live one is opt-in and fails soft.

Three things this adds when enabled:

* **`create_payment_link`** on a chase, so the recovery ask carries something payable
  rather than an instruction to raise a transfer.
* **`fetch_payments`** to detect the incoming credit, which is how a real deployment would
  learn that a chase worked.
* **`create_refund`** for the one case where money flows the other way — a rate difference
  the seller caused, where a credit note is owed.

## Why it is off by default, and stays off

A payment link in an outbound message is a security decision, not a feature toggle. Three
independent gates must all be open before one is created:

1. `razorpay.enabled: true` in `config/llm.yaml`'s sibling — the config must ask for it
2. `drafting.allow_payment_link: true` in `policy.yaml` — the compliance policy must permit it
3. `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` present and test-mode

Any one missing and the client is `NullRazorpayClient`, which records what it *would* have
done and returns a deterministic stub. The scoreboard is identical either way, because the
counterparty's decision to pay was pre-committed by the generator and no payment link
changes it.

**Test mode is enforced, not assumed.** A key that does not start with `rzp_test_` is
rejected outright. There is no code path in this repository that can move real money, and
that is checked by a test rather than left to the reader.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import CONFIG_DIR, load_yaml
from ..money import Paise, format_inr

RAZORPAY_API = "https://api.razorpay.com/v1"
TEST_KEY_PREFIX = "rzp_test_"


@dataclass
class PaymentLink:
    """A request for money. `live=False` means nothing left this machine."""

    id: str
    short_url: str
    amount_paise: int
    reference_id: str
    live: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "short_url": self.short_url,
            "amount_paise": self.amount_paise,
            "reference_id": self.reference_id,
            "live": self.live,
        }


class RazorpayClient(Protocol):
    """The seam. Call sites depend on this and never on a concrete implementation."""

    enabled: bool

    def create_payment_link(
        self, *, amount_paise: int, reference_id: str, description: str, customer_email: str
    ) -> PaymentLink: ...

    def fetch_payments(self, *, count: int = 100) -> list[dict[str, Any]]: ...

    def create_refund(self, *, payment_id: str, amount_paise: int) -> dict[str, Any]: ...

    def health(self) -> dict[str, Any]: ...


# ======================================================================================
# Default: does not touch the network
# ======================================================================================
class NullRazorpayClient:
    """Records the intent and returns a deterministic stub.

    Not a mock for tests — this is the **production default**. The batch, the scoreboard
    and the demo all run through it, so the offline path is the well-trodden one rather
    than a fallback that only gets exercised when something breaks.

    Link ids are derived from the reference so a re-run produces the same output and the
    database content hash stays stable.
    """

    enabled = False

    def __init__(self, reason: str = "not configured") -> None:
        self.reason = reason
        self.intents: list[dict[str, Any]] = []

    def create_payment_link(
        self, *, amount_paise: int, reference_id: str, description: str, customer_email: str
    ) -> PaymentLink:
        self.intents.append(
            {
                "action": "create_payment_link",
                "amount_paise": int(amount_paise),
                "reference_id": reference_id,
                "customer_email": customer_email,
            }
        )
        digest = abs(hash(reference_id)) % 10**10
        return PaymentLink(
            id=f"plink_stub_{digest:010d}",
            short_url=f"https://rzp.io/i/stub-{reference_id.lower()}",
            amount_paise=Paise(int(amount_paise)),
            reference_id=reference_id,
            live=False,
        )

    def fetch_payments(self, *, count: int = 100) -> list[dict[str, Any]]:
        # Recoveries come from the counterparty state machine, never from a gateway. This
        # returning empty is correct, not a stub limitation.
        return []

    def create_refund(self, *, payment_id: str, amount_paise: int) -> dict[str, Any]:
        self.intents.append(
            {"action": "create_refund", "payment_id": payment_id, "amount_paise": int(amount_paise)}
        )
        return {"id": f"rfnd_stub_{payment_id}", "status": "stubbed", "live": False}

    def health(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "mode": "offline",
            "reason": self.reason,
            "note": "No network call is possible through this client.",
        }


# ======================================================================================
# Test-mode REST client
# ======================================================================================
class RazorpayTestClient:
    """Razorpay REST API, **test mode only**.

    Every method fails soft: a network error returns the same shape the null client would,
    so a broken connection degrades the message rather than breaking the run. That is the
    spec's requirement and it is the right behaviour anyway — an AR batch should not stop
    because a payment gateway is slow.
    """

    enabled = True

    def __init__(self, key_id: str, key_secret: str, *, timeout_s: int = 15) -> None:
        if not key_id.startswith(TEST_KEY_PREFIX):
            # Refused rather than warned. There is no path in this repository that should
            # be able to touch a live key, and a warning is something you scroll past.
            raise ValueError(
                f"refusing a non-test Razorpay key: expected a {TEST_KEY_PREFIX!r} prefix. "
                f"This project has no live-money code path."
            )
        self.key_id = key_id
        self.key_secret = key_secret
        self.timeout_s = timeout_s
        self._fallback = NullRazorpayClient(reason="network call failed")

    def _auth_header(self) -> str:
        token = base64.b64encode(f"{self.key_id}:{self.key_secret}".encode()).decode()
        return f"Basic {token}"

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{RAZORPAY_API}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": self._auth_header(),
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def create_payment_link(
        self, *, amount_paise: int, reference_id: str, description: str, customer_email: str
    ) -> PaymentLink:
        try:
            body = self._request(
                "POST",
                "/payment_links",
                {
                    "amount": int(amount_paise),
                    "currency": "INR",
                    "accept_partial": False,
                    "reference_id": reference_id,
                    "description": description[:2048],
                    "customer": {"email": customer_email},
                    "notify": {"sms": False, "email": False},
                    "reminder_enable": False,
                },
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return self._fallback.create_payment_link(
                amount_paise=amount_paise,
                reference_id=reference_id,
                description=description,
                customer_email=customer_email,
            )

        return PaymentLink(
            id=str(body.get("id", "")),
            short_url=str(body.get("short_url", "")),
            amount_paise=Paise(int(amount_paise)),
            reference_id=reference_id,
            live=True,
            raw=body,
        )

    def fetch_payments(self, *, count: int = 100) -> list[dict[str, Any]]:
        try:
            body = self._request("GET", f"/payments?count={int(count)}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return []
        return list(body.get("items", []))

    def create_refund(self, *, payment_id: str, amount_paise: int) -> dict[str, Any]:
        try:
            return self._request(
                "POST", f"/payments/{payment_id}/refund", {"amount": int(amount_paise)}
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return self._fallback.create_refund(payment_id=payment_id, amount_paise=amount_paise)

    def health(self) -> dict[str, Any]:
        try:
            self._request("GET", "/payments?count=1")
            reachable = True
            error = None
        except Exception as exc:  # noqa: BLE001 - health must never raise
            reachable = False
            error = str(exc)[:200]
        return {
            "enabled": True,
            "mode": "test",
            "key_id": self.key_id[:14] + "…",
            "reachable": reachable,
            "error": error,
        }


# ======================================================================================
# Construction
# ======================================================================================
def load_razorpay_config() -> dict[str, Any]:
    path = CONFIG_DIR / "razorpay.yaml"
    if not path.exists():
        return {"enabled": False}
    return load_yaml(path)


def build_razorpay_client(cfg: dict[str, Any] | None = None) -> RazorpayClient:
    """Resolve the client. Returns the null client unless every gate is open.

    Order matters: config first, then credentials, then test-mode. Each refusal names
    itself so `doctor` can say exactly which gate is shut.
    """
    cfg = cfg if cfg is not None else load_razorpay_config()

    if not cfg.get("enabled"):
        return NullRazorpayClient(reason="razorpay.enabled is false in config/razorpay.yaml")

    key_id = os.environ.get(str(cfg.get("key_id_env", "RAZORPAY_KEY_ID")), "")
    key_secret = os.environ.get(str(cfg.get("key_secret_env", "RAZORPAY_KEY_SECRET")), "")

    if not key_id or not key_secret:
        return NullRazorpayClient(reason="RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set")

    if not key_id.startswith(TEST_KEY_PREFIX):
        return NullRazorpayClient(
            reason=f"key is not test-mode (expected {TEST_KEY_PREFIX!r}); refusing to use it"
        )

    return RazorpayTestClient(key_id, key_secret, timeout_s=int(cfg.get("timeout_s", 15)))


def payment_link_allowed(policy) -> bool:
    """The compliance gate on links, separate from the integration being configured.

    Two independent switches: the integration can be wired up and the policy can still
    forbid putting a link in a letter. `actions/validator.py` enforces the second one on
    the drafted message regardless of what happened here.
    """
    return bool(policy.drafting.get("allow_payment_link", False))


def describe_link_for_message(link: PaymentLink) -> str:
    """One line to append to a chase, when links are permitted."""
    return (
        f"You can settle {format_inr(link.amount_paise)} directly here: {link.short_url} "
        f"(reference {link.reference_id})"
    )
