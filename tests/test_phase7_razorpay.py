"""Phase 7 acceptance: the Razorpay integration cannot break the demo or move real money.

Two claims, both asserted rather than described:

* **The offline path is the default and cannot make a network call.** Not a fallback that
  only runs when something breaks — it is what the batch, the scoreboard and every measured
  number in this project actually ran on.
* **There is no live-money code path.** A key that is not test-mode is refused at
  construction, and the build fails if any module reaches for the Razorpay API outside the
  one client that enforces that.

The spec's requirement is that a broken network must never break the demo, so the failure
modes get tested as carefully as the success ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deduction_desk.actions.razorpay_client import (
    TEST_KEY_PREFIX,
    NullRazorpayClient,
    RazorpayTestClient,
    build_razorpay_client,
    describe_link_for_message,
    load_razorpay_config,
    payment_link_allowed,
)
from deduction_desk.config import load_policy

SRC = Path(__file__).resolve().parents[1] / "src" / "deduction_desk"


# ======================================================================================
# Default is offline
# ======================================================================================


def test_default_config_is_disabled() -> None:
    """Every measured number in this project was produced with this off."""
    assert load_razorpay_config().get("enabled") is False


def test_disabled_config_yields_the_null_client() -> None:
    client = build_razorpay_client({"enabled": False})
    assert isinstance(client, NullRazorpayClient)
    assert client.enabled is False
    assert client.health()["mode"] == "offline"


def test_missing_credentials_yield_the_null_client(monkeypatch) -> None:
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)

    client = build_razorpay_client({"enabled": True})
    assert isinstance(client, NullRazorpayClient)
    assert "not set" in client.health()["reason"]


def test_null_client_produces_a_usable_link_without_a_network_call() -> None:
    """The offline path has to be genuinely usable, not a stub that breaks the message."""
    client = NullRazorpayClient()
    link = client.create_payment_link(
        amount_paise=1_250_000,
        reference_id="CASE-0042-0",
        description="Short payment on INV/2026/0042",
        customer_email="ap@buyer.example",
    )

    assert link.live is False
    assert link.short_url
    assert link.amount_paise == 1_250_000
    assert client.intents[0]["action"] == "create_payment_link"


def test_null_client_links_are_deterministic() -> None:
    """A re-run must produce the same output, or the database content hash moves."""
    a = NullRazorpayClient().create_payment_link(
        amount_paise=100, reference_id="CASE-1", description="d", customer_email="e@x"
    )
    b = NullRazorpayClient().create_payment_link(
        amount_paise=100, reference_id="CASE-1", description="d", customer_email="e@x"
    )
    assert a.id == b.id and a.short_url == b.short_url


def test_null_client_reports_no_gateway_payments() -> None:
    """Recoveries come from the counterparty state machine, never from a gateway.

    Returning nothing here is correct rather than a stub limitation: if payments could
    arrive from an external source, the agent's outcome would no longer be pre-committed
    and the scoreboard would stop being defensible.
    """
    assert NullRazorpayClient().fetch_payments() == []


# ======================================================================================
# No live-money path
# ======================================================================================


def test_a_live_key_is_refused_at_construction() -> None:
    """Refused, not warned about. A warning is something you scroll past."""
    with pytest.raises(ValueError, match="non-test"):
        RazorpayTestClient("rzp_live_abcdef123456", "secret")


def test_a_live_key_downgrades_to_the_null_client(monkeypatch) -> None:
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_live_abcdef123456")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret")

    client = build_razorpay_client({"enabled": True})
    assert isinstance(client, NullRazorpayClient)
    assert "test-mode" in client.health()["reason"]


def test_test_mode_key_is_accepted() -> None:
    client = RazorpayTestClient(f"{TEST_KEY_PREFIX}abcdef123456", "secret")
    assert client.enabled is True
    assert client.health()["mode"] == "test"


def test_only_the_razorpay_client_module_touches_the_api() -> None:
    """No other module may reach the gateway directly.

    Otherwise the test-mode enforcement in this one client could be bypassed by any call
    site that felt like constructing its own request.
    """
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.name == "razorpay_client.py":
            continue
        text = path.read_text(encoding="utf-8")
        for needle in ("api.razorpay.com", "razorpay.com/v1", "payment_links"):
            if needle in text:
                offenders.append(f"{path.relative_to(SRC)}: {needle}")

    assert not offenders, f"Razorpay API reached outside its client: {offenders}"


# ======================================================================================
# The compliance gate is separate from the integration
# ======================================================================================


def test_policy_forbids_payment_links_by_default() -> None:
    """Two independent switches: wired up, and permitted. Both must be open."""
    assert payment_link_allowed(load_policy()) is False


def test_the_draft_validator_rejects_a_link_regardless_of_the_integration() -> None:
    """Even with the integration on, an unauthorised link never reaches a customer.

    The validator polices the drafted message, so it does not matter how the link got
    there — a model hallucinating one is caught by the same check.
    """
    from deduction_desk.actions.validator import validate_draft

    result = validate_draft(
        subject="Payment",
        body="Settle here: https://rzp.io/i/stub-case-1. Case CASE-1.",
        policy=load_policy(),
        case_amount_paise=1_250_000,
    )
    assert not result.ok
    assert "payment_link_not_authorised" in result.rejections


def test_link_description_renders_the_amount_and_reference() -> None:
    link = NullRazorpayClient().create_payment_link(
        amount_paise=1_250_000, reference_id="CASE-0042-0",
        description="d", customer_email="e@x",
    )
    text = describe_link_for_message(link)
    assert "Rs 12,500" in text
    assert "CASE-0042-0" in text
    assert link.short_url in text
