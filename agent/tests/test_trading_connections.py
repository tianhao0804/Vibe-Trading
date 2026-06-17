"""Tests for connector-first trading profile operations."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.live.mandate.model import (
    MANDATE_SCHEMA_VERSION,
    AssetClass,
    ConsentMeta,
    HardCaps,
    InstrumentType,
    Mandate,
    UniverseConstraint,
)
from src.trading import profiles, service
from src.tools import build_registry
from src.tools.trading_connector_tool import TradingSelectConnectionTool

pytestmark = pytest.mark.unit


def _agent_config(server) -> SimpleNamespace:
    return SimpleNamespace(mcp_servers={"robinhood": server})


def _live_mandate(account_ref: str = "RH_AGENTIC_TEST_ACCOUNT") -> Mandate:
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(days=7)
    return Mandate(
        schema_version=MANDATE_SCHEMA_VERSION,
        hard_caps=HardCaps(
            account_funding_usd=300.0,
            max_order_notional_usd=50.0,
            max_total_exposure_usd=150.0,
            max_leverage=1.0,
            allowed_instruments=(InstrumentType.EQUITY,),
            max_trades_per_day=3,
        ),
        universe=UniverseConstraint(
            asset_classes=(AssetClass.US_EQUITY,),
            min_market_cap_usd=None,
            min_avg_daily_volume_usd=None,
            exclude_symbols=(),
        ),
        consent=ConsentMeta(
            created_at=created_at.isoformat(),
            consent_token_sha256="test-consent",
            broker="robinhood",
            account_ref=account_ref,
            expires_at=expires_at.isoformat(),
        ),
    )


def test_remote_call_requires_enabled_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generic remote reads must respect the operator MCP allowlist."""
    server = SimpleNamespace(
        url="https://agent.robinhood.com/mcp/trading",
        enabled_tools=["get_portfolio"],
        auth=SimpleNamespace(cache_dir="/tmp/vibe-no-token"),
    )
    monkeypatch.setattr("src.config.loader.load_agent_config", lambda: _agent_config(server))
    monkeypatch.setattr("src.live.registry.has_cached_oauth_token", lambda *_: True)

    result = service.get_positions("robinhood-live-mcp")

    assert result["status"] == "error"
    assert "not enabled" in result["error"]


def test_remote_call_requires_cached_oauth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generic remote reads must not trigger OAuth from tool/API/MCP paths."""
    server = SimpleNamespace(
        url="https://agent.robinhood.com/mcp/trading",
        enabled_tools=["get_equity_positions"],
        auth=SimpleNamespace(cache_dir="/tmp/vibe-no-token"),
    )
    monkeypatch.setattr("src.config.loader.load_agent_config", lambda: _agent_config(server))
    monkeypatch.setattr("src.live.registry.has_cached_oauth_token", lambda *_: False)

    result = service.get_positions("robinhood-live-mcp")

    assert result["status"] == "not_authorized"
    assert "connector authorize robinhood-live-mcp" in result["error"]


def test_robinhood_remote_reads_use_current_account_scoped_tool_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Robinhood's current MCP catalog requires account_number for account reads."""
    calls: list[tuple[str, dict]] = []
    account_ref = "RH_AGENTIC_TEST_ACCOUNT"
    server = SimpleNamespace(
        url="https://agent.robinhood.com/mcp/trading",
        enabled_tools=[
            "get_portfolio",
            "get_equity_positions",
            "get_equity_orders",
            "get_equity_quotes",
        ],
        auth=SimpleNamespace(cache_dir="/tmp/vibe-token"),
    )
    monkeypatch.setattr("src.config.loader.load_agent_config", lambda: _agent_config(server))
    monkeypatch.setattr("src.live.registry.has_cached_oauth_token", lambda *_: True)

    class FakeAdapter:
        def __init__(self, server_name, server_config):
            assert server_name == "robinhood"
            assert server_config is server

        def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return {"status": "ok", "tool": name, "arguments": arguments}

    monkeypatch.setattr("src.tools.mcp.MCPServerAdapter", FakeAdapter)

    account = service.get_account("robinhood-live-mcp", account=account_ref)
    positions = service.get_positions("robinhood-live-mcp", account=account_ref)
    orders = service.get_open_orders("robinhood-live-mcp", account=account_ref)
    quote = service.get_quote("AAPL", "robinhood-live-mcp")

    assert account["tool"] == "get_portfolio"
    assert positions["tool"] == "get_equity_positions"
    assert orders["tool"] == "get_equity_orders"
    assert quote["tool"] == "get_equity_quotes"
    assert calls == [
        ("get_portfolio", {"account_number": account_ref}),
        ("get_equity_positions", {"account_number": account_ref}),
        ("get_equity_orders", {"account_number": account_ref}),
        ("get_equity_quotes", {"symbols": ["AAPL"]}),
    ]


def test_robinhood_account_scoped_reads_require_explicit_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not silently pick an account from get_accounts for per-account reads."""
    server = SimpleNamespace(
        url="https://agent.robinhood.com/mcp/trading",
        enabled_tools=["get_portfolio"],
        auth=SimpleNamespace(cache_dir="/tmp/vibe-token"),
    )
    monkeypatch.setattr("src.config.loader.load_agent_config", lambda: _agent_config(server))
    monkeypatch.setattr("src.live.registry.has_cached_oauth_token", lambda *_: True)

    result = service.get_account("robinhood-live-mcp")
    blank = service.get_account("robinhood-live-mcp", account="  ")

    assert result["status"] == "error"
    assert "account" in result["error"].lower()
    assert blank["status"] == "error"
    assert "account" in blank["error"].lower()


def test_robinhood_remote_place_order_uses_guarded_current_write_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote Robinhood live orders should pass through the mandate guard."""
    calls: list[tuple[str, dict]] = []
    increments: list[str] = []
    account_ref = "RH_AGENTIC_TEST_ACCOUNT"
    server = SimpleNamespace(
        url="https://agent.robinhood.com/mcp/trading",
        enabled_tools=[
            "get_portfolio",
            "get_equity_positions",
            "place_equity_order",
        ],
        auth=SimpleNamespace(cache_dir="/tmp/vibe-token"),
    )
    monkeypatch.setattr("src.config.loader.load_agent_config", lambda: _agent_config(server))
    monkeypatch.setattr("src.live.registry.has_cached_oauth_token", lambda *_: True)
    monkeypatch.setattr("src.live.order_guard.load_mandate", lambda broker: _live_mandate(account_ref))
    monkeypatch.setattr("src.live.order_guard.read_daily_count", lambda broker: 0)
    monkeypatch.setattr("src.live.order_guard.increment_daily_count", lambda broker: increments.append(broker))
    monkeypatch.setattr(
        "src.live.order_guard.write_live_action",
        lambda event, **_: event.to_record(),
    )

    class FakeAdapter:
        def __init__(self, server_name, server_config):
            assert server_name == "robinhood"
            assert server_config is server

        def call_tool(self, name, arguments, **_kwargs):
            calls.append((name, arguments))
            if name == "get_equity_positions":
                return {"status": "ok", "positions": []}
            if name == "get_portfolio":
                return {"status": "ok", "cash": 300.0, "equity": 300.0}
            if name == "place_equity_order":
                return {"status": "ok", "order_id": "rh_order_1", "state": "accepted"}
            raise AssertionError(f"unexpected tool {name}")

    monkeypatch.setattr("src.tools.mcp.MCPServerAdapter", FakeAdapter)

    result = service.place_order(
        "AAPL",
        "robinhood-live-mcp",
        side="buy",
        notional=25.0,
        order_type="market",
        time_in_force="day",
        market_hours="all_day_hours",
        session_id="sess-1",
        account=account_ref,
    )

    assert result["status"] == "ok"
    assert result["order_id"] == "rh_order_1"
    assert result["profile_id"] == "robinhood-live-mcp"
    assert result["live_action"]["remote_tool"] == "place_equity_order"
    assert increments == ["robinhood"]
    assert calls == [
        ("get_equity_positions", {"account_number": account_ref}),
        ("get_portfolio", {"account_number": account_ref}),
        (
            "place_equity_order",
            {
                "account_number": account_ref,
                "symbol": "AAPL",
                "side": "buy",
                "type": "market",
                "time_in_force": "gfd",
                "market_hours": "all_day_hours",
                "dollar_amount": "25",
            },
        ),
    ]


def test_robinhood_remote_place_order_requires_write_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default Robinhood MCP allowlist must stay read-only for orders."""
    server = SimpleNamespace(
        url="https://agent.robinhood.com/mcp/trading",
        enabled_tools=["get_portfolio", "get_equity_positions"],
        auth=SimpleNamespace(cache_dir="/tmp/vibe-token"),
    )
    monkeypatch.setattr("src.config.loader.load_agent_config", lambda: _agent_config(server))
    monkeypatch.setattr("src.live.registry.has_cached_oauth_token", lambda *_: True)

    result = service.place_order(
        "AAPL",
        "robinhood-live-mcp",
        side="buy",
        notional=25.0,
        session_id="sess-1",
    )

    assert result["status"] == "error"
    assert "place_equity_order" in result["error"]
    assert "not enabled" in result["error"]


def test_robinhood_remote_place_order_requires_explicit_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write-enabled Robinhood live orders must still name the account."""
    server = SimpleNamespace(
        url="https://agent.robinhood.com/mcp/trading",
        enabled_tools=["get_portfolio", "get_equity_positions", "place_equity_order"],
        auth=SimpleNamespace(cache_dir="/tmp/vibe-token"),
    )
    monkeypatch.setattr("src.config.loader.load_agent_config", lambda: _agent_config(server))
    monkeypatch.setattr("src.live.registry.has_cached_oauth_token", lambda *_: True)

    result = service.place_order(
        "AAPL",
        "robinhood-live-mcp",
        side="buy",
        quantity=1,
        order_type="limit",
        limit_price=1.0,
        session_id="sess-1",
    )

    assert result["status"] == "error"
    assert "account" in result["error"].lower()


def test_robinhood_remote_cancel_order_uses_current_tool_and_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote Robinhood cancellation should use cancel_equity_order directly."""
    calls: list[tuple[str, dict]] = []
    audits: list[dict] = []
    server = SimpleNamespace(
        url="https://agent.robinhood.com/mcp/trading",
        enabled_tools=["cancel_equity_order"],
        auth=SimpleNamespace(cache_dir="/tmp/vibe-token"),
    )
    monkeypatch.setattr("src.config.loader.load_agent_config", lambda: _agent_config(server))
    monkeypatch.setattr("src.live.registry.has_cached_oauth_token", lambda *_: True)

    class FakeAdapter:
        def __init__(self, server_name, server_config):
            assert server_name == "robinhood"
            assert server_config is server

        def call_tool(self, name, arguments, **_kwargs):
            calls.append((name, arguments))
            return {"status": "ok", "order_id": arguments["order_id"], "state": "cancelled"}

    monkeypatch.setattr("src.tools.mcp.MCPServerAdapter", FakeAdapter)
    monkeypatch.setattr(
        "src.live.audit.write_live_action",
        lambda event, **_: audits.append(event.to_record()) or event.to_record(),
    )

    result = service.cancel_order(
        "ord_123",
        "robinhood-live-mcp",
        symbol="AAPL",
        session_id="sess-1",
        account="RH_AGENTIC_TEST_ACCOUNT",
    )

    assert result["status"] == "ok"
    assert result["state"] == "cancelled"
    assert result["profile_id"] == "robinhood-live-mcp"
    assert calls == [
        (
            "cancel_equity_order",
            {
                "account_number": "RH_AGENTIC_TEST_ACCOUNT",
                "order_id": "ord_123",
                "symbol": "AAPL",
            },
        )
    ]
    assert audits[0]["kind"] == "order_cancelled"
    assert audits[0]["remote_tool"] == "cancel_equity_order"


def test_ibkr_official_profile_does_not_advertise_unknown_generic_reads() -> None:
    """IBKR official MCP stays honest until stable remote tool names are known."""
    profile = profiles.profile_by_id("ibkr-live-official-mcp-readonly")

    assert profile.capabilities == ("mcp.read.discovery",)
    result = service.get_account(profile.id)
    assert result["status"] == "error"
    assert "does not support" in result["error"]


def test_connector_profile_id_for_broker_prefers_live_remote_mcp() -> None:
    """Broker on-ramps should resolve through the centralized profile registry."""
    assert service.connector_profile_id_for_broker("robinhood") == "robinhood-live-mcp"
    assert service.connector_profile_id_for_broker("ibkr") == "ibkr-live-official-mcp-readonly"
    assert service.connector_profile_id_for_broker("futurebroker") == "futurebroker-live-mcp"


def test_select_connection_tool_returns_canonical_profile_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Selecting a profile should persist and return the canonical id."""
    monkeypatch.setattr(profiles, "get_runtime_root", lambda: tmp_path)

    result = TradingSelectConnectionTool().execute(connection="IBKR-PAPER-LOCAL")

    assert result
    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["selected_profile"] == "ibkr-paper-local"
    assert profiles.load_selected_profile_id() == "ibkr-paper-local"


def test_live_broker_mcp_wrappers_are_hidden_from_agent_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connector-first registry must not expose broker-specific mcp_* tools."""
    server = SimpleNamespace(
        url="https://agent.robinhood.com/mcp/trading",
        enabled_tools=["get_equity_positions"],
        auth=SimpleNamespace(cache_dir="/tmp/vibe-token"),
    )
    agent_config = SimpleNamespace(mcp_servers={"robinhood": server})
    monkeypatch.setattr("src.live.registry.is_live_broker", lambda *_: True)
    monkeypatch.setattr("src.live.registry.should_register_live_channel", lambda **_: True)

    def fail_build_wrappers(*_, **__):
        raise AssertionError("live broker wrappers should not be registered directly")

    monkeypatch.setattr("src.tools.mcp.build_mcp_tool_wrappers", fail_build_wrappers)

    registry = build_registry(agent_config=agent_config, include_shell_tools=False)

    assert "trading_positions" in registry.tool_names
    assert not any(name.startswith("mcp_robinhood_") for name in registry.tool_names)
