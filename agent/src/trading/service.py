"""Connector-first trading operations used by CLI, MCP, and agent tools."""

from __future__ import annotations

import json
from typing import Any

from src.trading.profiles import list_profiles, profile_by_id
from src.trading.types import TradingProfile

RUNNER_CAPABILITY = "runner.manage.requires_mandate"

#: Direct-SDK connectors (``broker_sdk`` transport) → their connector module.
#: Each module exposes a uniform read interface (``build_config``, ``check_status``,
#: ``get_account_snapshot``, ``get_positions``, ``get_open_orders``, ``get_quote``,
#: ``get_historical_bars``).
_SDK_CONNECTOR_MODULES = {
    "tiger": "src.trading.connectors.tiger.sdk",
    "longbridge": "src.trading.connectors.longbridge.sdk",
    "alpaca": "src.trading.connectors.alpaca.sdk",
    "okx": "src.trading.connectors.okx.sdk",
    "binance": "src.trading.connectors.binance.sdk",
    "futu": "src.trading.connectors.futu.sdk",
    "dhan": "src.trading.connectors.dhan.sdk",
    "shoonya": "src.trading.connectors.shoonya.sdk",
}


def _sdk_module(connector: str):
    """Import the SDK connector module for a ``broker_sdk`` connector key."""
    import importlib

    path = _SDK_CONNECTOR_MODULES.get(connector)
    if path is None:
        raise ValueError(f"no SDK connector module for '{connector}'")
    return importlib.import_module(path)


def check_connection(profile_id: str | None = None, **overrides: Any) -> dict[str, Any]:
    """Check a connector profile without mutating broker state."""
    profile = profile_by_id(profile_id)
    if profile.transport == "local_tws":
        from src.trading.connectors.ibkr.local import check_local_status

        cfg = _ibkr_config(profile, overrides)
        report = check_local_status(cfg)
        report["profile_id"] = profile.id
        report["connector"] = profile.connector
        report["environment"] = profile.environment
        report["transport"] = profile.transport
        return report

    if profile.transport == "broker_sdk":
        module = _sdk_module(profile.connector)
        report = module.check_status(module.build_config(profile.config, overrides))
        report["profile_id"] = profile.id
        report["connector"] = profile.connector
        report["environment"] = profile.environment
        report["transport"] = profile.transport
        return report

    return _remote_status(profile)


def get_account(profile_id: str | None = None, **overrides: Any) -> dict[str, Any]:
    """Read account summary for a connector profile."""
    profile = profile_by_id(profile_id)
    if profile.transport == "local_tws":
        from src.trading.connectors.ibkr.local import get_account_snapshot

        return _with_profile(profile, get_account_snapshot(_ibkr_config(profile, overrides)))
    if profile.transport == "broker_sdk":
        module = _sdk_module(profile.connector)
        return _with_profile(profile, module.get_account_snapshot(module.build_config(profile.config, overrides)))
    return _call_remote(profile, "account", overrides)


def get_positions(profile_id: str | None = None, **overrides: Any) -> dict[str, Any]:
    """Read positions for a connector profile."""
    profile = profile_by_id(profile_id)
    if profile.transport == "local_tws":
        from src.trading.connectors.ibkr.local import get_positions as _get_positions

        return _with_profile(profile, _get_positions(_ibkr_config(profile, overrides)))
    if profile.transport == "broker_sdk":
        module = _sdk_module(profile.connector)
        return _with_profile(profile, module.get_positions(module.build_config(profile.config, overrides)))
    return _call_remote(profile, "positions", overrides)


def get_open_orders(
    profile_id: str | None = None,
    *,
    include_executions: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Read open orders for a connector profile."""
    profile = profile_by_id(profile_id)
    if profile.transport == "local_tws":
        from src.trading.connectors.ibkr.local import get_open_orders as _get_open_orders

        return _with_profile(
            profile,
            _get_open_orders(_ibkr_config(profile, overrides), include_executions=include_executions),
        )
    if profile.transport == "broker_sdk":
        module = _sdk_module(profile.connector)
        return _with_profile(
            profile,
            module.get_open_orders(module.build_config(profile.config, overrides), include_executions=include_executions),
        )
    remote_args = {"include_executions": include_executions, **overrides}
    return _call_remote(profile, "orders", remote_args)


def get_quote(
    symbol: str,
    profile_id: str | None = None,
    *,
    exchange: str = "SMART",
    currency: str = "USD",
    sec_type: str = "STK",
    **overrides: Any,
) -> dict[str, Any]:
    """Read a quote for a connector profile."""
    profile = profile_by_id(profile_id)
    if profile.transport == "local_tws":
        from src.trading.connectors.ibkr.local import get_quote as _get_quote

        return _with_profile(
            profile,
            _get_quote(
                symbol,
                config=_ibkr_config(profile, overrides),
                exchange=exchange,
                currency=currency,
                sec_type=sec_type,
            ),
        )
    if profile.transport == "broker_sdk":
        module = _sdk_module(profile.connector)
        return _with_profile(profile, module.get_quote(symbol, config=module.build_config(profile.config, overrides)))
    return _call_remote(profile, "quote", {"symbols": [symbol], "symbol": symbol})


def get_history(
    symbol: str,
    profile_id: str | None = None,
    *,
    exchange: str = "SMART",
    currency: str = "USD",
    sec_type: str = "STK",
    duration: str = "30 D",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    period: str = "1d",
    limit: int = 90,
    **overrides: Any,
) -> dict[str, Any]:
    """Read historical bars for a connector profile.

    ``duration``/``bar_size``/``what_to_show``/``use_rth`` are the IBKR
    (``local_tws``) vocabulary. ``period`` (e.g. ``1m``/``5m``/``1h``/``1d``)
    and ``limit`` are the generic vocabulary every ``broker_sdk`` connector
    understands and maps to its own SDK tokens.
    """
    profile = profile_by_id(profile_id)
    if profile.transport == "local_tws":
        from src.trading.connectors.ibkr.local import get_historical_bars

        return _with_profile(
            profile,
            get_historical_bars(
                symbol,
                config=_ibkr_config(profile, overrides),
                exchange=exchange,
                currency=currency,
                sec_type=sec_type,
                duration=duration,
                bar_size=bar_size,
                what_to_show=what_to_show,
                use_rth=use_rth,
            ),
        )
    if profile.transport == "broker_sdk":
        module = _sdk_module(profile.connector)
        return _with_profile(
            profile,
            module.get_historical_bars(
                symbol,
                config=module.build_config(profile.config, overrides),
                period=period,
                limit=limit,
            ),
        )
    return _unsupported(profile, "history.read")


#: Connector → (instrument type, fixed asset class | None). ``None`` asset class
#: means "infer from the symbol's market" (multi-market equity connectors).
_CONNECTOR_INSTRUMENT = {
    "okx": ("crypto", "crypto"),
    "binance": ("crypto", "crypto"),
    "alpaca": ("equity", "us_equity"),
    "tiger": ("equity", None),
    "longbridge": ("equity", None),
    "futu": ("equity", None),
}


def _order_classification(connector: str, symbol: str):
    """Return ``(InstrumentType, AssetClass | None)`` for an order's mandate gate.

    Crypto connectors are unambiguous; multi-market equity connectors infer the
    asset class from the symbol's market tag (``.HK``/``HK.`` → HK, ``.US``/``US.``
    → US, ``.SH``/``.SZ``/``CN.`` → A-share). When the market cannot be inferred
    the asset class is ``None`` and the gate falls back to the US default — which
    only ever DENIES (never silently widens) when the user's mandate permits a
    non-US class, so the unknown case is fail-safe.
    """
    from src.live.mandate.model import AssetClass, InstrumentType

    instrument_name, asset_name = _CONNECTOR_INSTRUMENT.get(connector, ("equity", None))
    instrument = InstrumentType(instrument_name)
    if asset_name is not None:
        return instrument, AssetClass(asset_name)

    token = (symbol or "").strip().upper()
    if token.startswith("HK.") or token.endswith(".HK"):
        return instrument, AssetClass.HK_EQUITY
    if token.startswith("US.") or token.endswith(".US"):
        return instrument, AssetClass.US_EQUITY
    if token.startswith(("CN.", "SH.", "SZ.")) or token.endswith((".SH", ".SS", ".SZ")):
        return instrument, AssetClass.CN_EQUITY
    return instrument, None


def place_order(
    symbol: str,
    profile_id: str | None = None,
    *,
    side: str,
    quantity: float | None = None,
    notional: float | None = None,
    order_type: str = "market",
    limit_price: float | None = None,
    time_in_force: str = "day",
    session_id: str = "",
    **overrides: Any,
) -> dict[str, Any]:
    """Place an order via a connector profile.

    Paper profiles place directly against the broker's sandbox account. Live
    profiles route through the mandate gate (mandate + kill switch +
    fail-closed pre-trade checks + audit) before any order reaches the broker.
    Direct SDK profiles use ``execute_live_order``; live remote-MCP profiles
    use the same ``LiveOrderGuardTool`` wrapper as raw MCP registration.
    """
    profile = profile_by_id(profile_id)
    if profile.transport == "remote_mcp" and profile.environment == "live":
        return _place_remote_live_order(
            profile,
            symbol=symbol,
            side=side,
            quantity=quantity,
            notional=notional,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            session_id=session_id,
        )
    if profile.transport != "broker_sdk":
        return _unsupported(profile, "orders.place")

    module = _sdk_module(profile.connector)
    config = module.build_config(profile.config, overrides)
    place_kwargs = {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "notional": notional,
        "order_type": order_type,
        "limit_price": limit_price,
        "time_in_force": time_in_force,
    }

    if profile.environment == "paper":
        return _with_profile(profile, module.place_order(config, **place_kwargs))

    # Live: pre-trade mandate gate.
    from src.live.enforcement import OrderIntent
    from src.live.sdk_order_gate import execute_live_order

    instrument_type, asset_class = _order_classification(profile.connector, symbol)
    intent = OrderIntent(
        symbol=str(symbol or "").strip().upper(),
        side=str(side or "").strip().lower(),
        notional_usd=float(notional) if notional is not None else None,
        quantity=float(quantity) if quantity is not None else None,
        instrument_type=instrument_type,
        asset_class=asset_class,
    )
    result = execute_live_order(
        broker=profile.connector,
        connector_module=module,
        config=config,
        intent=intent,
        place_kwargs=place_kwargs,
        session_id=session_id,
    )
    return _with_profile(profile, result)


def cancel_order(
    order_id: str,
    profile_id: str | None = None,
    *,
    symbol: str | None = None,
    session_id: str = "",
    **overrides: Any,
) -> dict[str, Any]:
    """Cancel an order via a connector profile.

    Cancelling is risk-reducing, so it is not blocked by the mandate or the kill
    switch (a halt should still let the user cancel resting orders). But a live
    cancel IS a live action, so it is written to the audit ledger — every live
    action must be logged (Red Lines).
    """
    profile = profile_by_id(profile_id)
    if profile.transport == "remote_mcp" and profile.environment == "live":
        return _cancel_remote_live_order(
            profile,
            order_id=order_id,
            symbol=symbol,
            session_id=session_id,
        )
    if profile.transport != "broker_sdk":
        return _unsupported(profile, "orders.cancel")
    module = _sdk_module(profile.connector)
    config = module.build_config(profile.config, overrides)
    result = module.cancel_order(config, order_id, symbol=symbol)
    if profile.environment == "live":
        _audit_live_cancel(profile, order_id, symbol, result, session_id)
    return _with_profile(profile, result)


def _place_remote_live_order(
    profile: TradingProfile,
    *,
    symbol: str,
    side: str,
    quantity: float | None,
    notional: float | None,
    order_type: str,
    limit_price: float | None,
    time_in_force: str,
    session_id: str,
) -> dict[str, Any]:
    """Place a live remote-MCP order through the mandate guard."""
    from src.live.order_guard import LiveOrderGuardTool
    from src.tools.mcp import MCPRemoteToolSpec, MCPServerAdapter

    remote_name = runner_tool_name(profile.connector, "submit_order")
    if remote_name is None:
        return _unsupported(profile, "orders.place")

    context = _remote_write_context(profile, remote_name)
    if "error" in context:
        return context["error"]

    broker_request = _remote_order_arguments(
        symbol=symbol,
        side=side,
        quantity=quantity,
        notional=notional,
        order_type=order_type,
        limit_price=limit_price,
        time_in_force=time_in_force,
    )
    adapter = MCPServerAdapter(context["server_name"], context["server"])
    spec = MCPRemoteToolSpec(
        server_name=context["server_name"],
        remote_name=remote_name,
        local_name=f"mcp_{context['server_name']}_{remote_name}",
        description=f"Live {profile.connector} order placement via remote MCP.",
        parameters={"type": "object", "additionalProperties": True},
    )
    raw = LiveOrderGuardTool(
        adapter,
        spec,
        broker=profile.connector,
        session_id=session_id,
    ).execute(**broker_request)
    return _with_profile(profile, _json_object_payload(raw))


def _cancel_remote_live_order(
    profile: TradingProfile,
    *,
    order_id: str,
    symbol: str | None,
    session_id: str,
) -> dict[str, Any]:
    """Cancel a live remote-MCP order and audit the risk-reducing write."""
    from src.tools.mcp import MCPServerAdapter

    remote_name = runner_tool_name(profile.connector, "cancel_order")
    if remote_name is None:
        return _unsupported(profile, "orders.cancel")

    context = _remote_write_context(profile, remote_name)
    if "error" in context:
        return context["error"]

    broker_request: dict[str, Any] = {"order_id": order_id}
    if symbol:
        broker_request["symbol"] = symbol

    adapter = MCPServerAdapter(context["server_name"], context["server"])
    try:
        result = adapter.call_tool(remote_name, broker_request)
    except Exception as exc:  # noqa: BLE001 - cancel audit should capture failures
        result = {"status": "error", "error": str(exc)}
    _audit_live_cancel(
        profile,
        order_id,
        symbol,
        result,
        session_id,
        remote_tool=remote_name,
        broker_request=broker_request,
    )
    return _with_profile(profile, result)


def _remote_order_arguments(
    *,
    symbol: str,
    side: str,
    quantity: float | None,
    notional: float | None,
    order_type: str,
    limit_price: float | None,
    time_in_force: str,
) -> dict[str, Any]:
    """Map generic trading tool order fields to Robinhood's current MCP schema."""
    args: dict[str, Any] = {
        "symbol": str(symbol or "").strip().upper(),
        "side": str(side or "").strip().lower(),
        "type": str(order_type or "market").strip().lower(),
        "time_in_force": str(time_in_force or "day").strip().lower(),
    }
    if notional is not None:
        args["dollar_amount"] = float(notional)
    if quantity is not None:
        args["quantity"] = float(quantity)
    if limit_price is not None:
        args["limit_price"] = float(limit_price)
    return args


def _remote_write_context(profile: TradingProfile, remote_name: str) -> dict[str, Any]:
    """Resolve a live remote-MCP write channel, enforcing config + OAuth gates."""
    from src.config.loader import load_agent_config
    from src.live.registry import has_cached_oauth_token

    server_name = str(profile.config.get("server") or profile.connector)
    server = (load_agent_config().mcp_servers or {}).get(server_name)
    if server is None:
        return {
            "error": {
                "status": "error",
                "profile_id": profile.id,
                "connector": profile.connector,
                "environment": profile.environment,
                "transport": profile.transport,
                "error": f"remote MCP server '{server_name}' is not configured",
            }
        }

    enabled_tools = list(getattr(server, "enabled_tools", None) or [])
    if "*" not in enabled_tools and remote_name not in enabled_tools:
        return {
            "error": {
                "status": "error",
                "profile_id": profile.id,
                "connector": profile.connector,
                "environment": profile.environment,
                "transport": profile.transport,
                "error": f"remote tool '{remote_name}' is not enabled for connector profile '{profile.id}'",
                "enabled_tools": enabled_tools,
            }
        }

    auth = getattr(server, "auth", None)
    if auth is None:
        return {
            "error": {
                "status": "error",
                "profile_id": profile.id,
                "connector": profile.connector,
                "environment": profile.environment,
                "transport": profile.transport,
                "error": f"connector profile '{profile.id}' has no OAuth auth configured",
            }
        }
    if not has_cached_oauth_token(server.url, auth.cache_dir):
        return {
            "error": {
                "status": "not_authorized",
                "profile_id": profile.id,
                "connector": profile.connector,
                "environment": profile.environment,
                "transport": profile.transport,
                "error": (
                    f"connector profile '{profile.id}' is not authorized. "
                    f"Run `vibe-trading connector authorize {profile.id}` from a desktop session."
                ),
            }
        }
    return {"server_name": server_name, "server": server}


def _json_object_payload(raw: Any) -> dict[str, Any]:
    """Return a dict payload from a JSON string or arbitrary object."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"status": "error", "error": "remote live order returned invalid JSON"}
        if isinstance(parsed, dict):
            return parsed
        return {"status": "ok", "result": parsed}
    return {"status": "ok", "result": raw}


def _audit_live_cancel(
    profile,
    order_id,
    symbol,
    result,
    session_id,
    *,
    remote_tool: str = "cancel_order",
    broker_request: dict[str, Any] | None = None,
) -> None:
    """Write a live-action audit record for a live order cancellation (best-effort)."""
    try:
        from src.live.audit import LiveActionEvent, write_live_action

        ok = isinstance(result, dict) and str(result.get("status", "")).lower() == "ok"
        event = LiveActionEvent(
            kind="order_cancelled",
            session_id=session_id,
            outcome="accepted" if ok else "error",
            server=profile.connector,
            remote_tool=remote_tool,
            intent_normalized=f"cancel {order_id} {symbol or ''}".strip(),
            mandate_snapshot_ref=None,
            consent_record_ref=None,
            broker_request=broker_request or {"order_id": order_id, "symbol": symbol},
            broker_response=result if isinstance(result, dict) else {"raw": result},
            gate_decision={"allowed": True, "decision": "cancel"},
            error=None if ok else (result.get("error") if isinstance(result, dict) else "cancel failed"),
        )
        try:
            write_live_action(event, event_callback=None, trace_writer=None)
        except TypeError:
            write_live_action(event)
    except Exception:  # noqa: BLE001 - auditing must never block a cancel
        pass


def profile_supports_live_runner(profile: TradingProfile) -> bool:
    """Return whether a profile can run the managed live runner."""
    return (
        profile.environment == "live"
        and profile.transport == "remote_mcp"
        and RUNNER_CAPABILITY in profile.capabilities
    )


def live_runner_profile_for_broker(broker: str) -> TradingProfile | None:
    """Return the live-runner profile for a broker, if one exists."""
    key = str(broker or "").strip().lower()
    if not key:
        return None
    for profile in list_profiles():
        if profile.connector == key and profile_supports_live_runner(profile):
            return profile
    return None


def broker_supports_live_runner(broker: str) -> bool:
    """Return whether any configured profile exposes live runner management."""
    return live_runner_profile_for_broker(broker) is not None


def connector_profile_id_for_broker(broker: str) -> str:
    """Return the preferred connector profile id for a broker on-ramp."""
    key = str(broker or "").strip().lower()
    if not key:
        raise ValueError("broker must not be blank")

    candidates = [profile for profile in list_profiles() if profile.connector == key and profile.environment == "live"]
    for profile in candidates:
        if profile.transport == "remote_mcp":
            return profile.id
    if candidates:
        return candidates[0].id
    return f"{key}-live-mcp"


def runner_tool_name(connector: str, operation: str) -> str | None:
    """Map a runner operation to a connector-specific remote MCP tool name."""
    if connector == "robinhood":
        from src.trading.connectors.robinhood.mcp import runner_tool_name as _runner_tool_name

        return _runner_tool_name(operation)
    return None


def _with_profile(profile: TradingProfile, payload: dict[str, Any]) -> dict[str, Any]:
    """Add connector profile metadata to an operation payload."""
    result = dict(payload)
    result["profile_id"] = profile.id
    result["connector"] = profile.connector
    result["environment"] = profile.environment
    result["transport"] = profile.transport
    return result


def _ibkr_config(profile: TradingProfile, overrides: dict[str, Any]):
    """Build an IBKR local config from a trading profile and call overrides."""
    from src.trading.connectors.ibkr.local import IBKRLocalConfig, config_path, load_config

    default_cfg = IBKRLocalConfig.from_mapping(profile.config)
    base = load_config()
    if config_path().exists() and base.profile == default_cfg.profile:
        cfg = base
    else:
        cfg = default_cfg
    return cfg.with_overrides(
        host=_clean(overrides.get("host")),
        port=_int_or_none(overrides.get("port")),
        client_id=_int_or_none(overrides.get("client_id")),
        account=_clean(overrides.get("account")),
    )


def _remote_status(profile: TradingProfile) -> dict[str, Any]:
    """Return local authorization/config status for a remote MCP profile."""
    from src.config.loader import load_agent_config
    from src.live.registry import has_cached_oauth_token

    server_name = str(profile.config.get("server") or profile.connector)
    server = (load_agent_config().mcp_servers or {}).get(server_name)
    auth = getattr(server, "auth", None) if server is not None else None
    token_present = False
    if server is not None and auth is not None:
        token_present = has_cached_oauth_token(server.url, auth.cache_dir)
    return {
        "status": "ok" if token_present else "not_authorized",
        "profile_id": profile.id,
        "connector": profile.connector,
        "environment": profile.environment,
        "transport": profile.transport,
        "configured": server is not None,
        "oauth_token_present": token_present,
        "capabilities": list(profile.capabilities),
        "readonly": profile.readonly,
        "notes": profile.notes,
    }


def _call_remote(profile: TradingProfile, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a known read operation on a remote MCP connector profile."""
    from src.config.loader import load_agent_config
    from src.live.registry import has_cached_oauth_token
    from src.tools.mcp import MCPServerAdapter

    remote_name = _remote_tool_name(profile.connector, operation)
    if remote_name is None:
        return _unsupported(profile, f"{operation}.read")

    server_name = str(profile.config.get("server") or profile.connector)
    server = (load_agent_config().mcp_servers or {}).get(server_name)
    if server is None:
        return {
            "status": "error",
            "profile_id": profile.id,
            "connector": profile.connector,
            "environment": profile.environment,
            "transport": profile.transport,
            "error": f"remote MCP server '{server_name}' is not configured",
        }

    enabled_tools = list(getattr(server, "enabled_tools", None) or [])
    if "*" not in enabled_tools and remote_name not in enabled_tools:
        return {
            "status": "error",
            "profile_id": profile.id,
            "connector": profile.connector,
            "environment": profile.environment,
            "transport": profile.transport,
            "error": f"remote tool '{remote_name}' is not enabled for connector profile '{profile.id}'",
            "enabled_tools": enabled_tools,
        }

    auth = getattr(server, "auth", None)
    if profile.environment == "live" and auth is None:
        return {
            "status": "error",
            "profile_id": profile.id,
            "connector": profile.connector,
            "environment": profile.environment,
            "transport": profile.transport,
            "error": f"connector profile '{profile.id}' has no OAuth auth configured",
        }
    if auth is not None and not has_cached_oauth_token(server.url, auth.cache_dir):
        return {
            "status": "not_authorized",
            "profile_id": profile.id,
            "connector": profile.connector,
            "environment": profile.environment,
            "transport": profile.transport,
            "error": (
                f"connector profile '{profile.id}' is not authorized. "
                f"Run `vibe-trading connector authorize {profile.id}` from a desktop session."
            ),
        }

    try:
        remote_arguments = _remote_arguments(profile.connector, operation, arguments)
    except ValueError as exc:
        return {
            "status": "error",
            "profile_id": profile.id,
            "connector": profile.connector,
            "environment": profile.environment,
            "transport": profile.transport,
            "error": str(exc),
        }

    adapter = MCPServerAdapter(server_name, server)
    return _with_profile(
        profile,
        adapter.call_tool(remote_name, remote_arguments),
    )


def _remote_tool_name(connector: str, operation: str) -> str | None:
    """Map generic read operations to current remote MCP tool names."""
    if connector == "robinhood":
        from src.trading.connectors.robinhood.mcp import remote_tool_name

        return remote_tool_name(operation)
    return None


def _remote_arguments(connector: str, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize generic arguments for a remote MCP operation."""
    if connector == "robinhood":
        from src.trading.connectors.robinhood.mcp import remote_arguments

        return remote_arguments(operation, arguments)
    return {}


def _unsupported(profile: TradingProfile, capability: str) -> dict[str, Any]:
    """Return a standard unsupported-capability payload."""
    return {
        "status": "error",
        "profile_id": profile.id,
        "connector": profile.connector,
        "environment": profile.environment,
        "transport": profile.transport,
        "error": f"profile '{profile.id}' does not support {capability} through the generic trading tool yet",
        "capabilities": list(profile.capabilities),
    }


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
