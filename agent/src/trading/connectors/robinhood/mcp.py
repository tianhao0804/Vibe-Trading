"""Robinhood remote MCP generic-operation mapping."""

from __future__ import annotations

from typing import Any

_REMOTE_TOOL_NAMES = {
    "account": "get_portfolio",
    "positions": "get_equity_positions",
    "orders": "get_equity_orders",
    "quote": "get_equity_quotes",
}

_RUNNER_TOOL_NAMES = {
    "account": "get_portfolio",
    "positions": "get_equity_positions",
    "orders": "get_equity_orders",
    "quote": "get_equity_quotes",
    "submit_order": "place_equity_order",
    "cancel_order": "cancel_equity_order",
}

_ACCOUNT_SCOPED_OPERATIONS = {"account", "positions", "orders"}


def remote_tool_name(operation: str) -> str | None:
    """Return the Robinhood remote tool name for a generic operation."""
    return _REMOTE_TOOL_NAMES.get(operation)


def runner_tool_name(operation: str) -> str | None:
    """Return the Robinhood remote tool name used by live runner plumbing."""
    return _RUNNER_TOOL_NAMES.get(operation)


def remote_arguments(operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize generic arguments for a Robinhood remote MCP operation."""
    if operation in _ACCOUNT_SCOPED_OPERATIONS:
        account = arguments.get("account") or arguments.get("account_number")
        account_number = str(account).strip() if account is not None else ""
        if not account_number:
            raise ValueError(
                "Robinhood account, positions, and orders reads require an "
                "explicit account number. Pass --account <account_number> or "
                "the trading_* account parameter."
            )
        return {"account_number": account_number}
    if operation == "quote":
        symbol = arguments.get("symbol")
        symbols = arguments.get("symbols")
        return {"symbols": symbols or ([symbol] if symbol else [])}
    return {}
