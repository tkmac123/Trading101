"""Alpaca broker integration for paper and live trading."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

log = logging.getLogger(__name__)


@dataclass
class AlpacaAccount:
    """Account state snapshot."""

    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    is_paper: bool
    last_equity: float


@dataclass
class AlpacaPosition:
    """Open position snapshot."""

    symbol: str
    qty: float
    avg_fill_price: float
    current_price: float
    side: Literal["long", "short"]
    unrealized_pl: float
    unrealized_plpc: float


@dataclass
class AlpacaOrder:
    """Order submission result."""

    id: str
    symbol: str
    qty: float
    side: str  # "buy" or "sell"
    order_type: str  # "market" or "limit"
    price: float | None  # for limit orders
    status: str
    created_at: datetime


class AlpacaClient:
    """Wrapper around alpaca-py TradingClient."""

    def __init__(self, api_key: str, api_secret: str, base_url: str | None = None):
        """
        Initialize Alpaca client.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: Override base URL (defaults to paper trading)
        """
        self.client = TradingClient(api_key=api_key, secret_key=api_secret, paper=True)
        self._is_paper = True
        if base_url:
            self.client.base_url = base_url
            self._is_paper = "paper" in base_url.lower()

    def get_account(self) -> AlpacaAccount | None:
        """Fetch current account state."""
        try:
            acc = self.client.get_account()
            return AlpacaAccount(
                account_id=acc.id,
                buying_power=float(acc.buying_power),
                cash=float(acc.cash),
                portfolio_value=float(acc.portfolio_value),
                is_paper=self._is_paper,
                last_equity=float(acc.last_equity),
            )
        except Exception as exc:
            log.error("Failed to fetch account: %s", exc)
            return None

    def get_positions(self) -> list[AlpacaPosition] | None:
        """Fetch all open positions."""
        try:
            positions = self.client.get_all_positions()
            out = []
            for p in positions:
                out.append(AlpacaPosition(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_fill_price=float(p.avg_fill_price),
                    current_price=float(p.current_price),
                    side="long" if float(p.qty) > 0 else "short",
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_plpc=float(p.unrealized_plpc),
                ))
            return out
        except Exception as exc:
            log.error("Failed to fetch positions: %s", exc)
            return None

    def get_position(self, symbol: str) -> AlpacaPosition | None:
        """Fetch single position by symbol."""
        try:
            p = self.client.get_open_position(symbol)
            return AlpacaPosition(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_fill_price=float(p.avg_fill_price),
                current_price=float(p.current_price),
                side="long" if float(p.qty) > 0 else "short",
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
            )
        except Exception as exc:
            log.debug("No position for %s: %s", symbol, exc)
            return None

    def get_quotes(self, symbols: list[str]) -> dict[str, float] | None:
        """Fetch latest quotes for symbols.

        Returns dict of {symbol: price}.
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest

            client = StockHistoricalDataClient(self.client.api_key, self.client.secret_key)
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = client.get_stock_latest_quote(request)

            out = {}
            for symbol in symbols:
                if symbol in quotes:
                    out[symbol] = float(quotes[symbol].ask_price)
            return out
        except Exception as exc:
            log.error("Failed to fetch quotes: %s", exc)
            return None

    def place_market_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
    ) -> AlpacaOrder | None:
        """Place a market order."""
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=int(qty),
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(req)
            return AlpacaOrder(
                id=order.id,
                symbol=order.symbol,
                qty=float(order.qty),
                side=order.side.value,
                order_type="market",
                price=None,
                status=order.status.value,
                created_at=order.created_at,
            )
        except Exception as exc:
            log.error("Failed to place market order %s %s: %s", side, symbol, exc)
            return None

    def place_limit_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
        limit_price: float,
    ) -> AlpacaOrder | None:
        """Place a limit order."""
        try:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=int(qty),
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                limit_price=limit_price,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(req)
            return AlpacaOrder(
                id=order.id,
                symbol=order.symbol,
                qty=float(order.qty),
                side=order.side.value,
                order_type="limit",
                price=limit_price,
                status=order.status.value,
                created_at=order.created_at,
            )
        except Exception as exc:
            log.error("Failed to place limit order %s %s @ $%s: %s", side, symbol, limit_price, exc)
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID."""
        try:
            self.client.cancel_order_by_id(order_id)
            log.info("Cancelled order %s", order_id)
            return True
        except Exception as exc:
            log.error("Failed to cancel order %s: %s", order_id, exc)
            return False
