"""Market data adapter.

Default backend: yfinance (free, delayed). The interface is provider-agnostic
so a Polygon/Alpaca/Tradier adapter can be dropped in without touching the
analysis layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Snapshot of recent market activity for one ticker."""

    ticker: str
    history: pd.DataFrame  # daily OHLCV, most-recent last
    intraday: pd.DataFrame | None = None  # 1m bars for today (optional)
    info: dict = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def last_price(self) -> float:
        if self.intraday is not None and not self.intraday.empty:
            return float(self.intraday["Close"].iloc[-1])
        return float(self.history["Close"].iloc[-1])

    @property
    def previous_close(self) -> float:
        return float(self.history["Close"].iloc[-2]) if len(self.history) >= 2 else self.last_price

    @property
    def pct_change_today(self) -> float:
        prev = self.previous_close
        if prev == 0:
            return 0.0
        return (self.last_price - prev) / prev * 100.0

    @property
    def avg_volume_30d(self) -> float:
        if "Volume" not in self.history or len(self.history) < 5:
            return 0.0
        return float(self.history["Volume"].tail(30).mean())

    @property
    def relative_volume(self) -> float:
        avg = self.avg_volume_30d
        if avg == 0:
            return 0.0
        last_vol = float(self.history["Volume"].iloc[-1])
        return last_vol / avg

    @property
    def market_cap(self) -> float | None:
        return self.info.get("marketCap")

    @property
    def float_shares(self) -> float | None:
        return self.info.get("floatShares")

    @property
    def short_percent_of_float(self) -> float | None:
        return self.info.get("shortPercentOfFloat")


def get_market_data(
    ticker: str,
    *,
    history_days: int = 120,
    include_intraday: bool = False,
) -> MarketData | None:
    """Pull OHLCV history (and optional intraday) for a ticker via yfinance."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{history_days}d", interval="1d", auto_adjust=False)
        if hist.empty:
            log.warning("No history returned for %s", ticker)
            return None
        intraday = None
        if include_intraday:
            try:
                intraday = t.history(period="1d", interval="1m", auto_adjust=False)
                if intraday.empty:
                    intraday = None
            except Exception as exc:  # noqa: BLE001
                log.debug("Intraday fetch failed for %s: %s", ticker, exc)

        info: dict = {}
        try:
            # .info can be slow / rate-limited; treat as best-effort
            info = t.info or {}
        except Exception as exc:  # noqa: BLE001
            log.debug("Info fetch failed for %s: %s", ticker, exc)

        return MarketData(ticker=ticker, history=hist, intraday=intraday, info=info)
    except Exception as exc:  # noqa: BLE001
        log.error("Market data fetch failed for %s: %s", ticker, exc)
        return None


def get_many(tickers: list[str], **kwargs) -> dict[str, MarketData]:
    """Batch fetch — sequential to stay polite to the free API."""
    out: dict[str, MarketData] = {}
    for tk in tickers:
        md = get_market_data(tk, **kwargs)
        if md is not None:
            out[tk] = md
    return out
