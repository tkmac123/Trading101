"""Momentum scanner — detects 3–5 day burst potential.

Idea: find tickers that historically pop 8–40% in 3–5 day windows AND whose
current state (relative volume, pre-market gap, short-term trend) resembles
the typical pre-burst signature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.market import MarketData


@dataclass
class MomentumSignal:
    ticker: str
    score: float                # 0..1
    historical_burst_rate: float  # fraction of 5d windows with >=8% gain
    median_burst_size_pct: float  # median size of those bursts
    last_5d_return_pct: float
    pre_market_gap_pct: float
    rel_volume: float
    trend_alignment: bool
    notes: list[str]


def _historical_burst_stats(close: pd.Series, window: int = 5, threshold: float = 8.0) -> tuple[float, float]:
    """Fraction of `window`-day forward returns >= threshold% and median of those."""
    if len(close) < window + 30:
        return 0.0, 0.0
    fwd = close.shift(-window) / close - 1.0
    fwd_pct = fwd.dropna() * 100.0
    bursts = fwd_pct[fwd_pct >= threshold]
    rate = len(bursts) / max(1, len(fwd_pct))
    median = float(bursts.median()) if len(bursts) else 0.0
    return float(rate), median


def _pre_market_gap(md: MarketData) -> float:
    if md.intraday is None or md.intraday.empty:
        return 0.0
    open_today = float(md.intraday["Open"].iloc[0])
    prev_close = md.previous_close
    if prev_close == 0:
        return 0.0
    return (open_today - prev_close) / prev_close * 100.0


def score_momentum(md: MarketData) -> MomentumSignal | None:
    df = md.history.dropna()
    if len(df) < 35:
        return None

    close = df["Close"]
    burst_rate, burst_median = _historical_burst_stats(close, window=5, threshold=8.0)

    last_5d_return = float((close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0) if len(close) >= 6 else 0.0
    pm_gap = _pre_market_gap(md)
    rvol = md.relative_volume

    # Trend alignment: price > 20MA > 50MA
    sma20 = close.tail(20).mean()
    sma50 = close.tail(50).mean() if len(close) >= 50 else sma20
    trend_aligned = bool(close.iloc[-1] > sma20 > sma50)

    notes: list[str] = []
    score = 0.0

    if burst_rate >= 0.10:
        score += 0.25
        notes.append(f"historical 5d burst rate {burst_rate*100:.0f}% (median +{burst_median:.1f}%)")

    if rvol >= 2.0:
        score += 0.25
        notes.append(f"relative volume {rvol:.1f}x")
    elif rvol >= 1.3:
        score += 0.10

    if pm_gap >= 5.0:
        score += 0.20
        notes.append(f"pre-market gap +{pm_gap:.1f}%")
    elif pm_gap <= -5.0:
        notes.append(f"pre-market gap {pm_gap:.1f}% (caution)")

    if trend_aligned:
        score += 0.15
        notes.append("trend aligned (price > 20MA > 50MA)")

    if 1.0 <= last_5d_return <= 6.0:
        # Sweet spot — already moving but not extended
        score += 0.15
        notes.append(f"early 5d move +{last_5d_return:.1f}%")
    elif last_5d_return > 15.0:
        # Already extended — burst probably already happened
        score -= 0.10
        notes.append(f"extended 5d move +{last_5d_return:.1f}%")

    score = float(np.clip(score, 0.0, 1.0))

    return MomentumSignal(
        ticker=md.ticker,
        score=score,
        historical_burst_rate=burst_rate,
        median_burst_size_pct=burst_median,
        last_5d_return_pct=last_5d_return,
        pre_market_gap_pct=pm_gap,
        rel_volume=rvol,
        trend_alignment=trend_aligned,
        notes=notes,
    )
