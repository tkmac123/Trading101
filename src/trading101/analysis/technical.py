"""Technical analysis: indicators + pattern detection on daily OHLCV."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.market import MarketData


# ---------- indicators (pure functions, no external TA library) ----------

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ---------- snapshot ----------

@dataclass
class TechnicalSnapshot:
    ticker: str
    close: float
    rsi14: float
    sma20: float
    sma50: float
    ema9: float
    ema21: float
    atr14: float
    rel_volume: float
    resistance_20d: float
    support_20d: float
    distance_to_resistance_pct: float
    distance_to_support_pct: float
    patterns: list[str] = field(default_factory=list)
    bullish_score: float = 0.0  # -1.0..+1.0

    @property
    def is_breakout_setup(self) -> bool:
        return (
            "near_resistance" in self.patterns
            and self.rel_volume >= 1.3
            and self.rsi14 < 75
        )

    @property
    def is_capitulation_setup(self) -> bool:
        return "capitulation" in self.patterns and self.rsi14 < 35

    @property
    def is_bull_flag(self) -> bool:
        return "bull_flag" in self.patterns


# ---------- pattern detectors ----------

def _detect_patterns(df: pd.DataFrame, last: pd.Series) -> list[str]:
    patterns: list[str] = []
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    # Resistance / support — 20-day high/low excluding today
    if len(close) >= 22:
        recent = close.iloc[-21:-1]
        resistance = float(recent.max())
        support = float(recent.min())
        last_close = float(last["Close"])
        if last_close >= resistance * 0.98:
            patterns.append("near_resistance")
        if last_close > resistance and float(last["Volume"]) > vol.tail(20).mean() * 1.3:
            patterns.append("breakout_confirmed")
        if last_close <= support * 1.02:
            patterns.append("near_support")

    # Volume accumulation: last 5 vols all above 20-day avg
    if len(vol) >= 25:
        avg_vol = vol.tail(25).iloc[:-5].mean()
        if (vol.tail(5) > avg_vol).all():
            patterns.append("volume_accumulation")

    # Capitulation: sharp drop from a recent peak followed by stabilization
    if len(close) >= 8:
        pre_window = close.iloc[-8:-3]
        last_3 = close.tail(3)
        drop = (last_3.mean() / pre_window.max() - 1.0) * 100.0
        stabilization = (last_3.max() - last_3.min()) / last_3.mean() * 100.0
        if drop <= -8.0 and stabilization < 3.0:
            patterns.append("capitulation")

    # Bull flag: strong run-up followed by tight low-volume consolidation
    if len(close) >= 15:
        run = close.iloc[-15:-5]
        flag = close.tail(5)
        run_return = (run.iloc[-1] / run.iloc[0] - 1.0) * 100.0
        flag_range = (flag.max() - flag.min()) / flag.mean() * 100.0
        flag_vol_ratio = vol.tail(5).mean() / vol.iloc[-15:-5].mean()
        if run_return >= 8.0 and flag_range < 4.0 and flag_vol_ratio < 1.0:
            patterns.append("bull_flag")

    # Consolidation: tight range over 10 days
    if len(close) >= 10:
        win = close.tail(10)
        if (win.max() - win.min()) / win.mean() * 100.0 < 3.5:
            patterns.append("consolidation")

    # RSI divergence: lower lows in price, higher lows in RSI (or vice versa)
    if len(close) >= 30:
        r = rsi(close, 14)
        price_lows = close.tail(20).nsmallest(2).sort_index()
        rsi_lows = r.tail(20).nsmallest(2).sort_index()
        if (
            len(price_lows) == 2
            and len(rsi_lows) == 2
            and price_lows.iloc[1] < price_lows.iloc[0]
            and rsi_lows.iloc[1] > rsi_lows.iloc[0]
        ):
            patterns.append("bullish_divergence")

    return patterns


def _detect_crossovers(df: pd.DataFrame) -> list[str]:
    patterns: list[str] = []
    close = df["Close"]
    if len(close) < 51:
        return patterns
    e9 = ema(close, 9)
    e21 = ema(close, 21)
    s50 = sma(close, 50)

    # Golden / death cross of EMAs
    if e9.iloc[-2] <= e21.iloc[-2] and e9.iloc[-1] > e21.iloc[-1]:
        patterns.append("ema9_cross_above_ema21")
    if e9.iloc[-2] >= e21.iloc[-2] and e9.iloc[-1] < e21.iloc[-1]:
        patterns.append("ema9_cross_below_ema21")

    # Reclaim of 50 SMA
    if close.iloc[-2] <= s50.iloc[-2] and close.iloc[-1] > s50.iloc[-1]:
        patterns.append("reclaim_sma50")

    return patterns


def analyze_technicals(md: MarketData) -> TechnicalSnapshot | None:
    df = md.history.dropna()
    if len(df) < 30:
        return None

    close = df["Close"]
    last = df.iloc[-1]

    s20 = float(sma(close, 20).iloc[-1])
    s50 = float(sma(close, 50).iloc[-1]) if len(close) >= 50 else float(sma(close, len(close)).iloc[-1])
    e9 = float(ema(close, 9).iloc[-1])
    e21 = float(ema(close, 21).iloc[-1])
    r14 = float(rsi(close, 14).iloc[-1])
    a14 = float(atr(df["High"], df["Low"], close, 14).iloc[-1])

    recent = close.iloc[-21:-1] if len(close) >= 22 else close.iloc[:-1]
    resistance = float(recent.max())
    support = float(recent.min())
    last_close = float(last["Close"])
    dist_res = (resistance - last_close) / last_close * 100.0
    dist_sup = (last_close - support) / last_close * 100.0

    patterns = _detect_patterns(df, last) + _detect_crossovers(df)

    score = 0.0
    score += 0.15 if e9 > e21 else -0.15
    score += 0.10 if last_close > s20 else -0.10
    score += 0.10 if last_close > s50 else -0.10
    if r14 < 30:
        score += 0.20  # oversold bounce potential
    elif r14 > 70:
        score -= 0.10  # overbought caution
    if "breakout_confirmed" in patterns:
        score += 0.30
    if "near_resistance" in patterns and md.relative_volume > 1.3:
        score += 0.15
    if "capitulation" in patterns:
        score += 0.20
    if "bull_flag" in patterns:
        score += 0.20
    if "bullish_divergence" in patterns:
        score += 0.15
    if "ema9_cross_above_ema21" in patterns or "reclaim_sma50" in patterns:
        score += 0.10
    if "ema9_cross_below_ema21" in patterns:
        score -= 0.15
    score = max(-1.0, min(1.0, score))

    return TechnicalSnapshot(
        ticker=md.ticker,
        close=last_close,
        rsi14=r14,
        sma20=s20,
        sma50=s50,
        ema9=e9,
        ema21=e21,
        atr14=a14,
        rel_volume=md.relative_volume,
        resistance_20d=resistance,
        support_20d=support,
        distance_to_resistance_pct=dist_res,
        distance_to_support_pct=dist_sup,
        patterns=patterns,
        bullish_score=score,
    )
