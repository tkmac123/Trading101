"""Tests for the technical analysis engine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from trading101.analysis.technical import analyze_technicals, ema, rsi, sma
from trading101.data.market import MarketData


def _make_md(closes, volumes=None) -> MarketData:
    n = len(closes)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="D")
    if volumes is None:
        volumes = [1_000_000] * n
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": volumes,
        },
        index=idx,
    )
    return MarketData(ticker="TEST", history=df)


def test_sma_and_ema_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sma(s, 3).iloc[-1] == 4.0
    e = ema(s, 3).iloc[-1]
    assert 3.5 < e < 5.0  # ema biases toward recent values


def test_rsi_bounds():
    rng = np.random.default_rng(42)
    series = pd.Series(np.cumsum(rng.normal(0, 1, 200)) + 100)
    r = rsi(series, 14)
    assert ((r >= 0) & (r <= 100)).all()


def test_breakout_setup_detected():
    # Flat then a sharp push toward the 20-day high on big volume
    base = [100 + np.sin(i / 3) for i in range(40)]
    base[-1] = max(base) * 1.005  # right at resistance
    vols = [1_000_000] * 40
    vols[-1] = 5_000_000  # volume spike
    snap = analyze_technicals(_make_md(base, vols))
    assert snap is not None
    assert "near_resistance" in snap.patterns or "breakout_confirmed" in snap.patterns
    assert snap.rel_volume > 1.3


def test_capitulation_detected():
    # Big drop then stabilization
    closes = [100.0] * 30 + [85.0, 80.0, 79.0, 79.5, 79.8, 79.6]
    snap = analyze_technicals(_make_md(closes))
    assert snap is not None
    assert "capitulation" in snap.patterns


def test_score_in_range():
    rng = np.random.default_rng(7)
    closes = list(np.cumsum(rng.normal(0, 1, 80)) + 100)
    snap = analyze_technicals(_make_md(closes))
    assert snap is not None
    assert -1.0 <= snap.bullish_score <= 1.0
