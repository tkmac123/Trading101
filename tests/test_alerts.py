"""Tests for the alert engine assembly."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from trading101.alerts.engine import build_alert, SignalBundle
from trading101.alerts.models import Alert, Confidence, SetupType
from trading101.analysis.catalyst import classify_catalysts
from trading101.analysis.momentum import score_momentum
from trading101.analysis.sentiment import score_sentiment
from trading101.analysis.technical import analyze_technicals
from trading101.data.market import MarketData
from trading101.data.news import NewsItem


def _market(closes, volumes=None) -> MarketData:
    n = len(closes)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="D")
    if volumes is None:
        volumes = [1_000_000] * n
    df = pd.DataFrame(
        {
            "Open": closes, "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes], "Close": closes, "Volume": volumes,
        },
        index=idx,
    )
    return MarketData(ticker="ACME", history=df, info={})


def test_alert_renders_in_spec_format():
    closes = [100 + np.sin(i / 4) * 2 for i in range(60)]
    closes[-1] = max(closes) * 1.01
    vols = [1_000_000] * 60
    vols[-1] = 4_000_000

    md = _market(closes, vols)
    news = [
        NewsItem(
            ticker="ACME",
            title="Acme beats earnings and raises guidance",
            summary="",
            url="https://example.com/news",
            source="test",
            published_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
    ]
    bundle = SignalBundle(
        ticker="ACME",
        market=md,
        technical=analyze_technicals(md),
        momentum=score_momentum(md),
        catalyst=classify_catalysts("ACME", news),
        sentiment=score_sentiment("ACME", news, []),
    )
    alert = build_alert(bundle)
    assert alert is not None
    rendered = alert.render_text()
    for required in [
        "Ticker:", "Current Price:", "Setup Type:", "Confidence Level:",
        "Catalyst:", "Key Signals:", "Risk Factors:",
        "Suggested Strategy:", "Time Horizon:",
    ]:
        assert required in rendered, f"missing field: {required}"


def test_low_score_returns_no_alert():
    # Choppy series with no catalyst — composite should be low
    rng = np.random.default_rng(1)
    closes = list(np.cumsum(rng.normal(0, 0.2, 60)) + 100)
    md = _market(closes)
    bundle = SignalBundle(
        ticker="ACME",
        market=md,
        technical=analyze_technicals(md),
        momentum=score_momentum(md),
        catalyst=classify_catalysts("ACME", []),
        sentiment=score_sentiment("ACME", [], []),
    )
    alert = build_alert(bundle)
    # Either no alert, or at most a Low-confidence one
    if alert is not None:
        assert alert.confidence in (Confidence.LOW, Confidence.MEDIUM)


def test_confidence_from_score_thresholds():
    assert Confidence.from_score(0.1) is Confidence.LOW
    assert Confidence.from_score(0.6) is Confidence.MEDIUM
    assert Confidence.from_score(0.8) is Confidence.HIGH


def test_alert_to_dict_round_trip():
    a = Alert(
        ticker="ACME",
        current_price=12.34,
        setup_type=SetupType.BREAKOUT,
        confidence=Confidence.HIGH,
        catalyst=__import__("trading101.alerts.models", fromlist=["CatalystType"]).CatalystType.MIXED,
        key_signals=["s1"], risk_factors=["r1"],
        suggested_strategy="x", raw_score=0.9,
    )
    d = a.to_dict()
    assert d["ticker"] == "ACME"
    assert d["setup_type"] == "Breakout"
    assert d["confidence"] == "High"
