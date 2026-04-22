"""Tests for the news catalyst classifier."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading101.analysis.catalyst import classify_catalysts
from trading101.data.news import NewsItem


def _item(title: str, ticker: str = "ACME", hours_ago: float = 2.0) -> NewsItem:
    return NewsItem(
        ticker=ticker,
        title=title,
        summary="",
        url="https://example.com",
        source="test",
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
    )


def test_earnings_beat_positive_polarity():
    items = [_item("Acme beats earnings estimates and raises guidance")]
    sig = classify_catalysts("ACME", items)
    assert "earnings_surprise" in sig.categories
    assert sig.polarity > 0
    assert sig.has_fresh_catalyst


def test_fda_approval_classified():
    items = [_item("FDA grants approval for Acme drug")]
    sig = classify_catalysts("ACME", items)
    assert "fda_approval" in sig.categories


def test_negative_polarity_on_miss():
    items = [_item("Acme misses revenue estimates and lowers guidance")]
    sig = classify_catalysts("ACME", items)
    assert sig.polarity < 0


def test_no_relevant_news_returns_empty_signal():
    items = [_item("Some other ticker news", ticker="OTHER")]
    sig = classify_catalysts("ACME", items)
    assert sig.categories == []
    assert sig.headline is None
    assert not sig.has_fresh_catalyst


def test_freshness_calculated():
    items = [
        _item("Recent headline", hours_ago=1.0),
        _item("Older headline", hours_ago=72.0),
    ]
    sig = classify_catalysts("ACME", items)
    # Most-recent headline is the one we expose
    assert sig.headline == "Recent headline"
    assert sig.freshness_hours < 5.0
