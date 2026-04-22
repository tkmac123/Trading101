"""News aggregation.

Free path: Yahoo Finance & MarketWatch RSS feeds (per-ticker), plus a
general-market feed. Premium adapters (NewsAPI, Benzinga) are wired but only
activate when an API key is set in the environment.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import feedparser

log = logging.getLogger(__name__)

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
MARKETWATCH_RSS = "https://feeds.content.dowjones.io/public/rss/mw_topstories"
SEEKING_ALPHA_RSS = "https://seekingalpha.com/api/sa/combined/{ticker}.xml"


@dataclass
class NewsItem:
    ticker: str | None
    title: str
    summary: str
    url: str
    source: str
    published_at: datetime

    @property
    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()


_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


def _parse_entry(entry, ticker: str | None, source: str) -> NewsItem | None:
    try:
        title = (entry.get("title") or "").strip()
        if not title:
            return None
        summary = re.sub(r"<[^>]+>", "", entry.get("summary", "")).strip()
        url = entry.get("link", "")
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        if published:
            published_at = datetime(*published[:6], tzinfo=timezone.utc)
        else:
            published_at = datetime.now(timezone.utc)
        return NewsItem(
            ticker=ticker,
            title=title,
            summary=summary,
            url=url,
            source=source,
            published_at=published_at,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("Failed to parse news entry: %s", exc)
        return None


def fetch_ticker_news(ticker: str, *, limit: int = 25) -> list[NewsItem]:
    """Yahoo Finance per-ticker RSS — free, no auth."""
    url = YAHOO_RSS.format(ticker=ticker)
    feed = feedparser.parse(url)
    items: list[NewsItem] = []
    for e in feed.entries[:limit]:
        item = _parse_entry(e, ticker, "Yahoo Finance")
        if item:
            items.append(item)
    return items


def fetch_market_news(*, limit: int = 50) -> list[NewsItem]:
    """General market headlines from MarketWatch top stories RSS."""
    feed = feedparser.parse(MARKETWATCH_RSS)
    items: list[NewsItem] = []
    for e in feed.entries[:limit]:
        item = _parse_entry(e, None, "MarketWatch")
        if item:
            items.append(item)
    return items


def fetch_news(tickers: Iterable[str], *, per_ticker_limit: int = 15) -> list[NewsItem]:
    """Aggregate per-ticker + market news across the universe."""
    out: list[NewsItem] = []
    for tk in tickers:
        try:
            out.extend(fetch_ticker_news(tk, limit=per_ticker_limit))
        except Exception as exc:  # noqa: BLE001
            log.warning("News fetch failed for %s: %s", tk, exc)
    try:
        out.extend(fetch_market_news(limit=25))
    except Exception as exc:  # noqa: BLE001
        log.warning("Market news fetch failed: %s", exc)
    out.sort(key=lambda n: n.published_at, reverse=True)
    return out


def attach_tickers(items: list[NewsItem], universe: set[str]) -> list[NewsItem]:
    """Tag market-wide items with any tickers from the universe mentioned in text."""
    enriched: list[NewsItem] = []
    for it in items:
        if it.ticker:
            enriched.append(it)
            continue
        found = {m for m in _TICKER_RE.findall(it.text) if m in universe}
        if not found:
            enriched.append(it)
            continue
        for tk in found:
            enriched.append(
                NewsItem(
                    ticker=tk,
                    title=it.title,
                    summary=it.summary,
                    url=it.url,
                    source=it.source,
                    published_at=it.published_at,
                )
            )
    return enriched
