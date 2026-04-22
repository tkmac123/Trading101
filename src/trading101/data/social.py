"""Social sentiment adapters.

Free path: StockTwits public stream API (no auth, rate-limited). Premium
adapters (Twitter v2, Reddit / PRAW) require API keys and are stubbed but
called transparently when keys are present.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import requests

from ..config import load_settings

log = logging.getLogger(__name__)

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"


@dataclass
class SocialMention:
    ticker: str
    source: str
    body: str
    sentiment: str | None  # 'Bullish' | 'Bearish' | None (raw label if provided)
    created_at: datetime
    user: str | None = None


def fetch_stocktwits(ticker: str, *, limit: int = 30) -> list[SocialMention]:
    try:
        resp = requests.get(
            STOCKTWITS_URL.format(ticker=ticker),
            timeout=8,
            headers={"User-Agent": "trading101/0.1"},
        )
        if resp.status_code != 200:
            log.debug("StockTwits %s -> HTTP %s", ticker, resp.status_code)
            return []
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.debug("StockTwits fetch failed for %s: %s", ticker, exc)
        return []

    mentions: list[SocialMention] = []
    for msg in (payload.get("messages") or [])[:limit]:
        try:
            entities = msg.get("entities") or {}
            sentiment_obj = entities.get("sentiment") or {}
            sentiment = sentiment_obj.get("basic") if isinstance(sentiment_obj, dict) else None
            created = msg.get("created_at")
            try:
                created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                created_at = datetime.now(timezone.utc)
            mentions.append(
                SocialMention(
                    ticker=ticker,
                    source="StockTwits",
                    body=msg.get("body", ""),
                    sentiment=sentiment,
                    created_at=created_at,
                    user=(msg.get("user") or {}).get("username"),
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("Failed to parse stocktwits msg: %s", exc)
    return mentions


def fetch_twitter(ticker: str) -> list[SocialMention]:
    """Stub: Twitter v2 recent search. Activates only with TWITTER_BEARER_TOKEN."""
    settings = load_settings()
    if not settings.twitter_bearer_token:
        return []
    # Real implementation would call:
    # GET https://api.twitter.com/2/tweets/search/recent?query=${ticker} lang:en
    # with Bearer auth, then map results -> SocialMention.
    log.debug("Twitter adapter not implemented; skipping (key present).")
    return []


def fetch_reddit(ticker: str) -> list[SocialMention]:
    """Stub: Reddit /r/wallstreetbets via PRAW. Activates only with creds."""
    settings = load_settings()
    if not (settings.reddit_client_id and settings.reddit_client_secret):
        return []
    log.debug("Reddit adapter not implemented; skipping (creds present).")
    return []


def fetch_social_mentions(tickers: Iterable[str]) -> list[SocialMention]:
    out: list[SocialMention] = []
    for tk in tickers:
        out.extend(fetch_stocktwits(tk))
        out.extend(fetch_twitter(tk))
        out.extend(fetch_reddit(tk))
    return out
