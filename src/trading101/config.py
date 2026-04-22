"""Runtime configuration loaded from environment / .env."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
STATE_DIR = DATA_DIR / "state"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN",
    "NFLX", "AVGO", "SMCI", "PLTR", "COIN", "MARA", "RIOT", "SOFI",
    "GME", "AMC", "ARM", "SNOW",
]


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [t.strip().upper() for t in value.split(",") if t.strip()]


@dataclass(frozen=True)
class Settings:
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    min_confidence: str = "medium"
    log_level: str = "INFO"

    polygon_api_key: str | None = None
    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    tradier_token: str | None = None
    finnhub_api_key: str | None = None

    newsapi_key: str | None = None
    benzinga_api_key: str | None = None

    twitter_bearer_token: str | None = None
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str = "trading101/0.1"
    stocktwits_token: str | None = None

    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None


def load_settings() -> Settings:
    return Settings(
        universe=_split_csv(os.getenv("TRADING101_UNIVERSE"), DEFAULT_UNIVERSE),
        min_confidence=os.getenv("TRADING101_MIN_CONFIDENCE", "medium").lower(),
        log_level=os.getenv("TRADING101_LOG_LEVEL", "INFO").upper(),
        polygon_api_key=os.getenv("POLYGON_API_KEY") or None,
        alpaca_api_key=os.getenv("ALPACA_API_KEY") or None,
        alpaca_api_secret=os.getenv("ALPACA_API_SECRET") or None,
        tradier_token=os.getenv("TRADIER_TOKEN") or None,
        finnhub_api_key=os.getenv("FINNHUB_API_KEY") or None,
        newsapi_key=os.getenv("NEWSAPI_KEY") or None,
        benzinga_api_key=os.getenv("BENZINGA_API_KEY") or None,
        twitter_bearer_token=os.getenv("TWITTER_BEARER_TOKEN") or None,
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID") or None,
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET") or None,
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "trading101/0.1"),
        stocktwits_token=os.getenv("STOCKTWITS_TOKEN") or None,
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
    )


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
