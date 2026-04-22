"""Catalyst classifier — extracts and scores news catalysts per ticker."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..data.news import NewsItem


CATALYST_PATTERNS: dict[str, list[re.Pattern]] = {
    "earnings_surprise": [
        re.compile(r"\b(beats?|tops?|crush(?:e[sd])?|smash(?:e[sd])?)\b.*\b(estimat|expectation|consensus|earnings|EPS|revenue|guidance)\b", re.I),
        re.compile(r"\b(misses?|fell short of|disappoint(?:ed|ing|s)?)\b.*\b(estimat|expectation|consensus|earnings|EPS|revenue|guidance)\b", re.I),
        re.compile(r"\bguidance\b.*\b(rais|cut|lower|hik)\w*", re.I),
        re.compile(r"\b(rais|cut|lower|hik)\w*\b.*\bguidance\b", re.I),
    ],
    "fda_approval": [
        re.compile(r"\bFDA\b.*\b(approv\w*|clearance|reject\w*|denial|breakthrough designation)\b", re.I),
        re.compile(r"\bphase\s*(?:1|2|3|i{1,3})\b.*\b(trial|results)\b", re.I),
    ],
    "mna": [
        re.compile(r"\b(acquir\w*|merger|buyout|takeover|deal to acquire)\b", re.I),
        re.compile(r"\b(LBO|tender offer|spin[- ]off)\b", re.I),
        re.compile(r"\bto (?:acquire|buy)\s+(?:all of|the (?:remaining|outstanding)|\$)", re.I),
    ],
    "analyst_action": [
        re.compile(r"\b(upgrad|downgrad|initiat|reiterat)\w*\b.*\b(buy|sell|hold|outperform|underperform|overweight|neutral|price target)\b", re.I),
        re.compile(r"\bprice target\b.*\b(rais|cut|lower|hik)\w*", re.I),
    ],
    "partnership": [
        re.compile(r"\b(partnership|collaborat|joint venture|strategic alliance|signs deal)\b", re.I),
    ],
    "ai_tech_hype": [
        re.compile(r"\b(AI|artificial intelligence|GPU|LLM|generative)\b", re.I),
    ],
    "macro_sector": [
        re.compile(r"\b(Fed|FOMC|rate cut|rate hike|CPI|PCE|jobs report|nonfarm|GDP)\b", re.I),
        re.compile(r"\b(sector rotation|risk[- ]on|risk[- ]off)\b", re.I),
    ],
    "regulatory": [
        re.compile(r"\b(SEC|DOJ|investigation|probe|lawsuit|class action|antitrust)\b", re.I),
    ],
}

NEGATIVE_HINTS = re.compile(
    r"\b(miss(?:es|ed)?|disappoint|reject(?:ed)?|denial|cut|lower(?:s|ed)?|downgrad\w*|investigat\w*|probe|lawsuit|fraud|halt(?:ed)?|recall|warn(?:s|ed)?|bankruptc\w*)\b",
    re.I,
)
POSITIVE_HINTS = re.compile(
    r"\b(beat|tops?|crush\w*|approv\w*|breakthrough|surge|jump|soar|raises?|hike|upgrad\w*|outperform|partnership|wins?|record)\b",
    re.I,
)


@dataclass
class CatalystSignal:
    ticker: str
    categories: list[str] = field(default_factory=list)
    polarity: float = 0.0       # -1..+1
    freshness_hours: float = 999.0
    headline: str | None = None
    url: str | None = None
    items: list[NewsItem] = field(default_factory=list)

    @property
    def has_fresh_catalyst(self) -> bool:
        return bool(self.categories) and self.freshness_hours <= 48.0


def _classify(item: NewsItem) -> tuple[list[str], float]:
    text = item.text
    cats: list[str] = []
    for cat, patterns in CATALYST_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            cats.append(cat)

    pos = len(POSITIVE_HINTS.findall(text))
    neg = len(NEGATIVE_HINTS.findall(text))
    total = pos + neg
    polarity = 0.0 if total == 0 else (pos - neg) / total
    return cats, polarity


def classify_catalysts(ticker: str, items: list[NewsItem]) -> CatalystSignal:
    relevant = [it for it in items if it.ticker == ticker]
    if not relevant:
        return CatalystSignal(ticker=ticker)

    now = datetime.now(timezone.utc)
    relevant.sort(key=lambda n: n.published_at, reverse=True)

    all_cats: set[str] = set()
    polarity_sum = 0.0
    weighted_total = 0.0

    for it in relevant[:10]:
        cats, pol = _classify(it)
        age_h = (now - it.published_at).total_seconds() / 3600.0
        # Weight = exponential decay over 72h
        w = max(0.0, 1.0 - min(age_h / 72.0, 1.0))
        polarity_sum += pol * w
        weighted_total += w
        all_cats.update(cats)

    polarity = polarity_sum / weighted_total if weighted_total else 0.0
    headline = relevant[0]
    fresh_h = (now - headline.published_at).total_seconds() / 3600.0

    return CatalystSignal(
        ticker=ticker,
        categories=sorted(all_cats),
        polarity=round(polarity, 3),
        freshness_hours=round(fresh_h, 2),
        headline=headline.title,
        url=headline.url,
        items=relevant[:10],
    )
