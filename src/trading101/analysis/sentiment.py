"""Sentiment scoring across news + social using VADER."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..data.news import NewsItem
from ..data.social import SocialMention

_analyzer = SentimentIntensityAnalyzer()


@dataclass
class SentimentSignal:
    ticker: str
    news_score: float       # -1..+1
    social_score: float     # -1..+1
    mentions_24h: int
    hype_velocity: float    # mentions in last 6h / mentions in prior 18h
    label: str              # 'bullish' | 'bearish' | 'mixed' | 'neutral' | 'hype'

    @property
    def composite(self) -> float:
        return round(0.6 * self.news_score + 0.4 * self.social_score, 3)


def _vader(text: str) -> float:
    if not text:
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]


def _label(news: float, social: float, hype_velocity: float, mentions: int) -> str:
    if mentions >= 20 and hype_velocity >= 2.5:
        return "hype"
    composite = 0.6 * news + 0.4 * social
    if composite >= 0.25:
        return "bullish"
    if composite <= -0.25:
        return "bearish"
    if abs(news - social) >= 0.4:
        return "mixed"
    return "neutral"


def score_sentiment(
    ticker: str,
    news_items: list[NewsItem],
    social_mentions: list[SocialMention],
) -> SentimentSignal:
    now = datetime.now(timezone.utc)

    relevant_news = [n for n in news_items if n.ticker == ticker][:25]
    relevant_social = [s for s in social_mentions if s.ticker == ticker]

    news_scores = [_vader(n.text) for n in relevant_news]
    news_score = round(sum(news_scores) / len(news_scores), 3) if news_scores else 0.0

    social_scores: list[float] = []
    for s in relevant_social:
        if s.sentiment == "Bullish":
            social_scores.append(0.6)
        elif s.sentiment == "Bearish":
            social_scores.append(-0.6)
        else:
            social_scores.append(_vader(s.body))
    social_score = round(sum(social_scores) / len(social_scores), 3) if social_scores else 0.0

    last_24h = [s for s in relevant_social if (now - s.created_at).total_seconds() <= 86400]
    last_6h = [s for s in last_24h if (now - s.created_at).total_seconds() <= 21600]
    prior_18h = len(last_24h) - len(last_6h)
    hype_velocity = (len(last_6h) / prior_18h) if prior_18h else float(len(last_6h))

    label = _label(news_score, social_score, hype_velocity, len(last_24h))

    return SentimentSignal(
        ticker=ticker,
        news_score=news_score,
        social_score=social_score,
        mentions_24h=len(last_24h),
        hype_velocity=round(hype_velocity, 2),
        label=label,
    )
