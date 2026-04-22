"""Alert engine — fuses technical / momentum / catalyst / sentiment signals."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..analysis.catalyst import CatalystSignal, classify_catalysts
from ..analysis.momentum import MomentumSignal, score_momentum
from ..analysis.sentiment import SentimentSignal, score_sentiment
from ..analysis.technical import TechnicalSnapshot, analyze_technicals
from ..data.market import MarketData, get_market_data
from ..data.news import NewsItem, attach_tickers, fetch_news
from ..data.social import SocialMention, fetch_social_mentions
from .models import Alert, CatalystType, Confidence, SetupType, TimeHorizon

log = logging.getLogger(__name__)


# Default signal weights — the LearningTracker can override these.
DEFAULT_WEIGHTS = {
    "technical": 0.35,
    "momentum": 0.30,
    "catalyst": 0.20,
    "sentiment": 0.15,
}


@dataclass
class SignalBundle:
    ticker: str
    market: MarketData
    technical: TechnicalSnapshot | None
    momentum: MomentumSignal | None
    catalyst: CatalystSignal
    sentiment: SentimentSignal


def _composite_score(bundle: SignalBundle, weights: dict[str, float]) -> float:
    t = bundle.technical
    m = bundle.momentum
    tech_score = max(0.0, t.bullish_score) if t else 0.0
    mom_score = m.score if m else 0.0
    # Catalyst score: presence + positive polarity + freshness
    cat = bundle.catalyst
    cat_score = 0.0
    if cat.has_fresh_catalyst:
        cat_score = 0.4 + 0.6 * max(0.0, cat.polarity)
    sent_score = max(0.0, (bundle.sentiment.composite + 1) / 2.0)  # remap -1..1 -> 0..1, only positive helps

    return (
        weights["technical"] * tech_score
        + weights["momentum"] * mom_score
        + weights["catalyst"] * cat_score
        + weights["sentiment"] * sent_score
    )


def _classify_setup(bundle: SignalBundle) -> SetupType:
    t = bundle.technical
    m = bundle.momentum
    if t and t.is_capitulation_setup:
        return SetupType.CAPITULATION
    if t and (t.is_breakout_setup or "breakout_confirmed" in t.patterns):
        return SetupType.BREAKOUT
    if m and m.pre_market_gap_pct >= 5.0:
        return SetupType.PRE_MARKET
    if bundle.sentiment.label == "hype":
        return SetupType.HYPE
    if (m and m.rel_volume >= 2.5) or (t and t.rel_volume >= 2.5):
        return SetupType.UNUSUAL_ACTIVITY
    return SetupType.MOMENTUM


def _classify_catalyst_source(bundle: SignalBundle) -> CatalystType:
    has_news = bundle.catalyst.has_fresh_catalyst
    has_social = bundle.sentiment.mentions_24h >= 10 or bundle.sentiment.label == "hype"
    has_tech = bool(bundle.technical and bundle.technical.bullish_score >= 0.3)
    flags = sum([has_news, has_social, has_tech])
    if flags >= 2:
        return CatalystType.MIXED
    if has_news:
        return CatalystType.NEWS
    if has_social:
        return CatalystType.SOCIAL
    return CatalystType.TECHNICAL


def _key_signals(bundle: SignalBundle) -> list[str]:
    out: list[str] = []
    t = bundle.technical
    m = bundle.momentum

    if t:
        if t.patterns:
            out.append("Patterns: " + ", ".join(t.patterns))
        out.append(f"RSI(14): {t.rsi14:.1f}")
        if t.distance_to_resistance_pct <= 3.0:
            out.append(f"Within {t.distance_to_resistance_pct:.1f}% of 20-day resistance ${t.resistance_20d:,.2f}")
        if t.rel_volume >= 1.3:
            out.append(f"Relative volume {t.rel_volume:.1f}x average")

    if m:
        out.extend(m.notes)

    cat = bundle.catalyst
    if cat.categories:
        out.append("News catalysts: " + ", ".join(cat.categories))
    if cat.headline:
        out.append(f"Latest headline ({cat.freshness_hours:.0f}h ago): {cat.headline[:120]}")

    s = bundle.sentiment
    if s.mentions_24h >= 5:
        out.append(
            f"Social: {s.mentions_24h} mentions/24h, "
            f"hype velocity {s.hype_velocity:.1f}x, label={s.label}, composite={s.composite:+.2f}"
        )
    return out


def _risk_factors(bundle: SignalBundle) -> list[str]:
    risks: list[str] = []
    t = bundle.technical
    m = bundle.momentum
    s = bundle.sentiment
    cat = bundle.catalyst

    if t:
        if t.rsi14 >= 75:
            risks.append(f"RSI overbought at {t.rsi14:.1f} — risk of pullback")
        if t.distance_to_support_pct >= 8.0:
            risks.append(f"Support far below ({t.distance_to_support_pct:.1f}% to ${t.support_20d:,.2f})")
        if t.atr14 / max(t.close, 0.01) > 0.05:
            risks.append(f"High volatility (ATR {t.atr14:.2f} = {t.atr14/t.close*100:.1f}% of price)")

    if m and m.last_5d_return_pct > 20:
        risks.append(f"Already extended +{m.last_5d_return_pct:.1f}% over 5 days")
    if m and m.pre_market_gap_pct >= 10:
        risks.append("Large pre-market gap — high chance of fade")

    if s.label == "hype":
        risks.append("Hype-driven sentiment — risk of late entry near peak")
    if s.composite <= -0.25:
        risks.append("Negative sentiment composite — headwind")

    if cat.polarity <= -0.3:
        risks.append("Negative news polarity in recent headlines")

    mc = bundle.market.market_cap
    if mc and mc < 500_000_000:
        risks.append(f"Small-cap (mkt cap ${mc/1e6:.0f}M) — slippage / volatility")

    sp = bundle.market.short_percent_of_float
    if sp and sp >= 0.20:
        risks.append(f"High short interest ({sp*100:.0f}% of float) — squeeze risk both ways")

    if not risks:
        risks.append("Standard market risk; size positions appropriately")
    return risks


def _strategy(bundle: SignalBundle, setup: SetupType) -> tuple[str, TimeHorizon]:
    t = bundle.technical
    m = bundle.momentum
    price = bundle.market.last_price

    if setup is SetupType.BREAKOUT and t:
        entry = t.resistance_20d
        stop = max(t.support_20d, price - 1.5 * t.atr14)
        target = price + 2.0 * t.atr14
        return (
            f"Watch for break + hold above ${entry:,.2f} on volume; "
            f"entry on retest, stop ${stop:,.2f}, first target ${target:,.2f}",
            TimeHorizon.SHORT,
        )
    if setup is SetupType.CAPITULATION and t:
        stop = price - 1.0 * t.atr14
        target = t.sma20
        return (
            f"Reversal play: scale in near support ${t.support_20d:,.2f}, "
            f"stop ${stop:,.2f}, mean-reversion target ${target:,.2f}",
            TimeHorizon.SWING,
        )
    if setup is SetupType.PRE_MARKET and m:
        return (
            f"Wait for first 15-30 min of cash session to confirm direction; "
            f"avoid chasing the open; gap was {m.pre_market_gap_pct:+.1f}%",
            TimeHorizon.INTRADAY,
        )
    if setup is SetupType.HYPE:
        return (
            "Caution: hype-driven move. Define a hard stop (e.g. -5%) before entry; "
            "consider a small starter position only",
            TimeHorizon.INTRADAY,
        )
    if setup is SetupType.UNUSUAL_ACTIVITY:
        return (
            "Investigate cause of volume spike before entry; "
            "if confirmed by news, treat as breakout / momentum setup",
            TimeHorizon.SHORT,
        )
    return (
        "Add to watchlist; size small until trend / catalyst confirms",
        TimeHorizon.SWING,
    )


def build_alert(bundle: SignalBundle, weights: dict[str, float] | None = None) -> Alert | None:
    weights = weights or DEFAULT_WEIGHTS
    score = _composite_score(bundle, weights)
    if score < 0.30:  # below this we don't bother emitting
        return None

    setup = _classify_setup(bundle)
    catalyst_src = _classify_catalyst_source(bundle)
    strategy, horizon = _strategy(bundle, setup)

    return Alert(
        ticker=bundle.ticker,
        current_price=round(bundle.market.last_price, 4),
        setup_type=setup,
        confidence=Confidence.from_score(score),
        catalyst=catalyst_src,
        key_signals=_key_signals(bundle),
        risk_factors=_risk_factors(bundle),
        suggested_strategy=strategy,
        time_horizon=horizon,
        raw_score=round(score, 3),
        headline=bundle.catalyst.headline,
        headline_url=bundle.catalyst.url,
    )


def build_bundle(
    ticker: str,
    *,
    news: list[NewsItem],
    social: list[SocialMention],
    include_intraday: bool = False,
) -> SignalBundle | None:
    md = get_market_data(ticker, include_intraday=include_intraday)
    if md is None:
        return None
    return SignalBundle(
        ticker=ticker,
        market=md,
        technical=analyze_technicals(md),
        momentum=score_momentum(md),
        catalyst=classify_catalysts(ticker, news),
        sentiment=score_sentiment(ticker, news, social),
    )


def generate_alerts(
    tickers: list[str],
    *,
    weights: dict[str, float] | None = None,
    include_intraday: bool = False,
    fetch_social: bool = True,
) -> list[Alert]:
    universe = set(tickers)
    log.info("Fetching news for %d tickers", len(tickers))
    raw_news = fetch_news(tickers)
    news = attach_tickers(raw_news, universe)

    social: list[SocialMention] = []
    if fetch_social:
        log.info("Fetching social mentions")
        social = fetch_social_mentions(tickers)

    alerts: list[Alert] = []
    for tk in tickers:
        try:
            bundle = build_bundle(
                tk, news=news, social=social, include_intraday=include_intraday
            )
            if bundle is None:
                continue
            alert = build_alert(bundle, weights=weights)
            if alert:
                alerts.append(alert)
        except Exception as exc:  # noqa: BLE001
            log.warning("Alert generation failed for %s: %s", tk, exc)
    alerts.sort(key=lambda a: (a.confidence.rank, a.raw_score), reverse=True)
    return alerts


# Convenience re-export
scan_universe = generate_alerts
