"""Alert data model — matches the spec's required output format exactly."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum


class SetupType(str, Enum):
    BREAKOUT = "Breakout"
    CAPITULATION = "Capitulation"
    MOMENTUM = "Momentum"
    HYPE = "Hype"
    UNUSUAL_ACTIVITY = "Unusual Activity"
    PRE_MARKET = "Pre-Market Mover"


class Confidence(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

    @classmethod
    def from_score(cls, score: float) -> "Confidence":
        if score >= 0.75:
            return cls.HIGH
        if score >= 0.5:
            return cls.MEDIUM
        return cls.LOW

    @property
    def rank(self) -> int:
        return {"Low": 0, "Medium": 1, "High": 2}[self.value]


class CatalystType(str, Enum):
    NEWS = "News"
    TECHNICAL = "Technical"
    SOCIAL = "Social"
    MIXED = "Mixed"


class TimeHorizon(str, Enum):
    INTRADAY = "intraday"
    SHORT = "1-3 days"
    SWING = "3-5 days"


@dataclass
class Alert:
    ticker: str
    current_price: float
    setup_type: SetupType
    confidence: Confidence
    catalyst: CatalystType
    key_signals: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    suggested_strategy: str = ""
    time_horizon: TimeHorizon = TimeHorizon.SHORT
    raw_score: float = 0.0
    headline: str | None = None
    headline_url: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["setup_type"] = self.setup_type.value
        d["confidence"] = self.confidence.value
        d["catalyst"] = self.catalyst.value
        d["time_horizon"] = self.time_horizon.value
        d["created_at"] = self.created_at.isoformat()
        return d

    def render_text(self) -> str:
        """Render in the spec's exact output format."""
        lines = [
            f"Ticker: {self.ticker}",
            f"Current Price: ${self.current_price:,.2f}",
            f"Setup Type: {self.setup_type.value}",
            f"Confidence Level: {self.confidence.value}",
            f"Catalyst: {self.catalyst.value}",
            "Key Signals:",
            *[f"  - {s}" for s in self.key_signals],
            "Risk Factors:",
            *[f"  - {r}" for r in self.risk_factors],
            f"Suggested Strategy: {self.suggested_strategy}",
            f"Time Horizon: {self.time_horizon.value}",
        ]
        if self.headline:
            lines.append(f"Headline: {self.headline}")
            if self.headline_url:
                lines.append(f"Source: {self.headline_url}")
        return "\n".join(lines)
