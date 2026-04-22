"""Outcome logging + signal-weight adjustment.

Persists every emitted alert to a JSONL ledger. After enough time has passed
(configurable horizon), the tracker grades outcomes by pulling the realized
forward return from yfinance and adjusts signal weights using a simple
inverse-variance / win-rate heuristic.

Designed to be run on a schedule (e.g. nightly cron) — see CLI command
`trading101 grade-outcomes`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..alerts.models import Alert
from ..config import STATE_DIR
from ..data.market import get_market_data
from ..alerts.engine import DEFAULT_WEIGHTS

log = logging.getLogger(__name__)

ALERTS_LOG = STATE_DIR / "alerts.jsonl"
OUTCOMES_LOG = STATE_DIR / "outcomes.jsonl"
WEIGHTS_FILE = STATE_DIR / "weights.json"


@dataclass
class AlertOutcome:
    ticker: str
    setup_type: str
    confidence: str
    catalyst: str
    raw_score: float
    entry_price: float
    exit_price: float
    return_pct: float
    horizon_days: int
    won: bool
    created_at: str
    graded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LearningTracker:
    def __init__(
        self,
        alerts_path: Path = ALERTS_LOG,
        outcomes_path: Path = OUTCOMES_LOG,
        weights_path: Path = WEIGHTS_FILE,
    ) -> None:
        self.alerts_path = alerts_path
        self.outcomes_path = outcomes_path
        self.weights_path = weights_path

    # ---------- persistence ----------

    def log_alert(self, alert: Alert) -> None:
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        with self.alerts_path.open("a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")

    def log_alerts(self, alerts: list[Alert]) -> None:
        for a in alerts:
            self.log_alert(a)

    def _read_alerts(self) -> list[dict]:
        if not self.alerts_path.exists():
            return []
        out: list[dict] = []
        with self.alerts_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    def _read_outcomes(self) -> list[dict]:
        if not self.outcomes_path.exists():
            return []
        out: list[dict] = []
        with self.outcomes_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    # ---------- grading ----------

    def grade_pending(self, horizon_days: int = 5, win_threshold_pct: float = 3.0) -> list[AlertOutcome]:
        """Grade alerts that are at least `horizon_days` old and haven't been graded yet."""
        already_graded = {(o["ticker"], o["created_at"]) for o in self._read_outcomes()}
        cutoff = datetime.now(timezone.utc) - timedelta(days=horizon_days)
        pending = [
            a for a in self._read_alerts()
            if datetime.fromisoformat(a["created_at"]) <= cutoff
            and (a["ticker"], a["created_at"]) not in already_graded
        ]
        if not pending:
            return []

        outcomes: list[AlertOutcome] = []
        for a in pending:
            try:
                md = get_market_data(a["ticker"], history_days=horizon_days + 5)
                if md is None or md.history.empty:
                    continue
                exit_price = float(md.history["Close"].iloc[-1])
                entry_price = float(a["current_price"])
                ret = (exit_price - entry_price) / entry_price * 100.0
                outcome = AlertOutcome(
                    ticker=a["ticker"],
                    setup_type=a["setup_type"],
                    confidence=a["confidence"],
                    catalyst=a["catalyst"],
                    raw_score=a.get("raw_score", 0.0),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    return_pct=round(ret, 3),
                    horizon_days=horizon_days,
                    won=ret >= win_threshold_pct,
                    created_at=a["created_at"],
                )
                outcomes.append(outcome)
            except Exception as exc:  # noqa: BLE001
                log.warning("Grading failed for %s: %s", a.get("ticker"), exc)

        if outcomes:
            self.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
            with self.outcomes_path.open("a") as f:
                for o in outcomes:
                    f.write(json.dumps(asdict(o)) + "\n")
        return outcomes

    # ---------- weights ----------

    def load_weights(self) -> dict[str, float]:
        if not self.weights_path.exists():
            return dict(DEFAULT_WEIGHTS)
        try:
            data = json.loads(self.weights_path.read_text())
            # Sanity: must contain the four keys
            if set(data.keys()) >= set(DEFAULT_WEIGHTS.keys()):
                return {k: float(data[k]) for k in DEFAULT_WEIGHTS.keys()}
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to read weights, using defaults: %s", exc)
        return dict(DEFAULT_WEIGHTS)

    def save_weights(self, weights: dict[str, float]) -> None:
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.weights_path.write_text(json.dumps(weights, indent=2))

    def adjust_weights(self, learning_rate: float = 0.1) -> dict[str, float]:
        """Bias weights toward signal categories with higher historical win rates.

        We bucket each outcome by its dominant signal class — inferred from the
        Catalyst field on the alert. Categories with above-average win rate get
        weight added, below-average get weight removed; renormalised to sum=1.0.
        """
        outcomes = self._read_outcomes()
        if len(outcomes) < 10:
            log.info("Not enough outcomes to adjust weights (have %d, need 10).", len(outcomes))
            return self.load_weights()

        # Map alert.catalyst -> primary internal weight key
        category_to_key = {
            "News": "catalyst",
            "Technical": "technical",
            "Social": "sentiment",
            "Mixed": "momentum",
        }

        win_by_key: dict[str, list[bool]] = {k: [] for k in DEFAULT_WEIGHTS}
        for o in outcomes:
            key = category_to_key.get(o["catalyst"])
            if not key:
                continue
            win_by_key[key].append(bool(o["won"]))

        weights = self.load_weights()
        all_results = [w for results in win_by_key.values() for w in results]
        if not all_results:
            return weights
        global_rate = sum(all_results) / len(all_results)

        for key, results in win_by_key.items():
            if not results:
                continue
            rate = sum(results) / len(results)
            adjustment = learning_rate * (rate - global_rate)
            weights[key] = max(0.05, weights[key] + adjustment)

        # Normalize
        total = sum(weights.values())
        weights = {k: round(v / total, 4) for k, v in weights.items()}
        self.save_weights(weights)
        return weights

    # ---------- stats ----------

    def stats(self) -> dict:
        outcomes = self._read_outcomes()
        alerts = self._read_alerts()
        if not outcomes:
            return {
                "alerts_logged": len(alerts),
                "outcomes_graded": 0,
                "win_rate": None,
                "avg_return_pct": None,
                "by_setup": {},
                "current_weights": self.load_weights(),
            }
        wins = sum(1 for o in outcomes if o["won"])
        avg_ret = sum(o["return_pct"] for o in outcomes) / len(outcomes)

        by_setup: dict[str, dict] = {}
        for o in outcomes:
            s = o["setup_type"]
            entry = by_setup.setdefault(s, {"n": 0, "wins": 0, "sum_ret": 0.0})
            entry["n"] += 1
            entry["wins"] += int(o["won"])
            entry["sum_ret"] += o["return_pct"]
        for s, e in by_setup.items():
            e["win_rate"] = round(e["wins"] / e["n"], 3)
            e["avg_return_pct"] = round(e["sum_ret"] / e["n"], 3)

        return {
            "alerts_logged": len(alerts),
            "outcomes_graded": len(outcomes),
            "win_rate": round(wins / len(outcomes), 3),
            "avg_return_pct": round(avg_ret, 3),
            "by_setup": by_setup,
            "current_weights": self.load_weights(),
        }
