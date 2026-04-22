"""Tests for the learning tracker (no network)."""
from __future__ import annotations

import json
from pathlib import Path

from trading101.alerts.engine import DEFAULT_WEIGHTS
from trading101.alerts.models import (
    Alert, CatalystType, Confidence, SetupType, TimeHorizon,
)
from trading101.learning import LearningTracker


def _alert(ticker="ACME", score=0.6) -> Alert:
    return Alert(
        ticker=ticker,
        current_price=10.0,
        setup_type=SetupType.BREAKOUT,
        confidence=Confidence.MEDIUM,
        catalyst=CatalystType.NEWS,
        key_signals=["x"], risk_factors=["y"],
        suggested_strategy="watch",
        time_horizon=TimeHorizon.SHORT,
        raw_score=score,
    )


def test_log_and_read_roundtrip(tmp_path: Path):
    tracker = LearningTracker(
        alerts_path=tmp_path / "alerts.jsonl",
        outcomes_path=tmp_path / "outcomes.jsonl",
        weights_path=tmp_path / "weights.json",
    )
    tracker.log_alerts([_alert("AAA"), _alert("BBB")])
    raw = (tmp_path / "alerts.jsonl").read_text().strip().splitlines()
    assert len(raw) == 2
    assert json.loads(raw[0])["ticker"] == "AAA"


def test_default_weights_returned_if_no_file(tmp_path: Path):
    tracker = LearningTracker(
        alerts_path=tmp_path / "alerts.jsonl",
        outcomes_path=tmp_path / "outcomes.jsonl",
        weights_path=tmp_path / "weights.json",
    )
    assert tracker.load_weights() == DEFAULT_WEIGHTS


def test_save_then_load_weights(tmp_path: Path):
    tracker = LearningTracker(
        alerts_path=tmp_path / "alerts.jsonl",
        outcomes_path=tmp_path / "outcomes.jsonl",
        weights_path=tmp_path / "weights.json",
    )
    new = {"technical": 0.5, "momentum": 0.2, "catalyst": 0.2, "sentiment": 0.1}
    tracker.save_weights(new)
    assert tracker.load_weights() == new


def test_stats_empty(tmp_path: Path):
    tracker = LearningTracker(
        alerts_path=tmp_path / "alerts.jsonl",
        outcomes_path=tmp_path / "outcomes.jsonl",
        weights_path=tmp_path / "weights.json",
    )
    s = tracker.stats()
    assert s["alerts_logged"] == 0
    assert s["outcomes_graded"] == 0
    assert s["win_rate"] is None
