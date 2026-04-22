"""Trading101 CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .alerts import generate_alerts
from .alerts.models import Confidence
from .config import configure_logging, load_settings
from .learning import LearningTracker

console = Console()


@click.group()
@click.option("--log-level", default=None, help="DEBUG / INFO / WARNING / ERROR")
@click.pass_context
def main(ctx: click.Context, log_level: str | None) -> None:
    """AI-powered short-term trading intelligence agent."""
    settings = load_settings()
    configure_logging(log_level or settings.log_level)
    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings


@main.command()
@click.option("--tickers", "-t", default=None, help="Comma-separated tickers (defaults to TRADING101_UNIVERSE)")
@click.option("--min-confidence", default=None, type=click.Choice(["Low", "Medium", "High"], case_sensitive=False))
@click.option("--intraday/--no-intraday", default=False)
@click.option("--social/--no-social", default=True)
@click.option("--json-out", is_flag=True, help="Emit JSON instead of rich text")
@click.option("--log/--no-log", default=True, help="Persist alerts for the learning tracker")
@click.pass_context
def scan(
    ctx: click.Context,
    tickers: str | None,
    min_confidence: str | None,
    intraday: bool,
    social: bool,
    json_out: bool,
    log: bool,
) -> None:
    """Run a one-shot scan of the watchlist and emit alerts."""
    settings = ctx.obj["settings"]
    universe = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers else settings.universe
    )
    tracker = LearningTracker()
    weights = tracker.load_weights()

    console.log(f"Scanning {len(universe)} tickers with weights {weights}")
    alerts = generate_alerts(
        universe, weights=weights, include_intraday=intraday, fetch_social=social
    )
    if log:
        tracker.log_alerts(alerts)

    threshold = Confidence((min_confidence or settings.min_confidence).capitalize())
    alerts = [a for a in alerts if a.confidence.rank >= threshold.rank]

    if json_out:
        click.echo(json.dumps([a.to_dict() for a in alerts], indent=2, default=str))
        return

    if not alerts:
        console.print(Panel.fit(
            f"No alerts at confidence >= {threshold.value}.",
            title="Trading101", border_style="yellow",
        ))
        return

    table = Table(title=f"Trading101 Alerts (>= {threshold.value})", show_lines=True)
    for col in ("Ticker", "Price", "Setup", "Conf", "Catalyst", "Score", "Horizon"):
        table.add_column(col)
    for a in alerts:
        table.add_row(
            a.ticker,
            f"${a.current_price:,.2f}",
            a.setup_type.value,
            a.confidence.value,
            a.catalyst.value,
            f"{a.raw_score:.2f}",
            a.time_horizon.value,
        )
    console.print(table)

    for a in alerts:
        console.print(Panel(a.render_text(), title=f"{a.ticker} — {a.setup_type.value}", border_style="cyan"))


@main.command(name="grade-outcomes")
@click.option("--horizon", default=5, help="Forward window (days) used to grade past alerts")
@click.option("--win-pct", default=3.0, help="Return % considered a 'win'")
def grade_outcomes(horizon: int, win_pct: float) -> None:
    """Grade past alerts by their realized forward return."""
    tracker = LearningTracker()
    outcomes = tracker.grade_pending(horizon_days=horizon, win_threshold_pct=win_pct)
    console.print(f"Graded {len(outcomes)} outcomes.")
    if outcomes:
        wins = sum(1 for o in outcomes if o.won)
        avg = sum(o.return_pct for o in outcomes) / len(outcomes)
        console.print(f"Win rate this batch: {wins/len(outcomes)*100:.1f}%   avg return {avg:+.2f}%")


@main.command(name="adjust-weights")
@click.option("--learning-rate", default=0.1)
def adjust_weights(learning_rate: float) -> None:
    """Bias weights toward signal classes with higher historical win rate."""
    tracker = LearningTracker()
    weights = tracker.adjust_weights(learning_rate=learning_rate)
    console.print("Updated weights:")
    console.print_json(data=weights)


@main.command()
def stats() -> None:
    """Print learning-tracker stats."""
    tracker = LearningTracker()
    console.print_json(data=tracker.stats())


@main.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    here = Path(__file__).resolve().parent / "ui" / "dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(here)]
    console.print(f"Launching dashboard: {' '.join(cmd)}")
    subprocess.call(cmd)


if __name__ == "__main__":
    main()
