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
from .integrations.alpaca import AlpacaClient
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


@main.command()
def bot() -> None:
    """Run the interactive Telegram bot (requires TELEGRAM_BOT_TOKEN)."""
    from .notify import TelegramBot
    settings = load_settings()
    if not settings.telegram_bot_token:
        console.print(
            "[yellow]TELEGRAM_BOT_TOKEN not set. Follow the setup in .env.example, "
            "then run `trading101 bot` again.[/yellow]"
        )
        return
    TelegramBot().run()


@main.command()
@click.option("--tickers", "-t", default=None, help="Comma-separated tickers")
@click.option("--min-confidence", default="Medium", type=click.Choice(["Low", "Medium", "High"], case_sensitive=False))
def notify(tickers: str | None, min_confidence: str) -> None:
    """Run a scan and broadcast the results to your Telegram chat."""
    from .notify import send_alerts_to_telegram
    from .alerts.models import Confidence

    settings = load_settings()
    if not (settings.telegram_bot_token and settings.telegram_chat_id):
        console.print("[yellow]Telegram not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.[/yellow]")
        return
    universe = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers else settings.universe
    )
    tracker = LearningTracker()
    alerts = generate_alerts(universe, weights=tracker.load_weights())
    tracker.log_alerts(alerts)
    threshold = Confidence(min_confidence.capitalize())
    sent = send_alerts_to_telegram(alerts, min_confidence=threshold)
    console.print(f"Sent {sent} Telegram message(s).")


@main.group(name="alpaca")
@click.pass_context
def alpaca_group(ctx: click.Context) -> None:
    """Alpaca broker integration (paper + live trading)."""
    settings = ctx.obj["settings"]
    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        console.print("[red]ALPACA_API_KEY and ALPACA_API_SECRET not set.[/red]")
        sys.exit(1)
    ctx.obj["alpaca"] = AlpacaClient(
        api_key=settings.alpaca_api_key,
        api_secret=settings.alpaca_api_secret,
        base_url=settings.alpaca_base_url,
    )


@alpaca_group.command(name="account")
@click.pass_context
def alpaca_account(ctx: click.Context) -> None:
    """Show account status (buying power, cash, portfolio value)."""
    client: AlpacaClient = ctx.obj["alpaca"]
    acc = client.get_account()
    if not acc:
        console.print("[red]Failed to fetch account.[/red]")
        return
    mode = "[green]LIVE[/green]" if not acc.is_paper else "[yellow]PAPER[/yellow]"
    console.print(f"Account Mode: {mode}")
    table = Table(title="Account Status", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Portfolio Value", f"${acc.portfolio_value:,.2f}")
    table.add_row("Buying Power", f"${acc.buying_power:,.2f}")
    table.add_row("Cash", f"${acc.cash:,.2f}")
    table.add_row("Last Equity", f"${acc.last_equity:,.2f}")
    console.print(table)


@alpaca_group.command(name="positions")
@click.pass_context
def alpaca_positions(ctx: click.Context) -> None:
    """Show all open positions."""
    client: AlpacaClient = ctx.obj["alpaca"]
    positions = client.get_positions()
    if not positions:
        console.print("[yellow]No open positions.[/yellow]")
        return
    table = Table(title="Open Positions", show_lines=True)
    for col in ("Symbol", "Qty", "Avg Price", "Current", "P&L", "P&L %"):
        table.add_column(col)
    for pos in positions:
        pl_color = "green" if pos.unrealized_pl >= 0 else "red"
        table.add_row(
            pos.symbol,
            f"{pos.qty:.0f}",
            f"${pos.avg_fill_price:,.2f}",
            f"${pos.current_price:,.2f}",
            f"[{pl_color}]${pos.unrealized_pl:,.2f}[/{pl_color}]",
            f"[{pl_color}]{pos.unrealized_plpc:+.2f}%[/{pl_color}]",
        )
    console.print(table)


@alpaca_group.command(name="quote")
@click.argument("tickers")
@click.pass_context
def alpaca_quote(ctx: click.Context, tickers: str) -> None:
    """Get latest quotes for one or more tickers."""
    client: AlpacaClient = ctx.obj["alpaca"]
    symbols = [t.strip().upper() for t in tickers.split(",")]
    quotes = client.get_quotes(symbols)
    if not quotes:
        console.print("[red]Failed to fetch quotes.[/red]")
        return
    table = Table(title="Latest Quotes", show_lines=True)
    table.add_column("Ticker", style="cyan")
    table.add_column("Price", style="magenta")
    for sym in symbols:
        if sym in quotes:
            table.add_row(sym, f"${quotes[sym]:,.2f}")
        else:
            table.add_row(sym, "[red]N/A[/red]")
    console.print(table)


@alpaca_group.command(name="buy")
@click.argument("symbol")
@click.argument("qty", type=float)
@click.option("--limit", type=float, default=None, help="Limit price (omit for market order)")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def alpaca_buy(ctx: click.Context, symbol: str, qty: float, limit: float | None, confirm: bool) -> None:
    """Place a buy order."""
    client: AlpacaClient = ctx.obj["alpaca"]
    symbol = symbol.upper()
    if not confirm:
        if limit:
            console.print(f"[yellow]BUY {qty:.0f} {symbol} @ ${limit:.2f} (LIMIT)[/yellow]")
        else:
            console.print(f"[yellow]BUY {qty:.0f} {symbol} (MARKET)[/yellow]")
        if not click.confirm("Continue?"):
            return
    if limit:
        order = client.place_limit_order(symbol, qty, "buy", limit)
    else:
        order = client.place_market_order(symbol, qty, "buy")
    if not order:
        console.print("[red]Order failed.[/red]")
        return
    console.print(f"[green]Order placed: {order.id}[/green]")
    console.print(f"Status: {order.status}")


@alpaca_group.command(name="sell")
@click.argument("symbol")
@click.argument("qty", type=float)
@click.option("--limit", type=float, default=None, help="Limit price (omit for market order)")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def alpaca_sell(ctx: click.Context, symbol: str, qty: float, limit: float | None, confirm: bool) -> None:
    """Place a sell order."""
    client: AlpacaClient = ctx.obj["alpaca"]
    symbol = symbol.upper()
    if not confirm:
        if limit:
            console.print(f"[yellow]SELL {qty:.0f} {symbol} @ ${limit:.2f} (LIMIT)[/yellow]")
        else:
            console.print(f"[yellow]SELL {qty:.0f} {symbol} (MARKET)[/yellow]")
        if not click.confirm("Continue?"):
            return
    if limit:
        order = client.place_limit_order(symbol, qty, "sell", limit)
    else:
        order = client.place_market_order(symbol, qty, "sell")
    if not order:
        console.print("[red]Order failed.[/red]")
        return
    console.print(f"[green]Order placed: {order.id}[/green]")
    console.print(f"Status: {order.status}")


if __name__ == "__main__":
    main()
