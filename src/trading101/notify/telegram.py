"""Telegram bot interface.

Two modes:

1. One-shot broadcast — `send_alerts_to_telegram(alerts)` pushes the current
   scan results to a chat. Good for cron jobs.
2. Interactive bot — `TelegramBot.run()` listens for commands like
   /scan NVDA, /chart AAPL, /news TSLA, /stats, /alerts.

Both modes use the same TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from .env.
"""
from __future__ import annotations

import asyncio
import io
import logging
from typing import Iterable

import requests

from ..alerts import generate_alerts
from ..alerts.models import Alert, Confidence
from ..analysis.technical import analyze_technicals, ema, sma
from ..config import load_settings
from ..data.market import get_market_data
from ..data.news import fetch_ticker_news
from ..learning import LearningTracker

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


# ---------- One-shot sender (no extra deps beyond `requests`) ----------

def _esc(text: str) -> str:
    """Escape for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_alert(a: Alert) -> str:
    conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "⚪️"}[a.confidence.value]
    setup_emoji = {
        "Breakout": "🚀", "Capitulation": "🔁", "Momentum": "📈",
        "Hype": "🔥", "Unusual Activity": "⚡", "Pre-Market Mover": "🌅",
    }.get(a.setup_type.value, "•")
    lines = [
        f"{conf_emoji} <b>{_esc(a.ticker)}</b> — ${a.current_price:,.2f}",
        f"{setup_emoji} {_esc(a.setup_type.value)} · {_esc(a.catalyst.value)} · "
        f"{a.confidence.value} conf · score {a.raw_score:.2f}",
        f"⏱ {_esc(a.time_horizon.value)}",
        "",
        "<b>Key signals:</b>",
        *[f"• {_esc(s)}" for s in a.key_signals[:6]],
        "",
        "<b>Risks:</b>",
        *[f"• {_esc(r)}" for r in a.risk_factors[:4]],
        "",
        f"💡 <i>{_esc(a.suggested_strategy)}</i>",
    ]
    if a.headline:
        lines.append("")
        if a.headline_url:
            lines.append(f"📰 <a href=\"{_esc(a.headline_url)}\">{_esc(a.headline[:90])}</a>")
        else:
            lines.append(f"📰 {_esc(a.headline[:90])}")
    return "\n".join(lines)


def send_message(token: str, chat_id: str, text: str, *, parse_mode: str = "HTML") -> bool:
    try:
        resp = requests.post(
            TELEGRAM_API.format(token=token, method="sendMessage"),
            json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode,
                  "disable_web_page_preview": True},
            timeout=10,
        )
        if resp.status_code != 200:
            log.warning("Telegram send failed [%s]: %s", resp.status_code, resp.text)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Telegram send error: %s", exc)
        return False


def send_photo(token: str, chat_id: str, image_bytes: bytes, caption: str = "") -> bool:
    try:
        resp = requests.post(
            TELEGRAM_API.format(token=token, method="sendPhoto"),
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": ("chart.png", image_bytes, "image/png")},
            timeout=20,
        )
        return resp.status_code == 200
    except Exception as exc:  # noqa: BLE001
        log.error("Telegram photo send error: %s", exc)
        return False


def send_alerts_to_telegram(
    alerts: Iterable[Alert],
    *,
    token: str | None = None,
    chat_id: str | None = None,
    min_confidence: Confidence = Confidence.MEDIUM,
) -> int:
    """Push a batch of alerts. Returns the number of messages sent."""
    settings = load_settings()
    token = token or settings.telegram_bot_token
    chat_id = chat_id or settings.telegram_chat_id
    if not token or not chat_id:
        log.warning("Telegram not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return 0

    sent = 0
    filtered = [a for a in alerts if a.confidence.rank >= min_confidence.rank]
    if not filtered:
        send_message(token, chat_id, f"📭 No alerts at or above {min_confidence.value} confidence.")
        return 1

    header = f"🧠 <b>Trading101 scan</b> — {len(filtered)} alert(s)"
    send_message(token, chat_id, header)
    for a in filtered:
        if send_message(token, chat_id, _format_alert(a)):
            sent += 1
    return sent


# ---------- Interactive bot (requires python-telegram-bot) ----------


def _chart_png(ticker: str) -> bytes | None:
    """Render a small matplotlib chart and return it as PNG bytes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — charts disabled.")
        return None

    md = get_market_data(ticker, history_days=120)
    if md is None or md.history.empty:
        return None
    df = md.history
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(df.index, df["Close"], label="Close", linewidth=1.4)
    ax1.plot(df.index, sma(df["Close"], 20), label="SMA 20", linewidth=0.9)
    ax1.plot(df.index, sma(df["Close"], 50), label="SMA 50", linewidth=0.9)
    ax1.plot(df.index, ema(df["Close"], 9), label="EMA 9", linewidth=0.9, linestyle="--")
    ax1.set_title(f"{ticker} — last 120 days")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)

    from ..analysis.technical import rsi as _rsi
    r = _rsi(df["Close"], 14)
    ax2.plot(df.index, r, linewidth=1.0, color="#805ad5")
    ax2.axhline(70, color="red", linestyle=":", linewidth=0.8)
    ax2.axhline(30, color="green", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("RSI(14)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


class TelegramBot:
    """Polling-based Telegram bot that understands trading commands."""

    def __init__(self, token: str | None = None, allowed_chat_ids: list[str] | None = None):
        settings = load_settings()
        self.token = token or settings.telegram_bot_token
        if not self.token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN not set — see .env.example.")
        # If provided, the bot only answers these chats (basic auth).
        allowed = allowed_chat_ids or (
            [settings.telegram_chat_id] if settings.telegram_chat_id else []
        )
        self.allowed = {str(x) for x in allowed if x}
        self.tracker = LearningTracker()

    # -------- auth helper --------

    def _authorized(self, chat_id) -> bool:
        if not self.allowed:
            return True  # open bot
        return str(chat_id) in self.allowed

    # -------- command handlers --------

    async def _handle_start(self, update, _context):
        await update.message.reply_html(
            "👋 <b>Trading101</b> bot ready!\n\n"
            "Commands:\n"
            "/scan TICKERS — analyze one or more tickers (e.g. <code>/scan NVDA AMD</code>)\n"
            "/alerts — scan the default watchlist\n"
            "/chart TICKER — send a chart image\n"
            "/news TICKER — latest headlines\n"
            "/stats — learning-tracker stats\n"
            "/help — this message"
        )

    async def _handle_help(self, update, context):
        await self._handle_start(update, context)

    async def _handle_scan(self, update, context):
        if not self._authorized(update.effective_chat.id):
            return
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /scan NVDA AMD TSLA")
            return
        tickers = [t.upper() for t in args]
        await update.message.reply_text(f"🔎 Scanning {', '.join(tickers)}…")
        weights = self.tracker.load_weights()
        alerts = generate_alerts(tickers, weights=weights, fetch_social=True)
        self.tracker.log_alerts(alerts)

        if not alerts:
            await update.message.reply_text("No setups worth alerting on right now.")
            return
        for a in alerts:
            await update.message.reply_html(_format_alert(a), disable_web_page_preview=True)

    async def _handle_alerts(self, update, _context):
        if not self._authorized(update.effective_chat.id):
            return
        settings = load_settings()
        await update.message.reply_text(
            f"🔎 Scanning watchlist ({len(settings.universe)} tickers)…"
        )
        weights = self.tracker.load_weights()
        alerts = generate_alerts(settings.universe, weights=weights, fetch_social=True)
        self.tracker.log_alerts(alerts)
        high_med = [a for a in alerts if a.confidence.rank >= Confidence.MEDIUM.rank]
        if not high_med:
            await update.message.reply_text("No Medium+ alerts — market is quiet.")
            return
        await update.message.reply_text(
            f"Found {len(high_med)} alert(s) at Medium+ confidence."
        )
        for a in high_med[:10]:
            await update.message.reply_html(_format_alert(a), disable_web_page_preview=True)

    async def _handle_chart(self, update, context):
        if not self._authorized(update.effective_chat.id):
            return
        if not context.args:
            await update.message.reply_text("Usage: /chart NVDA")
            return
        ticker = context.args[0].upper()
        await update.message.reply_text(f"📊 Rendering {ticker}…")
        png = _chart_png(ticker)
        if png is None:
            await update.message.reply_text("Could not render chart (missing data or matplotlib).")
            return

        md = get_market_data(ticker)
        caption_parts = [f"<b>{_esc(ticker)}</b>"]
        if md is not None:
            caption_parts.append(
                f"${md.last_price:,.2f} ({md.pct_change_today:+.2f}% today) · "
                f"RelVol {md.relative_volume:.1f}x"
            )
        await update.message.reply_photo(
            photo=png, caption=" — ".join(caption_parts), parse_mode="HTML",
        )

    async def _handle_news(self, update, context):
        if not self._authorized(update.effective_chat.id):
            return
        if not context.args:
            await update.message.reply_text("Usage: /news AAPL")
            return
        ticker = context.args[0].upper()
        items = fetch_ticker_news(ticker, limit=5)
        if not items:
            await update.message.reply_text(f"No recent news for {ticker}.")
            return
        lines = [f"📰 <b>{_esc(ticker)}</b> — latest headlines:", ""]
        for n in items:
            ts = n.published_at.strftime("%m-%d %H:%M")
            if n.url:
                lines.append(f"[{ts}] <a href=\"{_esc(n.url)}\">{_esc(n.title)}</a>")
            else:
                lines.append(_esc(f"[{ts}] {n.title}"))
            lines.append("")
        await update.message.reply_html("\n".join(lines), disable_web_page_preview=True)

    async def _handle_stats(self, update, _context):
        if not self._authorized(update.effective_chat.id):
            return
        s = self.tracker.stats()
        msg = (
            f"<b>Learning Tracker</b>\n"
            f"Alerts logged: {s['alerts_logged']}\n"
            f"Outcomes graded: {s['outcomes_graded']}\n"
            f"Win rate: {(s['win_rate'] or 0)*100:.1f}%\n"
            f"Avg return: {s['avg_return_pct'] or 0:+.2f}%\n\n"
            f"Weights: {_esc(str(s['current_weights']))}"
        )
        await update.message.reply_html(msg)

    # -------- lifecycle --------

    def run(self) -> None:
        """Start polling. Blocks until interrupted."""
        try:
            from telegram.ext import (
                ApplicationBuilder, CommandHandler,
            )
        except ImportError as exc:
            raise RuntimeError(
                "python-telegram-bot not installed. Run:  pip install 'python-telegram-bot>=21.0'"
            ) from exc

        app = ApplicationBuilder().token(self.token).build()
        app.add_handler(CommandHandler("start", self._handle_start))
        app.add_handler(CommandHandler("help", self._handle_help))
        app.add_handler(CommandHandler("scan", self._handle_scan))
        app.add_handler(CommandHandler("alerts", self._handle_alerts))
        app.add_handler(CommandHandler("chart", self._handle_chart))
        app.add_handler(CommandHandler("news", self._handle_news))
        app.add_handler(CommandHandler("stats", self._handle_stats))
        log.info("Telegram bot starting (polling)…")
        app.run_polling(allowed_updates=["message"])
