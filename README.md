# Trading101

**AI-powered short-term trading intelligence agent for the US stock market.**

Identifies high-probability 3–5 day trade setups by fusing real-time price action,
financial news, social sentiment, and technical pattern recognition into ranked,
risk-aware alerts.

> ⚠️ Trading101 is a research tool. It does **not** guarantee returns and does **not**
> place trades. All output is data-driven probability and risk analysis, not investment advice.

---

## What it does

1. **Ingests data** for a watchlist of tickers
   - Daily OHLCV + optional 1-min intraday (yfinance, free / delayed)
   - Per-ticker headlines (Yahoo Finance RSS) + market headlines (MarketWatch RSS)
   - Social sentiment (StockTwits public stream; optional Twitter/Reddit with API keys)
2. **Analyzes** each ticker through four engines
   - **Technical** — RSI, EMAs (9/21), SMAs (20/50), ATR, plus pattern detectors:
     near-resistance, breakout-confirmed, capitulation, bull flag, consolidation,
     volume accumulation, RSI divergence, EMA crossover, SMA reclaim
   - **Momentum** — historical 5-day burst rate, relative volume, pre-market gap,
     trend alignment, "early-move" sweet-spot detection
   - **Catalyst** — regex-based classifier for earnings, FDA, M&A, analyst actions,
     partnerships, AI/tech hype, macro/sector, regulatory; scored for polarity + freshness
   - **Sentiment** — VADER on news + StockTwits with hype-velocity detection
3. **Generates alerts** with the exact spec output format (see below)
4. **Learns** from outcomes — logs every alert, grades realized 5-day forward returns,
   and adjusts signal weights toward categories with higher historical win rates
5. **Visualizes** everything in a Streamlit dashboard (alerts, charts, learning stats)

---

## Quick start

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. (optional) configure
cp .env.example .env
# edit .env to set TRADING101_UNIVERSE or premium API keys

# 3. scan from the CLI
trading101 scan --tickers NVDA,AMD,SMCI,PLTR --min-confidence Medium

# 4. or open the dashboard
trading101 dashboard
# → opens http://localhost:8501
```

---

## CLI

| Command                          | What it does                                                       |
| -------------------------------- | ------------------------------------------------------------------ |
| `trading101 scan`                | Scan watchlist, print ranked alerts (table + per-alert detail)     |
| `trading101 scan --json-out`     | Same, JSON output (for piping into Slack / cron / your own tools)  |
| `trading101 grade-outcomes`      | Pull realized 5-day forward returns for past alerts → outcomes log |
| `trading101 adjust-weights`      | Adjust signal weights from the outcomes ledger                     |
| `trading101 stats`               | Win rate, avg return, performance broken down by setup type        |
| `trading101 dashboard`           | Launch the Streamlit dashboard                                     |
| `trading101 bot`                 | Run the interactive Telegram bot                                   |
| `trading101 notify`              | One-shot scan → broadcast alerts to your Telegram chat             |

---

## Alert output format

Every alert renders in the spec's required format:

```
Ticker: NVDA
Current Price: $891.23
Setup Type: Breakout
Confidence Level: High
Catalyst: Mixed
Key Signals:
  - Patterns: near_resistance, volume_accumulation, ema9_cross_above_ema21
  - RSI(14): 62.4
  - Within 0.8% of 20-day resistance $895.10
  - Relative volume 2.3x average
  - historical 5d burst rate 18% (median +12.4%)
  - News catalysts: analyst_action, ai_tech_hype
  - Latest headline (3h ago): Goldman raises NVDA price target to $1,100
  - Social: 47 mentions/24h, hype velocity 3.1x, label=bullish, composite=+0.42
Risk Factors:
  - High volatility (ATR 28.40 = 3.2% of price)
  - Already extended +6.4% over 5 days
Suggested Strategy: Watch for break + hold above $895.10 on volume; entry on retest, stop $864.50, first target $948.00
Time Horizon: 1-3 days
```

---

## Telegram bot

Control the agent from your phone. Zero coding needed after setup.

### One-time setup (2 minutes)

1. On Telegram, search for **@BotFather** → send `/newbot` → follow prompts → copy the **HTTP API token**.
2. Start a chat with your new bot (send it any message, e.g. "hi").
3. Get your chat ID — open this in a browser:
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   Look for `"chat":{"id":123456789,...}` — that number is your chat ID.
4. Edit `.env`:
   ```
   TELEGRAM_BOT_TOKEN=123456:ABC...
   TELEGRAM_CHAT_ID=123456789
   ```

### Run the bot

```bash
trading101 bot
```

Now on Telegram, chat with your bot:

| Command           | What it does                                             |
| ----------------- | -------------------------------------------------------- |
| `/start`          | Welcome + list of commands                               |
| `/scan NVDA AMD`  | Scan one or more tickers and get full alerts back        |
| `/alerts`         | Scan your whole default watchlist                        |
| `/chart TSLA`     | Send a candlestick + RSI chart image                     |
| `/news AAPL`      | Latest 5 headlines for a ticker                          |
| `/stats`          | Learning-tracker stats (win rate, weights)               |

### Or just broadcast (no interactive bot)

Pair this with cron to get a morning scan pushed to your phone:

```bash
trading101 notify --min-confidence High
```

---

## Architecture

```
src/trading101/
├── config.py                  # env / settings / paths
├── data/
│   ├── market.py              # yfinance adapter (swappable for Polygon/Alpaca)
│   ├── news.py                # RSS aggregator + ticker tagging
│   └── social.py              # StockTwits + (Twitter/Reddit stubs)
├── analysis/
│   ├── technical.py           # indicators + pattern detection
│   ├── momentum.py            # 3–5 day burst scanner
│   ├── catalyst.py            # news classifier (regex + polarity + freshness)
│   └── sentiment.py           # VADER + hype-velocity
├── alerts/
│   ├── models.py              # Alert dataclass (spec output format)
│   └── engine.py              # weighted signal fusion → ranked alerts
├── learning/
│   └── tracker.py             # log, grade, adjust weights
├── ui/
│   └── dashboard.py           # Streamlit dashboard
└── cli.py                     # Click CLI entry point
```

The engine is provider-agnostic. The `data/` adapters can be swapped for paid
providers (Polygon, Alpaca, Tradier, Finnhub, NewsAPI, Benzinga, Twitter v2,
Reddit/PRAW) without touching the analysis or alert layer — drop in an API key
in `.env` and extend the corresponding adapter.

---

## What's free vs. what needs API keys

| Capability                                  | Free (default)                  | Premium upgrade           |
| ------------------------------------------- | ------------------------------- | ------------------------- |
| Daily OHLCV + 1-min intraday                | yfinance                        | Polygon / Alpaca / Tradier |
| Per-ticker news                             | Yahoo Finance RSS               | NewsAPI / Benzinga        |
| Social sentiment                            | StockTwits public stream        | Twitter v2 / Reddit (PRAW) |
| Options flow / unusual activity             | _not available_                 | Tradier / CBOE / Unusual Whales |
| Short interest / float                      | yfinance `.info` (best-effort)  | Finnhub / FinTel          |

The keys are listed in `.env.example`. Adapters detect them automatically.

---

## Learning loop

The system **logs every alert** to `data/state/alerts.jsonl`. Run nightly:

```bash
trading101 grade-outcomes --horizon 5     # pulls realized 5-day return per alert
trading101 adjust-weights                 # nudges signal weights toward winners
```

Weights are saved to `data/state/weights.json` and picked up on the next scan.
Once enough outcomes accumulate (>= 10), `adjust-weights` shifts allocation
between {technical, momentum, catalyst, sentiment} based on per-class win rate.

---

## Tests

```bash
pytest
```

Tests cover indicators, pattern detection, catalyst classification, alert
construction, and the learning tracker — all without network calls.

---

## Roadmap

- [ ] Polygon adapter for true real-time quotes + options chain
- [ ] Twitter v2 + Reddit/PRAW concrete implementations
- [ ] Per-ticker per-setup weighting (not just per-class)
- [ ] Backtest harness driven by the same alert engine
- [ ] Webhook / Slack / Discord notifier in addition to dashboard
- [ ] Sector rotation detector (intermarket)

---

## License

MIT.
