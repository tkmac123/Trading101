"""Streamlit dashboard.

Run with:  streamlit run src/trading101/ui/dashboard.py
Or via CLI: trading101 dashboard
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from trading101.alerts import generate_alerts
from trading101.alerts.models import Confidence
from trading101.analysis.technical import sma, ema, rsi
from trading101.config import load_settings
from trading101.data.market import get_market_data
from trading101.learning import LearningTracker

st.set_page_config(
    page_title="Trading101 — Short-Term Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- styling ----------
st.markdown(
    """
    <style>
        .alert-card { border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; }
        .conf-High   { background: #0f5132; color: #e8f5e9; }
        .conf-Medium { background: #664d03; color: #fff8e1; }
        .conf-Low    { background: #495057; color: #f1f3f5; }
        .signal-pill { display:inline-block; padding:2px 8px; border-radius:8px; background:#1e3a5f; color:#cfe2ff; margin-right:4px; font-size:0.85em;}
        .ticker-big { font-size: 1.6em; font-weight: 700; letter-spacing: 0.05em; }
        .price-big  { font-size: 1.3em; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- sidebar ----------
settings = load_settings()
with st.sidebar:
    st.title("⚙️ Configuration")
    universe_text = st.text_area(
        "Watchlist (comma-separated tickers)",
        value=", ".join(settings.universe),
        help="Tickers to scan. Pulls free Yahoo Finance data — no API key required.",
    )
    min_conf_label = st.selectbox(
        "Minimum Confidence",
        ["Low", "Medium", "High"],
        index={"low": 0, "medium": 1, "high": 2}[settings.min_confidence],
    )
    include_intraday = st.checkbox("Include intraday (1-min) bars", value=False)
    fetch_social = st.checkbox("Fetch StockTwits sentiment", value=True)
    st.divider()
    if st.button("🔄 Run Scan", use_container_width=True, type="primary"):
        st.session_state.pop("alerts", None)
    st.caption("Data: yfinance (delayed) + Yahoo/MarketWatch RSS + StockTwits public stream.")


def _normalize_universe(text: str) -> list[str]:
    return [t.strip().upper() for t in text.split(",") if t.strip()]


# ---------- run scan ----------

if "alerts" not in st.session_state:
    universe = _normalize_universe(universe_text)
    if not universe:
        st.warning("Add at least one ticker to the watchlist.")
        st.stop()
    tracker = LearningTracker()
    weights = tracker.load_weights()
    with st.spinner(f"Scanning {len(universe)} tickers across price, news, and sentiment…"):
        alerts = generate_alerts(
            universe,
            weights=weights,
            include_intraday=include_intraday,
            fetch_social=fetch_social,
        )
        tracker.log_alerts(alerts)
    st.session_state["alerts"] = alerts
    st.session_state["weights"] = weights

alerts = st.session_state["alerts"]
weights = st.session_state.get("weights", {})

min_conf = Confidence(min_conf_label)
filtered = [a for a in alerts if a.confidence.rank >= min_conf.rank]


# ---------- header ----------
col_t, col_t2, col_t3, col_t4 = st.columns([2, 1, 1, 1])
col_t.title("📈 Trading101")
col_t.caption("AI-powered short-term trading intelligence — for research, not trade execution.")
col_t2.metric("Alerts", len(filtered))
col_t3.metric("High conf.", sum(1 for a in filtered if a.confidence is Confidence.HIGH))
bull = sum(1 for a in alerts if a.raw_score >= 0.5)
total = max(len(alerts), 1)
col_t4.metric("Market Tone", f"{int(bull/total*100)}% bullish setups")

st.divider()

tab_alerts, tab_chart, tab_learning = st.tabs(["🚨 Alerts", "📊 Charts", "🧠 Learning"])


# ---------- alerts tab ----------

with tab_alerts:
    if not filtered:
        st.info("No alerts at the selected confidence level. Try lowering the threshold or expanding the watchlist.")
    for a in filtered:
        with st.container():
            st.markdown(
                f"<div class='alert-card conf-{a.confidence.value}'>"
                f"<span class='ticker-big'>{a.ticker}</span> &nbsp; "
                f"<span class='price-big'>${a.current_price:,.2f}</span> &nbsp; "
                f"<span class='signal-pill'>{a.setup_type.value}</span>"
                f"<span class='signal-pill'>{a.catalyst.value}</span>"
                f"<span class='signal-pill'>{a.time_horizon.value}</span>"
                f"<span class='signal-pill'>score {a.raw_score:.2f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Details", expanded=(a.confidence is Confidence.HIGH)):
                left, right = st.columns(2)
                left.markdown("**Key Signals**")
                for s in a.key_signals:
                    left.markdown(f"- {s}")
                right.markdown("**Risk Factors**")
                for r in a.risk_factors:
                    right.markdown(f"- {r}")
                st.markdown(f"**Suggested Strategy:** {a.suggested_strategy}")
                if a.headline:
                    if a.headline_url:
                        st.markdown(f"📰 [{a.headline}]({a.headline_url})")
                    else:
                        st.markdown(f"📰 {a.headline}")


# ---------- chart tab ----------

with tab_chart:
    tickers = [a.ticker for a in filtered] or _normalize_universe(universe_text)
    selected = st.selectbox("Ticker", tickers)
    if selected:
        md = get_market_data(selected, history_days=180)
        if md is None or md.history.empty:
            st.warning("No data available for this ticker.")
        else:
            df = md.history.copy()
            df["SMA20"] = sma(df["Close"], 20)
            df["SMA50"] = sma(df["Close"], 50)
            df["EMA9"] = ema(df["Close"], 9)
            df["RSI14"] = rsi(df["Close"], 14)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Price",
            ))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA 20", line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50", line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], name="EMA 9", line=dict(width=1, dash="dot")))
            fig.update_layout(
                height=520, xaxis_rangeslider_visible=False,
                template="plotly_dark", title=f"{selected} — daily",
            )
            st.plotly_chart(fig, use_container_width=True)

            cols = st.columns(4)
            cols[0].metric("Last", f"${md.last_price:,.2f}", f"{md.pct_change_today:+.2f}%")
            cols[1].metric("Rel. Volume", f"{md.relative_volume:.2f}x")
            cols[2].metric("RSI(14)", f"{df['RSI14'].iloc[-1]:.1f}")
            mc = md.market_cap
            cols[3].metric("Market Cap", f"${mc/1e9:,.1f}B" if mc else "—")

            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI", line=dict(width=1.5)))
            rsi_fig.add_hline(y=70, line_dash="dot", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dot", line_color="green")
            rsi_fig.update_layout(height=200, template="plotly_dark", title="RSI(14)")
            st.plotly_chart(rsi_fig, use_container_width=True)


# ---------- learning tab ----------

with tab_learning:
    tracker = LearningTracker()
    stats = tracker.stats()
    cols = st.columns(4)
    cols[0].metric("Alerts logged", stats["alerts_logged"])
    cols[1].metric("Outcomes graded", stats["outcomes_graded"])
    cols[2].metric("Win rate", f"{(stats['win_rate'] or 0)*100:.1f}%" if stats["win_rate"] is not None else "—")
    cols[3].metric("Avg return", f"{stats['avg_return_pct'] or 0:+.2f}%" if stats["avg_return_pct"] is not None else "—")

    st.subheader("Current signal weights")
    st.json(stats["current_weights"])

    if stats["by_setup"]:
        st.subheader("Performance by setup type")
        df_setup = pd.DataFrame(stats["by_setup"]).T
        st.dataframe(df_setup, use_container_width=True)

    st.divider()
    if st.button("📊 Grade pending outcomes (5d horizon)"):
        graded = tracker.grade_pending(horizon_days=5)
        st.success(f"Graded {len(graded)} outcomes.")
    if st.button("🧠 Adjust weights from history"):
        new_w = tracker.adjust_weights()
        st.success("Weights updated.")
        st.json(new_w)
