"""Step 5: Trading-style dashboard for shares, forex, crypto, treasuries and gold.

Run from the project folder (forecaster) with:
    streamlit run src/app.py
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

try:
    from src.fetch import fetch_prices, DATA_DIR
    from src.model import (build_dataset, walk_forward, score, latest_forecast,
                           forecast_range, available_models, model_columns,
                           range_coverage, MODEL_INFO, MIN_TRAIN_DAYS, GAP_DAYS, TEST_DAYS,
                           MIN_TEST_DAYS, default_models,
                           TECH_FEATURES, FUND_FEATURES, HORIZON)
    from src.catalog import CATALOG, NAME_OF, symbols
except ImportError:
    from fetch import fetch_prices, DATA_DIR
    from model import (build_dataset, walk_forward, score, latest_forecast,
                       forecast_range, available_models, model_columns,
                       range_coverage, MODEL_INFO, MIN_TRAIN_DAYS, GAP_DAYS, TEST_DAYS,
                           MIN_TEST_DAYS, default_models,
                       TECH_FEATURES, FUND_FEATURES, HORIZON)
    from catalog import CATALOG, NAME_OF, symbols

ALL, MINE = "All", "⭐ My added"
DAYS_NEEDED = MIN_TRAIN_DAYS + GAP_DAYS + MIN_TEST_DAYS   # 2 years training + gap + 1 quarter


def usable_days(data, features):
    """Rows where every feature and the 20-day target are known."""
    return data.dropna(subset=list(features) + ["target"])

WATCHLIST_FILE = DATA_DIR / "watchlist.json"

# ---------- Asset classes ----------
# Each class has its own header labels and symbol format for Yahoo Finance.
# The assets in each dropdown come from catalog.py.

ASSET_CLASSES = {
    "shares": {
        "button": "📈  Shares", "select": "Share", "add": "Add another share",
        "placeholder": "Ticker, e.g. AMZN, SAP.DE",
        "hint": "Use the stock ticker, e.g. AAPL. German shares need .DE (SAP.DE), "
                "London .L, Paris .PA.",
    },
    "forex": {
        "button": "💱  Forex", "select": "Currency pair", "add": "Add another pair",
        "placeholder": "e.g. EUR/MXN, GBP/INR",
        "hint": "Use two 3-letter currency codes, e.g. EUR/USD or GBPJPY.",
    },
    "crypto": {
        "button": "🪙  Crypto", "select": "Coin", "add": "Add another coin",
        "placeholder": "Coin symbol, e.g. KAS, RNDR",
        "hint": "Use the coin symbol, e.g. BTC or ETH. It's priced in US dollars.",
    },
    "treasuries": {
        "button": "🏛️  Treasuries", "select": "Bond / yield", "add": "Add another bond ETF",
        "placeholder": "Ticker, e.g. VCIT, MUB",
        "hint": "Use a bond ETF ticker (e.g. VCIT) or a yield: 13W, 5Y, 10Y, 30Y.",
    },
    "gold": {
        "button": "🥇  Gold & metals", "select": "Metal", "add": "Add another metal asset",
        "placeholder": "Ticker, e.g. PHYS, SIVR",
        "hint": "Use GOLD, SILVER, PLATINUM, COPPER, or an ETF ticker like GLD.",
    },
}

YIELD_ALIASES = {"13W": "^IRX", "3M": "^IRX", "5Y": "^FVX", "10Y": "^TNX", "30Y": "^TYX"}
METAL_ALIASES = {"GOLD": "GC=F", "SILVER": "SI=F", "PLATINUM": "PL=F",
                 "PALLADIUM": "PA=F", "COPPER": "HG=F", "ALUMINIUM": "ALI=F", "ALUMINUM": "ALI=F"}


def short_code(symbol: str) -> str:
    """Short code people recognise: EURUSD=X -> EUR/USD, BTC-USD -> BTC, SUI20947-USD -> SUI."""
    if symbol.endswith("=X"):
        pair = symbol[:-2]
        return f"{pair[:3]}/{pair[3:]}"
    m = re.match(r"^([A-Z]+?)\d*-(USD|EUR)$", symbol)
    if m:
        return m.group(1) if m.group(2) == "USD" else f"{m.group(1)}/EUR"
    return symbol


def find_in_catalog(text: str, asset: str):
    """Match what the user typed (ticker, code or name) against the catalogue."""
    t = text.strip().upper()
    if not t:
        return None
    for s in symbols(asset):
        if t in (s.upper(), short_code(s).upper(), short_code(s).replace("/", "").upper(),
                 NAME_OF[s].upper()):
            return s
    return None


def to_symbol(text: str, asset: str) -> str:
    """Turn what the user typed into a Yahoo Finance symbol."""
    s = text.strip().upper()
    letters = re.sub(r"[^A-Z]", "", s)
    if asset == "forex":
        if s.endswith("=X"):
            return s
        return f"{letters[:6]}=X" if len(letters) >= 6 else s
    if asset == "crypto":
        coin = re.split(r"[-/ ]", s)[0]
        return f"{coin}-USD"
    if asset == "treasuries":
        return YIELD_ALIASES.get(s.replace(" ", ""), s)
    if asset == "gold":
        return METAL_ALIASES.get(letters, s)
    return s


def display(symbol: str) -> str:
    """Short name, e.g. EURUSD=X -> EUR/USD, ^TNX -> 10-year Treasury yield."""
    if symbol.startswith("^") or symbol.endswith("=F"):
        return NAME_OF.get(symbol, symbol)
    return short_code(symbol)


def label(symbol: str) -> str:
    """Dropdown text: code plus name, so you can search by either."""
    name = NAME_OF.get(symbol)
    code = display(symbol)
    if not name or name == code or symbol.startswith("^") or symbol.endswith("=F"):
        return code
    if f"({code})" in name:                  # e.g. "20+ year Treasuries (TLT)"
        return name
    return f"{code}  ·  {name}"


def is_yield(symbol: str) -> bool:
    return symbol in ("^IRX", "^FVX", "^TNX", "^TYX")


# ---------- Watchlists (saved so they survive restarts) ----------

def load_watchlists() -> dict:
    """Assets the user added that aren't in the catalogue."""
    lists = {k: [] for k in ASSET_CLASSES}
    try:
        saved = json.loads(WATCHLIST_FILE.read_text())
        if isinstance(saved, list):          # older version saved shares only
            saved = {"shares": saved}
        for k, v in saved.items():
            if k in lists:
                built_in = set(symbols(k))
                lists[k] = [s for s in v if s not in built_in]
    except Exception:
        pass
    return lists


def save_watchlists(lists: dict):
    WATCHLIST_FILE.write_text(json.dumps(lists, indent=2))


# ---------- Colours and styling ----------

BG, PANEL, BORDER = "#131722", "#1b1f2b", "#2a2e39"
TEXT, MUTED, GRID = "#d1d4dc", "#787b86", "#1f2430"
UP, DOWN, ACCENT = "#26a69a", "#ef5350", "#2962ff"
CHART_HEIGHT = 640

CSS = f"""
<style>
.stApp {{ background: {BG}; color: {TEXT}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding: 1rem 1.2rem 1rem 1.2rem; max-width: 100%; }}
section[data-testid="stSidebar"] {{ background: #0f121a; border-right: 1px solid {BORDER}; }}

/* Pull the sidebar content up to the very top */
[data-testid="stSidebarHeader"] {{ padding: .4rem .6rem 0 .6rem; height: 2rem; min-height: 0; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0 !important; }}
.app-title {{ font-size: 26px; font-weight: 700; color: {TEXT}; line-height: 1.15;
              margin: -1.6rem 0 12px 0; letter-spacing: -.01em; }}
.app-title span {{ color: {ACCENT}; }}

/* Asset card under the title */
.asset-card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
               padding: 12px 14px; margin-bottom: 4px; }}
.asset-name {{ font-size: 21px; font-weight: 700; color: {TEXT}; line-height: 1.2; }}
.asset-sub {{ font-size: 12px; color: {MUTED}; margin: 2px 0 8px; }}
.asset-price {{ font-size: 24px; font-weight: 600; color: {TEXT}; line-height: 1.1; }}
.asset-price small {{ font-size: 12px; color: {MUTED}; font-weight: 400; }}
.asset-change {{ font-size: 13px; margin: 2px 0 10px; }}
.asset-signal {{ font-size: 12px; font-weight: 700; letter-spacing: .04em; text-align: center;
                 border-radius: 4px; padding: 6px; margin-bottom: 10px; }}
.asset-signal.bull {{ border: 1px solid {UP}; color: {UP}; background: rgba(38,166,154,.08); }}
.asset-signal.bear {{ border: 1px solid {DOWN}; color: {DOWN}; background: rgba(239,83,80,.08); }}
.asset-signal.flat {{ border: 1px solid {MUTED}; color: {MUTED}; }}
.asset-card .row {{ font-size: 12.5px; padding: 2px 0; gap: 10px; }}
.asset-card .row span:last-child {{ text-align: right; }}
section[data-testid="stSidebar"] .stButton button {{ justify-content: flex-start;
    border: 1px solid {BORDER}; padding: .55rem .8rem; }}
.side-label {{ font-size: 11px; letter-spacing: .08em; color: {MUTED};
               text-transform: uppercase; margin: 6px 0 4px; }}

/* Top stats bar */
.statbar {{ display: flex; flex-wrap: wrap; background: {PANEL};
           border: 1px solid {BORDER}; border-radius: 6px; margin: 4px 0 10px; }}
.stat {{ flex: 1 1 130px; padding: 10px 16px; border-right: 1px solid {BORDER}; }}
.stat:last-child {{ border-right: none; }}
.stat .label {{ font-size: 11px; letter-spacing: .06em; color: {MUTED}; text-transform: uppercase; }}
.stat .value {{ font-size: 17px; font-weight: 600; color: {TEXT}; margin-top: 2px; }}
.up {{ color: {UP} !important; }}
.down {{ color: {DOWN} !important; }}

/* Right-hand signal panel: same height as the chart, fills the column */
.panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px;
          padding: 14px 16px; min-height: {CHART_HEIGHT + 52}px; box-sizing: border-box;
          display: flex; flex-direction: column; }}
.panel h3 {{ margin: 0; font-size: 18px; color: {TEXT}; }}
.panel .sub {{ font-size: 12px; color: {MUTED}; }}
.panel .price {{ font-size: 24px; font-weight: 600; margin: 4px 0 12px; }}
.sides {{ display: flex; gap: 8px; margin-bottom: 10px; }}
.side {{ flex: 1; text-align: center; border-radius: 4px; padding: 8px 4px;
         border: 1px solid {BORDER}; color: {MUTED}; font-size: 12px; }}
.side b {{ display: block; font-size: 16px; margin-top: 2px; }}
.side.sell.on {{ border-color: {DOWN}; color: {DOWN}; background: rgba(239,83,80,.10); }}
.side.buy.on  {{ border-color: {UP};   color: {UP};   background: rgba(38,166,154,.10); }}
.signal {{ text-align: center; border-radius: 4px; padding: 10px; font-weight: 700;
           letter-spacing: .05em; margin-bottom: 12px; }}
.signal.bull {{ border: 1px solid {UP}; color: {UP}; background: rgba(38,166,154,.08); }}
.signal.bear {{ border: 1px solid {DOWN}; color: {DOWN}; background: rgba(239,83,80,.08); }}
.signal.flat {{ border: 1px solid {MUTED}; color: {MUTED}; }}
.summary-title {{ font-size: 13px; font-weight: 600; color: {TEXT};
                  border-top: 1px solid {BORDER}; padding-top: 10px; margin: 6px 0 6px; }}
.row {{ display: flex; justify-content: space-between; font-size: 13px; padding: 3px 0; }}
.row span:first-child {{ color: {MUTED}; }}
.bar {{ height: 6px; background: {GRID}; border-radius: 3px; margin: 2px 0 8px; position: relative; }}
.bar i {{ position: absolute; top: 0; height: 6px; border-radius: 3px; }}
.perf {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; margin-bottom: 4px; }}
.perf div {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 4px;
             text-align: center; padding: 6px 2px; font-size: 11px; color: {MUTED}; }}
.perf b {{ display: block; font-size: 13px; margin-top: 2px; }}
.range {{ position: relative; height: 6px; border-radius: 3px; margin: 8px 0 4px;
          background: linear-gradient(90deg, {DOWN}, #f5a623, {UP}); }}
.range i {{ position: absolute; top: -4px; width: 3px; height: 14px; background: {TEXT};
            border-radius: 2px; }}
.note {{ font-size: 11px; color: {MUTED}; margin-top: auto; padding-top: 12px; }}
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 10px 14px; }}
</style>
"""


# ---------- Calculations (cached so the page stays fast) ----------

STALE_AFTER_HOURS = 12
CACHE_SECONDS = 6 * 3600   # forget in-memory results after 6 hours so new prices show up


def update_prices_if_old(symbol: str):
    """Download fresh prices if the saved file is older than 12 hours.
    If the download fails (e.g. no internet), keep using the saved file."""
    path = DATA_DIR / f"{symbol}_prices.csv"
    if not path.exists():
        return
    age_hours = (pd.Timestamp.now().timestamp() - path.stat().st_mtime) / 3600
    if age_hours > STALE_AFTER_HOURS:
        try:
            fetch_prices(symbol, refresh=True)
        except Exception:
            pass


@st.cache_data(show_spinner="Loading data...", ttl=CACHE_SECONDS)
def load_data(symbol: str, asset: str, refresh: bool = False) -> pd.DataFrame:
    update_prices_if_old(symbol)
    df = build_dataset(symbol, refresh=refresh, with_fundamentals=(asset == "shares"))
    ohlcv = fetch_prices(symbol)[["Open", "High", "Low", "Volume"]]
    ohlcv.index = pd.DatetimeIndex(ohlcv.index).astype("datetime64[ns]")
    df = df.join(ohlcv)
    df["sma50"] = df["Close"].rolling(50).mean()
    df["sma200"] = df["Close"].rolling(200).mean()
    return df


INFO_KEYS = {"name": ("longName", "shortName"), "currency": ("currency",),
             "exchange": ("fullExchangeName", "exchange"), "sector": ("sector",),
             "industry": ("industry",), "market_cap": ("marketCap",),
             "ytd_return": ("ytdReturn",)}


@st.cache_data(show_spinner=False)
def fetch_info(symbol: str) -> dict:
    """Name, exchange, sector etc. from Yahoo. Saved to a file so it's only
    downloaded once. Returns an empty dict if Yahoo has nothing."""
    path = DATA_DIR / f"{symbol}_info.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    try:
        raw = yf.Ticker(symbol).info or {}
    except Exception:
        return {}
    info = {}
    for key, options in INFO_KEYS.items():
        for o in options:
            if raw.get(o) not in (None, ""):
                info[key] = raw[o]
                break
    if info:
        path.write_text(json.dumps(info, default=str))
    return info


def pick_features(data, use_fundamentals):
    features = TECH_FEATURES + FUND_FEATURES if use_fundamentals else TECH_FEATURES
    return data.dropna(subset=features), features


MODEL_CACHE_DIR = DATA_DIR / "model_cache"
MODEL_CACHE_VERSION = 3   # change this number after editing model settings, to ignore old results


def model_cache_file(symbol, use_fundamentals, models, data_date):
    """One file per asset + settings + latest price date. New prices = new file."""
    key = f"{MODEL_CACHE_VERSION}|{symbol}|{use_fundamentals}|{','.join(models)}|{TEST_DAYS}"
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", symbol)
    return MODEL_CACHE_DIR / f"{safe}__{data_date}__{digest}.pkl", safe


@st.cache_data(show_spinner="Testing the models on this asset (only the first time, "
                            "usually under a minute)...", ttl=CACHE_SECONDS)
def run_models(symbol, asset, use_fundamentals, models: tuple, data_date: str) -> dict:
    """Walk-forward test, today's forecast and the 80% range, in one go.
    Saved to data/model_cache, so reopening the asset (even after a restart) is instant
    until new prices arrive."""
    path, safe = model_cache_file(symbol, use_fundamentals, models, data_date)
    if path.exists():
        try:
            return pd.read_pickle(path)
        except Exception:
            pass   # damaged file: just recalculate

    data, features = pick_features(load_data(symbol, asset), use_fundamentals)
    result = {
        "preds": walk_forward(data, features, list(models)),
        "forecast": latest_forecast(data, features, list(models)),
        "range": forecast_range(data, features),
    }

    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    for old in MODEL_CACHE_DIR.glob(f"{safe}__*.pkl"):     # remove results for older prices
        if f"__{data_date}__" not in old.name:
            old.unlink(missing_ok=True)
    try:
        pd.to_pickle(result, path)
    except Exception:
        pass
    return result


def equity_curves(preds, close):
    """Hold on days the model predicts 'up', cash otherwise. No trading costs."""
    daily_ret = close.pct_change().shift(-1).reindex(preds.index).fillna(0)
    curves = {"Buy & hold": (1 + daily_ret).cumprod()}
    for name in model_columns(preds):
        if name != "Baseline":
            curves[name] = (1 + (preds[name] > 0).astype(float) * daily_ret).cumprod()
    return pd.DataFrame(curves)


def best_model(scores):
    if scores is None:
        return None, None
    s = scores.drop(index="Baseline", errors="ignore").dropna(subset=["IC"])
    return (s["IC"].idxmax(), s.loc[s["IC"].idxmax()]) if len(s) else (None, None)


def period_return(close, days):
    return close.iloc[-1] / close.iloc[-days - 1] - 1 if len(close) > days else np.nan


# ---------- Formatting ----------

def pct(x, digits=1, sign=False):
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"


def num(x, digits=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:,.{digits}f}"


def price(x):
    """More decimals for small prices (forex pairs, yields)."""
    if x is None or pd.isna(x):
        return "n/a"
    digits = 2 if abs(x) >= 10 else 4 if abs(x) >= 1 else 5
    return f"{x:,.{digits}f}"


def cls(x):
    return "" if x is None or pd.isna(x) else ("up" if x > 0 else "down" if x < 0 else "")


def dark(fig, height=420, title=None):
    fig.update_layout(
        template="plotly_dark", height=height, title=title,
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT, size=12),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        hovermode="x unified", legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showspikes=True, spikecolor=MUTED,
                     spikethickness=1, spikedash="dot")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, side="right")
    return fig


# ---------- Header: asset picker ----------

def ticker_picker(asset: str) -> str:
    """Header: [Group] [Asset, searchable] [Add another ___ Add] [Remove].
    The asset list comes from catalog.py plus anything the user added."""
    cfg = ASSET_CLASSES[asset]
    lists = st.session_state.watchlists
    mine = lists[asset]
    group_key, key = f"group_{asset}", f"selected_{asset}"

    c_group, c_select, c_add, c_remove = st.columns([1.3, 2.4, 2.2, 0.8],
                                                    vertical_alignment="bottom")

    # Handle the Add box first, so a new asset can be selected straight away
    with c_add:
        with st.form(f"add_{asset}", clear_on_submit=True, border=False):
            f1, f2 = st.columns([3, 1], vertical_alignment="bottom")
            typed = f1.text_input(cfg["add"], placeholder=cfg["placeholder"])
            added = f2.form_submit_button("Add", use_container_width=True)

    if added and typed.strip():
        found = find_in_catalog(typed, asset)
        symbol = found or to_symbol(typed, asset)
        if found or symbol in mine:
            st.session_state[group_key] = ALL
            st.session_state[key] = symbol
            if found:
                st.toast(f"{label(symbol)} is already in the list, selected it.")
        else:
            with st.spinner(f"Downloading {display(symbol)}..."):
                try:
                    load_data(symbol, asset)
                    mine.append(symbol)
                    save_watchlists(lists)
                    st.session_state[group_key] = MINE
                    st.session_state[key] = symbol
                except Exception:
                    st.error(f"Couldn't find **{typed.strip()}** on Yahoo Finance. {cfg['hint']}")

    # Group filter
    groups = [ALL] + list(CATALOG[asset]) + ([MINE] if mine else [])
    if st.session_state.get(group_key) not in groups:
        st.session_state[group_key] = ALL
    with c_group:
        group = st.selectbox("Group", groups, key=group_key)

    # Assets in that group (searchable: click the box and start typing)
    if group == ALL:
        options = symbols(asset) + mine
    elif group == MINE:
        options = mine
    else:
        options = symbols(asset, group)
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    with c_select:
        symbol = st.selectbox(f"{cfg['select']} ({len(options)}) · type to search",
                              options, key=key, format_func=label)

    # Only assets you added yourself can be removed
    with c_remove:
        is_mine = symbol in mine
        if st.button("Remove", use_container_width=True, disabled=not is_mine,
                     help=f"Remove {display(symbol)} from your added assets" if is_mine
                     else "Built-in assets can't be removed"):
            mine.remove(symbol)
            save_watchlists(lists)
            st.session_state.pop(key, None)
            st.rerun()
    return symbol


# ---------- Page sections ----------

def stat_bar(asset, data, forecast, scores):
    last, close = data.iloc[-1], data["Close"]
    change = close.iloc[-1] / close.iloc[-2] - 1
    avg_fc = np.mean(list(forecast.values()))
    _, best = best_model(scores)
    year = close.iloc[-252:]

    stats = [("Last", price(last["Close"]), ""),
             ("Change (1D)", pct(change, 2, True), cls(change))]
    if asset == "shares" and pd.notna(last.get("market_cap")):
        stats += [("Market cap", f"${last['market_cap'] / 1e9:,.0f} bn", ""),
                  ("P/E (TTM)", num(last.get("pe_ttm"), 1), "")]
    else:
        stats += [("52W high", price(year.max()), ""), ("52W low", price(year.min()), "")]
    stats += [(f"{HORIZON}D forecast", pct(avg_fc, 1, True), cls(avg_fc)),
              ("Best model hit rate", pct(best["Hit rate"]) if best is not None else "n/a", ""),
              ("Volatility (20D)", pct(last["vol_20d"]), "")]

    html = "".join(f'<div class="stat"><div class="label">{l}</div>'
                   f'<div class="value {c}">{v}</div></div>' for l, v, c in stats)
    st.markdown(f'<div class="statbar">{html}</div>', unsafe_allow_html=True)


def price_chart(asset, data, symbol, months, indicators, forecast=None, rng=None):
    d = data[data.index >= data.index[-1] - pd.DateOffset(months=months)]
    show_vol = "Volume" in indicators and d["Volume"].fillna(0).sum() > 0
    show_rsi = "RSI" in indicators

    heights = [1.0] + ([0.18] if show_vol else []) + ([0.22] if show_rsi else [])
    heights = [h / sum(heights) for h in heights]
    fig = make_subplots(rows=len(heights), cols=1, shared_xaxes=True,
                        row_heights=heights, vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name=display(symbol), increasing_line_color=UP, decreasing_line_color=DOWN,
        increasing_fillcolor=UP, decreasing_fillcolor=DOWN), row=1, col=1)
    if "SMA 50" in indicators:
        fig.add_trace(go.Scatter(x=d.index, y=d["sma50"], name="SMA 50",
                                 line=dict(width=1.2, color="#f5a623")), row=1, col=1)
    if "SMA 200" in indicators:
        fig.add_trace(go.Scatter(x=d.index, y=d["sma200"], name="SMA 200",
                                 line=dict(width=1.2, color=ACCENT)), row=1, col=1)

    last = d["Close"].iloc[-1]
    if forecast and rng:
        # Forecast cone: from today's price to the 80% range in 20 trading days
        t0 = d.index[-1]
        t1 = (t0 + pd.Timedelta(days=HORIZON) if asset == "crypto"
              else t0 + pd.tseries.offsets.BDay(HORIZON))
        lo, hi = rng
        avg = float(np.mean(list(forecast.values())))
        fig.add_trace(go.Scatter(
            x=[t0, t1, t1, t0], y=[last, last * (1 + hi), last * (1 + lo), last],
            fill="toself", fillcolor="rgba(41,98,255,0.15)", line=dict(width=0),
            name="80% range", hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[t0, t1], y=[last, last * (1 + avg)], name=f"Forecast ({HORIZON}d)",
            mode="lines+markers", marker=dict(size=[0, 7]),
            line=dict(color=ACCENT, width=2, dash="dash")), row=1, col=1)
    color = UP if last >= d["Open"].iloc[-1] else DOWN
    fig.add_hline(y=last, line=dict(color=color, width=1, dash="dot"), row=1, col=1,
                  annotation_text=f" {price(last)} ", annotation_position="right",
                  annotation=dict(bgcolor=color, font=dict(color="white", size=11)))

    r = 2
    if show_vol:
        colors = np.where(d["Close"] >= d["Open"], UP, DOWN)
        fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume",
                             marker_color=colors, opacity=0.6), row=r, col=1)
        r += 1
    if show_rsi:
        fig.add_trace(go.Scatter(x=d.index, y=d["rsi_14"], name="RSI 14",
                                 line=dict(width=1.2, color="#b39ddb")), row=r, col=1)
        fig.add_hline(y=70, line=dict(color=DOWN, width=0.8, dash="dot"), row=r, col=1)
        fig.add_hline(y=30, line=dict(color=UP, width=0.8, dash="dot"), row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)

    dark(fig, CHART_HEIGHT)
    fig.update_layout(xaxis_rangeslider_visible=False)
    if asset != "crypto":   # crypto trades at weekends, everything else doesn't
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, use_container_width=True)


def model_signal(forecast, symbol):
    """Overall signal from all models: (label, css class, average, models up, models down)."""
    values = list(forecast.values())
    avg = float(np.mean(values))
    n_up = sum(v > 0 for v in values)
    n_down = len(values) - n_up
    if avg > 0.005 and n_up > n_down:
        sig, sig_cls = "BULLISH", "bull"
    elif avg < -0.005 and n_down > n_up:
        sig, sig_cls = "BEARISH", "bear"
    else:
        sig, sig_cls = "NEUTRAL", "flat"
    if is_yield(symbol):   # rising yields = falling bond prices
        sig = {"BULLISH": "YIELD UP", "BEARISH": "YIELD DOWN"}.get(sig, sig)
    return sig, sig_cls, avg, n_up, n_down


def asset_card(asset, symbol, data, forecast):
    """Sidebar card: asset name in large letters plus the key facts for its market."""
    info = fetch_info(symbol)
    close = data["Close"]
    last = data.iloc[-1]
    change = close.iloc[-1] / close.iloc[-2] - 1
    year = close.iloc[-252:]
    sig, sig_cls, avg, _, _ = model_signal(forecast, symbol)

    if asset == "shares":                     # Yahoo's full company name is best here
        name = info.get("name") or NAME_OF.get(symbol) or symbol
    else:
        name = NAME_OF.get(symbol) or info.get("name") or display(symbol)
    if asset == "crypto" and name.endswith(" USD"):
        name = name[:-4]                     # "Bitcoin USD" -> "Bitcoin"
    if asset == "forex":
        name = display(symbol)
        sub = NAME_OF.get(symbol) or info.get("name") or "Currency pair"
    else:
        sub = symbol if name != symbol else ASSET_CLASSES[asset]["select"]

    currency = info.get("currency", "")
    exchange = info.get("exchange", "")

    if is_yield(symbol):
        price_line = f"{close.iloc[-1]:.2f}%"
        bp_1d = (close.iloc[-1] - close.iloc[-2]) * 100
        change_line = f"{bp_1d:+.0f} bp today"
        change_cls = cls(bp_1d)
    else:
        price_line = f"{price(close.iloc[-1])} <small>{currency}</small>"
        change_line = f"{pct(change, 2, True)} today"
        change_cls = cls(change)

    rows = []
    if asset == "shares":
        if info.get("sector"):
            rows.append(("Sector", info["sector"]))
        if info.get("industry"):
            rows.append(("Industry", info["industry"]))
        if pd.notna(last.get("market_cap")):
            rows.append(("Market cap", f"${last['market_cap'] / 1e9:,.0f} bn"))
        if pd.notna(last.get("pe_ttm")):
            rows.append(("P/E (TTM)", num(last["pe_ttm"], 1)))
    elif asset == "crypto":
        if info.get("market_cap"):
            rows.append(("Market cap", f"${info['market_cap'] / 1e9:,.0f} bn"))
        rows.append(("Trades", "24/7"))
    elif asset == "forex":
        pair = symbol[:-2]
        rows.append(("Meaning", f"1 {pair[:3]} = {price(close.iloc[-1])} {pair[3:]}"))
    elif is_yield(symbol):
        bp_1y = (close.iloc[-1] - year.iloc[0]) * 100
        rows.append(("1Y change", f"{bp_1y:+.0f} bp"))
        rows.append(("Bond prices", "fall when yield rises"))
    elif asset == "treasuries" and info.get("ytd_return") is not None:
        rows.append(("YTD return", pct(info["ytd_return"], 1, True)))

    if not is_yield(symbol):
        rows.append(("1Y return", pct(period_return(close, 252), 1, True)))
    if is_yield(symbol):
        rows.append(("52W range", f"{year.min():.2f}% – {year.max():.2f}%"))
    else:
        rows.append(("52W range", f"{price(year.min())} – {price(year.max())}"))
    rows.append(("Volatility", pct(last["vol_20d"], 0) + " a year"))
    if exchange:
        rows.append(("Exchange", exchange))
    rows.append(("Data up to", data.index[-1].strftime("%d %b %Y")))

    html_rows = "".join(f'<div class="row"><span>{k}</span><span>{v}</span></div>' for k, v in rows)
    return f"""
    <div class="asset-card">
      <div class="asset-name">{name}</div>
      <div class="asset-sub">{sub}</div>
      <div class="asset-price">{price_line}</div>
      <div class="asset-change {change_cls}">{change_line}</div>
      <div class="asset-signal {sig_cls}">{sig} · {pct(avg, 1, True)} in {HORIZON}d</div>
      {html_rows}
    </div>
    """


def signal_panel(asset, symbol, data, forecast, scores, rng=None, coverage=np.nan):
    close = data["Close"]
    last = data.iloc[-1]
    change = close.iloc[-1] / close.iloc[-2] - 1
    values = list(forecast.values())
    sig, sig_cls, avg, n_up, n_down = model_signal(forecast, symbol)

    name, best = best_model(scores)
    horizon = f"{HORIZON} days" if asset == "crypto" else f"{HORIZON} trading days"

    # One small bar per model, centred at zero
    scale = max(0.05, max(abs(v) for v in values))
    bars = ""
    for m, v in forecast.items():
        width = abs(v) / scale * 50
        left = 50 if v >= 0 else 50 - width
        bars += (f'<div class="row"><span>{m}</span><span class="{cls(v)}">{pct(v, 1, True)}</span></div>'
                 f'<div class="bar"><i style="left:{left}%;width:{width}%;'
                 f'background:{UP if v >= 0 else DOWN}"></i></div>')

    # Performance tiles
    perf = "".join(
        f'<div>{label}<b class="{cls(r)}">{pct(r, 1, True)}</b></div>'
        for label, r in [("1W", period_return(close, 5)), ("1M", period_return(close, 21)),
                         ("3M", period_return(close, 63)), ("6M", period_return(close, 126)),
                         ("1Y", period_return(close, 252))])

    # Where today's price sits in the 52-week range
    year = close.iloc[-252:]
    lo, hi = year.min(), year.max()
    pos = 0 if hi == lo else (close.iloc[-1] - lo) / (hi - lo) * 100

    rows = [("Horizon", horizon),
            ("Target", price(close.iloc[-1] * (1 + avg)))]
    if rng:
        rows += [("80% range", f"{price(close.iloc[-1] * (1 + rng[0]))} – "
                               f"{price(close.iloc[-1] * (1 + rng[1]))}"),
                 ("Range was right", pct(coverage, 0) + " of the time" if pd.notna(coverage)
                  else "n/a")]
    rows += [
            ("Best model", name or "n/a"),
            ("Hit rate", pct(best["Hit rate"]) if best is not None else "n/a"),
            ("IC", num(best["IC"], 3) if best is not None else "n/a"),
            ("RSI (14)", num(last["rsi_14"], 1)),
            ("vs 200-day avg", pct(last["dist_sma200"], 1, True))]
    summary = "".join(f'<div class="row"><span>{k}</span><span>{v}</span></div>' for k, v in rows)

    yield_note = ("<br>For yields, the forecast is the % change in the yield itself."
                  if is_yield(symbol) else "")

    st.markdown(f"""
    <div class="panel">
      <h3>{display(symbol)}</h3>
      <div class="sub">{symbol} · {ASSET_CLASSES[asset]['button'].split('  ')[1]}</div>
      <div class="price {cls(change)}">{price(close.iloc[-1])}
        <span style="font-size:13px">({pct(change, 2, True)})</span></div>
      <div class="sides">
        <div class="side sell {'on' if sig_cls == 'bear' else ''}">MODELS DOWN<b>{n_down}</b></div>
        <div class="side buy {'on' if sig_cls == 'bull' else ''}">MODELS UP<b>{n_up}</b></div>
      </div>
      <div class="signal {sig_cls}">{sig} · {pct(avg, 1, True)}</div>
      <div class="summary-title">Forecast by model</div>
      {bars}
      <div class="summary-title">Signal summary</div>
      {summary}
      <div class="summary-title">Performance</div>
      <div class="perf">{perf}</div>
      <div class="summary-title">52-week range</div>
      <div class="range"><i style="left:calc({pos:.1f}% - 1px)"></i></div>
      <div class="row"><span>{price(lo)}</span><span>{price(hi)}</span></div>
      <div class="note">Learning project, not investment advice. Check the Model
      performance page before trusting a signal.{yield_note}</div>
    </div>
    """, unsafe_allow_html=True)


def page_trade(asset, symbol, data, forecast, scores, rng=None, coverage=np.nan):
    left, right = st.columns([3, 1.1], gap="small")
    with left:
        c1, c2 = st.columns([1, 1.6])
        with c1:
            period = st.radio("Period", ["6M", "1Y", "3Y", "5Y", "10Y"], index=1,
                              horizontal=True, label_visibility="collapsed")
        with c2:
            options = ["SMA 50", "SMA 200", "RSI"]
            if data["Volume"].fillna(0).sum() > 0:
                options.insert(2, "Volume")
            indicators = st.multiselect("Indicators", options,
                                        default=[o for o in options if o != "RSI"],
                                        label_visibility="collapsed")
        months = {"6M": 6, "1Y": 12, "3Y": 36, "5Y": 60, "10Y": 120}[period]
        price_chart(asset, data, symbol, months, indicators, forecast, rng)
    with right:
        signal_panel(asset, symbol, data, forecast, scores, rng, coverage)


def page_fundamentals(asset, data):
    last = data.iloc[-1]
    if asset != "shares":
        st.info("Fundamentals (earnings, margins, P/E) only exist for company shares. "
                "Switch to **Shares** in the left menu to see them.")
        return
    if pd.isna(last.get("market_cap")):
        st.info("No fundamental data for this ticker (it may be an ETF or index).")
        return
    st.caption(f"Based on the report for the period ending "
               f"{pd.Timestamp(last['period_end']).date()} "
               f"({int(last['days_since_report'])} days since it became public)")

    c = st.columns(4)
    c[0].metric("Market cap", f"${last['market_cap'] / 1e9:,.0f} bn")
    c[1].metric("P/E (TTM)", num(last["pe_ttm"], 1))
    c[2].metric("P/S (TTM)", num(last["ps_ttm"], 1))
    c[3].metric("P/B", num(last["pb"], 1))
    c = st.columns(4)
    c[0].metric("Net margin", pct(last["net_margin"]))
    c[1].metric("ROE", pct(last["roe"]))
    c[2].metric("Revenue growth (YoY)", pct(last["revenue_growth_yoy"]))
    c[3].metric("FCF yield", pct(last["fcf_yield"]))

    left, right = st.columns(2)
    with left:
        d = data.dropna(subset=["pe_ttm"])
        fig = go.Figure(go.Scatter(x=d.index, y=d["pe_ttm"], name="P/E",
                                   line=dict(width=1.4, color=ACCENT)))
        st.plotly_chart(dark(fig, 360, "P/E ratio over time"), use_container_width=True)
    with right:
        m = data.dropna(subset=["net_margin"])
        fig = go.Figure()
        for col, label, color in [("gross_margin", "Gross", ACCENT),
                                  ("operating_margin", "Operating", "#f5a623"),
                                  ("net_margin", "Net", UP)]:
            fig.add_trace(go.Scatter(x=m.index, y=m[col], name=label,
                                     line=dict(width=1.4, color=color)))
        fig = dark(fig, 360, "Margins (TTM)")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


def page_models(preds, data):
    if preds.empty:
        usable = usable_days(data, TECH_FEATURES)
        since = f" (since {usable.index[0]:%b %Y})" if len(usable) else ""
        st.warning(f"Not enough history to test the models: this asset has {len(usable)} "
                   f"usable trading days{since}, and the test needs at least {DAYS_NEEDED} "
                   f"(2 years of training plus one test quarter). The first ~200 days of "
                   f"prices are also used up warming up the 200-day average, so an asset "
                   f"needs roughly 3 years of price history. The forecast still works, "
                   f"but treat it with extra caution.")
        return
    st.caption(f"Out-of-sample: tested {preds.index[0].date()} to {preds.index[-1].date()} "
               f"({len(preds)} trading days). Each prediction came from a model that had "
               f"never seen that period.")

    s = score(preds)
    st.dataframe(pd.DataFrame({
        "Hit rate": s["Hit rate"].map(pct),
        "IC": s["IC"].map(lambda x: num(x, 3)),
        "MAE": s["MAE"].map(pct),
        "% days long": s["% days long"].map(pct),
        f"Avg {HORIZON}d return when long": s["Avg ret when long"].map(pct),
    }), use_container_width=True)
    st.caption(f"Buy-and-hold average {HORIZON}-day return: {pct(preds['actual'].mean())}. "
               f"A useful model has an IC above ~0.05 and beats the Baseline row.")

    cov = range_coverage(preds)
    if pd.notna(cov):
        verdict = ("well calibrated" if abs(cov - 0.8) <= 0.05 else
                   "too narrow: real outcomes vary more than the model expects" if cov < 0.8
                   else "too wide: the model is more uncertain than it needs to be")
        st.caption(f"**80% forecast range:** the actual {HORIZON}-day return landed inside it "
                   f"**{pct(cov, 0)}** of the time ({verdict}).")

    with st.expander("About the models"):
        for name in model_columns(preds):
            if name in MODEL_INFO:
                st.markdown(f"**{name}** – {MODEL_INFO[name]}")
        st.markdown("**Baseline** – always predicts the historical average return. "
                    "Any model worth using should beat it.")
        st.markdown(f"**80% range** – two quantile models predict a pessimistic (10th "
                    f"percentile) and optimistic (90th percentile) {HORIZON}-day return.")

    models = [c for c in model_columns(preds) if c != "Baseline"]
    default = models.index("Ensemble") if "Ensemble" in models else 0
    model = st.selectbox("Model to inspect", models, index=default)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        if "_low" in preds:
            fig.add_trace(go.Scatter(x=preds.index, y=preds["_high"], line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=preds.index, y=preds["_low"], line=dict(width=0),
                                     fill="tonexty", fillcolor="rgba(41,98,255,0.12)",
                                     name="80% range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=preds.index, y=preds["actual"], name="Actual",
                                 line=dict(width=1, color=MUTED)))
        fig.add_trace(go.Scatter(x=preds.index, y=preds[model], name=model,
                                 line=dict(width=1.6, color=ACCENT)))
        fig = dark(fig, 380, f"Predicted vs actual {HORIZON}-day return")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        curves = equity_curves(preds, data["Close"])
        fig = go.Figure()
        for col in curves.columns:
            fig.add_trace(go.Scatter(x=curves.index, y=curves[col], name=col,
                                     line=dict(width=2 if col == "Buy & hold" else 1.4)))
        st.plotly_chart(dark(fig, 380, "Growth of $1 (long when model says up)"),
                        use_container_width=True)

    st.dataframe(pd.DataFrame({
        "Total return": curves.iloc[-1].sub(1).map(pct),
        "Max drawdown": curves.apply(lambda c: (c / c.cummax() - 1).min()).map(pct),
    }), use_container_width=True)
    st.caption("No trading costs or taxes. Past results don't predict future returns.")


# ---------- App ----------

def main():
    st.set_page_config(page_title="Stock Forecaster", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    if "watchlists" not in st.session_state:
        st.session_state.watchlists = load_watchlists()
    if "asset" not in st.session_state:
        st.session_state.asset = "shares"

    with st.sidebar:
        st.markdown('<div class="app-title">📈 Market <span>Forecaster</span></div>',
                    unsafe_allow_html=True)
        card_slot = st.empty()   # filled with the asset card once the data is loaded
        st.markdown('<div class="side-label">Markets</div>', unsafe_allow_html=True)
        for key, cfg in ASSET_CLASSES.items():
            active = st.session_state.asset == key
            if st.button(cfg["button"], key=f"btn_{key}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.asset = key
                st.rerun()

        st.markdown('<div class="side-label">View</div>', unsafe_allow_html=True)
        page = st.radio("View", ["📊  Chart & signal", "🏦  Fundamentals", "🧪  Model performance"],
                        label_visibility="collapsed")

        st.divider()
        asset = st.session_state.asset
        use_fundamentals = False
        if asset == "shares":
            use_fundamentals = st.toggle("Use fundamental features", value=True,
                                         help="Fundamentals cover only the last ~4-5 years, "
                                              "so the test period is shorter.")
        all_models = available_models()
        chosen = st.multiselect(
            "Models", all_models, default=default_models(), key="models",
            help="Untick a model to leave it out. The Ensemble (average) is added "
                 "automatically when 2 or more are selected. Results are saved, so each "
                 "combination is only calculated once per asset per day.")
        models = tuple(m for m in all_models if m in chosen) or ("Ridge",)
        refresh_clicked = st.button("⟳  Refresh this asset", use_container_width=True,
                                    help="Download the latest prices and reports now. "
                                         "Prices also update by themselves when older than 12 hours.")

    symbol = ticker_picker(asset)

    if refresh_clicked:
        with st.spinner(f"Downloading the latest data for {display(symbol)}..."):
            try:
                build_dataset(symbol, refresh=True, with_fundamentals=(asset == "shares"))
            except Exception as e:
                st.error(f"Refresh failed, showing saved data instead: {e}")
        st.cache_data.clear()

    try:
        data = load_data(symbol, asset)
    except Exception as e:
        st.error(f"Could not load data for {display(symbol)}: {e}")
        st.stop()

    if use_fundamentals:
        usable = usable_days(data, TECH_FEATURES + FUND_FEATURES)
        if len(usable) < DAYS_NEEDED:
            since = (f"only go back to {usable.index[0]:%b %Y} ({len(usable)} trading days)"
                     if len(usable) else "aren't available")
            st.info(f"Complete fundamentals for {display(symbol)} {since}, but testing the "
                    f"models needs at least {DAYS_NEEDED}. Using technical features only.")
            use_fundamentals = False

    result = run_models(symbol, asset, use_fundamentals, models, f"{data.index[-1]:%Y-%m-%d}")
    forecast, preds, rng = result["forecast"], result["preds"], result["range"]
    scores = score(preds) if not preds.empty else None
    coverage = range_coverage(preds) if not preds.empty else np.nan

    card_slot.markdown(asset_card(asset, symbol, data, forecast), unsafe_allow_html=True)
    stat_bar(asset, data, forecast, scores)

    if page.endswith("signal"):
        page_trade(asset, symbol, data, forecast, scores, rng, coverage)
    elif page.endswith("Fundamentals"):
        page_fundamentals(asset, data)
    else:
        page_models(preds, data)


if __name__ == "__main__":
    main()
