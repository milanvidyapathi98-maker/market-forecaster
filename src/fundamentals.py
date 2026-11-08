"""Step 3: Fundamental ratios, joined to daily prices without look-ahead bias."""
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from src.fetch import fetch_prices, fetch_fundamentals, DATA_DIR
except ImportError:
    from fetch import fetch_prices, fetch_fundamentals, DATA_DIR

# Days after period end before the numbers count as "public"
QUARTERLY_LAG_DAYS = 45
ANNUAL_LAG_DAYS = 60

# Features the model will use in Step 4 (all scale-free)
FUND_FEATURES = [
    "earnings_yield", "sales_yield", "book_yield", "fcf_yield",
    "gross_margin", "operating_margin", "net_margin",
    "roe", "debt_to_equity", "revenue_growth_yoy",
]

# Items summed over 4 quarters (TTM = trailing twelve months)
FLOW_ITEMS = {
    "revenue": "Total Revenue",
    "gross_profit": "Gross Profit",
    "operating_income": "Operating Income",
    "net_income": "Net Income",
    "fcf": "Free Cash Flow",
}
# Items that are a snapshot at period end (balance sheet)
STOCK_ITEMS = {
    "equity": "Stockholders Equity",
    "debt": "Total Debt",
}


# ---------- Data loading ----------

def _to_ns(index) -> pd.DatetimeIndex:
    """Make all dates the same type so they can be joined safely."""
    return pd.DatetimeIndex(pd.to_datetime(index)).astype("datetime64[ns]")


def fetch_annual_fundamentals(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Annual statements (usually ~4 years), cached like in Step 1."""
    path = DATA_DIR / f"{ticker}_fundamentals_annual.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    t = yf.Ticker(ticker)
    frames = [s.T for s in (t.income_stmt, t.balance_sheet, t.cashflow)
              if s is not None and not s.empty]
    if not frames:
        raise ValueError(f"No annual data returned for {ticker}")

    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()
    df.to_csv(path)
    return df


def fetch_splits(ticker: str, refresh: bool = False) -> pd.Series:
    """Stock split history, e.g. NVDA 10-for-1 in June 2024."""
    path = DATA_DIR / f"{ticker}_splits.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="Date", parse_dates=True)["Split"]

    s = yf.Ticker(ticker).splits
    if len(s) and getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s.index.name = "Date"
    s.name = "Split"
    s.to_csv(path)
    return s


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column, or an empty (NaN) one if this company doesn't report it."""
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def _shares(df: pd.DataFrame) -> pd.Series:
    return _col(df, "Ordinary Shares Number").fillna(_col(df, "Diluted Average Shares"))


# ---------- Building one table of reporting periods ----------

def build_periods(quarterly: pd.DataFrame, annual: pd.DataFrame,
                  splits: pd.Series) -> pd.DataFrame:
    quarterly = quarterly.copy()
    annual = annual.copy()
    quarterly.index = _to_ns(quarterly.index)
    annual.index = _to_ns(annual.index)

    # Annual reports: flow items already cover 12 months
    a = pd.DataFrame(index=annual.index)
    for key, name in {**FLOW_ITEMS, **STOCK_ITEMS}.items():
        a[key] = _col(annual, name)
    a["shares"] = _shares(annual)
    a["available"] = a.index + pd.Timedelta(days=ANNUAL_LAG_DAYS)

    # Quarterly reports: sum the last 4 quarters to get TTM
    q = pd.DataFrame(index=quarterly.index)
    for key, name in FLOW_ITEMS.items():
        q[key] = _col(quarterly, name).rolling(4, min_periods=4).sum()
    for key, name in STOCK_ITEMS.items():
        q[key] = _col(quarterly, name)
    q["shares"] = _shares(quarterly)
    q["available"] = q.index + pd.Timedelta(days=QUARTERLY_LAG_DAYS)

    # Combine; where a fiscal year-end appears in both, keep the annual version
    periods = pd.concat([a, q])
    periods = periods[~periods.index.duplicated(keep="first")]
    periods = periods.dropna(subset=["revenue"]).sort_index()
    periods.index.name = "period_end"

    periods["shares"] = split_adjust_shares(periods["shares"], splits)

    # Year-over-year revenue growth: match each period with the one ~1 year earlier
    prev = periods[["revenue"]].rename(columns={"revenue": "revenue_prev"})
    prev.index = prev.index + pd.DateOffset(years=1)
    prev.index = _to_ns(prev.index)
    periods = pd.merge_asof(periods, prev.sort_index(), left_index=True, right_index=True,
                            direction="nearest", tolerance=pd.Timedelta(days=20))
    periods["revenue_growth_yoy"] = periods["revenue"] / periods["revenue_prev"] - 1

    # Ratios that don't depend on the share price
    periods["gross_margin"] = periods["gross_profit"] / periods["revenue"]
    periods["operating_margin"] = periods["operating_income"] / periods["revenue"]
    periods["net_margin"] = periods["net_income"] / periods["revenue"]
    equity = periods["equity"].where(periods["equity"] > 0)   # negative equity -> undefined
    periods["roe"] = periods["net_income"] / equity
    periods["debt_to_equity"] = periods["debt"] / equity
    return periods


def split_adjust_shares(shares: pd.Series, splits: pd.Series) -> pd.Series:
    """Prices from Step 1 are split-adjusted, so share counts must be too.
    Some statements are already restated after a split and some aren't, so we
    only apply a split if it brings the count closer to today's level."""
    shares = shares.copy()
    valid = shares.dropna()
    if valid.empty or splits.empty:
        return shares

    latest_date, latest = valid.index[-1], valid.iloc[-1]
    reference = latest * splits[splits.index > latest_date].prod()

    for date in valid.index:
        value = shares[date]
        factor = splits[splits.index > date].prod()
        if factor != 1 and abs(np.log(value * factor / reference)) < abs(np.log(value / reference)):
            shares[date] = value * factor
    return shares


# ---------- Attaching fundamentals to daily prices ----------

def add_fundamental_features(prices: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    daily = prices[["Close"]].copy()
    daily.index = _to_ns(daily.index)
    daily = daily.rename_axis("Date").reset_index()

    p = periods.reset_index()
    p["available"] = p["available"].astype("datetime64[ns]")
    p = p.sort_values("available")

    # For each trading day, take the latest period that was already public
    merged = pd.merge_asof(daily, p, left_on="Date", right_on="available", direction="backward")
    merged = merged.set_index("Date")

    mcap = merged["Close"] * merged["shares"]
    merged["market_cap"] = mcap

    # Display versions (classic ratios)
    merged["pe_ttm"] = (mcap / merged["net_income"]).where(merged["net_income"] > 0)
    merged["ps_ttm"] = mcap / merged["revenue"]
    merged["pb"] = (mcap / merged["equity"]).where(merged["equity"] > 0)

    # Model versions ("yields" behave well even when earnings are negative)
    merged["earnings_yield"] = merged["net_income"] / mcap
    merged["sales_yield"] = merged["revenue"] / mcap
    merged["book_yield"] = merged["equity"] / mcap
    merged["fcf_yield"] = merged["fcf"] / mcap

    merged["days_since_report"] = (merged.index - merged["available"]).dt.days
    return merged.drop(columns=["Close"])


def load_fundamentals(ticker: str, refresh: bool = False):
    """Everything in one call. Returns (periods table, daily fundamentals)."""
    prices = fetch_prices(ticker)
    quarterly = fetch_fundamentals(ticker, refresh=refresh)
    annual = fetch_annual_fundamentals(ticker, refresh=refresh)
    splits = fetch_splits(ticker, refresh=refresh)
    periods = build_periods(quarterly, annual, splits)
    daily = add_fundamental_features(prices, periods)
    return periods, daily


# ---------- Run directly to check the results ----------

def _pct(x):
    return "n/a" if pd.isna(x) else f"{x:.1%}"


def _num(x, digits=1):
    return "n/a" if pd.isna(x) else f"{x:.{digits}f}"


if __name__ == "__main__":
    for ticker in ["NVDA", "GOOG"]:
        periods, daily = load_fundamentals(ticker, refresh=True)
        daily.to_csv(DATA_DIR / f"{ticker}_fundamental_daily.csv")

        print(f"\n===== {ticker} =====")
        print("Reporting periods (TTM, revenue in $ billions):")
        table = pd.DataFrame({
            "revenue_bn": (periods["revenue"] / 1e9).round(1),
            "net_margin": periods["net_margin"].map(_pct),
            "roe": periods["roe"].map(_pct),
            "rev_growth": periods["revenue_growth_yoy"].map(_pct),
            "public_from": periods["available"].dt.date,
        })
        print(table.tail(6).to_string())

        last = daily.iloc[-1]
        print(f"\nLatest ({daily.index[-1].date()}), based on period ending "
              f"{pd.Timestamp(last['period_end']).date()}:")
        print(f"  Market cap:        ${last['market_cap'] / 1e9:,.0f} bn")
        print(f"  P/E (TTM):         {_num(last['pe_ttm'])}")
        print(f"  P/S (TTM):         {_num(last['ps_ttm'])}")
        print(f"  P/B:               {_num(last['pb'])}")
        print(f"  FCF yield:         {_pct(last['fcf_yield'])}")
        print(f"  Gross margin:      {_pct(last['gross_margin'])}")
        print(f"  Operating margin:  {_pct(last['operating_margin'])}")
        print(f"  ROE:               {_pct(last['roe'])}")
        print(f"  Debt / equity:     {_num(last['debt_to_equity'], 2)}")
        print(f"  Revenue growth:    {_pct(last['revenue_growth_yoy'])}")

        covered = daily[FUND_FEATURES].dropna()
        if len(covered):
            print(f"  Fundamentals available from {covered.index[0].date()} "
                  f"({len(covered)} of {len(daily)} trading days)")
        else:
            print("  Not enough data for the full feature set yet")