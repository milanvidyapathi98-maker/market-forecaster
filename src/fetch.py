"""Step 1: Download and cache price and fundamental data."""
from pathlib import Path

import pandas as pd
import yfinance as yf

# The data/ folder sits next to src/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_prices(ticker: str, period: str = "10y", refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV prices, adjusted for splits and dividends."""
    path = DATA_DIR / f"{ticker}_prices.csv"

    # Use the saved file if we already downloaded it
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = df.index.tz_localize(None)   # remove timezone for easier joins later
    df.index.name = "Date"

    df.to_csv(path)
    return df


def fetch_fundamentals(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Quarterly income statement, balance sheet and cash flow in one table.
    One row per quarter-end date, one column per line item."""
    path = DATA_DIR / f"{ticker}_fundamentals.csv"

    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    t = yf.Ticker(ticker)
    statements = [t.quarterly_income_stmt, t.quarterly_balance_sheet, t.quarterly_cashflow]

    # yfinance gives line items as rows and dates as columns, so we transpose (.T)
    frames = [s.T for s in statements if s is not None and not s.empty]
    if not frames:
        raise ValueError(f"No fundamental data returned for {ticker}")

    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]   # drop items that appear in two statements
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    df.to_csv(path)
    return df


if __name__ == "__main__":
    KEY_ITEMS = ["Total Revenue", "Net Income", "Diluted EPS",
                 "Stockholders Equity", "Free Cash Flow"]

    for ticker in ["NVDA", "GOOG"]:
        print(f"\n===== {ticker} =====")

        prices = fetch_prices(ticker, refresh=True)
        print(f"Prices: {len(prices)} rows, "
              f"{prices.index.min().date()} to {prices.index.max().date()}")
        print(prices.tail(3))

        funda = fetch_fundamentals(ticker, refresh=True)
        print(f"\nFundamentals: {len(funda)} quarters, {funda.shape[1]} line items")
        available = [c for c in KEY_ITEMS if c in funda.columns]
        print(funda[available].tail(4))