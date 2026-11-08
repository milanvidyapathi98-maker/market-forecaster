"""Step 4: Combine features, define the target, and test models with walk-forward validation.

Target: the 20-trading-day forward return, Close(t+20) / Close(t) - 1.
Each model is always trained on the past and tested on the future, never the other way round.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.fetch import fetch_prices, DATA_DIR
    from src.fundamentals import load_fundamentals, FUND_FEATURES
except ImportError:
    from fetch import fetch_prices, DATA_DIR
    from fundamentals import load_fundamentals, FUND_FEATURES

# LightGBM is optional. "except Exception" also covers a DLL blocked by Windows security,
# in which case the app simply runs with the other two models.
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# Short descriptions, shown in the dashboard
MODEL_INFO = {
    "Ridge": "Linear regression with a penalty that keeps weights small. Simple, stable, "
             "hard to overfit: the benchmark the others must beat.",
    "GradBoost": "Gradient-boosted decision trees (scikit-learn). Builds many small trees, "
                 "each fixing the errors of the previous ones. Captures non-linear patterns.",
    "LightGBM": "Microsoft's gradient-boosting library. Grows trees leaf-by-leaf, so it's very "
                "fast and often slightly more accurate than other boosting libraries.",
    "Ensemble": "The average of all selected models. Averaging cancels out individual "
                "models' mistakes and is often the most reliable forecast.",
}

HORIZON = 20          # predict the return over the next 20 trading days (~1 month)
MIN_TRAIN_DAYS = 504  # at least ~2 years of history before the first test
TEST_DAYS = 126       # retrain every ~half-year; each test block is ~6 months
MIN_TEST_DAYS = 63    # need at least one quarter of test data to score the models
GAP_DAYS = HORIZON    # gap between train and test so 20-day targets don't overlap

TECH_FEATURES = [
    "ret_5d", "ret_20d", "ret_60d", "ret_120d",
    "vol_20d", "vol_60d",
    "dist_sma50", "dist_sma200", "sma50_vs_sma200",
    "rsi_14", "volume_ratio", "drawdown_252d",
]


# ---------- Features and target ----------

def add_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Price-based features. Each uses only data up to and including day t."""
    close, volume = prices["Close"], prices["Volume"]
    daily_ret = close.pct_change()
    f = pd.DataFrame(index=prices.index)

    for n in (5, 20, 60, 120):
        f[f"ret_{n}d"] = close.pct_change(n)
    f["vol_20d"] = daily_ret.rolling(20).std() * np.sqrt(252)   # annualised
    f["vol_60d"] = daily_ret.rolling(60).std() * np.sqrt(252)

    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    f["dist_sma50"] = close / sma50 - 1
    f["dist_sma200"] = close / sma200 - 1
    f["sma50_vs_sma200"] = sma50 / sma200 - 1

    # RSI(14) with Wilder smoothing
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss)

    # Forex and yields have no volume (always 0). Use 1.0 = "normal" so rows aren't dropped.
    avg_volume = volume.rolling(20).mean()
    f["volume_ratio"] = (volume / avg_volume).where(avg_volume > 0, 1.0)
    f["drawdown_252d"] = close / close.rolling(252, min_periods=1).max() - 1
    return f


def build_dataset(ticker: str, refresh: bool = False,
                  with_fundamentals: bool = True) -> pd.DataFrame:
    """One row per trading day: Close, technical features, fundamental features, target.
    Set with_fundamentals=False for forex, crypto, bonds and gold (no financial statements)."""
    prices = fetch_prices(ticker, refresh=refresh)
    prices.index = pd.DatetimeIndex(prices.index).astype("datetime64[ns]")

    tech = add_technical_features(prices)
    df = prices[["Close"]].join(tech)

    # Some tickers (ETFs, indices, some foreign stocks) have no financial statements.
    # Then we keep going with technical features only.
    if with_fundamentals:
        try:
            _, fund = load_fundamentals(ticker, refresh=refresh)
            df = df.join(fund)
        except Exception as e:
            print(f"{ticker}: no fundamentals ({e}), using technical features only")
    for col in FUND_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # The target looks 20 days ahead. It's NaN for the last 20 days (future unknown).
    df["target"] = df["Close"].shift(-HORIZON) / df["Close"] - 1
    return df


# ---------- Models ----------

def make_models(names=None) -> dict:
    """The three models (Ridge, GradBoost, LightGBM), or only the ones named in `names`.
    Settings are deliberately cautious (shallow trees, strong regularisation)
    because daily market data is very noisy and easy to overfit."""
    imp = lambda: SimpleImputer(strategy="median")   # fills gaps for models that can't handle NaN
    models = {
        "Ridge": make_pipeline(imp(), StandardScaler(), Ridge(alpha=10.0)),
        "GradBoost": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.03, max_depth=3,
            min_samples_leaf=50, l2_regularization=1.0, random_state=42),
    }
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, verbose=-1)

    if names is not None:
        models = {k: v for k, v in models.items() if k in names}
    return models


def available_models() -> list:
    return list(make_models())


def default_models() -> list:
    """All three models are ticked by default."""
    return available_models()


def make_quantile_models(low=0.1, high=0.9) -> dict:
    """Models that predict a low and a high outcome instead of the average.
    Together they give an 80% range: the return should land inside it ~8 times in 10."""
    q = lambda alpha: HistGradientBoostingRegressor(
        loss="quantile", quantile=alpha, max_iter=200, learning_rate=0.03, max_depth=3,
        min_samples_leaf=50, l2_regularization=1.0, random_state=42)
    return {"_low": q(low), "_high": q(high)}


# ---------- Walk-forward validation ----------

def walk_forward(data: pd.DataFrame, features: list, names=None,
                 with_range: bool = True) -> pd.DataFrame:
    """Expanding window: train on everything up to a cut-off, skip a 20-day gap,
    test on the next 63 days, then move forward and repeat.
    Returns one row per test day with the actual return, each model's prediction,
    the Ensemble (average of the models) and, optionally, the 80% range (_low, _high)."""
    data = data.dropna(subset=["target"])
    n = len(data)
    results = []

    start = MIN_TRAIN_DAYS
    while start + GAP_DAYS < n:
        train = data.iloc[:start]
        test = data.iloc[start + GAP_DAYS: start + GAP_DAYS + TEST_DAYS]
        if test.empty:
            break

        out = pd.DataFrame({"actual": test["target"]}, index=test.index)
        out["Baseline"] = train["target"].mean()   # "always predict the historical average"
        models = make_models(names)
        for name, model in models.items():
            model.fit(train[features], train["target"])
            out[name] = model.predict(test[features])
        if len(models) >= 2:
            out["Ensemble"] = out[list(models)].mean(axis=1)

        if with_range:
            for name, model in make_quantile_models().items():
                model.fit(train[features], train["target"])
                out[name] = model.predict(test[features])
            # Quantile models can occasionally cross; keep low <= high
            out["_low"], out["_high"] = (np.minimum(out["_low"], out["_high"]),
                                         np.maximum(out["_low"], out["_high"]))
        results.append(out)
        start += TEST_DAYS

    return pd.concat(results) if results else pd.DataFrame()


def model_columns(preds: pd.DataFrame) -> list:
    """Prediction columns that are models (not 'actual' or the hidden _low/_high range)."""
    return [c for c in preds.columns if c != "actual" and not c.startswith("_")]


def range_coverage(preds: pd.DataFrame) -> float:
    """How often the actual return landed inside the predicted 80% range."""
    if "_low" not in preds:
        return np.nan
    inside = (preds["actual"] >= preds["_low"]) & (preds["actual"] <= preds["_high"])
    return inside.mean()


def score(preds: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample scores for each model."""
    actual = preds["actual"]
    rows = {}
    for name in model_columns(preds):
        p = preds[name]
        long_only = actual[p > 0]
        rows[name] = {
            "MAE": (p - actual).abs().mean(),
            "Hit rate": (np.sign(p) == np.sign(actual)).mean(),
            "IC": p.corr(actual, method="spearman") if p.nunique() > 1 else np.nan,
            "% days long": (p > 0).mean(),
            "Avg ret when long": long_only.mean() if len(long_only) else np.nan,
        }
    return pd.DataFrame(rows).T


def latest_forecast(data: pd.DataFrame, features: list, names=None) -> dict:
    """Train each model on all rows with a known target, then predict from today's features."""
    known = data.dropna(subset=["target"])
    today = data.iloc[[-1]]
    return {name: float(m.fit(known[features], known["target"]).predict(today[features])[0])
            for name, m in make_models(names).items()}


def forecast_range(data: pd.DataFrame, features: list) -> tuple:
    """Today's 80% range for the 20-day return: (low, high)."""
    known = data.dropna(subset=["target"])
    today = data.iloc[[-1]]
    lo, hi = (float(m.fit(known[features], known["target"]).predict(today[features])[0])
              for m in make_quantile_models().values())
    return min(lo, hi), max(lo, hi)


# ---------- Run directly ----------

def _pct(x):
    return "n/a" if pd.isna(x) else f"{x:.1%}"


def _num(x, digits=1):
    return "n/a" if pd.isna(x) else f"{x:.{digits}f}"


def _print_scores(title: str, preds: pd.DataFrame):
    if preds.empty:
        print(f"\n{title}: not enough data for walk-forward testing")
        return
    folds = (len(preds) + TEST_DAYS - 1) // TEST_DAYS
    print(f"\n{title}")
    print(f"  Tested {preds.index[0].date()} to {preds.index[-1].date()} "
          f"({len(preds)} days, {folds} folds). Buy-and-hold avg 20d return: "
          f"{_pct(preds['actual'].mean())}")
    s = score(preds)
    table = pd.DataFrame({
        "MAE": s["MAE"].map(_pct),
        "Hit rate": s["Hit rate"].map(_pct),
        "IC": s["IC"].map(lambda x: _num(x, 3)),
        "% days long": s["% days long"].map(_pct),
        "Avg ret when long": s["Avg ret when long"].map(_pct),
    })
    print(table.to_string())
    cov = range_coverage(preds)
    if pd.notna(cov):
        print(f"  80% range: actual return landed inside it {cov:.0%} of the time "
              f"(well calibrated = close to 80%)")


if __name__ == "__main__":
    print("Models:", ", ".join(available_models()))
    if not HAS_LGBM:
        print("Note: lightgbm not available, running without it (pip install lightgbm)")

    for ticker in ["NVDA", "GOOG"]:
        data = build_dataset(ticker)
        print(f"\n===================== {ticker} =====================")
        print(f"Dataset: {len(data)} trading days, "
              f"{data.index[0].date()} to {data.index[-1].date()}")

        # ---- Latest-ratios block: sanity-check these against a site like Yahoo Finance ----
        last = data.iloc[-1]
        print(f"\nLatest values ({data.index[-1].date()}, close {last['Close']:.2f}):")
        print(f"  Market cap:        ${last['market_cap'] / 1e9:,.0f} bn")
        print(f"  P/E (TTM):         {_num(last['pe_ttm'])}")
        print(f"  P/S (TTM):         {_num(last['ps_ttm'])}")
        print(f"  P/B:               {_num(last['pb'])}")
        print(f"  Earnings yield:    {_pct(last['earnings_yield'])}")
        print(f"  FCF yield:         {_pct(last['fcf_yield'])}")
        print(f"  Net margin:        {_pct(last['net_margin'])}")
        print(f"  ROE:               {_pct(last['roe'])}")
        print(f"  Revenue growth:    {_pct(last['revenue_growth_yoy'])}")
        print(f"  20d return:        {_pct(last['ret_20d'])}")
        print(f"  Volatility (20d):  {_pct(last['vol_20d'])} annualised")
        print(f"  RSI(14):           {_num(last['rsi_14'])}")
        print(f"  vs 200-day avg:    {_pct(last['dist_sma200'])}")

        # ---- A) Technical features only: uses the full 10-year history ----
        tech_data = data.dropna(subset=TECH_FEATURES)
        preds_tech = walk_forward(tech_data, TECH_FEATURES)
        _print_scores("A) Technical only", preds_tech)

        # ---- B) Technical + fundamental: only where fundamentals exist ----
        all_features = TECH_FEATURES + FUND_FEATURES
        full_data = data.dropna(subset=all_features)
        if len(full_data):
            print(f"\n  (Fundamentals cover {full_data.index[0].date()} onwards, "
                  f"{len(full_data)} days)")
        preds_full = walk_forward(full_data, all_features)
        _print_scores("B) Technical + fundamental", preds_full)

        # Same test days for a fair comparison of A vs B
        if not preds_full.empty:
            common = preds_tech.index.intersection(preds_full.index)
            if len(common):
                _print_scores("A) Technical only, same test days as B",
                              preds_tech.loc[common])

        # ---- Forecast for the next 20 trading days ----
        use_full = len(full_data) and not full_data.index[-1] < data.index[-1] - pd.Timedelta(days=7)
        feats = all_features if use_full else TECH_FEATURES
        src = full_data if use_full else tech_data
        print(f"\nForecast, next {HORIZON} trading days "
              f"({'technical + fundamental' if use_full else 'technical only'}):")
        fc = latest_forecast(src, feats)
        for name, value in fc.items():
            print(f"  {name:<10} {value:+.1%}")
        print(f"  {'Ensemble':<10} {np.mean(list(fc.values())):+.1%}")
        lo, hi = forecast_range(src, feats)
        print(f"  80% range  {lo:+.1%} to {hi:+.1%}")

        # Save out-of-sample predictions for the dashboard (Step 5)
        out = preds_full if not preds_full.empty else preds_tech
        out.to_csv(DATA_DIR / f"{ticker}_walkforward.csv")
