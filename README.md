# 📈 Market Forecaster

An interactive dashboard that forecasts the **20-day return** of shares, currency pairs,
cryptocurrencies, US Treasuries and precious metals using machine learning on technical
and fundamental data.

**Live app:** _add your Streamlit link here_

> Learning and portfolio project. Not investment advice.

---

## Features

- **Five markets** – shares, forex, crypto, Treasuries (yields and bond ETFs), gold & metals
- **350+ built-in assets**, searchable by name or ticker, plus manual entry for any Yahoo Finance symbol
- **Trading-style dashboard** – candlestick chart, moving averages, volume, RSI, signal panel
- **Fundamentals for shares** – market cap, P/E, P/S, P/B, margins, ROE, revenue growth, FCF yield
- **Three ML models + Ensemble** – Ridge, Gradient Boosting and LightGBM; each is explained in the app
- **Forecast range** – an 80% range drawn as a cone on the chart, with a check of how often it was right
- **Honest model testing** – walk-forward validation, out-of-sample hit rate, IC and a simple strategy back-test

## How it works

| Step | File | What it does |
|---|---|---|
| 1 | `src/fetch.py` | Downloads and caches 10 years of daily prices and quarterly statements (Yahoo Finance) |
| 2 | `src/model.py` | Builds 12 technical features (momentum, volatility, moving averages, RSI, volume, drawdown) |
| 3 | `src/fundamentals.py` | Builds trailing-twelve-month ratios and joins them to prices **with reporting lags**, so the model never sees numbers before they were public |
| 4 | `src/model.py` | Target = 20-day forward return. Three models (Ridge regression, scikit-learn gradient boosting and LightGBM) plus their **Ensemble** are tested with **expanding-window walk-forward validation** and a 20-day gap between training and test data. Quantile models give an **80% forecast range**, and the app checks how often that range was right |
| 5 | `src/app.py` | Streamlit dashboard; `src/catalog.py` holds the built-in asset lists |

### Avoiding look-ahead bias
- Quarterly reports count as public 45 days after the period ends, annual reports after 60 days.
- Share counts are adjusted for stock splits to match split-adjusted prices.
- Every prediction shown in the performance view comes from a model trained only on earlier data.

## Run it on your own computer

```bash
git clone https://github.com/<your-username>/market-forecaster.git
cd market-forecaster
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run src/app.py
```

On Windows you can also run `create_shortcut.ps1` once to get a desktop icon.

## Tech stack

Python · pandas · NumPy · scikit-learn · LightGBM · Plotly · Streamlit · yfinance

## Limitations

- Stock returns are noisy; a hit rate a few points above 50% is already meaningful, and past results don't guarantee future ones.
- Yahoo Finance provides only ~4–5 years of financial statements, so fundamental features cover a shorter period than prices.
- No trading costs, taxes or slippage in the back-test.

## Author

Milan Vidyapathige · M.A. International Management and Information Systems · B.Sc. Finance
