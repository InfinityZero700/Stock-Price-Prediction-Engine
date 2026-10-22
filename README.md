# NVIDIA Stock Price Modeling & Forecasting

**Python · NumPy · pandas · SciPy · Matplotlib**

An educational stock-market analysis and mathematical modeling project developed during a **University of Washington summer math modeling course**. The project explores NVIDIA historical prices and trading returns, then fits a simple linear model to estimate the next **10 trading days** of average prices.

> **Modeling note:** This is an exploratory mathematical baseline, **not** a validated trading strategy, investment recommendation, or evidence of reliable future price prediction.

## What it does

- Loads and cleans historical trading data from a CSV.
- Calculates high/low average price, intraday range, intraday return, and daily closing return.
- Plots historical price movements and the distribution of daily returns.
- Fits a one-step linear relationship between consecutive high/low average prices using SciPy's `curve_fit`.
- Recursively applies that relationship to produce a configurable multi-day forecast.
- Exports a forecast CSV and three PNG charts.

## Prediction method

For each trading day, define `P[t] = (High[t] + Low[t]) / 2`. Fit:

```text
P[t] ≈ m * P[t-1] + c
```

The fitted slope `m` and intercept `c` are then applied recursively from the most recent observed average price to estimate subsequent trading days. This preserves the central forecasting idea in my original course script. Because a recursive forecast compounds estimation error and this fit is based on historical data, it should **not** be interpreted as a reliable market forecast.

**Important limitation:** The historical fitted line is **in-sample**. The project does not currently use a separate test period, report out-of-sample accuracy, or incorporate news, company fundamentals, or trading fees. The original exploration also experimented with sine-curve fits to volume; this standalone version focuses on the complete, runnable price model and return visualizations.

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Or for macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Get a historical stock CSV with these columns:

```text
Date,Close/Last,Volume,Open,High,Low
```

`Close` can be used instead of `Close/Last`. Dollar signs and commas are supported in numeric fields. Save your downloaded file as `data/nvidia.csv` (the data itself is not bundled here).

Run:

```bash
python stock_predictor.py --data data/nvidia.csv --days 10
```

Generated outputs in `outputs/`:

```text
historical_prices.png
daily_returns.png
forecast.png
forecast.csv
```

Change the forecast horizon using `--days` or the output directory using `--output`.

## Project structure

```text
nvidia-stock-modeling/
├── stock_predictor.py       # Data preparation, analysis, charts, linear forecast
├── requirements.txt         # Python dependencies
├── README.md                # Project overview and run instructions
├── .gitignore               # Ignore local environments, downloaded data, outputs
├── data/                    # Place your historical CSV here
├── outputs/                 # Generated figures and forecast CSV
└── tests/
    └── test_stock_predictor.py
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Background

I built the original modeling work in a University of Washington summer course, using Python to investigate historical NVIDIA trading patterns, visualize returns, and experiment with curve fitting and short-range forecasts. This repository is a cleaned, standalone version of that educational work with portable file paths and a command-line interface.

## Pipeline & Verification

The engine runs a multi-stage quantitative pipeline from raw CSV ingestion through to recursive extrapolation.

### Pipeline Architecture

[Raw Market CSV]
       │
       ▼
1. Data Sanitization & Normalization
   • Strips currency symbols & handles missing fields
   • Resolves column aliases (Close vs. Close/Last)
       │
       ▼
2. Metric Computation & Statistical Profiling
   • Calculates midpoint price: P[t] = (High[t] + Low[t]) / 2
   • Generates intraday ranges & daily return distributions
       │
       ▼
3. Non-Linear Parameter Optimization
   • Fits P[t] ≈ m * P[t-1] + c via SciPy curve_fit
       │
       ▼
4. Recursive Trajectory Forecasting
   • Extrapolates n-day horizon from terminal state P[0]
   • Logs cumulative recursive estimation drift
       │
       ▼
5. Output Serialization
   • Exports forecast tabular CSV & 3 diagnostic plots to outputs/
   
## Testing

1. Run unit test suite
python -m unittest discover -s tests -v

2. Execute full modeling pipeline (10-day forecast horizon)
python stock_predictor.py --data data/nvidia.csv --days 10

## Author & Contact

**Akshat Milan**
- **GitHub:** [@InfinityZero700](https://github.com/InfinityZero700)
- **Focus:** Applied Mathematics & Computer Science | Quantitative Analytics & ML
- **Repository:** [Stock-Price-Prediction-Engine](https://github.com/InfinityZero700/Stock-Price-Prediction-Engine)
  
## Disclaimer

For education and exploration only. Stock prices are unpredictable, and historical relationships may not hold in the future.
