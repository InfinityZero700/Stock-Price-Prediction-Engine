# NVIDIA Stock Price Modeling & Forecasting

**Python · NumPy · pandas · SciPy · Matplotlib**

An educational stock-market analysis and mathematical modeling project developed during a **University of Washington summer math modeling course**. The project explores historical NVIDIA equity prices and trading returns, then fits an autoregressive linear model to estimate the next 10 trading days of mid-market prices.

> **Modeling note:** This is an exploratory mathematical baseline, **not** a validated trading strategy, investment recommendation, or evidence of reliable future price prediction.

---

## What It Does

- Loads and cleans historical trading data from a CSV.
- Calculates high/low average midpoint price, intraday range, intraday return, and daily closing return.
- Plots historical price movements and empirical probability distributions of daily returns.
- Fits a one-step linear relationship between consecutive high/low average prices using SciPy's `curve_fit`.
- Recursively applies that relationship to generate a configurable multi-day forecast horizon.
- Exports a structured forecast CSV and three diagnostic visualization charts.

---

## Prediction Methodology

For each trading day, define the representative midpoint price $P_t$:

$$P_t = \frac{\text{High}_t + \text{Low}_t}{2}$$

We fit the first-order transition equation:

$$P_t \approx m \cdot P_{t-1} + c$$

The fitted slope $m$ and intercept $c$ are then applied recursively starting from the most recent observed average price $P_0$ to extrapolate subsequent trading days:

$$\hat{P}_k = m \cdot \hat{P}_{k-1} + c$$

Because a recursive forecast compounds estimation uncertainty over time and the parameters are fitted in-sample, this model serves as an educational mathematical demonstration rather than a reliable market prediction.

**Important limitation:** The fitted relationship is strictly **in-sample**. The baseline pipeline does not currently use an out-of-sample chronological holdout period, nor does it factor in market fundamentals, news sentiment, or execution friction.

---

## Pipeline & Architecture

```text
[Raw Market CSV]
       │
       ▼
1. Data Sanitization & Normalization
   • Strips currency symbols & handles missing values
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
```

---

## Project Structure

```text
Stock-Price-Prediction-Engine/
├── stock_predictor.py       # Data preparation, analysis, charts, linear forecast
├── requirements.txt         # Pinned Python dependencies
├── README.md                # System documentation & verification instructions
├── .gitignore               # Excludes virtual environments, downloaded data, caches
├── data/                    # Ingestion directory for historical CSVs
│   └── nvidia.csv           # Historical equity data (OHLCV)
├── outputs/                 # Generated figures and forecast CSV
└── tests/
    └── test_stock_predictor.py  # Automated unit test suite
```

---

## Run Locally

### 1. Environment Setup

Requires Python 3.10+.

```bash
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your historical equity CSV in `data/nvidia.csv` with standard market columns:

```text
Date,Close/Last,Volume,Open,High,Low
```
*(Handles `Close` vs. `Close/Last`, dollar signs `$`, and comma separators automatically).*

### 3. Execute Pipeline

```bash
python stock_predictor.py --data data/nvidia.csv --days 10
```

Generated outputs in `outputs/`:
- `historical_prices.png`: Temporal price movement across historical trading dates.
- `daily_returns.png`: Empirical probability distribution of daily percentage returns.
- `forecast.png`: Historical series plotted against the recursive projection trajectory.
- `forecast.csv`: Tabular projection values across the configured horizon.

*Optional flags:* Change the forecast horizon using `--days <int>` or the destination directory using `--output <dir>`.

---

## Testing & Verification

Run the automated unit test suite to verify data sanitization, metric computations, and projection consistency:

```bash
python -m unittest discover -s tests -v
```

---

## Background

I built the original modeling framework during a **University of Washington summer mathematical modeling course**, using Python to explore historical equity dynamics, visualize return distributions, and experiment with parameter curve fitting. This repository is a cleaned, standalone version of that educational coursework with portable file paths, comprehensive unit testing, and a command-line interface.

---

## Author & Contact

**Akshat Milan**  
- **GitHub:** [@InfinityZero700](https://github.com/InfinityZero700)  
- **Focus:** Applied Mathematics & Computer Science | Quantitative Analytics & ML  
- **Repository:** [Stock-Price-Prediction_Engine](https://github.com/InfinityZero700/Stock-Price-Prediction_Engine)

---

## Disclaimer

For education and mathematical exploration only. Equity prices are subject to non-stationary market regimes, and historical relationships may not hold in future periods.
