"""NVIDIA historical stock exploration and simple price forecasting.

Refactored from a University of Washington summer math modeling course project.
The forecasting approach is a fitted one-step linear relationship, recursively
applied over future trading days. It is an educational baseline, not a trading tool.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def load_stock_data(path: Path) -> pd.DataFrame:
    """Read a historical price CSV with Date, Open, High, Low, Close/Last and Volume."""
    df = pd.read_csv(path)
    required = {'Date', 'Open', 'High', 'Low', 'Volume'}
    missing = required - set(df.columns)
    if missing or not ({'Close/Last', 'Close'} & set(df.columns)):
        raise ValueError(f'Missing required columns: {sorted(missing)} or Close/Last (or Close)')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    numeric_cols = ['Open', 'High', 'Low', 'Volume']
    close_col = 'Close/Last' if 'Close/Last' in df else 'Close'
    for col in numeric_cols + [close_col]:
        cleaned = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
        df[col] = pd.to_numeric(cleaned, errors='coerce')
    df = df.rename(columns={close_col: 'Close'})
    df = df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df = df.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
    if len(df) < 3:
        raise ValueError('At least 3 valid trading days are required.')
    df['AvgPrice'] = (df['High'] + df['Low']) / 2
    df['Range'] = df['High'] - df['Low']
    df['Intraday_return'] = (df['Close'] - df['Open']) / df['Open'].replace(0, np.nan)
    df['Daily_return'] = df['Close'].pct_change()
    return df


def linear_model(x, slope, intercept):
    return slope * x + intercept


def forecast_prices(prices: np.ndarray, days: int = 10):
    """Fit P(t) = m P(t-1) + c and recursively forecast `days` steps."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 3 or not np.isfinite(prices).all():
        raise ValueError('Provide at least 3 finite historical prices.')
    if days < 1:
        raise ValueError('days must be positive')
    yesterday, today = prices[:-1], prices[1:]
    # A degenerate constant-price series needs a constant baseline instead.
    if np.allclose(yesterday, yesterday[0]):
        slope, intercept = 0.0, float(np.mean(today))
    else:
        (slope, intercept), _ = curve_fit(linear_model, yesterday, today)
    fitted = linear_model(yesterday, slope, intercept)
    future = []
    current = float(prices[-1])
    for _ in range(days):
        current = float(linear_model(current, slope, intercept))
        future.append(current)
    return (float(slope), float(intercept)), fitted, np.asarray(future)


def save_figures(df, fitted, future, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    plt.plot(df['Date'], df['Low'], label='Low', linewidth=.8)
    plt.plot(df['Date'], df['High'], label='High', linewidth=.8)
    plt.plot(df['Date'], df['AvgPrice'], label='High/Low average', linewidth=1)
    plt.title('Historical stock prices')
    plt.xlabel('Date'); plt.ylabel('Price (USD)'); plt.legend(); plt.tight_layout()
    plt.savefig(output / 'historical_prices.png', dpi=160); plt.close()

    returns = df['Daily_return'].dropna() * 100
    plt.figure(figsize=(9, 5))
    plt.hist(returns, bins=min(60, max(5, len(returns) // 3)), density=True, alpha=.75, label='Daily returns')
    if len(returns) > 2 and returns.std() > 0:
        xs = np.linspace(returns.min(), returns.max(), 300)
        mean, std = returns.mean(), returns.std()
        density = np.exp(-.5 * ((xs - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        plt.plot(xs, density, label='Matching normal curve')
    plt.title('Distribution of daily closing returns')
    plt.xlabel('Daily return (%)'); plt.ylabel('Density'); plt.legend(); plt.tight_layout()
    plt.savefig(output / 'daily_returns.png', dpi=160); plt.close()

    plt.figure(figsize=(11, 5))
    zoom = min(60, len(fitted))
    x_hist = np.arange(1, len(df))
    x_future = np.arange(len(df), len(df) + len(future))
    plt.plot(x_hist[-zoom:], df['AvgPrice'].to_numpy()[1:][-zoom:], label='Observed average price', linewidth=2)
    plt.plot(x_hist[-zoom:], fitted[-zoom:], '--', label='In-sample fitted price', linewidth=1.5)
    plt.plot(np.r_[len(df)-1, x_future], np.r_[df['AvgPrice'].iloc[-1], future],
             label=f'{len(future)}-trading-day recursive forecast', linewidth=2)
    plt.title('Linear-model stock price forecast')
    plt.xlabel('Trading-day index'); plt.ylabel('Price (USD)'); plt.grid(alpha=.2)
    plt.legend(); plt.tight_layout()
    plt.savefig(output / 'forecast.png', dpi=160); plt.close()


def main():
    parser = argparse.ArgumentParser(description='Historical stock analysis and simple linear forecast')
    parser.add_argument('--data', type=Path, required=True, help='Path to historical price CSV')
    parser.add_argument('--days', type=int, default=10, help='Future trading days (default: 10)')
    parser.add_argument('--output', type=Path, default=Path('outputs'), help='Output folder')
    args = parser.parse_args()
    data = load_stock_data(args.data)
    (slope, intercept), fitted, future = forecast_prices(data['AvgPrice'].to_numpy(), args.days)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'trading_day_ahead': np.arange(1, args.days + 1),
                  'predicted_avg_price': future}).to_csv(args.output / 'forecast.csv', index=False)
    save_figures(data, fitted, future, args.output)
    daily = data['Daily_return'].dropna() * 100
    print(f'Trading days loaded: {len(data)}')
    print(f'Historical date range: {data.Date.min().date()} to {data.Date.max().date()}')
    print(f'Linear model: next_avg_price = {slope:.6f} * previous_avg_price + {intercept:.6f}')
    print(f'Mean daily close return: {daily.mean():.3f}%')
    print(f'Standard deviation of daily close returns: {daily.std():.3f}%')
    print(f'Saved plots and {args.days}-day forecast to: {args.output.resolve()}')


if __name__ == '__main__':
    main()
