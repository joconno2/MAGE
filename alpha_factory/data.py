"""
Data loading for alpha evaluation.

Downloads OHLCV from Yahoo Finance, prepares aligned matrices for
cross-sectional alpha evaluation matching AlphaGen methodology.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

CACHE_DIR = Path.home() / "research" / "alpha-factory" / "data"

# S&P 100 proxy
SP100_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JPM",
    "JNJ", "V", "UNH", "PG", "HD", "MA", "DIS", "PYPL", "BAC", "INTC",
    "VZ", "CMCSA", "ADBE", "NFLX", "KO", "PEP", "T", "MRK", "ABT",
    "CRM", "CSCO", "XOM", "PFE", "NKE", "WMT", "TMO", "AVGO", "ACN",
    "COST", "LLY", "MCD", "DHR", "TXN", "NEE", "MDT", "LIN", "HON",
    "UNP", "AMGN", "BMY", "PM", "RTX", "LOW", "QCOM", "ORCL", "IBM",
    "CVX", "C", "GS", "CAT", "BLK", "ISRG", "GE", "MMM", "AXP",
    "BA", "SBUX", "GILD", "DE", "SYK", "MDLZ", "PLD", "ADP", "TGT",
    "BKNG", "LRCX", "MO", "CI", "ZTS", "CB", "SO", "DUK", "BDX",
    "CL", "CME", "USB", "TFC", "ICE", "APD", "ECL", "FIS", "NSC",
    "SHW", "ITW", "PNC", "AON", "WM", "EMR", "EW", "ATVI", "HUM", "F",
]


def download_ohlcv(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str = "2023-12-31",
    cache_name: str = "sp100",
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV. Caches to parquet."""
    cache_path = CACHE_DIR / f"{cache_name}_{start}_{end}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        result = {}
        for ticker in tickers:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    sub = df.xs(ticker, level=1, axis=1)
                else:
                    sub = df
                sub.columns = [c.lower() for c in sub.columns]
                sub = sub.dropna(how="all")
                if len(sub) > 100:
                    result[ticker] = sub
            except Exception:
                pass
        if result:
            return result

    import yfinance as yf
    print(f"Downloading {len(tickers)} tickers from Yahoo Finance...")
    data = yf.download(tickers, start=start, end=end, group_by="column", auto_adjust=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(cache_path)

    result = {}
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                sub = data.xs(ticker, level=1, axis=1)
            else:
                sub = data
            sub.columns = [c.lower() for c in sub.columns]
            sub = sub[["open", "high", "low", "close", "volume"]].dropna(how="all")
            if len(sub) > 100:
                result[ticker] = sub
        except Exception:
            pass
    print(f"Loaded {len(result)} tickers")
    return result


def prepare_eval_data(
    raw_data: dict[str, pd.DataFrame],
    train_start: str = "2010-01-01",
    train_end: str = "2019-12-31",
    val_start: str = "2020-01-01",
    val_end: str = "2020-12-31",
    test_start: str = "2021-01-01",
    test_end: str = "2022-12-31",
    return_horizon: int = 20,
) -> dict:
    """
    Prepare aligned numpy matrices for evaluation.

    Returns dict with keys "train", "val", "test", each containing:
        stock_data: {ticker: {feature: np.ndarray}}  per-stock for GP tree eval
        close_prices: (n_stocks, n_days)
        fwd_returns_1d: (n_stocks, n_days)  1-day forward returns
        fwd_returns_20d: (n_stocks, n_days) 20-day forward returns
        tickers: list[str]
        n_stocks: int
        n_days: int
    """
    splits = {
        "train": (train_start, train_end),
        "val": (val_start, val_end),
        "test": (test_start, test_end),
    }

    result = {}
    for split_name, (start, end) in splits.items():
        # Find common dates
        common_dates = None
        for ticker, df in raw_data.items():
            mask = (df.index >= start) & (df.index <= end)
            sub = df[mask].dropna(how="all")
            if len(sub) < 50:
                continue
            if common_dates is None:
                common_dates = sub.index
            else:
                common_dates = common_dates.intersection(sub.index)

        if common_dates is None or len(common_dates) < 60:
            continue

        # Build aligned arrays
        tickers = []
        stock_data = {}
        close_list = []

        for ticker, df in raw_data.items():
            sub = df.reindex(common_dates).ffill().bfill()
            if sub.isnull().any(axis=None):
                continue
            sub.columns = [c.lower() for c in sub.columns]

            tickers.append(ticker)
            stock_data[ticker] = {
                "open": sub["open"].values.astype(np.float64),
                "high": sub["high"].values.astype(np.float64),
                "low": sub["low"].values.astype(np.float64),
                "close": sub["close"].values.astype(np.float64),
                "volume": sub["volume"].values.astype(np.float64),
            }
            close_list.append(sub["close"].values.astype(np.float64))

        if len(tickers) < 20:
            continue

        close_prices = np.array(close_list)  # (n_stocks, n_days)
        n_stocks, n_days = close_prices.shape

        # Forward returns
        fwd_1d = np.full_like(close_prices, np.nan)
        fwd_1d[:, :-1] = close_prices[:, 1:] / np.maximum(close_prices[:, :-1], 1e-10) - 1

        fwd_20d = np.full_like(close_prices, np.nan)
        if return_horizon < n_days:
            fwd_20d[:, :-return_horizon] = (
                close_prices[:, return_horizon:] /
                np.maximum(close_prices[:, :-return_horizon], 1e-10) - 1
            )

        result[split_name] = {
            "stock_data": stock_data,
            "close_prices": close_prices,
            "fwd_returns_1d": fwd_1d,
            "fwd_returns_20d": fwd_20d,
            "tickers": tickers,
            "n_stocks": n_stocks,
            "n_days": n_days,
            "dates": common_dates,
        }

    return result
