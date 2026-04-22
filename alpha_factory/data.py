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

            o = sub["open"].values.astype(np.float64)
            h = sub["high"].values.astype(np.float64)
            l = sub["low"].values.astype(np.float64)
            c = sub["close"].values.astype(np.float64)
            v = sub["volume"].values.astype(np.float64)

            # Derived features
            ret = np.full_like(c, np.nan)
            ret[1:] = (c[1:] - c[:-1]) / np.maximum(np.abs(c[:-1]), 1e-10)

            log_ret = np.full_like(c, np.nan)
            log_ret[1:] = np.log(np.maximum(c[1:], 1e-10) / np.maximum(c[:-1], 1e-10))

            dollar_vol = c * v

            # adv20: 20-day average volume
            adv20 = np.full_like(v, np.nan)
            if len(v) >= 20:
                cs = np.cumsum(v)
                adv20[19:] = (cs[19:] - np.concatenate([[0], cs[:-20]])) / 20
            turnover_ratio = np.where(adv20 > 1e-10, v / adv20, 0.0)

            intraday_range = (h - l) / np.maximum(np.abs(c), 1e-10)

            gap = np.full_like(c, np.nan)
            gap[1:] = o[1:] / np.maximum(np.abs(c[:-1]), 1e-10) - 1

            upper_shadow = (h - np.maximum(o, c)) / np.maximum(np.abs(c), 1e-10)
            lower_shadow = (np.minimum(o, c) - l) / np.maximum(np.abs(c), 1e-10)
            body = (c - o) / np.maximum(np.abs(c), 1e-10)

            stock_data[ticker] = {
                "open": o, "high": h, "low": l, "close": c, "volume": v,
                "returns": ret,
                "log_return": log_ret,
                "dollar_volume": dollar_vol,
                "turnover_ratio": turnover_ratio,
                "intraday_range": intraday_range,
                "gap": gap,
                "upper_shadow": upper_shadow,
                "lower_shadow": lower_shadow,
                "body": body,
            }
            close_list.append(c)

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
