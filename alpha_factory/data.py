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

# S&P 500 constituents (as of early 2026)
SP500_TICKERS = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APP",
    "APTV", "ARE", "ARES", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP",
    "AZO", "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF-B", "BG",
    "BIIB", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO",
    "BSX", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CASY", "CAT", "CB",
    "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CEG", "CF", "CFG", "CHD",
    "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG",
    "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR",
    "COST", "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO",
    "CSGP", "CSX", "CTAS", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "D",
    "DAL", "DD", "DE", "DECK", "DELL", "DG", "DGX", "DHI",
    "DHR", "DIS", "DLR", "DLTR", "DOV", "DOW", "DPZ", "DRI", "DTE",
    "DUK", "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX",
    "EIX", "EL", "ELV", "EME", "EMR", "EOG", "EPAM", "EQIX", "EQR", "EQT",
    "ERIE", "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD",
    "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV",
    "FICO", "FIS", "FISV", "FITB", "FOX", "FOXA", "FRT", "FSLR", "FTNT",
    "FTV", "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL",
    "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW",
    "HAL", "HAS", "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HON",
    "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM",
    "IBM", "ICE", "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP",
    "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL",
    "JCI", "JKHY", "JNJ", "JPM", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR",
    "KLAC", "KMB", "KMI", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH",
    "LHX", "LIN", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU",
    "LUV", "LVS", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP",
    "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MKC", "MLM", "MMM",
    "MNST", "MO", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MS", "MSCI",
    "MSFT", "MSI", "MTB", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM",
    "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE",
    "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON",
    "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PAYX", "PCAR", "PCG", "PEG", "PEP",
    "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM",
    "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA",
    "PSX", "PTC", "PWR", "PYPL", "QCOM", "RCL", "REG", "REGN", "RF",
    "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY",
    "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA",
    "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX",
    "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG",
    "TDY", "TECH", "TEL", "TER", "TFC", "TGT", "TJX", "TMO", "TMUS",
    "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT",
    "TTD", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA",
    "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VLTO", "VMC",
    "VRSK", "VRSN", "VRT", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT",
    "WBD", "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB",
    "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM",
    "ZBH", "ZBRA", "ZTS",
]

# Keep old name for backward compat
SP100_TICKERS = SP500_TICKERS


def download_ohlcv(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str = "2023-12-31",
    cache_name: str = "sp500",
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
        # Find trading dates from the broadest stock, then keep stocks
        # that cover at least 90% of those dates. This avoids collapsing
        # the date range when recent IPOs have short histories.
        all_dates = set()
        per_ticker_dates = {}
        for ticker, df in raw_data.items():
            mask = (df.index >= start) & (df.index <= end)
            sub = df[mask].dropna(how="all")
            if len(sub) < 50:
                continue
            per_ticker_dates[ticker] = sub.index
            all_dates.update(sub.index)

        if not all_dates:
            continue

        # Use dates covered by at least 80% of stocks
        sorted_dates = sorted(all_dates)
        date_counts = {}
        for dates in per_ticker_dates.values():
            for d in dates:
                date_counts[d] = date_counts.get(d, 0) + 1
        n_tickers_total = len(per_ticker_dates)
        common_dates = pd.DatetimeIndex(sorted(
            d for d, c in date_counts.items() if c >= n_tickers_total * 0.8
        ))

        if len(common_dates) < 60:
            continue

        # Keep stocks with at least 90% coverage of common dates
        min_coverage = int(len(common_dates) * 0.9)

        # Build aligned arrays
        tickers = []
        stock_data = {}
        close_list = []

        for ticker, df in raw_data.items():
            sub = df.reindex(common_dates).ffill().bfill()
            n_valid = sub.notna().all(axis=1).sum()
            if n_valid < min_coverage:
                continue
            # Fill any remaining gaps
            sub = sub.ffill().bfill().fillna(0)
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
