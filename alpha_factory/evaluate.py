"""
Alpha evaluation pipeline.

Matches AlphaGen/AlphaForge methodology:
  - 20-day forward returns as target
  - Per-day cross-sectional IC (Pearson and Spearman)
  - Per-day normalization of alpha output (zero mean, unit L2 norm)
  - TopkDropout portfolio backtest for Sharpe
"""

import numpy as np
from scipy.stats import spearmanr, rankdata
from dataclasses import dataclass
from typing import Any


@dataclass
class AlphaMetrics:
    ic: float = 0.0
    rank_ic: float = 0.0
    icir: float = 0.0       # IC / std(IC)
    rank_icir: float = 0.0
    sharpe: float = 0.0     # portfolio Sharpe (annualized)
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    market_corr: float = 0.0
    valid: bool = False
    expression: str = ""
    n_days: int = 0


def normalize_alpha(signals: np.ndarray) -> np.ndarray:
    """
    Per-day normalization: zero mean, unit L2 norm.
    Matches AlphaGen's normalize_by_day().
    Input: (n_stocks, n_days)
    """
    out = signals.copy()
    for t in range(out.shape[1]):
        col = out[:, t]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            out[:, t] = 0.0
            continue
        m = np.nanmean(col)
        col_centered = col - m
        norm = np.sqrt(np.nansum(col_centered[valid] ** 2))
        if norm < 1e-10:
            out[:, t] = 0.0
        else:
            out[:, t] = col_centered / norm
            out[~valid, t] = 0.0
    return out


def compute_forward_returns(close: np.ndarray, horizon: int = 20) -> np.ndarray:
    """
    Compute forward returns: close[t+horizon] / close[t] - 1.
    Input: (n_stocks, n_days). Output: same shape with NaN at end.
    """
    fwd = np.full_like(close, np.nan)
    if horizon < close.shape[1]:
        fwd[:, :-horizon] = close[:, horizon:] / np.maximum(close[:, :-horizon], 1e-10) - 1
    return fwd


def compute_ic_series(
    signals: np.ndarray,
    forward_returns: np.ndarray,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute daily cross-sectional IC and Rank IC.

    Args:
        signals: (n_stocks, n_days) alpha values
        forward_returns: (n_stocks, n_days)
        normalize: whether to normalize signals per day

    Returns:
        (ic_series, rank_ic_series) each (n_days,) with NaN where insufficient data
    """
    if normalize:
        signals = normalize_alpha(signals)

    n_stocks, n_days = signals.shape
    ics = np.full(n_days, np.nan)
    rank_ics = np.full(n_days, np.nan)

    for t in range(n_days):
        sig = signals[:, t]
        ret = forward_returns[:, t]
        valid = ~(np.isnan(sig) | np.isnan(ret))
        if valid.sum() < 10:
            continue

        s, r = sig[valid], ret[valid]
        ic = np.corrcoef(s, r)[0, 1]
        if not np.isnan(ic):
            ics[t] = ic

        rho, _ = spearmanr(s, r)
        if not np.isnan(rho):
            rank_ics[t] = rho

    return ics, rank_ics


def long_short_backtest(
    signals: np.ndarray,
    forward_returns_1d: np.ndarray,
    quantile: float = 0.2,
) -> dict[str, Any]:
    """
    Long-short portfolio backtest.

    Long top quantile, short bottom quantile, equal-weight.
    Returns are market-neutral (long - short), isolating alpha signal
    from market beta.

    Args:
        signals: (n_stocks, n_days) normalized alpha values
        forward_returns_1d: (n_stocks, n_days) next-day returns
        quantile: fraction of stocks in each leg (0.2 = top/bottom 20%)

    Returns:
        dict with daily_returns, sharpe, annual_return, max_drawdown
    """
    n_stocks, n_days = signals.shape
    k = max(1, int(n_stocks * quantile))
    daily_returns = []

    for t in range(n_days - 1):
        sig = signals[:, t]
        ret = forward_returns_1d[:, t]

        valid = ~(np.isnan(sig) | np.isnan(ret))
        if valid.sum() < k * 2:
            daily_returns.append(0.0)
            continue

        # Mask invalid
        sig_v = np.where(valid, sig, -np.inf)
        ret_v = np.where(valid, ret, 0.0)

        ranked = np.argsort(sig_v)
        long_idx = ranked[-k:]    # top k
        short_idx = ranked[:k]    # bottom k

        long_ret = np.mean(ret_v[long_idx])
        short_ret = np.mean(ret_v[short_idx])
        daily_returns.append(long_ret - short_ret)

    daily_returns = np.array(daily_returns)

    if len(daily_returns) == 0 or np.std(daily_returns) < 1e-10:
        return {"daily_returns": daily_returns, "sharpe": 0.0,
                "annual_return": 0.0, "max_drawdown": 0.0, "n_days": 0}

    cumulative = np.cumprod(1 + daily_returns)
    annual_return = float((cumulative[-1] ** (252 / max(len(daily_returns), 1))) - 1)
    sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
    drawdown = cumulative / np.maximum.accumulate(cumulative) - 1
    max_dd = float(np.min(drawdown))

    return {
        "daily_returns": daily_returns,
        "sharpe": sharpe,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "n_days": len(daily_returns),
    }


def evaluate_signals(
    signals: np.ndarray,
    close_prices: np.ndarray,
    forward_returns_1d: np.ndarray,
    forward_returns_20d: np.ndarray,
    expression: str = "",
    min_active_frac: float = 0.5,
) -> AlphaMetrics:
    """
    Full evaluation of an alpha signal array.

    Args:
        signals: (n_stocks, n_days) raw alpha values
        close_prices: (n_stocks, n_days)
        forward_returns_1d: (n_stocks, n_days) 1-day forward returns
        forward_returns_20d: (n_stocks, n_days) 20-day forward returns
        expression: string representation of the alpha
        min_active_frac: reject signals where fewer than this fraction
            of days have meaningful cross-sectional variance

    Returns:
        AlphaMetrics with all standard metrics
    """
    # Reject degenerate signals: if most days have near-zero CS variance,
    # the backtest picks stocks by index order (meaningless).
    cs_std = np.nanstd(signals, axis=0)
    active_days = np.sum(cs_std > 1e-10) / max(signals.shape[1], 1)
    if active_days < min_active_frac:
        return AlphaMetrics(expression=expression)

    # IC against 20-day forward returns (standard)
    ics, rank_ics = compute_ic_series(signals, forward_returns_20d, normalize=True)

    valid_ics = ics[~np.isnan(ics)]
    valid_rank_ics = rank_ics[~np.isnan(rank_ics)]

    if len(valid_ics) < 20:
        return AlphaMetrics(expression=expression)

    mean_ic = float(np.mean(valid_ics))
    mean_rank_ic = float(np.mean(valid_rank_ics))
    icir = float(np.mean(valid_ics) / (np.std(valid_ics) + 1e-10))
    rank_icir = float(np.mean(valid_rank_ics) / (np.std(valid_rank_ics) + 1e-10))

    # Long-short portfolio backtest using 1-day returns
    norm_signals = normalize_alpha(signals)
    bt = long_short_backtest(norm_signals, forward_returns_1d)

    # Turnover from normalized signals
    daily_positions = []
    for t in range(signals.shape[1]):
        sig_t = norm_signals[:, t]
        ranked = rankdata(sig_t, nan_policy="omit") / max(np.sum(~np.isnan(sig_t)), 1)
        daily_positions.append(ranked)
    pos = np.array(daily_positions)
    turnover = float(np.nanmean(np.abs(np.diff(pos, axis=0)))) if len(daily_positions) > 1 else 0.0

    # Market correlation
    market_ret = np.nanmean(forward_returns_20d, axis=0)
    v = ~(np.isnan(valid_ics) | np.isnan(market_ret[:len(valid_ics)]))
    mkt_corr = 0.0
    if v.sum() > 10:
        mkt_corr = float(abs(np.corrcoef(valid_ics[v], market_ret[:len(valid_ics)][v])[0, 1]))

    return AlphaMetrics(
        ic=mean_ic,
        rank_ic=mean_rank_ic,
        icir=icir,
        rank_icir=rank_icir,
        sharpe=bt["sharpe"],
        annual_return=bt["annual_return"],
        max_drawdown=bt["max_drawdown"],
        turnover=turnover,
        market_corr=mkt_corr,
        valid=True,
        expression=expression,
        n_days=bt["n_days"],
    )
