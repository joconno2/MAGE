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
    valid = ~np.isnan(out)
    n_valid = valid.sum(axis=0)

    # Days with fewer than 2 valid stocks get zeroed
    degenerate = n_valid < 2
    out[:, degenerate] = 0.0

    # Center: subtract per-day mean (ignoring NaN)
    means = np.nanmean(out, axis=0, keepdims=True)
    out = out - means

    # L2 norm per day
    sq = np.where(valid, out ** 2, 0.0)
    norms = np.sqrt(sq.sum(axis=0, keepdims=True))
    norms = np.where(norms < 1e-10, 1.0, norms)
    out = out / norms

    # Zero out NaN positions and degenerate days
    out[~valid] = 0.0
    out[:, degenerate] = 0.0
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

    # Mask invalid positions
    valid = ~(np.isnan(signals) | np.isnan(forward_returns))
    n_valid = valid.sum(axis=0)

    # Zero out invalid positions for vectorized correlation
    sig = np.where(valid, signals, 0.0)
    ret = np.where(valid, forward_returns, 0.0)

    # Per-day means (only over valid stocks)
    n_safe = np.maximum(n_valid, 1).astype(np.float64)
    sig_mean = sig.sum(axis=0) / n_safe
    ret_mean = ret.sum(axis=0) / n_safe

    # Center
    sig_c = sig - sig_mean[np.newaxis, :]
    ret_c = ret - ret_mean[np.newaxis, :]
    sig_c[~valid] = 0.0
    ret_c[~valid] = 0.0

    # Pearson IC: cov / (std_sig * std_ret)
    cov = (sig_c * ret_c).sum(axis=0) / n_safe
    sig_var = (sig_c ** 2).sum(axis=0) / n_safe
    ret_var = (ret_c ** 2).sum(axis=0) / n_safe
    denom = np.sqrt(sig_var * ret_var)
    ics = np.where((denom > 1e-10) & (n_valid >= 10), cov / denom, np.nan)

    # Rank IC via argsort-based ranking on full matrix (no per-day loop)
    # Set invalid positions to -inf so they sort to bottom, then mask after
    sig_for_rank = np.where(valid, signals, -np.inf)
    ret_for_rank = np.where(valid, forward_returns, -np.inf)

    # argsort along axis=0 (across stocks for each day)
    sig_order = np.argsort(sig_for_rank, axis=0)
    ret_order = np.argsort(ret_for_rank, axis=0)

    # Assign ranks via fancy indexing
    sig_ranked = np.empty_like(signals, dtype=np.float64)
    ret_ranked = np.empty_like(forward_returns, dtype=np.float64)
    day_idx = np.arange(n_days)[np.newaxis, :]
    row_ranks = np.arange(1, n_stocks + 1, dtype=np.float64)[:, np.newaxis]
    sig_ranked[sig_order, day_idx] = row_ranks
    ret_ranked[ret_order, day_idx] = row_ranks

    # Mask invalid positions
    sig_ranked[~valid] = 0.0
    ret_ranked[~valid] = 0.0

    # Pearson on ranks = Spearman
    sr_mean = sig_ranked.sum(axis=0) / n_safe
    rr_mean = ret_ranked.sum(axis=0) / n_safe
    sr_c = sig_ranked - sr_mean[np.newaxis, :]
    rr_c = ret_ranked - rr_mean[np.newaxis, :]
    sr_c[~valid] = 0.0
    rr_c[~valid] = 0.0
    r_cov = (sr_c * rr_c).sum(axis=0) / n_safe
    r_denom = np.sqrt((sr_c ** 2).sum(axis=0) / n_safe * (rr_c ** 2).sum(axis=0) / n_safe)
    rank_ics = np.where((r_denom > 1e-10) & (n_valid >= 10), r_cov / r_denom, np.nan)

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

    # Work on all days except the last (no forward return)
    sig = signals[:, :-1]          # (n_stocks, n_days-1)
    ret = forward_returns_1d[:, :-1]

    valid = ~(np.isnan(sig) | np.isnan(ret))
    n_valid = valid.sum(axis=0)

    # Mask invalid signals to -inf so they sort to bottom
    sig_v = np.where(valid, sig, -np.inf)
    ret_v = np.where(valid, ret, 0.0)

    # argsort each day (along stocks axis)
    ranked = np.argsort(sig_v, axis=0)  # (n_stocks, n_days-1)

    # Top-k and bottom-k indices per day
    long_idx = ranked[-k:]    # (k, n_days-1)
    short_idx = ranked[:k]    # (k, n_days-1)

    # Gather returns for long and short legs
    days = np.arange(sig.shape[1])[np.newaxis, :]  # (1, n_days-1)
    long_ret = ret_v[long_idx, days].mean(axis=0)
    short_ret = ret_v[short_idx, days].mean(axis=0)

    daily_returns = long_ret - short_ret
    # Zero out days with insufficient valid stocks
    daily_returns = np.where(n_valid >= k * 2, daily_returns, 0.0)

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

    # Turnover from normalized signals (vectorized argsort ranking)
    n_stocks_t, n_days_t = norm_signals.shape
    valid_t = ~np.isnan(norm_signals)
    n_valid_per_day = valid_t.sum(axis=0)
    n_valid_safe = np.maximum(n_valid_per_day, 1).astype(np.float64)

    sig_for_rank = np.where(valid_t, norm_signals, -np.inf)
    order = np.argsort(sig_for_rank, axis=0)
    ranks = np.empty_like(norm_signals, dtype=np.float64)
    day_idx_t = np.arange(n_days_t)[np.newaxis, :]
    row_ranks_t = np.arange(1, n_stocks_t + 1, dtype=np.float64)[:, np.newaxis]
    ranks[order, day_idx_t] = row_ranks_t
    # Normalize to [0, 1] per day
    pos = (ranks / n_valid_safe[np.newaxis, :]).T  # (n_days, n_stocks)
    turnover = float(np.nanmean(np.abs(np.diff(pos, axis=0)))) if n_days_t > 1 else 0.0

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
