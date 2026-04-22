"""
Alpha expression operators.

Full operator set matching AlphaGen (KDD 2023) and Alpha101 (Kakushadze 2016).
28 operators total: 4 CS-unary, 7 CS-binary, 15 TS-unary, 2 TS-binary.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata


def _safe_div(a, b):
    return np.where(np.abs(b) < 1e-10, 0.0, a / b)


# ── Cross-sectional unary (operate on all stocks for one day) ──────────

def cs_abs(x): return np.abs(x)
def cs_log(x): return np.where(x > 0, np.log(x), 0.0)
def cs_sign(x): return np.sign(x)
def cs_rank(x):
    valid = ~np.isnan(x)
    n = valid.sum()
    if n < 2:
        return np.full_like(x, 0.5)
    ranks = rankdata(np.where(valid, x, 0), nan_policy="omit")
    return ranks / max(n, 1)


# ── Cross-sectional binary ─────────────────────────────────────────────

def cs_add(x, y): return x + y
def cs_sub(x, y): return x - y
def cs_mul(x, y): return x * y
def cs_div(x, y): return _safe_div(x, y)
def cs_pow(x, y): return np.sign(x) * np.power(np.abs(x) + 1e-10, np.clip(y, -3, 3))
def cs_greater(x, y): return np.maximum(x, y)
def cs_less(x, y): return np.minimum(x, y)


# ── Time-series unary (operate on one stock across days) ───────────────

def ts_ref(x, d):
    """Value d days ago."""
    out = np.full_like(x, np.nan)
    if 0 < d < len(x):
        out[d:] = x[:-d]
    return out

def ts_delta(x, d):
    return x - ts_ref(x, d)

def ts_mean(x, d):
    if d <= 0 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    cs = np.nancumsum(x)
    out[d-1:] = (cs[d-1:] - np.concatenate([[0], cs[:-d]])) / d
    return out

def ts_med(x, d):
    if d <= 0 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d)
    out[d-1:] = np.nanmedian(w, axis=1)
    return out

def ts_sum(x, d):
    if d <= 0 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    cs = np.nancumsum(x)
    out[d-1:] = cs[d-1:] - np.concatenate([[0], cs[:-d]])
    return out

def ts_std(x, d):
    if d <= 1 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d)
    out[d-1:] = np.nanstd(w, axis=1)
    return out

def ts_var(x, d):
    if d <= 1 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d)
    out[d-1:] = np.nanvar(w, axis=1)
    return out

def ts_skew(x, d):
    if d <= 2 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d)
    m = np.nanmean(w, axis=1, keepdims=True)
    s = np.nanstd(w, axis=1, keepdims=True)
    s = np.where(s < 1e-10, 1.0, s)
    z = (w - m) / s
    out[d-1:] = np.nanmean(z ** 3, axis=1)
    return out

def ts_kurt(x, d):
    if d <= 3 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d)
    m = np.nanmean(w, axis=1, keepdims=True)
    s = np.nanstd(w, axis=1, keepdims=True)
    s = np.where(s < 1e-10, 1.0, s)
    z = (w - m) / s
    out[d-1:] = np.nanmean(z ** 4, axis=1) - 3.0
    return out

def ts_max(x, d):
    if d <= 0 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d)
    out[d-1:] = np.nanmax(w, axis=1)
    return out

def ts_min(x, d):
    if d <= 0 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d)
    out[d-1:] = np.nanmin(w, axis=1)
    return out

def ts_mad(x, d):
    """Mean absolute deviation."""
    if d <= 0 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d)
    means = np.nanmean(w, axis=1, keepdims=True)
    out[d-1:] = np.nanmean(np.abs(w - means), axis=1)
    return out

def ts_rank(x, d):
    """Percentile rank of current value in d-day window."""
    if d <= 1 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d)
    # Vectorized: count how many values in each window are <= the last value
    last_vals = w[:, -1:]  # (n_windows, 1)
    out[d-1:] = np.nanmean(w <= last_vals, axis=1)
    return out

def ts_wma(x, d):
    """Weighted moving average with linearly decaying weights."""
    if d <= 0 or d > len(x): return np.full_like(x, np.nan)
    weights = np.arange(1, d + 1, dtype=np.float64)
    weights /= weights.sum()
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d)
    out[d-1:] = np.nansum(w * weights, axis=1)
    return out

def ts_ema(x, d):
    """Exponential moving average. Uses pandas for vectorized EWM."""
    if d <= 0: return np.full_like(x, np.nan)
    import pandas as pd
    return pd.Series(x).ewm(span=d, min_periods=1).mean().values


# ── Time-series binary ─────────────────────────────────────────────────

def ts_corr(x, y, d):
    if d <= 2 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    wx = sliding_window_view(x, d)
    wy = sliding_window_view(y, d)
    mx = np.nanmean(wx, axis=1, keepdims=True)
    my = np.nanmean(wy, axis=1, keepdims=True)
    sx = np.nanstd(wx, axis=1)
    sy = np.nanstd(wy, axis=1)
    cov = np.nanmean((wx - mx) * (wy - my), axis=1)
    denom = sx * sy
    out[d-1:] = np.where(denom < 1e-10, 0.0, cov / denom)
    return out

def ts_cov(x, y, d):
    if d <= 2 or d > len(x): return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    wx = sliding_window_view(x, d)
    wy = sliding_window_view(y, d)
    mx = np.nanmean(wx, axis=1, keepdims=True)
    my = np.nanmean(wy, axis=1, keepdims=True)
    out[d-1:] = np.nanmean((wx - mx) * (wy - my), axis=1)
    return out


# ── Helper ─────────────────────────────────────────────────────────────

def returns(close):
    ret = np.full_like(close, np.nan)
    ret[1:] = _safe_div(close[1:] - close[:-1], close[:-1])
    return ret

def safe_log(x): return np.where(x > 0, np.log(x), 0.0)
def safe_sqrt(x): return np.sqrt(np.abs(x))


# ── Registry ───────────────────────────────────────────────────────────

# (name, arity, function, has_window_param, window_range)
# arity: 1 = unary (1 series + optional window), 2 = binary (2 series + optional window)
TS_UNARY_OPS = [
    ("ts_ref", 1, ts_ref, True, (1, 20)),
    ("ts_delta", 1, ts_delta, True, (1, 20)),
    ("ts_mean", 1, ts_mean, True, (3, 60)),
    ("ts_med", 1, ts_med, True, (3, 60)),
    ("ts_sum", 1, ts_sum, True, (3, 60)),
    ("ts_std", 1, ts_std, True, (3, 60)),
    ("ts_var", 1, ts_var, True, (3, 60)),
    ("ts_skew", 1, ts_skew, True, (5, 60)),
    ("ts_kurt", 1, ts_kurt, True, (5, 60)),
    ("ts_max", 1, ts_max, True, (3, 60)),
    ("ts_min", 1, ts_min, True, (3, 60)),
    ("ts_mad", 1, ts_mad, True, (3, 60)),
    ("ts_rank", 1, ts_rank, True, (3, 60)),
    ("ts_wma", 1, ts_wma, True, (3, 60)),
    ("ts_ema", 1, ts_ema, True, (3, 60)),
]

TS_BINARY_OPS = [
    ("ts_corr", 2, ts_corr, True, (5, 60)),
    ("ts_cov", 2, ts_cov, True, (5, 60)),
]

CS_UNARY_OPS = [
    ("cs_abs", 1, cs_abs, False, None),
    ("cs_log", 1, cs_log, False, None),
    ("cs_sign", 1, cs_sign, False, None),
    ("cs_rank", 1, cs_rank, False, None),
]

CS_BINARY_OPS = [
    ("add", 2, cs_add, False, None),
    ("sub", 2, cs_sub, False, None),
    ("mul", 2, cs_mul, False, None),
    ("div", 2, cs_div, False, None),
    ("pow", 2, cs_pow, False, None),
    ("greater", 2, cs_greater, False, None),
    ("less", 2, cs_less, False, None),
]

ALL_OPS = TS_UNARY_OPS + TS_BINARY_OPS + CS_UNARY_OPS + CS_BINARY_OPS
OP_DICT = {op[0]: op for op in ALL_OPS}
UNARY_OPS = [op for op in ALL_OPS if op[1] == 1]
BINARY_OPS = [op for op in ALL_OPS if op[1] == 2]

# Input features (raw OHLCV, matching AlphaGen)
FEATURES = ["open", "high", "low", "close", "volume", "vwap"]

# Time window tokens (matching AlphaGen)
WINDOWS = [1, 5, 10, 20, 40]

# Constants (matching AlphaGen)
CONSTANTS = [-30, -10, -5, -2, -1, -0.5, -0.01, 0.01, 0.5, 1, 2, 5, 10, 30]
