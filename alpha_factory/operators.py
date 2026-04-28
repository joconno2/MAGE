"""
Alpha expression operators.

Full operator set matching AlphaGen (KDD 2023) and Alpha101 (Kakushadze 2016).
28 operators total: 4 CS-unary, 7 CS-binary, 15 TS-unary, 2 TS-binary.

All TS operators accept both 1D (n_days,) and 2D (n_stocks, n_days) input.
Operations apply along the last axis, so a single call on the full matrix
replaces the previous per-stock Python loop.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata


def _safe_div(a, b):
    return np.where(np.abs(b) < 1e-10, 0.0, a / b)


def _rolling_moments(x, d, max_power=2):
    """Compute rolling sums of x^1..x^max_power and valid count via cumsum.

    Returns dict: {1: sum_x, 2: sum_x2, ...} and 'count' key for non-NaN count.
    All arrays have NaN for warmup positions [0..d-2].
    O(n) per power level, no 3D intermediate arrays.
    """
    xf = np.where(np.isnan(x), 0.0, x).astype(np.float64)
    valid = (~np.isnan(x)).astype(np.float64)
    zeros = np.zeros(x.shape[:-1] + (1,), dtype=np.float64)

    def _rsum(arr):
        cs = np.cumsum(arr, axis=-1)
        out = np.full(x.shape, np.nan, dtype=np.float64)
        out[..., d-1:] = cs[..., d-1:] - np.concatenate([zeros, cs[..., :-d]], axis=-1)
        return out

    result = {'count': _rsum(valid)}
    xp = xf
    for p in range(1, max_power + 1):
        if p > 1:
            xp = xp * xf
        result[p] = _rsum(xp)
    return result


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


# ── Time-series unary (operate along last axis: days) ──────────────────

def ts_ref(x, d):
    """Value d days ago."""
    n = x.shape[-1]
    out = np.full_like(x, np.nan)
    if 0 < d < n:
        out[..., d:] = x[..., :-d]
    return out

def ts_delta(x, d):
    return x - ts_ref(x, d)

def ts_mean(x, d):
    n = x.shape[-1]
    if d <= 0 or d > n: return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    cs = np.nancumsum(x, axis=-1)
    zeros = np.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    out[..., d-1:] = (cs[..., d-1:] - np.concatenate([zeros, cs[..., :-d]], axis=-1)) / d
    return out

def ts_med(x, d):
    n = x.shape[-1]
    if d <= 0 or d > n: return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d, axis=-1)
    out[..., d-1:] = np.nanmedian(w, axis=-1)
    return out

def ts_sum(x, d):
    n = x.shape[-1]
    if d <= 0 or d > n: return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    cs = np.nancumsum(x, axis=-1)
    zeros = np.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    out[..., d-1:] = cs[..., d-1:] - np.concatenate([zeros, cs[..., :-d]], axis=-1)
    return out

def ts_std(x, d):
    n = x.shape[-1]
    if d <= 1 or d > n: return np.full_like(x, np.nan)
    m = _rolling_moments(x, d, max_power=2)
    cnt = np.maximum(m['count'], 1.0)
    mean = m[1] / cnt
    var = m[2] / cnt - mean ** 2
    out = np.sqrt(np.maximum(var, 0.0))
    out = np.where(m['count'] >= 2, out, np.nan)
    return out

def ts_var(x, d):
    n = x.shape[-1]
    if d <= 1 or d > n: return np.full_like(x, np.nan)
    m = _rolling_moments(x, d, max_power=2)
    cnt = np.maximum(m['count'], 1.0)
    mean = m[1] / cnt
    var = m[2] / cnt - mean ** 2
    out = np.maximum(var, 0.0)
    out = np.where(m['count'] >= 2, out, np.nan)
    return out

def ts_skew(x, d):
    n = x.shape[-1]
    if d <= 2 or d > n: return np.full_like(x, np.nan)
    m = _rolling_moments(x, d, max_power=3)
    cnt = np.maximum(m['count'], 1.0)
    m1 = m[1] / cnt
    m2 = m[2] / cnt
    m3 = m[3] / cnt
    var = m2 - m1 ** 2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    # skew = (m3 - 3*m1*m2 + 2*m1^3) / std^3
    num = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    denom = np.where(std < 1e-10, 1.0, std ** 3)
    out = np.where(std < 1e-10, 0.0, num / denom)
    out = np.where(m['count'] >= 3, out, np.nan)
    return out

def ts_kurt(x, d):
    n = x.shape[-1]
    if d <= 3 or d > n: return np.full_like(x, np.nan)
    m = _rolling_moments(x, d, max_power=4)
    cnt = np.maximum(m['count'], 1.0)
    m1 = m[1] / cnt
    m2 = m[2] / cnt
    m3 = m[3] / cnt
    m4 = m[4] / cnt
    var = m2 - m1 ** 2
    var = np.maximum(var, 0.0)
    # kurt = (m4 - 4*m1*m3 + 6*m1^2*m2 - 3*m1^4) / var^2 - 3
    num = m4 - 4.0 * m1 * m3 + 6.0 * m1 ** 2 * m2 - 3.0 * m1 ** 4
    var2 = np.where(var < 1e-10, 1.0, var ** 2)
    out = np.where(var < 1e-10, 0.0, num / var2 - 3.0)
    out = np.where(m['count'] >= 4, out, np.nan)
    return out

def ts_max(x, d):
    n = x.shape[-1]
    if d <= 0 or d > n: return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d, axis=-1)
    out[..., d-1:] = np.nanmax(w, axis=-1)
    return out

def ts_min(x, d):
    n = x.shape[-1]
    if d <= 0 or d > n: return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d, axis=-1)
    out[..., d-1:] = np.nanmin(w, axis=-1)
    return out

def ts_mad(x, d):
    """Mean absolute deviation."""
    n = x.shape[-1]
    if d <= 0 or d > n: return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d, axis=-1)
    means = np.nanmean(w, axis=-1, keepdims=True)
    out[..., d-1:] = np.nanmean(np.abs(w - means), axis=-1)
    return out

def ts_rank(x, d):
    """Percentile rank of current value in d-day window."""
    n = x.shape[-1]
    if d <= 1 or d > n: return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d, axis=-1)
    last_vals = w[..., -1:]
    out[..., d-1:] = np.nanmean(w <= last_vals, axis=-1)
    return out

def ts_wma(x, d):
    """Weighted moving average with linearly decaying weights."""
    n = x.shape[-1]
    if d <= 0 or d > n: return np.full_like(x, np.nan)
    weights = np.arange(1, d + 1, dtype=np.float64)
    weights /= weights.sum()
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d, axis=-1)
    out[..., d-1:] = np.nansum(w * weights, axis=-1)
    return out

def ts_ema(x, d):
    """Exponential moving average."""
    if d <= 0: return np.full_like(x, np.nan)
    import pandas as pd
    if x.ndim == 1:
        return pd.Series(x).ewm(span=d, min_periods=1).mean().values
    # 2D: pandas DataFrame ewm applies per-column, so transpose
    return pd.DataFrame(x.T).ewm(span=d, min_periods=1).mean().values.T


# ── Time-series binary ─────────────────────────────────────────────────

def ts_corr(x, y, d):
    n = x.shape[-1]
    if d <= 2 or d > n: return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    wx = sliding_window_view(x, d, axis=-1)
    wy = sliding_window_view(y, d, axis=-1)
    mx = np.nanmean(wx, axis=-1, keepdims=True)
    my = np.nanmean(wy, axis=-1, keepdims=True)
    sx = np.nanstd(wx, axis=-1)
    sy = np.nanstd(wy, axis=-1)
    cov = np.nanmean((wx - mx) * (wy - my), axis=-1)
    denom = sx * sy
    out[..., d-1:] = np.where(denom < 1e-10, 0.0, cov / denom)
    return out

def ts_cov(x, y, d):
    n = x.shape[-1]
    if d <= 2 or d > n: return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    wx = sliding_window_view(x, d, axis=-1)
    wy = sliding_window_view(y, d, axis=-1)
    mx = np.nanmean(wx, axis=-1, keepdims=True)
    my = np.nanmean(wy, axis=-1, keepdims=True)
    out[..., d-1:] = np.nanmean((wx - mx) * (wy - my), axis=-1)
    return out


# ── New TS operators (Tier 1 + Tier 2) ────────────────────────────────

def ts_argmax(x, d):
    """Index of max value in d-day window (0 = oldest, d-1 = newest)."""
    n = x.shape[-1]
    if d <= 0 or d > n: return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d, axis=-1)
    out[..., d-1:] = np.nanargmax(w, axis=-1).astype(np.float64)
    return out

def ts_argmin(x, d):
    """Index of min value in d-day window (0 = oldest, d-1 = newest)."""
    n = x.shape[-1]
    if d <= 0 or d > n: return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d, axis=-1)
    out[..., d-1:] = np.nanargmin(w, axis=-1).astype(np.float64)
    return out

def ts_product(x, d):
    """Rolling product over d days."""
    n = x.shape[-1]
    if d <= 0 or d > n: return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    xc = np.clip(x, -10, 10)
    w = sliding_window_view(xc, d, axis=-1)
    out[..., d-1:] = np.nanprod(w, axis=-1)
    return out

def ts_decay_linear(x, d):
    """Linearly decaying weighted average (most recent = weight d, oldest = weight 1)."""
    n = x.shape[-1]
    if d <= 0 or d > n: return np.full_like(x, np.nan)
    weights = np.arange(1, d + 1, dtype=np.float64)
    weights /= weights.sum()
    out = np.full_like(x, np.nan)
    w = sliding_window_view(x, d, axis=-1)
    out[..., d-1:] = np.nansum(w * weights, axis=-1)
    return out


# ── New CS unary operators (Tier 2) ───────────────────────────────────

def cs_scale(x):
    """Normalize to sum(abs(x)) = 1, preserving sign."""
    total = np.nansum(np.abs(x))
    if total < 1e-10:
        return np.zeros_like(x)
    return x / total

def cs_signedpower(x, e=2.0):
    """sign(x) * |x|^e, preserves sign with nonlinear scaling."""
    return np.sign(x) * np.power(np.abs(x) + 1e-10, e)


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
    ("ts_argmax", 1, ts_argmax, True, (3, 60)),
    ("ts_argmin", 1, ts_argmin, True, (3, 60)),
    ("ts_product", 1, ts_product, True, (2, 10)),
    ("ts_decay_linear", 1, ts_decay_linear, True, (3, 60)),
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
    ("cs_scale", 1, cs_scale, False, None),
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

# Input features: raw OHLCV + derived features that break price-ratio convergence
FEATURES = [
    "open", "high", "low", "close", "volume", "vwap",
    # Derived features (computed in data.py prepare_eval_data)
    "returns",          # daily returns: (close - prev_close) / prev_close
    "log_return",       # log(close / prev_close)
    "dollar_volume",    # close * volume (liquidity)
    "turnover_ratio",   # volume / adv20 (activity surprise)
    "intraday_range",   # (high - low) / close (volatility proxy)
    "gap",              # open / prev_close - 1 (overnight return)
    "upper_shadow",     # (high - max(open, close)) / close (selling pressure)
    "lower_shadow",     # (min(open, close) - low) / close (buying pressure)
    "body",             # (close - open) / close (directional conviction)
]

# Time window tokens (matching AlphaGen)
WINDOWS = [1, 5, 10, 20, 40]

# Constants (matching AlphaGen)
CONSTANTS = [-30, -10, -5, -2, -1, -0.5, -0.01, 0.01, 0.5, 1, 2, 5, 10, 30]
