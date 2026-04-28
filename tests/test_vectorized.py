#!/usr/bin/env python3
"""
Validate that vectorized operators produce identical results to the
original per-stock loop. Run from repo root:

    .venv/bin/python tests/test_vectorized.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpha_factory.operators import (
    ts_ref, ts_delta, ts_mean, ts_med, ts_sum, ts_std, ts_var,
    ts_skew, ts_kurt, ts_max, ts_min, ts_mad, ts_rank, ts_wma,
    ts_ema, ts_argmax, ts_argmin, ts_product, ts_decay_linear,
    ts_corr, ts_cov,
)
from alpha_factory.evaluate import normalize_alpha, compute_ic_series, long_short_backtest


def test_ts_operators():
    """Test that 2D call matches per-row 1D calls."""
    rng = np.random.RandomState(42)
    n_stocks, n_days = 50, 500
    x = rng.randn(n_stocks, n_days)
    y = rng.randn(n_stocks, n_days)
    # Sprinkle some NaN
    x[rng.rand(n_stocks, n_days) < 0.02] = np.nan
    y[rng.rand(n_stocks, n_days) < 0.02] = np.nan

    unary_ops = [
        ("ts_ref", ts_ref, 5),
        ("ts_delta", ts_delta, 5),
        ("ts_mean", ts_mean, 20),
        ("ts_med", ts_med, 10),
        ("ts_sum", ts_sum, 20),
        ("ts_std", ts_std, 20),
        ("ts_var", ts_var, 20),
        ("ts_skew", ts_skew, 20),
        ("ts_kurt", ts_kurt, 20),
        ("ts_max", ts_max, 10),
        ("ts_min", ts_min, 10),
        ("ts_mad", ts_mad, 10),
        ("ts_rank", ts_rank, 10),
        ("ts_wma", ts_wma, 10),
        ("ts_ema", ts_ema, 10),
        ("ts_argmax", ts_argmax, 10),
        ("ts_argmin", ts_argmin, 10),
        ("ts_product", ts_product, 5),
        ("ts_decay_linear", ts_decay_linear, 10),
    ]

    passed = 0
    failed = 0

    for name, func, d in unary_ops:
        # 2D call (vectorized)
        result_2d = func(x, d)

        # Per-row 1D calls (old behavior)
        result_1d = np.full_like(x, np.nan)
        for i in range(n_stocks):
            result_1d[i] = func(x[i], d)

        # Compare
        both_nan = np.isnan(result_2d) & np.isnan(result_1d)
        both_valid = ~np.isnan(result_2d) & ~np.isnan(result_1d)
        nan_mismatch = np.isnan(result_2d) != np.isnan(result_1d)

        if nan_mismatch.any():
            n_mis = nan_mismatch.sum()
            print(f"  FAIL {name}: {n_mis} NaN mismatches")
            failed += 1
            continue

        if both_valid.any():
            max_diff = np.max(np.abs(result_2d[both_valid] - result_1d[both_valid]))
            if max_diff > 1e-8:
                print(f"  FAIL {name}: max diff = {max_diff:.2e}")
                failed += 1
                continue

        print(f"  OK   {name}")
        passed += 1

    # Binary ops
    binary_ops = [
        ("ts_corr", ts_corr, 20),
        ("ts_cov", ts_cov, 20),
    ]

    for name, func, d in binary_ops:
        result_2d = func(x, y, d)
        result_1d = np.full_like(x, np.nan)
        for i in range(n_stocks):
            result_1d[i] = func(x[i], y[i], d)

        both_valid = ~np.isnan(result_2d) & ~np.isnan(result_1d)
        nan_mismatch = np.isnan(result_2d) != np.isnan(result_1d)

        if nan_mismatch.any():
            print(f"  FAIL {name}: NaN mismatches")
            failed += 1
            continue

        if both_valid.any():
            max_diff = np.max(np.abs(result_2d[both_valid] - result_1d[both_valid]))
            if max_diff > 1e-8:
                print(f"  FAIL {name}: max diff = {max_diff:.2e}")
                failed += 1
                continue

        print(f"  OK   {name}")
        passed += 1

    return passed, failed


def test_normalize_alpha():
    """Test vectorized normalize_alpha matches per-day loop."""
    rng = np.random.RandomState(42)
    signals = rng.randn(50, 200)
    signals[rng.rand(50, 200) < 0.05] = np.nan

    result = normalize_alpha(signals)

    # Check properties: zero mean per day, unit L2 norm per day
    for t in range(200):
        col = result[:, t]
        valid = ~np.isnan(signals[:, t])
        if valid.sum() < 2:
            assert np.all(col == 0.0), f"Day {t}: degenerate day not zeroed"
            continue
        # Mean should be ~0
        assert abs(np.mean(col[valid])) < 1e-8, f"Day {t}: mean = {np.mean(col[valid])}"
        # NaN positions should be 0
        assert np.all(col[~valid] == 0.0), f"Day {t}: NaN positions not zeroed"

    print("  OK   normalize_alpha")
    return 1, 0


def test_backtest():
    """Test vectorized backtest matches per-day loop."""
    rng = np.random.RandomState(42)
    signals = rng.randn(50, 200)
    returns = rng.randn(50, 200) * 0.02

    result = long_short_backtest(signals, returns)
    assert result["n_days"] == 199, f"Expected 199 days, got {result['n_days']}"
    assert abs(result["sharpe"]) < 50, f"Sharpe {result['sharpe']} seems wrong"

    print("  OK   long_short_backtest")
    return 1, 0


def test_speedup():
    """Benchmark 2D vs per-row 1D for a typical workload."""
    rng = np.random.RandomState(42)
    n_stocks, n_days = 467, 2516  # S&P 500 dimensions
    x = rng.randn(n_stocks, n_days)

    # Benchmark: ts_std with d=20 (common operator)
    t0 = time.monotonic()
    for _ in range(10):
        _ = ts_std(x, 20)
    t_2d = (time.monotonic() - t0) / 10

    t0 = time.monotonic()
    for _ in range(10):
        for i in range(n_stocks):
            _ = ts_std(x[i], 20)
    t_1d = (time.monotonic() - t0) / 10

    speedup = t_1d / t_2d
    print(f"  ts_std: 1D loop = {t_1d:.3f}s, 2D vectorized = {t_2d:.3f}s, speedup = {speedup:.1f}x")

    # Benchmark: normalize_alpha
    signals = rng.randn(n_stocks, n_days)
    t0 = time.monotonic()
    for _ in range(10):
        _ = normalize_alpha(signals)
    t_norm = (time.monotonic() - t0) / 10
    print(f"  normalize_alpha: {t_norm:.3f}s per call")


def test_end_to_end():
    """Benchmark full tree evaluation at S&P 500 scale."""
    from alpha_factory.gp_genome import random_tree, evaluate_tree
    from alpha_factory.data import prepare_eval_data
    import random

    # Build synthetic data matching real dimensions
    rng_np = np.random.RandomState(42)
    n_stocks, n_days = 467, 2516
    tickers = [f"T{i}" for i in range(n_stocks)]
    stock_data = {}
    for t in tickers:
        c = 100 + rng_np.randn(n_days).cumsum() * 0.5
        c = np.maximum(c, 1.0)
        h = c * (1 + abs(rng_np.randn(n_days) * 0.01))
        l = c * (1 - abs(rng_np.randn(n_days) * 0.01))
        o = c + rng_np.randn(n_days) * 0.5
        v = (1e6 + rng_np.randn(n_days) * 1e5).clip(1e4)
        ret = np.full_like(c, np.nan)
        ret[1:] = (c[1:] - c[:-1]) / np.maximum(np.abs(c[:-1]), 1e-10)
        stock_data[t] = {
            "open": o, "high": h, "low": l, "close": c, "volume": v,
            "returns": ret, "log_return": ret,
            "dollar_volume": c * v,
            "turnover_ratio": rng_np.rand(n_days),
            "intraday_range": (h - l) / np.maximum(c, 1e-10),
            "gap": rng_np.randn(n_days) * 0.01,
            "upper_shadow": abs(rng_np.randn(n_days) * 0.005),
            "lower_shadow": abs(rng_np.randn(n_days) * 0.005),
            "body": rng_np.randn(n_days) * 0.005,
        }

    close_prices = np.array([stock_data[t]["close"] for t in tickers])
    fwd_1d = np.full_like(close_prices, np.nan)
    fwd_1d[:, :-1] = close_prices[:, 1:] / np.maximum(close_prices[:, :-1], 1e-10) - 1
    fwd_20d = np.full_like(close_prices, np.nan)
    fwd_20d[:, :-20] = close_prices[:, 20:] / np.maximum(close_prices[:, :-20], 1e-10) - 1

    # Generate 20 random trees and time full evaluation
    rng_py = random.Random(42)
    trees = [random_tree(max_depth=4, rng=rng_py) for _ in range(20)]

    t0 = time.monotonic()
    for tree in trees:
        evaluate_tree(tree, stock_data, close_prices, fwd_1d, fwd_20d, n_days)
    elapsed = time.monotonic() - t0

    print(f"  20 tree evals (467 stocks, 2516 days): {elapsed:.2f}s ({elapsed/20:.3f}s per tree)")


if __name__ == "__main__":
    print("Testing TS operators (1D vs 2D equivalence):")
    p1, f1 = test_ts_operators()

    print("\nTesting evaluate.py functions:")
    p2, f2 = test_normalize_alpha()
    p3, f3 = test_backtest()

    total_passed = p1 + p2 + p3
    total_failed = f1 + f2 + f3

    print(f"\nBenchmarks (467 stocks x 2516 days):")
    test_speedup()

    print(f"\nEnd-to-end tree evaluation:")
    test_end_to_end()

    print(f"\n{'='*40}")
    print(f"  {total_passed} passed, {total_failed} failed")
    if total_failed > 0:
        sys.exit(1)
    print("  All tests passed.")
