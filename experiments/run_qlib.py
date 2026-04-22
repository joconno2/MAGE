#!/usr/bin/env python3
"""
Run Alpha Factory experiments on Qlib CSI300/CSI500 data.

Matches AlphaGen's exact data protocol:
  - Train: 2009-01-01 to 2018-12-31
  - Val: 2019-01-01 to 2019-12-31
  - Test: 2020-01-01 to 2021-12-31
  - Target: 20-day forward returns on Ref($close, -20)/$close - 1
  - Universe: CSI300 or CSI500 constituents

Usage:
    python experiments/run_qlib.py mapelites --market csi300 --output results/qlib_csi300_mapelites
    python experiments/run_qlib.py gp --market csi300 --output results/qlib_csi300_gp
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpha_factory.gp_genome import (
    Node, random_tree, mutate, crossover, compute_signals,
)
from alpha_factory.evaluate import (
    evaluate_signals, normalize_alpha, long_short_backtest,
)

# Find qlib data relative to this script, not home dir (works on any machine)
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
QLIB_DATA_DIR = _SCRIPT_DIR / "data" / "qlib" / "cn_data"


def load_qlib_data(market="csi300",
                    train_start="2009-01-01", train_end="2018-12-31",
                    val_start="2019-01-01", val_end="2019-12-31",
                    test_start="2020-01-01", test_end="2021-12-31"):
    """Load Qlib data and prepare splits matching AlphaGen."""
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(QLIB_DATA_DIR))

    # Get instrument list for this market
    inst_dict = D.instruments(market=market)
    inst_list = D.list_instruments(instruments=inst_dict, as_list=True)
    print(f"{market}: {len(inst_list)} instruments")

    fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]

    splits = {}
    for name, start, end in [
        ("train", train_start, train_end),
        ("val", val_start, val_end),
        ("test", test_start, test_end),
    ]:
        print(f"Loading {name}: {start} to {end}...")
        df = D.features(instruments=inst_list, fields=fields,
                        start_time=start, end_time=end)

        if df.empty:
            print(f"  No data for {name}, skipping")
            continue

        # Reshape from multi-index (instrument, datetime) to matrices
        df.columns = ["open", "high", "low", "close", "volume", "vwap"]
        tickers = df.index.get_level_values(0).unique().tolist()
        dates = df.index.get_level_values(1).unique().sort_values()

        # Filter to stocks with >80% coverage
        good_tickers = []
        for ticker in tickers:
            sub = df.loc[ticker] if ticker in df.index.get_level_values(0) else None
            if sub is not None and len(sub) > len(dates) * 0.8:
                good_tickers.append(ticker)

        if len(good_tickers) < 30:
            print(f"  Only {len(good_tickers)} stocks with sufficient data, skipping")
            continue

        tickers = good_tickers[:300]  # cap at 300 for memory
        n_stocks = len(tickers)
        n_days = len(dates)

        # Build matrices
        stock_data = {}
        close_list = []
        for ticker in tickers:
            try:
                sub = df.loc[ticker].reindex(dates).ffill().bfill()
                stock_data[ticker] = {
                    "open": sub["open"].values.astype(np.float64),
                    "high": sub["high"].values.astype(np.float64),
                    "low": sub["low"].values.astype(np.float64),
                    "close": sub["close"].values.astype(np.float64),
                    "volume": sub["volume"].values.astype(np.float64),
                    "vwap": sub["vwap"].values.astype(np.float64),
                }
                close_list.append(sub["close"].values.astype(np.float64))
            except Exception:
                pass

        if len(close_list) < 30:
            continue

        close_prices = np.array(close_list)
        n_stocks = close_prices.shape[0]

        # Forward returns
        fwd_1d = np.full_like(close_prices, np.nan)
        fwd_1d[:, :-1] = close_prices[:, 1:] / np.maximum(close_prices[:, :-1], 1e-10) - 1

        fwd_20d = np.full_like(close_prices, np.nan)
        if 20 < n_days:
            fwd_20d[:, :-20] = close_prices[:, 20:] / np.maximum(close_prices[:, :-20], 1e-10) - 1

        splits[name] = {
            "stock_data": stock_data,
            "close_prices": close_prices,
            "fwd_returns_1d": fwd_1d,
            "fwd_returns_20d": fwd_20d,
            "tickers": list(stock_data.keys()),
            "n_stocks": len(stock_data),
            "n_days": n_days,
        }
        print(f"  {name}: {len(stock_data)} stocks, {n_days} days")

    return splits


def _eval(tree, split):
    from alpha_factory.gp_genome import evaluate_tree
    return evaluate_tree(
        tree, split["stock_data"], split["close_prices"],
        split["fwd_returns_1d"], split["fwd_returns_20d"], split["n_days"],
    )


def run_mapelites(args, splits):
    train = splits["train"]
    rng = random.Random(args.seed)
    grid_size = args.grid_size
    grid = {}

    def cell(m):
        cx = min(grid_size - 1, max(0, int(m.turnover * grid_size * 5)))
        cy = min(grid_size - 1, max(0, int(m.market_corr * grid_size)))
        return (cx, cy)

    pop_size = args.pop
    population = [random_tree(max_depth=4, rng=rng) for _ in range(pop_size)]
    pop_fitness = [-999.0] * pop_size

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGP+MAP-Elites on {args.market} | pop={pop_size} gens={args.gens}")

    # Init
    for i, tree in enumerate(population):
        m = _eval(tree, train)
        pop_fitness[i] = m.sharpe if m.valid else -999
        if m.valid and m.sharpe > 0:
            c = cell(m)
            if c not in grid or m.sharpe > grid[c].sharpe:
                grid[c] = m
                grid[c]._tree = tree  # stash tree

    print(f"Initial coverage: {len(grid)}/{grid_size**2}")

    for gen in range(1, args.gens + 1):
        t0 = time.monotonic()
        inserted = 0
        new_pop, new_fit = [], []

        for _ in range(pop_size):
            src = rng.random()
            if src < 0.3 and grid:
                elite = rng.choice(list(grid.values()))
                child = mutate(elite._tree, rng=rng)
            elif src < 0.7:
                cands = rng.sample(range(len(population)), min(7, len(population)))
                winner = max(cands, key=lambda i: pop_fitness[i])
                child = mutate(population[winner], rng=rng)
            else:
                c1 = rng.sample(range(len(population)), min(7, len(population)))
                c2 = rng.sample(range(len(population)), min(7, len(population)))
                child = crossover(
                    population[max(c1, key=lambda i: pop_fitness[i])],
                    population[max(c2, key=lambda i: pop_fitness[i])],
                    rng=rng)

            m = _eval(child, train)
            f = m.sharpe if m.valid else -999
            new_pop.append(child)
            new_fit.append(f)

            if m.valid and m.sharpe > 0:
                c = cell(m)
                if c not in grid or m.sharpe > grid[c].sharpe:
                    m._tree = child
                    grid[c] = m
                    inserted += 1

        population = new_pop
        pop_fitness = new_fit
        dur = time.monotonic() - t0

        sharpes = [g.sharpe for g in grid.values()]
        best = max(sharpes) if sharpes else 0
        mean = float(np.mean(sharpes)) if sharpes else 0

        print(f"[gen {gen:3d}] coverage={len(grid)}/{grid_size**2} "
              f"best={best:.2f} mean={mean:.2f} +{inserted} dt={dur:.0f}s")

    # Save and evaluate on test
    print(f"\nTest set evaluation (top 10):")
    if "test" in splits:
        for g in sorted(grid.values(), key=lambda x: -x.sharpe)[:10]:
            m = _eval(g._tree, splits["test"])
            print(f"  train={g.sharpe:.2f} test={m.sharpe:.2f} IC={m.ic:.4f}")


def run_gp(args, splits):
    train = splits["train"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGP baseline on {args.market} | pop={args.pop} gens={args.gens} runs={args.n_runs}")

    for run in range(args.n_runs):
        t0 = time.monotonic()
        rng = random.Random(args.seed + run)
        pop = [random_tree(max_depth=4, rng=rng) for _ in range(args.pop)]
        pop_fitness = [(_eval(t, train).sharpe if _eval(t, train).valid else -999) for t in pop]

        for gen in range(args.gens):
            children = []
            for _ in range(args.pop):
                if rng.random() < 0.7:
                    cands = rng.sample(range(len(pop)), min(7, len(pop)))
                    children.append(mutate(pop[max(cands, key=lambda i: pop_fitness[i])], rng=rng))
                else:
                    c1 = rng.sample(range(len(pop)), min(7, len(pop)))
                    c2 = rng.sample(range(len(pop)), min(7, len(pop)))
                    children.append(crossover(
                        pop[max(c1, key=lambda i: pop_fitness[i])],
                        pop[max(c2, key=lambda i: pop_fitness[i])], rng=rng))
            pop = children
            pop_fitness = [(_eval(t, train).sharpe if _eval(t, train).valid else -999) for t in pop]

        best_idx = int(np.argmax(pop_fitness))
        best_m = _eval(pop[best_idx], train)
        dur = time.monotonic() - t0

        test_m = _eval(pop[best_idx], splits["test"]) if "test" in splits else None
        test_str = f" test={test_m.sharpe:.2f}" if test_m else ""

        print(f"  Run {run+1}/{args.n_runs}: Sharpe={best_m.sharpe:.2f} IC={best_m.ic:.4f}"
              f"{test_str} dt={dur:.0f}s | {best_m.expression[:50]}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="algorithm", required=True)

    def _shared(p):
        p.add_argument("--market", default="csi300", choices=["csi300", "csi500"])
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--output", required=True)

    me_p = sub.add_parser("mapelites")
    _shared(me_p)
    me_p.add_argument("--pop", type=int, default=100)
    me_p.add_argument("--gens", type=int, default=50)
    me_p.add_argument("--grid-size", type=int, default=20)

    gp_p = sub.add_parser("gp")
    _shared(gp_p)
    gp_p.add_argument("--pop", type=int, default=100)
    gp_p.add_argument("--gens", type=int, default=30)
    gp_p.add_argument("--n-runs", type=int, default=10)

    args = parser.parse_args()

    splits = load_qlib_data(market=args.market)
    if "train" not in splits:
        print("ERROR: Could not load training data")
        sys.exit(1)

    if args.algorithm == "mapelites":
        run_mapelites(args, splits)
    elif args.algorithm == "gp":
        run_gp(args, splits)


if __name__ == "__main__":
    main()
