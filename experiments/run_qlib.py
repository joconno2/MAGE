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


def _max_corr_with_archive(flat_sig, archive_signals, threshold=0.7):
    if not archive_signals:
        return 0.0, True
    valid = ~np.isnan(flat_sig)
    if valid.sum() < 100:
        return 1.0, False
    sig = np.where(valid, flat_sig, 0.0)
    max_corr = 0.0
    for archived in archive_signals.values():
        v = valid & ~np.isnan(archived)
        if v.sum() < 100:
            continue
        corr = abs(np.corrcoef(sig[v], archived[v])[0, 1])
        if not np.isnan(corr) and corr > max_corr:
            max_corr = corr
            if max_corr > threshold:
                return max_corr, False
    return max_corr, True

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

        # Build matrices with derived features (matches S&P 500 pipeline)
        stock_data = {}
        close_list = []
        for ticker in tickers:
            try:
                sub = df.loc[ticker].reindex(dates).ffill().bfill()
                o = sub["open"].values.astype(np.float64)
                h = sub["high"].values.astype(np.float64)
                l = sub["low"].values.astype(np.float64)
                c = sub["close"].values.astype(np.float64)
                v = sub["volume"].values.astype(np.float64)
                vw = sub["vwap"].values.astype(np.float64)

                ret = np.full_like(c, np.nan)
                ret[1:] = (c[1:] - c[:-1]) / np.maximum(np.abs(c[:-1]), 1e-10)
                log_ret = np.full_like(c, np.nan)
                log_ret[1:] = np.log(np.maximum(c[1:], 1e-10) / np.maximum(c[:-1], 1e-10))
                dollar_vol = c * v
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
                    "open": o, "high": h, "low": l, "close": c,
                    "volume": v, "vwap": vw,
                    "returns": ret, "log_return": log_ret,
                    "dollar_volume": dollar_vol, "turnover_ratio": turnover_ratio,
                    "intraday_range": intraday_range, "gap": gap,
                    "upper_shadow": upper_shadow, "lower_shadow": lower_shadow,
                    "body": body,
                }
                close_list.append(c)
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
    grid = {}  # (cx, cy) -> {tree, sharpe, ic, ...}
    archive_signals = {}  # (cx, cy) -> flattened signal
    corr_threshold = args.corr_threshold

    def cell(m):
        cx = min(grid_size - 1, max(0, int(m.turnover * grid_size * 5)))
        cy = min(grid_size - 1, max(0, int(m.market_corr * grid_size)))
        return (cx, cy)

    def try_insert(tree, m):
        c = cell(m)
        if c in grid and m.sharpe <= grid[c]["sharpe"]:
            return False
        signals = compute_signals(tree, train["stock_data"], train["n_days"])
        flat_sig = normalize_alpha(signals).flatten()
        max_corr, passes = _max_corr_with_archive(flat_sig, archive_signals, corr_threshold)
        if not passes:
            return False
        grid[c] = {
            "tree": tree, "tree_str": str(tree),
            "sharpe": m.sharpe, "ic": m.ic, "rank_ic": m.rank_ic,
            "icir": m.icir, "turnover": m.turnover,
            "market_corr": m.market_corr, "annual_return": m.annual_return,
            "max_drawdown": m.max_drawdown, "max_corr": max_corr,
        }
        archive_signals[c] = flat_sig
        return True

    pop_size = args.pop
    population = [random_tree(max_depth=4, rng=rng) for _ in range(pop_size)]
    pop_fitness = [-999.0] * pop_size

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {"algorithm": "gp_mapelites_qlib", "market": args.market,
              "pop_size": pop_size, "generations": args.gens,
              "grid_size": grid_size, "corr_threshold": corr_threshold}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nMAGE on {args.market} | pop={pop_size} gens={args.gens} gate={corr_threshold}")

    # Init
    for i, tree in enumerate(population):
        m = _eval(tree, train)
        pop_fitness[i] = m.sharpe if m.valid else -999
        if m.valid and m.sharpe > 0:
            try_insert(tree, m)

    print(f"Initial coverage: {len(grid)}/{grid_size**2}")
    history = []

    for gen in range(1, args.gens + 1):
        t0 = time.monotonic()
        inserted = 0
        new_pop, new_fit = [], []

        for _ in range(pop_size):
            src = rng.random()
            if src < 0.3 and grid:
                elite = rng.choice(list(grid.values()))
                child = mutate(elite["tree"], rng=rng)
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
                if try_insert(child, m):
                    inserted += 1

        population = new_pop
        pop_fitness = new_fit
        dur = time.monotonic() - t0

        sharpes = [g["sharpe"] for g in grid.values()]
        best = max(sharpes) if sharpes else 0
        mean = float(np.mean(sharpes)) if sharpes else 0
        history.append({"generation": gen, "coverage": len(grid),
                       "best_sharpe": best, "mean_sharpe": mean,
                       "inserted": inserted, "duration_s": dur})

        print(f"[gen {gen:3d}] coverage={len(grid)}/{grid_size**2} "
              f"best={best:.2f} mean={mean:.2f} +{inserted} dt={dur:.0f}s")

        if gen % 5 == 0 or gen == args.gens:
            grid_serial = {f"{k[0]},{k[1]}": {key: val for key, val in v.items() if key != "tree"}
                          for k, v in grid.items()}
            ckpt = {"generation": gen, "coverage": len(grid),
                    "grid": grid_serial, "history": history}
            (output_dir / "checkpoint.json").write_text(json.dumps(ckpt, indent=2))
            with open(output_dir / "grid.pkl", "wb") as f:
                pickle.dump(grid, f)

    # Test set evaluation
    if "test" in splits:
        print(f"\nTest set evaluation (top 20):")
        top = sorted(grid.values(), key=lambda x: -x["sharpe"])[:20]
        for i, g in enumerate(top):
            m = _eval(g["tree"], splits["test"])
            print(f"  #{i+1}: train={g['sharpe']:.2f} test={m.sharpe:.2f} IC={m.ic:.4f} | {g['tree_str'][:50]}")

    print(f"\nDone. Coverage: {len(grid)}/{grid_size**2}")


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
    me_p.add_argument("--corr-threshold", type=float, default=0.70)

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
