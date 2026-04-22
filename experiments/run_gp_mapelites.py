#!/usr/bin/env python3
"""
GP + MAP-Elites alpha factory.

GP trees for expressive search. MAP-Elites archive for behavioral diversity.
Full AlphaGen-compatible evaluation: 20-day forward returns, per-day normalization,
TopkDropout portfolio backtest.

Also runs GP-only and random baselines for comparison.

Usage:
    # GP + MAP-Elites (main contribution)
    python experiments/run_gp_mapelites.py mapelites --output results/gp_mapelites_v1

    # GP baseline (10 independent runs, measures collapse)
    python experiments/run_gp_mapelites.py gp --output results/gp_baseline_v1

    # Random search baseline (matched eval budget)
    python experiments/run_gp_mapelites.py random --output results/random_baseline
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

from alpha_factory.data import download_ohlcv, prepare_eval_data, SP100_TICKERS
from alpha_factory.gp_genome import (
    Node, random_tree, mutate, crossover, compute_signals, evaluate_tree,
)
from alpha_factory.evaluate import AlphaMetrics


def _eval(tree, split):
    """Evaluate a tree on a data split. Returns AlphaMetrics."""
    return evaluate_tree(
        tree,
        split["stock_data"],
        split["close_prices"],
        split["fwd_returns_1d"],
        split["fwd_returns_20d"],
        split["n_days"],
    )


# ── MAP-Elites ─────────────────────────────────────────────────────────

def run_mapelites(args):
    raw = download_ohlcv(SP100_TICKERS[:args.n_stocks])
    splits = prepare_eval_data(raw)
    train = splits["train"]
    print(f"Data: {train['n_stocks']} stocks, {train['n_days']} days")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    grid_size = args.grid_size
    grid = {}  # (cx, cy) -> {tree, metrics, tree_str}

    def cell(metrics: AlphaMetrics):
        # Axis 0: turnover [0, 0.2] -> [0, grid_size]
        cx = min(grid_size - 1, max(0, int(metrics.turnover * grid_size * 5)))
        # Axis 1: market_corr [0, 1] -> [0, grid_size]
        cy = min(grid_size - 1, max(0, int(metrics.market_corr * grid_size)))
        return (cx, cy)

    # Population for GP dynamics
    pop_size = args.pop
    population = [random_tree(max_depth=4, rng=rng) for _ in range(pop_size)]
    pop_fitness = [-999.0] * pop_size

    config = {
        "algorithm": "gp_mapelites",
        "pop_size": pop_size,
        "generations": args.gens,
        "grid_size": grid_size,
        "n_stocks": train["n_stocks"],
        "n_days": train["n_days"],
        "tournament": args.tournament,
        "eval": "20d_fwd_returns, normalized, topk50_drop5",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nGP+MAP-Elites | pop={pop_size} gens={args.gens} grid={grid_size}x{grid_size}")
    print(f"Output: {output_dir}\n")

    # Init
    print(f"Initializing ({pop_size} trees)...")
    for i, tree in enumerate(population):
        m = _eval(tree, train)
        pop_fitness[i] = m.sharpe if m.valid else -999
        if m.valid and m.sharpe > 0:
            c = cell(m)
            if c not in grid or m.sharpe > grid[c]["sharpe"]:
                grid[c] = {
                    "tree": tree,
                    "tree_str": str(tree),
                    "sharpe": m.sharpe,
                    "ic": m.ic,
                    "rank_ic": m.rank_ic,
                    "icir": m.icir,
                    "turnover": m.turnover,
                    "market_corr": m.market_corr,
                    "annual_return": m.annual_return,
                    "max_drawdown": m.max_drawdown,
                }

    print(f"Initial coverage: {len(grid)}/{grid_size**2}")

    history = []
    total_evals = pop_size

    for gen in range(1, args.gens + 1):
        t0 = time.monotonic()
        inserted = 0

        new_pop = []
        new_fitness = []

        for _ in range(pop_size):
            src = rng.random()

            if src < 0.3 and grid:
                # Parent from archive elite
                elite = rng.choice(list(grid.values()))
                child = mutate(elite["tree"], rng=rng)
            elif src < 0.7:
                # Tournament from population
                candidates = rng.sample(range(len(population)), min(args.tournament, len(population)))
                winner = max(candidates, key=lambda i: pop_fitness[i])
                child = mutate(population[winner], rng=rng)
            else:
                # Crossover
                c1 = rng.sample(range(len(population)), min(args.tournament, len(population)))
                c2 = rng.sample(range(len(population)), min(args.tournament, len(population)))
                p1 = population[max(c1, key=lambda i: pop_fitness[i])]
                p2 = population[max(c2, key=lambda i: pop_fitness[i])]
                child = crossover(p1, p2, rng=rng)

            m = _eval(child, train)
            total_evals += 1
            f = m.sharpe if m.valid else -999
            new_pop.append(child)
            new_fitness.append(f)

            if m.valid and m.sharpe > 0:
                c = cell(m)
                if c not in grid or m.sharpe > grid[c]["sharpe"]:
                    grid[c] = {
                        "tree": child,
                        "tree_str": str(child),
                        "sharpe": m.sharpe,
                        "ic": m.ic,
                        "rank_ic": m.rank_ic,
                        "icir": m.icir,
                        "turnover": m.turnover,
                        "market_corr": m.market_corr,
                        "annual_return": m.annual_return,
                        "max_drawdown": m.max_drawdown,
                    }
                    inserted += 1

        population = new_pop
        pop_fitness = new_fitness
        dur = time.monotonic() - t0

        sharpes = [g["sharpe"] for g in grid.values()]
        best = max(sharpes) if sharpes else 0
        mean = float(np.mean(sharpes)) if sharpes else 0

        gen_info = {
            "generation": gen,
            "coverage": len(grid),
            "best_sharpe": best,
            "mean_sharpe": mean,
            "pop_best": max(new_fitness),
            "inserted": inserted,
            "total_evals": total_evals,
            "duration_s": dur,
        }
        history.append(gen_info)

        print(f"[gen {gen:3d}] coverage={len(grid)}/{grid_size**2} "
              f"best={best:.2f} mean={mean:.2f} +{inserted} dt={dur:.0f}s")

        if gen % 5 == 0 or gen == args.gens:
            _save_checkpoint(output_dir, gen, grid, history, grid_size)

    # Final: evaluate top 10 on val set
    if "val" in splits:
        print(f"\nValidation (top 10 by train Sharpe):")
        top = sorted(grid.values(), key=lambda x: -x["sharpe"])[:10]
        for g in top:
            m = _eval(g["tree"], splits["val"])
            print(f"  train={g['sharpe']:.2f} val={m.sharpe:.2f} IC={m.ic:.4f} | {g['tree_str'][:60]}")

    print(f"\nDone. Coverage: {len(grid)}/{grid_size**2}, Total evals: {total_evals}")


# ── GP baseline ────────────────────────────────────────────────────────

def run_gp_baseline(args):
    raw = download_ohlcv(SP100_TICKERS[:args.n_stocks])
    splits = prepare_eval_data(raw)
    train = splits["train"]
    print(f"Data: {train['n_stocks']} stocks, {train['n_days']} days")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_runs = args.n_runs
    all_results = []
    all_trees = []
    all_signals = []

    config = {
        "algorithm": "gp_baseline",
        "pop_size": args.pop,
        "generations": args.gens,
        "n_runs": n_runs,
        "tournament": args.tournament,
        "n_stocks": train["n_stocks"],
        "n_days": train["n_days"],
        "eval": "20d_fwd_returns, normalized, topk50_drop5",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nGP baseline | pop={args.pop} gens={args.gens} runs={n_runs}")
    print(f"Output: {output_dir}\n")

    for run in range(n_runs):
        t0 = time.monotonic()
        rng = random.Random(args.seed + run)

        pop = [random_tree(max_depth=4, rng=rng) for _ in range(args.pop)]
        pop_fitness = []
        for tree in pop:
            m = _eval(tree, train)
            pop_fitness.append(m.sharpe if m.valid else -999)

        for gen in range(args.gens):
            new_pop = []
            new_fitness = []
            for _ in range(args.pop):
                if rng.random() < 0.7:
                    candidates = rng.sample(range(len(pop)), min(args.tournament, len(pop)))
                    winner = max(candidates, key=lambda i: pop_fitness[i])
                    child = mutate(pop[winner], rng=rng)
                else:
                    c1 = rng.sample(range(len(pop)), min(args.tournament, len(pop)))
                    c2 = rng.sample(range(len(pop)), min(args.tournament, len(pop)))
                    p1 = pop[max(c1, key=lambda i: pop_fitness[i])]
                    p2 = pop[max(c2, key=lambda i: pop_fitness[i])]
                    child = crossover(p1, p2, rng=rng)

                m = _eval(child, train)
                new_pop.append(child)
                new_fitness.append(m.sharpe if m.valid else -999)

            pop = new_pop
            pop_fitness = new_fitness

        best_idx = int(np.argmax(pop_fitness))
        best_tree = pop[best_idx]
        best_m = _eval(best_tree, train)
        dur = time.monotonic() - t0

        all_results.append({
            "run": run,
            "sharpe": best_m.sharpe,
            "ic": best_m.ic,
            "rank_ic": best_m.rank_ic,
            "icir": best_m.icir,
            "annual_return": best_m.annual_return,
            "turnover": best_m.turnover,
            "expression": str(best_tree),
        })
        all_trees.append(best_tree)

        # Store signals for pairwise correlation
        signals = compute_signals(best_tree, train["stock_data"], train["n_days"])
        all_signals.append(signals.flatten())

        print(f"  Run {run+1}/{n_runs}: Sharpe={best_m.sharpe:.2f} IC={best_m.ic:.4f} "
              f"ICIR={best_m.icir:.2f} dt={dur:.0f}s | {str(best_tree)[:50]}")

    # Pairwise correlation
    sig_matrix = np.array(all_signals)
    valid_cols = ~np.any(np.isnan(sig_matrix), axis=0)
    sig_clean = sig_matrix[:, valid_cols]
    if sig_clean.shape[1] > 10:
        corr = np.corrcoef(sig_clean)
        upper = corr[np.triu_indices_from(corr, k=1)]
        mean_corr = float(np.nanmean(upper))
        median_corr = float(np.nanmedian(upper))
    else:
        mean_corr = median_corr = 0.0

    unique = len(set(r["expression"] for r in all_results))

    print(f"\nMean pairwise corr: {mean_corr:.3f}, Unique: {unique}/{n_runs}")
    print(f"Best Sharpe: {max(r['sharpe'] for r in all_results):.2f}")

    summary = {
        "runs": all_results,
        "mean_pairwise_corr": mean_corr,
        "median_pairwise_corr": median_corr,
        "unique_programs": unique,
        "best_sharpe": max(r["sharpe"] for r in all_results),
        "mean_sharpe": float(np.mean([r["sharpe"] for r in all_results])),
    }
    (output_dir / "result.json").write_text(json.dumps(summary, indent=2))


# ── Random baseline ───────────────────────────────────────────────────

def run_random(args):
    raw = download_ohlcv(SP100_TICKERS[:args.n_stocks])
    splits = prepare_eval_data(raw)
    train = splits["train"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    n_samples = args.n_samples
    results = []

    print(f"\nRandom search | {n_samples} samples")
    for i in range(n_samples):
        tree = random_tree(max_depth=4, rng=rng)
        m = _eval(tree, train)
        if m.valid:
            results.append({
                "sharpe": m.sharpe, "ic": m.ic, "icir": m.icir,
                "turnover": m.turnover, "expression": str(tree),
            })
        if (i + 1) % 100 == 0:
            valid = [r for r in results if r["sharpe"] > 0]
            best = max(r["sharpe"] for r in valid) if valid else 0
            print(f"  {i+1}/{n_samples}: {len(valid)} valid, best Sharpe={best:.2f}")

    valid = [r for r in results if r["sharpe"] > 0]
    print(f"\nDone. {len(valid)}/{n_samples} valid, best={max(r['sharpe'] for r in valid):.2f}")
    (output_dir / "result.json").write_text(json.dumps({
        "n_samples": n_samples,
        "n_valid": len(valid),
        "best_sharpe": max(r["sharpe"] for r in valid) if valid else 0,
        "results": sorted(valid, key=lambda x: -x["sharpe"])[:50],
    }, indent=2))


# ── Checkpoint ─────────────────────────────────────────────────────────

def _save_checkpoint(output_dir, gen, grid, history, grid_size):
    # Serialize grid without tree objects (not JSON-safe)
    grid_serial = {}
    for k, v in grid.items():
        entry = {key: val for key, val in v.items() if key != "tree"}
        grid_serial[f"{k[0]},{k[1]}"] = entry

    ckpt = {
        "generation": gen,
        "coverage": len(grid),
        "total_cells": grid_size ** 2,
        "grid": grid_serial,
        "history": history,
    }
    (output_dir / "checkpoint.json").write_text(json.dumps(ckpt, indent=2))

    # Also pickle the full grid with trees for later val evaluation
    with open(output_dir / "grid.pkl", "wb") as f:
        pickle.dump(grid, f)


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Alpha Factory: GP + MAP-Elites")
    sub = parser.add_subparsers(dest="algorithm", required=True)

    def _shared(p):
        p.add_argument("--n-stocks", type=int, default=50)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--output", required=True)

    me_p = sub.add_parser("mapelites", help="GP + MAP-Elites")
    _shared(me_p)
    me_p.add_argument("--pop", type=int, default=100)
    me_p.add_argument("--gens", type=int, default=50)
    me_p.add_argument("--grid-size", type=int, default=20)
    me_p.add_argument("--tournament", type=int, default=7)

    gp_p = sub.add_parser("gp", help="GP baseline (multiple runs)")
    _shared(gp_p)
    gp_p.add_argument("--pop", type=int, default=100)
    gp_p.add_argument("--gens", type=int, default=30)
    gp_p.add_argument("--n-runs", type=int, default=10)
    gp_p.add_argument("--tournament", type=int, default=7)

    rand_p = sub.add_parser("random", help="Random search baseline")
    _shared(rand_p)
    rand_p.add_argument("--n-samples", type=int, default=5000)

    args = parser.parse_args()

    if args.algorithm == "mapelites":
        run_mapelites(args)
    elif args.algorithm == "gp":
        run_gp_baseline(args)
    elif args.algorithm == "random":
        run_random(args)


if __name__ == "__main__":
    main()
