#!/usr/bin/env python3
"""
Alpha Factory on Ray cluster.

Ships code via runtime_env, data via object store. Each worker evaluates
one GP tree on the full stock universe (~2s per eval). 600 CPUs = 300 evals/s.

Usage:
    python experiments/run_cluster.py mapelites --output results/cluster_mapelites
    python experiments/run_cluster.py gp --output results/cluster_gp
    python experiments/run_cluster.py random --output results/cluster_random
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
    Node, random_tree, mutate, crossover, compute_signals,
)
from alpha_factory.evaluate import evaluate_signals, AlphaMetrics


def _make_eval_fn():
    import ray

    @ray.remote(num_cpus=1)
    def eval_tree_remote(tree_bytes, stock_data, close, fwd_1d, fwd_20d, n_days):
        import pickle, sys, os
        from pathlib import Path
        try:
            # Add deployed code to path
            deploy = str(Path.home() / "alpha-factory-deploy")
            if deploy not in sys.path:
                sys.path.insert(0, deploy)
            tree = pickle.loads(tree_bytes)
            from alpha_factory.gp_genome import compute_signals
            from alpha_factory.evaluate import evaluate_signals
            signals = compute_signals(tree, stock_data, n_days)
            m = evaluate_signals(signals, close, fwd_1d, fwd_20d, expression=str(tree))
            return {
                "sharpe": m.sharpe, "ic": m.ic, "rank_ic": m.rank_ic,
                "icir": m.icir, "turnover": m.turnover, "market_corr": m.market_corr,
                "annual_return": m.annual_return, "max_drawdown": m.max_drawdown,
                "expression": m.expression, "valid": m.valid,
            }
        except Exception as e:
            return {"sharpe": -999, "ic": 0, "rank_ic": 0, "icir": 0,
                    "turnover": 0, "market_corr": 0, "annual_return": 0,
                    "max_drawdown": 0, "expression": str(e), "valid": False}

    return eval_tree_remote


def _eval_batch(trees, eval_fn, stock_ref, close_ref, fwd1d_ref, fwd20d_ref, n_days):
    """Evaluate a batch of trees in parallel on Ray."""
    import ray
    refs = [
        eval_fn.remote(pickle.dumps(t), stock_ref, close_ref, fwd1d_ref, fwd20d_ref, n_days)
        for t in trees
    ]
    return ray.get(refs)


def run_mapelites(args):
    import ray
    # Data
    raw = download_ohlcv(SP100_TICKERS[:args.n_stocks])
    splits = prepare_eval_data(raw)
    train = splits["train"]
    print(f"Data: {train['n_stocks']} stocks, {train['n_days']} days")

    # Connect
    if args.ray_address:
        ray.init(address=args.ray_address, namespace="alpha-factory")
    else:
        from aall_cluster import connect
        state, tunnel = connect(namespace="alpha-factory", verbose=True)
    print(f"Cluster: {ray.cluster_resources().get('CPU', 0)} CPUs")

    eval_fn = _make_eval_fn()

    # Put data in object store
    stock_ref = ray.put(train["stock_data"])
    close_ref = ray.put(train["close_prices"])
    fwd1d_ref = ray.put(train["fwd_returns_1d"])
    fwd20d_ref = ray.put(train["fwd_returns_20d"])
    n_days = train["n_days"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    grid_size = args.grid_size
    grid = {}

    def cell(r):
        cx = min(grid_size - 1, max(0, int(r["turnover"] * grid_size * 5)))
        cy = min(grid_size - 1, max(0, int(r["market_corr"] * grid_size)))
        return (cx, cy)

    pop_size = args.pop
    population = [random_tree(max_depth=4, rng=rng) for _ in range(pop_size)]
    pop_fitness = [-999.0] * pop_size

    config = {"algorithm": "gp_mapelites_cluster", "pop_size": pop_size,
              "generations": args.gens, "grid_size": grid_size,
              "n_stocks": train["n_stocks"], "n_days": n_days}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nGP+MAP-Elites on cluster | pop={pop_size} gens={args.gens} grid={grid_size}x{grid_size}")
    print(f"Output: {output_dir}\n")

    # Init: parallel eval of all initial trees
    print(f"Initializing ({pop_size} trees in parallel)...")
    t0 = time.monotonic()
    init_results = _eval_batch(population, eval_fn, stock_ref, close_ref, fwd1d_ref, fwd20d_ref, n_days)
    print(f"Init eval: {time.monotonic()-t0:.0f}s")

    for i, (tree, r) in enumerate(zip(population, init_results)):
        pop_fitness[i] = r["sharpe"] if r["valid"] else -999
        if r["valid"] and r["sharpe"] > 0:
            c = cell(r)
            if c not in grid or r["sharpe"] > grid[c]["sharpe"]:
                grid[c] = {"tree": tree, **r}

    print(f"Initial coverage: {len(grid)}/{grid_size**2}")

    history = []
    total_evals = pop_size

    for gen in range(1, args.gens + 1):
        t0 = time.monotonic()

        # Build batch of children
        children = []
        for _ in range(pop_size):
            src = rng.random()
            if src < 0.3 and grid:
                elite = rng.choice(list(grid.values()))
                children.append(mutate(elite["tree"], rng=rng))
            elif src < 0.7:
                candidates = rng.sample(range(len(population)), min(args.tournament, len(population)))
                winner = max(candidates, key=lambda i: pop_fitness[i])
                children.append(mutate(population[winner], rng=rng))
            else:
                c1 = rng.sample(range(len(population)), min(args.tournament, len(population)))
                c2 = rng.sample(range(len(population)), min(args.tournament, len(population)))
                p1 = population[max(c1, key=lambda i: pop_fitness[i])]
                p2 = population[max(c2, key=lambda i: pop_fitness[i])]
                children.append(crossover(p1, p2, rng=rng))

        # Parallel eval on cluster
        results = _eval_batch(children, eval_fn, stock_ref, close_ref, fwd1d_ref, fwd20d_ref, n_days)
        total_evals += pop_size

        # Update population and archive
        inserted = 0
        new_fitness = []
        for tree, r in zip(children, results):
            f = r["sharpe"] if r["valid"] else -999
            new_fitness.append(f)
            if r["valid"] and r["sharpe"] > 0:
                c = cell(r)
                if c not in grid or r["sharpe"] > grid[c]["sharpe"]:
                    grid[c] = {"tree": tree, **r}
                    inserted += 1

        population = children
        pop_fitness = new_fitness

        dur = time.monotonic() - t0
        sharpes = [g["sharpe"] for g in grid.values()]
        best = max(sharpes) if sharpes else 0
        mean = float(np.mean(sharpes)) if sharpes else 0

        history.append({"generation": gen, "coverage": len(grid),
                       "best_sharpe": best, "mean_sharpe": mean,
                       "inserted": inserted, "total_evals": total_evals,
                       "duration_s": dur})

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

    # Val eval top 10
    if "val" in splits:
        val = splits["val"]
        val_stock_ref = ray.put(val["stock_data"])
        val_close_ref = ray.put(val["close_prices"])
        val_fwd1d_ref = ray.put(val["fwd_returns_1d"])
        val_fwd20d_ref = ray.put(val["fwd_returns_20d"])

        top = sorted(grid.values(), key=lambda x: -x["sharpe"])[:10]
        print(f"\nValidation (top 10):")
        val_results = _eval_batch(
            [g["tree"] for g in top], eval_fn,
            val_stock_ref, val_close_ref, val_fwd1d_ref, val_fwd20d_ref, val["n_days"],
        )
        for g, vr in zip(top, val_results):
            print(f"  train={g['sharpe']:.2f} val={vr['sharpe']:.2f} IC={vr['ic']:.4f} | {g['expression'][:60]}")

    print(f"\nDone. Coverage: {len(grid)}/{grid_size**2}, Total evals: {total_evals}")
    ray.shutdown()


def run_gp(args):
    import ray
    raw = download_ohlcv(SP100_TICKERS[:args.n_stocks])
    splits = prepare_eval_data(raw)
    train = splits["train"]
    print(f"Data: {train['n_stocks']} stocks, {train['n_days']} days")

    if args.ray_address:
        ray.init(address=args.ray_address, namespace="alpha-factory")
    else:
        from aall_cluster import connect
        state, tunnel = connect(namespace="alpha-factory", verbose=True)

    eval_fn = _make_eval_fn()
    stock_ref = ray.put(train["stock_data"])
    close_ref = ray.put(train["close_prices"])
    fwd1d_ref = ray.put(train["fwd_returns_1d"])
    fwd20d_ref = ray.put(train["fwd_returns_20d"])
    n_days = train["n_days"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_signals = []

    config = {"algorithm": "gp_baseline_cluster", "pop_size": args.pop,
              "generations": args.gens, "n_runs": args.n_runs}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nGP baseline on cluster | pop={args.pop} gens={args.gens} runs={args.n_runs}")

    for run in range(args.n_runs):
        t0 = time.monotonic()
        rng = random.Random(args.seed + run)

        pop = [random_tree(max_depth=4, rng=rng) for _ in range(args.pop)]
        results = _eval_batch(pop, eval_fn, stock_ref, close_ref, fwd1d_ref, fwd20d_ref, n_days)
        pop_fitness = [r["sharpe"] if r["valid"] else -999 for r in results]

        for gen in range(args.gens):
            children = []
            for _ in range(args.pop):
                if rng.random() < 0.7:
                    cands = rng.sample(range(len(pop)), min(args.tournament, len(pop)))
                    winner = max(cands, key=lambda i: pop_fitness[i])
                    children.append(mutate(pop[winner], rng=rng))
                else:
                    c1 = rng.sample(range(len(pop)), min(args.tournament, len(pop)))
                    c2 = rng.sample(range(len(pop)), min(args.tournament, len(pop)))
                    p1 = pop[max(c1, key=lambda i: pop_fitness[i])]
                    p2 = pop[max(c2, key=lambda i: pop_fitness[i])]
                    children.append(crossover(p1, p2, rng=rng))

            results = _eval_batch(children, eval_fn, stock_ref, close_ref, fwd1d_ref, fwd20d_ref, n_days)
            pop = children
            pop_fitness = [r["sharpe"] if r["valid"] else -999 for r in results]

        best_idx = int(np.argmax(pop_fitness))
        best_r = results[best_idx]
        dur = time.monotonic() - t0

        all_results.append(best_r)

        # Store flattened signals for pairwise correlation
        signals = compute_signals(pop[best_idx], train["stock_data"], n_days)
        all_signals.append(signals.flatten())

        print(f"  Run {run+1}/{args.n_runs}: Sharpe={best_r['sharpe']:.2f} IC={best_r['ic']:.4f} "
              f"ICIR={best_r['icir']:.2f} dt={dur:.0f}s | {best_r['expression'][:50]}")

    # Pairwise correlation
    sig_matrix = np.array(all_signals)
    valid_cols = ~np.any(np.isnan(sig_matrix), axis=0)
    sig_clean = sig_matrix[:, valid_cols]
    if sig_clean.shape[1] > 10:
        corr = np.corrcoef(sig_clean)
        upper = corr[np.triu_indices_from(corr, k=1)]
        mean_corr = float(np.nanmean(upper))
    else:
        mean_corr = 0.0

    unique = len(set(r["expression"] for r in all_results))
    print(f"\nMean pairwise corr: {mean_corr:.3f}, Unique: {unique}/{args.n_runs}")
    print(f"Best Sharpe: {max(r['sharpe'] for r in all_results):.2f}")

    summary = {"runs": all_results, "mean_pairwise_corr": mean_corr,
               "unique_programs": unique,
               "best_sharpe": max(r["sharpe"] for r in all_results)}
    (output_dir / "result.json").write_text(json.dumps(summary, indent=2))
    ray.shutdown()


def run_random(args):
    import ray
    raw = download_ohlcv(SP100_TICKERS[:args.n_stocks])
    splits = prepare_eval_data(raw)
    train = splits["train"]

    if args.ray_address:
        ray.init(address=args.ray_address, namespace="alpha-factory")
    else:
        from aall_cluster import connect
        state, tunnel = connect(namespace="alpha-factory", verbose=True)

    eval_fn = _make_eval_fn()
    stock_ref = ray.put(train["stock_data"])
    close_ref = ray.put(train["close_prices"])
    fwd1d_ref = ray.put(train["fwd_returns_1d"])
    fwd20d_ref = ray.put(train["fwd_returns_20d"])
    n_days = train["n_days"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    n = args.n_samples
    batch_size = 200
    all_valid = []

    print(f"\nRandom search on cluster | {n} samples, batch={batch_size}")

    for i in range(0, n, batch_size):
        batch = [random_tree(max_depth=4, rng=rng) for _ in range(min(batch_size, n - i))]
        results = _eval_batch(batch, eval_fn, stock_ref, close_ref, fwd1d_ref, fwd20d_ref, n_days)

        for r in results:
            if r["valid"] and r["sharpe"] > 0:
                all_valid.append(r)

        done = min(i + batch_size, n)
        best = max(r["sharpe"] for r in all_valid) if all_valid else 0
        print(f"  {done}/{n}: {len(all_valid)} valid, best Sharpe={best:.2f}")

    print(f"\nDone. {len(all_valid)}/{n} valid, best={max(r['sharpe'] for r in all_valid):.2f}")
    (output_dir / "result.json").write_text(json.dumps({
        "n_samples": n, "n_valid": len(all_valid),
        "best_sharpe": max(r["sharpe"] for r in all_valid) if all_valid else 0,
        "results": sorted(all_valid, key=lambda x: -x["sharpe"])[:50],
    }, indent=2))
    ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Alpha Factory on Ray cluster")
    sub = parser.add_subparsers(dest="algorithm", required=True)

    def _shared(p):
        p.add_argument("--ray-address", default=None, help="Ray address (e.g. 'auto')")
        p.add_argument("--n-stocks", type=int, default=50)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--output", required=True)

    me_p = sub.add_parser("mapelites")
    _shared(me_p)
    me_p.add_argument("--pop", type=int, default=200)
    me_p.add_argument("--gens", type=int, default=100)
    me_p.add_argument("--grid-size", type=int, default=20)
    me_p.add_argument("--tournament", type=int, default=7)

    gp_p = sub.add_parser("gp")
    _shared(gp_p)
    gp_p.add_argument("--pop", type=int, default=200)
    gp_p.add_argument("--gens", type=int, default=50)
    gp_p.add_argument("--n-runs", type=int, default=10)
    gp_p.add_argument("--tournament", type=int, default=7)

    rand_p = sub.add_parser("random")
    _shared(rand_p)
    rand_p.add_argument("--n-samples", type=int, default=20000)

    args = parser.parse_args()

    if args.algorithm == "mapelites":
        run_mapelites(args)
    elif args.algorithm == "gp":
        run_gp(args)
    elif args.algorithm == "random":
        run_random(args)


if __name__ == "__main__":
    main()
