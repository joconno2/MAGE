#!/usr/bin/env python3
"""
Evaluate saved alphas on test set (2021-2022) and compute combined-model metrics.

Loads grid.pkl from MAP-Elites run and result.json from GP baseline.
Evaluates each alpha on the test split. Builds a combined linear model
from top-N alphas and reports combined IC/Sharpe.

Usage:
    python experiments/eval_test_set.py \
        --mapelites-grid ~/alpha-factory-results/cluster_mapelites/grid.pkl \
        --gp-result ~/alpha-factory-results/cluster_gp/result.json \
        --output ~/alpha-factory-results/test_eval
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpha_factory.data import download_ohlcv, prepare_eval_data, SP100_TICKERS
from alpha_factory.gp_genome import compute_signals, evaluate_tree, random_tree
from alpha_factory.evaluate import (
    evaluate_signals, normalize_alpha, long_short_backtest,
    compute_ic_series,
)


def eval_on_split(tree, split):
    return evaluate_tree(
        tree, split["stock_data"], split["close_prices"],
        split["fwd_returns_1d"], split["fwd_returns_20d"], split["n_days"],
    )


def combined_alpha_eval(trees, split, top_n=10):
    """
    Build a combined alpha from top-N trees using equal-weight averaging
    of their normalized signals. Evaluate on the given split.
    """
    n_stocks = split["n_stocks"]
    n_days = split["n_days"]

    all_signals = []
    for tree in trees[:top_n]:
        signals = compute_signals(tree, split["stock_data"], n_days)
        normed = normalize_alpha(signals)
        all_signals.append(normed)

    # Equal-weight combination
    combined = np.nanmean(np.array(all_signals), axis=0)

    # Evaluate combined signal
    m = evaluate_signals(
        combined, split["close_prices"],
        split["fwd_returns_1d"], split["fwd_returns_20d"],
        expression=f"combined_top{top_n}",
    )
    return m


def pairwise_correlation(trees, split, top_n=20):
    """Compute pairwise signal correlation across top-N trees."""
    n_days = split["n_days"]
    signals_flat = []
    for tree in trees[:top_n]:
        sig = compute_signals(tree, split["stock_data"], n_days)
        signals_flat.append(sig.flatten())

    sig_matrix = np.array(signals_flat)
    valid_cols = ~np.any(np.isnan(sig_matrix), axis=0)
    sig_clean = sig_matrix[:, valid_cols]

    if sig_clean.shape[1] < 10:
        return 0.0, np.array([])

    corr = np.corrcoef(sig_clean)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.nanmean(upper)), upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapelites-grid", required=True)
    parser.add_argument("--gp-result", default=None)
    parser.add_argument("--n-stocks", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    raw = download_ohlcv(SP100_TICKERS[:args.n_stocks])
    splits = prepare_eval_data(raw)

    for split_name in ["train", "val", "test"]:
        if split_name in splits:
            s = splits[split_name]
            print(f"{split_name}: {s['n_stocks']} stocks, {s['n_days']} days")

    # Load MAP-Elites grid
    print(f"\nLoading MAP-Elites grid from {args.mapelites_grid}")
    with open(args.mapelites_grid, "rb") as f:
        grid = pickle.load(f)
    print(f"Grid: {len(grid)} cells")

    # Sort by train Sharpe
    elites = sorted(grid.values(), key=lambda x: -x["sharpe"])
    elite_trees = [e["tree"] for e in elites]

    # Evaluate on all splits
    for split_name in ["train", "val", "test"]:
        if split_name not in splits:
            continue
        split = splits[split_name]
        print(f"\n{'='*60}")
        print(f"MAP-Elites on {split_name} set ({split['n_days']} days)")
        print(f"{'='*60}")

        results = []
        for i, elite in enumerate(elites[:20]):
            m = eval_on_split(elite["tree"], split)
            results.append({
                "rank": i + 1,
                "train_sharpe": elite["sharpe"],
                f"{split_name}_sharpe": m.sharpe,
                f"{split_name}_ic": m.ic,
                f"{split_name}_icir": m.icir,
                "turnover": m.turnover,
                "expression": m.expression[:80],
            })
            print(f"  #{i+1}: train={elite['sharpe']:.2f} {split_name}={m.sharpe:.2f} "
                  f"IC={m.ic:.4f} ICIR={m.icir:.2f} | {m.expression[:50]}")

        # Combined alpha
        for top_n in [5, 10, 20]:
            if len(elite_trees) >= top_n:
                m = combined_alpha_eval(elite_trees, split, top_n=top_n)
                print(f"\n  Combined top-{top_n}: Sharpe={m.sharpe:.2f} IC={m.ic:.4f} ICIR={m.icir:.2f}")

        # Pairwise correlation
        mean_corr, corr_vec = pairwise_correlation(elite_trees, split, top_n=20)
        print(f"\n  Pairwise corr (top 20): mean={mean_corr:.3f}")

        (output_dir / f"mapelites_{split_name}.json").write_text(
            json.dumps(results, indent=2))

    # GP baseline evaluation
    if args.gp_result and Path(args.gp_result).exists():
        print(f"\n{'='*60}")
        print(f"GP Baseline")
        print(f"{'='*60}")

        with open(args.gp_result) as f:
            gp_data = json.load(f)

        # GP result.json has expressions but not tree objects.
        # We can't re-evaluate without the trees.
        # Report the saved metrics.
        print(f"  Runs: {len(gp_data['runs'])}")
        print(f"  Best Sharpe (train): {gp_data['best_sharpe']:.2f}")
        print(f"  Pairwise corr: {gp_data['mean_pairwise_corr']:.3f}")

        for r in gp_data["runs"]:
            print(f"  Run: Sharpe={r['sharpe']:.2f} IC={r['ic']:.4f} ICIR={r['icir']:.2f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
