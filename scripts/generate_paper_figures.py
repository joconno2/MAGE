#!/usr/bin/env python3
"""Generate compound paper figures from MAGE sweep results."""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "figure.dpi": 300})

# ── Load all sweep data ──────────────────────────────────────────────

def load_checkpoint(path):
    with open(path) as f:
        return json.load(f)

def load_test_eval(path):
    """Load test eval results if available."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Gate sweep results
sweep = {}
sweep_configs = [
    (0.60, "mage_gate_060", "test_gate_060"),
    (0.65, "mage_gate_065", "test_gate_065"),
    (0.70, "mage_sp500_v2", "test_eval_sp500_v2"),
    (0.75, "mage_gate_075", None),
    (0.80, "mage_gate_080", None),
    (0.85, "mage_gate_085", None),
    (0.90, "mage_gate_090", "test_gate_090"),
    (0.95, "mage_gate_095", None),
]

for thresh, run_name, test_name in sweep_configs:
    ckpt_path = RESULTS / run_name / "checkpoint.json"
    if not ckpt_path.exists():
        continue
    d = load_checkpoint(ckpt_path)
    sharpes = [v["sharpe"] for v in d["grid"].values()]
    corrs = [v.get("max_corr", 0) for v in d["grid"].values()]

    test_data = None
    if test_name:
        test_path = RESULTS / test_name / "mapelites_test.json"
        test_data = load_test_eval(test_path)

    sweep[thresh] = {
        "coverage": d["coverage"],
        "generation": d["generation"],
        "best_sharpe": max(sharpes),
        "mean_sharpe": np.mean(sharpes),
        "median_corr": np.median(corrs),
        "mean_corr": np.mean(corrs),
        "test_data": test_data,
        "grid": d["grid"],
    }

print(f"Loaded {len(sweep)} sweep points: {sorted(sweep.keys())}")

# Best config for detailed figures
best_thresh = 0.70
best_grid = sweep[best_thresh]["grid"] if best_thresh in sweep else None


# ── Figure 1: Gate Sweep Summary (4-panel) ────────────────────────────

def fig_gate_sweep():
    thresholds = sorted(sweep.keys())
    if len(thresholds) < 3:
        print("  Skipping gate sweep (need 3+ points)")
        return

    covs = [sweep[t]["coverage"] for t in thresholds]
    bests = [sweep[t]["best_sharpe"] for t in thresholds]
    means = [sweep[t]["mean_sharpe"] for t in thresholds]
    corrs = [sweep[t]["median_corr"] for t in thresholds]

    # Test Sharpe where available
    test_thresholds = []
    test_combined = []
    test_best = []
    for t in thresholds:
        td = sweep[t].get("test_data")
        if td and len(td) >= 5:
            test_thresholds.append(t)
            test_sharpes = [r.get("test_sharpe", 0) for r in td[:20]]
            test_combined.append(np.mean(test_sharpes))
            test_best.append(max(test_sharpes))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("MAGE: Correlation Gate Threshold Sweep", fontsize=16, y=0.98)

    # Coverage
    ax = axes[0, 0]
    ax.plot(thresholds, covs, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.set_xlabel("Correlation Gate Threshold")
    ax.set_ylabel("Archive Coverage (cells)")
    ax.set_title("Coverage")

    # Train Sharpe
    ax = axes[0, 1]
    ax.plot(thresholds, bests, "s-", color="#4CAF50", linewidth=2, markersize=8, label="Best")
    ax.plot(thresholds, means, "^-", color="#8BC34A", linewidth=2, markersize=8, label="Mean")
    if test_thresholds:
        ax.plot(test_thresholds, test_best, "D--", color="#F44336", linewidth=2, markersize=8, label="Test Best")
        ax.plot(test_thresholds, test_combined, "v--", color="#FF9800", linewidth=2, markersize=8, label="Test Mean (top 20)")
    ax.set_xlabel("Correlation Gate Threshold")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Performance")
    ax.legend(fontsize=10)

    # Pairwise correlation
    ax = axes[1, 0]
    ax.plot(thresholds, corrs, "o-", color="#9C27B0", linewidth=2, markersize=8)
    ax.set_xlabel("Correlation Gate Threshold")
    ax.set_ylabel("Median Archive Pairwise Corr")
    ax.set_title("Archive Diversity")

    # Tradeoff: test Sharpe vs pairwise corr
    ax = axes[1, 1]
    if test_thresholds:
        for t in test_thresholds:
            tc = [r.get("test_sharpe", 0) for r in sweep[t]["test_data"][:20]]
            y = np.mean(tc)
            x = sweep[t]["median_corr"]
            ax.scatter(x, y, s=120, zorder=5)
            ax.annotate(f"{t:.2f}", (x, y), textcoords="offset points",
                       xytext=(8, 5), fontsize=11)
        ax.set_xlabel("Median Archive Pairwise Corr")
        ax.set_ylabel("Mean Test Sharpe (top 20)")
        ax.set_title("Diversity-Generalization Tradeoff")
    else:
        ax.text(0.5, 0.5, "Test evals\nnot available", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
        ax.set_title("Diversity-Generalization Tradeoff")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTDIR / "gate_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  gate_sweep.png")


# ── Figure 2: Train vs Test Sharpe per Alpha ──────────────────────────

def fig_train_vs_test():
    # Use gate=0.70 (best test) and gate=0.90 (best train)
    configs = []
    for thresh, test_name in [(0.70, "test_eval_sp500_v2"), (0.90, "test_gate_090")]:
        test_path = RESULTS / test_name / "mapelites_test.json"
        if test_path.exists():
            td = load_test_eval(test_path)
            if td:
                configs.append((thresh, td))

    if not configs:
        print("  Skipping train_vs_test (no test data)")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {0.70: "#2196F3", 0.90: "#F44336"}
    labels = {0.70: "Gate = 0.70 (strict)", 0.90: "Gate = 0.90 (loose)"}

    for thresh, td in configs:
        trains = [r.get("train_sharpe", 0) for r in td[:20]]
        tests = [r.get("test_sharpe", 0) for r in td[:20]]
        ax.scatter(trains, tests, s=80, alpha=0.7, c=colors[thresh],
                  label=labels[thresh], edgecolors="white", linewidth=0.5)

    # Diagonal line (no degradation)
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, linewidth=1)
    ax.axhline(0, color="gray", alpha=0.3, linewidth=1)
    ax.set_xlabel("Train Sharpe (2010-2019)")
    ax.set_ylabel("Test Sharpe (2021-2022)")
    ax.set_title("Per-Alpha Generalization: Train vs Test Sharpe")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTDIR / "train_vs_test.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  train_vs_test.png")


# ── Figure 3: Archive Composition ─────────────────────────────────────

def fig_archive_composition():
    if best_grid is None:
        print("  Skipping archive composition (no grid)")
        return

    # Count feature usage in expressions
    features = ["upper_shadow", "lower_shadow", "body", "gap", "returns",
                "log_return", "dollar_volume", "turnover_ratio", "intraday_range",
                "vwap", "close", "open", "high", "low", "volume"]
    feature_counts = {f: 0 for f in features}

    # Count operator usage
    operators = {}

    for cell in best_grid.values():
        expr = cell.get("expression", cell.get("tree_str", ""))
        for f in features:
            if f in expr:
                feature_counts[f] += 1
        # Root operator
        root = expr.split("(")[0] if "(" in expr else expr
        operators[root] = operators.get(root, 0) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Feature usage
    sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])
    sorted_features = [(f, c) for f, c in sorted_features if c > 0]
    names = [f for f, _ in sorted_features]
    counts = [c for _, c in sorted_features]

    bars = ax1.barh(range(len(names)), counts, color="#4CAF50", alpha=0.8)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel("Number of Archive Alphas Using Feature")
    ax1.set_title("Feature Usage Across Archive")
    ax1.invert_yaxis()

    # Operator usage (top 15)
    sorted_ops = sorted(operators.items(), key=lambda x: -x[1])[:15]
    op_names = [o for o, _ in sorted_ops]
    op_counts = [c for _, c in sorted_ops]

    ax2.barh(range(len(op_names)), op_counts, color="#2196F3", alpha=0.8)
    ax2.set_yticks(range(len(op_names)))
    ax2.set_yticklabels(op_names, fontsize=10)
    ax2.set_xlabel("Number of Archive Alphas")
    ax2.set_title("Root Operator Distribution")
    ax2.invert_yaxis()

    plt.suptitle(f"Archive Composition (gate=0.70, {len(best_grid)} cells)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTDIR / "archive_composition.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  archive_composition.png")


# ── Figure 4: Heatmap with Test Sharpe overlay ────────────────────────

def fig_heatmap_test():
    """MAP-Elites grid colored by test Sharpe instead of train Sharpe."""
    test_path = RESULTS / "test_eval_sp500_v2" / "mapelites_test.json"
    if not test_path.exists() or best_grid is None:
        print("  Skipping test heatmap (no data)")
        return

    test_data = load_test_eval(test_path)
    if not test_data:
        print("  Skipping test heatmap (empty)")
        return

    # Build lookup from expression to test sharpe
    test_lookup = {}
    for r in test_data:
        expr = r.get("expression", "")[:80]
        test_lookup[expr] = r.get("test_sharpe", 0)

    grid_size = 20
    train_grid = np.full((grid_size, grid_size), np.nan)
    test_grid = np.full((grid_size, grid_size), np.nan)

    for key, cell in best_grid.items():
        parts = key.split(",")
        r, c = int(parts[0]), int(parts[1])
        train_grid[r, c] = cell["sharpe"]
        expr = cell.get("expression", cell.get("tree_str", ""))[:80]
        if expr in test_lookup:
            test_grid[r, c] = test_lookup[expr]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Train
    im1 = ax1.imshow(train_grid.T, origin="lower", aspect="auto",
                     cmap="RdYlGn", interpolation="nearest",
                     extent=[0, 0.2, 0, 1])
    ax1.set_xlabel("Turnover")
    ax1.set_ylabel("Market Correlation")
    ax1.set_title("Train Sharpe (2010-2019)")
    plt.colorbar(im1, ax=ax1, label="Sharpe Ratio")

    # Test
    im2 = ax2.imshow(test_grid.T, origin="lower", aspect="auto",
                     cmap="RdYlGn", interpolation="nearest",
                     extent=[0, 0.2, 0, 1])
    ax2.set_xlabel("Turnover")
    ax2.set_ylabel("Market Correlation")
    ax2.set_title("Test Sharpe (2021-2022)")
    plt.colorbar(im2, ax=ax2, label="Sharpe Ratio")

    plt.suptitle("MAGE Archive: Train vs Test Performance by Behavioral Cell", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTDIR / "heatmap_train_vs_test.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  heatmap_train_vs_test.png")


# ── Figure 5: IC-Sharpe relationship ─────────────────────────────────

def fig_ic_sharpe():
    """Scatter of IC vs Sharpe, colored by turnover, for train and test."""
    test_path = RESULTS / "test_eval_sp500_v2" / "mapelites_test.json"
    if not test_path.exists():
        print("  Skipping IC-Sharpe (no test data)")
        return

    test_data = load_test_eval(test_path)
    if not test_data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    trains = [r.get("train_sharpe", 0) for r in test_data[:20]]
    tests = [r.get("test_sharpe", 0) for r in test_data[:20]]
    train_ics = [r.get("train_ic", r.get("ic", 0)) for r in test_data[:20]]
    test_ics = [r.get("test_ic", 0) for r in test_data[:20]]
    turnovers = [r.get("turnover", 0) for r in test_data[:20]]

    sc1 = ax1.scatter(train_ics, trains, c=turnovers, cmap="viridis",
                     s=80, alpha=0.8, edgecolors="white", linewidth=0.5)
    ax1.set_xlabel("IC")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("Train (2010-2019)")
    plt.colorbar(sc1, ax=ax1, label="Turnover")

    sc2 = ax2.scatter(test_ics, tests, c=turnovers, cmap="viridis",
                     s=80, alpha=0.8, edgecolors="white", linewidth=0.5)
    ax2.set_xlabel("IC")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Test (2021-2022)")
    ax2.axhline(0, color="gray", alpha=0.3)
    plt.colorbar(sc2, ax=ax2, label="Turnover")

    plt.suptitle("IC vs Sharpe Relationship (top 20 alphas)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTDIR / "ic_vs_sharpe.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ic_vs_sharpe.png")


# ── Run all ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating paper figures...")
    fig_gate_sweep()
    fig_train_vs_test()
    fig_archive_composition()
    fig_heatmap_test()
    fig_ic_sharpe()

    # Also regenerate the standard figures from the best checkpoint
    best_ckpt = RESULTS / "mage_sp500_v2" / "checkpoint.json"
    if best_ckpt.exists():
        print("\nRegenerating standard figures from gate=0.70...")
        os.system(f"python3 {BASE}/scripts/generate_figures.py {best_ckpt}")

    print("\nAll figures saved to", OUTDIR)
