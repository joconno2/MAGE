#!/usr/bin/env python3
"""Generate publication-quality figures from MAGE results."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# --- Config ---
CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/research/alpha-factory/results/mage_full_v1/checkpoint.json"
)
OUTDIR = os.path.expanduser("~/research/alpha-factory/figures")
os.makedirs(OUTDIR, exist_ok=True)

GRID_ROWS = 20
GRID_COLS = 20
DPI = 300

# Style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": DPI,
})

# --- Load data ---
with open(CHECKPOINT) as f:
    ckpt = json.load(f)

grid = ckpt["grid"]
history = ckpt["history"]

# Parse grid into arrays
sharpes, ics, turnovers, market_corrs, tree_strs = [], [], [], [], []
for key, cell in grid.items():
    sharpes.append(cell["sharpe"])
    ics.append(cell["ic"])
    turnovers.append(cell["turnover"])
    market_corrs.append(cell["market_corr"])
    tree_strs.append(cell.get("tree_str", cell.get("expression", "")))

sharpes = np.array(sharpes)
ics = np.array(ics)
turnovers = np.array(turnovers)
market_corrs = np.array(market_corrs)


# =============================================================================
# 1. MAP-Elites Heatmap
# =============================================================================
def make_heatmap():
    heatmap = np.full((GRID_ROWS, GRID_COLS), np.nan)
    for key, cell in grid.items():
        r, c = map(int, key.split(","))
        heatmap[r, c] = cell["sharpe"]

    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#f0f0f0")

    im = ax.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )

    # Axis labels as behavioral coordinates
    xticks = np.linspace(0, GRID_COLS - 1, 5)
    xlabels = [f"{v:.2f}" for v in np.linspace(0, 1, 5)]
    yticks = np.linspace(0, GRID_ROWS - 1, 5)
    ylabels = [f"{v:.2f}" for v in np.linspace(0, 0.2, 5)]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    ax.set_xlabel("Market Correlation")
    ax.set_ylabel("Turnover")
    ax.set_title("MAGE Archive: Sharpe Ratio by Behavioral Profile")

    cbar = fig.colorbar(im, ax=ax, label="Sharpe Ratio", shrink=0.85)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "mapelites_heatmap.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("  mapelites_heatmap.png")


# =============================================================================
# 2. Convergence Plot
# =============================================================================
def make_convergence():
    gens = [h["generation"] for h in history]
    coverage = [h["coverage"] for h in history]
    best_sh = [h["best_sharpe"] for h in history]
    mean_sh = [h["mean_sharpe"] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_cov = "#4C72B0"
    color_best = "#DD8452"
    color_mean = "#55A868"

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Coverage (filled cells)", color=color_cov)
    ax1.plot(gens, coverage, color=color_cov, linewidth=2, marker="o",
             markersize=4, label="Coverage")
    ax1.tick_params(axis="y", labelcolor=color_cov)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Sharpe Ratio", color=color_best)
    ax2.plot(gens, best_sh, color=color_best, linewidth=2, marker="s",
             markersize=4, label="Best Sharpe")
    ax2.plot(gens, mean_sh, color=color_mean, linewidth=2, marker="^",
             markersize=4, label="Mean Sharpe")
    ax2.tick_params(axis="y", labelcolor=color_best)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("MAGE Convergence: Coverage and Sharpe Over Generations")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "convergence.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("  convergence.png")


# =============================================================================
# 3. IC Distribution
# =============================================================================
def make_ic_dist():
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(ics, bins=25, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1, linestyle="-", label="IC = 0")
    mean_ic = np.mean(ics)
    ax.axvline(mean_ic, color="#DD8452", linewidth=1.5, linestyle="--",
               label=f"Mean IC = {mean_ic:.4f}")

    ax.set_xlabel("Information Coefficient (IC)")
    ax.set_ylabel("Count")
    ax.set_title("Information Coefficient Distribution Across Archive")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "ic_distribution.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("  ic_distribution.png")


# =============================================================================
# 4. Sharpe Distribution
# =============================================================================
def make_sharpe_dist():
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(sharpes, bins=25, color="#55A868", edgecolor="white", alpha=0.85)
    mean_s = np.mean(sharpes)
    med_s = np.median(sharpes)
    ax.axvline(mean_s, color="#DD8452", linewidth=1.5, linestyle="--",
               label=f"Mean = {mean_s:.3f}")
    ax.axvline(med_s, color="#C44E52", linewidth=1.5, linestyle="--",
               label=f"Median = {med_s:.3f}")

    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Sharpe Ratio Distribution Across Archive")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "sharpe_distribution.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("  sharpe_distribution.png")


# =============================================================================
# 5. Turnover vs Sharpe Scatter
# =============================================================================
def make_turnover_scatter():
    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(turnovers, sharpes, c=ics, cmap="viridis", s=50,
                    edgecolors="white", linewidths=0.5, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax, label="IC")

    ax.set_xlabel("Turnover")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Turnover vs. Sharpe Ratio (colored by IC)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "turnover_vs_sharpe.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("  turnover_vs_sharpe.png")


# =============================================================================
# 6. Expression Complexity
# =============================================================================
def tree_depth(expr):
    """Estimate tree depth by max nesting of parentheses."""
    depth = 0
    max_depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
    return max_depth


def make_complexity():
    depths = [tree_depth(s) for s in tree_strs]
    depth_arr = np.array(depths)

    unique_depths = sorted(set(depths))
    counts = [np.sum(depth_arr == d) for d in unique_depths]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(unique_depths, counts, color="#8172B2", edgecolor="white",
           alpha=0.85)

    ax.set_xlabel("Expression Tree Depth (max paren nesting)")
    ax.set_ylabel("Count")
    ax.set_title("Expression Complexity Distribution")
    ax.set_xticks(unique_depths)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "expression_complexity.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("  expression_complexity.png")


# =============================================================================
# Run all
# =============================================================================
if __name__ == "__main__":
    print("Generating MAGE figures...")
    make_heatmap()
    make_convergence()
    make_ic_dist()
    make_sharpe_dist()
    make_turnover_scatter()
    make_complexity()
    print(f"Done. {len(grid)} cells, {len(history)} generations.")
    print(f"Figures saved to {OUTDIR}/")
