#!/usr/bin/env python3
"""Generate 3D archive surface plots for MAGE."""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 300})


def load_grid(checkpoint_path):
    with open(checkpoint_path) as f:
        d = json.load(f)
    return d["grid"], d.get("coverage", 0)


def make_3d_surface(grid, title, color_field="sharpe", color_label="Sharpe Ratio",
                    cmap="viridis", filename="archive_3d.png", elev=25, azim=225):
    """3D surface: x=turnover, y=market_corr, z=Sharpe, colored by color_field."""
    grid_size = 20

    # Build arrays
    xs, ys, zs, cs = [], [], [], []
    for key, cell in grid.items():
        parts = key.split(",")
        r, c_idx = int(parts[0]), int(parts[1])
        turnover = (r + 0.5) / grid_size * 0.2
        mkt_corr = (c_idx + 0.5) / grid_size
        xs.append(turnover)
        ys.append(mkt_corr)
        zs.append(cell["sharpe"])
        cs.append(cell.get(color_field, cell["sharpe"]))

    xs, ys, zs, cs = np.array(xs), np.array(ys), np.array(zs), np.array(cs)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter with vertical stems
    norm = plt.Normalize(vmin=np.percentile(cs, 5), vmax=np.percentile(cs, 95))
    colors = cm.get_cmap(cmap)(norm(cs))

    # Draw stems from z=0 to each point
    for x, y, z, color in zip(xs, ys, zs, colors):
        ax.plot([x, x], [y, y], [0, z], color=color, alpha=0.3, linewidth=0.8)

    sc = ax.scatter(xs, ys, zs, c=cs, cmap=cmap, s=40, alpha=0.9,
                    edgecolors="white", linewidth=0.3, norm=norm)

    ax.set_xlabel("Turnover", labelpad=10)
    ax.set_ylabel("Market Correlation", labelpad=10)
    ax.set_zlabel("Sharpe Ratio", labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)
    ax.view_init(elev=elev, azim=azim)

    fig.colorbar(sc, ax=ax, label=color_label, shrink=0.6, pad=0.1)
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {filename}")


def make_3d_comparison(grid1, grid2, label1, label2, filename="archive_3d_compare.png"):
    """Side-by-side 3D views of two archives."""
    grid_size = 20

    fig = plt.figure(figsize=(18, 8))

    for idx, (grid, label) in enumerate([(grid1, label1), (grid2, label2)]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')

        xs, ys, zs, ics = [], [], [], []
        for key, cell in grid.items():
            parts = key.split(",")
            r, c_idx = int(parts[0]), int(parts[1])
            xs.append((r + 0.5) / grid_size * 0.2)
            ys.append((c_idx + 0.5) / grid_size)
            zs.append(cell["sharpe"])
            ics.append(cell.get("ic", 0))

        xs, ys, zs, ics = np.array(xs), np.array(ys), np.array(zs), np.array(ics)

        for x, y, z in zip(xs, ys, zs):
            ax.plot([x, x], [y, y], [0, z], color="gray", alpha=0.15, linewidth=0.5)

        sc = ax.scatter(xs, ys, zs, c=ics, cmap="RdYlGn", s=35, alpha=0.9,
                       edgecolors="white", linewidth=0.3,
                       vmin=-0.02, vmax=0.06)

        ax.set_xlabel("Turnover", labelpad=8)
        ax.set_ylabel("Mkt Corr", labelpad=8)
        ax.set_zlabel("Sharpe", labelpad=8)
        ax.set_title(f"{label}\n({len(grid)} cells)", fontsize=13)
        ax.view_init(elev=30, azim=220)
        ax.set_zlim(0, max(zs) * 1.1)

        fig.colorbar(sc, ax=ax, label="IC", shrink=0.5, pad=0.08)

    plt.suptitle("MAGE Archive: Strict vs Loose Correlation Gate", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTDIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {filename}")


def make_3d_ic_sharpe_turnover(grid, filename="3d_ic_sharpe_turnover.png"):
    """3D scatter: x=IC, y=turnover, z=Sharpe, colored by market_corr."""
    xs, ys, zs, cs = [], [], [], []
    for cell in grid.values():
        xs.append(cell.get("ic", 0))
        ys.append(cell.get("turnover", 0))
        zs.append(cell["sharpe"])
        cs.append(cell.get("market_corr", 0))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(xs, ys, zs, c=cs, cmap="plasma", s=40, alpha=0.8,
                   edgecolors="white", linewidth=0.3)

    ax.set_xlabel("IC", labelpad=10)
    ax.set_ylabel("Turnover", labelpad=10)
    ax.set_zlabel("Sharpe Ratio", labelpad=10)
    ax.set_title("Archive Alphas: IC, Turnover, Sharpe, Market Correlation", fontsize=13)
    ax.view_init(elev=20, azim=135)

    fig.colorbar(sc, ax=ax, label="Market Correlation", shrink=0.6, pad=0.1)
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {filename}")


if __name__ == "__main__":
    print("Generating 3D figures...")

    # Load grids
    grid_070, _ = load_grid(BASE / "results/mage_sp500_v2/checkpoint.json")
    grid_090, _ = load_grid(BASE / "results/mage_gate_090/checkpoint.json")

    # 3D archive surface (gate=0.70, colored by Sharpe)
    make_3d_surface(grid_070,
                    "MAGE Archive Landscape (gate=0.70, 221 cells)",
                    color_field="sharpe", color_label="Sharpe Ratio",
                    cmap="viridis", filename="archive_3d.png")

    # 3D archive colored by IC
    make_3d_surface(grid_070,
                    "MAGE Archive: Sharpe Height, IC Color (gate=0.70)",
                    color_field="ic", color_label="IC",
                    cmap="RdYlGn", filename="archive_3d_ic.png",
                    elev=30, azim=240)

    # Side-by-side 3D comparison: strict vs loose gate
    make_3d_comparison(grid_070, grid_090,
                       "Gate = 0.70 (strict)", "Gate = 0.90 (loose)")

    # 3D IC-Sharpe-Turnover scatter
    make_3d_ic_sharpe_turnover(grid_070)

    print(f"\nAll 3D figures saved to {OUTDIR}")
