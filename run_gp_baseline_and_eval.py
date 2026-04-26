#!/usr/bin/env python3
"""
Run GP baseline (10 runs, saves tree objects) then evaluate on test set.

Usage (any OS):
    python run_gp_baseline_and_eval.py

Takes ~2-4 hours on a modern multi-core machine (10 independent GP runs,
pop=200, 100 gens each, S&P 500 with 467 stocks). Data downloads
automatically on first run and is cached for the test eval step.

Output:
    results/gp_baseline_final/result.json   -- train metrics + expressions
    results/gp_baseline_final/trees.pkl     -- pickled tree objects
    results/test_eval_final/gp_*.json       -- test/val/train eval results
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

VENV_DIR = ROOT / ".venv"
GP_OUT = "results/gp_baseline_final"
EVAL_OUT = "results/test_eval_final"
ME_GRID = "results/mage_sp500_v2/grid.pkl"

# --- Locate or create venv ---

def get_python():
    """Return path to venv python, creating venv if needed."""
    if sys.platform == "win32":
        py = VENV_DIR / "Scripts" / "python.exe"
        pip = VENV_DIR / "Scripts" / "pip.exe"
    else:
        py = VENV_DIR / "bin" / "python"
        pip = VENV_DIR / "bin" / "pip"

    if not py.exists():
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        subprocess.check_call([str(pip), "install", "--upgrade", "pip"])
        subprocess.check_call([str(pip), "install", "-r", "requirements.txt"])

    return str(py)


def run(cmd):
    print(f"\n> {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: command exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    py = get_python()

    # --- Step 1: GP baseline ---
    print("")
    print("=" * 50)
    print("  Step 1: GP Baseline (10 runs, 100 gens)")
    print("=" * 50)

    run([py, "experiments/run_gp_mapelites.py", "gp",
         "--output", GP_OUT,
         "--pop", "200",
         "--gens", "100",
         "--n-runs", "10",
         "--n-stocks", "500",
         "--seed", "42"])

    trees_pkl = Path(GP_OUT) / "trees.pkl"
    if not trees_pkl.exists():
        print("ERROR: trees.pkl not created. GP run may have failed.")
        sys.exit(1)

    print(f"\nGP baseline done. Results in {GP_OUT}/")

    # --- Step 2: Test eval ---
    print("")
    print("=" * 50)
    print("  Step 2: Test Set Evaluation")
    print("=" * 50)

    if not Path(ME_GRID).exists():
        print(f"WARNING: MAP-Elites grid not found at {ME_GRID}")
        print("         Ask Jim for the grid.pkl if you need the full eval.")

    run([py, "experiments/eval_test_set.py",
         "--mapelites-grid", ME_GRID,
         "--gp-result", f"{GP_OUT}/result.json",
         "--n-stocks", "500",
         "--output", EVAL_OUT])

    print("")
    print("=" * 50)
    print(f"  Done. Results:")
    print(f"    GP baseline:  {GP_OUT}/")
    print(f"    Test eval:    {EVAL_OUT}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
