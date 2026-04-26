#!/usr/bin/env bash
#
# Run GP baseline (10 runs, saves tree objects) then evaluate on test set.
#
# Usage:
#   ./run_gp_baseline_and_eval.sh
#
# Takes ~2-4 hours on a modern multi-core machine (10 independent GP runs,
# pop=200, 100 gens each, S&P 500 with 467 stocks). Data downloads
# automatically on first run and is cached for the test eval step.
#
# Output:
#   results/gp_baseline_final/result.json   -- train metrics + expressions
#   results/gp_baseline_final/trees.pkl     -- pickled tree objects
#   results/test_eval_final/gp_*.json       -- test/val/train eval results

set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
GP_OUT="results/gp_baseline_final"
EVAL_OUT="results/test_eval_final"
ME_GRID="results/mage_sp500_v2/grid.pkl"

# --- Setup venv if needed ---
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r requirements.txt
fi

PY="$VENV/bin/python"

# --- Step 1: GP baseline ---
echo ""
echo "============================================"
echo "  Step 1: GP Baseline (10 runs, 100 gens)"
echo "============================================"
echo ""

"$PY" experiments/run_gp_mapelites.py gp \
    --output "$GP_OUT" \
    --pop 200 \
    --gens 100 \
    --n-runs 10 \
    --n-stocks 500 \
    --seed 42

# Sanity check
if [ ! -f "$GP_OUT/trees.pkl" ]; then
    echo "ERROR: trees.pkl not created. GP run may have failed."
    exit 1
fi
echo ""
echo "GP baseline done. Results in $GP_OUT/"

# --- Step 2: Test eval ---
echo ""
echo "============================================"
echo "  Step 2: Test Set Evaluation"
echo "============================================"
echo ""

if [ ! -f "$ME_GRID" ]; then
    echo "WARNING: MAP-Elites grid not found at $ME_GRID"
    echo "         Skipping MAP-Elites eval, running GP eval only."
    echo "         (Ask Jim for the grid.pkl if you need the full eval.)"
fi

"$PY" experiments/eval_test_set.py \
    --mapelites-grid "$ME_GRID" \
    --gp-result "$GP_OUT/result.json" \
    --n-stocks 500 \
    --output "$EVAL_OUT"

echo ""
echo "============================================"
echo "  Done. Results:"
echo "    GP baseline:  $GP_OUT/"
echo "    Test eval:    $EVAL_OUT/"
echo "============================================"
