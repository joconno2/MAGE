#!/bin/bash
# ──────────────────────────────────────────────────────────────
# MAGE CSI300/CSI500 Experiments
#
# One-command setup and run for Chinese equity experiments.
# Directly comparable to AlphaGen (KDD 2023) and AlphaForge (AAAI 2025).
#
# Usage:
#   ./experiments/run_csi.sh setup                # Install qlib + download data
#   ./experiments/run_csi.sh mapelites             # Run MAGE on CSI300
#   ./experiments/run_csi.sh gp                    # Run GP baseline on CSI300
#   ./experiments/run_csi.sh all                   # Run both + test eval
#   ./experiments/run_csi.sh all --market csi500   # Use CSI500 instead
#
# Requirements: Python 3.10+, ~5GB disk for Qlib data
# ──────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

MARKET="${MARKET:-csi300}"
POP="${POP:-200}"
GENS="${GENS:-100}"
SEED="${SEED:-42}"
GATE="${GATE:-0.70}"
VENV=".venv"

# Parse flags
CMD="${1:-help}"
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --market) MARKET="$2"; shift 2;;
        --pop) POP="$2"; shift 2;;
        --gens) GENS="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --gate) GATE="$2"; shift 2;;
        *) echo "Unknown flag: $1"; exit 1;;
    esac
done

OUTDIR="results/csi_${MARKET}"

# --- Ensure venv ---
ensure_venv() {
    if [ ! -d "$VENV" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$VENV"
        "$VENV/bin/pip" install --upgrade pip
        "$VENV/bin/pip" install -r requirements.txt
    fi
}

PY="$VENV/bin/python"

case "$CMD" in
setup)
    echo "=== Setting up Qlib for CSI experiments ==="
    ensure_venv
    "$VENV/bin/pip" install qlib 2>/dev/null || "$VENV/bin/pip" install pyqlib

    echo "Downloading Qlib CSI data (~5GB)..."
    mkdir -p data/qlib
    "$PY" -m qlib.run.get_data qlib_data --target_dir data/qlib/cn_data --region cn

    echo ""
    echo "Done. Data at data/qlib/cn_data/"
    echo "Next: $0 all --market $MARKET"
    ;;

mapelites)
    ensure_venv
    echo "=== MAGE MAP-Elites on $MARKET ==="
    echo "  pop=$POP gens=$GENS seed=$SEED gate=$GATE"
    "$PY" experiments/run_qlib.py mapelites \
        --market "$MARKET" \
        --pop "$POP" \
        --gens "$GENS" \
        --seed "$SEED" \
        --corr-threshold "$GATE" \
        --output "${OUTDIR}_mapelites"
    ;;

gp)
    ensure_venv
    echo "=== GP Baseline on $MARKET ==="
    "$PY" experiments/run_qlib.py gp \
        --market "$MARKET" \
        --pop "$POP" \
        --gens 100 \
        --n-runs 10 \
        --seed "$SEED" \
        --output "${OUTDIR}_gp"
    ;;

all)
    ensure_venv
    echo "=== Full CSI experiment suite on $MARKET ==="
    echo "  pop=$POP gens=$GENS seed=$SEED gate=$GATE"
    echo

    # Check qlib is installed
    "$PY" -c "import qlib" 2>/dev/null || {
        echo "Qlib not installed. Run: $0 setup"
        exit 1
    }

    # MAP-Elites
    echo "--- Step 1/2: MAGE MAP-Elites ---"
    "$PY" experiments/run_qlib.py mapelites \
        --market "$MARKET" \
        --pop "$POP" \
        --gens "$GENS" \
        --seed "$SEED" \
        --corr-threshold "$GATE" \
        --output "${OUTDIR}_mapelites"

    echo
    echo "--- Step 2/2: GP Baseline ---"
    "$PY" experiments/run_qlib.py gp \
        --market "$MARKET" \
        --pop "$POP" \
        --gens 100 \
        --n-runs 10 \
        --seed "$SEED" \
        --output "${OUTDIR}_gp"

    echo
    echo "============================================"
    echo "  Done. Results:"
    echo "    MAP-Elites:   ${OUTDIR}_mapelites/"
    echo "    GP baseline:  ${OUTDIR}_gp/"
    echo "============================================"
    ;;

help|*)
    cat <<EOF
MAGE CSI Experiments

Usage:
  $0 setup                     Install qlib + download CSI data (~5GB)
  $0 mapelites [flags]         Run MAGE on CSI300 (or CSI500)
  $0 gp [flags]                Run GP baseline
  $0 all [flags]               Run both MAP-Elites + GP baseline

Flags:
  --market csi300|csi500       Market universe (default: csi300)
  --pop N                      Population size (default: 200)
  --gens N                     Generations (default: 100)
  --seed N                     Random seed (default: 42)
  --gate F                     Correlation gate threshold (default: 0.70)

Steps:
  1. $0 setup
  2. $0 all --market csi300
  3. $0 all --market csi500
EOF
    ;;
esac
