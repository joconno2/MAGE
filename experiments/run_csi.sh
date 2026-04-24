#!/bin/bash
# ──────────────────────────────────────────────────────────────
# MAGE CSI300/CSI500 Experiments
#
# One-command setup and run for Chinese equity experiments.
# Directly comparable to AlphaGen (KDD 2023) and AlphaForge (AAAI 2025).
#
# Usage:
#   ./experiments/run_csi.sh setup          # Install qlib + download data
#   ./experiments/run_csi.sh mapelites      # Run MAGE on CSI300
#   ./experiments/run_csi.sh gp             # Run GP baseline on CSI300
#   ./experiments/run_csi.sh all            # Run both + eval
#   ./experiments/run_csi.sh all --market csi500  # Use CSI500 instead
#
# Requirements: Python 3.10+, pip, ~5GB disk for Qlib data
# ──────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

MARKET="${MARKET:-csi300}"
POP="${POP:-100}"
GENS="${GENS:-50}"
SEED="${SEED:-42}"
GATE="${GATE:-0.70}"

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

case "$CMD" in
setup)
    echo "=== Setting up Qlib for CSI experiments ==="
    pip install qlib pyqlib 2>/dev/null || pip install qlib

    echo "Downloading Qlib CSI data..."
    mkdir -p data/qlib
    python -m qlib.run.get_data qlib_data --target_dir data/qlib/cn_data --region cn

    echo "Done. Data at data/qlib/cn_data/"
    echo "Run: $0 all --market $MARKET"
    ;;

mapelites)
    echo "=== MAGE MAP-Elites on $MARKET ==="
    echo "  pop=$POP gens=$GENS seed=$SEED gate=$GATE"
    python experiments/run_qlib.py mapelites \
        --market "$MARKET" \
        --pop "$POP" \
        --gens "$GENS" \
        --seed "$SEED" \
        --output "${OUTDIR}_mapelites"
    ;;

gp)
    echo "=== GP Baseline on $MARKET ==="
    python experiments/run_qlib.py gp \
        --market "$MARKET" \
        --pop "$POP" \
        --gens 30 \
        --n-runs 10 \
        --seed "$SEED" \
        --output "${OUTDIR}_gp"
    ;;

all)
    echo "=== Full CSI experiment suite on $MARKET ==="
    echo "  pop=$POP gens=$GENS seed=$SEED gate=$GATE"
    echo

    # Check qlib is installed
    python -c "import qlib" 2>/dev/null || {
        echo "Qlib not installed. Run: $0 setup"
        exit 1
    }

    # MAP-Elites
    echo "--- MAGE MAP-Elites ---"
    python experiments/run_qlib.py mapelites \
        --market "$MARKET" \
        --pop "$POP" \
        --gens "$GENS" \
        --seed "$SEED" \
        --output "${OUTDIR}_mapelites"

    echo
    echo "--- GP Baseline ---"
    python experiments/run_qlib.py gp \
        --market "$MARKET" \
        --pop "$POP" \
        --gens 30 \
        --n-runs 10 \
        --seed "$SEED" \
        --output "${OUTDIR}_gp"

    echo
    echo "=== Done. Results in ${OUTDIR}_mapelites/ and ${OUTDIR}_gp/ ==="
    ;;

help|*)
    cat <<EOF
MAGE CSI Experiments

Usage:
  $0 setup                     Install qlib + download CSI data (~5GB)
  $0 mapelites [flags]         Run MAGE on CSI300 (or CSI500)
  $0 gp [flags]                Run GP baseline
  $0 all [flags]               Run both + test eval

Flags:
  --market csi300|csi500       Market universe (default: csi300)
  --pop N                      Population size (default: 100)
  --gens N                     Generations (default: 50)
  --seed N                     Random seed (default: 42)
  --gate F                     Correlation gate threshold (default: 0.70)

Examples:
  $0 setup
  $0 all --market csi300 --pop 200 --gens 100
  $0 mapelites --market csi500 --seed 1
EOF
    ;;
esac
