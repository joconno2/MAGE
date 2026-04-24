#!/usr/bin/env python3
"""
Overnight pipeline v2. Waits for sweep to finish, then runs test evals
and multi-seed experiments. Robust to tunnel drops (generous timeouts).
"""

import json
import subprocess
import time
from pathlib import Path

PYTHON = str(Path(__file__).resolve().parent.parent / ".venv/bin/python")
BASE = str(Path(__file__).resolve().parent.parent)
RESULTS = Path(BASE) / "results"

LOG = RESULTS / "overnight_v2.log"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(name, args, timeout=7200):
    log(f"START: {name}")
    try:
        r = subprocess.run([PYTHON] + args, cwd=BASE, timeout=timeout)
        log(f"DONE: {name} (exit={r.returncode})")
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT: {name} ({timeout}s)")
        return False
    except Exception as e:
        log(f"ERROR: {name}: {e}")
        return False


def wait_for_sweep():
    """Wait until no run_cluster.py mapelites processes are running."""
    log("Waiting for sweep to finish...")
    while True:
        r = subprocess.run(["pgrep", "-f", "run_cluster.py mapelites"],
                          capture_output=True)
        if r.returncode != 0:
            break
        time.sleep(60)
    log("Sweep finished (no more mapelites processes)")


def run_test_eval(name, grid_pkl, output_dir):
    if not Path(grid_pkl).exists():
        log(f"SKIP test eval {name}: no grid.pkl")
        return
    if (Path(output_dir) / "mapelites_test.json").exists():
        log(f"SKIP test eval {name}: already done")
        return
    run(f"test_eval_{name}",
        ["experiments/eval_test_set.py",
         "--mapelites-grid", grid_pkl,
         "--output", output_dir,
         "--n-stocks", "500"],
        timeout=1200)


def main():
    log("=" * 60)
    log("OVERNIGHT PIPELINE V2")
    log("=" * 60)

    # ── Phase 1: Wait for sweep ───────────────────────────────────
    wait_for_sweep()

    # ── Phase 2: Test evals on all gate sweep results ─────────────
    log("\n--- Phase 2: Test evals on sweep results ---")
    gate_configs = [
        ("060", "mage_gate_060"),
        ("065", "mage_gate_065"),
        ("070", "mage_sp500_v2"),
        ("075", "mage_gate_075"),
        ("080", "mage_gate_080"),
        ("085", "mage_gate_085"),
        ("090", "mage_gate_090"),
        ("095", "mage_gate_095"),
    ]
    for tag, run_name in gate_configs:
        grid_pkl = str(RESULTS / run_name / "grid.pkl")
        out_dir = str(RESULTS / f"test_gate_{tag}")
        run_test_eval(f"gate_{tag}", grid_pkl, out_dir)

    # ── Phase 3: MAGE multi-seed at gate=0.70 ────────────────────
    log("\n--- Phase 3: MAGE multi-seed (gate=0.70) ---")
    for seed in [1, 2, 3]:
        out = f"results/mage_seed{seed}"
        if (RESULTS / f"mage_seed{seed}" / "checkpoint.json").exists():
            log(f"SKIP MAGE seed={seed}: already done")
            continue
        run(f"MAGE seed={seed}",
            ["experiments/run_cluster.py", "mapelites",
             "--output", out,
             "--pop", "200", "--gens", "100", "--grid-size", "20",
             "--n-stocks", "500", "--seed", str(seed),
             "--corr-threshold", "0.70"],
            timeout=7200)

    # ── Phase 4: Test evals on multi-seed results ─────────────────
    log("\n--- Phase 4: Test evals on multi-seed ---")
    for seed in [1, 2, 3]:
        grid_pkl = str(RESULTS / f"mage_seed{seed}" / "grid.pkl")
        out_dir = str(RESULTS / f"test_mage_seed{seed}")
        run_test_eval(f"mage_seed{seed}", grid_pkl, out_dir)

    # ── Summary ───────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    # Gate sweep
    log("\nGate sweep (test evals):")
    for tag, run_name in gate_configs:
        test_file = RESULTS / f"test_gate_{tag}" / "mapelites_test.json"
        ckpt_file = RESULTS / run_name / "checkpoint.json"
        if test_file.exists() and ckpt_file.exists():
            td = json.load(open(test_file))
            cd = json.load(open(ckpt_file))
            test_sharpes = [r.get("test_sharpe", 0) for r in td[:20]]
            train_sharpes = [v["sharpe"] for v in cd["grid"].values()]
            pos = sum(1 for s in test_sharpes if s > 0)
            log(f"  gate=0.{tag[1:]}: cov={cd['coverage']} "
                f"train_best={max(train_sharpes):.2f} "
                f"test_mean={sum(test_sharpes)/len(test_sharpes):.2f} "
                f"test_best={max(test_sharpes):.2f} "
                f"{pos}/20 positive")
        elif ckpt_file.exists():
            cd = json.load(open(ckpt_file))
            log(f"  gate=0.{tag[1:]}: cov={cd['coverage']} gen={cd['generation']} (no test eval)")
        else:
            log(f"  gate=0.{tag[1:]}: missing")

    # Multi-seed
    log("\nMulti-seed (gate=0.70):")
    for seed in [42, 1, 2, 3]:
        if seed == 42:
            run_name = "mage_sp500_v2"
            test_name = "test_eval_sp500_v2"
        else:
            run_name = f"mage_seed{seed}"
            test_name = f"test_mage_seed{seed}"

        ckpt = RESULTS / run_name / "checkpoint.json"
        test_file = RESULTS / test_name / "mapelites_test.json"
        if ckpt.exists():
            cd = json.load(open(ckpt))
            s = [v["sharpe"] for v in cd["grid"].values()]
            msg = f"  seed={seed}: cov={cd['coverage']} best={max(s):.2f} mean={sum(s)/len(s):.2f}"
            if test_file.exists():
                td = json.load(open(test_file))
                ts = [r.get("test_sharpe", 0) for r in td[:20]]
                msg += f" test_mean={sum(ts)/len(ts):.2f} test_best={max(ts):.2f}"
            log(msg)
        else:
            log(f"  seed={seed}: missing")

    log(f"\nPipeline finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
