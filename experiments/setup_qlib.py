#!/usr/bin/env python3
"""
Set up Qlib with CSI300/CSI500 data for direct comparison to AlphaGen/AlphaForge.

Downloads and initializes the standard Qlib China A-shares dataset.
Prepares train/val/test splits matching AlphaGen's protocol.

Usage:
    python experiments/setup_qlib.py
    python experiments/setup_qlib.py --market csi500
"""

import argparse
import sys
from pathlib import Path

QLIB_DATA_DIR = Path.home() / "research" / "alpha-factory" / "data" / "qlib"


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="csi300", choices=["csi300", "csi500"])
    args = parser.parse_args()

    try:
        import qlib
    except ImportError:
        print("Installing qlib...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qlib"])
        import qlib

    # Download China A-shares data
    QLIB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = QLIB_DATA_DIR / "cn_data"

    if not data_path.exists():
        print("Downloading Qlib China A-shares data...")
        print("This uses the 1d frequency dataset from Qlib's data server.")
        from qlib.contrib.data.handler import Alpha158

        # Use qlib's data download utility
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "qlib.run.get_data",
             "qlib_data", "--target_dir", str(data_path),
             "--region", "cn"],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Download failed: {result.stderr}")
            print("\nAlternative: manually download from")
            print("  https://github.com/microsoft/qlib#data-preparation")
            print(f"  and extract to {data_path}")
            return
    else:
        print(f"Qlib data already at {data_path}")

    # Initialize Qlib
    qlib.init(provider_uri=str(data_path), region_type="cn")

    # Verify
    from qlib.data import D
    instruments = D.instruments(market=args.market)
    print(f"\n{args.market} instruments: {len(instruments)} stocks")

    # Date ranges matching AlphaGen
    print(f"\nAlphaGen date splits:")
    print(f"  Train: 2009-01-01 to 2018-12-31")
    print(f"  Val:   2019-01-01 to 2019-12-31")
    print(f"  Test:  2020-01-01 to 2021-12-31")

    # Load sample data to verify
    fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]
    df = D.features(
        instruments=instruments[:5] if isinstance(instruments, list) else instruments,
        fields=fields,
        start_time="2018-01-01",
        end_time="2018-01-31",
    )
    print(f"\nSample data shape: {df.shape}")
    print(df.head())

    print(f"\nQlib ready. Data at {data_path}")
    print(f"To run experiments: python experiments/run_qlib.py --market {args.market}")


if __name__ == "__main__":
    setup()
