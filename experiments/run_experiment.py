"""Run a single encoding experiment from a YAML config.

Usage:
    python experiments/run_experiment.py --config configs/xgb/scenario1_timestamp.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from encoding_experiments.config import load_config          # noqa: E402
from encoding_experiments.pipeline import run_experiment      # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one encoding experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
