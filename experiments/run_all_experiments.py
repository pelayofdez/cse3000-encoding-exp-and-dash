"""Run every config in a directory (default: ``configs/``).

Usage:
    python experiments/run_all_experiments.py
    python experiments/run_all_experiments.py --configs-dir configs --glob "scenario1_*.yaml"
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from encoding_experiments.config import load_config          # noqa: E402
from encoding_experiments.logging_utils import get_logger     # noqa: E402
from encoding_experiments.pipeline import run_experiment      # noqa: E402

logger = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all encoding experiments.")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument(
        "--glob", default="**/*.yaml",
        help="Glob under --configs-dir (recursive by default). E.g. 'xgb/*.yaml' for one "
             "model, '**/scenario2_*.yaml' for one scenario.",
    )
    parser.add_argument(
        "--exclude", default=None,
        help="Glob (under --configs-dir) of configs to SKIP, e.g. 'morais/*.yaml'.",
    )
    parser.add_argument(
        "--skip-on-error", action="store_true",
        help="Continue with the next config if one fails (e.g. a missing optional dependency).",
    )
    args = parser.parse_args()

    # Recursive glob, but never run anything under an ``ignore/`` directory.
    config_paths = [
        p for p in sorted(Path(args.configs_dir).glob(args.glob))
        if "ignore" not in p.parts
    ]
    if args.exclude:
        excluded = set(Path(args.configs_dir).glob(args.exclude))
        config_paths = [p for p in config_paths if p not in excluded]
        logger.info("Excluding %d config(s) matching %s", len(excluded), args.exclude)
    if not config_paths:
        logger.error("No configs matched %s/%s", args.configs_dir, args.glob)
        return

    succeeded, failed = [], []
    for path in config_paths:
        try:
            config = load_config(path)
            run_experiment(config)
            succeeded.append(path.name)
        except Exception as exc:  # noqa: BLE001
            failed.append((path.name, str(exc)))
            logger.error("FAILED %s: %s", path.name, exc)
            if not args.skip_on_error:
                traceback.print_exc()
                raise

    logger.info("Done. %d succeeded, %d failed.", len(succeeded), len(failed))
    for name, err in failed:
        logger.info("  - %s: %s", name, err)


if __name__ == "__main__":
    main()
