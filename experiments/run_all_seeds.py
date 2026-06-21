"""Run every config across multiple random seeds, appending all runs to the result CSVs.

For each (config, seed) pair the seed is injected into the split, the downstream model
(when the model accepts ``random_state``), and the encoder (harmless for encoders that
ignore it). Every run is a distinct row in ``results/metrics.csv`` carrying its
``random_state`` column, so all seeds appear side by side.

Usage:
    python experiments/run_all_seeds.py
    python experiments/run_all_seeds.py --seeds 1 2 3 42 --glob "xgb/*.yaml"
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

DEFAULT_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42]

# Both downstream models accept a ``random_state`` constructor arg, so the seed is injected
# into the model as well as into the split.
MODELS_WITH_RANDOM_STATE = {"xgboost", "xgb", "morais_nn", "position_nn"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all configs across multiple seeds.")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--glob", default="**/*.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--skip-on-error", action="store_true", default=True)
    args = parser.parse_args()

    config_paths = [
        p for p in sorted(Path(args.configs_dir).glob(args.glob))
        if "ignore" not in p.parts
    ]
    if not config_paths:
        logger.error("No configs matched %s/%s", args.configs_dir, args.glob)
        return

    total = len(config_paths) * len(args.seeds)
    logger.info("Running %d configs x %d seeds = %d runs.",
                len(config_paths), len(args.seeds), total)

    done = 0
    succeeded, failed = 0, 0
    for seed in args.seeds:
        for path in config_paths:
            done += 1
            try:
                config = load_config(path)
                # Inject the seed everywhere it matters.
                config.split.random_state = seed
                if config.model.name in MODELS_WITH_RANDOM_STATE:
                    config.model.params = {**config.model.params, "random_state": seed}
                config.encoder.params = {**config.encoder.params, "random_state": seed}
                logger.info("[%d/%d] seed=%d %s", done, total, seed, path.name)
                run_experiment(config)
                succeeded += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.error("FAILED seed=%d %s: %s", seed, path.name, exc)
                if not args.skip_on_error:
                    traceback.print_exc()
                    raise

    logger.info("Done. %d succeeded, %d failed (of %d).", succeeded, failed, total)


if __name__ == "__main__":
    main()
