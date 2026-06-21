"""A/B-compare the equirectangular vs UTM projection for one scenario / encoder / seed.

Runs the full pipeline twice on the SAME config, split seed, and model seed, changing only
the encoder's ``projection`` param ('equirectangular' -> 'utm'), and prints the val/test
metrics side by side with their differences. Results go to a throwaway temp dir, so the main
``results/`` tree is untouched.

Only the position-geometry encoders (``rolling``, ``bs_geometry``, ``bs_bearing``,
``bs_rolling``) actually project lat/lon, so the projection switch only changes their output;
for any other encoder the two runs are identical by construction.

Usage:
    python experiments/compare_projection.py                                  # scenario1, bs_geometry, morais_nn, seed 1
    python experiments/compare_projection.py --scenario 6 --encoder bs_rolling
    python experiments/compare_projection.py --model xgboost --seed 42
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from encoding_experiments.config import load_config          # noqa: E402
from encoding_experiments.pipeline import run_experiment      # noqa: E402

# model name -> config sub-folder under configs/
MODEL_DIR = {"morais_nn": "morais", "xgboost": "xgb", "xgb": "xgb"}
MODELS_WITH_RANDOM_STATE = {"xgboost", "xgb", "morais_nn"}
METRIC_KEYS = ["accuracy", "top_k_accuracy", "dba_score", "power_loss_db",
               "pr_top1", "pr_topk", "top_kb", "top_kk_ba"]


def _run(path: Path, seed: int, method: str, results_dir: Path) -> dict:
    cfg = load_config(path)
    cfg.split.random_state = seed
    if cfg.model.name in MODELS_WITH_RANDOM_STATE:
        cfg.model.params = {**cfg.model.params, "random_state": seed}
    cfg.encoder.params = {**cfg.encoder.params, "random_state": seed, "projection": method}
    cfg.results_dir = str(results_dir / method)
    rec = run_experiment(cfg)
    return rec["metrics"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="1", help="scenario number, e.g. 1 (default) or 6")
    ap.add_argument("--encoder", default="bs_geometry",
                    help="position-geometry encoder (bs_geometry/bs_bearing/bs_rolling/rolling)")
    ap.add_argument("--model", default="morais_nn", choices=sorted(MODEL_DIR))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    sub = MODEL_DIR[args.model]
    cfg_path = ROOT / "configs" / sub / f"scenario{args.scenario}_{args.encoder}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    with tempfile.TemporaryDirectory(prefix="proj_cmp_") as tmp:
        tmp = Path(tmp)
        eq = _run(cfg_path, args.seed, "equirectangular", tmp)
        ut = _run(cfg_path, args.seed, "utm", tmp)

    title = f"scenario{args.scenario} / {args.encoder} / {args.model} / seed {args.seed}"
    print(f"\n=== Projection A/B: {title} ===")
    for split in ("val", "test"):
        print(f"\n[{split}]  {'metric':>16}  {'equirect':>12}  {'utm':>12}  "
              f"{'abs diff':>12}  {'rel diff':>10}")
        for k in METRIC_KEYS:
            a, b = eq[split].get(k), ut[split].get(k)
            if a is None or b is None:
                continue
            d = b - a
            rel = d / a if a not in (0, None) else float("nan")
            print(f"     {k:>16}  {a:>12.6f}  {b:>12.6f}  {d:>+12.2e}  {rel:>+9.2%}")

    # one-line verdict on the headline metric
    da = ut["test"]["accuracy"] - eq["test"]["accuracy"]
    dd = ut["test"]["dba_score"] - eq["test"]["dba_score"]
    print(f"\nTest accuracy change: {da:+.4f}   |   Test DBA change: {dd:+.4f}")
    print("(0.0 differences mean the projection had no effect on this metric.)")


if __name__ == "__main__":
    main()
