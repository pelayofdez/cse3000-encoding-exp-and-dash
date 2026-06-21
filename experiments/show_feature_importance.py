"""Show the per-encoding feature-importance ranking from results/feature_importance.csv.

The pipeline records, for every run, the downstream model's own importance signal for
each encoded feature (XGBoost -> gain; morais_nn -> permutation importance), ranked most
-> least important. This script prints those rankings, one block per experiment (using
the latest run of each), and can be filtered by encoder / model / dataset.

Usage:
    python experiments/show_feature_importance.py
    python experiments/show_feature_importance.py --top 10 --model xgboost
    python experiments/show_feature_importance.py --encoder rolling
    python experiments/show_feature_importance.py --dataset deepsense_scenario1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Display feature-importance rankings.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--top", type=int, default=15, help="features to show per experiment")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--experiment", default=None)
    args = ap.parse_args()

    fp = Path(args.results_dir) / "feature_importance.csv"
    if not fp.exists():
        print(f"No feature_importance.csv under {args.results_dir!r}. Run an experiment first.")
        return

    df = pd.read_csv(fp)

    # Keep only the latest run of each experiment (the file is append-ordered).
    latest = df.groupby("experiment")["run_id"].transform("last")
    df = df[df["run_id"] == latest]

    for col, val in (("encoder", args.encoder), ("model", args.model),
                     ("dataset", args.dataset), ("experiment", args.experiment)):
        if val is not None:
            df = df[df[col] == val]
    if df.empty:
        print("No rows match the given filters.")
        return

    for experiment, g in df.sort_values(["experiment", "rank"]).groupby("experiment"):
        head = g.iloc[0]
        print(f"\n=== {experiment} ===")
        print(f"    dataset={head['dataset']}  regime={head['regime']}  "
              f"encoder={head['encoder']}  model={head['model']}")
        print(f"    importance={head['importance_kind']}  "
              f"total_features={head['n_features_total']}  "
              f"(showing top {min(args.top, len(g))} of {len(g)} recorded)")
        for _, r in g.head(args.top).iterrows():
            bar = "#" * max(1, int(round(r["importance_normalized"] * 50)))
            print(f"    #{int(r['rank']):<3} {r['feature']:<22} "
                  f"{r['importance']:.5f}  {r['importance_normalized']*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
