"""Export the per-encoder feature matrices for one scenario, so the encodings are
inspectable side by side instead of being a black box.

For the chosen scenario this reproduces *exactly* what the pipeline feeds the model:
load + clean the scenario CSV, make the deterministic grouped split, then for each
encoder fit on the TRAIN rows only and transform ALL rows. Each encoder's output is
written to its own CSV, prefixed with reference columns so every feature row is
traceable back to the original record:

    seq_index            -- the sequence (group) the row belongs to
    split                -- train / val / test (the split the model actually used)
    <target>             -- the label (unit1_beam_index)
    time_seconds_absolute-- ordering within a sequence (if present)
    <feature columns...> -- the encoded representation

Also writes ``_input_cleaned.csv`` (the frame the encoders consume) and ``_summary.csv``
(one row per encoder: shape + first feature names + any error), so you can read it as a
chain: original CSV -> _input_cleaned -> <encoder>.csv.

Usage:
    python experiments/export_encodings.py                      # scenario1, all encoders
    python experiments/export_encodings.py --scenario scenario3
    python experiments/export_encodings.py --encoders baseline rolling
    python experiments/export_encodings.py --max-feature-cols 50   # narrower preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make ``src`` importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from encoding_experiments.config import (  # noqa: E402
    DatasetConfig,
    EncoderConfig,
    SplitConfig,
    TaskConfig,
)
from encoding_experiments.data_loading import load_dataset  # noqa: E402
from encoding_experiments.encoders import build_encoder  # noqa: E402
from encoding_experiments.splitting import make_split  # noqa: E402

ALL_ENCODERS = ["baseline", "latlon", "timestamp", "lag_window", "rolling", "bs_geometry",
                "bs_bearing", "bs_rolling", "ple", "periodic"]
REFERENCE_COLS = ["seq_index", "time_seconds_absolute"]  # prepended for traceability


def _scenario_number(scenario: str) -> str:
    digits = "".join(c for c in scenario if c.isdigit())
    if not digits:
        raise ValueError(f"Could not parse a scenario number from {scenario!r}")
    return digits


def export(scenario: str, encoders: list[str], out_root: Path, max_feature_cols: int,
           rows: int | None) -> None:
    n = _scenario_number(scenario)
    dataset = DatasetConfig(
        kind=f"deepsense_{scenario}",
        path=f"datasets/deepsense/Scenario{n}",
        scenario_name=scenario,
    )
    task = TaskConfig(target="unit1_beam_index", group="seq_index", type="multiclass")
    split_cfg = SplitConfig()  # grouped, 0.30 / 0.50, seed 42 -- same as every config

    df, meta = load_dataset(dataset, task)
    split = make_split(df, task, split_cfg)
    split_label = split.as_label_series(df.index)
    print(f"Loaded {scenario}: {len(df)} rows, split sizes {split.sizes}")

    out_dir = out_root / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference columns present in this scenario, in a stable order.
    ref_cols = [c for c in REFERENCE_COLS if c in df.columns]

    def _decorate(features: pd.DataFrame) -> pd.DataFrame:
        """Prepend reference columns (seq_index / split / target / time) to a feature frame.

        Any feature column that collides with a reference column is dropped from the body
        first, so reference columns appear exactly once (the cleaned input frame still
        carries e.g. ``time_seconds_absolute``, which would otherwise duplicate the header).
        """
        head = pd.DataFrame(index=df.index)
        head["seq_index"] = df[task.group]
        head["split"] = split_label
        head[task.target] = df[task.target]
        for c in ref_cols:
            if c != "seq_index":
                head[c] = df[c]
        features = features.drop(columns=[c for c in head.columns if c in features.columns],
                                 errors="ignore")
        return pd.concat([head, features], axis=1)

    # The cleaned input the encoders actually consume.
    _decorate(df.drop(columns=[c for c in [task.group, task.target] if c in df.columns])) \
        .to_csv(out_dir / "_input_cleaned.csv", index=False)

    y = df[task.target]
    summary: list[dict] = []
    for name in encoders:
        try:
            enc = build_encoder(EncoderConfig(name=name, params={}), task)
            X = enc.fit_transform(df, split.train_idx, y)
            n_features_full = X.shape[1]

            truncated = n_features_full > max_feature_cols
            if truncated:
                X = X.iloc[:, :max_feature_cols]
            out = _decorate(X)
            if rows is not None:
                out = out.head(rows)

            out_path = out_dir / f"{name}.csv"
            out.to_csv(out_path, index=False)
            note = f"(showing first {max_feature_cols} of {n_features_full})" if truncated else ""
            print(f"  {name:11s} -> {out_path.name:18s} "
                  f"{out.shape[0]} rows x {n_features_full} features {note}")
            summary.append({
                "encoder": name,
                "n_rows": int(X.shape[0]),
                "n_features_full": int(n_features_full),
                "n_features_written": int(X.shape[1]),
                "truncated": truncated,
                "first_features": ", ".join(map(str, X.columns[:8])),
                "error": "",
            })
        except Exception as exc:  # noqa: BLE001 - report, don't abort the whole export
            print(f"  {name:11s} -> SKIPPED: {type(exc).__name__}: {exc}")
            summary.append({
                "encoder": name, "n_rows": 0, "n_features_full": 0,
                "n_features_written": 0, "truncated": False, "first_features": "",
                "error": f"{type(exc).__name__}: {exc}",
            })

    pd.DataFrame(summary).to_csv(out_dir / "_summary.csv", index=False)
    print(f"\nWrote {len(summary)} encoder CSVs + _input_cleaned.csv + _summary.csv to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Export per-encoder feature matrices for a scenario.")
    p.add_argument("--scenario", default="scenario1", help="e.g. scenario1, scenario3")
    p.add_argument("--encoders", nargs="+", default=ALL_ENCODERS,
                   help=f"subset of {ALL_ENCODERS}")
    p.add_argument("--out-dir", default=str(ROOT / "encoding_previews"),
                   help="output root (a per-scenario subfolder is created)")
    p.add_argument("--max-feature-cols", type=int, default=200,
                   help="cap feature columns written (keeps wide encodings openable)")
    p.add_argument("--rows", type=int, default=None, help="optional cap on rows written")
    args = p.parse_args()
    export(args.scenario, args.encoders, Path(args.out_dir), args.max_feature_cols, args.rows)


if __name__ == "__main__":
    main()
