"""End-to-end experiment pipeline.

Implements the 11 documented steps for one config:

    load config -> load dataset -> clean/validate -> split ->
    fit encoder (train only) -> transform train/val/test ->
    train model -> evaluate -> representation stats -> runtime -> save CSVs
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import numpy as np

from .config import ExperimentConfig
from .data_loading import load_dataset, load_deepsense_power
from .encoders import build_encoder
from .evaluation import evaluate_split, feature_importances, representation_stats
from .logging_utils import get_logger
from .models import build_model
from .splitting import make_split

logger = get_logger()


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_experiment(config: ExperimentConfig) -> dict:
    run_id = uuid.uuid4().hex  # unique per run; joins metrics/representation/runtime/json
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results_dir = Path(config.results_dir)
    logger.info("=== Experiment: %s (run_id=%s) ===", config.name, run_id)

    # 2-3. Load + clean/validate.
    df, data_meta = load_dataset(config.dataset, config.task)
    y = df[config.task.target]

    # 4. Split (deterministic; identical across encoders for a given seed).
    split = make_split(df, config.task, config.split)
    logger.info("Split sizes: %s", split.sizes)
    split_path = (
        results_dir / "splits"
        / f"{config.dataset.kind}_{config.split.strategy}_seed{config.split.random_state}.csv"
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split.as_label_series(df.index).to_frame("split").to_csv(split_path)

    # 5-6. Fit encoder on TRAIN only, then transform all rows.
    encoder = build_encoder(config.encoder, config.task)
    t0 = time.perf_counter()
    encoder.fit(df, split.train_idx, y)
    encode_fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    X = encoder.transform(df)
    encode_transform_seconds = time.perf_counter() - t0

    X_train, y_train = X.iloc[split.train_idx], y.iloc[split.train_idx]
    X_val, y_val = X.iloc[split.val_idx], y.iloc[split.val_idx]
    X_test, y_test = X.iloc[split.test_idx], y.iloc[split.test_idx]
    train_labels = y_train.unique()

    # 7. Train downstream model (imputer + classifier), fit on train only.
    model = build_model(config.model)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    # 8. Evaluate validation (tuning) and test (final headline).
    top_k = config.evaluation.top_k
    # DBA and power loss are beam-prediction metrics; compute them only for the DeepSense
    # task. The power matrix (received power of every beam per sample) and the scenario
    # noise floor P_n (mean of per-sample minima, computed over the WHOLE scenario for a
    # stable reference) feed the power-loss metric.
    is_beam = config.dataset.kind.startswith("deepsense_")
    power_full, noise_power = None, None
    if is_beam:
        try:
            power_full = load_deepsense_power(config.dataset, df)
            # Scenario noise floor P_n, exactly as Morais et al. 2022 define it: "the average
            # of the smallest power per sample", i.e. the mean over samples of each sample's
            # minimum beam power.
            noise_power = float(power_full.min(axis=1).mean())
        except Exception as exc:  # noqa: BLE001 - power loss is optional; degrade to NaN
            logger.warning("Power matrix unavailable (%s); power_loss_db will be NaN.", exc)
            power_full = None

    def _eval(X_eval, y_eval, idx):
        return evaluate_split(
            model, X_eval, y_eval, train_labels, top_k,
            compute_dba=is_beam,
            dba_max_k=config.evaluation.dba_max_k,
            dba_delta=config.evaluation.dba_delta,
            power_matrix=(power_full[idx] if power_full is not None else None),
            noise_power=noise_power,
        )

    t0 = time.perf_counter()
    val_metrics = _eval(X_val, y_val, split.val_idx)
    predict_val_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    test_metrics = _eval(X_test, y_test, split.test_idx)
    predict_test_seconds = time.perf_counter() - t0

    # 9. Representation statistics.
    rep = representation_stats(model, X_train, encoder.n_input_features(df))

    # 9b. Feature-importance ranking. XGBoost exposes a native signal; morais_nn has none and
    # falls back to permutation importance, computed on the VALIDATION split (held-out, no test
    # peeking) so every model produces a ranking. Use the names
    # emerging from the pre-classifier steps so they line up with the classifier's
    # coefficients/importances even if the imputer dropped an all-missing column.
    try:
        feat_names = list(model[:-1].get_feature_names_out())
    except Exception:  # noqa: BLE001 - some transformers can't name features; fall back
        feat_names = list(map(str, X_train.columns))
    importance_ranking, importance_kind = feature_importances(
        model, feat_names, top_n=config.evaluation.top_features,
        X=X_val, y=y_val, random_state=config.split.random_state,
    )

    # 10-11. Assemble + persist.
    common = {
        "run_id": run_id,
        "experiment": config.name,
        "dataset": config.dataset.kind,
        "regime": "time_series",
        "encoder": config.encoder.name,
        "model": config.model.name,
        "target": config.task.target,
        "n_train": int(len(split.train_idx)),
        "n_classes_train": int(len(train_labels)),
        "random_state": config.split.random_state,
        "timestamp_utc": started,
    }

    for split_name, m in (("val", val_metrics), ("test", test_metrics)):
        _append_csv(results_dir / "metrics.csv", {**common, "split": split_name, **m})

    _append_csv(results_dir / "representation_metrics.csv", {**common, **rep})

    # One row per ranked feature, so the central CSV is a tidy long table that joins
    # back to the other CSVs on run_id (and is filterable by encoder / model).
    for item in importance_ranking:
        _append_csv(results_dir / "feature_importance.csv", {
            **common,
            "importance_kind": importance_kind,
            "n_features_total": int(len(feat_names)),
            **item,
        })

    runtime_row = {
        **common,
        "encode_fit_seconds": encode_fit_seconds,
        "encode_transform_seconds": encode_transform_seconds,
        "model_fit_seconds": fit_seconds,
        "predict_val_seconds": predict_val_seconds,
        "predict_test_seconds": predict_test_seconds,
    }
    _append_csv(results_dir / "runtime.csv", runtime_row)

    run_record = {
        "run_id": run_id,
        "config": {
            "name": config.name,
            "source_path": config.source_path,
            "regime": "time_series",
            "dataset": vars(config.dataset),
            "task": vars(config.task),
            "split": vars(config.split),
            "encoder": {"name": config.encoder.name, "params": config.encoder.params},
            "model": {"name": config.model.name, "params": config.model.params},
            "evaluation": vars(config.evaluation),
        },
        "data_meta": data_meta,
        "split_sizes": split.sizes,
        "metrics": {"val": val_metrics, "test": test_metrics},
        "representation": rep,
        "feature_importance": {"kind": importance_kind, "ranking": importance_ranking},
        "runtime": {k: runtime_row[k] for k in runtime_row if k.endswith("seconds")},
    }
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{config.name}_{run_id}.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )

    logger.info(
        "VAL  acc=%.4f dba=%.4f ploss=%.3fdB pr1=%.4f pr%d=%.4f | "
        "TEST acc=%.4f dba=%.4f ploss=%.3fdB pr1=%.4f pr%d=%.4f",
        val_metrics["accuracy"], val_metrics["dba_score"], val_metrics["power_loss_db"],
        val_metrics["pr_top1"], top_k, val_metrics["pr_topk"],
        test_metrics["accuracy"], test_metrics["dba_score"], test_metrics["power_loss_db"],
        test_metrics["pr_top1"], top_k, test_metrics["pr_topk"],
    )
    return run_record
