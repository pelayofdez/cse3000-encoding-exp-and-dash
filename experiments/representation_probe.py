"""Probe representation quality (invariance) of each encoder, for one scenario.

Adapts the invariance axis of Plachouras et al. 2025 to the DeepSense position->beam task. For the
chosen scenario it fits each encoder on the (clean) TRAIN split, then:

* Invariance - injects Gaussian GPS noise into the TEST positions over a sweep of standard
  deviations (metres), re-encodes through the frozen encoder, and reports representation drift
  plus the downstream degradation in accuracy / DBA / power loss.

Perturbing only the TEST rows keeps each encoder's train-fitted parameters identical across noise
levels (fixed seed + unchanged train rows ⇒ effectively frozen), while exercising the realistic
case of a deployed model meeting noisy GPS. Input noise leaves the true beam labels and received
powers unchanged, so DBA / power-loss degradation is measured against the same ground truth.

Outputs (under <out-dir>/<scenario>/):
  invariance.csv - one row per (encoder, sigma_m): drift + clean/noisy/delta metrics

Usage:
  python experiments/representation_probe.py --scenario scenario1
  python experiments/representation_probe.py --scenario scenario3 \
      --encoders baseline rolling bs_geometry --sigmas 2 10 --seeds 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from encoding_experiments.config import (  # noqa: E402
    DatasetConfig, EncoderConfig, ModelConfig, SplitConfig, TaskConfig,
)
from encoding_experiments.data_loading import load_dataset, load_deepsense_power  # noqa: E402
from encoding_experiments.encoders import build_encoder  # noqa: E402
from encoding_experiments.evaluation import evaluate_split  # noqa: E402
from encoding_experiments.models import build_model  # noqa: E402
from encoding_experiments.representation_quality import (  # noqa: E402
    fit_drift_transformer, gps_noise_perturb, representation_drift,
)
from encoding_experiments.splitting import make_split  # noqa: E402

# Every encoder that operates on the UE position (the input this probe perturbs): the time-series
# DeepSense encoders plus the numeric-position re-encodings (ple / periodic), which encode the
# SAME unit2_lat/lon, just differently. Any listed encoder that cannot run on a scenario is
# skipped at runtime.
ALL_ENCODERS = [
    "baseline", "latlon", "latlon_time", "timestamp", "lag_window", "rolling",
    "bs_geometry", "bs_bearing", "bs_rolling", "ple", "periodic",
]


def _scenario_number(scenario: str) -> str:
    digits = "".join(c for c in scenario if c.isdigit())
    if not digits:
        raise ValueError(f"Could not parse a scenario number from {scenario!r}")
    return digits


def run(scenario: str, encoders: list[str], sigmas: list[float], seeds: int,
        model_name: str, top_k: int, out_root: Path) -> None:
    n = _scenario_number(scenario)
    dataset = DatasetConfig(kind=f"deepsense_{scenario}",
                            path=f"datasets/deepsense/Scenario{n}", scenario_name=scenario)
    task = TaskConfig(target="unit1_beam_index", group="seq_index", type="multiclass")

    df, _ = load_dataset(dataset, task)
    y = df[task.target]
    split = make_split(df, task, SplitConfig())

    # Ground truth for downstream degradation is invariant to input noise.
    power = load_deepsense_power(dataset, df)
    noise_power = float(power.min(axis=1).mean())
    power_test = power[split.test_idx]
    y_test = y.iloc[split.test_idx]
    train_labels = np.unique(y.iloc[split.train_idx])

    def _metrics(model, Z_test):
        m = evaluate_split(model, Z_test, y_test, train_labels, top_k,
                           compute_dba=True, power_matrix=power_test, noise_power=noise_power)
        return m["accuracy"], m["dba_score"], m["power_loss_db"]

    out_dir = out_root / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    inv_rows = []

    skipped = []
    for name in encoders:
        print(f"\n=== {name} ===")
        try:
            enc = build_encoder(EncoderConfig(name=name, params={}), task)
            Z_clean = enc.fit_transform(df, split.train_idx, y)
        except Exception as exc:  # noqa: BLE001 - encoder not applicable to this scenario
            print(f"  SKIPPED ({type(exc).__name__}: {exc})")
            skipped.append(name)
            continue
        # ---- Invariance (GPS-noise robustness) ----
        model = build_model(ModelConfig(name=model_name)).fit(
            Z_clean.iloc[split.train_idx], y.iloc[split.train_idx])
        drift_tf = fit_drift_transformer(Z_clean.iloc[split.train_idx])
        acc0, dba0, pl0 = _metrics(model, Z_clean.iloc[split.test_idx])
        print(f"  clean acc={acc0:.3f} dba={dba0:.3f} ploss={pl0:.2f}dB")

        for sigma in sigmas:
            cos_d, l2_d, accs, dbas, pls = [], [], [], [], []
            for seed in range(seeds):
                rng = np.random.default_rng(seed)
                df_p = gps_noise_perturb(df, split.test_idx, sigma, rng)
                enc_p = build_encoder(EncoderConfig(name=name, params={}), task)
                Z_p = enc_p.fit_transform(df_p, split.train_idx, y)
                c, l = representation_drift(
                    Z_clean.iloc[split.test_idx], Z_p.iloc[split.test_idx], drift_tf)
                a, d, p = _metrics(model, Z_p.iloc[split.test_idx])
                cos_d.append(c); l2_d.append(l); accs.append(a); dbas.append(d); pls.append(p)
            row = {
                "scenario": scenario, "encoder": name, "sigma_m": sigma, "n_seeds": seeds,
                "cos_drift": float(np.mean(cos_d)), "l2_drift": float(np.mean(l2_d)),
                "acc_clean": acc0, "acc_noisy": float(np.mean(accs)),
                "d_acc": acc0 - float(np.mean(accs)),
                "dba_clean": dba0, "dba_noisy": float(np.mean(dbas)),
                "d_dba": dba0 - float(np.mean(dbas)),
                "ploss_clean_db": pl0, "ploss_noisy_db": float(np.mean(pls)),
                "d_ploss_db": float(np.mean(pls)) - pl0,
            }
            inv_rows.append(row)
            print(f"  sigma={sigma:>4}m  cos_drift={row['cos_drift']:.3f} "
                  f"d_acc={row['d_acc']:+.3f} d_dba={row['d_dba']:+.3f} d_ploss={row['d_ploss_db']:+.2f}dB")

    pd.DataFrame(inv_rows).to_csv(out_dir / "invariance.csv", index=False)
    if skipped:
        print(f"Skipped (not applicable to {scenario}): {', '.join(skipped)}")
    print(f"\nWrote invariance to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Probe encoder representation quality.")
    p.add_argument("--scenario", default="scenario1")
    p.add_argument("--encoders", nargs="+", default=ALL_ENCODERS)
    p.add_argument("--sigmas", nargs="+", type=float, default=[1, 2, 5, 10],
                   help="GPS-noise standard deviations in metres")
    p.add_argument("--seeds", type=int, default=3, help="noise realisations averaged per sigma")
    p.add_argument("--model", default="xgboost", help="downstream model for the invariance probe")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--out-dir", default=str(ROOT / "representation_probes"))
    args = p.parse_args()
    run(args.scenario, args.encoders, args.sigmas, args.seeds, args.model, args.top_k,
        Path(args.out_dir))


if __name__ == "__main__":
    main()
