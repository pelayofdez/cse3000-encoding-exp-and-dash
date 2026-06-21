# CSE3000 - Experiments and Visualisation Environment

[![DOI](https://zenodo.org/badge/1275994806.svg)](https://doi.org/10.5281/zenodo.20784947)

Encoding experiments on beam-prediction task plus the Streamlit dashboard that visualises their results, all in one repository. 

Running an experiment writes result CSVs into `results/`, the folder the dashboard reads, so new results appear in the app with a refresh.

The study asks how the representation of a UE's GPS position affects downstream beam prediction. To keep comparisons fair, only the encoder changes between the runs you compare the dataset, split and task are held fixed.

## Setup

```
pip install -r requirements.txt
```

`numpy`, `pandas`, `scikit-learn`, `PyYAML`, `utm` are the core; `xgboost` and `torch` are
needed only for the two downstream models; `streamlit` + `altair` power the dashboard.

## Dashboard

```
streamlit run streamlit_app.py
```

The dashboard reads `results/*.csv` (merged on `run_id`) and the probe CSVs under
`representation_probes/`. Generate them by running experiments and the representation probes (see
below); the app picks them up on a refresh. Pick a scenario and a model, then compare encodings:

* **Performance / Feature Quality / Runtime** - grouped bars per encoder (mean ±1 std over the
  selected seeds), every metric selectable.
* **Feature Importance** - top encoded features for the selected encoder.
* **Feature Representation** - *invariance* (robustness to GPS noise).
* **Interactive y-axis** - every chart has an opt-in control to zoom in on small differences.

## Running experiments

**Run from the repository root**.
Experiments need the DeepSense scenario data under `datasets/deepsense/Scenario{n}/`.

```
# one experiment
python experiments/run_experiment.py --config configs/morais/scenario1_baseline.yaml

# everything (skip a config if an optional dependency is missing)
python experiments/run_all_experiments.py --skip-on-error

# the representation probes for a scenario (-> representation_probes/<scenario>/)
python experiments/representation_probe.py --scenario scenario1
```

The config tree is generated. `experiments/generate_configs.py` is the single source of
truth for which experiments exist (scenarios × encoders × models). Adding one is a one-line edit
there, re-running rebuilds `configs/`.

## Result CSVs

* `metrics.csv` - one row per (experiment, split): accuracy, top_k_accuracy, dba_score,
  power_loss_db, and the power-aware beam metrics (`pr_top1` / `pr_topk` / `top_kb` / `top_kk_ba`).
* `representation_metrics.csv` - n_raw_features, n_features_after_encoding, sparsity.
* `feature_importance.csv` - tidy table of the top-N encoded features ranked per run.
* `runtime.csv` - encode / fit / predict seconds.
* `runs/*.json`, `splits/*.csv` - full provenance and per-row split labels.

## Datasets, encoders & models

The DeepSense scenarios are sequential drives past a base station; the target is
`unit1_beam_index`, the optimal mmWave beam (multiclass).

**Encoders** (the independent variable): `baseline (Raw Sample)`, `latlon`, `timestamp`, `lag_window`,
`rolling`, `bs_geometry`, `bs_bearing`, `bs_rolling`, `ple`, `periodic`.

**Models**: `xgboost` (boosted trees) and `morais_nn` (anaptation of Morais et al.
2022, arXiv:2205.09054).

See `docs/` for the experiment overview and the metrics methodology.

---

**Study period:** 2026 Q4
