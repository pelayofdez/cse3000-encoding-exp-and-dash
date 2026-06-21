# Experiment Overview - Position-Aided Beam Prediction

Predicting the optimal mmWave beam from GPS position on the real-world DeepSense 6G dataset, and
measuring how the **feature encoding** of that position changes downstream performance and
representation quality.

---

## 1. Research focus

The encoding (representation) of the input is the **only** experimental variable; the dataset,
split, and task are held fixed, so any difference in performance is attributable to the
representation. The study asks:

- How do different encodings of GPS position affect mmWave beam-prediction quality?
- Does the answer hold across **models** (a tree vs. a neural net) and **scenarios** (different
  real-world locations)?
- Beyond accuracy, how do encodings compare on the **system objective** (received power) and on
  **representation properties** (what they preserve, how robust they are to GPS noise)?

---

## 2. Dataset - DeepSense 6G

Real-world co-located GPS + mmWave measurements. A vehicle (**unit2**) carrying a GPS receiver
drives past a fixed base station (**unit1**) with a 64-beam mmWave array; at each sample unit1
sweeps its codebook and records the received power per beam.

- **Scenarios:** 1, 2, 3, 4, 5, 6, 7, 8, 9. *Scenario 4 ships no `unit1_beam_index` column*, so
  its target is derived automatically: the beam with the highest received power
  (`argmax(unit1_pwr_60ghz) + 1`, 1-indexed) is labelled the optimal beam, exactly matching the
  beam-index definition used in every other scenario.
- **Sequences (`seq_index`):** individual drives; consecutive samples within a drive are
  milliseconds apart and nearly identical - the reason the split is sequence-grouped (§4).
- **Base features:** `unit2_lat`, `unit2_lon`, plus the GPS-quality columns a scenario provides
  (`unit2_PDOP`, `unit2_HDOP`, `unit2_num_sat`, `unit2_direction`).
- **Power vectors** (`unit1_pwr_60ghz`, 64 values per sample) feed the power metrics (§7).

---

## 3. Task

- **Target:** `unit1_beam_index` - the optimal beam, a **multiclass** label (≈ 58–62 active beam
  classes of 64). Labels are 1-indexed.
- **Granularity:** per timestep, using a causal window of the recent past (nowcasting the
  current-row beam). Identical across encoders so the comparison is fair.

---

## 4. Splitting & leakage control

- **Strategy:** `GroupShuffleSplit` grouped by `seq_index` - whole drives land entirely in train,
  val, or test. A row-wise split would leak (adjacent samples share near-identical GPS and beam).
- **Ratios:** ≈ 70 % train / 15 % val / 15 % test, identical across encoders for a fixed seed.
- **Encoder leakage policy:** anything learned (the projection origin, imputers) is fit on train
  rows only; deterministic per-sequence features are safe because sequences never span splits.

---

## 5. Encoders - the independent variable

| Encoder | What it adds on top of the base features |
|---|---|
| `baseline` | nothing - raw GPS features (reference) |
| `latlon` | raw lat/lon only |
| `timestamp` | elapsed/Δ seconds + cyclical sin/cos of time |
| `lag_window` | timestamp + lag-1/lag-2 of each base column |
| `rolling` | timestamp + motion (x/y metres, distance, speed, heading) + rolling mean/std |
| `bs_geometry` | base-station-relative geometry (range, bearing, local offsets) |
| `bs_bearing` | base-station bearing as sin/cos |
| `bs_rolling` | bs geometry + rolling statistics |
| `ple` | piecewise-linear encoding of lat/lon (Gorishniy et al. 2022) |
| `periodic` | Fourier-feature encoding of lat/lon (Tancik et al. 2020) |

All encoders share one interface (`fit` on train → `transform` all rows). Position-geometry
encoders project lat/lon to local metres via UTM (`projection.py`).

---

## 6. Downstream models

Two models, spanning two inductive biases:

| Model | Family |
|---|---|
| `xgboost` | boosted trees (scale-invariant) |
| `morais_nn` | the position-aided net of Morais et al. 2022 (scale-sensitive) |

Each is wrapped in a pipeline with a median imputer; `morais_nn` also gets a `StandardScaler`.
All fitting happens on the training split only.

---

## 7. Evaluation

### 7.1 Task-performance metrics (per val/test split)

| Metric | Captures | Direction |
|---|---|---|
| `accuracy` | exact-beam match | higher better |
| `top_k_accuracy` (k=3) | true beam among top-k - the operational beam-prediction goal | higher better |
| `dba_score` | Distance-Based Accuracy (DeepSense challenge): credits *near* beams | higher better |
| `power_loss_db` | received power given up by the predicted vs. optimal beam (Morais et al. 2022) | lower better (0 dB = optimal) |
| `pr_top1`, `pr_topk` | Power Ratio: selected-beam power ÷ best-beam power | higher better (1 = optimal) |
| `top_kb`, `top_kk_ba` | is the prediction among the K strongest-power beams? | higher better |

### 7.2 Representation-quality probes

Run separately via `experiments/representation_probe.py` (per scenario):

- **Informativeness** - how well a linear vs. an MLP probe recovers each factor of variation
  (position, speed, heading, BS-relative geometry) from the features the encoder *adds*; reported
  as R²/RMSE. The linear-vs-MLP gap shows whether a factor is linearly exposed.
- **Invariance** - inject Gaussian GPS noise into the test positions, re-encode through the frozen
  encoder, and report **representation drift** plus **downstream degradation** (Δaccuracy / ΔDBA /
  Δpower-loss). Operationalises the noisy-GPS robustness question of Morais et al. 2022.

### 7.3 Auxiliary outputs

Representation stats (`n_raw_features`, `n_features_after_encoding`, `sparsity_train`),
per-feature importance, and a runtime breakdown, all keyed by `run_id`.

---

## 8. Pipeline (per config)

```
load config → load + clean DeepSense CSV → grouped split →
fit encoder (train only) → transform all rows →
train model → evaluate val + test → representation stats →
feature importance → runtime → append result CSVs (+ per-run JSON)
```

Results land in `results/` as tidy long CSVs joinable on `run_id`: `metrics.csv`,
`representation_metrics.csv`, `feature_importance.csv`, `runtime.csv`.

---

## 9. Tooling

| Script | Purpose |
|---|---|
| `experiments/generate_configs.py` | single source of truth for the config matrix |
| `experiments/run_experiment.py` | run one config |
| `experiments/run_all_experiments.py` | run the full sweep |
| `experiments/export_encodings.py` | dump each encoder's feature matrix for a scenario |
| `experiments/representation_probe.py` | informativeness + invariance probes |

One seed (1) governs the split, model RNGs, and encoder RNGs. Adding a scenario/encoder/model is a
one-line change in `generate_configs.py`.

---

## 10. Key design decisions

- **Sequence-grouped split** - the single most important leakage guard for sequential GPS data.
- **Encoding is the only thing that varies** - fixed split + fixed task → clean attribution.
- **Two model biases** - encodings are judged across a tree and a neural net.
- **Beyond accuracy** - DBA and power loss capture the *system* objective; the probes capture
  *what* and *how robustly* each encoding encodes position.
