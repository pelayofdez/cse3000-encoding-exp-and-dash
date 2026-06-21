# Metrics & Methodology

Which metrics we report, why, and how to compare results across scenarios and the two downstream
models. Citations were checked against their DOI; the few marked **(verify details)** are
well-established works to confirm in your reference manager.

---

## 1. What we measure

We separate three things: **task performance** (how well the pipeline predicts),
**representation quality** (how usable an encoding is, independent of the model), and
**representation efficiency** (how compact it is).

### 1.1 Task-performance metrics

| metric | what it captures | why |
|---|---|---|
| **top-k accuracy** (k=3) | true beam among the top-k predicted beams | *the* standard beam-prediction metric: the goal is to shrink the beam sweep to a few candidates, so "true beam in top-k" maps to overhead reduction (DeepSense 6G; Alkhateeb et al. 2023; Demirhan & Alkhateeb 2021). The top-k convention comes from ImageNet (Russakovsky et al. 2015). |
| **accuracy** | fraction of exactly-correct beams | intuitive but **misleading under class imbalance** (58–62 beam classes, skewed), so never reported alone (He & Garcia 2009). |
| **DBA** (distance-based accuracy) | beam quality that **credits *near* beams**, averaged over top-1..3, δ=5 | the official metric of the DeepSense 6G beam-prediction challenge: adjacent beams overlap in angular coverage, so a neighbour of the true beam is almost as good as an exact hit. Higher is better, in [0, 1] (Alkhateeb et al. 2023). |
| **power loss (dB)** | received power the **predicted** beam gives up vs. the **optimal** beam (noise-subtracted) | scores the **system objective - received power - not label correctness**. 0 dB = always optimal; lower is better. Defined by Morais et al. 2022 (eq. 5). |
| **power ratio** (`pr_top1`, `pr_topk`) | selected-beam power ÷ best-beam power, for the top-1 and top-k predictions | a normalised power view; 1 = optimal (Vuckovic et al. 2024). |
| **top-KB / top-K,KBA** (`top_kb`, `top_kk_ba`) | is the prediction among the K strongest-power beams? | relaxes accuracy to a power-aware "within the K best beams" (Vuckovic et al. 2024). |

**Why several metrics, not one:** every scalar metric is blind to some kinds of error
(Sokolova & Lapalme 2009). Reporting them together covers exact-match, top-k shortlist utility,
ordinal beam-distance (DBA), and the physical-layer power objective (power loss / ratio). The
power metrics matter most operationally - they measure the received power the link actually keeps.

### 1.2 Representation quality - the probes

Representation quality is judged **independently of any single model**: a good encoding (i)
preserves the information that matters and (ii) is stable under realistic input noise. We measure
this with the probes of Plachouras et al. (2025), adapted to the position→beam setting (in
`representation_quality.py`, run by `experiments/representation_probe.py`):

* **Informativeness** - how well a linear vs. an MLP probe recovers a known factor of variation
  (position, speed, heading, BS-relative geometry) from the features the encoder *adds*; reported
  as R²/RMSE on held-out test rows. A small linear-vs-MLP gap with high linear R² is the strong
  case (the factor is *linearly* exposed).
* **Invariance** - perturb the test GPS with Gaussian noise, re-encode through the frozen encoder,
  and report **representation drift** (cosine / relative-L2) and **downstream degradation**
  (Δaccuracy / ΔDBA / Δpower-loss). Lower ⇒ more robust. Operationalises Morais et al. (2022):
  noisy GPS degrades position-aided beam prediction.

### 1.3 Representation efficiency

`n_features_after_encoding` and `sparsity_train` describe the *cost* of a representation
(dimensionality, density), not its quality. Report them as context next to performance.

### 1.4 Fair evaluation

* **Grouped (leave-sequence-out) split** prevents the optimistic bias of letting near-duplicate
  rows straddle train/test (Kaufman et al. 2012). Adjacent rows in a drive share near-identical GPS
  and the same beam, so a row-wise split would leak.
* **One fixed seed / identical split across encoders** isolates the encoding as the only variable.
