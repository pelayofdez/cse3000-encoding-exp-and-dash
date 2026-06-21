"""Representation-quality probe: **invariance**.

This evaluates an *encoding* (the feature matrix ``Z = E(x)``) beyond downstream beam accuracy,
adapting the **invariance** axis of Plachouras et al. 2025 ("Towards a Unified Representation
Evaluation Framework Beyond Downstream Tasks") to the DeepSense position->beam setting:

* **Invariance** - how stable ``Z`` (and the downstream beam prediction) is when the input GPS
  position is perturbed by realistic noise. This operationalises the real-world robustness question
  of Morais et al. 2022 (noisy GPS degrades beam prediction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_METRES_PER_DEG_LAT = 111_111.0  # ~constant; longitude is scaled by cos(latitude)


def gps_noise_perturb(df: pd.DataFrame, idx: np.ndarray, sigma_m: float,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of ``df`` with iid Gaussian GPS error (std ``sigma_m`` metres) added to
    ``unit2_lat``/``unit2_lon`` on rows ``idx`` (the test rows). Only position is perturbed;
    GPS-quality columns, the true beam, and the received powers are untouched."""
    out = df.copy()
    lat = out["unit2_lat"].to_numpy(dtype=float)
    lon = out["unit2_lon"].to_numpy(dtype=float)
    sel = np.asarray(idx)
    lat[sel] += rng.normal(0.0, sigma_m, size=sel.size) / _METRES_PER_DEG_LAT
    lon[sel] += rng.normal(0.0, sigma_m, size=sel.size) / (
        _METRES_PER_DEG_LAT * np.cos(np.radians(lat[sel])))
    out["unit2_lat"], out["unit2_lon"] = lat, lon
    return out


def fit_drift_transformer(Z_train: pd.DataFrame) -> Pipeline:
    """Impute+standardise fitted on the clean TRAIN representation, so drift is measured in a
    space where every feature contributes comparably (raw lat/lon would otherwise dominate)."""
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())]).fit(Z_train)


def representation_drift(Z_clean: pd.DataFrame, Z_pert: pd.DataFrame,
                         transformer: Pipeline) -> tuple[float, float]:
    """Mean row-wise drift between clean and perturbed representations, in standardised space.

    Returns ``(cosine_drift, l2_drift)`` where ``cosine_drift = mean(1 - cos)`` (0 = identical)
    and ``l2_drift`` is the mean relative L2 change. Higher ⇒ the encoding is more sensitive to
    GPS noise (less invariant).
    """
    a = transformer.transform(Z_clean)
    b = transformer.transform(Z_pert)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    l2 = np.linalg.norm(a - b, axis=1) / (np.linalg.norm(a, axis=1) + 1e-12)
    return float(np.mean(1.0 - cos)), float(np.mean(l2))
