"""Feature encoders.

Every encoder shares one interface so the rest of the pipeline never special-cases
a representation::

    enc = build_encoder(encoder_cfg, task)
    enc.fit(df, train_idx, y)          # learns params on the TRAIN rows only
    X = enc.transform(df)              # numeric feature matrix for ALL rows

The pipeline then slices ``X`` by the split indices. Downstream imputation lives
in the model pipeline, so encoders may emit NaNs (e.g. lag features at the start
of a sequence).

Leakage policy
--------------
* Encoders that learn anything from data (the ``rolling`` / ``bs_rolling``
  projection origin, the ``ple`` bin edges, the ``periodic`` min-max range and
  frequencies) fit strictly on the training rows.
* The purely deterministic per-sequence features (``timestamp``, ``lag_window``)
  are computed over the whole frame, which is safe **only because** the DeepSense
  split is grouped by sequence - a sequence never spans two splits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_RANDOM_STATE, EncoderConfig, TaskConfig
from .projection import DEFAULT_METHOD, to_local_xy

DEEPSENSE_BASE_FEATURES = [
    "unit2_lat", "unit2_lon", "unit2_PDOP", "unit2_HDOP", "unit2_num_sat", "unit2_direction",
]


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Encoder:
    name = "base"

    def __init__(self, task: TaskConfig, params: dict[str, Any] | None = None):
        self.task = task
        self.params = params or {}

    def fit(self, df: pd.DataFrame, train_idx: np.ndarray, y: pd.Series) -> "Encoder":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame, train_idx: np.ndarray, y: pd.Series) -> pd.DataFrame:
        return self.fit(df, train_idx, y).transform(df)

    def n_input_features(self, df: pd.DataFrame) -> int:
        """Number of *source* feature columns this encoder consumes, before any
        encoding/derivation. Compared against the encoded width in
        ``representation_stats`` to show how the encoding expands the feature space.
        Defaults to the base-feature universe (the DeepSense encoders)."""
        return len(self._base_features(df))

    # shared helpers ------------------------------------------------------- #
    def _base_features(self, df: pd.DataFrame) -> list[str]:
        requested = self.params.get("base_features", DEEPSENSE_BASE_FEATURES)
        cols = [c for c in requested if c in df.columns]
        if not cols:
            raise KeyError("None of the requested base features are present in the data.")
        return cols

    @staticmethod
    def _as_numeric(X: pd.DataFrame) -> pd.DataFrame:
        return X.apply(pd.to_numeric, errors="coerce")


# --------------------------------------------------------------------------- #
# DeepSense: deterministic per-sequence encoders
# --------------------------------------------------------------------------- #
def _timestamp_features(df: pd.DataFrame, group: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    seq_start = df.groupby(group)["time_seconds_absolute"].transform("min")
    out["elapsed_seconds"] = df["time_seconds_absolute"] - seq_start
    out["delta_seconds"] = df.groupby(group)["time_seconds_absolute"].diff().fillna(0)
    out["time_sin"] = np.sin(2 * np.pi * df["time_seconds_absolute"] / 86400)
    out["time_cos"] = np.cos(2 * np.pi * df["time_seconds_absolute"] / 86400)
    return out


def _rolling_window_stats(values: pd.DataFrame, groups: pd.Series, windows, stats) -> pd.DataFrame:
    """Per-sequence rolling-window statistics of each column in ``values``.

    ``groups`` (aligned to ``values.index``) keeps every window inside a single sequence - a
    window never spans two drives. Produces one column per ``(col, window, stat)`` named
    ``f"{col}_roll{window}_{stat}"``. ``mean``/``min``/``max`` use ``min_periods=1`` so the
    feature exists from the first sample; ``std`` uses ``min_periods=2`` (a lone point has no
    spread) with the leading NaN filled as 0. Shared by the ``rolling`` and ``bs_rolling``
    encoders so both compute their temporal stats identically.
    """
    grouped = pd.concat([groups.rename("__group__"), values], axis=1).groupby("__group__")
    out = pd.DataFrame(index=values.index)
    for window in windows:
        for col in values.columns:
            g = grouped[col]
            for stat in stats:
                name = f"{col}_roll{window}_{stat}"
                if stat == "mean":
                    out[name] = g.transform(lambda s: s.rolling(window, min_periods=1).mean())
                elif stat == "std":
                    out[name] = g.transform(lambda s: s.rolling(window, min_periods=2).std().fillna(0))
                elif stat == "min":
                    out[name] = g.transform(lambda s: s.rolling(window, min_periods=1).min())
                elif stat == "max":
                    out[name] = g.transform(lambda s: s.rolling(window, min_periods=1).max())
                else:
                    raise ValueError(f"Unknown rolling statistic: {stat!r}")
    return out


class BaselineEncoder(Encoder):
    name = "baseline"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._as_numeric(df[self._base_features(df)].copy())


class LatLonEncoder(BaselineEncoder):
    """Position-only input: the raw GPS pair ``(unit2_lat, unit2_lon)`` and nothing else -
    exactly the two-feature input of the position-aided beam-prediction NN of Morais et al.
    2022 (arXiv:2205.09054). Where ``baseline`` also passes the GNSS-quality channels
    (PDOP/HDOP/num_sat/direction), this strips the representation down to bare position so a
    model must, like the paper's net, infer the beam from latitude/longitude alone. The model
    pipeline's ``StandardScaler`` supplies the normalisation the paper did with min-max +
    200-bin quantisation. Needs GPS, so it is unavailable on GPS-less scenarios (e.g. Scenario 2)."""

    name = "latlon"

    def _base_features(self, df: pd.DataFrame) -> list[str]:
        cols = [c for c in ("unit2_lat", "unit2_lon") if c in df.columns]
        if len(cols) < 2:
            raise KeyError(
                "The 'latlon' encoder requires GPS columns unit2_lat/unit2_lon, which are "
                "unavailable for this dataset (e.g. DeepSense Scenario 2 has no GPS files)."
            )
        return cols


class LatLonTimeEncoder(LatLonEncoder):
    """Simple position **plus** lightweight time context: the raw ``(unit2_lat, unit2_lon)``
    pair (as in :class:`LatLonEncoder` / Morais et al. 2022) augmented with the four
    :func:`_timestamp_features` - ``elapsed_seconds`` (how far along its drive the sample is),
    ``delta_seconds`` (gap to the previous sample), and ``time_sin``/``time_cos`` (cyclical
    time-of-day). Tests whether handing the position-aided net a sense of *when* / *where in
    the trajectory* - without any movement/geometry engineering - improves on position alone.

    It is exactly :class:`TimestampEncoder` with the base restricted to lat/lon (the full
    ``timestamp`` encoder also carries the GNSS-quality channels), so the only difference from
    ``latlon`` is the added time block. Needs GPS + ``time_seconds_absolute``."""

    name = "latlon_time"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self._as_numeric(df[self._base_features(df)].copy())
        ts = _timestamp_features(df, self.task.group)
        return pd.concat([base, ts], axis=1)


class TimestampEncoder(Encoder):
    name = "timestamp"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self._as_numeric(df[self._base_features(df)].copy())
        ts = _timestamp_features(df, self.task.group)
        return pd.concat([base, ts], axis=1)


class LagWindowEncoder(Encoder):
    name = "lag_window"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        group = self.task.group
        base_cols = self._base_features(df)
        base = self._as_numeric(df[base_cols].copy())
        ts = _timestamp_features(df, group)

        lag_steps = self.params.get("lag_steps", [1, 2])
        lags = pd.DataFrame(index=df.index)
        for lag in lag_steps:
            for col in base_cols:
                lags[f"{col}_lag{lag}"] = df.groupby(group)[col].shift(lag)
        return pd.concat([base, ts, lags], axis=1)


class RollingEncoder(Encoder):
    name = "rolling"

    def fit(self, df: pd.DataFrame, train_idx: np.ndarray, y: pd.Series) -> "RollingEncoder":
        # Rolling features summarise movement, so they need GPS position. Scenario 2
        # ships no GPS files, so unit2_lat/unit2_lon are absent there - fail with a
        # clear message rather than a deep KeyError, and don't ship a rolling config
        # for GPS-less scenarios.
        if not {"unit2_lat", "unit2_lon"}.issubset(df.columns):
            raise KeyError(
                "The 'rolling' encoder requires GPS columns unit2_lat/unit2_lon, "
                "which are unavailable for this dataset (e.g. DeepSense Scenario 2 "
                "has no GPS files). Use a position-independent encoder instead."
            )
        # Projection origin learned on TRAIN rows only (the notebook used all rows).
        train = df.iloc[train_idx]
        self.lat0_ = float(train["unit2_lat"].mean())
        self.lon0_ = float(train["unit2_lon"].mean())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        group = self.task.group
        base_cols = self._base_features(df)
        base = self._as_numeric(df[base_cols].copy())
        ts = _timestamp_features(df, group)

        method = self.params.get("projection", DEFAULT_METHOD)
        x, y = to_local_xy(df["unit2_lat"], df["unit2_lon"], self.lat0_, self.lon0_, method)
        mv = pd.DataFrame(index=df.index)
        mv["unit2_x_m"] = x
        mv["unit2_y_m"] = y
        mv["distance_from_ref_m"] = np.sqrt(mv["unit2_x_m"] ** 2 + mv["unit2_y_m"] ** 2)

        joined = pd.concat([df[[group]], mv[["unit2_x_m", "unit2_y_m"]]], axis=1)
        mv["delta_x_m"] = joined.groupby(group)["unit2_x_m"].diff().fillna(0)
        mv["delta_y_m"] = joined.groupby(group)["unit2_y_m"].diff().fillna(0)
        mv["movement_distance_m"] = np.sqrt(mv["delta_x_m"] ** 2 + mv["delta_y_m"] ** 2)
        delta_seconds = ts["delta_seconds"].replace(0, np.nan)
        mv["speed_m_per_s"] = (mv["movement_distance_m"] / delta_seconds).replace([np.inf, -np.inf], np.nan).fillna(0)
        heading = np.arctan2(mv["delta_y_m"], mv["delta_x_m"])
        mv["heading_sin"] = np.sin(heading)
        mv["heading_cos"] = np.cos(heading)

        rolling_source = ["unit2_x_m", "unit2_y_m", "movement_distance_m", "speed_m_per_s"]
        windows = self.params.get("windows", [3, 5])
        stats = self.params.get("stats", ["mean", "std"])
        roll = _rolling_window_stats(mv[rolling_source], df[group], windows, stats)

        return pd.concat([base, ts, mv, roll], axis=1)


class BSGeometryEncoder(Encoder):
    """Base-station-relative geometry - the physical quantity the beam is selected by.

    The base station chooses the beam that points at the UE, so the optimal beam is largely a
    function of the **azimuth (bearing) from the fixed BS (unit1) to the UE (unit2)**. Rather
    than make a model rediscover that angle from raw lat/lon (as the position-aided NN of
    Morais et al. 2022 must), this encoder hands it over directly: the BS->UE bearing (as
    ``sin``/``cos`` to avoid the 0/360-degree wrap), the distance to the BS, and the local
    east/north offset. With ``include_rate=True`` (default) it also adds the per-sequence rate
    of change of that geometry (how the angle/distance evolve along a drive - the tracking cue).

    Deterministic (no fitting), but needs the BS coordinate: requires ``unit1_lat``/``unit1_lon``
    (added by the loader from unit1's GPS) and ``unit2_lat``/``unit2_lon`` (the UE). Standalone
    by default (``include_base=False``) so it is judged purely on the geometry it injects; set
    ``include_base=True`` to also pass through the raw base features.
    """

    name = "bs_geometry"
    _DEF_OFFSETS = True   # include the Cartesian east/north offsets
    _DEF_RATE = True      # include the per-sequence rate-of-change features
    _DEF_ANGLE = True     # include the raw bearing angle (near-linear in the beam index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        needed = {"unit1_lat", "unit1_lon", "unit2_lat", "unit2_lon"}
        if not needed.issubset(df.columns):
            raise KeyError(
                f"The '{self.name}' encoder requires the base-station coordinate "
                "(unit1_lat/unit1_lon) and the UE position (unit2_lat/unit2_lon); "
                "unit1 GPS is unavailable for this scenario."
            )
        # Local ENU offset of the UE relative to the BS, via the selected projection
        # (true 'utm' by default, or the 'equirectangular' small-angle approx).
        method = self.params.get("projection", DEFAULT_METHOD)
        east, north = to_local_xy(
            df["unit2_lat"].astype(float), df["unit2_lon"].astype(float),
            df["unit1_lat"].astype(float), df["unit1_lon"].astype(float), method,
        )
        east = pd.Series(east, index=df.index)
        north = pd.Series(north, index=df.index)
        bearing = np.arctan2(east, north)  # compass bearing of the UE as seen from the BS

        # Core polar coordinates of the UE about the BS: range + direction.
        out = pd.DataFrame(index=df.index)
        out["bs_distance_m"] = np.sqrt(east ** 2 + north ** 2)
        out["bs_bearing_sin"] = np.sin(bearing)
        out["bs_bearing_cos"] = np.cos(bearing)

        if bool(self.params.get("include_angle", self._DEF_ANGLE)):
            # The raw bearing angle is near-linear in the beam index (the beam is ~a quantised
            # azimuth), which the sin/cos pair hides - handing it over directly lets flexible
            # learners exploit that linearity. It wraps at +/-pi, but these single-BS scenes
            # span <180 deg from the BS so no wrap occurs; sin/cos remain for wrap-safety.
            out["bs_bearing_rad"] = bearing

        if bool(self.params.get("include_offsets", self._DEF_OFFSETS)):
            out["bs_east_m"] = east
            out["bs_north_m"] = north

        if bool(self.params.get("include_rate", self._DEF_RATE)):
            group = self.task.group
            cols = ["bs_bearing_sin", "bs_bearing_cos", "bs_distance_m"]
            g = pd.concat([df[[group]], out[cols]], axis=1)
            out["bs_bearing_sin_delta"] = g.groupby(group)["bs_bearing_sin"].diff().fillna(0.0)
            out["bs_bearing_cos_delta"] = g.groupby(group)["bs_bearing_cos"].diff().fillna(0.0)
            out["bs_distance_delta"] = g.groupby(group)["bs_distance_m"].diff().fillna(0.0)

        # Optional lags of the core geometry: the same quantities from the previous
        # ``lag_steps`` readings of the sequence (NaN at sequence starts, left for the
        # pipeline imputer). Off by default (``lag_steps=[]``); lets a candidate encoding
        # mix short-memory context with the instantaneous geometry. Available to the whole
        # BS family (bs_geometry / bs_bearing / bs_rolling) via params.
        lag_steps = self.params.get("lag_steps", [])
        if lag_steps:
            group = self.task.group
            lag_cols = ["bs_distance_m", "bs_bearing_sin", "bs_bearing_cos"]
            g = pd.concat([df[[group]], out[lag_cols]], axis=1)
            for lag in lag_steps:
                for col in lag_cols:
                    out[f"{col}_lag{lag}"] = g.groupby(group)[col].shift(lag)

        if bool(self.params.get("include_base", False)):
            base = self._as_numeric(df[self._base_features(df)].copy())
            out = pd.concat([base, out], axis=1)
        return out


class BSBearingEncoder(BSGeometryEncoder):
    """Minimal BS-relative encoding: only ``bs_distance_m`` + bearing (``sin``/``cos``) - the
    pure polar coordinates of the UE about the base station, with no Cartesian offsets and no
    rate-of-change features. Tests whether the bare physics (range + direction) is a *leaner,
    stronger* representation than the fuller ``bs_geometry`` or raw lat/lon."""

    name = "bs_bearing"
    _DEF_OFFSETS = False
    _DEF_RATE = False


class BSRollingEncoder(BSGeometryEncoder):
    """The best-performing **location** encoding (BS-relative geometry) **plus time context as
    rolling-window statistics** of that geometry along each drive.

    Rather than raw lat/lon, the position is the BS->UE polar geometry of
    :class:`BSGeometryEncoder` - ``bs_distance_m`` and the bearing ``sin``/``cos`` (+ the raw
    bearing angle), the representation that tops the ``morais_nn`` comparison. The temporal
    context is then added the way the ``rolling`` encoder does, but on this physically
    meaningful geometry instead of raw movement: per-sequence rolling **mean** and **std**
    (windows 3 & 5 by default) of ``bs_distance_m`` / ``bs_bearing_sin`` / ``bs_bearing_cos``.
    The mean smooths GPS jitter on the position cue; the std captures how steady the range /
    bearing are over the recent window (a tracking-stability signal). The single-step diffs of
    plain ``bs_geometry`` are dropped (``include_rate=False``) since the rolling stats subsume
    that local-dynamics role.

    Params: ``windows`` (default ``[3, 5]``), ``stats`` (default ``["mean", "std"]``), plus the
    inherited ``bs_geometry`` switches (``include_angle``/``include_offsets``/``include_rate``/
    ``include_base``). Needs unit1 + unit2 GPS (same requirement as ``bs_geometry``).
    """

    name = "bs_rolling"
    _DEF_OFFSETS = False   # lean geometry; the rolling stats carry the temporal signal instead
    _DEF_RATE = False      # single-step diffs are superseded by rolling mean/std

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        geom = super().transform(df)  # bs_distance_m, bs_bearing_sin/cos [, bs_bearing_rad]
        roll_source = ["bs_distance_m", "bs_bearing_sin", "bs_bearing_cos"]
        windows = self.params.get("windows", [3, 5])
        stats = self.params.get("stats", ["mean", "std"])
        roll = _rolling_window_stats(geom[roll_source], df[self.task.group], windows, stats)
        return pd.concat([geom, roll], axis=1)


# --------------------------------------------------------------------------- #
# Numerical-feature encoders (literature encodings for *continuous* features).
# Non-temporal / per-row: they re-represent the continuous DeepSense position
# (unit2_lat / unit2_lon) WITHOUT discarding its order or metric.
# --------------------------------------------------------------------------- #
class _PositionFeatureEncoder(Encoder):
    """Shared plumbing for the numeric position encoders: resolves the continuous source
    columns (default ``unit2_lat`` / ``unit2_lon``) and reports them for representation stats.
    Override the columns with ``params['features']``."""

    _DEFAULT_FEATURES = ["unit2_lat", "unit2_lon"]

    def _features(self, df: pd.DataFrame) -> list[str]:
        requested = self.params.get("features", self._DEFAULT_FEATURES)
        cols = [c for c in requested if c in df.columns]
        if not cols:
            raise KeyError(
                f"The {self.name!r} encoder needs continuous feature columns {requested}, "
                "none of which are present in this scenario."
            )
        return cols

    def n_input_features(self, df: pd.DataFrame) -> int:
        return len(self._features(df))


class PiecewiseLinearEncoder(_PositionFeatureEncoder):
    """Piecewise-linear encoding (PLE) of Gorishniy et al., "On Embeddings for Numerical
    Features in Tabular Deep Learning" (NeurIPS 2022).

    Each continuous feature is split into ``n_bins`` quantile bins with edges
    ``b_0 < ... < b_T`` fit on the TRAIN rows. A value ``x`` is encoded as a T-dim vector whose
    t-th component is the *fraction of bin t that x has passed*::

        PLE_t(x) = clip( (x - b_{t-1}) / (b_t - b_{t-1}), 0, 1 )

    so the components saturate to 1 left-to-right as ``x`` grows - a monotone "thermometer with
    a linear ramp in the active bin". Unlike one-hot-on-bins this **preserves the ordering and
    metric** of the original feature (the property nominal/target binning destroys), while still
    giving a flexible bin-local representation. Deterministic given the train-fit edges.
    ``n_bins`` default 10; ``features`` selects the source columns (default unit2_lat/lon)."""

    name = "ple"

    def fit(self, df, train_idx, y):
        n_bins = int(self.params.get("n_bins", 10))
        train = df.iloc[np.asarray(train_idx)]
        self._edges_: dict[str, np.ndarray] = {}
        for col in self._features(df):
            x = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype=float)
            x = x[~np.isnan(x)]
            if len(x) == 0:
                edges = np.array([0.0, 1.0])
            else:
                edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
            if len(edges) < 2:  # constant feature -> a single trivial bin
                edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
            self._edges_[col] = edges
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self._features(df):
            edges = self._edges_[col]
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)[:, None]
            lo, hi = edges[:-1], edges[1:]
            comp = np.clip((x - lo) / (hi - lo), 0.0, 1.0)  # (n, T)
            for t in range(comp.shape[1]):
                out[f"{col}_ple{t}"] = comp[:, t]
        return out


class PeriodicEncoder(_PositionFeatureEncoder):
    """Periodic / Fourier-feature encoding of continuous features - the "periodic" embedding of
    Gorishniy et al. (NeurIPS 2022) and the Gaussian random Fourier features of Tancik et al.,
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    (NeurIPS 2020). The optimal beam is a *high-frequency* function of a *low-dimensional*
    coordinate (lat/lon) - exactly the regime where Fourier features help.

    Each feature is min-max scaled to [0, 1] (range fit on TRAIN), then mapped to
    ``2 * n_frequencies`` columns ``[sin(2π f_i x), cos(2π f_i x)]`` with frequencies
    ``f_i ~ N(0, sigma^2)`` drawn once with a fixed seed. Generalises the single sin/cos pair the
    geometry encoders already use for the BS bearing to a multi-frequency basis over raw
    position. ``n_frequencies`` default 6, ``sigma`` default 2.0; ``features`` default
    unit2_lat/lon."""

    name = "periodic"

    def fit(self, df, train_idx, y):
        k = int(self.params.get("n_frequencies", 6))
        sigma = float(self.params.get("sigma", 2.0))
        seed = int(self.params.get("random_state", DEFAULT_RANDOM_STATE))
        rng = np.random.default_rng(seed)
        train = df.iloc[np.asarray(train_idx)]
        self._minmax_: dict[str, tuple[float, float]] = {}
        self._freqs_: dict[str, np.ndarray] = {}
        for col in self._features(df):
            x = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype=float)
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
            self._minmax_[col] = (lo, hi if hi > lo else lo + 1.0)
            self._freqs_[col] = rng.normal(0.0, sigma, size=k)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self._features(df):
            lo, hi = self._minmax_[col]
            xs = (pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) - lo) / (hi - lo)
            for i, f in enumerate(self._freqs_[col]):
                ang = 2 * np.pi * f * xs
                out[f"{col}_sin{i}"] = np.sin(ang)
                out[f"{col}_cos{i}"] = np.cos(ang)
        return out


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, type[Encoder]] = {
    # time-series (DeepSense)
    "baseline": BaselineEncoder,
    "latlon": LatLonEncoder,
    "latlon_time": LatLonTimeEncoder,
    "timestamp": TimestampEncoder,
    "lag_window": LagWindowEncoder,
    "rolling": RollingEncoder,
    "bs_geometry": BSGeometryEncoder,
    "bs_bearing": BSBearingEncoder,
    "bs_rolling": BSRollingEncoder,
    # numerical-feature encoders for continuous position (non-temporal)
    "ple": PiecewiseLinearEncoder,
    "periodic": PeriodicEncoder,
}


def build_encoder(cfg: EncoderConfig, task: TaskConfig) -> Encoder:
    if cfg.name not in _REGISTRY:
        raise ValueError(f"Unknown encoder {cfg.name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[cfg.name](task=task, params=cfg.params)
