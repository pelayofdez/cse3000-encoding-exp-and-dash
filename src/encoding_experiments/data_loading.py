"""Dataset loading, cleaning and validation.

Each loader returns a tidy ``pandas.DataFrame`` plus a small ``dict`` of
metadata. The loaders perform the *shared* preprocessing only - feature
engineering that defines an encoding (timestamp/lag/rolling/...) lives
in :mod:`encoding_experiments.encoders`.

Two defensive correctness rules enforced here (neither currently fires on the
real data, but they make future column drops safe):

* Duplicate-suffix columns (``foo.1`` produced by pandas on repeated headers)
  are dropped.
* After cleaning we assert that no remaining column is identical to the target.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DatasetConfig, TaskConfig
from .logging_utils import get_logger

logger = get_logger()

# Matches the ".1", ".2" suffixes pandas appends to duplicated column names.
_DUP_SUFFIX = re.compile(r"\.\d+$")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _drop_duplicate_suffix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose name ends in a pandas duplicate suffix (``.1`` etc.)."""
    dup_cols = [c for c in df.columns if _DUP_SUFFIX.search(str(c))]
    if dup_cols:
        logger.info("Dropping duplicate-suffix columns: %s", dup_cols)
        df = df.drop(columns=dup_cols)
    return df


def _assert_no_target_copy(df: pd.DataFrame, target: str) -> None:
    """Fail loudly if any feature column is identical to the target column."""
    if target not in df.columns:
        return
    y = df[target]
    for col in df.columns:
        if col == target:
            continue
        try:
            if (df[col].astype(str).values == y.astype(str).values).all():
                raise ValueError(
                    f"Column {col!r} is identical to the target {target!r}; "
                    "this would leak the label. Remove it from the dataset."
                )
        except ValueError as exc:
            if "leak the label" in str(exc):
                raise


def _drop_constant_columns(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    constant_cols = [
        c for c in df.columns
        if c not in keep and df[c].nunique(dropna=False) == 1
    ]
    if constant_cols:
        logger.info("Dropping constant columns: %s", constant_cols)
        df = df.drop(columns=constant_cols)
    return df


# --------------------------------------------------------------------------- #
# DeepSense (any scenario)
# --------------------------------------------------------------------------- #
def _find_single(root: Path, pattern: str) -> Path | None:
    """Find one match for ``pattern`` under ``root``, preferring the shallowest.

    Picking the shallowest path makes resolution robust to recursive/duplicate
    directory symlinks (e.g. a ``datasets`` symlink nested inside the dataset
    folder), which would otherwise let ``rglob`` return a deep duplicate first.
    """
    matches = sorted(root.rglob(pattern), key=lambda p: (len(p.parts), str(p)))
    return matches[0] if matches else None


def _parse_custom_timestamp_seconds(value: object) -> float:
    """Parse a ``['HH-MM-SS-mmm']`` or ``['HH-MM-SS']`` timestamp into absolute seconds.

    Scenarios differ in precision: most ship ``HH-MM-SS-mmm`` (4 parts), but some
    (e.g. Scenario 6) ship ``HH-MM-SS`` (3 parts, no milliseconds). Both are accepted;
    the milliseconds default to 0 when absent.
    """
    numbers = re.findall(r"\d+", str(value))
    if len(numbers) == 4:
        hour, minute, second, millisecond = (int(n) for n in numbers)
    elif len(numbers) == 3:
        hour, minute, second = (int(n) for n in numbers)
        millisecond = 0
    else:
        raise ValueError(f"Unexpected timestamp format: {value!r}")
    return hour * 3600 + minute * 60 + second + millisecond / 1000.0


def _load_gps(df: pd.DataFrame, scenario_root: Path, gps_path_col: str) -> pd.DataFrame:
    """Resolve the per-row GPS text files and add ``unit2_lat`` / ``unit2_lon``.

    Reads are cached by the (unique) relative path so repeated values are only
    read once.
    """
    cache: dict[str, tuple[float, float]] = {}

    def resolve(rel: str) -> Path:
        text = str(rel).strip().replace("./", "")
        candidate = scenario_root / text
        if candidate.exists():
            return candidate
        found = _find_single(scenario_root, Path(text).name)
        if found is None:
            raise FileNotFoundError(f"Could not resolve GPS path {rel!r}")
        return found

    def read(rel: str) -> tuple[float, float]:
        if rel in cache:
            return cache[rel]
        path = resolve(rel)
        values = [float(v.strip()) for v in path.read_text().splitlines() if v.strip()]
        if len(values) != 2:
            raise ValueError(f"Expected 2 GPS values in {path}, found {len(values)}")
        cache[rel] = (values[0], values[1])
        return cache[rel]

    latlon = df[gps_path_col].map(read)
    df = df.copy()
    df["unit2_lat"] = [p[0] for p in latlon]
    df["unit2_lon"] = [p[1] for p in latlon]
    return df


def _load_bs_location(scenario_root: Path) -> tuple[float, float] | None:
    """Resolve the fixed base-station (unit1) GPS coordinate, if present.

    The BS does not move, so its position is a single ``unit1/GPS_data/gps_location.txt``
    (two lines: lat, lon) rather than one file per row. Returns ``(lat, lon)`` or ``None``
    when unavailable (some scenarios ship no unit1 GPS). ``gps_location.txt`` (no index) is
    distinct from the per-row ``gps_location_<n>.txt`` files, so the search is unambiguous.
    """
    candidate: Path | None = scenario_root / "unit1" / "GPS_data" / "gps_location.txt"
    if not candidate.exists():
        candidate = _find_single(scenario_root, "gps_location.txt")
    if candidate is None or not candidate.exists():
        return None
    values = [float(v.strip()) for v in candidate.read_text().splitlines() if v.strip()]
    if len(values) != 2:
        return None
    return values[0], values[1]


def _derive_beam_index(df: pd.DataFrame, scenario_root: Path,
                        power_col: str, target_col: str) -> pd.DataFrame:
    """Derive ``target_col`` as ``argmax(power_vector) + 1`` (1-indexed) when it is absent.

    Used for Scenario 4, which ships ``unit1_pwr_60ghz`` power files but no
    ``unit1_beam_index`` column. The argmax of the 64-element power vector is the
    optimal beam, exactly matching the definition used in all other scenarios.
    """
    def resolve(rel: str) -> Path:
        text = str(rel).strip().replace("./", "")
        candidate = scenario_root / text
        if candidate.exists():
            return candidate
        found = _find_single(scenario_root, Path(text).name)
        if found is None:
            raise FileNotFoundError(f"Could not resolve power file {rel!r}")
        return found

    beam_indices = []
    for rel in df[power_col]:
        vec = np.asarray(resolve(rel).read_text().split(), dtype=float)
        beam_indices.append(int(np.argmax(vec)) + 1)  # 1-indexed to match other scenarios
    df = df.copy()
    df[target_col] = beam_indices
    logger.info("Derived %s from %s (argmax + 1) for %d rows.", target_col, power_col, len(df))
    return df


def load_deepsense(
    dataset: DatasetConfig,
    task: TaskConfig,
    timestamp_col: str = "time_stamp[UTC]",
    gps_path_col: str = "unit2_loc",
) -> tuple[pd.DataFrame, dict]:
    """Load any DeepSense scenario CSV (Scenario1, Scenario2, ...).

    The scenario is selected entirely by ``dataset.scenario_name`` and
    ``dataset.path``; the body is scenario-agnostic. GPS lat/lon are loaded only
    when the per-row GPS text files are present (Scenario 1 has them, Scenario 2
    does not), so position-dependent encoders must tolerate their absence.
    Scenarios that ship no ``unit1_beam_index`` column (e.g. Scenario 4) have it
    derived automatically from ``unit1_pwr_60ghz`` (argmax + 1).
    """
    data_dir = Path(dataset.path)
    # Prefer the expected direct location; fall back to a shallow recursive search.
    csv_path = data_dir / f"{dataset.scenario_name}.csv"
    if not csv_path.exists():
        csv_path = _find_single(data_dir, f"{dataset.scenario_name}.csv")
    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find {dataset.scenario_name}.csv under {data_dir}"
        )
    logger.info("Loading %s", csv_path)
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 5", ""], errors="ignore")
    df = _drop_duplicate_suffix_columns(df)

    # Derive the beam-index target from power files when the column is absent.
    # Must happen before _drop_constant_columns so the derived target survives the keep list.
    if task.target not in df.columns and "unit1_pwr_60ghz" in df.columns:
        df = _derive_beam_index(df, csv_path.parent, "unit1_pwr_60ghz", task.target)

    # Prefer the calibrated UE GPS track when the scenario ships one. Scenarios 8 & 9 carry a
    # ``unit2_loc_cal`` column (pointing at ``unit2/GPS_data_calibrated/``) next to the raw
    # ``unit2_loc``; the calibrated fixes correct a systematic GPS offset, so they are the
    # positions to use whenever present. No other scenario has this column, so this is a no-op
    # everywhere else. An explicit, non-default ``gps_path_col`` still wins (caller override).
    if gps_path_col == "unit2_loc" and "unit2_loc_cal" in df.columns \
            and df["unit2_loc_cal"].notna().any():
        gps_path_col = "unit2_loc_cal"
        logger.info("Using calibrated UE GPS column 'unit2_loc_cal' for %s.", dataset.scenario_name)

    keep = [c for c in [task.target, task.group, timestamp_col, gps_path_col] if c]
    df = _drop_constant_columns(df, keep=keep)

    # Resolve GPS lat/lon from the per-row text files. The scenario root is the
    # directory that contains ``unit2/GPS_data``; prefer the CSV's own folder.
    scenario_root = None
    if (csv_path.parent / "unit2" / "GPS_data").exists():
        scenario_root = csv_path.parent
    else:
        gps_file = _find_single(data_dir, "gps_location_0.txt")
        if gps_file is not None:
            scenario_root = gps_file.parent.parent.parent

    if scenario_root is not None and gps_path_col in df.columns:
        df = _load_gps(df, scenario_root, gps_path_col)
    else:
        logger.warning("GPS files not found; unit2_lat/unit2_lon will be unavailable.")

    # Fixed base-station (unit1) position, for BS-relative geometry encoders. Constant across
    # rows (the BS does not move), so it is added AFTER the constant-column drop above.
    if scenario_root is not None:
        bs = _load_bs_location(scenario_root)
        if bs is not None:
            df["unit1_lat"], df["unit1_lon"] = bs[0], bs[1]
        else:
            logger.warning("Base-station (unit1) GPS not found; unit1_lat/unit1_lon unavailable.")

    # Absolute time in seconds, then a stable per-sequence ordering.
    df["time_seconds_absolute"] = df[timestamp_col].map(_parse_custom_timestamp_seconds)
    sort_cols = [c for c in [task.group, "time_seconds_absolute"] if c]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Target -> int, drop rows with missing target/group.
    df[task.target] = pd.to_numeric(df[task.target], errors="coerce")
    subset = [c for c in [task.target, task.group] if c]
    df = df.dropna(subset=subset).reset_index(drop=True)
    df[task.target] = df[task.target].astype(int)

    _assert_no_target_copy(df, task.target)
    has_gps = "unit2_lat" in df.columns
    meta = {"csv_path": str(csv_path), "n_rows": len(df), "has_gps": has_gps}
    logger.info(
        "DeepSense %s loaded: %s rows, %s columns (gps=%s)",
        dataset.scenario_name, df.shape[0], df.shape[1], has_gps,
    )
    return df, meta


def load_deepsense_power(
    dataset: DatasetConfig,
    df: pd.DataFrame,
    power_col: str = "unit1_pwr_60ghz",
) -> np.ndarray:
    """Load the per-sample mmWave receive-power vectors, aligned row-for-row to ``df``.

    Each DeepSense row points (via ``unit1_pwr_60ghz``) to a text file holding the received
    power of every codebook beam swept for that sample. These are required for the
    power-loss metric: the beam *index* alone is not enough - we need the actual power at
    the predicted vs. the optimal beam. Vectors are cached on disk keyed by their relative
    path, so the thousands of small reads happen only once per scenario (the cache is shared
    across runs and survives the row re-ordering done in :func:`load_deepsense`, because the
    key is the path, not the row position).

    Returns an ``(n_rows, n_beams)`` float array in the same order as ``df``. Note DeepSense
    beam labels are 1-indexed while these vectors are 0-indexed, so the power of beam ``b`` is
    ``matrix[:, b - 1]`` (see ``power_loss_db``).
    """
    data_dir = Path(dataset.path)
    csv_path = data_dir / f"{dataset.scenario_name}.csv"
    if not csv_path.exists():
        csv_path = _find_single(data_dir, f"{dataset.scenario_name}.csv")
    if csv_path is None:
        raise FileNotFoundError(f"Could not find {dataset.scenario_name}.csv under {data_dir}")
    scenario_root = csv_path.parent

    if power_col not in df.columns:
        raise KeyError(f"Power column {power_col!r} not present in the DeepSense frame.")

    cache_path = data_dir / "_power_cache.pkl"
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        try:
            cache = pd.read_pickle(cache_path)
        except Exception:  # noqa: BLE001 - rebuild on any cache read problem
            cache = {}

    def resolve(rel: str) -> Path:
        text = str(rel).strip().replace("./", "")
        candidate = scenario_root / text
        if candidate.exists():
            return candidate
        found = _find_single(scenario_root, Path(text).name)
        if found is None:
            raise FileNotFoundError(f"Could not resolve power file {rel!r}")
        return found

    vectors: list[np.ndarray] = []
    new_reads = 0
    for rel in df[power_col]:
        key = str(rel)
        vec = cache.get(key)
        if vec is None:
            vec = np.asarray(resolve(rel).read_text().split(), dtype=float)
            cache[key] = vec
            new_reads += 1
        vectors.append(vec)

    lengths = {v.shape[0] for v in vectors}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent beam-vector lengths across power files: {sorted(lengths)}")

    if new_reads:
        try:
            pd.to_pickle(cache, cache_path)
        except Exception as exc:  # noqa: BLE001 - caching is best-effort
            logger.warning("Could not write power cache (%s); continuing.", exc)

    matrix = np.vstack(vectors)
    logger.info("Loaded power matrix %s for %s (%d new file reads).",
                matrix.shape, dataset.scenario_name, new_reads)
    return matrix


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
# Every DeepSense scenario shares one loader (the scenario is chosen by
# ``dataset.scenario_name``), so adding a new scenario needs only a new config - no code
# change here. To onboard a *different* dataset family, add one loader above and dispatch
# to it here.
def load_dataset(dataset: DatasetConfig, task: TaskConfig) -> tuple[pd.DataFrame, dict]:
    if dataset.kind.startswith("deepsense_"):
        return load_deepsense(dataset, task)
    raise ValueError(
        f"Unknown dataset kind: {dataset.kind!r}. Expected a 'deepsense_*' kind."
    )
