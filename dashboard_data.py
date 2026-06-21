"""Data helpers for the linked-results dashboard.

Every result CSV in the ``results/`` folder shares a ``run_id`` column. The
wide tables (performance, representation quality, runtime) are merged on that
key into a single row-per-evaluation dataframe, while the long-format feature
importance table (many ranked rows per run) is loaded separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


# --- column / value vocabulary -------------------------------------------------

LINK_KEY = "run_id"
RESULTS_DIRECTORY_NAME = "results"
SEED_COLUMN_HINTS = ("random_state", "seed")

# Long-format feature-importance tables carry these two columns together.
FEATURE_COLUMN = "feature"
IMPORTANCE_COLUMN = "importance"

DATETIME_NAMES = ("date", "datetime", "time", "created", "updated", "recorded")

# Feature-representation probes (the "feature representation" results). Each probe type is a
# CSV under ``representation_probes/<scenario>/<probe>.csv`` keyed by (scenario, encoder).
REPRESENTATION_PROBES_DIRECTORY_NAME = "representation_probes"
PROBE_TYPES = ("invariance",)

# Numeric columns that are bookkeeping counts / identifiers rather than metrics.
NON_METRIC_HINTS = ("n_train", "n_classes", "n_eval", "n_unseen", "n_features_total")
IDENTIFIER_NAMES = ("run_id", "random_state", "seed", "fold", "rank", "index")

# Substrings that mean "lower is better" for sort direction.
LOWER_IS_BETTER_HINTS = ("loss", "error", "latency", "cost", "time", "memory", "seconds")


# --- generic helpers -----------------------------------------------------------

def pretty_label(value: object) -> str:
    """Convert a snake_case value into a compact UI label."""

    return str(value).strip().replace("_", " ").title()


def _contains_hint(value: object, hints: Iterable[str]) -> bool:
    normalized = str(value).casefold()
    return any(hint in normalized for hint in hints)


def _looks_like_identifier(column: str) -> bool:
    normalized = column.casefold()
    return (
        normalized in IDENTIFIER_NAMES
        or normalized.endswith("_id")
        or normalized.endswith("_index")
    )


def _looks_like_datetime_column(column: str) -> bool:
    normalized = column.casefold()
    return (
        normalized in DATETIME_NAMES
        or "timestamp" in normalized
        or normalized.endswith(("_at", "_date", "_datetime"))
        or normalized.startswith(("date_", "datetime_"))
    )


def _unique_column_names(columns: Iterable[object]) -> list[str]:
    """Return stripped, non-empty, unique column names."""

    unique_names: list[str] = []
    used_names: set[str] = set()

    for index, column in enumerate(columns, start=1):
        base_name = str(column).strip() or f"column_{index}"
        name = base_name
        suffix = 2
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(name)
        unique_names.append(name)

    return unique_names


def prepare_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and parse timestamp-like text columns."""

    prepared = dataframe.copy()
    prepared.columns = _unique_column_names(prepared.columns)

    for column in prepared.select_dtypes(include=["object", "string"]).columns:
        if not _looks_like_datetime_column(column):
            continue
        non_null_count = int(prepared[column].notna().sum())
        if not non_null_count:
            continue
        converted = pd.to_datetime(prepared[column], errors="coerce", utc=True)
        if converted.notna().sum() / non_null_count >= 0.8:
            prepared[column] = converted

    return prepared


# --- loading and merging -------------------------------------------------------

def discover_linked_result_csvs(directory: Path) -> tuple[Path, ...]:
    """Return CSVs in ``directory`` whose header contains the link key column."""

    if not directory.is_dir():
        return ()

    matches: list[Path] = []
    for path in sorted(directory.glob("*.csv"), key=lambda p: p.name.casefold()):
        try:
            header = pd.read_csv(path, nrows=0).columns
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        if LINK_KEY in header:
            matches.append(path)
    return tuple(matches)


def _is_long_feature_table(frame: pd.DataFrame) -> bool:
    """Detect a long-format feature-importance table (many rows per run)."""

    return FEATURE_COLUMN in frame.columns and IMPORTANCE_COLUMN in frame.columns


def load_linked_results(
    directory: Path,
    *,
    link_key: str = LINK_KEY,
) -> tuple[pd.DataFrame, Mapping[str, tuple[str, ...]]]:
    """Merge the wide result CSVs in ``directory`` on ``link_key``.

    Long-format feature-importance tables are excluded here (load them with
    :func:`load_feature_importance`) so they cannot explode the wide merge. The
    table with the most rows per run (e.g. one row per evaluation split) is used
    as the merge base; single-row-per-run sources are left-joined onto it.
    """

    frames: list[tuple[str, pd.DataFrame]] = []
    for path in discover_linked_result_csvs(directory):
        frame = prepare_dataframe(pd.read_csv(path))
        if link_key not in frame.columns or frame.empty:
            continue
        if _is_long_feature_table(frame):
            continue
        frames.append((path.name, frame))

    if not frames:
        return pd.DataFrame(), {}

    rows_per_id = {
        name: frame.groupby(link_key, dropna=False).size().max()
        for name, frame in frames
    }
    frames.sort(key=lambda item: rows_per_id[item[0]], reverse=True)

    base_name, base_frame = frames[0]
    merged = base_frame.copy()
    contributions: dict[str, tuple[str, ...]] = {
        base_name: tuple(column for column in base_frame.columns if column != link_key)
    }

    for name, frame in frames[1:]:
        shared = [
            column
            for column in frame.columns
            if column != link_key and column in merged.columns
        ]
        right = frame.drop(columns=shared)
        merged = merged.merge(right, on=link_key, how="left")
        contributions[name] = tuple(column for column in right.columns if column != link_key)

    return merged, contributions


def load_feature_importance(
    directory: Path,
    *,
    link_key: str = LINK_KEY,
) -> pd.DataFrame:
    """Load and concatenate the long-format feature-importance table(s)."""

    frames: list[pd.DataFrame] = []
    for path in discover_linked_result_csvs(directory):
        frame = prepare_dataframe(pd.read_csv(path))
        if link_key in frame.columns and not frame.empty and _is_long_feature_table(frame):
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --- seeds ---------------------------------------------------------------------

def seed_column(dataframe: pd.DataFrame) -> str | None:
    """Return the seed-like column in ``dataframe`` if present."""

    for hint in SEED_COLUMN_HINTS:
        if hint in dataframe.columns:
            return hint
    for column in dataframe.columns:
        if _contains_hint(column, SEED_COLUMN_HINTS):
            return column
    return None


def available_seeds(dataframe: pd.DataFrame) -> tuple[object, ...]:
    """Return the sorted set of seed values observed in ``dataframe``."""

    column = seed_column(dataframe)
    if column is None:
        return ()
    values = dataframe[column].dropna().unique().tolist()
    try:
        values.sort()
    except TypeError:
        values.sort(key=str)
    return tuple(values)


# --- metrics / aggregation -----------------------------------------------------

def metric_value_columns(dataframe: pd.DataFrame) -> tuple[str, ...]:
    """Return numeric metric columns useful for mean/std aggregation."""

    columns: list[str] = []
    for column in dataframe.columns:
        series = dataframe[column]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        if not series.notna().any():
            continue
        if _looks_like_identifier(column) or _contains_hint(column, SEED_COLUMN_HINTS):
            continue
        if _contains_hint(column, NON_METRIC_HINTS):
            continue
        columns.append(column)
    return tuple(columns)


def metric_higher_is_better(metric: str) -> bool:
    """Return whether increasing a metric usually indicates an improvement."""

    return not _contains_hint(metric, LOWER_IS_BETTER_HINTS)


def aggregate_runs(
    dataframe: pd.DataFrame,
    group_columns: Iterable[str],
    metric_columns: Iterable[str],
) -> pd.DataFrame:
    """Aggregate ``metric_columns`` per group with mean, std, count and seeds.

    Every numeric metric yields ``<metric>_mean`` and ``<metric>_std`` columns.
    ``runs`` counts the rows in each group and ``seeds`` lists the unique seeds
    that contributed when a seed column is available.
    """

    selected_groups = [column for column in group_columns if column in dataframe.columns]
    selected_metrics = [
        column
        for column in metric_columns
        if column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    if not selected_groups or not selected_metrics or dataframe.empty:
        return pd.DataFrame()

    grouped = dataframe.groupby(selected_groups, dropna=False)
    mean_frame = grouped[selected_metrics].mean()
    std_frame = grouped[selected_metrics].std(ddof=0)
    count_frame = grouped.size().rename("runs")

    aggregated = mean_frame.add_suffix("_mean").join(std_frame.add_suffix("_std"))
    aggregated = aggregated.join(count_frame)

    seed = seed_column(dataframe)
    if seed is not None:
        aggregated["seeds"] = grouped[seed].apply(
            lambda values: sorted(values.dropna().unique().tolist())
        )

    aggregated = aggregated.reset_index()
    aggregated["configuration"] = aggregated[selected_groups].astype("string").agg(
        " + ".join, axis=1
    )
    return aggregated


# --- datasets ------------------------------------------------------------------

def available_datasets(
    dataframe: pd.DataFrame, *, dataset_column: str = "dataset"
) -> tuple[str, ...]:
    """Return the sorted unique dataset names present in ``dataframe``."""

    if dataframe.empty or dataset_column not in dataframe.columns:
        return ()
    return tuple(sorted(dataframe[dataset_column].dropna().astype(str).unique(), key=str))


# --- feature-representation probes ---------------------------------------------

def load_representation_probes(directory: Path) -> dict[str, pd.DataFrame]:
    """Load the feature-representation probe CSVs into one frame per probe type.

    The probes live under ``<directory>/<scenario>/<probe>.csv`` (written by
    ``experiments/representation_probe.py``) and are keyed by ``(scenario, encoder)`` plus a
    per-probe axis (``factor`` / ``sigma_m`` / ``transform``). Each probe type's CSVs are
    concatenated across scenarios; every row already carries a ``scenario`` column, so the
    folder name is only a fallback. Returns ``{probe_type: dataframe}``, skipping any probe
    type with no CSVs on disk.
    """

    if not directory.is_dir():
        return {}

    frames: dict[str, list[pd.DataFrame]] = {probe: [] for probe in PROBE_TYPES}
    for scenario_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        for probe in PROBE_TYPES:
            path = scenario_dir / f"{probe}.csv"
            if not path.is_file():
                continue
            try:
                frame = prepare_dataframe(pd.read_csv(path))
            except (OSError, pd.errors.ParserError, UnicodeDecodeError):
                continue
            if frame.empty:
                continue
            if "scenario" not in frame.columns:
                frame = frame.assign(scenario=scenario_dir.name)
            frames[probe].append(frame)

    return {
        probe: pd.concat(parts, ignore_index=True)
        for probe, parts in frames.items()
        if parts
    }
