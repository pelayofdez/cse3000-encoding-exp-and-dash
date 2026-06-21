"""Single-page Streamlit dashboard for linked encoding-experiment results.

The wide result CSVs in the ``results/`` folder are merged on ``run_id`` and shown as one
table. You choose a dataset (DeepSense scenario) and a model, then compare encodings:
vertical bar charts use the encoder name on the x-axis and show the mean with ±1
standard-deviation error bars across the selected seeds, separated by evaluation split. The
page also renders the top features from the long-format feature-importance table and a
**Feature Representation** section driven by the encoder representation-quality probes under
``representation_probes/``. Every quantitative chart has an optional y-axis range control for
zooming in on small differences.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dashboard_data import (
    PROBE_TYPES,
    REPRESENTATION_PROBES_DIRECTORY_NAME,
    RESULTS_DIRECTORY_NAME,
    aggregate_runs,
    available_datasets,
    available_seeds,
    load_feature_importance,
    load_linked_results,
    load_representation_probes,
    metric_higher_is_better,
    metric_value_columns,
    pretty_label,
    seed_column,
)


APP_DIRECTORY = Path(__file__).resolve().parent
RESULTS_DIRECTORY = APP_DIRECTORY / RESULTS_DIRECTORY_NAME
REPRESENTATION_PROBES_DIRECTORY = APP_DIRECTORY / REPRESENTATION_PROBES_DIRECTORY_NAME
PERFORMANCE_SOURCE = "metrics.csv"
ENCODER_COLUMN = "encoder"
SOURCE_SECTION_TITLES = {
    "metrics.csv": "Performance",
    "representation_metrics.csv": "Feature Quality (Representation)",
    "runtime.csv": "Runtime / Cost",
}
PERFORMANCE_DEFAULT_METRICS = ("top_k_accuracy", "dba_score", "power_loss_db")
PREFERRED_SPLIT_ORDER = ("val", "test")
FEATURE_COUNT_METRIC = "n_features_after_encoding"
MODEL_COLUMN = "model"
DATASET_COLUMN = "dataset"
DEFAULT_TOP_FEATURES = 15

# Per-probe metric vocabulary for the Feature Representation section.
PROBE_SECTION_TITLES = {
    "informativeness": "Informativeness - recovering position factors",
    "invariance": "Invariance - robustness to GPS noise",
}
INVARIANCE_METRICS = ("d_acc", "d_dba", "d_ploss_db", "cos_drift", "l2_drift")


st.set_page_config(page_title="6G Encoding Results Dashboard", page_icon=":bar_chart:", layout="wide")


@st.cache_data(show_spinner=False)
def load_linked_results_cached(
    directory: str, fingerprint: tuple[tuple[str, int], ...]
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Load the merged linked-results dataframe, refreshing on file changes."""

    del fingerprint
    merged, contributions = load_linked_results(Path(directory))
    return merged, dict(contributions)


@st.cache_data(show_spinner=False)
def load_feature_importance_cached(
    directory: str, fingerprint: tuple[tuple[str, int], ...]
) -> pd.DataFrame:
    """Load the long-format feature-importance table, refreshing on changes."""

    del fingerprint
    return load_feature_importance(Path(directory))


@st.cache_data(show_spinner=False)
def load_representation_probes_cached(
    directory: str, fingerprint: tuple[tuple[str, int], ...]
) -> dict[str, pd.DataFrame]:
    """Load the feature-representation probe frames, refreshing on changes."""

    del fingerprint
    return load_representation_probes(Path(directory))


def linked_results_fingerprint(directory: Path) -> tuple[tuple[str, int], ...]:
    """Return a (name, mtime_ns) tuple per CSV used to invalidate the cache."""

    if not directory.is_dir():
        return ()
    fingerprints: list[tuple[str, int]] = []
    for path in sorted(directory.glob("*.csv"), key=lambda p: p.name.casefold()):
        try:
            fingerprints.append((path.name, path.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(fingerprints)


def representation_probes_fingerprint(directory: Path) -> tuple[tuple[str, int], ...]:
    """Return a (relative-path, mtime_ns) tuple per probe CSV (recursive)."""

    if not directory.is_dir():
        return ()
    fingerprints: list[tuple[str, int]] = []
    for path in sorted(directory.glob("*/*.csv"), key=lambda p: str(p).casefold()):
        try:
            fingerprints.append((path.name + "@" + path.parent.name, path.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(fingerprints)


def apply_seed_filter(merged: pd.DataFrame) -> pd.DataFrame:
    """Render the global seed filter in the sidebar and apply it."""

    st.sidebar.header("Filters")
    seed_col = seed_column(merged)
    seeds = available_seeds(merged)
    if not seeds:
        st.sidebar.caption("No seed column detected.")
        return merged

    chosen = st.sidebar.multiselect(
        f"Seeds ({seed_col})",
        options=list(seeds),
        default=list(seeds),
        key="filter_seeds",
    )
    st.sidebar.caption("Bars aggregate the runs across the selected seeds.")
    if not chosen:
        st.sidebar.warning("Select at least one seed.")
        return merged.iloc[0:0]
    return merged[merged[seed_col].isin(chosen)]


# --- y-axis range control ------------------------------------------------------

def _axis_range_control(container, values, key: str, *, label: str = "y-axis"):
    """Opt-in numeric range control for an Altair quantitative axis.

    Renders a checkbox; when ticked, two number inputs (min / max) prefilled with the data's
    range appear. Returns ``(lo, hi)`` to clamp the axis, or ``None`` for the automatic range.
    """

    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if numeric.empty:
        return None
    lo, hi = float(numeric.min()), float(numeric.max())
    if not (hi > lo):
        pad = abs(hi) * 0.05 or 1.0
        lo, hi = lo - pad, hi + pad
    span = hi - lo
    step = float(f"{span / 50:.1g}") if span > 0 else 0.01
    if step <= 0:
        step = 0.01

    if not container.checkbox(f"Set {label} range", value=False, key=f"{key}__rng"):
        return None
    cols = container.columns(2)
    new_lo = cols[0].number_input(f"{label} min", value=lo, step=step, key=f"{key}__min", format="%.4f")
    new_hi = cols[1].number_input(f"{label} max", value=hi, step=step, key=f"{key}__max", format="%.4f")
    if not (new_hi > new_lo):
        container.caption("Range max must exceed min - showing the automatic range.")
        return None
    return (float(new_lo), float(new_hi))


def _quant_scale(domain):
    """Return an Altair Scale clamped to ``domain``, or ``alt.Undefined`` for automatic."""

    if domain is None:
        return alt.Undefined
    return alt.Scale(domain=list(domain), clamp=True)


# --- tables / merged view ------------------------------------------------------

def render_merged_table(merged: pd.DataFrame, contributions: dict[str, tuple[str, ...]]) -> None:
    """Show the single merged table that combines every linked CSV."""

    st.subheader("Merged results")
    st.caption(
        f"{len(merged):,} rows × {len(merged.columns):,} columns, joined on `run_id` "
        f"from {len(contributions)} source CSV(s)."
    )
    with st.expander("Columns contributed by each source CSV", expanded=False):
        for filename, columns in contributions.items():
            st.markdown(f"- **{filename}** - {', '.join(columns)}")
    st.dataframe(merged, hide_index=True, width="stretch")
    st.download_button(
        "Download merged CSV",
        data=merged.to_csv(index=False).encode("utf-8"),
        file_name="merged_results.csv",
        mime="text/csv",
        key="download_merged",
    )


# --- experiments view ----------------------------------------------------------

def render_experiments(
    merged: pd.DataFrame,
    contributions: dict[str, tuple[str, ...]],
    metric_options: set[str],
    feature_importance: pd.DataFrame,
) -> None:
    """Render the per-dataset / per-model experiment comparison (all runs are time-series)."""

    datasets = list(available_datasets(merged))
    if not datasets:
        st.info("No datasets are present in the results.")
        return

    controls = st.columns(2)
    dataset = controls[0].selectbox(
        "Dataset", options=datasets, format_func=pretty_label, key="dataset"
    )
    scoped = merged[merged[DATASET_COLUMN].astype(str) == str(dataset)]

    model = None
    if MODEL_COLUMN in scoped.columns:
        models = sorted(scoped[MODEL_COLUMN].dropna().astype(str).unique().tolist())
        if models:
            model = controls[1].selectbox(
                "Model", options=models, format_func=pretty_label, key="model"
            )
            scoped = scoped[scoped[MODEL_COLUMN].astype(str) == model]

    if scoped.empty:
        st.warning("No rows match the current filters.")
        return

    key = f"{dataset}__{model}"
    for source_filename, columns in contributions.items():
        source_metrics = [column for column in columns if column in metric_options]
        if not source_metrics:
            continue
        if source_filename == PERFORMANCE_SOURCE:
            render_performance_section(scoped, source_metrics, key)
        else:
            per_run = scoped.drop_duplicates("run_id")
            render_per_run_section(per_run, source_filename, source_metrics, key)

    render_feature_importance_section(feature_importance, dataset=str(dataset), model=model)


def render_performance_section(scoped: pd.DataFrame, performance_metrics: list[str], key: str) -> None:
    """Compare performance metrics together, separated by evaluation split."""

    st.subheader(SOURCE_SECTION_TITLES[PERFORMANCE_SOURCE])
    st.caption(
        "From `metrics.csv`. Bars show the mean across the selected seeds with a ±1 "
        "standard-deviation error bar; metrics are grouped per encoding and the "
        "evaluation splits are shown separately. Every metric is selectable below."
    )

    default_metrics = [
        metric for metric in PERFORMANCE_DEFAULT_METRICS if metric in performance_metrics
    ] or performance_metrics[:1]
    controls = st.columns([3, 2])
    selected = controls[0].multiselect(
        "Performance metrics (shown together)",
        options=performance_metrics,
        default=default_metrics,
        format_func=pretty_label,
        key=f"performance_metrics_{key}",
    )
    has_feature_count = FEATURE_COUNT_METRIC in scoped.columns
    show_feature_count = False
    if has_feature_count:
        show_feature_count = controls[1].checkbox(
            f"Overlay {pretty_label(FEATURE_COUNT_METRIC)} (right axis)",
            value=True,
            key=f"feature_overlay_{key}",
        )
    if not selected:
        st.caption("Pick at least one performance metric.")
        return

    for split in _ordered_splits(scoped):
        subset = scoped if split is None else scoped[scoped["split"].astype(str) == split]
        aggregated = aggregate_runs(subset, [ENCODER_COLUMN], selected)
        if aggregated.empty:
            continue
        st.markdown(f"**{pretty_label(split) if split else 'All rows'}**")
        long_df = _aggregated_to_long(aggregated, selected)
        feature_counts = _feature_count_frame(subset) if has_feature_count else pd.DataFrame()
        encoder_order = _encoder_order_by_features(feature_counts)
        _render_grouped_bars(
            long_df,
            feature_counts if show_feature_count else pd.DataFrame(),
            encoder_order=encoder_order,
            key=f"perf_{key}_{split}",
        )
        _render_mean_std_table(aggregated, selected, encoder_order=encoder_order)


def render_per_run_section(
    per_run: pd.DataFrame,
    source_filename: str,
    source_metrics: list[str],
    key: str,
) -> None:
    """Compare representation-quality or runtime metrics (one value per run)."""

    title = SOURCE_SECTION_TITLES.get(source_filename, pretty_label(Path(source_filename).stem))
    st.subheader(title)
    st.caption(
        f"From `{source_filename}`. One value per run, averaged across the selected "
        "seeds with a ±1 standard-deviation error bar."
    )

    selected = st.multiselect(
        "Metrics",
        options=source_metrics,
        default=source_metrics,
        format_func=pretty_label,
        key=f"metrics_{source_filename}_{key}",
    )
    if not selected:
        st.caption("Pick at least one metric.")
        return

    aggregated = aggregate_runs(per_run, [ENCODER_COLUMN], selected)
    if aggregated.empty:
        st.warning("No rows to aggregate with the current filters.")
        return

    for metric in selected:
        _render_single_metric_bars(aggregated, metric, key=f"perrun_{key}_{source_filename}_{metric}")
    _render_mean_std_table(aggregated, selected)


def render_feature_importance_section(
    feature_importance: pd.DataFrame,
    *,
    dataset: str,
    model: str | None,
) -> None:
    """Show the top features by importance for the selected dataset + model."""

    st.subheader("Feature Importance")
    if feature_importance.empty:
        st.info("No `feature_importance.csv` was found in the results folder.")
        return

    subset = feature_importance[feature_importance[DATASET_COLUMN].astype(str) == dataset]
    if model is not None and MODEL_COLUMN in subset.columns:
        subset = subset[subset[MODEL_COLUMN].astype(str) == str(model)]
    if subset.empty:
        st.info("No feature-importance rows for this dataset and model.")
        return

    kinds = sorted(subset["importance_kind"].dropna().astype(str).unique().tolist())
    kind_note = f" ({', '.join(pretty_label(kind) for kind in kinds)})" if kinds else ""
    st.caption(
        f"Top features ranked by importance{kind_note}. The importance kind is set by "
        "the model - e.g. gain for boosted trees and permutation importance for the others."
    )

    encoders = sorted(subset[ENCODER_COLUMN].dropna().astype(str).unique().tolist())
    if not encoders:
        st.info("No encoders with feature-importance data here.")
        return

    controls = st.columns([3, 2])
    encoder = controls[0].selectbox(
        "Encoder",
        options=encoders,
        format_func=pretty_label,
        key=f"fi_encoder_{dataset}",
    )
    encoder_rows = subset[subset[ENCODER_COLUMN].astype(str) == encoder]
    max_available = int(encoder_rows["rank"].nunique()) if "rank" in encoder_rows.columns else len(encoder_rows)
    max_slider = max(1, min(max_available, 30))
    top_n = controls[1].slider(
        "Top features",
        min_value=1,
        max_value=max_slider,
        value=min(DEFAULT_TOP_FEATURES, max_slider),
        key=f"fi_topn_{dataset}",
    )

    value_column = (
        "importance_normalized"
        if "importance_normalized" in encoder_rows.columns
        else "importance"
    )
    ranked = _top_features(encoder_rows, value_column, top_n)
    if ranked.empty:
        st.info("No ranked features to plot.")
        return

    value_title = pretty_label(value_column)
    domain = _axis_range_control(st, ranked[value_column], key=f"fi_x_{dataset}", label="importance-axis")
    chart = (
        alt.Chart(ranked)
        .mark_bar(cornerRadiusEnd=4, color="#4c78a8")
        .encode(
            x=alt.X(field=value_column, type="quantitative", title=value_title, scale=_quant_scale(domain)),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip(field=value_column, type="quantitative", title=value_title, format=".4f"),
                alt.Tooltip("rank:Q", title="Rank") if "rank" in ranked.columns else alt.Tooltip("feature:N"),
                alt.Tooltip("runs:Q", title="Runs"),
            ],
        )
        .properties(height=min(max(28 * len(ranked), 120), 700))
    )
    st.altair_chart(chart, width="stretch")


# --- feature representation (probes) -------------------------------------------

def render_feature_representation(probes: dict[str, pd.DataFrame]) -> None:
    """Render the encoder representation-quality probes (the 'feature representation' results)."""

    st.header("Feature Representation")
    st.caption(
        "Encoder representation-quality probes (adapted from Plachouras et al. 2025) on the "
        "DeepSense position→beam task, from `representation_probes/`: how well the encoded "
        "features expose ground-truth position factors (informativeness), and how robust the "
        "representation is to GPS noise (invariance)."
    )

    scenarios = sorted(
        {str(s) for frame in probes.values() for s in frame.get("scenario", pd.Series(dtype=str)).unique()}
    )
    if not scenarios:
        st.info("No probe scenarios found.")
        return
    scenario = st.selectbox(
        "Scenario", options=scenarios, format_func=pretty_label, key="probe_scenario"
    )

    renderers = {
        "informativeness": _render_informativeness,
        "invariance": _render_invariance,
    }
    for probe_type in PROBE_TYPES:
        frame = probes.get(probe_type)
        if frame is None:
            continue
        rows = frame[frame["scenario"].astype(str) == scenario]
        if rows.empty:
            continue
        st.subheader(PROBE_SECTION_TITLES.get(probe_type, pretty_label(probe_type)))
        renderers[probe_type](rows, scenario)


def _render_probe_bars(
    rows: pd.DataFrame, metric_columns: list[str], *, key: str, value_title: str
) -> None:
    """Grouped vertical bars per encoder for one or more probe metric columns."""

    present = [m for m in metric_columns if m in rows.columns]
    if rows.empty or not present:
        st.caption("No data to plot.")
        return
    long_df = (
        rows[[ENCODER_COLUMN, *present]]
        .melt(id_vars=[ENCODER_COLUMN], value_vars=present, var_name="metric", value_name="value")
        .dropna(subset=["value"])
    )
    if long_df.empty:
        st.caption("No data to plot.")
        return
    long_df["metric"] = long_df["metric"].map(pretty_label)

    domain = _axis_range_control(st, long_df["value"], key=key, label="y-axis")
    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{ENCODER_COLUMN}:N", title="Encoding", axis=alt.Axis(labelAngle=-40)),
            xOffset="metric:N",
            y=alt.Y("value:Q", title=value_title, scale=_quant_scale(domain)),
            color=alt.Color("metric:N", title="Probe"),
            tooltip=[
                alt.Tooltip(f"{ENCODER_COLUMN}:N", title="Encoding"),
                alt.Tooltip("metric:N", title="Probe"),
                alt.Tooltip("value:Q", title=value_title, format=".4f"),
            ],
        )
    )
    st.altair_chart(chart, width="stretch")


def _render_informativeness(rows: pd.DataFrame, scenario: str) -> None:
    st.caption(
        "R² of a linear vs MLP probe recovering each ground-truth factor from the features the "
        "encoder *adds* (raw GPS pass-through excluded). A linear–MLP gap means the factor is "
        "present but not linearly exposed."
    )
    factors = sorted(rows["factor"].dropna().astype(str).unique().tolist()) if "factor" in rows.columns else []
    if not factors:
        st.caption("No factors recorded.")
        return
    factor = st.selectbox(
        "Factor", options=factors, format_func=pretty_label, key=f"info_factor_{scenario}"
    )
    factor_rows = rows[rows["factor"].astype(str) == factor]
    _render_probe_bars(
        factor_rows, ["r2_linear", "r2_mlp"], key=f"info_{scenario}_{factor}", value_title="R²"
    )


def _render_invariance(rows: pd.DataFrame, scenario: str) -> None:
    st.caption(
        "Degradation as Gaussian GPS noise (σ, in metres) is injected into the test positions and "
        "re-encoded: representation drift and the downstream drop in accuracy / DBA / power loss. "
        "Lower is better - a flatter line is a more robust encoding."
    )
    metric_options = [c for c in INVARIANCE_METRICS if c in rows.columns]
    if not metric_options or "sigma_m" not in rows.columns:
        st.caption("No invariance sweep recorded.")
        return
    metric = st.selectbox(
        "Degradation metric", options=metric_options, format_func=pretty_label,
        key=f"inv_metric_{scenario}",
    )
    plot = rows[[ENCODER_COLUMN, "sigma_m", metric]].dropna()
    if plot.empty:
        st.caption("No data to plot.")
        return
    domain = _axis_range_control(st, plot[metric], key=f"inv_y_{scenario}_{metric}", label="y-axis")
    chart = (
        alt.Chart(plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("sigma_m:Q", title="GPS noise σ (m)"),
            y=alt.Y(f"{metric}:Q", title=pretty_label(metric), scale=_quant_scale(domain)),
            color=alt.Color(f"{ENCODER_COLUMN}:N", title="Encoding"),
            tooltip=[
                alt.Tooltip(f"{ENCODER_COLUMN}:N", title="Encoding"),
                alt.Tooltip("sigma_m:Q", title="σ (m)"),
                alt.Tooltip(f"{metric}:Q", title=pretty_label(metric), format=".4f"),
            ],
        )
    )
    st.altair_chart(chart, width="stretch")


# --- shared chart helpers ------------------------------------------------------

def _top_features(encoder_rows: pd.DataFrame, value_column: str, top_n: int) -> pd.DataFrame:
    """Average importance per feature across seeds and keep the top ``top_n``."""

    if value_column not in encoder_rows.columns:
        return pd.DataFrame()
    grouped = encoder_rows.groupby("feature", dropna=False)
    ranked = grouped[value_column].mean().reset_index()
    ranked["runs"] = grouped.size().to_numpy()
    if "rank" in encoder_rows.columns:
        ranked = ranked.merge(
            grouped["rank"].mean().reset_index().rename(columns={"rank": "rank"}),
            on="feature",
            how="left",
        )
    ranked = ranked.sort_values(value_column, ascending=False, kind="stable").head(top_n)
    return ranked.reset_index(drop=True)


def _ordered_splits(scoped: pd.DataFrame) -> list[str | None]:
    if "split" not in scoped.columns:
        return [None]
    present = set(scoped["split"].dropna().astype(str))
    ordered = [split for split in PREFERRED_SPLIT_ORDER if split in present]
    ordered += [split for split in sorted(present) if split not in ordered]
    return ordered or [None]


def _aggregated_to_long(aggregated: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Reshape mean/std columns into long form for grouped bar charts."""

    frames = []
    for metric in metrics:
        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"
        if mean_column not in aggregated.columns or aggregated[mean_column].notna().sum() == 0:
            continue
        part = aggregated[[ENCODER_COLUMN, mean_column, std_column, "runs"]].copy()
        part = part.rename(columns={mean_column: "mean", std_column: "std"})
        part["metric"] = pretty_label(metric)
        part["std"] = part["std"].fillna(0.0)
        part["lower"] = part["mean"] - part["std"]
        part["upper"] = part["mean"] + part["std"]
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _feature_count_frame(subset: pd.DataFrame) -> pd.DataFrame:
    """Return the mean feature count after encoding, one row per encoder."""

    if FEATURE_COUNT_METRIC not in subset.columns:
        return pd.DataFrame()
    counts = (
        subset.groupby(ENCODER_COLUMN, dropna=False)[FEATURE_COUNT_METRIC]
        .mean()
        .reset_index(name="features")
    )
    counts = counts[counts["features"].notna()]
    return counts


def _encoder_order_by_features(feature_counts: pd.DataFrame) -> list[str] | None:
    """Return encoders ordered by ascending feature count, when available."""

    if feature_counts is None or feature_counts.empty:
        return None
    ordered = feature_counts.sort_values("features", kind="stable")
    return ordered[ENCODER_COLUMN].astype(str).tolist()


def _render_grouped_bars(
    long_df: pd.DataFrame,
    feature_counts: pd.DataFrame | None = None,
    *,
    encoder_order: list[str] | None = None,
    key: str = "grouped",
) -> None:
    """Vertical grouped bars per encoder, with an optional feature-count overlay.

    The feature count lives on a different scale (an integer column), so it is drawn as a
    labelled line + points against an independent right-hand axis. When ``encoder_order`` is
    given the bars follow that order (e.g. ascending feature count). An opt-in y-axis range
    control lets the user zoom the primary (mean) axis for precision.
    """

    if long_df.empty:
        st.caption("No data to plot for this split.")
        return

    domain = _axis_range_control(
        st, pd.concat([long_df["lower"], long_df["upper"]]), key=key, label="y-axis"
    )
    y_scale = _quant_scale(domain)

    x_sort = encoder_order if encoder_order else None
    x_axis = alt.X(
        f"{ENCODER_COLUMN}:N", title="Encoding", sort=x_sort, axis=alt.Axis(labelAngle=-40)
    )
    x_offset = alt.XOffset("metric:N")
    bars = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=x_axis,
            xOffset=x_offset,
            y=alt.Y("mean:Q", title="Mean", scale=y_scale),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=[
                alt.Tooltip(f"{ENCODER_COLUMN}:N", title="Encoding"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("mean:Q", title="Mean", format=".4f"),
                alt.Tooltip("std:Q", title="Std", format=".4f"),
                alt.Tooltip("runs:Q", title="Runs"),
            ],
        )
    )
    error_bars = (
        alt.Chart(long_df)
        .mark_rule()
        .encode(x=x_axis, xOffset=x_offset, y=alt.Y("lower:Q", scale=y_scale), y2="upper:Q")
    )
    layers = bars + error_bars

    if feature_counts is not None and not feature_counts.empty:
        feature_axis = alt.Y(
            "features:Q",
            title=pretty_label(FEATURE_COUNT_METRIC),
            axis=alt.Axis(titleColor="#555", orient="right"),
            scale=alt.Scale(zero=False, nice=True),
        )
        feature_tooltip = [
            alt.Tooltip(f"{ENCODER_COLUMN}:N", title="Encoding"),
            alt.Tooltip("features:Q", title=pretty_label(FEATURE_COUNT_METRIC), format=".0f"),
        ]
        feature_base = alt.Chart(feature_counts).encode(
            x=x_axis, y=feature_axis, tooltip=feature_tooltip
        )
        feature_line = feature_base.mark_line(color="#444", strokeDash=[4, 3], point=False)
        feature_points = feature_base.mark_point(
            color="#444", filled=True, size=70, opacity=1
        )
        feature_labels = feature_base.mark_text(
            color="#444", dy=-12, fontWeight="bold"
        ).encode(text=alt.Text("features:Q", format=".0f"))
        feature_overlay = feature_line + feature_points + feature_labels
        layers = alt.layer(layers, feature_overlay).resolve_scale(y="independent")

    st.altair_chart(layers, width="stretch")


def _render_single_metric_bars(aggregated: pd.DataFrame, metric: str, *, key: str = "single") -> None:
    """Vertical bar chart for a single metric with a ±1 std error bar."""

    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    if mean_column not in aggregated.columns or aggregated[mean_column].notna().sum() == 0:
        st.caption(f"No data for {pretty_label(metric)}.")
        return

    data = aggregated.copy()
    data[std_column] = data[std_column].fillna(0.0)
    data["lower"] = data[mean_column] - data[std_column]
    data["upper"] = data[mean_column] + data[std_column]
    sort_order = "-y" if metric_higher_is_better(metric) else "y"

    domain = _axis_range_control(
        st, pd.concat([data["lower"], data["upper"]]), key=key, label=f"{pretty_label(metric)} y-axis"
    )
    y_scale = _quant_scale(domain)

    x_axis = alt.X(
        f"{ENCODER_COLUMN}:N",
        sort=sort_order,
        title="Encoding",
        axis=alt.Axis(labelAngle=-40),
    )
    base = alt.Chart(data)
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        x=x_axis,
        y=alt.Y(field=mean_column, type="quantitative", title=f"Mean {pretty_label(metric)}", scale=y_scale),
        color=alt.Color(f"{ENCODER_COLUMN}:N", title="Encoding", legend=None),
        tooltip=[
            alt.Tooltip(f"{ENCODER_COLUMN}:N", title="Encoding"),
            alt.Tooltip(field=mean_column, type="quantitative", title="Mean", format=".4f"),
            alt.Tooltip(field=std_column, type="quantitative", title="Std", format=".4f"),
            alt.Tooltip("runs:Q", title="Runs"),
        ],
    )
    error_bars = base.mark_rule().encode(x=x_axis, y=alt.Y("lower:Q", scale=y_scale), y2="upper:Q")
    st.altair_chart((bars + error_bars).properties(title=pretty_label(metric)), width="stretch")


def _render_mean_std_table(
    aggregated: pd.DataFrame,
    metrics: list[str],
    *,
    encoder_order: list[str] | None = None,
) -> None:
    """Show a compact table with `mean ± std` per encoder."""

    if encoder_order:
        rank = {encoder: index for index, encoder in enumerate(encoder_order)}
        aggregated = aggregated.assign(
            _order=aggregated[ENCODER_COLUMN].astype(str).map(rank)
        ).sort_values("_order", kind="stable", na_position="last")

    table = aggregated[[ENCODER_COLUMN]].copy()
    for metric in metrics:
        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"
        if mean_column not in aggregated.columns:
            continue
        means = aggregated[mean_column]
        stds = aggregated[std_column].fillna(0.0)
        table[pretty_label(metric)] = [
            f"{mean:.4f} ± {std:.4f}" if pd.notna(mean) else "-"
            for mean, std in zip(means, stds)
        ]
    table["runs"] = aggregated["runs"]
    if "seeds" in aggregated.columns:
        table["seeds"] = aggregated["seeds"].apply(
            lambda values: ", ".join(str(value) for value in values)
        )
    st.dataframe(
        table.rename(columns={column: pretty_label(column) for column in table.columns}),
        hide_index=True,
        width="stretch",
    )


# --- orchestration -------------------------------------------------------------

fingerprint = linked_results_fingerprint(RESULTS_DIRECTORY)
if not fingerprint:
    st.title("6G Encoding Results Dashboard")
    st.error(
        f"No CSVs found in `{RESULTS_DIRECTORY_NAME}/`. Add result CSVs that share a "
        "`run_id` column to populate the dashboard."
    )
    st.stop()

merged_data, source_contributions = load_linked_results_cached(str(RESULTS_DIRECTORY), fingerprint)
if merged_data.empty:
    st.title("6G Encoding Results Dashboard")
    st.error(f"No CSVs with a `run_id` column were found in `{RESULTS_DIRECTORY_NAME}/`.")
    st.stop()

feature_importance_data = load_feature_importance_cached(str(RESULTS_DIRECTORY), fingerprint)
probe_fingerprint = representation_probes_fingerprint(REPRESENTATION_PROBES_DIRECTORY)
probe_data = load_representation_probes_cached(str(REPRESENTATION_PROBES_DIRECTORY), probe_fingerprint)
metric_value_set = set(metric_value_columns(merged_data))
present_seeds = available_seeds(merged_data)

st.title("6G Encoding Results Dashboard")
st.caption(
    f"{len(source_contributions)} linked CSVs in `{RESULTS_DIRECTORY_NAME}/` · "
    f"seeds present: {', '.join(str(seed) for seed in present_seeds) or 'n/a'}"
)

filtered_data = apply_seed_filter(merged_data)
if filtered_data.empty:
    st.warning("No rows match the current seed selection.")
    st.stop()

# The feature-importance table follows the same seed filter as the merged data.
seed_col = seed_column(feature_importance_data) if not feature_importance_data.empty else None
if seed_col is not None:
    kept_seeds = filtered_data[seed_col].unique() if seed_col in filtered_data.columns else []
    feature_importance_data = feature_importance_data[
        feature_importance_data[seed_col].isin(kept_seeds)
    ]

render_merged_table(filtered_data, source_contributions)

st.header("Experiments")
render_experiments(filtered_data, source_contributions, metric_value_set, feature_importance_data)

if probe_data:
    render_feature_representation(probe_data)
