from pathlib import Path
import unittest

import pandas as pd

from dashboard_data import (
    FEATURE_COLUMN,
    IMPORTANCE_COLUMN,
    PROBE_TYPES,
    aggregate_runs,
    available_datasets,
    available_seeds,
    discover_linked_result_csvs,
    load_feature_importance,
    load_linked_results,
    load_representation_probes,
    metric_higher_is_better,
    metric_value_columns,
    prepare_dataframe,
    seed_column,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"
REPRESENTATION_PROBES_DIRECTORY = REPOSITORY_ROOT / "representation_probes"


class LinkedResultsTests(unittest.TestCase):
    def test_discover_picks_only_csvs_with_run_id(self) -> None:
        names = {path.name for path in discover_linked_result_csvs(RESULTS_DIRECTORY)}

        # All four result CSVs carry run_id (the long-format one is split out later).
        self.assertIn("metrics.csv", names)
        self.assertIn("representation_metrics.csv", names)
        self.assertIn("runtime.csv", names)
        self.assertIn("feature_importance.csv", names)

    def test_wide_merge_excludes_long_feature_importance(self) -> None:
        merged, contributions = load_linked_results(RESULTS_DIRECTORY)
        metrics = prepare_dataframe(pd.read_csv(RESULTS_DIRECTORY / "metrics.csv"))

        self.assertFalse(merged.empty)
        # metrics.csv is the base (most rows per run) so its row count is kept;
        # the long feature-importance table must not inflate the merge.
        self.assertEqual(len(merged), len(metrics))
        self.assertNotIn("feature_importance.csv", contributions)
        self.assertNotIn(FEATURE_COLUMN, merged.columns)
        self.assertIn("accuracy", merged.columns)
        self.assertIn("sparsity_train", merged.columns)
        self.assertIn("encode_fit_seconds", merged.columns)
        # Single-row-per-run sources stay constant within a run.
        for _, run_rows in merged.groupby("run_id"):
            self.assertEqual(run_rows["sparsity_train"].nunique(dropna=False), 1)

    def test_results_are_time_series_only(self) -> None:
        # Only DeepSense beam-prediction runs are present.
        merged, _ = load_linked_results(RESULTS_DIRECTORY)
        datasets = available_datasets(merged)
        self.assertTrue(datasets)
        self.assertTrue(all(name.startswith("deepsense_") for name in datasets))
        if "regime" in merged.columns:
            self.assertEqual(set(merged["regime"].dropna().unique()), {"time_series"})

    def test_load_feature_importance_returns_long_table(self) -> None:
        fi = load_feature_importance(RESULTS_DIRECTORY)
        raw = pd.read_csv(RESULTS_DIRECTORY / "feature_importance.csv")

        self.assertEqual(len(fi), len(raw))
        self.assertIn(FEATURE_COLUMN, fi.columns)
        self.assertIn(IMPORTANCE_COLUMN, fi.columns)
        self.assertIn("importance_normalized", fi.columns)
        self.assertIn("rank", fi.columns)
        # Many ranked rows per run is what makes it "long".
        self.assertGreater(fi.groupby("run_id").size().max(), 1)

    def test_metric_value_columns_keep_metrics_drop_bookkeeping(self) -> None:
        merged, _ = load_linked_results(RESULTS_DIRECTORY)
        metrics = set(metric_value_columns(merged))

        # Downstream, representation and runtime metrics are kept.
        self.assertIn("accuracy", metrics)
        self.assertIn("top_k_accuracy", metrics)
        self.assertIn("sparsity_train", metrics)
        self.assertIn("encode_fit_seconds", metrics)
        # Counts / identifiers / seeds are excluded.
        self.assertNotIn("random_state", metrics)
        self.assertNotIn("n_train", metrics)
        self.assertNotIn("n_classes_train", metrics)

    def test_aggregate_runs_returns_mean_std_runs_and_seeds(self) -> None:
        merged, _ = load_linked_results(RESULTS_DIRECTORY)
        test_split = merged[merged["split"] == "test"]
        aggregated = aggregate_runs(test_split, ["dataset", "encoder"], ["accuracy"])

        self.assertIn("accuracy_mean", aggregated.columns)
        self.assertIn("accuracy_std", aggregated.columns)
        self.assertIn("runs", aggregated.columns)
        self.assertIn("seeds", aggregated.columns)
        for (dataset, encoder), group in test_split.groupby(["dataset", "encoder"]):
            row = aggregated[
                (aggregated["dataset"] == dataset) & (aggregated["encoder"] == encoder)
            ].iloc[0]
            self.assertAlmostEqual(row["accuracy_mean"], group["accuracy"].mean())
            self.assertEqual(row["runs"], len(group))

    def test_aggregate_runs_handles_missing_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "dataset": ["a", "a", "b"],
                "encoder": ["x", "x", "y"],
                "random_state": [1, 2, 1],
                "accuracy": [0.8, 0.9, 0.6],
            }
        )
        aggregated = aggregate_runs(
            frame, ["dataset", "encoder", "missing_column"], ["accuracy", "missing_metric"]
        )

        row = aggregated[(aggregated["dataset"] == "a") & (aggregated["encoder"] == "x")].iloc[0]
        self.assertAlmostEqual(row["accuracy_mean"], 0.85)
        self.assertAlmostEqual(row["accuracy_std"], 0.05)
        self.assertEqual(row["runs"], 2)
        self.assertEqual(row["seeds"], [1, 2])

    def test_seed_helpers_use_random_state_column(self) -> None:
        merged, _ = load_linked_results(RESULTS_DIRECTORY)
        expected = tuple(sorted(merged["random_state"].dropna().unique().tolist()))

        self.assertEqual(seed_column(merged), "random_state")
        self.assertEqual(available_seeds(merged), expected)

    def test_metric_higher_is_better(self) -> None:
        self.assertTrue(metric_higher_is_better("accuracy"))
        self.assertTrue(metric_higher_is_better("top_k_accuracy"))
        self.assertFalse(metric_higher_is_better("encode_fit_seconds"))
        self.assertFalse(metric_higher_is_better("power_loss_db"))


class RepresentationProbeTests(unittest.TestCase):
    def test_load_probes_discovers_each_type_with_scenario_and_encoder(self) -> None:
        probes = load_representation_probes(REPRESENTATION_PROBES_DIRECTORY)

        # Every probe CSV that ships should be discovered and keyed by probe type.
        self.assertTrue(probes)
        self.assertTrue(set(probes).issubset(set(PROBE_TYPES)))
        for probe_type, frame in probes.items():
            self.assertFalse(frame.empty, probe_type)
            self.assertIn("scenario", frame.columns)
            self.assertIn("encoder", frame.columns)

    def test_load_probes_concatenates_across_scenarios(self) -> None:
        probes = load_representation_probes(REPRESENTATION_PROBES_DIRECTORY)
        # The shipped probes cover scenario1 and scenario3; at least one type spans both.
        spans = {
            probe_type: set(frame["scenario"].astype(str).unique())
            for probe_type, frame in probes.items()
        }
        self.assertTrue(any(len(scenarios) >= 2 for scenarios in spans.values()), spans)

    def test_load_probes_returns_empty_for_missing_directory(self) -> None:
        self.assertEqual(load_representation_probes(REPOSITORY_ROOT / "does_not_exist"), {})


if __name__ == "__main__":
    unittest.main()
