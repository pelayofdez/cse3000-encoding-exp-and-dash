"""Experiment configuration: typed loading of YAML config files.

A config describes one experiment = one (dataset, encoder, model, split) point.
Comparisons (baseline vs encoding) are produced by running several configs and
aggregating the central result CSVs (see ``notebooks/01_explore_results.ipynb``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ----------------------------------------------------------------------------- #
# Single source of truth for randomness across the whole pipeline. Change this
# one value to reseed every component at once: the train/val/test split, the
# downstream model, and the encoders' internal RNGs. A config may still override
# any of these by setting ``random_state`` explicitly in its split / model.params /
# encoder.params, but none of the active configs do, so this constant governs them all.
# ----------------------------------------------------------------------------- #
DEFAULT_RANDOM_STATE = 1


@dataclass
class DatasetConfig:
    kind: str                       # "deepsense_scenario{1..9}"
    path: str                       # dataset folder
    scenario_name: str = "scenario1"


@dataclass
class TaskConfig:
    target: str
    type: str                       # "multiclass" | "binary"
    group: str | None = None        # grouping column for grouped splits (or None)


@dataclass
class SplitConfig:
    strategy: str = "grouped"       # "grouped" (DeepSense) | "contiguous" (per-group blocks)
    test_size: float = 0.30         # fraction held out for (val + test)
    val_size_of_temp: float = 0.50  # fraction of the held-out part used as test
    random_state: int = DEFAULT_RANDOM_STATE
    order_col: str | None = None    # 'contiguous' only: column giving the within-group order


@dataclass
class EncoderConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "xgboost"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    top_k: int = 3
    top_features: int = 20          # how many ranked features to persist per run
    # DBA (DeepSense Distance-Based Accuracy) - beam task only. Defaults match the
    # DeepSense 6G challenge: average over top-1/2/3 with a 5-beam distance tolerance.
    dba_max_k: int = 3
    dba_delta: float = 5.0


@dataclass
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    task: TaskConfig
    split: SplitConfig
    encoder: EncoderConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    results_dir: str = "results"
    source_path: str | None = None  # path the config was loaded from (provenance)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source_path: str | None = None) -> "ExperimentConfig":
        return cls(
            name=raw["name"],
            dataset=DatasetConfig(**raw["dataset"]),
            task=TaskConfig(**raw["task"]),
            split=SplitConfig(**raw.get("split", {})),
            encoder=EncoderConfig(**raw["encoder"]),
            model=ModelConfig(**raw.get("model", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            results_dir=raw.get("results_dir", "results"),
            source_path=source_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw, source_path=str(path))


def load_config(path: str | Path) -> ExperimentConfig:
    """Convenience wrapper used by the entry-point scripts."""
    return ExperimentConfig.load(path)
