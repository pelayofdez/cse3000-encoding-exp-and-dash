"""Train / validation / test splitting.

The split produces a deterministic 3-way partition that is *identical* across
encoders for a given (dataset, seed). This is what makes the encoder comparison
fair: only the representation changes, never the split.

The split is dictated by the **data's structure** via ``task.group``: for DeepSense the
group is the sequence (one drive), so whole sequences stay in one partition - a row-wise
split would leak, because adjacent rows in a sequence share near-identical GPS and the
same beam index. ``grouped`` is the strategy every config uses; a per-group ``contiguous``
block split is also available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import SplitConfig, TaskConfig


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def as_label_series(self, index: pd.Index) -> pd.Series:
        labels = pd.Series("unused", index=index, dtype=object)
        labels.iloc[self.train_idx] = "train"
        labels.iloc[self.val_idx] = "val"
        labels.iloc[self.test_idx] = "test"
        return labels

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": int(len(self.train_idx)),
            "val": int(len(self.val_idx)),
            "test": int(len(self.test_idx)),
        }


def _grouped_split(y: pd.Series, groups: pd.Series, cfg: SplitConfig) -> Split:
    n = len(y)
    idx = np.arange(n)

    first = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    train_idx, temp_idx = next(first.split(idx, y, groups=groups))

    second = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size_of_temp, random_state=cfg.random_state)
    val_rel, test_rel = next(second.split(temp_idx, y.iloc[temp_idx], groups=groups.iloc[temp_idx]))
    return Split(train_idx=train_idx, val_idx=temp_idx[val_rel], test_idx=temp_idx[test_rel])


def _contiguous_split(df: pd.DataFrame, task: TaskConfig, cfg: SplitConfig) -> Split:
    """Per-group **contiguous-block** split: hold out the *tail* of each group's sequence.

    For datasets where each group is one recording of consecutive timesteps, a random row
    split leaks - adjacent timesteps are near-duplicates, so a near-neighbour of every test
    row sits in train and accuracy looks better than it generalises. Instead, within each
    group the rows are ordered (by ``cfg.order_col`` if given, else their current order) and
    cut into three contiguous blocks::

        [ ---- train (1 - test_size) ---- | -- val -- | -- test -- ]

    with ``val``/``test`` splitting the final ``test_size`` fraction by ``val_size_of_temp``.
    Every group contributes to all three splits, so all classes stay present in train/val/test
    (unlike ``grouped``, which would hold out whole groups - impossible when the group *is* the
    label). Deterministic: no RNG, the cut points are fixed fractions.
    """
    if not task.group:
        raise ValueError("contiguous split requires task.group to be set")
    train_idx, val_idx, test_idx = [], [], []
    for _, sub in df.groupby(task.group, sort=True):
        order = sub.sort_values(cfg.order_col).index if cfg.order_col else sub.index
        pos = np.asarray([df.index.get_loc(i) for i in order])
        n = len(pos)
        n_temp = int(round(n * cfg.test_size))
        n_train = n - n_temp
        n_val = int(round(n_temp * cfg.val_size_of_temp))
        train_idx.extend(pos[:n_train])
        val_idx.extend(pos[n_train:n_train + n_val])
        test_idx.extend(pos[n_train + n_val:])
    return Split(
        train_idx=np.asarray(train_idx, dtype=int),
        val_idx=np.asarray(val_idx, dtype=int),
        test_idx=np.asarray(test_idx, dtype=int),
    )


def make_split(df: pd.DataFrame, task: TaskConfig, cfg: SplitConfig) -> Split:
    y = df[task.target]
    if cfg.strategy == "grouped":
        if not task.group:
            raise ValueError("grouped split requires task.group to be set")
        return _grouped_split(y, df[task.group], cfg)
    if cfg.strategy == "contiguous":
        return _contiguous_split(df, task, cfg)
    raise ValueError(
        f"Unsupported split strategy {cfg.strategy!r}; expected 'grouped' or 'contiguous'."
    )
