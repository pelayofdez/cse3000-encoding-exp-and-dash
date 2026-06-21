"""Metrics: predictive performance, representation statistics, runtime."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def _class_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return a (n_samples, n_classes) score matrix using predict_proba if available,
    otherwise decision_function. For binary decision_function (1-D output), scores
    are stacked into 2 columns so downstream argsort logic stays uniform."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    scores = model.decision_function(X)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores


def top_k_accuracy(model, X: pd.DataFrame, y_true: pd.Series, k: int) -> float:
    """Fraction of rows whose true label is among the model's top-k predictions."""
    classes = model.named_steps["classifier"].classes_
    if len(classes) <= k:
        return float("nan")  # top-k is trivial when there are <= k classes
    scores = _class_scores(model, X)
    top_k = classes[np.argsort(scores, axis=1)[:, -k:]]
    y_arr = np.asarray(y_true)
    return float(np.mean([true in row for true, row in zip(y_arr, top_k)]))


def dba_score(model, X: pd.DataFrame, y_true: pd.Series,
              max_k: int = 3, delta: float = 5.0) -> float:
    """DeepSense Distance-Based Accuracy (DBA): beam-prediction quality that credits
    *near* beams, not just exact hits. Higher is better, in ``[0, 1]``.

    For each cutoff ``k = 1..max_k``, take the ``k`` best-scoring predicted beams; the
    per-sample penalty is the beam-index distance from the true beam to the closest of
    those ``k``, divided by ``delta`` and capped at 1 (so a beam ``>= delta`` indices away
    is a full miss). The k-score is ``1 - mean(penalty)``, and DBA is the average over
    ``k = 1..max_k``. This is the metric of the DeepSense 6G Multi-Modal Beam Prediction
    Challenge (``delta=5``, ``max_k=3``). It is only meaningful when the labels are beam
    indices (ordinal), so callers gate it to the DeepSense task.
    """
    classes = model.named_steps["classifier"].classes_
    scores = _class_scores(model, X)
    # Beam indices ranked best-first per sample: (n_samples, n_classes).
    ranked = classes[np.argsort(scores, axis=1)[:, ::-1]].astype(float)
    y_arr = np.asarray(y_true, dtype=float)
    n = len(y_arr)
    if n == 0:
        return float("nan")
    k_max = min(max_k, ranked.shape[1])
    per_k = np.empty(k_max)
    for k in range(k_max):
        # Distance to the closest of the top-(k+1) predicted beams, per sample.
        dist = np.min(np.abs(ranked[:, :k + 1] - y_arr[:, None]), axis=1)
        per_k[k] = 1.0 - np.minimum(dist / delta, 1.0).mean()
    return float(per_k.mean())


def power_loss_db(pred, y_true, power_matrix: np.ndarray, noise_power: float,
                  beam_offset: int = 1) -> float:
    """Average power loss in dB (Morais et al. 2022, eq. 5): how much receive power the
    *predicted* beam gives up relative to the *optimal* beam, noise-subtracted and averaged.

        P_L = 10 * log10( (1/K) * sum_k (P_opt^k - P_n) / (P_pred^k - P_n) )

    where ``P_opt`` is the ground-truth beam's power, ``P_pred`` the predicted beam's power,
    and ``P_n`` the scenario noise floor. Following the paper, ``P_n`` is "the average of the
    smallest power per sample" - the mean over samples of each sample's minimum beam power
    (computed in the pipeline and passed in here).
    ``0 dB`` = always selecting the optimal beam; larger (more positive) = worse, so lower is
    better. This complements accuracy/DBA: it scores the *system* objective (received power),
    so a confident wrong guess at a beam that still carries strong power is barely penalised.

    ``power_matrix`` is ``(n_eval, n_beams)`` aligned to ``pred``/``y_true``. DeepSense beam
    labels are 1-indexed, so beam ``b``'s power is ``power_matrix[:, b - beam_offset]``.
    The averaging is over the linear power ratios *before* the log, exactly as in the paper.
    """
    pred = np.asarray(pred)
    y = np.asarray(y_true)
    n_beams = power_matrix.shape[1]
    rows = np.arange(len(pred))
    # 1-indexed beam label -> 0-indexed power column; clip defensively (predictions are
    # always valid training labels, so clipping is a no-op in practice).
    pred_idx = np.clip(pred - beam_offset, 0, n_beams - 1)
    true_idx = np.clip(y - beam_offset, 0, n_beams - 1)
    p_opt = power_matrix[rows, true_idx]    # ground-truth beam power (== per-row max)
    p_pred = power_matrix[rows, pred_idx]   # predicted beam power
    ratio = (p_opt - noise_power) / (p_pred - noise_power)
    mean_ratio = float(np.mean(ratio))
    # The formula is only well-defined while the predicted beam stays above the noise floor
    # P_n (the *average* of per-sample minimum powers): a prediction below it makes that
    # sample's denominator negative. This holds for a decent model (as in the paper) but is
    # violated by a near-random one (e.g. scenario 9). If the aggregate goes non-finite or
    # non-positive, print the offending values and report NaN rather than a complex/inf dB.
    if not np.isfinite(mean_ratio) or mean_ratio <= 0:
        bad = ~np.isfinite(ratio) | (ratio <= 0)
        print(
            f"[power_loss_db] -> NaN: mean_ratio={mean_ratio:.4g}, "
            f"noise_power={noise_power:.4g}, n_eval={len(pred)}, n_bad={int(bad.sum())}\n"
            f"  p_pred[bad]={np.array2string(p_pred[bad], precision=4, threshold=20)}\n"
            f"  p_opt[bad] ={np.array2string(p_opt[bad], precision=4, threshold=20)}\n"
            f"  ratio[bad] ={np.array2string(ratio[bad], precision=4, threshold=20)}"
        )
        return float("nan")
    return float(10.0 * np.log10(mean_ratio))


def power_ratio(power_matrix: np.ndarray, *, classes: np.ndarray | None = None,
                scores: np.ndarray | None = None, pred=None, k: int = 1,
                beam_offset: int = 1) -> float:
    """Average Power Ratio (Vuckovic et al. 2024, MILCOM): how strong the *selected* beam is
    relative to the *best* beam, in linear power. Higher is better, in ``(0, 1]``.

        PR = mean_n( P(selected beam) / P(best beam) )

    where ``P(best beam)`` is the per-sample maximum received power (the ground-truth beam).
    Unlike :func:`power_loss_db` this is the raw linear ratio the paper reports - no noise
    subtraction and no log, so it cannot go non-finite and needs no NaN guard.

    * ``k <= 1`` scores the single predicted beam (``pred`` required): **top-1 PR**.
    * ``k > 1`` takes the *highest-power* beam among the model's ``k`` highest-scoring beams
      (``classes`` + ``scores`` required): **top-k PR**. This credits a model that ranks a
      strong beam within its top-k even when the argmax beam is weak.

    The paper's argument is that near-equal-power beams make exact-beam accuracy misleading;
    a PR near 1 means the selected beam carries nearly the optimal power regardless of index.
    ``power_matrix`` is ``(n_eval, n_beams)``; DeepSense beams are 1-indexed, so beam ``b`` is
    column ``b - beam_offset``.
    """
    n, n_beams = power_matrix.shape
    if n == 0:
        return float("nan")
    rows = np.arange(n)
    p_best = power_matrix.max(axis=1)  # per-row max == ground-truth beam power
    if k <= 1:
        sel = np.clip(np.asarray(pred) - beam_offset, 0, n_beams - 1)
        p_sel = power_matrix[rows, sel]
    else:
        # Beam *labels* of the k highest-scoring classes -> power columns; take the best power.
        topk_labels = classes[np.argsort(scores, axis=1)[:, -k:]]
        cols = np.clip(topk_labels.astype(int) - beam_offset, 0, n_beams - 1)
        p_sel = power_matrix[rows[:, None], cols].max(axis=1)
    return float(np.mean(p_sel / p_best))


def topk_beams_accuracy(classes: np.ndarray, scores: np.ndarray, power_matrix: np.ndarray,
                        k1: int, k2: int, beam_offset: int = 1) -> float:
    """top-K1,K2 Beams-Accuracy (Vuckovic et al. 2024): predict the ``k2`` highest-scoring
    beams and count the sample correct if **any** of them lands among the ``k1`` highest-power
    (true) beams. A relaxed, power-aware accuracy that stops penalising a confident pick of a
    beam whose power nearly matches the optimum. Higher is better, in ``[0, 1]``.

    * ``top-KB`` (top-K,1BA) is ``k2 = 1``: is the single predicted beam within the ``k1`` best
      by power?
    * ``top-3,3BA`` is ``k1 = k2 = 3``: a headline metric in the paper's Table I.
    """
    n, n_beams = power_matrix.shape
    if n == 0:
        return float("nan")
    k1 = min(k1, n_beams)
    k2 = min(k2, scores.shape[1])
    pred_cols = np.clip(classes[np.argsort(scores, axis=1)[:, -k2:]].astype(int) - beam_offset,
                        0, n_beams - 1)
    best_cols = np.argsort(power_matrix, axis=1)[:, -k1:]  # k1 strongest beams, 0-indexed
    hit = [len(set(p.tolist()) & set(b.tolist())) > 0 for p, b in zip(pred_cols, best_cols)]
    return float(np.mean(hit))


def evaluate_split(model, X_eval, y_eval, train_labels, top_k: int,
                   compute_dba: bool = False, dba_max_k: int = 3, dba_delta: float = 5.0,
                   power_matrix: np.ndarray | None = None, noise_power: float | None = None) -> dict:
    pred = model.predict(X_eval)
    train_label_set = set(np.asarray(train_labels))
    # DBA and the power metrics are beam-task metrics (ordinal beam indices / received power);
    # NaN keeps the CSV schema uniform when they cannot be computed (e.g. no power matrix).
    compute_power = power_matrix is not None and noise_power is not None
    metrics = {
        "accuracy": float(accuracy_score(y_eval, pred)),
        "top_k_accuracy": top_k_accuracy(model, X_eval, y_eval, k=top_k),
        "dba_score": (dba_score(model, X_eval, y_eval, max_k=dba_max_k, delta=dba_delta)
                      if compute_dba else float("nan")),
        "power_loss_db": (power_loss_db(pred, y_eval, power_matrix, noise_power)
                          if compute_power else float("nan")),
    }
    # Power-aware beam metrics (Vuckovic et al. 2024): PR scores the selected beam's power
    # vs the optimum; top-KB / top-K,KBA relax accuracy to "within the K strongest beams".
    # These need only the power matrix (no noise floor), but we gate them with the rest of
    # the beam metrics so the CSV stays NaN when the power matrix is unavailable.
    if power_matrix is not None:
        classes = model.named_steps["classifier"].classes_
        scores = _class_scores(model, X_eval)
        metrics.update({
            "pr_top1": power_ratio(power_matrix, pred=pred, k=1),
            "pr_topk": power_ratio(power_matrix, classes=classes, scores=scores, k=top_k),
            "top_kb": topk_beams_accuracy(classes, scores, power_matrix, k1=top_k, k2=1),
            "top_kk_ba": topk_beams_accuracy(classes, scores, power_matrix, k1=top_k, k2=top_k),
        })
    else:
        metrics.update({"pr_top1": float("nan"), "pr_topk": float("nan"),
                        "top_kb": float("nan"), "top_kk_ba": float("nan")})
    metrics.update({
        "n_eval": int(len(X_eval)),
        "n_unseen_classes": int(len(set(np.unique(y_eval)) - train_label_set)),
    })
    return metrics


def _native_importance_kind(clf) -> str:
    """Name the *native* importance signal the model exposes via ``feature_importances_``.

    XGBoost's ``feature_importances_`` defaults to average ``"gain"``. It carries a known bias
    toward high-cardinality / high-variance features, so use it *within* an encoding, not to
    rank encoders (use the permutation path for cross-model / cross-encoder comparison).
    """
    return "gain"


def feature_importances(model, feature_names, top_n: int | None = None, *,
                        X: pd.DataFrame | None = None, y=None, scoring=None,
                        n_repeats: int = 5, random_state: int = 0) -> tuple[list[dict], str | None]:
    """Rank the encoded features by an importance signal. Returns ``(ranked, kind)`` where
    ``ranked`` is a list of dicts sorted by descending importance (``rank`` 1 = most
    important) and ``kind`` names the signal used:

    * ``"gain"`` – XGBoost's **native** tree importance (average gain; see
      :func:`_native_importance_kind`). It carries a known bias toward high-cardinality /
      high-variance features, so use it *within* an encoding, not to rank encoders.
    * ``"permutation"`` – **model-agnostic fallback** for estimators with no native signal
      (morais_nn): the drop in score when each feature column is shuffled, via
      ``sklearn.inspection.permutation_importance`` on the **whole pipeline**. Unlike the
      tree signal it is unbiased w.r.t. cardinality, so it is the one importance that is
      fair to compare across models and encoders. Requires ``X``/``y`` (a held-out split,
      e.g. validation); without them the fallback is skipped and ``([], None)`` returned.

    ``feature_names`` must align with the classifier's input columns (pass
    ``model[:-1].get_feature_names_out()``). For the permutation path the names instead come
    from ``X.columns`` (the raw encoded inputs that get permuted). ``importance_normalized``
    is each feature's share of the total |importance|. ``top_n`` truncates the ranking.
    """
    clf = model.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
        kind = _native_importance_kind(clf)
        names = list(map(str, feature_names))
    elif X is not None and y is not None:
        # Model-agnostic: permute each input column of the fitted pipeline and measure the
        # score drop. Runs on the held-out split the caller passes (not train), so it reflects
        # generalisation rather than memorisation.
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            model, X, y, scoring=scoring, n_repeats=n_repeats,
            random_state=random_state, n_jobs=-1,
        )
        imp = np.asarray(result.importances_mean, dtype=float)
        kind = "permutation"
        names = list(map(str, X.columns))
    else:
        return [], None

    if len(names) != len(imp):  # defensive: fall back to positional names on any mismatch
        names = [f"f{i}" for i in range(len(imp))]
    # |importance| sum keeps normalisation well-defined when permutation yields negatives
    # (a feature whose shuffle *improves* the score); identical to the old sum for the
    # non-negative tree / coef signals.
    total = float(np.abs(imp).sum()) or 1.0
    order = np.argsort(imp)[::-1]
    ranked = [
        {
            "rank": rank + 1,
            "feature": names[i],
            "importance": float(imp[i]),
            "importance_normalized": float(imp[i] / total),
        }
        for rank, i in enumerate(order)
    ]
    return (ranked[:top_n] if top_n is not None else ranked), kind


def representation_stats(model, X_train: pd.DataFrame, n_raw_features: int) -> dict:
    """Shape / sparsity of the representation actually fed to the classifier.

    ``n_raw_features`` is the count of *source* columns the encoder consumed (before
    encoding), reported by the encoder. ``X_train`` is the already-encoded matrix, so
    ``n_features_after_encoding`` is its width; the two differ when the encoding
    adds features (e.g. rolling, periodic) and match when it does not (e.g. baseline).
    """
    transformed = model.named_steps["imputer"].transform(X_train)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    return {
        "n_raw_features": int(n_raw_features),
        "n_features_after_encoding": int(transformed.shape[1]),
        "sparsity_train": float(np.mean(transformed == 0)),
    }
