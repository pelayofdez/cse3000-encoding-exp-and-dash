"""Downstream model factory.

The representation (encoder) is the primary experimental variable; the study compares it
across two downstream models chosen to span **different inductive biases**, so an
encoding's value is judged against the way a model actually consumes features:

* ``xgboost`` - boosted trees, split on thresholds and scale-invariant.
* ``morais_nn`` - the exact PyTorch network of Morais et al. 2022
  (arXiv:2205.09054), the position-aided beam-prediction baseline; scale-sensitive.

Every model is wrapped in a pipeline whose first step is a median imputer (encoders may
emit NaNs, e.g. lag features at the start of a sequence). The scale-sensitive ``morais_nn``
additionally gets a ``StandardScaler``; the tree model does not (it is scale-invariant).
Each pipeline keeps the step names ``imputer`` and ``classifier`` so ``evaluation`` can
find them uniformly. All fitting (imputer / scaler statistics, the classifier) happens on
the training split only, inside ``Pipeline.fit``.
"""

from __future__ import annotations

from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_RANDOM_STATE, ModelConfig

# XGBoost has no `class_weight`; balanced sample weights are applied in the wrapper instead
# (see build_model). `tree_method=hist` keeps it fast on these small datasets.
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300, "random_state": DEFAULT_RANDOM_STATE,
    "n_jobs": -1, "tree_method": "hist", "eval_metric": "mlogloss", "verbosity": 0,
}
# The fully-connected network of Morais et al. 2022 ("Position Aided Beam Prediction in the
# Real World", arXiv:2205.09054), Table I. sklearn's MLPClassifier cannot express the paper's
# step LR schedule or best-val-accuracy checkpointing, so this is implemented in PyTorch (see
# `_MoraisBeamNN`). Defaults below reproduce Table I verbatim; a config may override any of
# them via model.params.
DEFAULT_MORAIS_PARAMS: dict[str, Any] = {
    "hidden_layers": 3, "hidden_units": 256, "lr": 0.01,
    "lr_milestones": (20, 40), "lr_gamma": 0.2, "batch_size": 32, "epochs": 60,
    "val_fraction": 0.1, "random_state": DEFAULT_RANDOM_STATE,
}


class _MoraisBeamNN:
    """Faithful re-implementation of the position-aided beam-prediction network of
    Morais et al. 2022 (arXiv:2205.09054), Table I, as a scikit-learn-style estimator.

    Architecture / training (all from the paper):
      * ``hidden_layers`` fully-connected layers of ``hidden_units`` ReLU units (paper: 3×256),
        then a linear head of ``n_classes`` logits.
      * Adam, initial ``lr=0.01``; the LR is multiplied by ``lr_gamma=0.2`` at each epoch in
        ``lr_milestones=(20, 40)`` (a ``MultiStepLR`` step schedule).
      * Cross-entropy loss, ``batch_size=32``, ``epochs=60``.
      * **Best-validation-accuracy checkpoint**: an internal validation split (``val_fraction``)
        is carved from the training data; the weights from the epoch with the highest val
        accuracy are restored at the end (the paper's "model checkpoint at epoch with highest
        validation accuracy"). The pipeline's outer val/test splits are never touched here.

    Implemented in PyTorch (CPU). Exposes exactly the surface the pipeline + metrics use:
    ``fit`` / ``predict`` / ``predict_proba`` / ``classes_``. Labels are label-encoded to
    ``0..n-1`` for training and mapped back to the real beam indices on predict, so DBA / top-k
    stay correct even when beam labels have gaps. No native ``feature_importances_`` / ``coef_``,
    so importance falls back to the model-agnostic permutation path.
    """

    def __init__(self, hidden_layers: int = 3, hidden_units: int = 256, lr: float = 0.01,
                 lr_milestones=(20, 40), lr_gamma: float = 0.2, batch_size: int = 32,
                 epochs: int = 60, val_fraction: float = 0.1,
                 random_state: int = DEFAULT_RANDOM_STATE):
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.lr = lr
        self.lr_milestones = tuple(lr_milestones)
        self.lr_gamma = lr_gamma
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_fraction = val_fraction
        self.random_state = random_state

    def _build_net(self, n_features: int, n_classes: int):
        import torch.nn as nn
        layers: list = []
        in_dim = n_features
        for _ in range(self.hidden_layers):
            layers += [nn.Linear(in_dim, self.hidden_units), nn.ReLU()]
            in_dim = self.hidden_units
        layers.append(nn.Linear(in_dim, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import numpy as np
        import torch
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_  # real beam labels, in predict_proba column order
        y_enc = self._le.transform(y).astype(np.int64)
        n_classes = len(self.classes_)

        # Internal validation split for best-checkpoint selection. With too few samples to
        # carve a validation set, fall back to scoring on the training data itself.
        if len(X) >= 10 and self.val_fraction and self.val_fraction > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y_enc, test_size=self.val_fraction,
                random_state=self.random_state, shuffle=True,
            )
        else:
            X_tr, X_val, y_tr, y_val = X, X, y_enc, y_enc

        net = self._build_net(X.shape[1], n_classes)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(self.lr_milestones), gamma=self.lr_gamma,
        )
        criterion = torch.nn.CrossEntropyLoss()

        X_tr_t, y_tr_t = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
        X_val_t, y_val_t = torch.from_numpy(X_val), torch.from_numpy(y_val)
        n_tr = len(X_tr_t)
        rng = np.random.default_rng(self.random_state)
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        best_val_acc = -1.0

        for _epoch in range(self.epochs):
            net.train()
            order = rng.permutation(n_tr)  # reshuffle minibatches each epoch
            for start in range(0, n_tr, self.batch_size):
                idx = order[start:start + self.batch_size]
                bx, by = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad()
                loss = criterion(net(bx), by)
                loss.backward()
                optimizer.step()
            scheduler.step()
            net.eval()
            with torch.no_grad():
                val_acc = (net(X_val_t).argmax(1) == y_val_t).float().mean().item()
            if val_acc >= best_val_acc:  # keep the latest best (ties -> later epoch)
                best_val_acc = val_acc
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        net.load_state_dict(best_state)
        net.eval()
        self._net = net
        self.best_val_accuracy_ = best_val_acc
        return self

    def predict_proba(self, X):
        import numpy as np
        import torch
        X = np.asarray(X, dtype=np.float32)
        self._net.eval()
        with torch.no_grad():
            return torch.softmax(self._net(torch.from_numpy(X)), dim=1).numpy()

    def predict(self, X):
        import numpy as np
        return self._le.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

    def score(self, X, y):
        """Mean accuracy - the default ``ClassifierMixin.score``. Needed so the pipeline is
        scorable for ``permutation_importance`` (the model-agnostic importance fallback)."""
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(y, self.predict(X)))


def _tree_pipeline(classifier) -> Pipeline:
    """Imputer -> classifier (tree models are scale-invariant, no scaling)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier),
    ])


def _scaled_pipeline(classifier) -> Pipeline:
    """Imputer -> StandardScaler -> classifier (scale-sensitive models)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


def build_model(cfg: ModelConfig) -> Pipeline:
    name = cfg.name
    if name in ("xgboost", "xgb"):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'xgboost' model requires the xgboost package (`pip install xgboost`)."
            ) from exc
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight

        class _BalancedXGB:
            """Composition wrapper around XGBoost. XGBoost requires labels in ``0..n-1`` (the
            beam indices are not), so we label-encode for fitting and map predictions back to
            the real beam indices - keeping DBA/top-k correct even when beams have gaps. We
            also apply balanced sample weights, since XGBoost has no ``class_weight``. Exposes
            the sklearn surface the pipeline/metrics use."""

            def __init__(self, **params):
                self._params = params

            def fit(self, X, y):
                self._le = LabelEncoder().fit(y)
                self.classes_ = self._le.classes_  # real beam labels, in predict_proba column order
                self._model = XGBClassifier(**self._params)
                self._model.fit(X, self._le.transform(y),
                                sample_weight=compute_sample_weight("balanced", y))
                return self

            def predict(self, X):
                return self._le.inverse_transform(self._model.predict(X))

            def predict_proba(self, X):
                return self._model.predict_proba(X)

            @property
            def feature_importances_(self):
                return self._model.feature_importances_

        return _tree_pipeline(_BalancedXGB(**{**DEFAULT_XGB_PARAMS, **cfg.params}))
    if name in ("morais_nn", "position_nn"):
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'morais_nn' model requires PyTorch (`pip install torch`)."
            ) from exc
        return _scaled_pipeline(_MoraisBeamNN(**{**DEFAULT_MORAIS_PARAMS, **cfg.params}))
    raise ValueError(f"Unknown model {name!r}. Available: xgboost, morais_nn.")
