# -*- coding: utf-8 -*-
"""
LightGBM learned baseline (evaluation.md grade B: "beats LightGBM").

A GBDT regressor that predicts each stop's normalized visit rank from its node
features; the route is the stops ordered by ascending predicted rank. Unlike the
heuristics it needs a fit() on the train split before evaluation on test.

`lightgbm` is optional — `is_available()` reports whether it can be used so eval
can skip the grade-B LightGBM clause gracefully.
"""
import numpy as np


def is_available():
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def _rank_target(visit_seq, n):
    """input-position -> normalized visit rank in [0,1] (0 = visited first)."""
    rank = np.argsort(visit_seq)            # rank[input_pos] = visit step
    return rank / max(n - 1, 1)


class LightGBMBaseline:
    def __init__(self, params=None):
        self.params = params or dict(
            objective='regression', n_estimators=300, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, random_state=2024, n_jobs=-1)
        self.model = None

    def fit(self, data):
        """data: a dataset dict (train.npy) with V, label, length."""
        import lightgbm as lgb
        V, label, length = data['V'], data['label'], data['length']
        X, y = [], []
        for i in range(len(length)):
            Ln = int(length[i])
            X.append(V[i, :Ln])
            y.append(_rank_target(label[i, :Ln], Ln))
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict_seq(self, V_traj, length):
        """V_traj: (>=length, F); returns visit order (input indices) of length `length`."""
        Ln = int(length)
        scores = self.model.predict(V_traj[:Ln])
        return np.argsort(scores).astype(np.int64)   # ascending: earliest rank first
