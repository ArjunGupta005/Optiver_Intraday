"""Baselines, ridge with train-fitted preprocessing, and LightGBM (L1). All serialisable."""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np


class ZeroModel:
    kind = "zero"

    def fit(self, X, y, **_):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.float64)

    def to_dict(self):
        return {"kind": self.kind}


class MedianModel:
    kind = "median"

    def __init__(self, value: float = 0.0):
        self.value = value

    def fit(self, X, y, **_):
        self.value = float(np.nanmedian(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=np.float64)

    def to_dict(self):
        return {"kind": self.kind, "value": self.value}


class RidgeModel:
    """Median imputation + quantile clipping + standardisation, all fitted on training rows; closed-form ridge."""
    kind = "ridge"

    def __init__(self, alpha: float = 1.0, clip_q: float = 0.001, max_fit_rows: int = 2_000_000, seed: int = 0):
        self.alpha, self.clip_q, self.max_fit_rows, self.seed = alpha, clip_q, max_fit_rows, seed
        self.params: dict = {}

    def _prep(self, X: np.ndarray) -> np.ndarray:
        p = self.params
        Z = np.where(np.isfinite(X), X, p["median"]).astype(np.float64)
        Z = np.clip(Z, p["lo"], p["hi"])
        return (Z - p["mean"]) / p["std"]

    def fit(self, X, y, **_):
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(X))
        if len(idx) > self.max_fit_rows:
            idx = np.sort(rng.choice(idx, self.max_fit_rows, replace=False))
        Xs, ys = X[idx].astype(np.float64), y[idx].astype(np.float64)
        med = np.nanmedian(Xs, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xs = np.where(np.isfinite(Xs), Xs, med)
        lo = np.quantile(Xs, self.clip_q, axis=0)
        hi = np.quantile(Xs, 1 - self.clip_q, axis=0)
        Xs = np.clip(Xs, lo, hi)
        mu, sd = Xs.mean(0), Xs.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        self.params = {"median": med, "lo": lo, "hi": hi, "mean": mu, "std": sd}
        Z = (Xs - mu) / sd
        ym = ys.mean()
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef = np.linalg.solve(A, Z.T @ (ys - ym))
        self.intercept = float(ym)
        return self

    def predict(self, X):
        out = np.empty(len(X))
        for i in range(0, len(X), 500_000):
            out[i:i + 500_000] = self._prep(X[i:i + 500_000]) @ self.coef + self.intercept
        return out

    def to_dict(self):
        return {"kind": self.kind, "alpha": self.alpha, "clip_q": self.clip_q, "coef": self.coef.tolist(),
                "intercept": self.intercept, **{k: v.tolist() for k, v in self.params.items()}}

    @classmethod
    def from_dict(cls, d):
        m = cls(alpha=d["alpha"], clip_q=d["clip_q"])
        m.params = {k: np.asarray(d[k]) for k in ("median", "lo", "hi", "mean", "std")}
        m.coef, m.intercept = np.asarray(d["coef"]), d["intercept"]
        return m


class LGBMModel:
    kind = "lgbm"

    def __init__(self, params: dict, n_estimators: int = 3000, early_stopping_rounds: int = 100,
                 categorical: list[str] | None = None, n_threads: int = 4, seed: int = 42, log_every: int = 100):
        self.params = dict(params)
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.categorical = categorical or []
        self.n_threads, self.seed, self.log_every = n_threads, seed, log_every
        self.booster: lgb.Booster | None = None
        self.best_iteration: int | None = None
        self.fit_seconds: float | None = None
        self.curve: list[float] = []

    def lgb_params(self) -> dict:
        p = {k: v for k, v in self.params.items() if k != "name"}
        p.update({"objective": p.get("objective", "l1"), "metric": "l1", "verbosity": -1,
                  "num_threads": self.n_threads, "seed": self.seed, "deterministic": True,
                  "force_col_wise": True})
        return p

    def fit(self, X, y, feature_names: list[str], X_val=None, y_val=None, num_boost_round: int | None = None):
        m = np.isfinite(y)
        cat = [c for c in self.categorical if c in feature_names]
        dtrain = lgb.Dataset(X[m], y[m], feature_name=feature_names, categorical_feature=cat, free_raw_data=True)
        valid, callbacks, evals = [], [], {}
        if X_val is not None:
            mv = np.isfinite(y_val)
            dval = lgb.Dataset(X_val[mv], y_val[mv], reference=dtrain, feature_name=feature_names, categorical_feature=cat)
            valid = [dval]
            callbacks.append(lgb.record_evaluation(evals))
            if num_boost_round is None:
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(self.log_every))
        t = time.time()
        self.booster = lgb.train(self.lgb_params(), dtrain, num_boost_round=num_boost_round or self.n_estimators,
                                 valid_sets=valid, valid_names=["val"] if valid else None, callbacks=callbacks)
        self.fit_seconds = time.time() - t
        self.best_iteration = self.booster.best_iteration or self.booster.current_iteration()
        if evals:
            self.curve = evals["val"]["l1"]
        return self

    def predict(self, X):
        return self.booster.predict(X, num_iteration=self.best_iteration, num_threads=self.n_threads)

    def save(self, path: Path):
        self.booster.save_model(str(path), num_iteration=self.best_iteration)

    @staticmethod
    def load(path: Path, n_threads: int = 1) -> "LoadedBooster":
        return LoadedBooster(lgb.Booster(model_file=str(path)), n_threads)


class LoadedBooster:
    kind = "lgbm"

    def __init__(self, booster: lgb.Booster, n_threads: int):
        self.booster, self.n_threads = booster, n_threads

    def predict(self, X):
        return self.booster.predict(X, num_threads=self.n_threads)


def save_linear(model, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(model.to_dict(), f)


def load_linear(path: Path):
    with open(path) as f:
        d = json.load(f)
    if d["kind"] == "zero":
        return ZeroModel()
    if d["kind"] == "median":
        return MedianModel(d["value"])
    return RidgeModel.from_dict(d)
