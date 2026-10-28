"""Development-fold experiments: baselines, LightGBM configs, feature-family ablations.

Every run appends one JSON line per (experiment, fold) to artifacts/<name>/experiments.jsonl and
writes out-of-fold predictions with row identifiers and split provenance to artifacts/<name>/oof/.
Nothing in this module reads holdout dates.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from . import features as F
from . import metrics as M
from .config import Config, write_json
from .data import manifest_digest
from .featurestore import load_frame, load_matrix
from .models import LGBMModel, MedianModel, RidgeModel, ZeroModel
from .splits import Fold, dev_folds

log = logging.getLogger(__name__)
META_COLS = ["row_id", "time_id", "stock_id", "date_id", "seconds_in_bucket", "target"]
STOCK_INPUTS = ["stock_id", "log_matched", "spread_bps", "log_quoted", "wap_ret_10"]


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    columns: list[str]


def config_hash(cfg: Config) -> str:
    return hashlib.sha256(json.dumps(cfg.raw, sort_keys=True, default=str).encode()).hexdigest()[:12]


def sample_dates(lo: int, hi: int, frac: float, seed: int) -> list[int] | None:
    if frac >= 1.0:
        return None
    rng = np.random.default_rng(seed)
    d = np.arange(lo, hi + 1)
    return sorted(rng.choice(d, max(1, int(round(len(d) * frac))), replace=False).tolist())


def fit_stats_for(cfg: Config, lo: int, hi: int) -> pl.DataFrame:
    """Per-stock statistics from TRAINING dates [lo, hi] only."""
    return F.fit_stock_stats(load_frame(cfg, lo, hi, STOCK_INPUTS))


def load_split(cfg: Config, lo: int, hi: int, columns: list[str], stock_stats: pl.DataFrame | None,
               dates: list[int] | None = None) -> SplitData:
    unfitted = [c for c in columns if c not in F.STOCK]
    X = load_matrix(cfg, lo, hi, unfitted, dates)
    meta = load_frame(cfg, lo, hi, META_COLS, dates)
    stock_cols = [c for c in columns if c in F.STOCK]
    if stock_cols:
        assert stock_stats is not None
        sf = F.stock_features(load_frame(cfg, lo, hi, STOCK_INPUTS, dates), stock_stats)
        S = sf.select([pl.col(c).cast(pl.Float32) for c in stock_cols]).to_numpy()
        X = np.concatenate([X, S], axis=1)
        del S, sf
    order = unfitted + stock_cols
    if order != columns:
        idx = [order.index(c) for c in columns]
        X = X[:, idx]
    y = meta["target"].to_numpy().astype(np.float64)
    y = np.where(np.isnan(y), np.nan, y)
    return SplitData(X=X, y=y, meta=meta.to_pandas(), columns=columns)


def make_model(spec: dict, cfg: Config):
    kind = spec["kind"]
    if kind == "zero":
        return ZeroModel()
    if kind == "median":
        return MedianModel()
    if kind == "ridge":
        return RidgeModel(alpha=spec["alpha"], clip_q=cfg["ridge"]["clip_quantile"], seed=cfg["seed"])
    if kind == "lgbm":
        lg = cfg["lgbm"]
        params = {"objective": lg["objective"], **{k: v for k, v in spec["params"].items() if k != "name"}}
        return LGBMModel(params, n_estimators=spec.get("n_estimators", lg["n_estimators"]),
                         early_stopping_rounds=lg["early_stopping_rounds"], categorical=F.CATEGORICAL,
                         n_threads=cfg.n_threads, seed=cfg["seed"], log_every=lg["log_every"])
    raise ValueError(kind)


def evaluate(meta: pd.DataFrame, pred: np.ndarray) -> dict:
    df = meta[["date_id", "seconds_in_bucket", "stock_id", "target"]].copy()
    df["pred"] = pred
    return M.summary(df)


def append_record(cfg: Config, rec: dict) -> None:
    p = cfg.artifacts_dir / "experiments.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def save_oof(cfg: Config, exp: str, fold: Fold, meta: pd.DataFrame, pred: np.ndarray) -> Path:
    d = cfg.artifacts_dir / "oof"
    d.mkdir(parents=True, exist_ok=True)
    out = meta[["row_id", "time_id", "stock_id", "date_id", "seconds_in_bucket", "target"]].copy()
    out["pred"] = pred.astype(np.float64)
    out["experiment"] = exp
    out["split"] = fold.name
    out["train_dates"] = f"{fold.train_start}-{fold.train_end}"
    p = d / f"{exp}__{fold.name}.parquet"
    out.to_parquet(p, index=False)
    return p


def done_set(cfg: Config) -> set[tuple[str, str]]:
    p = cfg.artifacts_dir / "experiments.jsonl"
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.add((r["experiment"], r["fold"]))
    return out


def run_experiments(cfg: Config, specs: list[dict], folds: list[Fold] | None = None, resume: bool = True) -> None:
    """specs: [{"name", "kind", "families", ...model params..., "date_fraction"}]. Resumable."""
    folds = folds or dev_folds(cfg)
    done = done_set(cfg) if resume else set()
    for fold in folds:
        todo = [s for s in specs if (s["name"], fold.name) not in done]
        if not todo:
            continue
        stats = fit_stats_for(cfg, fold.train_start, fold.train_end)
        write_json(stats.to_dicts(), cfg.artifacts_dir / "stock_stats" / f"{fold.name}.json")
        # group specs by (families, date_fraction) so each matrix is loaded once
        groups: dict[tuple, list[dict]] = {}
        for s in todo:
            groups.setdefault((tuple(s["families"]), s.get("date_fraction", 1.0), tuple(s.get("exclude", []))), []).append(s)
        for (fams, frac, excl), group in groups.items():
            cols = [c for c in F.feature_columns(list(fams)) if c not in excl]
            tr_dates = sample_dates(fold.train_start, fold.train_end, frac, cfg["seed"])
            t0 = time.time()
            tr = load_split(cfg, fold.train_start, fold.train_end, cols, stats, tr_dates)
            va = load_split(cfg, fold.val_start, fold.val_end, cols, stats)
            log.info("%s %s: loaded train %s val %s in %.1fs", fold.name, fams, tr.X.shape, va.X.shape, time.time() - t0)
            for spec in group:
                model = make_model(spec, cfg)
                t = time.time()
                if spec["kind"] == "lgbm":
                    model.fit(tr.X, tr.y, cols, va.X, va.y)
                else:
                    m = np.isfinite(tr.y)
                    model.fit(tr.X[m], tr.y[m])
                fit_s = time.time() - t
                pred = model.predict(va.X)
                met = evaluate(va.meta, pred)
                rec = {
                    "experiment": spec["name"], "fold": fold.name, "kind": spec["kind"], "families": list(fams),
                    "n_features": len(cols), "excluded": list(excl), "date_fraction": frac, "n_train_rows": int(np.isfinite(tr.y).sum()),
                    "train_dates": [fold.train_start, fold.train_end], "val_dates": [fold.val_start, fold.val_end],
                    "params": spec.get("params") or {k: v for k, v in spec.items() if k in ("alpha",)},
                    "best_iteration": getattr(model, "best_iteration", None), "fit_seconds": round(fit_s, 1),
                    "metrics": met, "config_hash": config_hash(cfg), "data": manifest_digest(cfg),
                    "seed": cfg["seed"], "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                if spec["kind"] == "lgbm":
                    rec["val_curve_every_50"] = model.curve[::50]
                    imp_dir = cfg.artifacts_dir / "importance"
                    imp_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({"feature": cols,
                                  "gain": model.booster.feature_importance("gain", iteration=model.best_iteration),
                                  "split": model.booster.feature_importance("split", iteration=model.best_iteration)}
                                 ).to_csv(imp_dir / f"{spec['name']}__{fold.name}.csv", index=False)
                append_record(cfg, rec)
                save_oof(cfg, spec["name"], fold, va.meta, pred)
                log.info("%s %s MAE=%.5f it=%s fit=%.0fs", spec["name"], fold.name, met["mae"], rec["best_iteration"], fit_s)
                del model
                gc.collect()
            del tr, va
            gc.collect()


def load_records(cfg: Config) -> pd.DataFrame:
    p = cfg.artifacts_dir / "experiments.jsonl"
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    flat = []
    for r in rows:
        f = {k: v for k, v in r.items() if k not in ("metrics", "val_curve_every_50", "data")}
        f.update(r["metrics"])
        flat.append(f)
    return pd.DataFrame(flat)


def load_oof(cfg: Config, exp: str) -> pd.DataFrame:
    d = cfg.artifacts_dir / "oof"
    return pd.concat([pd.read_parquet(p) for p in sorted(d.glob(f"{exp}__fold*.parquet"))], ignore_index=True)
