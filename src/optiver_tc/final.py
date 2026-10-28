"""Freeze every development choice, then fit on all pre-holdout dates and score the holdout ONCE.

`freeze` reads only development artifacts (experiment log + OOF predictions) and writes
frozen_choices.json. `run_final` refuses to run without it and refuses to overwrite existing
holdout predictions unless OPTIVER_ALLOW_HOLDOUT_RERUN=1 (a rerun is then logged).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import numpy as np
import pandas as pd

from . import features as F
from . import metrics as M
from .backtest import select_rule
from .config import Config, environment_info, read_json, write_json
from .data import manifest_digest
from .experiments import config_hash, fit_stats_for, load_oof, load_records, load_split, make_model
from .models import MedianModel
from .splits import dev_folds, final_split
from .streaming import ModelBundle

log = logging.getLogger(__name__)


def _blend_oof(cfg: Config, lgb_name: str, ridge_name: str | None) -> pd.DataFrame:
    a = load_oof(cfg, lgb_name)
    if ridge_name:
        b = load_oof(cfg, ridge_name)[["row_id", "pred"]].rename(columns={"pred": "pred_ridge"})
        a = a.merge(b, on="row_id", how="left")
    return a.rename(columns={"pred": "pred_lgb"})


def _pp_apply(p: np.ndarray, df: pd.DataFrame, demean: float, clip: float | None) -> np.ndarray:
    out = p.copy()
    if demean:
        key = df["date_id"].to_numpy(np.int64) * 10_000 + df["seconds_in_bucket"].to_numpy(np.int64)
        out = out - demean * pd.Series(out).groupby(key).transform("mean").to_numpy()
    if clip:
        out = np.clip(out, -clip, clip)
    return out


def freeze(cfg: Config) -> dict:
    sel = read_json(cfg.artifacts_dir / "selection" / "dev_selection.json")
    rec = load_records(cfg)
    folds = [f.name for f in dev_folds(cfg)]
    lgb = rec[rec["experiment"] == "lgb_final_dev"].drop_duplicates("fold", keep="last")
    n_iter = int(round(lgb["best_iteration"].mean()))
    # best ridge from stage A (mean dev MAE)
    ridge = rec[rec["experiment"].str.startswith("ridge_")].groupby("experiment")["mae"].mean()
    ridge_name = ridge.idxmin() if len(ridge) else None
    oof = _blend_oof(cfg, "lgb_final_dev", ridge_name)
    y = oof["target"].to_numpy(float)
    # --- blend weight on LightGBM (pooled dev OOF, 0.05 grid)
    blend = []
    for w in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        p = w * oof["pred_lgb"].to_numpy() + (1 - w) * (oof["pred_ridge"].to_numpy() if ridge_name else 0)
        blend.append({"w_lgb": float(w), "mae": M.mae(y, p),
                      "by_fold": {f: M.mae(y[oof["split"] == f], p[oof["split"] == f]) for f in folds}})
    best_blend = min(blend, key=lambda b: b["mae"])
    w = best_blend["w_lgb"] if ridge_name else 1.0
    base_pred = w * oof["pred_lgb"].to_numpy() + (1 - w) * (oof["pred_ridge"].to_numpy() if ridge_name else 0)
    # --- post-processing: demeaning strength and clipping (pooled dev OOF; must also win in >= 2/3 folds)
    pp_grid = []
    base_by_fold = {f: M.mae(y[oof["split"] == f], base_pred[oof["split"] == f]) for f in folds}
    for dm in (0.0, 0.5, 1.0):
        for clip in (None, 30.0, 15.0):
            p = _pp_apply(base_pred, oof, dm, clip)
            by = {f: M.mae(y[oof["split"] == f], p[oof["split"] == f]) for f in folds}
            pp_grid.append({"demean": dm, "clip": clip, "mae": M.mae(y, p), "by_fold": by,
                            "folds_better_than_none": int(sum(by[f] < base_by_fold[f] for f in folds))})
    none = next(g for g in pp_grid if g["demean"] == 0.0 and g["clip"] is None)
    cands = [g for g in pp_grid if g is none or (g["mae"] < none["mae"] and g["folds_better_than_none"] >= 2)]
    pp = min(cands, key=lambda g: g["mae"])
    final_oof = oof[["row_id", "time_id", "stock_id", "date_id", "seconds_in_bucket", "target", "split"]].copy()
    final_oof["pred"] = _pp_apply(base_pred, oof, pp["demean"], pp["clip"])
    final_oof.to_parquet(cfg.artifacts_dir / "oof" / "frozen_dev_oof.parquet", index=False)
    # --- backtest rule on dev OOF
    bt_sel = select_rule(final_oof, cfg)
    choices = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_hash": config_hash(cfg),
        "data": manifest_digest(cfg),
        "families": sel["families"],
        "feature_columns": F.feature_columns(sel["families"]),
        "lgbm_params": sel["params"],
        "lgbm_num_boost_round": n_iter,
        "lgbm_dev_best_iterations": lgb.set_index("fold")["best_iteration"].to_dict(),
        "ridge": {"experiment": ridge_name, "alpha": float(ridge_name.split("_a")[1]) if ridge_name else None},
        "blend": {"w_lgb": w, "grid": blend},
        "postprocess": {"demean": pp["demean"], "clip": pp["clip"], "grid": pp_grid,
                        "rule": "pooled dev MAE, must beat 'none' in >=2/3 folds"},
        "backtest": bt_sel,
        "holdout": final_split(cfg).to_dict(),
    }
    blob = json.dumps({k: v for k, v in choices.items() if k != "frozen_at"}, sort_keys=True, default=str)
    choices["sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    write_json(choices, cfg.artifacts_dir / "frozen_choices.json")
    log.info("frozen: families=%s iters=%s w_lgb=%s pp=%s rule=%s", sel["families"], n_iter, w,
             (pp["demean"], pp["clip"]), bt_sel["selected"])
    return choices


def run_final(cfg: Config) -> dict:
    fz_path = cfg.artifacts_dir / "frozen_choices.json"
    if not fz_path.exists():
        raise RuntimeError("run `freeze` first: holdout evaluation requires frozen choices")
    fz = read_json(fz_path)
    out_pred = cfg.artifacts_dir / "predictions" / "holdout_predictions.parquet"
    if out_pred.exists() and os.environ.get("OPTIVER_ALLOW_HOLDOUT_RERUN") != "1":
        raise RuntimeError(f"holdout already evaluated ({out_pred}); set OPTIVER_ALLOW_HOLDOUT_RERUN=1 to recompute")
    hs = final_split(cfg)
    cols = fz["feature_columns"]
    stats = fit_stats_for(cfg, hs.train_start, hs.train_end)
    tr = load_split(cfg, hs.train_start, hs.train_end, cols, stats)
    m = np.isfinite(tr.y)
    models, specs = [], []
    t0 = time.time()
    lgbm = make_model({"kind": "lgbm", "params": fz["lgbm_params"]}, cfg)
    lgbm.fit(tr.X, tr.y, cols, num_boost_round=fz["lgbm_num_boost_round"])
    lgbm.best_iteration = fz["lgbm_num_boost_round"]
    lgb_s = time.time() - t0
    models.append(lgbm)
    specs.append({"kind": "lgbm", "file": "lgbm.txt", "weight": fz["blend"]["w_lgb"], "columns": None})
    ridge_cols = [c for c in cols if c != "stock_id"]
    if fz["ridge"]["experiment"] and fz["blend"]["w_lgb"] < 1.0:
        ridge = make_model({"kind": "ridge", "alpha": fz["ridge"]["alpha"]}, cfg)
        idx = [cols.index(c) for c in ridge_cols]
        ridge.fit(tr.X[m][:, idx], tr.y[m])
        models.append(ridge)
        specs.append({"kind": "ridge", "file": "ridge.json", "weight": 1 - fz["blend"]["w_lgb"], "columns": ridge_cols})
    train_median = float(np.nanmedian(tr.y))
    del tr
    schema = {
        "feature_columns": cols, "families": fz["families"], "categorical": F.CATEGORICAL, "models": specs,
        "postprocess": {"demean": bool(fz["postprocess"]["demean"]), "demean_strength": fz["postprocess"]["demean"],
                        "clip": fz["postprocess"]["clip"]},
        "fallback": train_median, "history_seconds": F.HISTORY_SECONDS, "train_dates": [hs.train_start, hs.train_end],
        "frozen_sha256": fz["sha256"], "feature_version": "v1",
    }
    model_dir = cfg.artifacts_dir / "model"
    ModelBundle.save(model_dir, schema, models, stats)
    bundle = ModelBundle.load(model_dir, cfg.n_threads)
    # ---- holdout scoring (research batch path through the saved bundle)
    va = load_split(cfg, hs.val_start, hs.val_end, cols, stats)
    pred = bundle.predict_matrix(va.X, va.meta["date_id"].to_numpy(), va.meta["seconds_in_bucket"].to_numpy())
    preds = va.meta[["row_id", "time_id", "stock_id", "date_id", "seconds_in_bucket", "target"]].copy()
    preds["pred"] = pred
    preds["pred_lgb_raw"] = models[0].predict(va.X)
    if len(models) > 1:
        preds["pred_ridge_raw"] = models[1].predict(va.X[:, [cols.index(c) for c in ridge_cols]])
    preds["pred_median"] = train_median
    preds["split"] = "holdout"
    preds["train_dates"] = f"{hs.train_start}-{hs.train_end}"
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_pred, index=False)
    res = {"holdout_dates": [hs.val_start, hs.val_end], "train_dates": [hs.train_start, hs.train_end],
           "frozen_sha256": fz["sha256"], "lgbm_fit_seconds": round(lgb_s, 1), "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "final_model": M.summary(preds, "pred"),
           "lgbm_only_raw": M.summary(preds, "pred_lgb_raw"),
           "zero": {"mae": M.mae(preds["target"], np.zeros(len(preds))), "rmse": M.rmse(preds["target"], np.zeros(len(preds)))},
           "train_median": {"value": train_median, "mae": M.mae(preds["target"], preds["pred_median"])},
           "environment": environment_info()}
    if "pred_ridge_raw" in preds:
        res["ridge_only"] = M.summary(preds, "pred_ridge_raw")
    res["improvement_vs_zero_pct"] = 100 * (1 - res["final_model"]["mae"] / res["zero"]["mae"])
    write_json(res, cfg.artifacts_dir / "holdout_metrics.json")
    for key in ("date_id", "stock_id", "seconds_in_bucket"):
        M.mae_by(preds, key).to_csv(cfg.artifacts_dir / "predictions" / f"holdout_mae_by_{key}.csv", index=False)
    log.info("HOLDOUT MAE final=%.5f zero=%.5f median=%.5f", res["final_model"]["mae"], res["zero"]["mae"], res["train_median"]["mae"])
    return res
