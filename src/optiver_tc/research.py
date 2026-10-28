"""Staged, bounded development research with pre-declared, automated selection rules.

Stage A  baselines: zero, train median, ridge (alpha grid) on all three dev folds.
Stage B  feature-family ablation (LightGBM 'lgb_base', learning_rate 0.1, 50% of training dates):
         all families vs leave-one-family-out. Rule: drop a family only if removing it lowers the
         mean dev MAE AND lowers MAE in at least 2 of 3 folds.
Stage C  LightGBM configuration search on the selected families (configured learning rate, 50% dates).
         Rule: lowest mean dev MAE.
Stage D  selected configuration on 100% of each fold's training dates -> development OOF predictions
         used for post-processing, blending, iteration count and backtest-rule selection.
All rules only read development-fold records.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import features as F
from .config import Config, write_json
from .experiments import load_records, run_experiments

log = logging.getLogger(__name__)


def _mean_mae(rec: pd.DataFrame, name: str, n_folds: int) -> tuple[float, pd.Series]:
    r = rec[rec["experiment"] == name].drop_duplicates(["experiment", "fold"], keep="last").set_index("fold")["mae"]
    if len(r) < n_folds:
        raise RuntimeError(f"{name}: only {len(r)} folds recorded")
    return float(r.mean()), r


def stage_a(cfg: Config) -> None:
    fams = list(F.FAMILIES)
    specs = [{"name": "zero", "kind": "zero", "families": fams},
             {"name": "median", "kind": "median", "families": fams}]
    for a in cfg["ridge"]["alphas"]:
        specs.append({"name": f"ridge_a{a:g}", "kind": "ridge", "alpha": float(a), "families": fams, "exclude": ["stock_id"]})
    run_experiments(cfg, specs)


def _base_params(cfg: Config) -> dict:
    return dict(cfg["lgbm"]["candidates"][0])


def stage_b(cfg: Config) -> list[str]:
    fams = cfg["features"]["families"]
    frac = cfg["resources"]["ablation_date_fraction"]
    p = {**_base_params(cfg), "learning_rate": 0.1}
    specs = [{"name": "abl_all", "kind": "lgbm", "params": p, "families": fams, "date_fraction": frac}]
    for f in fams:
        specs.append({"name": f"abl_no_{f}", "kind": "lgbm", "params": p,
                      "families": [x for x in fams if x != f], "date_fraction": frac})
    run_experiments(cfg, specs)
    rec = load_records(cfg)
    n = len(cfg["splits"]["folds"])
    base_mean, base = _mean_mae(rec, "abl_all", n)
    keep, table = [], []
    for f in fams:
        m, r = _mean_mae(rec, f"abl_no_{f}", n)
        wins = int((r < base.reindex(r.index)).sum())
        drop = (m < base_mean) and wins >= 2
        table.append({"family": f, "mae_without": m, "delta_vs_all": m - base_mean, "folds_better_without": wins, "dropped": drop})
        if not drop:
            keep.append(f)
    write_json({"rule": "drop family iff mean MAE without it < with it AND better in >=2/3 folds",
                "mae_all": base_mean, "table": table, "selected_families": keep},
               cfg.artifacts_dir / "selection" / "stage_b_families.json")
    return keep


def stage_c(cfg: Config, fams: list[str]) -> dict:
    frac = cfg["resources"]["ablation_date_fraction"]
    specs = [{"name": f"cfg_{c['name']}", "kind": "lgbm", "params": dict(c), "families": fams, "date_fraction": frac}
             for c in cfg["lgbm"]["candidates"]]
    run_experiments(cfg, specs)
    rec = load_records(cfg)
    n = len(cfg["splits"]["folds"])
    res = []
    for s, c in zip(specs, cfg["lgbm"]["candidates"]):
        m, r = _mean_mae(rec, s["name"], n)
        res.append({"config": c["name"], "mean_mae": m, "by_fold": r.to_dict()})
    best = min(res, key=lambda x: x["mean_mae"])
    chosen = next(c for c in cfg["lgbm"]["candidates"] if c["name"] == best["config"])
    write_json({"rule": "lowest mean dev MAE", "results": res, "selected": dict(chosen)},
               cfg.artifacts_dir / "selection" / "stage_c_config.json")
    return dict(chosen)


def stage_d(cfg: Config, fams: list[str], params: dict) -> None:
    run_experiments(cfg, [{"name": "lgb_final_dev", "kind": "lgbm", "params": params, "families": fams, "date_fraction": 1.0}])


def run_all(cfg: Config, stages: str = "ABCD") -> dict:
    if "A" in stages:
        stage_a(cfg)
    fams = stage_b(cfg) if "B" in stages else cfg["features"]["families"]
    params = stage_c(cfg, fams) if "C" in stages else _base_params(cfg)
    if "D" in stages:
        stage_d(cfg, fams, params)
    out = {"families": fams, "params": params}
    write_json(out, cfg.artifacts_dir / "selection" / "dev_selection.json")
    return out
