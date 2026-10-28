"""End-to-end smoke: feature store -> dev research -> freeze -> final holdout -> backtest -> streaming replay,
on a small synthetic chronological panel written to a temp directory (no dependence on the real data)."""
import copy

import numpy as np
import pandas as pd
import pytest

from optiver_tc.config import load_config, read_json
from optiver_tc.featurestore import build_feature_store
from optiver_tc.final import freeze, run_final
from optiver_tc.reporting import backtest_outputs, build_reports
from optiver_tc.replay_cli import run_replay
from optiver_tc.research import run_all

from conftest import make_raw


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    c = load_config("configs/smoke.toml")
    c = copy.deepcopy(c)
    r = c.raw
    r["name"] = "e2e"
    r["artifacts_dir"] = str(tmp / "artifacts")
    r["reports_dir"] = str(tmp / "reports")
    r["data"]["processed_dir"] = str(tmp / "processed")
    r["data"]["max_date_id"] = -1
    r["resources"]["n_threads"] = 1
    r["splits"].update({"holdout_start": 26, "holdout_end": 29,
                        "folds": [{"name": "fold1", "val_start": 17, "val_end": 19},
                                  {"name": "fold2", "val_start": 20, "val_end": 22},
                                  {"name": "fold3", "val_start": 23, "val_end": 25}]})
    r["lgbm"]["n_estimators"] = 25
    r["lgbm"]["early_stopping_rounds"] = 5
    for cand in r["lgbm"]["candidates"]:
        cand["min_data_in_leaf"] = 50
    r["lgbm"]["candidates"] = r["lgbm"]["candidates"][:2]
    r["ridge"]["alphas"] = [10.0]
    r["backtest"].update({"min_names_per_side": 5, "bootstrap_reps": 50, "weight_caps": [0.2], "selection_quantiles": [1.0, 0.4]})
    (tmp / "processed").mkdir()
    make_raw(n_dates=30, n_stocks=24, seed=7).write_parquet(tmp / "processed" / "train.parquet")
    return c


def test_end_to_end_pipeline(cfg):
    build_feature_store(cfg, force=True)
    sel = run_all(cfg)
    assert set(sel) == {"families", "params"}
    fz = freeze(cfg)
    assert fz["lgbm_num_boost_round"] >= 1 and (cfg.artifacts_dir / "frozen_choices.json").exists()
    res = run_final(cfg)
    assert np.isfinite(res["final_model"]["mae"])
    with pytest.raises(RuntimeError):            # holdout is scored once
        run_final(cfg)
    preds = pd.read_parquet(cfg.artifacts_dir / "predictions" / "holdout_predictions.parquet")
    assert preds["date_id"].between(26, 29).all() and preds["row_id"].is_unique
    oof = pd.read_parquet(cfg.artifacts_dir / "oof" / "frozen_dev_oof.parquet")
    assert oof["date_id"].max() < 26                # dev predictions never touch the holdout
    bt = backtest_outputs(cfg)
    assert "holdout" in bt["results"]
    rep = run_replay(cfg, source="holdout")
    assert rep["all_finite"] and rep["rows"] == len(preds)
    assert rep["max_abs_diff_stream_vs_batch_frame"] < 1e-9
    build_reports(cfg)
    assert (cfg.reports_dir / "validation_report_e2e.md").exists()
    # reported metric is reproducible from the saved predictions
    m = read_json(cfg.artifacts_dir / "holdout_metrics.json")
    assert abs(m["final_model"]["mae"] - float(np.nanmean(np.abs(preds["target"] - preds["pred"])))) < 1e-9
