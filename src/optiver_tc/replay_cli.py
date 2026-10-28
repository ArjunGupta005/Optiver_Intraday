"""Offline replay: (1) the vendor-style mock API over the example test files, (2) the holdout dates.

Both are OFFLINE MOCK REPLAYS (the compiled Kaggle API cannot load on this machine). The Kaggle path
is `scripts/kaggle_inference.py`, which uses `optiver2023.make_env()` and the same predictor.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import polars as pl

from .config import Config, read_json, write_json
from .data import load_train
from .mock_api import make_env
from .streaming import ModelBundle, StreamingPredictor, replay

log = logging.getLogger(__name__)


def predict_loop(env, predictor: StreamingPredictor) -> dict:
    """The exact loop used on Kaggle (see scripts/kaggle_inference.py)."""
    n, rows = 0, 0
    for test, revealed_targets, sample_prediction in env.iter_test():
        # revealed_targets are deliberately ignored by the primary model (no target-derived features)
        pred = pd.Series(predictor.predict(test), index=test["row_id"].to_numpy())
        sample_prediction["target"] = pred.reindex(sample_prediction["row_id"].to_numpy()).to_numpy()
        assert np.isfinite(sample_prediction["target"]).all() and len(sample_prediction) == len(test)
        env.predict(sample_prediction)
        n += 1
        rows += len(test)
    return {"batches": n, "rows": rows}


def run_replay(cfg: Config, source: str = "example") -> dict:
    model_dir = cfg.artifacts_dir / "model"
    bundle = ModelBundle.load(model_dir, n_threads=1)
    out_dir = cfg.artifacts_dir / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"label": "OFFLINE MOCK REPLAY (not the official Kaggle API)", "source": source}
    if source == "example":
        sub_path = out_dir / "submission.csv"
        env = make_env(cfg.raw_dir, sub_path)
        t = time.time()
        res.update(predict_loop(env, StreamingPredictor(bundle)))
        res["seconds"] = round(time.time() - t, 2)
        sub = pd.read_csv(sub_path)
        samp = pd.read_csv(cfg.raw_dir / "example_test_files" / "sample_submission.csv")
        test = pd.read_csv(cfg.raw_dir / "example_test_files" / "test.csv")
        res["submission_rows"] = len(sub)
        res["submission_schema_matches_sample"] = list(sub.columns) == list(samp.columns)
        res["row_ids_match_sample_order"] = bool((sub["row_id"].to_numpy() == samp["row_id"].to_numpy()).all())
        res["all_finite"] = bool(np.isfinite(sub["target"]).all())
        batch = bundle.predict_batch_frame(pl.from_pandas(test))
        res["max_abs_diff_vs_batch_features"] = float(np.max(np.abs(sub["target"].to_numpy() - batch)))
    else:
        fz = read_json(cfg.artifacts_dir / "frozen_choices.json")
        hs = fz["holdout"]
        raw = load_train(cfg).filter(pl.col("date_id").is_between(hs["val_start"], hs["val_end"]))
        timings: list[float] = []
        t = time.time()
        stream = replay(StreamingPredictor(bundle), raw.drop("target"), timings)
        res["seconds"] = round(time.time() - t, 1)
        res["batches"] = len(timings)
        res["rows"] = stream.height
        res["all_finite"] = bool(np.isfinite(stream["pred"].to_numpy()).all())
        batch = raw.select("row_id").with_columns(pl.Series("pred_batch", bundle.predict_batch_frame(raw)))
        m = stream.join(batch, on="row_id")
        res["max_abs_diff_stream_vs_batch_frame"] = float((m["pred"] - m["pred_batch"]).abs().max())
        hp = pd.read_parquet(cfg.artifacts_dir / "predictions" / "holdout_predictions.parquet", columns=["row_id", "pred"])
        m2 = m.to_pandas().merge(hp, on="row_id", suffixes=("", "_store"))
        d = (m2["pred"] - m2["pred_store"]).abs()
        res["vs_feature_store_predictions"] = {"max_abs_diff": float(d.max()), "p99_abs_diff": float(d.quantile(0.99)),
                                               "share_rows_diff_gt_1e-6": float((d > 1e-6).mean())}
        res["note"] = ("Streaming and batch-frame paths share features.py and ModelBundle and compute in float64; "
                       "the research feature store holds float32 copies, so its predictions can differ by tiny amounts "
                       "where a tree split threshold falls between the float32 and float64 value of a feature.")
        stream.write_parquet(out_dir / "holdout_stream_predictions.parquet")
    write_json(res, out_dir / f"replay_{source}.json")
    log.info("replay %s: %s", source, res)
    return res
