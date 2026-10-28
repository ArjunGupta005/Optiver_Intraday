"""Model bundle (serialised models + preprocessing + schema) and the stateful streaming predictor.

Batch research and streaming both call `ModelBundle.predict_features` on frames produced by
`features.build_features`, so there is a single implementation of features, blending and post-processing.

Streaming state policy
----------------------
* State = raw rows of the CURRENT date with seconds_in_bucket >= t - history_seconds (bounded by
  n_stocks x (history_seconds/10 + 1) rows). It is cleared whenever date_id changes.
* A batch may contain one or more timestamps; each timestamp is processed in ascending order using
  only state strictly earlier than it, then appended to the state.
* Duplicate batch (same date_id and seconds as the latest processed timestamp): recomputed with the
  previous rows for that timestamp replaced; deterministic for identical input.
* Out-of-order batch (earlier seconds on the current date, or an earlier date_id): strict mode raises;
  default mode predicts using only state strictly before that timestamp (history is filtered) and,
  for an earlier date, resets state as for any date change. Future batches are never read.
* Duplicate stock rows inside a batch: the first occurrence is used for features; every input row
  still gets a prediction (looked up by stock_id), preserving the incoming row order.
* Non-finite predictions (e.g. a row with no book data) fall back to the training median, then the
  configured clip is applied. Exactly one finite float is returned per input row.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from . import features as F
from .config import read_json, write_json
from .data import SCHEMA
from .models import LGBMModel, load_linear, save_linear

log = logging.getLogger(__name__)
RAW_INPUTS = ["stock_id", "date_id", "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag",
              "reference_price", "matched_size", "far_price", "near_price", "bid_price", "bid_size",
              "ask_price", "ask_size", "wap"]


class ModelBundle:
    def __init__(self, schema: dict, models: list, stock_stats: pl.DataFrame | None):
        self.schema, self.models, self.stock_stats = schema, models, stock_stats
        self.columns: list[str] = schema["feature_columns"]

    # ---------------------------------------------------------------- persistence
    @classmethod
    def load(cls, d: str | Path, n_threads: int = 1) -> "ModelBundle":
        d = Path(d)
        schema = read_json(d / "schema.json")
        models = []
        for m in schema["models"]:
            if m["kind"] == "lgbm":
                models.append(LGBMModel.load(d / m["file"], n_threads))
            else:
                models.append(load_linear(d / m["file"]))
        st = None
        if (d / "stock_stats.json").exists():
            st = pl.DataFrame(read_json(d / "stock_stats.json")).with_columns(pl.col("stock_id").cast(pl.Int16))
        return cls(schema, models, st)

    @staticmethod
    def save(d: str | Path, schema: dict, models: list, stock_stats: pl.DataFrame | None) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        for spec, model in zip(schema["models"], models):
            if spec["kind"] == "lgbm":
                model.save(d / spec["file"])
            else:
                save_linear(model, d / spec["file"])
        if stock_stats is not None:
            write_json(stock_stats.to_dicts(), d / "stock_stats.json")
        write_json(schema, d / "schema.json")

    # ---------------------------------------------------------------- prediction
    def raw_model_outputs(self, feats: pl.DataFrame) -> np.ndarray:
        X = feats.select([pl.col(c).cast(pl.Float32) for c in self.columns]).to_numpy()
        return self.raw_from_matrix(X)

    def raw_from_matrix(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(len(X))
        for spec, model in zip(self.schema["models"], self.models):
            cols = spec.get("columns")
            Xi = X if cols is None else X[:, [self.columns.index(c) for c in cols]]
            out += spec["weight"] * model.predict(Xi)
        return out

    def postprocess(self, pred: np.ndarray, date_id: np.ndarray, seconds: np.ndarray) -> np.ndarray:
        pp = self.schema.get("postprocess", {})
        p = pred.astype(np.float64).copy()
        fb = float(self.schema.get("fallback", 0.0))
        p[~np.isfinite(p)] = fb
        if pp.get("demean"):
            key = date_id.astype(np.int64) * 10_000 + seconds.astype(np.int64)
            _, inv = np.unique(key, return_inverse=True)
            means = np.bincount(inv, weights=p) / np.bincount(inv)
            p = p - float(pp.get("demean_strength", 1.0)) * means[inv]
        clip = pp.get("clip")
        if clip:
            p = np.clip(p, -clip, clip)
        p[~np.isfinite(p)] = fb
        return p

    def predict_matrix(self, X: np.ndarray, date_id: np.ndarray, seconds: np.ndarray) -> np.ndarray:
        return self.postprocess(self.raw_from_matrix(X), date_id, seconds)

    def predict_features(self, feats: pl.DataFrame) -> np.ndarray:
        raw = self.raw_model_outputs(feats)
        return self.postprocess(raw, feats["date_id"].to_numpy(), feats["seconds_in_bucket"].to_numpy())

    def predict_batch_frame(self, raw: pl.DataFrame) -> np.ndarray:
        """Research path: features for whole dates at once, then the same prediction code."""
        feats = F.build_features(_coerce(raw), self.stock_stats)
        return self.predict_features(feats)


def _coerce(df: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    exprs = []
    for c in RAW_INPUTS:
        if c not in df.columns:
            exprs.append(pl.lit(None, dtype=SCHEMA[c]).alias(c))
        else:
            e = pl.col(c).cast(SCHEMA[c], strict=False)
            if SCHEMA[c] == pl.Float64:
                e = e.fill_nan(None)
            exprs.append(e)
    return df.select(exprs)


class StreamingPredictor:
    def __init__(self, bundle: ModelBundle, history_seconds: int | None = None, strict_order: bool = False):
        self.bundle = bundle
        self.history_seconds = int(history_seconds or bundle.schema.get("history_seconds", F.HISTORY_SECONDS))
        if self.history_seconds < F.HISTORY_SECONDS:
            raise ValueError(f"history_seconds must be >= {F.HISTORY_SECONDS}")
        self.strict = strict_order
        self.state: pl.DataFrame | None = None
        self.cur_date: int | None = None
        self.last_sec: int | None = None
        self.n_batches = 0
        self.events: list[str] = []

    @classmethod
    def from_dir(cls, d: str | Path, n_threads: int = 1, **kw) -> "StreamingPredictor":
        return cls(ModelBundle.load(d, n_threads), **kw)

    def reset(self) -> None:
        self.state, self.cur_date, self.last_sec = None, None, None

    def _predict_timestamp(self, rows: pl.DataFrame, date: int, sec: int) -> pl.DataFrame:
        hist = None
        if date != self.cur_date:
            if self.cur_date is not None and date < self.cur_date:
                self._order_issue(f"date {date} after {self.cur_date}")
            self.state, self.cur_date, self.last_sec = None, date, None
        elif self.last_sec is not None and sec < self.last_sec:
            self._order_issue(f"seconds {sec} after {self.last_sec} on date {date}")
        mutate = self.last_sec is None or sec >= self.last_sec
        if self.state is not None:
            hist = self.state.filter((pl.col("seconds_in_bucket") < sec) &
                                     (pl.col("seconds_in_bucket") >= sec - self.history_seconds))
        uniq = rows.unique(subset=["stock_id"], keep="first", maintain_order=True)
        combined = uniq if hist is None or hist.height == 0 else pl.concat([hist, uniq], how="vertical")
        cur = F.build_features(combined, self.bundle.stock_stats, output_seconds=sec)
        pred = self.bundle.predict_features(cur)
        out = cur.select("stock_id").with_columns(pl.Series("pred", pred))
        if mutate:
            base = self.state.filter(pl.col("seconds_in_bucket") != sec) if self.state is not None else None
            new = uniq if base is None else pl.concat([base, uniq], how="vertical")
            self.state = new.filter(pl.col("seconds_in_bucket") >= sec - self.history_seconds)
            if self.last_sec is not None and sec == self.last_sec:
                self.events.append(f"duplicate timestamp {date}/{sec} recomputed")
            self.last_sec = sec
        return out

    def _order_issue(self, msg: str) -> None:
        if self.strict:
            raise ValueError(f"out-of-order batch: {msg}")
        log.warning("out-of-order batch handled causally: %s", msg)
        self.events.append(f"out-of-order: {msg}")

    def predict(self, test: pd.DataFrame | pl.DataFrame) -> np.ndarray:
        """One finite prediction per input row, in the input row order."""
        raw = _coerce(test)
        n = raw.height
        if n == 0:
            return np.zeros(0)
        raw = raw.with_row_index("__pos")
        result = np.full(n, np.nan)
        stamps = raw.select(["date_id", "seconds_in_bucket"]).unique(maintain_order=True)
        stamps = stamps.sort(["date_id", "seconds_in_bucket"])  # process in chronological order within a call
        for date, sec in stamps.iter_rows():
            rows = raw.filter((pl.col("date_id") == date) & (pl.col("seconds_in_bucket") == sec))
            out = self._predict_timestamp(rows.drop("__pos"), int(date), int(sec))
            m = rows.select(["__pos", "stock_id"]).join(out, on="stock_id", how="left", maintain_order="left")
            result[m["__pos"].to_numpy()] = m["pred"].to_numpy()
        self.n_batches += 1
        fb = float(self.bundle.schema.get("fallback", 0.0))
        result[~np.isfinite(result)] = fb
        return result


# ------------------------------------------------------------------------------------- replay
def iter_timestamps(df: pl.DataFrame):
    """Chronological batches: one (date_id, seconds_in_bucket) cross-section at a time."""
    df = df.sort(["date_id", "seconds_in_bucket"], maintain_order=True)
    for _, g in df.group_by(["date_id", "seconds_in_bucket"], maintain_order=True):
        yield g


def replay(predictor: StreamingPredictor, df: pl.DataFrame, timings: list | None = None) -> pl.DataFrame:
    """Offline chronological replay over a raw frame. Returns row_id + prediction."""
    outs = []
    for batch in iter_timestamps(df):
        t = time.perf_counter()
        p = predictor.predict(batch)
        if timings is not None:
            timings.append(time.perf_counter() - t)
        outs.append(batch.select(["row_id", "date_id", "seconds_in_bucket", "stock_id"]).with_columns(pl.Series("pred", p)))
    return pl.concat(outs)
