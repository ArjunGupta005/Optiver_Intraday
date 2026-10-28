"""Batch feature store: unfitted features computed once, chunked by whole dates.

Chunking by date is exact because no feature crosses a date boundary (intraday state resets).
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.dataset as ds

from . import features as F
from .config import Config, write_json
from .data import load_train

ID_COLS = ["row_id", "time_id", "stock_id", "date_id", "seconds_in_bucket", "target"]
# raw columns kept in the store for the WAP-based backtest robustness check (not model inputs)
EXTRA_RAW = ["wap"]


def store_path(cfg: Config) -> Path:
    tag = cfg["name"]
    return cfg.processed_dir / f"features_{tag}.parquet"


def unfitted_columns() -> list[str]:
    cols = []
    for f in F.UNFITTED_FAMILIES:
        cols += F.FAMILIES[f]
    return cols


def build_feature_store(cfg: Config, chunk_dates: int = 40, force: bool = False) -> Path:
    out = store_path(cfg)
    if out.exists() and not force:
        return out
    raw = load_train(cfg)
    dates = np.sort(raw["date_id"].unique().to_numpy())
    feat_cols = unfitted_columns()
    parts = []
    tmp_dir = out.parent / f"_tmp_{cfg['name']}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(dates), chunk_dates):
        sub = raw.filter(pl.col("date_id").is_between(dates[i], dates[min(i + chunk_dates, len(dates)) - 1]))
        f = F.build_unfitted(sub)
        f = f.select(ID_COLS + EXTRA_RAW + [pl.col(c).cast(pl.Float32) for c in feat_cols])
        p = tmp_dir / f"part_{i:04d}.parquet"
        f.write_parquet(p, compression="zstd", row_group_size=100_000)
        parts.append(p)
        del sub, f
        gc.collect()
    del raw
    gc.collect()
    pl.scan_parquet(parts).sink_parquet(out, compression="zstd", row_group_size=100_000)
    for p in parts:
        p.unlink()
    tmp_dir.rmdir()
    write_json({"columns": ID_COLS + EXTRA_RAW + feat_cols, "feature_version": "v1"}, out.with_suffix(".schema.json"))
    return out


def load_frame(cfg: Config, date_lo: int, date_hi: int, columns: list[str], dates: list[int] | None = None) -> pl.DataFrame:
    lf = pl.scan_parquet(store_path(cfg)).filter(pl.col("date_id").is_between(date_lo, date_hi))
    if dates is not None:
        lf = lf.filter(pl.col("date_id").is_in(dates))
    return lf.select(columns).collect()


def load_matrix(cfg: Config, date_lo: int, date_hi: int, columns: list[str], dates: list[int] | None = None) -> np.ndarray:
    """Fill a preallocated float32 matrix column by column to cap peak memory."""
    dset = ds.dataset(store_path(cfg), format="parquet")
    filt = (ds.field("date_id") >= date_lo) & (ds.field("date_id") <= date_hi)
    if dates is not None:
        filt = filt & ds.field("date_id").isin(dates)
    n = dset.count_rows(filter=filt)
    X = np.empty((n, len(columns)), dtype=np.float32)
    for j, c in enumerate(columns):
        col = dset.to_table(columns=[c], filter=filt).column(0)
        X[:, j] = col.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return X
