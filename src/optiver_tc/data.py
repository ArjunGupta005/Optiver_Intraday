"""Safe archive extraction, checksums/manifest, and typed Parquet cache.

Raw files are written once under data/raw and never modified afterwards.
"""
from __future__ import annotations

import hashlib
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

import polars as pl

from .config import Config, read_json, write_json

RAW_COLUMNS = [
    "stock_id", "date_id", "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag",
    "reference_price", "matched_size", "far_price", "near_price", "bid_price", "bid_size",
    "ask_price", "ask_size", "wap", "target", "time_id", "row_id",
]
PRICE_COLS = ["reference_price", "far_price", "near_price", "bid_price", "ask_price", "wap"]
SIZE_COLS = ["imbalance_size", "matched_size", "bid_size", "ask_size"]

SCHEMA = {
    "stock_id": pl.Int16, "date_id": pl.Int16, "seconds_in_bucket": pl.Int16,
    "imbalance_size": pl.Float64, "imbalance_buy_sell_flag": pl.Int8,
    "reference_price": pl.Float64, "matched_size": pl.Float64, "far_price": pl.Float64,
    "near_price": pl.Float64, "bid_price": pl.Float64, "bid_size": pl.Float64,
    "ask_price": pl.Float64, "ask_size": pl.Float64, "wap": pl.Float64,
    "target": pl.Float64, "time_id": pl.Int32, "row_id": pl.Utf8,
}


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _safe_member_path(dest: Path, name: str) -> Path:
    """Reject absolute paths, drive letters, '..' components and symlinks escaping dest."""
    if not name or name.startswith(("/", "\\")) or ":" in name:
        raise ValueError(f"unsafe archive member path: {name!r}")
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if any(p in ("..", "") for p in parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    target = (dest / Path(*parts)).resolve()
    if dest.resolve() not in target.parents and target != dest.resolve():
        raise ValueError(f"archive member escapes destination: {name!r}")
    return target


def safe_extract(zip_path: str | Path, dest: str | Path) -> list[dict]:
    """Extract a ZIP after validating every member path. Existing files are verified, not overwritten."""
    zip_path, dest = Path(zip_path), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        # validate everything before writing anything
        targets = [(m, _safe_member_path(dest, m.filename)) for m in members]
        for m, target in targets:
            mode = (m.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError(f"symlink member refused: {m.filename}")
            if m.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == m.file_size:
                action = "verified-existing"
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 22)
                action = "extracted"
            # raw inputs are immutable: mark read-only
            target.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            records.append({
                "path": str(target.relative_to(dest)).replace("\\", "/"),
                "bytes": m.file_size,
                "crc32": f"{m.CRC:08x}",
                "action": action,
            })
    return records


def build_manifest(cfg: Config, zip_path: str | Path, extra_files: list[Path] | None = None) -> dict:
    raw_dir = cfg.raw_dir
    files = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file():
            files.append({
                "path": str(p.relative_to(raw_dir)).replace("\\", "/"),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    manifest = {
        "source_zip": str(zip_path),
        "source_zip_sha256": sha256_file(zip_path),
        "source_zip_bytes": Path(zip_path).stat().st_size,
        "raw_files": files,
        "external_files": [
            {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in (extra_files or []) if p.exists()
        ],
    }
    write_json(manifest, cfg.processed_dir / "data_manifest.json")
    return manifest


def train_parquet_path(cfg: Config) -> Path:
    return cfg.processed_dir / "train.parquet"


def build_parquet_cache(cfg: Config) -> Path:
    """Typed Parquet copy of train.csv (sorted by date, seconds, stock). Raw CSV untouched."""
    out = train_parquet_path(cfg)
    src = cfg.raw_dir / "train.csv"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pl.read_csv(src, schema_overrides=SCHEMA, null_values=["", "NA", "nan"])
    assert df.columns == RAW_COLUMNS, df.columns
    # CSV may contain literal "nan" tokens; represent every missing float as null.
    df = df.with_columns([pl.col(c).fill_nan(None) for c, t in df.schema.items() if t == pl.Float64])
    df = df.sort(["date_id", "seconds_in_bucket", "stock_id"])
    df.write_parquet(out, compression="zstd", statistics=True, row_group_size=200_000)
    write_json({"source": str(src), "source_sha256": sha256_file(src), "rows": df.height}, out.with_suffix(".meta.json"))
    return out


def load_train(cfg: Config, columns: list[str] | None = None) -> pl.DataFrame:
    """Load the typed train table, honouring the config's date/stock restrictions."""
    lf = pl.scan_parquet(train_parquet_path(cfg))
    max_date = cfg["data"].get("max_date_id", -1)
    if max_date is not None and max_date >= 0:
        lf = lf.filter(pl.col("date_id") <= max_date)
    stocks = cfg["data"].get("stock_subset") or []
    if stocks:
        lf = lf.filter(pl.col("stock_id").is_in(stocks))
    if columns:
        lf = lf.select(columns)
    return lf.collect()


def load_example_test(cfg: Config) -> dict[str, pl.DataFrame]:
    d = cfg.raw_dir / "example_test_files"
    return {
        "test": pl.read_csv(d / "test.csv", schema_overrides={k: v for k, v in SCHEMA.items() if k != "target"}),
        "revealed_targets": pl.read_csv(d / "revealed_targets.csv"),
        "sample_submission": pl.read_csv(d / "sample_submission.csv"),
    }


def manifest_digest(cfg: Config) -> dict:
    """Short pointer to the data manifest, embedded in every experiment record."""
    p = cfg.processed_dir / "data_manifest.json"
    if not p.exists():
        return {}
    m = read_json(p)
    train = next((f for f in m["raw_files"] if f["path"] == "train.csv"), None)
    return {"source_zip_sha256": m["source_zip_sha256"], "train_csv_sha256": train["sha256"] if train else None}
