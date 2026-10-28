"""Data audit: structure, cadence, missingness, invalid values, target distribution, example-file overlap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from .config import Config, write_json
from .data import PRICE_COLS, SIZE_COLS, load_example_test, load_train


def _md_table(df: pl.DataFrame, floatfmt: str = "{:.4g}") -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def inspect_compiled_api(raw_dir: Path) -> dict:
    so = raw_dir / "optiver2023" / "competition.cpython-310-x86_64-linux-gnu.so"
    info = {"path": str(so.relative_to(raw_dir)).replace("\\", "/"), "exists": so.exists()}
    if so.exists():
        head = so.read_bytes()[:20]
        info["elf_magic"] = head[:4] == b"\x7fELF"
        info["elf_class_64bit"] = head[4] == 2
        info["e_machine_x86_64"] = int.from_bytes(head[18:20], "little") == 62
    import platform
    import sys
    info["this_interpreter"] = f"{platform.python_implementation()} {sys.version.split()[0]} on {platform.system()} {platform.machine()}"
    info["loadable_here"] = platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64") and sys.version_info[:2] == (3, 10)
    return info


def run_audit(cfg: Config) -> dict:
    df = load_train(cfg)
    out: dict = {}
    out["shape"] = list(df.shape)
    out["dtypes"] = {c: str(t) for c, t in df.schema.items()}
    out["n_stocks"] = df["stock_id"].n_unique()
    out["stock_id_range"] = [int(df["stock_id"].min()), int(df["stock_id"].max())]
    out["n_dates"] = df["date_id"].n_unique()
    out["date_id_range"] = [int(df["date_id"].min()), int(df["date_id"].max())]
    dates = np.sort(df["date_id"].unique().to_numpy())
    out["date_gaps"] = [int(d) for d in np.setdiff1d(np.arange(dates.min(), dates.max() + 1), dates)]
    secs = np.sort(df["seconds_in_bucket"].unique().to_numpy())
    out["seconds_in_bucket_values"] = {"min": int(secs.min()), "max": int(secs.max()), "n": len(secs),
                                       "unique_steps": sorted(set(np.diff(secs).tolist()))}
    # time_id consistency: one time_id per (date, seconds)
    tid = df.group_by(["date_id", "seconds_in_bucket"]).agg(pl.col("time_id").n_unique().alias("k"))
    out["time_id_unique_per_timestamp"] = bool((tid["k"] == 1).all())
    tmap = df.select(["date_id", "seconds_in_bucket", "time_id"]).unique().sort(["date_id", "seconds_in_bucket"])
    out["time_id_monotone_in_date_seconds"] = bool((tmap["time_id"].diff().drop_nulls() > 0).all())
    # duplicates
    dup = df.group_by(["stock_id", "date_id", "seconds_in_bucket"]).len().filter(pl.col("len") > 1)
    out["duplicate_keys"] = dup.height
    out["duplicate_row_id"] = df.height - df["row_id"].n_unique()
    rid = df.select((pl.col("date_id").cast(pl.Utf8) + "_" + pl.col("seconds_in_bucket").cast(pl.Utf8) + "_" + pl.col("stock_id").cast(pl.Utf8)).alias("r"))
    out["row_id_matches_date_sec_stock"] = bool((rid["r"] == df["row_id"]).all())

    # coverage: stocks per date, buckets per stock-day
    spd = df.group_by("date_id").agg(pl.col("stock_id").n_unique().alias("n_stocks")).sort("date_id")
    out["stocks_per_date"] = {"min": int(spd["n_stocks"].min()), "max": int(spd["n_stocks"].max()),
                              "median": float(spd["n_stocks"].median()),
                              "dates_with_fewer_than_max": int((spd["n_stocks"] < spd["n_stocks"].max()).sum())}
    bps = df.group_by(["stock_id", "date_id"]).len()
    full = len(secs)
    out["buckets_per_stock_day"] = {"expected": full, "n_stock_days": bps.height,
                                    "incomplete_stock_days": int((bps["len"] < full).sum()),
                                    "min": int(bps["len"].min())}
    first_date = df.group_by("stock_id").agg(pl.col("date_id").min().alias("first"), pl.col("date_id").n_unique().alias("ndays"))
    out["stocks_missing_some_dates"] = int((first_date["ndays"] < out["n_dates"]).sum())
    out["stocks_entering_after_date0"] = first_date.filter(pl.col("first") > 0).sort("first").to_dicts()

    # missingness overall and by time bucket
    out["null_counts"] = {c: int(v) for c, v in zip(df.columns, df.null_count().row(0))}
    miss_bucket = (df.group_by("seconds_in_bucket").agg(
        [pl.col(c).is_null().mean().alias(c) for c in ["far_price", "near_price", "reference_price", "wap", "bid_price", "imbalance_size", "target"]]
    ).sort("seconds_in_bucket"))
    out["missing_by_bucket"] = miss_bucket.to_dicts()

    # invalid values
    inv = {}
    for c in SIZE_COLS:
        inv[f"{c}_negative"] = int((df[c] < 0).sum())
        inv[f"{c}_zero"] = int((df[c] == 0).sum())
    for c in PRICE_COLS:
        inv[f"{c}_nonpositive"] = int((df[c] <= 0).sum())
    inv["crossed_book_bid_gt_ask"] = int((df["bid_price"] > df["ask_price"]).sum())
    inv["locked_book_bid_eq_ask"] = int((df["bid_price"] == df["ask_price"]).sum())
    inv["imbalance_flag_values"] = sorted(df["imbalance_buy_sell_flag"].drop_nulls().unique().to_list())
    inv["flag_zero_but_imbalance_positive"] = int(((df["imbalance_buy_sell_flag"] == 0) & (df["imbalance_size"] > 0)).sum())
    inv["wap_at_second0_not_1"] = int(df.filter(pl.col("seconds_in_bucket") == 0).select((pl.col("wap") - 1.0).abs() > 1e-9).sum().item())
    inv["far_price_gt_2_or_lt_0.5"] = int(((df["far_price"] > 2) | (df["far_price"] < 0.5)).sum())
    out["invalid_values"] = inv
    out["far_price_quantiles"] = {str(q): float(df["far_price"].quantile(q)) for q in [0.0001, 0.01, 0.5, 0.99, 0.9999]}

    # rows with missing book entirely
    rows_no_wap = df.filter(pl.col("wap").is_null())
    out["rows_missing_wap"] = rows_no_wap.height
    out["rows_missing_wap_stock_dates"] = rows_no_wap.select(["stock_id", "date_id"]).unique().sort(["date_id", "stock_id"]).to_dicts()[:50]

    # target distribution
    t = df["target"].drop_nulls()
    out["target"] = {
        "n": t.len(), "n_missing": int(df["target"].is_null().sum()),
        "mean": float(t.mean()), "std": float(t.std()), "median": float(t.median()),
        "mae_vs_zero": float(t.abs().mean()),
        "quantiles": {str(q): float(t.quantile(q)) for q in [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]},
        "min": float(t.min()), "max": float(t.max()),
        "skew": float(t.skew()), "kurtosis": float(t.kurtosis()),
    }
    by_bucket = df.group_by("seconds_in_bucket").agg(
        pl.col("target").mean().alias("mean"), pl.col("target").std().alias("std"),
        pl.col("target").abs().mean().alias("mean_abs")).sort("seconds_in_bucket")
    out["target_by_bucket"] = by_bucket.to_dicts()
    by_date = df.group_by("date_id").agg(pl.col("target").abs().mean().alias("mean_abs"), pl.col("target").mean().alias("mean")).sort("date_id")
    out["target_by_date_summary"] = {
        "mean_abs_min": float(by_date["mean_abs"].min()), "mean_abs_max": float(by_date["mean_abs"].max()),
        "mean_abs_median": float(by_date["mean_abs"].median()),
    }
    by_stock = df.group_by("stock_id").agg(pl.col("target").abs().mean().alias("mean_abs"))
    out["target_by_stock_summary"] = {
        "mean_abs_min": float(by_stock["mean_abs"].min()), "mean_abs_max": float(by_stock["mean_abs"].max()),
        "mean_abs_median": float(by_stock["mean_abs"].median()),
    }
    # does target sum to ~0 cross-sectionally (index-relative)? simple mean per timestamp
    xs = df.group_by(["date_id", "seconds_in_bucket"]).agg(pl.col("target").mean().alias("m"))
    out["cross_sectional_mean_target"] = {"mean_abs_of_timestamp_means": float(xs["m"].abs().mean()),
                                          "std_of_timestamp_means": float(xs["m"].std())}

    # example files overlap
    ex = load_example_test(cfg)
    test, rev, sub = ex["test"], ex["revealed_targets"], ex["sample_submission"]
    ex_info = {
        "test_shape": list(test.shape), "test_columns": test.columns,
        "test_dates": sorted(test["date_id"].unique().to_list()),
        "revealed_shape": list(rev.shape), "revealed_columns": rev.columns,
        "revealed_dates": sorted(rev["revealed_date_id"].drop_nulls().unique().to_list()),
        "revealed_null_rows": int(rev["revealed_date_id"].is_null().sum()),
        "submission_shape": list(sub.shape), "submission_columns": sub.columns,
        "n_test_batches_time_id": test["time_id"].n_unique(),
        "currently_scored_values": test["currently_scored"].value_counts().to_dicts() if "currently_scored" in test.columns else None,
    }
    joined = test.join(df.select(["row_id", *PRICE_COLS, *SIZE_COLS]), on="row_id", how="left", suffix="_train")
    ex_info["test_rows_found_in_train"] = int(joined["wap_train"].is_not_null().sum())
    same = True
    for c in PRICE_COLS + SIZE_COLS:
        a, b = joined[c].to_numpy(), joined[f"{c}_train"].to_numpy()
        m = ~(np.isnan(a) & np.isnan(b))
        same &= bool(np.allclose(a[m], b[m], equal_nan=False, atol=1e-9)) if m.any() else True
    ex_info["test_feature_values_identical_to_train"] = same
    rj = rev.join(df.select(["stock_id", "date_id", "seconds_in_bucket", "target"]),
                  left_on=["stock_id", "revealed_date_id", "seconds_in_bucket"],
                  right_on=["stock_id", "date_id", "seconds_in_bucket"], how="left")
    m = rj["target"].is_not_null() & rj["revealed_target"].is_not_null()
    ex_info["revealed_targets_match_train_targets_max_abs_diff"] = float((rj.filter(m)["revealed_target"] - rj.filter(m)["target"]).abs().max())
    ex_info["sub_row_ids_equal_test_row_ids_in_order"] = bool((sub["row_id"] == test["row_id"]).all())
    out["example_files"] = ex_info
    out["compiled_api"] = inspect_compiled_api(cfg.raw_dir)

    write_json(out, cfg.reports_dir / "data_audit.json")
    _write_md(cfg, out, miss_bucket, by_bucket)
    return out


def _write_md(cfg: Config, a: dict, miss_bucket: pl.DataFrame, by_bucket: pl.DataFrame) -> None:
    ex = a["example_files"]
    t = a["target"]
    lines = [
        "# Data audit",
        "",
        "Generated by `optiver-tc audit` from `data/processed/train.parquet` (typed copy of `data/raw/train.csv`).",
        "Full machine-readable output: `reports/data_audit.json`. Checksums: `data/processed/data_manifest.json`.",
        "",
        "## Structure",
        f"- Shape: **{a['shape'][0]:,} rows x {a['shape'][1]} columns**.",
        f"- Stocks: {a['n_stocks']} (ids {a['stock_id_range'][0]}..{a['stock_id_range'][1]}). Dates: {a['n_dates']} (date_id {a['date_id_range'][0]}..{a['date_id_range'][1]}, gaps: {a['date_gaps'] or 'none'}).",
        f"- seconds_in_bucket: {a['seconds_in_bucket_values']['min']}..{a['seconds_in_bucket_values']['max']}, {a['seconds_in_bucket_values']['n']} values, step(s) {a['seconds_in_bucket_values']['unique_steps']} s.",
        f"- One time_id per (date_id, seconds_in_bucket): {a['time_id_unique_per_timestamp']}; time_id strictly increasing in (date, seconds): {a['time_id_monotone_in_date_seconds']}.",
        f"- Duplicate (stock, date, seconds) keys: {a['duplicate_keys']}; duplicate row_id: {a['duplicate_row_id']}; row_id == `date_seconds_stock`: {a['row_id_matches_date_sec_stock']}.",
        f"- Stocks per date: min {a['stocks_per_date']['min']}, median {a['stocks_per_date']['median']}, max {a['stocks_per_date']['max']}; dates with fewer than max: {a['stocks_per_date']['dates_with_fewer_than_max']}.",
        f"- Stock-days: {a['buckets_per_stock_day']['n_stock_days']:,}; incomplete (fewer than {a['buckets_per_stock_day']['expected']} buckets): {a['buckets_per_stock_day']['incomplete_stock_days']} (min {a['buckets_per_stock_day']['min']}).",
        f"- Stocks absent on at least one date: {a['stocks_missing_some_dates']}. Stocks first appearing after date 0: {len(a['stocks_entering_after_date0'])}.",
        "",
        "Interpretation: the data is a panel of 10-second snapshots during the 10-minute closing auction window; a",
        "timestamp's cross-section is exactly one API batch. Missing stocks on a date are real absences, not padding.",
        "",
        "## Missing values",
        "Null counts: " + ", ".join(f"`{k}`={v:,}" for k, v in a["null_counts"].items() if v),
        "",
        f"Rows with missing WAP (whole book missing): {a['rows_missing_wap']:,}.",
        "",
        "Missing fraction by time bucket (selected columns):",
        "",
        _md_table(miss_bucket.select(["seconds_in_bucket", "far_price", "near_price", "reference_price", "wap", "target"])),
        "",
        "far_price and near_price are structurally missing during the first 300 s (they are only published once the",
        "auction's crossing information is disseminated), so missingness indicators are informative and are kept as features.",
        "",
        "## Invalid / unusual values",
        "```json",
        json.dumps(a["invalid_values"], indent=1),
        "```",
        f"far_price quantiles: {a['far_price_quantiles']} (heavy tails: far_price can be far from 1 when little volume is matched).",
        "",
        "## Target",
        f"- Non-missing: {t['n']:,}; missing: {t['n_missing']} (excluded from fitting and scoring, never imputed).",
        f"- Mean {t['mean']:.4f}, median {t['median']:.4f}, std {t['std']:.3f} bps; MAE of the zero predictor {t['mae_vs_zero']:.4f}.",
        f"- Quantiles: " + ", ".join(f"{k}: {v:.2f}" for k, v in t["quantiles"].items()),
        f"- min {t['min']:.1f}, max {t['max']:.1f}, skew {t['skew']:.3f}, excess kurtosis {t['kurtosis']:.1f} (fat tails motivate an L1 objective).",
        f"- Mean |target| by date ranges {a['target_by_date_summary']['mean_abs_min']:.2f}..{a['target_by_date_summary']['mean_abs_max']:.2f}; by stock {a['target_by_stock_summary']['mean_abs_min']:.2f}..{a['target_by_stock_summary']['mean_abs_max']:.2f}.",
        f"- Cross-sectional (equal-weight) timestamp mean of target: mean |.| = {a['cross_sectional_mean_target']['mean_abs_of_timestamp_means']:.3f} bps. It is not exactly zero because the synthetic index uses unknown (non-equal) weights.",
        "",
        "Target by auction stage:",
        "",
        _md_table(by_bucket),
        "",
        "## Example test files (integration fixtures)",
        f"- test.csv: {ex['test_shape']}, dates {ex['test_dates']}, {ex['n_test_batches_time_id']} batches (time_id). Columns: {ex['test_columns']}.",
        f"- revealed_targets.csv: {ex['revealed_shape']}, revealed dates {ex['revealed_dates']}. Columns: {ex['revealed_columns']}.",
        f"- sample_submission.csv: {ex['submission_shape']}, columns {ex['submission_columns']}; row_id order equals test order: {ex['sub_row_ids_equal_test_row_ids_in_order']}.",
        f"- test rows found in train: {ex['test_rows_found_in_train']:,} of {ex['test_shape'][0]:,}; feature values identical: {ex['test_feature_values_identical_to_train']}.",
        f"- revealed targets vs train targets max |diff|: {ex['revealed_targets_match_train_targets_max_abs_diff']:.2e}.",
        "",
        "Conclusion: the example test dates are the final training dates. They are used **only** as integration",
        "fixtures for the streaming/replay path, never as validation (they fall inside the final holdout window anyway).",
        "",
        "## Compiled competition API",
        "```json",
        json.dumps(a["compiled_api"], indent=1),
        "```",
        "The `.so` is a Linux x86-64 CPython 3.10 extension and cannot be imported on this machine; offline runs use",
        "`optiver_tc.mock_api` (a repaired, configured copy of the vendor MockApi) and are labelled as mock replays.",
    ]
    p = cfg.reports_dir / "data_audit.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
