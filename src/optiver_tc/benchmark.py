"""End-to-end streaming latency (features + model + post-processing) per batch, on this machine only."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time

import numpy as np
import polars as pl
import psutil

from .config import Config, environment_info, read_json, write_json
from .data import load_train
from .streaming import ModelBundle, StreamingPredictor, iter_timestamps

log = logging.getLogger(__name__)

COLD_SNIPPET = r"""
import time, json, sys
t0 = time.perf_counter()
import optiver_tc
from optiver_tc.streaming import StreamingPredictor
import polars as pl
t1 = time.perf_counter()
p = StreamingPredictor.from_dir(sys.argv[1], n_threads=1)
t2 = time.perf_counter()
df = pl.read_parquet(sys.argv[2])
t3 = time.perf_counter()
p.predict(df)
t4 = time.perf_counter()
print(json.dumps({"import_s": t1 - t0, "load_bundle_s": t2 - t1, "first_batch_s": t4 - t3}))
"""


def _pct(x) -> dict:
    a = np.asarray(x) * 1000
    return {"n": int(len(a)), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max()), "mean_ms": float(a.mean())}


def run_benchmark(cfg: Config, n_days: int = 5) -> dict:
    fz = read_json(cfg.artifacts_dir / "frozen_choices.json")
    hs = fz["holdout"]
    model_dir = cfg.artifacts_dir / "model"
    days = list(range(hs["val_start"], min(hs["val_start"] + n_days, hs["val_end"] + 1)))
    from .data import train_parquet_path
    raw = pl.scan_parquet(train_parquet_path(cfg)).filter(pl.col("date_id").is_in(days)).drop("target").collect()
    import gc
    gc.collect()
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t = time.perf_counter()
    bundle = ModelBundle.load(model_dir, n_threads=1)
    load_s = time.perf_counter() - t
    pred = StreamingPredictor(bundle)
    first_of_day, warm, first_call = [], [], None
    for b in iter_timestamps(raw):
        t = time.perf_counter()
        pred.predict(b)
        dt = time.perf_counter() - t
        if first_call is None:
            first_call = dt
        elif int(b["seconds_in_bucket"][0]) == 0:
            first_of_day.append(dt)
        else:
            warm.append(dt)
    rss1 = proc.memory_info().rss
    # cold start in a fresh interpreter (imports + bundle load + first batch)
    tmp = cfg.artifacts_dir / "replay" / "_bench_batch.parquet"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    next(iter_timestamps(raw)).write_parquet(tmp)
    colds = []
    for _ in range(3):
        r = subprocess.run([sys.executable, "-c", COLD_SNIPPET, str(model_dir), str(tmp)], capture_output=True, text=True,
                           env={**__import__("os").environ, "PYTHONPATH": str(cfg.resolve("src"))})
        if r.returncode == 0:
            colds.append(json.loads(r.stdout.strip().splitlines()[-1]))
        else:
            log.warning("cold subprocess failed: %s", r.stderr[-500:])
    tmp.unlink(missing_ok=True)
    res = {
        "what": "wall-clock per predict() call: input coercion + features (history+batch) + model + post-processing",
        "days": days, "stocks_per_batch_mean": float(raw.group_by(["date_id", "seconds_in_bucket"]).len()["len"].mean()),
        "model_threads": 1, "bundle_load_s": load_s, "first_call_in_process_ms": first_call * 1000,
        "warm": _pct(warm), "first_batch_of_day_after_reset": _pct(first_of_day),
        "cold_fresh_process": colds,
        "rss_mb_before_load": rss0 / 1e6, "rss_mb_after_replay": rss1 / 1e6,
        "state_rows_at_end": pred.state.height if pred.state is not None else 0,
        "model_files_mb": sum(p.stat().st_size for p in model_dir.glob("*")) / 1e6,
        "environment": environment_info(),
        "caveat": "Single machine, single run, other applications running. Not a latency guarantee.",
    }
    write_json(res, cfg.artifacts_dir / "streaming_benchmark.json")
    sfx = "" if cfg["name"] == "full" else "_" + cfg["name"]
    w, f = res["warm"], res["first_batch_of_day_after_reset"]
    cold = (f"{np.mean([c['import_s'] for c in colds]):.2f} s imports + {np.mean([c['load_bundle_s'] for c in colds]):.2f} s bundle load + "
            f"{np.mean([c['first_batch_s'] for c in colds]) * 1000:.0f} ms first batch (mean of {len(colds)} fresh processes)") if colds else "n/a"
    env = res["environment"]
    md = f"""# Streaming benchmark

Measured by `optiver-tc benchmark` on {len(days)} holdout days ({w['n'] + f['n'] + 1} batches, ~{res['stocks_per_batch_mean']:.0f} stocks each).
Each timing is one `StreamingPredictor.predict(batch)` call end to end: input coercion, feature computation on
(bounded history + batch), model inference (LightGBM, 1 thread{', + ridge' if len(bundle.models) > 1 else ''}), post-processing.

| measurement | p50 ms | p95 ms | p99 ms | max ms | n |
|---|---|---|---|---|---|
| warm (batch with intraday history) | {w['p50_ms']:.1f} | {w['p95_ms']:.1f} | {w['p99_ms']:.1f} | {w['max_ms']:.1f} | {w['n']} |
| first batch of a day (state just reset) | {f['p50_ms']:.1f} | {f['p95_ms']:.1f} | {f['p99_ms']:.1f} | {f['max_ms']:.1f} | {f['n']} |

* First call in a warmed process: {res['first_call_in_process_ms']:.1f} ms; bundle load {load_s:.2f} s.
* Cold start: {cold}.
* Process RSS {res['rss_mb_before_load']:.0f} MB before bundle load (imports + {len(days)} days of raw data) -> {res['rss_mb_after_replay']:.0f} MB after replay;
  model files {res['model_files_mb']:.1f} MB; streaming state at the end: {res['state_rows_at_end']} rows.
* Environment: {env['processor']} ({env['physical_cpus']} cores / {env['logical_cpus']} threads), {env['ram_total_gb']} GB RAM,
  {env['platform']}, Python {env['python']}, polars {env['packages']['polars']}, lightgbm {env['packages']['lightgbm']}.

{res['caveat']}
"""
    (cfg.reports_dir / f"streaming_benchmark{sfx}.md").write_text(md, encoding="utf-8")
    log.info("benchmark warm p50 %.1f ms p95 %.1f ms", w["p50_ms"], w["p95_ms"])
    return res
