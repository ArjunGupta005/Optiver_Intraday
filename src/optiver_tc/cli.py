"""Command line entry point: `optiver-tc <command> --config configs/default.toml`."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def _setup(args) -> "object":
    from .config import load_config, seed_everything, set_threads

    cfg = load_config(args.config)
    set_threads(cfg.n_threads)
    seed_everything(cfg["seed"])
    log_dir = cfg.artifacts_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir / f"{args.command}.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers, force=True)
    return cfg


def cmd_extract(cfg, args):
    from .data import build_manifest, build_parquet_cache, safe_extract

    zip_path = args.zip or cfg["data"]["zip_path"]
    recs = safe_extract(zip_path, cfg.raw_dir)
    for r in recs:
        logging.info("%s", r)
    extra = [Path(p) for p in (args.extra or [])]
    build_manifest(cfg, zip_path, extra)
    logging.info("parquet cache: %s", build_parquet_cache(cfg))


def cmd_audit(cfg, args):
    from .audit import run_audit

    a = run_audit(cfg)
    logging.info("audit written: %s rows, %s stocks, %s dates", a["shape"][0], a["n_stocks"], a["n_dates"])


def cmd_features(cfg, args):
    from .featurestore import build_feature_store

    logging.info("feature store: %s", build_feature_store(cfg, force=args.force))


def cmd_dev(cfg, args):
    from .config import environment_info, write_json
    from .featurestore import build_feature_store
    from .research import run_all
    from .splits import split_table

    build_feature_store(cfg)
    write_json({"splits": split_table(cfg), "environment": environment_info(), "config": cfg.raw},
               cfg.artifacts_dir / "run_manifest_dev.json")
    sel = run_all(cfg, args.stages)
    logging.info("dev selection: %s", sel)


def cmd_freeze(cfg, args):
    from .final import freeze

    freeze(cfg)


def cmd_final(cfg, args):
    from .final import run_final

    run_final(cfg)


def cmd_backtest(cfg, args):
    from .backtest import run_backtest_reports

    run_backtest_reports(cfg)


def cmd_replay(cfg, args):
    from .replay_cli import run_replay

    run_replay(cfg, source=args.source)


def cmd_benchmark(cfg, args):
    from .benchmark import run_benchmark

    run_benchmark(cfg)


def cmd_report(cfg, args):
    from .reporting import build_reports

    build_reports(cfg)


def cmd_smoke(cfg, args):
    """Full vertical slice on a small chronological subset (configs/smoke.toml)."""
    from .featurestore import build_feature_store
    from .final import freeze, run_final
    from .research import run_all
    from .backtest import run_backtest_reports
    from .replay_cli import run_replay
    from .benchmark import run_benchmark
    from .reporting import build_reports

    t = time.time()
    build_feature_store(cfg, force=True)
    run_all(cfg, args.stages)
    freeze(cfg)
    run_final(cfg)
    run_backtest_reports(cfg)
    run_replay(cfg, source="holdout")
    run_replay(cfg, source="example")
    run_benchmark(cfg)
    build_reports(cfg)
    logging.info("smoke finished in %.1fs", time.time() - t)


COMMANDS = {
    "extract": cmd_extract, "audit": cmd_audit, "features": cmd_features, "dev": cmd_dev, "freeze": cmd_freeze,
    "final": cmd_final, "backtest": cmd_backtest, "replay": cmd_replay, "benchmark": cmd_benchmark,
    "report": cmd_report, "smoke": cmd_smoke,
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="optiver-tc")
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--config", default="configs/default.toml")
    ap.add_argument("--zip", default=None, help="path to optiver-trading-at-the-close.zip")
    ap.add_argument("--extra", nargs="*", help="extra files to checksum into the manifest")
    ap.add_argument("--stages", default="ABCD", help="dev research stages to run")
    ap.add_argument("--source", default="example", choices=["example", "holdout"], help="replay source")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = _setup(args)
    COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    main()
