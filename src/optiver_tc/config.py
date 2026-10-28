"""Configuration loading, project paths, thread settings and environment capture."""
from __future__ import annotations

import json
import os
import platform
import random
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 (e.g. Kaggle)
    import tomli as tomllib

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve(self.raw["artifacts_dir"])

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.raw["reports_dir"])

    @property
    def processed_dir(self) -> Path:
        return self.resolve(self.raw["data"]["processed_dir"])

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.raw["data"]["raw_dir"])

    @property
    def n_threads(self) -> int:
        return int(os.environ.get("OPTIVER_THREADS", self.raw["resources"]["n_threads"]))


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Config(raw=raw, path=path)


def set_threads(n: int) -> None:
    """Cap BLAS/OpenMP/polars threads. Must run before heavy imports to fully apply."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ.setdefault(var, str(n))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def environment_info() -> dict[str, Any]:
    import psutil

    pkgs = {}
    for p in ["numpy", "pandas", "polars", "pyarrow", "lightgbm", "scikit-learn", "scipy", "matplotlib", "psutil", "pytest"]:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    vm = psutil.virtual_memory()
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "ram_available_gb_at_start": round(vm.available / 1e9, 2),
        "packages": pkgs,
    }


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
