"""Chronological, whole-date splits. Never random rows."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import Config


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: int
    train_end: int      # inclusive
    val_start: int
    val_end: int        # inclusive

    def to_dict(self) -> dict:
        return asdict(self)


def dev_folds(cfg: Config) -> list[Fold]:
    s = cfg["splits"]
    emb = int(s.get("embargo_days", 0))
    folds = [Fold(f["name"], s["train_start"], f["val_start"] - 1 - emb, f["val_start"], f["val_end"]) for f in s["folds"]]
    validate_folds(folds, s["holdout_start"])
    return folds


def final_split(cfg: Config) -> Fold:
    """Final model trains on every pre-holdout date and is scored once on the holdout."""
    s = cfg["splits"]
    emb = int(s.get("embargo_days", 0))
    return Fold("holdout", s["train_start"], s["holdout_start"] - 1 - emb, s["holdout_start"], s["holdout_end"])


def validate_folds(folds: list[Fold], holdout_start: int) -> None:
    prev_val_end = None
    for f in folds:
        if not (f.train_start <= f.train_end < f.val_start <= f.val_end):
            raise ValueError(f"fold {f.name} is not chronological: {f}")
        if f.val_end >= holdout_start:
            raise ValueError(f"fold {f.name} touches the final holdout (starts {holdout_start})")
        if prev_val_end is not None and f.val_start <= prev_val_end:
            raise ValueError(f"validation windows overlap at {f.name}")
        prev_val_end = f.val_end


def split_table(cfg: Config) -> list[dict]:
    rows = [{**f.to_dict(), "n_train_dates": f.train_end - f.train_start + 1, "n_val_dates": f.val_end - f.val_start + 1}
            for f in dev_folds(cfg)]
    h = final_split(cfg)
    rows.append({**h.to_dict(), "n_train_dates": h.train_end - h.train_start + 1, "n_val_dates": h.val_end - h.val_start + 1})
    return rows
