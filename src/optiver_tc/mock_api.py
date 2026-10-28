"""Repaired, configurable copy of the vendor `public_timeseries_testing_util.MockApi`.

The vendor file (kept unmodified in data/raw/public_timeseries_testing_util.py) intentionally leaves
`input_paths`, `group_id_column` and `export_group_id_column` as invalid empty assignments. This copy
takes them as constructor arguments and otherwise mirrors the vendor serving logic: all rows sharing
the current group id are served together, `predict` must be called before the next group, and the
collected predictions are written to a CSV at the end.

Running this is an OFFLINE MOCK REPLAY, not the official Kaggle API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd


class MockApi:
    def __init__(self, input_paths: Sequence[str | Path], group_id_column: str = "time_id",
                 export_group_id_column: bool = True, output_path: str | Path = "submission.csv"):
        self.input_paths = [str(p) for p in input_paths]
        self.group_id_column = group_id_column
        self.export_group_id_column = export_group_id_column
        self.output_path = Path(output_path)
        # iter_test is only designed to support at least two dataframes, such as test and sample_submission
        assert len(self.input_paths) >= 2
        self._status = "initialized"
        self.predictions: list[pd.DataFrame] = []

    def iter_test(self) -> Iterator[tuple[pd.DataFrame, ...]]:
        if self._status != "initialized":
            raise Exception("WARNING: the real API can only iterate over `iter_test()` once.")
        dataframes = [pd.read_csv(p, low_memory=False) for p in self.input_paths]
        group_order = dataframes[0][self.group_id_column].drop_duplicates().tolist()
        dataframes = [df.set_index(self.group_id_column) for df in dataframes]
        for group_id in group_order:
            self._status = "prediction_needed"
            current = []
            for df in dataframes:
                # revealed_targets is keyed by the time_id of the batch it is released with; groups
                # absent from a frame are served as empty frames (vendor code would raise KeyError).
                if group_id in df.index:
                    cur = df.loc[[group_id]].copy()
                else:
                    cur = df.iloc[0:0].copy()
                cur = cur.reset_index(drop=not self.export_group_id_column)
                current.append(cur)
            yield tuple(current)
            if self._status != "prediction_received":
                raise RuntimeError("You must call `predict()` successfully before you can continue with `iter_test()`")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(self.predictions).to_csv(self.output_path, index=False)
        self._status = "finished"

    def predict(self, user_predictions: pd.DataFrame) -> None:
        if self._status == "finished":
            raise Exception("You have already made predictions for the full test set.")
        if self._status != "prediction_needed":
            raise Exception("You must get the next test sample from `iter_test()` first.")
        if not isinstance(user_predictions, pd.DataFrame):
            raise Exception("You must provide a DataFrame.")
        self.predictions.append(user_predictions)
        self._status = "prediction_received"


def make_env(raw_dir: str | Path, output_path: str | Path) -> MockApi:
    d = Path(raw_dir) / "example_test_files"
    return MockApi([d / "test.csv", d / "revealed_targets.csv", d / "sample_submission.csv"],
                   group_id_column="time_id", export_group_id_column=True, output_path=output_path)
