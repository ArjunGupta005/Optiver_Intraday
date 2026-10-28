"""Kaggle entry point (requires the competition's CPython 3.10 / Linux x86-64 environment).

Upload `src/optiver_tc` and `artifacts/full/model/` as a Kaggle dataset, then run this script in a
notebook attached to the competition. Not executable on this Windows machine: the compiled
`optiver2023` module only loads on Linux x86-64 CPython 3.10. Offline equivalent:
`optiver-tc replay --source example` (labelled as a mock replay).
"""
import os
import sys

import numpy as np
import pandas as pd

SRC = os.environ.get("OPTIVER_TC_SRC", "/kaggle/input/optiver-tc/src")
MODEL_DIR = os.environ.get("OPTIVER_TC_MODEL", "/kaggle/input/optiver-tc/artifacts/full/model")
sys.path.insert(0, SRC)

import optiver_tc  # noqa: E402,F401  (imports lightgbm before polars)
from optiver_tc.streaming import StreamingPredictor  # noqa: E402

import optiver2023  # noqa: E402

predictor = StreamingPredictor.from_dir(MODEL_DIR, n_threads=1)
env = optiver2023.make_env()
for test, revealed_targets, sample_prediction in env.iter_test():
    pred = pd.Series(predictor.predict(test), index=test["row_id"].to_numpy())
    sample_prediction["target"] = pred.reindex(sample_prediction["row_id"].to_numpy()).to_numpy()
    assert np.isfinite(sample_prediction["target"]).all()
    env.predict(sample_prediction)
