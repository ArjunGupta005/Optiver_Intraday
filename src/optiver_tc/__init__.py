"""Optiver Trading at the Close research package."""
# Import order matters on Windows: loading LightGBM's OpenMP runtime AFTER polars/pyarrow caused an
# access violation inside LGBM_DatasetSetField (reproduced on this machine). Importing it first avoids that.
import lightgbm  # noqa: F401

__version__ = "0.1.0"
