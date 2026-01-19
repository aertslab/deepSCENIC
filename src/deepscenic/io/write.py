"""Write functions for deepSCENIC data."""

from pathlib import Path

import mudata as md
import numpy as np
from scipy.sparse import issparse

from ..data.schema import validate_schema


def _prepare_for_write(mdata: md.MuData) -> md.MuData:
    """
    Prepare MuData for h5mu serialization.

    Converts data types that h5py cannot handle:
    - Sparse matrices in uns -> dict with indices, data, shape
    - Python lists -> numpy arrays
    - pd.NA values -> np.nan (for numeric) or empty string (for object)

    Returns a copy with serializable data types.
    """
    import pandas as pd

    mdata = mdata.copy()

    def _fix_uns_value(value):
        """Recursively fix uns values for h5py serialization."""
        if value is None:
            return "None"  # h5py can't store None directly
        elif isinstance(value, dict):
            return {k: _fix_uns_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Convert lists to numpy arrays
            if len(value) > 0 and isinstance(value[0], str):
                return np.array(value, dtype=object)
            elif len(value) > 0 and isinstance(value[0], (int, np.integer)):
                return np.array(value, dtype=np.int64)
            elif len(value) > 0 and isinstance(value[0], (float, np.floating)):
                return np.array(value, dtype=np.float64)
            else:
                return np.array(value, dtype=object)
        elif issparse(value):
            # Sparse matrices need special handling - convert to dict of arrays
            coo = value.tocoo()
            return {
                "_sparse_matrix": True,
                "data": coo.data.astype(np.float32),
                "row": coo.row.astype(np.int64),
                "col": coo.col.astype(np.int64),
                "shape": np.array(coo.shape, dtype=np.int64),
            }
        else:
            return value

    # Fix all uns dictionaries (top-level and per-modality)
    mdata.uns = {k: _fix_uns_value(v) for k, v in mdata.uns.items()}
    for mod in mdata.mod.values():
        mod.uns = {k: _fix_uns_value(v) for k, v in mod.uns.items()}

    # Special handling for r2g to flatten nested config
    if "r2g" in mdata.uns:
        r2g = mdata.uns["r2g"]

        # Flatten config dict (h5py doesn't handle nested dicts well)
        if "config" in r2g and isinstance(r2g["config"], dict):
            config = r2g["config"]
            r2g["config_max_distance"] = np.int64(config.get("max_distance", 0))
            r2g["config_sigma"] = np.int64(config.get("sigma", 0))
            r2g["config_method"] = str(config.get("method", "gaussian"))
            r2g["config_n_links"] = np.int64(config.get("n_links", 0))
            r2g["config_density"] = np.float64(config.get("density", 0.0))
            del r2g["config"]

    # Handle pd.NA and other problematic values in var/obs DataFrames
    # h5py cannot serialize pd.NA, so we convert to numpy-compatible types
    def _fix_column(series: pd.Series) -> pd.Series:
        """Convert pd.NA to h5py-compatible types."""
        if not bool(series.isna().any()):
            return series

        # Check if column contains any strings (even mixed with NA)
        non_null = series.dropna()
        if len(non_null) == 0:
            # All NA - convert to float (np.nan)
            return pd.Series([np.nan] * len(series), index=series.index)

        first_val = non_null.iloc[0]

        # Check for string values
        if isinstance(first_val, str):
            return series.fillna("").astype(object)

        # Check for numeric values (int, float, numpy numeric types)
        # Try to convert to float - this handles pd.NA + int/float mixtures
        try:
            return series.astype(float)
        except (ValueError, TypeError):
            pass

        # Check for boolean
        if isinstance(first_val, (bool, np.bool_)):
            return series.fillna(False).astype(bool)

        # Fallback: convert to string, replacing NA markers
        result = series.astype(str)
        result = result.replace({"nan": "", "<NA>": "", "None": ""})
        return result

    for mod in mdata.mod.values():
        # Fix var columns
        for col in mod.var.columns:
            mod.var[col] = _fix_column(mod.var[col])

        # Fix obs columns
        for col in mod.obs.columns:
            mod.obs[col] = _fix_column(mod.obs[col])

    # Fix top-level obs columns
    for col in mdata.obs.columns:
        mdata.obs[col] = _fix_column(mdata.obs[col])

    return mdata


def write(
    mdata: md.MuData,
    path: str | Path,
    validate: bool = True,
    strict: bool = True,
    compression: str = "gzip",
) -> None:
    """
    Write MuData to file.

    Parameters
    ----------
    mdata
        Data to write
    path
        Output path (should end in .h5mu)
    validate
        Validate schema before writing
    strict
        Use strict validation (raise Exceptions instead of logging warnings)
    compression
        Compression for h5 file

    Examples
    --------
    >>> import deepscenic as ds
    >>> ds.write(mdata, "processed_dataset.h5mu")
    """
    path = Path(path)

    if validate:
        validate_schema(mdata, strict=strict)

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for serialization (converts sparse matrices, handles pd.NA)
    mdata_prepared = _prepare_for_write(mdata)

    mdata_prepared.write(str(path), compression=compression)
