"""Write functions for deepSCENIC data."""

from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from ..data.schema import validate_schema

# Enable nullable string serialization (supported in anndata >= 0.11)
ad.settings.allow_write_nullable_strings = True


def _convert_nullable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object-dtype columns with NA to proper nullable pandas dtypes.

    h5py cannot serialize object-dtype columns containing pd.NA. This function
    converts them to proper nullable dtypes (string, Int64, Float64) which
    AnnData can serialize correctly.

    Parameters
    ----------
    df
        DataFrame to convert.

    Returns
    -------
    DataFrame with converted columns.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue
        if not df[col].isna().any():  # type: ignore
            continue

        non_null = df[col].dropna()
        if len(non_null) == 0:
            # All NA - default to string dtype
            df[col] = df[col].astype("string")
            continue

        first_val = non_null.iloc[0]
        if isinstance(first_val, str):
            df[col] = df[col].astype("string")
        elif isinstance(first_val, (int, np.integer)):
            df[col] = df[col].astype("Int64")
        elif isinstance(first_val, (float, np.floating)):
            df[col] = df[col].astype("Float64")
        elif isinstance(first_val, (bool, np.bool_)):
            df[col] = df[col].astype("boolean")

    return df


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
        Data to write.
    path
        Output path (should end in .h5mu).
    validate
        Validate schema before writing.
    strict
        Use strict validation (raise Exceptions instead of logging warnings).
    compression
        Compression for h5 file.

    Examples
    --------
    >>> import deepscenic as ds
    >>> ds.write(mdata, "processed_dataset.h5mu")
    """
    path = Path(path)

    if validate:
        validate_schema(mdata, strict=strict)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Create a copy to avoid modifying the original
    mdata = mdata.copy()

    # Convert object-dtype columns with pd.NA to proper nullable dtypes
    mdata.obs = _convert_nullable_columns(mdata.obs)
    for mod in mdata.mod.values():
        mod.var = _convert_nullable_columns(mod.var)
        mod.obs = _convert_nullable_columns(mod.obs)

    mdata.write(str(path), compression=compression)
