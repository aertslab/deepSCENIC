"""Write functions for deepSCENIC data."""

from pathlib import Path

import mudata as md

from ..data.schema import validate_schema


def write(
    mdata: md.MuData,
    path: str | Path,
    validate: bool = True,
    compression: str = "gzip",
) -> None:
    """
    Write MuData to file.

    Parameters
    ----------
    mdata : MuData
        Data to write
    path : str or Path
        Output path (should end in .h5mu)
    validate : bool, default=True
        Validate schema before writing
    compression : str, default="gzip"
        Compression for h5 file

    Examples
    --------
    >>> import deepscenic as ds
    >>> ds.write(mdata, "processed_dataset.h5mu")
    """
    path = Path(path)

    if validate:
        validate_schema(mdata, strict=True)

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    mdata.write(path, compression=compression)
