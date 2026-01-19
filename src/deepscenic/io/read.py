"""Read functions for deepSCENIC data."""

from pathlib import Path

import mudata as md

from ..data.schema import validate_schema

md.set_options(pull_on_update=False)  # adopt new mudata behaviour


def read(
    path: str | Path,
    validate: bool = True,
    strict: bool = False,
    backed: str | bool | None = None,
) -> md.MuData:
    """
    Read a deepSCENIC MuData file.

    Wrapper around mudata.read() with schema validation for required deepSCENIC fields.

    Parameters
    ----------
    path
        Path to .h5mu file.
    validate
        Whether to validate schema after loading.
    strict
        Whether to log warnings (strict: False) or raise Exception when not following schema.
    backed
        Load in backed mode for memory efficiency.
        See anndata/mudata official documentation for more info.

    Returns
    -------
    Loaded and optionally validated MuData.

    Examples
    --------
    >>> import deepscenic as ds
    >>> mdata = ds.read("my_dataset.h5mu")
    >>> mdata
    MuData object with n_obs × n_vars = 39470 × 455165
      rna: 39470 × 16437
      atac: 39470 × 438728
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix != ".h5mu":
        raise ValueError(f"Expected .h5mu file, got: {path.suffix}")

    mdata = md.read(path, backed=backed)

    if validate:
        validate_schema(mdata, strict=strict)

    return mdata
