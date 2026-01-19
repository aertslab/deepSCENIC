"""Read functions for deepSCENIC data."""

from pathlib import Path

import mudata as md
import numpy as np
from scipy.sparse import csr_matrix

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

    Simple wrapper around `mudata.read(...)` with schema validation for required DeepSCENIC fields.

    Parameters
    ----------
    path
        Path to .h5mu file
    validate
        Whether to validate schema after loading
    strict
        Whether to log warnings (strict: False) or raise Exception when not following schema.
    backed
        Load in backed mode for memory efficiency
        See anndata/mudata official documentation for more info.

    Returns
    -------
    MuData
        Loaded and optionally validated MuData

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

    if path.suffix not in [".h5mu"]:
        raise ValueError(f"Expected .h5mu file, got: {path.suffix}")

    mdata = md.read(path, backed=backed)
    if isinstance(mdata, md.AnnData):
        mdata = md.MuData(mdata)

    # Restore sparse matrices and other data types
    _restore_after_read(mdata)

    if validate:
        validate_schema(mdata, strict=strict)

    return mdata


def _restore_after_read(mdata: md.MuData) -> None:
    """
    Restore data types after reading from h5mu.

    Reconstructs sparse matrices and nested dicts from serialized components.
    Modifies mdata in-place.
    """

    def _restore_uns_value(value):
        """Recursively restore uns values after reading."""
        if isinstance(value, dict):
            # Check if this is a serialized sparse matrix
            if value.get("_sparse_matrix") is True:
                shape = tuple(value["shape"])
                return csr_matrix(
                    (value["data"], (value["row"], value["col"])),
                    shape=shape,
                )
            # Otherwise recursively restore dict values
            return {k: _restore_uns_value(v) for k, v in value.items()}
        elif isinstance(value, np.ndarray) and value.dtype == object:
            # Convert object arrays back to lists (for region_names, gene_names, etc.)
            return value.tolist()
        elif isinstance(value, str) and value == "None":
            # Restore None values
            return None
        else:
            return value

    # Restore all uns dictionaries (top-level and per-modality)
    mdata.uns = {k: _restore_uns_value(v) for k, v in mdata.uns.items()}
    for mod in mdata.mod.values():
        mod.uns = {k: _restore_uns_value(v) for k, v in mod.uns.items()}

    # Special handling: reconstruct r2g config from flattened keys
    if "r2g" in mdata.uns:
        r2g = mdata.uns["r2g"]
        if "config_max_distance" in r2g:
            r2g["config"] = {
                "max_distance": int(r2g["config_max_distance"]),
                "sigma": int(r2g["config_sigma"]),
                "method": str(r2g["config_method"]),
                "n_links": int(r2g["config_n_links"]),
                "density": float(r2g["config_density"]),
            }
            del r2g["config_max_distance"]
            del r2g["config_sigma"]
            del r2g["config_method"]
            del r2g["config_n_links"]
            del r2g["config_density"]
