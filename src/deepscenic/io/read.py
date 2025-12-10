"""Read functions for deepSCENIC data."""

from pathlib import Path

import mudata as md
from scipy.sparse import csr_matrix, load_npz

from ..data.schema import validate_schema

md.set_options(pull_on_update=False)


def read(
    path: str | Path,
    validate: bool = True,
    backed: str | None = None,
) -> md.MuData:
    """
    Read a deepSCENIC MuData file.

    Parameters
    ----------
    path : str or Path
        Path to .h5mu file
    validate : bool, default=True
        Whether to validate schema after loading
    backed : {'r', 'r+'}, optional
        Load in backed mode for memory efficiency

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

    if path.suffix not in [".h5mu", ".h5ad"]:
        raise ValueError(f"Expected .h5mu file, got: {path.suffix}")

    mdata = md.read(path, backed=backed)

    if validate:
        validate_schema(mdata, strict=False)

    return mdata


def read_legacy(
    rna_path: str | Path,
    atac_path: str | Path,
    r2g_path: str | Path,
    tf_list_path: str | Path | None = None,
) -> md.MuData:
    """
    Read legacy deepSCENIC format (separate files) into MuData.

    This is a convenience function for migrating from the old format.
    For new projects, use `ds.pp.create_mudata()` instead.

    Parameters
    ----------
    rna_path : str or Path
        Path to RNA h5ad file (e.g., raw_exprMat.h5ad)
    atac_path : str or Path
        Path to ATAC h5ad file (e.g., fragment_matrix.h5ad)
    r2g_path : str or Path
        Path to r2g penalty matrix (e.g., r2gpenalty_train.npz)
    tf_list_path : str or Path, optional
        Path to TF list file

    Returns
    -------
    MuData
        Combined MuData (note: may need further preprocessing)

    Notes
    -----
    This loads data but does NOT apply the full schema.
    Use preprocessing functions like `ds.pp.mark_tfs()`, `ds.pp.parse_region_coordinates()`,
    and `ds.pp.compute_r2g_penalty()` to complete preprocessing.
    """
    import scanpy as sc

    # Load AnnData objects
    adata_rna = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)

    # Load r2g matrix
    r2g_sparse = load_npz(r2g_path)

    # Create MuData
    mdata = md.MuData({"rna": adata_rna, "atac": adata_atac})

    # Store r2g in uns (partial - just train for now)
    mdata.uns["r2g"] = {
        "train": csr_matrix(r2g_sparse),
        "config": {"source": "legacy_import"},
    }

    # Load TF list if provided
    if tf_list_path is not None:
        with open(tf_list_path) as f:
            tf_names = [line.strip() for line in f if line.strip()]

        # Mark TFs in var
        mdata.mod["rna"].var["is_tf"] = mdata.mod["rna"].var_names.isin(tf_names)
        mdata.mod["rna"].uns["tf_order"] = [tf for tf in tf_names if tf in mdata.mod["rna"].var_names]

    return mdata
