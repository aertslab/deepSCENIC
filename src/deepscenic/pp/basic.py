"""Basic preprocessing functions for deepSCENIC."""

import numpy as np
from anndata import AnnData


def remove_zero_variance_genes(
    adata: AnnData,
    inplace: bool = True,
) -> AnnData | None:
    """
    Remove genes with zero variance across cells.

    This is useful after filtering to remove genes that have become
    constant (e.g., all zeros) across the remaining cells.

    Parameters
    ----------
    adata : AnnData
        Gene expression data.
    inplace : bool, default=True
        Whether to modify in-place.

    Returns
    -------
    AnnData or None
        If inplace=False, returns filtered AnnData.

    Examples
    --------
    >>> import scanpy as sc
    >>> import deepscenic as ds
    >>> # Standard preprocessing with scanpy
    >>> sc.pp.filter_genes(adata, min_cells=10)
    >>> # Remove any zero-variance genes that remain
    >>> ds.pp.remove_zero_variance_genes(adata)
    """
    if not inplace:
        adata = adata.copy()

    if hasattr(adata.X, "toarray"):
        var = np.array(adata.X.toarray().std(axis=0)).flatten()
    else:
        var = np.array(adata.X.std(axis=0)).flatten()

    keep = var > 0
    if not keep.all():
        n_removed = (~keep).sum()
        adata._inplace_subset_var(keep)
        print(f"Removed {n_removed} zero-variance genes")

    if not inplace:
        return adata
    return None


def filter_regions_by_celltype(
    adata: AnnData,
    celltype_key: str,
    min_fraction: float = 0.1,
    inplace: bool = True,
) -> AnnData | None:
    """
    Filter ATAC regions: keep if present in >min_fraction cells of ANY cell type.

    This mimics the filtering in format_data.ipynb where regions are kept if
    they pass the threshold in at least one cell type.

    Parameters
    ----------
    adata : AnnData
        ATAC accessibility data
    celltype_key : str
        Key in adata.obs for cell type labels
    min_fraction : float, default=0.1
        Minimum fraction of cells per cell type
    inplace : bool, default=True
        Whether to modify in-place

    Returns
    -------
    AnnData or None
        If inplace=False, returns filtered AnnData
    """
    from tqdm import tqdm

    if not inplace:
        adata = adata.copy()

    celltypes = adata.obs[celltype_key].unique()
    keep_regions: set = set()

    for ct in tqdm(celltypes, desc="Filtering regions by cell type"):
        mask = adata.obs[celltype_key] == ct
        ct_adata = adata[mask]
        n_cells_ct = ct_adata.n_obs
        min_cells = int(n_cells_ct * min_fraction)

        # Count cells per region
        if hasattr(ct_adata.X, "toarray"):
            nonzero_counts = np.array((ct_adata.X > 0).sum(axis=0)).flatten()
        else:
            nonzero_counts = np.array((ct_adata.X > 0).sum(axis=0)).flatten()

        passing = ct_adata.var_names[nonzero_counts >= min_cells]
        keep_regions.update(passing)

    keep_mask = adata.var_names.isin(keep_regions)
    adata._inplace_subset_var(keep_mask)

    print(f"Kept {keep_mask.sum()} / {len(keep_mask)} regions")

    if not inplace:
        return adata
    return None
