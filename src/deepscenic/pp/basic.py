"""Basic preprocessing functions for deepSCENIC."""

import numpy as np
import scanpy as sc
from anndata import AnnData


def filter_genes(
    adata: AnnData,
    min_cells: int = 10,
    inplace: bool = True,
) -> AnnData | None:
    """
    Filter genes expressed in too few cells.

    Parameters
    ----------
    adata : AnnData
        RNA expression data
    min_cells : int, default=10
        Minimum number of cells a gene must be expressed in
    inplace : bool, default=True
        Whether to modify in-place

    Returns
    -------
    AnnData or None
        If inplace=False, returns filtered AnnData
    """
    if not inplace:
        adata = adata.copy()

    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Also remove zero-variance genes
    if hasattr(adata.X, "toarray"):
        var = np.array(adata.X.toarray().std(axis=0)).flatten()
    else:
        var = np.array(adata.X.std(axis=0)).flatten()

    keep = var > 0
    if not keep.all():
        adata._inplace_subset_var(keep)

    if not inplace:
        return adata
    return None


def normalize_rna(
    adata: AnnData,
    target_sum: float | None = None,
    log: bool = True,
    inplace: bool = True,
) -> AnnData | None:
    """
    Normalize RNA data (total count + log1p).

    Parameters
    ----------
    adata : AnnData
        RNA expression data
    target_sum : float, optional
        Target sum for normalization. If None, uses median.
    log : bool, default=True
        Whether to apply log1p transformation
    inplace : bool, default=True
        Whether to modify in-place

    Returns
    -------
    AnnData or None
        If inplace=False, returns normalized AnnData
    """
    if not inplace:
        adata = adata.copy()

    sc.pp.normalize_total(adata, target_sum=target_sum)
    if log:
        sc.pp.log1p(adata)

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
