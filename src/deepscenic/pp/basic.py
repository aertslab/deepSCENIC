"""Basic preprocessing functions for deepSCENIC. Functions in basic.py operate on anndatas."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse


def _to_dense(X: Any) -> np.ndarray:
    """
    Convert any array-like to dense numpy array.

    Handles sparse matrices, anndata array views (CSRDataset, CSCDataset),
    H5Array, and regular numpy arrays.
    """
    if X is None:
        raise ValueError("Cannot convert None to dense array")

    if sparse.issparse(X):
        return X.toarray()  # type: ignore[union-attr]

    if hasattr(X, "toarray"):
        # Handles CSRDataset, CSCDataset, and other anndata views
        return X.toarray()  # type: ignore[union-attr]

    # Already dense
    return np.asarray(X)


def remove_zero_variance_genes(
    adata: AnnData,
    inplace: bool = True,
) -> AnnData | None:
    """
    Remove genes (features) with zero variance across cells (samples).

    Expects genes in adata.X columns and cells in rows.
    This is useful after filtering to remove genes that have become
    constant (e.g., all zeros) across the remaining cells.

    Parameters
    ----------
    adata
        Gene expression data.
    inplace
        Whether to modify in-place.

    Returns
    -------
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

    # Convert to dense and compute variance
    X_dense = _to_dense(adata.X)
    var = X_dense.std(axis=0).flatten()

    keep = var > 0
    if not keep.all():
        n_removed = (~keep).sum()
        adata._inplace_subset_var(keep)
        print(f"Removed {n_removed} zero-variance genes")

    if not inplace:
        return adata
    return None


def mark_dars(
    adata: AnnData,
    *,
    dar_dir: str | Path | None = None,
    dar_dict: dict[str, list[str]] | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """
    Mark Differentially Accessible Regions in ATAC data.

    DARs can be loaded from a directory of BED files (pyCisTopic output)
    or provided directly as a dictionary. DARs are typically more
    informative for training and may be upweighted during sampling.

    Parameters
    ----------
    adata
        ATAC accessibility data (single modality).
    dar_dir
        Directory containing DAR BED files (one per cell type).
        Files should be named: {celltype}.bed
        BED format: chrom, start, end (tab-separated, no header).
    dar_dict
        Dictionary mapping cell type names to lists of region names.
        Region names should match adata.var_names format.
    inplace
        Whether to modify adata in-place.

    Returns
    -------
    If inplace=False, returns modified AnnData.

    Notes
    -----
    DARs are typically identified using pyCisTopic's differential
    accessibility analysis. This function simply marks regions as DARs
    based on external files.

    Examples
    --------
    >>> # From pyCisTopic BED files
    >>> ds.pp.mark_dars(adata_atac, dar_dir="/path/to/DARs/")

    >>> # From dictionary
    >>> dars = {"Astro": ["chr1:1000-2000", ...], "Oligo": [...]}
    >>> ds.pp.mark_dars(adata_atac, dar_dict=dars)

    >>> # Check result
    >>> adata_atac.var['is_dar'].sum()
    5000
    """
    if not inplace:
        adata = adata.copy()

    # Initialize DAR column
    adata.var["is_dar"] = False

    # Get DARs from either source
    if dar_dir is not None:
        dar_dict = _load_dars_from_dir(Path(dar_dir))
    elif dar_dict is None:
        raise ValueError("Must provide either dar_dir or dar_dict")

    # Collect all unique DAR regions
    all_dars = set()
    for _, regions in dar_dict.items():
        all_dars.update(regions)

    # Mark DARs that exist in our data
    dars_in_data = all_dars & set(adata.var_names)
    adata.var["is_dar"] = adata.var.index.isin(dars_in_data)

    n_marked = adata.var["is_dar"].sum()
    n_provided = len(all_dars)
    n_missing = n_provided - len(dars_in_data)

    print(f"Marked {n_marked} / {len(adata.var)} regions as DARs")
    if n_missing > 0:
        print(f"  ({n_missing} DAR regions not found in data)")

    if not inplace:
        return adata
    return None


def _load_dars_from_dir(dar_dir: Path) -> dict[str, list[str]]:
    """
    Load DARs from directory of BED files.

    Parameters
    ----------
    dar_dir : Path
        Directory containing .bed files.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping celltype name to list of region names.
    """
    from ..io.bed import read_bed

    dar_dict = {}

    for fname in os.listdir(dar_dir):
        if not fname.endswith(".bed"):
            continue

        celltype = fname.replace(".bed", "")
        fpath = dar_dir / fname

        # Use new read_bed function
        df = read_bed(fpath, chromsizes=None)
        dar_dict[celltype] = df["region"].tolist()

    if not dar_dict:
        raise ValueError(f"No .bed files found in {dar_dir}")

    return dar_dict


def mark_tfs(
    adata: AnnData,
    tf_list: list[str] | str | Path,
    case_sensitive: bool = False,
    inplace: bool = True,
) -> AnnData | None:
    """
    Mark transcription factors in RNA modality.

    Parameters
    ----------
    adata
        RNA expression anndata.
    tf_list
        List of TF names or path to file with one TF per line
    case_sensitive
        Whether to match TF names case-sensitively.
        If False (default), matches genes case-insensitively.
        This handles cases where gene names have different capitalization
        (e.g., 'SOX10' vs 'Sox10').
    inplace
        Whether to modify in-place

    Returns
    -------
    If inplace=False, returns modified AnnData

    Examples
    --------
    >>> import deepscenic as ds
    >>> tfs = ds.datasets.fetch_tf_collection(species="mouse")
    >>> ds.pp.mark_tfs(rna, tfs)
    >>> print(f"Marked {rna.var['is_tf'].sum()} TFs")
    """
    if not inplace:
        adata = adata.copy()

    # Load TF list if path
    if isinstance(tf_list, str | Path):
        with open(tf_list) as f:
            tf_names = [line.strip() for line in f if line.strip()]
    else:
        tf_names = list(tf_list)

    if case_sensitive:
        # Direct matching
        tf_names_in_data = [tf for tf in tf_names if tf in adata.var_names]
    else:
        # Case-insensitive matching
        # Build lookup: lowercase -> original gene name in data
        gene_name_lookup = {g.lower(): g for g in adata.var_names}
        tf_names_in_data = []
        for tf in tf_names:
            tf_lower = tf.lower()
            if tf_lower in gene_name_lookup:
                # Use the gene name as it appears in the data
                tf_names_in_data.append(gene_name_lookup[tf_lower])

    # Mark TFs
    adata.var["is_tf"] = adata.var_names.isin(tf_names_in_data)
    adata.uns["tf_order"] = tf_names_in_data

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
    Filter ATAC regions: keep region if present in >min_fraction cells of ANY cell type.

    Parameters
    ----------
    adata
        ATAC accessibility data
    celltype_key
        Key in adata.obs for cell type labels
    min_fraction
        Minimum fraction of cells per cell type
    inplace
        Whether to modify in-place

    Returns
    -------
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
        # Convert to dense for boolean operations to avoid sparse matrix issues
        X_dense = _to_dense(ct_adata.X)
        nonzero_counts = (X_dense > 0).sum(axis=0).flatten()

        passing = ct_adata.var_names[nonzero_counts >= min_cells]
        keep_regions.update(passing)

    keep_mask = adata.var_names.isin(keep_regions)
    adata._inplace_subset_var(keep_mask)

    print(f"Kept {keep_mask.sum()} / {len(keep_mask)} regions")

    if not inplace:
        return adata
    return None


def add_gene_annotation(
    adata: AnnData,
    gene_annotation: pd.DataFrame,
    columns: list[str] | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """
    Add gene annotation columns to RNA var.

    Maps gene annotation columns (e.g., chromosome, TSS) to the AnnData var DataFrame.
    This is typically used before creating MuData and splitting features by chromosome.

    Parameters
    ----------
    adata
        RNA expression data.
    gene_annotation
        Gene annotation DataFrame with gene names as index.
        Expected columns (case-insensitive):

        - Chromosome / chromosome
        - Transcription_Start_Site / TSS / tss
        - Start / start
        - End / end
        - Strand / strand

        Use ``ds.datasets.fetch_gene_annotation()`` to download.
    columns
        Columns to add to var. Default: ``['chromosome', 'tss']``.
        Available: chromosome, tss, start, end, strand.
    inplace
        Whether to modify in-place.

    Returns
    -------
    If inplace=False, returns modified AnnData.

    Examples
    --------
    >>> import deepscenic as ds
    >>> annot, _ = ds.datasets.fetch_gene_annotation(species="mmusculus")
    >>> ds.pp.add_gene_annotation(adata_rna, annot)
    >>> # Now adata_rna.var has 'chromosome' and 'tss' columns
    >>> print(adata_rna.var['chromosome'].value_counts())
    """
    if not inplace:
        adata = adata.copy()

    # Default columns to add
    if columns is None:
        columns = ["chromosome", "tss"]

    # Column name mapping (target -> possible source names)
    col_mapping = {
        "chromosome": ["Chromosome", "chromosome", "chrom", "chr"],
        "tss": ["Transcription_Start_Site", "TSS", "tss", "transcription_start_site"],
        "start": ["Start", "start"],
        "end": ["End", "end"],
        "strand": ["Strand", "strand"],
    }

    # Find genes present in both
    genes_in_data = adata.var_names.intersection(gene_annotation.index)
    n_missing = len(adata.var_names) - len(genes_in_data)

    if len(genes_in_data) == 0:
        raise ValueError("No genes from annotation found in RNA data. Check index alignment.")

    # Add requested columns
    for col in columns:
        if col not in col_mapping:
            raise ValueError(f"Unknown column: {col}. Options: {list(col_mapping.keys())}")

        # Find matching column in annotation (case-insensitive)
        annot_col = None
        for option in col_mapping[col]:
            if option in gene_annotation.columns:
                annot_col = option
                break

        if annot_col is None:
            raise ValueError(f"Column '{col}' not found in annotation. Available: {list(gene_annotation.columns)}")

        # Add to var - initialize with NA, then fill matched genes
        adata.var[col] = pd.NA
        adata.var.loc[genes_in_data, col] = gene_annotation.loc[genes_in_data, annot_col].values  # type: ignore

    print(f"Added {columns} to {len(genes_in_data)} / {len(adata.var)} genes")
    if n_missing > 0:
        print(f"  ({n_missing} genes not found in annotation)")

    if not inplace:
        return adata
    return None
