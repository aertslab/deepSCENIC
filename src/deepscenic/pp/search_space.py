"""Region-gene search space and distance penalty computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .._constants import DEFAULT_R2G_MAX_DISTANCE, DEFAULT_R2G_SIGMA

if TYPE_CHECKING:
    import mudata as md


def _normalize_column_name(df: pd.DataFrame, options: list[str]) -> str | None:
    """Find a column name from a list of options (case-insensitive)."""
    df_cols_lower: dict[str, str] = {str(c).lower(): str(c) for c in df.columns}
    for opt in options:
        if opt.lower() in df_cols_lower:
            return df_cols_lower[opt.lower()]
    return None


def _compute_r2g_matrix(
    regions: pd.DataFrame,
    genes: pd.DataFrame,
    max_distance: int,
    sigma: int,
    method: str,
) -> tuple[csr_matrix, dict]:
    """
    Compute R2G penalty matrix from DataFrames.

    Parameters
    ----------
    regions : DataFrame
        Must have columns: chromosome, start, end. Index = region names.
    genes : DataFrame
        Must have columns: chromosome, tss. Index = gene names.
    max_distance : int
        Maximum distance to consider.
    sigma : int
        Gaussian sigma for penalty.
    method : str
        Penalty method ('gaussian' or 'linear').

    Returns
    -------
    r2g : csr_matrix
        Sparse penalty matrix (n_regions x n_genes)
    config : dict
        Computation parameters and stats
    """
    from tqdm import tqdm

    # Pre-compute region centers
    regions = regions.copy()
    regions["center"] = (regions["start"] + regions["end"]) // 2

    # Group by chromosome for efficiency
    region_by_chr = regions.groupby("chromosome")
    gene_by_chr = genes.groupby("chromosome")

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    region_names = regions.index.tolist()
    gene_names = genes.index.tolist()
    region_to_idx = {r: i for i, r in enumerate(region_names)}
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Process each chromosome
    chromosomes = set(regions["chromosome"]) & set(genes["chromosome"])

    for chrom in tqdm(sorted(chromosomes), desc="Computing R2G matrix"):
        if chrom not in region_by_chr.groups or chrom not in gene_by_chr.groups:
            continue

        chr_regions = region_by_chr.get_group(chrom)
        chr_genes = gene_by_chr.get_group(chrom)

        # Vectorized approach for speed
        region_centers = chr_regions["center"].values
        gene_tss = chr_genes["tss"].values

        # Compute all pairwise distances
        # Using broadcasting: (n_regions, 1) - (1, n_genes)
        distances = np.abs(region_centers[:, np.newaxis] - gene_tss[np.newaxis, :])

        # Find pairs within max_distance
        valid_mask = distances <= max_distance

        # Get indices of valid pairs
        region_local_idx, gene_local_idx = np.where(valid_mask)

        for r_local, g_local in zip(region_local_idx, gene_local_idx, strict=False):
            region_name = chr_regions.index[r_local]
            gene_name = chr_genes.index[g_local]
            distance = distances[r_local, g_local]

            region_idx = region_to_idx[region_name]
            gene_idx = gene_to_idx[gene_name]

            if method == "gaussian":
                penalty = 1 - np.exp(-(distance**2) / (2 * sigma**2))
            elif method == "linear":
                penalty = distance / max_distance
            else:
                raise ValueError(f"Unknown method: {method}")

            rows.append(region_idx)
            cols.append(gene_idx)
            values.append(penalty)

    # Create sparse matrix
    r2g = csr_matrix((values, (rows, cols)), shape=(len(region_names), len(gene_names)))

    config = {
        "max_distance": max_distance,
        "sigma": sigma,
        "method": method,
        "n_links": len(values),
        "density": len(values) / (len(region_names) * len(gene_names))
        if (len(region_names) * len(gene_names)) > 0
        else 0,
    }

    return r2g, config


def compute_r2g_penalty(
    mdata: md.MuData,
    gene_annotation: pd.DataFrame,
    max_distance: int = DEFAULT_R2G_MAX_DISTANCE,
    sigma: int = DEFAULT_R2G_SIGMA,
    method: str = "gaussian",
    key_added: str = "r2g",
    copy: bool = False,
) -> md.MuData | None:
    """
    Compute region-to-gene distance penalty matrix and store in MuData.

    For each region-gene pair within max_distance, compute a penalty
    value where 0 = very close (no penalty), 1 = far (high penalty).

    Parameters
    ----------
    mdata : MuData
        Must have 'atac' modality with parsed coordinates.
        Run ``ds.pp.parse_region_coordinates(mdata)`` first.
    gene_annotation : DataFrame
        Gene annotation with columns (case-insensitive):
        - 'tss' or 'Transcription_Start_Site': transcription start site
        - 'chromosome' or 'Chromosome': chromosome name
        Index should be gene names.
    max_distance : int, default=1_000_000
        Maximum distance to consider (1Mb).
    sigma : int, default=100_000
        Gaussian sigma for distance penalty (100kb).
    method : {'gaussian', 'linear'}
        Penalty function.
    key_added : str, default='r2g'
        Key in ``mdata.uns`` to store results.
    copy : bool, default=False
        If True, return a modified copy instead of modifying in-place.

    Returns
    -------
    MuData or None
        If ``copy=True``, returns modified MuData. Otherwise None.

    Stores
    ------
    mdata.uns[key_added] : dict
        - 'matrix': sparse penalty matrix (n_regions x n_genes)
        - 'config': computation parameters (max_distance, sigma, method, n_links, density)
        - 'region_names': ordered region names (list)
        - 'gene_names': ordered gene names (list)

    Notes
    -----
    The penalty formula (Gaussian):
        penalty = 1 - exp(-d^2 / (2*sigma^2))

    Where d is distance from region center to gene TSS.

    Examples
    --------
    >>> import deepscenic as ds
    >>> ds.pp.parse_region_coordinates(mdata)
    >>> annot, _ = ds.datasets.fetch_gene_annotation(species="mmusculus")
    >>> ds.pp.compute_r2g_penalty(mdata, annot)
    >>> # Results stored in mdata.uns['r2g']
    >>> print(f"Created {mdata.uns['r2g']['config']['n_links']} region-gene links")
    """
    if copy:
        mdata = mdata.copy()

    # Validate prerequisites
    if "atac" not in mdata.mod:
        raise ValueError("MuData must have 'atac' modality")

    atac = mdata["atac"]
    if "chromosome" not in atac.var.columns:
        raise ValueError(
            "ATAC modality must have parsed coordinates. "
            "Run ds.pp.parse_region_coordinates(mdata) first."
        )

    # Extract and normalize regions DataFrame
    regions = atac.var[["chromosome", "start", "end"]].copy()

    # Normalize gene annotation column names
    genes = gene_annotation.copy()
    gene_chrom_col = _normalize_column_name(
        genes, ["chromosome", "Chromosome", "chrom", "chr"]
    )
    gene_tss_col = _normalize_column_name(
        genes, ["tss", "Transcription_Start_Site", "transcription_start_site", "TSS"]
    )

    if gene_chrom_col is None or gene_tss_col is None:
        raise ValueError(
            f"Gene annotation must have columns for chromosome and TSS. "
            f"Found columns: {list(genes.columns)}. "
            f"Expected: chromosome/Chromosome and tss/Transcription_Start_Site"
        )

    # Rename to standard lowercase names for internal use
    genes = genes.rename(columns={gene_chrom_col: "chromosome", gene_tss_col: "tss"})

    # Compute matrix using internal function
    r2g_matrix, config = _compute_r2g_matrix(
        regions, genes, max_distance, sigma, method
    )

    # Store in MuData
    mdata.uns[key_added] = {
        "matrix": r2g_matrix,
        "config": config,
        "region_names": regions.index.tolist(),
        "gene_names": genes.index.tolist(),
    }

    return mdata if copy else None


def split_r2g_by_chromosome(
    r2g: csr_matrix,
    regions_df: pd.DataFrame,
    genes_df: pd.DataFrame,
    test_chromosomes: list[str],
) -> tuple[csr_matrix, csr_matrix, dict]:
    """
    Split R2G matrix into train/test by chromosome.

    Parameters
    ----------
    r2g : csr_matrix
        Full R2G matrix (n_regions x n_genes)
    regions_df : DataFrame
        Must have 'chromosome' column, index = region names
    genes_df : DataFrame
        Must have 'chromosome' column, index = gene names
    test_chromosomes : list
        Chromosomes for test set (e.g., ['chr7', 'chr11', 'chr18', 'chr19'])

    Returns
    -------
    r2g_train : csr_matrix
        Train regions x train genes
    r2g_test : csr_matrix
        Test regions x test genes
    split_info : dict
        Information about the split including ordering
    """
    # Create masks based on chromosome
    train_region_mask = ~regions_df["chromosome"].isin(test_chromosomes)
    test_region_mask = regions_df["chromosome"].isin(test_chromosomes)

    train_gene_mask = ~genes_df["chromosome"].isin(test_chromosomes)
    test_gene_mask = genes_df["chromosome"].isin(test_chromosomes)

    # Get ordered names for each split
    train_regions = regions_df.index[train_region_mask].tolist()
    test_regions = regions_df.index[test_region_mask].tolist()
    train_genes = genes_df.index[train_gene_mask].tolist()
    test_genes = genes_df.index[test_gene_mask].tolist()

    # Subset matrices using boolean indexing
    # Need to convert to arrays for scipy sparse indexing
    train_r_idx = np.where(train_region_mask)[0]
    test_r_idx = np.where(test_region_mask)[0]
    train_g_idx = np.where(train_gene_mask)[0]
    test_g_idx = np.where(test_gene_mask)[0]

    r2g_train = r2g[train_r_idx][:, train_g_idx]
    r2g_test = r2g[test_r_idx][:, test_g_idx]

    split_info = {
        "test_chromosomes": test_chromosomes,
        "n_train_regions": len(train_regions),
        "n_test_regions": len(test_regions),
        "n_train_genes": len(train_genes),
        "n_test_genes": len(test_genes),
        "region_order_train": train_regions,
        "region_order_test": test_regions,
        "gene_order_train": train_genes,
        "gene_order_test": test_genes,
    }

    return r2g_train, r2g_test, split_info
