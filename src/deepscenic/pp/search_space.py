"""Scenicplus-like region-gene search space and distance penalty computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

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
    method: Literal["gaussian", "linear"],
) -> tuple[csr_matrix, dict]:
    """
    Compute R2G penalty matrix from DataFrames.

    Parameters
    ----------
    regions
        Must have columns: chromosome, start, end. Index = region names.
    genes
        Must have columns: chromosome, tss. Index = gene names.
    max_distance
        Maximum distance to consider.
    sigma
        Gaussian sigma for penalty.
    method
        Penalty method.

    Returns
    -------
    r2g
        Sparse penalty matrix (n_regions x n_genes)
    config
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
    region_to_idx: dict[str, int] = {r: i for i, r in enumerate(region_names)}
    gene_to_idx: dict[str, int] = {g: i for i, g in enumerate(gene_names)}

    # Process each chromosome
    chromosomes = set(regions["chromosome"]) & set(genes["chromosome"])

    for chrom in tqdm(sorted(chromosomes), desc="Computing R2G matrix"):
        if chrom not in region_by_chr.groups or chrom not in gene_by_chr.groups:
            continue

        chr_regions = pd.DataFrame(region_by_chr.get_group(chrom))
        chr_genes = pd.DataFrame(gene_by_chr.get_group(chrom))

        # Vectorized approach for speed
        region_centers = np.array(chr_regions["center"].values)
        gene_tss = np.array(chr_genes["tss"].values)

        # Compute all pairwise distances
        # Using broadcasting: (n_regions, 1) - (1, n_genes)
        distances = np.abs(region_centers[:, np.newaxis] - gene_tss[np.newaxis, :])

        # Find pairs within max_distance
        valid_mask = distances <= max_distance

        # Get indices of valid pairs
        region_local_idx, gene_local_idx = np.where(valid_mask)

        for r_local, g_local in zip(region_local_idx, gene_local_idx, strict=False):
            region_name = str(chr_regions.index[r_local])
            gene_name = str(chr_genes.index[g_local])
            distance = distances[r_local, g_local]

            region_idx = region_to_idx[region_name]
            gene_idx = gene_to_idx[gene_name]

            if method == "gaussian":
                penalty = float(1 - np.exp(-(distance**2) / (2 * sigma**2)))
            elif method == "linear":
                penalty = float(distance / max_distance)

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
    gene_annotation: pd.DataFrame | None = None,
    max_distance: int = 1_000_000,
    sigma: int = 100_000,
    method: Literal["gaussian", "linear"] = "gaussian",
    filter_to_rna_genes: bool = True,
    key_added: str = "r2g",
    inplace: bool = True,
) -> md.MuData | None:
    """
    Compute region-to-gene distance penalty matrix and store in MuData.

    For each region-gene pair within max_distance, compute a penalty
    value where 0 = very close (no penalty), 1 = far (high penalty).

    Parameters
    ----------
    mdata
        Must have 'atac' modality with parsed coordinates.
        Use ``ds.pp.create_mudata()`` to create the MuData (auto-parses coordinates).
    gene_annotation
        Gene annotation with columns (case-insensitive):

        - 'tss' or 'Transcription_Start_Site': transcription start site
        - 'chromosome' or 'Chromosome': chromosome name

        Index should be gene names.

        If None (default), reads from ``rna.var['chromosome']`` and
        ``rna.var['tss']``. Use ``ds.pp.add_gene_annotation()`` to populate
        these columns first.
    max_distance
        Maximum distance to consider (1Mb).
    sigma
        Gaussian sigma for distance penalty (100kb).
    method
        Penalty function.
    filter_to_rna_genes
        If True (default), filter gene_annotation to only include genes
        present in the RNA modality. This ensures the R2G matrix columns
        match the genes in the training data. Set to False to include all
        genes from the annotation. Ignored when gene_annotation is None.
    key_added
        Key in ``mdata.uns`` to store results.
    inplace
        If False, return a modified copy instead of modifying in-place.

    Returns
    -------
    If ``inplace=False``, returns modified MuData. Otherwise None.

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
    >>> mdata = ds.pp.create_mudata(rna=adata_rna, atac=adata_atac)
    >>> # Option 1: Provide annotation directly
    >>> annot, _ = ds.datasets.fetch_gene_annotation(species="mmusculus")
    >>> ds.pp.compute_r2g_penalty(mdata, annot)
    >>>
    >>> # Option 2: Use pre-populated rna.var columns
    >>> ds.pp.add_gene_annotation(adata_rna, annot)  # Before create_mudata
    >>> mdata = ds.pp.create_mudata(rna=adata_rna, atac=adata_atac)
    >>> ds.pp.compute_r2g_penalty(mdata)  # No annotation needed
    """
    if not inplace:
        mdata = mdata.copy()

    # Validate prerequisites
    if "atac" not in mdata.mod:
        raise ValueError("MuData must have 'atac' modality")

    atac = mdata["atac"]
    if "chromosome" not in atac.var.columns:
        raise ValueError(
            "ATAC modality must have parsed coordinates (chromosome, start, end columns). "
            "Use ds.pp.create_mudata() to create the MuData, which auto-parses coordinates."
        )

    # Extract and normalize regions DataFrame
    regions = pd.DataFrame(atac.var[["chromosome", "start", "end"]].copy())

    # Build genes DataFrame from annotation or rna.var
    if gene_annotation is not None:
        # Use provided annotation
        genes = gene_annotation.copy()
        gene_chrom_col = _normalize_column_name(genes, ["chromosome", "Chromosome", "chrom", "chr"])
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

        # Filter to RNA genes if requested (default: True for compatibility with training)
        if filter_to_rna_genes:
            if "rna" not in mdata.mod:
                raise ValueError(
                    "MuData must have 'rna' modality when filter_to_rna_genes=True. "
                    "Set filter_to_rna_genes=False to use all genes from annotation."
                )
            rna_gene_names = set(mdata["rna"].var_names)
            genes = genes.loc[genes.index.isin(rna_gene_names)]
            if len(genes) == 0:
                raise ValueError(
                    "No genes from annotation found in RNA modality. "
                    "Check that gene annotation index contains gene names matching RNA var_names."
                )
    else:
        # Read from rna.var
        if "rna" not in mdata.mod:
            raise ValueError("MuData must have 'rna' modality when gene_annotation is None.")
        rna = mdata["rna"]
        if "chromosome" not in rna.var.columns or "tss" not in rna.var.columns:
            raise ValueError(
                "gene_annotation not provided and rna.var missing 'chromosome'/'tss' columns. "
                "Either pass gene_annotation or call ds.pp.add_gene_annotation() first."
            )
        genes = pd.DataFrame(
            {
                "chromosome": rna.var["chromosome"],
                "tss": rna.var["tss"],
            },
            index=rna.var_names,
        )
        # Drop genes without chromosome/tss info
        genes = genes.dropna()

    # Compute matrix using internal function
    r2g_matrix, config = _compute_r2g_matrix(regions, genes, max_distance, sigma, method)

    # Filter RNA modality to genes in R2G matrix, but ALWAYS keep TFs
    # TFs are needed as encoder input even if they lack chromosome annotation
    # This matches legacy behavior where TFs are added to R2G with zero links
    r2g_gene_set = set(genes.index)
    rna = mdata.mod["rna"]

    # Determine which genes to keep: R2G genes + all TFs
    if "is_tf" in rna.var.columns:
        tf_mask = rna.var["is_tf"].fillna(False)
        genes_to_keep = [g for g in rna.var_names if g in r2g_gene_set or tf_mask[g]]
        n_tfs_without_r2g = sum(1 for g in rna.var_names if tf_mask[g] and g not in r2g_gene_set)
    else:
        genes_to_keep = [g for g in rna.var_names if g in r2g_gene_set]
        n_tfs_without_r2g = 0

    n_removed = rna.n_vars - len(genes_to_keep)

    if n_removed > 0:
        msg = f"Filtering RNA modality to {len(genes_to_keep)} genes"
        if n_tfs_without_r2g > 0:
            msg += f" ({n_tfs_without_r2g} TFs kept without R2G links)"
        msg += f" ({n_removed} non-TF genes without annotation removed)"
        print(msg)
        # Filter RNA in-place within MuData
        mdata.mod["rna"] = rna[:, genes_to_keep].copy()

    # Store in MuData
    mdata.uns[key_added] = {
        "matrix": r2g_matrix,
        "config": config,
        "region_names": regions.index.tolist(),
        "gene_names": genes.index.tolist(),
    }

    return None if inplace else mdata
