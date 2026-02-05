"""Scenicplus-like region-gene search space and distance penalty computation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    import mudata as md

log = logging.getLogger(__name__)


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

    This function also reorders the RNA modality to place TFs first
    (matching legacy behavior), which ensures TF indices are simply
    [0, 1, 2, ..., n_tfs-1] during training.

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
        present in the RNA modality. This ensures only RNA genes are used
        when computing links. Set to False to include all genes from the
        annotation. Ignored when gene_annotation is None.
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
        - 'matrix': sparse penalty matrix (n_regions x n_genes), where n_genes
          is TFs + non-TF genes with annotation. TFs without annotation have
          empty columns (0 links). Non-TF genes without annotation are removed.
        - 'config': computation parameters (max_distance, sigma, method, n_links, density)
        - 'region_names': ordered region names (list)
        - 'gene_names': ordered gene names (list), TFs first then other genes

    mdata.mod["rna"].uns["tf_order"] : list
        List of TF gene names in their new order (first n_tfs genes)

    Notes
    -----
    The penalty formula (Gaussian):
        penalty = 1 - exp(-d^2 / (2*sigma^2))

    Where d is distance from region center to gene TSS.

    The RNA modality is reordered and filtered to match legacy behavior:
    - All TFs are kept (first n_tfs genes), even without annotation (0 links)
    - Non-TF genes are only kept if they have annotation
    This ensures TF indices are [0, ..., n_tfs-1] during training.

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
    # This only includes genes that have chromosome/tss annotation
    r2g_matrix, config = _compute_r2g_matrix(regions, genes, max_distance, sigma, method)

    # Get computed gene names (those with annotation)
    computed_gene_names = genes.index.tolist()
    genes_with_annotation = set(computed_gene_names)

    # Build final gene order: TFs first, then other genes WITH ANNOTATION (legacy behavior)
    # - All TFs are kept (even without annotation, they get 0 links)
    # - Non-TF genes are only kept if they have annotation
    # This ensures TF indices are simply [0, 1, 2, ..., n_tfs-1] after reordering
    rna = mdata.mod["rna"]
    all_gene_names = list(rna.var_names)

    if "is_tf" in rna.var.columns:
        tf_mask = rna.var["is_tf"].fillna(False).values
        tf_genes = [g for i, g in enumerate(all_gene_names) if tf_mask[i]]
        # Only keep non-TF genes that have annotation (legacy behavior)
        non_tf_genes = [g for i, g in enumerate(all_gene_names) if not tf_mask[i] and g in genes_with_annotation]
    else:
        tf_genes = []
        # Only keep genes that have annotation
        non_tf_genes = [g for g in all_gene_names if g in genes_with_annotation]

    # Final gene order: TFs first, then non-TFs with annotation
    final_gene_order = tf_genes + non_tf_genes

    # Build mapping for final gene order
    final_gene_to_idx = {g: i for i, g in enumerate(final_gene_order)}

    # Expand r2g matrix to include ALL genes from RNA modality
    # Genes without annotation will have 0 links (empty columns)
    n_regions = len(regions)
    n_genes_final = len(final_gene_order)

    # Convert to COO for efficient iteration and remapping
    r2g_coo = r2g_matrix.tocoo()

    new_rows = []
    new_cols = []
    new_data = []

    for row, col, val in zip(r2g_coo.row, r2g_coo.col, r2g_coo.data, strict=False):
        # Get the gene name from the computed matrix
        gene_name = computed_gene_names[col]

        # Map to the new column index in final gene order
        if gene_name in final_gene_to_idx:
            new_rows.append(row)
            new_cols.append(final_gene_to_idx[gene_name])
            new_data.append(val)

    # Create expanded sparse matrix with all genes
    r2g_expanded = csr_matrix(
        (new_data, (new_rows, new_cols)),
        shape=(n_regions, n_genes_final),
    )

    # Update config with new stats
    config["n_links"] = len(new_data)
    config["density"] = len(new_data) / (n_regions * n_genes_final) if (n_regions * n_genes_final) > 0 else 0

    # Log TFs without annotation and removed non-TF genes
    tfs_without_annotation = [g for g in tf_genes if g not in genes_with_annotation]
    tf_set = set(tf_genes)
    n_removed_non_tfs = len([g for g in all_gene_names if g not in tf_set and g not in genes_with_annotation])

    if tfs_without_annotation:
        log.info(f"{len(tfs_without_annotation)} TFs without annotation will have 0 links")
    if n_removed_non_tfs > 0:
        log.info(f"Removed {n_removed_non_tfs} non-TF genes without annotation from RNA modality")

    # Reorder RNA modality to match final_gene_order (TFs first)
    # This is critical because training uses rna.var_names order
    mdata.mod["rna"] = rna[:, final_gene_order].copy()

    # Store TF order in uns for easy access during training
    mdata.mod["rna"].uns["tf_order"] = tf_genes

    log.info(f"Reordered RNA modality: {len(tf_genes)} TFs first, then {len(non_tf_genes)} other genes")

    # Store in MuData
    mdata.uns[key_added] = {
        "matrix": r2g_expanded,
        "config": config,
        "region_names": regions.index.tolist(),
        "gene_names": final_gene_order,  # TFs first, then others
    }

    return None if inplace else mdata
