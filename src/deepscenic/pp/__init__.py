"""Preprocessing functions for deepSCENIC."""

from pathlib import Path

import mudata as md
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.model_selection import train_test_split

from .._constants import (
    DEFAULT_CELL_SPLIT_SEED,
    DEFAULT_CELL_TEST_FRACTION,
    DEFAULT_TEST_CHROMOSOMES,
)
from .basic import filter_regions_by_celltype, remove_zero_variance_genes
from .dars import load_dars_from_bed, mark_dars
from .ppi import build_ppi_network, load_string_ppi
from .search_space import compute_r2g_penalty, split_r2g_by_chromosome


def create_mudata(
    adata_rna: AnnData,
    adata_atac: AnnData,
    copy: bool = True,
) -> md.MuData:
    """
    Create MuData from RNA and ATAC AnnData objects.

    Parameters
    ----------
    adata_rna : AnnData
        RNA expression data (log-normalized recommended).
    adata_atac : AnnData
        ATAC accessibility data.

        **IMPORTANT**: ATAC data should be imputed accessibility from
        pyCisTopic, NOT raw fragment counts. pyCisTopic performs:

        - Topic modeling on the accessibility matrix
        - Imputation of accessibility scores per cell per region
        - Region filtering and quality control

        The imputed matrix provides continuous accessibility values
        that are better suited for the VAE architecture than sparse
        binary fragment counts.

        See: https://pycistopic.readthedocs.io/
    copy : bool, default=True
        Whether to copy the input data.

    Returns
    -------
    MuData
        Combined multimodal data.

    Examples
    --------
    >>> import deepscenic as ds
    >>> mdata = ds.pp.create_mudata(adata_rna, adata_atac)
    """
    if copy:
        adata_rna = adata_rna.copy()
        adata_atac = adata_atac.copy()

    # Verify cell alignment
    if not (adata_rna.obs_names == adata_atac.obs_names).all():
        raise ValueError("Cell names must match between RNA and ATAC. Consider using `mudata.intersect_obs()` first.")

    mdata = md.MuData({"rna": adata_rna, "atac": adata_atac})

    return mdata


def mark_tfs(
    mdata: md.MuData,
    tf_list: list[str] | str | Path,
    case_sensitive: bool = False,
    inplace: bool = True,
) -> md.MuData | None:
    """
    Mark transcription factors in RNA modality.

    Parameters
    ----------
    mdata : MuData
        Input multimodal data
    tf_list : list or path
        List of TF names or path to file with one TF per line
    case_sensitive : bool, default=False
        Whether to match TF names case-sensitively.
        If False (default), matches genes case-insensitively.
        This handles cases where gene names have different capitalization
        (e.g., 'SOX10' vs 'Sox10').
    inplace : bool, default=True
        Whether to modify in-place

    Returns
    -------
    MuData or None
        If inplace=False, returns modified MuData

    Examples
    --------
    >>> import deepscenic as ds
    >>> tfs = ds.datasets.fetch_tf_collection(species="mouse")
    >>> ds.pp.mark_tfs(mdata, tfs)
    >>> print(f"Marked {mdata['rna'].var['is_tf'].sum()} TFs")
    """
    if not inplace:
        mdata = mdata.copy()

    # Load TF list if path
    if isinstance(tf_list, str | Path):
        with open(tf_list) as f:
            tf_names = [line.strip() for line in f if line.strip()]
    else:
        tf_names = list(tf_list)

    rna = mdata.mod["rna"]

    if case_sensitive:
        # Direct matching
        tf_names_in_data = [tf for tf in tf_names if tf in rna.var_names]
    else:
        # Case-insensitive matching
        # Build lookup: lowercase -> original gene name in data
        gene_name_lookup = {g.lower(): g for g in rna.var_names}
        tf_names_in_data = []
        for tf in tf_names:
            tf_lower = tf.lower()
            if tf_lower in gene_name_lookup:
                # Use the gene name as it appears in the data
                tf_names_in_data.append(gene_name_lookup[tf_lower])

    # Mark TFs
    rna.var["is_tf"] = rna.var_names.isin(tf_names_in_data)
    rna.var["tf_index"] = -1
    for i, tf in enumerate(tf_names_in_data):
        rna.var.loc[tf, "tf_index"] = i
    rna.uns["tf_order"] = tf_names_in_data

    if not inplace:
        return mdata
    return None


def parse_region_coordinates(
    mdata: md.MuData,
    inplace: bool = True,
) -> md.MuData | None:
    """
    Parse ATAC region names to extract chromosome, start, end coordinates.

    Expects region names in format: 'chr1:1000-2000'

    Parameters
    ----------
    mdata : MuData
        Input multimodal data
    inplace : bool, default=True
        Whether to modify in-place

    Returns
    -------
    MuData or None
        If inplace=False, returns modified MuData
    """
    if not inplace:
        mdata = mdata.copy()

    atac = mdata.mod["atac"]

    def parse_region(region_str: str) -> tuple[str, int, int]:
        chrom, coords = region_str.split(":")
        start, end = map(int, coords.split("-"))
        return chrom, start, end

    coords = [parse_region(r) for r in atac.var_names]
    atac.var["chromosome"] = [c[0] for c in coords]
    atac.var["start"] = [c[1] for c in coords]
    atac.var["end"] = [c[2] for c in coords]

    if not inplace:
        return mdata
    return None


def split_cells(
    mdata: md.MuData,
    test_fraction: float = DEFAULT_CELL_TEST_FRACTION,
    seed: int = DEFAULT_CELL_SPLIT_SEED,
    stratify_key: str | None = None,
    inplace: bool = True,
) -> md.MuData | None:
    """
    Split cells into train/test sets.

    Parameters
    ----------
    mdata : MuData
        Input multimodal data
    test_fraction : float, default=0.2
        Fraction of cells for test set
    seed : int, default=42
        Random seed
    stratify_key : str, optional
        Key in obs for stratified splitting (e.g., 'cell_type')
    inplace : bool, default=True
        Whether to modify in-place

    Returns
    -------
    MuData or None
        If inplace=False, returns modified MuData
    """
    if not inplace:
        mdata = mdata.copy()

    stratify = mdata.obs[stratify_key] if stratify_key else None

    train_idx, test_idx = train_test_split(
        np.arange(mdata.n_obs),
        test_size=test_fraction,
        random_state=seed,
        stratify=stratify,
    )

    split_labels = np.array(["train"] * mdata.n_obs)
    split_labels[test_idx] = "test"

    mdata.obs["split"] = pd.Categorical(split_labels, categories=["train", "test"])

    if not inplace:
        return mdata
    return None


def split_features_by_chromosome(
    mdata: md.MuData,
    test_chromosomes: list[str] = DEFAULT_TEST_CHROMOSOMES,
    inplace: bool = True,
) -> md.MuData | None:
    """
    Split features (genes/regions) by chromosome.

    Parameters
    ----------
    mdata : MuData
        Input multimodal data
    test_chromosomes : list, default=['chr7', 'chr11', 'chr18', 'chr19']
        Chromosomes for test set
    inplace : bool, default=True
        Whether to modify in-place

    Returns
    -------
    MuData or None
        If inplace=False, returns modified MuData
    """
    if not inplace:
        mdata = mdata.copy()

    # ATAC regions: use parsed chromosome
    atac = mdata.mod["atac"]
    atac.var["split"] = pd.Categorical(
        ["test" if c in test_chromosomes else "train" for c in atac.var["chromosome"]],
        categories=["train", "test"],
    )

    # RNA genes: need gene annotation with TSS chromosome
    # For now, mark all as 'train' - user should add chromosome info to rna.var
    rna = mdata.mod["rna"]
    if "chromosome" in rna.var.columns:
        rna.var["split"] = pd.Categorical(
            ["test" if c in test_chromosomes else "train" for c in rna.var["chromosome"]],
            categories=["train", "test"],
        )
    else:
        # Default: all train (user should add chromosome info to rna.var for proper splitting)
        rna.var["split"] = pd.Categorical(["train"] * rna.n_vars, categories=["train", "test"])

    if not inplace:
        return mdata
    return None


__all__ = [
    # Basic
    "filter_regions_by_celltype",
    "remove_zero_variance_genes",
    # Search space
    "compute_r2g_penalty",
    "split_r2g_by_chromosome",
    # TF handling
    "mark_tfs",
    # DAR handling
    "mark_dars",
    "load_dars_from_bed",
    # PPI network
    "build_ppi_network",
    "load_string_ppi",
    # Region parsing
    "parse_region_coordinates",
    # Splits
    "split_cells",
    "split_features_by_chromosome",
    # High-level
    "create_mudata",
]
