"""Preprocessing functions for deepSCENIC."""

from datetime import datetime
from pathlib import Path

import mudata as md
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.model_selection import train_test_split

import deepscenic

from .._constants import (
    DEFAULT_CELL_SPLIT_SEED,
    DEFAULT_CELL_TEST_FRACTION,
    DEFAULT_R2G_MAX_DISTANCE,
    DEFAULT_R2G_SIGMA,
    DEFAULT_TEST_CHROMOSOMES,
)
from .basic import filter_genes, filter_regions_by_celltype, normalize_rna
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
    inplace : bool, default=True
        Whether to modify in-place

    Returns
    -------
    MuData or None
        If inplace=False, returns modified MuData
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

    # Filter to TFs in data
    tf_names_in_data = [tf for tf in tf_names if tf in rna.var_names]

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
    # For now, mark all as 'train' - user should provide gene_annotation
    # or use prepare_for_training which handles this
    rna = mdata.mod["rna"]
    if "chromosome" in rna.var.columns:
        rna.var["split"] = pd.Categorical(
            ["test" if c in test_chromosomes else "train" for c in rna.var["chromosome"]],
            categories=["train", "test"],
        )
    else:
        # Default: all train (will be updated by prepare_for_training)
        rna.var["split"] = pd.Categorical(["train"] * rna.n_vars, categories=["train", "test"])

    if not inplace:
        return mdata
    return None


def prepare_for_training(
    mdata: md.MuData,
    tf_list: list[str] | str | Path,
    gene_annotation: pd.DataFrame | None = None,
    test_chromosomes: list[str] = DEFAULT_TEST_CHROMOSOMES,
    cell_test_fraction: float = DEFAULT_CELL_TEST_FRACTION,
    cell_split_seed: int = DEFAULT_CELL_SPLIT_SEED,
    max_r2g_distance: int = DEFAULT_R2G_MAX_DISTANCE,
    r2g_sigma: int = DEFAULT_R2G_SIGMA,
    inplace: bool = True,
) -> md.MuData | None:
    """
    Prepare MuData for deepSCENIC training.

    This is the main preprocessing function that:
    1. Marks TFs in the RNA var
    2. Parses ATAC region coordinates
    3. Splits cells and features by chromosome
    4. Computes R2G penalty matrix
    5. Adds all required metadata

    Parameters
    ----------
    mdata : MuData
        Input data (RNA + ATAC)
    tf_list : list or path
        List of TF names or path to file
    gene_annotation : DataFrame, optional
        Gene annotation with 'tss' and 'chromosome' columns.
        Index should be gene names.
    test_chromosomes : list
        Chromosomes for test set
    cell_test_fraction : float
        Fraction of cells for test set
    cell_split_seed : int
        Random seed for cell split
    max_r2g_distance : int
        Maximum R2G distance in bp
    r2g_sigma : int
        Gaussian sigma for R2G penalty in bp
    inplace : bool
        Whether to modify in-place

    Returns
    -------
    MuData or None
        Prepared MuData if inplace=False

    Examples
    --------
    >>> import deepscenic as ds
    >>> mdata = ds.pp.create_mudata(adata_rna, adata_atac)
    >>> ds.pp.prepare_for_training(
    ...     mdata,
    ...     tf_list="allTFs_mm.txt",
    ...     gene_annotation=gene_df,
    ... )
    >>> ds.write(mdata, "prepared_dataset.h5mu")
    """
    if not inplace:
        mdata = mdata.copy()

    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]

    # 1. Mark TFs
    mark_tfs(mdata, tf_list, inplace=True)

    # 2. Parse region coordinates
    parse_region_coordinates(mdata, inplace=True)

    # 3. Split cells
    split_cells(
        mdata,
        test_fraction=cell_test_fraction,
        seed=cell_split_seed,
        inplace=True,
    )

    # 4. Split features by chromosome
    # First, mark ATAC regions
    atac.var["split"] = pd.Categorical(
        ["test" if c in test_chromosomes else "train" for c in atac.var["chromosome"]],
        categories=["train", "test"],
    )

    # For genes, need annotation
    if gene_annotation is not None:
        # Filter to genes in data
        genes_in_data = [g for g in rna.var_names if g in gene_annotation.index]

        # Add chromosome info to RNA var
        rna.var["chromosome"] = None
        rna.var["tss"] = None
        for gene in genes_in_data:
            rna.var.loc[gene, "chromosome"] = gene_annotation.loc[gene, "chromosome"]
            rna.var.loc[gene, "tss"] = gene_annotation.loc[gene, "tss"]

        # Mark split
        rna.var["split"] = pd.Categorical(
            ["test" if rna.var.loc[g, "chromosome"] in test_chromosomes else "train" for g in rna.var_names],
            categories=["train", "test"],
        )

        # 5. Compute R2G matrix
        regions_df = atac.var[["chromosome", "start", "end"]].copy()
        genes_df = gene_annotation.loc[genes_in_data, ["tss", "chromosome"]].copy()

        r2g_full, r2g_config = compute_r2g_penalty(
            regions_df,
            genes_df,
            max_distance=max_r2g_distance,
            sigma=r2g_sigma,
        )

        # Split R2G
        r2g_train, r2g_test, split_info = split_r2g_by_chromosome(
            r2g_full,
            regions_df,
            genes_df,
            test_chromosomes,
        )

        # Store R2G in uns
        mdata.uns["r2g"] = {
            "train": r2g_train,
            "test": r2g_test,
            "config": r2g_config,
            "region_order_train": split_info["region_order_train"],
            "region_order_test": split_info["region_order_test"],
            "gene_order_train": split_info["gene_order_train"],
            "gene_order_test": split_info["gene_order_test"],
        }
    else:
        # No gene annotation - skip R2G computation
        # User will need to add R2G manually
        rna.var["split"] = pd.Categorical(["train"] * rna.n_vars, categories=["train", "test"])

    # 6. Add metadata
    mdata.uns["deepscenic_version"] = deepscenic.__version__
    mdata.uns["preprocessing"] = {
        "rna_normalized": True,  # Assume already done
        "atac_imputed": True,  # Assume pyCisTopic output
        "steps": ["prepare_for_training"],
    }
    mdata.uns["splits"] = {
        "cell_split_seed": cell_split_seed,
        "cell_split_ratio": cell_test_fraction,
        "region_split_chromosomes": test_chromosomes,
        "created_at": datetime.now().isoformat(),
    }

    if not inplace:
        return mdata
    return None


__all__ = [
    # Basic
    "filter_genes",
    "normalize_rna",
    "filter_regions_by_celltype",
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
    "prepare_for_training",
]
