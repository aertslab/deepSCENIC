"""Preprocessing functions for deepSCENIC. Functions in the __init__ operate on mudatas."""

import logging

import mudata as md
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.model_selection import train_test_split

from .basic import add_gene_annotation, filter_regions_by_celltype, mark_dars, mark_tfs, remove_zero_variance_genes
from .search_space import compute_r2g_penalty

log = logging.getLogger("deepscenic.pp")


def create_mudata(
    *,
    rna: AnnData,
    atac: AnnData,
    copy: bool = True,
) -> md.MuData:
    """
    Create MuData from RNA and ATAC AnnData objects.

    Simple wrapper around `mudata.MuData` with extra region coordinates parsing.
    This function requires all keyword arguments for safekeeping.

    Parameters
    ----------
    rna
        RNA expression data (log-normalized recommended).
    atac
        ATAC accessibility data.

        **IMPORTANT**: It is strongly advised that ATAC data should be imputed accessibility from pyCisTopic, NOT raw fragment counts. pyCisTopic performs:

        - Topic modeling on the accessibility matrix
        - Imputation of accessibility scores per cell per region
        - Region filtering and quality control

        The imputed matrix provides continuous accessibility values
        that are better suited for the VAE architecture than sparse
        binary fragment counts.

        See: https://pycistopic.readthedocs.io/
    copy
        Whether to copy the input data.

    Returns
    -------
    Combined multimodal data.

    Examples
    --------
    >>> import deepscenic as ds
    >>> mdata = ds.pp.create_mudata(adata_rna, adata_atac)
    """
    if copy:
        rna = rna.copy()
        atac = atac.copy()

    # Verify cell alignment
    if not (rna.obs_names == atac.obs_names).all():
        raise ValueError("Observation names must match exactly between RNA and ATAC.")

    # Parse region coordinates from ATAC var_names
    _parse_region_coordinates(atac)

    mdata = md.MuData({"rna": rna, "atac": atac})

    return mdata


def _parse_region_coordinates(adata: AnnData) -> None:
    """
    Parse region names to extract chromosome, start, end coordinates.

    Expects region names in var_names with format: 'chr1:1000-2000'

    Parameters
    ----------
    adata
        AnnData with region names in var_names. Modified in-place.
    """

    def parse_region(region_str: str) -> tuple[str, int, int]:
        chrom, coords = region_str.split(":")
        start, end = map(int, coords.split("-"))
        return chrom, start, end

    coords = [parse_region(r) for r in adata.var_names]
    adata.var["chromosome"] = [c[0] for c in coords]
    adata.var["start"] = [c[1] for c in coords]
    adata.var["end"] = [c[2] for c in coords]


def split_cells(
    mdata: md.MuData,
    test_fraction: float = 0.2,
    seed: int = 42,
    stratify_key: str | None = None,
    inplace: bool = True,
) -> md.MuData | None:
    """
    Split cells into train/test sets. Adds a 'split' column to `mdata.obs`.

    Parameters
    ----------
    mdata
        Input multimodal data
    test_fraction
        Fraction of cells for test set
    seed
        Random seed passed to `sklearn.model_selection.train_test_split`
    stratify_key
        Key in obs for stratified splitting (e.g., 'cell_type')
        If None, will not stratify.
    inplace
        Whether to modify in-place

    Returns
    -------
    If inplace=False, returns modified MuData
    """
    if not inplace:
        mdata = mdata.copy()

    stratify = mdata.obs[stratify_key] if stratify_key else None

    _, test_idx = train_test_split(
        np.arange(mdata.n_obs),
        test_size=test_fraction,
        random_state=seed,
        stratify=stratify,
    )

    split_labels = np.array(["train"] * mdata.n_obs)
    split_labels[test_idx] = "test"

    mdata.obs["split"] = pd.Categorical(split_labels, categories=["train", "test"])

    n_train = (split_labels == "train").sum()
    n_test = (split_labels == "test").sum()
    log.info(f"Split cells: {n_train} train, {n_test} test")

    if not inplace:
        return mdata
    return None


def split_features_by_chromosome(
    mdata: md.MuData,
    test_chromosomes: list[str] | None = None,
    keep_tfs_in_both: bool = True,
    inplace: bool = True,
) -> md.MuData | None:
    """
    Split features (genes/regions) by chromosome for training and evaluation.

    Adds a 'split' column to both `atac.var` and `rna.var` inside `mdata` object.
    When ``keep_tfs_in_both=True`` (default), TFs get ``split='both'`` to indicate
    they should be included in both train and test views.

    Note: The feature split affects both training and evaluation:
    - **Reconstruction loss** is computed only on TRAIN genes and TRAIN regions
    - **E2 sparsity loss** is applied to ALL region→gene links (including test genes)
    - The split enables evaluation on held-out chromosomes to assess generalization

    Expects `mdata["rna"].var` to contain "chromosome" column.
    Run `ds.pp.add_gene_annotation` first to assign gene positions based on TSS.

    Parameters
    ----------
    mdata
        Input multimodal data
    test_chromosomes
        Chromosomes for test set. Defaults to ["chr7", "chr11", "chr18", "chr19"].
    keep_tfs_in_both
        If True (default), transcription factors get ``split='both'`` regardless
        of their chromosome location. This ensures all TFs are available as
        encoder input in both train and test views.
        Set to False for strict chromosome-only splitting.
    inplace
        Whether to modify in-place

    Returns
    -------
    If inplace=False, returns modified MuData
    """
    if test_chromosomes is None:
        test_chromosomes = ["chr7", "chr11", "chr18", "chr19"]

    if not inplace:
        mdata = mdata.copy()

    # Helper to handle NA chromosome values (genes without annotation)
    def _assign_split(chrom: str) -> str:
        if pd.isna(chrom):
            return "train"  # Unannotated features go to train
        return "test" if chrom in test_chromosomes else "train"

    # ATAC: strict chromosome-based split (no "both" - regions are never shared)
    atac = mdata.mod["atac"]
    atac.var["split"] = pd.Categorical(
        [_assign_split(c) for c in atac.var["chromosome"]],
        categories=["train", "test"],
    )

    # RNA: chromosome-based split with optional TF handling
    rna = mdata.mod["rna"]
    if "chromosome" not in rna.var.columns:
        raise ValueError(
            "Expects 'chromosome' column in rna.var. Run `ds.pp.add_gene_annotation` first to annotate gene positions based on TSS."
        )

    # Initial chromosome-based assignment
    splits = [_assign_split(c) for c in rna.var["chromosome"]]

    # Override TFs to "both" if requested
    if keep_tfs_in_both and "is_tf" in rna.var.columns:
        tf_mask = rna.var["is_tf"].fillna(False)
        splits = ["both" if tf_mask.iloc[i] else splits[i] for i in range(len(splits))]
        n_tfs = tf_mask.sum()
        log.info(f"{n_tfs} TFs assigned to 'both' splits")

        rna.var["split"] = pd.Categorical(splits, categories=["train", "test", "both"])
    else:
        rna.var["split"] = pd.Categorical(splits, categories=["train", "test"])
        if keep_tfs_in_both and "is_tf" not in rna.var.columns:
            log.warning(
                "keep_tfs_in_both=True but 'is_tf' column not found in rna.var. "
                "Run `ds.pp.mark_tfs` first to mark transcription factors."
            )

    # Remove legacy in_both_splits column if it exists
    if "in_both_splits" in rna.var.columns:
        del rna.var["in_both_splits"]

    # Warn about unannotated genes
    n_unannotated = rna.var["chromosome"].isna().sum()
    if n_unannotated > 0:
        log.info(f"{n_unannotated} genes without chromosome annotation assigned to 'train' split")

    # Log summary
    n_atac_test = (atac.var["split"] == "test").sum()
    n_rna_test = (rna.var["split"] == "test").sum()
    n_rna_both = (rna.var["split"] == "both").sum() if "both" in rna.var["split"].cat.categories else 0
    log.info(f"Split features by chromosome: {n_atac_test} ATAC regions, {n_rna_test} genes in test set")
    if n_rna_both > 0:
        log.info(f"  ({n_rna_both} TFs assigned to 'both' splits)")

    if not inplace:
        return mdata
    return None


__all__ = [
    # Basic
    "filter_regions_by_celltype",
    "remove_zero_variance_genes",
    # Search space
    "compute_r2g_penalty",
    # TF handling
    "mark_tfs",
    # Gene annotation
    "add_gene_annotation",
    # DAR handling
    "mark_dars",
    # Splits
    "split_cells",
    "split_features_by_chromosome",
    # High-level
    "create_mudata",
]
