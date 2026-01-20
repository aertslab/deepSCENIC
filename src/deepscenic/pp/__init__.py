"""Preprocessing functions for deepSCENIC. Functions in the __init__ operate on mudatas."""

import logging

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
from .basic import add_gene_annotation, filter_regions_by_celltype, mark_dars, mark_tfs, remove_zero_variance_genes
from .ppi import build_ppi_network, load_string_ppi
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
    test_fraction: float = DEFAULT_CELL_TEST_FRACTION,
    seed: int = DEFAULT_CELL_SPLIT_SEED,
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
    test_chromosomes: list[str] = DEFAULT_TEST_CHROMOSOMES,
    inplace: bool = True,
) -> md.MuData | None:
    """
    Split features (genes/regions) by chromosome. Adds a 'split' column to both `atac.var` and `rna.var` inside `mdata` object.

    Expects `mdata["rna"].var` to contain "chromosome" column.
    Run `ds.pp.add_gene_annotation` first to asign gene positions based on TSS.

    Parameters
    ----------
    mdata
        Input multimodal data
    test_chromosomes
        Chromosomes for test set
    inplace
        Whether to modify in-place

    Returns
    -------
    If inplace=False, returns modified MuData
    """
    if not inplace:
        mdata = mdata.copy()

    # Helper to handle NA chromosome values (genes without annotation)
    def _assign_split(chrom: str) -> str:
        if pd.isna(chrom):
            return "train"  # Unannotated features go to train (matches legacy behavior)
        return "test" if chrom in test_chromosomes else "train"

    atac = mdata.mod["atac"]
    atac.var["split"] = pd.Categorical(
        [_assign_split(c) for c in atac.var["chromosome"]],
        categories=["train", "test"],
    )

    rna = mdata.mod["rna"]
    if "chromosome" in rna.var.columns:
        rna.var["split"] = pd.Categorical(
            [_assign_split(c) for c in rna.var["chromosome"]],
            categories=["train", "test"],
        )
        # Warn about unannotated genes
        n_unannotated = rna.var["chromosome"].isna().sum()
        if n_unannotated > 0:
            log.info(f"{n_unannotated} genes without chromosome annotation assigned to 'train' split")
    else:
        raise ValueError(
            "Expects 'chromosome' column in rna.var. Run `ds.pp.add_gene_annotation` first to annotate gene positions based on TSS."
        )

    n_atac_test = (atac.var["split"] == "test").sum()
    n_rna_test = (rna.var["split"] == "test").sum()
    log.info(f"Split features by chromosome: {n_atac_test} ATAC regions, {n_rna_test} genes in test set")

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
    # PPI network
    "build_ppi_network",
    "load_string_ppi",
    # Splits
    "split_cells",
    "split_features_by_chromosome",
    # High-level
    "create_mudata",
]
