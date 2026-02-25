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


def filter_by_genome(
    mdata: md.MuData,
    genome=None,
    inplace: bool = True,
) -> md.MuData | None:
    """
    Filter ATAC regions to chromosomes present in the genome FASTA.

    Removes regions on chromosomes not found in the registered/provided genome.
    This commonly happens with unplaced scaffolds, alternate sequences, or patches
    that are present in ATAC data but absent from your FASTA file.

    Must be called after :func:`create_mudata` (which parses region coordinates).
    Can be called before or after :func:`compute_r2g_penalty` — if r2g has already
    been computed, it is subsetted in-place to stay consistent with the filtered
    ATAC regions.

    Parameters
    ----------
    mdata
        Input multimodal data. ATAC regions must have a ``'chromosome'`` column
        (added automatically by :func:`create_mudata`).
    genome
        :class:`~deepscenic.Genome` instance to filter against. If ``None``, uses
        the globally registered genome (see :func:`~deepscenic.register_genome`).
    inplace
        Whether to modify in-place.

    Returns
    -------
    If ``inplace=False``, returns filtered MuData.

    Raises
    ------
    RuntimeError
        If no genome is registered and ``genome`` is ``None``.

    Examples
    --------
    >>> ds.register_genome("/path/to/hg38.fa")
    >>> ds.pp.filter_by_genome(mdata)
    """
    from deepscenic._genome import get_genome

    if genome is None:
        genome = get_genome()
        if genome is None:
            raise RuntimeError("No genome registered. Either pass `genome=` or call `ds.register_genome()` first.")

    if not inplace:
        mdata = mdata.copy()

    atac = mdata.mod["atac"]

    # _chrom_map covers both "chr1" and "1" variants, so we can check either convention
    valid_chroms = set(genome._chrom_map.keys())

    atac_chroms = atac.var["chromosome"]
    keep = atac_chroms.isin(valid_chroms)
    n_removed = (~keep).sum()

    if n_removed > 0:
        removed_chroms = sorted(atac_chroms[~keep].unique())
        log.info(
            f"Removed {n_removed} ATAC regions on chromosomes not found in genome "
            f"({len(removed_chroms)} chromosomes): {removed_chroms}"
        )
        atac._inplace_subset_var(keep.values)

        # Keep r2g consistent: its rows align positionally with atac.var_names
        if "r2g" in mdata.uns:
            r2g = mdata.uns["r2g"]
            region_names = r2g.get("region_names", [])
            if len(region_names) > 0:
                keep_set = set(atac.var_names)  # already filtered by _inplace_subset_var
                keep_idx = np.array([i for i, name in enumerate(region_names) if name in keep_set])
                new_matrix = r2g["matrix"][keep_idx, :]
                mdata.uns["r2g"]["matrix"] = new_matrix
                mdata.uns["r2g"]["region_names"] = [region_names[i] for i in keep_idx]
                if "config" in r2g:
                    n_r, n_g = new_matrix.shape
                    n_links = new_matrix.nnz
                    mdata.uns["r2g"]["config"]["n_links"] = n_links
                    mdata.uns["r2g"]["config"]["density"] = n_links / (n_r * n_g) if (n_r * n_g) > 0 else 0
                log.info(f"Updated r2g matrix: {new_matrix.shape[0]} regions, {new_matrix.nnz} links remaining")
    else:
        log.info("All ATAC regions are on chromosomes present in the genome.")

    if not inplace:
        return mdata
    return None


def split_features_by_chromosome(
    mdata: md.MuData,
    test_chromosomes: list[str] | None = None,
    inplace: bool = True,
) -> md.MuData | None:
    """
    Split features (genes/regions) by chromosome for training and evaluation.

    Adds a 'split' column to both ``atac.var`` and ``rna.var`` inside ``mdata``.
    All genes (including TFs) are assigned to ``'train'`` or ``'test'`` based on
    their chromosome. TFs are always available as encoder input regardless of split
    (determined by ``is_tf``), but only genes in the current split contribute to
    reconstruction loss.

    Note: The feature split affects training and evaluation:

    - **Phase 1**: Reconstruction loss on train-chromosome genes/regions only
    - **Phase 3 (r2g finetuning)**: Reconstruction loss on test-chromosome genes/regions
    - **r2g sparsity loss** is applied to train-chromosome links only (matching reconstruction scope)

    Expects ``mdata["rna"].var`` to contain a ``"chromosome"`` column.
    Run :func:`~deepscenic.pp.add_gene_annotation` first to assign gene positions based on TSS.

    Parameters
    ----------
    mdata
        Input multimodal data
    test_chromosomes
        Chromosomes for test set. Defaults to ``["chr7", "chr11", "chr18", "chr19"]``.
    inplace
        Whether to modify in-place

    Returns
    -------
    If ``inplace=False``, returns modified MuData.
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

    # RNA: chromosome-based split (TFs follow chromosome split like all other genes)
    rna = mdata.mod["rna"]
    if "chromosome" not in rna.var.columns:
        raise ValueError(
            "Expects 'chromosome' column in rna.var. Run `ds.pp.add_gene_annotation` first to annotate gene positions based on TSS."
        )

    # Assign splits based on chromosome
    splits = [_assign_split(c) for c in rna.var["chromosome"]]
    rna.var["split"] = pd.Categorical(splits, categories=["train", "test"])

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
    if "is_tf" in rna.var.columns:
        tf_mask = rna.var["is_tf"].fillna(False)
        n_tf_test = (rna.var.loc[tf_mask, "split"] == "test").sum()
        log.info(
            f"Split features by chromosome: {n_atac_test} ATAC regions, "
            f"{n_rna_test} genes ({n_tf_test} TFs) in test set"
        )
    else:
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
    # Splits
    "split_cells",
    "split_features_by_chromosome",
    "filter_by_genome",
    # High-level
    "create_mudata",
]
