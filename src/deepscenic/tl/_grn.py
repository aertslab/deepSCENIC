"""
GRN extraction functions for deepSCENIC.

This module provides functions to extract Gene Regulatory Networks (GRNs)
from trained deepSCENIC models.

Basic Extraction
----------------
- extract_grn: Extract tf2r and r2g matrices as DataFrames
- extract_tf2r_matrix: Extract TF→region binding matrix
- extract_r2g_matrix: Extract region→gene regulatory matrix

Query Functions
---------------
- get_tf_targets: Get target genes for a specific TF
- get_gene_regulators: Get TFs that regulate a specific gene

Cell-Type Aware Extraction
--------------------------
For cell-type specific GRN analysis, use these functions in order:

1. identify_active_enhancers: Wilcoxon differential analysis to find active enhancers
2. compute_tf_activity_scores: TF activity weighted by tf2r to active enhancers
3. identify_key_tfs: Otsu threshold to identify key TFs
4. build_grn_for_tfs: Build GRN for selected TFs

Example
-------
>>> import deepscenic as ds
>>>
>>> # Load model and data
>>> model = ds.tl.load_model("model.pt")
>>> mdata = ds.read("data.h5mu")
>>>
>>> # Basic GRN extraction
>>> grn = ds.tl.extract_grn(model)
>>> sox10_targets = ds.tl.get_tf_targets(model, "SOX10")
>>>
>>> # Cell-type aware workflow
>>> active_enh = ds.tl.identify_active_enhancers(model, mdata, "celltype")
>>> tf_scores = ds.tl.compute_tf_activity_scores(model, mdata, "celltype", active_enh)
>>> key_tfs = ds.tl.identify_key_tfs(tf_scores)
>>> grn_df = ds.tl.build_grn_for_tfs(model, key_tfs)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    import mudata as md

    from ._model import DeepSCENICModel

logger = logging.getLogger(__name__)

# Column definitions for typed DataFrames
_TF_TARGETS_COLUMNS = ["tf", "region", "tf2r_weight", "gene", "r2g_weight", "combined_weight"]
_GENE_REGULATORS_COLUMNS = ["gene", "tf", "region", "tf2r_weight", "r2g_weight", "combined_weight"]


def _otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Compute Otsu's threshold for a 1D array.

    Finds the threshold that minimizes intra-class variance.

    Parameters
    ----------
    values
        1D array of values to threshold.
    nbins
        Number of histogram bins.

    Returns
    -------
    float
        Optimal threshold value.
    """
    counts, bin_edges = np.histogram(values, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Cumulative sums for background (class 0) and foreground (class 1)
    weight_bg = np.cumsum(counts).astype(np.float64)
    weight_fg = weight_bg[-1] - weight_bg

    # Cumulative means (suppress divide-by-zero; np.where guards the zeros)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = np.cumsum(counts * bin_centers)
        mean_bg = np.where(weight_bg > 0, mean_bg / weight_bg, 0)

        mean_fg_cumsum = np.cumsum((counts * bin_centers)[::-1])[::-1]
        mean_fg = np.where(weight_fg > 0, mean_fg_cumsum / weight_fg, 0)

    # Between-class variance
    variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

    # Degenerate case: all values identical → variance is zero everywhere
    if variance_between.max() == 0:
        return float(np.mean(values))

    idx = np.argmax(variance_between)
    return float(bin_centers[idx])


def _get_otsu_threshold(values: np.ndarray, fallback: float) -> tuple[float, bool]:
    """Compute Otsu threshold on positive values, with fallback on failure.

    Returns
    -------
    tuple[float, bool]
        (threshold_value, used_otsu) - the threshold and whether Otsu was used
    """
    positive_values = values[values > 0]
    if len(positive_values) > 1:
        try:
            return float(_otsu_threshold(positive_values)), True
        except ValueError:
            return fallback, False
    return fallback, False


def extract_grn(model: DeepSCENICModel) -> dict[str, pd.DataFrame]:
    """
    Extract GRN matrices (tf2r and r2g) as DataFrames with names.

    Parameters
    ----------
    model
        Trained DeepSCENICModel

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with:
        - 'tf2r': DataFrame (n_regions, n_tfs) with region/TF names
        - 'r2g': DataFrame (sparse format) with region/gene links

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> grn = ds.tl.extract_grn(model)
    >>> tf2r = grn['tf2r']  # TF → region weights
    >>> r2g = grn['r2g']  # region → gene weights
    """
    return {
        "tf2r": extract_tf2r_matrix(model),
        "r2g": extract_r2g_matrix(model, as_edgelist=True),
    }


def extract_tf2r_matrix(model: DeepSCENICModel) -> pd.DataFrame:
    """
    Extract TF→region (tf2r) matrix as DataFrame.

    Parameters
    ----------
    model
        Trained DeepSCENICModel

    Returns
    -------
    pd.DataFrame
        tf2r matrix (n_regions, n_tfs) with region/TF names
    """
    with torch.no_grad():
        E1_values = model.adj_tf2r.cpu().numpy()

    return pd.DataFrame(
        E1_values,
        index=model.region_names,  # type: ignore[arg-type]
        columns=model.tf_names,  # type: ignore[arg-type]
    )


def extract_r2g_matrix(
    model: DeepSCENICModel,
    as_edgelist: bool = True,
) -> pd.DataFrame:
    """
    Extract region→gene (r2g) matrix.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    as_edgelist
        If True (default), return edge list format with columns (region, gene, weight).
        If False, return dense DataFrame (n_regions × n_genes). Warning: may be very large.

    Returns
    -------
    pd.DataFrame
        r2g matrix in edge list or dense format
    """
    with torch.no_grad():
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[operator]
        r2g_vals = model.vae.adj_r2g.abs().cpu().numpy()  # type: ignore[operator]

    if as_edgelist:
        # Edge list format: (region, gene, weight) rows
        return pd.DataFrame(
            {
                "region": [model.region_names[idx] for idx in r2g_indices[0]],
                "gene": [model.gene_names[idx] for idx in r2g_indices[1]],
                "weight": r2g_vals,
            }
        )
    else:
        # Dense format using torch sparse tensor (faster than Python loop)
        # Warning: may be very large for real datasets
        r2g_sparse = torch.sparse_coo_tensor(
            torch.from_numpy(r2g_indices),
            torch.from_numpy(r2g_vals),
            size=(len(model.region_names), len(model.gene_names)),
        )
        r2g_dense = r2g_sparse.to_dense().numpy()

        return pd.DataFrame(
            r2g_dense,
            index=model.region_names,  # type: ignore[arg-type]
            columns=model.gene_names,  # type: ignore[arg-type]
        )


def get_tf_targets(
    model: DeepSCENICModel,
    tf_name: str,
    tf2r_threshold: float | None = None,
    r2g_threshold: float = 0.0,
    top_k: int | None = None,
) -> pd.DataFrame:
    """
    Get target regions and genes for a specific TF.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    tf_name
        Name of the TF
    tf2r_threshold
        Threshold for tf2r (TF→region) weights. If None (default), uses Otsu's
        method to automatically determine threshold.
    r2g_threshold
        Threshold for r2g (region→gene) weights. Default 0.0.
    top_k
        If specified, return only top K targets

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: tf, region, tf2r_weight, gene, r2g_weight, combined_weight

    Notes
    -----
    - tf2r values can be positive or negative (activation/repression)
    - Thresholding is applied to absolute values
    - Results are sorted by absolute combined_weight (strongest effects first)
    - The original sign is preserved in tf2r_weight and combined_weight columns
    """
    if tf_name not in model.tf_names:
        raise ValueError(f"TF '{tf_name}' not found")

    tf_idx = model.tf_names.index(tf_name)

    with torch.no_grad():
        # Get tf2r weights for this TF
        tf2r_tf = model.adj_tf2r[:, tf_idx].cpu().numpy()

        # Get r2g sparse info
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[operator]
        r2g_vals = model.vae.adj_r2g.abs().cpu().numpy()  # type: ignore[operator]

    # Determine tf2r threshold (Otsu by default, computed on absolute values)
    tf2r_tf_abs = np.abs(tf2r_tf)
    if tf2r_threshold is None:
        tf2r_thresh, used_otsu = _get_otsu_threshold(tf2r_tf_abs, fallback=0.0)
        if used_otsu:
            logger.info(f"TF '{tf_name}': using Otsu tf2r threshold = {tf2r_thresh:.4f}")
        else:
            logger.info(f"TF '{tf_name}': Otsu failed, using tf2r threshold = {tf2r_thresh:.4f}")
    else:
        tf2r_thresh = tf2r_threshold
        logger.info(f"TF '{tf_name}': using manual tf2r threshold = {tf2r_thresh:.4f}")

    # Find regions with tf2r weight above threshold (using absolute values)
    results = []
    for region_idx in np.where(tf2r_tf_abs > tf2r_thresh)[0]:
        tf2r_w = tf2r_tf[region_idx]
        region_name = model.region_names[region_idx]

        # Find genes linked to this region
        link_mask = r2g_indices[0] == region_idx
        if link_mask.any():
            gene_indices = r2g_indices[1, link_mask]
            r2g_weights = r2g_vals[link_mask]

            for gene_idx, r2g_w in zip(gene_indices, r2g_weights, strict=False):
                if r2g_w > r2g_threshold:
                    results.append(
                        {
                            "tf": tf_name,
                            "region": region_name,
                            "tf2r_weight": tf2r_w,
                            "gene": model.gene_names[gene_idx],
                            "r2g_weight": r2g_w,
                            "combined_weight": tf2r_w * r2g_w,
                        }
                    )

    if not results:
        return pd.DataFrame(columns=_TF_TARGETS_COLUMNS)
    df = pd.DataFrame(results)

    if len(df) > 0:
        # Sort by absolute combined_weight (strongest effects first, regardless of sign)
        df = df.assign(_abs_weight=df["combined_weight"].abs())
        df = df.sort_values("_abs_weight", ascending=False).drop(columns=["_abs_weight"])

        if top_k is not None:
            df = df.head(top_k)

    return df


def get_gene_regulators(
    model: DeepSCENICModel,
    gene_name: str,
    tf2r_threshold: float | None = None,
    r2g_threshold: float = 0.0,
    top_k: int | None = None,
) -> pd.DataFrame:
    """
    Get TFs and regions that regulate a specific gene.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    gene_name
        Name of the gene
    tf2r_threshold
        Threshold for tf2r (TF→region) weights. If None (default), uses Otsu's
        method to automatically determine threshold from linked regions.
    r2g_threshold
        Threshold for r2g (region→gene) weights. Default 0.0.
    top_k
        If specified, return only top K regulators

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gene, tf, region, tf2r_weight, r2g_weight, combined_weight

    Notes
    -----
    - tf2r values can be positive or negative (activation/repression)
    - Thresholding is applied to absolute values
    - Results are sorted by absolute combined_weight (strongest effects first)
    - The original sign is preserved in tf2r_weight and combined_weight columns
    """
    if gene_name not in model.gene_names:
        raise ValueError(f"Gene '{gene_name}' not found")

    gene_idx = model.gene_names.index(gene_name)

    with torch.no_grad():
        tf2r_matrix = model.adj_tf2r.cpu().numpy()
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[operator]
        r2g_vals = model.vae.adj_r2g.abs().cpu().numpy()  # type: ignore[operator]

    # Find regions linked to this gene
    link_mask = r2g_indices[1] == gene_idx
    if not link_mask.any():
        return pd.DataFrame(columns=_GENE_REGULATORS_COLUMNS)

    region_indices = r2g_indices[0, link_mask]
    r2g_weights_for_gene = r2g_vals[link_mask]

    # Pre-compute per-TF thresholds if using automatic thresholding
    if tf2r_threshold is None:
        # Compute Otsu threshold for each TF (same as get_tf_targets)
        tf_thresholds = {}
        for tf_idx in range(tf2r_matrix.shape[1]):
            tf2r_tf_abs = np.abs(tf2r_matrix[:, tf_idx])
            tf_thresh, _ = _get_otsu_threshold(tf2r_tf_abs, fallback=0.0)
            tf_thresholds[tf_idx] = tf_thresh
        logger.info(f"Gene '{gene_name}': using per-TF Otsu thresholds")
    else:
        tf_thresholds = None  # Use manual threshold for all TFs
        logger.info(f"Gene '{gene_name}': using manual tf2r threshold = {tf2r_threshold:.4f}")

    results = []
    for region_idx, r2g_w in zip(region_indices, r2g_weights_for_gene, strict=False):
        if r2g_w <= r2g_threshold:
            continue

        region_name = model.region_names[region_idx]
        tf2r_weights_region = tf2r_matrix[region_idx, :]
        tf2r_weights_region_abs = np.abs(tf2r_weights_region)

        for tf_idx in range(len(model.tf_names)):
            # Use per-TF threshold or manual threshold
            thresh = tf_thresholds[tf_idx] if tf_thresholds else tf2r_threshold

            if tf2r_weights_region_abs[tf_idx] > thresh:
                tf2r_w = tf2r_weights_region[tf_idx]
                results.append(
                    {
                        "gene": gene_name,
                        "tf": model.tf_names[tf_idx],
                        "region": region_name,
                        "tf2r_weight": tf2r_w,
                        "r2g_weight": r2g_w,
                        "combined_weight": tf2r_w * r2g_w,
                    }
                )

    if not results:
        return pd.DataFrame(columns=_GENE_REGULATORS_COLUMNS)
    df = pd.DataFrame(results)

    if len(df) > 0:
        # Sort by absolute combined_weight (strongest effects first, regardless of sign)
        df = df.assign(_abs_weight=df["combined_weight"].abs())
        df = df.sort_values("_abs_weight", ascending=False).drop(columns=["_abs_weight"])

        if top_k is not None:
            df = df.head(top_k)

    return df


# =============================================================================
# Cell-Type Aware GRN Extraction
# =============================================================================


def identify_active_enhancers(
    model: DeepSCENICModel,
    mdata: md.MuData,
    celltype_key: str,
    top_n: int | None = 3000,
    logfc_threshold: float | None = None,
    pval_threshold: float = 0.05,
    key_prefix: str = "X_deepscenic_",
) -> dict[str, list[str]]:
    """
    Identify active enhancers per cell type using differential analysis.

    Uses the Wilcoxon rank-sum test (via scanpy's rank_genes_groups) to find
    enhancers that are differentially active in each cell type compared to
    all others.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    mdata
        MuData with RNA modality
    celltype_key
        Column in mdata.obs containing cell type labels
    top_n
        Maximum number of active enhancers per cell type.
        Applied after logfc/pval filtering. If None, no limit. Default 3000.
    logfc_threshold
        Minimum log fold change threshold. If None, no logFC filter.
    pval_threshold
        Maximum adjusted p-value threshold. Default 0.05.
    key_prefix
        Prefix for obsm keys. Default: "X_deepscenic_"

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping cell type → list of active region names
    """
    import scanpy as sc

    from ._inference import to_latent

    # Get enhancer activity, computing if needed
    enh_act_key = f"{key_prefix}enh_act"
    if enh_act_key not in mdata.obsm:
        to_latent(model, mdata, key_prefix=key_prefix)
    enh_act = mdata.obsm[enh_act_key]

    # Build AnnData of per-cell enhancer activity
    ad_enh = sc.AnnData(
        X=enh_act,
        obs=mdata.obs[[celltype_key]].copy(),
    )
    ad_enh.var_names = list(model.region_names)

    # Run Wilcoxon rank-sum test
    sc.tl.rank_genes_groups(ad_enh, celltype_key, method="wilcoxon")

    # Extract active enhancers per cell type
    active_enhancers = {}
    for ct in ad_enh.obs[celltype_key].unique():
        df = sc.get.rank_genes_groups_df(ad_enh, group=ct)

        # Apply filters
        if pval_threshold is not None:
            df = df[df["pvals_adj"] < pval_threshold]
        if logfc_threshold is not None:
            df = df[df["logfoldchanges"] > logfc_threshold]
        if top_n is not None:
            df = df.head(top_n)

        active_enhancers[ct] = df["names"].tolist()
        logger.info(f"Cell type '{ct}': {len(active_enhancers[ct])} active enhancers")

    return active_enhancers


def compute_tf_activity_scores(
    model: DeepSCENICModel,
    mdata: md.MuData,
    celltype_key: str,
    active_enhancers: dict[str, list[str]],
    key_prefix: str = "X_deepscenic_",
) -> pd.DataFrame:
    """
    Compute TF activity scores weighted by binding to active enhancers.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    mdata
        MuData with RNA modality
    celltype_key
        Column in mdata.obs containing cell type labels
    active_enhancers
        Active enhancers per cell type from identify_active_enhancers()
    key_prefix
        Prefix for obsm keys. Default: "X_deepscenic_"

    Returns
    -------
    pd.DataFrame
        TF activity scores (n_tfs, n_celltypes)
        Index: TF names, Columns: cell type names

    Notes
    -----
    Score formula: mean(|TF_activity × E1[active_enhancers]|)
    """
    from ._inference import to_latent

    # Get TF activity from mdata.obsm, computing if needed
    z_tf_key = f"{key_prefix}z_tf"
    if z_tf_key not in mdata.obsm:
        to_latent(model, mdata, key_prefix=key_prefix)
    tf_act = mdata.obsm[z_tf_key]  # (n_cells, n_tfs)

    # Get E1 matrix
    with torch.no_grad():
        E1 = model.adj_tf2r.cpu().numpy()  # (n_regions, n_tfs)

    # Get cell type labels
    celltypes = mdata.obs[celltype_key]
    unique_celltypes = celltypes.unique()

    # Build region name to index mapping
    region_to_idx = {name: idx for idx, name in enumerate(model.region_names)}

    # Compute TF scores per cell type
    results = {}
    for ct in unique_celltypes:
        mask = celltypes == ct
        mean_tf_act = tf_act[mask.values].mean(axis=0)  # (n_tfs,)

        # Get active enhancer indices for this cell type
        active_regions = active_enhancers.get(ct, [])
        if not active_regions:
            results[ct] = np.zeros(len(model.tf_names))
            continue

        region_indices = [region_to_idx[r] for r in active_regions if r in region_to_idx]

        if not region_indices:
            results[ct] = np.zeros(len(model.tf_names))
            continue

        # E1 subset for active enhancers: (n_active, n_tfs)
        E1_active = E1[region_indices, :]

        # TF score = mean(|mean_tf_act × E1_active|) across enhancers
        # mean_tf_act: (n_tfs,), E1_active.T: (n_tfs, n_active)
        tf_scores = np.abs(mean_tf_act[:, None] * E1_active.T).mean(axis=1)
        results[ct] = tf_scores

    return pd.DataFrame(
        results,
        index=model.tf_names,  # type: ignore[arg-type]
    )


def identify_key_tfs(
    tf_activity_scores: pd.DataFrame,
    method: str = "otsu",
) -> list[str]:
    """
    Identify key TFs across all cell types using Otsu thresholding.

    Parameters
    ----------
    tf_activity_scores
        TF activity scores from compute_tf_activity_scores()
    method
        Thresholding method, currently only "otsu" supported

    Returns
    -------
    list[str]
        Unique list of key TF names across all cell types
    """
    if method != "otsu":
        raise ValueError(f"Unknown method: {method}. Only 'otsu' is supported.")

    key_tfs: set[str] = set()

    for celltype in tf_activity_scores.columns:
        scores = tf_activity_scores[celltype].abs()
        thresh, used_otsu = _get_otsu_threshold(scores.values, fallback=0.0)

        if used_otsu:
            logger.info(f"Cell type '{celltype}': Otsu TF threshold = {thresh:.4f}")
        else:
            logger.info(f"Cell type '{celltype}': Otsu failed, using threshold = {thresh:.4f}")

        ct_key_tfs = scores[scores > thresh].index.tolist()
        key_tfs.update(ct_key_tfs)

    return sorted(key_tfs)


def build_grn_for_tfs(
    model: DeepSCENICModel,
    tf_names: list[str],
    r2g_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Build GRN DataFrame for a specific set of TFs.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    tf_names
        List of TF names to include
    r2g_threshold
        Minimum r2g weight to include links. Default 0.0.

    Returns
    -------
    pd.DataFrame
        GRN with columns: TF, region, gene, tf2r_score, r2g_score, tf2g_score
    """
    _BUILD_GRN_COLUMNS = ["TF", "region", "gene", "tf2r_score", "r2g_score", "tf2g_score"]
    results = []

    for tf_name in tf_names:
        if tf_name not in model.tf_names:
            logger.warning(f"TF '{tf_name}' not found in model, skipping")
            continue

        # Get targets with per-TF Otsu threshold
        targets = get_tf_targets(model, tf_name, tf2r_threshold=None, r2g_threshold=r2g_threshold)

        if len(targets) > 0:
            # Rename columns to match legacy format
            targets = targets.rename(
                columns={
                    "tf": "TF",
                    "tf2r_weight": "tf2r_score",
                    "r2g_weight": "r2g_score",
                    "combined_weight": "tf2g_score",
                }
            )
            results.append(targets[["TF", "region", "gene", "tf2r_score", "r2g_score", "tf2g_score"]])

    if not results:
        return pd.DataFrame(columns=_BUILD_GRN_COLUMNS)

    return pd.concat(results, ignore_index=True)
