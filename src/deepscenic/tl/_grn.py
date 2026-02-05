"""
GRN extraction functions for deepSCENIC.

This module provides functions to extract Gene Regulatory Networks (GRNs)
from trained deepSCENIC models.

Basic Extraction
----------------
- extract_grn: Extract E1 and E2 matrices as DataFrames
- extract_e1_matrix: Extract TF→region binding matrix
- extract_e2_matrix: Extract region→gene regulatory matrix

Query Functions
---------------
- get_tf_targets: Get target genes for a specific TF
- get_gene_regulators: Get TFs that regulate a specific gene

Cell-Type Aware Extraction
--------------------------
For cell-type specific GRN analysis, use these functions in order:

1. compute_celltype_enhancer_activity: Mean enhancer activity per cell type
2. identify_active_enhancers: Otsu threshold to find active enhancers
3. compute_tf_activity_scores: TF activity weighted by E1 to active enhancers
4. identify_key_tfs: Otsu threshold to identify key TFs
5. build_grn_for_tfs: Build GRN for selected TFs

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
>>> enh_activity = ds.tl.compute_celltype_enhancer_activity(model, mdata, "celltype")
>>> active_enh = ds.tl.identify_active_enhancers(enh_activity)
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
from skimage.filters import threshold_otsu

if TYPE_CHECKING:
    import mudata as md

    from ._model import DeepSCENICModel

logger = logging.getLogger(__name__)

# Column definitions for typed DataFrames
_TF_TARGETS_COLUMNS = ["tf", "region", "E1_weight", "gene", "E2_weight", "combined_weight"]
_GENE_REGULATORS_COLUMNS = ["gene", "tf", "region", "E1_weight", "E2_weight", "combined_weight"]


def _get_otsu_threshold(values: np.ndarray, fallback: float) -> tuple[float, bool]:
    """
    Compute Otsu threshold on positive values, with fallback on failure.

    Returns
    -------
    tuple[float, bool]
        (threshold_value, used_otsu) - the threshold and whether Otsu was used
    """
    positive_values = values[values > 0]
    if len(positive_values) > 1:
        try:
            return float(threshold_otsu(positive_values)), True
        except ValueError:
            # Otsu fails if all values are the same
            return fallback, False
    return fallback, False


def extract_grn(model: DeepSCENICModel) -> dict[str, pd.DataFrame]:
    """
    Extract GRN matrices as DataFrames with names.

    Parameters
    ----------
    model
        Trained DeepSCENICModel

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with:
        - 'E1': DataFrame (n_regions, n_tfs) with region/TF names
        - 'E2': DataFrame (sparse format) with region/gene links

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> grn = ds.tl.extract_grn(model)
    >>> E1 = grn['E1']  # TF → region weights
    >>> E2 = grn['E2']  # region → gene weights
    """
    return {
        "E1": extract_e1_matrix(model),
        "E2": extract_e2_matrix(model, as_edgelist=True),
    }


def extract_e1_matrix(model: DeepSCENICModel) -> pd.DataFrame:
    """
    Extract E1 (TF→region) matrix as DataFrame.

    Parameters
    ----------
    model
        Trained DeepSCENICModel

    Returns
    -------
    pd.DataFrame
        E1 matrix (n_regions, n_tfs) with region/TF names
    """
    with torch.no_grad():
        E1_values = model.adj_E1.cpu().numpy()

    return pd.DataFrame(
        E1_values,
        index=model.region_names,  # type: ignore[arg-type]
        columns=model.tf_names,  # type: ignore[arg-type]
    )


def extract_e2_matrix(
    model: DeepSCENICModel,
    as_edgelist: bool = True,
) -> pd.DataFrame:
    """
    Extract E2 (region→gene) matrix.

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
        E2 matrix in edge list or dense format
    """
    with torch.no_grad():
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[operator]
        adj_E2 = model.vae.adj_E2.abs().cpu().numpy()  # type: ignore[operator]

    if as_edgelist:
        # Edge list format: (region, gene, weight) rows
        return pd.DataFrame(
            {
                "region": [model.region_names[idx] for idx in r2g_indices[0]],
                "gene": [model.gene_names[idx] for idx in r2g_indices[1]],
                "weight": adj_E2,
            }
        )
    else:
        # Dense format using torch sparse tensor (faster than Python loop)
        # Warning: may be very large for real datasets
        E2_sparse = torch.sparse_coo_tensor(
            torch.from_numpy(r2g_indices),
            torch.from_numpy(adj_E2),
            size=(len(model.region_names), len(model.gene_names)),
        )
        E2_dense = E2_sparse.to_dense().numpy()

        return pd.DataFrame(
            E2_dense,
            index=model.region_names,  # type: ignore[arg-type]
            columns=model.gene_names,  # type: ignore[arg-type]
        )


def get_tf_targets(
    model: DeepSCENICModel,
    tf_name: str,
    e1_threshold: float | None = None,
    e2_threshold: float = 0.0,
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
    e1_threshold
        Threshold for E1 (TF→region) weights. If None (default), uses Otsu's
        method to automatically determine threshold.
    e2_threshold
        Threshold for E2 (region→gene) weights. Default 0.0.
    top_k
        If specified, return only top K targets

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: tf, region, E1_weight, gene, E2_weight, combined_weight

    Notes
    -----
    - E1 values can be positive or negative (activation/repression)
    - Thresholding is applied to absolute values
    - Results are sorted by absolute combined_weight (strongest effects first)
    - The original sign is preserved in E1_weight and combined_weight columns
    """
    if tf_name not in model.tf_names:
        raise ValueError(f"TF '{tf_name}' not found")

    tf_idx = model.tf_names.index(tf_name)

    with torch.no_grad():
        # Get E1 weights for this TF
        E1_tf = model.adj_E1[:, tf_idx].cpu().numpy()

        # Get E2 sparse info
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[operator]
        adj_E2 = model.vae.adj_E2.abs().cpu().numpy()  # type: ignore[operator]

    # Determine E1 threshold (Otsu by default, computed on absolute values)
    E1_tf_abs = np.abs(E1_tf)
    if e1_threshold is None:
        e1_thresh, used_otsu = _get_otsu_threshold(E1_tf_abs, fallback=0.0)
        if used_otsu:
            logger.info(f"TF '{tf_name}': using Otsu E1 threshold = {e1_thresh:.4f}")
        else:
            logger.info(f"TF '{tf_name}': Otsu failed, using E1 threshold = {e1_thresh:.4f}")
    else:
        e1_thresh = e1_threshold
        logger.info(f"TF '{tf_name}': using manual E1 threshold = {e1_thresh:.4f}")

    # Find regions with E1 weight above threshold (using absolute values)
    results = []
    for region_idx in np.where(E1_tf_abs > e1_thresh)[0]:
        e1_weight = E1_tf[region_idx]
        region_name = model.region_names[region_idx]

        # Find genes linked to this region
        link_mask = r2g_indices[0] == region_idx
        if link_mask.any():
            gene_indices = r2g_indices[1, link_mask]
            e2_weights = adj_E2[link_mask]

            for gene_idx, e2_weight in zip(gene_indices, e2_weights, strict=False):
                if e2_weight > e2_threshold:
                    results.append(
                        {
                            "tf": tf_name,
                            "region": region_name,
                            "E1_weight": e1_weight,
                            "gene": model.gene_names[gene_idx],
                            "E2_weight": e2_weight,
                            "combined_weight": e1_weight * e2_weight,
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
    e1_threshold: float | None = None,
    e2_threshold: float = 0.0,
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
    e1_threshold
        Threshold for E1 (TF→region) weights. If None (default), uses Otsu's
        method to automatically determine threshold from linked regions.
    e2_threshold
        Threshold for E2 (region→gene) weights. Default 0.0.
    top_k
        If specified, return only top K regulators

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gene, tf, region, E1_weight, E2_weight, combined_weight

    Notes
    -----
    - E1 values can be positive or negative (activation/repression)
    - Thresholding is applied to absolute values
    - Results are sorted by absolute combined_weight (strongest effects first)
    - The original sign is preserved in E1_weight and combined_weight columns
    """
    if gene_name not in model.gene_names:
        raise ValueError(f"Gene '{gene_name}' not found")

    gene_idx = model.gene_names.index(gene_name)

    with torch.no_grad():
        E1 = model.adj_E1.cpu().numpy()
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[operator]
        adj_E2 = model.vae.adj_E2.abs().cpu().numpy()  # type: ignore[operator]

    # Find regions linked to this gene
    link_mask = r2g_indices[1] == gene_idx
    if not link_mask.any():
        return pd.DataFrame(columns=_GENE_REGULATORS_COLUMNS)

    region_indices = r2g_indices[0, link_mask]
    e2_weights_for_gene = adj_E2[link_mask]

    # Pre-compute per-TF thresholds if using automatic thresholding
    if e1_threshold is None:
        # Compute Otsu threshold for each TF (same as get_tf_targets)
        tf_thresholds = {}
        for tf_idx in range(E1.shape[1]):
            E1_tf_abs = np.abs(E1[:, tf_idx])
            tf_thresh, _ = _get_otsu_threshold(E1_tf_abs, fallback=0.0)
            tf_thresholds[tf_idx] = tf_thresh
        logger.info(f"Gene '{gene_name}': using per-TF Otsu thresholds")
    else:
        tf_thresholds = None  # Use manual threshold for all TFs
        logger.info(f"Gene '{gene_name}': using manual E1 threshold = {e1_threshold:.4f}")

    results = []
    for region_idx, e2_weight in zip(region_indices, e2_weights_for_gene, strict=False):
        if e2_weight <= e2_threshold:
            continue

        region_name = model.region_names[region_idx]
        e1_weights_region = E1[region_idx, :]
        e1_weights_region_abs = np.abs(e1_weights_region)

        for tf_idx in range(len(model.tf_names)):
            # Use per-TF threshold or manual threshold
            thresh = tf_thresholds[tf_idx] if tf_thresholds else e1_threshold

            if e1_weights_region_abs[tf_idx] > thresh:
                e1_weight = e1_weights_region[tf_idx]
                results.append(
                    {
                        "gene": gene_name,
                        "tf": model.tf_names[tf_idx],
                        "region": region_name,
                        "E1_weight": e1_weight,
                        "E2_weight": e2_weight,
                        "combined_weight": e1_weight * e2_weight,
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


def compute_celltype_enhancer_activity(
    model: DeepSCENICModel,
    mdata: md.MuData,
    celltype_key: str,
) -> pd.DataFrame:
    """
    Compute mean enhancer activity per cell type.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    mdata
        MuData with RNA modality
    celltype_key
        Column in mdata.obs containing cell type labels

    Returns
    -------
    pd.DataFrame
        Mean enhancer activity (n_regions, n_celltypes)
        Index: region names, Columns: cell type names
    """
    from ._inference import to_latent

    # Get enhancer activity from VAE forward pass
    latent = to_latent(model, mdata)
    enh_act = latent["enh_act"]  # (n_cells, n_regions)

    # Get cell type labels
    celltypes = mdata.obs[celltype_key]
    unique_celltypes = celltypes.unique()

    # Compute mean activity per cell type
    results = {}
    for ct in unique_celltypes:
        mask = celltypes == ct
        results[ct] = enh_act[mask.values].mean(axis=0)

    return pd.DataFrame(
        results,
        index=model.region_names,  # type: ignore[arg-type]
    )


def identify_active_enhancers(
    enhancer_activity: pd.DataFrame,
    method: str = "otsu",
) -> dict[str, list[str]]:
    """
    Identify active enhancers per cell type using Otsu thresholding.

    Parameters
    ----------
    enhancer_activity
        Mean enhancer activity from compute_celltype_enhancer_activity()
    method
        Thresholding method, currently only "otsu" supported

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping cell type → list of active region names
    """
    if method != "otsu":
        raise ValueError(f"Unknown method: {method}. Only 'otsu' is supported.")

    active_enhancers = {}

    for celltype in enhancer_activity.columns:
        activity = enhancer_activity[celltype].abs()
        thresh, used_otsu = _get_otsu_threshold(activity.values, fallback=0.0)

        if used_otsu:
            logger.info(f"Cell type '{celltype}': Otsu enhancer threshold = {thresh:.4f}")
        else:
            logger.info(f"Cell type '{celltype}': Otsu failed, using threshold = {thresh:.4f}")

        active = activity[activity > thresh].index.tolist()
        active_enhancers[celltype] = active

    return active_enhancers


def compute_tf_activity_scores(
    model: DeepSCENICModel,
    mdata: md.MuData,
    celltype_key: str,
    active_enhancers: dict[str, list[str]],
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

    # Get TF activity from VAE forward pass
    latent = to_latent(model, mdata)
    tf_act = latent["z_tf"]  # (n_cells, n_tfs)

    # Get E1 matrix
    with torch.no_grad():
        E1 = model.adj_E1.cpu().numpy()  # (n_regions, n_tfs)

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
    e2_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Build GRN DataFrame for a specific set of TFs.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    tf_names
        List of TF names to include
    e2_threshold
        Minimum E2 weight to include links. Default 0.0.

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
        targets = get_tf_targets(model, tf_name, e1_threshold=None, e2_threshold=e2_threshold)

        if len(targets) > 0:
            # Rename columns to match legacy format
            targets = targets.rename(
                columns={
                    "tf": "TF",
                    "E1_weight": "tf2r_score",
                    "E2_weight": "r2g_score",
                    "combined_weight": "tf2g_score",
                }
            )
            results.append(targets[["TF", "region", "gene", "tf2r_score", "r2g_score", "tf2g_score"]])

    if not results:
        return pd.DataFrame(columns=_BUILD_GRN_COLUMNS)

    return pd.concat(results, ignore_index=True)
