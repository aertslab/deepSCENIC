"""
GRN extraction functions for deepSCENIC.

This module provides functions to extract Gene Regulatory Networks (GRNs)
from trained deepSCENIC models.

Basic Extraction
----------------
- extract_grn: Extract tf2r and r2g matrices as DataFrames
- extract_tf2r_matrix: Extract TF→region binding matrix
- extract_r2g_matrix: Extract region→gene regulatory matrix
- compute_celltype_tf2r: Weight tf2r links by mean TF activity in a cell class
- compute_celltype_r2g: Weight r2g links by mean enhancer activity in a cell class
- compute_celltype_tf2g: Weight combined tf2g links by mean TF activity in a cell class

Query Functions
---------------
- get_tf_targets: Get target genes for a specific TF
- get_gene_regulators: Get TFs that regulate a specific gene
- get_region_info: Get top active TFs and target genes for a set of regions

Cell-Type Aware Extraction
--------------------------
For cell-type specific GRN analysis, use these functions in order:

1. identify_active_enhancers: Wilcoxon differential analysis to find active enhancers
2. compute_tf_activity_scores: TF activity weighted by tf2r to active enhancers
3. build_grn_for_tfs: Build GRN for user-selected TFs

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
>>> active_enh = ds.tl.identify_active_enhancers(model, mdata, class_key="celltype")
>>> tf_scores = ds.tl.compute_tf_activity_scores(
...     model, mdata, class_key="celltype", active_enhancers=active_enh
... )
>>> selected_tfs = ["SOX10", "MITF"]
>>> grn_df = ds.tl.build_grn_for_tfs(model, selected_tfs)
>>>
>>> # Region-centric query
>>> region_info = ds.tl.get_region_info(model, "chr3:69926370-69926870")
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
_REGION_INFO_COLUMNS = ["region", "tf", "tf2r_weight", "gene", "r2g_weight"]


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


def _normalize_r2g_per_gene(
    r2g_vals: np.ndarray,
    r2g_indices: np.ndarray,
) -> np.ndarray:
    """Normalize r2g weights per gene so each gene's weights sum to 1.

    Parameters
    ----------
    r2g_vals
        Absolute r2g weights, shape (n_links,)
    r2g_indices
        Sparse indices, shape (2, n_links). Row 0 = region indices, row 1 = gene indices.

    Returns
    -------
    np.ndarray
        Normalized weights, shape (n_links,). Each gene's weights sum to 1,
        making them interpretable as fractional contributions of each region
        to that gene's regulatory signal.
    """
    normalized = r2g_vals.copy()
    gene_indices = r2g_indices[1]
    unique_genes = np.unique(gene_indices)
    for gene_idx in unique_genes:
        mask = gene_indices == gene_idx
        total = r2g_vals[mask].sum()
        if total > 0:
            normalized[mask] = r2g_vals[mask] / total
    return normalized


def extract_grn(model: DeepSCENICModel, normalize_r2g: bool = True) -> dict[str, pd.DataFrame]:
    """
    Extract GRN matrices (tf2r and r2g) as DataFrames with names.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    normalize_r2g
        If True (default), normalize r2g weights per gene so they sum to 1.
        This makes weights comparable across genes (fractional contributions).

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with:
        - 'tf2r': DataFrame (n_tfs, n_regions) with TF/region names
        - 'r2g': DataFrame (sparse format) with region/gene links

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> grn = ds.tl.extract_grn(model)
    >>> tf2r = grn['tf2r']  # TF → region weights
    >>> r2g = grn['r2g']  # region → gene weights (normalized per gene)
    """
    return {
        "tf2r": extract_tf2r_matrix(model),
        "r2g": extract_r2g_matrix(model, as_edgelist=True, normalize=normalize_r2g),
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
        tf2r matrix (n_tfs, n_regions) with TF/region names
    """
    with torch.no_grad():
        E1_values = model.adj_tf2r.cpu().numpy()

    return pd.DataFrame(
        E1_values.T,
        index=model.tf_names,  # type: ignore[arg-type]
        columns=model.region_names,  # type: ignore[arg-type]
    )


def extract_r2g_matrix(
    model: DeepSCENICModel,
    as_edgelist: bool = True,
    normalize: bool = True,
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
    normalize
        If True (default), normalize r2g weights per gene so they sum to 1.
        This makes weights comparable across genes — each weight represents the
        fractional contribution of that region to the gene's regulatory signal,
        in [0, 1] range.

    Returns
    -------
    pd.DataFrame
        r2g matrix in edge list or dense format
    """
    with torch.no_grad():
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[operator]
        r2g_vals = model.vae.adj_r2g.abs().cpu().numpy()  # type: ignore[operator]

    if normalize:
        r2g_vals = _normalize_r2g_per_gene(r2g_vals, r2g_indices)

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


def compute_celltype_tf2r(
    model: DeepSCENICModel,
    mdata: md.MuData,
    class_key: str,
    class_label: str,
    *,
    as_edgelist: bool = False,
    key_prefix: str = "X_deepscenic_",
) -> pd.DataFrame:
    """
    Compute TF-activity-weighted tf2r scores for one cell class.

    This reproduces the legacy calculation
    ``mean_tf_activity[:, None] * E1``. TF activity is averaged over cells in
    ``class_label`` and multiplied into each corresponding TF→region weight.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    mdata
        MuData containing the cells to aggregate.
    class_key
        Column in ``mdata.obs`` containing class labels.
    class_label
        Class whose mean TF activity should be used.
    as_edgelist
        If False (default), return the legacy-compatible dense TF × region
        matrix. If True, return a long-form edge list.
    key_prefix
        Prefix for model-derived values in ``mdata.obsm``. Default
        ``"X_deepscenic_"``.

    Returns
    -------
    pd.DataFrame
        If ``as_edgelist=False``, a TF × region matrix of activity-weighted
        tf2r scores. Otherwise, an edge list with columns ``tf``, ``region``,
        ``tf2r_weight``, ``tf_activity``, and ``celltype_tf2r_weight``.

    Examples
    --------
    >>> oligo_tf2r = ds.tl.compute_celltype_tf2r(
    ...     model,
    ...     mdata,
    ...     class_key="subclass_Bakken_2022",
    ...     class_label="Oligo",
    ... )
    """
    from ._inference import to_latent

    if class_key not in mdata.obs:
        raise KeyError(f"Class key {class_key!r} not found in mdata.obs")

    class_mask = mdata.obs[class_key].eq(class_label).to_numpy()
    if not class_mask.any():
        raise ValueError(f"No cells found with {class_key}={class_label!r}")

    tf_act_key = f"{key_prefix}z_tf"
    if tf_act_key not in mdata.obsm:
        to_latent(model, mdata, key_prefix=key_prefix)

    tf_activity = np.asarray(mdata.obsm[tf_act_key])
    if tf_activity.shape[1] != len(model.tf_names):
        raise ValueError(
            f"{tf_act_key!r} has {tf_activity.shape[1]} TFs, "
            f"but the model has {len(model.tf_names)}"
        )

    mean_activity = tf_activity[class_mask].mean(axis=0)
    tf2r = extract_tf2r_matrix(model)
    weighted_tf2r = tf2r.mul(mean_activity, axis=0)

    if not as_edgelist:
        return weighted_tf2r

    result = tf2r.rename_axis("tf").reset_index().melt(
        id_vars="tf",
        var_name="region",
        value_name="tf2r_weight",
    )
    tf_to_activity = dict(zip(model.tf_names, mean_activity, strict=True))
    result["tf_activity"] = result["tf"].map(tf_to_activity)
    result["celltype_tf2r_weight"] = result["tf_activity"] * result["tf2r_weight"]
    return result


def compute_celltype_r2g(
    model: DeepSCENICModel,
    mdata: md.MuData,
    class_key: str,
    class_label: str,
    *,
    as_edgelist: bool = False,
    normalize_r2g: bool = False,
    key_prefix: str = "X_deepscenic_",
) -> pd.DataFrame:
    """
    Compute enhancer-activity-weighted r2g scores for one cell class.

    This reproduces the legacy calculation
    ``mean_enhancer_activity[:, None] * E2``. Enhancer activity is averaged
    over cells in ``class_label`` and multiplied into each corresponding
    region→gene weight.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    mdata
        MuData containing the cells to aggregate.
    class_key
        Column in ``mdata.obs`` containing class labels.
    class_label
        Class whose mean enhancer activity should be used.
    as_edgelist
        If False (default), return the legacy-compatible dense region × gene
        matrix. If True, return only modeled r2g links as an edge list, which
        is more memory efficient for large datasets.
    normalize_r2g
        If True, normalize r2g weights per gene before weighting them by
        enhancer activity. Default False, matching the legacy raw E2 matrix.
    key_prefix
        Prefix for model-derived values in ``mdata.obsm``. Default
        ``"X_deepscenic_"``.

    Returns
    -------
    pd.DataFrame
        If ``as_edgelist=False``, a region × gene matrix of activity-weighted
        r2g scores. Otherwise, an edge list with columns ``region``, ``gene``,
        ``r2g_weight``, ``enhancer_activity``, and ``celltype_r2g_weight``.

    Examples
    --------
    >>> vip_r2g = ds.tl.compute_celltype_r2g(
    ...     model,
    ...     mdata,
    ...     class_key="subclass_Bakken_2022",
    ...     class_label="Vip",
    ... )
    """
    from ._inference import to_latent

    if class_key not in mdata.obs:
        raise KeyError(f"Class key {class_key!r} not found in mdata.obs")

    class_mask = mdata.obs[class_key].eq(class_label).to_numpy()
    if not class_mask.any():
        raise ValueError(f"No cells found with {class_key}={class_label!r}")

    enh_act_key = f"{key_prefix}enh_act"
    if enh_act_key not in mdata.obsm:
        to_latent(model, mdata, key_prefix=key_prefix)

    enhancer_activity = np.asarray(mdata.obsm[enh_act_key])
    if enhancer_activity.shape[1] != len(model.region_names):
        raise ValueError(
            f"{enh_act_key!r} has {enhancer_activity.shape[1]} regions, "
            f"but the model has {len(model.region_names)}"
        )

    mean_activity = enhancer_activity[class_mask].mean(axis=0)

    r2g = extract_r2g_matrix(
        model,
        as_edgelist=as_edgelist,
        normalize=normalize_r2g,
    )

    if not as_edgelist:
        return r2g.mul(mean_activity, axis=0)

    region_to_activity = dict(zip(model.region_names, mean_activity, strict=True))
    r2g = r2g.rename(columns={"weight": "r2g_weight"})
    r2g["enhancer_activity"] = r2g["region"].map(region_to_activity)
    r2g["celltype_r2g_weight"] = r2g["enhancer_activity"] * r2g["r2g_weight"]
    return r2g


def compute_celltype_tf2g(
    model: DeepSCENICModel,
    mdata: md.MuData,
    class_key: str,
    class_label: str,
    *,
    as_edgelist: bool = False,
    normalize_r2g: bool = False,
    key_prefix: str = "X_deepscenic_",
) -> pd.DataFrame:
    """
    Compute TF-activity-weighted tf2g scores for one cell class.

    This reproduces the legacy calculation
    ``mean_tf_activity[:, None] * (E1 @ E2)``. The TF→gene matrix is obtained
    by multiplying the TF × region tf2r matrix by the region × gene r2g
    matrix, then weighting each TF row by its mean activity in ``class_label``.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    mdata
        MuData containing the cells to aggregate.
    class_key
        Column in ``mdata.obs`` containing class labels.
    class_label
        Class whose mean TF activity should be used.
    as_edgelist
        If False (default), return the legacy-compatible dense TF × gene
        matrix. If True, return a long-form edge list.
    normalize_r2g
        If True, normalize r2g weights per gene before computing E1 @ E2.
        Default False, matching the legacy raw E2 matrix.
    key_prefix
        Prefix for model-derived values in ``mdata.obsm``. Default
        ``"X_deepscenic_"``.

    Returns
    -------
    pd.DataFrame
        If ``as_edgelist=False``, a TF × gene matrix of activity-weighted tf2g
        scores. Otherwise, an edge list with columns ``tf``, ``gene``,
        ``tf2g_weight``, ``tf_activity``, and ``celltype_tf2g_weight``.

    Examples
    --------
    >>> oligo_tf2g = ds.tl.compute_celltype_tf2g(
    ...     model,
    ...     mdata,
    ...     class_key="subclass_Bakken_2022",
    ...     class_label="Oligo",
    ... )
    """
    from ._inference import to_latent

    if class_key not in mdata.obs:
        raise KeyError(f"Class key {class_key!r} not found in mdata.obs")

    class_mask = mdata.obs[class_key].eq(class_label).to_numpy()
    if not class_mask.any():
        raise ValueError(f"No cells found with {class_key}={class_label!r}")

    tf_act_key = f"{key_prefix}z_tf"
    if tf_act_key not in mdata.obsm:
        to_latent(model, mdata, key_prefix=key_prefix)

    tf_activity = np.asarray(mdata.obsm[tf_act_key])
    if tf_activity.shape[1] != len(model.tf_names):
        raise ValueError(
            f"{tf_act_key!r} has {tf_activity.shape[1]} TFs, "
            f"but the model has {len(model.tf_names)}"
        )

    mean_activity = tf_activity[class_mask].mean(axis=0)
    from scipy.sparse import coo_matrix

    tf2r = extract_tf2r_matrix(model)
    with torch.no_grad():
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[union-attr]
        r2g_values = model.vae.adj_r2g.abs().cpu().numpy()  # type: ignore[union-attr]
    if normalize_r2g:
        r2g_values = _normalize_r2g_per_gene(r2g_values, r2g_indices)
    r2g = coo_matrix(
        (r2g_values, (r2g_indices[0], r2g_indices[1])),
        shape=(len(model.region_names), len(model.gene_names)),
    ).tocsr()
    tf2g_values = tf2r.to_numpy() @ r2g
    tf2g = pd.DataFrame(
        tf2g_values.toarray() if hasattr(tf2g_values, "toarray") else np.asarray(tf2g_values),
        index=model.tf_names,
        columns=model.gene_names,
    )
    weighted_tf2g = tf2g.mul(mean_activity, axis=0)

    if not as_edgelist:
        return weighted_tf2g

    result = tf2g.rename_axis("tf").reset_index().melt(
        id_vars="tf",
        var_name="gene",
        value_name="tf2g_weight",
    )
    tf_to_activity = dict(zip(model.tf_names, mean_activity, strict=True))
    result["tf_activity"] = result["tf"].map(tf_to_activity)
    result["celltype_tf2g_weight"] = result["tf_activity"] * result["tf2g_weight"]
    return result


def get_tf_targets(
    model: DeepSCENICModel,
    tf_name: str,
    tf2r_threshold: float | None = None,
    r2g_threshold: float = 0.0,
    top_k: int | None = None,
    normalize_r2g: bool = True,
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
        Threshold for r2g (region→gene) weights. Default 0.0. When
        ``normalize_r2g=True``, this is in [0, 1] (fractional contribution).
    top_k
        If specified, return only top K targets
    normalize_r2g
        If True (default), normalize r2g weights per gene so they sum to 1.
        This makes weights comparable across genes and ``r2g_threshold``
        interpretable as a minimum fractional contribution.

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

    if normalize_r2g:
        r2g_vals = _normalize_r2g_per_gene(r2g_vals, r2g_indices)

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
    normalize_r2g: bool = True,
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
        Threshold for r2g (region→gene) weights. Default 0.0. When
        ``normalize_r2g=True``, this is in [0, 1] (fractional contribution).
    top_k
        If specified, return only top K regulators
    normalize_r2g
        If True (default), normalize r2g weights per gene so they sum to 1.
        This makes weights comparable across genes and ``r2g_threshold``
        interpretable as a minimum fractional contribution.

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

    if normalize_r2g:
        r2g_vals = _normalize_r2g_per_gene(r2g_vals, r2g_indices)

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


def get_region_info(
    model: DeepSCENICModel,
    regions: str | list[str],
    tf2r_threshold: float | None = None,
    r2g_threshold: float = 0.0,
    top_k_tfs: int | None = 10,
    top_k_genes: int | None = 10,
    normalize_r2g: bool = True,
) -> pd.DataFrame:
    """
    Get top active TFs and target genes for a set of regions.

    For each queried region, returns the TFs with the highest binding potential
    (from tf2r) and the target genes linked via r2g. Each row represents one
    TF→region→gene path.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    regions
        Region name(s) to query (e.g., ``"chr1:1000-2000"`` or a list of names)
    tf2r_threshold
        Threshold for tf2r (TF→region) weights (absolute value). If None
        (default), uses Otsu's method per region to automatically determine
        the threshold.
    r2g_threshold
        Threshold for r2g (region→gene) weights. Default 0.0. When
        ``normalize_r2g=True``, this is in [0, 1] (fractional contribution).
    top_k_tfs
        Maximum number of TFs to return per region, ranked by absolute
        tf2r weight. Default 10. Set to None for no limit.
    top_k_genes
        Maximum number of target genes to return per region, ranked by r2g
        weight. Default 10. Set to None for no limit.
    normalize_r2g
        If True (default), normalize r2g weights per gene so they sum to 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: region, tf, tf2r_weight, gene, r2g_weight.
        Each row is one TF→region→gene path.

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> region_info = ds.tl.get_region_info(model, "chr3:69926370-69926870")
    >>> # Query multiple regions
    >>> region_info = ds.tl.get_region_info(model, ["chr3:69926370-69926870", "chr9:6781416-6781916"])
    """
    if isinstance(regions, str):
        regions = [regions]

    unknown = [r for r in regions if r not in model.region_names]
    if unknown:
        raise ValueError(f"Regions not found in model: {unknown}")

    with torch.no_grad():
        tf2r_matrix = model.adj_tf2r.cpu().numpy()  # (n_regions, n_tfs)
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[operator]
        r2g_vals = model.vae.adj_r2g.abs().cpu().numpy()  # type: ignore[operator]

    if normalize_r2g:
        r2g_vals = _normalize_r2g_per_gene(r2g_vals, r2g_indices)

    region_to_idx = {name: idx for idx, name in enumerate(model.region_names)}

    results = []
    for region_name in regions:
        region_idx = region_to_idx[region_name]

        # --- Top TFs for this region ---
        tf2r_region = tf2r_matrix[region_idx, :]  # (n_tfs,)
        tf2r_region_abs = np.abs(tf2r_region)

        # Determine tf2r threshold
        if tf2r_threshold is None:
            thresh, used_otsu = _get_otsu_threshold(tf2r_region_abs, fallback=0.0)
            if used_otsu:
                logger.debug(f"Region '{region_name}': Otsu tf2r threshold = {thresh:.4f}")
        else:
            thresh = tf2r_threshold

        # Select TFs above threshold
        tf_mask = tf2r_region_abs > thresh
        tf_idxs = np.where(tf_mask)[0]

        # Sort by descending absolute weight and apply top_k
        tf_order = tf_idxs[np.argsort(tf2r_region_abs[tf_idxs])[::-1]]
        if top_k_tfs is not None:
            tf_order = tf_order[:top_k_tfs]

        # --- Target genes for this region ---
        link_mask = r2g_indices[0] == region_idx
        if not link_mask.any():
            # No r2g links: emit TF rows with no gene
            for tf_idx in tf_order:
                results.append(
                    {
                        "region": region_name,
                        "tf": model.tf_names[tf_idx],
                        "tf2r_weight": float(tf2r_region[tf_idx]),
                        "gene": None,
                        "r2g_weight": None,
                    }
                )
            continue

        gene_idxs = r2g_indices[1, link_mask]
        gene_r2g = r2g_vals[link_mask]

        # Filter by r2g threshold
        gene_mask = gene_r2g > r2g_threshold
        gene_idxs = gene_idxs[gene_mask]
        gene_r2g = gene_r2g[gene_mask]

        # Sort by descending r2g weight and apply top_k
        gene_order = np.argsort(gene_r2g)[::-1]
        if top_k_genes is not None:
            gene_order = gene_order[:top_k_genes]
        gene_idxs = gene_idxs[gene_order]
        gene_r2g = gene_r2g[gene_order]

        # Cross-product: each TF × each gene
        for tf_idx in tf_order:
            for g_idx, g_r2g in zip(gene_idxs, gene_r2g, strict=False):
                results.append(
                    {
                        "region": region_name,
                        "tf": model.tf_names[tf_idx],
                        "tf2r_weight": float(tf2r_region[tf_idx]),
                        "gene": model.gene_names[g_idx],
                        "r2g_weight": float(g_r2g),
                    }
                )

    if not results:
        return pd.DataFrame(columns=_REGION_INFO_COLUMNS)

    return pd.DataFrame(results)


# =============================================================================
# Cell-Type Aware GRN Extraction
# =============================================================================


def identify_active_enhancers(
    model: DeepSCENICModel,
    mdata: md.MuData,
    class_key: str,
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
    class_key
        Column in mdata.obs containing class labels
    top_n
        Maximum number of active enhancers per cell type.
        Applied after logfc/pval filtering. If None, no limit. Default 3000.
    logfc_threshold
        Minimum Scanpy log-fold-change threshold. If None, no logFC filter.
        Model-derived enhancer activities can be negative, so Scanpy may report
        undefined log-fold changes for some regions. The Wilcoxon scores and
        p-values remain valid because they are rank-based.
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

    # Build AnnData from the original signed, model-derived enhancer activity.
    ad_enh = sc.AnnData(
        X=enh_act,
        obs=mdata.obs[[class_key]].copy(),
    )
    ad_enh.var_names = list(model.region_names)

    # Run Wilcoxon rank-sum test
    sc.tl.rank_genes_groups(ad_enh, class_key, method="wilcoxon", use_raw=False)

    # Extract active enhancers per cell type
    active_enhancers = {}
    for ct in ad_enh.obs[class_key].unique():
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
    class_key: str,
    active_enhancers: dict[str, list[str]],
    key_prefix: str = "X_deepscenic_",
    zscore: bool = False,
) -> pd.DataFrame:
    """
    Compute TF activity scores weighted by binding to active enhancers.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    mdata
        MuData with RNA modality
    class_key
        Column in mdata.obs containing class labels
    active_enhancers
        Active enhancers per cell type from identify_active_enhancers()
    key_prefix
        Prefix for obsm keys. Default: "X_deepscenic_"
    zscore
        If True, z-score each TF across classes. Each TF row is centered to
        mean zero and scaled to unit population standard deviation. TFs with
        zero variance across classes are set to zero. Default: False.

    Returns
    -------
    pd.DataFrame
        TF activity scores (n_tfs, n_celltypes)
        Index: TF names, Columns: cell type names

    Notes
    -----
    Score formula: mean(TF_activity × E1[active_enhancers])

    Scores retain their sign: positive and negative values represent the signed
    contribution implied by TF activity and tf2r weights.
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
    celltypes = mdata.obs[class_key]
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

        # TF score = mean(mean_tf_act × E1_active) across enhancers
        # mean_tf_act: (n_tfs,), E1_active.T: (n_tfs, n_active)
        tf_scores = (mean_tf_act[:, None] * E1_active.T).mean(axis=1)
        results[ct] = tf_scores

    scores = pd.DataFrame(
        results,
        index=model.tf_names,  # type: ignore[arg-type]
    )

    if zscore:
        row_mean = scores.mean(axis=1)
        row_std = scores.std(axis=1, ddof=0).replace(0, np.nan)
        scores = scores.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0.0)

    return scores


def build_grn_for_tfs(
    model: DeepSCENICModel,
    tf_names: list[str],
    r2g_threshold: float = 0.0,
    normalize_r2g: bool = True,
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
        Minimum r2g weight to include links. Default 0.0. When
        ``normalize_r2g=True``, this is in [0, 1] (fractional contribution).
    normalize_r2g
        If True (default), normalize r2g weights per gene so they sum to 1.

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
        targets = get_tf_targets(
            model, tf_name, tf2r_threshold=None, r2g_threshold=r2g_threshold, normalize_r2g=normalize_r2g
        )

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
