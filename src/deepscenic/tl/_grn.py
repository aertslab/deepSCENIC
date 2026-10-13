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
    # Group links per gene with a stable sort: each gene's weights stay in link
    # order, so every per-gene sum is computed exactly as on a boolean mask.
    order, starts, ends = _group_slices(r2g_indices[1])
    sorted_vals = r2g_vals[order]
    for start, end in zip(starts, ends, strict=True):
        total = sorted_vals[start:end].sum()
        if total > 0:
            normalized[order[start:end]] = sorted_vals[start:end] / total
    return normalized


def _group_slices(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable-sort ``keys`` and return the order plus start/end of each run of equal keys."""
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]) if len(keys) else np.array([], int)
    ends = np.r_[starts[1:], len(keys)].astype(int)
    return order, starts, ends


def _r2g_links(model: DeepSCENICModel, normalize: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return r2g link indices (2, n_links) and absolute (optionally per-gene normalized) weights."""
    with torch.no_grad():
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[union-attr]
        r2g_vals = model.vae.adj_r2g.abs().cpu().numpy()  # type: ignore[union-attr]
    if normalize:
        r2g_vals = _normalize_r2g_per_gene(r2g_vals, r2g_indices)
    return r2g_indices, r2g_vals


def _r2g_edgelist_for_genes(model: DeepSCENICModel, genes: list[str], normalize: bool) -> pd.DataFrame:
    """r2g edge list restricted to ``genes``, identical to filtering ``extract_r2g_matrix``'s edge list.

    Normalization is per gene, so normalizing only the kept links gives the same weights.
    """
    with torch.no_grad():
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[union-attr]
        r2g_vals = model.vae.adj_r2g.abs().cpu().numpy()  # type: ignore[union-attr]
    gene_names = _name_array(model.gene_names)
    keep = pd.Index(gene_names).isin(genes)[r2g_indices[1]]
    r2g_indices = r2g_indices[:, keep]
    r2g_vals = r2g_vals[keep]
    if normalize:
        r2g_vals = _normalize_r2g_per_gene(r2g_vals, r2g_indices)
    return pd.DataFrame(
        {
            "region": _name_array(model.region_names)[r2g_indices[0]],
            "gene": gene_names[r2g_indices[1]],
            "weight": r2g_vals,
        }
    )


def _name_array(names: list[str]) -> np.ndarray:
    """Names as an object array, so they can be gathered with integer indices."""
    return np.asarray(names, dtype=object)


def _links_of_regions(r2g_indices: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """Positions of links whose region is in ``region_mask``, ordered by region, then link order."""
    sel = np.flatnonzero(region_mask[r2g_indices[0]])
    return sel[np.argsort(r2g_indices[0, sel], kind="stable")]


def _r2g_dense(model: DeepSCENICModel, r2g_indices: np.ndarray, r2g_vals: np.ndarray) -> np.ndarray:
    """Dense (n_regions, n_genes) r2g matrix from link indices and weights."""
    r2g_sparse = torch.sparse_coo_tensor(
        torch.from_numpy(r2g_indices),
        torch.from_numpy(r2g_vals),
        size=(len(model.region_names), len(model.gene_names)),
    )
    return r2g_sparse.to_dense().numpy()


def _tf_edgelist(
    tf_by_x: np.ndarray,
    tf_names: list[str],
    x_names: list[str],
    x_col: str,
    value_col: str,
    mean_activity: np.ndarray,
    weighted_col: str,
) -> pd.DataFrame:
    """Long-form edge list of a TF × X matrix, weighted by mean TF activity.

    Rows follow ``DataFrame.melt`` order: all TFs for the first X, then the next X.
    """
    n_tfs, n_x = tf_by_x.shape
    values = tf_by_x.ravel(order="F")
    tf_activity = np.tile(mean_activity, n_x)
    return pd.DataFrame(
        {
            "tf": np.tile(_name_array(tf_names), n_x),
            x_col: np.repeat(_name_array(x_names), n_tfs),
            value_col: values,
            "tf_activity": tf_activity,
            weighted_col: tf_activity * values,
        }
    )


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
    r2g_indices, r2g_vals = _r2g_links(model, normalize)

    if as_edgelist:
        # Edge list format: (region, gene, weight) rows
        return pd.DataFrame(
            {
                "region": _name_array(model.region_names)[r2g_indices[0]],
                "gene": _name_array(model.gene_names)[r2g_indices[1]],
                "weight": r2g_vals,
            }
        )
    else:
        # Dense format using torch sparse tensor (faster than Python loop)
        # Warning: may be very large for real datasets
        return pd.DataFrame(
            _r2g_dense(model, r2g_indices, r2g_vals),
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
        raise ValueError(f"{tf_act_key!r} has {tf_activity.shape[1]} TFs, but the model has {len(model.tf_names)}")

    mean_activity = tf_activity[class_mask].mean(axis=0)
    tf2r = extract_tf2r_matrix(model)

    if not as_edgelist:
        return tf2r.mul(mean_activity, axis=0)

    return _tf_edgelist(
        tf2r.to_numpy(),
        model.tf_names,
        model.region_names,
        "region",
        "tf2r_weight",
        mean_activity,
        "celltype_tf2r_weight",
    )


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
            f"{enh_act_key!r} has {enhancer_activity.shape[1]} regions, but the model has {len(model.region_names)}"
        )

    mean_activity = enhancer_activity[class_mask].mean(axis=0)

    r2g_indices, r2g_vals = _r2g_links(model, normalize_r2g)

    if not as_edgelist:
        # Weight the dense matrix in place when the dtype allows it, saving a full copy
        r2g_dense = _r2g_dense(model, r2g_indices, r2g_vals)
        weights = mean_activity[:, None]
        if np.result_type(r2g_dense, weights) == r2g_dense.dtype:
            np.multiply(r2g_dense, weights, out=r2g_dense)
        else:
            r2g_dense = r2g_dense * weights
        return pd.DataFrame(
            r2g_dense,
            index=model.region_names,  # type: ignore[arg-type]
            columns=model.gene_names,  # type: ignore[arg-type]
        )

    enhancer_activity = mean_activity[r2g_indices[0]]
    return pd.DataFrame(
        {
            "region": _name_array(model.region_names)[r2g_indices[0]],
            "gene": _name_array(model.gene_names)[r2g_indices[1]],
            "r2g_weight": r2g_vals,
            "enhancer_activity": enhancer_activity,
            "celltype_r2g_weight": enhancer_activity * r2g_vals,
        }
    )


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
        raise ValueError(f"{tf_act_key!r} has {tf_activity.shape[1]} TFs, but the model has {len(model.tf_names)}")

    mean_activity = tf_activity[class_mask].mean(axis=0)
    from scipy.sparse import coo_matrix

    tf2r = extract_tf2r_matrix(model)
    r2g_indices, r2g_values = _r2g_links(model, normalize_r2g)
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

    if not as_edgelist:
        return tf2g.mul(mean_activity, axis=0)

    return _tf_edgelist(
        tf2g.to_numpy(), model.tf_names, model.gene_names, "gene", "tf2g_weight", mean_activity, "celltype_tf2g_weight"
    )


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

    r2g_indices, r2g_vals = _r2g_links(model, normalize_r2g)
    return _tf_targets(model, tf_name, tf2r_threshold, r2g_threshold, top_k, r2g_indices, r2g_vals)


def _tf_targets(
    model: DeepSCENICModel,
    tf_name: str,
    tf2r_threshold: float | None,
    r2g_threshold: float,
    top_k: int | None,
    r2g_indices: np.ndarray,
    r2g_vals: np.ndarray,
) -> pd.DataFrame:
    """Body of :func:`get_tf_targets`, on precomputed r2g links."""
    tf_idx = model.tf_names.index(tf_name)

    with torch.no_grad():
        # Get tf2r weights for this TF
        tf2r_tf = model.adj_tf2r[:, tf_idx].cpu().numpy()

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

    # Links from regions with tf2r weight above threshold (absolute values),
    # by region and then link order, keeping those above the r2g threshold
    links = _links_of_regions(r2g_indices, tf2r_tf_abs > tf2r_thresh)
    links = links[r2g_vals[links] > r2g_threshold]
    if len(links) == 0:
        return pd.DataFrame(columns=_TF_TARGETS_COLUMNS)

    region_idx = r2g_indices[0, links]
    tf2r_w = tf2r_tf[region_idx]
    r2g_w = r2g_vals[links]
    df = pd.DataFrame(
        {
            "tf": np.full(len(links), tf_name, dtype=object),
            "region": _name_array(model.region_names)[region_idx],
            "tf2r_weight": tf2r_w,
            "gene": _name_array(model.gene_names)[r2g_indices[1, links]],
            "r2g_weight": r2g_w,
            "combined_weight": tf2r_w * r2g_w,
        }
    )

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
    r2g_indices, r2g_vals = _r2g_links(model, normalize_r2g)

    # Find regions linked to this gene
    link_mask = r2g_indices[1] == gene_idx
    if not link_mask.any():
        return pd.DataFrame(columns=_GENE_REGULATORS_COLUMNS)

    region_indices = r2g_indices[0, link_mask]
    r2g_weights_for_gene = r2g_vals[link_mask]

    # Per-TF thresholds if using automatic thresholding, else the manual threshold
    if tf2r_threshold is None:
        # Otsu threshold for each TF (same as get_tf_targets). Cast to the weight dtype:
        # comparing a float32 weight with a Python float threshold happens in float32.
        thresholds = np.array(_tf2r_otsu_thresholds(tf2r_matrix), dtype=tf2r_matrix.dtype)
        logger.info(f"Gene '{gene_name}': using per-TF Otsu thresholds")
    else:
        thresholds = tf2r_threshold
        logger.info(f"Gene '{gene_name}': using manual tf2r threshold = {tf2r_threshold:.4f}")

    # Regions above the r2g threshold (written as "not <=" so NaN weights are kept, as before)
    keep = ~(r2g_weights_for_gene <= r2g_threshold)
    region_indices = region_indices[keep]
    r2g_weights_for_gene = r2g_weights_for_gene[keep]

    # (region, TF) pairs above threshold, by region and then TF order
    tf2r_sub = tf2r_matrix[region_indices]
    row, tf_idx = np.nonzero(np.abs(tf2r_sub) > thresholds)
    if len(row) == 0:
        return pd.DataFrame(columns=_GENE_REGULATORS_COLUMNS)

    tf2r_w = tf2r_sub[row, tf_idx]
    r2g_w = r2g_weights_for_gene[row]
    df = pd.DataFrame(
        {
            "gene": np.full(len(row), gene_name, dtype=object),
            "tf": _name_array(model.tf_names)[tf_idx],
            "region": _name_array(model.region_names)[region_indices[row]],
            "tf2r_weight": tf2r_w,
            "r2g_weight": r2g_w,
            "combined_weight": tf2r_w * r2g_w,
        }
    )

    # Sort by absolute combined_weight (strongest effects first, regardless of sign)
    df = df.assign(_abs_weight=df["combined_weight"].abs())
    df = df.sort_values("_abs_weight", ascending=False).drop(columns=["_abs_weight"])

    if top_k is not None:
        df = df.head(top_k)

    return df


def _tf2r_otsu_thresholds(tf2r_matrix: np.ndarray, chunk_bytes: int = 64 * 2**20) -> list[float]:
    """Otsu threshold of each TF's absolute tf2r weights (columns of ``tf2r_matrix``).

    Columns are copied to contiguous blocks of at most ``chunk_bytes``, which is much
    faster than strided column access on a (n_regions, n_tfs) matrix and gives the same values.
    """
    n_regions, n_tfs = tf2r_matrix.shape
    chunk_size = max(1, chunk_bytes // max(1, n_regions * tf2r_matrix.itemsize))
    thresholds = []
    for start in range(0, n_tfs, chunk_size):
        block = np.ascontiguousarray(np.abs(tf2r_matrix[:, start : start + chunk_size].T))  # (chunk, n_regions)
        for column in block:
            thresholds.append(_get_otsu_threshold(column, fallback=0.0)[0])
    return thresholds


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
    r2g_indices, r2g_vals = _r2g_links(model, normalize_r2g)

    # Links grouped per region (stable, so each region keeps its link order)
    link_order = np.argsort(r2g_indices[0], kind="stable")
    sorted_link_regions = r2g_indices[0, link_order]

    region_to_idx = {name: idx for idx, name in enumerate(model.region_names)}
    tf_names = _name_array(model.tf_names)
    gene_names = _name_array(model.gene_names)

    columns: dict[str, list] = {c: [] for c in _REGION_INFO_COLUMNS}
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
        lo, hi = np.searchsorted(sorted_link_regions, [region_idx, region_idx + 1])
        if lo == hi:
            # No r2g links: emit TF rows with no gene
            n_rows = len(tf_order)
            columns["region"] += [region_name] * n_rows
            columns["tf"] += tf_names[tf_order].tolist()
            columns["tf2r_weight"] += tf2r_region[tf_order].tolist()
            columns["gene"] += [None] * n_rows
            columns["r2g_weight"] += [None] * n_rows
            continue

        links = link_order[lo:hi]
        gene_idxs = r2g_indices[1, links]
        gene_r2g = r2g_vals[links]

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
        n_tfs, n_genes = len(tf_order), len(gene_idxs)
        columns["region"] += [region_name] * (n_tfs * n_genes)
        columns["tf"] += np.repeat(tf_names[tf_order], n_genes).tolist()
        columns["tf2r_weight"] += np.repeat(tf2r_region[tf_order], n_genes).tolist()
        columns["gene"] += np.tile(gene_names[gene_idxs], n_tfs).tolist()
        columns["r2g_weight"] += np.tile(gene_r2g, n_tfs).tolist()

    if not columns["region"]:
        return pd.DataFrame(columns=_REGION_INFO_COLUMNS)

    return pd.DataFrame(columns)


# =============================================================================
# Cell-Type Aware GRN Extraction
# =============================================================================


def identify_active_enhancers(
    model: DeepSCENICModel,
    mdata: md.MuData,
    class_key: str,
    top_n: int | None = 3000,
    min_mean_diff: float | None = None,
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
    min_mean_diff
        Minimum difference in mean enhancer activity between the cell type and
        all other cells. If None, no effect-size filter. Enhancer activities are
        signed model outputs, not log counts, so a mean difference is used
        instead of Scanpy's log-fold change, which is undefined for them.
    pval_threshold
        Maximum adjusted p-value threshold. Default 0.05.
    key_prefix
        Prefix for obsm keys. Default: ``"X_deepscenic_"``.

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

    # Run Wilcoxon rank-sum test. Scanpy's log-fold changes assume log counts and
    # are invalid (often NaN) for signed activities, so their warning is silenced.
    with np.errstate(invalid="ignore", divide="ignore"):
        sc.tl.rank_genes_groups(ad_enh, class_key, method="wilcoxon", use_raw=False)

    labels = ad_enh.obs[class_key].to_numpy()
    enh_act_dense = np.asarray(enh_act)

    # Extract active enhancers per cell type
    active_enhancers = {}
    for ct in ad_enh.obs[class_key].unique():
        df = sc.get.rank_genes_groups_df(ad_enh, group=ct)

        # Apply filters
        if pval_threshold is not None:
            df = df[df["pvals_adj"] < pval_threshold]
        if min_mean_diff is not None:
            in_group = labels == ct
            mean_diff = pd.Series(
                enh_act_dense[in_group].mean(axis=0) - enh_act_dense[~in_group].mean(axis=0),
                index=ad_enh.var_names,
            )
            df = df[mean_diff.loc[df["names"]].to_numpy() > min_mean_diff]
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
        Prefix for obsm keys. Default: ``"X_deepscenic_"``.
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
    r2g_indices, r2g_vals = _r2g_links(model, normalize_r2g)

    for tf_name in tf_names:
        if tf_name not in model.tf_names:
            logger.warning(f"TF '{tf_name}' not found in model, skipping")
            continue

        # Get targets with per-TF Otsu threshold
        targets = _tf_targets(model, tf_name, None, r2g_threshold, None, r2g_indices, r2g_vals)

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
