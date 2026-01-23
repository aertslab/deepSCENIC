"""GRN extraction functions for deepSCENIC."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from skimage.filters import threshold_otsu

if TYPE_CHECKING:
    from ._model import DeepSCENICModel

logger = logging.getLogger(__name__)


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
        DataFrame with columns: region, E1_weight, gene, E2_weight, combined_weight
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

    df = pd.DataFrame(results)

    if len(df) > 0:
        df = df.sort_values("combined_weight", ascending=False)

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
        DataFrame with columns: tf, region, E1_weight, E2_weight, combined_weight
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
        return pd.DataFrame()

    region_indices = r2g_indices[0, link_mask]
    e2_weights_for_gene = adj_E2[link_mask]

    # Determine E1 threshold (Otsu by default, computed from linked regions using absolute values)
    if e1_threshold is None:
        E1_linked = np.abs(E1[region_indices, :]).ravel()
        e1_thresh, used_otsu = _get_otsu_threshold(E1_linked, fallback=0.0)
        if used_otsu:
            logger.info(f"Gene '{gene_name}': using Otsu E1 threshold = {e1_thresh:.4f}")
        else:
            logger.info(f"Gene '{gene_name}': Otsu failed, using E1 threshold = {e1_thresh:.4f}")
    else:
        e1_thresh = e1_threshold
        logger.info(f"Gene '{gene_name}': using manual E1 threshold = {e1_thresh:.4f}")

    results = []
    for region_idx, e2_weight in zip(region_indices, e2_weights_for_gene, strict=False):
        if e2_weight <= e2_threshold:
            continue

        region_name = model.region_names[region_idx]

        # Vectorized: Get all TF weights for this region at once (using absolute values)
        e1_weights_region = E1[region_idx, :]
        e1_weights_region_abs = np.abs(e1_weights_region)
        tf_mask = e1_weights_region_abs > e1_thresh
        tf_indices = np.where(tf_mask)[0]

        for tf_idx in tf_indices:
            e1_weight = e1_weights_region[tf_idx]  # Keep original sign for output
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

    df = pd.DataFrame(results)

    if len(df) > 0:
        df = df.sort_values("combined_weight", ascending=False)

        if top_k is not None:
            df = df.head(top_k)

    return df
