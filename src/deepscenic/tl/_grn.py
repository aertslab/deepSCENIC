"""GRN extraction functions for deepSCENIC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from ._model import DeepSCENICModel


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
    with torch.no_grad():
        # E1: TF → region (from adj_E1 cache)
        E1_values = model.adj_E1.cpu().numpy()

        E1_df = pd.DataFrame(
            E1_values,
            index=model.region_names,
            columns=model.tf_names,
        )

        # E2: region → gene (sparse)
        # Get sparse indices and weights from VAE
        r2g_indices = model.vae.r2g_indices.cpu().numpy()
        adj_E2 = model.vae.adj_E2.abs().cpu().numpy()

        E2_data = []
        for i in range(len(adj_E2)):
            region_idx = r2g_indices[0, i]
            gene_idx = r2g_indices[1, i]
            weight = adj_E2[i]

            E2_data.append({
                "region": model.region_names[region_idx],
                "gene": model.gene_names[gene_idx],
                "weight": weight,
            })

        E2_df = pd.DataFrame(E2_data)

    return {"E1": E1_df, "E2": E2_df}


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
        index=model.region_names,
        columns=model.tf_names,
    )


def extract_e2_matrix(
    model: DeepSCENICModel,
    as_sparse: bool = True,
) -> pd.DataFrame:
    """
    Extract E2 (region→gene) matrix.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    as_sparse
        If True, return sparse format (region, gene, weight).
        If False, return dense DataFrame.

    Returns
    -------
    pd.DataFrame
        E2 matrix in sparse or dense format
    """
    with torch.no_grad():
        r2g_indices = model.vae.r2g_indices.cpu().numpy()
        adj_E2 = model.vae.adj_E2.abs().cpu().numpy()

    if as_sparse:
        data = []
        for i in range(len(adj_E2)):
            region_idx = r2g_indices[0, i]
            gene_idx = r2g_indices[1, i]
            weight = adj_E2[i]
            data.append({
                "region": model.region_names[region_idx],
                "gene": model.gene_names[gene_idx],
                "weight": weight,
            })
        return pd.DataFrame(data)
    else:
        # Dense format (warning: may be very large)
        E2_dense = np.zeros((len(model.region_names), len(model.gene_names)))
        for i in range(len(adj_E2)):
            region_idx = r2g_indices[0, i]
            gene_idx = r2g_indices[1, i]
            E2_dense[region_idx, gene_idx] = adj_E2[i]

        return pd.DataFrame(
            E2_dense,
            index=model.region_names,
            columns=model.gene_names,
        )


def get_tf_targets(
    model: DeepSCENICModel,
    tf_name: str,
    threshold: float = 0.0,
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
    threshold
        Minimum weight threshold
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
        r2g_indices = model.vae.r2g_indices.cpu().numpy()
        adj_E2 = model.vae.adj_E2.abs().cpu().numpy()

    # Find regions with non-zero E1 weight
    results = []
    for region_idx in np.where(E1_tf > threshold)[0]:
        e1_weight = E1_tf[region_idx]
        region_name = model.region_names[region_idx]

        # Find genes linked to this region
        link_mask = r2g_indices[0] == region_idx
        if link_mask.any():
            gene_indices = r2g_indices[1, link_mask]
            e2_weights = adj_E2[link_mask]

            for gene_idx, e2_weight in zip(gene_indices, e2_weights, strict=False):
                if e2_weight > threshold:
                    results.append({
                        "tf": tf_name,
                        "region": region_name,
                        "E1_weight": e1_weight,
                        "gene": model.gene_names[gene_idx],
                        "E2_weight": e2_weight,
                        "combined_weight": e1_weight * e2_weight,
                    })

    df = pd.DataFrame(results)

    if len(df) > 0:
        df = df.sort_values("combined_weight", ascending=False)

        if top_k is not None:
            df = df.head(top_k)

    return df


def get_gene_regulators(
    model: DeepSCENICModel,
    gene_name: str,
    threshold: float = 0.0,
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
    threshold
        Minimum weight threshold
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
        r2g_indices = model.vae.r2g_indices.cpu().numpy()
        adj_E2 = model.vae.adj_E2.abs().cpu().numpy()

    # Find regions linked to this gene
    link_mask = r2g_indices[1] == gene_idx
    if not link_mask.any():
        return pd.DataFrame()

    region_indices = r2g_indices[0, link_mask]
    e2_weights = adj_E2[link_mask]

    results = []
    for region_idx, e2_weight in zip(region_indices, e2_weights, strict=False):
        if e2_weight <= threshold:
            continue

        region_name = model.region_names[region_idx]

        # Get TFs that bind this region
        for tf_idx in range(len(model.tf_names)):
            e1_weight = E1[region_idx, tf_idx]
            if e1_weight > threshold:
                results.append({
                    "gene": gene_name,
                    "tf": model.tf_names[tf_idx],
                    "region": region_name,
                    "E1_weight": e1_weight,
                    "E2_weight": e2_weight,
                    "combined_weight": e1_weight * e2_weight,
                })

    df = pd.DataFrame(results)

    if len(df) > 0:
        df = df.sort_values("combined_weight", ascending=False)

        if top_k is not None:
            df = df.head(top_k)

    return df
