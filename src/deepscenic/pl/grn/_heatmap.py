"""GRN heatmap visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from deepscenic.tl._model import DeepSCENICModel

from .._utils import savefig_or_show, setup_axes


def heatmap_r2g(
    model: DeepSCENICModel,
    *,
    genes: list[str] | None = None,
    top_k: int | None = 50,
    normalize: bool = True,
    ax: Axes | None = None,
    cmap: str = "viridis",
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (10, 8),
    **kwargs,
) -> Axes | Figure | None:
    """
    Plot r2g (region→gene) heatmap.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    genes
        Genes to include.
    top_k
        Number of top genes/regions to show.
    normalize
        If True (default), normalize r2g weights per gene so they sum to 1.
        This makes the color scale represent fractional contributions in [0, 1],
        enabling cross-gene comparison.
    ax
        Pre-existing axes.
    cmap
        Colormap.
    show
        Display figure.
    save
        Save figure.
    return_fig
        Return Figure instead of Axes.
    figsize
        Figure size.
    **kwargs
        Passed to seaborn.heatmap.

    Returns
    -------
    Axes, Figure, or None depending on parameters.

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> ds.pl.heatmap_r2g(model, top_k=30)
    """
    from deepscenic.tl import extract_r2g_matrix

    # Get r2g as edge list, pivot to dense for selected genes
    E2_sparse = extract_r2g_matrix(model, as_edgelist=True, normalize=normalize)

    # Filter genes
    if genes is not None:
        E2_sparse = E2_sparse[E2_sparse["gene"].isin(genes)]
    elif top_k is not None:
        gene_weights = E2_sparse.groupby("gene")["weight"].sum()
        top_genes = gene_weights.nlargest(top_k).index.tolist()
        E2_sparse = E2_sparse[E2_sparse["gene"].isin(top_genes)]

    # Pivot to matrix
    E2_pivot = E2_sparse.pivot_table(index="region", columns="gene", values="weight", fill_value=0)

    # Limit regions
    if top_k is not None and len(E2_pivot) > top_k:
        region_weights = E2_pivot.sum(axis=1)
        top_regions = region_weights.nlargest(top_k).index.tolist()
        E2_pivot = E2_pivot.loc[top_regions]

    # Create plot
    fig, ax = setup_axes(ax, figsize=figsize)
    sns.heatmap(E2_pivot, ax=ax, cmap=cmap, **kwargs)
    ax.set_xlabel("Genes")
    ax.set_ylabel("Regions")
    title = "Region → Gene Links (r2g)"
    if normalize:
        title += " [normalized per gene]"
    ax.set_title(title)

    savefig_or_show("heatmap_r2g", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None


def heatmap_grn(
    model: DeepSCENICModel,
    *,
    tfs: list[str] | None = None,
    genes: list[str] | None = None,
    top_k: int | None = 30,
    normalize: str | None = "zscore",
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
    center: float = 0,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (12, 10),
    **kwargs,
) -> Axes | Figure | None:
    """
    Plot combined TF->gene GRN heatmap (E1 @ E2).

    Computes TF-gene regulatory weights by multiplying E1 (TF->region) and
    E2 (region->gene) matrices. By default, values are z-score normalized
    per TF to highlight gene-specific regulation patterns.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    tfs
        TFs to include.
    genes
        Genes to include.
    top_k
        Number of top TFs/genes to show.
    normalize
        Normalization method for the GRN matrix.
        ``'zscore'`` (default): Z-score per TF (row), matching legacy workflow.
        ``None``: Raw E1 @ E2 values.
    ax
        Pre-existing axes.
    cmap
        Colormap.
    center
        Center value for diverging colormap.
    show
        Display figure.
    save
        Save figure.
    return_fig
        Return Figure instead of Axes.
    figsize
        Figure size.
    **kwargs
        Passed to seaborn.heatmap.

    Returns
    -------
    Axes, Figure, or None depending on parameters.

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> ds.pl.heatmap_grn(model, top_k=20)
    """
    import numpy as np
    import pandas as pd
    import torch
    from scipy.sparse import coo_matrix

    from deepscenic.tl import extract_tf2r_matrix

    # Get E1 as dense DataFrame (n_regions, n_tfs)
    E1_df = extract_tf2r_matrix(model)

    # Build sparse E2 matrix using scipy (avoids dense materialization)
    with torch.no_grad():
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[union-attr]
        adj_r2g = model.vae.adj_r2g.abs().cpu().numpy()  # type: ignore[union-attr]

    E2_scipy = coo_matrix(
        (adj_r2g, (r2g_indices[0], r2g_indices[1])),
        shape=(len(model.region_names), len(model.gene_names)),
    ).tocsr()

    # Sparse matrix multiplication: GRN = E1.T @ E2 -> (n_tfs, n_genes)
    grn_values = E1_df.values.T @ E2_scipy

    grn_matrix = pd.DataFrame(
        grn_values.toarray() if hasattr(grn_values, "toarray") else np.asarray(grn_values),
        index=model.tf_names,
        columns=model.gene_names,
    )

    # Remove all-zero columns (genes with no E2 links)
    grn_matrix = grn_matrix.loc[:, (grn_matrix != 0).any(axis=0)]

    # Normalize
    if normalize == "zscore":
        # Z-score per TF (row): (value - row_mean) / row_std
        row_mean = grn_matrix.mean(axis=1)
        row_std = grn_matrix.std(axis=1)
        # Avoid division by zero for TFs with zero variance
        row_std = row_std.replace(0, np.nan)
        grn_matrix = grn_matrix.sub(row_mean, axis=0).div(row_std, axis=0)
        # Drop TFs with zero variance (all NaN after zscore)
        grn_matrix = grn_matrix.dropna(how="all")
    elif normalize is not None:
        raise ValueError(f"Unknown normalize method: {normalize!r}. Use 'zscore' or None.")

    # Filter TFs
    if tfs is not None:
        grn_matrix = grn_matrix.loc[[t for t in tfs if t in grn_matrix.index]]
    elif top_k is not None:
        tf_var = grn_matrix.var(axis=1)
        top_tfs = tf_var.nlargest(top_k).index.tolist()
        grn_matrix = grn_matrix.loc[top_tfs]

    # Filter genes
    if genes is not None:
        valid_genes = [g for g in genes if g in grn_matrix.columns]
        if valid_genes:
            grn_matrix = grn_matrix[valid_genes]
    elif top_k is not None:
        gene_var = grn_matrix.var()
        top_genes = gene_var.nlargest(top_k).index.tolist()
        grn_matrix = grn_matrix[top_genes]

    # Create plot
    fig, ax = setup_axes(ax, figsize=figsize)
    sns.heatmap(grn_matrix, ax=ax, cmap=cmap, center=center, **kwargs)
    ax.set_xlabel("Target Genes")
    ax.set_ylabel("Transcription Factors")
    title = "GRN: TF -> Gene Regulatory Weights"
    if normalize:
        title += f" ({normalize})"
    ax.set_title(title)

    savefig_or_show("heatmap_grn", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None
