"""GRN heatmap visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from deepscenic.tl._model import DeepSCENICModel

from .._utils import savefig_or_show, setup_axes


def heatmap_e1(
    model: DeepSCENICModel,
    *,
    tfs: list[str] | None = None,
    regions: list[str] | None = None,
    top_k: int | None = 50,
    ax: Axes | None = None,
    cmap: str = "viridis",
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (10, 8),
    **kwargs,
) -> Axes | Figure | None:
    """
    Plot E1 (TF->region) heatmap.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    tfs
        TFs to include. If None, use top by variance.
    regions
        Regions to include. If None, use top by variance.
    top_k
        Number of top TFs/regions to show if not specified.
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
    >>> ds.pl.heatmap_e1(model, top_k=30)
    """
    from deepscenic.tl import extract_e1_matrix

    # Get E1 matrix
    E1_df = extract_e1_matrix(model)

    # Filter TFs
    if tfs is not None:
        E1_df = E1_df[tfs]
    elif top_k is not None:
        tf_var = E1_df.var()
        top_tfs = tf_var.nlargest(top_k).index.tolist()
        E1_df = E1_df[top_tfs]

    # Filter regions
    if regions is not None:
        E1_df = E1_df.loc[regions]
    elif top_k is not None:
        region_var = E1_df.var(axis=1)
        top_regions = region_var.nlargest(top_k).index.tolist()
        E1_df = E1_df.loc[top_regions]

    # Create plot
    fig, ax = setup_axes(ax, figsize=figsize)
    sns.heatmap(E1_df, ax=ax, cmap=cmap, **kwargs)
    ax.set_xlabel("Transcription Factors")
    ax.set_ylabel("Regions")
    ax.set_title("E1: TF -> Region Binding")

    savefig_or_show("heatmap_e1", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None


def heatmap_e2(
    model: DeepSCENICModel,
    *,
    genes: list[str] | None = None,
    top_k: int | None = 50,
    ax: Axes | None = None,
    cmap: str = "viridis",
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (10, 8),
    **kwargs,
) -> Axes | Figure | None:
    """
    Plot E2 (region->gene) heatmap.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    genes
        Genes to include.
    top_k
        Number of top genes/regions to show.
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
    >>> ds.pl.heatmap_e2(model, top_k=30)
    """
    from deepscenic.tl import extract_e2_matrix

    # Get E2 as sparse df, pivot to dense for selected genes
    E2_sparse = extract_e2_matrix(model, as_sparse=True)

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
    ax.set_title("E2: Region -> Gene Links")

    savefig_or_show("heatmap_e2", show=show, save=save)

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
    from deepscenic.tl import extract_e1_matrix, extract_e2_matrix

    # Get matrices
    E1_df = extract_e1_matrix(model)
    E2_sparse = extract_e2_matrix(model, as_sparse=True)

    # Compute TF->gene via matrix multiplication
    # First need dense E2
    E2_pivot = E2_sparse.pivot_table(index="region", columns="gene", values="weight", fill_value=0)
    # Align indices
    common_regions = E1_df.index.intersection(E2_pivot.index)
    E1_aligned = E1_df.loc[common_regions]
    E2_aligned = E2_pivot.loc[common_regions]

    # TF->gene = E1.T @ E2
    grn_matrix = E1_aligned.T @ E2_aligned

    # Filter TFs
    if tfs is not None:
        grn_matrix = grn_matrix.loc[tfs]
    elif top_k is not None:
        tf_var = grn_matrix.var(axis=1)
        top_tfs = tf_var.nlargest(top_k).index.tolist()
        grn_matrix = grn_matrix.loc[top_tfs]

    # Filter genes
    if genes is not None:
        # Only keep genes that exist in the matrix
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
    ax.set_title("GRN: TF -> Gene Regulatory Weights")

    savefig_or_show("heatmap_grn", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None
