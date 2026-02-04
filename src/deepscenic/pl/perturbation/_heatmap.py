"""Heatmap and dotplot for perturbation results."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .._utils import savefig_or_show, setup_axes


def heatmap_perturbation(
    results: pd.DataFrame,
    *,
    tfs: list[str] | None = None,
    genes: list[str] | None = None,
    value_col: str = "log2fc",
    tf_col: str = "tf",
    gene_col: str = "gene",
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    cmap: str = "RdBu_r",
    center: float = 0,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (12, 10),
    **kwargs,
) -> Figure | None:
    """
    Heatmap of perturbation effects across TFs and genes.

    Parameters
    ----------
    results
        DataFrame with TF, gene, and effect columns.
    tfs
        TFs to include.
    genes
        Genes to include.
    value_col
        Column for heatmap values.
    tf_col
        Column for TF names.
    gene_col
        Column for gene names.
    cluster_rows
        Cluster TF rows.
    cluster_cols
        Cluster gene columns.
    cmap
        Colormap.
    center
        Center value for colormap.
    show
        Display figure.
    save
        Save figure.
    return_fig
        Return Figure.
    figsize
        Figure size.
    **kwargs
        Passed to seaborn.clustermap.

    Returns
    -------
    Figure or None

    Examples
    --------
    >>> import deepscenic as ds
    >>> results = ds.tl.perturb_all_tfs(model, mdata)    >>> ds.pl.heatmap_perturbation(results)
    """
    df = results.copy()

    # Filter
    if tfs is not None:
        df = df[df[tf_col].isin(tfs)]
    if genes is not None:
        df = df[df[gene_col].isin(genes)]

    # Pivot
    pivot = df.pivot_table(
        index=tf_col,
        columns=gene_col,
        values=value_col,
        fill_value=0,
    )

    # Clustermap
    g = sns.clustermap(
        pivot,
        cmap=cmap,
        center=center,
        figsize=figsize,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        **kwargs,
    )

    g.ax_heatmap.set_xlabel("Target Genes")
    g.ax_heatmap.set_ylabel("Perturbed TFs")
    g.fig.suptitle("Perturbation Effects", y=1.02)

    savefig_or_show("heatmap_perturbation", show=show, save=save)

    if return_fig:
        return cast("Figure", g.fig)
    return None


def dotplot_perturbation(
    results: pd.DataFrame,
    *,
    tfs: list[str] | None = None,
    genes: list[str] | None = None,
    size_col: str = "abs_log2fc",
    color_col: str = "log2fc",
    tf_col: str = "tf",
    gene_col: str = "gene",
    size_range: tuple[int, int] = (10, 200),
    cmap: str = "RdBu_r",
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (12, 8),
) -> Axes | Figure | None:
    """
    Dot plot of perturbation effects.

    Parameters
    ----------
    results
        DataFrame with TF, gene, and effect columns.
    tfs
        TFs to include.
    genes
        Genes to include.
    size_col
        Column for dot size (e.g., absolute effect).
    color_col
        Column for dot color (e.g., signed effect).
    tf_col
        Column for TF names.
    gene_col
        Column for gene names.
    size_range
        Min and max dot sizes.
    cmap
        Colormap.
    ax
        Pre-existing axes.
    show
        Display figure.
    save
        Save figure.
    return_fig
        Return Figure.
    figsize
        Figure size.

    Returns
    -------
    Axes, Figure, or None

    Examples
    --------
    >>> import deepscenic as ds
    >>> results = ds.tl.perturb_all_tfs(model, mdata)    >>> ds.pl.dotplot_perturbation(results, tfs=["SOX2", "NANOG"])
    """
    df = results.copy()

    # Filter
    if tfs is not None:
        df = df[df[tf_col].isin(tfs)]
    if genes is not None:
        df = df[df[gene_col].isin(genes)]

    # Create size column if needed
    if size_col not in df.columns and "log2fc" in df.columns:
        df["abs_log2fc"] = df["log2fc"].abs()
        size_col = "abs_log2fc"

    # Get unique values for positioning
    unique_tfs = df[tf_col].unique()
    unique_genes = df[gene_col].unique()
    tf_to_y = {tf: i for i, tf in enumerate(unique_tfs)}
    gene_to_x = {gene: i for i, gene in enumerate(unique_genes)}

    df["x"] = df[gene_col].map(gene_to_x)
    df["y"] = df[tf_col].map(tf_to_y)

    # Normalize sizes
    size_min, size_max = size_range
    sizes = df[size_col]
    sizes_norm = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-10)
    df["size"] = sizes_norm * (size_max - size_min) + size_min

    fig, ax = setup_axes(ax, figsize=figsize)

    scatter = ax.scatter(
        df["x"],
        df["y"],
        s=df["size"],
        c=df[color_col],
        cmap=cmap,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )

    # Colorbar
    plt.colorbar(scatter, ax=ax, label=color_col)

    # Labels
    ax.set_xticks(range(len(unique_genes)))
    ax.set_xticklabels(unique_genes, rotation=90)
    ax.set_yticks(range(len(unique_tfs)))
    ax.set_yticklabels(unique_tfs)
    ax.set_xlabel("Target Genes")
    ax.set_ylabel("Perturbed TFs")
    ax.set_title("Perturbation Effects")

    savefig_or_show("dotplot_perturbation", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None
