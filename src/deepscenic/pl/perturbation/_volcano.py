"""Waterfall plot for perturbation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .._utils import COLORS, savefig_or_show, setup_axes


def waterfall_perturbation(
    results: pd.DataFrame,
    *,
    top_n: int = 20,
    logfc_col: str = "log2fc",
    gene_col: str = "gene",
    highlight_genes: list[str] | None = None,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (8, 6),
    title: str | None = None,
) -> Axes | Figure | None:
    """
    Waterfall plot of top up- and down-regulated genes from perturbation.

    Shows the top N most affected genes ranked by effect size (log fold
    change), with upregulated genes on the right and downregulated genes
    on the left. This is the recommended visualization for perturbation
    results, as standard volcano plots are not meaningful for deterministic
    model simulations where p-values are always ~0.

    Parameters
    ----------
    results
        DataFrame from :func:`~deepscenic.tl.process_perturbation_results`
        with ``log2fc`` and ``gene`` columns.
    top_n
        Number of top genes to show per direction (up and down).
    logfc_col
        Column name for log fold change values.
    gene_col
        Column name for gene names.
    highlight_genes
        Specific genes to highlight with bold labels.
    ax
        Pre-existing axes.
    show
        Display figure.
    save
        Save figure.
    return_fig
        Return Figure instead of Axes.
    figsize
        Figure size.
    title
        Plot title.

    Returns
    -------
    Axes, Figure, or None

    Examples
    --------
    >>> import deepscenic as ds
    >>> _, logFC = ds.tl.simulate_perturbation(model, mdata, "SOX10")
    >>> results = ds.tl.process_perturbation_results(logFC, mdata, "SOX10")
    >>> ds.pl.waterfall_perturbation(results, highlight_genes=["MITF", "DCT"])
    """
    df = results.copy()

    # Get top up and down regulated genes
    top_up = df.nlargest(top_n, logfc_col)
    top_down = df.nsmallest(top_n, logfc_col)

    # Combine and sort by logfc (ascending so most downregulated at bottom)
    import pandas as pd

    combined = pd.concat([top_up, top_down]).drop_duplicates(subset=[gene_col]).sort_values(logfc_col, ascending=True)

    fig, ax = setup_axes(ax, figsize=figsize)

    # Color bars by direction
    colors = [COLORS["gene"] if v < 0 else COLORS["tf"] for v in combined[logfc_col]]

    y_pos = np.arange(len(combined))
    ax.barh(y_pos, combined[logfc_col].values, color=colors, alpha=0.8, edgecolor="none")

    # Gene labels
    gene_names = combined[gene_col].values
    for i, gene in enumerate(gene_names):
        is_highlight = highlight_genes and gene in highlight_genes
        ax.text(
            0,
            i,
            f"  {gene}  ",
            va="center",
            ha="right" if combined[logfc_col].iloc[i] >= 0 else "left",
            fontsize=8,
            fontweight="bold" if is_highlight else "normal",
            color="black" if is_highlight else "dimgray",
        )

    # Reference line at 0
    ax.axvline(0, color="black", linewidth=0.8)

    # Clean up axes
    ax.set_yticks([])
    ax.set_xlabel("Effect size (log fold change)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if title:
        ax.set_title(title)
    else:
        # Infer TF name from results if available
        if "tf" in df.columns:
            tf_name = df["tf"].iloc[0]
            ax.set_title(f"{tf_name} Perturbation — Top Affected Genes")
        else:
            ax.set_title("Perturbation — Top Affected Genes")

    savefig_or_show("waterfall_perturbation", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None
