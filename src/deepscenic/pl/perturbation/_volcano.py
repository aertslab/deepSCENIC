"""Volcano plot for perturbation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .._utils import savefig_or_show, setup_axes


def volcano_perturbation(
    results: pd.DataFrame,
    *,
    logfc_col: str = "log2fc",
    pval_col: str = "pvalue",
    gene_col: str = "gene",
    logfc_threshold: float = 0.5,
    pval_threshold: float = 0.05,
    top_n_labels: int = 10,
    highlight_genes: list[str] | None = None,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (8, 6),
    title: str | None = None,
) -> Axes | Figure | None:
    """
    Volcano plot of perturbation effects.

    Parameters
    ----------
    results
        DataFrame with log2fc, pvalue, and gene columns.
    logfc_col
        Column name for log2 fold change.
    pval_col
        Column name for p-value.
    gene_col
        Column name for gene names.
    logfc_threshold
        Threshold for significant fold change.
    pval_threshold
        Threshold for significant p-value.
    top_n_labels
        Number of top genes to label.
    highlight_genes
        Specific genes to highlight.
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
    """
    df = results.copy()

    # Compute -log10(pval)
    df["neg_log_pval"] = -np.log10(df[pval_col].clip(lower=1e-300))

    # Classify points
    df["significant"] = (df[logfc_col].abs() >= logfc_threshold) & (df[pval_col] <= pval_threshold)
    df["direction"] = np.where(df[logfc_col] > 0, "up", "down")

    fig, ax = setup_axes(ax, figsize=figsize)

    # Non-significant points
    non_sig = df[~df["significant"]]
    ax.scatter(non_sig[logfc_col], non_sig["neg_log_pval"], c="gray", alpha=0.5, s=20, label="Not significant")

    # Significant up
    sig_up = df[df["significant"] & (df["direction"] == "up")]
    ax.scatter(sig_up[logfc_col], sig_up["neg_log_pval"], c="#E64B35", alpha=0.7, s=30, label="Up-regulated")

    # Significant down
    sig_down = df[df["significant"] & (df["direction"] == "down")]
    ax.scatter(sig_down[logfc_col], sig_down["neg_log_pval"], c="#4DBBD5", alpha=0.7, s=30, label="Down-regulated")

    # Threshold lines
    ax.axhline(-np.log10(pval_threshold), linestyle="--", color="gray", alpha=0.5)
    ax.axvline(logfc_threshold, linestyle="--", color="gray", alpha=0.5)
    ax.axvline(-logfc_threshold, linestyle="--", color="gray", alpha=0.5)

    # Labels for top genes
    if top_n_labels > 0 and len(df[df["significant"]]) > 0:
        sig_df = df[df["significant"]].nlargest(top_n_labels, "neg_log_pval")
        for _, row in sig_df.iterrows():
            ax.annotate(
                row[gene_col],
                (row[logfc_col], row["neg_log_pval"]),
                fontsize=8,
                alpha=0.8,
            )

    # Highlight specific genes
    if highlight_genes:
        for gene in highlight_genes:
            if gene in df[gene_col].values:
                row = df[df[gene_col] == gene].iloc[0]
                ax.scatter(row[logfc_col], row["neg_log_pval"], c="black", s=100, marker="*", zorder=10)
                ax.annotate(
                    gene,
                    (row[logfc_col], row["neg_log_pval"]),
                    fontsize=10,
                    fontweight="bold",
                )

    ax.set_xlabel("log2(Fold Change)")
    ax.set_ylabel("-log10(p-value)")
    ax.legend(loc="upper right")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Perturbation Effects")

    savefig_or_show("volcano_perturbation", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None
