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
    normalize: str | None = "tf",
    cluster_tfs: bool = True,
    cluster_genes: bool = True,
    cmap: str = "RdBu_r",
    center: float = 0,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (12, 10),
    **kwargs,
) -> Axes | Figure | None:
    """
    Plot a clustered combined TF->gene GRN heatmap (E1 @ E2).

    Computes TF-gene regulatory weights by multiplying E1 (TF->region) and
    E2 (region->gene) matrices. By default, values are z-score normalized
    per TF to highlight gene-specific regulation patterns. Alternatively,
    values can be z-score normalized per gene to compare TF-specific effects.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    tfs
        TFs to include. Their requested order is retained when
        ``cluster_tfs=False``. If omitted, select ``top_k`` TFs by their mean
        absolute TF→gene weight across the selected genes (or across all genes
        when ``genes`` is also omitted).
    genes
        Genes to include. Their requested order is retained when
        ``cluster_genes=False``. If omitted, select ``top_k`` genes by their
        mean absolute TF→gene weight across the selected TFs.
    top_k
        Number of features to select for each axis whose explicit list is
        omitted. Selection uses raw weights before optional normalization.
    normalize
        Axis used to z-score the GRN matrix:

        - ``'tf'``: Z-score each TF across genes (rows).
        - ``'gene'``: Z-score each gene across TFs (columns).
        - ``None``: Plot raw E1 @ E2 values.
    cluster_tfs
        Whether to hierarchically cluster TF rows. Default is True.
    cluster_genes
        Whether to hierarchically cluster gene columns. Default is True.
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
        Passed to seaborn.clustermap.

    Returns
    -------
    Axes, Figure, or None depending on parameters.

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> ds.pl.heatmap_grn(model, top_k=20)
    >>> ds.pl.heatmap_grn(
    ...     model,
    ...     tfs=["MITF", "SOX10"],
    ...     genes=["TYR", "DCT", "MLANA"],
    ...     normalize="gene",
    ...     cluster_tfs=False,
    ... )
    """
    import numpy as np
    import pandas as pd
    import torch
    from scipy.sparse import coo_matrix

    from deepscenic.tl import extract_tf2r_matrix

    # Get E1 as dense DataFrame (n_tfs, n_regions)
    E1_df = extract_tf2r_matrix(model)

    # Build sparse E2 matrix using scipy (avoids dense materialization)
    with torch.no_grad():
        r2g_indices = model.vae.r2g_indices.cpu().numpy()  # type: ignore[union-attr]
        adj_r2g = model.vae.adj_r2g.abs().cpu().numpy()  # type: ignore[union-attr]

    E2_scipy = coo_matrix(
        (adj_r2g, (r2g_indices[0], r2g_indices[1])),
        shape=(len(model.region_names), len(model.gene_names)),
    ).tocsr()

    # Sparse matrix multiplication: GRN = E1 @ E2 -> (n_tfs, n_genes)
    grn_values = E1_df.values @ E2_scipy

    grn_matrix = pd.DataFrame(
        grn_values.toarray() if hasattr(grn_values, "toarray") else np.asarray(grn_values),
        index=model.tf_names,
        columns=model.gene_names,
    )

    # Resolve explicit selections first, preserving the requested order.
    if tfs is not None:
        missing_tfs = [tf for tf in tfs if tf not in grn_matrix.index]
        if missing_tfs:
            raise ValueError(f"TFs not found in model: {missing_tfs}")
        selected_tfs = list(tfs)
    elif top_k is not None:
        candidate_genes = genes if genes is not None else list(grn_matrix.columns)
        missing_genes = [gene for gene in candidate_genes if gene not in grn_matrix.columns]
        if missing_genes:
            raise ValueError(f"Genes not found in model: {missing_genes}")
        tf_scores = grn_matrix.loc[:, candidate_genes].abs().mean(axis=1)
        selected_tfs = tf_scores.nlargest(top_k).index.tolist()
    else:
        selected_tfs = list(grn_matrix.index)

    if genes is not None:
        missing_genes = [gene for gene in genes if gene not in grn_matrix.columns]
        if missing_genes:
            raise ValueError(f"Genes not found in model: {missing_genes}")
        selected_genes = list(genes)
    elif top_k is not None:
        gene_scores = grn_matrix.loc[selected_tfs].abs().mean(axis=0)
        selected_genes = gene_scores.nlargest(top_k).index.tolist()
    else:
        selected_genes = list(grn_matrix.columns)

    # Normalize the full matrix after feature ranking, then subset for plotting.
    if normalize == "tf":
        # Z-score per TF (row): (value - row_mean) / row_std
        row_mean = grn_matrix.mean(axis=1)
        row_std = grn_matrix.std(axis=1)
        # Avoid division by zero for TFs with zero variance
        row_std = row_std.replace(0, np.nan)
        grn_matrix = grn_matrix.sub(row_mean, axis=0).div(row_std, axis=0)
        # A constant TF has no relative preference across genes.
        grn_matrix = grn_matrix.fillna(0.0)
        normalization_label = "z-score per TF"
    elif normalize == "gene":
        # Z-score per gene (column): (value - column_mean) / column_std
        column_mean = grn_matrix.mean(axis=0)
        column_std = grn_matrix.std(axis=0)
        # Avoid division by zero for genes with zero variance
        column_std = column_std.replace(0, np.nan)
        grn_matrix = grn_matrix.sub(column_mean, axis=1).div(column_std, axis=1)
        # A constant gene has no relative preference across TFs.
        grn_matrix = grn_matrix.fillna(0.0)
        normalization_label = "z-score per gene"
    elif normalize is None:
        normalization_label = None
    else:
        raise ValueError(f"Unknown normalize method: {normalize!r}. Use 'tf', 'gene', or None.")

    grn_matrix = grn_matrix.loc[selected_tfs, selected_genes]

    # A single observation cannot be hierarchically clustered.
    row_cluster = cluster_tfs and len(grn_matrix.index) > 1
    col_cluster = cluster_genes and len(grn_matrix.columns) > 1

    # Create clustered plot. Unlike sns.heatmap, clustermap owns its figure and
    # cannot be drawn into a pre-existing Axes.
    grid = sns.clustermap(
        grn_matrix,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cmap=cmap,
        center=center,
        figsize=figsize,
        **kwargs,
    )
    grid.ax_heatmap.set_xlabel("Target Genes")
    grid.ax_heatmap.set_ylabel("Transcription Factors")
    title = "GRN: TF -> Gene Regulatory Weights"
    if normalization_label is not None:
        title += f" ({normalization_label})"
    grid.figure.suptitle(title, y=1.02)

    savefig_or_show("heatmap_grn", show=show, save=save)

    if return_fig:
        return grid.figure
    if show is False:
        return grid.ax_heatmap
    return None
