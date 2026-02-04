"""Cell-type specific GRN visualization functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

from .._utils import savefig_or_show


def upset_active_enhancers(
    active_enhancers: dict[str, list[str]],
    *,
    min_subset_size: int = 1,
    show_counts: bool = True,
    sort_by: str = "cardinality",
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (12, 6),
) -> Figure | None:
    """
    UpSet plot showing enhancer overlap across cell types.

    Visualizes which enhancers are shared between cell types and which are
    cell-type specific. Useful for identifying common vs unique regulatory
    elements.

    Parameters
    ----------
    active_enhancers
        Dictionary mapping cell type names to lists of active enhancer region names.
        Typically from identify_active_enhancers().
    min_subset_size
        Minimum number of enhancers in a subset to display.
    show_counts
        Show count numbers on the bars.
    sort_by
        How to sort subsets: 'cardinality' (size) or 'degree' (number of sets).
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
    Figure or None.

    Examples
    --------
    >>> import deepscenic as ds
    >>> enhancer_activity = ds.tl.compute_celltype_enhancer_activity(model, mdata, "celltype")
    >>> active_enhancers = ds.tl.identify_active_enhancers(enhancer_activity)
    >>> ds.pl.upset_active_enhancers(active_enhancers)

    Notes
    -----
    Requires the upsetplot package. Install with: pip install upsetplot
    or pip install deepscenic[celltype]
    """
    try:
        from upsetplot import UpSet, from_memberships
    except ImportError as e:
        raise ImportError(
            "upsetplot is required for upset_active_enhancers. "
            "Install with: pip install upsetplot or pip install deepscenic[celltype]"
        ) from e

    import matplotlib.pyplot as plt

    # Convert dict to membership format for upsetplot
    # from_memberships expects: [("A", "B"), ("A",), ("B", "C"), ...]
    # where each tuple is the set memberships of an element

    # Build region -> cell types mapping
    region_memberships: dict[str, list[str]] = {}
    for celltype, regions in active_enhancers.items():
        for region in regions:
            if region not in region_memberships:
                region_memberships[region] = []
            region_memberships[region].append(celltype)

    # Convert to membership tuples
    memberships = [tuple(sorted(cts)) for cts in region_memberships.values()]

    if len(memberships) == 0:
        raise ValueError("No active enhancers found in any cell type")

    # Create UpSet data
    upset_data = from_memberships(memberships)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Create UpSet plot
    upset = UpSet(
        upset_data,
        subset_size="count",
        show_counts=show_counts,
        sort_by=sort_by,
        min_subset_size=min_subset_size,
    )
    upset.plot(fig=fig)

    fig.suptitle("Active Enhancer Overlap Across Cell Types", y=1.02)

    savefig_or_show("upset_active_enhancers", show=show, save=save)

    if return_fig:
        return fig
    return None


def heatmap_celltype_activity(
    activity: pd.DataFrame,
    *,
    top_k: int | None = 100,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    cmap: str = "viridis",
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (10, 12),
    **kwargs,
) -> Figure | None:
    """
    Heatmap of enhancer activity across cell types.

    Visualizes the enhancer activity matrix with regions as rows and cell types
    as columns. Useful for identifying cell-type specific enhancer patterns.

    Parameters
    ----------
    activity
        DataFrame with regions as index and cell types as columns.
        Typically from compute_celltype_enhancer_activity().
    top_k
        Number of top regions to show (by variance across cell types).
        If None, show all regions.
    cluster_rows
        Cluster region rows.
    cluster_cols
        Cluster cell type columns.
    cmap
        Colormap.
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
    Figure or None.

    Examples
    --------
    >>> import deepscenic as ds
    >>> enhancer_activity = ds.tl.compute_celltype_enhancer_activity(model, mdata, "celltype")
    >>> ds.pl.heatmap_celltype_activity(enhancer_activity, top_k=50)
    """
    import seaborn as sns

    df = activity.copy()

    # Filter to top_k regions by variance
    if top_k is not None and top_k < len(df):
        variances = df.var(axis=1)
        top_regions = variances.nlargest(top_k).index
        df = df.loc[top_regions]

    # Create clustermap
    g = sns.clustermap(
        df,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        cmap=cmap,
        figsize=figsize,
        xticklabels=True,
        yticklabels=False,  # Too many regions to show labels
        dendrogram_ratio=(0.1, 0.05),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        **kwargs,
    )
    g.ax_heatmap.set_xlabel("Cell Type")
    g.ax_heatmap.set_ylabel("Enhancer Regions")
    g.fig.suptitle("Enhancer Activity by Cell Type", y=1.02)

    savefig_or_show("heatmap_celltype_activity", show=show, save=save)

    if return_fig:
        return g.fig
    return None
