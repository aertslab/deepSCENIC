"""Arc plot for region-gene links."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .._utils import savefig_or_show, setup_axes


def arc_plot(
    links: pd.DataFrame,
    *,
    region_col: str = "region",
    gene_col: str = "gene",
    weight_col: str = "weight",
    gene_positions: dict[str, int] | None = None,
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
    cmap: str = "Blues",
    arc_height: float = 0.5,
    linewidth: float = 1.0,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (12, 4),
    title: str | None = None,
) -> Axes | Figure | None:
    """
    Plot region-gene links as arcs (Signac-style).

    Parameters
    ----------
    links
        DataFrame with region, gene, and weight columns.
        Region format: 'chr1:1000-2000'.
    region_col
        Column for region coordinates.
    gene_col
        Column for gene names.
    weight_col
        Column for link weights.
    gene_positions
        Optional dict mapping gene names to genomic positions (TSS).
        If not provided, arcs will be drawn to fixed offset from region.
    chrom
        Chromosome to filter.
    start
        Start position to filter.
    end
        End position to filter.
    cmap
        Colormap for arc colors.
    arc_height
        Height scaling for arcs.
    linewidth
        Arc line width.
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
    title
        Plot title.

    Returns
    -------
    Axes, Figure, or None
    """
    import pandas as pd
    from matplotlib import cm
    from matplotlib.colors import Normalize

    df = links.copy()

    # Parse regions
    def parse_region(r: str) -> tuple[str, int, int]:
        chrom_r, coords = r.split(":")
        s, e = coords.split("-")
        return chrom_r, int(s), int(e)

    parsed = df[region_col].apply(lambda x: pd.Series(parse_region(x)))
    df[["chrom", "reg_start", "reg_end"]] = parsed
    df["reg_center"] = (df["reg_start"] + df["reg_end"]) // 2

    # Filter by coordinates
    if chrom is not None:
        df = df[df["chrom"] == chrom]
    if start is not None:
        df = df[df["reg_center"] >= start]
    if end is not None:
        df = df[df["reg_center"] <= end]

    if len(df) == 0:
        raise ValueError("No links in specified region")

    fig, ax = setup_axes(ax, figsize=figsize)

    # Color by weight
    norm = Normalize(vmin=df[weight_col].min(), vmax=df[weight_col].max())
    cmap_obj = cm.get_cmap(cmap)

    # Draw arcs
    for _, row in df.iterrows():
        pos_region = row["reg_center"]

        # Get gene position
        if gene_positions is not None and row[gene_col] in gene_positions:
            pos_gene = gene_positions[row[gene_col]]
        else:
            # Default: offset from region center
            pos_gene = pos_region + 50000

        # Determine arc direction based on relative positions
        if pos_gene > pos_region:
            rad = -arc_height
        else:
            rad = arc_height

        arc = mpatches.FancyArrowPatch(
            (pos_region, 0),
            (pos_gene, 0),
            connectionstyle=f"arc3,rad={rad}",
            color=cmap_obj(norm(row[weight_col])),
            linewidth=linewidth,
            alpha=0.7,
        )
        ax.add_patch(arc)

    # Set limits
    all_positions = list(df["reg_center"])
    if gene_positions:
        all_positions.extend([gene_positions.get(g, 0) for g in df[gene_col] if g in gene_positions])

    if start and end:
        ax.set_xlim(start, end)
    else:
        margin = 10000
        ax.set_xlim(min(all_positions) - margin, max(all_positions) + margin)

    ax.set_ylim(-1, 0.1)

    ax.set_xlabel("Genomic Position")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if title:
        ax.set_title(title)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=weight_col, orientation="vertical", fraction=0.02)

    savefig_or_show("arc_plot", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None
