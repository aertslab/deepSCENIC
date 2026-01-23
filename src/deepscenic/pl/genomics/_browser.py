"""Genome browser visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure
    from mudata import MuData

from .._utils import COLORS, parse_region, savefig_or_show


def genome_browser(
    mdata: MuData | None = None,
    *,
    chrom: str,
    start: int,
    end: int,
    atac_data: np.ndarray | None = None,
    region_names: list[str] | None = None,
    genes: pd.DataFrame | None = None,
    links: pd.DataFrame | None = None,
    cell_groups: dict[str, np.ndarray] | None = None,
    group_colors: dict[str, str] | None = None,
    show_regions: bool = True,
    show_genes: bool = True,
    show_links: bool = True,
    link_cmap: str = "Blues",
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure | None:
    """
    Multi-track genome browser view.

    Parameters
    ----------
    mdata
        MuData object with ATAC data. If provided, extracts data automatically.
    chrom
        Chromosome to display.
    start
        Start position.
    end
        End position.
    atac_data
        ATAC accessibility matrix (cells x regions). Alternative to mdata.
    region_names
        Region names in format 'chr:start-end'. Required if atac_data provided.
    genes
        DataFrame with gene annotations (columns: gene, chrom, start, end, strand).
    links
        DataFrame with region-gene links (columns: region, gene, weight).
    cell_groups
        Dict mapping group names to cell indices for aggregated tracks.
    group_colors
        Dict mapping group names to colors.
    show_regions
        Show region accessibility track.
    show_genes
        Show gene annotation track.
    show_links
        Show region-gene link arcs.
    link_cmap
        Colormap for link arcs.
    show
        Display figure.
    save
        Save figure.
    return_fig
        Return Figure.
    figsize
        Figure size. Auto-calculated if None.
    title
        Plot title.

    Returns
    -------
    Figure or None

    Examples
    --------
    >>> import deepscenic as ds
    >>> ds.pl.genome_browser(
    ...     mdata, chrom="chr1", start=1000, end=5000
    ... )    """
    import pandas as pd

    # Extract data from mdata if provided
    if mdata is not None:
        atac_adata = mdata.mod["atac"]
        atac_data = atac_adata.X
        region_names = atac_adata.var_names.tolist()

        # Try to get gene info from RNA modality
        if genes is None and "rna" in mdata.mod:
            rna_var = mdata.mod["rna"].var
            if all(col in rna_var.columns for col in ["chromosome", "tss"]):
                genes = pd.DataFrame(
                    {
                        "gene": rna_var.index,
                        "chrom": rna_var["chromosome"],
                        "start": rna_var["tss"],
                        "end": rna_var["tss"] + 1000,  # Approximate gene body
                        "strand": rna_var.get("strand", "+"),
                    }
                )

    # Filter regions in view
    region_mask = []
    region_coords = []
    if region_names is not None:
        for i, r in enumerate(region_names):
            r_chrom, r_start, r_end = parse_region(r)
            if r_chrom == chrom and r_start < end and r_end > start:
                region_mask.append(i)
                region_coords.append((r_start, r_end))

    # Calculate number of tracks
    n_tracks = 0
    if show_regions and len(region_mask) > 0:
        if cell_groups:
            n_tracks += len(cell_groups)
        else:
            n_tracks += 1  # Aggregate track
    if show_genes and genes is not None:
        n_tracks += 1
    if show_links and links is not None:
        n_tracks += 1

    if n_tracks == 0:
        raise ValueError("No tracks to display")

    # Set up figure
    if figsize is None:
        figsize = (12, 2 * n_tracks)

    fig, axes = plt.subplots(n_tracks, 1, figsize=figsize, sharex=True)
    if n_tracks == 1:
        axes = [axes]

    track_idx = 0

    # Region accessibility tracks
    if show_regions and len(region_mask) > 0 and atac_data is not None:
        if cell_groups:
            for group_name, cell_idx in cell_groups.items():
                ax = axes[track_idx]

                # Aggregate accessibility for this group
                group_signal = np.mean(atac_data[cell_idx][:, region_mask], axis=0)

                # Plot as bars
                for (r_start, r_end), signal in zip(region_coords, group_signal, strict=False):
                    color = group_colors.get(group_name, COLORS["gene"]) if group_colors else COLORS["gene"]
                    ax.fill_between([r_start, r_end], 0, signal, alpha=0.7, color=color)

                ax.set_ylabel(group_name, fontsize=8)
                ax.set_xlim(start, end)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                track_idx += 1
        else:
            ax = axes[track_idx]

            # Aggregate all cells
            agg_signal = np.mean(atac_data[:, region_mask], axis=0)

            for (r_start, r_end), signal in zip(region_coords, agg_signal, strict=False):
                ax.fill_between([r_start, r_end], 0, signal, alpha=0.7, color=COLORS["gene"])

            ax.set_ylabel("Accessibility", fontsize=8)
            ax.set_xlim(start, end)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            track_idx += 1

    # Gene annotation track
    if show_genes and genes is not None:
        ax = axes[track_idx]
        genes_in_view = genes[(genes["chrom"] == chrom) & (genes["start"] < end) & (genes["end"] > start)]

        for _, gene in genes_in_view.iterrows():
            g_start = max(gene["start"], start)
            g_end = min(gene["end"], end)
            strand = gene.get("strand", "+")

            # Draw gene body
            y_pos = 0.5
            ax.plot([g_start, g_end], [y_pos, y_pos], color=COLORS["tf"], linewidth=3)

            # Add arrow for direction
            if strand == "+":
                ax.annotate(
                    "",
                    xy=(g_end, y_pos),
                    xytext=(g_end - (end - start) * 0.02, y_pos),
                    arrowprops={"arrowstyle": "->", "color": COLORS["tf"]},
                )
            else:
                ax.annotate(
                    "",
                    xy=(g_start, y_pos),
                    xytext=(g_start + (end - start) * 0.02, y_pos),
                    arrowprops={"arrowstyle": "->", "color": COLORS["tf"]},
                )

            # Gene name
            ax.text((g_start + g_end) / 2, y_pos + 0.2, gene["gene"], ha="center", fontsize=7)

        ax.set_ylim(0, 1)
        ax.set_xlim(start, end)
        ax.set_ylabel("Genes", fontsize=8)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        track_idx += 1

    # Link arcs track
    if show_links and links is not None:
        import matplotlib.patches as mpatches
        from matplotlib import cm
        from matplotlib.colors import Normalize

        ax = axes[track_idx]

        # Filter links in view
        links_in_view = []
        for _, link in links.iterrows():
            r_chrom, r_start, r_end = parse_region(link["region"])
            if r_chrom == chrom and r_start < end and r_end > start:
                links_in_view.append(link)

        if links_in_view:
            links_df = pd.DataFrame(links_in_view)
            norm = Normalize(vmin=links_df["weight"].min(), vmax=links_df["weight"].max())
            cmap_obj = cm.get_cmap(link_cmap)

            for _, link in links_df.iterrows():
                r_chrom, r_start, r_end = parse_region(link["region"])
                pos_region = (r_start + r_end) // 2

                # Try to get gene position from genes DataFrame
                pos_gene = pos_region + 50000  # Default offset
                if genes is not None:
                    gene_row = genes[genes["gene"] == link["gene"]]
                    if len(gene_row) > 0:
                        pos_gene = gene_row.iloc[0]["start"]

                arc = mpatches.FancyArrowPatch(
                    (pos_region, 0),
                    (pos_gene, 0),
                    connectionstyle="arc3,rad=-0.3",
                    color=cmap_obj(norm(link["weight"])),
                    linewidth=1,
                    alpha=0.7,
                )
                ax.add_patch(arc)

        ax.set_ylim(-1, 0.1)
        ax.set_xlim(start, end)
        ax.set_ylabel("Links", fontsize=8)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        track_idx += 1

    # X-axis formatting
    axes[-1].set_xlabel(f"Position on {chrom}")

    # Title
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f"{chrom}:{start:,}-{end:,}")

    plt.tight_layout()

    savefig_or_show("genome_browser", show=show, save=save)

    if return_fig:
        return fig
    return None
