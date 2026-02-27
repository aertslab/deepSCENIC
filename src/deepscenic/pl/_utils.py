"""Plotting utilities for deepSCENIC."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# Default color palette (NPG-inspired)
COLORS = {
    "tf": "#E64B35",  # TF nodes, E1 weights, up-regulated
    "gene": "#4DBBD5",  # Gene nodes, E2 weights, down-regulated
    "region": "#00A087",  # Region nodes
}


def setup_axes(
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """
    Get or create axes for plotting.

    Parameters
    ----------
    ax
        Pre-existing axes. If None, create new figure.
    figsize
        Figure size if creating new figure.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes objects.
    """
    fig: Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        maybe_fig = ax.get_figure()
        if maybe_fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = cast("Figure", maybe_fig)
    return fig, ax


def savefig_or_show(
    writekey: str,
    show: bool | None = None,
    save: str | bool | None = None,
    ext: str = "pdf",
    dpi: int = 150,
    close: bool = True,
) -> None:
    """
    Scanpy-compatible save/show handler.

    Parameters
    ----------
    writekey
        Base filename (e.g., 'heatmap_grn').
    show
        Display figure. If None, show if not saving.
    save
        Save figure. True for default name, str for suffix.
    ext
        File extension.
    dpi
        Resolution for saved figure.
    close
        Whether to call plt.close() when not showing. Set to False when the
        caller is plotting into a user-provided axes so the figure stays alive.
    """
    if save:
        suffix = save if isinstance(save, str) else ""
        try:
            import scanpy as sc

            filepath = f"{sc.settings.figdir}/{writekey}{suffix}.{ext}"
        except (ImportError, AttributeError):
            filepath = f"{writekey}{suffix}.{ext}"
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")

    if show or (show is None and not save):
        plt.show()
    elif not show and close:
        plt.close()


def get_cmap_colors(n: int, cmap: str = "tab20") -> list[str]:
    """
    Get n colors from a colormap.

    Parameters
    ----------
    n
        Number of colors to generate.
    cmap
        Name of matplotlib colormap.

    Returns
    -------
    list[str]
        List of hex color strings.
    """
    import matplotlib.cm as cm
    from matplotlib.colors import rgb2hex

    cmap_obj = cm.get_cmap(cmap)
    return [rgb2hex(cmap_obj(i / max(n - 1, 1))) for i in range(n)]


def parse_region(region: str) -> tuple[str, int, int]:
    """
    Parse a genomic region string into chromosome, start, and end.

    Parameters
    ----------
    region
        Region string in format 'chr:start-end' (e.g., 'chr1:1000-2000').

    Returns
    -------
    tuple[str, int, int]
        Tuple of (chromosome, start, end).

    Raises
    ------
    ValueError
        If region string is not in expected format.

    Examples
    --------
    >>> from deepscenic.pl import parse_region
    >>> parse_region("chr1:1000-2000")
    ('chr1', 1000, 2000)
    """
    try:
        chrom, coords = region.split(":")
        start_str, end_str = coords.split("-")
        return chrom, int(start_str), int(end_str)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid region format '{region}'. Expected 'chr:start-end'.") from e
