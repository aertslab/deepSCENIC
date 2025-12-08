"""Plotting utilities for deepSCENIC."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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
    elif not show:
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
