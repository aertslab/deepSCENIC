"""In-silico mutagenesis visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .._utils import savefig_or_show, setup_axes


def ism_heatmap(
    ism_scores: np.ndarray,
    *,
    start: int = 0,
    end: int | None = None,
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
    center: float = 0,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (14, 3),
    title: str | None = None,
) -> Axes | Figure | None:
    """
    Plot ISM scores as a heatmap.

    Parameters
    ----------
    ism_scores
        ISM delta scores, shape (seq_len, 4) or (4, seq_len).
        Values represent change in prediction when mutating to each nucleotide.
    start
        Start position to plot.
    end
        End position to plot.
    ax
        Pre-existing axes.
    cmap
        Colormap (diverging recommended).
    center
        Center value for colormap.
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
    >>> import numpy as np
    >>> import deepscenic as ds
    >>> ism_scores = np.random.randn(100, 4)
    >>> ds.pl.ism_heatmap(ism_scores)  # doctest: +SKIP
    """
    # Ensure shape is (4, seq_len) for heatmap
    if ism_scores.shape[1] == 4 and ism_scores.shape[0] != 4:
        ism_scores = ism_scores.T

    # Slice
    if end is None:
        end = ism_scores.shape[1]
    ism_scores = ism_scores[:, start:end]

    fig, ax = setup_axes(ax, figsize=figsize)

    sns.heatmap(
        ism_scores,
        ax=ax,
        cmap=cmap,
        center=center,
        yticklabels=["A", "C", "G", "T"],
        cbar_kws={"label": "Delta Prediction"},
    )

    ax.set_xlabel("Position")
    ax.set_ylabel("Mutated to")
    if title:
        ax.set_title(title)

    savefig_or_show("ism_heatmap", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None
