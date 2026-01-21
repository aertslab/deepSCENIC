"""Sequence logo visualizations using tangermeme/logomaker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .._utils import savefig_or_show, setup_axes


def logo_attribution(
    attributions: np.ndarray,
    *,
    sequence: np.ndarray | None = None,
    start: int = 0,
    end: int | None = None,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (12, 3),
    title: str | None = None,
) -> Axes | Figure | None:
    """
    Plot sequence attribution logo using tangermeme.

    Parameters
    ----------
    attributions
        Attribution values, shape (seq_len, 4) or (4, seq_len).
    sequence
        One-hot encoded sequence for masking.
    start
        Start position to plot.
    end
        End position to plot.
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

    Raises
    ------
    ImportError
        If tangermeme is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> import deepscenic as ds
    >>> attrs = np.random.randn(100, 4)  # (seq_len, 4)
    >>> ds.pl.logo_attribution(attrs)  # doctest: +SKIP
    """
    try:
        from tangermeme.plot import plot_logo
    except ImportError as e:
        raise ImportError("tangermeme is required for logo_attribution. Install with: pip install tangermeme") from e

    # Ensure shape is (seq_len, 4)
    if attributions.shape[0] == 4 and attributions.shape[1] != 4:
        attributions = attributions.T

    # Slice if needed
    if end is None:
        end = attributions.shape[0]
    attributions = attributions[start:end]

    # Mask by sequence if provided
    if sequence is not None:
        if sequence.shape[0] == 4:
            sequence = sequence.T
        sequence = sequence[start:end]
        attributions = attributions * sequence

    fig, ax = setup_axes(ax, figsize=figsize)
    plot_logo(attributions, ax=ax)

    if title:
        ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Attribution")

    savefig_or_show("logo_attribution", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None


def logo_motif(
    pwm: np.ndarray,
    *,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (8, 2),
    title: str | None = None,
) -> Axes | Figure | None:
    """
    Plot motif logo from position weight matrix using logomaker.

    Parameters
    ----------
    pwm
        Position weight matrix, shape (seq_len, 4) or (4, seq_len).
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

    Raises
    ------
    ImportError
        If logomaker is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> import deepscenic as ds
    >>> pwm = np.random.rand(20, 4)
    >>> pwm = pwm / pwm.sum(axis=1, keepdims=True)
    >>> ds.pl.logo_motif(pwm)  # doctest: +SKIP
    """
    try:
        import logomaker
    except ImportError as e:
        raise ImportError("logomaker is required for logo_motif. Install with: pip install logomaker") from e

    import pandas as pd

    # Ensure shape is (seq_len, 4)
    if pwm.shape[0] == 4 and pwm.shape[1] != 4:
        pwm = pwm.T

    # Create DataFrame with ACGT columns
    pwm_df = pd.DataFrame(pwm, columns=["A", "C", "G", "T"])

    fig, ax = setup_axes(ax, figsize=figsize)
    logomaker.Logo(pwm_df, ax=ax)

    if title:
        ax.set_title(title)

    savefig_or_show("logo_motif", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None
