"""Color management for deepSCENIC plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mudata import MuData

from ._utils import get_cmap_colors


def set_colors(
    mdata: MuData,
    key: str,
    colors: list[str] | dict[str, str] | None = None,
    cmap: str = "tab20",
) -> None:
    """
    Set colors for a categorical variable in mdata.uns.

    Parameters
    ----------
    mdata
        MuData object.
    key
        Key in mdata.obs to color.
    colors
        Colors to use. If None, generate from cmap.
        Can be list (ordered) or dict (category -> color).
    cmap
        Colormap to use if colors is None.

    Raises
    ------
    KeyError
        If key not found in mdata.obs or any modality's obs.
    ValueError
        If number of colors doesn't match number of categories.

    Examples
    --------
    >>> import deepscenic as ds
    >>> ds.pl.set_colors(mdata, "celltype")    >>> ds.pl.set_colors(mdata, "celltype", colors=["red", "blue"])
    """
    # Get categories
    if key in mdata.obs.columns:
        categories = mdata.obs[key].cat.categories.tolist()
    elif key in mdata.mod["rna"].obs.columns:
        categories = mdata.mod["rna"].obs[key].cat.categories.tolist()
    else:
        raise KeyError(f"Key '{key}' not found in mdata.obs")

    # Generate or validate colors
    if colors is None:
        color_list = get_cmap_colors(len(categories), cmap)
    elif isinstance(colors, dict):
        color_list = [colors[cat] for cat in categories]
    else:
        color_list = list(colors)

    if len(color_list) != len(categories):
        raise ValueError(f"Need {len(categories)} colors, got {len(color_list)}")

    # Store in mdata.uns
    mdata.uns[f"{key}_colors"] = np.array(color_list)


def get_colors(
    mdata: MuData,
    key: str,
    cmap: str = "tab20",
) -> dict[str, str]:
    """
    Get colors for a categorical variable.

    Parameters
    ----------
    mdata
        MuData object.
    key
        Key in mdata.obs.
    cmap
        Colormap to use if colors not stored.

    Returns
    -------
    Mapping from category to color.

    Raises
    ------
    KeyError
        If key not found in mdata.obs or any modality's obs.

    Examples
    --------
    >>> import deepscenic as ds
    >>> colors = ds.pl.get_colors(mdata, "celltype")
    """
    # Get categories
    if key in mdata.obs.columns:
        categories = mdata.obs[key].cat.categories.tolist()
    elif key in mdata.mod["rna"].obs.columns:
        categories = mdata.mod["rna"].obs[key].cat.categories.tolist()
    else:
        raise KeyError(f"Key '{key}' not found in mdata.obs")

    # Check if colors stored
    color_key = f"{key}_colors"
    if color_key in mdata.uns:
        color_list = mdata.uns[color_key]
    else:
        color_list = get_cmap_colors(len(categories), cmap)
        mdata.uns[color_key] = np.array(color_list)

    return dict(zip(categories, color_list, strict=True))
