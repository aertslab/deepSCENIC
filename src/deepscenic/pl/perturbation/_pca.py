"""PCA visualization for paired perturbation simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import issparse

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .._utils import savefig_or_show, setup_axes


def perturbation_pca(
    original: np.ndarray,
    perturbed: np.ndarray,
    *,
    obs: pd.DataFrame | None = None,
    color: str | None = None,
    palette: list[str] | dict[object, str] | None = None,
    cmap: str = "viridis",
    n_components: int = 2,
    max_arrows: int | None = 500,
    original_label: str = "Original",
    perturbed_label: str = "Perturbed",
    original_color: str = "lightgray",
    perturbed_color: str = "#EF6F6C",
    point_size: float = 20,
    original_alpha: float = 0.35,
    perturbed_alpha: float = 1.0,
    arrow_alpha: float = 0.35,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (6, 5),
    title: str | None = None,
) -> Axes | Figure | None:
    """Plot paired original and perturbed cells in a shared PCA space.

    PCA is fitted only on the original expression matrix. The perturbed
    matrix is then projected with the same centering and components, so each
    arrow represents the predicted displacement of one cell after
    perturbation.

    Parameters
    ----------
    original
        Original expression matrix with shape ``(n_cells, n_features)``.
    perturbed
        Perturbed expression matrix with the same shape and cell order as
        ``original``.
    obs
        Cell annotations in the same order as the matrix rows.
    color
        Column in ``obs`` used to color cells. Categorical annotations use a
        discrete legend; numeric annotations use ``cmap`` and a colorbar.
    palette
        Colors for categorical annotations, either in category order or as a
        mapping from category to color.
    cmap
        Colormap for numeric annotations.
    n_components
        Number of PCA components to compute. Must be at least 2.
    max_arrows
        Maximum number of paired displacement arrows to draw. Cells are
        selected at evenly spaced indices. Set to ``None`` to draw all arrows.
    original_label
        Legend label for original cells.
    perturbed_label
        Legend label for perturbed cells.
    original_color
        Color of original cells.
    perturbed_color
        Color of perturbed cells and displacement arrows.
    point_size
        Scatter point size.
    original_alpha
        Opacity of original cells. Defaults to 0.35 so perturbed cells remain
        visually prominent.
    perturbed_alpha
        Opacity of perturbed cells. Defaults to 1.0.
    arrow_alpha
        Displacement arrow opacity.
    ax
        Pre-existing axes.
    show
        Display the figure.
    save
        Save the figure using the package plotting configuration.
    return_fig
        Return the Figure instead of the Axes when ``show=False``.
    figsize
        Figure size when creating new axes.
    title
        Plot title. Defaults to ``"Perturbation in PCA space"``.

    Returns
    -------
    Axes, Figure, or None
        Figure when ``return_fig=True``; Axes when ``show=False``; otherwise
        None.

    Examples
    --------
    >>> import numpy as np
    >>> import deepscenic as ds
    >>> from scipy.sparse import issparse
    >>> original = mdata.mod["rna"].X
    >>> original = original.toarray() if issparse(original) else np.asarray(original)
    >>> perturbed, logFC = ds.tl.simulate_perturbation(
    ...     model, mdata, "SOX10", level=0
    ... )
    >>> ds.pl.perturbation_pca(
    ...     original,
    ...     perturbed,
    ...     perturbed_label="SOX10 knockout",
    ...     obs=mdata.obs,
    ...     color="cell_state",
    ... )
    """
    from sklearn.decomposition import PCA

    original_array = original.toarray() if issparse(original) else np.asarray(original)
    perturbed_array = perturbed.toarray() if issparse(perturbed) else np.asarray(perturbed)

    if original_array.ndim != 2 or perturbed_array.ndim != 2:
        raise ValueError("original and perturbed must be two-dimensional matrices")
    if original_array.shape != perturbed_array.shape:
        raise ValueError(f"Shape mismatch: original {original_array.shape} != perturbed {perturbed_array.shape}")
    if n_components < 2:
        raise ValueError("n_components must be at least 2")
    max_components = min(original_array.shape)
    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} exceeds the maximum {max_components} "
            f"for an original matrix with shape {original_array.shape}"
        )
    if max_arrows is not None and max_arrows < 0:
        raise ValueError("max_arrows must be non-negative or None")
    if color is not None:
        if obs is None:
            raise ValueError("obs is required when color is specified")
        if color not in obs.columns:
            raise KeyError(f"Annotation {color!r} not found in obs")
        if len(obs) != original_array.shape[0]:
            raise ValueError(f"obs has {len(obs)} rows, but the matrices have {original_array.shape[0]} cells")

    pca = PCA(n_components=n_components)
    original_pca = pca.fit_transform(original_array)
    perturbed_pca = pca.transform(perturbed_array)
    displacement = perturbed_pca - original_pca

    fig, ax = setup_axes(ax, figsize=figsize)
    annotation_colors = None
    annotation_handles = []
    if color is None:
        ax.scatter(
            original_pca[:, 0],
            original_pca[:, 1],
            color=original_color,
            s=point_size,
            alpha=original_alpha,
            marker="o",
        )
        ax.scatter(
            perturbed_pca[:, 0],
            perturbed_pca[:, 1],
            color=perturbed_color,
            s=point_size,
            alpha=perturbed_alpha,
            marker="^",
        )
    else:
        import pandas as pd
        from matplotlib.lines import Line2D

        values = obs[color]
        if pd.api.types.is_numeric_dtype(values):
            if palette is not None:
                raise ValueError("palette is only supported for categorical annotations")
            original_scatter = ax.scatter(
                original_pca[:, 0],
                original_pca[:, 1],
                c=values.to_numpy(),
                cmap=cmap,
                s=point_size,
                alpha=original_alpha,
                marker="o",
            )
            ax.scatter(
                perturbed_pca[:, 0],
                perturbed_pca[:, 1],
                c=values.to_numpy(),
                cmap=cmap,
                s=point_size,
                alpha=perturbed_alpha,
                marker="^",
            )
            fig.colorbar(original_scatter, ax=ax, label=color)
        else:
            from .._utils import get_cmap_colors

            values = values.astype("string").fillna("<NA>")
            categories = values.drop_duplicates().tolist()
            if palette is None:
                category_colors = dict(zip(categories, get_cmap_colors(len(categories)), strict=True))
            elif isinstance(palette, dict):
                missing = [category for category in categories if category not in palette]
                if missing:
                    raise ValueError(f"palette has no colors for categories: {missing}")
                category_colors = {category: palette[category] for category in categories}
            else:
                if len(palette) != len(categories):
                    raise ValueError(f"Need {len(categories)} colors, got {len(palette)}")
                category_colors = dict(zip(categories, palette, strict=True))
            annotation_colors = values.map(category_colors).to_numpy()
            ax.scatter(
                original_pca[:, 0],
                original_pca[:, 1],
                color=annotation_colors,
                s=point_size,
                alpha=original_alpha,
                marker="o",
            )
            ax.scatter(
                perturbed_pca[:, 0],
                perturbed_pca[:, 1],
                color=annotation_colors,
                s=point_size,
                alpha=perturbed_alpha,
                marker="^",
            )
            annotation_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=category_colors[category],
                    markeredgecolor="none",
                    label=str(category),
                )
                for category in categories
            ]

    n_cells = original_array.shape[0]
    if max_arrows is None or max_arrows >= n_cells:
        arrow_indices = np.arange(n_cells)
    elif max_arrows == 0:
        arrow_indices = np.array([], dtype=int)
    else:
        arrow_indices = np.linspace(0, n_cells - 1, max_arrows, dtype=int)

    if arrow_indices.size:
        ax.quiver(
            original_pca[arrow_indices, 0],
            original_pca[arrow_indices, 1],
            displacement[arrow_indices, 0],
            displacement[arrow_indices, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=(annotation_colors[arrow_indices] if annotation_colors is not None else perturbed_color),
            alpha=arrow_alpha,
            width=0.002,
        )

    explained_variance = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1f}%)")
    ax.set_title(title or "Perturbation in PCA space")
    from matplotlib.lines import Line2D

    condition_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="gray", label=original_label),
        Line2D([0], [0], marker="^", linestyle="none", color="gray", label=perturbed_label),
    ]
    ax.legend(handles=condition_handles + annotation_handles, frameon=False)
    ax.grid(False)

    savefig_or_show("perturbation_pca", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None
