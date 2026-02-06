"""Co-embedding visualization for perturbation analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from .._utils import savefig_or_show


def perturbation_coembedding(
    original: np.ndarray,
    perturbed: np.ndarray,
    obs: pd.DataFrame | None = None,
    *,
    color: str | list[str] | None = None,
    condition_labels: tuple[str, str] = ("Original", "Perturbed"),
    n_pcs: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    title: str | None = None,
    ncols: int = 2,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> Figure | None:
    """
    Co-embedding visualization of original and perturbed expression.

    Projects both original and perturbed expression profiles into a shared
    UMAP space to visualize phenotype shifts from perturbation.

    Parameters
    ----------
    original
        Original expression matrix, shape (n_cells, n_features).
    perturbed
        Perturbed expression matrix, shape (n_cells, n_features).
    obs
        DataFrame with cell annotations (e.g., cell_state).
        Will be duplicated for both conditions.
    color
        Keys for annotations to color by. 'condition' is always added.
    condition_labels
        Labels for original and perturbed conditions.
    n_pcs
        Number of PCA components. Set to 0 to skip PCA.
    n_neighbors
        Number of neighbors for UMAP computation.
    min_dist
        Minimum distance parameter for UMAP.
    title
        Overall plot title.
    ncols
        Number of columns for subplot layout.
    show
        Display figure.
    save
        Save figure.
    return_fig
        Return Figure instead of None.
    figsize
        Figure size. Defaults to (6*ncols, 5).
    **kwargs
        Additional arguments passed to scanpy.pl.umap().

    Returns
    -------
    Figure or None
        Figure if return_fig=True, otherwise None.

    Raises
    ------
    ValueError
        If original and perturbed have different shapes.

    Examples
    --------
    >>> import deepscenic as ds
    >>> ds.tl.to_latent(model, mdata)
    >>> original = mdata.obsm["X_deepscenic_z_rna"]
    >>> logFC = ds.tl.simulate_perturbation(model, mdata, "SOX10", level=0)
    >>> perturbed = original + logFC
    >>> ds.pl.perturbation_coembedding(
    ...     original, perturbed,
    ...     obs=mdata.obs,
    ...     color=["condition", "cell_state"],
    ...     condition_labels=["Original", "SOX10 KD"],
    ... )
    """
    import anndata as ad
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc

    # Validate shapes
    if original.shape != perturbed.shape:
        raise ValueError(f"Shape mismatch: original {original.shape} != perturbed {perturbed.shape}")

    n_cells = original.shape[0]

    # Stack data
    combined = np.vstack([original, perturbed])

    # Create AnnData
    adata = ad.AnnData(combined)

    # Add condition labels
    adata.obs["condition"] = [condition_labels[0]] * n_cells + [condition_labels[1]] * n_cells
    adata.obs["condition"] = pd.Categorical(adata.obs["condition"], categories=list(condition_labels))

    # Duplicate obs annotations if provided
    if obs is not None:
        for col in obs.columns:
            if col != "condition":
                values = list(obs[col]) * 2
                if hasattr(obs[col], "cat"):
                    adata.obs[col] = pd.Categorical(values, categories=obs[col].cat.categories)
                else:
                    adata.obs[col] = values

    # PCA if needed
    n_features = original.shape[1]
    if n_pcs > 0 and n_features > n_pcs:
        sc.pp.pca(adata, n_comps=n_pcs)

    # Compute neighbors and UMAP
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca" if "X_pca" in adata.obsm else None)
    sc.tl.umap(adata, min_dist=min_dist)

    # Prepare color list
    if color is None:
        color_list = ["condition"]
    elif isinstance(color, str):
        color_list = [color]
    else:
        color_list = list(color)

    # Ensure condition is first if not already present
    if "condition" not in color_list:
        color_list = ["condition"] + color_list

    # Set up figure
    n_plots = len(color_list)
    nrows = (n_plots + ncols - 1) // ncols
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Plot each color
    for i, c in enumerate(color_list):
        sc.pl.umap(adata, color=c, ax=axes[i], show=False, **kwargs)

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    # Handle save/show
    savefig_or_show("perturbation_coembedding", show=show, save=save)

    if return_fig:
        return fig
    return None
