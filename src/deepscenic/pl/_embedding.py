"""Embedding visualization functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mudata import MuData

from ._utils import savefig_or_show, setup_axes


def embedding_umap(
    mdata: MuData,
    key: str = "X_deepscenic_enh_act",
    *,
    color: str | list[str] | None = None,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    store_umap: bool = False,
    umap_key: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> Figure | None:
    """
    UMAP visualization of embeddings stored in mdata.obsm.

    Creates a temporary AnnData from the specified embedding, computes PCA
    (if needed), neighbors, and UMAP, then plots with scanpy.

    Parameters
    ----------
    mdata
        MuData object containing embeddings in .obsm.
    key
        Key in mdata.obsm to visualize. Default is 'X_deepscenic_enh_act'.
    color
        Keys for annotations of observations, e.g., 'cell_state'.
    n_pcs
        Number of PCA components. Set to 0 to skip PCA.
    n_neighbors
        Number of neighbors for UMAP computation.
    min_dist
        Minimum distance parameter for UMAP.
    store_umap
        If True, store the computed UMAP coordinates back in mdata.obsm.
    umap_key
        Key for storing UMAP coordinates. Defaults to '{key}_umap'.
    title
        Plot title.
    ax
        Pre-existing axes for plotting.
    show
        Display figure.
    save
        Save figure.
    return_fig
        Return Figure instead of None.
    figsize
        Figure size.
    **kwargs
        Additional arguments passed to scanpy.pl.umap().

    Returns
    -------
    Figure or None
        Figure if return_fig=True, otherwise None.

    Raises
    ------
    KeyError
        If the specified key is not found in mdata.obsm.

    Examples
    --------
    >>> import deepscenic as ds
    >>> ds.tl.to_latent(model, mdata)
    >>> ds.pl.embedding_umap(mdata, key="X_deepscenic_enh_act", color="cell_state")

    Store UMAP coordinates for later use:

    >>> ds.pl.embedding_umap(mdata, key="X_deepscenic_z_rna", store_umap=True)
    >>> mdata.obsm["X_deepscenic_z_rna_umap"]  # UMAP coordinates now stored
    """
    import anndata as ad
    import scanpy as sc

    if key not in mdata.obsm:
        raise KeyError(f"Key '{key}' not found in mdata.obsm. Available keys: {list(mdata.obsm.keys())}")

    # Create temporary AnnData
    adata = ad.AnnData(mdata.obsm[key])
    adata.obs = mdata.obs.copy()

    # PCA if needed (skip if embedding is already low-dimensional or n_pcs=0)
    n_features = mdata.obsm[key].shape[1]
    if n_pcs > 0 and n_features > n_pcs:
        sc.pp.pca(adata, n_comps=n_pcs)

    # Compute neighbors and UMAP
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca" if "X_pca" in adata.obsm else None)
    sc.tl.umap(adata, min_dist=min_dist)

    # Optionally store UMAP coordinates back
    if store_umap:
        if umap_key is None:
            umap_key = f"{key}_umap"
        mdata.obsm[umap_key] = adata.obsm["X_umap"]

    # Always suppress scanpy's show — we handle display via savefig_or_show below
    kwargs["show"] = False

    # Track whether the caller supplied their own axes (before setup_axes may reassign ax)
    user_provided_ax = ax is not None

    # Set up axes if provided, otherwise let scanpy create them
    fig = None
    if ax is not None:
        fig, ax = setup_axes(ax, figsize=figsize)
        kwargs["ax"] = ax
    elif figsize is not None:
        # Create figure with specified size
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        kwargs["ax"] = ax

    # Plot
    sc.pl.umap(adata, color=color, title=title, **kwargs)

    # Handle save/show
    writekey = key.replace("X_deepscenic_", "embedding_")
    if user_provided_ax:
        # User owns the figure lifecycle — don't auto-show or close between calls.
        # Treat show=None as False; respect an explicit show=True.
        effective_show = show if show is not None else False
        savefig_or_show(writekey, show=effective_show, save=save, close=False)
    else:
        savefig_or_show(writekey, show=show, save=save)

    if return_fig:
        if fig is None:
            import matplotlib.pyplot as plt

            fig = plt.gcf()
        return fig
    return None
