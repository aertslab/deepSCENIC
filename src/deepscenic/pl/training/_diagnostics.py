"""Training diagnostic plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mudata import MuData

    from deepscenic.tl._model import DeepSCENICModel

from .._utils import COLORS, savefig_or_show, setup_axes


def loss_curves(
    history: pd.DataFrame | dict | DeepSCENICModel,
    *,
    metrics: list[str] | None = None,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (10, 6),
    log_scale: bool = True,
) -> Axes | Figure | None:
    """
    Plot training loss curves.

    Parameters
    ----------
    history
        Training history as DataFrame, dict, or DeepSCENICModel containing history.
    metrics
        Specific metrics to plot. If None, plot all available.
    ax
        Pre-existing axes for the plot.
    show
        Whether to display the figure.
    save
        Path to save figure, or True for default path.
    return_fig
        Whether to return the Figure object.
    figsize
        Figure dimensions as (width, height).
    log_scale
        Whether to use logarithmic scale for y-axis.

    Returns
    -------
    Axes, Figure, or None depending on show and return_fig parameters.

    Examples
    --------
    >>> model = ds.tl.train(mdata, epochs=100)
    >>> ds.pl.loss_curves(model)
    >>> ds.pl.loss_curves(model.history.to_dict())
    """
    import pandas as pd

    from deepscenic.tl._model import DeepSCENICModel

    # Handle model input
    if isinstance(history, DeepSCENICModel):
        if history.history is None:
            raise ValueError(
                "Model has no training history. "
                "History is only available for models that were just trained "
                "or loaded from files that include history."
            )
        history = history.history.to_dict()

    # Flatten nested dict structure if needed (train/test split)
    if isinstance(history, dict) and "train" in history:
        flat: dict[str, list[float]] = {}
        for key, values in history.get("train", {}).items():
            flat[f"train_{key}"] = values
        for key, values in history.get("test", {}).items():
            flat[f"test_{key}"] = values
        history = flat

    if isinstance(history, dict):
        history = pd.DataFrame(history)

    if metrics is None:
        metrics = [c for c in history.columns if c != "epoch"]

    fig, ax = setup_axes(ax, figsize=figsize)

    for metric in metrics:
        if metric in history.columns:
            ax.plot(history[metric], label=metric)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()

    if log_scale:
        ax.set_yscale("log")

    savefig_or_show("loss_curves", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None


def sparsity_histogram(
    model: DeepSCENICModel,
    *,
    which: str = "both",
    bins: int = 50,
    threshold: float = 0.01,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (10, 4),
) -> Axes | Figure | None:
    """
    Histogram of E1/E2 weight distributions.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    which
        'E1', 'E2', or 'both'.
    bins
        Number of histogram bins.
    threshold
        Threshold for sparsity calculation.
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

    Returns
    -------
    Axes, Figure, or None depending on parameters.

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")    >>> ds.pl.sparsity_histogram(model)    """
    import torch

    with torch.no_grad():
        E1_vals = model.adj_E1.flatten().cpu().numpy()
        E2_vals = model.vae.adj_E2.abs().flatten().cpu().numpy()

    if which == "both":
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].hist(E1_vals, bins=bins, alpha=0.7, color=COLORS["tf"])
        axes[0].set_xlabel("Weight")
        axes[0].set_ylabel("Count")
        e1_sparsity = (E1_vals < threshold).mean()
        axes[0].set_title(f"E1 (TF->Region)\nSparsity: {e1_sparsity:.1%}")

        axes[1].hist(E2_vals, bins=bins, alpha=0.7, color=COLORS["gene"])
        axes[1].set_xlabel("Weight")
        axes[1].set_ylabel("Count")
        e2_sparsity = (E2_vals < threshold).mean()
        axes[1].set_title(f"E2 (Region->Gene)\nSparsity: {e2_sparsity:.1%}")

        plt.tight_layout()
        savefig_or_show("sparsity_histogram", show=show, save=save)

        if return_fig:
            return fig
        return None
    else:
        fig, ax = setup_axes(ax, figsize=(figsize[0] // 2, figsize[1]))
        vals = E1_vals if which == "E1" else E2_vals
        color = COLORS["tf"] if which == "E1" else COLORS["gene"]

        ax.hist(vals, bins=bins, alpha=0.7, color=color)
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")
        sparsity = (vals < threshold).mean()
        ax.set_title(f"{which} Weights\nSparsity: {sparsity:.1%}")

        savefig_or_show(f"sparsity_{which}", show=show, save=save)

        if return_fig:
            return fig
        if show is False:
            return ax
        return None


def latent_umap(
    model: DeepSCENICModel,
    mdata: MuData,
    *,
    color: str | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (8, 6),
    **kwargs,
) -> Axes | Figure | None:
    """
    UMAP of latent TF activity space.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    mdata
        MuData with cell annotations.
    color
        Column in mdata.obs to color by.
    n_neighbors
        UMAP n_neighbors parameter.
    min_dist
        UMAP min_dist parameter.
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
    **kwargs
        Passed to scatter plot.

    Returns
    -------
    Axes, Figure, or None depending on parameters.

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")    >>> ds.pl.latent_umap(model, mdata, color="celltype")    """
    import torch
    from sklearn.preprocessing import StandardScaler

    try:
        from umap import UMAP
    except ImportError as e:
        raise ImportError("umap-learn is required for latent_umap. Install with: pip install umap-learn") from e

    # Get latent representations
    with torch.no_grad():
        # Get TF expression from RNA modality
        rna_data = mdata.mod["rna"].X
        if hasattr(rna_data, "toarray"):
            rna_data = rna_data.toarray()

        # Get TF indices and extract TF expression
        tf_indices = model.vae.tf_indices.cpu().numpy()
        tf_expression = rna_data[:, tf_indices]

        # Pass through encoder to get latent (returns z, mu, logvar)
        tf_tensor = torch.FloatTensor(tf_expression)
        _, mu, _ = model.vae.encoder(tf_tensor, use_mean=True)
        z_tf = mu.squeeze(-1).cpu().numpy()

    # Standardize
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z_tf)

    # UMAP
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(z_scaled)

    fig, ax = setup_axes(ax, figsize=figsize)

    # Color by annotation if provided
    if color is not None and color in mdata.obs.columns:
        categories = mdata.obs[color]
        if hasattr(categories, "cat"):
            # Categorical
            unique_cats = categories.cat.categories
            cmap = plt.colormaps.get_cmap("tab20")
            colors_map = cmap(np.linspace(0, 1, len(unique_cats)))
            for i, cat in enumerate(unique_cats):
                mask = categories == cat
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[colors_map[i]],
                    label=cat,
                    alpha=0.7,
                    s=10,
                    **kwargs,
                )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        else:
            # Continuous
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1], c=categories, cmap="viridis", alpha=0.7, s=10, **kwargs
            )
            plt.colorbar(scatter, ax=ax, label=color)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=10, **kwargs)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Latent TF Activity Space")

    savefig_or_show("latent_umap", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None


def enhancer_activity_histogram(
    model: DeepSCENICModel,
    mdata: MuData,
    *,
    log_scale: bool = True,
    bins: int = 100,
    exclude_zeros: bool = True,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (8, 5),
) -> Axes | Figure | None:
    """
    Histogram of enhancer activity distribution.

    Use to diagnose if the model learned meaningful enhancer activity profiles.
    A well-trained model should show a smooth, roughly normal distribution
    (possibly with long tails) rather than uniform or degenerate distributions.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    mdata
        MuData with cell annotations.
    log_scale
        Use logarithmic scale for y-axis.
    bins
        Number of histogram bins.
    exclude_zeros
        Exclude zero-activity enhancers from histogram.
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

    Returns
    -------
    Axes, Figure, or None depending on parameters.

    Examples
    --------
    >>> import deepscenic as ds
    >>> model = ds.tl.load_model("model.pt")    >>> ds.pl.enhancer_activity_histogram(model, mdata)    """
    import torch

    # Get enhancer activity from model forward pass
    with torch.no_grad():
        rna_data = mdata.mod["rna"].X
        if hasattr(rna_data, "toarray"):
            rna_data = rna_data.toarray()

        # Get TF expression and compute enhancer activity
        tf_indices = model.vae.tf_indices.cpu().numpy()
        tf_expression = rna_data[:, tf_indices]

        # Enhancer activity = TF expression @ E1
        tf_tensor = torch.FloatTensor(tf_expression)

        # Get z_tf from encoder
        _, mu, _ = model.vae.encoder(tf_tensor, use_mean=True)
        z_tf = mu.squeeze(-1)

        # Compute enhancer activity: z_tf @ E1
        enh_act = (z_tf @ model.adj_E1).cpu().numpy()

    # Flatten and optionally exclude zeros
    values = enh_act.flatten()
    if exclude_zeros:
        values = values[values != 0]

    fig, ax = setup_axes(ax, figsize=figsize)

    ax.hist(values, bins=bins, alpha=0.7, color=COLORS["region"], edgecolor="white")
    ax.set_xlabel("Enhancer Activity")
    ax.set_ylabel("Count")
    ax.set_title("Enhancer Activity Distribution")

    if log_scale:
        ax.set_yscale("log")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig_or_show("enhancer_activity_histogram", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None


def tf_activity_clustermap(
    model: DeepSCENICModel,
    mdata: MuData,
    *,
    groupby: str,
    top_n_tfs: int | None = 50,
    tfs: list[str] | None = None,
    cluster_rows: bool = True,
    cluster_cols: bool = False,
    cmap: str = "RdBu_r",
    center: float = 0,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (12, 8),
    **kwargs,
) -> Figure | None:
    """
    Clustered heatmap of TF activities per cell group.

    Use to diagnose if the model learned cell-type-specific TF activity patterns.
    A well-trained model should show distinct TF activity profiles that
    correlate with known biology (e.g., lineage-specific TFs).

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    mdata
        MuData with cell annotations.
    groupby
        Column in mdata.obs to group cells by (e.g., 'celltype').
    top_n_tfs
        Number of top TFs to show (by variance across groups).
        Ignored if tfs is provided.
    tfs
        Specific TFs to include.
    cluster_rows
        Cluster TF rows.
    cluster_cols
        Cluster cell group columns.
    cmap
        Colormap.
    center
        Center value for diverging colormap.
    show
        Display figure.
    save
        Save figure.
    return_fig
        Return Figure.
    figsize
        Figure size.
    **kwargs
        Passed to seaborn.clustermap.

    Returns
    -------
    Figure or None.

    Examples
    --------
    >>> import deepscenic as ds
    >>> model = ds.tl.load_model("model.pt")    >>> ds.pl.tf_activity_clustermap(model, mdata, groupby="celltype")    """
    import pandas as pd
    import seaborn as sns
    import torch

    # Get TF activity from model
    with torch.no_grad():
        rna_data = mdata.mod["rna"].X
        if hasattr(rna_data, "toarray"):
            rna_data = rna_data.toarray()

        tf_indices = model.vae.tf_indices.cpu().numpy()
        tf_expression = rna_data[:, tf_indices]

        # Get z_tf from encoder
        tf_tensor = torch.FloatTensor(tf_expression)
        _, mu, _ = model.vae.encoder(tf_tensor, use_mean=True)
        z_tf = mu.squeeze(-1).cpu().numpy()

    # Get TF names
    tf_names = model.tf_names

    # Create DataFrame with TF activity per cell
    df_tf = pd.DataFrame(z_tf, index=mdata.obs_names, columns=tf_names)

    # Group by annotation and compute mean
    groups = mdata.obs[groupby]
    df_grouped = df_tf.groupby(groups).mean().T

    # Select TFs to display
    if tfs is not None:
        df_grouped = df_grouped.loc[df_grouped.index.isin(tfs)]
    elif top_n_tfs is not None and top_n_tfs < len(df_grouped):
        # Select top TFs by variance across groups
        variances = df_grouped.var(axis=1)
        top_tfs = variances.nlargest(top_n_tfs).index
        df_grouped = df_grouped.loc[top_tfs]

    # Create clustermap
    g = sns.clustermap(
        df_grouped,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        cmap=cmap,
        center=center,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.1, 0.05),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        **kwargs,
    )
    g.ax_heatmap.set_xlabel(groupby)
    g.ax_heatmap.set_ylabel("TF")
    g.fig.suptitle(f"TF Activity by {groupby}", y=1.02)

    savefig_or_show("tf_activity_clustermap", show=show, save=save)

    if return_fig:
        return g.fig
    return None
