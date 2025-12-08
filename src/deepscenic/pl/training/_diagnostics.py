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

from .._utils import savefig_or_show, setup_axes


def loss_curves(
    history: pd.DataFrame | dict,
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
        Training history DataFrame or dict with columns/keys
        like 'loss', 'val_loss', 'loss_rec_rna', etc.
    metrics
        Specific metrics to plot. If None, plot all.
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
    log_scale
        Use log scale for y-axis.

    Returns
    -------
    Axes, Figure, or None
    """
    import pandas as pd

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
    Axes, Figure, or None
    """
    import torch

    with torch.no_grad():
        E1_vals = model.adj_E1.flatten().cpu().numpy()
        E2_vals = model.vae.adj_E2.abs().flatten().cpu().numpy()

    if which == "both":
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].hist(E1_vals, bins=bins, alpha=0.7, color="#E64B35")
        axes[0].set_xlabel("Weight")
        axes[0].set_ylabel("Count")
        e1_sparsity = (E1_vals < threshold).mean()
        axes[0].set_title(f"E1 (TF->Region)\nSparsity: {e1_sparsity:.1%}")

        axes[1].hist(E2_vals, bins=bins, alpha=0.7, color="#4DBBD5")
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
        color = "#E64B35" if which == "E1" else "#4DBBD5"

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
    Axes, Figure, or None
    """
    import torch
    from sklearn.preprocessing import StandardScaler

    try:
        from umap import UMAP
    except ImportError as e:
        raise ImportError("umap-learn is required for latent_umap. " "Install with: pip install umap-learn") from e

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
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=categories, cmap="viridis", alpha=0.7, s=10, **kwargs)
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
