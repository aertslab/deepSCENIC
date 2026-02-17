"""Training diagnostic plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

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
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] | None = None,
    log_scale: bool = True,
    ncols: int = 3,
) -> Figure | None:
    """
    Plot training loss curves with one subplot per metric.

    Each metric gets its own subplot, with train and test curves shown together.
    This makes it easier to compare train/test performance and identify issues
    like overfitting.

    Parameters
    ----------
    history
        Training history as DataFrame, dict, or DeepSCENICModel containing history.
    metrics
        Specific metrics to plot. If None, plot all available.
    show
        Whether to display the figure.
    save
        Path to save figure, or True for default path.
    return_fig
        Whether to return the Figure object.
    figsize
        Figure dimensions as (width, height). If None, auto-calculated based on
        number of metrics.
    log_scale
        Whether to use logarithmic scale for y-axis.
    ncols
        Number of columns in subplot grid.

    Returns
    -------
    Figure or None depending on show and return_fig parameters.

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
    train_metrics: dict[str, list[float]] = {}
    test_metrics: dict[str, list[float]] = {}

    if isinstance(history, dict) and "train" in history:
        train_metrics = history.get("train", {})
        test_metrics = history.get("test", {})
    elif isinstance(history, dict):
        # Already flat dict with train_/test_ prefixes
        for key, values in history.items():
            if key.startswith("train_"):
                train_metrics[key[6:]] = values
            elif key.startswith("test_"):
                test_metrics[key[5:]] = values
            elif key != "epoch":
                train_metrics[key] = values
    elif isinstance(history, pd.DataFrame):
        for col in history.columns:
            if col.startswith("train_"):
                train_metrics[col[6:]] = history[col].tolist()
            elif col.startswith("test_"):
                test_metrics[col[5:]] = history[col].tolist()
            elif col != "epoch":
                train_metrics[col] = history[col].tolist()

    # Get all unique metric names
    all_metric_names = set(train_metrics.keys()) | set(test_metrics.keys())

    if metrics is not None:
        all_metric_names = {m for m in all_metric_names if m in metrics}

    if not all_metric_names:
        raise ValueError("No metrics found in history")

    # Sort metrics for consistent ordering
    metric_names = sorted(all_metric_names)
    n_metrics = len(metric_names)

    # Calculate grid layout
    nrows = (n_metrics + ncols - 1) // ncols

    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, metric in enumerate(metric_names):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        has_data = False

        # Plot train curve
        if metric in train_metrics:
            ax.plot(train_metrics[metric], label="train", color=COLORS.get("tf", "#1f77b4"))
            has_data = True

        # Plot test curve
        if metric in test_metrics:
            ax.plot(
                test_metrics[metric],
                label="test",
                color=COLORS.get("gene", "#ff7f0e"),
                linestyle="--",
            )
            has_data = True

        if has_data:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(metric)
            ax.legend(loc="upper right", fontsize=8)

            if log_scale:
                ax.set_yscale("log")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for idx in range(n_metrics, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    savefig_or_show("loss_curves", show=show, save=save)

    if return_fig:
        return fig
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
    Histogram of tf2r/r2g weight distributions.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    which
        'tf2r', 'r2g', or 'both'.
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
    >>> model = ds.tl.load_model("model.pt")
    >>> ds.pl.sparsity_histogram(model)
    """
    import torch

    with torch.no_grad():
        tf2r_vals = model.adj_tf2r.flatten().cpu().numpy()
        r2g_vals = model.vae.adj_r2g.abs().flatten().cpu().numpy()

    if which == "both":
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].hist(tf2r_vals, bins=bins, alpha=0.7, color=COLORS["tf"])
        axes[0].set_xlabel("Weight")
        axes[0].set_ylabel("Count")
        tf2r_sparsity = (tf2r_vals < threshold).mean()
        axes[0].set_title(f"TF->Region (tf2r)\nSparsity: {tf2r_sparsity:.1%}")

        axes[1].hist(r2g_vals, bins=bins, alpha=0.7, color=COLORS["gene"])
        axes[1].set_xlabel("Weight")
        axes[1].set_ylabel("Count")
        r2g_sparsity = (r2g_vals < threshold).mean()
        axes[1].set_title(f"Region->Gene (r2g)\nSparsity: {r2g_sparsity:.1%}")

        plt.tight_layout()
        savefig_or_show("sparsity_histogram", show=show, save=save)

        if return_fig:
            return fig
        return None
    else:
        fig, ax = setup_axes(ax, figsize=(figsize[0] // 2, figsize[1]))
        vals = tf2r_vals if which == "tf2r" else r2g_vals
        color = COLORS["tf"] if which == "tf2r" else COLORS["gene"]

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
    >>> model = ds.tl.load_model("model.pt")
    >>> ds.pl.enhancer_activity_histogram(model, mdata)
    """
    import torch

    # Get enhancer activity from model forward pass
    with torch.no_grad():
        # Get device from model
        device = model.adj_tf2r.device

        rna_data = mdata.mod["rna"].X
        if hasattr(rna_data, "toarray"):
            rna_data = rna_data.toarray()

        # Get TF expression and compute enhancer activity
        tf_indices = model.vae.tf_indices.cpu().numpy()
        tf_expression = rna_data[:, tf_indices]

        # Enhancer activity = TF expression @ E1
        tf_tensor = torch.tensor(tf_expression, dtype=torch.float32, device=device)

        # Get z_tf from encoder
        _, mu, _ = model.vae.encoder(tf_tensor, use_mean=True)
        z_tf = mu.squeeze(-1)

        # Compute enhancer activity: z_tf @ E1
        enh_act = (z_tf @ model.adj_tf2r).cpu().numpy()

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
    scores: pd.DataFrame,
    *,
    xlabel: str = "Cell Type",
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

    Use to visualize cell-type-specific TF activity patterns. Requires
    pre-computed TF activity scores from ``ds.tl.compute_tf_activity_scores()``.

    Parameters
    ----------
    scores
        TF activity scores as DataFrame with TF names as index and cell groups
        as columns. Typically from ``compute_tf_activity_scores()``.
    xlabel
        Label for x-axis (cell group axis). Default: "Cell Type"
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
    >>> model = ds.tl.load_model("model.pt")
    >>>
    >>> # Compute TF activity scores
    >>> enh_activity = ds.tl.compute_celltype_enhancer_activity(model, mdata, "celltype")
    >>> active_enh = ds.tl.identify_active_enhancers(enh_activity)
    >>> tf_scores = ds.tl.compute_tf_activity_scores(model, mdata, "celltype", active_enh)
    >>>
    >>> # Plot clustermap
    >>> ds.pl.tf_activity_clustermap(tf_scores, xlabel="celltype")
    """
    import pandas as pd
    import seaborn as sns

    df_grouped = scores.copy()
    if not isinstance(df_grouped.index, pd.Index):
        raise ValueError("scores must have TF names as index")

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
    g.ax_heatmap.set_xlabel(xlabel)
    g.ax_heatmap.set_ylabel("TF")
    g.figure.suptitle(f"TF Activity by {xlabel}", y=1.02)

    savefig_or_show("tf_activity_clustermap", show=show, save=save)

    if return_fig:
        return g.figure
    return None
