"""Training diagnostic visualizations."""

from ._diagnostics import latent_umap, loss_curves, sparsity_histogram

__all__ = [
    "loss_curves",
    "latent_umap",
    "sparsity_histogram",
]
