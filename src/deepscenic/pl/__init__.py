"""Plotting functions for deepSCENIC."""

from ._colors import get_colors, set_colors
from ._utils import savefig_or_show, setup_axes
from .genomics import arc_plot, genome_browser
from .grn import (
    heatmap_e1,
    heatmap_e2,
    heatmap_grn,
    network_gene_regulators,
    network_grn,
    network_tf_targets,
)
from .perturbation import (
    dotplot_perturbation,
    heatmap_perturbation,
    volcano_perturbation,
)
from .sequence import ism_heatmap, logo_attribution, logo_motif
from .training import latent_umap, loss_curves, sparsity_histogram

__all__ = [
    # Utilities
    "setup_axes",
    "savefig_or_show",
    "set_colors",
    "get_colors",
    # GRN
    "heatmap_e1",
    "heatmap_e2",
    "heatmap_grn",
    "network_grn",
    "network_tf_targets",
    "network_gene_regulators",
    # Sequence
    "logo_attribution",
    "logo_motif",
    "ism_heatmap",
    # Perturbation
    "volcano_perturbation",
    "heatmap_perturbation",
    "dotplot_perturbation",
    # Genomics
    "genome_browser",
    "arc_plot",
    # Training
    "loss_curves",
    "sparsity_histogram",
    "latent_umap",
]
