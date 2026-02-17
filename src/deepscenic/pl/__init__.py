"""Plotting functions for deepSCENIC."""

from ._colors import get_colors, set_colors
from ._embedding import embedding_umap
from ._utils import get_cmap_colors, parse_region, savefig_or_show, setup_axes
from .genomics import arc_plot, genome_browser
from .grn import (
    heatmap_celltype_activity,
    heatmap_grn,
    heatmap_r2g,
    network_gene_regulators,
    network_grn,
    network_tf_targets,
    upset_active_enhancers,
)
from .perturbation import (
    dotplot_perturbation,
    heatmap_perturbation,
    waterfall_perturbation,
)
from .sequence import ism_heatmap, logo_attribution, logo_motif
from .training import (
    enhancer_activity_histogram,
    loss_curves,
    sparsity_histogram,
    tf_activity_clustermap,
)

__all__ = [
    # Utilities
    "setup_axes",
    "savefig_or_show",
    "get_cmap_colors",
    "parse_region",
    "set_colors",
    "get_colors",
    # Embedding Visualization
    "embedding_umap",
    # GRN
    "heatmap_r2g",
    "heatmap_grn",
    "network_grn",
    "network_tf_targets",
    "network_gene_regulators",
    # GRN - Cell-type specific
    "upset_active_enhancers",
    "heatmap_celltype_activity",
    # Sequence
    "logo_attribution",
    "logo_motif",
    "ism_heatmap",
    # Perturbation
    "waterfall_perturbation",
    "heatmap_perturbation",
    "dotplot_perturbation",
    # Genomics
    "genome_browser",
    "arc_plot",
    # Training / Diagnostics
    "loss_curves",
    "sparsity_histogram",
    "enhancer_activity_histogram",
    "tf_activity_clustermap",
]
