"""Tools for deepSCENIC (training, inference, perturbation)."""

from ._grn import (
    extract_e1_matrix,
    extract_e2_matrix,
    extract_grn,
    get_gene_regulators,
    get_tf_targets,
)
from ._inference import to_latent
from ._model import DeepSCENICModel, load_model
from ._perturbation import simulate_multi_perturbation, simulate_perturbation
from ._train import train

__all__ = [
    # Training
    "train",
    # Model
    "DeepSCENICModel",
    "load_model",
    # Inference
    "to_latent",
    # Perturbation
    "simulate_perturbation",
    "simulate_multi_perturbation",
    # GRN extraction
    "extract_grn",
    "extract_e1_matrix",
    "extract_e2_matrix",
    "get_tf_targets",
    "get_gene_regulators",
]
