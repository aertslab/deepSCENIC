"""Tools for deepSCENIC (training, inference, perturbation)."""

from ._grn import (
    extract_e1_matrix,
    extract_e2_matrix,
    extract_grn,
    get_gene_regulators,
    get_tf_targets,
)
from ._inference import to_latent
from ._model import (
    DeepSCENICModel,
    TrainingState,
    load_legacy_data,
    load_legacy_model,
    load_model,
)
from ._perturbation import simulate_multi_perturbation, simulate_perturbation
from ._train import finetune_e2, train
from ._training_state import (
    EarlyStopping,
    ModelConfig,
    TrainingHistory,
)

__all__ = [
    # Training stages
    "train",  # Phase 1: Full model training
    "finetune_e2",  # Phases 2-3: E2 finetuning
    # Model
    "DeepSCENICModel",
    "TrainingState",
    "load_model",
    "load_legacy_data",
    "load_legacy_model",
    # Config
    "ModelConfig",
    "TrainingHistory",
    "EarlyStopping",
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
