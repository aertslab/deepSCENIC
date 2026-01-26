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
    PretrainedModel,
    load_legacy_data,
    load_legacy_model,
    load_model,
    load_pretrained,
)
from ._perturbation import simulate_multi_perturbation, simulate_perturbation
from ._train import finetune, pretrain, train
from ._training_state import (
    Checkpoint,
    EarlyStopping,
    FinetuneConfig,
    PretrainConfig,
    TrainingConfig,
    TrainingHistory,
)

__all__ = [
    # Training stages
    "pretrain",
    "train",
    "finetune",
    # Model
    "DeepSCENICModel",
    "PretrainedModel",
    "load_model",
    "load_pretrained",
    "load_legacy_data",
    "load_legacy_model",
    # Config
    "PretrainConfig",
    "TrainingConfig",
    "FinetuneConfig",
    "TrainingHistory",
    "Checkpoint",
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
