"""Tools for deepSCENIC (training, inference, perturbation)."""

from ._grn import (
    build_grn_for_tfs,
    compute_celltype_enhancer_activity,
    compute_tf_activity_scores,
    extract_grn,
    extract_r2g_matrix,
    extract_tf2r_matrix,
    get_gene_regulators,
    get_tf_targets,
    identify_active_enhancers,
    identify_key_tfs,
)
from ._inference import to_latent
from ._model import (
    DeepSCENICModel,
    TrainingState,
    load_legacy_data,
    load_legacy_model,
    load_model,
)
from ._perturbation import (
    process_perturbation_results,
    simulate_multi_perturbation,
    simulate_perturbation,
)
from ._train import finetune_r2g, train
from ._training_state import (
    EarlyStopping,
    ModelConfig,
    TrainingHistory,
)

__all__ = [
    # Training stages
    "train",  # Phase 1: Full model training
    "finetune_r2g",  # Phases 2-3: E2 finetuning
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
    "process_perturbation_results",
    # GRN extraction
    "extract_grn",
    "extract_tf2r_matrix",
    "extract_r2g_matrix",
    "get_tf_targets",
    "get_gene_regulators",
    # Cell-type aware GRN
    "compute_celltype_enhancer_activity",
    "identify_active_enhancers",
    "compute_tf_activity_scores",
    "identify_key_tfs",
    "build_grn_for_tfs",
]
