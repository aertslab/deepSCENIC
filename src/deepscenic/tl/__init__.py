"""Tools for deepSCENIC (training, inference, perturbation)."""

from ._grn import (
    build_grn_for_tfs,
    compute_celltype_r2g,
    compute_celltype_tf2g,
    compute_celltype_tf2r,
    compute_tf_activity_scores,
    extract_grn,
    extract_r2g_matrix,
    extract_tf2r_matrix,
    get_gene_regulators,
    get_region_info,
    get_tf_targets,
    identify_active_enhancers,
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
from ._sequence import ISMResult, in_silico_mutagenesis
from ._train import finetune_r2g, train
from ._training_state import (
    EarlyStopping,
    ModelConfig,
    TrainingHistory,
)

__all__ = [
    # Training stages
    "train",  # Phase 1: Full model training
    "finetune_r2g",  # Phase 2: r2g finetuning
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
    # Sequence interpretation
    "ISMResult",
    "in_silico_mutagenesis",
    # GRN extraction
    "extract_grn",
    "extract_tf2r_matrix",
    "extract_r2g_matrix",
    "compute_celltype_tf2r",
    "compute_celltype_r2g",
    "compute_celltype_tf2g",
    "get_tf_targets",
    "get_gene_regulators",
    "get_region_info",
    # Cell-type aware GRN
    "identify_active_enhancers",
    "compute_tf_activity_scores",
    "build_grn_for_tfs",
]
