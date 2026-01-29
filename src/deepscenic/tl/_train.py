"""Main training loop for deepSCENIC."""

from __future__ import annotations

import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm.auto import tqdm

from ..models import DeepSCENICVAE, MotifNet
from ._dataloaders import build_cell_dataloader, build_sequence_dataloader, build_test_sequence_dataloader
from ._logging import get_logger
from ._loss import compute_test_chromosome_loss, compute_total_loss, f1_score_binary
from ._model import DeepSCENICModel, load_model
from ._training_state import (
    EarlyStopping,
    ModelConfig,
    TrainingHistory,
)

if TYPE_CHECKING:
    import mudata as md

log = logging.getLogger("deepscenic.tl")

# Global thread pool for async checkpoint saving (single worker to serialize saves)
_checkpoint_executor: ThreadPoolExecutor | None = None


def _get_checkpoint_executor() -> ThreadPoolExecutor:
    """Get or create the global checkpoint executor."""
    global _checkpoint_executor
    if _checkpoint_executor is None:
        _checkpoint_executor = ThreadPoolExecutor(max_workers=1)
    return _checkpoint_executor


def _save_checkpoint_data(
    path: Path,
    vae_state_dict: dict,
    tf2rnet_state_dict: dict,
    enformer_state_dict: dict,
    adj_E1: torch.Tensor,
    config: ModelConfig,
    tf_names: list[str],
    gene_names: list[str],
    region_names: list[str],
    vae_metadata: dict,
    history: TrainingHistory | None,
    optimizer_state_dict: dict,
    scheduler_state_dict: dict | None,
    epoch: int,
    best_loss: float,
    adj_E1_test: torch.Tensor | None,
    is_custom_sequence_model: bool,
) -> None:
    """Save model checkpoint asynchronously in a background thread.

    This builds the same format as DeepSCENICModel.save() but takes
    pre-copied state dicts for async saving during training.
    """
    data = {
        "vae_state_dict": vae_state_dict,
        "tf2rnet_state_dict": tf2rnet_state_dict,
        "enformer_state_dict": enformer_state_dict,
        "adj_E1": adj_E1,
        "config": config.to_dict(),
        "tf_names": tf_names,
        "gene_names": gene_names,
        "region_names": region_names,
        # VAE metadata for reconstruction
        **vae_metadata,
        # Custom sequence model flag
        "is_custom_sequence_model": is_custom_sequence_model,
        # Training state
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "epoch": epoch,
        "best_loss": best_loss,
        "adj_E1_test": adj_E1_test,
    }
    if history is not None:
        data["history"] = history.to_dict()
    torch.save(data, path)


def _init_enformer(device: str, emb_len: int) -> torch.nn.Module:
    """Initialize Enformer model."""
    from enformer_pytorch import Enformer

    enformer = Enformer.from_pretrained(
        "EleutherAI/enformer-official-rough",
        dropout_rate=0.1,
        target_length=emb_len,  # default TF2rNet embedding
    )
    enformer.to(device)
    return enformer  # type: ignore[no-any-return]


def _is_enformer(model: torch.nn.Module) -> bool:
    """Check if model is an Enformer instance.

    Parameters
    ----------
    model
        Sequence embedding model.

    Returns
    -------
    bool
        True if model is Enformer, False otherwise.
    """
    return type(model).__name__ == "Enformer"


def _get_sequence_embeddings(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    bottleneck_size: int,
    emb_len: int,
) -> torch.Tensor:
    """Get embeddings from sequence model.

    Handles both Enformer (which requires special calling convention) and
    custom models that return flattened embeddings directly.

    Parameters
    ----------
    model
        Sequence embedding model (Enformer or custom nn.Module).
    sequences
        Input sequences with shape (batch, seq_len, 4).
    bottleneck_size
        Embedding dimension per position.
    emb_len
        Output sequence length (positions).

    Returns
    -------
    torch.Tensor
        Flattened embeddings with shape (batch, bottleneck_size * emb_len).
    """
    if _is_enformer(model):
        # Enformer returns (batch, emb_len, bottleneck_size)
        output = model(sequences, return_only_embeddings=True)
        # Flatten: (batch, emb_len, bottleneck) -> (batch, bottleneck * emb_len)
        return output.reshape(-1, bottleneck_size * emb_len).float()  # type: ignore[no-any-return]
    else:
        # Custom models return flat embeddings directly
        return model(sequences).float()  # type: ignore[no-any-return]


def _init_e1_cache(
    sequence_model: torch.nn.Module,
    tf2rnet: MotifNet,
    seq_dataloader: torch.utils.data.DataLoader,
    device: str,
    config: ModelConfig,
) -> torch.Tensor:
    """Initialize E1 cache by running all sequences through sequence model + TF2rNet."""
    sequence_model.eval()
    tf2rnet.eval()

    n_regions = len(seq_dataloader.dataset)  # type: ignore[arg-type]
    adj_E1 = torch.zeros(n_regions, tf2rnet.n_tfs, device=device)

    with torch.no_grad():
        for sequences, seq_idx in tqdm(seq_dataloader, desc="Initializing E1"):
            sequences = sequences.to(device)
            emb = _get_sequence_embeddings(sequence_model, sequences, config.bottleneck_size, config.emb_len)
            tf_pred = tf2rnet(emb)
            adj_E1[seq_idx] = tf_pred

    return adj_E1


def _init_e1_test_cache(
    sequence_model: torch.nn.Module,
    tf2rnet: MotifNet,
    test_seq_dataloader: torch.utils.data.DataLoader,
    device: str,
    config: ModelConfig,
) -> torch.Tensor:
    """Initialize E1 cache for TEST regions (no augmentation)."""
    sequence_model.eval()
    tf2rnet.eval()

    n_test_regions = len(test_seq_dataloader.dataset)  # type: ignore[arg-type]
    adj_E1_test = torch.zeros(n_test_regions, tf2rnet.n_tfs, device=device)

    with torch.no_grad():
        for sequences, local_idx in tqdm(test_seq_dataloader, desc="Caching E1 (test)"):
            sequences = sequences.to(device)
            emb = _get_sequence_embeddings(sequence_model, sequences, config.bottleneck_size, config.emb_len)
            tf_pred = tf2rnet(emb)
            adj_E1_test[local_idx] = tf_pred

    return adj_E1_test


class SequenceIterator:
    """Iterator over sequence dataloader that auto-resets when exhausted."""

    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self._iterator = iter(dataloader)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next batch, auto-resetting if exhausted."""
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)


def _evaluate_test_chromosomes(
    vae: DeepSCENICVAE,
    adj_E1_test: torch.Tensor,
    adj_E1_cache: torch.Tensor,
    test_region_indices: torch.Tensor,
    test_gene_indices: torch.Tensor,
    test_cell_loader: torch.utils.data.DataLoader,
    device: str,
    *,
    loss_atac: str = "cosine",
    loss_rna: str = "mae",
    alpha: float = 1e-2,
    atac_tau: float = 1.0,
    rna_tau: float = 1.0,
    ppi_device: str | None = None,
    use_ppi: bool = True,
) -> dict[str, float]:
    """Evaluate ATAC and RNA reconstruction on test chromosomes.

    Computes TF activity from encoder (with PPI modulation if enabled),
    then predicts ATAC for test regions using adj_E1_test and RNA
    reconstruction for test genes.

    This matches legacy behavior where the full VAE forward pass (including
    PPI network) is used to compute test chromosome reconstruction.
    """
    vae.eval()
    metrics: dict[str, float] = {"total": 0.0, "atac_recon": 0.0, "e1_l1": 0.0, "rna_recon": 0.0}
    n_batches = 0

    with torch.no_grad():
        for batch in test_cell_loader:
            x_rna = batch["rna"].to(device)
            x_atac = batch["atac"].to(device)

            # Extract TF expression
            x_rna_tfs = x_rna[:, vae.tf_indices]

            # Apply PPI modulation if enabled (matches legacy vae.predict behavior)
            if vae.use_ppi and use_ppi and vae.ppi is not None:
                from ..models._ppi import build_ppi_batch, extract_tf_weights

                batch_ppi = build_ppi_batch(
                    x_rna, vae.ppi_genes_idx, vae.ppi_edge_index, vae.ppi_node_ids, device
                )
                ppi_out = vae.ppi(batch_ppi.x, batch_ppi.edge_index, batch_ppi.node_ids)
                x_rna_ppi = extract_tf_weights(
                    ppi_out,
                    x_rna.shape[0],
                    len(vae.ppi_genes_idx),
                    vae.ppi_tfs_idx_keys,
                    vae.ppi_tfs_idx_values,
                    device,
                )
                x_rna_tfs = x_rna_tfs * x_rna_ppi

            # Encode TF activity
            z_tf, _, _ = vae.encoder(x_rna_tfs, use_mean=True)

            # Predict test region activity via E1_test
            enh_act_test = z_tf @ adj_E1_test.T  # (n_cells, n_test_regions)

            # Decode ATAC for test regions
            x_atac_rec_test = vae.decoder_atac(enh_act_test)

            # Compute RNA reconstruction (full forward pass with cached E1)
            output = vae(
                x_rna,
                adj_E1_cache,
                use_ppi=use_ppi,
                use_mean=True,
                ppi_device=ppi_device,
            )

            # Compute loss
            losses = compute_test_chromosome_loss(
                x_atac=x_atac,
                x_atac_rec_test=x_atac_rec_test,
                adj_E1_test=adj_E1_test,
                test_region_indices=test_region_indices,
                loss_atac=loss_atac,
                alpha=alpha,
                atac_tau=atac_tau,
                x_rna=x_rna,
                x_rna_rec=output.x_rna_rec,
                test_gene_indices=test_gene_indices,
                loss_rna=loss_rna,
                rna_tau=rna_tau,
            )

            for key in metrics:
                metrics[key] += losses[key].item()
            n_batches += 1

    return {k: v / n_batches for k, v in metrics.items()}


def train(
    mdata: md.MuData,
    *,
    config: ModelConfig | None = None,
    # Training loop parameters
    epochs: int = 100,
    batch_size: int = 64,
    seq_batch_size: int = 1000,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    use_scheduler: bool = False,
    # Loss settings
    loss_rna: str = "mae",
    loss_atac: str = "cosine",
    dropout_mask_rna: bool = False,
    dropout_mask_atac: bool = False,
    beta: float = 1e-2,
    alpha: float = 1e-2,
    gamma: float = 1.0,
    rna_tau: float = 1.0,
    atac_tau: float = 1.0,
    # Runtime settings
    device: str = "cuda",
    ppi_device: str | None = None,
    num_workers: int = 0,
    # Data settings
    batch_key: str | None = None,
    balance_class: bool = False,
    class_key: str | None = None,
    balance_dars: bool = False,
    # Logging and checkpointing
    logger: str = "dict",
    log_dir: str = "./runs",
    logger_kwargs: dict | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 0,
    save_best_checkpoints: bool | None = None,
    resume_from: str | None = None,
    early_stopping_patience: int | None = None,
) -> DeepSCENICModel:
    """Train deepSCENIC model on multiome data.

    Phase 1 of the training workflow. Uses train cells and train features.
    Trains the full model: sequence model (Enformer), MotifNet (TF2rNet),
    and VAE encoder/decoders. PPInet is frozen during this phase.

    For PPI training (Phase 4), use :func:`train_ppi` after E2 finetuning.

    Parameters
    ----------
    mdata
        MuData with preprocessed RNA + ATAC data.
    config
        Model architecture configuration. Use this to set n_hidden, use_ppi,
        binary_atac, and sequence model settings. If None, uses defaults.
    epochs
        Total training epochs.
    batch_size
        Cells per batch.
    seq_batch_size
        Sequences per batch for E1 cache updates.
    lr
        Learning rate for all trainable components (VAE, MotifNet, Enformer).
    weight_decay
        Weight decay for optimizer.
    use_scheduler
        Whether to use CosineAnnealingLR scheduler.
    loss_rna
        RNA loss type: 'mse', 'mae', 'cosine'.
    loss_atac
        ATAC loss type: 'mse', 'mae', 'bce', 'cosine'.
    dropout_mask_rna
        Whether to mask RNA loss on zeros.
    dropout_mask_atac
        Whether to mask ATAC loss on zeros.
    beta
        KL divergence weight.
    alpha
        E1 sparsity + PPI weight.
    gamma
        E2 sparsity weight.
    rna_tau
        RNA reconstruction weight.
    atac_tau
        ATAC reconstruction weight.
    device
        Training device ('cuda' or 'cpu').
    ppi_device
        Separate device for PPI network (optional).
    num_workers
        Number of workers for parallel data loading.
    batch_key
        Column in obs for batch correction.
    balance_class
        Whether to use weighted sampling for class balance.
    class_key
        Column in obs for class balancing.
    balance_dars
        Whether to upweight DAR regions during sequence sampling (1.5x weight).
        Requires DARs to be marked via ``ds.pp.mark_dars()`` first.
    logger
        Logging backend: 'dict', 'tensorboard', 'wandb'.
    log_dir
        Directory for tensorboard/wandb logs.
    logger_kwargs
        Additional keyword arguments passed to the logger backend.
    checkpoint_dir
        Directory for checkpoints (None = no checkpoints).
    checkpoint_every
        Save checkpoint every N epochs. Set to 0 to disable periodic checkpointing.
    save_best_checkpoints
        Save checkpoint when validation loss improves. If None (default),
        enabled when checkpoint_dir is set. Saves to 'best.pt' in checkpoint_dir.
    resume_from
        Path to checkpoint to resume from.
    early_stopping_patience
        Stop if no improvement for N epochs (None = disabled).

    Returns
    -------
    DeepSCENICModel
        Trained model (self-contained, ready for inference).

    Notes
    -----
    A genome must be registered before training using
    ``ds.genome.register_genome(fasta_file)``. The regions for sequence
    extraction are derived from ``mdata.mod["atac"].var_names``.

    Examples
    --------
    Simple training with defaults:

    >>> import deepscenic as ds
    >>> ds.genome.register_genome("/path/to/hg38.fa")
    >>> mdata = ds.read("preprocessed.h5mu")
    >>> model = ds.tl.train(mdata, epochs=100, device="cuda")

    Training with custom architecture:

    >>> from deepscenic.tl import ModelConfig
    >>> config = ModelConfig(n_hidden=256, use_ppi=True)
    >>> model = ds.tl.train(mdata, config=config, epochs=100, lr=1e-4)

    Training with custom sequence model:

    >>> config = ModelConfig(
    ...     sequence_model=my_custom_model,
    ...     bottleneck_size=256,
    ...     emb_len=60,
    ... )
    >>> model = ds.tl.train(mdata, config=config, epochs=100)
    """
    # Build model config from provided config or defaults
    model_config = config if config is not None else ModelConfig()

    # Setup logging — use phase-specific subdirectory so TensorBoard shows
    # each phase as a separate named run instead of merging them into "."
    _phase_log_dir = str(Path(log_dir) / "train")
    _logger_kwargs = {"log_dir": _phase_log_dir}
    if logger_kwargs:
        _logger_kwargs.update(logger_kwargs)
    training_logger = get_logger(logger, **_logger_kwargs)
    # Log both architecture and training params
    hyperparams = {
        **model_config.to_dict(),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "loss_rna": loss_rna,
        "loss_atac": loss_atac,
        "beta": beta,
        "alpha": alpha,
        "gamma": gamma,
    }
    training_logger.log_hyperparams(hyperparams)

    # Extract metadata from MuData
    rna = mdata.mod["rna"]
    tf_mask = rna.var["is_tf"]
    tf_names = rna.var_names[tf_mask].tolist()
    gene_names = list(rna.var_names)
    region_names = list(mdata.mod["atac"].var_names)

    n_tfs = len(tf_names)
    n_genes = len(gene_names)
    n_regions = len(region_names)

    # Compute TF indices from is_tf mask
    tf_indices = torch.tensor(np.where(tf_mask)[0])

    # Validate split columns exist
    if "split" not in rna.var.columns:
        raise ValueError("RNA modality missing 'split' column in var. Run ds.pp.split_features_by_chromosome() first.")
    atac = mdata.mod["atac"]
    if "split" not in atac.var.columns:
        raise ValueError("ATAC modality missing 'split' column in var. Run ds.pp.split_features_by_chromosome() first.")

    # Compute gene_indices for reconstruction loss: only TRAIN genes (split='train' or 'both')
    # This matches legacy behavior where reconstruction loss only evaluates on train features,
    # while E2 matrix can span ALL genes (E2 sparsity loss applies to all links).
    gene_split = rna.var["split"]
    # Include genes with split='train' or 'both' (TFs get 'both' by default)
    train_gene_mask = gene_split.isin(["train", "both"])
    gene_indices = torch.tensor(np.where(train_gene_mask)[0])
    log.info(f"Using {len(gene_indices)}/{n_genes} genes for reconstruction (train + 'both' splits)")

    # Compute region_indices for ATAC reconstruction loss: only TRAIN regions
    region_split = atac.var["split"]
    train_region_mask = region_split == "train"
    region_indices = torch.tensor(np.where(train_region_mask)[0])
    log.info(f"Using {len(region_indices)}/{n_regions} regions for ATAC reconstruction (train split)")

    # Compute test_gene_indices for test chromosome RNA reconstruction evaluation
    test_gene_mask = gene_split == "test"
    test_gene_indices = torch.tensor(np.where(test_gene_mask)[0])
    log.info(f"Using {len(test_gene_indices)}/{n_genes} genes for test chromosome RNA evaluation")

    # Get r2g sparse matrix info (convert CSR to COO for indices)
    r2g_coo = mdata.uns["r2g"]["matrix"].tocoo()
    r2g_indices = torch.tensor(np.array([r2g_coo.row, r2g_coo.col]))
    r2g_distances = torch.tensor(r2g_coo.data).float()

    # Get PPI info if present
    ppi_edge_index = None
    ppi_genes_idx = None
    ppi_tfs_idx_keys = None
    ppi_tfs_idx_values = None
    use_ppi = model_config.use_ppi and "ppi_edge_index" in mdata.uns

    if use_ppi:
        ppi_edge_index = torch.tensor(mdata.uns["ppi_edge_index"])
        ppi_genes_idx = torch.tensor(mdata.uns["ppi_genes_idx"])
        ppi_tfs_idx_keys = torch.tensor(mdata.uns["ppi_tfs_idx_keys"])
        ppi_tfs_idx_values = torch.tensor(mdata.uns["ppi_tfs_idx_values"])

    # Batch correction
    n_batches = 0
    if batch_key is not None:
        n_batches = int(mdata.obs[batch_key].nunique())

    # Initialize models
    vae = DeepSCENICVAE(
        n_tfs=n_tfs,
        n_genes=n_genes,
        n_regions=n_regions,
        r2g_indices=r2g_indices,
        r2g_distances=r2g_distances,
        tf_indices=tf_indices,
        gene_indices=gene_indices,
        region_indices=region_indices,
        ppi_edge_index=ppi_edge_index,
        ppi_genes_idx=ppi_genes_idx,
        ppi_tfs_idx_keys=ppi_tfs_idx_keys,
        ppi_tfs_idx_values=ppi_tfs_idx_values,
        n_hidden=model_config.n_hidden,
        use_ppi=use_ppi,
        binary_atac=model_config.binary_atac,
        n_batches=n_batches,
    ).to(device)

    # Initialize sequence models (custom or default Enformer)
    if model_config.sequence_model is not None:
        # User provided custom sequence model
        tf2rnet = MotifNet(
            n_tfs=n_tfs,
            bottleneck_size=model_config.bottleneck_size,
            emb_len=model_config.emb_len,
        ).to(device)
        sequence_model = model_config.sequence_model.to(device)
    else:
        # Default: use Enformer
        tf2rnet = MotifNet(
            n_tfs=n_tfs,
            bottleneck_size=model_config.bottleneck_size,
            emb_len=model_config.emb_len,
        ).to(device)
        sequence_model = _init_enformer(device, model_config.emb_len)

    # Build dataloaders
    train_cell_loader = build_cell_dataloader(
        mdata,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        balance_class=balance_class,
        class_key=class_key,
        num_workers=num_workers,
    )

    test_cell_loader = build_cell_dataloader(
        mdata,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Sequence dataloader (for TF2rNet training)
    # Regions are derived from ATAC var_names (chr:start-end format)
    region_names = list(mdata.mod["atac"].var_names)

    # Get DAR indices if balance_dars is enabled
    dar_indices = None
    if balance_dars and "atac" in mdata.mod:
        atac_var = mdata.mod["atac"].var
        if "is_dar" in atac_var.columns:
            dar_indices = np.where(atac_var["is_dar"].values)[0]

    train_seq_loader = build_sequence_dataloader(
        regions=region_names,
        batch_size=seq_batch_size,
        shuffle=True,
        shift_augs=(-3, 3),
        rc_aug=True,
        context_length=model_config.seq_len,
        num_workers=num_workers,
        balance_dars=balance_dars,
        dar_indices=dar_indices,
    )

    # Build test sequence dataloader for test chromosome evaluation
    test_region_mask = mdata.mod["atac"].var["split"] == "test"
    n_test_regions = test_region_mask.sum()
    if n_test_regions == 0:
        raise ValueError(
            "No test regions found (atac.var['split'] == 'test'). "
            "Run ds.pp.split_features_by_chromosome() first."
        )

    test_seq_loader, test_region_indices = build_test_sequence_dataloader(
        mdata,
        batch_size=seq_batch_size,
        context_length=model_config.seq_len,
        num_workers=num_workers,
    )
    test_region_indices = test_region_indices.to(device)
    test_gene_indices = test_gene_indices.to(device)

    log.info(f"Test chromosome evaluation: {n_test_regions} test regions")

    # Initialize E1 cache
    adj_E1_cache = _init_e1_cache(sequence_model, tf2rnet, train_seq_loader, device, model_config)

    # Initialize test E1 cache (always computed, no pretrained cache for test regions)
    adj_E1_test_cache = _init_e1_test_cache(sequence_model, tf2rnet, test_seq_loader, device, model_config)

    # Initialize sequence iterator for dynamic E1 updates (legacy behavior)
    seq_iterator = SequenceIterator(train_seq_loader)

    n_cells = mdata.n_obs
    log.info(f"Starting training: {epochs} epochs, {n_cells} cells, {n_genes} genes, {n_tfs} TFs, {n_regions} regions")

    optimizer = Adam(
        [
            {"params": vae.parameters(), "lr": lr},
            {"params": tf2rnet.parameters(), "lr": lr},
            {"params": sequence_model.parameters(), "lr": lr},
        ],
        weight_decay=weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) if use_scheduler else None

    # History and early stopping
    history = TrainingHistory()
    early_stopping = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience else None

    # Resolve save_best_checkpoints
    if save_best_checkpoints is None:
        save_best_checkpoints = checkpoint_dir is not None
    if save_best_checkpoints and checkpoint_dir is None:
        raise ValueError("save_best_checkpoints=True requires checkpoint_dir")

    # Log checkpoint configuration
    if checkpoint_dir is not None:
        if checkpoint_every > 0:
            log.info(f"Periodic checkpointing: every {checkpoint_every} epochs to {checkpoint_dir}")
        if save_best_checkpoints:
            log.info("Best checkpoint saving enabled: saving to 'best.pt' on improvement")
        if checkpoint_every <= 0 and not save_best_checkpoints:
            log.info("checkpoint_dir set but no checkpointing enabled (checkpoint_every=0, save_best_checkpoints=False)")
    else:
        log.info("Checkpointing disabled (checkpoint_dir=None)")

    best_loss = float("inf")

    # Resume from checkpoint
    start_epoch = 0
    if resume_from is not None:
        log.info(f"Resuming from {resume_from}")
        resumed_model = load_model(resume_from, device=device)

        # Load state dicts into our freshly initialized models
        vae.load_state_dict(resumed_model.vae.state_dict())
        tf2rnet.load_state_dict(resumed_model.tf2rnet.state_dict())
        sequence_model.load_state_dict(resumed_model.enformer.state_dict())

        # Restore E1 cache
        adj_E1_cache = resumed_model.adj_E1.to(device)

        # Restore training state if present
        if resumed_model.training_state is not None:
            optimizer.load_state_dict(resumed_model.training_state.optimizer_state_dict)
            if resumed_model.training_state.scheduler_state_dict and scheduler is not None:
                scheduler.load_state_dict(resumed_model.training_state.scheduler_state_dict)
            start_epoch = resumed_model.training_state.epoch + 1
            best_loss = resumed_model.training_state.best_loss
            # Load test E1 cache from checkpoint if available
            if resumed_model.training_state.adj_E1_test is not None:
                adj_E1_test_cache = resumed_model.training_state.adj_E1_test.to(device)
        else:
            log.warning(
                "Checkpoint does not contain training state. "
                "Starting from epoch 0 with fresh optimizer state."
            )

        # Restore history if present
        if resumed_model.history is not None:
            history = resumed_model.history

    # Freeze PPInet - it's trained separately via train_ppi()
    if vae.ppi is not None:
        for param in vae.ppi.parameters():
            param.requires_grad = False

    # Training loop
    for epoch in range(start_epoch, epochs):
        # Set all models to train mode
        vae.train()
        tf2rnet.train()
        sequence_model.train()

        epoch_metrics: dict[str, float] = {
            "total": 0.0,
            "rna_recon": 0.0,
            "atac_recon": 0.0,
            "kl_div": 0.0,
            "e1_l1": 0.0,
            "e2_l1": 0.0,
            "ppi_reg": 0.0,
        }
        if model_config.binary_atac:
            epoch_metrics["f1_atac"] = 0.0
        n_batches_seen = 0

        pbar = tqdm(train_cell_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            x_rna = batch["rna"].to(device)
            x_atac = batch["atac"].to(device)

            # Update E1 with fresh sequence predictions (legacy behavior)
            # Clone persistent cache, then insert fresh predictions WITH gradients
            # This allows gradients to flow back through Enformer/MotifNet
            sequences, seq_idx = seq_iterator.next()
            sequences = sequences.to(device)

            # Clone persistent cache for this iteration
            adj_E1_batch = adj_E1_cache.clone()

            # Compute fresh TF predictions WITH gradients (trains Enformer + MotifNet)
            emb = _get_sequence_embeddings(sequence_model, sequences, model_config.bottleneck_size, model_config.emb_len)
            tf_pred = tf2rnet(emb)
            adj_E1_batch[seq_idx] = tf_pred  # Gradients flow through these predictions

            # VAE forward (PPI is frozen during train(), trained via train_ppi())
            optimizer.zero_grad()
            output = vae(
                x_rna,
                adj_E1_batch,
                use_ppi=False,
                use_mean=False,
                ppi_device=ppi_device,
            )

            # Compute losses (sparsity losses always included in train())
            losses = compute_total_loss(
                x_rna=x_rna,
                x_atac=x_atac,
                x_rna_rec=output.x_rna_rec,
                x_atac_rec=output.x_atac_rec,
                mu=output.mu,
                logvar=output.logvar,
                adj_E1_batch=adj_E1_batch,
                adj_E2=vae.adj_E2,
                r2g_distances=vae.r2g_distances,  # type: ignore[arg-type]
                x_rna_ppi=output.x_rna_ppi,
                gene_indices=vae.gene_indices,  # type: ignore[arg-type]
                region_indices=vae.region_indices,  # type: ignore[arg-type]
                loss_rna=loss_rna,
                loss_atac=loss_atac,
                dropout_mask_rna=dropout_mask_rna,
                dropout_mask_atac=dropout_mask_atac,
                beta=beta,
                alpha=alpha,
                gamma=gamma,
                rna_tau=rna_tau,
                atac_tau=atac_tau,
                use_ppi=False,
                include_e1_sparsity=True,
                ppi_phase=False,
                seq_idx=seq_idx,  # E1 sparsity only on sampled batch (legacy behavior)
            )
            total_loss = losses["total"]

            # Backward
            total_loss.backward()
            optimizer.step()

            # Update persistent E1 cache with fresh predictions (no_grad for persistence)
            with torch.no_grad():
                adj_E1_cache[seq_idx] = tf_pred.detach()

            # Compute F1 score for binary ATAC predictions
            if model_config.binary_atac:
                f1 = f1_score_binary(output.x_atac_rec, x_atac)
                epoch_metrics["f1_atac"] += f1.item()

            # Accumulate metrics
            for key in epoch_metrics:
                if key in losses:
                    epoch_metrics[key] += losses[key].item()
            epoch_metrics["total"] += total_loss.item()
            n_batches_seen += 1

            pbar.set_postfix(loss=total_loss.item())

        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches_seen

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Log metrics
        history.log("train", epoch_metrics)
        training_logger.log_metrics({f"train/{k}": v for k, v in epoch_metrics.items()}, step=epoch)

        # Validation
        vae.eval()
        tf2rnet.eval()
        sequence_model.eval()

        # Recompute test E1 each epoch (legacy behavior)
        with torch.no_grad():
            for sequences, local_idx in test_seq_loader:
                sequences = sequences.to(device)
                emb = _get_sequence_embeddings(sequence_model, sequences, model_config.bottleneck_size, model_config.emb_len)
                tf_pred = tf2rnet(emb)
                adj_E1_test_cache[local_idx] = tf_pred

        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in test_cell_loader:
                x_rna = batch["rna"].to(device)
                x_atac = batch["atac"].to(device)

                # Validation (PPI is frozen during train())
                output = vae(
                    x_rna,
                    adj_E1_cache,
                    use_ppi=False,
                    use_mean=True,
                    ppi_device=ppi_device,
                )

                losses = compute_total_loss(
                    x_rna=x_rna,
                    x_atac=x_atac,
                    x_rna_rec=output.x_rna_rec,
                    x_atac_rec=output.x_atac_rec,
                    mu=output.mu,
                    logvar=output.logvar,
                    adj_E1_batch=adj_E1_cache,
                    adj_E2=vae.adj_E2,
                    r2g_distances=vae.r2g_distances,  # type: ignore[arg-type]
                    x_rna_ppi=output.x_rna_ppi,
                    gene_indices=vae.gene_indices,  # type: ignore[arg-type]
                    region_indices=vae.region_indices,  # type: ignore[arg-type]
                    loss_rna=loss_rna,
                    loss_atac=loss_atac,
                    beta=beta,
                    alpha=alpha,
                    gamma=gamma,
                    use_ppi=False,
                    ppi_phase=False,
                )

                val_loss += losses["total"].item()
                n_val_batches += 1

        val_loss /= n_val_batches
        history.log("val", {"total": val_loss})
        training_logger.log_metrics({"val/total": val_loss}, step=epoch)

        # Validation on test chromosomes (generalization metric)
        test_chrom_metrics = _evaluate_test_chromosomes(
            vae=vae,
            adj_E1_test=adj_E1_test_cache,
            adj_E1_cache=adj_E1_cache,
            test_region_indices=test_region_indices,
            test_gene_indices=test_gene_indices,
            test_cell_loader=test_cell_loader,
            device=device,
            loss_atac=loss_atac,
            loss_rna=loss_rna,
            alpha=alpha,
            atac_tau=atac_tau,
            rna_tau=rna_tau,
            ppi_device=ppi_device,
            use_ppi=False,
        )

        history.log("val_chrom", test_chrom_metrics)
        training_logger.log_metrics(
            {f"val_chrom/{k}": v for k, v in test_chrom_metrics.items()},
            step=epoch,
        )

        # Use test chromosome ATAC reconstruction for early stopping
        early_stop_loss = test_chrom_metrics["atac_recon"]

        # Early stopping (based on test chromosome ATAC loss)
        if early_stopping is not None:
            if early_stopping(early_stop_loss):
                log.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Best checkpoint saving based on test chromosome ATAC reconstruction (async to not block training)
        if save_best_checkpoints and (test_chrom_metrics["atac_recon"] < best_loss) and (checkpoint_dir is not None):
            best_loss = test_chrom_metrics["atac_recon"]
            best_path = Path(checkpoint_dir) / "best.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            # Deep copy state dicts for async save (training continues modifying them)
            vae_metadata = {
                "n_tfs": vae.n_tfs,
                "n_genes": vae.n_genes,
                "n_regions": vae.n_regions,
                "n_hidden": vae.n_hidden,
                "n_batches": vae.n_batches,
                "tf_indices": vae.tf_indices,
                "gene_indices": vae.gene_indices,
                "region_indices": vae.region_indices,
                "r2g_indices": vae.r2g_indices,
                "r2g_distances": vae.r2g_distances,
                "use_ppi": vae.use_ppi,
                "ppi_edge_index": getattr(vae, "ppi_edge_index", None),
                "ppi_genes_idx": getattr(vae, "ppi_genes_idx", None),
                "ppi_tfs_idx_keys": getattr(vae, "ppi_tfs_idx_keys", None),
                "ppi_tfs_idx_values": getattr(vae, "ppi_tfs_idx_values", None),
            }
            _get_checkpoint_executor().submit(
                _save_checkpoint_data,
                best_path,
                copy.deepcopy(vae.state_dict()),
                copy.deepcopy(tf2rnet.state_dict()),
                copy.deepcopy(sequence_model.state_dict()),
                adj_E1_cache.clone(),
                model_config,
                tf_names,
                gene_names,
                region_names,
                vae_metadata,
                copy.deepcopy(history),
                copy.deepcopy(optimizer.state_dict()),
                copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
                epoch,
                best_loss,
                adj_E1_test_cache.clone(),
                not _is_enformer(sequence_model),
            )
            log.info(f"New best val_chrom/atac_recon: {test_chrom_metrics['atac_recon']:.6f} - saving to {best_path}")

        # Periodic checkpointing (async to not block training)
        if checkpoint_dir is not None and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = Path(checkpoint_dir) / f"epoch_{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            # Deep copy state dicts for async save (training continues modifying them)
            vae_metadata = {
                "n_tfs": vae.n_tfs,
                "n_genes": vae.n_genes,
                "n_regions": vae.n_regions,
                "n_hidden": vae.n_hidden,
                "n_batches": vae.n_batches,
                "tf_indices": vae.tf_indices,
                "gene_indices": vae.gene_indices,
                "region_indices": vae.region_indices,
                "r2g_indices": vae.r2g_indices,
                "r2g_distances": vae.r2g_distances,
                "use_ppi": vae.use_ppi,
                "ppi_edge_index": getattr(vae, "ppi_edge_index", None),
                "ppi_genes_idx": getattr(vae, "ppi_genes_idx", None),
                "ppi_tfs_idx_keys": getattr(vae, "ppi_tfs_idx_keys", None),
                "ppi_tfs_idx_values": getattr(vae, "ppi_tfs_idx_values", None),
            }
            _get_checkpoint_executor().submit(
                _save_checkpoint_data,
                checkpoint_path,
                copy.deepcopy(vae.state_dict()),
                copy.deepcopy(tf2rnet.state_dict()),
                copy.deepcopy(sequence_model.state_dict()),
                adj_E1_cache.clone(),
                model_config,
                tf_names,
                gene_names,
                region_names,
                vae_metadata,
                copy.deepcopy(history),
                copy.deepcopy(optimizer.state_dict()),
                copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
                epoch,
                best_loss,
                adj_E1_test_cache.clone(),
                not _is_enformer(sequence_model),
            )

    # Wait for any pending async checkpoint saves to complete
    if checkpoint_dir is not None:
        executor = _get_checkpoint_executor()
        executor.shutdown(wait=True)
        # Reset the global executor for future use
        global _checkpoint_executor
        _checkpoint_executor = None

    # Close logger
    training_logger.close()

    # Return trained model
    vae.eval()
    tf2rnet.eval()
    sequence_model.eval()

    log.info("Training completed")

    return DeepSCENICModel(
        vae=vae,
        tf2rnet=tf2rnet,
        enformer=sequence_model,
        adj_E1=adj_E1_cache,
        config=model_config,
        tf_names=tf_names,
        gene_names=gene_names,
        region_names=region_names,
        history=history,
    )


def finetune_e2(
    model: DeepSCENICModel,
    mdata: md.MuData,
    *,
    cell_split: str = "test",
    feature_split: str = "train",
    epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    lr_patience: int = 50,
    reinit_e2: bool = True,
    gamma: float = 1.0,
    loss_rna: str = "mae",
    dropout_mask_rna: bool = False,
    use_ppi: bool = True,
    device: str = "cuda",
    num_workers: int = 0,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 0,
    save_best_checkpoints: bool | None = None,
    logger: str = "dict",
    log_dir: str = "./runs",
    logger_kwargs: dict | None = None,
) -> DeepSCENICModel:
    """Finetune the E2 (region→gene) matrix.

    Phases 2 and 3 of the training workflow. Freezes all model parameters
    except E2 and trains with ReduceLROnPlateau scheduler.

    Parameters
    ----------
    model
        Trained DeepSCENICModel from ds.tl.train().
    mdata
        MuData with preprocessed RNA + ATAC data.
    cell_split
        Which cells to use: "train" or "test".
        - Phase 2: cell_split="test" (cross-cell-type generalization)
        - Phase 3: cell_split="train" (test chromosome)
    feature_split
        Which features to use: "train" or "test".
        - Phase 2: feature_split="train" (train regions/genes)
        - Phase 3: feature_split="test" (test regions/genes from test chromosome)
    epochs
        Finetuning epochs.
    batch_size
        Cells per batch.
    lr
        Learning rate.
    lr_patience
        Epochs before reducing LR if no improvement.
    reinit_e2
        If True, reinitialize E2 to near-zero before finetuning.
    gamma
        E2 sparsity weight.
    loss_rna
        RNA loss type: 'mse', 'mae', 'cosine'.
    dropout_mask_rna
        Whether to mask RNA loss on zeros.
    use_ppi
        Whether to use PPI modulation (if available in model).
    device
        Training device.
    num_workers
        Number of workers for parallel data loading.
    checkpoint_dir
        Directory for checkpoints (None = no checkpoints).
    checkpoint_every
        Save checkpoint every N epochs. Set to 0 to disable periodic checkpointing.
    save_best_checkpoints
        Save checkpoint when validation loss improves. If None (default),
        enabled when checkpoint_dir is set.
    logger
        Logging backend: 'dict', 'tensorboard', 'wandb'.
    log_dir
        Directory for logs.
    logger_kwargs
        Additional keyword arguments passed to the logger backend.

    Returns
    -------
    DeepSCENICModel
        Model with finetuned E2 matrix.

    Examples
    --------
    Phase 2: E2 finetuning on cross-cell-type validation:

    >>> model = ds.tl.train(mdata, epochs=100)
    >>> model = ds.tl.finetune_e2(model, mdata, cell_split="test", epochs=500)

    Phase 3: E2 finetuning on test chromosome:

    >>> model = ds.tl.finetune_e2(
    ...     model, mdata,
    ...     cell_split="train",
    ...     feature_split="test",
    ...     epochs=500,
    ... )
    """
    # Determine if PPI should be active during finetuning
    use_ppi_finetune = model.vae.use_ppi and use_ppi
    if use_ppi_finetune:
        log.info("PPInet active (but frozen) during E2 finetuning")

    # Setup logging — use phase-specific subdirectory so TensorBoard shows
    # each phase as a separate named run
    _phase_name = f"finetune_e2_cell-{cell_split}_feat-{feature_split}"
    _phase_log_dir = str(Path(log_dir) / _phase_name)
    _logger_kwargs = {"log_dir": _phase_log_dir}
    if logger_kwargs:
        _logger_kwargs.update(logger_kwargs)
    training_logger = get_logger(logger, **_logger_kwargs)
    hyperparams = {
        "cell_split": cell_split,
        "feature_split": feature_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "gamma": gamma,
        "loss_rna": loss_rna,
        "reinit_e2": reinit_e2,
    }
    training_logger.log_hyperparams(hyperparams)

    # Move model to device
    vae = model.vae.to(device)
    adj_E1 = model.adj_E1.to(device)

    # Freeze all parameters except adj_E2
    for name, param in vae.named_parameters():
        if name == "adj_E2":
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Compute link mask and gene_indices based on feature_split
    # This ensures we only train E2 weights for the appropriate region-gene links,
    # matching legacy behavior where separate r2g matrices were used per phase.
    rna_var = mdata.mod["rna"].var
    atac_var = mdata.mod["atac"].var

    if "split" not in rna_var.columns:
        raise ValueError("RNA modality missing 'split' column in var. Run ds.pp.split_features_by_chromosome() first.")
    if "split" not in atac_var.columns:
        raise ValueError("ATAC modality missing 'split' column in var. Run ds.pp.split_features_by_chromosome() first.")

    # Determine which genes and regions match the feature_split
    if feature_split == "train":
        # Include 'train' and 'both' (TFs get 'both' by default)
        gene_mask = rna_var["split"].isin(["train", "both"]).values
        region_mask = (atac_var["split"] == "train").values
    elif feature_split == "test":
        gene_mask = (rna_var["split"] == "test").values
        region_mask = (atac_var["split"] == "test").values
    else:
        raise ValueError(f"Invalid feature_split: {feature_split}. Must be 'train' or 'test'.")

    # Compute gene_indices for loss computation
    gene_indices = torch.tensor(np.where(gene_mask)[0], device=device)
    log.info(f"Using {len(gene_indices)}/{len(rna_var)} genes for loss (feature_split='{feature_split}')")

    # Compute E2 link mask: only links where BOTH region AND gene match the split
    # r2g_indices: (2, n_links) - row 0 = region indices, row 1 = gene indices
    r2g_region_idx = vae.r2g_indices[0].cpu().numpy()  # type: ignore
    r2g_gene_idx = vae.r2g_indices[1].cpu().numpy()  # type: ignore

    # For each link, check if both region AND gene are in the target split
    link_region_in_split = region_mask[r2g_region_idx]
    link_gene_in_split = gene_mask[r2g_gene_idx]
    e2_link_mask = torch.tensor(link_region_in_split & link_gene_in_split, device=device)

    n_active_links = e2_link_mask.sum().item()
    log.info(f"Using {n_active_links}/{len(e2_link_mask)} E2 links for training (feature_split='{feature_split}')")

    # Optionally reinitialize ONLY the active E2 links (matching legacy behavior)
    if reinit_e2:
        with torch.no_grad():
            vae.adj_E2[e2_link_mask] = 1e-8

    # Register gradient mask hook to only update matching E2 links
    # This prevents data leakage by not training E2 on cross-split links
    def e2_grad_hook(grad: torch.Tensor) -> torch.Tensor:
        return grad * e2_link_mask.float()

    e2_hook_handle = vae.adj_E2.register_hook(e2_grad_hook)

    # Build dataloaders WITHOUT feature filtering
    # The VAE needs all genes for TF indexing and E2 computation.
    # The feature_split is applied via gene_indices (loss) and e2_link_mask (gradients).
    train_cell_loader = build_cell_dataloader(
        mdata,
        split=cell_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        feature_split=None,  # Don't filter - VAE needs all genes
    )

    # For validation, use the opposite cell split
    val_cell_split = "train" if cell_split == "test" else "test"
    val_cell_loader = build_cell_dataloader(
        mdata,
        split=val_cell_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        feature_split=None,  # Don't filter - VAE needs all genes
    )

    log.info(f"E2 finetuning: cells={cell_split}, features={feature_split}")

    # Single optimizer for E2 only
    optimizer = Adam([vae.adj_E2], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=lr_patience, factor=0.5)

    # History tracking
    history = TrainingHistory()
    best_loss = float("inf")
    best_e2 = vae.adj_E2.data.clone()

    # Resolve save_best_checkpoints
    if save_best_checkpoints is None:
        save_best_checkpoints = checkpoint_dir is not None
    if save_best_checkpoints and checkpoint_dir is None:
        raise ValueError("save_best_checkpoints=True requires checkpoint_dir")

    # Log checkpoint configuration
    if checkpoint_dir is not None:
        if checkpoint_every > 0:
            log.info(f"Periodic checkpointing: every {checkpoint_every} epochs to {checkpoint_dir}")
        if save_best_checkpoints:
            log.info("Best checkpoint saving enabled: saving to 'finetune_best.pt' on improvement")
        if checkpoint_every <= 0 and not save_best_checkpoints:
            log.info("checkpoint_dir set but no checkpointing enabled (checkpoint_every=0, save_best_checkpoints=False)")
    else:
        log.info("Checkpointing disabled (checkpoint_dir=None)")

    log.info(f"Starting E2 finetuning: {epochs} epochs")

    # Import loss functions
    from ._loss import e2_sparsity_loss, reconstruction_loss

    # Training loop
    for epoch in range(epochs):
        vae.train()

        epoch_metrics = {"total": 0.0, "rna_recon": 0.0, "e2_l1": 0.0}
        n_batches_seen = 0

        pbar = tqdm(train_cell_loader, desc=f"Finetune E2 {epoch + 1}/{epochs}")
        for batch in pbar:
            x_rna = batch["rna"].to(device)

            optimizer.zero_grad()

            # VAE forward (PPInet is frozen but may modulate TF inputs)
            output = vae(
                x_rna,
                adj_E1,
                use_ppi=use_ppi_finetune,
                use_mean=False,
            )

            # Compute RNA reconstruction loss only on genes matching feature_split
            # This prevents data leakage by not evaluating reconstruction on test genes during train finetune
            loss_rec_rna = reconstruction_loss(
                output.x_rna_rec[:, gene_indices],
                x_rna[:, gene_indices],
                loss_type=loss_rna,
                dropout_mask=dropout_mask_rna,
            )

            # E2 sparsity only on active links (matching feature_split)
            loss_e2_sparse = e2_sparsity_loss(
                vae.adj_E2[e2_link_mask], vae.r2g_distances[e2_link_mask]  # type: ignore[arg-type]
            )
            total_loss = loss_rec_rna + loss_e2_sparse * gamma

            total_loss.backward()
            optimizer.step()

            # Accumulate metrics
            epoch_metrics["total"] += total_loss.item()
            epoch_metrics["rna_recon"] += loss_rec_rna.item()
            epoch_metrics["e2_l1"] += loss_e2_sparse.item()
            n_batches_seen += 1

            pbar.set_postfix(loss=total_loss.item())

        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches_seen

        # Log metrics
        history.log("train", epoch_metrics)
        training_logger.log_metrics({f"train/{k}": v for k, v in epoch_metrics.items()}, step=epoch)

        # Validation
        vae.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_cell_loader:
                x_rna = batch["rna"].to(device)

                output = vae(
                    x_rna,
                    adj_E1,
                    use_ppi=use_ppi_finetune,
                    use_mean=True,
                )

                loss_rec_rna = reconstruction_loss(
                    output.x_rna_rec[:, gene_indices],
                    x_rna[:, gene_indices],
                    loss_type=loss_rna,
                    dropout_mask=dropout_mask_rna,
                )
                loss_e2_sparse = e2_sparsity_loss(
                    vae.adj_E2[e2_link_mask], vae.r2g_distances[e2_link_mask]  # type: ignore[arg-type]
                )

                val_loss += (loss_rec_rna + loss_e2_sparse * gamma).item()
                n_val_batches += 1

        val_loss /= n_val_batches
        history.log("val", {"total": val_loss})
        training_logger.log_metrics({"val/total": val_loss}, step=epoch)

        # Step scheduler
        scheduler.step(val_loss)

        # Track best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_e2 = vae.adj_E2.data.clone()
            # Save best checkpoint
            if save_best_checkpoints and (checkpoint_dir is not None):
                best_path = Path(checkpoint_dir) / "finetune_best.pt"
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "adj_E2": vae.adj_E2.data,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "history": history.to_dict(),
                        "best_loss": best_loss,
                    },
                    best_path,
                )
                log.info(f"New best val loss: {val_loss:.6f} - saved to {best_path}")

        # Periodic checkpointing
        if checkpoint_dir is not None and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = Path(checkpoint_dir) / f"finetune_epoch_{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "adj_E2": vae.adj_E2.data,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "history": history.to_dict(),
                    "best_loss": best_loss,
                },
                checkpoint_path,
            )

    # Remove gradient hook
    e2_hook_handle.remove()

    # Close logger
    training_logger.close()

    # Restore best E2
    vae.adj_E2.data = best_e2
    vae.eval()

    log.info("E2 finetuning completed")

    # Return updated model with new history
    return DeepSCENICModel(
        vae=vae,
        tf2rnet=model.tf2rnet,
        enformer=model.enformer,
        adj_E1=adj_E1,
        config=model.config,
        tf_names=model.tf_names,
        gene_names=model.gene_names,
        region_names=model.region_names,
        history=history,
    )


def train_ppi(
    model: DeepSCENICModel,
    mdata: md.MuData,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    seq_batch_size: int = 1000,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    # Loss settings
    loss_rna: str = "mae",
    loss_atac: str = "cosine",
    beta: float = 1e-2,
    rna_tau: float = 1.0,
    atac_tau: float = 1.0,
    # Runtime settings
    device: str = "cuda",
    ppi_device: str | None = None,
    num_workers: int = 0,
    # Logging and checkpointing
    logger: str = "dict",
    log_dir: str = "./runs",
    logger_kwargs: dict | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 0,
    save_best_checkpoints: bool | None = None,
    early_stopping_patience: int | None = None,
) -> DeepSCENICModel:
    """Train the PPI network while keeping VAE and E2 frozen.

    Phase 4 of the training workflow. Only the PPInet is trainable.
    Uses train cells and train features.

    Parameters
    ----------
    model
        Trained DeepSCENICModel from ds.tl.train() or ds.tl.finetune_e2().
    mdata
        MuData with preprocessed RNA + ATAC data.
    epochs
        Training epochs.
    batch_size
        Cells per batch.
    seq_batch_size
        Sequences per batch for E1 cache updates.
    lr
        Learning rate for PPI network.
    weight_decay
        Weight decay for optimizer.
    loss_rna
        RNA loss type: 'mse', 'mae', 'cosine'.
    loss_atac
        ATAC loss type: 'mse', 'mae', 'bce', 'cosine'.
    beta
        KL divergence weight.
    rna_tau
        RNA reconstruction weight.
    atac_tau
        ATAC reconstruction weight.
    device
        Training device ('cuda' or 'cpu').
    ppi_device
        Separate device for PPI network (optional).
    num_workers
        Number of workers for parallel data loading.
    logger
        Logging backend: 'dict', 'tensorboard', 'wandb'.
    log_dir
        Directory for tensorboard/wandb logs.
    logger_kwargs
        Additional keyword arguments passed to the logger backend.
    checkpoint_dir
        Directory for checkpoints (None = no checkpoints).
    checkpoint_every
        Save checkpoint every N epochs. Set to 0 to disable periodic checkpointing.
    save_best_checkpoints
        Save checkpoint when validation loss improves.
    early_stopping_patience
        Stop if no improvement for N epochs (None = disabled).

    Returns
    -------
    DeepSCENICModel
        Model with trained PPI network.

    Examples
    --------
    >>> model = ds.tl.train(mdata, epochs=100)
    >>> model = ds.tl.finetune_e2(model, mdata, cell_split="test", epochs=500)
    >>> model = ds.tl.finetune_e2(model, mdata, cell_split="train", feature_split="test", epochs=500)
    >>> model = ds.tl.train_ppi(model, mdata, epochs=100)
    """
    if not model.vae.use_ppi or model.vae.ppi is None:
        raise ValueError(
            "Model does not have PPI network enabled. "
            "Create model with config=ModelConfig(use_ppi=True) and ensure PPI data is in mdata.uns."
        )

    log.info("Starting Phase 4: PPI training (VAE and E2 frozen)")

    # Setup logging — use phase-specific subdirectory so TensorBoard shows
    # each phase as a separate named run
    _phase_log_dir = str(Path(log_dir) / "train_ppi")
    _logger_kwargs = {"log_dir": _phase_log_dir}
    if logger_kwargs:
        _logger_kwargs.update(logger_kwargs)
    training_logger = get_logger(logger, **_logger_kwargs)
    hyperparams = {
        "phase": "ppi",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "loss_rna": loss_rna,
        "loss_atac": loss_atac,
        "beta": beta,
    }
    training_logger.log_hyperparams(hyperparams)

    # Move model to device
    vae = model.vae.to(device)
    tf2rnet = model.tf2rnet.to(device)
    sequence_model = model.enformer.to(device)
    adj_E1_cache = model.adj_E1.to(device)

    # Freeze everything except PPInet
    for param in vae.parameters():
        param.requires_grad = False
    for param in vae.ppi.parameters():
        param.requires_grad = True
    for param in tf2rnet.parameters():
        param.requires_grad = False
    for param in sequence_model.parameters():
        param.requires_grad = False

    # Build dataloaders
    train_cell_loader = build_cell_dataloader(
        mdata,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_cell_loader = build_cell_dataloader(
        mdata,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Build sequence dataloader for E1 cache updates
    region_names = list(mdata.mod["atac"].var_names)
    train_seq_loader = build_sequence_dataloader(
        regions=region_names,
        batch_size=seq_batch_size,
        shuffle=True,
        shift_augs=(-3, 3),
        rc_aug=True,
        context_length=model.config.seq_len,
        num_workers=num_workers,
    )
    seq_iterator = SequenceIterator(train_seq_loader)

    # Optimizer for PPI only
    optimizer = Adam(vae.ppi.parameters(), lr=lr, weight_decay=weight_decay)

    # History and early stopping
    history = TrainingHistory()
    early_stopping = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience else None

    # Resolve save_best_checkpoints
    if save_best_checkpoints is None:
        save_best_checkpoints = checkpoint_dir is not None
    if save_best_checkpoints and checkpoint_dir is None:
        raise ValueError("save_best_checkpoints=True requires checkpoint_dir")

    best_loss = float("inf")

    log.info(f"Starting PPI training: {epochs} epochs")

    # Training loop
    for epoch in range(epochs):
        vae.train()

        epoch_metrics: dict[str, float] = {
            "total": 0.0,
            "rna_recon": 0.0,
            "atac_recon": 0.0,
            "kl_div": 0.0,
            "ppi_reg": 0.0,
        }
        n_batches_seen = 0

        pbar = tqdm(train_cell_loader, desc=f"PPI Train {epoch + 1}/{epochs}")
        for batch in pbar:
            x_rna = batch["rna"].to(device)
            x_atac = batch["atac"].to(device)

            # Update E1 cache with fresh sequence predictions
            sequences, seq_idx = seq_iterator.next()
            sequences = sequences.to(device)
            with torch.no_grad():
                emb = _get_sequence_embeddings(sequence_model, sequences, model.config.bottleneck_size, model.config.emb_len)
                tf_pred = tf2rnet(emb)
                adj_E1_cache[seq_idx] = tf_pred

            # VAE forward with PPI active
            optimizer.zero_grad()
            output = vae(
                x_rna,
                adj_E1_cache,
                use_ppi=True,
                use_mean=False,
                ppi_device=ppi_device,
            )

            # Compute losses (PPI phase - no sparsity losses)
            losses = compute_total_loss(
                x_rna=x_rna,
                x_atac=x_atac,
                x_rna_rec=output.x_rna_rec,
                x_atac_rec=output.x_atac_rec,
                mu=output.mu,
                logvar=output.logvar,
                adj_E1_batch=adj_E1_cache,
                adj_E2=vae.adj_E2,
                r2g_distances=vae.r2g_distances,  # type: ignore[arg-type]
                x_rna_ppi=output.x_rna_ppi,
                gene_indices=vae.gene_indices,  # type: ignore[arg-type]
                region_indices=vae.region_indices,  # type: ignore[arg-type]
                loss_rna=loss_rna,
                loss_atac=loss_atac,
                beta=beta,
                alpha=0.0,  # No sparsity in PPI phase
                gamma=0.0,  # No sparsity in PPI phase
                rna_tau=rna_tau,
                atac_tau=atac_tau,
                use_ppi=True,
                include_e1_sparsity=False,
                ppi_phase=True,
            )
            total_loss = losses["total"]

            # Backward
            total_loss.backward()
            optimizer.step()

            # Accumulate metrics
            for key in epoch_metrics:
                if key in losses:
                    epoch_metrics[key] += losses[key].item()
            epoch_metrics["total"] += total_loss.item()
            n_batches_seen += 1

            pbar.set_postfix(loss=total_loss.item())

        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches_seen

        # Log metrics
        history.log("train", epoch_metrics)
        training_logger.log_metrics({f"train/{k}": v for k, v in epoch_metrics.items()}, step=epoch)

        # Validation
        vae.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in test_cell_loader:
                x_rna = batch["rna"].to(device)
                x_atac = batch["atac"].to(device)

                output = vae(
                    x_rna,
                    adj_E1_cache,
                    use_ppi=True,
                    use_mean=True,
                    ppi_device=ppi_device,
                )

                losses = compute_total_loss(
                    x_rna=x_rna,
                    x_atac=x_atac,
                    x_rna_rec=output.x_rna_rec,
                    x_atac_rec=output.x_atac_rec,
                    mu=output.mu,
                    logvar=output.logvar,
                    adj_E1_batch=adj_E1_cache,
                    adj_E2=vae.adj_E2,
                    r2g_distances=vae.r2g_distances,  # type: ignore[arg-type]
                    x_rna_ppi=output.x_rna_ppi,
                    gene_indices=vae.gene_indices,  # type: ignore[arg-type]
                    region_indices=vae.region_indices,  # type: ignore[arg-type]
                    loss_rna=loss_rna,
                    loss_atac=loss_atac,
                    beta=beta,
                    alpha=0.0,
                    gamma=0.0,
                    use_ppi=True,
                    ppi_phase=True,
                )

                val_loss += losses["total"].item()
                n_val_batches += 1

        val_loss /= n_val_batches
        history.log("val", {"total": val_loss})
        training_logger.log_metrics({"val/total": val_loss}, step=epoch)

        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_loss):
                log.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Best checkpoint saving
        if save_best_checkpoints and (val_loss < best_loss) and (checkpoint_dir is not None):
            best_loss = val_loss
            best_path = Path(checkpoint_dir) / "ppi_best.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "ppi_state_dict": vae.ppi.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history.to_dict(),
                    "best_loss": best_loss,
                },
                best_path,
            )
            log.info(f"New best val loss: {val_loss:.6f} - saved to {best_path}")

        # Periodic checkpointing
        if checkpoint_dir is not None and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = Path(checkpoint_dir) / f"ppi_epoch_{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "ppi_state_dict": vae.ppi.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history.to_dict(),
                    "best_loss": best_loss,
                },
                checkpoint_path,
            )

    # Close logger
    training_logger.close()

    vae.eval()
    log.info("PPI training completed")

    # Return updated model
    return DeepSCENICModel(
        vae=vae,
        tf2rnet=tf2rnet,
        enformer=sequence_model,
        adj_E1=adj_E1_cache,
        config=model.config,
        tf_names=model.tf_names,
        gene_names=model.gene_names,
        region_names=model.region_names,
        history=history,
    )
