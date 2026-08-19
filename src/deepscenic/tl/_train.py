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


def _resolve_device(device: str | None) -> str:
    """Resolve device, auto-detecting CUDA availability if None."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    motifnet_state_dict: dict,
    enformer_state_dict: dict,
    adj_tf2r: torch.Tensor,
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
    is_custom_sequence_model: bool,
) -> None:
    """Save model checkpoint asynchronously."""
    data = {
        "vae_state_dict": vae_state_dict,
        "motifnet_state_dict": motifnet_state_dict,
        "sequence_model_state_dict": enformer_state_dict,
        "adj_tf2r": adj_tf2r,
        "config": config.to_dict(),
        "tf_names": tf_names,
        "gene_names": gene_names,
        "region_names": region_names,
        **vae_metadata,
        "is_custom_sequence_model": is_custom_sequence_model,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "epoch": epoch,
        "best_loss": best_loss,
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
        target_length=emb_len,
    )
    enformer.to(device)  # type: ignore
    return enformer  # type: ignore[no-any-return]


def _is_enformer(model: torch.nn.Module) -> bool:
    """Check if model is an Enformer instance."""
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


def _init_tf2r_cache(
    sequence_model: torch.nn.Module,
    motifnet: MotifNet,
    train_seq_dataloader: torch.utils.data.DataLoader,
    test_seq_dataloader: torch.utils.data.DataLoader,
    train_region_global_indices: np.ndarray,
    test_region_indices: torch.Tensor,
    device: str,
    config: ModelConfig,
    n_total_regions: int,
) -> torch.Tensor:
    """Initialize complete E1 cache for ALL regions (train + test)."""
    sequence_model.eval()
    motifnet.eval()

    adj_tf2r = torch.zeros(n_total_regions, motifnet.n_tfs, device=device)

    with torch.no_grad():
        # Compute TF->region for train regions
        for sequences, seq_idx in tqdm(train_seq_dataloader, desc="Init tf2r (train)"):
            sequences = sequences.to(device)
            emb = _get_sequence_embeddings(sequence_model, sequences, config.bottleneck_size, config.emb_len)
            tf_pred = motifnet(emb)
            global_idx = train_region_global_indices[seq_idx.numpy()]
            adj_tf2r[global_idx] = tf_pred

        # Compute TF->region for test regions
        for sequences, local_idx in tqdm(test_seq_dataloader, desc="Init tf2r (test)"):
            sequences = sequences.to(device)
            emb = _get_sequence_embeddings(sequence_model, sequences, config.bottleneck_size, config.emb_len)
            tf_pred = motifnet(emb)
            global_idx = test_region_indices[local_idx]
            adj_tf2r[global_idx] = tf_pred

    return adj_tf2r


def _recompute_tf2r(
    model: DeepSCENICModel,
    mdata: md.MuData,
    device: str,
    batch_size: int = 200,
) -> torch.Tensor:
    """Recompute tf2r matrix from scratch using the model's sequence model and MotifNet.

    After Phase 1 training, the cached tf2r matrix may be partially stale for
    train regions (each region was last updated at different points during the
    final epoch). This function runs ALL regions through the sequence model and
    MotifNet with the final trained weights to produce a fully up-to-date tf2r.

    Parameters
    ----------
    model
        Trained DeepSCENICModel with sequence_model and motifnet.
    mdata
        MuData with ATAC modality containing region coordinates.
    device
        Device to use for computation.
    batch_size
        Number of sequences per batch for inference.

    Returns
    -------
    torch.Tensor
        Fresh tf2r matrix of shape (n_regions, n_tfs).
    """
    sequence_model = model.sequence_model.to(device)
    motifnet = model.motifnet.to(device)
    sequence_model.eval()
    motifnet.eval()

    all_region_names = list(mdata.mod["atac"].var_names)
    n_regions = len(all_region_names)

    seq_loader = build_sequence_dataloader(
        regions=all_region_names,
        batch_size=batch_size,
        shuffle=False,
        shift_augs=(0, 0),
        rc_aug=False,
        context_length=model.config.seq_len,
    )

    adj_tf2r = torch.zeros(n_regions, motifnet.n_tfs, device=device)

    with torch.no_grad():
        for sequences, seq_idx in tqdm(seq_loader, desc="Recomputing tf2r"):
            sequences = sequences.to(device)
            emb = _get_sequence_embeddings(
                sequence_model, sequences, model.config.bottleneck_size, model.config.emb_len
            )
            tf_pred = motifnet(emb)
            adj_tf2r[seq_idx] = tf_pred

    # Free GPU memory — sequence model and motifnet not needed during finetuning
    sequence_model.cpu()
    motifnet.cpu()
    if device != "cpu":
        torch.cuda.empty_cache()

    return adj_tf2r


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
    adj_tf2r: torch.Tensor,
    test_region_indices: torch.Tensor,
    test_cell_loader: torch.utils.data.DataLoader,
    device: str,
    *,
    loss_atac: str = "cosine",
    alpha: float = 1e-2,
    atac_tau: float = 1.0,
) -> dict[str, float]:
    """Evaluate ATAC reconstruction on test chromosomes."""
    vae.eval()
    metrics: dict[str, float] = {"total": 0.0, "atac_recon": 0.0, "tf2r_l1": 0.0}
    n_batches = 0

    # Extract test TF->region from global cache
    adj_tf2r_test = adj_tf2r[test_region_indices]

    with torch.no_grad():
        for batch in test_cell_loader:
            x_rna = batch["rna"].to(device)
            x_atac = batch["atac"].to(device)

            x_rna_tfs = x_rna[:, vae.tf_indices]
            z_tf, _, _ = vae.encoder(x_rna_tfs, use_mean=True)
            enh_act_test = z_tf @ adj_tf2r_test.T
            x_atac_rec_test = vae.decoder_atac(enh_act_test)

            losses = compute_test_chromosome_loss(
                x_atac=x_atac,
                x_atac_rec_test=x_atac_rec_test,
                adj_tf2r_test=adj_tf2r_test,
                test_region_indices=test_region_indices,
                loss_atac=loss_atac,
                alpha=alpha,
                atac_tau=atac_tau,
            )

            for key in metrics:
                if key in losses:
                    metrics[key] += losses[key].item()
            n_batches += 1

    return {k: v / n_batches for k, v in metrics.items()}


def train(
    mdata: md.MuData,
    *,
    config: ModelConfig | None = None,
    # Training loop parameters
    epochs: int = 100,
    batch_size: int = 16,
    seq_batch_size: int = 200,
    lr: float = 1e-4,
    use_scheduler: bool = False,
    # Loss settings
    loss_rna: str = "mae",
    loss_atac: str = "cosine",
    beta: float = 1e-2,
    alpha: float = 1e-2,
    gamma: float = 1.0,
    rna_tau: float = 1.0,
    atac_tau: float = 1.0,
    # Runtime settings
    device: str | None = None,
    num_workers: int = 0,
    # Data settings
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
    Trains the full model: sequence model (Enformer), MotifNet (MotifNet),
    and VAE encoder/decoders.

    Parameters
    ----------
    mdata
        MuData with preprocessed RNA + ATAC data.
    config
        Model architecture configuration. Use this to set n_hidden,
        binary_atac, and sequence model settings. If None, uses defaults.
    epochs
        Total training epochs.
    batch_size
        Cells per batch.
    seq_batch_size
        Sequences per batch for E1 cache updates.
    lr
        Learning rate for all trainable components (VAE, MotifNet, Enformer).
    use_scheduler
        Whether to use CosineAnnealingLR scheduler.
    loss_rna
        RNA loss type: 'mse', 'mae', 'cosine'.
    loss_atac
        ATAC loss type: 'mse', 'mae', 'bce', 'cosine'.
    beta
        KL divergence weight.
    alpha
        E1 sparsity weight.
    gamma
        E2 sparsity weight.
    rna_tau
        RNA reconstruction weight.
    atac_tau
        ATAC reconstruction weight.
    device
        Training device ('cuda', 'cpu', or None for auto-detect).
    num_workers
        Number of workers for parallel data loading.
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
    >>> config = ModelConfig(n_hidden=256)
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

    # Resolve device (auto-detect if None)
    device = _resolve_device(device)
    log.info(f"Using device: {device}")

    # Validate MuData structure
    if "rna" not in mdata.mod:
        raise ValueError("MuData missing 'rna' modality. Ensure your data has both 'rna' and 'atac' modalities.")
    if "atac" not in mdata.mod:
        raise ValueError("MuData missing 'atac' modality. Ensure your data has both 'rna' and 'atac' modalities.")

    # Setup logging - use phase-specific subdirectory so TensorBoard shows
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
    atac = mdata.mod["atac"]
    tf_mask = rna.var["is_tf"]
    tf_names = rna.var_names[tf_mask].tolist()
    gene_names = list(rna.var_names)
    region_names = list(mdata.mod["atac"].var_names)

    n_tfs = len(tf_names)
    n_genes = len(gene_names)
    n_regions = len(region_names)

    # Compute TF indices from is_tf mask
    tf_indices = torch.tensor(np.where(tf_mask)[0], dtype=torch.long)

    # Validate split columns exist
    if "split" not in rna.var.columns:
        raise ValueError("RNA modality missing 'split' column in var. Run ds.pp.split_features_by_chromosome() first.")
    if "split" not in atac.var.columns:
        raise ValueError("ATAC modality missing 'split' column in var. Run ds.pp.split_features_by_chromosome() first.")

    # Compute gene_indices for reconstruction loss: only TRAIN genes
    # (r2g sparsity loss is also scoped to train-only links below)
    gene_split = rna.var["split"]
    train_gene_mask = gene_split == "train"
    gene_indices = torch.tensor(np.where(train_gene_mask)[0], dtype=torch.long)
    log.info(f"Using {len(gene_indices)}/{n_genes} genes for reconstruction (train split)")

    # Compute region_indices for ATAC reconstruction loss: only TRAIN regions
    region_split = atac.var["split"]
    train_region_mask = region_split == "train"
    region_indices = torch.tensor(np.where(train_region_mask)[0], dtype=torch.long)
    log.info(f"Using {len(region_indices)}/{n_regions} regions for ATAC reconstruction (train split)")

    # Get r2g sparse matrix info (convert CSR to COO for indices)
    r2g_coo = mdata.uns["r2g"]["matrix"].tocoo()
    r2g_indices = torch.tensor(np.array([r2g_coo.row, r2g_coo.col]), dtype=torch.long)
    r2g_distances = torch.tensor(r2g_coo.data).float()

    # Compute train link mask for r2g sparsity loss.
    # During Phase 1, sparsity only applies to links where BOTH region AND gene
    # are in the train split (matching legacy behavior where the r2g penalty file
    # only contained train-region rows on train chromosomes).
    train_r2g_link_mask = torch.tensor(train_region_mask.values[r2g_coo.row] & train_gene_mask.values[r2g_coo.col])
    n_train_links = train_r2g_link_mask.sum().item()
    log.info(f"R2G sparsity on {n_train_links}/{len(r2g_distances)} links (train split)")

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
        n_hidden=model_config.n_hidden,
        binary_atac=model_config.binary_atac,
        n_batches=0,
    ).to(device)
    train_r2g_link_mask = train_r2g_link_mask.to(device)

    # Initialize sequence models (custom or default Enformer)
    if model_config.sequence_model is not None:
        # User provided custom sequence model
        motifnet = MotifNet(
            n_tfs=n_tfs,
            bottleneck_size=model_config.bottleneck_size,
            emb_len=model_config.emb_len,
        ).to(device)
        sequence_model = model_config.sequence_model.to(device)
    else:
        # Default: use Enformer
        motifnet = MotifNet(
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

    # Train sequence dataloader (for sequencenet + motifnet training)
    train_region_mask = mdata.mod["atac"].var["split"] == "train"
    train_region_names = list(mdata.mod["atac"].var_names[train_region_mask])
    # Map from train-only indices (0..n_train-1) to global indices (into full adj_tf2r_cache)
    train_region_global_indices = np.where(train_region_mask)[0]

    # Get DAR indices if balance_dars is enabled (remap to train-only indexing)
    dar_indices = None
    if balance_dars:
        atac_var = mdata.mod["atac"].var
        if "is_dar" not in atac_var.columns:
            raise ValueError(
                "balance_dars=True requires 'is_dar' column in atac.var. "
                "Run ds.pp.mark_dars() first to identify differentially accessible regions."
            )
        global_dar = set(np.where(atac_var["is_dar"].values)[0])
        dar_indices = np.array([i for i, g in enumerate(train_region_global_indices) if g in global_dar])
        if len(dar_indices) == 0:
            log.warning("No DAR regions found in train split. DAR balancing will have no effect.")
            dar_indices = None

    train_seq_loader = build_sequence_dataloader(
        regions=train_region_names,
        batch_size=seq_batch_size,
        shuffle=True,
        shift_augs=(-3, 3),
        rc_aug=True,
        context_length=model_config.seq_len,
        balance_dars=balance_dars,
        dar_indices=dar_indices,
    )

    # Build test sequence dataloader for test chromosome evaluation
    test_region_mask = mdata.mod["atac"].var["split"] == "test"
    n_test_regions = test_region_mask.sum()
    if n_test_regions == 0:
        raise ValueError(
            "No test regions found (atac.var['split'] == 'test'). Run ds.pp.split_features_by_chromosome() first."
        )

    test_seq_loader, test_region_indices = build_test_sequence_dataloader(
        mdata,
        batch_size=seq_batch_size,
        context_length=model_config.seq_len,
        num_workers=num_workers,
    )
    test_region_indices = test_region_indices.to(device)

    log.info(f"Test chromosome evaluation: {n_test_regions} test regions")

    # Initialize complete E1 cache (all regions)
    n_all_regions = len(mdata.mod["atac"].var_names)
    adj_tf2r_cache = _init_tf2r_cache(
        sequence_model=sequence_model,
        motifnet=motifnet,
        train_seq_dataloader=train_seq_loader,
        test_seq_dataloader=test_seq_loader,
        train_region_global_indices=train_region_global_indices,
        test_region_indices=test_region_indices,
        device=device,
        config=model_config,
        n_total_regions=n_all_regions,
    )

    # Tensor mapping from train-local indices to global region indices
    train_region_global_idx = torch.tensor(train_region_global_indices, dtype=torch.long, device=device)

    # Initialize sequence iterator for dynamic E1 updates
    seq_iterator = SequenceIterator(train_seq_loader)

    n_cells = mdata.n_obs
    log.info(f"Starting training: {epochs} epochs, {n_cells} cells, {n_genes} genes, {n_tfs} TFs, {n_regions} regions")

    optimizer = Adam(
        [
            {"params": vae.parameters(), "lr": lr},
            {"params": motifnet.parameters(), "lr": lr},
            {"params": sequence_model.parameters(), "lr": lr},
        ],
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
            log.info(
                "checkpoint_dir set but no checkpointing enabled (checkpoint_every=0, save_best_checkpoints=False)"
            )
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
        motifnet.load_state_dict(resumed_model.motifnet.state_dict())
        sequence_model.load_state_dict(resumed_model.sequence_model.state_dict())

        # Restore E1 cache
        adj_tf2r_cache = resumed_model.adj_tf2r.to(device)

        # Restore training state if present
        if resumed_model.training_state is not None:
            optimizer.load_state_dict(resumed_model.training_state.optimizer_state_dict)
            if resumed_model.training_state.scheduler_state_dict and scheduler is not None:
                scheduler.load_state_dict(resumed_model.training_state.scheduler_state_dict)
            start_epoch = resumed_model.training_state.epoch + 1
            best_loss = resumed_model.training_state.best_loss
        else:
            log.warning("Checkpoint does not contain training state. Starting from epoch 0 with fresh optimizer state.")

        # Restore history if present
        if resumed_model.history is not None:
            history = resumed_model.history

    # Training loop
    for epoch in range(start_epoch, epochs):
        # Set all models to train mode
        vae.train()
        motifnet.train()
        sequence_model.train()

        epoch_metrics: dict[str, float] = {
            "total": 0.0,
            "rna_recon": 0.0,
            "atac_recon": 0.0,
            "kl_div": 0.0,
            "tf2r_l1": 0.0,
            "r2g_l1": 0.0,
        }
        if model_config.binary_atac:
            epoch_metrics["f1_atac"] = 0.0
        n_batches_seen = 0

        pbar = tqdm(train_cell_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            x_rna = batch["rna"].to(device)
            x_atac = batch["atac"].to(device)
            # Update E1 with fresh sequence predictions
            # Clone persistent cache, then insert fresh predictions WITH gradients
            sequences, seq_idx_local = seq_iterator.next()
            sequences = sequences.to(device)

            seq_idx = train_region_global_idx[seq_idx_local]
            adj_tf2r_batch = adj_tf2r_cache.clone()
            emb = _get_sequence_embeddings(
                sequence_model, sequences, model_config.bottleneck_size, model_config.emb_len
            )
            tf_pred = motifnet(emb)
            adj_tf2r_batch[seq_idx] = tf_pred  # Gradients flow through these predictions

            # VAE forward
            optimizer.zero_grad()
            output = vae(
                x_rna,
                adj_tf2r_batch,
                use_mean=False,
            )

            # Compute losses
            losses = compute_total_loss(
                x_rna=x_rna,
                x_atac=x_atac,
                x_rna_rec=output.x_rna_rec,
                x_atac_rec=output.x_atac_rec,
                mu=output.mu,
                logvar=output.logvar,
                adj_tf2r_batch=adj_tf2r_batch,
                adj_r2g=vae.adj_r2g[train_r2g_link_mask],
                r2g_distances=vae.r2g_distances[train_r2g_link_mask],  # type: ignore[arg-type]
                gene_indices=vae.gene_indices,  # type: ignore[arg-type]
                region_indices=vae.region_indices,  # type: ignore[arg-type]
                loss_rna=loss_rna,
                loss_atac=loss_atac,
                beta=beta,
                alpha=alpha,
                gamma=gamma,
                rna_tau=rna_tau,
                atac_tau=atac_tau,
                include_tf2r_sparsity=True,
                seq_idx=seq_idx,  # E1 sparsity only on sampled batch (legacy behavior)
            )
            total_loss = losses["total"]

            # Backward
            total_loss.backward()
            optimizer.step()

            # Update persistent E1 cache with fresh predictions (no_grad for persistence)
            with torch.no_grad():
                adj_tf2r_cache[seq_idx] = tf_pred.detach()

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
        motifnet.eval()
        sequence_model.eval()

        # Recompute test E1 each epoch
        with torch.no_grad():
            for sequences, local_idx in test_seq_loader:
                sequences = sequences.to(device)
                emb = _get_sequence_embeddings(
                    sequence_model, sequences, model_config.bottleneck_size, model_config.emb_len
                )
                tf_pred = motifnet(emb)
                global_idx = test_region_indices[local_idx]
                adj_tf2r_cache[global_idx] = tf_pred

        val_metrics: dict[str, float] = {
            "total": 0.0,
            "rna_recon": 0.0,
            "atac_recon": 0.0,
            "kl_div": 0.0,
            "tf2r_l1": 0.0,
            "r2g_l1": 0.0,
        }
        n_val_batches = 0

        with torch.no_grad():
            for batch in test_cell_loader:
                x_rna = batch["rna"].to(device)
                x_atac = batch["atac"].to(device)
                # Validation
                output = vae(
                    x_rna,
                    adj_tf2r_cache,
                    use_mean=True,
                )

                losses = compute_total_loss(
                    x_rna=x_rna,
                    x_atac=x_atac,
                    x_rna_rec=output.x_rna_rec,
                    x_atac_rec=output.x_atac_rec,
                    mu=output.mu,
                    logvar=output.logvar,
                    adj_tf2r_batch=adj_tf2r_cache,
                    adj_r2g=vae.adj_r2g[train_r2g_link_mask],
                    r2g_distances=vae.r2g_distances[train_r2g_link_mask],  # type: ignore[arg-type]
                    gene_indices=vae.gene_indices,  # type: ignore[arg-type]
                    region_indices=vae.region_indices,  # type: ignore[arg-type]
                    loss_rna=loss_rna,
                    loss_atac=loss_atac,
                    beta=beta,
                    alpha=alpha,
                    gamma=gamma,
                )

                for key in val_metrics:
                    if key in losses:
                        val_metrics[key] += losses[key].item()
                val_metrics["total"] += losses["total"].item()
                n_val_batches += 1

        val_metrics = {k: v / n_val_batches for k, v in val_metrics.items()}
        history.log("val_cells", val_metrics)
        training_logger.log_metrics({f"val_cells/{k}": v for k, v in val_metrics.items()}, step=epoch)

        # Validation on test chromosomes (generalization metric)
        test_chrom_metrics = _evaluate_test_chromosomes(
            vae=vae,
            adj_tf2r=adj_tf2r_cache,
            test_region_indices=test_region_indices,
            test_cell_loader=test_cell_loader,
            device=device,
            loss_atac=loss_atac,
            alpha=alpha,
            atac_tau=atac_tau,
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
            }
            _get_checkpoint_executor().submit(
                _save_checkpoint_data,
                best_path,
                copy.deepcopy(vae.state_dict()),
                copy.deepcopy(motifnet.state_dict()),
                copy.deepcopy(sequence_model.state_dict()),
                adj_tf2r_cache.clone().cpu(),
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
                not _is_enformer(sequence_model),
            )
            log.info(
                f"New best val_chrom/atac_recon (checkpoint): {test_chrom_metrics['atac_recon']:.6f} - saving to {best_path}"
            )

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
            }
            _get_checkpoint_executor().submit(
                _save_checkpoint_data,
                checkpoint_path,
                copy.deepcopy(vae.state_dict()),
                copy.deepcopy(motifnet.state_dict()),
                copy.deepcopy(sequence_model.state_dict()),
                adj_tf2r_cache.clone().cpu(),
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
    motifnet.eval()
    sequence_model.eval()

    log.info("Training completed")

    return DeepSCENICModel(
        vae=vae,
        motifnet=motifnet,
        sequence_model=sequence_model,
        adj_tf2r=adj_tf2r_cache,
        config=model_config,
        tf_names=tf_names,
        gene_names=gene_names,
        region_names=region_names,
        history=history,
    )


def finetune_r2g(
    model: DeepSCENICModel,
    mdata: md.MuData,
    *,
    epochs: int = 500,
    batch_size: int = 16,
    lr: float = 1e-3,
    use_scheduler: bool = False,
    lr_patience: int = 50,
    reinit_r2g: bool = True,
    gamma: float = 1.0,
    loss_rna: str = "mae",
    device: str | None = None,
    num_workers: int = 0,
    balance_class: bool = False,
    class_key: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 0,
    save_best_checkpoints: bool | None = None,
    logger: str = "dict",
    log_dir: str = "./runs",
    logger_kwargs: dict | None = None,
) -> DeepSCENICModel:
    """Finetune the region->gene matrix.

    The second stage of the training workflow. Recomputes tf2r for all regions
    using the trained sequence model and MotifNet, then freezes all model
    parameters except the region->gene weights. All r2g links are optimized
    together using reconstruction loss over all genes. The chromosome split is
    used only during main training; this stage trains on train cells and
    validates on test cells.

    Parameters
    ----------
    model
        Trained DeepSCENICModel from ds.tl.train().
    mdata
        MuData with preprocessed RNA + ATAC data.
    epochs
        Finetuning epochs.
    batch_size
        Cells per batch.
    lr
        Learning rate.
    use_scheduler
        Whether to reduce the learning rate when validation loss plateaus.
        Disabled by default.
    lr_patience
        Epochs before reducing the learning rate when ``use_scheduler=True``.
    reinit_r2g
        If True, reinitialize region->gene weights to near-zero before finetuning.
    gamma
        R2G sparsity weight.
    loss_rna
        RNA loss type: 'mse', 'mae', 'cosine'.
    device
        Training device ('cuda', 'cpu', or None for auto-detect).
    num_workers
        Number of workers for parallel data loading.
    balance_class
        Whether to use weighted sampling to balance classes among training cells.
    class_key
        Column in ``mdata.obs`` containing the labels used when
        ``balance_class=True``.
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
        Model with finetuned r2g matrix and recomputed tf2r.

    Examples
    --------
    Finetune all r2g links with a freshly recomputed tf2r matrix:

    >>> model = ds.tl.train(mdata, epochs=100)
    >>> model = ds.tl.finetune_r2g(model, mdata, epochs=500)
    """
    # Resolve device (auto-detect if None)
    device = _resolve_device(device)
    log.info(f"Using device: {device}")

    # Validate MuData structure
    if "rna" not in mdata.mod:
        raise ValueError("MuData missing 'rna' modality. Ensure your data has both 'rna' and 'atac' modalities.")
    if "atac" not in mdata.mod:
        raise ValueError("MuData missing 'atac' modality. Ensure your data has both 'rna' and 'atac' modalities.")

    # Setup logging - use phase-specific subdirectory so TensorBoard shows
    # each phase as a separate named run
    _phase_name = "finetune_r2g"
    _phase_log_dir = str(Path(log_dir) / _phase_name)
    _logger_kwargs = {"log_dir": _phase_log_dir}
    if logger_kwargs:
        _logger_kwargs.update(logger_kwargs)
    training_logger = get_logger(logger, **_logger_kwargs)
    hyperparams = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "use_scheduler": use_scheduler,
        "lr_patience": lr_patience,
        "gamma": gamma,
        "loss_rna": loss_rna,
        "reinit_r2g": reinit_r2g,
        "balance_class": balance_class,
        "class_key": class_key,
    }
    training_logger.log_hyperparams(hyperparams)

    # Recompute tf2r from scratch using final model weights
    log.info("Recomputing tf2r matrix from sequence model...")
    adj_tf2r = _recompute_tf2r(model, mdata, device)

    # Move VAE to device
    vae = model.vae.to(device)

    # Freeze all parameters except adj_r2g
    for name, param in vae.named_parameters():
        if name == "adj_r2g":
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Reconstruct every gene and optimize every r2g link. The chromosome split
    # remains relevant to main training only.
    n_genes = mdata.mod["rna"].n_vars
    gene_indices = torch.arange(n_genes, dtype=torch.long, device=device)
    e2_link_mask = torch.ones_like(vae.adj_r2g, dtype=torch.bool, device=device)
    log.info(f"Using all {n_genes} genes for reconstruction")
    log.info(f"Using all {len(e2_link_mask)} r2g links for finetuning")

    # Optionally reinitialize all r2g links.
    if reinit_r2g:
        with torch.no_grad():
            vae.adj_r2g[e2_link_mask] = 1e-8

    # Build dataloaders WITHOUT feature filtering
    # The VAE needs all genes for TF indexing and E2 computation.
    # Train on train cells and validate on test cells.
    train_cell_loader = build_cell_dataloader(
        mdata,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        balance_class=balance_class,
        class_key=class_key,
        num_workers=num_workers,
        feature_split=None,  # Don't filter - VAE needs all genes
    )

    val_cell_loader = build_cell_dataloader(
        mdata,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        feature_split=None,  # Don't filter - VAE needs all genes
    )

    log.info("R2G finetuning: all genes, regions, and links")

    # Single optimizer for E2 only
    optimizer = Adam([vae.adj_r2g], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=lr_patience, factor=0.5) if use_scheduler else None

    # History tracking
    history = TrainingHistory()
    best_loss = float("inf")
    best_r2g = vae.adj_r2g.data.clone()

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
            log.info(
                "checkpoint_dir set but no checkpointing enabled (checkpoint_every=0, save_best_checkpoints=False)"
            )
    else:
        log.info("Checkpointing disabled (checkpoint_dir=None)")

    log.info(f"Starting R2G finetuning: {epochs} epochs")

    def _save_finetune_checkpoint(path: Path, epoch: int) -> None:
        """Save a complete model plus state needed to resume finetuning."""
        checkpoint_model = DeepSCENICModel(
            vae=vae,
            motifnet=model.motifnet,
            sequence_model=model.sequence_model,
            adj_tf2r=adj_tf2r,
            config=model.config,
            tf_names=model.tf_names,
            gene_names=model.gene_names,
            region_names=model.region_names,
            history=history,
        )
        checkpoint_model.save(
            path,
            include_training_state=True,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_loss=best_loss,
        )

    # Import loss functions
    from ._loss import r2g_sparsity_loss, reconstruction_loss

    # Training loop
    for epoch in range(epochs):
        vae.train()

        epoch_metrics = {"total": 0.0, "rna_recon": 0.0, "r2g_l1": 0.0}
        n_batches_seen = 0

        pbar = tqdm(train_cell_loader, desc=f"Finetune R2G {epoch + 1}/{epochs}")
        for batch in pbar:
            x_rna = batch["rna"].to(device)

            optimizer.zero_grad()

            # VAE forward
            output = vae(
                x_rna,
                adj_tf2r,
                use_mean=False,
            )

            # Compute RNA reconstruction loss over all genes.
            loss_rec_rna = reconstruction_loss(
                output.x_rna_rec[:, gene_indices],
                x_rna[:, gene_indices],
                loss_type=loss_rna,
            )

            # Apply sparsity regularization to all r2g links.
            loss_r2g_sparse = r2g_sparsity_loss(
                vae.adj_r2g[e2_link_mask],
                vae.r2g_distances[e2_link_mask],  # type: ignore[arg-type]
            )
            total_loss = loss_rec_rna + loss_r2g_sparse * gamma

            total_loss.backward()
            optimizer.step()

            # Accumulate metrics
            epoch_metrics["total"] += total_loss.item()
            epoch_metrics["rna_recon"] += loss_rec_rna.item()
            epoch_metrics["r2g_l1"] += loss_r2g_sparse.item()
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
                    adj_tf2r,
                    use_mean=True,
                )

                loss_rec_rna = reconstruction_loss(
                    output.x_rna_rec[:, gene_indices],
                    x_rna[:, gene_indices],
                    loss_type=loss_rna,
                )
                loss_r2g_sparse = r2g_sparsity_loss(
                    vae.adj_r2g[e2_link_mask],
                    vae.r2g_distances[e2_link_mask],  # type: ignore[arg-type]
                )

                val_loss += (loss_rec_rna + loss_r2g_sparse * gamma).item()
                n_val_batches += 1

        val_loss /= n_val_batches
        history.log("val_cells", {"total": val_loss})
        training_logger.log_metrics({"val_cells/total": val_loss}, step=epoch)

        # Step scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Track best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_r2g = vae.adj_r2g.data.clone()
            # Save a complete, directly loadable model checkpoint.
            if save_best_checkpoints and (checkpoint_dir is not None):
                best_path = Path(checkpoint_dir) / "finetune_best.pt"
                _save_finetune_checkpoint(best_path, epoch)
                log.info(f"New best val loss: {val_loss:.6f} - saved to {best_path}")

        # Periodic checkpointing
        if checkpoint_dir is not None and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = Path(checkpoint_dir) / f"finetune_epoch_{epoch + 1}.pt"
            _save_finetune_checkpoint(checkpoint_path, epoch)

    # Close logger
    training_logger.close()

    # Restore best E2
    vae.adj_r2g.data = best_r2g
    vae.eval()

    log.info("R2G finetuning completed")

    # Return updated model with new history
    return DeepSCENICModel(
        vae=vae,
        motifnet=model.motifnet,
        sequence_model=model.sequence_model,
        adj_tf2r=adj_tf2r,
        config=model.config,
        tf_names=model.tf_names,
        gene_names=model.gene_names,
        region_names=model.region_names,
        history=history,
    )
