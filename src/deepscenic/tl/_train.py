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
from ._loss import compute_test_chromosome_loss, compute_total_loss, e1_sparsity_loss
from ._model import DeepSCENICModel, PretrainedModel
from ._training_state import (
    Checkpoint,
    EarlyStopping,
    FinetuneConfig,
    PretrainConfig,
    TrainingConfig,
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


def _save_checkpoint_async(checkpoint: Checkpoint, path: Path) -> None:
    """Save checkpoint asynchronously in a background thread.

    The checkpoint data is already deep-copied before this function is called,
    so it's safe to save in the background while training continues.
    """
    checkpoint.save(path)


def _save_sequence_model_state(
    sequence_model: torch.nn.Module,
    tf2rnet: torch.nn.Module,
    checkpoint_dir: Path,
) -> Path:
    """Save sequence model state once at training start.

    This enables checkpoint storage optimization: individual checkpoints
    skip the large Enformer state dict, and resume logic loads from this
    one-time file instead.

    Parameters
    ----------
    sequence_model
        Sequence embedding model (Enformer or custom).
    tf2rnet
        MotifNet model.
    checkpoint_dir
        Directory to save the sequence model state.

    Returns
    -------
    Path
        Path to the saved file.
    """
    seq_model_path = checkpoint_dir / "sequence_model.pt"
    if not seq_model_path.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "enformer_state_dict": sequence_model.state_dict(),
                "tf2rnet_state_dict": tf2rnet.state_dict(),
            },
            seq_model_path,
        )
        log.info(f"Saved sequence model state to {seq_model_path}")
    return seq_model_path


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
    config: TrainingConfig | PretrainConfig,
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
    config: TrainingConfig | PretrainConfig,
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


def _evaluate_test_chromosomes(
    vae: DeepSCENICVAE,
    adj_E1_test: torch.Tensor,
    test_region_indices: torch.Tensor,
    test_cell_loader: torch.utils.data.DataLoader,
    device: str,
    config: TrainingConfig,
    use_ppi: bool = True,
) -> dict[str, float]:
    """Evaluate ATAC reconstruction on test chromosomes.

    Computes TF activity from encoder (with PPI modulation if enabled),
    then predicts ATAC for test regions using adj_E1_test.

    This matches legacy behavior where the full VAE forward pass (including
    PPI network) is used to compute test chromosome ATAC reconstruction.
    """
    vae.eval()
    metrics: dict[str, float] = {"total": 0.0, "atac_recon": 0.0, "e1_l1": 0.0}
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

                batch_ppi = build_ppi_batch(x_rna, vae.ppi_genes_idx, vae.ppi_edge_index, device)
                ppi_out = vae.ppi(batch_ppi.x, batch_ppi.edge_index)
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

            # Compute loss
            losses = compute_test_chromosome_loss(
                x_atac=x_atac,
                x_atac_rec_test=x_atac_rec_test,
                adj_E1_test=adj_E1_test,
                test_region_indices=test_region_indices,
                loss_atac=config.loss_atac,
                alpha=config.alpha,
                atac_tau=config.atac_tau,
            )

            for key in metrics:
                metrics[key] += losses[key].item()
            n_batches += 1

    return {k: v / n_batches for k, v in metrics.items()}


def train(
    mdata: md.MuData,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cuda",
    logger: str = "dict",
    log_dir: str = "./runs",
    logger_kwargs: dict | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 10,
    save_best_checkpoints: bool | None = None,
    resume_from: str | None = None,
    early_stopping_patience: int | None = None,
    seq_batch_size: int = 1000,
    balance_dars: bool = False,
    num_workers: int = 0,
    pretrained_model: PretrainedModel | None = None,
    config: TrainingConfig | None = None,
) -> DeepSCENICModel:
    """Train deepSCENIC model on multiome data.

    Parameters
    ----------
    mdata
        MuData with preprocessed RNA + ATAC data.
    epochs
        Total training epochs.
    batch_size
        Cells per batch.
    lr
        Learning rate.
    device
        Training device ('cuda' or 'cpu').
    logger
        Logging backend: 'dict', 'tensorboard', 'wandb'.
    log_dir
        Directory for tensorboard/wandb logs.
    logger_kwargs
        Additional keyword arguments passed to the logger backend. For wandb,
        this can include 'project', 'name', 'tags', etc. The default wandb
        project is 'deepscenic'.
    checkpoint_dir
        Directory for checkpoints (None = no checkpoints).
    checkpoint_every
        Save checkpoint every N epochs.
    save_best_checkpoints
        Save checkpoint when validation loss improves. If None (default),
        enabled when checkpoint_dir is set. Saves to 'best.pt' in checkpoint_dir.
    resume_from
        Path to checkpoint to resume from.
    early_stopping_patience
        Stop if no improvement for N epochs (None = disabled).
    seq_batch_size
        Sequences per batch for TF2rNet updates.
    balance_dars
        Whether to upweight DAR regions during sequence sampling (1.5x weight).
        Requires DARs to be marked via ``ds.pp.mark_dars()`` first.
    num_workers
        Number of workers for parallel data loading. Use "auto" (default) to
        automatically detect optimal value based on CPU count, or 0 for main
        process only.
    pretrained_model
        Optional pretrained model from ds.tl.pretrain(). If provided, TF2rNet
        and sequence model weights are initialized from the pretrained model.
    config
        Advanced configuration options. When provided, explicit function
        parameters take precedence over config values. Use config.sequence_model
        to provide a custom sequence embedding model instead of Enformer.

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

    Training with pretrained model:
    >>> pretrained = ds.tl.pretrain(mdata, epochs=50)
    >>> model = ds.tl.train(mdata, pretrained_model=pretrained)

    Training with custom sequence model:
    >>> from deepscenic.tl import TrainingConfig
    >>> config = TrainingConfig(
    ...     sequence_model=my_custom_model,
    ...     bottleneck_size=256,
    ...     emb_len=60,
    ... )
    >>> model = ds.tl.train(mdata, config=config)
    """
    # Build config: start from provided config or defaults, then override with explicit params
    if config is not None:
        # Start with provided config and override with explicit parameters
        base_config = config
    else:
        base_config = TrainingConfig()

    # Explicit function parameters override config values
    final_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        seq_batch_size=seq_batch_size,
        lr=lr,
        lr_ppi=base_config.lr_ppi,
        weight_decay=base_config.weight_decay,
        beta=base_config.beta,
        alpha=base_config.alpha,
        gamma=base_config.gamma,
        rna_tau=base_config.rna_tau,
        atac_tau=base_config.atac_tau,
        loss_rna=base_config.loss_rna,
        loss_atac=base_config.loss_atac,
        dropout_mask_rna=base_config.dropout_mask_rna,
        dropout_mask_atac=base_config.dropout_mask_atac,
        n_hidden=base_config.n_hidden,
        use_ppi=base_config.use_ppi,
        binary_atac=base_config.binary_atac,
        device=device,
        ppi_device=base_config.ppi_device,
        batch_key=base_config.batch_key,
        balance_class=base_config.balance_class,
        class_key=base_config.class_key,
        balance_dars=balance_dars,
        num_workers=num_workers,
        bottleneck_size=base_config.bottleneck_size,
        emb_len=base_config.emb_len,
        seq_len=base_config.seq_len,
        sequence_model=base_config.sequence_model,
    )
    config = final_config

    # Setup logging
    _logger_kwargs = {"log_dir": log_dir}
    if logger_kwargs:
        _logger_kwargs.update(logger_kwargs)
    training_logger = get_logger(logger, **_logger_kwargs)
    training_logger.log_hyperparams(config.to_dict())

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

    # Get r2g sparse matrix info (convert CSR to COO for indices)
    r2g_coo = mdata.uns["r2g"]["matrix"].tocoo()
    r2g_indices = torch.tensor(np.array([r2g_coo.row, r2g_coo.col]))
    r2g_distances = torch.tensor(r2g_coo.data).float()

    # Get PPI info if present
    ppi_edge_index = None
    ppi_genes_idx = None
    ppi_tfs_idx_keys = None
    ppi_tfs_idx_values = None
    use_ppi = config.use_ppi and "ppi_edge_index" in mdata.uns

    if use_ppi:
        ppi_edge_index = torch.tensor(mdata.uns["ppi_edge_index"])
        ppi_genes_idx = torch.tensor(mdata.uns["ppi_genes_idx"])
        ppi_tfs_idx_keys = torch.tensor(mdata.uns["ppi_tfs_idx_keys"])
        ppi_tfs_idx_values = torch.tensor(mdata.uns["ppi_tfs_idx_values"])

    # Batch correction
    n_batches = 0
    if config.batch_key is not None:
        n_batches = int(mdata.obs[config.batch_key].nunique())

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
        n_hidden=config.n_hidden,
        use_ppi=use_ppi,
        binary_atac=config.binary_atac,
        n_batches=n_batches,
    ).to(device)

    # Initialize sequence models (or use pretrained/custom)
    if pretrained_model is not None:
        tf2rnet = pretrained_model.tf2rnet.to(device)
        sequence_model = pretrained_model.enformer.to(device)
    elif config.sequence_model is not None:
        # User provided custom sequence model
        tf2rnet = MotifNet(
            n_tfs=n_tfs,
            bottleneck_size=config.bottleneck_size,
            emb_len=config.emb_len,
        ).to(device)
        sequence_model = config.sequence_model.to(device)
    else:
        # Default: use Enformer
        tf2rnet = MotifNet(
            n_tfs=n_tfs,
            bottleneck_size=config.bottleneck_size,
            emb_len=config.emb_len,
        ).to(device)
        sequence_model = _init_enformer(device, config.emb_len)

    # Build dataloaders
    train_cell_loader = build_cell_dataloader(
        mdata,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        balance_class=config.balance_class,
        class_key=config.class_key,
        num_workers=config.num_workers,
    )

    test_cell_loader = build_cell_dataloader(
        mdata,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Sequence dataloader (for TF2rNet training)
    # Regions are derived from ATAC var_names (chr:start-end format)
    region_names = list(mdata.mod["atac"].var_names)

    # Get DAR indices if balance_dars is enabled
    dar_indices = None
    if config.balance_dars and "atac" in mdata.mod:
        atac_var = mdata.mod["atac"].var
        if "is_dar" in atac_var.columns:
            dar_indices = np.where(atac_var["is_dar"].values)[0]

    train_seq_loader = build_sequence_dataloader(
        regions=region_names,
        batch_size=seq_batch_size,
        shuffle=True,
        shift_augs=(-3, 3),
        rc_aug=True,
        context_length=config.seq_len,
        num_workers=config.num_workers,
        balance_dars=config.balance_dars,
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
        context_length=config.seq_len,
        num_workers=config.num_workers,
    )
    test_region_indices = test_region_indices.to(device)

    log.info(f"Test chromosome evaluation: {n_test_regions} test regions")

    # Initialize E1 cache (use pretrained if available)
    if pretrained_model is not None:
        adj_E1_cache = pretrained_model.adj_E1.to(device)
    else:
        adj_E1_cache = _init_e1_cache(sequence_model, tf2rnet, train_seq_loader, device, config)

    # Initialize test E1 cache (always computed, no pretrained cache for test regions)
    adj_E1_test_cache = _init_e1_test_cache(sequence_model, tf2rnet, test_seq_loader, device, config)

    # Save sequence model state once (for checkpoint resume)
    # This enables storage optimization: checkpoints skip Enformer state dict
    if checkpoint_dir is not None:
        _save_sequence_model_state(sequence_model, tf2rnet, Path(checkpoint_dir))

    n_cells = mdata.n_obs
    log.info(f"Starting training: {epochs} epochs, {n_cells} cells, {n_genes} genes, {n_tfs} TFs, {n_regions} regions")

    # Optimizer (only for VAE - TF2rNet is frozen)
    optimizer_vae = Adam(vae.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    scheduler = CosineAnnealingLR(optimizer_vae, T_max=epochs) if config.use_scheduler else None

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
        log.info(f"Checkpointing enabled: saving every {checkpoint_every} epochs to {checkpoint_dir}")
        if save_best_checkpoints:
            log.info("Best checkpoint saving enabled: saving to 'best.pt' on improvement")
    else:
        log.info("Checkpointing disabled (checkpoint_dir=None)")

    best_loss = float("inf")

    # Resume from checkpoint
    start_epoch = 0
    if resume_from is not None:
        checkpoint = Checkpoint.load(resume_from, map_location=device)
        vae.load_state_dict(checkpoint.vae_state_dict)

        # Load sequence model state: from checkpoint, from one-time file, or keep initialized
        if checkpoint.enformer_state_dict is not None:
            # Checkpoint has sequence model (old-style checkpoint)
            sequence_model.load_state_dict(checkpoint.enformer_state_dict)
            tf2rnet.load_state_dict(checkpoint.tf2rnet_state_dict)
        elif checkpoint_dir is not None:
            # Try loading from one-time sequence_model.pt file
            seq_model_path = Path(checkpoint_dir) / "sequence_model.pt"
            if seq_model_path.exists():
                seq_data = torch.load(seq_model_path, map_location=device)
                sequence_model.load_state_dict(seq_data["enformer_state_dict"])
                tf2rnet.load_state_dict(seq_data["tf2rnet_state_dict"])
                log.info(f"Loaded sequence model from {seq_model_path}")
            else:
                # No sequence model file - load tf2rnet from checkpoint, keep sequence model initialized
                tf2rnet.load_state_dict(checkpoint.tf2rnet_state_dict)
                log.warning(
                    "No sequence model state in checkpoint or sequence_model.pt. "
                    "Using initialized model - ensure this matches original training."
                )
        else:
            # No checkpoint_dir - load tf2rnet from checkpoint only
            tf2rnet.load_state_dict(checkpoint.tf2rnet_state_dict)
            log.warning(
                "No checkpoint_dir provided for resume. Cannot load sequence model state. "
                "Provide checkpoint_dir or ensure pretrained_model matches original training."
            )

        optimizer_vae.load_state_dict(checkpoint.optimizer_state_dict)
        if checkpoint.scheduler_state_dict and scheduler is not None:
            scheduler.load_state_dict(checkpoint.scheduler_state_dict)
        history = checkpoint.history
        adj_E1_cache = checkpoint.adj_E1.to(device)
        # Load test E1 cache from checkpoint if available, otherwise recompute
        if checkpoint.adj_E1_test is not None:
            adj_E1_test_cache = checkpoint.adj_E1_test.to(device)
        start_epoch = checkpoint.epoch + 1
        best_loss = checkpoint.best_loss

    # TF2rNet is always frozen during train() - use pretrain() to train TF2rNet
    for param in tf2rnet.parameters():
        param.requires_grad = False
    for param in sequence_model.parameters():
        param.requires_grad = False

    # Training loop
    for epoch in range(start_epoch, epochs):
        vae.train()

        epoch_metrics = {
            "total": 0.0,
            "rna_recon": 0.0,
            "atac_recon": 0.0,
            "kl_div": 0.0,
            "e1_l1": 0.0,
            "e2_l1": 0.0,
            "ppi_reg": 0.0,
        }
        n_batches_seen = 0

        pbar = tqdm(train_cell_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            x_rna = batch["rna"].to(device)
            x_atac = batch["atac"].to(device)

            # VAE forward (E1 is frozen, computed once at initialization)
            optimizer_vae.zero_grad()
            output = vae(
                x_rna,
                adj_E1_cache,
                use_ppi=use_ppi,
                use_mean=False,
                ppi_device=config.ppi_device,
            )

            # Compute losses
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
                loss_rna=config.loss_rna,
                loss_atac=config.loss_atac,
                dropout_mask_rna=config.dropout_mask_rna,
                dropout_mask_atac=config.dropout_mask_atac,
                beta=config.beta,
                alpha=config.alpha,
                gamma=config.gamma,
                rna_tau=config.rna_tau,
                atac_tau=config.atac_tau,
                use_ppi=use_ppi,
                include_e1_sparsity=True,
            )
            total_loss = losses["total"]

            # Backward
            total_loss.backward()
            optimizer_vae.step()

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
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in test_cell_loader:
                x_rna = batch["rna"].to(device)
                x_atac = batch["atac"].to(device)

                output = vae(
                    x_rna,
                    adj_E1_cache,
                    use_ppi=use_ppi,
                    use_mean=True,
                    ppi_device=config.ppi_device,
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
                    loss_rna=config.loss_rna,
                    loss_atac=config.loss_atac,
                    beta=config.beta,
                    alpha=config.alpha,
                    gamma=config.gamma,
                    use_ppi=use_ppi,
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
            test_region_indices=test_region_indices,
            test_cell_loader=test_cell_loader,
            device=device,
            config=config,
            use_ppi=use_ppi,
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
            # Skip Enformer state dict - it's frozen and unchanged during train()
            # Deep copy state dicts for async save (training continues modifying them)
            best_checkpoint = Checkpoint(
                epoch=epoch,
                vae_state_dict=copy.deepcopy(vae.state_dict()),
                tf2rnet_state_dict=copy.deepcopy(tf2rnet.state_dict()),
                enformer_state_dict=None,
                optimizer_state_dict=copy.deepcopy(optimizer_vae.state_dict()),
                scheduler_state_dict=copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
                history=copy.deepcopy(history),
                config=config,
                adj_E1=adj_E1_cache.clone(),
                adj_E1_test=adj_E1_test_cache.clone(),
                best_loss=best_loss,
            )
            _get_checkpoint_executor().submit(_save_checkpoint_async, best_checkpoint, best_path)
            log.info(f"New best val_chrom/atac_recon: {test_chrom_metrics['atac_recon']:.6f} - saving to {best_path}")

        # Periodic checkpointing (async to not block training)
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = Path(checkpoint_dir) / f"epoch_{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            # Skip Enformer state dict - it's frozen and unchanged during train()
            # Deep copy state dicts for async save (training continues modifying them)
            periodic_checkpoint = Checkpoint(
                epoch=epoch,
                vae_state_dict=copy.deepcopy(vae.state_dict()),
                tf2rnet_state_dict=copy.deepcopy(tf2rnet.state_dict()),
                enformer_state_dict=None,
                optimizer_state_dict=copy.deepcopy(optimizer_vae.state_dict()),
                scheduler_state_dict=copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
                history=copy.deepcopy(history),
                config=config,
                adj_E1=adj_E1_cache.clone(),
                adj_E1_test=adj_E1_test_cache.clone(),
                best_loss=best_loss,
            )
            _get_checkpoint_executor().submit(_save_checkpoint_async, periodic_checkpoint, checkpoint_path)

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
        config=config,
        tf_names=tf_names,
        gene_names=gene_names,
        region_names=region_names,
        history=history,
    )


def finetune(
    model: DeepSCENICModel,
    mdata: md.MuData,
    *,
    epochs: int = 10000,
    batch_size: int = 64,
    lr: float = 1e-6,
    lr_patience: int = 50,
    device: str = "cuda",
    reinit_e2: bool = True,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 100,
    save_best_checkpoints: bool | None = None,
    logger: str = "dict",
    log_dir: str = "./runs",
    logger_kwargs: dict | None = None,
    num_workers: int = 0,
    config: FinetuneConfig | None = None,
) -> DeepSCENICModel:
    """Fine-tune E2 matrix on training data.

    Freezes all model parameters except E2 (region-to-gene weights) and
    trains with a very low learning rate using ReduceLROnPlateau scheduler.

    Parameters
    ----------
    model
        Trained DeepSCENICModel from ds.tl.train().
    mdata
        MuData with train/test split.
    epochs
        Finetuning epochs.
    batch_size
        Cells per batch.
    lr
        Learning rate (very small, typically 1e-6).
    lr_patience
        Epochs before reducing LR if no improvement.
    device
        Training device.
    reinit_e2
        If True, reinitialize E2 to near-zero before finetuning.
    checkpoint_dir
        Directory for checkpoints (None = no checkpoints).
    checkpoint_every
        Save checkpoint every N epochs.
    save_best_checkpoints
        Save checkpoint when validation loss improves. If None (default),
        enabled when checkpoint_dir is set. Saves to 'finetune_best.pt'.
    logger
        Logging backend: 'dict', 'tensorboard', 'wandb'.
    log_dir
        Directory for logs.
    logger_kwargs
        Additional keyword arguments passed to the logger backend. For wandb,
        this can include 'project', 'name', 'tags', etc. The default wandb
        project is 'deepscenic'.
    num_workers
        Number of workers for parallel data loading. Use "auto" (default) to
        automatically detect optimal value based on CPU count, or 0 for main
        process only.
    config
        Advanced configuration. Explicit parameters override config values.

    Returns
    -------
    DeepSCENICModel
        Model with finetuned E2 matrix.

    Examples
    --------
    >>> model = ds.tl.train(mdata, epochs=100)
    >>> model = ds.tl.finetune(model, mdata, epochs=10000, lr=1e-6)
    """
    # Build config
    if config is not None:
        base_config = config
    else:
        base_config = FinetuneConfig()

    final_config = FinetuneConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lr_patience=lr_patience,
        reinit_e2=reinit_e2,
        gamma=base_config.gamma,
        loss_rna=base_config.loss_rna,
        dropout_mask_rna=base_config.dropout_mask_rna,
        device=device,
        num_workers=num_workers,
        batch_key=base_config.batch_key,
    )
    finetune_config = final_config

    # Setup logging
    _logger_kwargs = {"log_dir": log_dir}
    if logger_kwargs:
        _logger_kwargs.update(logger_kwargs)
    training_logger = get_logger(logger, **_logger_kwargs)
    training_logger.log_hyperparams(finetune_config.to_dict())

    # Move model to device
    vae = model.vae.to(device)
    adj_E1 = model.adj_E1.to(device)

    # Freeze all parameters except adj_E2
    for name, param in vae.named_parameters():
        if name == "adj_E2":
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Optionally reinitialize E2
    if finetune_config.reinit_e2:
        with torch.no_grad():
            vae.adj_E2.fill_(1e-8)

    # Build dataloaders
    train_cell_loader = build_cell_dataloader(
        mdata,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=finetune_config.num_workers,
    )

    test_cell_loader = build_cell_dataloader(
        mdata,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=finetune_config.num_workers,
    )

    # Single optimizer for E2 only
    optimizer = Adam([vae.adj_E2], lr=finetune_config.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=finetune_config.lr_patience, factor=0.5)

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
        log.info(f"Checkpointing enabled: saving every {checkpoint_every} epochs to {checkpoint_dir}")
        if save_best_checkpoints:
            log.info("Best checkpoint saving enabled: saving to 'finetune_best.pt' on improvement")
    else:
        log.info("Checkpointing disabled (checkpoint_dir=None)")

    log.info(f"Starting finetuning: {epochs} epochs")

    # Training loop
    for epoch in range(finetune_config.epochs):
        vae.train()

        epoch_metrics = {"total": 0.0, "rna_recon": 0.0, "e2_l1": 0.0}
        n_batches_seen = 0

        pbar = tqdm(train_cell_loader, desc=f"Finetune {epoch + 1}/{epochs}")
        for batch in pbar:
            x_rna = batch["rna"].to(device)

            optimizer.zero_grad()

            # VAE forward (no PPI, use_mean=False for training)
            output = vae(
                x_rna,
                adj_E1,
                use_ppi=False,
                use_mean=False,
            )

            # Compute RNA reconstruction loss only
            from ._loss import e2_sparsity_loss, reconstruction_loss

            loss_rec_rna = reconstruction_loss(
                output.x_rna_rec,
                x_rna,
                loss_type=finetune_config.loss_rna,
                dropout_mask=finetune_config.dropout_mask_rna,
            )

            loss_e2_sparse = e2_sparsity_loss(vae.adj_E2, vae.r2g_distances)  # type: ignore[arg-type]
            total_loss = loss_rec_rna + loss_e2_sparse * finetune_config.gamma

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
            for batch in test_cell_loader:
                x_rna = batch["rna"].to(device)

                output = vae(
                    x_rna,
                    adj_E1,
                    use_ppi=False,
                    use_mean=True,
                )

                loss_rec_rna = reconstruction_loss(
                    output.x_rna_rec,
                    x_rna,
                    loss_type=finetune_config.loss_rna,
                    dropout_mask=finetune_config.dropout_mask_rna,
                )
                loss_e2_sparse = e2_sparsity_loss(vae.adj_E2, vae.r2g_distances)  # type: ignore[arg-type]

                val_loss += (loss_rec_rna + loss_e2_sparse * finetune_config.gamma).item()
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
                        "config": finetune_config.to_dict(),
                        "best_loss": best_loss,
                    },
                    best_path,
                )
                log.info(f"New best val loss: {val_loss:.6f} - saved to {best_path}")

        # Periodic checkpointing
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = Path(checkpoint_dir) / f"finetune_epoch_{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "adj_E2": vae.adj_E2.data,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "history": history.to_dict(),
                    "config": finetune_config.to_dict(),
                    "best_loss": best_loss,
                },
                checkpoint_path,
            )

    # Close logger
    training_logger.close()

    # Restore best E2
    vae.adj_E2.data = best_e2
    vae.eval()

    log.info("Finetuning completed")

    # Return updated model
    return DeepSCENICModel(
        vae=vae,
        tf2rnet=model.tf2rnet,
        enformer=model.enformer,
        adj_E1=adj_E1,
        config=model.config,
        tf_names=model.tf_names,
        gene_names=model.gene_names,
        region_names=model.region_names,
    )


def pretrain(
    mdata: md.MuData,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: str = "cuda",
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 10,
    logger: str = "dict",
    log_dir: str = "./runs",
    logger_kwargs: dict | None = None,
    num_workers: int = 0,
    config: PretrainConfig | None = None,
) -> PretrainedModel:
    """Pretrain TF2rNet on DNA sequences.

    Trains the sequence-to-TF prediction model (Enformer + MotifNet) using
    L1 sparsity loss to learn biologically meaningful TF binding patterns.

    Parameters
    ----------
    mdata
        MuData with regions in ATAC modality.
    epochs
        Pretraining epochs.
    batch_size
        Sequences per batch.
    lr
        Learning rate.
    device
        Training device.
    checkpoint_dir
        Directory for checkpoints.
    checkpoint_every
        Save checkpoint every N epochs.
    logger
        Logging backend.
    log_dir
        Directory for logs.
    logger_kwargs
        Additional keyword arguments passed to the logger backend. For wandb,
        this can include 'project', 'name', 'tags', etc. The default wandb
        project is 'deepscenic'.
    num_workers
        Number of workers for parallel data loading. Use "auto" (default) to
        automatically detect optimal value based on CPU count, or 0 for main
        process only.
    config
        Advanced configuration.

    Returns
    -------
    PretrainedModel
        Pretrained sequence models with cached E1 matrix.

    Examples
    --------
    >>> pretrained = ds.tl.pretrain(mdata, epochs=100)
    >>> model = ds.tl.train(mdata, pretrained_model=pretrained)
    """
    # Build config
    if config is not None:
        base_config = config
    else:
        base_config = PretrainConfig()

    final_config = PretrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=base_config.weight_decay,
        alpha=base_config.alpha,
        device=device,
        num_workers=num_workers,
        bottleneck_size=base_config.bottleneck_size,
        emb_len=base_config.emb_len,
        seq_len=base_config.seq_len,
        sequence_model=base_config.sequence_model,
    )
    pretrain_config = final_config

    # Setup logging
    _logger_kwargs = {"log_dir": log_dir}
    if logger_kwargs:
        _logger_kwargs.update(logger_kwargs)
    training_logger = get_logger(logger, **_logger_kwargs)
    training_logger.log_hyperparams(pretrain_config.to_dict())

    # Extract metadata from MuData
    rna = mdata.mod["rna"]
    tf_mask = rna.var["is_tf"]
    tf_names = rna.var_names[tf_mask].tolist()
    region_names = list(mdata.mod["atac"].var_names)
    n_tfs = len(tf_names)

    # Initialize models
    tf2rnet = MotifNet(
        n_tfs=n_tfs,
        bottleneck_size=pretrain_config.bottleneck_size,
        emb_len=pretrain_config.emb_len,
    ).to(device)

    # Initialize sequence model (custom or default Enformer)
    if pretrain_config.sequence_model is not None:
        sequence_model = pretrain_config.sequence_model.to(device)
    else:
        sequence_model = _init_enformer(device, pretrain_config.emb_len)

    # Build sequence dataloader
    seq_loader = build_sequence_dataloader(
        regions=region_names,
        batch_size=batch_size,
        shuffle=True,
        shift_augs=(-3, 3),
        rc_aug=True,
        context_length=pretrain_config.seq_len,
        num_workers=pretrain_config.num_workers,
    )

    # Single optimizer for both models
    optimizer = Adam(
        list(tf2rnet.parameters()) + list(sequence_model.parameters()),
        lr=pretrain_config.lr,
        weight_decay=pretrain_config.weight_decay,
    )

    # History tracking
    history = TrainingHistory()

    n_regions = len(region_names)
    log.info(f"Starting pretraining: {epochs} epochs, {n_regions} regions, {n_tfs} TFs")

    # Training loop
    for epoch in range(pretrain_config.epochs):
        tf2rnet.train()
        sequence_model.train()

        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(seq_loader, desc=f"Pretrain {epoch + 1}/{epochs}")
        for sequences, _ in pbar:
            sequences = sequences.to(device)

            optimizer.zero_grad()

            # Forward through sequence model + TF2rNet
            emb = _get_sequence_embeddings(
                sequence_model, sequences, pretrain_config.bottleneck_size, pretrain_config.emb_len
            )
            tf_pred = tf2rnet(emb)

            # Loss: L1 sparsity on TF predictions
            loss = e1_sparsity_loss(tf_pred) * pretrain_config.alpha

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item())

        # Average epoch loss
        avg_loss = epoch_loss / n_batches
        history.log("train", {"total": avg_loss, "e1_l1": avg_loss})
        training_logger.log_metrics({"train/total": avg_loss}, step=epoch)

        # Checkpointing
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = Path(checkpoint_dir) / f"pretrain_epoch_{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "tf2rnet_state_dict": tf2rnet.state_dict(),
                    "enformer_state_dict": sequence_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history.to_dict(),
                    "config": pretrain_config.to_dict(),
                },
                checkpoint_path,
            )

    # Close logger
    training_logger.close()

    # Compute final E1 cache
    tf2rnet.eval()
    sequence_model.eval()
    adj_E1 = _init_e1_cache(sequence_model, tf2rnet, seq_loader, device, pretrain_config)

    log.info("Pretraining completed")

    return PretrainedModel(
        tf2rnet=tf2rnet,
        enformer=sequence_model,
        adj_E1=adj_E1,
        config=pretrain_config,
        tf_names=tf_names,
        region_names=region_names,
    )
