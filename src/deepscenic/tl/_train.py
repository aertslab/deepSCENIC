"""Main training loop for deepSCENIC."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm.auto import tqdm

from ..models import DeepSCENICVAE, MotifNet
from ._dataloaders import build_cell_dataloader, build_sequence_dataloader
from ._logging import get_logger
from ._loss import compute_total_loss, e1_sparsity_loss
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


def _init_enformer(device: str) -> torch.nn.Module:
    """Initialize Enformer model."""
    from enformer_pytorch import Enformer

    enformer = Enformer.from_pretrained(
        "EleutherAI/enformer-official-rough",
        target_length=-1,
    )
    enformer.to(device)
    return enformer


def _get_enformer_embeddings(
    enformer: torch.nn.Module,
    sequences: torch.Tensor,
    bottleneck_size: int,
    emb_len: int,
) -> torch.Tensor:
    """Get Enformer embeddings for sequences."""
    with torch.cuda.amp.autocast(enabled=True):
        output = enformer(sequences, return_only_embeddings=True)
    # Flatten: (batch, emb_len, bottleneck) -> (batch, bottleneck * emb_len)
    return output.reshape(-1, bottleneck_size * emb_len)


def _init_e1_cache(
    enformer: torch.nn.Module,
    tf2rnet: MotifNet,
    seq_dataloader: torch.utils.data.DataLoader,
    device: str,
    config: TrainingConfig,
) -> torch.Tensor:
    """Initialize E1 cache by running all sequences through Enformer + TF2rNet."""
    enformer.eval()
    tf2rnet.eval()

    n_regions = len(seq_dataloader.dataset)
    adj_E1 = torch.zeros(n_regions, tf2rnet.n_tfs, device=device)

    with torch.no_grad():
        for (sequences,), seq_idx in tqdm(seq_dataloader, desc="Initializing E1"):
            sequences = sequences.to(device)
            emb = _get_enformer_embeddings(enformer, sequences, config.bottleneck_size, config.emb_len)
            tf_pred = tf2rnet(emb)
            adj_E1[seq_idx] = tf_pred

    return adj_E1


def train(
    mdata: md.MuData,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
    warmup_vae: int = 10,
    warmup_grn: int = 50,
    logger: str = "dict",
    log_dir: str = "./runs",
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 10,
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
        Base learning rate (applied to all models).
    device
        Training device ('cuda' or 'cpu').
    warmup_vae
        Epochs before enabling TF2rNet updates.
    warmup_grn
        Epochs before enabling PPI network.
    logger
        Logging backend: 'dict', 'tensorboard', 'wandb'.
    log_dir
        Directory for tensorboard/wandb logs.
    checkpoint_dir
        Directory for checkpoints (None = no checkpoints).
    checkpoint_every
        Save checkpoint every N epochs.
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
        Number of workers for parallel data loading (0 = main process only).
    pretrained_model
        Optional pretrained model from ds.tl.pretrain(). If provided, TF2rNet
        and Enformer weights are initialized from the pretrained model.
    config
        Advanced configuration options. When provided, explicit function
        parameters take precedence over config values.

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
    >>> ds.genome.register_genome("/path/to/hg38.fa")  # doctest: +SKIP
    >>> mdata = ds.read("preprocessed.h5mu")  # doctest: +SKIP
    >>> model = ds.tl.train(mdata, epochs=100, device="cuda")  # doctest: +SKIP

    Training with pretrained model:

    >>> pretrained = ds.tl.pretrain(mdata, epochs=50)  # doctest: +SKIP
    >>> model = ds.tl.train(mdata, pretrained_model=pretrained)  # doctest: +SKIP

    Training with advanced configuration:

    >>> from deepscenic.tl import TrainingConfig
    >>> config = TrainingConfig(beta=0.01, alpha=0.02)  # doctest: +SKIP
    >>> model = ds.tl.train(mdata, epochs=200, config=config)  # doctest: +SKIP
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
        warmup_vae=warmup_vae,
        warmup_grn=warmup_grn,
        lr_vae=lr,
        lr_tf2rnet=base_config.lr_tf2rnet,
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
    )
    config = final_config

    # Setup logging
    training_logger = get_logger(logger, log_dir=log_dir)
    training_logger.log_hyperparams(config.to_dict())

    # Extract metadata from MuData
    tf_names = list(mdata.uns["tfs"])
    gene_names = list(mdata.mod["rna"].var_names)
    region_names = list(mdata.mod["atac"].var_names)

    n_tfs = len(tf_names)
    n_genes = len(gene_names)
    n_regions = len(region_names)

    # Get indices from MuData
    tf_indices = torch.tensor(mdata.uns["tf_indices"])
    gene_indices = torch.arange(n_genes)

    # Get r2g sparse matrix info
    r2g_mask = mdata.uns["r2g_mask"]
    r2g_indices = torch.tensor(np.array([r2g_mask.row, r2g_mask.col]))
    r2g_distances = torch.tensor(r2g_mask.data).float()

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
        ppi_edge_index=ppi_edge_index,
        ppi_genes_idx=ppi_genes_idx,
        ppi_tfs_idx_keys=ppi_tfs_idx_keys,
        ppi_tfs_idx_values=ppi_tfs_idx_values,
        n_hidden=config.n_hidden,
        use_ppi=use_ppi,
        binary_atac=config.binary_atac,
        n_batches=n_batches,
    ).to(device)

    # Initialize sequence models (or use pretrained)
    if pretrained_model is not None:
        tf2rnet = pretrained_model.tf2rnet.to(device)
        enformer = pretrained_model.enformer.to(device)
    else:
        tf2rnet = MotifNet(
            n_tfs=n_tfs,
            bottleneck_size=config.bottleneck_size,
            emb_len=config.emb_len,
        ).to(device)
        enformer = _init_enformer(device)

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

    # Initialize E1 cache (use pretrained if available)
    if pretrained_model is not None:
        adj_E1_cache = pretrained_model.adj_E1.to(device)
    else:
        adj_E1_cache = _init_e1_cache(enformer, tf2rnet, train_seq_loader, device, config)

    # Optimizers
    optimizer_vae = Adam(vae.parameters(), lr=config.lr_vae, weight_decay=config.weight_decay)
    optimizer_tf2rnet = Adam(
        list(tf2rnet.parameters()) + list(enformer.parameters()),
        lr=config.lr_tf2rnet,
        weight_decay=config.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer_vae, T_max=epochs)

    # History and early stopping
    history = TrainingHistory()
    early_stopping = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience else None

    # Resume from checkpoint
    start_epoch = 0
    if resume_from is not None:
        checkpoint = Checkpoint.load(resume_from, map_location=device)
        vae.load_state_dict(checkpoint.vae_state_dict)
        tf2rnet.load_state_dict(checkpoint.tf2rnet_state_dict)
        if checkpoint.enformer_state_dict:
            enformer.load_state_dict(checkpoint.enformer_state_dict)
        optimizer_vae.load_state_dict(checkpoint.optimizer_state_dict)
        if checkpoint.scheduler_state_dict:
            scheduler.load_state_dict(checkpoint.scheduler_state_dict)
        history = checkpoint.history
        adj_E1_cache = checkpoint.adj_E1.to(device)
        start_epoch = checkpoint.epoch + 1

    # Sequence iterator for cycling
    seq_iterator = iter(train_seq_loader)

    # Training loop
    for epoch in range(start_epoch, epochs):
        vae.train()
        tf2rnet.train()
        enformer.train()

        epoch_metrics = {
            "loss": 0.0,
            "rec_rna": 0.0,
            "rec_atac": 0.0,
            "kl": 0.0,
            "e1_sparse": 0.0,
            "e2_sparse": 0.0,
            "ppi": 0.0,
        }
        n_batches_seen = 0

        pbar = tqdm(train_cell_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            x_rna = batch["rna"].to(device)
            x_atac = batch["atac"].to(device)

            # Clone E1 cache
            adj_E1 = adj_E1_cache.clone()

            # Update E1 for a batch of sequences (after warmup)
            if epoch >= warmup_vae:
                # Get next sequence batch (cycle if exhausted)
                try:
                    (sequences,), seq_idx = next(seq_iterator)
                except StopIteration:
                    seq_iterator = iter(train_seq_loader)
                    (sequences,), seq_idx = next(seq_iterator)

                sequences = sequences.to(device)

                # Forward through Enformer + TF2rNet
                optimizer_tf2rnet.zero_grad()
                emb = _get_enformer_embeddings(enformer, sequences, config.bottleneck_size, config.emb_len)
                tf_pred = tf2rnet(emb)

                # Update E1 for this batch
                adj_E1[seq_idx] = tf_pred

                # Update cache (no gradients)
                with torch.no_grad():
                    adj_E1_cache[seq_idx] = tf_pred.detach()

            # VAE forward
            optimizer_vae.zero_grad()
            use_ppi_this_batch = use_ppi and epoch >= warmup_grn
            output = vae(
                x_rna,
                x_atac,
                adj_E1,
                use_ppi=use_ppi_this_batch,
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
                adj_E1_batch=adj_E1[seq_idx] if epoch >= warmup_vae else adj_E1,
                adj_E2=vae.adj_E2,
                r2g_distances=vae.r2g_distances,
                x_rna_ppi=output.x_rna_ppi,
                gene_indices=vae.gene_indices,
                loss_rna=config.loss_rna,
                loss_atac=config.loss_atac,
                dropout_mask_rna=config.dropout_mask_rna,
                dropout_mask_atac=config.dropout_mask_atac,
                beta=config.beta,
                alpha=config.alpha,
                gamma=config.gamma,
                rna_tau=config.rna_tau,
                atac_tau=config.atac_tau,
                use_ppi=use_ppi_this_batch,
            )

            # Add E1 sparsity loss (only for updated batch)
            if epoch >= warmup_vae:
                e1_loss = e1_sparsity_loss(tf_pred) * config.alpha
                total_loss = losses["total"] + e1_loss
            else:
                total_loss = losses["total"]

            # Backward
            total_loss.backward()
            optimizer_vae.step()
            if epoch >= warmup_vae:
                optimizer_tf2rnet.step()

            # Accumulate metrics
            for key in epoch_metrics:
                if key in losses:
                    epoch_metrics[key] += losses[key].item()
            epoch_metrics["loss"] += total_loss.item()
            n_batches_seen += 1

            pbar.set_postfix(loss=total_loss.item())

        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches_seen

        # Step scheduler
        scheduler.step()

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
                    x_atac,
                    adj_E1_cache,
                    use_ppi=use_ppi and epoch >= warmup_grn,
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
                    r2g_distances=vae.r2g_distances,
                    x_rna_ppi=output.x_rna_ppi,
                    gene_indices=vae.gene_indices,
                    loss_rna=config.loss_rna,
                    loss_atac=config.loss_atac,
                    beta=config.beta,
                    alpha=config.alpha,
                    gamma=config.gamma,
                    use_ppi=use_ppi and epoch >= warmup_grn,
                )

                val_loss += losses["total"].item()
                n_val_batches += 1

        val_loss /= n_val_batches
        history.log("test", {"loss": val_loss})
        training_logger.log_metrics({"test/loss": val_loss}, step=epoch)

        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Checkpointing
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = Path(checkpoint_dir) / f"epoch_{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = Checkpoint(
                epoch=epoch,
                vae_state_dict=vae.state_dict(),
                tf2rnet_state_dict=tf2rnet.state_dict(),
                enformer_state_dict=enformer.state_dict(),
                optimizer_state_dict=optimizer_vae.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                history=history,
                config=config,
                adj_E1=adj_E1_cache,
                best_loss=val_loss,
            )
            checkpoint.save(checkpoint_path)

    # Close logger
    training_logger.close()

    # Return trained model
    vae.eval()
    tf2rnet.eval()
    enformer.eval()

    return DeepSCENICModel(
        vae=vae,
        tf2rnet=tf2rnet,
        enformer=enformer,
        adj_E1=adj_E1_cache,
        config=config,
        tf_names=tf_names,
        gene_names=gene_names,
        region_names=region_names,
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
    logger: str = "dict",
    log_dir: str = "./runs",
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
    logger
        Logging backend: 'dict', 'tensorboard', 'wandb'.
    log_dir
        Directory for logs.
    num_workers
        Number of workers for data loading.
    config
        Advanced configuration. Explicit parameters override config values.

    Returns
    -------
    DeepSCENICModel
        Model with finetuned E2 matrix.

    Examples
    --------
    >>> model = ds.tl.train(mdata, epochs=100)  # doctest: +SKIP
    >>> model = ds.tl.finetune(model, mdata, epochs=10000, lr=1e-6)  # doctest: +SKIP
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
    training_logger = get_logger(logger, log_dir=log_dir)
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

    # Training loop
    for epoch in range(finetune_config.epochs):
        vae.train()

        epoch_metrics = {"loss": 0.0, "rec_rna": 0.0, "e2_sparse": 0.0}
        n_batches_seen = 0

        pbar = tqdm(train_cell_loader, desc=f"Finetune {epoch + 1}/{epochs}")
        for batch in pbar:
            x_rna = batch["rna"].to(device)
            x_atac = batch["atac"].to(device)

            optimizer.zero_grad()

            # VAE forward (no PPI, use_mean=False for training)
            output = vae(
                x_rna,
                x_atac,
                adj_E1,
                use_ppi=False,
                use_mean=False,
            )

            # Compute RNA reconstruction loss only
            from ._loss import e2_sparsity_loss, reconstruction_loss

            loss_rec_rna = reconstruction_loss(
                x_rna,
                output.x_rna_rec,
                loss_type=finetune_config.loss_rna,
                dropout_mask=finetune_config.dropout_mask_rna,
            )

            loss_e2_sparse = e2_sparsity_loss(vae.adj_E2, vae.r2g_distances)
            total_loss = loss_rec_rna + loss_e2_sparse * finetune_config.gamma

            total_loss.backward()
            optimizer.step()

            # Accumulate metrics
            epoch_metrics["loss"] += total_loss.item()
            epoch_metrics["rec_rna"] += loss_rec_rna.item()
            epoch_metrics["e2_sparse"] += loss_e2_sparse.item()
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
                    x_atac,
                    adj_E1,
                    use_ppi=False,
                    use_mean=True,
                )

                loss_rec_rna = reconstruction_loss(
                    x_rna,
                    output.x_rna_rec,
                    loss_type=finetune_config.loss_rna,
                    dropout_mask=finetune_config.dropout_mask_rna,
                )
                loss_e2_sparse = e2_sparsity_loss(vae.adj_E2, vae.r2g_distances)

                val_loss += (loss_rec_rna + loss_e2_sparse * finetune_config.gamma).item()
                n_val_batches += 1

        val_loss /= n_val_batches
        history.log("test", {"loss": val_loss})
        training_logger.log_metrics({"test/loss": val_loss}, step=epoch)

        # Step scheduler
        scheduler.step(val_loss)

        # Track best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_e2 = vae.adj_E2.data.clone()

        # Checkpointing
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
    num_workers
        Number of workers for data loading.
    config
        Advanced configuration.

    Returns
    -------
    PretrainedModel
        Pretrained sequence models with cached E1 matrix.

    Examples
    --------
    >>> pretrained = ds.tl.pretrain(mdata, epochs=100)  # doctest: +SKIP
    >>> model = ds.tl.train(mdata, pretrained_model=pretrained)  # doctest: +SKIP
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
    )
    pretrain_config = final_config

    # Setup logging
    training_logger = get_logger(logger, log_dir=log_dir)
    training_logger.log_hyperparams(pretrain_config.to_dict())

    # Extract metadata from MuData
    tf_names = list(mdata.uns["tfs"])
    region_names = list(mdata.mod["atac"].var_names)
    n_tfs = len(tf_names)

    # Initialize models
    tf2rnet = MotifNet(
        n_tfs=n_tfs,
        bottleneck_size=pretrain_config.bottleneck_size,
        emb_len=pretrain_config.emb_len,
    ).to(device)

    enformer = _init_enformer(device)

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
        list(tf2rnet.parameters()) + list(enformer.parameters()),
        lr=pretrain_config.lr,
        weight_decay=pretrain_config.weight_decay,
    )

    # History tracking
    history = TrainingHistory()

    # Training loop
    for epoch in range(pretrain_config.epochs):
        tf2rnet.train()
        enformer.train()

        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(seq_loader, desc=f"Pretrain {epoch + 1}/{epochs}")
        for (sequences,), _seq_idx in pbar:
            sequences = sequences.to(device)

            optimizer.zero_grad()

            # Forward through Enformer + TF2rNet
            emb = _get_enformer_embeddings(
                enformer, sequences, pretrain_config.bottleneck_size, pretrain_config.emb_len
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
        history.log("train", {"loss": avg_loss, "e1_sparse": avg_loss})
        training_logger.log_metrics({"train/loss": avg_loss}, step=epoch)

        # Checkpointing
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = Path(checkpoint_dir) / f"pretrain_epoch_{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "tf2rnet_state_dict": tf2rnet.state_dict(),
                    "enformer_state_dict": enformer.state_dict(),
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
    enformer.eval()
    adj_E1 = _init_e1_cache(enformer, tf2rnet, seq_loader, device, pretrain_config)

    return PretrainedModel(
        tf2rnet=tf2rnet,
        enformer=enformer,
        adj_E1=adj_E1,
        config=pretrain_config,
        tf_names=tf_names,
        region_names=region_names,
    )
