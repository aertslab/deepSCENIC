"""Training state management for deepSCENIC."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class TrainingConfig:
    """Configuration for training.

    Parameters
    ----------
    epochs
        Total training epochs
    batch_size
        Cells per batch
    seq_batch_size
        Sequences per batch
    warmup_vae
        Epochs before enabling TF2rNet
    warmup_grn
        Epochs before enabling PPI
    lr_vae
        Learning rate for VAE
    lr_tf2rnet
        Learning rate for TF2rNet
    lr_ppi
        Learning rate for PPI network
    weight_decay
        Weight decay for optimizer
    beta
        KL divergence weight
    alpha
        E1 sparsity + PPI weight
    gamma
        E2 sparsity weight
    rna_tau
        RNA reconstruction weight
    atac_tau
        ATAC reconstruction weight
    loss_rna
        RNA loss type: 'mse', 'mae', 'cosine'
    loss_atac
        ATAC loss type: 'mse', 'mae', 'bce', 'cosine'
    dropout_mask_rna
        Whether to mask RNA loss on zeros
    dropout_mask_atac
        Whether to mask ATAC loss on zeros
    n_hidden
        MLP hidden dimension
    use_ppi
        Whether to use PPI network
    binary_atac
        Whether ATAC is binary
    device
        Training device
    ppi_device
        Separate device for PPI (optional)
    batch_key
        Column in obs for batch correction
    balance_class
        Whether to use weighted sampling
    class_key
        Column in obs for class balancing
    bottleneck_size
        Enformer embedding dimension
    emb_len
        Enformer output sequence length
    seq_len
        DNA sequence length (bp)
    """

    # Training
    epochs: int = 100
    batch_size: int = 64
    seq_batch_size: int = 1000

    # Warmup
    warmup_vae: int = 10
    warmup_grn: int = 50

    # Learning rates
    lr_vae: float = 1e-3
    lr_tf2rnet: float = 1e-3
    lr_ppi: float = 1e-3
    weight_decay: float = 0.0

    # Loss weights
    beta: float = 1e-2  # KL divergence
    alpha: float = 1e-2  # E1 sparsity + PPI
    gamma: float = 1.0  # E2 sparsity
    rna_tau: float = 1.0  # RNA reconstruction
    atac_tau: float = 1.0  # ATAC reconstruction

    # Loss types
    loss_rna: str = "mse"  # 'mse', 'mae', 'cosine'
    loss_atac: str = "mse"  # 'mse', 'mae', 'bce', 'cosine'
    dropout_mask_rna: bool = False
    dropout_mask_atac: bool = False

    # Model
    n_hidden: int = 128
    use_ppi: bool = True
    binary_atac: bool = False

    # Device
    device: str = "cuda"
    ppi_device: str | None = None  # Separate device for PPI

    # Data
    batch_key: str | None = None
    balance_class: bool = False
    class_key: str | None = None
    balance_dars: bool = False  # Whether to upweight DARs in sequence sampling
    num_workers: int = 0  # Number of data loading workers

    # Enformer
    bottleneck_size: int = 3072
    emb_len: int = 5
    seq_len: int = 640

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingHistory:
    """Container for training metrics history."""

    train: dict[str, list[float]] = field(default_factory=dict)
    test: dict[str, list[float]] = field(default_factory=dict)

    def log(self, split: str, metrics: dict[str, float]) -> None:
        """Log metrics for a split."""
        storage = self.train if split == "train" else self.test
        for key, value in metrics.items():
            if key not in storage:
                storage[key] = []
            storage[key].append(value)

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        """Convert to dictionary."""
        return {"train": self.train, "test": self.test}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingHistory:
        """Create from dictionary."""
        history = cls()
        history.train = d.get("train", {})
        history.test = d.get("test", {})
        return history


@dataclass
class Checkpoint:
    """Training checkpoint.

    Stores all state needed to resume training.
    """

    epoch: int
    vae_state_dict: dict[str, Any]
    tf2rnet_state_dict: dict[str, Any]
    enformer_state_dict: dict[str, Any] | None
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None
    history: TrainingHistory
    config: TrainingConfig
    adj_E1: torch.Tensor
    best_loss: float = float("inf")

    def save(self, path: str | Path) -> None:
        """Save checkpoint to file."""
        torch.save(
            {
                "epoch": self.epoch,
                "vae_state_dict": self.vae_state_dict,
                "tf2rnet_state_dict": self.tf2rnet_state_dict,
                "enformer_state_dict": self.enformer_state_dict,
                "optimizer_state_dict": self.optimizer_state_dict,
                "scheduler_state_dict": self.scheduler_state_dict,
                "history": self.history.to_dict(),
                "config": self.config.to_dict(),
                "adj_E1": self.adj_E1,
                "best_loss": self.best_loss,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> Checkpoint:
        """Load checkpoint from file."""
        data = torch.load(path, map_location=map_location)
        history = TrainingHistory.from_dict(data["history"])
        config = TrainingConfig.from_dict(data["config"])
        return cls(
            epoch=data["epoch"],
            vae_state_dict=data["vae_state_dict"],
            tf2rnet_state_dict=data["tf2rnet_state_dict"],
            enformer_state_dict=data.get("enformer_state_dict"),
            optimizer_state_dict=data["optimizer_state_dict"],
            scheduler_state_dict=data.get("scheduler_state_dict"),
            history=history,
            config=config,
            adj_E1=data["adj_E1"],
            best_loss=data.get("best_loss", float("inf")),
        )


class EarlyStopping:
    """Early stopping handler.

    Parameters
    ----------
    patience
        Number of epochs to wait before stopping
    min_delta
        Minimum improvement to consider as progress

    Examples
    --------
    >>> early_stop = EarlyStopping(patience=10)
    >>> for epoch in range(100):
    ...     loss = train_one_epoch()
    ...     if early_stop(loss):
    ...         print("Early stopping!")
    ...         break
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.should_stop = False

    def __call__(self, loss: float) -> bool:
        """Check if training should stop."""
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
