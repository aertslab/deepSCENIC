"""Training state management for deepSCENIC."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import nn


@dataclass
class TrainingConfig:
    """Configuration for training.

    Parameters
    ----------
    epochs
        Total training epochs.
    batch_size
        Cells per batch.
    seq_batch_size
        Sequences per batch.
    warmup_vae
        Epochs before enabling TF2rNet.
    warmup_grn
        Epochs before enabling PPI.
    lr_vae
        Learning rate for VAE.
    lr_tf2rnet
        Learning rate for TF2rNet.
    lr_ppi
        Learning rate for PPI network.
    weight_decay
        Weight decay for optimizer.
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
    loss_rna
        RNA loss type: 'mse', 'mae', 'cosine'.
    loss_atac
        ATAC loss type: 'mse', 'mae', 'bce', 'cosine'.
    dropout_mask_rna
        Whether to mask RNA loss on zeros.
    dropout_mask_atac
        Whether to mask ATAC loss on zeros.
    n_hidden
        MLP hidden dimension.
    use_ppi
        Whether to use PPI network.
    binary_atac
        Whether ATAC is binary.
    device
        Training device.
    ppi_device
        Separate device for PPI (optional).
    batch_key
        Column in obs for batch correction.
    balance_class
        Whether to use weighted sampling.
    class_key
        Column in obs for class balancing.
    bottleneck_size
        Sequence model embedding dimension per position.
    emb_len
        Sequence model output length (positions).
    seq_len
        DNA sequence length in base pairs. Default is 640bp for Enformer
        compatibility (target_length=5). Set to match your sequence_model's input.
    sequence_model
        Custom sequence embedding model. If None (default), uses Enformer.
        Custom models must accept (batch, seq_len, 4) input and return
        (batch, bottleneck_size * emb_len) flattened embeddings.

    Examples
    --------
    Using default Enformer:

    >>> config = TrainingConfig(epochs=100)
    >>> model = ds.tl.train(mdata, config=config)
    Using a custom sequence model:

    >>> import torch
    >>> class MyModel(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv = torch.nn.Conv1d(4, 256, kernel_size=15, padding=7)
    ...         self.pool = torch.nn.AdaptiveAvgPool1d(60)
    ...
    ...     def forward(self, x):
    ...         # x: (batch, seq_len, 4)
    ...         x = x.transpose(1, 2)  # (batch, 4, seq_len)
    ...         x = self.conv(x)  # (batch, 256, seq_len)
    ...         x = self.pool(x)  # (batch, 256, 60)
    ...         return x.flatten(1)  # (batch, 15360)
    ...
    >>> config = TrainingConfig(
    ...     sequence_model=MyModel(),
    ...     bottleneck_size=256,
    ...     emb_len=60,
    ...     seq_len=640,
    ... )
    >>> model = ds.tl.train(mdata, config=config)    """

    # Training
    epochs: int = 100
    batch_size: int = 64
    seq_batch_size: int = 1000

    # Warmup
    warmup_vae: int = 0
    warmup_grn: int = 0

    # Learning rates
    lr_vae: float = 1e-4
    lr_tf2rnet: float = 1e-4
    lr_ppi: float = 1e-3
    weight_decay: float = 0.0

    # Loss weights
    beta: float = 1e-2  # KL divergence
    alpha: float = 1e-2  # E1 sparsity + PPI
    gamma: float = 1.0  # E2 sparsity
    rna_tau: float = 1.0  # RNA reconstruction
    atac_tau: float = 1.0  # ATAC reconstruction

    # Loss types
    loss_rna: str = "mae"  # 'mse', 'mae', 'cosine'
    loss_atac: str = "cosine"  # 'mse', 'mae', 'bce', 'cosine'
    dropout_mask_rna: bool = False
    dropout_mask_atac: bool = False

    # Model
    n_hidden: int = 128
    use_ppi: bool = True
    binary_atac: bool = False

    # Device
    device: str = "cuda"
    ppi_device: str | None = None  # Separate device for PPI

    # Scheduler
    use_scheduler: bool = False  # Whether to use CosineAnnealingLR scheduler

    # Data
    batch_key: str | None = None
    balance_class: bool = False
    class_key: str | None = None
    balance_dars: bool = False  # Whether to upweight DARs in sequence sampling
    num_workers: int = 0  # Number of data loading workers

    # Sequence model
    bottleneck_size: int = 3072
    emb_len: int = 5
    seq_len: int = 640
    sequence_model: nn.Module | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes non-serializable sequence_model)."""
        d = dict(self.__dict__)
        d.pop("sequence_model", None)
        return d

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


@dataclass
class PretrainConfig:
    """Configuration for sequence model pretraining.

    Parameters
    ----------
    epochs
        Pretraining epochs.
    batch_size
        Sequences per batch.
    lr
        Learning rate.
    weight_decay
        Optimizer weight decay.
    alpha
        E1 sparsity weight.
    device
        Training device.
    num_workers
        Number of data loading workers.
    bottleneck_size
        Sequence model embedding dimension per position.
    emb_len
        Sequence model output length (positions).
    seq_len
        DNA sequence length in base pairs. Default is 640bp for Enformer
        compatibility (target_length=5). Set to match your sequence_model's input.
    sequence_model
        Custom sequence embedding model. If None (default), uses Enformer.
        Custom models must accept (batch, seq_len, 4) input and return
        (batch, bottleneck_size * emb_len) flattened embeddings.

    Examples
    --------
    >>> config = PretrainConfig(epochs=100, lr=1e-4)
    >>> pretrained = ds.tl.pretrain(mdata, config=config)    """

    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.0
    alpha: float = 1e-2
    device: str = "cuda"
    num_workers: int = 0
    bottleneck_size: int = 3072
    emb_len: int = 5
    seq_len: int = 640
    sequence_model: nn.Module | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes non-serializable sequence_model)."""
        d = dict(self.__dict__)
        d.pop("sequence_model", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PretrainConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FinetuneConfig:
    """Configuration for E2 finetuning.

    Parameters
    ----------
    epochs
        Finetuning epochs (typically 10000).
    batch_size
        Cells per batch.
    lr
        Learning rate (very small, e.g., 1e-6).
    lr_patience
        Epochs before reducing LR (ReduceLROnPlateau).
    reinit_e2
        Reinitialize E2 to zeros before finetuning.
    gamma
        E2 sparsity weight.
    loss_rna
        RNA loss type: 'mse', 'mae', 'cosine'.
    dropout_mask_rna
        Whether to mask RNA loss on zeros.
    device
        Training device.
    num_workers
        Number of data loading workers.
    batch_key
        Column in obs for batch correction.

    Examples
    --------
    >>> config = FinetuneConfig(epochs=10000, lr=1e-6)
    >>> model = ds.tl.finetune(model, mdata, config=config)    """

    epochs: int = 10000
    batch_size: int = 64
    lr: float = 1e-6
    lr_patience: int = 50
    reinit_e2: bool = True
    gamma: float = 1.0
    loss_rna: str = "mse"
    dropout_mask_rna: bool = False
    device: str = "cuda"
    num_workers: int = 0
    batch_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FinetuneConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class EarlyStopping:
    """Early stopping handler.

    Parameters
    ----------
    patience
        Number of epochs to wait before stopping.
    min_delta
        Minimum improvement to consider as progress.

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
