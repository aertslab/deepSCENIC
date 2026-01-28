"""Training state management for deepSCENIC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn


@dataclass
class ModelConfig:
    """Model architecture configuration.

    These settings affect the model structure and must be set before training.
    Once a model is created, these cannot be changed.

    Parameters
    ----------
    n_hidden
        MLP hidden dimension.
    use_ppi
        Whether to use PPI network.
    binary_atac
        Whether ATAC is binary.
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
    Using default settings:

    >>> config = ModelConfig()
    >>> model = ds.tl.train(mdata, config=config, epochs=100)

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
    >>> config = ModelConfig(
    ...     sequence_model=MyModel(),
    ...     bottleneck_size=256,
    ...     emb_len=60,
    ...     seq_len=640,
    ... )
    >>> model = ds.tl.train(mdata, config=config, epochs=100)
    """

    # Model architecture
    n_hidden: int = 128
    use_ppi: bool = True
    binary_atac: bool = False

    # Sequence model settings
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
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingHistory:
    """Container for training metrics history.

    Splits:
    - train: Training metrics (train cells, train features)
    - val: Validation metrics (test cells, train features)
    - val_chrom: Chromosome generalization (test cells, test features)
    """

    train: dict[str, list[float]] = field(default_factory=dict)
    val: dict[str, list[float]] = field(default_factory=dict)
    val_chrom: dict[str, list[float]] = field(default_factory=dict)

    def log(self, split: str, metrics: dict[str, float]) -> None:
        """Log metrics for a split."""
        if split == "train":
            storage = self.train
        elif split == "val_chrom":
            storage = self.val_chrom
        else:
            storage = self.val
        for key, value in metrics.items():
            if key not in storage:
                storage[key] = []
            storage[key].append(value)

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        """Convert to dictionary."""
        return {
            "train": self.train,
            "val": self.val,
            "val_chrom": self.val_chrom,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingHistory:
        """Create from dictionary."""
        history = cls()
        history.train = d.get("train", {})
        history.val = d.get("val", {})
        history.val_chrom = d.get("val_chrom", {})
        return history


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
