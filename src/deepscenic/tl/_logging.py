"""Abstract logging interface for training metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TrainingLogger(ABC):
    """Abstract base class for training loggers."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log a dictionary of metrics at given step."""
        pass

    @abstractmethod
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the logger."""
        pass


class DictLogger(TrainingLogger):
    """Simple logger that stores metrics in a dictionary.

    Useful for testing and when no external logging is needed.

    Examples
    --------
    >>> logger = DictLogger()
    >>> logger.log_metrics({"loss": 0.5}, step=1)
    >>> logger.history["loss"]
    [0.5]
    """

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {}
        self.hyperparams: dict[str, Any] = {}

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        self.hyperparams.update(params)

    def close(self) -> None:
        pass


class TensorboardLogger(TrainingLogger):
    """Tensorboard logger.

    Parameters
    ----------
    log_dir
        Directory for tensorboard logs

    Examples
    --------
    >>> logger = TensorboardLogger(log_dir="./runs/experiment1")
    >>> logger.log_metrics({"loss": 0.5}, step=1)
    >>> logger.close()
    """

    def __init__(self, log_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        # Log hyperparams as text
        lines = ["| Param | Value |", "|-------|-------|"]
        for k, v in params.items():
            lines.append(f"| {k} | {v} |")
        self.writer.add_text("hyperparams", "\n".join(lines))

    def close(self) -> None:
        self.writer.close()


class WandbLogger(TrainingLogger):
    """Weights & Biases logger.

    Parameters
    ----------
    project
        W&B project name
    name
        Run name (optional)
    **kwargs
        Additional arguments passed to wandb.init()

    Examples
    --------
    >>> logger = WandbLogger(project="deepscenic", name="run1")
    >>> logger.log_metrics({"loss": 0.5}, step=1)
    >>> logger.close()
    """

    def __init__(self, project: str, name: str | None = None, **kwargs: Any) -> None:
        import wandb

        self.run = wandb.init(project=project, name=name, **kwargs)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        import wandb

        wandb.log(metrics, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        import wandb

        wandb.config.update(params)

    def close(self) -> None:
        import wandb

        wandb.finish()


def get_logger(
    backend: str = "dict",
    **kwargs: Any,
) -> TrainingLogger:
    """
    Create a training logger.

    Parameters
    ----------
    backend
        One of: 'dict', 'tensorboard', 'wandb'
    **kwargs
        Backend-specific arguments:
        - tensorboard: log_dir (str)
        - wandb: project (str), name (str, optional)

    Returns
    -------
    TrainingLogger
        Logger instance

    Examples
    --------
    >>> logger = get_logger("dict")
    >>> logger = get_logger("tensorboard", log_dir="./runs")
    >>> logger = get_logger("wandb", project="my_project")
    """
    if backend == "dict":
        return DictLogger()
    elif backend == "tensorboard":
        return TensorboardLogger(kwargs.get("log_dir", "./runs"))
    elif backend == "wandb":
        project = kwargs.pop("project", "deepscenic")
        name = kwargs.pop("name", None)
        # Remove log_dir as it's not used by wandb
        kwargs.pop("log_dir", None)
        return WandbLogger(project=project, name=name, **kwargs)
    else:
        raise ValueError(f"Unknown logger backend: {backend}")
