"""deepSCENIC: Deep learning for single-cell Gene Regulatory Networks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

try:
    from importlib.metadata import version

    __version__ = version("deepscenic")
except Exception:
    __version__ = "0.1.0"  # Fallback for development

# Eager imports (lightweight, frequently used)
from ._data import (
    SchemaError,
    SchemaWarning,
    is_valid_schema,
    validate_schema,
)
from ._datasets import (
    clear_cache,
    fetch_gene_annotation,
    fetch_tf_collection,
    get_cache_info,
)
from ._genome import (
    Genome,
    GenomeIntervalDataset,
    clear_genome,
    get_genome,
    register_genome,
)
from ._io import read, read_bed, write

if TYPE_CHECKING:
    from . import models, pl, pp, tl


def __getattr__(name: str):
    """Lazy load heavy submodules."""
    import importlib

    if name in {"pp", "tl", "pl", "models"}:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__


__all__ = [
    "__version__",
    # I/O
    "read",
    "read_bed",
    "write",
    # Datasets
    "fetch_tf_collection",
    "fetch_gene_annotation",
    "clear_cache",
    "get_cache_info",
    # Genome
    "Genome",
    "GenomeIntervalDataset",
    "register_genome",
    "get_genome",
    "clear_genome",
    # Data utilities
    "validate_schema",
    "is_valid_schema",
    "SchemaError",
    "SchemaWarning",
    # Submodules (lazy loaded)
    "pp",
    "tl",
    "pl",
    "models",
]

# Configure package-level logging to show INFO messages by default
_logger = logging.getLogger("deepscenic")
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    _logger.addHandler(_handler)
