"""deepSCENIC: Deep learning for single-cell Gene Regulatory Networks."""

try:
    from importlib.metadata import version

    __version__ = version("deepscenic")
except Exception:
    __version__ = "0.1.0"  # Fallback for development

from . import data, datasets, models, pl, pp, tl
from .io import read, read_bed, write

__all__ = [
    "__version__",
    "read",
    "read_bed",
    "write",
    "pp",
    "tl",
    "pl",
    "data",
    "datasets",
    "models",
]
