"""I/O functions for deepSCENIC."""

from .bed import read_bed
from .read import read
from .write import write

__all__ = ["read", "read_bed", "read_legacy", "write"]
