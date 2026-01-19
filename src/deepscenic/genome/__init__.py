"""Genome handling for deepSCENIC.

This module provides genome sequence extraction and registration for use
across deepSCENIC. The key pattern is to register a genome once at the
start of a session, then all functions will automatically use it.

Examples
--------
>>> import deepscenic as ds

Register a genome globally:

>>> ds.genome.register_genome("/path/to/hg38.fa")  # doctest: +SKIP

Now training functions will automatically use this genome:

>>> model = ds.tl.train(mdata, epochs=100)  # doctest: +SKIP

You can also use the genome directly:

>>> genome = ds.genome.get_genome()  # doctest: +SKIP
>>> seq = genome.fetch("chr1", 1000, 2000)  # doctest: +SKIP
"""

from ._dataset import GenomeIntervalDataset
from ._genome import Genome, clear_genome, get_genome, register_genome

__all__ = [
    "Genome",
    "GenomeIntervalDataset",
    "clear_genome",
    "get_genome",
    "register_genome",
]
