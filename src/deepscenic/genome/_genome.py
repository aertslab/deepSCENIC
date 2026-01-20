"""Genome class and registration for sequence extraction."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import torch

log = logging.getLogger("deepscenic.genome")

if TYPE_CHECKING:
    pass

# Module-level global genome (CREsted-style singleton pattern)
_genome: Genome | None = None


class Genome:
    """Genome class for extracting DNA sequences from FASTA files.

    Wraps pyfaidx for lazy-loaded FASTA access with sequence extraction
    and one-hot encoding capabilities.

    Parameters
    ----------
    fasta_file
        Path to genome FASTA file (must be indexed with samtools faidx).

    Attributes
    ----------
    fasta_file : Path
        Path to the FASTA file.
    name : str
        Genome name derived from filename.

    Examples
    --------
    Register a genome globally for use across deepSCENIC:

    >>> import deepscenic as ds
    >>> ds.genome.register_genome("/path/to/hg38.fa")
    >>> # Now training and other functions will use this genome automatically
    >>> model = ds.tl.train(mdata, epochs=100)  # doctest: +SKIP

    Create a Genome instance directly:

    >>> genome = ds.genome.Genome("/path/to/hg38.fa")  # doctest: +SKIP
    >>> seq = genome.fetch("chr1", 1000, 2000)  # doctest: +SKIP
    >>> onehot = genome.fetch_onehot("chr1", 1000, 2000)  # doctest: +SKIP

    Notes
    -----
    The FASTA file must be indexed with ``samtools faidx`` before use.
    The index file (.fai) should be in the same directory as the FASTA.
    """

    def __init__(self, fasta_file: str | Path) -> None:
        self.fasta_file = Path(fasta_file)
        if not self.fasta_file.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_file}")
        self.name = self.fasta_file.stem

    @cached_property
    def _fasta(self):
        """Lazy-load FASTA file using pyfaidx."""
        import pyfaidx

        return pyfaidx.Fasta(str(self.fasta_file))

    @property
    def chromosomes(self) -> list[str]:
        """List of chromosome names in the FASTA."""
        return list(self._fasta.keys())

    def fetch(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        rc: bool = False,
    ) -> str:
        """Fetch DNA sequence for a genomic region.

        Parameters
        ----------
        chrom
            Chromosome name.
        start
            Start position (0-based, inclusive).
        end
            End position (0-based, exclusive).
        rc
            If True, return reverse complement.

        Returns
        -------
        str
            DNA sequence (uppercase).

        Examples
        --------
        >>> genome = ds.genome.Genome("/path/to/hg38.fa")  # doctest: +SKIP
        >>> genome.fetch("chr1", 1000, 1010)  # doctest: +SKIP
        'ACGTACGTAC'
        """
        chrom_len = len(self._fasta[chrom])

        # Calculate padding for out-of-bounds coordinates
        pad_left = max(0, -start)
        pad_right = max(0, end - chrom_len)

        # Clamp to valid range
        fetch_start = max(0, start)
        fetch_end = min(chrom_len, end)

        seq = str(self._fasta[chrom][fetch_start:fetch_end]).upper()
        seq = "N" * pad_left + seq + "N" * pad_right

        if rc:
            seq = self._reverse_complement(seq)

        return seq

    def fetch_onehot(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        rc: bool = False,
    ) -> torch.Tensor:
        """Fetch one-hot encoded sequence.

        Returns tensor of shape (4, length) with channels A, C, G, T.
        N bases are encoded as 0.25 for each channel.

        Parameters
        ----------
        chrom
            Chromosome name.
        start
            Start position (0-based, inclusive).
        end
            End position (0-based, exclusive).
        rc
            If True, return reverse complement.

        Returns
        -------
        torch.Tensor
            One-hot encoded sequence with shape (4, length).

        Examples
        --------
        >>> genome = ds.genome.Genome("/path/to/hg38.fa")  # doctest: +SKIP
        >>> onehot = genome.fetch_onehot("chr1", 1000, 1640)  # doctest: +SKIP
        >>> onehot.shape  # doctest: +SKIP
        torch.Size([4, 640])
        """
        seq = self.fetch(chrom, start, end, rc=rc)
        return self._seq_to_onehot(seq)

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Return reverse complement of DNA sequence."""
        complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        return "".join(complement.get(base, "N") for base in reversed(seq))

    @staticmethod
    def _seq_to_onehot(seq: str) -> torch.Tensor:
        """Convert DNA sequence to one-hot encoding.

        Parameters
        ----------
        seq
            DNA sequence string (uppercase).

        Returns
        -------
        torch.Tensor
            One-hot encoded tensor with shape (4, length).
            Channels are ordered A, C, G, T.
            N bases are encoded as 0.25 for each channel.
        """
        base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

        length = len(seq)
        onehot = torch.zeros(4, length)

        for i, base in enumerate(seq):
            idx = base_to_idx.get(base)
            if idx is not None:
                onehot[idx, i] = 1.0
            else:
                # N or unknown bases: uniform 0.25 distribution
                onehot[:, i] = 0.25

        return onehot


def register_genome(fasta_file: str | Path | Genome) -> None:
    """Register a genome globally for use across deepSCENIC.

    Once registered, functions like ``ds.tl.train()`` will automatically
    use this genome for sequence extraction without needing to pass
    the fasta_file parameter.

    Parameters
    ----------
    fasta_file
        Path to genome FASTA file, or an existing Genome instance.

    Examples
    --------
    >>> import deepscenic as ds
    >>> ds.genome.register_genome("/path/to/hg38.fa")  # doctest: +SKIP
    >>> # Now all functions will use this genome
    >>> model = ds.tl.train(mdata, epochs=100)  # doctest: +SKIP
    """
    global _genome

    if isinstance(fasta_file, Genome):
        _genome = fasta_file
    else:
        _genome = Genome(fasta_file)

    log.info(f"Registered genome: {_genome.name} ({_genome.fasta_file})")


def get_genome() -> Genome:
    """Get the globally registered genome.

    Returns
    -------
    Genome
        The registered genome instance.

    Raises
    ------
    RuntimeError
        If no genome has been registered.

    Examples
    --------
    >>> import deepscenic as ds
    >>> ds.genome.register_genome("/path/to/hg38.fa")  # doctest: +SKIP
    >>> genome = ds.genome.get_genome()  # doctest: +SKIP
    >>> genome.chromosomes[:3]  # doctest: +SKIP
    ['chr1', 'chr2', 'chr3']
    """
    if _genome is None:
        raise RuntimeError(
            "No genome registered. Call ds.genome.register_genome(fasta_file) first."
        )
    return _genome


def clear_genome() -> None:
    """Clear the globally registered genome.

    Useful for testing or switching between different genomes.
    """
    global _genome
    _genome = None
