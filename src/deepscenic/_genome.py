"""Genome class and registration for sequence extraction.

The GenomeIntervalDataset class is adapted from enformer-pytorch
(https://github.com/lucidrains/enformer-pytorch) by Phil Wang (lucidrains).

MIT License

Copyright (c) 2021 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import logging
import random
from functools import cached_property
from pathlib import Path

import torch
from torch.utils.data import Dataset

__all__ = [
    "Genome",
    "GenomeIntervalDataset",
    "register_genome",
    "get_genome",
    "clear_genome",
]

log = logging.getLogger("deepscenic.genome")

# Module-level global genome
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
    fasta_file
        Path to the FASTA file.
    name
        Genome name derived from filename.

    Examples
    --------
    Register a genome globally for use across deepSCENIC:

    >>> import deepscenic as ds
    >>> ds.register_genome("/path/to/hg38.fa")
    >>> # Now training and other functions will use this genome automatically
    >>> model = ds.tl.train(mdata, epochs=100)

    Create a Genome instance directly:
    >>> genome = ds.Genome("/path/to/hg38.fa")
    >>> seq = genome.fetch("chr1", 1000, 2000)
    >>> onehot = genome.fetch_onehot("chr1", 1000, 2000)
    """

    def __init__(self, fasta_file: str | Path) -> None:
        self.fasta_file = Path(fasta_file)
        if not self.fasta_file.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_file}")
        self.name = self.fasta_file.stem

    def __getstate__(self) -> dict:
        """Prepare for pickling - exclude pyfaidx.Fasta (not pickle-safe)."""
        state = self.__dict__.copy()
        # Remove cached properties that contain unpicklable objects
        state.pop("_fasta", None)
        state.pop("_chrom_map", None)
        state.pop("_has_chr_prefix", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore from pickling - cached properties will be recreated on first access."""
        self.__dict__.update(state)

    @cached_property
    def _fasta(self):
        """Lazy-load FASTA file using pyfaidx."""
        import pyfaidx

        return pyfaidx.Fasta(str(self.fasta_file))

    @cached_property
    def _has_chr_prefix(self) -> bool:
        """Check if FASTA chromosomes use 'chr' prefix."""
        chroms = list(self._fasta.keys())
        # Check if any standard chromosome has chr prefix
        return any(c.startswith("chr") for c in chroms if c.lstrip("chr") in ("1", "2", "X", "Y"))

    @cached_property
    def _chrom_map(self) -> dict[str, str]:
        """Build mapping from query chromosome names to FASTA chromosome names.

        Handles chr prefix mismatches automatically.
        """
        fasta_chroms = set(self._fasta.keys())
        chrom_map = {}

        for chrom in fasta_chroms:
            # Map exact match
            chrom_map[chrom] = chrom

            # Map with/without chr prefix
            if chrom.startswith("chr"):
                chrom_map[chrom[3:]] = chrom  # "1" -> "chr1"
            else:
                chrom_map[f"chr{chrom}"] = chrom  # "chr1" -> "1"

        return chrom_map

    def _normalize_chrom(self, chrom: str) -> str:
        """Normalize chromosome name to match FASTA convention.

        Parameters
        ----------
        chrom
            Input chromosome name (e.g., "chr1" or "1").

        Returns
        -------
        str
            Chromosome name as it appears in the FASTA file.

        Raises
        ------
        KeyError
            If chromosome cannot be found in FASTA (even with prefix normalization).
        """
        if chrom in self._chrom_map:
            return self._chrom_map[chrom]
        raise KeyError(f"Chromosome '{chrom}' not found in {self.fasta_file} (tried with/without 'chr' prefix)")

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
        >>> genome = ds.Genome("/path/to/hg38.fa")
        >>> genome.fetch("chr1", 1000, 1010)
        'ACGTACGTAC'
        """
        # Normalize chromosome name (handles chr prefix mismatch)
        chrom = self._normalize_chrom(chrom)
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
        >>> genome = ds.Genome("/path/to/hg38.fa")
        >>> onehot = genome.fetch_onehot("chr1", 1000, 1640)
        >>> onehot.shape
        torch.Size([4, 640])
        """
        seq = self.fetch(chrom, start, end, rc=rc)
        return self._seq_to_onehot(seq)

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Return reverse complement of DNA sequence.

        Parameters
        ----------
        seq
            DNA sequence string (uppercase).

        Returns
        -------
        str
            Reverse complement sequence.

        Examples
        --------
        >>> Genome._reverse_complement("ACGT")
        'ACGT'
        >>> Genome._reverse_complement("AAAA")
        'TTTT'
        """
        from tangermeme.utils import reverse_complement

        return reverse_complement(seq)

    @staticmethod
    def _seq_to_onehot(seq: str) -> torch.Tensor:
        """Convert DNA sequence to one-hot encoding.

        Uses tangermeme's encoding with numba-compiled backend.

        Parameters
        ----------
        seq
            DNA sequence string (uppercase).

        Returns
        -------
        torch.Tensor
            One-hot encoded tensor with shape (4, length).
            Channels are ordered A, C, G, T.
            N bases are encoded as all zeros.

        Examples
        --------
        >>> onehot = Genome._seq_to_onehot("ACGT")
        >>> onehot.shape
        torch.Size([4, 4])
        """
        from tangermeme.utils import one_hot_encode

        return one_hot_encode(seq, dtype=torch.float32)


class GenomeIntervalDataset(Dataset):
    """Dataset for loading DNA sequences from genomic intervals.

    Extracts one-hot encoded DNA sequences for a list of genomic regions,
    with optional shift and reverse complement augmentation.

    Parameters
    ----------
    regions
        List of region strings in "chr:start-end" format.
    genome
        Genome instance for sequence extraction.
    context_length
        Sequence length in base pairs to extract. Regions are centered and
        padded/trimmed to this length. Should match your model's expected input.
    shift_augs
        Tuple of (min_shift, max_shift) for random position augmentation.
        Set to (0, 0) for no augmentation.
    rc_aug
        If True, randomly return reverse complement (50% probability).

    Examples
    --------
    >>> import deepscenic as ds
    >>> genome = ds.Genome("/path/to/hg38.fa")
    >>> regions = ["chr1:1000-1640", "chr1:2000-2640", "chr1:3000-3640"]
    >>> dataset = ds.GenomeIntervalDataset(
    ...     regions=regions,
    ...     genome=genome,
    ...     context_length=640,
    ...     shift_augs=(-3, 3),
    ...     rc_aug=True,
    ... )
    >>> len(dataset)
    3
    >>> (seq,) = dataset[0]
    >>> seq.shape
    torch.Size([640, 4])
    """

    def __init__(
        self,
        regions: list[str],
        genome: Genome,
        context_length: int,
        shift_augs: tuple[int, int] = (0, 0),
        rc_aug: bool = False,
    ) -> None:
        self.regions = regions
        self.genome = genome
        self.context_length = context_length
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug

        # Pre-parse all regions for efficiency
        self._parsed_regions = [self._parse_region(r) for r in regions]

    @staticmethod
    def _parse_region(region: str) -> tuple[str, int, int]:
        """Parse region string to (chrom, start, end).

        Parameters
        ----------
        region
            Region string in "chr:start-end" format.

        Returns
        -------
        tuple[str, int, int]
            Chromosome name, start position, end position.
        """
        chrom, coords = region.split(":")
        start, end = map(int, coords.split("-"))
        return chrom, start, end

    def __len__(self) -> int:
        """Return the number of regions in the dataset."""
        return len(self.regions)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get one-hot encoded sequence for region.

        Parameters
        ----------
        idx
            Region index.

        Returns
        -------
        torch.Tensor
            One-hot encoded sequence with shape (context_length, 4).
        """
        chrom, start, end = self._parsed_regions[idx]

        # Center the region
        center = (start + end) // 2
        half_len = self.context_length // 2

        # Apply random shift augmentation
        if self.shift_augs != (0, 0):
            shift = random.randint(self.shift_augs[0], self.shift_augs[1])
            center += shift

        # Calculate sequence boundaries
        seq_start = center - half_len
        seq_end = center + half_len

        # Apply reverse complement augmentation
        rc = self.rc_aug and random.random() < 0.5

        # Fetch one-hot encoded sequence: (4, context_length)
        onehot = self.genome.fetch_onehot(chrom, seq_start, seq_end, rc=rc)

        # Transpose to (context_length, 4) for Enformer compatibility
        return onehot.T


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
    >>> ds.register_genome("/path/to/hg38.fa")
    >>> # Now all functions will use this genome
    >>> model = ds.tl.train(mdata, epochs=100)
    """
    global _genome

    if isinstance(fasta_file, Genome):
        _genome = fasta_file
    else:
        _genome = Genome(fasta_file)

    # Log chromosome naming convention
    chr_style = "with 'chr' prefix" if _genome._has_chr_prefix else "without 'chr' prefix"
    log.info(f"Registered genome: {_genome.name} ({_genome.fasta_file}), chromosomes {chr_style}")


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
    >>> ds.register_genome("/path/to/hg38.fa")
    >>> genome = ds.get_genome()
    >>> genome.chromosomes[:3]
    ['chr1', 'chr2', 'chr3']
    """
    if _genome is None:
        raise RuntimeError("No genome registered. Call ds.register_genome(fasta_file) first.")
    return _genome


def clear_genome() -> None:
    """Clear the globally registered genome.

    Useful for testing or switching between different genomes.
    """
    global _genome
    _genome = None
