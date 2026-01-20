"""Sequence dataset for genomic interval loading."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from ._genome import Genome


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
        Desired sequence length. Regions are centered and padded/trimmed
        symmetrically to this length.
    shift_augs
        Tuple of (min_shift, max_shift) for random position augmentation.
        Set to (0, 0) for no augmentation.
    rc_aug
        If True, randomly return reverse complement (50% probability).

    Examples
    --------
    >>> import deepscenic as ds
    >>> genome = ds.genome.Genome("/path/to/hg38.fa")  # doctest: +SKIP
    >>> regions = ["chr1:1000-1640", "chr1:2000-2640", "chr1:3000-3640"]
    >>> dataset = ds.genome.GenomeIntervalDataset(
    ...     regions=regions,
    ...     genome=genome,
    ...     context_length=640,
    ...     shift_augs=(-3, 3),
    ...     rc_aug=True,
    ... )  # doctest: +SKIP
    >>> len(dataset)  # doctest: +SKIP
    3
    >>> (seq,) = dataset[0]  # doctest: +SKIP
    >>> seq.shape  # doctest: +SKIP
    torch.Size([640, 4])
    """

    def __init__(
        self,
        regions: list[str],
        genome: Genome,
        context_length: int = 640,
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        """Get one-hot encoded sequence for region.

        Parameters
        ----------
        idx
            Region index.

        Returns
        -------
        tuple[torch.Tensor]
            Tuple containing one-hot encoded sequence with shape
            (context_length, 4), matching Enformer's expected input format.
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
        return (onehot.T,)
