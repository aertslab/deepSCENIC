"""Dataloaders for deepSCENIC training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..pp.basic import _to_dense

if TYPE_CHECKING:
    import mudata as md
    from numpy.typing import NDArray

    from .._genome import Genome


class CellDataset(Dataset):
    """Dataset for cell batches (RNA + ATAC).

    Parameters
    ----------
    rna
        RNA expression matrix (n_cells, n_genes)
    atac
        ATAC accessibility matrix (n_cells, n_regions)
    """

    def __init__(
        self,
        rna: NDArray,
        atac: NDArray,
    ) -> None:
        rna = _to_dense(rna)
        atac = _to_dense(atac)

        self.rna = torch.Tensor(rna)
        self.atac = torch.Tensor(atac)
        self.n_cells: int = rna.shape[0]

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "rna": self.rna[idx],
            "atac": self.atac[idx],
            "idx": torch.tensor(idx),
        }
        return item


class SequenceDatasetWithIndex(Dataset):
    """Wrapper that adds index to GenomeIntervalDataset.

    Parameters
    ----------
    genome_dataset
        GenomeIntervalDataset instance.
    """

    def __init__(self, genome_dataset: Dataset) -> None:
        self.dataset = genome_dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq = self.dataset[idx]
        return seq, idx


def build_cell_dataloader(
    mdata: md.MuData,
    split: str = "train",
    batch_size: int = 64,
    shuffle: bool = True,
    balance_class: bool = False,
    class_key: str | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build dataloader for cell batches.

    Parameters
    ----------
    mdata
        MuData with rna and atac modalities.
    split
        'train' or 'test'.
    batch_size
        Cells per batch.
    shuffle
        Whether to shuffle.
    balance_class
        Whether to use weighted sampling for class balance.
    class_key
        Column in obs for class balancing.
    num_workers
        Number of data loading workers.

    Returns
    -------
    DataLoader
        Cell dataloader.
    """
    # Get split mask
    mask = mdata.obs["split"] == split

    # Extract data
    rna = mdata.mod["rna"][mask].X
    atac = mdata.mod["atac"][mask].X

    dataset = CellDataset(rna, atac)

    # Weighted sampling for class balance
    sampler = None
    if balance_class and class_key is not None:
        from sklearn.utils.class_weight import compute_class_weight

        classes = mdata.obs.loc[mask, class_key].values
        unique_classes = np.unique(classes)
        weights = compute_class_weight("balanced", classes=unique_classes, y=classes)
        sample_weights = [weights[np.where(unique_classes == c)[0][0]] for c in classes]
        sampler = WeightedRandomSampler(sample_weights, len(dataset))  # type: ignore
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_sequence_dataloader(
    regions: list[str],
    genome: Genome | None = None,
    batch_size: int = 1000,
    shuffle: bool = True,
    shift_augs: tuple[int, int] = (-3, 3),
    rc_aug: bool = True,
    *,
    context_length: int,
    num_workers: int = 0,
    balance_dars: bool = False,
    dar_indices: NDArray | None = None,
) -> DataLoader:
    """
    Build dataloader for DNA sequences.

    Parameters
    ----------
    regions
        List of region strings in "chr:start-end" format.
        Typically from ``mdata.mod["atac"].var_names``.
    genome
        Genome instance for sequence extraction. If None, uses the
        globally registered genome from ``ds.register_genome()``.
    batch_size
        Sequences per batch.
    shuffle
        Whether to shuffle.
    shift_augs
        Random shift range for augmentation (training only).
    rc_aug
        Whether to use reverse complement augmentation.
    context_length
        Sequence length in base pairs. Should match your model's expected input.
    num_workers
        Number of data loading workers.
    balance_dars
        Whether to upweight DARs.
    dar_indices
        Indices of DAR regions for upweighting.

    Returns
    -------
    DataLoader
        Sequence dataloader.

    Examples
    --------
    >>> import deepscenic as ds
    >>> ds.genome.register_genome("/path/to/hg38.fa")
    >>> regions = mdata.mod["atac"].var_names.tolist()
    >>> loader = ds.tl._dataloaders.build_sequence_dataloader(
    ...     regions=regions,
    ...     batch_size=1000,
    ... )
    """
    from .._genome import GenomeIntervalDataset, get_genome

    if genome is None:
        genome = get_genome()

    ds = GenomeIntervalDataset(
        regions=regions,
        genome=genome,
        context_length=context_length,
        shift_augs=shift_augs,
        rc_aug=rc_aug,
    )

    dataset = SequenceDatasetWithIndex(ds)

    # DAR upweighting
    sampler = None
    if balance_dars and dar_indices is not None:
        weights = np.ones(len(dataset))
        weights[dar_indices] = 1.5
        sampler = WeightedRandomSampler(weights, len(dataset))  # type: ignore
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
