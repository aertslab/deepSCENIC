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


class CellDataset(Dataset):
    """Dataset for cell batches (RNA + ATAC).

    Parameters
    ----------
    rna
        RNA expression matrix (n_cells, n_genes)
    atac
        ATAC accessibility matrix (n_cells, n_regions)
    batch_ids
        Optional one-hot batch IDs (n_cells, n_batches)
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
        self.n_cells = rna.shape[0]

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
    """Wrapper that adds index to enformer-pytorch GenomeIntervalDataset.

    Parameters
    ----------
    genome_dataset
        GenomeIntervalDataset from enformer-pytorch
    """

    def __init__(self, genome_dataset: Dataset) -> None:
        self.dataset = genome_dataset

    def __len__(self) -> int:
        return len(self.dataset)

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
    """
    Build dataloader for cell batches.

    Parameters
    ----------
    mdata
        MuData with rna and atac modalities
    split
        'train' or 'test'
    batch_size
        Cells per batch
    shuffle
        Whether to shuffle
    balance_class
        Whether to use weighted sampling for class balance
    class_key
        Column in obs for class balancing
    num_workers
        Number of data loading workers

    Returns
    -------
    DataLoader
        Cell dataloader
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
    bed_file: str,
    fasta_file: str,
    batch_size: int = 1000,
    shuffle: bool = True,
    shift_augs: tuple[int, int] = (-3, 3),
    rc_aug: bool = True,
    context_length: int = 640,
    num_workers: int = 0,
    balance_dars: bool = False,
    dar_indices: NDArray | None = None,
) -> DataLoader:
    """
    Build dataloader for DNA sequences.

    Parameters
    ----------
    bed_file
        Path to BED file with region coordinates
    fasta_file
        Path to genome FASTA file
    batch_size
        Sequences per batch
    shuffle
        Whether to shuffle
    shift_augs
        Random shift range for augmentation (training only)
    rc_aug
        Whether to use reverse complement augmentation
    context_length
        Sequence length (bp)
    num_workers
        Number of data loading workers
    balance_dars
        Whether to upweight DARs
    dar_indices
        Indices of DAR regions for upweighting

    Returns
    -------
    DataLoader
        Sequence dataloader
    """
    from enformer_pytorch import GenomeIntervalDataset

    ds = GenomeIntervalDataset(
        bed_file=bed_file,
        fasta_file=fasta_file,
        shift_augs=shift_augs,
        rc_aug=rc_aug,
        context_length=context_length,
    )

    dataset = SequenceDatasetWithIndex(ds)

    # DAR upweighting
    sampler = None
    if balance_dars and dar_indices is not None:
        weights = np.ones(len(dataset))
        weights[dar_indices] = 1.5
        sampler = WeightedRandomSampler(weights, len(dataset))
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def collate_cell_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for cell batches.

    Parameters
    ----------
    batch
        List of dictionaries from CellDataset

    Returns
    -------
    dict[str, torch.Tensor]
        Batched tensors
    """
    result = {
        "rna": torch.stack([item["rna"] for item in batch]),
        "atac": torch.stack([item["atac"] for item in batch]),
        "idx": torch.stack([item["idx"] for item in batch]),
    }
    if "batch_id" in batch[0]:
        result["batch_id"] = torch.stack([item["batch_id"] for item in batch])
    return result
