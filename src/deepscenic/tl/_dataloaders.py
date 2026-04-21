"""Dataloaders for deepSCENIC training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .._types import LazyImpute
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
    batch_id
        One-hot encoded batch IDs (n_cells, n_batches), optional.
    """

    def __init__(
        self,
        rna: NDArray,
        atac: NDArray | LazyImpute,
        batch_id: NDArray | None = None,
    ) -> None:
        rna = _to_dense(rna)

        if not isinstance(atac, LazyImpute):
            atac = _to_dense(atac)

        self.rna = torch.Tensor(rna)

        self.atac: torch.Tensor | LazyImpute
        if not isinstance(atac, LazyImpute):
            self.atac = torch.Tensor(atac)
        else:
            # pass atac as lazy compute and create Tensor with calling
            # __getitem__
            self.atac = atac

        self.batch_id = torch.Tensor(batch_id) if batch_id is not None else None
        self.n_cells: int = rna.shape[0]

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "rna": self.rna[idx],
            "atac": self.atac[idx] if not isinstance(self.atac, LazyImpute) else torch.Tensor(self.atac[idx]),
            "idx": torch.tensor(idx),
        }
        if self.batch_id is not None:
            item["batch_id"] = self.batch_id[idx]
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
    batch_size: int = 16,
    shuffle: bool = True,
    balance_class: bool = False,
    class_key: str | None = None,
    batch_key: str | None = None,
    num_workers: int = 0,
    feature_split: str | None = None,
) -> DataLoader:
    """Build dataloader for cell batches.

    Parameters
    ----------
    mdata
        MuData with rna and atac modalities.
    split
        Cell split: 'train' or 'test'.
    batch_size
        Cells per batch.
    shuffle
        Whether to shuffle.
    balance_class
        Whether to use weighted sampling for class balance.
    class_key
        Column in obs for class balancing.
    batch_key
        Column in obs for batch correction. When specified, one-hot encodes
        batch IDs and includes them in each batch for the VAE batch correction layer.
    num_workers
        Number of data loading workers.
    feature_split
        Feature split: 'train', 'test', or None (all features).
        When specified, filters genes/regions by var['split'].

    Returns
    -------
    DataLoader
        Cell dataloader.
    """
    # Validate split parameter
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    # Get cell split mask
    cell_mask = mdata.obs["split"] == split

    # Extract data with optional feature filtering
    rna_adata = mdata.mod["rna"][cell_mask]
    atac_adata = mdata.mod["atac"][cell_mask]

    if feature_split is not None:
        # Filter to features matching the specified split
        # For RNA, include 'both' split (TFs) along with the requested split
        if "split" not in rna_adata.var.columns:
            raise ValueError(
                f"Cannot filter by feature_split='{feature_split}': "
                "RNA modality is missing 'split' column in var. "
                "Run ds.pp.split_features_by_chromosome() first."
            )
        if "split" not in atac_adata.var.columns:
            raise ValueError(
                f"Cannot filter by feature_split='{feature_split}': "
                "ATAC modality is missing 'split' column in var. "
                "Run ds.pp.split_features_by_chromosome() first."
            )

        rna_var_mask = rna_adata.var["split"] == feature_split
        atac_var_mask = atac_adata.var["split"] == feature_split

        rna = rna_adata[:, rna_var_mask].X
        atac = atac_adata[:, atac_var_mask].X
    else:
        rna = rna_adata.X
        atac = atac_adata.X

    # Create batch_id one-hot encoding for batch correction
    batch_id = None
    if batch_key is not None:
        import pandas as pd

        batch_values = mdata.obs.loc[cell_mask, batch_key].values
        batch_encoder = pd.get_dummies(batch_values)
        batch_id = batch_encoder.values.astype(np.float32)

    dataset = CellDataset(rna, atac, batch_id=batch_id)

    # Weighted sampling for class balance
    sampler = None
    if balance_class:
        if class_key is None:
            raise ValueError(
                "balance_class=True requires class_key to be set. Provide the obs column name containing class labels."
            )
        from sklearn.utils.class_weight import compute_class_weight

        classes = mdata.obs.loc[cell_mask, class_key].values
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
    batch_size: int = 200,
    shuffle: bool = True,
    shift_augs: tuple[int, int] = (-3, 3),
    rc_aug: bool = True,
    *,
    context_length: int,
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
    ...     batch_size=200,
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

    # Force num_workers=0 for sequence dataloaders - pyfaidx file handles
    # don't survive multiprocessing reliably, causing corrupted sequences
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )


def build_test_sequence_dataloader(
    mdata: md.MuData,
    genome: Genome | None = None,
    batch_size: int = 200,
    *,
    context_length: int,
    num_workers: int = 0,
) -> tuple[DataLoader, torch.Tensor]:
    """Build dataloader for TEST region sequences (no augmentation).

    Parameters
    ----------
    mdata
        MuData with split column in atac.var
    genome
        Genome instance (uses global if None)
    batch_size
        Sequences per batch
    context_length
        Sequence length for model
    num_workers
        DataLoader workers

    Returns
    -------
    tuple[DataLoader, torch.Tensor]
        - DataLoader yielding (sequence, local_idx) tuples
        - test_region_indices: tensor mapping local idx to global region idx
    """
    from .._genome import GenomeIntervalDataset, get_genome

    if genome is None:
        genome = get_genome()

    # Get test region mask and indices
    atac_var = mdata.mod["atac"].var
    test_mask = atac_var["split"] == "test"
    test_region_indices = torch.tensor(np.where(test_mask)[0], dtype=torch.long)
    test_region_names = mdata.mod["atac"].var_names[test_mask].tolist()

    # Build dataset WITHOUT augmentation
    ds = GenomeIntervalDataset(
        regions=test_region_names,
        genome=genome,
        context_length=context_length,
        shift_augs=(0, 0),  # No augmentation
        rc_aug=False,  # No reverse complement
    )
    dataset = SequenceDatasetWithIndex(ds)

    # Force num_workers=0 for sequence dataloaders - pyfaidx file handles
    # don't survive multiprocessing reliably, causing corrupted sequences
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic for evaluation
        num_workers=0,
        pin_memory=True,
    )

    return dataloader, test_region_indices
