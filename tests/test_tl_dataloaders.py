"""Tests for dataloaders."""

import random

import numpy as np
import pytest

from deepscenic._genome import clear_genome, register_genome
from deepscenic.tl._dataloaders import CellDataset


class TestCellDataset:
    """Tests for CellDataset."""

    def test_length(self):
        """Length should match number of cells."""
        rna = np.random.randn(100, 50)
        atac = np.random.randn(100, 200)
        dataset = CellDataset(rna, atac)
        assert len(dataset) == 100

    def test_getitem_structure(self):
        """Getitem should return dict with correct keys."""
        rna = np.random.randn(100, 50)
        atac = np.random.randn(100, 200)
        dataset = CellDataset(rna, atac)

        item = dataset[0]
        assert "rna" in item
        assert "atac" in item
        assert "idx" in item

    def test_sparse_input(self):
        """Should handle sparse matrices."""
        from scipy.sparse import csr_matrix

        rna = csr_matrix(np.random.randn(100, 50))
        atac = csr_matrix(np.random.randn(100, 200))
        dataset = CellDataset(rna, atac)  # type: ignore

        assert len(dataset) == 100
        item = dataset[0]
        assert item["rna"].shape == (50,)


class TestBuildCellDataloader:
    """Tests for build_cell_dataloader function."""

    @pytest.fixture
    def mock_mdata(self):
        """Create mock MuData for testing."""
        pytest.importorskip("mudata")
        pytest.importorskip("anndata")
        import warnings

        import anndata as ad
        import mudata as md

        n_cells = 100
        n_genes = 50
        n_regions = 200

        rna = ad.AnnData(np.random.randn(n_cells, n_genes).astype(np.float32))
        rna.var_names = [f"Gene_{i}" for i in range(n_genes)]

        atac = ad.AnnData(np.random.randn(n_cells, n_regions).astype(np.float32))
        atac.var_names = [f"Region_{i}" for i in range(n_regions)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mdata = md.MuData({"rna": rna, "atac": atac})

        mdata.obs["split"] = ["train"] * 80 + ["test"] * 20

        return mdata

    def test_builds_dataloader(self, mock_mdata):
        """Should build working dataloader."""
        from deepscenic.tl._dataloaders import build_cell_dataloader

        loader = build_cell_dataloader(mock_mdata, split="train", batch_size=16)

        # Should be iterable
        batch = next(iter(loader))
        assert "rna" in batch
        assert "atac" in batch

    def test_respects_split(self, mock_mdata):
        """Should only include cells from specified split."""
        from deepscenic.tl._dataloaders import build_cell_dataloader

        train_loader = build_cell_dataloader(mock_mdata, split="train", batch_size=100)
        test_loader = build_cell_dataloader(mock_mdata, split="test", batch_size=100)

        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))

        assert train_batch["rna"].shape[0] == 80
        assert test_batch["rna"].shape[0] == 20

    def test_balances_training_classes(self, mock_mdata):
        """Class balancing should use a weighted sampler over training cells."""
        from torch.utils.data import WeightedRandomSampler

        from deepscenic.tl._dataloaders import build_cell_dataloader

        mock_mdata.obs["class"] = ["major"] * 75 + ["minor"] * 5 + ["test"] * 20
        loader = build_cell_dataloader(
            mock_mdata,
            split="train",
            balance_class=True,
            class_key="class",
        )

        assert isinstance(loader.sampler, WeightedRandomSampler)

    def test_class_balancing_requires_class_key(self, mock_mdata):
        """A class column must be specified when balancing is enabled."""
        from deepscenic.tl._dataloaders import build_cell_dataloader

        with pytest.raises(ValueError, match="requires class_key"):
            build_cell_dataloader(mock_mdata, split="train", balance_class=True)


class TestBuildSequenceDataloader:
    """Tests for build_sequence_dataloader function."""

    @pytest.fixture
    def tmp_fasta(self, tmp_path):
        """Create a temporary FASTA file for testing."""
        fasta_path = tmp_path / "test.fa"

        # Generate a deterministic sequence for reproducible tests
        random.seed(42)
        bases = "ACGT"
        seq = "".join(random.choice(bases) for _ in range(10000))

        with open(fasta_path, "w") as f:
            f.write(">chr1\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")

        # Create the index file using pyfaidx
        import pyfaidx

        pyfaidx.Fasta(str(fasta_path))

        return fasta_path

    @pytest.fixture(autouse=True)
    def reset_genome(self):
        """Reset global genome state before each test."""
        clear_genome()
        yield
        clear_genome()

    def test_builds_with_regions(self, tmp_fasta):
        """Test building dataloader from region list."""
        from deepscenic.tl._dataloaders import build_sequence_dataloader

        register_genome(tmp_fasta)

        regions = ["chr1:0-640", "chr1:100-740", "chr1:200-840"]
        loader = build_sequence_dataloader(
            regions=regions,
            batch_size=2,
            shuffle=False,
            context_length=640,
        )

        batch = next(iter(loader))
        seqs, indices = batch
        # Shape is (batch, context_length, 4) - batch=2, context_length=640, channels=4
        assert seqs.shape[0] == 2
        assert seqs.shape[-2] == 640

    def test_respects_context_length(self, tmp_fasta):
        """Should use specified context_length."""
        from deepscenic.tl._dataloaders import build_sequence_dataloader

        register_genome(tmp_fasta)

        regions = ["chr1:0-320"]
        loader = build_sequence_dataloader(
            regions=regions,
            batch_size=1,
            context_length=320,
            shuffle=False,
        )

        batch = next(iter(loader))
        seqs, indices = batch
        # Shape is (batch, context_length, 4)
        assert seqs.shape[-2] == 320

    def test_returns_indices(self, tmp_fasta):
        """Should return region indices with sequences."""
        from deepscenic.tl._dataloaders import build_sequence_dataloader

        register_genome(tmp_fasta)

        regions = ["chr1:0-640", "chr1:100-740", "chr1:200-840"]
        loader = build_sequence_dataloader(
            regions=regions,
            batch_size=3,
            shuffle=False,
            context_length=640,
        )

        batch = next(iter(loader))
        seqs, indices = batch
        assert len(indices) == 3
        assert set(indices.tolist()) == {0, 1, 2}

    def test_raises_without_genome(self):
        """Should raise error when no genome registered."""
        from deepscenic.tl._dataloaders import build_sequence_dataloader

        regions = ["chr1:0-640"]

        with pytest.raises(RuntimeError, match="No genome registered"):
            build_sequence_dataloader(regions=regions, context_length=640)
