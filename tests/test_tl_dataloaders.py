"""Tests for dataloaders."""

import numpy as np
import pytest
import torch

from deepscenic.tl._dataloaders import CellDataset, SequenceDatasetWithIndex, collate_cell_batch


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

    def test_getitem_shapes(self):
        """Getitem should return correct shapes."""
        n_genes, n_regions = 50, 200
        rna = np.random.randn(100, n_genes)
        atac = np.random.randn(100, n_regions)
        dataset = CellDataset(rna, atac)

        item = dataset[0]
        assert item["rna"].shape == (n_genes,)
        assert item["atac"].shape == (n_regions,)

    def test_with_batch_ids(self):
        """Should include batch_id when provided."""
        rna = np.random.randn(100, 50)
        atac = np.random.randn(100, 200)
        batch_ids = np.zeros((100, 3))
        batch_ids[:, 0] = 1  # All cells in batch 0
        dataset = CellDataset(rna, atac, batch_ids)

        item = dataset[0]
        assert "batch_id" in item
        assert item["batch_id"].shape == (3,)

    def test_sparse_input(self):
        """Should handle sparse matrices."""
        from scipy.sparse import csr_matrix

        rna = csr_matrix(np.random.randn(100, 50))
        atac = csr_matrix(np.random.randn(100, 200))
        dataset = CellDataset(rna, atac)

        assert len(dataset) == 100
        item = dataset[0]
        assert item["rna"].shape == (50,)

    def test_tensors_are_float(self):
        """Tensors should be float type."""
        rna = np.random.randn(10, 5).astype(np.float64)
        atac = np.random.randn(10, 20).astype(np.float64)
        dataset = CellDataset(rna, atac)

        item = dataset[0]
        assert item["rna"].dtype == torch.float32
        assert item["atac"].dtype == torch.float32


class TestSequenceDatasetWithIndex:
    """Tests for SequenceDatasetWithIndex."""

    def test_wraps_dataset(self):
        """Should wrap underlying dataset and add index."""

        # Mock dataset
        class MockDataset:
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return torch.randn(4, 640)  # (channels, length)

        mock = MockDataset()
        dataset = SequenceDatasetWithIndex(mock)

        assert len(dataset) == 100

        seq, idx = dataset[42]
        assert seq.shape == (4, 640)
        assert idx == 42


class TestCollateCellBatch:
    """Tests for collate_cell_batch function."""

    def test_collates_rna_atac(self):
        """Should stack rna and atac tensors."""
        batch = [
            {"rna": torch.randn(50), "atac": torch.randn(200), "idx": torch.tensor(0)},
            {"rna": torch.randn(50), "atac": torch.randn(200), "idx": torch.tensor(1)},
        ]
        result = collate_cell_batch(batch)

        assert result["rna"].shape == (2, 50)
        assert result["atac"].shape == (2, 200)
        assert result["idx"].shape == (2,)

    def test_collates_batch_id(self):
        """Should include batch_id when present."""
        batch = [
            {
                "rna": torch.randn(50),
                "atac": torch.randn(200),
                "idx": torch.tensor(0),
                "batch_id": torch.tensor([1.0, 0.0, 0.0]),
            },
            {
                "rna": torch.randn(50),
                "atac": torch.randn(200),
                "idx": torch.tensor(1),
                "batch_id": torch.tensor([0.0, 1.0, 0.0]),
            },
        ]
        result = collate_cell_batch(batch)

        assert "batch_id" in result
        assert result["batch_id"].shape == (2, 3)


class TestBuildCellDataloader:
    """Tests for build_cell_dataloader function."""

    @pytest.fixture
    def mock_mdata(self):
        """Create mock MuData for testing."""
        pytest.importorskip("mudata")
        pytest.importorskip("anndata")
        import anndata as ad
        import mudata as md
        import warnings

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

    def test_batch_correction(self, mock_mdata):
        """Should include batch_id when batch_key specified."""
        from deepscenic.tl._dataloaders import build_cell_dataloader

        mock_mdata.obs["sample"] = ["A"] * 50 + ["B"] * 50

        loader = build_cell_dataloader(
            mock_mdata, split="train", batch_size=16, batch_key="sample"
        )

        batch = next(iter(loader))
        assert "batch_id" in batch
        assert batch["batch_id"].shape[1] == 2  # Two batches
