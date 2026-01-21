"""Tests for PPI network."""

import pytest
import torch

# Skip all tests if torch_geometric is not available
pytest.importorskip("torch_geometric")

from deepscenic.models._ppi import NoPPI, PPIgnn, build_ppi_batch, extract_tf_weights


class TestPPIgnn:
    """Tests for PPIgnn (Graph Attention Network)."""

    def test_output_shape(self):
        """Output shape should match input nodes."""
        ppi = PPIgnn(hidden_channels=16, heads=2)
        n_nodes = 100
        x = torch.randn(n_nodes, 1)
        # Simple chain graph
        edge_index = torch.tensor([[i, i + 1] for i in range(n_nodes - 1)]).T
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Undirected

        out = ppi(x, edge_index)
        assert out.shape == (n_nodes, 1)


class TestNoPPI:
    """Tests for NoPPI (dummy module)."""

    def test_returns_ones(self):
        """Should return tensor of ones matching input shape."""
        no_ppi = NoPPI()
        x = torch.randn(100, 1)
        edge_index = torch.randint(0, 100, (2, 200))

        out = no_ppi(x, edge_index)
        assert out.shape == x.shape
        assert torch.allclose(out, torch.ones_like(x))


class TestBuildPPIBatch:
    """Tests for build_ppi_batch function."""

    def test_batch_structure(self):
        """Should create proper PyG batch."""
        n_cells = 8
        n_genes = 1000
        n_ppi_genes = 500
        x_rna = torch.randn(n_cells, n_genes)
        ppi_genes_idx = torch.randperm(n_genes)[:n_ppi_genes]
        edge_index = torch.randint(0, n_ppi_genes, (2, 1000))
        device = torch.device("cpu")

        batch = build_ppi_batch(x_rna, ppi_genes_idx, edge_index, device)

        # Batch should have n_cells * n_ppi_genes total nodes
        assert batch.x.shape[0] == n_cells * n_ppi_genes
        assert batch.x.shape[1] == 1


class TestExtractTFWeights:
    """Tests for extract_tf_weights function."""

    def test_output_shape(self):
        """Output shape should be (n_cells, n_tfs)."""
        n_cells = 8
        n_ppi_genes = 100
        n_tfs = 50
        ppi_output = torch.randn(n_cells * n_ppi_genes, 1)
        ppi_tfs_idx_keys = torch.randperm(n_ppi_genes)[:n_tfs]
        ppi_tfs_idx_values = torch.arange(n_tfs)
        device = torch.device("cpu")

        result = extract_tf_weights(ppi_output, n_cells, n_ppi_genes, ppi_tfs_idx_keys, ppi_tfs_idx_values, device)

        assert result.shape == (n_cells, n_tfs)

    def test_sigmoid_applied(self):
        """Output should be in [0, 1] range (sigmoid applied)."""
        n_cells = 4
        n_ppi_genes = 50
        n_tfs = 20
        ppi_output = torch.randn(n_cells * n_ppi_genes, 1) * 10  # Large values
        ppi_tfs_idx_keys = torch.arange(n_tfs)
        ppi_tfs_idx_values = torch.arange(n_tfs)
        device = torch.device("cpu")

        result = extract_tf_weights(ppi_output, n_cells, n_ppi_genes, ppi_tfs_idx_keys, ppi_tfs_idx_values, device)

        assert (result >= 0).all()
        assert (result <= 1).all()
