"""Tests for the main VAE model."""

import pytest
import torch

from deepscenic.models._vae import DeepSCENICVAE, VAEOutput


def create_test_vae(
    n_tfs: int = 50,
    n_genes: int = 100,
    n_regions: int = 200,
    n_links: int = 500,
    n_hidden: int = 32,
    use_ppi: bool = False,
    n_batches: int = 0,
) -> DeepSCENICVAE:
    """Create a small VAE for testing."""
    # Create sparse r2g indices (random links within bounds)
    r2g_rows = torch.randint(0, n_regions, (n_links,))
    r2g_cols = torch.randint(0, n_genes, (n_links,))
    r2g_indices = torch.stack([r2g_rows, r2g_cols])
    r2g_distances = torch.rand(n_links)

    # TF and gene indices
    tf_indices = torch.randperm(n_genes)[:n_tfs]
    gene_indices = torch.arange(n_genes)

    return DeepSCENICVAE(
        n_tfs=n_tfs,
        n_genes=n_genes,
        n_regions=n_regions,
        r2g_indices=r2g_indices,
        r2g_distances=r2g_distances,
        tf_indices=tf_indices,
        gene_indices=gene_indices,
        n_hidden=n_hidden,
        use_ppi=use_ppi,
        n_batches=n_batches,
    )


class TestDeepSCENICVAE:
    """Tests for DeepSCENICVAE model."""

    def test_forward_output_type(self):
        """Forward should return VAEOutput dataclass."""
        vae = create_test_vae()
        n_cells = 8
        x_rna = torch.randn(n_cells, 100)
        x_atac = torch.randn(n_cells, 200)
        adj_E1 = torch.randn(200, 50)

        output = vae(x_rna, x_atac, adj_E1)
        assert isinstance(output, VAEOutput)

    def test_output_shapes(self):
        """All output shapes should be correct."""
        n_tfs, n_genes, n_regions = 50, 100, 200
        vae = create_test_vae(n_tfs=n_tfs, n_genes=n_genes, n_regions=n_regions)
        n_cells = 8
        x_rna = torch.randn(n_cells, n_genes)
        x_atac = torch.randn(n_cells, n_regions)
        adj_E1 = torch.randn(n_regions, n_tfs)

        output = vae(x_rna, x_atac, adj_E1)

        assert output.x_rna_rec.shape == (n_cells, n_genes)
        assert output.x_atac_rec.shape == (n_cells, n_regions)
        assert output.z_tf.shape == (n_cells, n_tfs)
        assert output.mu.shape == (n_cells, n_tfs)
        assert output.logvar.shape == (n_cells, n_tfs)
        assert output.enh_act.shape == (n_cells, n_regions)
        assert output.z_rna.shape == (n_cells, n_genes)
        assert output.x_rna_ppi.shape == (n_cells, n_tfs)

    def test_use_mean_deterministic(self):
        """When use_mean=True, output should be deterministic."""
        vae = create_test_vae()
        x_rna = torch.randn(4, 100)
        x_atac = torch.randn(4, 200)
        adj_E1 = torch.randn(200, 50)

        out1 = vae(x_rna, x_atac, adj_E1, use_mean=True)
        out2 = vae(x_rna, x_atac, adj_E1, use_mean=True)

        assert torch.allclose(out1.z_tf, out2.z_tf)
        assert torch.allclose(out1.x_rna_rec, out2.x_rna_rec)

    def test_sampling_stochastic(self):
        """When use_mean=False, sampling should be stochastic."""
        vae = create_test_vae()
        x_rna = torch.randn(4, 100)
        x_atac = torch.randn(4, 200)
        adj_E1 = torch.randn(200, 50)

        out1 = vae(x_rna, x_atac, adj_E1, use_mean=False)
        out2 = vae(x_rna, x_atac, adj_E1, use_mean=False)

        # With high probability, samples should differ
        assert not torch.allclose(out1.z_tf, out2.z_tf)

    def test_batch_correction(self):
        """Batch correction should work when enabled."""
        n_batches = 3
        vae = create_test_vae(n_batches=n_batches)
        n_cells = 8
        x_rna = torch.randn(n_cells, 100)
        x_atac = torch.randn(n_cells, 200)
        adj_E1 = torch.randn(200, 50)
        batch_id = torch.zeros(n_cells, n_batches)
        batch_id[:, 0] = 1  # All cells in batch 0

        output = vae(x_rna, x_atac, adj_E1, batch_id=batch_id)
        assert output.x_rna_rec.shape == (n_cells, 100)

    def test_no_ppi_returns_ones(self):
        """Without PPI, x_rna_ppi should be all ones."""
        vae = create_test_vae(use_ppi=False)
        x_rna = torch.randn(4, 100)
        x_atac = torch.randn(4, 200)
        adj_E1 = torch.randn(200, 50)

        output = vae(x_rna, x_atac, adj_E1)
        assert torch.allclose(output.x_rna_ppi, torch.ones_like(output.x_rna_ppi))

    def test_e2_parameter(self):
        """adj_E2 should be a learnable parameter."""
        vae = create_test_vae()
        assert isinstance(vae.adj_E2, torch.nn.Parameter)
        assert vae.adj_E2.requires_grad

    def test_buffers_not_parameters(self):
        """Index tensors should be buffers, not parameters."""
        vae = create_test_vae()
        param_names = [name for name, _ in vae.named_parameters()]
        assert "tf_indices" not in param_names
        assert "gene_indices" not in param_names
        assert "r2g_indices" not in param_names

    def test_batch_size_one(self):
        """Should work with batch size 1."""
        vae = create_test_vae()
        x_rna = torch.randn(1, 100)
        x_atac = torch.randn(1, 200)
        adj_E1 = torch.randn(200, 50)

        output = vae(x_rna, x_atac, adj_E1)
        assert output.x_rna_rec.shape == (1, 100)


class TestVAEWithPPI:
    """Tests for VAE with PPI network enabled."""

    @pytest.fixture
    def vae_with_ppi(self):
        """Create VAE with PPI network."""
        pytest.importorskip("torch_geometric")

        n_tfs = 50
        n_genes = 100
        n_regions = 200
        n_links = 500
        n_ppi_genes = 80

        # Create sparse r2g indices
        r2g_indices = torch.stack([
            torch.randint(0, n_regions, (n_links,)),
            torch.randint(0, n_genes, (n_links,)),
        ])
        r2g_distances = torch.rand(n_links)

        # TF and gene indices
        tf_indices = torch.randperm(n_genes)[:n_tfs]
        gene_indices = torch.arange(n_genes)

        # PPI indices
        ppi_genes_idx = torch.randperm(n_genes)[:n_ppi_genes]
        ppi_edge_index = torch.randint(0, n_ppi_genes, (2, 200))
        ppi_tfs_idx_keys = torch.randperm(n_ppi_genes)[:n_tfs]
        ppi_tfs_idx_values = torch.arange(n_tfs)

        return DeepSCENICVAE(
            n_tfs=n_tfs,
            n_genes=n_genes,
            n_regions=n_regions,
            r2g_indices=r2g_indices,
            r2g_distances=r2g_distances,
            tf_indices=tf_indices,
            gene_indices=gene_indices,
            ppi_edge_index=ppi_edge_index,
            ppi_genes_idx=ppi_genes_idx,
            ppi_tfs_idx_keys=ppi_tfs_idx_keys,
            ppi_tfs_idx_values=ppi_tfs_idx_values,
            n_hidden=32,
            use_ppi=True,
        )

    def test_ppi_modulates_output(self, vae_with_ppi):
        """PPI should modulate the output."""
        x_rna = torch.randn(4, 100)
        x_atac = torch.randn(4, 200)
        adj_E1 = torch.randn(200, 50)

        out_with_ppi = vae_with_ppi(x_rna, x_atac, adj_E1, use_ppi=True)
        out_without_ppi = vae_with_ppi(x_rna, x_atac, adj_E1, use_ppi=False)

        # Outputs should differ when PPI is used
        assert not torch.allclose(out_with_ppi.z_tf, out_without_ppi.z_tf)

    def test_ppi_weights_in_range(self, vae_with_ppi):
        """PPI weights should be in [0, 1] due to sigmoid."""
        x_rna = torch.randn(4, 100)
        x_atac = torch.randn(4, 200)
        adj_E1 = torch.randn(200, 50)

        output = vae_with_ppi(x_rna, x_atac, adj_E1, use_ppi=True)
        assert (output.x_rna_ppi >= 0).all()
        assert (output.x_rna_ppi <= 1).all()
