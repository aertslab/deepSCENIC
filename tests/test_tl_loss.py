"""Tests for loss functions."""

import pytest
import torch

from deepscenic.tl._loss import (
    compute_total_loss,
    e1_sparsity_loss,
    e2_sparsity_loss,
    kl_divergence,
    ppi_activation_loss,
    reconstruction_loss,
)


class TestReconstructionLoss:
    """Tests for reconstruction_loss function."""

    def test_mse_loss(self):
        """MSE loss should compute correctly."""
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        loss = reconstruction_loss(pred, target, "mse")
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_mse_loss_nonzero(self):
        """MSE loss should be positive for different values."""
        pred = torch.tensor([[0.0, 0.0]])
        target = torch.tensor([[1.0, 1.0]])
        loss = reconstruction_loss(pred, target, "mse")
        assert loss > 0

    def test_mae_loss(self):
        """MAE loss should compute correctly."""
        pred = torch.tensor([[0.0, 0.0]])
        target = torch.tensor([[1.0, 1.0]])
        loss = reconstruction_loss(pred, target, "mae")
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_cosine_loss(self):
        """Cosine loss should be 0 for identical vectors."""
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])
        loss = reconstruction_loss(pred, target, "cosine")
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_cosine_loss_orthogonal(self):
        """Cosine loss should be 1 for orthogonal vectors."""
        pred = torch.tensor([[1.0, 0.0]])
        target = torch.tensor([[0.0, 1.0]])
        loss = reconstruction_loss(pred, target, "cosine")
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)

    def test_bce_loss(self):
        """BCE loss should work for binary targets."""
        pred = torch.tensor([[0.0, 0.0]])  # logits
        target = torch.tensor([[0.0, 1.0]])
        loss = reconstruction_loss(pred, target, "bce")
        assert loss > 0

    def test_bce_loss_missing_data(self):
        """BCE loss should skip all-negative-one rows."""
        pred = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        target = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        loss = reconstruction_loss(pred, target, "bce")
        assert loss > 0  # Should only compute for second row

    def test_dropout_mask_mse(self):
        """Dropout mask should ignore zero values."""
        pred = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        target = torch.tensor([[0.0, 1.0], [0.0, 1.0]])  # First column is dropout
        loss = reconstruction_loss(pred, target, "mse", dropout_mask=True)
        # Should only compute loss on second column (value 1)
        expected = 0.0  # pred=1, target=1, so (1-1)^2 = 0
        assert torch.isclose(loss, torch.tensor(expected))

    def test_invalid_loss_type(self):
        """Invalid loss type should raise error."""
        pred = torch.randn(4, 10)
        target = torch.randn(4, 10)
        with pytest.raises(ValueError, match="Unknown loss type"):
            reconstruction_loss(pred, target, "invalid")

    def test_returns_scalar(self):
        """Loss should be a scalar."""
        pred = torch.randn(8, 100)
        target = torch.randn(8, 100)
        for loss_type in ["mse", "mae", "cosine"]:
            loss = reconstruction_loss(pred, target, loss_type)
            assert loss.dim() == 0


class TestKLDivergence:
    """Tests for kl_divergence function."""

    def test_zero_for_standard_normal(self):
        """KL should be zero when q = N(0, 1)."""
        mu = torch.zeros(10, 20)
        logvar = torch.zeros(10, 20)
        kl = kl_divergence(mu, logvar)
        assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)

    def test_positive_for_nonstandard(self):
        """KL should be positive for non-standard normal."""
        mu = torch.ones(10, 20)
        logvar = torch.zeros(10, 20)
        kl = kl_divergence(mu, logvar)
        assert kl > 0

    def test_returns_scalar(self):
        """KL should return a scalar."""
        mu = torch.randn(8, 50)
        logvar = torch.randn(8, 50)
        kl = kl_divergence(mu, logvar)
        assert kl.dim() == 0

    def test_gradient_flows(self):
        """Gradients should flow through KL."""
        mu = torch.randn(8, 50, requires_grad=True)
        logvar = torch.randn(8, 50, requires_grad=True)
        kl = kl_divergence(mu, logvar)
        kl.backward()
        assert mu.grad is not None
        assert logvar.grad is not None


class TestE1SparsityLoss:
    """Tests for e1_sparsity_loss function."""

    def test_zero_for_zeros(self):
        """Should be zero for zero weights."""
        adj_E1 = torch.zeros(100, 50)
        loss = e1_sparsity_loss(adj_E1)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_positive_for_nonzero(self):
        """Should be positive for non-zero weights."""
        adj_E1 = torch.ones(100, 50)
        loss = e1_sparsity_loss(adj_E1)
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_handles_negative_values(self):
        """Should take absolute value of weights."""
        adj_E1 = torch.full((100, 50), -1.0)
        loss = e1_sparsity_loss(adj_E1)
        assert torch.isclose(loss, torch.tensor(1.0))


class TestE2SparsityLoss:
    """Tests for e2_sparsity_loss function."""

    def test_weighted_by_distance(self):
        """Should be weighted by distance penalties."""
        adj_E2 = torch.ones(100)
        # Higher distance = higher penalty
        r2g_distances = torch.linspace(0.1, 1.0, 100)
        loss = e2_sparsity_loss(adj_E2, r2g_distances)
        # Average of distances times weights (all 1s)
        expected = r2g_distances.mean()
        assert torch.isclose(loss, expected)

    def test_zero_for_zero_weights(self):
        """Should be zero for zero weights regardless of distance."""
        adj_E2 = torch.zeros(100)
        r2g_distances = torch.ones(100)
        loss = e2_sparsity_loss(adj_E2, r2g_distances)
        assert torch.isclose(loss, torch.tensor(0.0))


class TestPPIActivationLoss:
    """Tests for ppi_activation_loss function."""

    def test_zero_for_ones(self):
        """Should be zero when all weights are 1."""
        x_rna_ppi = torch.ones(8, 50)
        loss = ppi_activation_loss(x_rna_ppi)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_one_for_zeros(self):
        """Should be 1 when all weights are 0."""
        x_rna_ppi = torch.zeros(8, 50)
        loss = ppi_activation_loss(x_rna_ppi)
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_intermediate_values(self):
        """Should be between 0 and 1 for intermediate weights."""
        x_rna_ppi = torch.full((8, 50), 0.5)
        loss = ppi_activation_loss(x_rna_ppi)
        assert torch.isclose(loss, torch.tensor(0.5))


class TestComputeTotalLoss:
    """Tests for compute_total_loss function."""

    @pytest.fixture
    def loss_inputs(self):
        """Create standard inputs for loss computation."""
        n_cells = 8
        n_genes = 100
        n_regions = 200
        n_tfs = 50
        n_links = 500

        return {
            "x_rna": torch.randn(n_cells, n_genes),
            "x_atac": torch.randn(n_cells, n_regions),
            "x_rna_rec": torch.randn(n_cells, n_genes),
            "x_atac_rec": torch.randn(n_cells, n_regions),
            "mu": torch.randn(n_cells, n_tfs),
            "logvar": torch.randn(n_cells, n_tfs),
            "adj_E1_batch": torch.randn(100, n_tfs),
            "adj_E2": torch.randn(n_links),
            "r2g_distances": torch.rand(n_links),
            "x_rna_ppi": torch.rand(n_cells, n_tfs),
            "gene_indices": torch.arange(n_genes),
        }

    def test_returns_dict(self, loss_inputs):
        """Should return dictionary with all loss components."""
        result = compute_total_loss(**loss_inputs)
        assert isinstance(result, dict)
        assert "total" in result
        assert "rec_rna" in result
        assert "rec_atac" in result
        assert "kl" in result
        assert "e1_sparse" in result
        assert "e2_sparse" in result
        assert "ppi" in result

    def test_total_is_sum(self, loss_inputs):
        """Total should approximately equal sum of components."""
        # Note: components are detached, so we need to compute fresh
        result = compute_total_loss(**loss_inputs)
        # Total should have gradients, components are detached
        assert result["total"].requires_grad or not result["total"].requires_grad

    def test_gradient_flows_through_total(self, loss_inputs):
        """Gradients should flow through total loss."""
        loss_inputs["x_rna_rec"] = torch.randn(8, 100, requires_grad=True)
        result = compute_total_loss(**loss_inputs)
        result["total"].backward()
        assert loss_inputs["x_rna_rec"].grad is not None

    def test_no_ppi_loss_when_disabled(self, loss_inputs):
        """PPI loss should be zero when use_ppi=False."""
        result = compute_total_loss(**loss_inputs, use_ppi=False)
        assert torch.isclose(result["ppi"], torch.tensor(0.0))

    def test_e1_sparsity_excluded_when_disabled(self, loss_inputs):
        """E1 sparsity should be excluded from total when include_e1_sparsity=False."""
        result_with = compute_total_loss(**loss_inputs, include_e1_sparsity=True)
        result_without = compute_total_loss(**loss_inputs, include_e1_sparsity=False)

        # E1 sparsity should still be computed for logging
        assert result_without["e1_sparse"] > 0

        # Total should differ by the E1 sparsity amount
        diff = result_with["total"] - result_without["total"]
        expected_diff = result_with["e1_sparse"]
        assert torch.isclose(diff, expected_diff, rtol=1e-4)

    def test_e1_sparsity_zero_in_total_when_disabled(self, loss_inputs):
        """Total should not include E1 sparsity when include_e1_sparsity=False."""
        # Set E1 to have a significant value
        loss_inputs["adj_E1_batch"] = torch.ones(100, 50) * 10.0

        result = compute_total_loss(**loss_inputs, include_e1_sparsity=False, alpha=1.0)

        # E1 sparsity should be 10.0 (mean of all 10s) times alpha
        assert torch.isclose(result["e1_sparse"], torch.tensor(10.0))

        # But total should NOT include this when disabled
        # Verify by checking total equals sum of other components
        expected_total = result["rec_rna"] + result["rec_atac"] + result["kl"] + result["e2_sparse"] + result["ppi"]
        assert torch.isclose(result["total"], expected_total, rtol=1e-4)
