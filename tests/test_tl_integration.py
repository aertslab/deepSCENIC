"""Integration tests for training loop components."""

import pytest
import torch

from deepscenic.models import DeepSCENICVAE, MotifNet
from deepscenic.tl._loss import compute_total_loss


@pytest.fixture
def training_setup(minimal_dims):
    """Create a complete training setup with minimal dimensions."""
    d = minimal_dims
    n_cells = d["n_cells"] * 2

    r2g_indices = torch.stack(
        [
            torch.randint(0, d["n_regions"], (d["n_links"],)),
            torch.randint(0, d["n_genes"], (d["n_links"],)),
        ]
    )
    r2g_distances = torch.rand(d["n_links"])
    tf_indices = torch.randperm(d["n_genes"])[: d["n_tfs"]]
    gene_indices = torch.arange(d["n_genes"])
    region_indices = torch.arange(d["n_regions"])

    vae = DeepSCENICVAE(
        n_tfs=d["n_tfs"],
        n_genes=d["n_genes"],
        n_regions=d["n_regions"],
        r2g_indices=r2g_indices,
        r2g_distances=r2g_distances,
        tf_indices=tf_indices,
        gene_indices=gene_indices,
        region_indices=region_indices,
        n_hidden=d["n_hidden"],
        use_ppi=False,
        n_batches=0,
    )

    tf2rnet = MotifNet(
        n_tfs=d["n_tfs"],
        bottleneck_size=d["bottleneck_size"],
        emb_len=d["emb_len"],
    )

    adj_E1 = torch.abs(torch.randn(d["n_regions"], d["n_tfs"])) + 0.1

    x_rna = torch.rand(n_cells, d["n_genes"])
    x_atac = torch.rand(n_cells, d["n_regions"])

    return {
        "vae": vae,
        "tf2rnet": tf2rnet,
        "adj_E1": adj_E1,
        "x_rna": x_rna,
        "x_atac": x_atac,
        "dims": d,
        "n_cells": n_cells,
        "tf_indices": tf_indices,
        "gene_indices": gene_indices,
        "region_indices": region_indices,
        "r2g_distances": r2g_distances,
    }


class TestGradientFlow:
    """Test gradient flow through full training loop."""

    def test_vae_backward_computes_gradients(self, training_setup):
        """VAE backward pass should compute gradients."""
        vae = training_setup["vae"]
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        optimizer.zero_grad()

        output = vae(
            training_setup["x_rna"],
            training_setup["adj_E1"],
            use_ppi=False,
            use_mean=False,
        )

        losses = compute_total_loss(
            x_rna=training_setup["x_rna"],
            x_atac=training_setup["x_atac"],
            x_rna_rec=output.x_rna_rec,
            x_atac_rec=output.x_atac_rec,
            mu=output.mu,
            logvar=output.logvar,
            adj_E1_batch=training_setup["adj_E1"],
            adj_E2=vae.adj_E2,
            r2g_distances=training_setup["r2g_distances"],
            x_rna_ppi=output.x_rna_ppi,
            gene_indices=training_setup["gene_indices"],
            region_indices=training_setup["region_indices"],
        )

        losses["total"].backward()

        has_gradients = False
        for param in vae.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        assert has_gradients


class TestLossComputation:
    """Test loss computation during training."""

    def test_loss_is_scalar(self, training_setup):
        """Total loss should be a scalar."""
        vae = training_setup["vae"]
        output = vae(
            training_setup["x_rna"],
            training_setup["adj_E1"],
            use_ppi=False,
            use_mean=False,
        )

        losses = compute_total_loss(
            x_rna=training_setup["x_rna"],
            x_atac=training_setup["x_atac"],
            x_rna_rec=output.x_rna_rec,
            x_atac_rec=output.x_atac_rec,
            mu=output.mu,
            logvar=output.logvar,
            adj_E1_batch=training_setup["adj_E1"],
            adj_E2=vae.adj_E2,
            r2g_distances=training_setup["r2g_distances"],
            x_rna_ppi=output.x_rna_ppi,
            gene_indices=training_setup["gene_indices"],
            region_indices=training_setup["region_indices"],
        )

        assert losses["total"].dim() == 0


class TestOptimizerStep:
    """Test optimizer step changes parameters."""

    def test_optimizer_step_changes_params(self, training_setup):
        """Optimizer step should change model parameters."""
        vae = training_setup["vae"]
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        initial_params = {name: param.clone() for name, param in vae.named_parameters()}

        output = vae(
            training_setup["x_rna"],
            training_setup["adj_E1"],
            use_ppi=False,
            use_mean=False,
        )

        losses = compute_total_loss(
            x_rna=training_setup["x_rna"],
            x_atac=training_setup["x_atac"],
            x_rna_rec=output.x_rna_rec,
            x_atac_rec=output.x_atac_rec,
            mu=output.mu,
            logvar=output.logvar,
            adj_E1_batch=training_setup["adj_E1"],
            adj_E2=vae.adj_E2,
            r2g_distances=training_setup["r2g_distances"],
            x_rna_ppi=output.x_rna_ppi,
            gene_indices=training_setup["gene_indices"],
            region_indices=training_setup["region_indices"],
        )

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        params_changed = False
        for name, param in vae.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break

        assert params_changed


class TestEndToEndMockTraining:
    """End-to-end test with mock components."""

    def test_mini_training_loop(self, training_setup):
        """Run a minimal training loop to verify all components work together."""
        vae = training_setup["vae"]
        tf2rnet = training_setup["tf2rnet"]
        adj_E1_cache = training_setup["adj_E1"].clone()
        d = training_setup["dims"]

        optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)
        optimizer_tf2rnet = torch.optim.Adam(tf2rnet.parameters(), lr=1e-4)

        x_rna = training_setup["x_rna"]
        x_atac = training_setup["x_atac"]

        for _ in range(3):
            emb = torch.randn(4, d["bottleneck_size"] * d["emb_len"])
            seq_idx = torch.randint(0, d["n_regions"], (4,))

            optimizer_tf2rnet.zero_grad()
            tf_pred = tf2rnet(emb)

            adj_E1 = adj_E1_cache.clone()
            adj_E1[seq_idx] = tf_pred

            with torch.no_grad():
                adj_E1_cache[seq_idx] = tf_pred.detach()

            optimizer_vae.zero_grad()
            output = vae(x_rna, adj_E1, use_ppi=False, use_mean=False)

            losses = compute_total_loss(
                x_rna=x_rna,
                x_atac=x_atac,
                x_rna_rec=output.x_rna_rec,
                x_atac_rec=output.x_atac_rec,
                mu=output.mu,
                logvar=output.logvar,
                adj_E1_batch=adj_E1[seq_idx],
                adj_E2=vae.adj_E2,
                r2g_distances=training_setup["r2g_distances"],
                x_rna_ppi=output.x_rna_ppi,
                gene_indices=training_setup["gene_indices"],
                region_indices=training_setup["region_indices"],
            )

            e1_sparse = tf_pred.abs().mean() * 0.01
            total_loss = losses["total"] + e1_sparse

            total_loss.backward()
            optimizer_vae.step()
            optimizer_tf2rnet.step()

        vae.eval()
        output = vae(x_rna, adj_E1_cache, use_ppi=False, use_mean=True)
        assert torch.isfinite(output.x_rna_rec).all()
        assert torch.isfinite(output.x_atac_rec).all()
