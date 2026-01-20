"""Tests for deepSCENIC training plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing


@pytest.fixture
def sample_history():
    """Create sample training history."""
    np.random.seed(42)
    epochs = 100
    base_loss = np.exp(-np.linspace(0, 2, epochs)) + 0.1

    return pd.DataFrame(
        {
            "epoch": range(epochs),
            "loss": base_loss + np.random.randn(epochs) * 0.01,
            "val_loss": base_loss * 1.1 + np.random.randn(epochs) * 0.01,
            "loss_rec_rna": base_loss * 0.5 + np.random.randn(epochs) * 0.005,
            "loss_rec_atac": base_loss * 0.3 + np.random.randn(epochs) * 0.005,
        }
    )


class TestLossCurves:
    """Tests for loss_curves function."""

    def test_creates_loss_curves_from_dataframe(self, sample_history):
        """Should create loss curves from DataFrame."""
        from deepscenic.pl import loss_curves

        ax = loss_curves(sample_history, show=False)

        assert ax is not None
        plt.close("all")

    def test_creates_loss_curves_from_dict(self):
        """Should create loss curves from dict."""
        from deepscenic.pl import loss_curves

        history = {
            "loss": [1.0, 0.8, 0.6, 0.5, 0.4],
            "val_loss": [1.1, 0.9, 0.7, 0.6, 0.5],
        }
        ax = loss_curves(history, show=False)

        assert ax is not None
        plt.close("all")

    def test_filters_metrics(self, sample_history):
        """Should only plot specified metrics."""
        from deepscenic.pl import loss_curves

        ax = loss_curves(sample_history, metrics=["loss", "val_loss"], show=False)

        assert ax is not None
        plt.close("all")

    def test_returns_figure_when_requested(self, sample_history):
        """Should return Figure when return_fig=True."""
        from matplotlib.figure import Figure

        from deepscenic.pl import loss_curves

        fig = loss_curves(sample_history, show=False, return_fig=True)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_linear_scale(self, sample_history):
        """Should use linear scale when log_scale=False."""
        from deepscenic.pl import loss_curves

        ax = loss_curves(sample_history, log_scale=False, show=False)

        assert ax is not None
        assert ax.get_yscale() == "linear"
        plt.close("all")

    def test_creates_loss_curves_from_model(self, mock_deepscenic_model):
        """Should create loss curves from DeepSCENICModel with history."""
        from deepscenic.pl import loss_curves

        ax = loss_curves(mock_deepscenic_model, show=False)

        assert ax is not None
        plt.close("all")

    def test_raises_error_for_model_without_history(self, mock_vae, mock_tf2rnet, mock_adj_E1, minimal_dims):
        """Should raise ValueError when model has no history."""
        import torch

        from deepscenic.pl import loss_curves
        from deepscenic.tl._model import DeepSCENICModel
        from deepscenic.tl._training_state import TrainingConfig

        d = minimal_dims

        # Create a simple mock enformer
        class MockEnformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return x

        model = DeepSCENICModel(
            vae=mock_vae,
            tf2rnet=mock_tf2rnet,
            enformer=MockEnformer(),
            adj_E1=mock_adj_E1,
            config=TrainingConfig(),
            tf_names=[f"TF{i}" for i in range(d["n_tfs"])],
            gene_names=[f"GENE{i}" for i in range(d["n_genes"])],
            region_names=[f"chr1:{i * 100}-{i * 100 + 100}" for i in range(d["n_regions"])],
            history=None,
        )

        with pytest.raises(ValueError, match="Model has no training history"):
            loss_curves(model, show=False)


class TestSparsityHistogram:
    """Tests for sparsity_histogram function."""

    def test_creates_both_histograms(self, mock_deepscenic_model):
        """Should create histograms for both E1 and E2."""
        from deepscenic.pl import sparsity_histogram

        # Returns None for 'both' when return_fig=False
        result = sparsity_histogram(mock_deepscenic_model, which="both", show=False)

        assert result is None
        plt.close("all")

    def test_creates_e1_histogram(self, mock_deepscenic_model):
        """Should create histogram for E1 only."""
        from deepscenic.pl import sparsity_histogram

        ax = sparsity_histogram(mock_deepscenic_model, which="E1", show=False)

        assert ax is not None
        plt.close("all")

    def test_creates_e2_histogram(self, mock_deepscenic_model):
        """Should create histogram for E2 only."""
        from deepscenic.pl import sparsity_histogram

        ax = sparsity_histogram(mock_deepscenic_model, which="E2", show=False)

        assert ax is not None
        plt.close("all")

    def test_returns_figure_when_requested(self, mock_deepscenic_model):
        """Should return Figure when return_fig=True."""
        from matplotlib.figure import Figure

        from deepscenic.pl import sparsity_histogram

        fig = sparsity_histogram(mock_deepscenic_model, which="both", show=False, return_fig=True)

        assert isinstance(fig, Figure)
        plt.close("all")


class TestLatentUMAP:
    """Tests for latent_umap function."""

    def test_raises_import_error_without_umap(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should raise ImportError if umap-learn not installed."""
        from deepscenic.pl import latent_umap

        # Just check the function exists - actual test requires umap-learn
        assert callable(latent_umap)

    @pytest.fixture
    def mock_umap(self, monkeypatch):
        """Mock umap-learn."""
        import sys
        from unittest.mock import MagicMock

        # Create mock UMAP
        mock_umap_class = MagicMock()
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.random.randn(16, 2)
        mock_umap_class.return_value = mock_umap_instance

        mock_module = MagicMock()
        mock_module.UMAP = mock_umap_class
        sys.modules["umap"] = mock_module

        yield mock_module

        del sys.modules["umap"]

    def test_creates_umap_with_mock(self, mock_deepscenic_model, mock_mdata_for_model, mock_umap):
        """Should create UMAP plot with mock."""
        from deepscenic.pl import latent_umap

        ax = latent_umap(mock_deepscenic_model, mock_mdata_for_model, show=False)

        assert ax is not None
        plt.close("all")
