"""Tests for model wrapper (_model.py)."""

import tempfile
from pathlib import Path

import pytest
import torch


class TestDeepSCENICModel:
    """Tests for DeepSCENICModel class."""

    def test_has_required_attributes(self, mock_deepscenic_model):
        """Model should have all required attributes."""
        assert hasattr(mock_deepscenic_model, "vae")
        assert hasattr(mock_deepscenic_model, "tf2rnet")
        assert hasattr(mock_deepscenic_model, "enformer")
        assert hasattr(mock_deepscenic_model, "adj_E1")
        assert hasattr(mock_deepscenic_model, "config")
        assert hasattr(mock_deepscenic_model, "tf_names")
        assert hasattr(mock_deepscenic_model, "gene_names")
        assert hasattr(mock_deepscenic_model, "region_names")
        assert hasattr(mock_deepscenic_model, "history")


class TestModelSaveLoad:
    """Tests for model save/load functionality."""

    def test_save_creates_file(self, mock_deepscenic_model):
        """Save should create a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            mock_deepscenic_model.save(path)
            assert path.exists()

    def test_save_file_is_loadable(self, mock_deepscenic_model):
        """Saved file should be loadable by torch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            mock_deepscenic_model.save(path)
            data = torch.load(path)
            assert "vae_state_dict" in data
            assert "tf2rnet_state_dict" in data
            assert "adj_E1" in data
            assert "config" in data
            assert "tf_names" in data
            assert "gene_names" in data
            assert "region_names" in data

    def test_save_contains_architecture_info(self, mock_deepscenic_model):
        """Saved file should contain architecture info for reconstruction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            mock_deepscenic_model.save(path)
            data = torch.load(path)
            assert "n_tfs" in data
            assert "n_genes" in data
            assert "n_regions" in data
            assert "n_hidden" in data

    def test_save_contains_buffers(self, mock_deepscenic_model):
        """Saved file should contain buffer tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            mock_deepscenic_model.save(path)
            data = torch.load(path)
            assert "tf_indices" in data
            assert "gene_indices" in data
            assert "r2g_indices" in data
            assert "r2g_distances" in data


class TestModelHistory:
    """Tests for training history on DeepSCENICModel."""

    def test_model_has_history_attribute(self, mock_deepscenic_model):
        """Model should have history attribute."""
        assert hasattr(mock_deepscenic_model, "history")
        assert mock_deepscenic_model.history is not None

    def test_history_default_is_none(self, mock_vae, mock_tf2rnet, mock_adj_E1, minimal_dims):
        """Model should accept None history by default."""
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
        )
        assert model.history is None


class TestHistorySaveLoad:
    """Tests for history persistence in save/load."""

    def test_save_includes_history_when_present(self, mock_deepscenic_model):
        """Save should include history when model has it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            mock_deepscenic_model.save(path)
            data = torch.load(path)

            assert "history" in data
            assert data["history"]["train"]["total"] == [1.0, 0.8]
            assert data["history"]["val"]["total"] == [1.1, 0.9]

    def test_save_omits_history_when_none(self, mock_vae, mock_tf2rnet, mock_adj_E1, minimal_dims):
        """Save should not include history key when None."""
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

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(path)
            data = torch.load(path)

            assert "history" not in data


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_returns_correct_type(self, mock_deepscenic_model):
        """load_model should return DeepSCENICModel (skip if Enformer needed)."""
        pytest.skip("Requires Enformer model download")
